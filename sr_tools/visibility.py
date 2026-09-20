"""Triangle visibility -> occupied 4096 voxels -> fixed C256 ancestry."""

import torch
from . import common


def coordinate_keys(coords, grid):
    xyz = coords[:, -3:].long()
    return (xyz[:, 0] * grid + xyz[:, 1]) * grid + xyz[:, 2]


def lookup_rows(source, query, grid):
    keys, order = coordinate_keys(source, grid).sort()
    q = coordinate_keys(query, grid)
    if not len(keys):
        return torch.zeros(len(q), dtype=torch.long), torch.zeros(
            len(q), dtype=torch.bool
        )
    pos = torch.searchsorted(keys, q).clamp_max(len(keys) - 1)
    return order[pos], keys[pos] == q


def latent_visibility(voxels, visible_voxels, c256):
    """Any visible source face in any occupied descendant makes a point visible."""
    rows, valid = lookup_rows(voxels, visible_voxels, 4096)
    voxel_visible = torch.zeros(len(voxels), dtype=torch.bool)
    voxel_visible[rows[valid]] = True
    parents = torch.div(voxels, 16, rounding_mode="floor")
    parent_rows, found = lookup_rows(c256, parents, 256)
    assert found.all(), "Voxel ancestry escaped encoded C256"
    assert len(parent_rows.unique()) == len(c256), "C256 has no voxel ancestor"
    visible = torch.zeros(len(c256), dtype=torch.bool)
    visible[parent_rows[voxel_visible]] = True
    return visible, voxel_visible, parent_rows


@torch.no_grad()
def visible_faces(mesh, camera, resolution=4096, face_chunk=1_000_000):
    """Z-buffer visibility at input camera; rasterized face IDs remain exact integers."""
    import nvdiffrast.torch as dr
    from pixal3d.renderers.mesh_renderer import intrinsics_to_projection
    from pixal3d.utils.render_utils import proj_camera_to_render_params

    ext, intr = proj_camera_to_render_params(
        camera["camera_angle_x"], camera["distance"]
    )
    vertices = mesh.vertices.cuda().float()
    homogeneous = torch.cat((vertices, torch.ones_like(vertices[:, :1])), 1)
    clip = (homogeneous @ (intrinsics_to_projection(intr, 0.01, 100) @ ext).T)[
        None
    ].contiguous()
    context = dr.RasterizeCudaContext()
    depth = torch.full((resolution, resolution), float("inf"), device="cuda")
    winner = torch.full((resolution, resolution), -1, dtype=torch.long, device="cuda")
    for start in range(0, len(mesh.faces), face_chunk):
        faces = mesh.faces[start : start + face_chunk].cuda().int().contiguous()
        rast, _ = dr.rasterize(context, clip, faces, (resolution, resolution))
        hit = (rast[0, ..., 3] > 0) & (rast[0, ..., 2] < depth)
        depth[hit] = rast[0, ..., 2][hit]
        winner[hit] = rast[0, ..., 3][hit].long() - 1 + start
        del rast, faces
    ids = winner[winner >= 0].unique().cpu()
    del vertices, homogeneous, clip, context, depth, winner
    common.empty_cuda()
    return ids


@torch.no_grad()
def voxelize(mesh, faces=None):
    import o_voxel

    return o_voxel.convert.mesh_to_flexible_dual_grid(
        mesh.vertices.float().cpu(),
        (mesh.faces if faces is None else faces).long().cpu(),
        grid_size=4096,
        aabb=[[-0.5] * 3, [0.5] * 3],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
        timing=True,
    )


@torch.no_grad()
def encode_geometry(pipe, mesh, camera, out):
    """Full and visible-face voxelizations share exactly the same world lattice."""
    from pixal3d import models
    from .baseline import ENCODER

    out.mkdir(parents=True, exist_ok=True)
    target = out / "encoded.pt"
    if target.exists():
        return common.load_payload(target)
    voxel_path = out / "voxels.pt"
    if not voxel_path.exists():
        indices, dual, intersections = voxelize(mesh)
        common.save(
            voxel_path,
            coords=indices.cpu(),
            dual=dual.cpu(),
            intersections=intersections.cpu(),
        )
    vox = common.load_payload(voxel_path)
    coords = torch.cat((torch.zeros_like(vox["coords"][:, :1]), vox["coords"]), 1).int()
    vertices = common.SparseTensor(
        (vox["dual"] * 4096 - vox["coords"]).cuda(), coords.cuda()
    )
    intersections = vertices.replace(vox["intersections"].cuda())
    encoder = models.from_pretrained(str(ENCODER)).eval().cuda()
    encoded = encoder(vertices, intersections, sample_posterior=False)
    c256, raw = encoded.coords.cpu(), encoded.feats.float().cpu()
    mean, std = common.normalization_tensors(pipe.shape_slat_normalization, raw.device)
    normalized = (raw - mean) / std
    assert torch.isfinite(normalized).all()
    del encoded, encoder, vertices, intersections
    common.empty_cuda()
    face_path = out / "visible_faces.pt"
    if face_path.exists():
        face_ids = common.load_payload(face_path)["face_ids"]
    else:
        face_ids = visible_faces(mesh, camera)
        common.save(face_path, face_ids=face_ids, camera=camera, resolution=4096)
    assert len(face_ids) > 0
    visible_path = out / "visible_voxels.pt"
    if visible_path.exists():
        visible_voxels = common.load_payload(visible_path)["coords"]
    else:
        visible_voxels, _, _ = voxelize(mesh, mesh.faces[face_ids])
        visible_voxels = visible_voxels.cpu()
        common.save(visible_path, coords=visible_voxels)
    visible, voxel_visible, parent_rows = latent_visibility(
        vox["coords"], visible_voxels, c256
    )
    common.save(
        out / "ancestry.pt",
        voxel_visible=voxel_visible,
        voxel_to_c256=parent_rows,
        visible_faces_file=str(face_path),
        rule="union of voxels intersected by raster-visible source faces; any visible descendant",
    )
    payload = dict(coords=c256, features=normalized, normalized=True, visible=visible)
    common.save(target, **payload)
    common.js(
        out / "visibility.json",
        dict(
            total_faces=len(mesh.faces),
            visible_faces=len(face_ids),
            voxels=len(coords),
            visible_voxels=int(voxel_visible.sum()),
            latent_points=len(c256),
            visible_latent_points=int(visible.sum()),
            raster_resolution=4096,
            rule="ANY source face visible; same voxelization for all source faces and visible source faces",
        ),
    )
    return payload


@torch.no_grad()
def derive_visibility_from_shape_support(mesh, camera, out, target):
    """Build visibility/ancestry without running the large dual-grid VAE.

    The VAE encoder is only used by the original path to recover a C256
    support and its features.  During SR preparation the fixed shape endpoint
    already supplies both.  The C256 support induced by the 4096 voxelization
    is therefore used as a temporary source lattice, and its visibility is
    remapped to the fixed shape support by nearest C256 coordinate.  This
    avoids a multi-gigabyte encoder activation peak for very dense meshes;
    voxelization, raster visibility, and the exact source-face ancestry are
    unchanged.
    """
    from scipy.spatial import cKDTree

    out.mkdir(parents=True, exist_ok=True)
    target = dict(
        coords=target["coords"].int().cpu(),
        features=target["features"].float().cpu(),
        normalized=True,
    )
    target_path = out / "encoded.pt"
    if target_path.exists():
        cached = common.load_payload(target_path)
        if not torch.equal(cached["coords"], target["coords"]):
            raise RuntimeError("cached visibility target support differs")
        return cached

    voxel_path = out / "voxels.pt"
    if not voxel_path.exists():
        indices, dual, intersections = voxelize(mesh)
        common.save(
            voxel_path,
            coords=indices.cpu(),
            dual=dual.cpu(),
            intersections=intersections.cpu(),
        )
    vox = common.load_payload(voxel_path)
    source_xyz = torch.div(vox["coords"].int(), 16, rounding_mode="floor").unique(dim=0)
    source_c256 = torch.cat(
        (torch.zeros_like(source_xyz[:, :1]), source_xyz), dim=1
    ).int()

    face_path = out / "visible_faces.pt"
    if face_path.exists():
        face_ids = common.load_payload(face_path)["face_ids"]
    else:
        face_ids = visible_faces(mesh, camera)
        common.save(face_path, face_ids=face_ids, camera=camera, resolution=4096)
    if len(face_ids) == 0:
        raise RuntimeError("visibility rasterization found no visible faces")

    visible_path = out / "visible_voxels.pt"
    if visible_path.exists():
        visible_voxels = common.load_payload(visible_path)["coords"]
    else:
        visible_voxels, _, _ = voxelize(mesh, mesh.faces[face_ids])
        visible_voxels = visible_voxels.cpu()
        common.save(visible_path, coords=visible_voxels)

    source_visible, voxel_visible, source_parent = latent_visibility(
        vox["coords"], visible_voxels, source_c256
    )
    source_xyz_np = source_c256[:, 1:].numpy()
    target_xyz_np = target["coords"][:, 1:].numpy()
    source_to_target = cKDTree(target_xyz_np).query(source_xyz_np, k=1)[1]
    target_to_source = cKDTree(source_xyz_np).query(target_xyz_np, k=1)[1]
    mapped_parent = torch.as_tensor(
        source_to_target[source_parent.numpy()], dtype=torch.long
    )
    visible = source_visible[torch.as_tensor(target_to_source).long()]
    common.save(
        out / "ancestry.pt",
        voxel_visible=voxel_visible,
        voxel_to_c256=mapped_parent,
        visible_faces_file=str(face_path),
        rule=(
            "4096 full/visible voxelization; source C256 is voxel-parent lattice; "
            "nearest C256 remap to fixed shape support"
        ),
    )
    payload = dict(
        coords=target["coords"],
        features=target["features"],
        normalized=True,
        visible=visible,
    )
    common.save(target_path, **payload)
    common.js(
        out / "visibility.json",
        dict(
            total_faces=len(mesh.faces),
            visible_faces=len(face_ids),
            voxels=len(vox["coords"]),
            visible_voxels=int(voxel_visible.sum()),
            source_latent_points=len(source_c256),
            latent_points=len(target["coords"]),
            visible_latent_points=int(visible.sum()),
            raster_resolution=4096,
            rule=(
                "ANY source face visible; source voxel-parent visibility remapped "
                "to fixed shape endpoint support"
            ),
        ),
    )
    return payload


def classify_blocks(data, visible, out):
    records = []
    for block in data["blocks"]:
        ids = block["global_ids"]
        block["visible_mask"] = visible[ids].bool().clone()
        fraction = float(visible[ids].float().mean()) if len(ids) else 0.0
        block["conditional"] = bool(10 * int(visible[ids].sum()) > 3 * len(ids))
        block["visible_fraction"] = fraction
        records.append(
            dict(
                block_id=block["block_id"],
                points=len(ids),
                visible_fraction=fraction,
                conditional=block["conditional"],
            )
        )
    common.js(
        out / "block_visibility.json",
        dict(
            threshold=0.30,
            comparison="strict >",
            denominator="all sparse points in block, including image padding context",
            blocks=records,
        ),
    )
    return records
