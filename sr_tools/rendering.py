"""Global mesh/material decoding and aligned native-resolution multi-view renders."""

import gc
import math
import numpy as np
import torch
from . import common
from .common import SparseTensor


class UnlitEnvironment:
    def shade(self, pos, normal, kd, ks, view_pos, specular=True):
        return kd


@torch.no_grad()
def decode_geometry(pipe, out, coords, features, resolution):
    marker = out / "geometry_mesh.pt"
    if marker.exists() and (out / "topology.pt").exists():
        return common.load_payload(marker)["mesh"]
    raw = common.denormalize_shape(pipe, features)
    meshes, subs = pipe.decode_shape_slat(
        SparseTensor(raw.to(pipe.device), coords.to(pipe.device)), resolution
    )
    mesh = meshes[0].cpu()
    common.save(out / "topology.pt", coords=coords, subs=cpu_topology(subs))
    common.save_geometry_mesh(mesh, out)
    common.js(
        out / "decode.json",
        dict(
            input_points=len(coords),
            resolution=resolution,
            coordinates="original global indices; native world mesh, no post-transform",
            vertices=len(mesh.vertices),
            faces=len(mesh.faces),
            global_decode=True,
        ),
    )
    del raw, meshes, subs
    common.empty_cuda()
    return mesh


def norm(pipe, kind, value, inverse=False):
    mean, std = common.normalization_tensors(
        getattr(pipe, f"{kind}_slat_normalization"), value.device
    )
    return value * std + mean if inverse else (value - mean) / std


def cpu_topology(subs):
    # SparseTensor.cpu() retains spatial caches, including CUDA convolution
    # maps. Persist only subdivision features/coords, never those caches.
    return [
        common.SparseTensor(s.feats.detach().cpu(), s.coords.detach().cpu())
        for s in subs
    ]


@torch.no_grad()
def decode_material(pipe, source, out, shape, texture_features):
    from pixal3d.representations.mesh import MeshWithVoxel

    target = out / "textured_mesh.pt"
    if target.exists():
        return common.load_payload(target)["mesh"]
    # Reuse the subdivisions returned by the geometry decode. No second trunk pass.
    topology_path = source / "final_mesh/topology.pt"
    topology = common.load_payload(topology_path)
    assert torch.equal(topology["coords"], shape["coords"])
    assert len(texture_features) == len(shape["coords"])
    tex = SparseTensor(
        norm(pipe, "tex", texture_features.to(pipe.device), inverse=True),
        shape["coords"].to(pipe.device),
    )
    print("[texture] Decoding complete merged material at resolution 4096", flush=True)
    field = pipe.decode_tex_slat(tex, [s.to(pipe.device) for s in topology["subs"]])
    assert torch.isfinite(field.feats).all()
    coords, attrs = field.coords[:, 1:].cpu(), field.feats.cpu()
    del field, tex, topology
    gc.collect()
    common.empty_cuda()
    mesh = common.load_payload(source / "final_mesh/geometry_mesh.pt")["mesh"]
    mesh = MeshWithVoxel(
        mesh.vertices,
        mesh.faces,
        [-0.5] * 3,
        1 / 4096,
        coords,
        attrs,
        torch.Size([1, attrs.shape[1], 4096, 4096, 4096]),
        pipe.pbr_attr_layout,
    )
    common.save(target, mesh=mesh)
    common.js(
        out / "decode.json",
        dict(
            resolution=4096,
            material_decode_calls=1,
            latent_points=len(shape["coords"]),
            material_voxels=len(coords),
            vertices=len(mesh.vertices),
            faces=len(mesh.faces),
            geometry_source=str(source),
            topology="subdivisions saved by the single global geometry decode",
        ),
    )
    return mesh


def valid_image(path, resolution):
    if not path.exists():
        return False
    with common.Image.open(path) as image:
        return image.size == (resolution, resolution)


@torch.no_grad()
def render(pipe, mesh, out, resolution=2048, camera=None):
    import utils3d
    from pixal3d.renderers import PbrMeshRenderer, MeshRenderer
    from pixal3d.representations import Mesh
    from pixal3d.utils.render_utils import proj_camera_to_render_params

    if camera is None:
        raise ValueError("An explicit input camera is required")
    cam = camera
    _, intr = proj_camera_to_render_params(cam["camera_angle_x"], cam["distance"])
    renderer = PbrMeshRenderer(
        dict(
            resolution=resolution,
            near=0.01,
            far=100,
            ssaa=1,
            peel_layers=1,
            face_chunk_size=2_000_000,
        ),
        device=pipe.device,
    )
    geometry_renderer = MeshRenderer(
        dict(
            resolution=resolution,
            near=0.01,
            far=100,
            ssaa=1,
            chunk_size=2_000_000,
            antialias=False,
        ),
        device=pipe.device,
    )
    mesh = mesh.to(pipe.device)
    geometry_mesh = Mesh(mesh.vertices, mesh.faces)
    views = out / f"views_{resolution}"
    views.mkdir(exist_ok=True)
    cameras = []
    for yaw in common.YAWS:
        a = math.radians(yaw)
        eye = (
            torch.tensor([math.sin(a), 0.0, math.cos(a)], device=pipe.device)
            * cam["distance"]
        )
        ext = utils3d.torch.extrinsics_look_at(
            eye,
            torch.zeros(3, device=pipe.device),
            torch.tensor([0.0, 1.0, 0.0], device=pipe.device),
        )
        cameras.append(
            dict(yaw=yaw, extrinsics=ext.cpu().tolist(), intrinsics=intr.cpu().tolist())
        )
        if not all(
            valid_image(views / f"{kind}_{yaw:03d}.png", resolution)
            for kind in ("base_color", "normal")
        ):
            rendered = renderer.render(mesh, ext, intr, envmap=UnlitEnvironment())
            for kind in ("base_color", "normal"):
                image = (
                    (rendered[kind].permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255)
                    .round()
                    .astype(np.uint8)
                )
                common.Image.fromarray(image).save(views / f"{kind}_{yaw:03d}.png")
            del rendered
            common.empty_cuda()
        if not valid_image(views / f"geometry_normal_{yaw:03d}.png", resolution):
            # Exactly the same mesh, extrinsics, intrinsics and pixel grid as
            # the material render; normals are computed by geometry rasterization.
            rendered = geometry_renderer.render(
                geometry_mesh, ext, intr, return_types=["normal", "mask"]
            )
            normal = rendered["normal"] * rendered["mask"]
            image = (
                (normal.permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255)
                .round()
                .astype(np.uint8)
            )
            common.Image.fromarray(image).save(views / f"geometry_normal_{yaw:03d}.png")
            del rendered, normal
            common.empty_cuda()
        print(
            f"[texture + geometry render] yaw {yaw} complete at {resolution}x{resolution}",
            flush=True,
        )
    common.js(
        views / "cameras.json",
        dict(
            resolution=resolution,
            views=cameras,
            alignment="identical mesh, extrinsics, intrinsics, near/far and pixel grid for material and geometry",
            geometry_normal="camera-space face normals; RGB=(normal+1)/2; black background",
        ),
    )
    outputs = {}
    for kind in ("base_color", "normal", "geometry_normal"):
        # Preserve every view at native resolution inside the contact sheet.
        sheet = common.Image.new("RGB", (len(common.YAWS) * resolution, resolution))
        for i, yaw in enumerate(common.YAWS):
            with common.Image.open(views / f"{kind}_{yaw:03d}.png") as im:
                assert im.size == (resolution, resolution)
                sheet.paste(im, (i * resolution, 0))
        name = f"{kind}_six_views_{resolution}.png"
        sheet.save(out / name)
        outputs[kind] = name
    common.js(
        out / f"render_{resolution}.json",
        dict(
            status="COMPLETE",
            resolution=resolution,
            sheet_size=[len(common.YAWS) * resolution, resolution],
            cameras=str(views / "cameras.json"),
            outputs=outputs,
        ),
    )
