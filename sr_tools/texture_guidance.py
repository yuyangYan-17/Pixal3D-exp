"""Baseline endpoint transfer and visibility-routed texture sampling."""

from pathlib import Path
import torch
from . import common, rendering, texture, shape as sync
from .visibility import lookup_rows

TEX_ENCODER = Path(
    "/home/nvme04/yyyan/download/model/TRELLIS.2-4B/microsoft/TRELLIS___2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"
)


@torch.no_grad()
def baseline_fields(pipe, baseline, canonical, camera, out, *, decode_fields=True):
    """Replay the first four native texture updates, decode each predicted clean endpoint."""
    out.mkdir(parents=True, exist_ok=True)
    if decode_fields and all((out / f"field_{k}.pt").exists() for k in range(4)):
        return
    if not decode_fields and all((out / f"endpoint_{k}.pt").exists() for k in range(4)):
        return
    p = common.load_payload(baseline / "baseline/shape_c64_shape1024_denormalized.pt")
    raw = common.SparseTensor(p["features"].cuda(), p["coords"].cuda())
    cond = pipe.get_proj_cond_shape(
        pipe.image_cond_model_tex_1024,
        [canonical["image_1024"]],
        raw.coords,
        camera_angle_x=camera["camera_angle_x"],
        distance=camera["distance"],
        mesh_scale=1.0,
        grid_resolution_override=64,
    )
    normalized = raw.replace(rendering.norm(pipe, "shape", raw.feats))
    model = pipe.models["tex_slat_flow_model_1024"]
    sampler = pipe.tex_slat_sampler
    params = dict(pipe.tex_slat_sampler_params)
    params.pop("steps", None)
    times = sampler.timestep_schedule(12, params.pop("rescale_t", 1.0))
    # Match sample_tex_slat: CPU torch.randn, then transfer to the model device.
    noise = torch.randn(
        (len(raw.coords), model.in_channels - raw.feats.shape[1]),
        generator=torch.Generator().manual_seed(46),
    )
    x = raw.replace(noise.cuda())
    model.cuda()
    try:
        for k in range(4):
            v = sampler._inference_model(
                model, x, times[k], concat_cond=normalized, **cond, **params
            )
            endpoint = sampler._pred_to_xstart(x, times[k], v)
            common.save(
                out / f"endpoint_{k}.pt",
                coords=p["coords"],
                features=endpoint.feats.cpu(),
                t=times[k],
                t_next=times[k + 1],
                normalized=True,
                sigma_min=sampler.sigma_min,
                state=x.feats.cpu(),
                velocity=v.feats.cpu(),
            )
            x = x - (times[k] - times[k + 1]) * v
    finally:
        model.cpu()
    del cond, x, normalized, endpoint, v
    common.empty_cuda()
    if not decode_fields:
        del raw
        common.empty_cuda()
        return
    meshes, subs = pipe.decode_shape_slat(raw, 1024)
    subs = rendering.cpu_topology(subs)
    del meshes, raw
    common.empty_cuda()
    for k in range(4):
        target = out / f"field_{k}.pt"
        if target.exists():
            continue
        endpoint = common.load_payload(out / f"endpoint_{k}.pt")
        latent = common.SparseTensor(
            rendering.norm(pipe, "tex", endpoint["features"].cuda(), inverse=True),
            p["coords"].cuda(),
        )
        field = pipe.decode_tex_slat(latent, [sub.cuda() for sub in subs])
        assert torch.isfinite(field.feats).all()
        common.save(
            target,
            coords=field.coords.cpu(),
            attrs=field.feats.cpu(),
            resolution=1024,
            endpoint_t=endpoint["t"],
            sigma_min=endpoint["sigma_min"],
        )
        del field, latent
        common.empty_cuda()


@torch.no_grad()
def lift_guides_from_c64_endpoints(endpoint_dir, c256, out):
    """Lift native C64 texture endpoints onto the fixed C256 support.

    Native baseline texture flow runs on the C64 support produced by the
    baseline shape path, while the visibility experiment runs texture flow on
    the denser fixed C256 support.  The full 4096-field -> texture-encoder
    replay is prohibitively large for some assets (tens of millions of sparse
    voxels).  This deterministic world-coordinate nearest-3 lift preserves
    the native endpoint trajectory and supplies a finite guide at every C256
    point; it is only used for the first ``n`` guide steps.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    endpoint_dir = Path(endpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = torch.as_tensor(c256).detach().cpu().int()
    if target.ndim != 2 or target.shape[1] != 4:
        raise ValueError("c256 must have shape [N,4]")
    target_xyz = (target[:, 1:].numpy().astype(np.float64) + 0.5) / 256.0
    for k in range(4):
        guide_path = out / f"guide_{k}.pt"
        coverage_path = out / f"coverage_{k}.json"
        if guide_path.exists() and coverage_path.exists():
            cached = common.load_payload(guide_path)
            if torch.equal(cached["coords"], target):
                continue
        endpoint = common.load_payload(endpoint_dir / f"endpoint_{k}.pt")
        source_coords = endpoint["coords"].detach().cpu().int()
        source_features = endpoint["features"].detach().cpu().float()
        source_xyz = (source_coords[:, 1:].numpy().astype(np.float64) + 0.5) / 64.0
        tree = cKDTree(source_xyz)
        query_k = min(3, len(source_coords))
        distance, rows = tree.query(target_xyz, k=query_k, workers=-1)
        if query_k == 1:
            distance = distance[:, None]
            rows = rows[:, None]
        distance = np.asarray(distance, dtype=np.float64)
        rows = np.asarray(rows, dtype=np.int64)
        weights = 1.0 / np.maximum(distance, 1e-8)
        exact = distance <= 1e-8
        exact_rows = exact.any(axis=1)
        if exact_rows.any():
            weights[exact_rows] = 0.0
            weights[exact_rows, np.argmax(exact[exact_rows], axis=1)] = 1.0
        regular = ~exact_rows
        if regular.any():
            weights[regular] /= weights[regular].sum(axis=1, keepdims=True)
        features = (source_features.numpy()[rows] * weights[..., None]).sum(axis=1)
        features = torch.from_numpy(features.astype(np.float32, copy=False))
        common.save(
            guide_path,
            coords=target,
            features=features,
            normalized=True,
            baseline_step=k,
            query_method="native_c64_endpoint_nearest3_lift_v1",
            source_points=len(source_coords),
            mean_lattice_distance=float((distance * weights).sum(axis=1).mean()),
        )
        common.save(
            out / f"coverage_{k}.pt",
            valid=torch.ones(len(target), dtype=torch.bool),
            trilinear_valid=torch.ones(len(target), dtype=torch.bool),
        )
        common.js(
            coverage_path,
            dict(
                points=len(target),
                valid_points=len(target),
                valid_fraction=1.0,
                method="native_c64_endpoint_nearest3_lift_v1",
                query="three nearest native C64 endpoint latent points in world coordinates",
                weighting="inverse Euclidean distance",
                missing_policy="none; every fixed C256 point receives a lifted endpoint",
                source_points=len(source_coords),
                mean_lattice_distance=float((distance * weights).sum(axis=1).mean()),
            ),
        )


class _Nearest3Index:
    """CPU index for the sparse 1024 material field.

    The decoded material field is sparse, so a dense 1024^3 grid sample can have
    no support even when a nearby surface voxel has a perfectly good material.
    The index stores active voxel centres in the same continuous coordinate system
    used by the old sampler and returns the three closest active centres.
    """

    def __init__(self, coords, resolution=1024):
        import numpy as np
        from scipy.spatial import cKDTree

        coords = torch.as_tensor(coords).detach().cpu().int()
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(f"field coords must have shape [N,4], got {tuple(coords.shape)}")
        if not len(coords):
            raise ValueError("cannot query an empty sparse material field")
        if bool((coords[:, 0] != 0).any()):
            raise ValueError("material field coordinates must contain batch index zero")
        self.coords = coords
        self.resolution = int(resolution)
        centres = coords[:, 1:].numpy().astype(np.float64, copy=False) + 0.5
        self.tree = cKDTree(centres)

    def query(self, attrs, xyz, chunk=262144):
        import numpy as np

        attrs = torch.as_tensor(attrs).detach().cpu().float()
        if len(attrs) != len(self.coords):
            raise ValueError("field attributes and coordinates have different lengths")
        if not torch.isfinite(attrs).all():
            raise ValueError("material field contains non-finite attributes")
        xyz = torch.as_tensor(xyz).detach().cpu().float()
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"query points must have shape [N,3], got {tuple(xyz.shape)}")

        result = torch.empty((len(xyz), attrs.shape[1]), dtype=torch.float32)
        # This is the distance to the weighted support, rather than a binary
        # trilinear support flag.  It is persisted for diagnostics by encode_guides.
        distance = torch.empty(len(xyz), dtype=torch.float32)
        query_k = min(3, len(self.coords))
        attrs_np = attrs.numpy()
        for start in range(0, len(xyz), chunk):
            end = min(start + chunk, len(xyz))
            # xyz is in world coordinates [-.5,.5]; field centres are integer
            # lattice coordinates plus .5, exactly matching the old q transform.
            q = ((xyz[start:end].numpy().astype(np.float64, copy=False) + 0.5)
                 * self.resolution)
            dist, rows = self.tree.query(q, k=query_k, workers=-1)
            if query_k == 1:
                dist = dist[:, None]
                rows = rows[:, None]
            dist = np.asarray(dist, dtype=np.float64)
            rows = np.asarray(rows, dtype=np.int64)
            # Inverse-distance weights are well-defined at an exact field centre
            # by assigning all mass to the exact neighbour.
            exact = dist <= 1e-8
            weights = 1.0 / np.maximum(dist, 1e-8)
            exact_rows = exact.any(axis=1)
            if exact_rows.any():
                weights[exact_rows] = 0.0
                first = np.argmax(exact[exact_rows], axis=1)
                weights[exact_rows, first] = 1.0
            else:
                weights /= weights.sum(axis=1, keepdims=True)
            if exact_rows.any() and (~exact_rows).any():
                regular = ~exact_rows
                weights[regular] /= weights[regular].sum(axis=1, keepdims=True)
            values = (attrs_np[rows] * weights[..., None]).sum(axis=1)
            result[start:end] = torch.from_numpy(values.astype(np.float32, copy=False))
            distance[start:end] = torch.from_numpy(
                (dist * weights).sum(axis=1).astype(np.float32, copy=False)
            )
        # Every query has a finite fallback material value.  This intentionally
        # differs from the former zero-filled invalid-support path.
        valid = torch.ones(len(xyz), dtype=torch.bool)
        assert torch.isfinite(result).all() and torch.isfinite(distance).all()
        return result, valid, distance


class _InferenceSparseTrilinearIndex:
    """Match the sparse trilinear sampler used by the native renderer.

    ``inference.py`` ultimately renders a ``MeshWithVoxel`` through
    ``flex_gemm.ops.grid_sample.grid_sample_3d(..., mode='trilinear')``.
    That kernel looks up the eight lattice centres around each query, drops
    missing sparse entries, and accumulates the remaining weighted values
    without requiring all eight entries to exist.  Appending a constant-one
    channel lets us recover the kernel's support weight and identify the
    genuinely unsupported queries for the mesh-face fallback.
    """

    method = "inference_sparse_trilinear_v1"

    def __init__(self, coords, resolution=1024, device="cuda:0", chunk=1 << 20):
        from flex_gemm.ops.grid_sample import grid_sample_3d

        coords = torch.as_tensor(coords).detach().cpu().int()
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(
                f"field coords must have shape [N,4], got {tuple(coords.shape)}"
            )
        if not len(coords):
            raise ValueError("cannot query an empty sparse material field")
        if bool((coords[:, 0] != 0).any()):
            raise ValueError("material field coordinates must contain batch index zero")
        self.coords = coords
        self.resolution = int(resolution)
        self.device = torch.device(device)
        self.chunk = int(chunk)
        self._grid_sample_3d = grid_sample_3d
        self.coords_device = coords.to(self.device)
        self.ones_device = torch.ones(
            (len(coords), 1), dtype=torch.float32, device=self.device
        )
        self.shape = torch.Size([1, 1, self.resolution, self.resolution, self.resolution])

    def _sample_augmented(self, augmented, xyz):
        q = ((xyz + 0.5) * self.resolution).float().contiguous()
        sampled = self._grid_sample_3d(
            augmented,
            self.coords_device,
            torch.Size(
                [
                    1,
                    augmented.shape[1],
                    self.resolution,
                    self.resolution,
                    self.resolution,
                ]
            ),
            q.reshape(1, -1, 3),
            mode="trilinear",
        )[0]
        return sampled[:, :-1], sampled[:, -1]

    @torch.no_grad()
    def query(self, attrs, xyz, fallback=None):
        """Sample attrs and replace zero-support points through ``fallback``.

        ``xyz`` is in the normalized world coordinate system ``[-.5, .5]``.
        The returned ``trilinear_valid`` mask describes the native sparse
        trilinear query before fallback; ``valid`` includes successful face
        fallbacks as well.
        """
        attrs = torch.as_tensor(attrs).detach().cpu().float().contiguous()
        if len(attrs) != len(self.coords):
            raise ValueError("material field attributes and coordinates differ")
        if not torch.isfinite(attrs).all():
            raise ValueError("material field contains non-finite attributes")
        xyz = torch.as_tensor(xyz).detach().cpu().float().contiguous()
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"query points must have shape [N,3], got {tuple(xyz.shape)}")

        channels = attrs.shape[1]
        result = torch.empty((len(xyz), channels), dtype=torch.float32)
        trilinear_valid = torch.zeros(len(xyz), dtype=torch.bool)
        fallback_valid = torch.zeros(len(xyz), dtype=torch.bool)
        fallback_distance = torch.zeros(len(xyz), dtype=torch.float32)
        augmented = torch.cat(
            (attrs.to(self.device), self.ones_device), dim=1
        ).contiguous()
        for start in range(0, len(xyz), self.chunk):
            end = min(start + self.chunk, len(xyz))
            points = xyz[start:end].to(self.device)
            values, support = self._sample_augmented(augmented, points)
            native = support > 0
            trilinear_valid[start:end] = native.cpu()
            if fallback is not None and bool((~native).any()):
                bad = ~native
                face_values, face_valid, face_distance = fallback.query(points[bad])
                values[bad] = face_values
                bad_rows = torch.nonzero(bad, as_tuple=False).flatten().cpu()
                fallback_valid[start:end].index_copy_(0, bad_rows, face_valid.cpu())
                fallback_distance[start:end].index_copy_(
                    0, bad_rows, face_distance.cpu()
                )
            result[start:end] = values.float().cpu()
        valid = trilinear_valid | fallback_valid
        return result, valid, trilinear_valid, fallback_distance


class _NearestMeshFaceColor:
    """CUDA nearest-face lookup on the native 1024 textured mesh.

    The closest point on the selected 1024 triangle is sampled with the same
    sparse trilinear material sampler.  Thus the fallback is a real mesh-face
    color lookup, not another nearest-voxel interpolation.
    """

    def __init__(self, mesh_path, device="cuda:0"):
        import cumesh
        from flex_gemm.ops.grid_sample import grid_sample_3d

        mesh = common.load_payload(Path(mesh_path))["mesh"]
        self.device = torch.device(device)
        self._grid_sample_3d = grid_sample_3d
        vertices = mesh.vertices.detach().cpu().float().contiguous()
        faces = mesh.faces.detach().cpu().long().contiguous()
        if len(faces) <= 8:
            raise ValueError("native 1024 mesh must contain more than eight faces")
        self.bvh = cumesh.cuBVH(vertices, faces)
        self.vertices = vertices.to(self.device)
        self.faces = faces.to(self.device)
        self.attrs = mesh.attrs.detach().cpu().float().contiguous().to(self.device)
        coords = mesh.coords.detach().cpu().int()
        self.coords = torch.cat((torch.zeros_like(coords[:, :1]), coords), dim=1).to(
            self.device
        )
        self.resolution = int(round(1.0 / float(mesh.voxel_size)))
        self.shape = torch.Size(
            [1, self.attrs.shape[1], self.resolution, self.resolution, self.resolution]
        )
        self.origin = mesh.origin.detach().cpu().float().to(self.device)
        self.voxel_size = float(mesh.voxel_size)
        self.attrs_augmented = torch.cat(
            (self.attrs, torch.ones((len(self.attrs), 1), device=self.device)), dim=1
        ).contiguous()
        self.shape_augmented = torch.Size(
            [
                1,
                self.attrs_augmented.shape[1],
                self.resolution,
                self.resolution,
                self.resolution,
            ]
        )

    @torch.no_grad()
    def query(self, points):
        points = torch.as_tensor(points, device=self.device).float().contiguous()
        distance, face_id, uvw = self.bvh.unsigned_distance(
            points, return_uvw=True
        )
        face_id = face_id.long()
        tri = self.vertices[self.faces[face_id]]
        closest = (tri * uvw.unsqueeze(-1)).sum(dim=1)
        grid = ((closest - self.origin) / self.voxel_size).reshape(1, -1, 3)
        sampled = self._grid_sample_3d(
            self.attrs_augmented,
            self.coords,
            self.shape_augmented,
            grid,
            mode="trilinear",
        )[0]
        return sampled[:, :-1], sampled[:, -1] > 0, distance


@torch.no_grad()
def query_field(field, xyz, chunk=262144, index=None, return_distance=False):
    """Query the nearest three sparse field points with inverse-distance weights.

    ``xyz`` uses world coordinates.  The returned tensors stay on CPU, as did the
    old query helper, while the index is reusable for all four baseline endpoints.
    Set ``return_distance`` to also receive the weighted support distance in 1024
    lattice units.
    """
    if index is None:
        index = _Nearest3Index(field["coords"], field.get("resolution", 1024))
    if not torch.equal(index.coords, torch.as_tensor(field["coords"]).cpu().int()):
        raise ValueError("all endpoint fields must share the same sparse support")
    result, valid, distance = index.query(field["attrs"], xyz, chunk=chunk)
    if return_distance:
        return result, valid, distance
    return result, valid


@torch.no_grad()
def encode_guides(
    pipe,
    voxel_path,
    c256,
    fields,
    out,
    *,
    query_method="nearest3_inverse_distance_v1",
    fallback_mesh=None,
):
    """Encode four texture guides on the exact same 4096 sparse support.

    ``nearest3_inverse_distance_v1`` is retained for old experiments.  The
    inference-compatible method uses the native sparse trilinear sampler and
    sends only zero-support queries to the nearest face of the 1024 baseline
    textured mesh.
    """
    if query_method not in (
        "nearest3_inverse_distance_v1",
        _InferenceSparseTrilinearIndex.method,
    ):
        raise ValueError(f"unknown guide query method: {query_method}")
    if query_method == _InferenceSparseTrilinearIndex.method and fallback_mesh is None:
        raise ValueError("trilinear guide queries require fallback_mesh")
    from pixal3d import models

    out.mkdir(parents=True, exist_ok=True)
    vox = common.load_payload(voxel_path)
    coords = torch.cat((torch.zeros_like(vox["coords"][:, :1]), vox["coords"]), 1).int()
    query_index = None
    face_fallback = None
    for k in range(4):
        target = out / f"guide_{k}.pt"
        coverage_json = out / f"coverage_{k}.json"
        if target.exists() and coverage_json.exists():
            cached = common.load_payload(target)
            metadata = __import__("json").loads(coverage_json.read_text())
            if (
                torch.equal(cached["coords"], c256)
                and metadata.get("method") == query_method
                and cached.get("query_method") == query_method
            ):
                continue
        field = common.load_payload(fields / f"field_{k}.pt")
        if query_method == _InferenceSparseTrilinearIndex.method:
            if query_index is None:
                query_index = _InferenceSparseTrilinearIndex(
                    field["coords"],
                    field.get("resolution", 1024),
                    device=pipe.device,
                )
                face_fallback = _NearestMeshFaceColor(
                    fallback_mesh,
                    device=pipe.device,
                )
            attrs, valid, trilinear_valid, face_distance = query_index.query(
                field["attrs"],
                vox["dual"] - 0.5,
                fallback=face_fallback,
            )
            common.save(
                out / f"coverage_{k}.pt",
                valid=valid,
                trilinear_valid=trilinear_valid,
            )
            used_face = ~trilinear_valid
            face_distance_used = face_distance[used_face]
            common.js(
                out / f"coverage_{k}.json",
                dict(
                    points=len(valid),
                    valid_points=int(valid.sum()),
                    valid_fraction=float(valid.float().mean()),
                    trilinear_points=int(trilinear_valid.sum()),
                    trilinear_fraction=float(trilinear_valid.float().mean()),
                    face_fallback_points=int(used_face.sum()),
                    face_fallback_fraction=float(used_face.float().mean()),
                    method=query_method,
                    query="same sparse trilinear flow as inference.py/grid_sample_3d",
                    weighting="native eight-neighbour weights; missing neighbours contribute zero",
                    fallback="nearest point on 1024 baseline mesh face via cuBVH, then native mesh trilinear color query",
                    fallback_distance_mean=(
                        float(face_distance_used.mean()) if len(face_distance_used) else 0.0
                    ),
                    fallback_distance_p95=(
                        float(torch.quantile(face_distance_used[:: max(1, len(face_distance_used) // 100_000)], 0.95))
                        if len(face_distance_used)
                        else 0.0
                    ),
                    missing_policy="nearest 1024 mesh-face color; no zero-fill",
                ),
            )
        else:
            if query_index is None:
                query_index = _Nearest3Index(field["coords"], field.get("resolution", 1024))
            attrs, valid, distance = query_field(
                field,
                vox["dual"] - 0.5,
                index=query_index,
                return_distance=True,
            )
            common.save(out / f"coverage_{k}.pt", valid=valid)
            # torch.quantile has a hard size limit on some builds.  A deterministic
            # bounded sample is sufficient for the diagnostic percentile and keeps
            # the full-resolution query result untouched.
            p95_sample = distance[:: max(1, len(distance) // 100_000)]
            common.js(
                out / f"coverage_{k}.json",
                dict(
                    points=len(valid),
                    valid_points=int(valid.sum()),
                    valid_fraction=float(valid.float().mean()),
                    method=query_method,
                    query="three nearest active 1024-field voxel centres",
                    weighting="inverse Euclidean distance in 1024 lattice units",
                    distance_mean=float(distance.mean()),
                    distance_p95=float(torch.quantile(p95_sample, 0.95)),
                    missing_policy="nearest sparse material fallback; never zero-fill",
                ),
            )
        del field
        common.empty_cuda()
        encoder = models.from_pretrained(str(TEX_ENCODER)).eval().cuda()
        encoded = encoder(
            common.SparseTensor((attrs * 2 - 1).cuda(), coords.cuda()),
            sample_posterior=False,
        )
        rows, found = lookup_rows(encoded.coords.cpu(), c256, 256)
        # A geometry flow can retain a few C256 support points that are not
        # emitted by the bridge encoder after voxelization.  Those points are
        # deliberately marked unsupported and receive no endpoint guidance;
        # feeding a zero feature into the flow would turn absence into black
        # material.  Exact support remains the common path.
        raw = torch.zeros(
            len(c256), encoded.feats.shape[1], dtype=torch.float32
        )
        raw[found] = encoded.feats.float().cpu()[rows[found]]
        normalized = rendering.norm(pipe, "tex", raw)
        assert torch.isfinite(normalized).all()
        common.save(
            target,
            coords=c256,
            features=normalized,
            normalized=True,
            baseline_step=k,
            endpoint_t=common.load_payload(fields / f"endpoint_{k}.pt")["t"],
            support_fraction=float(found.float().mean()),
            query_method=query_method,
        )
        del encoder, encoded, raw, normalized, attrs
        common.empty_cuda()


def routed_predictions(pipe, model, group, x, shape, bank, t, params, step, n, guide):
    """Separate full-context forwards, then select velocities per point."""
    result = {}
    if any("visible_mask" not in b for b in group):
        raise ValueError("mapping lacks point visibility; rerun classify_blocks")
    high = [b for b in group if b["conditional"]]
    low = [b for b in group if not b["conditional"]]
    if high:
        values = texture.predict_safe(pipe, model, high, x, shape, bank, t, params)
        result.update((b["block_id"], v) for b, v in zip(high, values))
    if low:
        mixed = [b for b in low if bool(b["visible_mask"].any())]
        conditional = {}
        if mixed:
            values = texture.predict_safe(pipe, model, mixed, x, shape, bank, t, params)
            conditional = {b["block_id"]: v for b, v in zip(mixed, values)}
        uncond_blocks = mixed if step < n else low
        unconditional = {}
        if uncond_blocks:
            values = texture.predict_safe(
                pipe, model, uncond_blocks, x, shape, bank, t, params, unconditional=True
            )
            unconditional = {b["block_id"]: v for b, v in zip(uncond_blocks, values)}
        if step < n:
            for b in low:
                ids = b["global_ids"]
                result[b["block_id"]] = pipe.tex_slat_sampler._xstart_to_pred(
                    x[ids], t, guide[ids]
                )
        else:
            result.update(unconditional)
        for b in mixed:
            mask = b["visible_mask"]
            result[b["block_id"]][mask] = conditional[b["block_id"]][mask]
    return [result[b["block_id"]] for b in group]


@torch.no_grad()
def flow(pipe, data, shape, bank, guides, out, args, n):
    out.mkdir(parents=True, exist_ok=True)
    model = pipe.models["tex_slat_flow_model_1024"]
    sampler = pipe.tex_slat_sampler
    params = dict(pipe.tex_slat_sampler_params)
    params.pop("steps", None)
    times = sampler.timestep_schedule(12, params.pop("rescale_t", 1.0))
    x = torch.randn(
        (len(data["coords"]), model.in_channels - shape.shape[1]),
        generator=torch.Generator().manual_seed(46),
    )
    identity = dict(
        n=n,
        shape_hash=common.tensor_hash(shape),
        routing="block_threshold_point_selection_v2",
        visibility=[b["conditional"] for b in data["blocks"]],
        times=times,
    )
    manifest = out / "sampler.json"
    if manifest.exists():
        assert __import__("json").loads(manifest.read_text()) == identity
    common.js(manifest, identity)
    groups = list(sync.groups(data["blocks"], args.batch_size, args.max_tokens))
    model.cuda()
    try:
        for step, (t, tn) in enumerate(zip(times, times[1:])):
            target = out / f"step_{step:02d}.pt"
            before = common.tensor_hash(x)
            if target.exists():
                p = common.load_payload(target)
                assert (
                    p["input_hash"] == before
                    and p["n"] == n
                    and torch.equal(p["coords"], data["coords"])
                )
                x = p["features"]
                continue
            guide = None
            if step < n:
                p = common.load_payload(guides / f"guide_{step}.pt")
                assert (
                    torch.equal(p["coords"], data["coords"])
                    and abs(p["endpoint_t"] - t) < 1e-10
                )
                guide = p["features"]
            total = torch.zeros_like(x)
            counts = torch.zeros(len(x), dtype=torch.int32)
            for group in groups:
                values = routed_predictions(
                    pipe, model, group, x, shape, bank, t, params, step, n, guide
                )
                sync.reduce_predictions(total, counts, group, values)
            assert torch.equal(counts, data["counts"]) and (counts > 0).all()
            assert before == common.tensor_hash(x)
            x = x - (t - tn) * total / counts[:, None]
            assert torch.isfinite(x).all()
            common.save(
                target,
                coords=data["coords"],
                features=x,
                input_hash=before,
                n=n,
                t=t,
                t_next=tn,
            )
            print("VISIBILITY TEXTURE", n, step + 1, "/12", flush=True)
    finally:
        model.cpu()
        common.empty_cuda()
    common.save(out / "endpoint.pt", coords=data["coords"], features=x, normalized=True)
    return x
