#!/usr/bin/env python3
"""Protect exactly the 4096 material voxels queried by the front render.

For every foreground pixel, rasterize the final mesh position, enumerate its
eight trilinear lattice corners, and retain the full-conditional material at
those sparse field entries.  All other material entries come from the hidden
candidate.  This avoids both C256 block overreach and decoder-output boundary
guessing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_gpu_arg = "4"
if "--gpu" in sys.argv:
    _gpu_arg = sys.argv[sys.argv.index("--gpu") + 1]
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_arg

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", default="4")
    p.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/sr_0_img_trilinear_face_n3_20260919/fixed"),
    )
    p.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("outputs/sr_point_visibility_20260919"),
    )
    p.add_argument("--base-variant", default="seed_48_n_0")
    p.add_argument("--candidate-variant", default="seed_48_n_4_vctx4")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--face-chunk", type=int, default=2_000_000)
    p.add_argument(
        "--appearance-gate",
        action="store_true",
        help="diagnostic: choose the endpoint closer to the input per front query",
    )
    return p.parse_args()


@torch.no_grad()
def raster_front_positions(mesh, camera, resolution, face_chunk, device):
    import nvdiffrast.torch as dr
    from pixal3d.renderers.mesh_renderer import intrinsics_to_projection
    from pixal3d.utils.render_utils import proj_camera_to_render_params

    ext, intr = proj_camera_to_render_params(
        camera["camera_angle_x"], camera["distance"]
    )
    vertices = mesh.vertices.to(device=device, dtype=torch.float32)
    homogeneous = torch.cat(
        (vertices, torch.ones_like(vertices[:, :1])), dim=1
    )
    clip = (
        homogeneous
        @ (intrinsics_to_projection(intr.to(device), 0.01, 100) @ ext.to(device)).T
    )[None].contiguous()
    context = dr.RasterizeCudaContext(device=device)
    depth = torch.full(
        (resolution, resolution), float("inf"), device=device, dtype=torch.float32
    )
    positions = torch.zeros(
        (resolution, resolution, 3), device=device, dtype=torch.float32
    )
    hit_any = torch.zeros((resolution, resolution), device=device, dtype=torch.bool)
    for start in range(0, len(mesh.faces), face_chunk):
        faces = mesh.faces[start : start + face_chunk].to(device=device).int().contiguous()
        rast, _ = dr.rasterize(context, clip, faces, (resolution, resolution))
        hit = rast[0, ..., 3] > 0
        closer = hit & (rast[0, ..., 2] < depth)
        if bool(closer.any()):
            pos = dr.interpolate(vertices[None], rast, faces)[0][0]
            depth[closer] = rast[0, ..., 2][closer]
            positions[closer] = pos[closer]
            hit_any[closer] = True
            del pos
        del faces, rast, hit, closer
    pixel_ids = hit_any.flatten().nonzero(as_tuple=False).flatten().cpu()
    out = positions[hit_any].cpu()
    del vertices, homogeneous, clip, context, depth, positions, hit_any
    torch.cuda.empty_cache()
    return out, pixel_ids


def field_query_support(coords, positions, resolution):
    """Return material-field rows touched by all visible pixel trilinear queries."""
    del resolution  # the query list already comes from the requested raster size
    xyz = coords[:, 1:].long()
    keys = ((xyz[:, 0].long() * 4096 + xyz[:, 1]) * 4096 + xyz[:, 2]).contiguous()
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    # flex_gemm's kernel uses base = floor(q - 0.5), because integer sparse
    # coordinates denote voxel centers rather than cell corners.
    grid = (((positions + 0.5) * 4096.0) - 0.5).floor().long()
    corner_offsets = torch.tensor(
        [
            (dx, dy, dz)
            for dx in (0, 1)
            for dy in (0, 1)
            for dz in (0, 1)
        ],
        dtype=torch.long,
    )
    touched = torch.zeros(len(coords), dtype=torch.bool)
    chunk = 250_000
    for start in range(0, len(grid), chunk):
        q = grid[start : start + chunk, None, :] + corner_offsets[None]
        valid = (q >= 0).all(dim=-1) & (q < 4096).all(dim=-1)
        qkeys = ((q[..., 0] * 4096 + q[..., 1]) * 4096 + q[..., 2]).reshape(-1)
        valid = valid.reshape(-1)
        pos = torch.searchsorted(sorted_keys, qkeys.clamp_min(0))
        in_range = pos < len(sorted_keys)
        pos = pos.clamp_max(len(sorted_keys) - 1)
        found = valid & in_range & (sorted_keys[pos] == qkeys)
        if bool(found.any()):
            touched[order[pos[found]]] = True
    return touched


def field_query_scores(coords, positions, pixel_scores):
    """Accumulate front-pixel candidate-vs-base gains onto sparse field rows."""
    xyz = coords[:, 1:].long()
    keys = ((xyz[:, 0].long() * 4096 + xyz[:, 1]) * 4096 + xyz[:, 2]).contiguous()
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    query = ((positions + 0.5) * 4096.0).float()
    base = torch.floor(query - 0.5).long()
    offsets = torch.tensor(
        [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)],
        dtype=torch.long,
    )
    score = torch.zeros(len(coords), dtype=torch.float64)
    weight_sum = torch.zeros(len(coords), dtype=torch.float64)
    chunk = 250_000
    for start in range(0, len(query), chunk):
        q = query[start : start + chunk]
        b = base[start : start + chunk]
        corners = b[:, None, :] + offsets[None]
        valid = (corners >= 0).all(dim=-1) & (corners < 4096).all(dim=-1)
        qkeys = (
            (corners[..., 0] * 4096 + corners[..., 1]) * 4096 + corners[..., 2]
        ).reshape(-1)
        valid_flat = valid.reshape(-1)
        pos = torch.searchsorted(sorted_keys, qkeys.clamp_min(0))
        in_range = pos < len(sorted_keys)
        pos = pos.clamp_max(len(sorted_keys) - 1)
        found = valid_flat & in_range & (sorted_keys[pos] == qkeys)
        found = found.reshape(-1, 8)
        rows = order[pos].reshape(-1, 8)
        weights = torch.stack(
            [
                1.0 - (q[:, 0] - (b[:, 0] + offsets[i, 0]) - 0.5).abs()
                for i in range(8)
            ],
            dim=1,
        )
        weights = weights * torch.stack(
            [
                1.0 - (q[:, 1] - (b[:, 1] + offsets[i, 1]) - 0.5).abs()
                for i in range(8)
            ],
            dim=1,
        )
        weights = weights * torch.stack(
            [
                1.0 - (q[:, 2] - (b[:, 2] + offsets[i, 2]) - 0.5).abs()
                for i in range(8)
            ],
            dim=1,
        )
        weights = weights * found.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        flat_rows = rows[found]
        flat_weights = weights[found]
        flat_scores = pixel_scores[start : start + len(q), None].expand(-1, 8)[found]
        score.index_add_(0, flat_rows, flat_scores.double() * flat_weights.double())
        weight_sum.index_add_(0, flat_rows, flat_weights.double())
    return score, weight_sum


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    sys.argv = ["sr_point_visibility_texture.py", "--gpu", str(args.gpu)]
    import sr_point_visibility_texture as exp

    from pixal3d.representations.mesh import MeshWithVoxel

    exp.ARGS.render_resolution = args.resolution
    root = args.root.resolve()
    experiment_dir = args.experiment_dir.resolve()
    output = experiment_dir / (
        f"pixel_{'gate_' if args.appearance_gate else ''}merge_"
        f"{args.base_variant}_with_{args.candidate_variant}"
    )
    output.mkdir(parents=True, exist_ok=True)
    pipe = exp.common.setup(experiment_dir)
    shape_payload = exp.common.load_payload(root / "shared/shape1024/endpoint.pt")
    base = exp.common.load_payload(
        experiment_dir / args.base_variant / "texture/endpoint.pt"
    )
    candidate = exp.common.load_payload(
        experiment_dir / args.candidate_variant / "texture/endpoint.pt"
    )
    topology = exp.common.load_payload(root / "shared/final_mesh/topology.pt")
    geometry = exp.common.load_payload(
        root / "shared/final_mesh/geometry_mesh.pt"
    )["mesh"]
    if not exp.torch.equal(base["coords"], candidate["coords"]):
        raise RuntimeError("base and candidate C256 supports differ")

    @exp.torch.no_grad()
    def decode_field(features):
        tex = exp.common.SparseTensor(
            exp.rendering.norm(
                pipe, "tex", features.to(pipe.device), inverse=True
            ),
            base["coords"].to(pipe.device),
        )
        started = time.perf_counter()
        field = pipe.decode_tex_slat(
            tex, [sub.to(pipe.device) for sub in topology["subs"]]
        )
        elapsed = time.perf_counter() - started
        coords = field.coords.detach().cpu().int()
        attrs = field.feats.detach().cpu().float()
        voxel_shape = exp.torch.Size([*field.shape, *field.spatial_shape])
        del field, tex
        exp.common.empty_cuda()
        return coords, attrs, voxel_shape, elapsed

    base_coords, base_attrs, voxel_shape, base_decode_seconds = decode_field(
        base["features"]
    )
    cand_coords, cand_attrs, cand_shape, candidate_decode_seconds = decode_field(
        candidate["features"]
    )
    if not exp.torch.equal(base_coords, cand_coords):
        raise RuntimeError("decoded material supports differ")
    if voxel_shape != cand_shape:
        raise RuntimeError("decoded material shapes differ")

    import json

    camera = json.loads((root / "baseline/camera.json").read_text())
    positions, pixel_ids = raster_front_positions(
        geometry,
        camera,
        args.resolution,
        args.face_chunk,
        pipe.device,
    )
    if args.appearance_gate:
        from PIL import Image
        import numpy as np

        reference_np = np.asarray(
            Image.open(root / "baseline/image_4096.png")
            .convert("RGB")
            .resize((args.resolution, args.resolution), Image.Resampling.LANCZOS),
            dtype=np.float32,
        ) / 255.0
        base_np = np.asarray(
            Image.open(
                experiment_dir
                / args.base_variant
                / f"render/views_{args.resolution}/base_color_000.png"
            ).convert("RGB"),
            dtype=np.float32,
        ) / 255.0
        candidate_np = np.asarray(
            Image.open(
                experiment_dir
                / args.candidate_variant
                / f"render/views_{args.resolution}/base_color_000.png"
            ).convert("RGB"),
            dtype=np.float32,
        ) / 255.0
        flat_ref = torch.from_numpy(reference_np.reshape(-1, 3))
        flat_base = torch.from_numpy(base_np.reshape(-1, 3))
        flat_candidate = torch.from_numpy(candidate_np.reshape(-1, 3))
        pixel_scores = (
            (flat_base[pixel_ids] - flat_ref[pixel_ids]).square().mean(dim=1)
            - (flat_candidate[pixel_ids] - flat_ref[pixel_ids])
            .square()
            .mean(dim=1)
        )
        score, weight_sum = field_query_scores(
            base_coords, positions, pixel_scores
        )
        touched = weight_sum > 0
        use_candidate = score > 0
        merged = cand_attrs.clone()
        keep_base = touched & ~use_candidate
        merged[keep_base] = base_attrs[keep_base]
        print(
            f"[appearance gate] candidate_rows={int(use_candidate.sum()):,}/"
            f"{int(touched.sum()):,} touched; score_sum={float(score.sum()):.6f}",
            flush=True,
        )
    else:
        touched = field_query_support(base_coords, positions, args.resolution)
        use_candidate = ~touched
        merged = cand_attrs.clone()
        merged[touched] = base_attrs[touched]
    print(
        f"[support] pixels={len(positions):,} field_rows={int(touched.sum()):,}/"
        f"{len(touched):,}",
        flush=True,
    )
    exp.common.save(
        output / "query_support.pt",
        positions=positions,
        touched=touched,
        use_candidate=use_candidate,
        pixel_ids=pixel_ids,
    )

    mesh = MeshWithVoxel(
        geometry.vertices,
        geometry.faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1 / 4096,
        coords=base_coords[:, 1:],
        attrs=merged,
        voxel_shape=voxel_shape,
        layout=pipe.pbr_attr_layout,
    )
    baseline_views = exp.ensure_baseline_render(pipe, root, experiment_dir)
    from PIL import Image

    reference = Image.open(root / "baseline/image_4096.png").convert("RGB").resize(
        (args.resolution, args.resolution), Image.Resampling.LANCZOS
    )
    mask = Image.open(root / "baseline/foreground_mask_4096.png").resize(
        (args.resolution, args.resolution), Image.Resampling.LANCZOS
    )
    import lpips

    lpips_model = lpips.LPIPS(
        net="alex", version="0.1", pretrained=True, pnet_rand=False, spatial=True
    ).eval().to(pipe.device)
    report = exp.render_and_measure(
        pipe,
        mesh,
        root,
        output,
        baseline_views,
        {"reference": reference, "mask": mask},
        lpips_model,
    )
    exp.save_json(
        output / "result.json",
        dict(
            variant=output.name,
            base_variant=args.base_variant,
            candidate_variant=args.candidate_variant,
            visible_query_pixels=len(positions),
            touched_field_rows=int(touched.sum()),
            total_field_rows=len(touched),
            touched_fraction=float(touched.float().mean()),
            base_decode_seconds=base_decode_seconds,
            candidate_decode_seconds=candidate_decode_seconds,
            metrics=report,
        ),
    )
    print(exp.json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
