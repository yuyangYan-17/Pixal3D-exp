#!/usr/bin/env python3
"""Visibility-aware material-field merge diagnostic.

Decode a full-conditional texture endpoint and a visibility-routed endpoint on
the same fixed geometry.  At the decoded 4096 material field, retain the
full-conditional material for C256 parents marked visible and the routed
material elsewhere.  This isolates decoder cross-talk from flow routing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch as exp_torch


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
    p.add_argument("--dilations", default="0")
    p.add_argument(
        "--visibility-mode",
        choices=("parent", "exact", "final", "final_grid"),
        default="parent",
        help="use C256-parent, bridge-4096, final-mesh, or grid-dilated final visibility",
    )
    return p.parse_args()


def sparse_grid_dilate(mask, coords, radius, grid=4096):
    """Chebyshev dilation on a sparse 3-D integer support.

    The renderer's trilinear query can use all 8 neighboring lattice corners,
    so a one-step dilation includes the 26 neighbors, not only mesh-adjacent
    vertices.  Queries are chunked to keep peak CPU memory bounded.
    """
    radius = int(radius)
    if radius <= 0:
        return mask.clone()
    xyz = coords[:, 1:].to(dtype=exp_torch.int64)
    keys = ((xyz[:, 0] * grid + xyz[:, 1]) * grid + xyz[:, 2]).contiguous()
    order = exp_torch.argsort(keys)
    sorted_keys = keys[order]
    active = mask.clone()
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    chunk = 1_000_000
    for _ in range(radius):
        rows = exp_torch.nonzero(active, as_tuple=False).flatten()
        expanded = active.clone()
        for dx, dy, dz in offsets:
            delta = exp_torch.tensor((dx, dy, dz), dtype=xyz.dtype)
            for start in range(0, len(rows), chunk):
                active_rows = rows[start : start + chunk]
                query_xyz = xyz[active_rows] + delta
                valid = (query_xyz >= 0).all(dim=1) & (query_xyz < grid).all(dim=1)
                if not bool(valid.any()):
                    continue
                query = ((query_xyz[:, 0] * grid + query_xyz[:, 1]) * grid + query_xyz[:, 2])
                positions = exp_torch.searchsorted(sorted_keys, query.clamp_min(0))
                in_range = positions < len(sorted_keys)
                positions = positions.clamp_max(len(sorted_keys) - 1)
                hit = valid & in_range & (sorted_keys[positions] == query)
                if bool(hit.any()):
                    expanded[order[positions[hit]]] = True
        active = expanded
    return active


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    sys.argv = ["sr_point_visibility_texture.py", "--gpu", str(args.gpu)]
    import sr_point_visibility_texture as exp

    from pixal3d.representations.mesh import MeshWithVoxel
    from sr_tools.visibility import lookup_rows

    exp.ARGS.render_resolution = args.resolution
    root = args.root.resolve()
    experiment_dir = args.experiment_dir.resolve()
    output = experiment_dir / (
        f"material_merge_{args.base_variant}_with_{args.candidate_variant}"
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
    if not exp.torch.equal(base["coords"], candidate["coords"]):
        raise RuntimeError("base and candidate C256 supports differ")

    @exp.torch.no_grad()
    def decode_field(features):
        topology = exp.common.load_payload(root / "shared/final_mesh/topology.pt")
        tex = exp.common.SparseTensor(
            exp.rendering.norm(
                pipe,
                "tex",
                features.to(pipe.device),
                inverse=True,
            ),
            base["coords"].to(pipe.device),
        )
        started = time.perf_counter()
        field = pipe.decode_tex_slat(
            tex,
            [sub.to(pipe.device) for sub in topology["subs"]],
        )
        elapsed = time.perf_counter() - started
        coords = field.coords.detach().cpu().int()
        attrs = field.feats.detach().cpu().float()
        voxel_shape = exp.torch.Size([*field.shape, *field.spatial_shape])
        if not exp.torch.isfinite(attrs).all():
            raise RuntimeError("decoded material contains non-finite values")
        del field, tex, topology
        exp.common.empty_cuda()
        return coords, attrs, voxel_shape, elapsed

    base_coords, base_attrs, voxel_shape, base_decode_seconds = decode_field(
        base["features"]
    )
    cand_coords, cand_attrs, cand_shape, cand_decode_seconds = decode_field(
        candidate["features"]
    )
    if not exp.torch.equal(base_coords, cand_coords):
        raise RuntimeError("decoded material supports differ")
    if voxel_shape != cand_shape or base_attrs.shape != cand_attrs.shape:
        raise RuntimeError("decoded material layouts differ")

    encoded = exp.common.load_payload(root / "shared/bridge/encoded.pt")
    parents = base_coords.clone()
    parents[:, 1:] //= 16
    rows, found = lookup_rows(base["coords"], parents, 256)
    exact_visible = None
    if args.visibility_mode == "exact":
        bridge_voxels = exp.common.load_payload(root / "shared/bridge/voxels.pt")
        ancestry = exp.common.load_payload(root / "shared/bridge/ancestry.pt")
        field_rows, field_found = lookup_rows(
            bridge_voxels["coords"], base_coords[:, 1:], 4096
        )
        if not bool(field_found.all()):
            raise RuntimeError(
                "4096 material field has coordinates outside bridge voxel support: "
                f"missing={int((~field_found).sum())}"
            )
        exact_visible = ancestry["voxel_visible"][field_rows]
        del bridge_voxels, ancestry, field_rows, field_found
    dilations = [int(v.strip()) for v in args.dilations.split(",") if v.strip()]
    if any(v < 0 for v in dilations):
        raise ValueError("dilations must be non-negative")

    geometry = exp.common.load_payload(
        root / "shared/final_mesh/geometry_mesh.pt"
    )["mesh"]
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
    for dilation in dilations:
        if args.visibility_mode == "parent":
            visible_c256 = exp.dilate_visible_mask(
                encoded["visible"].bool(), base["coords"], dilation
            )
            visible_field = found & visible_c256[rows]
        else:
            if args.visibility_mode in ("final", "final_grid"):
                final_path = (
                    root
                    / "shared/final_mesh/final_visibility"
                    / f"visible_vertices_d{0 if args.visibility_mode == 'final_grid' else dilation}.pt"
                )
                if not final_path.exists():
                    raise FileNotFoundError(final_path)
                final_visibility = exp.common.load_payload(final_path)
                visible_field = final_visibility["visible_vertices"].bool()
                if len(visible_field) != len(base_coords):
                    raise RuntimeError(
                        "final-mesh visibility length does not match decoded material: "
                        f"{len(visible_field)} != {len(base_coords)}"
                    )
                del final_visibility
                if args.visibility_mode == "final_grid" and dilation:
                    visible_field = sparse_grid_dilate(
                        visible_field, base_coords, dilation
                    )
                continue_visibility = True
            else:
                continue_visibility = False
            if args.visibility_mode == "exact" and dilation:
                raise ValueError(
                    "exact/final visibility currently supports only dilation=0; "
                    "dilate the 4096 mask only after this mapping is validated"
                )
            if not continue_visibility:
                visible_field = exact_visible
        merged = cand_attrs.clone()
        merged[visible_field] = base_attrs[visible_field]
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
        target = output / f"dilation_{dilation}"
        report = exp.render_and_measure(
            pipe,
            mesh,
            root,
            target,
            baseline_views,
            {"reference": reference, "mask": mask},
            lpips_model,
        )
        exp.save_json(
            target / "result.json",
            dict(
                variant=target.name,
                base_variant=args.base_variant,
                candidate_variant=args.candidate_variant,
                dilation=dilation,
                visibility_mode=args.visibility_mode,
                decoded_material_voxels=len(base_coords),
                visible_parent_field_voxels=int(visible_field.sum()),
                visible_parent_fraction=float(visible_field.float().mean()),
                base_decode_seconds=base_decode_seconds,
                candidate_decode_seconds=cand_decode_seconds,
                metrics=report,
            ),
        )
        print(
            exp.json.dumps(
                dict(dilation=dilation, visible_fraction=float(visible_field.float().mean()), metrics=report),
                ensure_ascii=False,
            ),
            flush=True,
        )
        del mesh, merged
        exp.common.empty_cuda()


if __name__ == "__main__":
    main()
