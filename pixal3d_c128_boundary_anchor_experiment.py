#!/usr/bin/env python3
"""Test baseline boundary anchors for a fresh block-local C128 geometry.

The local Shape512/Shape1024 route is allowed to change the interior support.
Here, the fresh local-flow state is augmented only near the three internal
C64 split planes with the original baseline C128 endpoint support and raw
features.  The augmented state is decoded once by the global C128 decoder.
This isolates whether a small amount of cross-block low-frequency support is
enough to remove the seams without replacing the local high-frequency result.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import numpy as np
import torch
from PIL import Image

from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import Mesh

from pixal3d_c128_block_local_cascade_shape_renoise_2048 import (
    DEFAULT_BASELINE,
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL_PATH,
    empty_cuda,
    normal_metrics,
)
from pixal3d_c128_local_decode_merge_experiment import (
    atomic_json,
    atomic_save,
    load_decoder_pipeline,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_FRESH_STATE = (
    ROOT
    / "outputs/c128_block_local_cascade_shape_renoise_2048_fresh_n8"
    / "conditional/n_08/shape_c64_final_denormalized.pt"
)
DEFAULT_OUTPUT = ROOT / "outputs/c128_boundary_anchor_fresh_n8_cuda4"
GLOBAL_GRID = 128


def load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def key(coords: torch.Tensor) -> torch.Tensor:
    xyz = coords[:, 1:].long() if coords.shape[1] == 4 else coords.long()
    return (xyz[:, 0] * GLOBAL_GRID + xyz[:, 1]) * GLOBAL_GRID + xyz[:, 2]


def boundary_mask(xyz: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return torch.zeros(xyz.shape[0], dtype=torch.bool)
    return (
        ((xyz[:, 0] - 64).abs() < width)
        | ((xyz[:, 1] - 64).abs() < width)
        | ((xyz[:, 2] - 64).abs() < width)
    )


def make_augmented_state(
    fresh_coords: torch.Tensor,
    fresh_features: torch.Tensor,
    baseline_coords: torch.Tensor,
    baseline_features: torch.Tensor,
    width: int,
    feature_policy: str,
    feature_width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Union fresh support with baseline points in an internal boundary shell.

    ``width`` controls which baseline coordinates are added to the decoder
    support.  ``feature_width`` independently controls where common
    coordinates take baseline features.  This lets the decoder see a wide
    continuous support halo while retaining fresh features farther from the
    actual split plane.
    """
    if feature_width is None:
        feature_width = width
    if int(feature_width) < 0 or int(feature_width) > 32:
        raise ValueError("feature_width must be in 0..32")
    if int(feature_width) > int(width):
        raise ValueError("feature_width cannot exceed support width")
    fresh_keys = key(fresh_coords)
    baseline_keys = key(baseline_coords)
    fresh_map = {int(k): i for i, k in enumerate(fresh_keys.tolist())}
    baseline_map = {int(k): i for i, k in enumerate(baseline_keys.tolist())}
    selected_baseline = boundary_mask(baseline_coords[:, 1:], int(width))
    selected_features = boundary_mask(baseline_coords[:, 1:], int(feature_width))
    selected_keys = {
        int(k)
        for k, selected in zip(baseline_keys.tolist(), selected_baseline.tolist())
        if selected
    }
    union_keys = set(fresh_map) | selected_keys
    ordered = sorted(union_keys)
    out_coords: list[torch.Tensor] = []
    out_features: list[torch.Tensor] = []
    fresh_count = 0
    anchor_count = 0
    replaced_count = 0
    for k in ordered:
        baseline_row = baseline_map.get(k)
        fresh_row = fresh_map.get(k)
        if fresh_row is not None:
            out_coords.append(fresh_coords[fresh_row])
            fresh_value = fresh_features[fresh_row]
            if (
                baseline_row is not None
                and bool(selected_features[baseline_row])
                and feature_policy == "blend"
            ):
                out_features.append((fresh_value + baseline_features[baseline_row]) * 0.5)
                replaced_count += 1
            elif (
                baseline_row is not None
                and bool(selected_features[baseline_row])
                and feature_policy == "baseline"
            ):
                out_features.append(baseline_features[baseline_row])
                replaced_count += 1
            else:
                out_features.append(fresh_value)
            fresh_count += 1
        else:
            if baseline_row is None:
                raise RuntimeError("boundary selection produced an unknown key")
            out_coords.append(baseline_coords[baseline_row])
            out_features.append(baseline_features[baseline_row])
            anchor_count += 1
    coords = torch.stack(out_coords, 0).int()
    features = torch.stack(out_features, 0).float()
    if torch.unique(key(coords)).numel() != coords.shape[0]:
        raise RuntimeError("augmented support contains duplicate coordinates")
    metadata = {
        "boundary_width": int(width),
        "feature_width": int(feature_width),
        "feature_policy": feature_policy,
        "fresh_tokens": int(fresh_coords.shape[0]),
        "baseline_tokens": int(baseline_coords.shape[0]),
        "augmented_tokens": int(coords.shape[0]),
        "new_baseline_anchor_tokens": int(anchor_count),
        "fresh_tokens_retained": int(fresh_count),
        "common_boundary_features_replaced_or_blended": int(replaced_count),
    }
    return coords, features, metadata


@torch.no_grad()
def decode_global(
    pipeline: Any,
    coords: torch.Tensor,
    raw_features: torch.Tensor,
    resolution: int,
    device: torch.device,
) -> Mesh:
    slat = SparseTensor(raw_features.to(device), coords.to(device))
    meshes = None
    substructures = None
    mesh = None
    try:
        meshes, substructures = pipeline.decode_shape_slat(slat, int(resolution))
        if len(meshes) != 1:
            raise RuntimeError(f"decoder returned {len(meshes)} meshes")
        mesh = meshes[0]
        result = Mesh(mesh.vertices.detach().cpu(), mesh.faces.detach().cpu())
        return result
    finally:
        del meshes, substructures, mesh, slat
        empty_cuda()


def render(
    mesh: Mesh,
    camera: Mapping[str, Any],
    output: Path,
    device: torch.device,
    reference: Path,
    mask: Path,
    angles: Sequence[int],
    render_resolution: int,
    render_chunk_size: int,
) -> dict[str, Any]:
    from pixal3d_baseline1024_c128_8xc64_geometry import render_geometry

    live = mesh.to(device)
    try:
        result = render_geometry(
            live,
            camera,
            reference,
            mask,
            output,
            int(render_resolution),
            tuple(int(v) % 360 for v in angles),
            int(render_chunk_size),
        )
    finally:
        del live
        empty_cuda()
    front = output / f"multiview_{render_resolution}/view_000_camera_normal.png"
    normal_ref = DEFAULT_BASELINE / "raw_ovoxel_render/normal.png"
    normal_mask = DEFAULT_BASELINE / "raw_ovoxel_render/alpha.png"
    normal_reference = {
        "kind": "raw_ovoxel_baseline_render_proxy",
        "is_ground_truth": False,
        "image": str(normal_ref.resolve()),
        "mask": str(normal_mask.resolve()),
        "comparison": "RGB encoded normal, resized to 1024, pixel MSE/PSNR/SSIM/MAE",
    }
    normal_reference_metrics = normal_metrics(front, normal_ref, normal_mask)
    return {
        **result,
        "normal_reference": normal_reference,
        "normal_reference_metrics": normal_reference_metrics,
        # Kept for readers of earlier experiment records.
        "baseline_normal_metrics": normal_reference_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-state", type=Path, default=DEFAULT_FRESH_STATE)
    parser.add_argument("--endpoint-dir", type=Path, default=DEFAULT_ENDPOINT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--camera", type=Path, default=DEFAULT_BASELINE / "global_camera.json")
    parser.add_argument("--reference", type=Path, default=DEFAULT_BASELINE / "canonical_1024.png")
    parser.add_argument("--reference-mask", type=Path, default=DEFAULT_BASELINE / "raw_ovoxel_render/alpha.png")
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--resolution", type=int, choices=(1024, 2048), default=2048)
    parser.add_argument("--widths", default="0,1,2,4,8")
    parser.add_argument(
        "--feature-width",
        type=int,
        default=None,
        help="baseline feature replacement band; defaults to the support width",
    )
    parser.add_argument("--feature-policy", choices=("fresh", "baseline", "blend"), default="baseline")
    parser.add_argument("--angles", default="0,180")
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--render-chunk-size", type=int, default=200_000)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.cuda_device):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r}, expected physical CUDA {args.cuda_device}"
        )
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    camera_payload = json.loads(args.camera.read_text(encoding="utf-8"))
    camera = camera_payload.get("camera", camera_payload)
    fresh = load(args.fresh_state.resolve())
    endpoint = load(args.endpoint_dir.resolve() / "support/fixed_c128_shape_slat.pt")
    fresh_coords = fresh["coords"].int()
    fresh_features = fresh["features"].float()
    baseline_coords = endpoint["coords"].int()
    baseline_features = endpoint["raw_features"].float()
    if fresh_features.shape[1] != baseline_features.shape[1]:
        raise RuntimeError("fresh and baseline channel counts differ")
    widths = tuple(sorted({int(v.strip()) for v in args.widths.split(",") if v.strip()}))
    if not widths or any(v < 0 or v > 32 for v in widths):
        raise ValueError("widths must be a comma-separated subset of 0..32")
    if args.feature_width is not None and not 0 <= args.feature_width <= 32:
        raise ValueError("feature-width must be in 0..32")
    if args.feature_width is not None and any(args.feature_width > width for width in widths):
        raise ValueError("feature-width cannot exceed every requested support width")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pipeline = load_decoder_pipeline(args.model_path.resolve(), device)
    all_rows = []
    for width in widths:
        variant = output / f"width_{width:02d}_{args.feature_policy}"
        variant.mkdir(parents=True, exist_ok=True)
        coords, features, support_meta = make_augmented_state(
            fresh_coords,
            fresh_features,
            baseline_coords,
            baseline_features,
            width,
            args.feature_policy,
            args.feature_width,
        )
        atomic_save(
            variant / "shape_c128_augmented.pt",
            {"coords": coords, "raw_features": features, **support_meta},
        )
        print(
            f"[variant width={width}] decode global C128 tokens={coords.shape[0]:,}",
            flush=True,
        )
        started = time.perf_counter()
        mesh = decode_global(pipeline, coords, features, int(args.resolution), device)
        mesh_path = variant / "geometry_mesh.pt"
        atomic_save(mesh_path, {"mesh": mesh, **support_meta})
        try:
            import trimesh

            trimesh.Trimesh(
                vertices=mesh.vertices.numpy(),
                faces=mesh.faces.numpy(),
                process=False,
            ).export(variant / "geometry_mesh.glb")
        except Exception as exc:
            atomic_json(variant / "glb_export_error.json", {"error": repr(exc)})
        render_meta = render(
            mesh,
            camera,
            variant,
            device,
            args.reference.resolve(),
            args.reference_mask.resolve(),
            tuple(int(v.strip()) for v in args.angles.split(",") if v.strip()),
            int(args.render_resolution),
            int(args.render_chunk_size),
        )
        row = {
            "status": "complete",
            **support_meta,
            "decode_resolution": int(args.resolution),
            "vertices": int(mesh.vertices.shape[0]),
            "faces": int(mesh.faces.shape[0]),
            "seconds": float(time.perf_counter() - started),
            "mesh": str(mesh_path.resolve()),
            "render": render_meta,
        }
        atomic_json(variant / "summary.json", row)
        all_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del mesh, coords, features
        empty_cuda()
    atomic_json(
        output / "summary.json",
        {
            "format": "pixal3d_c128_boundary_anchor_v1",
            "fresh_state": str(args.fresh_state.resolve()),
            "endpoint": str(args.endpoint_dir.resolve()),
            "resolution": int(args.resolution),
            "feature_policy": args.feature_policy,
            "results": all_rows,
        },
    )


if __name__ == "__main__":
    main()
