#!/usr/bin/env python3
"""Test a crop-level C32 support against the independent C32 baseline.

The source is the completed global C128 support.  We select only points whose
global-camera projection falls inside the independent 1024 crop, map their
3-D bbox into the *observed interior* support range of the independent C32,
and then run the normal C32 -> C64 geometry cascade.  Image features come
from that same canonical_4096 1024 crop, resized to Shape512's 512 input.

This is a one-crop diagnostic, not a full tiled reconstruction.  It isolates
the support distribution while preserving the source image rows.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    str(Path(__file__).with_name("autotune_cache.json")),
)

import numpy as np
import torch
from PIL import Image

import pixal3d_c128_context32_hr_crop_local_geometry as local
import run_c32_block_distribution_ablation_cuda4 as ab


ROOT = Path(__file__).resolve().parent
DEFAULT_BLOCK_ROOT = ROOT / "outputs/c128_context32_local_cascade_4096_crop_geometry_cuda4"
DEFAULT_CANONICAL = ROOT / (
    "outputs/c64_to_c128_two_stage_block_flow_geometry_cuda4/inputs/canonical_4096.png"
)
DEFAULT_CAMERA = ROOT / "outputs/baseline1024_c128_8xc64_geometry_cuda4/camera.json"
INDEPENDENT_ROOT = ROOT / (
    "outputs/independent_subimage_baseline_compare_cuda4/"
    "back_headtop_inner4096_baseline1024"
)
INDEPENDENT_META = ROOT / (
    "outputs/independent_subimage_baseline_compare_cuda4/inputs/"
    "back_headtop_inner4096_bbox1024.json"
)
GLOBAL_GRID = 128
SHAPE512_GRID = 128
SHAPE1024_GRID = 256
STEPS = 12
TARGET_LO = torch.tensor([1, 3, 1], dtype=torch.float32)
TARGET_HI = torch.tensor([29, 31, 26], dtype=torch.float32)


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def support_stats(coords: torch.Tensor, grid: int) -> dict[str, Any]:
    xyz = coords[:, 1:].int() if coords.ndim == 2 and coords.shape[1] == 4 else coords.int()
    if not len(xyz):
        return {"tokens": 0}
    lo = xyz.amin(0)
    hi = xyz.amax(0)
    distance = torch.minimum(xyz, (grid - 1) - xyz).amin(1)
    boundary = distance == 0
    return {
        "tokens": int(len(xyz)),
        "min": lo.tolist(),
        "max": hi.tolist(),
        "span": (hi - lo + 1).tolist(),
        "bbox_fill": float(len(xyz) / max(1, int(torch.prod(hi - lo + 1)))),
        "boundary_tokens": int(boundary.sum()),
        "boundary_fraction": float(boundary.float().mean()),
        "boundary_by_axis": [
            int(((xyz[:, axis] == 0) | (xyz[:, axis] == grid - 1)).sum())
            for axis in range(3)
        ],
    }


def project_c128_to_4096(coords: torch.Tensor, camera: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU copy of ProjGrid.project_grid_indices for C128 coordinates."""
    xyz = coords[:, 1:].int()
    one = torch.linspace(-1.0, 1.0, GLOBAL_GRID)
    p = one[xyz]
    world = torch.stack((p[:, 0], -p[:, 2], p[:, 1]), dim=-1) / 2.0
    transform = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -float(camera["distance"])],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    homogeneous = torch.cat((world, torch.ones((len(world), 1))), dim=1)
    camera_xyz = homogeneous @ torch.linalg.inv(transform).T
    x, y, z = camera_xyz[:, 0], camera_xyz[:, 1], camera_xyz[:, 2]
    depth = -z
    focal = 16.0 / math.tan(float(camera["camera_angle_x"]) / 2.0) * 512.0 / 32.0
    uv512 = torch.stack(
        (focal * x / (-z + 1e-8) + 256.0, -focal * y / (-z + 1e-8) + 256.0),
        dim=1,
    )
    uv4096 = (uv512 + 0.5) * 8.0 - 0.5
    finite = torch.isfinite(uv4096).all(1) & torch.isfinite(depth) & (depth > 0)
    return uv4096, finite


def crop_info(box: list[int]) -> dict[str, Any]:
    x0, y0, x1, y1 = box
    return {
        "crop_box_4096": box,
        "projection_crop_box": [x0 / 4096.0, y0 / 4096.0, x1 / 4096.0, y1 / 4096.0],
        "size": [1024, 1024],
        "alignment": 16,
        "source": "independent subimage selected inner crop",
    }


def make_record(coords_local: torch.Tensor, projection: torch.Tensor) -> dict[str, Any]:
    n = int(len(coords_local))
    return {
        "cube_id": -4000,
        "start": (0, 0, 0),
        "global_row_ids": torch.arange(n, dtype=torch.long),
        "local_xyz": coords_local.int(),
        "local_coords": torch.cat((torch.zeros((n, 1), dtype=torch.int32), coords_local.int()), dim=1),
        "projection_coords": projection.int(),
        "owned_row_ids": torch.arange(n, dtype=torch.long),
        "tokens": n,
    }


def extract_fixed_condition(
    pipeline: Any,
    canonical: Image.Image,
    camera: Mapping[str, Any],
    record: Mapping[str, Any],
    info: Mapping[str, Any],
    stage: str,
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if stage == "shape512":
        image_model = pipeline.image_cond_model_shape_512
        grid = SHAPE512_GRID
        input_size = 512
    elif stage == "shape1024":
        image_model = pipeline.image_cond_model_shape_1024
        grid = SHAPE1024_GRID
        input_size = 1024
    else:
        raise ValueError(stage)
    projection = record["projection_coords"].int().cpu()
    uv4096, depth, finite = local.project_condition_points(image_model, projection, grid, camera)
    x0, y0, x1, y1 = [int(v) for v in info["crop_box_4096"]]
    region = canonical.crop((x0, y0, x1, y1)).convert("RGB")
    feature_image = region if input_size == 1024 else region.resize((512, 512), Image.Resampling.LANCZOS)
    image_dir = output / "inputs" / f"{stage}_crop_1024_input_{input_size}"
    image_dir.mkdir(parents=True, exist_ok=True)
    feature_image.save(image_dir / "crop.png")
    condition = pipeline.get_proj_cond_shape(
        image_model,
        [feature_image],
        projection.to(pipeline.device),
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera.get("mesh_scale", 1.0)),
        grid_resolution_override=grid,
        projection_crop_box=info["projection_crop_box"],
        preserve_image_resolution=True,
    )["cond"]
    global_feature = condition["global"].detach().cpu().contiguous()
    projected_feature = condition["proj"].feats.detach().cpu().contiguous()
    if global_feature.shape[0] != 1 or projected_feature.shape[0] != len(projection):
        raise RuntimeError(f"{stage}: condition row mismatch")
    inside = (
        finite
        & (uv4096[:, 0] >= x0)
        & (uv4096[:, 0] < x1)
        & (uv4096[:, 1] >= y0)
        & (uv4096[:, 1] < y1)
    )
    payload = {
        "cubes": {
            int(record["cube_id"]): {
                "global_row_ids": record["global_row_ids"].clone(),
                "global": global_feature,
                "proj": projected_feature,
                "projection_coords": projection,
                "crop": dict(info),
                "source": f"canonical_4096 1024 crop; Shape{stage[-4:]} input {input_size}",
            }
        },
        "source": "fixed independent subimage crop",
    }
    coverage = {
        "finite_tokens": int(finite.sum()),
        "inside_tokens": int(inside.sum()),
        "coverage": float(inside[finite].float().mean()) if int(finite.sum()) else 0.0,
        "uv_bbox_4096": [*uv4096[finite].amin(0).tolist(), *uv4096[finite].amax(0).tolist()] if int(finite.sum()) else None,
    }
    del condition, region, feature_image, global_feature, projected_feature
    empty_cuda()
    return payload, coverage


def map_global_c256_from_model_c64(model_c64: torch.Tensor, affine_a: torch.Tensor, affine_b: torch.Tensor) -> torch.Tensor:
    """Inverse of model C32 = global C128*a+b, in C256 coordinates."""
    return ((model_c64.float() - 2.0 * affine_b[None]) / affine_a[None]).round().int().clamp(0, 255)


def map_mesh_to_global(mesh: Any, affine_a: torch.Tensor, affine_b: torch.Tensor) -> Any:
    model_index = (mesh.vertices.float().cpu() + 0.5) * 32.0
    global_c128 = (model_index - affine_b[None]) / affine_a[None].clamp_min(1e-6)
    global_q = global_c128 / GLOBAL_GRID - 0.5
    return type(mesh)(global_q, mesh.faces.int().cpu())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block-root", type=Path, default=DEFAULT_BLOCK_ROOT)
    p.add_argument("--canonical-4096", type=Path, default=DEFAULT_CANONICAL)
    p.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/c32_crop_support_equivalence_cuda4")
    p.add_argument("--model-path", type=Path, default=Path("/home/nvme04/yyyan/download/model/Pixal3D"))
    p.add_argument("--cuda-device", type=int, default=4)
    p.add_argument("--seed-c32", type=int, default=72101)
    p.add_argument("--seed-c64", type=int, default=72102)
    p.add_argument("--max-flow-tokens", type=int, default=30000)
    p.add_argument("--render-resolution", type=int, default=1024)
    p.add_argument("--render-chunk-size", type=int, default=200000)
    p.add_argument("--angles", default="0,60,120,180,240,300")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.cuda_device):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}, expected {args.cuda_device}")
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    started = time.perf_counter()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    block_root = args.block_root.resolve()
    global_coords = load(block_root / "support/baseline_derived_c128_support.pt")["coords"].int()
    camera = dict(read_json(args.camera))
    camera.setdefault("mesh_scale", 1.0)
    meta = read_json(INDEPENDENT_META)
    box = [int(v) for v in meta["selected_inner_crop_4096"]]
    info = crop_info(box)
    uv4096, finite = project_c128_to_4096(global_coords, camera)
    inside = finite & (uv4096[:, 0] >= box[0]) & (uv4096[:, 0] < box[2]) & (uv4096[:, 1] >= box[1]) & (uv4096[:, 1] < box[3])
    source_rows = torch.where(inside)[0].long()
    source_xyz = global_coords[source_rows, 1:].int()
    source_lo = source_xyz.amin(0).float()
    source_hi = source_xyz.amax(0).float()
    affine_a = (TARGET_HI - TARGET_LO) / (source_hi - source_lo).clamp_min(1.0)
    affine_b = TARGET_LO - affine_a * source_lo
    model_xyz = torch.round(source_xyz.float() * affine_a[None] + affine_b[None]).int()
    model_xyz, selected_rows = ab.sort_unique_with_rows(model_xyz, source_rows, 32)
    projection = global_coords.index_select(0, selected_rows).int()
    record32 = make_record(model_xyz, projection)
    independent = load(INDEPENDENT_ROOT / "support/coords_c32.pt")["coords"].int()
    independent_stats = support_stats(independent, 32)
    source_stats = support_stats(torch.cat((torch.zeros((len(source_xyz), 1), dtype=torch.int32),
                                            source_xyz - source_lo.int()[None]), dim=1), 32)
    model_stats = support_stats(record32["local_coords"], 32)
    atomic_json(
        output / "config.json",
        {
            "format": "pixal3d_c32_crop_support_equivalence_cuda4_v1",
            "status": "running",
            "route": "global C128 projected crop support -> affine interior C32 -> C32 Flow -> C64 -> Shape1024 Flow",
            "crop_box_4096": box,
            "projection_source_tokens": int(inside.sum()),
            "selected_model_tokens": int(len(model_xyz)),
            "source_bbox_c128": [source_lo.tolist(), source_hi.tolist()],
            "model_bbox_c32": model_stats,
            "independent_c32": independent_stats,
            "source_bbox_relative_stats": source_stats,
            "affine_source_c128_to_model_c32": {"a": affine_a.tolist(), "b": affine_b.tolist()},
            "image_condition": "same canonical_4096 1024 crop resized to Shape512 512; Shape1024 uses native 1024 crop",
            "camera": camera,
        },
    )
    atomic_save(output / "support/shape512_input.pt", {"local_coords": record32["local_coords"], "projection_coords": projection, "source_rows": selected_rows, "stats": model_stats})
    print(f"[crop-support] source projected C128={int(inside.sum())} model C32={len(model_xyz)} independent C32={len(independent)}", flush=True)
    print(f"[crop-support] model support min/max={model_stats['min']}/{model_stats['max']} boundary={model_stats['boundary_fraction']:.4f}", flush=True)
    print("[model] loading shape/image pipeline", flush=True)
    pipeline = ab.cascade.init_shape_pipeline(args.model_path, device)
    pipeline.shape_slat_sampler_params["steps"] = STEPS

    canonical = Image.open(args.canonical_4096).convert("RGB")
    condition32, cov32 = extract_fixed_condition(pipeline, canonical, camera, record32, info, "shape512", output)
    state32 = local.run_complete_local_flow(pipeline, [record32], condition32, "shape_slat_flow_model_512", device, args.seed_c32, "shape512", output, args.max_flow_tokens, 1)
    local64 = ab.decode_stage1_support(pipeline, record32, state32, "round", device)
    global_c256 = map_global_c256_from_model_c64(local64, affine_a, affine_b)
    record64 = make_record(local64, torch.cat((torch.zeros((len(global_c256), 1), dtype=torch.int32), global_c256), dim=1))
    c64_stats = support_stats(record64["local_coords"], 64)
    atomic_save(output / "support/shape1024_input.pt", {"local_coords": record64["local_coords"], "projection_coords": record64["projection_coords"], "stats": c64_stats})
    print(f"[crop-support] decoded model C64={len(local64)}", flush=True)
    condition64, cov64 = extract_fixed_condition(pipeline, canonical, camera, record64, info, "shape1024", output)
    state64 = local.run_complete_local_flow(pipeline, [record64], condition64, "shape_slat_flow_model_1024", device, args.seed_c64, "shape1024", output, args.max_flow_tokens, 1)
    mesh_local = ab.decode_mesh(pipeline, record64, state64, device)
    mesh_global = map_mesh_to_global(mesh_local, affine_a, affine_b)
    atomic_save(output / "final/geometry_mesh.pt", {"mesh": mesh_global, "mesh_local": mesh_local, "affine_a": affine_a, "affine_b": affine_b})
    glb = None
    try:
        import trimesh
        glb_path = output / "final/geometry_mesh.glb"
        trimesh.Trimesh(vertices=mesh_global.vertices.numpy(), faces=mesh_global.faces.numpy(), process=False).export(glb_path)
        glb = str(glb_path.resolve())
    except Exception as exc:
        atomic_json(output / "final/glb_export_error.json", {"error": repr(exc)})
    angles = tuple(int(v) % 360 for v in args.angles.split(",") if v.strip())
    print("[render] crop-support mesh in global camera", flush=True)
    render_global = local.render_normals(mesh_global.to(device), camera, output / "final_global", args.render_resolution, angles, args.render_chunk_size)
    independent_camera = read_json(INDEPENDENT_ROOT / "camera.json")
    print("[render] crop-support local frame with independent camera", flush=True)
    render_local = local.render_normals(mesh_local.to(device), independent_camera, output / "final_independent_frame", args.render_resolution, angles, args.render_chunk_size)
    summary = {
        "format": "pixal3d_c32_crop_support_equivalence_cuda4_v1",
        "status": "complete",
        "crop_box_4096": box,
        "projection_source_tokens": int(inside.sum()),
        "selected_model_c32": model_stats,
        "independent_c32": independent_stats,
        "decoded_model_c64": c64_stats,
        "affine_source_c128_to_model_c32": {"a": affine_a.tolist(), "b": affine_b.tolist()},
        "stage1_coverage": cov32,
        "stage2_coverage": cov64,
        "vertices": int(len(mesh_global.vertices)),
        "faces": int(len(mesh_global.faces)),
        "seconds": time.perf_counter() - started,
        "mesh_pt": str((output / "final/geometry_mesh.pt").resolve()),
        "mesh_glb": glb,
        "render_global": render_global,
        "render_independent_frame": render_local,
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "config.json", {**read_json(output / "config.json"), "status": "complete"})
    print(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
