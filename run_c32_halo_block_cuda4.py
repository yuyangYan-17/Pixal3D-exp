#!/usr/bin/env python3
"""Run a single C32 halo experiment for the geometry cascade.

The ordinary route gives Shape512 one 32^3 context block.  This script gives
it a 64^3 C128 physical neighborhood compressed to a 32^3 model grid, then
keeps only the target 32^3 C128 block after the C32 decoder.  Shape1024 still
runs on the target block alone.  It is therefore a focused test of whether
the hard 0/31 walls in a clipped C32 fragment are suppressing local geometry.

The Shape512 image input follows the requested convention: use the complete
1024x1024 region selected from canonical_4096 and resize that region to 512.
Only one block is processed per invocation.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

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
DEFAULT_MODEL = Path("/home/nvme04/yyyan/download/model/Pixal3D")
DEFAULT_BLOCK_ROOT = ROOT / "outputs/c128_context32_local_cascade_4096_crop_geometry_cuda4"
DEFAULT_CANONICAL = ROOT / (
    "outputs/c64_to_c128_two_stage_block_flow_geometry_cuda4/inputs/canonical_4096.png"
)
DEFAULT_CAMERA = ROOT / "outputs/baseline1024_c128_8xc64_geometry_cuda4/camera.json"
INDEPENDENT_ROOT = ROOT / (
    "outputs/independent_subimage_baseline_compare_cuda4/"
    "back_headtop_inner4096_baseline1024"
)
INDEPENDENT_CAMERA = INDEPENDENT_ROOT / "camera.json"
GLOBAL_GRID = 128
TARGET_GRID = 32
STAGE2_GRID = 64
STEPS = 12


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_stats(coords: torch.Tensor, grid: int) -> dict[str, Any]:
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
        "margin1_or_more": int((distance >= 1).sum()),
        "margin2_or_more": int((distance >= 2).sum()),
        "margin3_or_more": int((distance >= 3).sum()),
    }


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


def get_block(layout: Mapping[str, Any], block_id: int) -> Mapping[str, Any]:
    for block in layout["blocks"]:
        if int(block["cube_id"]) == block_id:
            return block
    raise KeyError(f"block {block_id} is not active")


def make_halo_record(
    global_coords: torch.Tensor,
    target_start: Sequence[int],
    halo_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pack a 64-cell C128 halo into C32 and keep source projection rows."""
    if halo_size != 64:
        raise ValueError("the focused experiment currently supports halo_size=64")
    target_start_t = torch.tensor(target_start, dtype=torch.int32)
    half = (halo_size - 32) // 2
    halo_start = (target_start_t - half).clamp(0, GLOBAL_GRID - halo_size)
    halo_end = halo_start + halo_size
    xyz = global_coords[:, 1:].int()
    in_halo = ((xyz >= halo_start) & (xyz < halo_end)).all(1)
    source_rows = torch.where(in_halo)[0].long()
    source_xyz = xyz.index_select(0, source_rows)
    # Two adjacent C128 cells map to one C32 model cell.  Deduplication is
    # deliberate: the model sees one support token per sparse cell, while the
    # first source row supplies its projected image feature.
    compressed = torch.div(
        source_xyz - halo_start[None], 2, rounding_mode="floor"
    ).int()
    compressed, kept_rows = ab.sort_unique_with_rows(
        compressed, source_rows, TARGET_GRID
    )
    projection = global_coords.index_select(0, kept_rows).int()
    n = int(len(compressed))
    local_coords = torch.cat(
        (torch.zeros((n, 1), dtype=torch.int32), compressed), dim=1
    )
    record = {
        "cube_id": int(-1000 - int(target_start_t[0]) * 10000 - int(target_start_t[1]) * 100 - int(target_start_t[2])),
        "start": tuple(int(v) for v in target_start_t.tolist()),
        "global_row_ids": torch.arange(n, dtype=torch.long),
        "local_xyz": compressed,
        "local_coords": local_coords,
        "projection_coords": projection,
        "owned_row_ids": torch.arange(n, dtype=torch.long),
        "tokens": n,
    }
    target_mask = ((source_xyz >= target_start_t) & (source_xyz < target_start_t + 32)).all(1)
    meta = {
        "target_start_c128": target_start_t.tolist(),
        "target_end_c128_exclusive": (target_start_t + 32).tolist(),
        "halo_start_c128": halo_start.tolist(),
        "halo_end_c128_exclusive": halo_end.tolist(),
        "halo_source_tokens_before_compression": int(in_halo.sum()),
        "halo_model_c32_tokens": n,
        "halo_source_target_block_tokens": int(target_mask.sum()),
        "target_offset_c32": ((target_start_t - halo_start) // 2).tolist(),
        "target_extent_c32": [16, 16, 16],
        "support": tensor_stats(local_coords, TARGET_GRID),
    }
    return record, meta


@torch.no_grad()
def extract_fixed_stage1_condition(
    pipeline: Any,
    canonical: Image.Image,
    camera: Mapping[str, Any],
    record: Mapping[str, Any],
    crop_info: Mapping[str, Any],
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use target block's complete 1024 region, resized to Shape512's 512 input."""
    image_model = pipeline.image_cond_model_shape_512
    projection_coords = record["projection_coords"].int().cpu()
    uv4096, depth, finite = local.project_condition_points(
        image_model, projection_coords, GLOBAL_GRID, camera
    )
    box = tuple(int(v) for v in crop_info["crop_box_4096"])
    region = canonical.crop(box).convert("RGB")
    if region.size != (1024, 1024):
        raise RuntimeError(f"invalid target stage1 region: {region.size}")
    feature_image = region.resize((512, 512), Image.Resampling.LANCZOS)
    image_dir = output / "inputs/shape512_region_1024_input_512_from_canonical4096"
    image_dir.mkdir(parents=True, exist_ok=True)
    feature_image.save(image_dir / "target.png")
    condition = pipeline.get_proj_cond_shape(
        image_model,
        [feature_image],
        projection_coords.to(pipeline.device),
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera.get("mesh_scale", 1.0)),
        grid_resolution_override=GLOBAL_GRID,
        projection_crop_box=crop_info["projection_crop_box"],
        preserve_image_resolution=True,
    )["cond"]
    global_feature = condition["global"].detach().cpu().contiguous()
    projected_feature = condition["proj"].feats.detach().cpu().contiguous()
    if global_feature.shape[0] != 1 or projected_feature.shape[0] != len(projection_coords):
        raise RuntimeError("stage1 halo condition row mismatch")
    x0, y0, x1, y1 = [float(v) for v in crop_info["crop_box_4096"]]
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
                "projection_coords": projection_coords,
                "crop": dict(crop_info),
                "source": "canonical_4096 target 1024 region resized to 512",
            }
        },
        "source": "target block crop 1024 -> Shape512 input 512",
    }
    coverage = {
        "finite_projection_tokens": int(finite.sum()),
        "target_crop_covered_tokens": int(inside.sum()),
        "target_crop_coverage_fraction": float(inside[finite].float().mean()) if int(finite.sum()) else 0.0,
        "projected_uv_bbox_4096": [
            *uv4096[finite].amin(0).tolist(), *uv4096[finite].amax(0).tolist()
        ] if int(finite.sum()) else None,
    }
    del condition, region, feature_image, global_feature, projected_feature
    empty_cuda()
    return payload, coverage


def target_c64_record(
    halo_c64: torch.Tensor,
    target_start: Sequence[int],
    halo_start: Sequence[int],
) -> tuple[dict[str, Any], torch.Tensor, dict[str, Any]]:
    """Keep the target physical C128 block and recenter it to local C64."""
    target_start_t = torch.tensor(target_start, dtype=torch.int32)
    halo_start_t = torch.tensor(halo_start, dtype=torch.int32)
    # The halo was compressed from 64 C128 cells to the Shape512 C32 grid.
    # Consequently decoder C64 index n represents halo C128 position
    # halo_start+n (one C128 cell per output position), rather than the native
    # block route's C256 position start*2+n.  Select the target in that
    # physical C128 frame first, then expand its local positions to the
    # target Shape1024 C64/C256 frame.
    global_c128 = halo_start_t[None] + halo_c64.int()
    low_c128 = target_start_t
    high_c128 = target_start_t + 32
    keep = ((global_c128 >= low_c128) & (global_c128 < high_c128)).all(1)
    selected_c128 = global_c128[keep].int()
    local_c128 = (selected_c128 - low_c128[None]).int().unique(dim=0)
    local64 = local_c128 * 2
    if not len(local64):
        raise RuntimeError("halo C32 decoder produced no target C64 support")
    # ``unique`` may reorder rows; regenerate the physical rows from the
    # canonical local support so condition rows remain one-to-one.
    low_c256 = target_start_t * 2
    selected_global = local64 + low_c256[None]
    n = int(len(local64))
    local_coords = torch.cat(
        (torch.zeros((n, 1), dtype=torch.int32), local64), dim=1
    )
    record = {
        "cube_id": int(-2000 - int(target_start_t[0]) * 10000 - int(target_start_t[1]) * 100 - int(target_start_t[2])),
        "start": tuple(int(v) for v in target_start_t.tolist()),
        "global_row_ids": torch.arange(n, dtype=torch.long),
        "local_xyz": local64,
        "local_coords": local_coords,
        "projection_coords": torch.cat(
            (torch.zeros((n, 1), dtype=torch.int32), selected_global), dim=1
        ),
        "owned_row_ids": torch.arange(n, dtype=torch.long),
        "tokens": n,
    }
    meta = {
        "halo_c64_tokens": int(len(halo_c64)),
        "target_selected_c64_tokens": n,
        "target_selected_fraction": float(keep.float().mean()),
        "target_local_c64": tensor_stats(local_coords, STAGE2_GRID),
        "target_global_c256_bbox": [
            *selected_global.amin(0).tolist(), *selected_global.amax(0).tolist()
        ],
        "halo_c64_interpreted_as_c128_physical": True,
    }
    return record, selected_global, meta


def map_target_mesh(mesh: Any, target_start: Sequence[int]) -> Any:
    """Map a normal target-block local mesh into the global C128 cube."""
    vertices = mesh.vertices.float().cpu()
    q = vertices
    start = torch.tensor(target_start, dtype=torch.float32)
    center = -0.5 + (start + 16.0) / GLOBAL_GRID
    scale = 32.0 / GLOBAL_GRID
    return type(mesh)(center[None] + scale * q, mesh.faces.int().cpu())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block-id", type=int, default=57)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--block-root", type=Path, default=DEFAULT_BLOCK_ROOT)
    p.add_argument("--canonical-4096", type=Path, default=DEFAULT_CANONICAL)
    p.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/c32_halo_block_cuda4",
    )
    p.add_argument("--cuda-device", type=int, default=4)
    p.add_argument("--seed-c32", type=int, default=72101)
    p.add_argument("--seed-c64", type=int, default=72102)
    p.add_argument("--max-flow-tokens", type=int, default=30000)
    p.add_argument("--max-decode-tokens", type=int, default=30000)
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
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r}, expected physical CUDA {args.cuda_device}"
        )
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    started = time.perf_counter()

    block_root = args.block_root.resolve()
    layout = ab.read_json(block_root / "support/context32_block_layout.json")
    block = get_block(layout, int(args.block_id))
    target_start = tuple(int(v) for v in block["start"])
    global_payload = load(block_root / "support/baseline_derived_c128_support.pt")
    global_coords = global_payload["coords"].int()
    source_c32_stats = ab.support_stats(
        global_coords[
            ((global_coords[:, 1:] >= torch.tensor(target_start))
             & (global_coords[:, 1:] < torch.tensor(target_start) + 32)).all(1),
            1:,
        ] - torch.tensor(target_start),
        TARGET_GRID,
    )
    record32, halo_meta = make_halo_record(global_coords, target_start, 64)
    independent = load(INDEPENDENT_ROOT / "support/coords_c32.pt")["coords"].int()
    independent_stats = ab.support_stats(independent[:, 1:], TARGET_GRID)
    target_crop = load(
        block_root / "conditions/shape512" / f"cube_{args.block_id:02d}.pt"
    ).get("crop")
    if not target_crop:
        raise RuntimeError("target block has no cached Shape512 crop metadata")
    canonical = Image.open(args.canonical_4096).convert("RGB")
    camera_payload = read_json(args.camera)
    camera = dict(camera_payload.get("camera", camera_payload))
    camera.setdefault("mesh_scale", 1.0)
    output = args.output_dir.resolve() / f"cube_{args.block_id:02d}"
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "config.json",
        {
            "format": "pixal3d_c32_halo_block_cuda4_v1",
            "status": "running",
            "block_id": int(args.block_id),
            "target_start_c128": list(target_start),
            "route": "global C128 64-cell halo -> local C32 Flow -> decoder C64 -> center target C64 -> local C64 Flow -> mesh",
            "stage1_image_condition": "canonical_4096 target block 1024 region resized to 512",
            "source_c32_target_block": source_c32_stats,
            "independent_c32": independent_stats,
            "halo": halo_meta,
            "target_crop_4096": target_crop,
            "camera": camera,
            "seeds": {"c32": args.seed_c32, "c64": args.seed_c64},
        },
    )
    print(
        f"[halo] target cube={args.block_id} start={target_start} "
        f"target_c32={source_c32_stats['tokens']} halo_c32={halo_meta['halo_model_c32_tokens']} "
        f"independent_c32={independent_stats['tokens']}",
        flush=True,
    )
    print("[model] loading shape/image pipeline", flush=True)
    pipeline = ab.cascade.init_shape_pipeline(args.model_path, device)
    pipeline.shape_slat_sampler_params["steps"] = STEPS

    condition32, stage1_coverage = extract_fixed_stage1_condition(
        pipeline, canonical, camera, record32, target_crop, output
    )
    atomic_save(
        output / "support/shape512_input.pt",
        {
            "local_coords": record32["local_coords"],
            "projection_coords": record32["projection_coords"],
            "metadata": halo_meta,
        },
    )
    atomic_json(output / "support/stage1_projection_coverage.json", stage1_coverage)
    print(
        f"[halo] stage1 C32={record32['tokens']} "
        f"target-crop coverage over halo={stage1_coverage['target_crop_coverage_fraction']:.3f}",
        flush=True,
    )
    state32 = local.run_complete_local_flow(
        pipeline,
        [record32],
        condition32,
        "shape_slat_flow_model_512",
        device,
        args.seed_c32,
        "shape512_halo",
        output,
        args.max_flow_tokens,
        1,
    )
    halo_c64 = ab.decode_stage1_support(
        pipeline, record32, state32, "round", device
    )
    halo_start = halo_meta["halo_start_c128"]
    record64, selected_global, stage2_meta = target_c64_record(
        halo_c64, target_start, halo_start
    )
    atomic_save(
        output / "support/shape1024_input.pt",
        {
            "halo_local_c64": halo_c64,
            "target_local_c64": record64["local_coords"],
            "target_projection_coords_c256": record64["projection_coords"],
            "metadata": stage2_meta,
        },
    )
    atomic_json(output / "support/stage2_support_stats.json", stage2_meta)
    print(
        f"[halo] decoded halo C64={len(halo_c64)}; target C64={record64['tokens']}",
        flush=True,
    )
    condition64 = local.extract_conditions(
        pipeline,
        canonical,
        camera,
        [record64],
        GLOBAL_GRID * 2,
        "shape1024",
        output,
        crop_size=1024,
        input_size=None,
    )
    state64 = local.run_complete_local_flow(
        pipeline,
        [record64],
        condition64,
        "shape_slat_flow_model_1024",
        device,
        args.seed_c64,
        "shape1024_target",
        output,
        args.max_flow_tokens,
        1,
    )
    mesh_local = ab.decode_mesh(pipeline, record64, state64, device)
    mesh_global = map_target_mesh(mesh_local, target_start)
    mesh_path = output / "final/geometry_mesh.pt"
    atomic_save(
        mesh_path,
        {
            "format": "pixal3d_c32_halo_block_cuda4_v1",
            "mesh": mesh_global,
            "mesh_local": mesh_local,
            "target_start_c128": list(target_start),
            "halo": halo_meta,
        },
    )
    try:
        import trimesh

        tri = trimesh.Trimesh(
            vertices=mesh_global.vertices.numpy(),
            faces=mesh_global.faces.numpy(),
            process=False,
        )
        glb_path = output / "final/geometry_mesh.glb"
        glb_path.parent.mkdir(parents=True, exist_ok=True)
        tri.export(glb_path)
    except Exception as exc:
        glb_path = None
        atomic_json(output / "final/glb_export_error.json", {"error": repr(exc)})

    angles = tuple(int(v) % 360 for v in args.angles.split(",") if v.strip())
    print("[render] target mesh in global camera", flush=True)
    render_global = local.render_normals(
        mesh_global.to(device),
        camera,
        output / "final_global",
        args.render_resolution,
        angles,
        args.render_chunk_size,
    )
    independent_camera = read_json(INDEPENDENT_CAMERA)
    print("[render] target mesh in independent crop camera", flush=True)
    render_local = local.render_normals(
        mesh_local.to(device),
        independent_camera,
        output / "final_independent_frame",
        args.render_resolution,
        angles,
        args.render_chunk_size,
    )
    summary = {
        "format": "pixal3d_c32_halo_block_cuda4_v1",
        "status": "complete",
        "block_id": int(args.block_id),
        "target_start_c128": list(target_start),
        "source_target_c32": source_c32_stats,
        "independent_c32": independent_stats,
        "halo": halo_meta,
        "stage1_projection_coverage": stage1_coverage,
        "stage2": stage2_meta,
        "vertices": int(len(mesh_global.vertices)),
        "faces": int(len(mesh_global.faces)),
        "seconds": time.perf_counter() - started,
        "mesh_pt": str(mesh_path.resolve()),
        "mesh_glb": str(glb_path.resolve()) if glb_path else None,
        "render_global": render_global,
        "render_independent_frame": render_local,
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "config.json", {**read_json(output / "config.json"), "status": "complete"})
    print(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
