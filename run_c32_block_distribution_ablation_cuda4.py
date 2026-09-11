#!/usr/bin/env python3
"""Run geometry-only C32 support-distribution ablations on one global block.

The control is the existing C128->context32 route.  Every variant keeps the
same projected image rows and changes only the C32 spatial support seen by the
Shape512/Shape1024 Flow models:

  raw_floor / raw_round
  trim1_*                 remove the one-cell boundary shell
  affine_independent_*   map the block bbox into the independent-C32 bbox

For the affine variant, image projection remains tied to the original global
coordinates; only the local Flow coordinates are reparameterized.  The final
mesh is mapped back through the inverse affine transform before rendering.

The Shape512 condition resolution is configurable.  The historical control
uses a native 1024 crop, while ``--stage1-crop-size 512`` matches the native
independent C32 baseline and exposes the resolution mismatch separately.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
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
from PIL import Image, ImageDraw

import pixal3d_c128_block_local_cascade_shape_renoise_2048 as cascade
import pixal3d_c128_context32_hr_crop_local_geometry as local
from pixal3d.modules.sparse import SparseTensor


ROOT = Path(__file__).resolve().parent
BLOCK_ROOT = ROOT / "outputs/c128_context32_local_cascade_4096_crop_geometry_cuda4"
DEFAULT_MODEL = Path("/home/nvme04/yyyan/download/model/Pixal3D")
DEFAULT_CANONICAL = ROOT / "outputs/c64_to_c128_two_stage_block_flow_geometry_cuda4/inputs/canonical_4096.png"
DEFAULT_CAMERA = ROOT / "outputs/baseline1024_c128_8xc64_geometry_cuda4/camera.json"
INDEPENDENT_ROOT = ROOT / (
    "outputs/independent_subimage_baseline_compare_cuda4/"
    "back_headtop_inner4096_baseline1024"
)
INDEPENDENT_CROP_META = ROOT / (
    "outputs/independent_subimage_baseline_compare_cuda4/inputs/"
    "back_headtop_inner4096_bbox1024.json"
)
GLOBAL_GRID = 128
BLOCK_GRID = 32
STAGE2_GRID = 64
STAGE1_RESOLUTION = 512
STAGE2_RESOLUTION = 1024
STEPS = 12
CANONICAL_SIZE = 4096
DINO_PATCH = 16


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
    if isinstance(value, float) and not math.isfinite(value):
        return None
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


def support_stats(local_xyz: torch.Tensor, grid: int) -> dict[str, Any]:
    xyz = local_xyz.int().cpu()
    if not len(xyz):
        return {"tokens": 0}
    lo = xyz.amin(0)
    hi = xyz.amax(0)
    distance = torch.minimum(xyz, (grid - 1) - xyz).amin(1)
    boundary = distance == 0
    volume = int(torch.prod(hi - lo + 1))
    return {
        "tokens": int(len(xyz)),
        "min": lo.tolist(),
        "max": hi.tolist(),
        "span": (hi - lo + 1).tolist(),
        "bbox_fill": float(len(xyz) / max(1, volume)),
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


def sort_unique_with_rows(
    xyz: torch.Tensor, source_rows: torch.Tensor, grid: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort transformed support and retain the first source row per cell."""
    xyz = xyz.int().cpu()
    source_rows = source_rows.long().cpu()
    keys = (xyz[:, 0].long() * grid + xyz[:, 1]) * grid + xyz[:, 2]
    order = torch.argsort(keys, stable=True)
    keys = keys.index_select(0, order)
    keep = torch.ones(len(order), dtype=torch.bool)
    if len(order) > 1:
        keep[1:] = keys[1:] != keys[:-1]
    order = order[keep]
    return xyz.index_select(0, order), source_rows.index_select(0, order)


def variant_support(
    name: str,
    source_local: torch.Tensor,
    source_projection_coords: torch.Tensor,
    independent_stats: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build model-local C32 support while preserving source projection rows."""
    if name.endswith("_floor"):
        quantization = "floor"
        base = name[: -len("_floor")]
    elif name.endswith("_round"):
        quantization = "round"
        base = name[: -len("_round")]
    else:
        raise ValueError(f"variant must end in _floor or _round: {name}")

    source_rows = torch.arange(len(source_local), dtype=torch.long)
    if base == "raw":
        model_xyz = source_local.clone()
        affine_a = torch.ones(3)
        affine_b = torch.zeros(3)
    elif base == "trim1":
        keep = ((source_local >= 1) & (source_local <= 30)).all(1)
        source_rows = source_rows[keep]
        model_xyz = source_local[keep].clone()
        affine_a = torch.ones(3)
        affine_b = torch.zeros(3)
    elif base == "affine_independent":
        source_lo = source_local.amin(0).float()
        source_hi = source_local.amax(0).float()
        target_lo = torch.tensor(independent_stats["min"], dtype=torch.float32)
        target_hi = torch.tensor(independent_stats["max"], dtype=torch.float32)
        affine_a = (target_hi - target_lo) / (source_hi - source_lo).clamp_min(1.0)
        affine_b = target_lo - affine_a * source_lo
        model_xyz = torch.round(source_local.float() * affine_a + affine_b).int()
    else:
        raise ValueError(f"unknown support variant: {base}")

    if len(model_xyz):
        if bool(((model_xyz < 0) | (model_xyz >= BLOCK_GRID)).any()):
            raise RuntimeError(f"{name}: transformed support escaped C32")
        model_xyz, source_rows = sort_unique_with_rows(
            model_xyz, source_rows, BLOCK_GRID
        )
    projection = source_projection_coords.index_select(0, source_rows).int()
    metadata = {
        "name": name,
        "base": base,
        "quantization": quantization,
        "source_tokens": int(len(source_local)),
        "selected_source_rows": int(len(source_rows)),
        "model_support": support_stats(model_xyz, BLOCK_GRID),
        "source_support": support_stats(source_local, BLOCK_GRID),
        "local_affine_source_to_model": {
            "a": affine_a.tolist(),
            "b": affine_b.tolist(),
        },
    }
    return model_xyz, projection, metadata


def make_record(
    cube_id: int,
    start: Sequence[int],
    local_xyz: torch.Tensor,
    projection_coords: torch.Tensor,
) -> dict[str, Any]:
    n = int(len(local_xyz))
    rows = torch.arange(n, dtype=torch.long)
    local_coords = torch.cat(
        (torch.zeros((n, 1), dtype=torch.int32), local_xyz.int()), dim=1
    )
    return {
        "cube_id": int(cube_id),
        "start": tuple(int(v) for v in start),
        "global_row_ids": rows,
        "local_xyz": local_xyz.int().cpu(),
        "local_coords": local_coords.cpu(),
        "projection_coords": projection_coords.int().cpu(),
        "owned_row_ids": rows,
        "tokens": n,
    }


def make_stage1_conditions(
    cube_id: int,
    record: Mapping[str, Any],
    cached: Mapping[str, Any],
    source_rows: torch.Tensor,
) -> dict[str, Any]:
    rows = record["global_row_ids"].long()
    return {
        "cubes": {
            int(cube_id): {
                "global_row_ids": rows,
                "global": cached["global"].detach().cpu().contiguous(),
                "proj": cached["proj"].index_select(0, source_rows).detach().cpu().contiguous(),
            }
        }
    }


def choose_crop_for_size(
    uv4096: torch.Tensor,
    depth: torch.Tensor,
    finite: torch.Tensor,
    crop_size: int,
) -> dict[str, Any]:
    """Choose a patch-aligned canonical crop of the requested native size."""
    if crop_size <= 0 or crop_size > CANONICAL_SIZE:
        raise ValueError(f"invalid crop size: {crop_size}")
    points = uv4096[finite].double()
    if not len(points):
        raise RuntimeError("no finite projected support points")
    raw_lo = points.amin(0)
    raw_hi = points.amax(0)
    clipped_lo = torch.maximum(raw_lo, torch.zeros(2, dtype=torch.float64))
    clipped_hi = torch.minimum(
        raw_hi, torch.full((2,), float(CANONICAL_SIZE), dtype=torch.float64)
    )
    intersects = bool((clipped_hi > clipped_lo).all())
    center = (
        (clipped_lo + clipped_hi) * 0.5
        if intersects
        else (raw_lo + raw_hi) * 0.5
    ).clamp(0.0, float(CANONICAL_SIZE))
    starts: list[int] = []
    contains_bbox = True
    for axis in range(2):
        extent = float(clipped_hi[axis] - clipped_lo[axis])
        lower = max(0, int(math.ceil(float(clipped_hi[axis])) - crop_size))
        upper = min(
            CANONICAL_SIZE - crop_size,
            int(math.floor(float(clipped_lo[axis]))),
        )
        preferred = int(
            round((float(center[axis]) - crop_size / 2.0) / DINO_PATCH)
            * DINO_PATCH
        )
        if extent <= crop_size and lower <= upper:
            starts.append(min(max(preferred, lower), upper))
        else:
            contains_bbox = False
            starts.append(
                min(
                    max(preferred, 0),
                    CANONICAL_SIZE - crop_size,
                )
            )
    x0, y0 = starts
    inside = (
        finite
        & (uv4096[:, 0] >= x0)
        & (uv4096[:, 0] < x0 + crop_size)
        & (uv4096[:, 1] >= y0)
        & (uv4096[:, 1] < y0 + crop_size)
    )
    inside_finite = inside[finite]
    return {
        "crop_box_4096": [x0, y0, x0 + crop_size, y0 + crop_size],
        "projection_crop_box": [
            x0 / CANONICAL_SIZE,
            y0 / CANONICAL_SIZE,
            (x0 + crop_size) / CANONICAL_SIZE,
            (y0 + crop_size) / CANONICAL_SIZE,
        ],
        "raw_bbox_pixel_edges_4096": [*raw_lo.tolist(), *raw_hi.tolist()],
        "clipped_bbox_pixel_edges_4096": [
            *clipped_lo.tolist(),
            *clipped_hi.tolist(),
        ],
        "projected_extent_4096": [
            max(0.0, float(clipped_hi[0] - clipped_lo[0])),
            max(0.0, float(clipped_hi[1] - clipped_lo[1])),
        ],
        "finite_projected_tokens": int(finite.sum()),
        "crop_covered_finite_tokens": int(inside_finite.sum()),
        "crop_coverage_fraction": (
            float(inside_finite.float().mean()) if len(inside_finite) else 0.0
        ),
        "contains_projected_bbox": bool(contains_bbox),
        "intersects_canonical_image": intersects,
        "depth_range": [float(depth[finite].min()), float(depth[finite].max())],
        "size": [crop_size, crop_size],
        "alignment": DINO_PATCH,
        "source": "direct native crop from canonical_4096",
    }


@torch.no_grad()
def extract_native_stage1_condition(
    pipeline: Any,
    canonical: Image.Image,
    camera: Mapping[str, Any],
    cube_id: int,
    record: Mapping[str, Any],
    projection_grid: int,
    crop_size: int,
    input_size: int,
    output: Path,
) -> dict[str, Any]:
    """Extract Shape512 condition from a region resized to ``input_size``."""
    image_model = pipeline.image_cond_model_shape_512
    projection_coords = record["projection_coords"].int().cpu()
    uv4096, depth, finite = local.project_condition_points(
        image_model, projection_coords, projection_grid, camera
    )
    crop_info = choose_crop_for_size(uv4096, depth, finite, crop_size)
    box = tuple(int(x) for x in crop_info["crop_box_4096"])
    region = canonical.crop(box).convert("RGB")
    if region.size != (crop_size, crop_size):
        raise RuntimeError(f"cube {cube_id}: invalid native region {region.size}")
    crop = region.resize((input_size, input_size), Image.Resampling.LANCZOS)
    crop_dir = output / "inputs" / (
        f"shape512_region_{crop_size}_input_{input_size}_from_canonical4096"
    )
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop.save(crop_dir / f"cube_{cube_id:02d}.png")
    condition = pipeline.get_proj_cond_shape(
        image_model,
        [crop],
        projection_coords.to(pipeline.device),
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera.get("mesh_scale", 1.0)),
        grid_resolution_override=int(projection_grid),
        projection_crop_box=crop_info["projection_crop_box"],
        preserve_image_resolution=True,
    )["cond"]
    global_feature = condition["global"].detach().cpu().contiguous()
    projected_feature = condition["proj"].feats.detach().cpu().contiguous()
    if global_feature.shape[0] != 1 or projected_feature.shape[0] != len(projection_coords):
        raise RuntimeError(f"cube {cube_id}: native condition row mismatch")
    payload = {
        "cubes": {
            int(cube_id): {
                "global_row_ids": record["global_row_ids"].clone(),
                "global": global_feature,
                "proj": projected_feature,
                "projection_coords": projection_coords,
                "crop": crop_info,
                "source": (
                    f"DINO/NAF from canonical_4096 {crop_size} region "
                    f"resized to native {input_size} input"
                ),
            }
        },
        "source": f"direct canonical_4096 {crop_size} crop",
    }
    del condition, crop, global_feature, projected_feature
    local.empty_cuda()
    return payload


@torch.no_grad()
def decode_stage1_support(
    pipeline: Any,
    record: Mapping[str, Any],
    state32: torch.Tensor,
    quantization: str,
    device: torch.device,
) -> torch.Tensor:
    decoder = pipeline.models["shape_slat_decoder"]
    decoder.to(device)
    decoder.low_vram = True
    try:
        raw = local.denormalize(
            state32.to(device), pipeline.shape_slat_normalization
        )
        packed = SparseTensor(raw, record["local_coords"].to(device))
        candidates = decoder.upsample(packed, upsample_times=4)
        candidate_xyz = candidates[:, 1:].float()
        if quantization == "floor":
            local64 = torch.div(
                (candidate_xyz + 0.5) * STAGE2_GRID,
                STAGE1_RESOLUTION,
                rounding_mode="floor",
            ).int()
        elif quantization == "round":
            local64 = (
                (candidate_xyz + 0.5)
                / float(STAGE1_RESOLUTION)
                * (STAGE2_GRID - 1)
            ).round().int()
        else:
            raise ValueError(quantization)
        local64 = local64.unique(dim=0).int().cpu()
    finally:
        decoder.cpu()
        decoder.low_vram = False
    if not len(local64):
        raise RuntimeError("Shape512 decoder produced empty C64 support")
    if bool(((local64 < 0) | (local64 >= STAGE2_GRID)).any()):
        raise RuntimeError("Shape512 decoder produced out-of-range C64 support")
    return local64


def physical_stage2_projection(
    model_local64: torch.Tensor,
    start: Sequence[int],
    affine_a: torch.Tensor,
    affine_b: torch.Tensor,
) -> torch.Tensor:
    """Map model-local C64 cells back to global physical C256 cells."""
    start_t = torch.tensor(start, dtype=torch.float32)
    raw_local64 = ((model_local64.float() - 2.0 * affine_b[None]) / affine_a[None]).round().int()
    raw_local64 = raw_local64.clamp(0, STAGE2_GRID - 1)
    global_xyz = start_t.int()[None] * 2 + raw_local64
    return torch.cat(
        (torch.zeros((len(global_xyz), 1), dtype=torch.int32), global_xyz.int()),
        dim=1,
    )


@torch.no_grad()
def decode_mesh(
    pipeline: Any,
    record: Mapping[str, Any],
    state64: torch.Tensor,
    device: torch.device,
) -> Any:
    decoder = pipeline.models["shape_slat_decoder"]
    decoder.to(device)
    decoder.low_vram = True
    try:
        decoder.set_resolution(1024)
        raw = local.denormalize(
            state64.to(device), pipeline.shape_slat_normalization
        )
        packed = SparseTensor(raw, record["local_coords"].to(device))
        decoded = decoder(packed, return_subs=True)
        meshes = decoded[0] if isinstance(decoded, tuple) else decoded
        if not isinstance(meshes, (list, tuple)) or len(meshes) != 1:
            raise RuntimeError("decoder did not return one local mesh")
        return meshes[0].cpu()
    finally:
        decoder.cpu()
        decoder.low_vram = False


def map_mesh_to_global(
    mesh: Any,
    start: Sequence[int],
    affine_a: torch.Tensor,
    affine_b: torch.Tensor,
) -> Any:
    """Inverse-reparameterize local decoder vertices and place in C128 space."""
    vertices = mesh.vertices.float().cpu()
    # Decoder q is normalized to a local [-0.5, 0.5] cube.  Convert to the
    # continuous C32 index coordinate before applying the inverse affine.
    model_index = (vertices + 0.5) * BLOCK_GRID
    raw_index = (model_index - affine_b[None]) / affine_a[None].clamp_min(1e-6)
    raw_q = raw_index / BLOCK_GRID - 0.5
    start_t = torch.tensor(start, dtype=torch.float32)
    center = -0.5 + (start_t + BLOCK_GRID / 2.0) / GLOBAL_GRID
    scale = BLOCK_GRID / GLOBAL_GRID
    mapped = center[None] + scale * raw_q
    return type(mesh)(mapped, mesh.faces.int().cpu())


def crop_render(
    render_path: Path,
    box_4096: Sequence[int],
    output: Path,
) -> None:
    image = Image.open(render_path).convert("RGB")
    # The camera-normal render is 1024 while the canonical image is 4096.
    box = tuple(
        max(0, min(image.width, int(round(float(value) / 4.0))))
        for value in box_4096
    )
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return
    image.crop(box).resize((1024, 1024), Image.Resampling.NEAREST).save(output)


def make_variant_sheet(
    independent_render: Path,
    variant_dirs: Sequence[tuple[str, Path]],
    crop_box: Sequence[int],
    output: Path,
) -> None:
    items: list[tuple[str, Image.Image]] = []
    if independent_render.is_file():
        items.append(("independent C32 baseline", Image.open(independent_render).convert("RGB")))
    for name, path in variant_dirs:
        if not path.is_file():
            continue
        source = Image.open(path).convert("RGB")
        box = tuple(max(0, min(source.width, int(round(float(v) / 4.0)))) for v in crop_box)
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            continue
        items.append((name, source.crop(box).resize((1024, 1024), Image.Resampling.NEAREST)))
    if not items:
        return
    header = 34
    cols = min(3, len(items))
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 1024, rows * (1024 + header)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, image) in enumerate(items):
        x = (i % cols) * 1024
        y = (i // cols) * (1024 + header)
        sheet.paste(image, (x, y + header))
        draw.text((x + 8, y + 9), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-id", type=int, default=54)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--block-root", type=Path, default=BLOCK_ROOT)
    parser.add_argument("--canonical-4096", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/c128_context32_block_distribution_ablation_cuda4",
    )
    parser.add_argument(
        "--variants",
        default="raw_floor,raw_round,trim1_floor,trim1_round,affine_independent_floor,affine_independent_round",
    )
    parser.add_argument("--seed-c32", type=int, default=72101)
    parser.add_argument("--seed-c64", type=int, default=72102)
    parser.add_argument(
        "--stage1-crop-size",
        type=int,
        choices=(512, 1024),
        default=1024,
        help="native canonical crop size for the Shape512 condition",
    )
    parser.add_argument(
        "--stage1-input-size",
        type=int,
        choices=(512, 1024),
        default=None,
        help="resize the Shape512 region to this image input size",
    )
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--max-flow-tokens", type=int, default=30000)
    parser.add_argument("--max-decode-tokens", type=int, default=30000)
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--render-chunk-size", type=int, default=200000)
    parser.add_argument("--angles", default="0,60,120,180,240,300")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if not variants:
        raise ValueError("no variants")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.cuda_device):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r}, expected physical CUDA {args.cuda_device}"
        )
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)

    block_root = args.block_root.resolve()
    block_id = int(args.block_id)
    layout = read_json(block_root / "support/context32_block_layout.json")
    block = next(row for row in layout["blocks"] if int(row["cube_id"]) == block_id)
    start = tuple(int(v) for v in block["start"])
    cached512 = load(block_root / "conditions/shape512" / f"cube_{block_id:02d}.pt")
    source_projection = cached512["projection_coords"].int()
    source_local = source_projection[:, 1:] - torch.tensor(start, dtype=torch.int32)
    if bool(((source_local < 0) | (source_local >= BLOCK_GRID)).any()):
        raise RuntimeError("cached projection coords do not belong to requested block")
    if len(source_local) != len(source_local.unique(dim=0)):
        raise RuntimeError("cached block support has duplicate coordinates")
    independent = load(INDEPENDENT_ROOT / "support/coords_c32.pt")["coords"].int()
    independent_meta = read_json(INDEPENDENT_CROP_META)
    independent_stats = support_stats(independent[:, 1:], BLOCK_GRID)
    canonical = Image.open(args.canonical_4096).convert("RGB")
    camera_payload = json.loads(args.camera.read_text())
    camera = dict(camera_payload.get("camera", camera_payload))
    camera.setdefault("mesh_scale", 1.0)
    output = args.output_dir.resolve() / f"cube_{block_id:02d}"
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "config.json",
        {
            "format": "c32_block_distribution_ablation_cuda4_v1",
            "status": "running",
            "block_id": block_id,
            "block_start_c128": list(start),
            "route": "baseline C64 -> global C128 -> one context32 block -> local C32 Flow -> local C64 -> local Shape1024 Flow -> local mesh",
            "projected_image_rows_preserved": True,
            "stage1_crop_size": int(args.stage1_crop_size),
            "stage1_input_size": int(args.stage1_input_size or args.stage1_crop_size),
            "stage1_condition_source": (
                "fresh native crop condition"
                if (
                    args.stage1_crop_size != 1024
                    or args.stage1_input_size is not None
                )
                else "cached historical 1024 crop condition"
            ),
            "canonical_4096": args.canonical_4096,
            "camera": camera,
            "variants": variants,
            "independent_c32": independent_stats,
            "independent_crop_box_4096": independent_meta["selected_inner_crop_4096"],
            "source_c32": support_stats(source_local, BLOCK_GRID),
        },
    )
    print(
        f"[block] cube={block_id} start={start} source_c32={len(source_local)} "
        f"independent_c32={len(independent[:, 1:])}",
        flush=True,
    )
    print("[model] loading shape/image pipeline", flush=True)
    pipeline = cascade.init_shape_pipeline(args.model_path, device)
    pipeline.shape_slat_sampler_params["steps"] = STEPS
    schedule = pipeline.shape_slat_sampler.timestep_schedule(
        STEPS, float(pipeline.shape_slat_sampler_params.get("rescale_t", 3.0))
    )
    del schedule
    variant_render_paths: list[tuple[str, Path]] = []
    summaries: list[dict[str, Any]] = []

    for variant in variants:
        variant_dir = output / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        summary_path = variant_dir / "summary.json"
        if summary_path.is_file():
            cached_summary = read_json(summary_path)
            if cached_summary.get("status") == "complete":
                render_path = variant_dir / "final/multiview_1024/view_000_camera_normal.png"
                variant_render_paths.append((variant, render_path))
                summaries.append(cached_summary)
                print(f"[resume] {variant}", flush=True)
                continue
        started = time.perf_counter()
        model_xyz, projection, support_meta = variant_support(
            variant, source_local, source_projection, independent_stats
        )
        # The projection rows are sorted/selected in variant_support; recover
        # the exact source-row ids through coordinate keys instead of relying
        # on row order from the cached support.
        source_key = (source_projection[:, 1].long() * 128 + source_projection[:, 2].long()) * 128 + source_projection[:, 3].long()
        projection_key = (projection[:, 1].long() * 128 + projection[:, 2].long()) * 128 + projection[:, 3].long()
        source_rows = torch.searchsorted(torch.sort(source_key).values, projection_key)
        _, sorted_rows = torch.sort(source_key)
        source_rows = sorted_rows[source_rows]
        if not torch.equal(source_key.index_select(0, source_rows), projection_key):
            raise RuntimeError(f"{variant}: failed to recover condition source rows")
        # variant_support's affine parameters are needed to map generated
        # C64/mesh geometry back to the original block.
        affine = support_meta["local_affine_source_to_model"]
        affine_a = torch.tensor(affine["a"], dtype=torch.float32)
        affine_b = torch.tensor(affine["b"], dtype=torch.float32)
        record32 = make_record(block_id, start, model_xyz, projection)
        use_cached_stage1 = (
            args.stage1_crop_size == 1024 and args.stage1_input_size is None
        )
        if use_cached_stage1:
            condition32 = make_stage1_conditions(
                block_id, record32, cached512, source_rows
            )
        else:
            condition32 = extract_native_stage1_condition(
                pipeline,
                canonical,
                camera,
                block_id,
                record32,
                GLOBAL_GRID,
                args.stage1_crop_size,
                int(args.stage1_input_size or args.stage1_crop_size),
                variant_dir,
            )
        atomic_save(
            variant_dir / "support/shape512_input.pt",
            {
                "variant": variant,
                "local_coords": record32["local_coords"],
                "projection_coords": record32["projection_coords"],
                "source_rows": source_rows,
                "metadata": support_meta,
            },
        )
        atomic_json(variant_dir / "support/shape512_input_stats.json", support_meta)
        print(
            f"\n[variant {variant}] C32={len(model_xyz)} "
            f"boundary={support_meta['model_support'].get('boundary_fraction', 0.0):.3f}",
            flush=True,
        )
        state32 = local.run_complete_local_flow(
            pipeline,
            [record32],
            condition32,
            "shape_slat_flow_model_512",
            device,
            args.seed_c32,
            "shape512",
            variant_dir,
            args.max_flow_tokens,
            1,
        )
        local64_model = decode_stage1_support(
            pipeline,
            record32,
            state32,
            support_meta["quantization"],
            device,
        )
        record64 = make_record(
            block_id,
            start,
            local64_model,
            physical_stage2_projection(local64_model, start, affine_a, affine_b),
        )
        atomic_save(
            variant_dir / "support/shape1024_input.pt",
            {
                "variant": variant,
                "local_coords": record64["local_coords"],
                "projection_coords": record64["projection_coords"],
                "source_c64_model_tokens": int(len(local64_model)),
                "source_c64_model_stats": support_stats(local64_model, STAGE2_GRID),
            },
        )
        print(
            f"[variant {variant}] local C64={len(local64_model)}; extracting Shape1024 condition",
            flush=True,
        )
        condition64 = local.extract_conditions(
            pipeline,
            canonical,
            camera,
            [record64],
            GLOBAL_GRID * 2,
            "shape1024",
            variant_dir,
        )
        state64 = local.run_complete_local_flow(
            pipeline,
            [record64],
            condition64,
            "shape_slat_flow_model_1024",
            device,
            args.seed_c64,
            "shape1024",
            variant_dir,
            args.max_flow_tokens,
            1,
        )
        mesh_local = decode_mesh(pipeline, record64, state64, device)
        mesh_global = map_mesh_to_global(mesh_local, start, affine_a, affine_b)
        mesh_path = variant_dir / "final/geometry_mesh.pt"
        atomic_save(
            mesh_path,
            {
                "format": "c32_block_distribution_ablation_cuda4_v1",
                "variant": variant,
                "block_id": block_id,
                "mesh": mesh_global,
                "mesh_local": mesh_local,
                "local_to_global": {
                    "start_c128": list(start),
                    "affine_a": affine_a,
                    "affine_b": affine_b,
                },
            },
        )
        try:
            import trimesh

            tri = trimesh.Trimesh(
                vertices=mesh_global.vertices.numpy(),
                faces=mesh_global.faces.numpy(),
                process=False,
            )
            glb_path = variant_dir / "final/geometry_mesh.glb"
            glb_path.parent.mkdir(parents=True, exist_ok=True)
            tri.export(glb_path)
        except Exception as exc:
            glb_path = None
            atomic_json(variant_dir / "final/glb_export_error.json", {"error": repr(exc)})
        print(f"[variant {variant}] rendering normals", flush=True)
        render = local.render_normals(
            mesh_global.to(device),
            camera,
            variant_dir / "final",
            args.render_resolution,
            tuple(int(v) % 360 for v in args.angles.split(",") if v.strip()),
            args.render_chunk_size,
        )
        summary = {
            "format": "c32_block_distribution_ablation_cuda4_v1",
            "status": "complete",
            "variant": variant,
            "block_id": block_id,
            "block_start_c128": list(start),
            "support": support_meta,
            "shape1024_model_support": support_stats(local64_model, STAGE2_GRID),
            "vertices": int(len(mesh_global.vertices)),
            "faces": int(len(mesh_global.faces)),
            "seconds": time.perf_counter() - started,
            "mesh_pt": str(mesh_path.resolve()),
            "mesh_glb": str(glb_path.resolve()) if glb_path else None,
            "render": render,
            "stage1_crop_size": int(args.stage1_crop_size),
            "stage1_input_size": int(args.stage1_input_size or args.stage1_crop_size),
            "stage2_crop": load(variant_dir / f"conditions/shape1024/cube_{block_id:02d}.pt").get("crop") if (variant_dir / f"conditions/shape1024/cube_{block_id:02d}.pt").is_file() else None,
        }
        atomic_json(summary_path, summary)
        summaries.append(summary)
        render_path = variant_dir / "final/multiview_1024/view_000_camera_normal.png"
        variant_render_paths.append((variant, render_path))
        del state32, state64, condition32, condition64, record32, record64, mesh_local, mesh_global
        empty_cuda()

    independent_render = INDEPENDENT_ROOT / "multiview_1024/view_000_camera_normal.png"
    independent_box = independent_meta["selected_inner_crop_4096"]
    make_variant_sheet(
        independent_render,
        variant_render_paths,
        independent_box,
        output / "head_region_view0_variant_comparison.png",
    )
    atomic_json(
        output / "summary.json",
        {
            "format": "c32_block_distribution_ablation_cuda4_v1",
            "status": "complete",
            "block_id": block_id,
            "block_start_c128": list(start),
            "independent": independent_stats,
            "source": support_stats(source_local, BLOCK_GRID),
            "variants": summaries,
            "comparison_sheet": str((output / "head_region_view0_variant_comparison.png").resolve()),
        },
    )
    print(json.dumps(jsonable(read_json(output / "summary.json")), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
