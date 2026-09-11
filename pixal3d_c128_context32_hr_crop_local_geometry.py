#!/usr/bin/env python3
"""Geometry-only C128 context32 local cascade with direct canonical 4096 crops.

Route:
  complete baseline C64 endpoint
    -> decoder.upsample(4) C1024 support
    -> floor quantization to C128 support
    -> context=32/stride=32 local blocks
    -> local C32 Shape512 Flow
    -> decoder.upsample(4) to local C64 support
    -> local C64 Shape1024 Flow
    -> local geometry decode at 1024
    -> map each local mesh to global object space and render camera normals.

Only the unchanged baseline C64 endpoint is used to generate the first
support.  After every support change the Flow starts from fresh noise; no
latent is copied onto a different coordinate set.  DINO/NAF receives a
native 1024x1024 crop directly cut from canonical_4096.  Texture is absent.
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
import pixal3d_global4096_tile_endpoint_rollout_sync as legacy
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import Mesh


ROOT = Path(__file__).resolve().parent
FORMAT = "pixal3d_c128_context32_local_cascade_4096_crop_geometry_v1"
BASELINE_GRID = 64
GLOBAL_GRID = 128
BLOCK_GRID = 32
C512_GRID = 512
LOCAL_C64 = 64
CANONICAL_SIZE = 4096
CROP_SIZE = 1024
DINO_PATCH = 16
STEPS = 12
DECODE_RESOLUTION = 1024

DEFAULT_MODEL = Path("/home/nvme04/yyyan/download/model/Pixal3D")
DEFAULT_BASELINE = ROOT / (
    "outputs/baseline1024_c128_8xc64_geometry_cuda4/"
    "baseline/shape_c64_denormalized.pt"
)
DEFAULT_CANONICAL = ROOT / (
    "outputs/c64_to_c128_two_stage_block_flow_geometry_cuda4/"
    "inputs/canonical_4096.png"
)
DEFAULT_CAMERA = ROOT / "outputs/baseline1024_c128_8xc64_geometry_cuda4/camera.json"
DEFAULT_OUTPUT = ROOT / "outputs/c128_context32_local_cascade_4096_crop_geometry_cuda4"


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_camera(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    camera = dict(payload.get("camera", payload))
    camera.setdefault("mesh_scale", 1.0)
    for key in ("camera_angle_x", "distance"):
        if key not in camera:
            raise KeyError(f"camera is missing {key}: {path}")
    return camera


def parse_angles(value: str) -> tuple[int, ...]:
    angles = tuple(int(item.strip()) % 360 for item in value.split(",") if item.strip())
    if not angles:
        raise ValueError("angles must not be empty")
    return angles


def validate_coords(coords: torch.Tensor, grid: int, name: str) -> torch.Tensor:
    coords = torch.as_tensor(coords).int().contiguous()
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"{name}: expected [N,4], got {tuple(coords.shape)}")
    if coords.numel() and bool((coords[:, 0] != 0).any()):
        raise ValueError(f"{name}: batch column must be zero")
    if coords.numel() and bool(((coords[:, 1:] < 0) | (coords[:, 1:] >= grid)).any()):
        raise ValueError(f"{name}: coordinate outside C{grid}")
    if len(coords) != len(coords.unique(dim=0)):
        raise ValueError(f"{name}: duplicate coordinates")
    return coords


def sort_coords(coords: torch.Tensor, grid: int) -> torch.Tensor:
    if not len(coords):
        return coords.reshape(0, 4).int()
    xyz = coords[:, 1:].long()
    key = (xyz[:, 0] * grid + xyz[:, 1]) * grid + xyz[:, 2]
    return coords[torch.argsort(key, stable=True)]


def denormalize(features: torch.Tensor, spec: Mapping[str, Any]) -> torch.Tensor:
    mean = torch.as_tensor(spec["mean"], dtype=features.dtype, device=features.device)[None]
    std = torch.as_tensor(spec["std"], dtype=features.dtype, device=features.device)[None]
    return features * std + mean


def seed_noise(shape: Sequence[int], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(tuple(int(x) for x in shape), generator=generator)


def load_baseline(path: Path, pipeline: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"complete baseline C64 endpoint not found: {path}")
    payload = load(path)
    coords = validate_coords(payload["coords"], BASELINE_GRID, "baseline C64")
    features = torch.as_tensor(payload["features"]).float().contiguous()
    if bool(payload.get("normalized", False)):
        features = denormalize(features, pipeline.shape_slat_normalization).cpu()
    if features.ndim != 2 or features.shape[0] != len(coords):
        raise ValueError("baseline features are not aligned with baseline coords")
    return coords.cpu(), features.cpu()


@torch.no_grad()
def baseline_upsample_to_c128(
    pipeline: Any,
    coords_c64: torch.Tensor,
    features_c64: torch.Tensor,
    device: torch.device,
    output: Path,
) -> torch.Tensor:
    slat = SparseTensor(features_c64.to(device), coords_c64.to(device))
    decoder = pipeline.models["shape_slat_decoder"]
    decoder.to(device)
    decoder.low_vram = True
    print(
        f"[support] baseline C64 -> decoder.upsample(4) C1024, "
        f"tokens={len(coords_c64):,}",
        flush=True,
    )
    coords_c1024 = decoder.upsample(slat, upsample_times=4)
    decoder.cpu()
    decoder.low_vram = False
    coords_c1024 = validate_coords(
        coords_c1024.detach().cpu(), 1024, "baseline C1024 support"
    )
    # This is the native cascade's support quantization convention.
    xyz_c128 = torch.div(
        (coords_c1024[:, 1:].float() + 0.5) * GLOBAL_GRID,
        1024.0,
        rounding_mode="floor",
    ).int()
    coords_c128 = torch.cat(
        (torch.zeros((len(xyz_c128), 1), dtype=torch.int32), xyz_c128), dim=1
    )
    coords_c128 = sort_coords(
        validate_coords(
            coords_c128.unique(dim=0), GLOBAL_GRID, "baseline-derived C128"
        ),
        GLOBAL_GRID,
    )
    atomic_save(
        output / "support/baseline_c1024_coords.pt",
        {
            "format": FORMAT,
            "coords": coords_c1024,
            "features_carried": False,
            "source": "complete baseline C64 endpoint decoder.upsample(4)",
        },
    )
    atomic_save(
        output / "support/baseline_derived_c128_support.pt",
        {
            "format": FORMAT,
            "coords": coords_c128,
            "features_carried": False,
            "quantization": "floor((coord+0.5)/1024*128)",
        },
    )
    atomic_json(
        output / "support/promotion_summary.json",
        {
            "baseline_c64_tokens": int(len(coords_c64)),
            "c1024_tokens": int(len(coords_c1024)),
            "c128_tokens": int(len(coords_c128)),
            "baseline_features_used_for": "decoder.upsample only",
            "changed_support_latent_transferred": False,
        },
    )
    del slat, coords_c1024
    empty_cuda()
    return coords_c128


def make_blocks(c128: torch.Tensor) -> list[dict[str, Any]]:
    c128 = validate_coords(c128, GLOBAL_GRID, "global C128")
    xyz = c128[:, 1:]
    coverage = torch.zeros(len(c128), dtype=torch.int16)
    records: list[dict[str, Any]] = []
    row_offset = 0
    cube_id = 0
    for sx in range(0, GLOBAL_GRID, BLOCK_GRID):
        for sy in range(0, GLOBAL_GRID, BLOCK_GRID):
            for sz in range(0, GLOBAL_GRID, BLOCK_GRID):
                start = torch.tensor((sx, sy, sz), dtype=torch.int32)
                rows = torch.where(
                    ((xyz >= start) & (xyz < start + BLOCK_GRID)).all(dim=1)
                )[0].long()
                if not len(rows):
                    cube_id += 1
                    continue
                local_xyz = xyz.index_select(0, rows) - start
                local_coords = torch.cat(
                    (
                        torch.zeros((len(rows), 1), dtype=torch.int32),
                        local_xyz,
                    ),
                    dim=1,
                )
                local_rows = torch.arange(
                    row_offset, row_offset + len(rows), dtype=torch.long
                )
                records.append(
                    {
                        "cube_id": int(cube_id),
                        "start": tuple(int(x) for x in start.tolist()),
                        "global_row_ids": local_rows,
                        "source_c128_rows": rows,
                        "local_xyz": local_xyz.int(),
                        "local_coords": local_coords.int(),
                        "projection_coords": c128.index_select(0, rows).int(),
                        "owned_row_ids": local_rows,
                        "tokens": int(len(rows)),
                    }
                )
                row_offset += len(rows)
                coverage.index_add_(
                    0, rows, torch.ones(len(rows), dtype=torch.int16)
                )
                cube_id += 1
    if not torch.all(coverage == 1):
        raise RuntimeError("C128 support is not partitioned exactly once")
    if row_offset != len(c128):
        raise RuntimeError("local C32 row table does not cover C128 support")
    return records


def pack_sparse_values(
    values: Sequence[SparseTensor],
    device: torch.device,
    label: str,
) -> SparseTensor:
    if not values:
        raise ValueError(f"{label}: empty values")
    feats: list[torch.Tensor] = []
    coords: list[torch.Tensor] = []
    for batch_id, value in enumerate(values):
        local = value.coords.detach().cpu().int()
        if local.ndim != 2 or local.shape[1] != 4:
            raise ValueError(f"{label}: invalid coords")
        if len(local) and bool((local[:, 0] != 0).any()):
            raise ValueError(f"{label}: local batch column must be zero")
        local = local.clone()
        local[:, 0] = int(batch_id)
        feats.append(value.feats)
        coords.append(local)
    return SparseTensor(
        torch.cat(feats, dim=0).to(device),
        torch.cat(coords, dim=0).to(device),
    )


def make_groups(
    records: Sequence[Mapping[str, Any]],
    max_tokens: int,
    max_blocks: int,
) -> list[list[Mapping[str, Any]]]:
    if max_tokens <= 0 or max_blocks <= 0:
        raise ValueError("batch limits must be positive")
    groups: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    tokens = 0
    for record in records:
        count = int(record["tokens"])
        if count > max_tokens:
            raise RuntimeError(
                f"cube {record['cube_id']} has {count} tokens > {max_tokens}"
            )
        if current and (len(current) >= max_blocks or tokens + count > max_tokens):
            groups.append(current)
            current = []
            tokens = 0
        current.append(record)
        tokens += count
    if current:
        groups.append(current)
    return groups


def exact_feature_order(
    actual_coords: torch.Tensor,
    actual_features: torch.Tensor,
    expected_coords: torch.Tensor,
    grid: int,
    label: str,
) -> torch.Tensor:
    actual_coords = actual_coords.int().cpu()
    expected_coords = expected_coords.int().cpu()
    actual_features = actual_features.float().cpu()
    actual_xyz = actual_coords[:, 1:]
    expected_xyz = expected_coords[:, 1:]
    actual_key = (actual_xyz[:, 0].long() * grid + actual_xyz[:, 1]) * grid + actual_xyz[:, 2]
    expected_key = (expected_xyz[:, 0].long() * grid + expected_xyz[:, 1]) * grid + expected_xyz[:, 2]
    actual_order = torch.argsort(actual_key, stable=True)
    expected_order = torch.argsort(expected_key, stable=True)
    if not torch.equal(
        actual_xyz.index_select(0, actual_order),
        expected_xyz.index_select(0, expected_order),
    ):
        raise RuntimeError(f"{label}: Flow changed sparse support")
    inverse = torch.empty_like(expected_order)
    inverse[expected_order] = torch.arange(len(expected_order))
    return actual_features.index_select(0, actual_order).index_select(0, inverse)


def project_condition_points(
    image_model: Any,
    projection_coords: torch.Tensor,
    projection_grid: int,
    camera: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use the same ProjGrid projection as feature extraction, scaled to C4096."""
    grid = image_model.proj_grid
    device = grid.grid_points.device
    coords = projection_coords.to(device=device, dtype=torch.int32)
    image_points, depth, _ = grid.project_grid_indices(
        camera_angle_x=torch.tensor(
            [float(camera["camera_angle_x"])], device=device
        ),
        distance=torch.tensor([float(camera["distance"])], device=device),
        mesh_scale=torch.tensor(
            [float(camera.get("mesh_scale", 1.0))], device=device
        ),
        grid_indices=coords[:, 1:],
        grid_resolution=int(projection_grid),
    )
    image_points = image_points[0].detach().cpu().float()
    depth = depth[0].detach().cpu().float()
    normalized = (
        image_points + 0.5
    ) / float(grid.image_resolution)
    uv4096 = normalized * CANONICAL_SIZE - 0.5
    finite = (
        torch.isfinite(uv4096).all(dim=1)
        & torch.isfinite(depth)
        & (depth > 0)
    )
    return uv4096, depth, finite


def choose_crop(
    uv4096: torch.Tensor,
    depth: torch.Tensor,
    finite: torch.Tensor,
    crop_size: int = CROP_SIZE,
) -> dict[str, Any]:
    """Center a patch-aligned native crop and record support coverage."""
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
    )
    center = center.clamp(0.0, float(CANONICAL_SIZE))
    starts: list[int] = []
    contains_bbox = True
    for axis in range(2):
        extent = float(clipped_hi[axis] - clipped_lo[axis])
        if intersects and extent <= crop_size:
            lower = max(0, int(math.ceil(float(clipped_hi[axis])) - crop_size))
            upper = min(
                CANONICAL_SIZE - crop_size,
                int(math.floor(float(clipped_lo[axis]))),
            )
            lower = int(math.ceil(lower / DINO_PATCH) * DINO_PATCH)
            upper = int(math.floor(upper / DINO_PATCH) * DINO_PATCH)
            if lower <= upper:
                preferred = int(
                    round(
                        (float(center[axis]) - crop_size / 2.0) / DINO_PATCH
                    )
                    * DINO_PATCH
                )
                starts.append(min(max(preferred, lower), upper))
                continue
        preferred = int(
            round(
                (float(center[axis]) - crop_size / 2.0) / DINO_PATCH
            )
            * DINO_PATCH
        )
        starts.append(min(max(preferred, 0), CANONICAL_SIZE - crop_size))
        contains_bbox = False
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
        "clipped_bbox_pixel_edges_4096": [*clipped_lo.tolist(), *clipped_hi.tolist()],
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
def extract_conditions(
    pipeline: Any,
    image_4096: Image.Image,
    camera: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    projection_grid: int,
    stage: str,
    output: Path,
    crop_size: int = CROP_SIZE,
    input_size: int | None = None,
) -> dict[str, Any]:
    if stage == "shape512":
        image_model = pipeline.image_cond_model_shape_512
    elif stage == "shape1024":
        image_model = pipeline.image_cond_model_shape_1024
    else:
        raise ValueError(stage)
    condition_dir = output / "conditions" / stage
    crop_dir = output / "inputs" / f"{stage}_crops_{crop_size}_from_canonical4096"
    cubes: dict[int, dict[str, Any]] = {}
    for record in records:
        cube_id = int(record["cube_id"])
        projection_coords = record["projection_coords"].int().cpu()
        uv4096, depth, finite = project_condition_points(
            image_model, projection_coords, projection_grid, camera
        )
        crop_info = choose_crop(uv4096, depth, finite, crop_size)
        box = tuple(int(x) for x in crop_info["crop_box_4096"])
        crop = image_4096.crop(box).convert("RGB")
        if crop.size != (crop_size, crop_size):
            raise RuntimeError(f"{stage} cube {cube_id}: invalid crop {crop.size}")
        feature_image = (
            crop
            if input_size is None or input_size == crop_size
            else crop.resize((input_size, input_size), Image.Resampling.LANCZOS)
        )
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop.save(crop_dir / f"cube_{cube_id:02d}.png")
        condition = pipeline.get_proj_cond_shape(
            image_model,
            [feature_image],
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
            raise RuntimeError(f"{stage} cube {cube_id}: condition row mismatch")
        payload = {
            "format": FORMAT,
            "stage": stage,
            "cube_id": cube_id,
            "global_row_ids": record["global_row_ids"].clone(),
            "global": global_feature,
            "proj": projected_feature,
            "projection_coords": projection_coords,
            "crop": crop_info,
            "source": (
                f"DINO/NAF from canonical_4096 {crop_size} region"
                + (f" resized to {input_size}" if input_size else "")
            ),
        }
        atomic_save(condition_dir / f"cube_{cube_id:02d}.pt", payload)
        cubes[cube_id] = payload
        print(
            f"[condition {stage}] cube={cube_id:02d} tokens={len(projection_coords):,} "
            f"crop={crop_info['crop_box_4096']} "
            f"coverage={crop_info['crop_coverage_fraction']:.3f}",
            flush=True,
        )
        del condition, crop, global_feature, projected_feature
        if input_size is not None and input_size != crop_size:
            del feature_image
        empty_cuda()
    atomic_json(
        output / "conditions" / f"{stage}_manifest.json",
        {
            "format": FORMAT,
            "stage": stage,
            "source": f"canonical_4096 direct {crop_size} region",
            "actual_image_input": [
                int(input_size or crop_size),
                int(input_size or crop_size),
            ],
            "region_size": [crop_size, crop_size],
            "projection_grid": int(projection_grid),
            "blocks": [
                {
                    "cube_id": int(record["cube_id"]),
                    "tokens": int(record["tokens"]),
                    "crop": cubes[int(record["cube_id"])]["crop"],
                }
                for record in records
            ],
        },
    )
    return {"cubes": cubes, "source": "direct canonical_4096 crop"}


def run_complete_local_flow(
    pipeline: Any,
    records: Sequence[Mapping[str, Any]],
    conditions: Mapping[str, Any],
    model_key: str,
    device: torch.device,
    seed: int,
    stage: str,
    output: Path,
    max_batch_tokens: int,
    max_batch_blocks: int,
) -> torch.Tensor:
    """Use the existing Jacobi local-batch implementation from step 0 to 12."""
    if not records:
        raise RuntimeError(f"{stage}: no active blocks")
    feature_channels = int(pipeline.models[model_key].in_channels)
    state_parts = [
        seed_noise(
            (int(record["tokens"]), feature_channels),
            seed + int(record["cube_id"]) * 1009,
        )
        for record in records
    ]
    state = torch.cat(state_parts, dim=0).float().cpu()
    # cascade.run_suffix uses the same native sampler and per-block local
    # packing.  n=0 with fresh noise is a complete 12-step Flow.
    schedule = pipeline.shape_slat_sampler.timestep_schedule(
        STEPS,
        float(pipeline.shape_slat_sampler_params.get("rescale_t", 3.0)),
    )
    state, timings = cascade.run_suffix(
        pipeline,
        records,
        state,
        conditions,
        model_key,
        0,
        schedule,
        "conditional",
        device,
        int(max_batch_tokens),
    )
    if state.shape[0] != sum(int(r["tokens"]) for r in records):
        raise RuntimeError(f"{stage}: returned state has wrong row count")
    atomic_save(
        output / "flow" / stage / "final_state_normalized.pt",
        {
            "format": FORMAT,
            "stage": stage,
            "features_normalized": state,
            "seed": int(seed),
            "complete_steps": STEPS,
            "local_coordinates": "zero-based",
        },
    )
    atomic_json(
        output / "flow" / stage / "summary.json",
        {
            "format": FORMAT,
            "stage": stage,
            "model": model_key,
            "steps": STEPS,
            "tokens": int(state.shape[0]),
            "blocks": len(records),
            "max_batch_tokens": int(max_batch_tokens),
            "max_batch_blocks": int(max_batch_blocks),
            "mode": "fresh complete local Flow",
            "baseline_features_copied": False,
            "step_timings": timings,
        },
    )
    return state


@torch.no_grad()
def decode_c32_to_c64(
    pipeline: Any,
    records: Sequence[Mapping[str, Any]],
    state32: torch.Tensor,
    device: torch.device,
    output: Path,
    max_tokens: int,
    max_blocks: int,
) -> list[dict[str, Any]]:
    groups = make_groups(records, max_tokens, max_blocks)
    decoder = pipeline.models["shape_slat_decoder"]
    decoder.to(device)
    decoder.low_vram = True
    result: list[dict[str, Any]] = []
    try:
        for group_index, group in enumerate(groups):
            values: list[SparseTensor] = []
            for record in group:
                rows = record["global_row_ids"].long()
                raw = denormalize(
                    state32.index_select(0, rows).to(device),
                    pipeline.shape_slat_normalization,
                )
                values.append(SparseTensor(raw, record["local_coords"].int()))
            packed = pack_sparse_values(
                values, device, "local Shape512 decoder input"
            )
            print(
                f"[decode support] group={group_index + 1}/{len(groups)} "
                f"blocks={len(group)} tokens={len(packed.coords):,} C32->C512",
                flush=True,
            )
            candidates = decoder.upsample(packed, upsample_times=4)
            for batch_id, record in enumerate(group):
                part = candidates[candidates[:, 0] == batch_id].detach().cpu()
                if not len(part):
                    raise RuntimeError(f"cube {record['cube_id']}: empty C512 support")
                local_xyz = torch.div(
                    (part[:, 1:].float() + 0.5) * LOCAL_C64,
                    C512_GRID,
                    rounding_mode="floor",
                ).int()
                local_coords = torch.cat(
                    (
                        torch.zeros((len(local_xyz), 1), dtype=torch.int32),
                        local_xyz,
                    ),
                    dim=1,
                )
                local_coords = sort_coords(
                    validate_coords(
                        local_coords.unique(dim=0),
                        LOCAL_C64,
                        f"cube {record['cube_id']} local C64",
                    ),
                    LOCAL_C64,
                )
                start = torch.tensor(record["start"], dtype=torch.int32)
                # Local C64 spans one C128 context32 block, so its exact
                # physical projection lattice is C256.
                projection_xyz = start[None] * 2 + local_coords[:, 1:]
                projection_coords = torch.cat(
                    (
                        torch.zeros((len(local_coords), 1), dtype=torch.int32),
                        projection_xyz,
                    ),
                    dim=1,
                )
                result.append(
                    {
                        "cube_id": int(record["cube_id"]),
                        "start": tuple(int(x) for x in record["start"]),
                        "global_row_ids": torch.arange(
                            0, len(local_coords), dtype=torch.long
                        ),
                        "local_xyz": local_coords[:, 1:].int(),
                        "local_coords": local_coords,
                        "projection_coords": projection_coords.int(),
                        "owned_row_ids": torch.arange(
                            0, len(local_coords), dtype=torch.long
                        ),
                        "tokens": int(len(local_coords)),
                        "source_c32_tokens": int(record["tokens"]),
                    }
                )
                atomic_save(
                    output / "support/stage1_local_c64"
                    / f"cube_{int(record['cube_id']):02d}.pt",
                    {
                        "format": FORMAT,
                        "cube_id": int(record["cube_id"]),
                        "input_local_c32": record["local_coords"],
                        "output_local_c64": local_coords,
                        "projection_coords_c256": projection_coords,
                        "features_carried": False,
                        "quantization": "floor((coord+0.5)/512*64)",
                    },
                )
            del candidates, packed, values
            empty_cuda()
    finally:
        decoder.cpu()
        decoder.low_vram = False
    # Rebuild contiguous row ids across all generated C64 blocks.
    offset = 0
    for record in sorted(result, key=lambda item: int(item["cube_id"])):
        rows = torch.arange(offset, offset + int(record["tokens"]), dtype=torch.long)
        record["global_row_ids"] = rows
        record["owned_row_ids"] = rows
        offset += int(record["tokens"])
    result.sort(key=lambda item: int(item["cube_id"]))
    atomic_json(
        output / "support/stage1_local_c64/summary.json",
        {
            "format": FORMAT,
            "blocks": len(result),
            "input_local_c32_tokens": int(sum(int(r["tokens"]) for r in records)),
            "output_local_c64_tokens": int(sum(int(r["tokens"]) for r in result)),
            "projection_grid": GLOBAL_GRID * 2,
            "features_carried": False,
        },
    )
    return result


@torch.no_grad()
def decode_local_meshes(
    pipeline: Any,
    records: Sequence[Mapping[str, Any]],
    state64: torch.Tensor,
    device: torch.device,
    output: Path,
    max_tokens: int,
    max_blocks: int,
) -> Mesh:
    groups = make_groups(records, max_tokens, max_blocks)
    decoder = pipeline.models["shape_slat_decoder"]
    decoder.to(device)
    decoder.low_vram = True
    vertices: list[torch.Tensor] = []
    faces: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    vertex_offset = 0
    started = time.perf_counter()
    try:
        decoder.set_resolution(DECODE_RESOLUTION)
        for group_index, group in enumerate(groups):
            values: list[SparseTensor] = []
            for record in group:
                raw = denormalize(
                    state64.index_select(0, record["global_row_ids"]).to(device),
                    pipeline.shape_slat_normalization,
                )
                values.append(SparseTensor(raw, record["local_coords"].int()))
            packed = pack_sparse_values(
                values, device, "local Shape1024 decoder input"
            )
            print(
                f"[decode mesh] group={group_index + 1}/{len(groups)} "
                f"blocks={len(group)} tokens={len(packed.coords):,}",
                flush=True,
            )
            decoded = decoder(packed, return_subs=True)
            meshes = decoded[0] if isinstance(decoded, tuple) else decoded
            if not isinstance(meshes, (list, tuple)) or len(meshes) != len(group):
                raise RuntimeError(
                    f"decoder returned invalid mesh batch for group {group_index}"
                )
            for record, local_mesh in zip(group, meshes):
                local_vertices = local_mesh.vertices.detach().cpu().float()
                local_faces = local_mesh.faces.detach().cpu().int()
                if not len(local_vertices) or not len(local_faces):
                    rows.append(
                        {
                            "cube_id": int(record["cube_id"]),
                            "vertices": int(len(local_vertices)),
                            "faces": int(len(local_faces)),
                            "skipped": True,
                        }
                    )
                    continue
                start = torch.tensor(record["start"], dtype=torch.float32)
                center = -0.5 + (start + BLOCK_GRID / 2.0) / GLOBAL_GRID
                scale = BLOCK_GRID / GLOBAL_GRID
                vertices.append(center[None] + scale * local_vertices)
                faces.append(local_faces + int(vertex_offset))
                vertex_offset += int(len(local_vertices))
                rows.append(
                    {
                        "cube_id": int(record["cube_id"]),
                        "vertices": int(len(local_vertices)),
                        "faces": int(len(local_faces)),
                        "skipped": False,
                        "global_center": center.tolist(),
                        "global_scale": scale,
                    }
                )
            del decoded, meshes, packed, values
            empty_cuda()
    finally:
        decoder.cpu()
        decoder.low_vram = False
    if not vertices:
        raise RuntimeError("all local Shape1024 decodes are empty")
    merged = Mesh(torch.cat(vertices, dim=0), torch.cat(faces, dim=0))
    atomic_save(
        output / "final/local_mesh_manifest.pt",
        {
            "format": FORMAT,
            "mesh_rows": rows,
            "merge": "concatenate local meshes in global object space; no welding",
        },
    )
    atomic_json(
        output / "final/local_mesh_summary.json",
        {
            "format": FORMAT,
            "blocks": rows,
            "vertices": int(len(merged.vertices)),
            "faces": int(len(merged.faces)),
            "seconds": time.perf_counter() - started,
        },
    )
    return merged


def make_contact_sheet(paths: Sequence[tuple[str, Path]], output: Path) -> None:
    if not paths:
        return
    images = [(label, Image.open(path).convert("RGB")) for label, path in paths]
    panel = images[0][1].width
    header = 32
    cols = min(3, len(images))
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * panel, rows * (panel + header)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(images):
        x = (index % cols) * panel
        y = (index // cols) * (panel + header)
        sheet.paste(image, (x, y + header))
        draw.text((x + 8, y + 8), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def render_normals(
    mesh: Mesh,
    camera: Mapping[str, Any],
    output: Path,
    resolution: int,
    angles: Sequence[int],
    chunk_size: int,
) -> dict[str, Any]:
    from pixal3d_baseline1024_pbr_mesh_compare import _make_camera_views
    from pixal3d.utils import render_utils

    extrinsics, intrinsics, _ = _make_camera_views(
        float(camera["camera_angle_x"]), float(camera["distance"]), angles
    )
    rendered = render_utils.render_frames(
        mesh,
        [extrinsics[a].to(mesh.device) for a in angles],
        [intrinsics.to(mesh.device) for _ in angles],
        options={
            "resolution": int(resolution),
            "near": 0.01,
            "far": float(camera["distance"]) + 10.0,
            "ssaa": 1,
            "chunk_size": int(chunk_size),
        },
        return_types=["normal", "mask"],
        verbose=True,
    )
    render_dir = output / f"multiview_{resolution}"
    render_dir.mkdir(parents=True, exist_ok=True)
    paths: list[tuple[str, Path]] = []
    for index, angle in enumerate(angles):
        normal = np.asarray(rendered["normal"][index])
        mask = np.asarray(rendered["mask"][index])
        normal_path = render_dir / f"view_{int(angle):03d}_camera_normal.png"
        mask_path = render_dir / f"view_{int(angle):03d}_mask.png"
        Image.fromarray(normal).convert("RGB").save(normal_path)
        Image.fromarray(mask).convert("L").save(mask_path)
        paths.append((f"yaw {angle} camera normal", normal_path))
    sheet = render_dir / "camera_normal_contact_sheet.png"
    make_contact_sheet(paths, sheet)
    return {
        "camera_normal_contact_sheet": str(sheet.resolve()),
        "views": [str(path.resolve()) for _, path in paths],
        "masks": [
            str((render_dir / f"view_{int(angle):03d}_mask.png").resolve())
            for angle in angles
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--baseline-c64", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--canonical-4096", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--steps", type=int, default=STEPS)
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
    parser.add_argument("--max-flow-tokens", type=int, default=20000)
    parser.add_argument("--max-flow-blocks", type=int, default=2)
    parser.add_argument("--max-decode-tokens", type=int, default=20000)
    parser.add_argument("--max-decode-blocks", type=int, default=1)
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--render-chunk-size", type=int, default=200000)
    parser.add_argument("--angles", default="0,60,120,180,240,300")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.steps != STEPS:
        raise ValueError("this experiment uses the native 12-step schedule")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.cuda_device):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r}; expected physical CUDA {args.cuda_device}"
        )
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    camera = parse_camera(args.camera)
    image_4096 = Image.open(args.canonical_4096).convert("RGB")
    if image_4096.size != (CANONICAL_SIZE, CANONICAL_SIZE):
        raise ValueError(f"canonical image must be 4096x4096, got {image_4096.size}")
    angles = parse_angles(args.angles)
    started = time.perf_counter()

    atomic_json(
        output / "config.json",
        {
            "format": FORMAT,
            "status": "running",
            "cuda_device": int(args.cuda_device),
            "route": (
                "baseline C64 -> decoder C1024 support -> C128 -> context32 "
                "local C32 Shape512 -> decoder local C64 -> local Shape1024 "
                "-> local decode1024"
            ),
            "baseline_c64": args.baseline_c64,
            "canonical_4096": args.canonical_4096,
            "actual_feature_input": "native crop region directly from canonical_4096",
            "stage1_crop_size": int(args.stage1_crop_size),
            "stage1_input_size": int(args.stage1_input_size or args.stage1_crop_size),
            "condition_projection_grids": {
                "shape512": GLOBAL_GRID,
                "shape1024": GLOBAL_GRID * 2,
            },
            "flow_coordinate_ranges": {
                "shape512": [0, 31],
                "shape1024": [0, 63],
            },
            "changed_support_latent_policy": "fresh noise after each support change",
            "texture": False,
            "steps": int(args.steps),
            "seed_c32": int(args.seed_c32),
            "seed_c64": int(args.seed_c64),
        },
    )

    print("[model] loading geometry-only pipeline", flush=True)
    pipeline = cascade.init_shape_pipeline(args.model_path, device)
    coords_c64, features_c64 = load_baseline(args.baseline_c64, pipeline)
    atomic_save(
        output / "baseline/shape_c64_denormalized.pt",
        {
            "format": FORMAT,
            "coords": coords_c64,
            "features": features_c64,
            "source": str(args.baseline_c64.resolve()),
            "complete_endpoint": True,
            "used_after_support_change": False,
        },
    )
    c128 = baseline_upsample_to_c128(
        pipeline, coords_c64, features_c64, device, output
    )
    records32 = make_blocks(c128)
    atomic_json(
        output / "support/context32_block_layout.json",
        {
            "format": FORMAT,
            "grid": GLOBAL_GRID,
            "context": BLOCK_GRID,
            "stride": BLOCK_GRID,
            "tokens": int(len(c128)),
            "active_blocks": len(records32),
            "all_blocks": (GLOBAL_GRID // BLOCK_GRID) ** 3,
            "blocks": [
                {
                    "cube_id": int(r["cube_id"]),
                    "start": list(r["start"]),
                    "tokens": int(r["tokens"]),
                    "local_indices": [0, 31],
                }
                for r in records32
            ],
        },
    )
    print(
        f"[support] C128 tokens={len(c128):,}, "
        f"active context32 blocks={len(records32)}",
        flush=True,
    )

    cond32 = extract_conditions(
        pipeline,
        image_4096,
        camera,
        records32,
        GLOBAL_GRID,
        "shape512",
        output,
        args.stage1_crop_size,
        args.stage1_input_size,
    )
    state32 = run_complete_local_flow(
        pipeline,
        records32,
        cond32,
        "shape_slat_flow_model_512",
        device,
        args.seed_c32,
        "shape512",
        output,
        args.max_flow_tokens,
        args.max_flow_blocks,
    )
    records64 = decode_c32_to_c64(
        pipeline,
        records32,
        state32,
        device,
        output,
        args.max_decode_tokens,
        args.max_decode_blocks,
    )
    local_c64_tokens = int(sum(int(r["tokens"]) for r in records64))
    del cond32, state32, records32
    empty_cuda()

    cond64 = extract_conditions(
        pipeline, image_4096, camera, records64, GLOBAL_GRID * 2, "shape1024", output
    )
    state64 = run_complete_local_flow(
        pipeline,
        records64,
        cond64,
        "shape_slat_flow_model_1024",
        device,
        args.seed_c64,
        "shape1024",
        output,
        args.max_flow_tokens,
        args.max_flow_blocks,
    )
    shape1024_tokens = int(state64.shape[0])
    merged_mesh = decode_local_meshes(
        pipeline,
        records64,
        state64,
        device,
        output,
        args.max_decode_tokens,
        args.max_decode_blocks,
    )
    del cond64, state64, records64
    empty_cuda()

    final_dir = output / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = final_dir / "geometry_mesh.pt"
    atomic_save(
        mesh_path,
        {
            "format": FORMAT,
            "mesh": merged_mesh,
            "mesh_coordinate_system": "global normalized object q [-0.5,0.5]",
            "local_to_global": "center + (32/128) * local_q",
        },
    )
    glb_path = final_dir / "geometry_mesh.glb"
    glb_error: str | None = None
    try:
        import trimesh

        trimesh.Trimesh(
            vertices=merged_mesh.vertices.numpy(),
            faces=merged_mesh.faces.numpy(),
            process=False,
        ).export(glb_path)
    except Exception as exc:
        glb_error = repr(exc)
        atomic_json(final_dir / "glb_export_error.json", {"error": glb_error})

    render = render_normals(
        merged_mesh.to(device),
        camera,
        final_dir,
        args.render_resolution,
        angles,
        args.render_chunk_size,
    )
    summary = {
        "format": FORMAT,
        "status": "complete",
        "texture_executed": False,
        "cuda_device": int(args.cuda_device),
        "seconds": time.perf_counter() - started,
        "baseline_c64_tokens": int(len(coords_c64)),
        "baseline_derived_c128_tokens": int(len(c128)),
        "context32_active_blocks": len(
            load_json(output / "support/context32_block_layout.json")["blocks"]
        ),
        "shape512_flow_tokens": int(
            load_json(output / "flow/shape512/summary.json")["tokens"]
        ),
        "local_c64_support_tokens_after_shape512": local_c64_tokens,
        "shape1024_flow_tokens": shape1024_tokens,
        "mesh_vertices": int(len(merged_mesh.vertices)),
        "mesh_faces": int(len(merged_mesh.faces)),
        "mesh_pt": str(mesh_path.resolve()),
        "mesh_glb": str(glb_path.resolve()) if glb_error is None else None,
        "glb_error": glb_error,
        "camera_normal_render": render,
        "condition_source": "direct native crop from canonical_4096",
        "condition_crop_size": [args.stage1_crop_size, args.stage1_crop_size],
        "condition_input_size": [
            int(args.stage1_input_size or args.stage1_crop_size),
            int(args.stage1_input_size or args.stage1_crop_size),
        ],
        "stage2_condition_crop_size": [CROP_SIZE, CROP_SIZE],
        "projection_grids": {"shape512": GLOBAL_GRID, "shape1024": GLOBAL_GRID * 2},
        "local_coordinate_ranges": {"shape512": [0, 31], "shape1024": [0, 63]},
        "baseline_features_copied_after_support_change": False,
    }
    atomic_json(output / "summary.json", summary)
    config = load_json(output / "config.json")
    config["status"] = "complete"
    atomic_json(output / "config.json", config)
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
