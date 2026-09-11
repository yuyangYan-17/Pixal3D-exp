#!/usr/bin/env python3
"""Geometry-only two-stage C64 -> C128 block-local shape Flow.

The support hierarchy follows the requested cascade exactly:

1. Run the ordinary image-to-3D baseline through a complete global C64 shape
   SLat.  Only its C64 coordinates are used as the first support; its latent
   features are never copied into a local Flow state.
2. Partition that C64 support into context=32, stride=32 blocks.  Each block
   is relabelled to local C32 coordinates [0, 31].  A fresh Shape512 Flow is
   sampled with a block-specific 1024 crop from the 2048 image.  The local
   C32 endpoint is sent through decoder.upsample(..., 4), producing C512
   decoder coordinates that are requantized to local C64.  Those local C64
   supports are placed into disjoint global C128 regions.
3. Partition the assembled global C128 support into context=64, stride=64
   blocks.  Each block is relabelled to local C64 coordinates [0, 63].  A
   fresh Shape1024 Flow is sampled with a newly extracted 1024 crop.  The
   resulting local C64 features are scattered back to global C128.
4. Decode the assembled global C128 shape SLat at resolution 2048 and render
   camera-space normal maps.  Texture is not loaded, sampled, or decoded.

The image condition is projected with the *physical global support* and only
the Flow sparse coordinates are relabelled.  This keeps the crop location
correct while making every local Flow input use the expected zero-based
coordinate range.
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

from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as camera_core


ROOT = Path(__file__).resolve().parent
FORMAT = "pixal3d_c64_to_c128_two_stage_block_flow_geometry_v1"
BASELINE_GRID = 64
GLOBAL_GRID = 128
STAGE1_CONTEXT = 32
STAGE1_LOCAL_GRID = 32
STAGE1_DECODER_GRID = 512
STAGE1_OUTPUT_LOCAL_GRID = 64
STAGE2_CONTEXT = 64
STAGE2_LOCAL_GRID = 64
CONDITION_IMAGE_GRID = 2048
CROP_SIZE = 1024
DINO_PATCH = 16
DECODE_RESOLUTION = 2048
DEFAULT_IMAGE = ROOT / "assets/choose/0_img.png"
DEFAULT_CAMERA = ROOT / "outputs/baseline1024_c128_8xc64_geometry_cuda4/camera.json"
DEFAULT_BASELINE_C64 = ROOT / "outputs/baseline1024_c128_8xc64_geometry_cuda4/baseline/shape_c64_denormalized.pt"
DEFAULT_OUTPUT = ROOT / "outputs/c64_to_c128_two_stage_block_flow_geometry_cuda4"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    tmp.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def seed_all(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def parse_angles(value: str) -> tuple[int, ...]:
    angles = tuple(int(item.strip()) % 360 for item in value.split(",") if item.strip())
    if not angles:
        raise ValueError("angles must contain at least one integer")
    return angles


def camera_from_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    camera = payload.get("camera", payload)
    camera = dict(camera)
    camera.setdefault("mesh_scale", 1.0)
    for key in ("camera_angle_x", "distance", "mesh_scale"):
        if key not in camera:
            raise KeyError(f"camera is missing {key}: {path}")
    return camera


def validate_coords(coords: torch.Tensor, grid: int, name: str) -> torch.Tensor:
    coords = torch.as_tensor(coords).int().contiguous()
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"{name}: expected [N,4], got {tuple(coords.shape)}")
    if coords.numel() and bool((coords[:, 0] != 0).any()):
        raise ValueError(f"{name}: expected a single batch")
    if coords.numel() and bool(((coords[:, 1:] < 0) | (coords[:, 1:] >= grid)).any()):
        raise ValueError(f"{name}: coordinates outside C{grid}")
    if len(coords) != len(coords.unique(dim=0)):
        raise ValueError(f"{name}: duplicate coordinates")
    return coords


def sort_rows(coords: torch.Tensor) -> torch.Tensor:
    if not len(coords):
        return coords.reshape(0, 4).int()
    key = (
        coords[:, 1].long() * 1_000_000
        + coords[:, 2].long() * 1_000
        + coords[:, 3].long()
    )
    return coords[torch.argsort(key)]


def normalize_shape(features: torch.Tensor, spec: Mapping[str, Any]) -> torch.Tensor:
    mean = torch.as_tensor(spec["mean"], dtype=features.dtype, device=features.device)[None]
    std = torch.as_tensor(spec["std"], dtype=features.dtype, device=features.device)[None]
    return (features - mean) / std


def denormalize_shape(features: torch.Tensor, spec: Mapping[str, Any]) -> torch.Tensor:
    mean = torch.as_tensor(spec["mean"], dtype=features.dtype, device=features.device)[None]
    std = torch.as_tensor(spec["std"], dtype=features.dtype, device=features.device)[None]
    return features * std + mean


def load_baseline_c64(path: Path) -> tuple[torch.Tensor, torch.Tensor | None]:
    payload = load(path)
    coords = validate_coords(payload["coords"], BASELINE_GRID, "baseline C64")
    features = payload.get("features")
    if features is not None:
        features = torch.as_tensor(features).float().contiguous()
        if features.ndim != 2 or features.shape[0] != len(coords):
            raise ValueError("baseline C64 features are not row-aligned")
    return coords, features


@torch.no_grad()
def generate_baseline_c64(
    pipeline: Any,
    canonical: Mapping[str, Image.Image],
    camera: Mapping[str, Any],
    device: torch.device,
    steps: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the ordinary full-image 512/C32 -> 1024/C64 shape baseline."""
    seed_all(seed)
    print("[baseline] sparse structure -> C32", flush=True)
    cond_ss = pipeline.get_proj_cond_ss(
        [canonical["image_512"]],
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
    )
    coords_c32 = pipeline.sample_sparse_structure(cond_ss, 32)
    del cond_ss
    empty_cuda()

    print("[baseline] complete Shape512 C32 Flow", flush=True)
    cond_c32 = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_512,
        [canonical["image_512"]],
        coords_c32,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
        grid_resolution_override=32,
    )
    shape_c32 = pipeline.sample_shape_slat(
        cond_c32,
        pipeline.models["shape_slat_flow_model_512"],
        coords_c32,
        {"steps": int(steps)},
    )
    del cond_c32, coords_c32
    empty_cuda()

    print("[baseline] decoder upsample -> C64, complete Shape1024 C64 Flow", flush=True)
    decoder = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        decoder.to(device)
        decoder.low_vram = True
    hr_coords = decoder.upsample(shape_c32, upsample_times=4)
    if pipeline.low_vram:
        decoder.cpu()
        decoder.low_vram = False
    coords_c64 = torch.cat(
        (
            hr_coords[:, :1].int(),
            ((hr_coords[:, 1:].float() + 0.5) / 512.0 * (BASELINE_GRID - 1)).round().int(),
        ),
        dim=1,
    ).unique(dim=0)
    coords_c64 = validate_coords(coords_c64.cpu(), BASELINE_GRID, "generated baseline C64")
    del hr_coords, shape_c32
    empty_cuda()

    cond_c64 = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [canonical["image_1024"]],
        coords_c64.to(device),
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
        grid_resolution_override=BASELINE_GRID,
    )
    shape_c64 = pipeline.sample_shape_slat(
        cond_c64,
        pipeline.models["shape_slat_flow_model_1024"],
        coords_c64.to(device),
        {"steps": int(steps)},
    )
    return coords_c64.cpu(), shape_c64.feats.detach().cpu().float()


def make_blocks(
    coords: torch.Tensor,
    input_grid: int,
    context: int,
    output_grid: int,
    name: str,
) -> list[dict[str, Any]]:
    """Split support and relabel each context cube to zero-based local coords."""
    coords = validate_coords(coords, input_grid, name)
    if context <= 0 or input_grid % context:
        raise ValueError(f"{name}: context must divide input grid")
    starts = tuple(range(0, input_grid, context))
    if context not in (STAGE1_CONTEXT, STAGE2_CONTEXT):
        raise ValueError("unexpected context")
    scale = output_grid // input_grid
    if output_grid % input_grid:
        raise ValueError("output grid must be an integer multiple of input grid")
    xyz = coords[:, 1:].int()
    coverage = torch.zeros(len(coords), dtype=torch.int16)
    records: list[dict[str, Any]] = []
    cube_id = 0
    for sx in starts:
        for sy in starts:
            for sz in starts:
                start = torch.tensor((sx, sy, sz), dtype=torch.int32)
                inside = ((xyz >= start) & (xyz < start + context)).all(1)
                rows = torch.where(inside)[0].long()
                if not len(rows):
                    cube_id += 1
                    continue
                local_xyz = xyz.index_select(0, rows) - start
                if bool(((local_xyz < 0) | (local_xyz >= context)).any()):
                    raise RuntimeError(f"{name}: local coordinate escaped 0..{context - 1}")
                local_coords = torch.cat(
                    (torch.zeros((len(local_xyz), 1), dtype=torch.int32), local_xyz),
                    dim=1,
                )
                projection_coords = coords.index_select(0, rows).int()
                output_start = start * scale
                records.append(
                    {
                        "cube_id": cube_id,
                        "start": tuple(int(x) for x in start.tolist()),
                        "output_start": tuple(int(x) for x in output_start.tolist()),
                        "global_row_ids": rows,
                        "local_xyz": local_xyz.int(),
                        "local_coords": local_coords,
                        "projection_coords": projection_coords,
                        "tokens": int(len(rows)),
                    }
                )
                coverage.index_add_(0, rows, torch.ones(len(rows), dtype=torch.int16))
                cube_id += 1
    if not torch.all(coverage == 1):
        raise RuntimeError(
            f"{name}: support is not covered exactly once; "
            f"min={int(coverage.min())} max={int(coverage.max())}"
        )
    return records


def crop_for_support(
    projection_coords: torch.Tensor,
    projection_grid: int,
    camera: Mapping[str, Any],
) -> dict[str, Any]:
    """Project physical support to 2048 and choose a containing 1024 crop."""
    xyz = projection_coords[:, 1:].float()
    q = 2.0 * (xyz + 0.5) / float(projection_grid) - 1.0
    uv, depth, finite = camera_core._project_global_q_to_image(
        q,
        global_camera=camera,
        image_width=CONDITION_IMAGE_GRID,
        image_height=CONDITION_IMAGE_GRID,
    )
    finite = finite & torch.isfinite(uv).all(1)
    if not bool(finite.any()):
        raise RuntimeError("block support has no finite image projection")
    points = uv[finite].double()
    raw_lo = points.amin(0)
    raw_hi = points.amax(0)
    lo = raw_lo.clamp(0.0, float(CONDITION_IMAGE_GRID))
    hi = raw_hi.clamp(0.0, float(CONDITION_IMAGE_GRID))
    extent = hi - lo
    if bool((hi <= lo).any()):
        raise RuntimeError("block support does not intersect the image")
    if float(extent.max()) > CROP_SIZE + 1e-4:
        raise RuntimeError(
            f"block projection {extent.tolist()} is larger than the fixed {CROP_SIZE} crop"
        )
    center = (lo + hi) * 0.5
    starts: list[int] = []
    for axis in range(2):
        lower = max(0, int(math.ceil(float(hi[axis]))) - CROP_SIZE)
        upper = min(
            int(math.floor(float(lo[axis]))),
            CONDITION_IMAGE_GRID - CROP_SIZE,
        )
        if lower > upper:
            raise RuntimeError("cannot place a 1024 crop containing projected support")
        preferred = int(round(float(center[axis]) - CROP_SIZE / 2.0))
        starts.append(min(max(preferred, lower), upper))
    x0, y0 = starts
    box = (x0, y0, x0 + CROP_SIZE, y0 + CROP_SIZE)
    return {
        "projection_grid": int(projection_grid),
        "crop_box_2048": list(box),
        "projection_crop_box": [
            x0 / CONDITION_IMAGE_GRID,
            y0 / CONDITION_IMAGE_GRID,
            (x0 + CROP_SIZE) / CONDITION_IMAGE_GRID,
            (y0 + CROP_SIZE) / CONDITION_IMAGE_GRID,
        ],
        "raw_bbox_2048": [*raw_lo.tolist(), *raw_hi.tolist()],
        "clipped_bbox_2048": [*lo.tolist(), *hi.tolist()],
        "projected_extent_2048": extent.tolist(),
        "depth_range": [float(depth[finite].min()), float(depth[finite].max())],
        "crop_size": [CROP_SIZE, CROP_SIZE],
        "crop_alignment": DINO_PATCH,
    }


@torch.no_grad()
def extract_block_conditions(
    pipeline: Any,
    image_2048: Image.Image,
    camera: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    projection_grid: int,
    stage: str,
    output: Path,
) -> dict[int, dict[str, Any]]:
    """Re-extract DINO/NAF features independently for every block crop."""
    if stage == "c32":
        model = pipeline.image_cond_model_shape_512
    elif stage == "c64":
        model = pipeline.image_cond_model_shape_1024
    else:
        raise ValueError(stage)
    root = output / stage / "conditions"
    crop_root = output / stage / "image_crops_1024_from_2048"
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        cube_id = int(record["cube_id"])
        projection_coords = record["projection_coords"].int().cpu()
        crop_info = crop_for_support(projection_coords, projection_grid, camera)
        box = tuple(int(x) for x in crop_info["crop_box_2048"])
        crop = image_2048.crop(box).convert("RGB")
        if crop.size != (CROP_SIZE, CROP_SIZE):
            raise RuntimeError(f"{stage} cube {cube_id}: crop size is {crop.size}")
        crop_root.mkdir(parents=True, exist_ok=True)
        crop.save(crop_root / f"cube_{cube_id:02d}.png")

        cond = pipeline.get_proj_cond_shape(
            model,
            [crop],
            projection_coords.to(pipeline.device),
            camera_angle_x=float(camera["camera_angle_x"]),
            distance=float(camera["distance"]),
            mesh_scale=float(camera["mesh_scale"]),
            grid_resolution_override=int(projection_grid),
            projection_crop_box=crop_info["projection_crop_box"],
            preserve_image_resolution=True,
        )["cond"]
        global_feature = cond["global"].detach().cpu().contiguous()
        projection_feature = cond["proj"].feats.detach().cpu().contiguous()
        if global_feature.shape[0] != 1:
            raise RuntimeError(f"{stage} cube {cube_id}: invalid global condition shape")
        if projection_feature.shape[0] != len(projection_coords):
            raise RuntimeError(
                f"{stage} cube {cube_id}: projection rows {projection_feature.shape[0]} "
                f"!= support rows {len(projection_coords)}"
            )
        payload = {
            "format": FORMAT,
            "stage": stage,
            "cube_id": cube_id,
            "global_row_ids": record["global_row_ids"].clone(),
            "global": global_feature,
            "proj": projection_feature,
            "crop": crop_info,
            "projection_grid": int(projection_grid),
            "condition_source": "per-block crop from canonical 4096 resized to 2048",
            "global_token_source": "this block crop",
            "projected_token_source": "this block crop DINO/NAF projection",
        }
        atomic_save(root / f"cube_{cube_id:02d}.pt", payload)
        result[cube_id] = payload
        print(
            f"[condition {stage}] cube={cube_id:02d} "
            f"tokens={len(projection_coords):,} crop={CROP_SIZE} "
            f"bbox={crop_info['projected_extent_2048']}",
            flush=True,
        )
        del cond, crop, global_feature, projection_feature
        empty_cuda()
    return result


def pack_local_records(
    records: Sequence[Mapping[str, Any]],
    conditions: Mapping[int, Mapping[str, Any]],
    device: torch.device,
    seed: int,
    channels: int,
) -> tuple[SparseTensor, dict[str, Any]]:
    """Create a fresh-noise local sparse batch and attach row-aligned cond."""
    active = [record for record in records if len(record["local_xyz"])]
    if not active:
        raise RuntimeError("no active local blocks")
    coords_parts: list[torch.Tensor] = []
    global_parts: list[torch.Tensor] = []
    proj_parts: list[torch.Tensor] = []
    for batch_id, record in enumerate(active):
        local = record["local_coords"].clone().int()
        local[:, 0] = batch_id
        coords_parts.append(local)
        payload = conditions[int(record["cube_id"])]
        global_parts.append(payload["global"])
        proj_parts.append(payload["proj"])
        if payload["proj"].shape[0] != len(local):
            raise RuntimeError("local condition is not aligned with local support")
    packed_coords = torch.cat(coords_parts, dim=0).int()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(
        (len(packed_coords), int(channels)),
        generator=generator,
        dtype=torch.float32,
    )
    packed_noise = SparseTensor(noise.to(device), packed_coords.to(device))
    global_cond = torch.cat(global_parts, dim=0).to(device)
    proj_cond = SparseTensor(
        torch.cat(proj_parts, dim=0).to(device),
        packed_coords.to(device),
    )
    cond = {
        "cond": {"global": global_cond, "proj": proj_cond},
        "neg_cond": {
            "global": torch.zeros_like(global_cond),
            "proj": SparseTensor(torch.zeros_like(proj_cond.feats), packed_coords.to(device)),
        },
    }
    return packed_noise, cond


@torch.no_grad()
def run_fresh_local_flow(
    pipeline: Any,
    records: Sequence[Mapping[str, Any]],
    conditions: Mapping[int, Mapping[str, Any]],
    model_key: str,
    device: torch.device,
    steps: int,
    seed: int,
    stage: str,
    output: Path,
) -> SparseTensor:
    """Run a complete native Flow from fresh noise on zero-based local coords."""
    model = pipeline.models[model_key]
    channels = int(model.in_channels)
    noise, cond = pack_local_records(records, conditions, device, seed, channels)
    if pipeline.low_vram:
        model.to(device)
    params = dict(pipeline.shape_slat_sampler_params)
    params["steps"] = int(steps)
    print(
        f"[flow {stage}] fresh noise, model={model_key}, "
        f"blocks={len([r for r in records if len(r['local_xyz'])])}, "
        f"tokens={len(noise.coords):,}, local_indices=zero_based",
        flush=True,
    )
    started = time.perf_counter()
    sampled = pipeline.shape_slat_sampler.sample(
        model,
        noise,
        **cond,
        **params,
        verbose=True,
        tqdm_desc=f"Sampling local {stage} Shape Flow",
    ).samples
    seconds = time.perf_counter() - started
    if pipeline.low_vram:
        model.cpu()
    norm = sampled.feats.detach().cpu().float()
    coords = sampled.coords.detach().cpu().int()
    atomic_save(
        output / stage / "local_flow_normalized.pt",
        {"format": FORMAT, "coords": coords, "features": norm, "seed": seed},
    )
    atomic_json(
        output / stage / "flow_summary.json",
        {
            "format": FORMAT,
            "stage": stage,
            "model": model_key,
            "mode": "fresh_noise_complete_flow",
            "blocks": len([r for r in records if len(r["local_xyz"])]),
            "tokens": int(len(coords)),
            "steps": int(steps),
            "seconds": seconds,
            "coordinates": "local zero-based; no baseline latent feature is copied",
            "sampler_params": params,
        },
    )
    del noise, cond, sampled
    empty_cuda()
    return SparseTensor(norm, coords)


def check_and_reorder_batch(
    sampled: SparseTensor,
    records: Sequence[Mapping[str, Any]],
) -> list[torch.Tensor]:
    """Return each batch's feature rows in the record's local-coordinate order."""
    parts: list[torch.Tensor] = []
    active = [record for record in records if len(record["local_xyz"])]
    for batch_id, record in enumerate(active):
        mask = sampled.coords[:, 0].cpu() == batch_id
        actual_coords = sampled.coords[mask].detach().cpu().int()
        actual_feats = sampled.feats[mask].detach().cpu().float()
        expected = record["local_coords"].int()
        if len(actual_coords) != len(expected):
            raise RuntimeError(
                f"batch {batch_id}: Flow changed token count "
                f"{len(expected)} -> {len(actual_coords)}"
            )
        actual_xyz = actual_coords[:, 1:]
        if torch.equal(actual_xyz, expected[:, 1:]):
            parts.append(actual_feats)
            continue
        def key(x: torch.Tensor) -> torch.Tensor:
            return x[:, 0].long() * 1_000_000 + x[:, 1].long() * 1_000 + x[:, 2].long()
        expected_order = torch.argsort(key(expected[:, 1:]))
        actual_order = torch.argsort(key(actual_xyz))
        if not torch.equal(
            actual_xyz[actual_order], expected[:, 1:][expected_order]
        ):
            raise RuntimeError(f"batch {batch_id}: Flow changed local support")
        inverse = torch.empty_like(expected_order)
        inverse[expected_order] = torch.arange(len(expected_order))
        parts.append(actual_feats[actual_order][inverse])
    return parts


@torch.no_grad()
def decode_c32_flow_to_local_c64(
    pipeline: Any,
    sampled_norm: SparseTensor,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    output: Path,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """C32 Flow endpoint -> decoder C512 coordinates -> local C64 support."""
    parts_features = check_and_reorder_batch(sampled_norm, records)
    active = [record for record in records if len(record["local_xyz"])]
    # Rebuild the exact local batch coordinate order.  Features are already
    # aligned to these rows by check_and_reorder_batch.
    coord_parts: list[torch.Tensor] = []
    raw_parts: list[torch.Tensor] = []
    for batch_id, (record, norm_features) in enumerate(zip(active, parts_features)):
        local = record["local_coords"].clone().int()
        local[:, 0] = batch_id
        coord_parts.append(local)
        raw_parts.append(denormalize_shape(norm_features.to(device), pipeline.shape_slat_normalization).cpu())
    packed_coords = torch.cat(coord_parts, dim=0).int()
    packed_raw = torch.cat(raw_parts, dim=0).float()
    slat = SparseTensor(packed_raw.to(device), packed_coords.to(device))
    decoder = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        decoder.to(device)
        decoder.low_vram = True
    print("[stage1] decoder.upsample C32 -> C512", flush=True)
    candidates = decoder.upsample(slat, upsample_times=4)
    if pipeline.low_vram:
        decoder.cpu()
        decoder.low_vram = False
    local_supports: list[torch.Tensor] = []
    generated_records: list[dict[str, Any]] = []
    for batch_id, record in enumerate(active):
        part = candidates[candidates[:, 0] == batch_id]
        if not len(part):
            raise RuntimeError(f"stage1 block {record['cube_id']} decoder returned no C512 rows")
        local = (
            (part[:, 1:].float() + 0.5) / float(STAGE1_DECODER_GRID)
            * float(STAGE1_OUTPUT_LOCAL_GRID - 1)
        ).round().int()
        local = sort_rows(
            torch.cat((torch.zeros((len(local), 1), dtype=torch.int32, device=local.device), local), dim=1)
            .unique(dim=0)
            .cpu()
        )
        local = validate_coords(local, STAGE1_OUTPUT_LOCAL_GRID, f"stage1 local C64 block {record['cube_id']}")
        local_xyz = local[:, 1:].contiguous()
        output_start = torch.tensor(record["output_start"], dtype=torch.int32)
        global_xyz = local_xyz + output_start
        if bool(((global_xyz < 0) | (global_xyz >= GLOBAL_GRID)).any()):
            raise RuntimeError("stage1 generated support escaped global C128")
        global_coords = torch.cat(
            (torch.zeros((len(global_xyz), 1), dtype=torch.int32), global_xyz), dim=1
        )
        local_supports.append(global_coords)
        generated_records.append(
            {
                "cube_id": int(record["cube_id"]),
                "start": record["start"],
                "output_start": record["output_start"],
                "local_xyz": local_xyz,
                "local_coords": local,
                "projection_coords": global_coords,
                "tokens": int(len(local)),
            }
        )
        atomic_save(
            output / "stage1_c32_to_c64" / f"cube_{int(record['cube_id']):02d}_support.pt",
            {
                "format": FORMAT,
                "cube_id": int(record["cube_id"]),
                "input_local_c32": record["local_coords"],
                "output_local_c64": local,
                "output_global_c128": global_coords,
                "decoder_grid": STAGE1_DECODER_GRID,
                "quantization": "round((coord+0.5)/512*63)",
            },
        )
        print(
            f"[stage1 support] cube={record['cube_id']:02d} "
            f"C32={len(record['local_xyz']):,} -> local C64={len(local):,} "
            f"global_offset={record['output_start']}",
            flush=True,
        )
    del slat, candidates, packed_coords, packed_raw
    empty_cuda()
    concatenated = torch.cat(local_supports, dim=0).int()
    global_coords = validate_coords(
        concatenated.unique(dim=0), GLOBAL_GRID, "stage1 global C128 support"
    )
    if len(global_coords) != len(concatenated):
        raise RuntimeError("stage1 block outputs overlap in global C128")
    global_coords = sort_rows(global_coords)
    atomic_save(
        output / "stage1_c32_to_c64" / "global_c128_support.pt",
        {"format": FORMAT, "coords": global_coords},
    )
    atomic_json(
        output / "stage1_c32_to_c64" / "support_summary.json",
        {
            "format": FORMAT,
            "input": "baseline global C64 support split context=32 stride=32",
            "local_coordinates": "0..31 for Shape512 Flow",
            "output": "each local C32 decoder C512 down-quantized to local C64, mapped to global C128",
            "tokens": int(len(global_coords)),
            "blocks": [
                {"cube_id": int(r["cube_id"]), "tokens": int(r["tokens"]), "output_start": r["output_start"]}
                for r in generated_records
            ],
        },
    )
    return global_coords, generated_records


def assemble_global_c128(
    sampled_norm: SparseTensor,
    records: Sequence[Mapping[str, Any]],
    pipeline: Any,
    output: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter local C64 Flow features into the stage1 global C128 support."""
    parts_features = check_and_reorder_batch(sampled_norm, records)
    coord_parts: list[torch.Tensor] = []
    feature_parts: list[torch.Tensor] = []
    active = [record for record in records if len(record["local_xyz"])]
    for record, norm_features in zip(active, parts_features):
        local_xyz = record["local_xyz"].int()
        output_start = torch.tensor(record["output_start"], dtype=torch.int32)
        global_xyz = local_xyz + output_start
        global_coords = torch.cat(
            (torch.zeros((len(global_xyz), 1), dtype=torch.int32), global_xyz), dim=1
        )
        coord_parts.append(global_coords)
        feature_parts.append(
            denormalize_shape(norm_features, pipeline.shape_slat_normalization).float()
        )
    coords = torch.cat(coord_parts, dim=0).int()
    features = torch.cat(feature_parts, dim=0).float()
    if len(coords) != len(coords.unique(dim=0)):
        raise RuntimeError("stage2 local C64 blocks produced duplicate global C128 coordinates")
    order = torch.argsort(
        coords[:, 1].long() * GLOBAL_GRID * GLOBAL_GRID
        + coords[:, 2].long() * GLOBAL_GRID
        + coords[:, 3].long()
    )
    coords = validate_coords(coords[order], GLOBAL_GRID, "final global C128")
    features = features[order].contiguous()
    atomic_save(
        output / "stage2_c64_to_c128" / "global_c128_shape_slat.pt",
        {"format": FORMAT, "coords": coords, "features": features, "normalized": False},
    )
    return coords, features


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


@torch.no_grad()
def render_camera_normals(
    mesh: Any,
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
    device = mesh.device
    result = render_utils.render_frames(
        mesh,
        [extrinsics[angle].to(device) for angle in angles],
        [intrinsics.to(device) for _ in angles],
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
        normal = np.asarray(result["normal"][index])
        mask = np.asarray(result["mask"][index])
        normal_path = render_dir / f"view_{angle:03d}_camera_normal.png"
        mask_path = render_dir / f"view_{angle:03d}_mask.png"
        Image.fromarray(normal).convert("RGB").save(normal_path)
        Image.fromarray(mask).convert("L").save(mask_path)
        paths.append((f"yaw {angle} camera normal", normal_path))
    sheet = render_dir / "camera_normal_contact_sheet.png"
    make_contact_sheet(paths, sheet)
    return {
        "camera_normal_contact_sheet": str(sheet.resolve()),
        "views": [str(path.resolve()) for _, path in paths],
        "masks": [str((render_dir / f"view_{angle:03d}_mask.png").resolve()) for angle in angles],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--baseline-c64", type=Path, default=DEFAULT_BASELINE_C64)
    parser.add_argument("--force-baseline", action="store_true")
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--baseline-seed", type=int, default=42)
    parser.add_argument("--stage1-seed", type=int, default=52001)
    parser.add_argument("--stage2-seed", type=int, default=52002)
    parser.add_argument("--angles", default="0,60,120,180,240,300")
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--render-chunk-size", type=int, default=200_000)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.cuda_device):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r}, expected physical CUDA {args.cuda_device}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    angles = parse_angles(args.angles)
    camera = camera_from_file(args.camera)
    started = time.perf_counter()

    atomic_json(
        output / "config.json",
        {
            "format": FORMAT,
            "status": "running",
            "cuda_device": args.cuda_device,
            "image": args.image,
            "camera": args.camera,
            "steps": args.steps,
            "path": (
                "complete baseline C64 support -> context32/stride32 local C32 fresh Shape512 Flow "
                "-> decoder C512 -> local C64 -> global C128 support -> context64/stride64 local C64 "
                "fresh Shape1024 Flow -> global C128 -> geometry decode2048"
            ),
            "baseline_features_policy": "baseline C64 features are never copied into either local Flow",
            "stage1_coordinate_policy": "global C64 block coords minus start -> local 0..31; output local C64 offset by start*2 into C128",
            "stage2_coordinate_policy": "global C128 block coords minus start -> local 0..63; output offset by start",
            "condition_policy": "physical global support projected to 2048; per-block 1024 crop; DINO/NAF re-extracted",
            "texture": "not loaded, sampled, or decoded",
        },
    )

    print("[model] loading Pixal3D pipeline (geometry stages only are executed)", flush=True)
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=True)
    canonical = pipeline.preprocess_canonical_images(Image.open(args.image).convert("RGB"))
    image_4096 = canonical["image_4096"].convert("RGB")
    image_2048 = image_4096.resize(
        (CONDITION_IMAGE_GRID, CONDITION_IMAGE_GRID), Image.Resampling.LANCZOS
    )
    input_dir = output / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    image_4096.save(input_dir / "canonical_4096.png")
    image_2048.save(input_dir / "canonical_2048.png")
    canonical["image_512"].save(input_dir / "canonical_512.png")
    canonical["image_1024"].save(input_dir / "canonical_1024.png")
    atomic_json(output / "camera.json", camera)

    baseline_path = output / "baseline" / "shape_c64_baseline.pt"
    if args.baseline_c64.is_file() and not args.force_baseline:
        print(f"[baseline] reuse complete C64 shape SLat support: {args.baseline_c64}", flush=True)
        baseline_coords, baseline_features = load_baseline_c64(args.baseline_c64)
        baseline_source = str(args.baseline_c64.resolve())
    else:
        baseline_coords, baseline_features = generate_baseline_c64(
            pipeline, canonical, camera, device, args.steps, args.baseline_seed
        )
        baseline_source = "generated_in_this_run"
    atomic_save(
        baseline_path,
        {
            "format": FORMAT,
            "coords": baseline_coords,
            "features": baseline_features,
            "normalized": False,
            "source": baseline_source,
            "features_used_by_stage1": False,
        },
    )
    atomic_json(
        output / "baseline" / "summary.json",
        {
            "format": FORMAT,
            "source": baseline_source,
            "complete_baseline_flow": True,
            "c64_tokens": int(len(baseline_coords)),
            "features_loaded": baseline_features is not None,
            "features_used_by_local_flow": False,
        },
    )
    print(f"[baseline] C64 tokens={len(baseline_coords):,}; features are support-only for this cascade", flush=True)

    # Stage 1: C64 support -> local C32 Flow -> decoder C512 -> local C64 -> C128 support.
    stage1_records = make_blocks(
        baseline_coords,
        input_grid=BASELINE_GRID,
        context=STAGE1_CONTEXT,
        output_grid=GLOBAL_GRID,
        name="stage1 baseline C64 split",
    )
    atomic_json(
        output / "stage1_c32_to_c64" / "block_layout.json",
        {
            "format": FORMAT,
            "input_grid": BASELINE_GRID,
            "context": STAGE1_CONTEXT,
            "stride": STAGE1_CONTEXT,
            "local_flow_grid": STAGE1_LOCAL_GRID,
            "output_decoder_grid": STAGE1_DECODER_GRID,
            "output_local_grid": STAGE1_OUTPUT_LOCAL_GRID,
            "global_output_grid": GLOBAL_GRID,
            "blocks": [
                {
                    "cube_id": int(r["cube_id"]),
                    "start": r["start"],
                    "output_start": r["output_start"],
                    "tokens": int(r["tokens"]),
                }
                for r in stage1_records
            ],
        },
    )
    stage1_conditions = extract_block_conditions(
        pipeline, image_2048, camera, stage1_records, BASELINE_GRID, "c32", output
    )
    stage1_flow = run_fresh_local_flow(
        pipeline,
        stage1_records,
        stage1_conditions,
        "shape_slat_flow_model_512",
        device,
        args.steps,
        args.stage1_seed,
        "stage1_c32_to_c64",
        output,
    )
    stage1_global_c128, stage1_records_out = decode_c32_flow_to_local_c64(
        pipeline, stage1_flow, stage1_records, device, output
    )
    del stage1_conditions, stage1_flow, stage1_records
    empty_cuda()

    # Stage 2: global C128 support -> local C64 Flow -> assembled global C128 SLat.
    stage2_records = make_blocks(
        stage1_global_c128,
        input_grid=GLOBAL_GRID,
        context=STAGE2_CONTEXT,
        output_grid=GLOBAL_GRID,
        name="stage2 generated C128 split",
    )
    atomic_json(
        output / "stage2_c64_to_c128" / "block_layout.json",
        {
            "format": FORMAT,
            "input_grid": GLOBAL_GRID,
            "context": STAGE2_CONTEXT,
            "stride": STAGE2_CONTEXT,
            "local_flow_grid": STAGE2_LOCAL_GRID,
            "global_output_grid": GLOBAL_GRID,
            "blocks": [
                {
                    "cube_id": int(r["cube_id"]),
                    "start": r["start"],
                    "output_start": r["output_start"],
                    "tokens": int(r["tokens"]),
                }
                for r in stage2_records
            ],
        },
    )
    stage2_conditions = extract_block_conditions(
        pipeline, image_2048, camera, stage2_records, GLOBAL_GRID, "c64", output
    )
    stage2_flow = run_fresh_local_flow(
        pipeline,
        stage2_records,
        stage2_conditions,
        "shape_slat_flow_model_1024",
        device,
        args.steps,
        args.stage2_seed,
        "stage2_c64_to_c128",
        output,
    )
    final_coords, final_features = assemble_global_c128(
        stage2_flow, stage2_records, pipeline, output
    )
    del stage2_conditions, stage2_flow, stage2_records, stage1_records_out
    empty_cuda()

    print(
        f"[decode] global C128 tokens={len(final_coords):,} -> geometry at {DECODE_RESOLUTION}",
        flush=True,
    )
    shape_slat = SparseTensor(final_features.to(device), final_coords.to(device))
    meshes, _ = pipeline.decode_shape_slat(shape_slat, DECODE_RESOLUTION)
    if len(meshes) != 1:
        raise RuntimeError(f"expected one decoded mesh, got {len(meshes)}")
    mesh = meshes[0]
    final_dir = output / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = final_dir / "geometry_mesh.pt"
    atomic_save(
        mesh_path,
        {
            "format": FORMAT,
            "mesh": mesh.cpu(),
            "shape_coords": final_coords,
            "shape_features": final_features,
        },
    )
    glb_path = final_dir / "geometry_mesh.glb"
    glb_error: str | None = None
    try:
        import trimesh

        trimesh.Trimesh(
            vertices=mesh.vertices.detach().cpu().numpy(),
            faces=mesh.faces.detach().cpu().numpy(),
            process=False,
        ).export(glb_path)
    except Exception as exc:  # pragma: no cover - optional export dependency/path
        glb_error = repr(exc)
        atomic_json(final_dir / "glb_export_error.json", {"error": glb_error})

    render = render_camera_normals(
        mesh,
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
        "baseline_c64_tokens": int(len(baseline_coords)),
        "stage1_global_c128_tokens": int(len(stage1_global_c128)),
        "final_global_c128_tokens": int(len(final_coords)),
        "mesh_vertices": int(mesh.vertices.shape[0]),
        "mesh_faces": int(mesh.faces.shape[0]),
        "mesh_pt": str(mesh_path.resolve()),
        "mesh_glb": str(glb_path.resolve()) if glb_error is None else None,
        "glb_error": glb_error,
        "camera_normal_render": render,
        "seconds": time.perf_counter() - started,
        "cuda_device": int(args.cuda_device),
        "baseline_features_used": False,
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "config.json", {**load_json(output / "config.json"), "status": "complete"})
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), flush=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
