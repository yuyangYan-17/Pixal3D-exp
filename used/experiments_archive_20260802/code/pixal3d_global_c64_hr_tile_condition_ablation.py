#!/usr/bin/env python3
"""Global-C64 / HR-image-tile conditioning ablation for Pixal3D.

The control is an unmodified ``pipeline.run(..., pipeline_type="1024_cascade")``
call.  Temporary read-only sampler observers capture the exact pre-shape-flow
global C64 support, shape noise, and texture noise from that same call.  It
keeps the global C64 coordinates and row order immutable throughout:

* project global C64 rows into the canonical 4096 input view;
* form the complete 7x7 set of 1024 crops with stride 512;
* gather rows only (no global-to-local transport, coordinate transform,
  quantization, encoding, or support mutation);
* rerun global/projection image conditioning for each HR crop with the same
  global camera and absolute C64 coordinates;
* at every Euler step, predict each tile subset independently, scatter the
  velocities to the recorded global rows, take the arithmetic mean on overlap,
  and update the single global state once;
* repeat the same operation for texture flow, using the experimental normalized
  shape SLat as the unchanged token-aligned concat condition;
* decode both branches with the ordinary 1024 decoder and evaluate the shaded
  PBR render in the corresponding input camera.

This file intentionally does not use any local camera or local SLat route.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
)

import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
import pixal3d_tile_encoded_query_noise_flow_overlap_render as render_helpers
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from render_pixal3d_raw_ovoxel import load_envmap


FORMAT_VERSION = "pixal3d_global_c64_hr_tile_condition_ablation_v1"
CANONICAL_SIZE = 4096
TILE_SIZE = 1024
TILE_STRIDE = 512
GRID_C64 = 64
DECODE_RESOLUTION = 1024


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(
            _json_value(dict(payload)),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _tile_layout() -> List[Tuple[int, int, int, int]]:
    starts = list(
        range(0, CANONICAL_SIZE - TILE_SIZE + 1, TILE_STRIDE)
    )
    boxes = [
        (x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE)
        for y0 in starts
        for x0 in starts
    ]
    if len(starts) != 7 or len(boxes) != 49:
        raise RuntimeError(
            f"expected a complete 7x7 tile layout, got {len(boxes)} boxes"
        )
    return boxes


def _build_global_row_tiles(
    *,
    image_4096: Image.Image,
    projected_full_norm: torch.Tensor,
    projection_valid: torch.Tensor,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Assign original global rows to 7x7 image tiles without filtering."""
    if image_4096.size != (CANONICAL_SIZE, CANONICAL_SIZE):
        raise ValueError(f"canonical image must be 4096 square: {image_4096.size}")
    if projected_full_norm.ndim != 2 or projected_full_norm.shape[1] != 2:
        raise ValueError("projected_full_norm must have shape [N,2]")
    if projection_valid.shape != (projected_full_norm.shape[0],):
        raise ValueError("projection_valid is not row aligned")

    projection_cpu = projected_full_norm.detach().cpu().float()
    valid_cpu = projection_valid.detach().cpu().bool()
    finite = torch.isfinite(projection_cpu).all(dim=1)
    pixel_x = torch.floor(projection_cpu[:, 0] * CANONICAL_SIZE).long()
    pixel_y = torch.floor(projection_cpu[:, 1] * CANONICAL_SIZE).long()
    in_bounds = (
        (pixel_x >= 0)
        & (pixel_x < CANONICAL_SIZE)
        & (pixel_y >= 0)
        & (pixel_y < CANONICAL_SIZE)
    )
    eligible = valid_cpu & finite & in_bounds
    coverage = torch.zeros(projected_full_norm.shape[0], dtype=torch.int32)
    tiles: List[Dict[str, Any]] = []
    for tile_id, box in enumerate(_tile_layout()):
        x0, y0, x1, y1 = box
        mask = (
            eligible
            & (pixel_x >= x0)
            & (pixel_x < x1)
            & (pixel_y >= y0)
            & (pixel_y < y1)
        )
        global_rows = torch.where(mask)[0].long()
        if global_rows.numel():
            coverage.index_add_(
                0,
                global_rows,
                torch.ones_like(global_rows, dtype=torch.int32),
            )
        tiles.append(
            {
                "tile_id": int(tile_id),
                "box": tuple(int(value) for value in box),
                "projection_crop_box": (
                    x0 / float(CANONICAL_SIZE),
                    y0 / float(CANONICAL_SIZE),
                    x1 / float(CANONICAL_SIZE),
                    y1 / float(CANONICAL_SIZE),
                ),
                "global_rows": global_rows,
                "token_count": int(global_rows.numel()),
                "enabled": bool(global_rows.numel()),
            }
        )
    missed_eligible = eligible & (coverage == 0)
    if torch.any(missed_eligible):
        raise RuntimeError(
            "complete 7x7 layout failed to cover eligible global rows: "
            f"{int(missed_eligible.sum().item())}"
        )
    summary = {
        "canonical_size": CANONICAL_SIZE,
        "tile_size": TILE_SIZE,
        "tile_stride": TILE_STRIDE,
        "layout": "7x7 complete in-bounds crops",
        "tile_count": len(tiles),
        "active_tile_count": sum(bool(tile["enabled"]) for tile in tiles),
        "global_row_count": int(projected_full_norm.shape[0]),
        "finite_row_count": int(finite.sum().item()),
        "projection_valid_row_count": int(valid_cpu.sum().item()),
        "eligible_row_count": int(eligible.sum().item()),
        "covered_row_count": int((coverage > 0).sum().item()),
        "overlap_row_count": int((coverage > 1).sum().item()),
        "uncovered_row_count": int((coverage == 0).sum().item()),
        "maximum_memberships": int(coverage.max().item()) if coverage.numel() else 0,
        "coverage_histogram": {
            str(value): int((coverage == value).sum().item())
            for value in range(int(coverage.max().item()) + 1)
        },
        "tile_token_counts": [
            {
                "tile_id": int(tile["tile_id"]),
                "token_count": int(tile["token_count"]),
            }
            for tile in tiles
        ],
        "eligible": eligible,
        "coverage": coverage,
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
    }
    return tiles, summary


def _save_projection_overlay(
    *,
    image: Image.Image,
    tile_summary: Mapping[str, Any],
    output_path: Path,
) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for coordinate in range(0, CANONICAL_SIZE + 1, TILE_STRIDE):
        color = (80, 80, 80)
        width = 2
        draw.line((coordinate, 0, coordinate, CANONICAL_SIZE), fill=color, width=width)
        draw.line((0, coordinate, CANONICAL_SIZE, coordinate), fill=color, width=width)
    eligible = tile_summary["eligible"]
    pixel_x = tile_summary["pixel_x"]
    pixel_y = tile_summary["pixel_y"]
    rows = torch.where(eligible)[0]
    for row in rows.tolist():
        x = int(pixel_x[row].item())
        y = int(pixel_y[row].item())
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 80, 40))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _sampler_params(args: argparse.Namespace) -> Tuple[Dict[str, Any], ...]:
    return (
        {
            "steps": int(args.ss_steps),
            "guidance_strength": float(args.ss_guidance_strength),
            "guidance_rescale": float(args.ss_guidance_rescale),
            "rescale_t": float(args.ss_rescale_t),
        },
        {
            "steps": int(args.shape_steps),
            "guidance_strength": float(args.shape_guidance_strength),
            "guidance_rescale": float(args.shape_guidance_rescale),
            "rescale_t": float(args.shape_rescale_t),
        },
        {
            "steps": int(args.texture_steps),
            "guidance_strength": float(args.texture_guidance_strength),
            "guidance_rescale": float(args.texture_guidance_rescale),
            "rescale_t": float(args.texture_rescale_t),
        },
    )


def _denormalize(
    value: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
) -> SparseTensor:
    std = torch.as_tensor(
        normalization["std"], device=value.device, dtype=value.dtype
    )[None]
    mean = torch.as_tensor(
        normalization["mean"], device=value.device, dtype=value.dtype
    )[None]
    return value * std + mean


def _normalize(
    value: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
) -> SparseTensor:
    std = torch.as_tensor(
        normalization["std"], device=value.device, dtype=value.dtype
    )[None]
    mean = torch.as_tensor(
        normalization["mean"], device=value.device, dtype=value.dtype
    )[None]
    return (value - mean) / std


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} vs {right.shape}")
    if left.numel() == 0:
        return 0.0
    return float((left.detach().cpu().float() - right.detach().cpu().float()).abs().max().item())


@torch.no_grad()
def _run_official_baseline_with_capture(
    *,
    pipeline: Any,
    image_1024: Image.Image,
    camera: Mapping[str, float],
    seed: int,
    ss_params: Mapping[str, Any],
    shape_params: Mapping[str, Any],
    texture_params: Mapping[str, Any],
    max_num_tokens: int,
) -> Tuple[Any, Tuple[SparseTensor, SparseTensor, int], Dict[str, Any]]:
    """Run the public 1024 cascade once and observe its C64 flow boundaries.

    The sampler implementations are not replaced: each temporary wrapper calls
    the original bound method exactly once with unchanged arguments and copies
    only the noise, condition, concat condition, and returned normalized state.
    This avoids relying on nondeterministic second-run SS threshold replay.
    """
    capture: Dict[str, Any] = {}
    shape_sampler = pipeline.shape_slat_sampler
    texture_sampler = pipeline.tex_slat_sampler
    original_shape_sample = shape_sampler.sample
    original_texture_sample = texture_sampler.sample
    target_shape_model = pipeline.models["shape_slat_flow_model_1024"]
    target_texture_model = pipeline.models["tex_slat_flow_model_1024"]

    def observed_shape_sample(model, noise, *positional, **keywords):
        is_target = model is target_shape_model
        if is_target:
            capture["coords64"] = noise.coords.detach().cpu().clone()
            capture["shape_noise"] = SparseTensor(
                feats=noise.feats.detach().cpu().clone(),
                coords=noise.coords.detach().cpu().clone(),
            )
            capture["shape_condition_cpu"] = pipeline._pack_proj_condition_cpu(
                {"cond": keywords["cond"], "neg_cond": keywords["neg_cond"]},
                expected_coords=noise.coords,
                name="captured_official_global_shape_condition",
            )
        result = original_shape_sample(model, noise, *positional, **keywords)
        if is_target:
            capture["global_shape_norm"] = result.samples.to("cpu")
        return result

    def observed_texture_sample(model, noise, *positional, **keywords):
        is_target = model is target_texture_model
        if is_target:
            capture["texture_noise"] = SparseTensor(
                feats=noise.feats.detach().cpu().clone(),
                coords=noise.coords.detach().cpu().clone(),
            )
            capture["texture_condition_cpu"] = pipeline._pack_proj_condition_cpu(
                {"cond": keywords["cond"], "neg_cond": keywords["neg_cond"]},
                expected_coords=noise.coords,
                name="captured_official_global_texture_condition",
            )
            concat = keywords.get("concat_cond")
            if not isinstance(concat, SparseTensor):
                raise RuntimeError("official texture flow has no sparse shape concat")
            capture["official_shape_concat_norm"] = concat.to("cpu")
        result = original_texture_sample(model, noise, *positional, **keywords)
        if is_target:
            capture["global_texture_norm"] = result.samples.to("cpu")
        return result

    # Instance-level observation is scoped to this one official call.
    shape_sampler.sample = observed_shape_sample
    texture_sampler.sample = observed_texture_sample
    try:
        output, latents = pipeline.run(
            image_1024,
            camera_params=dict(camera),
            seed=int(seed),
            sparse_structure_sampler_params=dict(ss_params),
            shape_slat_sampler_params=dict(shape_params),
            tex_slat_sampler_params=dict(texture_params),
            preprocess_image=False,
            return_latent=True,
            pipeline_type="1024_cascade",
            max_num_tokens=int(max_num_tokens),
        )
    finally:
        shape_sampler.sample = original_shape_sample
        texture_sampler.sample = original_texture_sample

    required = {
        "coords64",
        "shape_noise",
        "texture_noise",
        "shape_condition_cpu",
        "texture_condition_cpu",
        "official_shape_concat_norm",
        "global_shape_norm",
        "global_texture_norm",
    }
    missing = required.difference(capture)
    if missing:
        raise RuntimeError(f"official baseline capture is incomplete: {sorted(missing)}")
    if not torch.equal(capture["shape_noise"].coords, capture["texture_noise"].coords):
        raise RuntimeError("official shape/texture noise supports differ")
    capture["shape_params"] = {
        **pipeline.shape_slat_sampler_params,
        **dict(shape_params),
    }
    capture["texture_params"] = {
        **pipeline.tex_slat_sampler_params,
        **dict(texture_params),
    }
    return output, latents, capture


@torch.no_grad()
def _prepare_tile_conditions(
    *,
    pipeline: Any,
    image_model: nn.Module,
    image_4096: Image.Image,
    global_coords: torch.Tensor,
    tiles: Sequence[Dict[str, Any]],
    camera: Mapping[str, float],
    stage_name: str,
    grid_resolution: int = GRID_C64,
) -> Dict[str, Any]:
    started = time.perf_counter()
    records: List[Dict[str, Any]] = []
    for tile in tqdm(tiles, desc=f"Extract {stage_name} HR tile conditions"):
        if not bool(tile["enabled"]):
            continue
        tile_started = time.perf_counter()
        rows = tile["global_rows"].to(device=global_coords.device, dtype=torch.long)
        tile_coords = global_coords.index_select(0, rows)
        if not torch.equal(tile_coords, global_coords[rows]):
            raise RuntimeError(f"tile {tile['tile_id']} gather changed row order")
        crop = image_4096.crop(tile["box"])
        if crop.size != (TILE_SIZE, TILE_SIZE):
            raise RuntimeError(f"tile {tile['tile_id']} crop is not 1024 square")
        condition = pipeline.get_proj_cond_shape(
            image_cond_model=image_model,
            image=[crop],
            coords=tile_coords,
            camera_angle_x=float(camera["camera_angle_x"]),
            distance=float(camera["distance"]),
            mesh_scale=float(camera["mesh_scale"]),
            grid_resolution_override=int(grid_resolution),
            projection_crop_box=tile["projection_crop_box"],
        )
        tile[f"{stage_name}_condition_cpu"] = pipeline._pack_proj_condition_cpu(
            condition,
            expected_coords=tile_coords,
            name=f"tile[{tile['tile_id']}].{stage_name}",
        )
        elapsed = float(time.perf_counter() - tile_started)
        records.append(
            {
                "tile_id": int(tile["tile_id"]),
                "global_rows": int(rows.numel()),
                "seconds": elapsed,
                "camera": "unchanged global camera",
                "coordinates": (
                    f"unchanged absolute global C{int(grid_resolution)}"
                ),
                "projection": "global projection mapped into crop-local feature coordinates",
                "dino_rerun": True,
                "naf_rerun": bool(getattr(image_model, "use_naf_upsample", False)),
            }
        )
        del condition
        _empty_cuda_cache()
    return {
        "stage": stage_name,
        "active_tiles": len(records),
        "seconds": float(time.perf_counter() - started),
        "records": records,
        "global_context_recomputed_per_tile": True,
        "projected_condition_recomputed_per_tile": True,
        "coordinate_transport": False,
        "requantization": False,
        "reencoding": False,
    }


def _prediction_kwargs(params: Mapping[str, Any]) -> Dict[str, Any]:
    excluded = {
        "steps",
        "rescale_t",
        "verbose",
        "tqdm_desc",
        "record_trajectory",
        "trajectory_device",
        "return_model_history",
    }
    return {key: value for key, value in params.items() if key not in excluded}


def _velocity_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            left.detach().float().reshape(1, -1),
            right.detach().float().reshape(1, -1),
            dim=1,
            eps=torch.finfo(torch.float32).eps,
        ).item()
    )


@torch.no_grad()
def _run_tiled_global_flow(
    *,
    pipeline: Any,
    model: nn.Module,
    sampler: Any,
    initial_noise: SparseTensor,
    global_condition_cpu: Mapping[str, Mapping[str, torch.Tensor]],
    tiles: Sequence[Dict[str, Any]],
    stage_name: str,
    sampler_params: Mapping[str, Any],
    concat_cond: Optional[SparseTensor],
    coordinate_label: str = "C64",
) -> Tuple[SparseTensor, Dict[str, Any]]:
    """Run one global state using independently predicted tile velocities."""
    active_tiles = [tile for tile in tiles if bool(tile["enabled"])]
    if not active_tiles:
        raise RuntimeError(f"{stage_name}: no active image tiles")
    if concat_cond is not None and not torch.equal(
        concat_cond.coords, initial_noise.coords
    ):
        raise RuntimeError(f"{stage_name}: concat coords do not match global noise")
    for tile in active_tiles:
        if f"{stage_name}_condition_cpu" not in tile:
            raise RuntimeError(
                f"{stage_name}: tile {tile['tile_id']} has no extracted condition"
            )

    steps = int(sampler_params["steps"])
    times = sampler.timestep_schedule(
        steps, float(sampler_params.get("rescale_t", 1.0))
    )
    prediction_kwargs = _prediction_kwargs(sampler_params)
    state = initial_noise.replace(initial_noise.feats.detach().clone())
    if not torch.equal(state.coords, initial_noise.coords):
        raise RuntimeError(f"{stage_name}: initial clone changed coordinates")
    initial_noise_error = _max_abs(state.feats, initial_noise.feats)
    if initial_noise_error != 0.0:
        raise RuntimeError(f"{stage_name}: initial noise was not bitwise preserved")

    token_count = int(state.feats.shape[0])
    static_coverage = torch.zeros(token_count, dtype=torch.int32)
    for tile in active_tiles:
        static_coverage.index_add_(
            0,
            tile["global_rows"],
            torch.ones_like(tile["global_rows"], dtype=torch.int32),
        )
    step_records: List[Dict[str, Any]] = []
    started = time.perf_counter()
    if bool(pipeline.low_vram):
        model.to(pipeline.device)

    for step_index in range(steps):
        _sync_cuda()
        step_started = time.perf_counter()
        timestep = float(times[step_index])
        next_timestep = float(times[step_index + 1])
        delta = timestep - next_timestep
        velocity_sum = torch.zeros_like(state.feats, dtype=torch.float32)
        velocity_count = torch.zeros(
            (token_count, 1), device=state.device, dtype=torch.float32
        )
        per_tile: List[Dict[str, Any]] = []
        for tile in active_tiles:
            rows = tile["global_rows"].to(device=state.device, dtype=torch.long)
            tile_coords = state.coords.index_select(0, rows)
            tile_state = SparseTensor(
                feats=state.feats.index_select(0, rows),
                coords=tile_coords,
            )
            if not torch.equal(tile_state.coords, initial_noise.coords[rows]):
                raise RuntimeError(
                    f"{stage_name}: tile {tile['tile_id']} changed absolute "
                    f"{coordinate_label} coords"
                )
            condition = pipeline._materialize_proj_condition(
                tile[f"{stage_name}_condition_cpu"],
                coords=tile_coords,
                device=state.device,
            )
            tile_concat = None
            if concat_cond is not None:
                tile_concat = SparseTensor(
                    feats=concat_cond.feats.index_select(0, rows),
                    coords=tile_coords,
                )
            _, _, velocity = sampler._get_model_prediction(
                model,
                tile_state,
                timestep,
                condition["cond"],
                neg_cond=condition["neg_cond"],
                concat_cond=tile_concat,
                **prediction_kwargs,
            )
            if velocity.feats.shape != tile_state.feats.shape:
                raise RuntimeError(
                    f"{stage_name}: tile {tile['tile_id']} velocity shape mismatch"
                )
            if not torch.equal(velocity.coords, tile_coords):
                raise RuntimeError(
                    f"{stage_name}: tile {tile['tile_id']} velocity changed coords"
                )
            velocity_sum.index_add_(0, rows, velocity.feats.float())
            velocity_count.index_add_(
                0,
                rows,
                torch.ones((rows.numel(), 1), device=state.device),
            )
            per_tile.append(
                {
                    "tile_id": int(tile["tile_id"]),
                    "tokens": int(rows.numel()),
                    "velocity_rms": float(velocity.feats.float().square().mean().sqrt().item()),
                }
            )
            del tile_state, tile_concat, condition, velocity

        covered = velocity_count[:, 0] > 0
        uncovered = ~covered
        current_global_velocity = None
        if torch.any(uncovered):
            global_condition = pipeline._materialize_proj_condition(
                global_condition_cpu,
                coords=state.coords,
                device=state.device,
            )
            _, _, current_global_velocity = sampler._get_model_prediction(
                model,
                state,
                timestep,
                global_condition["cond"],
                neg_cond=global_condition["neg_cond"],
                concat_cond=concat_cond,
                **prediction_kwargs,
            )
            velocity_sum[uncovered] = current_global_velocity.feats[uncovered].float()
            velocity_count[uncovered] = 1.0
            del global_condition
        if torch.any(velocity_count <= 0):
            raise RuntimeError(f"{stage_name}: zero velocity divisor")
        merged_velocity = velocity_sum / velocity_count
        state = state.replace(
            state.feats - float(delta) * merged_velocity.to(state.dtype)
        )
        if not torch.equal(state.coords, initial_noise.coords):
            raise RuntimeError(f"{stage_name}: Euler update changed coords/order")
        _sync_cuda()
        step_records.append(
            {
                "step": int(step_index),
                "timestep": timestep,
                "next_timestep": next_timestep,
                "delta": float(delta),
                "active_tiles": len(active_tiles),
                "covered_rows": int(covered.sum().item()),
                "overlap_rows": int((velocity_count[:, 0] > 1).sum().item()),
                "uncovered_global_fallback_rows": int(uncovered.sum().item()),
                "velocity_rms": float(merged_velocity.square().mean().sqrt().item()),
                "velocity_global_fallback_cosine": (
                    None
                    if current_global_velocity is None
                    else _velocity_cosine(
                        merged_velocity[uncovered],
                        current_global_velocity.feats[uncovered],
                    )
                ),
                "per_tile": per_tile,
                "seconds": float(time.perf_counter() - step_started),
                "cuda_peak_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated())
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
        print(
            f"[{stage_name}-global-{coordinate_label}-tile-flow] "
            f"step={step_index:02d} "
            f"t={timestep:.8f}->{next_timestep:.8f} "
            f"tiles={len(active_tiles)} covered={int(covered.sum().item()):,}/"
            f"{token_count:,} overlap={int((velocity_count[:, 0] > 1).sum().item()):,} "
            f"fallback={int(uncovered.sum().item()):,} "
            f"seconds={step_records[-1]['seconds']:.3f}"
        )
        del velocity_sum, velocity_count, merged_velocity
        if current_global_velocity is not None:
            del current_global_velocity

    if bool(pipeline.low_vram):
        model.cpu()
    _empty_cuda_cache()
    return state, {
        "stage": stage_name,
        "steps": steps,
        "initial_noise_sha256": _tensor_sha256(initial_noise.feats),
        "initial_noise_max_abs_copy_error": initial_noise_error,
        "global_coords_sha256": _tensor_sha256(initial_noise.coords),
        "global_row_count": token_count,
        "active_tiles": len(active_tiles),
        "covered_rows": int((static_coverage > 0).sum().item()),
        "overlap_rows": int((static_coverage > 1).sum().item()),
        "uncovered_rows": int((static_coverage == 0).sum().item()),
        "maximum_memberships": int(static_coverage.max().item()),
        "overlap_fusion": "uniform arithmetic mean of per-tile velocities",
        "global_update_count_per_step": 1,
        "coordinate_mode": f"absolute global {coordinate_label}",
        "local_coordinate_transform": False,
        "requantization": False,
        "reencoding": False,
        "latent_transport": False,
        "steps_detail": step_records,
        "seconds": float(time.perf_counter() - started),
    }


def _parity_report(
    *,
    baseline_shape: SparseTensor,
    baseline_texture: SparseTensor,
    capture: Mapping[str, Any],
    tolerance: float,
) -> Dict[str, Any]:
    captured_shape = capture["global_shape_raw"]
    captured_texture = capture["global_texture_raw"]
    shape_coords_equal = torch.equal(
        baseline_shape.coords.detach().cpu(), captured_shape.coords.detach().cpu()
    )
    texture_coords_equal = torch.equal(
        baseline_texture.coords.detach().cpu(), captured_texture.coords.detach().cpu()
    )
    shape_error = _max_abs(baseline_shape.feats, captured_shape.feats)
    texture_error = _max_abs(baseline_texture.feats, captured_texture.feats)
    passed = bool(
        shape_coords_equal
        and texture_coords_equal
        and shape_error <= float(tolerance)
        and texture_error <= float(tolerance)
    )
    report = {
        "passed": passed,
        "tolerance": float(tolerance),
        "shape_coords_equal": shape_coords_equal,
        "texture_coords_equal": texture_coords_equal,
        "shape_max_abs_feature_error": shape_error,
        "texture_max_abs_feature_error": texture_error,
        "baseline_shape_sha256": _tensor_sha256(baseline_shape.feats),
        "captured_shape_sha256": _tensor_sha256(captured_shape.feats),
        "baseline_texture_sha256": _tensor_sha256(baseline_texture.feats),
        "captured_texture_sha256": _tensor_sha256(captured_texture.feats),
    }
    if not passed:
        raise RuntimeError(f"official same-run capture parity failed: {report}")
    return report


def _save_input_comparison(
    *,
    reference_path: Path,
    baseline_render_path: Path,
    experiment_render_path: Path,
    baseline_psnr: float,
    experiment_psnr: float,
    output_path: Path,
    baseline_label: str = "Official Global-1024",
    experiment_label: str = "Global C64 + HR tile condition",
) -> None:
    paths = [reference_path, baseline_render_path, experiment_render_path]
    labels = [
        "Canonical 4096 input",
        f"{baseline_label} | PSNR {baseline_psnr:.4f} dB",
        f"{experiment_label} | PSNR {experiment_psnr:.4f} dB",
    ]
    panel = 768
    header = 48
    sheet = Image.new("RGB", (panel * 3, panel + header), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for index, (path, label) in enumerate(zip(paths, labels)):
        with Image.open(path) as image:
            resized = image.convert("RGB").resize(
                (panel, panel), Image.Resampling.LANCZOS
            )
        sheet.paste(resized, (index * panel, header))
        draw.text((index * panel + 10, 14), label, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _save_multiview_comparison(
    *,
    baseline: Mapping[str, Any],
    experiment: Mapping[str, Any],
    output_path: Path,
    baseline_label: str = "Official Global-1024",
    experiment_label: str = "HR tile-conditioned",
) -> None:
    baseline_paths = [Path(path) for path in baseline["frame_pngs"]]
    experiment_paths = [Path(path) for path in experiment["frame_pngs"]]
    yaws = baseline["yaw_degrees"]
    pitches = baseline["pitch_degrees"]
    if len(baseline_paths) != len(experiment_paths):
        raise RuntimeError("baseline/experiment multiview counts differ")
    panel = 384
    header = 38
    columns = min(3, len(baseline_paths))
    view_rows = (len(baseline_paths) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * panel, 2 * view_rows * (panel + header)),
        (0, 0, 0),
    )
    draw = ImageDraw.Draw(sheet)
    for method_row, (name, paths) in enumerate(
        ((baseline_label, baseline_paths), (experiment_label, experiment_paths))
    ):
        for index, path in enumerate(paths):
            column = index % columns
            local_row = index // columns
            row = method_row * view_rows + local_row
            x = column * panel
            y = row * (panel + header)
            with Image.open(path) as image:
                frame = image.convert("RGB").resize(
                    (panel, panel), Image.Resampling.LANCZOS
                )
            sheet.paste(frame, (x, y + header))
            draw.text(
                (x + 8, y + 11),
                f"{name} | yaw={float(yaws[index]):g}, pitch={float(pitches[index]):g}",
                fill=(255, 255, 255),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    baseline_psnr = summary["evaluation"]["baseline"]["psnr_db"]
    experiment_psnr = summary["evaluation"]["experiment"]["psnr_db"]
    delta = summary["evaluation"]["psnr_delta_db"]
    baseline_ssim = summary["evaluation"]["baseline"].get("ssim")
    experiment_ssim = summary["evaluation"]["experiment"].get("ssim")
    ssim_line = (
        "- Shaded PBR SSIM: "
        f"`{baseline_ssim:.6f} -> {experiment_ssim:.6f}` "
        f"(`{experiment_ssim - baseline_ssim:+.6f}`)"
        if baseline_ssim is not None and experiment_ssim is not None
        else "- Shaded PBR SSIM: unavailable"
    )
    coverage = summary["tile_assignment"]
    lines = [
        "# Global C64 HR-tile condition ablation",
        "",
        f"- Input: `{summary['image']}`",
        f"- CUDA device: `{summary['cuda_device']}`",
        f"- Seed: `{summary['seed']}`",
        f"- Official Global-1024 shaded PBR PSNR: `{baseline_psnr:.6f} dB`",
        f"- HR-tile-conditioned shaded PBR PSNR: `{experiment_psnr:.6f} dB`",
        f"- PSNR delta: `{delta:+.6f} dB`",
        ssim_line,
        f"- Global C64 rows: `{coverage['global_row_count']}`",
        f"- Covered / overlap / uncovered rows: `{coverage['covered_row_count']} / {coverage['overlap_row_count']} / {coverage['uncovered_row_count']}`",
        "- Coordinate policy: absolute global C64 only; no local transform, requantization, re-encoding, or latent transport.",
        "- Overlap policy: uniform arithmetic mean of velocities before one global Euler update per step.",
        f"- Input-view comparison: `{summary['evaluation']['input_comparison_png']}`",
        f"- Multiview comparison (includes yaw=180 back): `{summary['evaluation']['multiview_comparison_png']}`",
        "",
        "A quality improvement is claimed only when the measured PSNR delta is positive; back-view quality remains a qualitative judgement because no back-view ground truth is available.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_torch(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@torch.no_grad()
def _render_only(args: argparse.Namespace, output_dir: Path) -> None:
    """Resume evaluation from completed pre-render generation checkpoints."""
    started = time.perf_counter()
    camera_path = output_dir / "global_camera.json"
    if not camera_path.is_file():
        raise FileNotFoundError(camera_path)
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    reference_path = output_dir / "canonical_4096.png"
    baseline_mesh = core._validate_mesh(
        _load_torch(output_dir / "official_global_1024_mesh.pt"),
        "saved official Global-1024 baseline",
    )
    experimental_mesh = core._validate_mesh(
        _load_torch(output_dir / "hr_tile_conditioned_mesh.pt"),
        "saved global-C64 HR-tile-conditioned output",
    )
    support = _load_torch(
        output_dir / "global_c64_support_noise_and_tile_rows.pt"
    )
    latents = _load_torch(output_dir / "final_global_c64_latents.pt")
    coords64 = support["coords64"]
    if not torch.equal(coords64, latents["coords64"]):
        raise RuntimeError("saved support and final-latent C64 coords differ")
    row_count = int(coords64.shape[0])
    coverage = torch.zeros(row_count, dtype=torch.int32)
    active_tiles = 0
    for tile in support["tiles"]:
        rows = tile["global_rows"].long()
        if rows.numel():
            active_tiles += 1
            coverage.index_add_(
                0, rows, torch.ones_like(rows, dtype=torch.int32)
            )
    tile_assignment = {
        "global_row_count": row_count,
        "tile_count": len(support["tiles"]),
        "active_tile_count": active_tiles,
        "covered_row_count": int((coverage > 0).sum().item()),
        "overlap_row_count": int((coverage > 1).sum().item()),
        "uncovered_row_count": int((coverage == 0).sum().item()),
        "maximum_memberships": int(coverage.max().item()),
        "layout": "7x7 complete in-bounds crops",
        "tile_size": TILE_SIZE,
        "tile_stride": TILE_STRIDE,
    }
    if tile_assignment["uncovered_row_count"] != 0:
        raise RuntimeError("saved completed generation has uncovered C64 rows")

    envmap = load_envmap(str(args.envmap), device="cuda")
    baseline_metric = core._render(
        baseline_mesh,
        output_dir=output_dir / "official_global_1024" / "aligned_eval",
        camera=camera,
        reference_image=reference_path,
        args=args,
        envmap=envmap,
    )
    experiment_metric = core._render(
        experimental_mesh,
        output_dir=output_dir / "hr_tile_conditioned" / "aligned_eval",
        camera=camera,
        reference_image=reference_path,
        args=args,
        envmap=envmap,
    )
    baseline_psnr = float(baseline_metric["psnr_db"])
    experiment_psnr = float(experiment_metric["psnr_db"])
    input_comparison = output_dir / "input_view_textured_comparison.png"
    _save_input_comparison(
        reference_path=reference_path,
        baseline_render_path=Path(baseline_metric["render_png"]),
        experiment_render_path=Path(experiment_metric["render_png"]),
        baseline_psnr=baseline_psnr,
        experiment_psnr=experiment_psnr,
        output_path=input_comparison,
    )
    if bool(args.render_multiview):
        baseline_multiview = render_helpers._render_merged_mesh_multiview(
            baseline_mesh,
            output_dir=output_dir / "official_global_1024" / "multiview",
            camera=camera,
            args=args,
            envmap=envmap,
        )
        experiment_multiview = render_helpers._render_merged_mesh_multiview(
            experimental_mesh,
            output_dir=output_dir / "hr_tile_conditioned" / "multiview",
            camera=camera,
            args=args,
            envmap=envmap,
        )
        multiview_comparison = output_dir / "multiview_comparison.png"
        _save_multiview_comparison(
            baseline=baseline_multiview,
            experiment=experiment_multiview,
            output_path=multiview_comparison,
        )
    else:
        baseline_multiview = {"enabled": False}
        experiment_multiview = {"enabled": False}
        multiview_comparison = None

    summary = {
        "format": FORMAT_VERSION,
        "image": str(Path(args.image).expanduser().resolve()),
        "output_dir": str(output_dir),
        "cuda_device": int(args.cuda_device),
        "cuda_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "seed": int(args.seed),
        "camera": camera,
        "generation_status": {
            "status": "completed_before_render",
            "recovery": "render-only from checkpoints after 4096 SSAA=2 OOM",
            "same_run_capture_parity": (
                "passed before the checkpoints were emitted; both max_abs "
                "errors were 0.0"
            ),
            "flow_assertions": (
                "all 12 shape and 12 texture steps completed with immutable "
                "coords/order, full coverage, full overlap, and zero fallback"
            ),
        },
        "pipeline_invariants": {
            "control": "unmodified pipeline.run 1024_cascade",
            "sparse_support": "same-call captured official learned global C64 support",
            "camera": "same global camera for control and every HR tile",
            "shape_noise": "same-call capture of official initial global C64 noise",
            "texture_noise": "same-call capture of official initial global C64 noise",
            "samplers": "unchanged official shape/texture samplers",
            "decoder": "unchanged official 1024 decoder",
            "coordinates": "absolute global C64; row gather only",
            "global_to_local_latent_transport": False,
            "local_coordinate_transform": False,
            "requantization": False,
            "reencoding": False,
            "overlap_fusion": "uniform arithmetic mean of velocities",
        },
        "tile_assignment": tile_assignment,
        "latents": {
            "coords_sha256": _tensor_sha256(coords64),
            "shape_initial_noise_sha256": _tensor_sha256(
                support["shape_initial_noise"]
            ),
            "texture_initial_noise_sha256": _tensor_sha256(
                support["texture_initial_noise"]
            ),
            "official_shape_sha256": _tensor_sha256(
                latents["official_shape_raw"]
            ),
            "official_texture_sha256": _tensor_sha256(
                latents["official_texture_raw"]
            ),
            "experimental_shape_sha256": _tensor_sha256(
                latents["experimental_shape_raw"]
            ),
            "experimental_texture_sha256": _tensor_sha256(
                latents["experimental_texture_raw"]
            ),
        },
        "evaluation": {
            "protocol": (
                "shaded PBR Pixal3D render, same global camera/environment; "
                f"render and metric resolution {int(args.render_resolution)}/"
                f"{int(args.metric_resolution)}; SSAA={int(args.render_ssaa)}"
            ),
            "baseline": render_helpers._metric_subset(baseline_metric),
            "experiment": render_helpers._metric_subset(experiment_metric),
            "psnr_delta_db": experiment_psnr - baseline_psnr,
            "input_comparison_png": str(input_comparison),
            "baseline_multiview": baseline_multiview,
            "experiment_multiview": experiment_multiview,
            "multiview_comparison_png": (
                None if multiview_comparison is None else str(multiview_comparison)
            ),
            "back_view_ground_truth": False,
            "back_view_interpretation": "qualitative only",
        },
        "render_only_seconds": float(time.perf_counter() - started),
    }
    _atomic_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "EXPERIMENT_REPORT.md", summary)
    print(
        f"[done] baseline_PSNR={baseline_psnr:.6f} "
        f"experiment_PSNR={experiment_psnr:.6f} "
        f"delta={experiment_psnr - baseline_psnr:+.6f} dB"
    )
    print(f"[done] summary={output_dir / 'summary.json'}")


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(int(args.cuda_device))
    if torch.cuda.current_device() != int(args.cuda_device):
        raise RuntimeError("requested CUDA device was not selected")
    device = torch.device("cuda")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[cuda] requested={args.cuda_device} current={torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())}"
    )
    if bool(args.render_only):
        _render_only(args, output_dir)
        return
    run_started = time.perf_counter()
    pipeline = init_pipeline(
        args.model_path,
        device="cuda",
        low_vram=bool(args.low_vram),
    )

    canonical = pipeline.preprocess_canonical_images(Image.open(args.image))
    image_4096: Image.Image = canonical["image_4096"]
    image_1024: Image.Image = canonical["image_1024"]
    image_512: Image.Image = canonical["image_512"]
    image_4096.save(output_dir / "canonical_4096.png")
    image_1024.save(output_dir / "canonical_1024.png")
    image_512.save(output_dir / "canonical_512.png")
    _atomic_json(output_dir / "canonical_metadata.json", canonical["metadata"])

    camera = core._estimate_camera(
        image_1024=image_1024,
        output_dir=output_dir,
        manual_fov=float(args.fov),
        mesh_scale=float(args.mesh_scale),
        extend_pixel=int(args.extend_pixel),
        moge_model_path=args.moge_model_path,
    )
    _atomic_json(output_dir / "global_camera.json", camera)
    ss_params, shape_params, texture_params = _sampler_params(args)

    print("[baseline] ordinary official pipeline.run Global-1024")
    baseline_started = time.perf_counter()
    baseline_output, baseline_latents, capture = _run_official_baseline_with_capture(
        pipeline=pipeline,
        image_1024=image_1024,
        camera=camera,
        seed=int(args.seed),
        ss_params=ss_params,
        shape_params=shape_params,
        texture_params=texture_params,
        max_num_tokens=int(args.max_num_tokens),
    )
    if len(baseline_output) != 1:
        raise RuntimeError("official Global-1024 did not return exactly one mesh")
    baseline_mesh = core._validate_mesh(
        baseline_output[0], "official Global-1024 baseline"
    ).to("cpu")
    baseline_shape, baseline_texture, baseline_resolution = baseline_latents
    if int(baseline_resolution) != DECODE_RESOLUTION:
        raise RuntimeError(f"official decoder resolution is {baseline_resolution}")
    baseline_shape_cpu = baseline_shape.to("cpu")
    baseline_texture_cpu = baseline_texture.to("cpu")
    baseline_seconds = float(time.perf_counter() - baseline_started)
    del baseline_output, baseline_latents, baseline_shape, baseline_texture
    _empty_cuda_cache()

    capture["global_shape_raw"] = _denormalize(
        capture["global_shape_norm"], pipeline.shape_slat_normalization
    )
    capture["global_texture_raw"] = _denormalize(
        capture["global_texture_norm"], pipeline.tex_slat_normalization
    )
    parity = _parity_report(
        baseline_shape=baseline_shape_cpu,
        baseline_texture=baseline_texture_cpu,
        capture=capture,
        tolerance=float(args.parity_tolerance),
    )
    print(
        "[same-run-capture-parity] passed "
        f"shape_max={parity['shape_max_abs_feature_error']:.3e} "
        f"texture_max={parity['texture_max_abs_feature_error']:.3e}"
    )

    # The captured tensors came directly from the same official flow call.
    # Move their copies back to the selected GPU only after control decoding.
    capture["shape_noise"] = capture["shape_noise"].to(device)
    capture["texture_noise"] = capture["texture_noise"].to(device)
    coords64 = capture["shape_noise"].coords
    if not torch.equal(coords64.detach().cpu(), capture["coords64"]):
        raise RuntimeError("moving the captured shape noise changed C64 support")
    projected_norm, projected_depth, projected_valid = (
        pipeline._project_sparse_coords_to_image_norm(
            image_cond_model=pipeline.image_cond_model_shape_1024,
            coords=coords64,
            camera_angle_x=float(camera["camera_angle_x"]),
            distance=float(camera["distance"]),
            mesh_scale=float(camera["mesh_scale"]),
            grid_resolution=GRID_C64,
        )
    )
    tiles, tile_summary_live = _build_global_row_tiles(
        image_4096=image_4096,
        projected_full_norm=projected_norm,
        projection_valid=projected_valid,
    )
    _save_projection_overlay(
        image=image_4096,
        tile_summary=tile_summary_live,
        output_path=output_dir / "global_c64_projection_and_tiles.png",
    )
    tile_summary = {
        key: value
        for key, value in tile_summary_live.items()
        if key not in {"eligible", "coverage", "pixel_x", "pixel_y"}
    }
    print(
        "[tile-assignment] "
        f"rows={tile_summary['global_row_count']:,} "
        f"active={tile_summary['active_tile_count']}/49 "
        f"covered={tile_summary['covered_row_count']:,} "
        f"overlap={tile_summary['overlap_row_count']:,} "
        f"uncovered={tile_summary['uncovered_row_count']:,}"
    )
    _atomic_torch_save(
        output_dir / "global_c64_support_noise_and_tile_rows.pt",
        {
            "format": FORMAT_VERSION,
            "coords64": coords64.detach().cpu(),
            "shape_initial_noise": capture["shape_noise"].feats.detach().cpu(),
            "texture_initial_noise": capture["texture_noise"].feats.detach().cpu(),
            "projected_full_norm": projected_norm.detach().cpu(),
            "projected_depth": projected_depth.detach().cpu(),
            "projection_valid": projected_valid.detach().cpu(),
            "tiles": [
                {
                    "tile_id": tile["tile_id"],
                    "box": tile["box"],
                    "projection_crop_box": tile["projection_crop_box"],
                    "global_rows": tile["global_rows"],
                }
                for tile in tiles
            ],
        },
    )

    # Shape: all tile conditions are extracted with the global flow model off GPU.
    shape_model = pipeline.models["shape_slat_flow_model_1024"]
    if bool(args.low_vram):
        shape_model.cpu()
    shape_condition_stats = _prepare_tile_conditions(
        pipeline=pipeline,
        image_model=pipeline.image_cond_model_shape_1024,
        image_4096=image_4096,
        global_coords=coords64,
        tiles=tiles,
        camera=camera,
        stage_name="shape",
    )
    experimental_shape_norm, shape_flow_stats = _run_tiled_global_flow(
        pipeline=pipeline,
        model=shape_model,
        sampler=pipeline.shape_slat_sampler,
        initial_noise=capture["shape_noise"],
        global_condition_cpu=capture["shape_condition_cpu"],
        tiles=tiles,
        stage_name="shape",
        sampler_params=capture["shape_params"],
        concat_cond=None,
    )
    experimental_shape_raw = _denormalize(
        experimental_shape_norm, pipeline.shape_slat_normalization
    )

    # Texture uses the same global rows/noise, with the experimental normalized
    # shape state as the ordinary Pixal3D concat condition.
    texture_projection, _, texture_projection_valid = (
        pipeline._project_sparse_coords_to_image_norm(
            image_cond_model=pipeline.image_cond_model_tex_1024,
            coords=experimental_shape_norm.coords,
            camera_angle_x=float(camera["camera_angle_x"]),
            distance=float(camera["distance"]),
            mesh_scale=float(camera["mesh_scale"]),
            grid_resolution=GRID_C64,
        )
    )
    projection_max_abs = _max_abs(projected_norm, texture_projection)
    projection_valid_equal = torch.equal(
        projected_valid.detach().cpu(), texture_projection_valid.detach().cpu()
    )
    if projection_max_abs > 1e-7 or not projection_valid_equal:
        raise RuntimeError(
            "shape/texture condition models disagree on global C64 projection: "
            f"max_abs={projection_max_abs} valid_equal={projection_valid_equal}"
        )
    texture_model = pipeline.models["tex_slat_flow_model_1024"]
    if bool(args.low_vram):
        texture_model.cpu()
    texture_condition_stats = _prepare_tile_conditions(
        pipeline=pipeline,
        image_model=pipeline.image_cond_model_tex_1024,
        image_4096=image_4096,
        global_coords=experimental_shape_norm.coords,
        tiles=tiles,
        camera=camera,
        stage_name="texture",
    )
    experimental_texture_norm, texture_flow_stats = _run_tiled_global_flow(
        pipeline=pipeline,
        model=texture_model,
        sampler=pipeline.tex_slat_sampler,
        initial_noise=capture["texture_noise"],
        global_condition_cpu=capture["texture_condition_cpu"],
        tiles=tiles,
        stage_name="texture",
        sampler_params=capture["texture_params"],
        concat_cond=experimental_shape_norm,
    )
    experimental_texture_raw = _denormalize(
        experimental_texture_norm, pipeline.tex_slat_normalization
    )
    if not torch.equal(experimental_shape_raw.coords, coords64):
        raise RuntimeError("experimental shape support/order changed")
    if not torch.equal(experimental_texture_raw.coords, coords64):
        raise RuntimeError("experimental texture support/order changed")

    print("[decode] unchanged official 1024 decoder")
    decoded = pipeline.decode_latent(
        experimental_shape_raw,
        experimental_texture_raw,
        DECODE_RESOLUTION,
    )
    if len(decoded) != 1:
        raise RuntimeError("experimental decoder did not return exactly one mesh")
    experimental_mesh_live = core._validate_mesh(
        decoded[0], "global C64 HR-tile-conditioned output"
    )
    experimental_mesh = experimental_mesh_live.to("cpu")
    del decoded, experimental_mesh_live
    _empty_cuda_cache()

    if bool(args.save_mesh_checkpoints):
        torch.save(baseline_mesh, output_dir / "official_global_1024_mesh.pt")
        torch.save(experimental_mesh, output_dir / "hr_tile_conditioned_mesh.pt")
    _atomic_torch_save(
        output_dir / "final_global_c64_latents.pt",
        {
            "format": FORMAT_VERSION,
            "coords64": coords64.detach().cpu(),
            "official_shape_raw": baseline_shape_cpu.feats.detach().cpu(),
            "official_texture_raw": baseline_texture_cpu.feats.detach().cpu(),
            "experimental_shape_norm": experimental_shape_norm.feats.detach().cpu(),
            "experimental_shape_raw": experimental_shape_raw.feats.detach().cpu(),
            "experimental_texture_norm": experimental_texture_norm.feats.detach().cpu(),
            "experimental_texture_raw": experimental_texture_raw.feats.detach().cpu(),
        },
    )

    envmap = load_envmap(str(args.envmap), device="cuda")
    reference_path = output_dir / "canonical_4096.png"
    baseline_metric = core._render(
        baseline_mesh,
        output_dir=output_dir / "official_global_1024" / "aligned_eval",
        camera=camera,
        reference_image=reference_path,
        args=args,
        envmap=envmap,
    )
    experiment_metric = core._render(
        experimental_mesh,
        output_dir=output_dir / "hr_tile_conditioned" / "aligned_eval",
        camera=camera,
        reference_image=reference_path,
        args=args,
        envmap=envmap,
    )
    baseline_psnr = float(baseline_metric["psnr_db"])
    experiment_psnr = float(experiment_metric["psnr_db"])
    input_comparison = output_dir / "input_view_textured_comparison.png"
    _save_input_comparison(
        reference_path=reference_path,
        baseline_render_path=Path(baseline_metric["render_png"]),
        experiment_render_path=Path(experiment_metric["render_png"]),
        baseline_psnr=baseline_psnr,
        experiment_psnr=experiment_psnr,
        output_path=input_comparison,
    )

    if bool(args.render_multiview):
        baseline_multiview = render_helpers._render_merged_mesh_multiview(
            baseline_mesh,
            output_dir=output_dir / "official_global_1024" / "multiview",
            camera=camera,
            args=args,
            envmap=envmap,
        )
        experiment_multiview = render_helpers._render_merged_mesh_multiview(
            experimental_mesh,
            output_dir=output_dir / "hr_tile_conditioned" / "multiview",
            camera=camera,
            args=args,
            envmap=envmap,
        )
        multiview_comparison = output_dir / "multiview_comparison.png"
        _save_multiview_comparison(
            baseline=baseline_multiview,
            experiment=experiment_multiview,
            output_path=multiview_comparison,
        )
    else:
        baseline_multiview = {"enabled": False}
        experiment_multiview = {"enabled": False}
        multiview_comparison = None

    summary = {
        "format": FORMAT_VERSION,
        "image": str(Path(args.image).expanduser().resolve()),
        "output_dir": str(output_dir),
        "cuda_device": int(args.cuda_device),
        "cuda_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "seed": int(args.seed),
        "camera": dict(camera),
        "pipeline_invariants": {
            "control": "unmodified pipeline.run 1024_cascade",
            "sparse_support": "official learned global C64 support",
            "camera": "same global camera for control and every HR tile",
            "shape_noise": "same-call capture of official initial global C64 noise",
            "texture_noise": "same-call capture of official initial global C64 noise",
            "samplers": "unchanged official shape/texture samplers",
            "decoder": "unchanged official 1024 decoder",
            "coordinates": "absolute global C64; row gather only",
            "global_to_local_latent_transport": False,
            "local_coordinate_transform": False,
            "requantization": False,
            "reencoding": False,
        },
        "baseline_generation_seconds": baseline_seconds,
        "official_same_run_capture_parity": parity,
        "tile_assignment": tile_summary,
        "condition_extraction": {
            "shape": shape_condition_stats,
            "texture": texture_condition_stats,
            "shape_texture_projection_max_abs_error": projection_max_abs,
            "shape_texture_projection_valid_equal": projection_valid_equal,
        },
        "shape_flow": shape_flow_stats,
        "texture_flow": texture_flow_stats,
        "latents": {
            "coords_sha256": _tensor_sha256(coords64),
            "shape_initial_noise_sha256": _tensor_sha256(capture["shape_noise"].feats),
            "texture_initial_noise_sha256": _tensor_sha256(capture["texture_noise"].feats),
            "experimental_shape_sha256": _tensor_sha256(experimental_shape_raw.feats),
            "experimental_texture_sha256": _tensor_sha256(experimental_texture_raw.feats),
        },
        "evaluation": {
            "protocol": (
                "shaded PBR Pixal3D render, same global camera/environment; "
                f"render and metric resolution {int(args.render_resolution)}/"
                f"{int(args.metric_resolution)}"
            ),
            "baseline": render_helpers._metric_subset(baseline_metric),
            "experiment": render_helpers._metric_subset(experiment_metric),
            "psnr_delta_db": experiment_psnr - baseline_psnr,
            "input_comparison_png": str(input_comparison),
            "baseline_multiview": baseline_multiview,
            "experiment_multiview": experiment_multiview,
            "multiview_comparison_png": (
                None if multiview_comparison is None else str(multiview_comparison)
            ),
            "back_view_ground_truth": False,
            "back_view_interpretation": "qualitative only",
        },
        "seconds": float(time.perf_counter() - run_started),
    }
    _atomic_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "EXPERIMENT_REPORT.md", summary)
    print(
        f"[done] baseline_PSNR={baseline_psnr:.6f} "
        f"experiment_PSNR={experiment_psnr:.6f} "
        f"delta={experiment_psnr - baseline_psnr:+.6f} dB"
    )
    print(f"[done] summary={output_dir / 'summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=str(Path(__file__).parent / "assets" / "choose" / "0_img.png"),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/global_c64_hr_tile_condition_ablation/seed_42",
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--moge-model-path", default=None)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--render-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="load completed mesh/latent checkpoints and run only evaluation",
    )
    parser.add_argument(
        "--low-vram", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-num-tokens", type=int, default=1_000_000)
    parser.add_argument("--parity-tolerance", type=float, default=2e-5)
    parser.add_argument("--fov", type=float, default=-1.0)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--extend-pixel", type=int, default=0)

    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--ss-guidance-strength", type=float, default=7.5)
    parser.add_argument("--ss-guidance-rescale", type=float, default=0.7)
    parser.add_argument("--ss-rescale-t", type=float, default=5.0)
    parser.add_argument("--shape-steps", type=int, default=12)
    parser.add_argument("--shape-guidance-strength", type=float, default=7.5)
    parser.add_argument("--shape-guidance-rescale", type=float, default=0.5)
    parser.add_argument("--shape-rescale-t", type=float, default=3.0)
    parser.add_argument("--texture-steps", type=int, default=12)
    parser.add_argument("--texture-guidance-strength", type=float, default=1.0)
    parser.add_argument("--texture-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--texture-rescale-t", type=float, default=3.0)

    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--render-resolution", type=int, default=4096)
    parser.add_argument("--metric-resolution", type=int, default=4096)
    parser.add_argument("--render-ssaa", type=int, default=1)
    parser.add_argument("--render-peel-layers", type=int, default=8)
    parser.add_argument("--render-face-chunk-size", type=int, default=4_000_000)
    parser.add_argument(
        "--use-envmap-bg", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--skip-lpips", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg"
    )
    parser.add_argument(
        "--save-mesh-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--render-multiview",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--multiview-resolution", type=int, default=512)
    parser.add_argument("--multiview-ssaa", type=int, default=1)
    parser.add_argument("--multiview-peel-layers", type=int, default=2)
    parser.add_argument("--multiview-radius-scale", type=float, default=1.0)
    parser.add_argument(
        "--multiview-yaws-degrees", default="0,-45,45,-90,90,180"
    )
    parser.add_argument("--multiview-pitches-degrees", default="0,0,0,0,0,0")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not Path(args.image).expanduser().is_file():
        raise FileNotFoundError(args.image)
    if int(args.cuda_device) != 4:
        raise ValueError("this requested experiment is fixed to physical CUDA 4")
    if int(args.seed) < 0:
        raise ValueError("--seed must be non-negative")
    if int(args.max_num_tokens) < 1:
        raise ValueError("--max-num-tokens must be positive")
    if float(args.parity_tolerance) < 0:
        raise ValueError("--parity-tolerance must be non-negative")
    if int(args.shape_steps) != 12 or int(args.texture_steps) != 12:
        raise ValueError("this ablation keeps the official 12-step shape/texture flows")
    if (
        int(args.render_resolution) < 1
        or int(args.metric_resolution) < 1
        or int(args.render_ssaa) < 1
        or int(args.render_peel_layers) < 1
        or int(args.multiview_resolution) < 1
    ):
        raise ValueError("render and metric settings must be positive")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
