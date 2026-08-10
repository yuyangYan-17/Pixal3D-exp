#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pixal3D projective-tile 2048 super-resolution.

This experiment has one global C128 latent trajectory.  Geometry support is
constructed before either the high-resolution shape or texture flow:

  1. Run the complete image through SS C32 -> shape512 -> C64 -> shape1024.
  2. Subdivide the global shape1024 latent once to obtain global C1024 support.
  3. For every 4096-image crop, reproduce the projective evaluation route up to
     the deduplicated tile C64 union:

       projected global C1024 -> tile C32/C64
       tile SS C32 -> shape512 -> native tile C64
       tile C64 = union(projected C64, native C64)

     No tile shape1024 flow, texture flow, or decoder is run at this stage.
  4. Invert the corrected centered-tile camera transform for every tile C64
     point, quantize it on the global C1024 lattice, concatenate it with the
     native global C1024 support, and downsample the complete source list to one
     global C128 support using Pixal3D's fixed-2048 quantizer.
  5. Record a lossless CSR provenance table from every global C128 row to all
     contributing global-C1024 and tile-C64 source rows.

At every shape-flow Euler step, the global C128 model queries the resized
global 1024 image.  Each tile then gathers the *current* global state using its
C64->C128 provenance, queries its own raw 4096 crop, and predicts a local C64
velocity at the same time.  All tile velocities that target one C128 row are
averaged uniformly.  Uncovered rows fall back to the global velocity, and the
union velocity performs exactly one global Euler update.  Texture repeats the
same procedure while gathering the final normalized shape latent as
``concat_cond``.

The script also runs the ordinary Pixal3D 2048 baseline

    SS C32 -> shape512 -> fixed C128 -> shape/texture -> decode2048

and renders/evaluates the ordinary and projective-tile results against the same
canonical full-image reference.

Camera derivation and the bidirectional point transform are imported from
``pixal3d_projective_tile_generation_eval.py`` and are documented in
``GLOBAL_MOGE_TO_LOCAL_TILE_CAMERA.md``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
)
os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "1")

from inference import MODEL_PATH, init_pipeline  # noqa: E402
from pixal3d.modules.sparse import SparseTensor  # noqa: E402

import pixal3d_projective_tile_generation_eval as projective  # noqa: E402


GRID_SS = 32
GRID_TILE = 64
GRID_MASTER = 128
GRID_GLOBAL_SUPPORT = 1024
IMAGE_LR = 512
IMAGE_CONDITION = 1024
IMAGE_CANONICAL = 4096
DECODE_RESOLUTION = 2048
DEFAULT_TILE_SIZE = 1024
DEFAULT_TILE_STRIDE = 512


@dataclass
class GlobalPreparation:
    coords32: torch.Tensor
    shape512_norm: SparseTensor
    shape512_denorm: SparseTensor
    decoder_candidates512: torch.Tensor
    coords64: torch.Tensor
    shape1024_norm: SparseTensor
    shape1024_denorm: SparseTensor
    coords1024: torch.Tensor
    ordinary_coords128: torch.Tensor
    timings: Dict[str, float]
    upsample_stats: Dict[str, Any]


@dataclass
class TileExpert:
    tile_id: int
    box: Tuple[int, int, int, int]
    transform: projective.TileCameraTransform
    local_coords64: torch.Tensor
    global_q: torch.Tensor
    global_coords1024: torch.Tensor
    shape_condition_cpu: Mapping[str, Any]
    texture_condition_cpu: Mapping[str, Any]
    support_stats: Dict[str, Any]
    local_to_master: Optional[torch.Tensor] = None


@dataclass
class FusedFlowResult:
    samples: SparseTensor
    times: List[float]
    time_intervals: List[float]
    step_records: List[Dict[str, Any]]
    states: List[torch.Tensor]
    velocities: List[torch.Tensor]


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _randn(
    rows: int,
    channels: int,
    *,
    device: torch.device,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if rows <= 0 or channels <= 0:
        raise ValueError(f"invalid random tensor shape ({rows}, {channels})")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn(
        rows,
        channels,
        generator=generator,
        device=device,
        dtype=dtype,
    )


def _tree_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, SparseTensor):
        return SparseTensor(
            feats=value.feats.detach().to(device="cpu", copy=True),
            coords=value.coords.detach().to(device="cpu", copy=True),
        )
    if isinstance(value, Mapping):
        return {key: _tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_tree_to_cpu(item) for item in value]
    return value


def _tree_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=False)
    if isinstance(value, SparseTensor):
        return SparseTensor(
            feats=value.feats.to(device=device, non_blocking=False),
            coords=value.coords.to(device=device, non_blocking=False),
        )
    if isinstance(value, Mapping):
        return {key: _tree_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_tree_to_device(item, device) for item in value]
    return value


def _features(value: Any) -> torch.Tensor:
    return value.feats if hasattr(value, "feats") else value


def _denormalize_sparse(
    value: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
) -> SparseTensor:
    std = torch.as_tensor(
        normalization["std"], device=value.device, dtype=value.dtype
    )[None]
    mean = torch.as_tensor(
        normalization["mean"], device=value.device, dtype=value.dtype
    )[None]
    return value.replace(value.feats * std + mean)


def _sample_once_kwargs(params: Mapping[str, Any]) -> Dict[str, Any]:
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


def _official_fixed_2048_downsample(
    coords1024: torch.Tensor,
) -> torch.Tensor:
    """Map global C1024 integer IDs to fixed C128 IDs like Pixal3D 2048.

    This intentionally uses ``int((i + 0.5) / 1024 * 128)`` rather than an
    endpoint-aligned ``round(... * 127)``.  It is the fixed 2048 branch in
    ``Pixal3DImageTo3DPipeline.run`` and partitions C1024 into eight-cell bins.
    Row order is preserved.
    """
    if coords1024.ndim != 2 or coords1024.shape[1] != 4:
        raise ValueError(f"expected [N,4] C1024 coordinates, got {coords1024.shape}")
    coords_i32 = coords1024.to(torch.int32)
    valid1024 = (
        (coords_i32[:, 0] == 0)
        & (coords_i32[:, 1:] >= 0).all(dim=1)
        & (coords_i32[:, 1:] < GRID_GLOBAL_SUPPORT).all(dim=1)
    )
    if not bool(valid1024.all().item()):
        raise ValueError(
            f"C1024 contains {int((~valid1024).sum().item())} invalid rows"
        )
    xyz128 = (
        (coords_i32[:, 1:].to(torch.float32) + 0.5)
        / float(GRID_GLOBAL_SUPPORT)
        * float(GRID_MASTER)
    ).to(torch.int32)
    if bool(((xyz128 < 0) | (xyz128 >= GRID_MASTER)).any().item()):
        raise RuntimeError("fixed-2048 downsample produced out-of-range C128 IDs")
    return torch.cat([coords_i32[:, :1], xyz128], dim=1)


def _quantize_decoder_candidates_to_official_c128(
    candidates: torch.Tensor,
) -> torch.Tensor:
    """Official ordinary-2048 support from the shape512 decoder candidates."""
    if candidates.ndim != 2 or candidates.shape[1] != 4:
        raise ValueError(f"decoder candidates must be [N,4], got {candidates.shape}")
    xyz = (
        (candidates[:, 1:].to(torch.float32) + 0.5)
        / float(IMAGE_LR)
        * float(GRID_MASTER)
    ).to(torch.int32)
    coords = torch.cat([candidates[:, :1].to(torch.int32), xyz], dim=1)
    valid = (
        (coords[:, 0] == 0)
        & (coords[:, 1:] >= 0).all(dim=1)
        & (coords[:, 1:] < GRID_MASTER).all(dim=1)
    )
    coords = torch.unique(coords[valid], dim=0)
    if coords.numel() == 0:
        raise RuntimeError("shape512 decoder produced no valid ordinary C128 support")
    return coords


def _run_global_preparation(
    *,
    pipeline: Any,
    image_512: Image.Image,
    image_1024: Image.Image,
    camera: Mapping[str, float],
    params: Mapping[str, Mapping[str, Any]],
    seed: int,
    max_num_tokens: int,
    max_support_tokens: int,
) -> GlobalPreparation:
    """Run shared global geometry and retain both baseline and SR supports."""
    condition_ss = pipeline.get_proj_cond_ss(
        [image_512],
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
    )
    _seed_everything(seed)
    started = time.perf_counter()
    coords32 = pipeline.sample_sparse_structure(
        condition_ss,
        resolution=GRID_SS,
        sampler_params=dict(params["ss"]),
    )
    _sync_cuda()
    ss_seconds = time.perf_counter() - started
    del condition_ss
    if coords32.numel() == 0:
        raise RuntimeError("global sparse structure is empty")
    print(f"[global-preparation] C32={coords32.shape[0]:,}")

    condition512 = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_512,
        [image_512],
        coords32,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
        grid_resolution_override=GRID_SS,
    )
    model512 = pipeline.models["shape_slat_flow_model_512"]
    noise512 = SparseTensor(
        feats=_randn(
            coords32.shape[0],
            int(model512.in_channels),
            device=pipeline.device,
            seed=seed + 101,
        ),
        coords=coords32,
    )
    shape512_norm, shape512_seconds = projective._run_sampler_full(
        pipeline=pipeline,
        sampler=pipeline.shape_slat_sampler,
        model=model512,
        noise=noise512,
        condition=condition512,
        params=params["shape"],
        description="Global shared shape SLat 512",
    )
    shape512_denorm = _denormalize_sparse(
        shape512_norm, pipeline.shape_slat_normalization
    )
    del condition512, noise512

    decoder = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        decoder.to(pipeline.device)
        decoder.low_vram = True
    started = time.perf_counter()
    try:
        decoder_candidates512 = decoder.upsample(
            shape512_denorm, upsample_times=4
        )
    finally:
        if pipeline.low_vram:
            decoder.cpu()
            decoder.low_vram = False
    _sync_cuda()
    shape512_upsample_seconds = time.perf_counter() - started

    coords64 = projective._quantize_shape512_candidates_to_c64(
        decoder_candidates512
    )
    ordinary_coords128 = _quantize_decoder_candidates_to_official_c128(
        decoder_candidates512
    )
    if ordinary_coords128.shape[0] > int(max_num_tokens):
        raise RuntimeError(
            f"ordinary C128 has {ordinary_coords128.shape[0]:,} tokens, "
            f"exceeding --max-num-tokens={int(max_num_tokens):,}"
        )
    print(
        f"[global-preparation] learned C64={coords64.shape[0]:,}; "
        f"ordinary C128={ordinary_coords128.shape[0]:,}"
    )

    shape1024_norm, shape1024_denorm, shape1024_seconds = (
        projective._run_shape1024(
            pipeline=pipeline,
            image_1024=image_1024,
            coords64=coords64,
            camera=camera,
            params=params,
            seed=seed + 201,
            description="Global support-producing shape SLat 1024",
        )
    )
    started = time.perf_counter()
    coords1024, upsample_stats = (
        projective._learned_upsample_shape1024_to_c1024(
            pipeline, shape1024_denorm
        )
    )
    _sync_cuda()
    shape1024_upsample_seconds = time.perf_counter() - started
    if coords1024.shape[0] > int(max_support_tokens):
        raise RuntimeError(
            f"global C1024 has {coords1024.shape[0]:,} tokens, "
            f"exceeding --max-support-tokens={int(max_support_tokens):,}"
        )
    print(f"[global-preparation] global C1024={coords1024.shape[0]:,}")
    _empty_cuda_cache()
    return GlobalPreparation(
        coords32=coords32,
        shape512_norm=shape512_norm,
        shape512_denorm=shape512_denorm,
        decoder_candidates512=decoder_candidates512,
        coords64=coords64,
        shape1024_norm=shape1024_norm,
        shape1024_denorm=shape1024_denorm,
        coords1024=coords1024,
        ordinary_coords128=ordinary_coords128,
        timings={
            "ss_seconds": float(ss_seconds),
            "shape512_seconds": float(shape512_seconds),
            "shape512_upsample_seconds": float(shape512_upsample_seconds),
            "shape1024_seconds": float(shape1024_seconds),
            "shape1024_upsample_seconds": float(shape1024_upsample_seconds),
        },
        upsample_stats=dict(upsample_stats),
    )


def _local_c64_to_global_c1024(
    *,
    local_coords64: torch.Tensor,
    global_camera: Mapping[str, float],
    transform: projective.TileCameraTransform,
    boundary_epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Inverse-project tile C64 and keep only representable global C1024 rows."""
    q_local = projective._endpoint_indices_to_q(
        local_coords64[:, 1:4], GRID_TILE
    ).to(local_coords64.device)
    q_global, _, uv_full, inverse_stats = (
        projective._centered_tile_q_to_global_q(
            q_local,
            global_camera=global_camera,
            transform=transform,
            validate_roundtrip=True,
        )
    )
    overflow = (q_global.abs() - 1.0).clamp_min(0.0)
    strict_inside = (q_global.abs() <= 1.0).all(dim=1)
    hard_outside = (overflow > float(boundary_epsilon)).any(dim=1)
    numeric_outside = (~strict_inside) & (~hard_outside)
    keep = strict_inside
    if not bool(keep.any().item()):
        raise RuntimeError(
            f"tile {transform.tile_id}: no C64 point is representable in global C1024"
        )
    q_kept = q_global[keep]
    local_kept = local_coords64[keep]
    ids1024 = projective._q_to_endpoint_indices(
        q_kept, GRID_GLOBAL_SUPPORT
    )
    if bool(((ids1024 < 0) | (ids1024 >= GRID_GLOBAL_SUPPORT)).any().item()):
        raise RuntimeError("global C1024 quantization produced out-of-range IDs")
    coords1024 = torch.cat(
        [
            torch.zeros(
                (ids1024.shape[0], 1),
                device=ids1024.device,
                dtype=torch.int32,
            ),
            ids1024.to(torch.int32),
        ],
        dim=1,
    )
    stats = {
        **inverse_stats,
        "input_tile_c64_rows": int(local_coords64.shape[0]),
        "global_c1024_rows": int(coords1024.shape[0]),
        "global_c1024_unique_rows": int(torch.unique(coords1024, dim=0).shape[0]),
        "global_hard_outside_rows_dropped": int(hard_outside.sum().item()),
        "global_numeric_boundary_rows_dropped": int(numeric_outside.sum().item()),
        "global_boundary_epsilon": float(boundary_epsilon),
        "global_q_min_before_filter": [
            float(value) for value in q_global.amin(dim=0).detach().cpu().tolist()
        ],
        "global_q_max_before_filter": [
            float(value) for value in q_global.amax(dim=0).detach().cpu().tolist()
        ],
        "full_uv_min_kept": [
            float(value) for value in uv_full[keep].amin(dim=0).detach().cpu().tolist()
        ],
        "full_uv_max_kept": [
            float(value) for value in uv_full[keep].amax(dim=0).detach().cpu().tolist()
        ],
    }
    return local_kept, q_kept, coords1024, stats


def _prepare_tile_expert(
    *,
    pipeline: Any,
    tile_id: int,
    box: Tuple[int, int, int, int],
    image_4096: Image.Image,
    foreground_mask_4096: Image.Image,
    selected_global_coords1024: torch.Tensor,
    global_camera: Mapping[str, float],
    params: Mapping[str, Mapping[str, Any]],
    seed: int,
    output_dir: Path,
    boundary_epsilon: float,
    extend_pixel: int,
    blender_shift_y_sign: int,
    max_support_tokens: int,
) -> TileExpert:
    tile_dir = output_dir / "tiles" / f"tile_{tile_id:02d}"
    reference = projective._prepare_tile_reference(image_4096, box, tile_dir)
    foreground_mask = projective._prepare_tile_foreground_mask(
        foreground_mask_4096, box, tile_dir
    )
    transform = projective._derive_tile_camera(
        tile_id=tile_id,
        box=box,
        global_camera=global_camera,
        extend_pixel=int(extend_pixel),
        blender_shift_y_sign=int(blender_shift_y_sign),
    )
    tile_camera = {
        "camera_angle_x": float(transform.camera_angle_x),
        "distance": float(transform.distance),
        "mesh_scale": float(transform.mesh_scale),
    }
    projective._atomic_json(tile_dir / "tile_camera.json", asdict(transform))
    print(
        f"[tile {tile_id:02d}] selected global C1024={selected_global_coords1024.shape[0]:,} "
        f"fov={math.degrees(transform.camera_angle_x):.6f}deg "
        f"distance={transform.distance:.8f} box={box}"
    )

    support_dir = tile_dir / "support"
    projected_coords32, projected_coords64, support_stats = (
        projective._prepare_projective_tile_supports(
            reference=reference,
            selected_coords128=selected_global_coords1024,
            global_camera=global_camera,
            transform=transform,
            output_dir=support_dir,
            boundary_epsilon=float(boundary_epsilon),
        )
    )
    native_coords32, native_stats = (
        projective._sample_and_constrain_tile_native_c32(
            pipeline=pipeline,
            tile_image=reference,
            foreground_mask=foreground_mask,
            projected_coords32=projected_coords32,
            tile_camera=tile_camera,
            transform=transform,
            params=params,
            seed=seed + 11,
            reference=reference,
            output_dir=support_dir,
        )
    )
    native_coords64, shape512_seconds = (
        projective._run_shape512_and_upsample_c64(
            pipeline=pipeline,
            image_512=reference.resize(
                (IMAGE_LR, IMAGE_LR), Image.Resampling.LANCZOS
            ),
            coords32=native_coords32,
            camera=tile_camera,
            params=params,
            seed=seed + 101,
            description=f"Tile {tile_id:02d} native shape SLat 512",
        )
    )
    fused_coords64, fusion_stats = projective._merge_projected_and_native_c64(
        projected_coords64=projected_coords64,
        native_coords64=native_coords64,
        transform=transform,
        reference=reference,
        output_dir=support_dir,
    )
    if fused_coords64.shape[0] > int(max_support_tokens):
        raise RuntimeError(
            f"tile {tile_id}: fused C64={fused_coords64.shape[0]:,} exceeds "
            f"--max-support-tokens={int(max_support_tokens):,}"
        )

    local_coords64, global_q, global_coords1024, inverse_stats = (
        _local_c64_to_global_c1024(
            local_coords64=fused_coords64,
            global_camera=global_camera,
            transform=transform,
            boundary_epsilon=float(boundary_epsilon),
        )
    )
    tile_rgb = reference.convert("RGB")
    shape_condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [tile_rgb],
        local_coords64,
        camera_angle_x=float(transform.camera_angle_x),
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
        grid_resolution_override=GRID_TILE,
    )
    texture_condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [tile_rgb],
        local_coords64,
        camera_angle_x=float(transform.camera_angle_x),
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
        grid_resolution_override=GRID_TILE,
    )
    shape_condition_cpu = _tree_to_cpu(shape_condition)
    texture_condition_cpu = _tree_to_cpu(texture_condition)
    support_stats = {
        **dict(support_stats),
        "tile_native_c32_constraint": dict(native_stats),
        "tile_c64_fusion": dict(fusion_stats),
        "tile_c64_to_global_c1024": dict(inverse_stats),
        "native_shape512_seconds": float(shape512_seconds),
    }
    projective._atomic_json(support_dir / "superresolution_support_stats.json", support_stats)
    torch.save(
        {
            "format": "pixal3d_projective_tile_c64_global_c1024_support_v1",
            "tile_id": int(tile_id),
            "box": list(box),
            "transform": asdict(transform),
            "projected_coords32": projected_coords32.detach().cpu(),
            "projected_coords64": projected_coords64.detach().cpu(),
            "native_coords32": native_coords32.detach().cpu(),
            "native_coords64": native_coords64.detach().cpu(),
            "fused_coords64_before_global_domain_filter": fused_coords64.detach().cpu(),
            "local_coords64": local_coords64.detach().cpu(),
            "global_q": global_q.detach().cpu(),
            "global_coords1024": global_coords1024.detach().cpu(),
            "support_stats": support_stats,
        },
        support_dir / "tile_c64_to_global_c1024.pt",
    )
    del shape_condition, texture_condition
    _empty_cuda_cache()
    return TileExpert(
        tile_id=int(tile_id),
        box=tuple(int(value) for value in box),
        transform=transform,
        local_coords64=local_coords64.detach().cpu(),
        global_q=global_q.detach().cpu(),
        global_coords1024=global_coords1024.detach().cpu(),
        shape_condition_cpu=shape_condition_cpu,
        texture_condition_cpu=texture_condition_cpu,
        support_stats=support_stats,
    )


def _build_master_and_provenance(
    *,
    global_coords1024: torch.Tensor,
    tile_experts: Sequence[TileExpert],
    max_num_tokens: int,
    output_path: Path,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Jointly downsample all C1024 sources and publish row-level provenance."""
    device = global_coords1024.device
    global_source = global_coords1024.to(device=device, dtype=torch.int32)
    source_coords1024 = [global_source]
    source_kind = [
        torch.zeros(global_source.shape[0], device=device, dtype=torch.int8)
    ]
    source_tile_id = [
        torch.full(
            (global_source.shape[0],), -1, device=device, dtype=torch.int16
        )
    ]
    source_local_row = [
        torch.arange(global_source.shape[0], device=device, dtype=torch.int64)
    ]
    tile_slices: Dict[int, Tuple[int, int]] = {}
    cursor = int(global_source.shape[0])
    for expert in tile_experts:
        coords = expert.global_coords1024.to(device=device, dtype=torch.int32)
        count = int(coords.shape[0])
        source_coords1024.append(coords)
        source_kind.append(
            torch.ones(count, device=device, dtype=torch.int8)
        )
        source_tile_id.append(
            torch.full(
                (count,), int(expert.tile_id), device=device, dtype=torch.int16
            )
        )
        source_local_row.append(
            torch.arange(count, device=device, dtype=torch.int64)
        )
        tile_slices[int(expert.tile_id)] = (cursor, cursor + count)
        cursor += count

    all_coords1024 = torch.cat(source_coords1024, dim=0)
    all_down128 = _official_fixed_2048_downsample(all_coords1024)
    master_coords128, source_to_master = torch.unique(
        all_down128, dim=0, return_inverse=True
    )
    source_to_master = source_to_master.to(torch.long)
    if master_coords128.shape[0] > int(max_num_tokens):
        raise RuntimeError(
            f"joint master C128 has {master_coords128.shape[0]:,} tokens, "
            f"exceeding --max-num-tokens={int(max_num_tokens):,}"
        )

    for expert in tile_experts:
        begin, end = tile_slices[int(expert.tile_id)]
        mapping = source_to_master[begin:end]
        if mapping.shape[0] != expert.local_coords64.shape[0]:
            raise RuntimeError(f"tile {expert.tile_id}: provenance row mismatch")
        expert.local_to_master = mapping.detach().cpu()

    source_kind_tensor = torch.cat(source_kind, dim=0)
    source_tile_tensor = torch.cat(source_tile_id, dim=0)
    source_local_tensor = torch.cat(source_local_row, dim=0)
    sort_order = torch.argsort(source_to_master, stable=True)
    counts = torch.bincount(
        source_to_master, minlength=master_coords128.shape[0]
    )
    offsets = torch.zeros(
        master_coords128.shape[0] + 1, device=device, dtype=torch.int64
    )
    offsets[1:] = torch.cumsum(counts.to(torch.int64), dim=0)

    tile_mask = source_kind_tensor == 1
    global_mask = ~tile_mask
    tile_counts = torch.bincount(
        source_to_master[tile_mask], minlength=master_coords128.shape[0]
    )
    global_counts = torch.bincount(
        source_to_master[global_mask], minlength=master_coords128.shape[0]
    )
    stats = {
        "global_c1024_source_rows": int(global_source.shape[0]),
        "tile_c64_source_rows": int(tile_mask.sum().item()),
        "all_source_rows": int(all_coords1024.shape[0]),
        "master_c128_rows": int(master_coords128.shape[0]),
        "master_rows_with_global_source": int((global_counts > 0).sum().item()),
        "master_rows_with_tile_source": int((tile_counts > 0).sum().item()),
        "master_rows_global_only": int(
            ((global_counts > 0) & (tile_counts == 0)).sum().item()
        ),
        "master_rows_tile_only": int(
            ((global_counts == 0) & (tile_counts > 0)).sum().item()
        ),
        "master_rows_global_and_tile": int(
            ((global_counts > 0) & (tile_counts > 0)).sum().item()
        ),
        "maximum_sources_per_master": int(counts.max().item()),
        "maximum_tile_sources_per_master": int(tile_counts.max().item()),
        "tile_count": len(tile_experts),
        "downsample_rule": "int((global_C1024_id + 0.5) / 1024 * 128)",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "pixal3d_global_c1024_tile_c64_to_master_c128_provenance_v1",
            "description": {
                "source_kind": "0=global_C1024, 1=tile_C64",
                "source_local_row": (
                    "global C1024 row when kind=0; filtered tile local C64 row when kind=1"
                ),
                "master_source_query": (
                    "for master row m, take source_sort_order["
                    "master_source_offsets[m]:master_source_offsets[m+1]]"
                ),
            },
            "stats": stats,
            "master_coords128": master_coords128.detach().cpu(),
            "source_coords1024": all_coords1024.detach().cpu(),
            "source_downsampled_coords128": all_down128.detach().cpu(),
            "source_to_master": source_to_master.detach().cpu(),
            "source_kind": source_kind_tensor.detach().cpu(),
            "source_tile_id": source_tile_tensor.detach().cpu(),
            "source_local_row": source_local_tensor.detach().cpu(),
            "source_sort_order": sort_order.detach().cpu(),
            "master_source_offsets": offsets.detach().cpu(),
            "master_source_counts": counts.detach().cpu(),
            "master_global_source_counts": global_counts.detach().cpu(),
            "master_tile_source_counts": tile_counts.detach().cpu(),
            "tiles": [
                {
                    "tile_id": int(expert.tile_id),
                    "box": list(expert.box),
                    "transform": asdict(expert.transform),
                    "local_coords64": expert.local_coords64,
                    "global_q": expert.global_q,
                    "global_coords1024": expert.global_coords1024,
                    "local_to_master": expert.local_to_master,
                }
                for expert in tile_experts
            ],
        },
        output_path,
    )
    print(
        f"[master-support] sources={all_coords1024.shape[0]:,} "
        f"(global={global_source.shape[0]:,}, tile={int(tile_mask.sum().item()):,}) "
        f"-> C128={master_coords128.shape[0]:,}; "
        f"tile-covered={stats['master_rows_with_tile_source']:,}"
    )
    return master_coords128, stats


@torch.no_grad()
def _run_fused_flow(
    *,
    pipeline: Any,
    sampler: Any,
    model: torch.nn.Module,
    initial_state: SparseTensor,
    global_condition: Mapping[str, Any],
    tile_experts: Sequence[TileExpert],
    params: Mapping[str, Any],
    stage: str,
    global_concat_cond: Optional[SparseTensor] = None,
    save_step_states: bool,
) -> FusedFlowResult:
    """Run global and tile experts at every Euler step with a uniform mean."""
    if stage not in {"shape", "texture"}:
        raise ValueError(stage)
    steps = int(params.get("steps", 12))
    times = [
        float(value)
        for value in sampler.timestep_schedule(
            steps, float(params.get("rescale_t", 1.0))
        )
    ]
    if len(times) != steps + 1:
        raise RuntimeError(
            f"{stage}: timestep schedule has {len(times)} values for {steps} steps"
        )
    intervals = [times[index] - times[index + 1] for index in range(steps)]
    step_kwargs = _sample_once_kwargs(params)
    if pipeline.low_vram:
        model.to(pipeline.device)
    device = initial_state.device
    global_condition_device = _tree_to_device(global_condition, device)
    if global_concat_cond is not None and not torch.equal(
        initial_state.coords, global_concat_cond.coords
    ):
        raise RuntimeError(f"{stage}: global state and concat coordinates differ")
    for expert in tile_experts:
        if expert.local_to_master is None:
            raise RuntimeError(f"tile {expert.tile_id}: missing C64->C128 mapping")

    current = initial_state
    state_trace: List[torch.Tensor] = []
    velocity_trace: List[torch.Tensor] = []
    if save_step_states:
        state_trace.append(current.feats.detach().cpu().clone())
    records: List[Dict[str, Any]] = []
    progress = tqdm(
        range(steps),
        desc=f"Global C128 + tile C64 {stage}",
        dynamic_ncols=True,
    )
    for step in progress:
        t = times[step]
        t_next = times[step + 1]
        dt = intervals[step]
        global_call: Dict[str, Any] = {
            **global_condition_device,
            **step_kwargs,
        }
        if global_concat_cond is not None:
            global_call["concat_cond"] = global_concat_cond
        global_out = sampler.sample_once(
            model, current, t, t_next, **global_call
        )
        global_velocity = _features(global_out.pred_v).to(torch.float32)
        tile_velocity_sum = torch.zeros_like(global_velocity)
        tile_velocity_count = torch.zeros(
            (global_velocity.shape[0], 1),
            device=device,
            dtype=torch.float32,
        )
        local_rows_evaluated = 0

        for expert in tile_experts:
            mapping = expert.local_to_master.to(device=device, dtype=torch.long)
            local_coords = expert.local_coords64.to(device=device)
            local_state = SparseTensor(
                feats=current.feats.index_select(0, mapping),
                coords=local_coords,
            )
            condition_cpu = (
                expert.shape_condition_cpu
                if stage == "shape"
                else expert.texture_condition_cpu
            )
            tile_call: Dict[str, Any] = {
                **_tree_to_device(condition_cpu, device),
                **step_kwargs,
            }
            if global_concat_cond is not None:
                tile_call["concat_cond"] = SparseTensor(
                    feats=global_concat_cond.feats.index_select(0, mapping),
                    coords=local_coords,
                )
            tile_out = sampler.sample_once(
                model, local_state, t, t_next, **tile_call
            )
            local_velocity = _features(tile_out.pred_v).to(torch.float32)
            tile_velocity_sum.index_add_(0, mapping, local_velocity)
            tile_velocity_count.index_add_(
                0,
                mapping,
                torch.ones(
                    (mapping.shape[0], 1),
                    device=device,
                    dtype=torch.float32,
                ),
            )
            local_rows_evaluated += int(mapping.shape[0])
            del local_state, tile_call, tile_out, local_velocity

        covered = tile_velocity_count[:, 0] > 0
        fused_velocity = global_velocity.clone()
        if bool(covered.any().item()):
            fused_velocity[covered] = (
                tile_velocity_sum[covered] / tile_velocity_count[covered]
            )
        next_state = current.replace(
            current.feats - float(dt) * fused_velocity.to(current.dtype)
        )
        if not bool(torch.isfinite(next_state.feats).all().item()):
            raise RuntimeError(f"{stage}: non-finite state after step {step}")

        covered_count = int(covered.sum().item())
        if covered_count:
            tile_mean = (
                tile_velocity_sum[covered] / tile_velocity_count[covered]
            )
            global_covered = global_velocity[covered]
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    tile_mean.flatten()[None],
                    global_covered.flatten()[None],
                ).item()
            )
            norm_ratio = float(
                tile_mean.norm().item()
                / max(global_covered.norm().item(), 1e-12)
            )
        else:
            cosine = 1.0
            norm_ratio = 1.0
        record = {
            "step": int(step),
            "t": float(t),
            "t_next": float(t_next),
            "dt": float(dt),
            "master_rows": int(current.feats.shape[0]),
            "tile_covered_rows": covered_count,
            "global_fallback_rows": int(current.feats.shape[0] - covered_count),
            "tile_local_rows_evaluated": int(local_rows_evaluated),
            "tile_source_multiplicity_mean_covered": (
                float(tile_velocity_count[covered].mean().item())
                if covered_count
                else 0.0
            ),
            "tile_source_multiplicity_max": int(
                tile_velocity_count.max().item()
            ),
            "tile_mean_vs_global_cosine": cosine,
            "tile_mean_to_global_norm_ratio": norm_ratio,
            "fusion": "uniform mean of all tile-C64 velocities; global fallback",
        }
        records.append(record)
        progress.set_postfix(
            covered=f"{covered_count}/{current.feats.shape[0]}",
            local=local_rows_evaluated,
            maxn=record["tile_source_multiplicity_max"],
            cos=f"{cosine:.4f}",
        )
        current = next_state
        if save_step_states:
            velocity_trace.append(fused_velocity.detach().cpu().clone())
            state_trace.append(current.feats.detach().cpu().clone())
        del (
            global_out,
            global_velocity,
            tile_velocity_sum,
            tile_velocity_count,
            fused_velocity,
        )

    if pipeline.low_vram:
        model.cpu()
        _empty_cuda_cache()
    return FusedFlowResult(
        samples=current,
        times=times,
        time_intervals=intervals,
        step_records=records,
        states=state_trace,
        velocities=velocity_trace,
    )


def _run_global_shape_texture(
    *,
    pipeline: Any,
    coords128: torch.Tensor,
    image_1024: Image.Image,
    camera: Mapping[str, float],
    params: Mapping[str, Mapping[str, Any]],
    seed: int,
    label: str,
) -> Tuple[SparseTensor, SparseTensor, SparseTensor, SparseTensor, Dict[str, float]]:
    """Ordinary global C128 shape and texture flow for the 2048 baseline."""
    shape_condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [image_1024],
        coords128,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
        grid_resolution_override=GRID_MASTER,
    )
    shape_model = pipeline.models["shape_slat_flow_model_1024"]
    shape_noise = SparseTensor(
        feats=_randn(
            coords128.shape[0],
            int(shape_model.in_channels),
            device=pipeline.device,
            seed=seed + 202,
        ),
        coords=coords128,
    )
    shape_norm, shape_seconds = projective._run_sampler_full(
        pipeline=pipeline,
        sampler=pipeline.shape_slat_sampler,
        model=shape_model,
        noise=shape_noise,
        condition=shape_condition,
        params=params["shape"],
        description=f"{label} shape 2048",
    )
    shape_denorm = _denormalize_sparse(
        shape_norm, pipeline.shape_slat_normalization
    )
    del shape_condition, shape_noise
    _empty_cuda_cache()

    texture_condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [image_1024],
        coords128,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
        grid_resolution_override=GRID_MASTER,
    )
    texture_model = pipeline.models["tex_slat_flow_model_1024"]
    texture_channels = int(texture_model.in_channels) - int(
        shape_norm.feats.shape[1]
    )
    if texture_channels <= 0:
        raise RuntimeError(f"invalid texture noise channel count {texture_channels}")
    texture_noise = SparseTensor(
        feats=_randn(
            coords128.shape[0],
            texture_channels,
            device=pipeline.device,
            seed=seed + 303,
        ),
        coords=coords128,
    )
    texture_norm, texture_seconds = projective._run_sampler_full(
        pipeline=pipeline,
        sampler=pipeline.tex_slat_sampler,
        model=texture_model,
        noise=texture_noise,
        condition=texture_condition,
        params=params["texture"],
        description=f"{label} texture 2048",
        concat_cond=shape_norm,
    )
    texture_denorm = _denormalize_sparse(
        texture_norm, pipeline.tex_slat_normalization
    )
    del texture_condition, texture_noise
    _empty_cuda_cache()
    return (
        shape_norm,
        shape_denorm,
        texture_norm,
        texture_denorm,
        {
            "shape_seconds": float(shape_seconds),
            "texture_seconds": float(texture_seconds),
        },
    )


def _save_flow_trace(
    *,
    path: Path,
    flow: FusedFlowResult,
    final_normalized: SparseTensor,
    stage: str,
    save_step_states: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "pixal3d_projective_tile_uniform_velocity_flow_v1",
            "stage": stage,
            "times": torch.as_tensor(flow.times),
            "time_intervals": torch.as_tensor(flow.time_intervals),
            "step_records": flow.step_records,
            "states": flow.states if save_step_states else [],
            "velocities": flow.velocities if save_step_states else [],
            "coords": final_normalized.coords.detach().cpu(),
            "final_normalized_feats": final_normalized.feats.detach().cpu(),
        },
        path,
    )


def _decode_and_save_cache(
    *,
    pipeline: Any,
    shape_denorm: SparseTensor,
    texture_denorm: SparseTensor,
    output_dir: Path,
    camera: Mapping[str, float],
    label: str,
    experiment: str,
    seed: int,
    texture_size: int,
    decimation_target: int,
    export_glb: bool,
    glb_extension_webp: bool,
) -> Dict[str, Any]:
    meshes = pipeline.decode_latent(
        shape_denorm, texture_denorm, DECODE_RESOLUTION
    )
    mesh = meshes[0]
    vertices = int(mesh.vertices.shape[0])
    faces = int(mesh.faces.shape[0])
    print(f"[decode] {label}: vertices={vertices:,} faces={faces:,}")
    effective_target = (
        faces
        if int(decimation_target) <= 0
        else min(faces, int(decimation_target))
    )
    export_kwargs = {
        "vertices": mesh.vertices,
        "faces": mesh.faces,
        "attr_volume": mesh.attrs,
        "coords": mesh.coords,
        "attr_layout": pipeline.pbr_attr_layout,
        "grid_size": DECODE_RESOLUTION,
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "decimation_target": effective_target,
        "texture_size": int(texture_size),
        "remesh": False,
        "use_tqdm": True,
        "verbose": False,
    }
    from pixal3d_directory_texture_eval import save_to_glb_cache

    cache_dir = output_dir / "postprocess_cache"
    manifest = save_to_glb_cache(
        cache_dir,
        export_kwargs,
        extra_metadata={
            "camera_params": dict(camera),
            "pipeline_resolution": DECODE_RESOLUTION,
            "actual_grid_resolution": GRID_MASTER,
            "seed": int(seed),
            "decoder_vertices": vertices,
            "decoder_faces": faces,
            "experiment": experiment,
            "label": label,
        },
        overwrite=True,
    )
    manifest_path = cache_dir / "manifest.json"
    published = json.loads(manifest_path.read_text(encoding="utf-8"))
    published["grid_size"] = DECODE_RESOLUTION
    published["aabb"] = export_kwargs["aabb"]
    projective._atomic_json(manifest_path, published)

    glb_path: Optional[str] = None
    if export_glb:
        try:
            import o_voxel.postprocess
        except Exception as exc:
            raise RuntimeError("cannot import o_voxel.postprocess") from exc
        path = output_dir / "mesh.glb"
        scene = o_voxel.postprocess.to_glb(**export_kwargs)
        scene.export(str(path), extension_webp=bool(glb_extension_webp))
        glb_path = str(path)
        del scene
    del meshes, mesh
    _empty_cuda_cache()
    return {
        "cache_dir": str(cache_dir),
        "manifest": manifest,
        "decoder_vertices": vertices,
        "decoder_faces": faces,
        "effective_decimation_target": effective_target,
        "glb": glb_path,
    }


def _metric_values(row: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    def _value(name: str) -> Optional[float]:
        value = row.get(name)
        return None if value is None else float(value)

    return {
        "psnr_db": _value("psnr_db"),
        "ssim": _value("ssim"),
        "lpips": _value("lpips"),
    }


def _save_three_way_comparison(
    *,
    reference_path: Path,
    baseline_render_path: Path,
    superres_render_path: Path,
    baseline_metrics: Mapping[str, Any],
    superres_metrics: Mapping[str, Any],
    output_path: Path,
) -> None:
    images = [
        projective._composite_on_black(Image.open(reference_path)),
        projective._composite_on_black(Image.open(baseline_render_path)),
        projective._composite_on_black(Image.open(superres_render_path)),
    ]
    target_size = images[0].size
    images = [
        image
        if image.size == target_size
        else image.resize(target_size, Image.Resampling.LANCZOS)
        for image in images
    ]
    width, height = target_size
    header = 72
    canvas = Image.new("RGB", (width * 3, height + header), (18, 18, 18))
    labels = [
        "canonical reference",
        (
            "ordinary 2048\n"
            f"PSNR {baseline_metrics.get('psnr_db')}  "
            f"SSIM {baseline_metrics.get('ssim')}  "
            f"LPIPS {baseline_metrics.get('lpips')}"
        ),
        (
            "projective-tile SR 2048\n"
            f"PSNR {superres_metrics.get('psnr_db')}  "
            f"SSIM {superres_metrics.get('ssim')}  "
            f"LPIPS {superres_metrics.get('lpips')}"
        ),
    ]
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = index * width
        canvas.paste(image, (x, header))
        draw.multiline_text((x + 10, 8), label, fill=(255, 255, 255), spacing=4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _parse_tile_ids(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not value.strip():
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def run(args: argparse.Namespace) -> None:
    if int(args.tile_size) != DEFAULT_TILE_SIZE:
        raise ValueError(f"this experiment requires --tile-size={DEFAULT_TILE_SIZE}")
    if int(args.tile_stride) != DEFAULT_TILE_STRIDE:
        raise ValueError(
            f"this experiment requires --tile-stride={DEFAULT_TILE_STRIDE}"
        )
    if int(args.shape_steps) != int(args.texture_steps):
        raise ValueError("shape and texture must use the same number of Euler steps")

    repository_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = init_pipeline(
        args.model_path, device="cuda", low_vram=bool(args.low_vram)
    )
    canonical = pipeline.preprocess_canonical_images(Image.open(args.image))
    image_4096: Image.Image = canonical["image_4096"]
    image_1024: Image.Image = canonical["image_1024"]
    image_512: Image.Image = canonical["image_512"]
    foreground_mask_4096: Image.Image = canonical[
        "foreground_mask_4096"
    ].convert("L")
    image_4096.save(output_dir / "canonical_4096.png")
    image_1024.save(output_dir / "canonical_1024.png")
    image_512.save(output_dir / "canonical_512.png")
    foreground_mask_4096.save(output_dir / "canonical_foreground_mask_4096.png")
    projective._atomic_json(
        output_dir / "canonical_metadata.json", canonical["metadata"]
    )
    metric_reference = output_dir / "metric_reference_rgb.png"
    projective._composite_on_black(image_1024).save(metric_reference)

    global_camera = projective._estimate_camera(
        image_1024=image_1024,
        output_dir=output_dir,
        manual_fov=float(args.fov),
        mesh_scale=float(args.mesh_scale),
        extend_pixel=int(args.extend_pixel),
        image_resolution=int(args.camera_image_resolution),
        moge_model_path=args.moge_model_path,
    )
    projective._atomic_json(output_dir / "global_camera.json", global_camera)
    print(
        f"[global-camera] fov={global_camera['camera_angle_x']:.8f} "
        f"distance={global_camera['distance']:.8f} "
        f"mesh_scale={global_camera['mesh_scale']:.8f}"
    )
    params = projective._sampler_params(args, pipeline)
    preparation = _run_global_preparation(
        pipeline=pipeline,
        image_512=image_512,
        image_1024=image_1024,
        camera=global_camera,
        params=params,
        seed=int(args.seed),
        max_num_tokens=int(args.max_num_tokens),
        max_support_tokens=int(args.max_support_tokens),
    )
    preparation_timings = dict(preparation.timings)
    global_dir = output_dir / "global_support"
    global_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "pixal3d_projective_tile_sr_global_support_v1",
            "coords32": preparation.coords32.detach().cpu(),
            "shape512_normalized": preparation.shape512_norm.feats.detach().cpu(),
            "shape512_denormalized": preparation.shape512_denorm.feats.detach().cpu(),
            "decoder_candidates512": preparation.decoder_candidates512.detach().cpu(),
            "coords64": preparation.coords64.detach().cpu(),
            "shape1024_normalized": preparation.shape1024_norm.feats.detach().cpu(),
            "shape1024_denormalized": preparation.shape1024_denorm.feats.detach().cpu(),
            "coords1024": preparation.coords1024.detach().cpu(),
            "ordinary_coords128": preparation.ordinary_coords128.detach().cpu(),
            "timings": preparation.timings,
            "upsample_stats": preparation.upsample_stats,
        },
        global_dir / "global_support.pt",
    )
    global_coords1024 = preparation.coords1024
    ordinary_coords128 = preparation.ordinary_coords128
    global_support_stats = {
        "global_c32_rows": int(preparation.coords32.shape[0]),
        "global_c64_rows": int(preparation.coords64.shape[0]),
        "global_c1024_rows": int(global_coords1024.shape[0]),
        "ordinary_c128_rows": int(ordinary_coords128.shape[0]),
        "shape1024_to_c1024": dict(preparation.upsample_stats),
    }
    # Only the two coordinate supports are needed after this point.  In
    # particular, do not retain global shape512/shape1024 latents throughout
    # all tile SS/shape512 runs.
    del preparation
    _empty_cuda_cache()

    q_global1024 = projective._endpoint_indices_to_q(
        global_coords1024[:, 1:4], GRID_GLOBAL_SUPPORT
    ).to(global_coords1024.device)
    _, _, uv_full4096, _, finite_global = (
        projective._project_global_q_to_1024_and_4096(
            q_global1024, global_camera=global_camera
        )
    )
    boxes = projective._tile_layout(
        IMAGE_CANONICAL, int(args.tile_size), int(args.tile_stride)
    )
    selected_ids = _parse_tile_ids(args.tile_ids)
    if selected_ids is not None:
        invalid = sorted(
            tile_id
            for tile_id in selected_ids
            if tile_id < 0 or tile_id >= len(boxes)
        )
        if invalid:
            raise ValueError(
                f"invalid tile IDs {invalid}; valid range is 0-{len(boxes)-1}"
            )

    tile_experts: List[TileExpert] = []
    tile_records: List[Dict[str, Any]] = []
    attempted = 0
    for tile_id, box in enumerate(boxes):
        if selected_ids is not None and tile_id not in selected_ids:
            continue
        if args.max_tiles is not None and attempted >= int(args.max_tiles):
            break
        attempted += 1
        rows = projective._rows_inside_tile(uv_full4096, finite_global, box)
        if rows.numel() == 0:
            record = {
                "status": "skipped",
                "tile_id": int(tile_id),
                "box": list(box),
                "reason": "no projected global C1024 point inside tile",
            }
            tile_records.append(record)
            projective._atomic_json(
                output_dir / "tiles" / f"tile_{tile_id:02d}" / "summary.json",
                record,
            )
            continue
        try:
            expert = _prepare_tile_expert(
                pipeline=pipeline,
                tile_id=tile_id,
                box=box,
                image_4096=image_4096,
                foreground_mask_4096=foreground_mask_4096,
                selected_global_coords1024=global_coords1024.index_select(0, rows),
                global_camera=global_camera,
                params=params,
                seed=int(args.seed) + tile_id * 1000 + 1,
                output_dir=output_dir,
                boundary_epsilon=float(args.boundary_epsilon),
                extend_pixel=int(args.extend_pixel),
                blender_shift_y_sign=int(args.blender_shift_y_sign),
                max_support_tokens=int(args.max_support_tokens),
            )
            if expert.local_coords64.shape[0] < int(args.min_tile_tokens):
                raise RuntimeError(
                    f"usable C64={expert.local_coords64.shape[0]} below "
                    f"--min-tile-tokens={int(args.min_tile_tokens)}"
                )
            tile_experts.append(expert)
            record = {
                "status": "success",
                "tile_id": int(tile_id),
                "box": list(box),
                "selected_global_c1024_rows": int(rows.numel()),
                "tile_c64_rows": int(expert.local_coords64.shape[0]),
                "tile_global_c1024_unique": int(
                    torch.unique(expert.global_coords1024, dim=0).shape[0]
                ),
                "support_stats": expert.support_stats,
            }
        except Exception as exc:
            if args.strict_tiles:
                raise
            record = {
                "status": "failed",
                "tile_id": int(tile_id),
                "box": list(box),
                "selected_global_c1024_rows": int(rows.numel()),
                "reason": f"{type(exc).__name__}: {exc}",
            }
            print(f"[tile-error] tile={tile_id:02d}: {record['reason']}")
            _empty_cuda_cache()
        tile_records.append(record)
        projective._atomic_json(
            output_dir / "tiles" / f"tile_{tile_id:02d}" / "summary.json",
            record,
        )
    if not tile_experts:
        raise RuntimeError("no usable tile expert was prepared")

    master_coords128, provenance_stats = _build_master_and_provenance(
        global_coords1024=global_coords1024,
        tile_experts=tile_experts,
        max_num_tokens=int(args.max_num_tokens),
        output_path=output_dir / "provenance" / "master_c128_provenance.pt",
    )
    projective._atomic_json(
        output_dir / "provenance" / "summary.json", provenance_stats
    )

    global_shape_condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [image_1024],
        master_coords128,
        camera_angle_x=float(global_camera["camera_angle_x"]),
        distance=float(global_camera["distance"]),
        mesh_scale=float(global_camera["mesh_scale"]),
        grid_resolution_override=GRID_MASTER,
    )
    shape_model = pipeline.models["shape_slat_flow_model_1024"]
    superres_shape_noise = SparseTensor(
        feats=_randn(
            master_coords128.shape[0],
            int(shape_model.in_channels),
            device=pipeline.device,
            seed=int(args.seed) + 202,
        ),
        coords=master_coords128,
    )
    superres_shape_flow = _run_fused_flow(
        pipeline=pipeline,
        sampler=pipeline.shape_slat_sampler,
        model=shape_model,
        initial_state=superres_shape_noise,
        global_condition=global_shape_condition,
        tile_experts=tile_experts,
        params=params["shape"],
        stage="shape",
        save_step_states=bool(args.save_step_states),
    )
    superres_shape_norm = superres_shape_flow.samples
    superres_shape_denorm = _denormalize_sparse(
        superres_shape_norm, pipeline.shape_slat_normalization
    )
    _save_flow_trace(
        path=output_dir / "traces" / "superres_shape_flow.pt",
        flow=superres_shape_flow,
        final_normalized=superres_shape_norm,
        stage="shape",
        save_step_states=bool(args.save_step_states),
    )
    del global_shape_condition, superres_shape_noise
    _empty_cuda_cache()

    global_texture_condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [image_1024],
        master_coords128,
        camera_angle_x=float(global_camera["camera_angle_x"]),
        distance=float(global_camera["distance"]),
        mesh_scale=float(global_camera["mesh_scale"]),
        grid_resolution_override=GRID_MASTER,
    )
    texture_model = pipeline.models["tex_slat_flow_model_1024"]
    texture_channels = int(texture_model.in_channels) - int(
        superres_shape_norm.feats.shape[1]
    )
    if texture_channels <= 0:
        raise RuntimeError(f"invalid texture noise channel count {texture_channels}")
    superres_texture_noise = SparseTensor(
        feats=_randn(
            master_coords128.shape[0],
            texture_channels,
            device=pipeline.device,
            seed=int(args.seed) + 303,
        ),
        coords=master_coords128,
    )
    superres_texture_flow = _run_fused_flow(
        pipeline=pipeline,
        sampler=pipeline.tex_slat_sampler,
        model=texture_model,
        initial_state=superres_texture_noise,
        global_condition=global_texture_condition,
        tile_experts=tile_experts,
        params=params["texture"],
        stage="texture",
        global_concat_cond=superres_shape_norm,
        save_step_states=bool(args.save_step_states),
    )
    superres_texture_norm = superres_texture_flow.samples
    superres_texture_denorm = _denormalize_sparse(
        superres_texture_norm, pipeline.tex_slat_normalization
    )
    _save_flow_trace(
        path=output_dir / "traces" / "superres_texture_flow.pt",
        flow=superres_texture_flow,
        final_normalized=superres_texture_norm,
        stage="texture",
        save_step_states=bool(args.save_step_states),
    )
    torch.save(
        {
            "format": "pixal3d_projective_tile_superres_2048_final_latents_v1",
            "coords128": master_coords128.detach().cpu(),
            "shape_normalized": superres_shape_norm.feats.detach().cpu(),
            "shape_denormalized": superres_shape_denorm.feats.detach().cpu(),
            "texture_normalized": superres_texture_norm.feats.detach().cpu(),
            "texture_denormalized": superres_texture_denorm.feats.detach().cpu(),
            "provenance": str(
                output_dir / "provenance" / "master_c128_provenance.pt"
            ),
        },
        output_dir / "traces" / "superres_2048_final_latents.pt",
    )
    del global_texture_condition, superres_texture_noise
    _empty_cuda_cache()

    (
        baseline_shape_norm,
        baseline_shape_denorm,
        baseline_texture_norm,
        baseline_texture_denorm,
        baseline_flow_timings,
    ) = _run_global_shape_texture(
        pipeline=pipeline,
        coords128=ordinary_coords128,
        image_1024=image_1024,
        camera=global_camera,
        params=params,
        seed=int(args.seed),
        label="Ordinary global",
    )
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "pixal3d_ordinary_2048_final_latents_v1",
            "coords128": ordinary_coords128.detach().cpu(),
            "shape_normalized": baseline_shape_norm.feats.detach().cpu(),
            "shape_denormalized": baseline_shape_denorm.feats.detach().cpu(),
            "texture_normalized": baseline_texture_norm.feats.detach().cpu(),
            "texture_denormalized": baseline_texture_denorm.feats.detach().cpu(),
            "timings": baseline_flow_timings,
        },
        traces_dir / "ordinary_2048_final_latents.pt",
    )

    baseline_dir = output_dir / "ordinary_2048"
    superres_dir = output_dir / "projective_tile_superres_2048"
    baseline_cache: Dict[str, Any] = {}
    superres_cache: Dict[str, Any] = {}
    baseline_eval: Dict[str, Any] = {}
    superres_eval: Dict[str, Any] = {}
    comparison_path: Optional[Path] = None
    if not args.no_decode:
        baseline_cache = _decode_and_save_cache(
            pipeline=pipeline,
            shape_denorm=baseline_shape_denorm,
            texture_denorm=baseline_texture_denorm,
            output_dir=baseline_dir,
            camera=global_camera,
            label="Ordinary global 2048",
            experiment="ordinary_global_2048",
            seed=int(args.seed),
            texture_size=int(args.texture_size),
            decimation_target=int(args.decimation_target),
            export_glb=bool(args.export_glb),
            glb_extension_webp=bool(args.glb_extension_webp),
        )
        superres_cache = _decode_and_save_cache(
            pipeline=pipeline,
            shape_denorm=superres_shape_denorm,
            texture_denorm=superres_texture_denorm,
            output_dir=superres_dir,
            camera=global_camera,
            label="Projective tile super-resolution 2048",
            experiment="projective_tile_uniform_velocity_superresolution_2048",
            seed=int(args.seed),
            texture_size=int(args.texture_size),
            decimation_target=int(args.decimation_target),
            export_glb=bool(args.export_glb),
            glb_extension_webp=bool(args.glb_extension_webp),
        )

        # Rendering happens in child processes. Release the model and all GPU
        # latents first so Blender/Cycles and LPIPS get the available device.
        del (
            baseline_shape_norm,
            baseline_shape_denorm,
            baseline_texture_norm,
            baseline_texture_denorm,
            superres_shape_norm,
            superres_shape_denorm,
            superres_texture_norm,
            superres_texture_denorm,
            superres_shape_flow,
            superres_texture_flow,
            tile_experts,
            pipeline,
        )
        _empty_cuda_cache()

        if args.render_eval:
            baseline_eval = projective._run_evaluator(
                repository_dir=repository_dir,
                cache_dir=Path(baseline_cache["cache_dir"]),
                output_dir=baseline_dir / "aligned_eval",
                reference_image=metric_reference,
                args=args,
                render_resolution_override=int(args.render_resolution),
            )
            superres_eval = projective._run_evaluator(
                repository_dir=repository_dir,
                cache_dir=Path(superres_cache["cache_dir"]),
                output_dir=superres_dir / "aligned_eval",
                reference_image=metric_reference,
                args=args,
                render_resolution_override=int(args.render_resolution),
            )
            baseline_row = dict(baseline_eval["metrics_row"])
            superres_row = dict(superres_eval["metrics_row"])
            projective._save_extra_comparisons(
                metric_reference,
                Path(str(baseline_row["render_png"])),
                baseline_dir / "comparisons",
            )
            projective._save_extra_comparisons(
                metric_reference,
                Path(str(superres_row["render_png"])),
                superres_dir / "comparisons",
            )
            comparison_path = output_dir / "comparison_original_ordinary_superres.png"
            _save_three_way_comparison(
                reference_path=metric_reference,
                baseline_render_path=Path(str(baseline_row["render_png"])),
                superres_render_path=Path(str(superres_row["render_png"])),
                baseline_metrics=baseline_row,
                superres_metrics=superres_row,
                output_path=comparison_path,
            )
    else:
        print("[done] --no-decode: supports, provenance, and final latents saved")

    baseline_metrics = (
        _metric_values(baseline_eval["metrics_row"]) if baseline_eval else {}
    )
    superres_metrics = (
        _metric_values(superres_eval["metrics_row"]) if superres_eval else {}
    )
    metric_delta: Dict[str, Optional[float]] = {}
    if baseline_metrics and superres_metrics:
        metric_delta = {
            "psnr_gain_db": (
                float(superres_metrics["psnr_db"]) - float(baseline_metrics["psnr_db"])
                if superres_metrics["psnr_db"] is not None
                and baseline_metrics["psnr_db"] is not None
                else None
            ),
            "ssim_gain": (
                float(superres_metrics["ssim"]) - float(baseline_metrics["ssim"])
                if superres_metrics["ssim"] is not None
                and baseline_metrics["ssim"] is not None
                else None
            ),
            "lpips_reduction": (
                float(baseline_metrics["lpips"]) - float(superres_metrics["lpips"])
                if superres_metrics["lpips"] is not None
                and baseline_metrics["lpips"] is not None
                else None
            ),
        }
    summary = {
        "format": "pixal3d_projective_tile_superresolution_2048_summary_v1",
        "image": str(Path(args.image).expanduser().resolve()),
        "global_camera": global_camera,
        "ordinary_route": (
            "global SS C32 -> shape512 -> official fixed C128 -> "
            "global shape/texture -> decode2048"
        ),
        "superresolution_route": (
            "global shape1024 -> global C1024; per-tile projected+native C64 union "
            "-> inverse camera -> global C1024; joint downsample to C128; "
            "per-step global/tile uniform velocity fusion -> decode2048"
        ),
        "tile_velocity_weight": "uniform (one contribution per tile C64 row)",
        "global_fallback": "global V128 for C128 rows without tile contributors",
        "provenance": str(
            output_dir / "provenance" / "master_c128_provenance.pt"
        ),
        "provenance_stats": provenance_stats,
        "global_support_stats": global_support_stats,
        "attempted_tiles": attempted,
        "usable_tiles": sum(row.get("status") == "success" for row in tile_records),
        "failed_tiles": sum(row.get("status") == "failed" for row in tile_records),
        "skipped_tiles": sum(row.get("status") == "skipped" for row in tile_records),
        "tiles": tile_records,
        "global_preparation_timings": preparation_timings,
        "ordinary_flow_timings": baseline_flow_timings,
        "ordinary_cache": baseline_cache,
        "superres_cache": superres_cache,
        "ordinary_metrics": baseline_metrics,
        "superres_metrics": superres_metrics,
        "metric_delta_superres_minus_ordinary": metric_delta,
        "comparison_png": None if comparison_path is None else str(comparison_path),
    }
    projective._atomic_json(output_dir / "summary.json", summary)
    print(f"[summary] {output_dir / 'summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--moge-model-path", default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--tile-stride", type=int, default=DEFAULT_TILE_STRIDE)
    parser.add_argument(
        "--tile-ids",
        default=None,
        help="comma-separated tile IDs; omitted means all 49 canonical tiles",
    )
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument(
        "--min-tile-tokens",
        type=int,
        default=100,
        help="minimum globally representable fused tile-C64 rows",
    )
    parser.add_argument(
        "--strict-tiles", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=262_144,
        help=(
            "safety limit for the final joint C128 flow support; no token "
            "subsampling or silent resolution reduction is performed"
        ),
    )
    parser.add_argument(
        "--max-support-tokens",
        type=int,
        default=8_000_000,
        help=(
            "safety limit for dense global C1024 or one tile C64 support; "
            "this is separate from the much smaller C128 flow-token limit"
        ),
    )
    parser.add_argument("--boundary-epsilon", type=float, default=1e-5)
    parser.add_argument("--blender-shift-y-sign", type=int, choices=(-1, 1), default=1)

    parser.add_argument("--fov", type=float, default=-1.0)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument("--camera-image-resolution", type=int, default=1024)

    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--shape-steps", type=int, default=12)
    parser.add_argument("--texture-steps", type=int, default=12)
    parser.add_argument("--ss-guidance-strength", type=float, default=7.5)
    parser.add_argument("--ss-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--ss-rescale-t", type=float, default=1.0)
    parser.add_argument("--shape-guidance-strength", type=float, default=7.5)
    parser.add_argument("--shape-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--shape-rescale-t", type=float, default=1.0)
    parser.add_argument("--texture-guidance-strength", type=float, default=1.0)
    parser.add_argument("--texture-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--texture-rescale-t", type=float, default=1.0)
    parser.add_argument(
        "--save-step-states", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--low-vram", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument("--no-decode", action="store_true")
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--decimation-target", type=int, default=0)
    parser.add_argument(
        "--export-glb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--glb-extension-webp",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--render-eval", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--light", default="studio")
    parser.add_argument("--render-engine", choices=("cycles", "eevee"), default="cycles")
    parser.add_argument("--render-resolution", type=int, default=2048)
    parser.add_argument("--metric-resolution", type=int, default=1024)
    parser.add_argument("--blender-samples", type=int, default=64)
    parser.add_argument("--render-max-faces", type=int, default=0)
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--metric-device", default="cuda")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--blender", default="blender")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
