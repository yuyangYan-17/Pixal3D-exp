#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate local Pixal3D generation from a true global geometry-1024 prior.

Experiment-only pipeline:

1. Canonicalize one input image to 4096/1024/512.
2. Run the official global geometry path:
      sparse structure C32
      -> shape 512 flow
      -> learned C64 support
      -> shape 1024 flow
   Stop before global texture generation and global mesh decoding.
3. Feed the global shape-1024 C64 latent through FOUR learned decoder
   subdivisions:
      C64 -> four learned doublings -> geometry lattice 1024.
   The resulting coordinates are true 1024-resolution voxel coordinates.
4. Project every global geometry-1024 point with the global camera. For each
   1024 crop (stride 512), select points whose projection lies in that crop and
   re-express them in the crop camera's canonical frame.
5. Quantize the selected points first to the tile's own geometry-1024 lattice,
   then reduce those geometry points to the C64 support required by the official
   tile shape-1024 flow. The local shape/texture path itself is unmodified.
6. Skip a tile when its final unique C64 support has fewer than
   --min-tile-tokens entries (default: 1000).
7. Run official tile shape-1024 and texture-1024 generation, decode, render,
   compute PSNR/SSIM/LPIPS, and export one GLB per successful tile by default.
8. Save geometry-1024/C64 support diagnostics and comparison sheets.

This file contains only this experiment. It does not run the old one-step
intermediate-resolution prior, Route A/B/C comparisons, flow fusion, or final
multi-tile fusion.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
)
os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "1")

from inference import (  # noqa: E402
    MODEL_PATH,
    distance_from_fov,
    get_camera_params_wild_moge,
    init_pipeline,
    load_moge_model,
)
from pixal3d.modules.sparse import SparseTensor  # noqa: E402

try:
    import o_voxel  # type: ignore
except Exception:  # pragma: no cover
    o_voxel = None


GRID_SS = 32
GRID_SHAPE_1024 = 64
GEOMETRY_RESOLUTION = 1024
IMAGE_LR = 512
IMAGE_FLOW = 1024
IMAGE_CANONICAL = 4096
DECODE_TILE = 1024
DEFAULT_TILE_SIZE = 1024
DEFAULT_TILE_STRIDE = 512

PIXAL3D_EXPORT_ROTATION = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class TileCameraTransform:
    tile_id: int
    box: Tuple[int, int, int, int]
    output_size: int
    camera_angle_x: float
    distance: float
    mesh_scale: float
    full_focal_pixels: float
    tile_focal_pixels: float
    crop_scale_x: float
    crop_scale_y: float


@dataclass
class ShapeResult:
    shape_norm: SparseTensor
    shape_denorm: SparseTensor
    shape512_seconds: float
    shape1024_seconds: float


@dataclass
class ModelResult:
    shape_norm: SparseTensor
    shape_denorm: SparseTensor
    texture_norm: SparseTensor
    texture_denorm: SparseTensor
    shape_seconds: float
    texture_seconds: float


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


def _normalization(
    values: Mapping[str, Sequence[float]],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    std = torch.as_tensor(values["std"], device=device, dtype=dtype)[None]
    mean = torch.as_tensor(values["mean"], device=device, dtype=dtype)[None]
    return std, mean


def _denormalize_sparse(
    value: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
) -> SparseTensor:
    std, mean = _normalization(normalization, value.device, value.dtype)
    return value.replace(value.feats * std + mean)


def _endpoint_indices_to_q(indices: torch.Tensor, resolution: int) -> torch.Tensor:
    if resolution <= 1:
        raise ValueError("resolution must exceed one")
    return indices.to(torch.float32) * (2.0 / float(resolution - 1)) - 1.0


def _q_to_endpoint_indices(q: torch.Tensor, resolution: int) -> torch.Tensor:
    if resolution <= 1:
        raise ValueError("resolution must exceed one")
    return torch.round((q + 1.0) * (float(resolution - 1) / 2.0)).to(torch.int32)


def _voxel_indices_to_q(indices: torch.Tensor, resolution: int) -> torch.Tensor:
    """Map integer voxel indices to normalized coordinates at voxel centers."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    return (indices.to(torch.float32) + 0.5) * (2.0 / float(resolution)) - 1.0


def _q_to_voxel_indices(q: torch.Tensor, resolution: int) -> torch.Tensor:
    """Map normalized coordinates to containing voxel indices."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    result = torch.floor((q + 1.0) * (float(resolution) / 2.0)).to(torch.int32)
    return result.clamp(0, resolution - 1)


def _geometry1024_to_c64_indices(indices1024: torch.Tensor) -> torch.Tensor:
    """Use the official endpoint quantization from geometry 1024 to C64."""
    return torch.round(
        (indices1024.to(torch.float32) + 0.5)
        / float(GEOMETRY_RESOLUTION)
        * float(GRID_SHAPE_1024 - 1)
    ).to(torch.int32).clamp(0, GRID_SHAPE_1024 - 1)


def _quantize_shape512_candidates_to_c64(candidates: torch.Tensor) -> torch.Tensor:
    """Official-style 512 candidate lattice -> C64 support quantization."""
    if candidates.ndim != 2 or candidates.shape[1] != 4:
        raise ValueError(f"decoder candidates must be [N,4], got {tuple(candidates.shape)}")
    xyz = torch.round(
        (candidates[:, 1:].to(torch.float32) + 0.5)
        / float(IMAGE_LR)
        * float(GRID_SHAPE_1024 - 1)
    ).to(torch.int32)
    quantized = torch.cat([candidates[:, :1].to(torch.int32), xyz], dim=1)
    valid = (
        (quantized[:, 1:] >= 0)
        & (quantized[:, 1:] < GRID_SHAPE_1024)
    ).all(dim=1)
    quantized = torch.unique(quantized[valid], dim=0)
    if quantized.numel() == 0:
        raise RuntimeError("shape512 learned upsample produced no valid C64 coordinates")
    return quantized


def _learned_upsample_shape512_to_c64(
    pipeline: Any,
    shape512_denorm: SparseTensor,
) -> torch.Tensor:
    decoder = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        decoder.to(pipeline.device)
        decoder.low_vram = True
    try:
        candidates = decoder.upsample(shape512_denorm, upsample_times=4)
    finally:
        if pipeline.low_vram:
            decoder.cpu()
            decoder.low_vram = False
            _empty_cuda_cache()
    return _quantize_shape512_candidates_to_c64(candidates)


def _learned_upsample_shape1024_to_geometry1024(
    pipeline: Any,
    shape1024_denorm: SparseTensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Run four decoder subdivisions: C64 shape latent -> geometry 1024."""
    decoder = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        decoder.to(pipeline.device)
        decoder.low_vram = True
    try:
        candidates = decoder.upsample(shape1024_denorm, upsample_times=4)
    finally:
        if pipeline.low_vram:
            decoder.cpu()
            decoder.low_vram = False
            _empty_cuda_cache()

    if candidates.ndim != 2 or candidates.shape[1] != 4:
        raise RuntimeError(
            "shape_slat_decoder.upsample(..., upsample_times=4) must return "
            f"[N,4], got {tuple(candidates.shape)}"
        )
    candidates_float = candidates.to(torch.float32)
    if not torch.isfinite(candidates_float).all():
        raise RuntimeError("four-step shape1024 upsample returned non-finite coordinates")

    rounded = torch.round(candidates_float[:, 1:])
    max_fractional_error = float(
        (candidates_float[:, 1:] - rounded).abs().max().item()
    )
    coords_all = torch.cat(
        [candidates[:, :1].to(torch.int32), rounded.to(torch.int32)],
        dim=1,
    )
    valid = (
        (coords_all[:, 1:] >= 0)
        & (coords_all[:, 1:] < GEOMETRY_RESOLUTION)
    ).all(dim=1)
    coords1024 = torch.unique(coords_all[valid], dim=0)
    if coords1024.numel() == 0:
        raise RuntimeError("four-step upsample produced no valid geometry-1024 points")

    stats = {
        "source_c64_tokens": int(shape1024_denorm.coords.shape[0]),
        "upsample_times": 4,
        "target_geometry_resolution": GEOMETRY_RESOLUTION,
        "candidate_rows": int(candidates.shape[0]),
        "valid_candidate_rows": int(valid.sum().item()),
        "discarded_out_of_range_rows": int((~valid).sum().item()),
        "unique_geometry1024_points": int(coords1024.shape[0]),
        "unique_merge_rows": int(valid.sum().item() - coords1024.shape[0]),
        "max_fractional_coordinate_error": max_fractional_error,
        "min_xyz": [int(v) for v in coords1024[:, 1:].amin(dim=0).cpu().tolist()],
        "max_xyz": [int(v) for v in coords1024[:, 1:].amax(dim=0).cpu().tolist()],
    }
    return coords1024, stats

def _focal_pixels(camera_angle_x: float, resolution: int) -> float:
    return float(resolution) / (2.0 * math.tan(float(camera_angle_x) / 2.0))


def _global_geometry1024_to_camera(
    coords1024: torch.Tensor,
    *,
    camera: Mapping[str, float],
) -> torch.Tensor:
    q = _voxel_indices_to_q(
        coords1024[:, 1:4],
        GEOMETRY_RESOLUTION,
    ).to(coords1024.device)
    center = torch.tensor(
        [0.0, 0.0, -float(camera["distance"])],
        device=coords1024.device,
        dtype=q.dtype,
    )
    return center[None] + q / (2.0 * float(camera["mesh_scale"]))

def _project_camera_points(
    camera_points: torch.Tensor,
    camera_angle_x: float,
    resolution: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points = camera_points
    if points.ndim == 2:
        points = points.unsqueeze(0)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("camera_points must be [K,3] or [B,K,3]")

    focal = _focal_pixels(float(camera_angle_x), int(resolution))
    depth = -points[..., 2]
    denom = depth.clamp_min(1e-8)
    u = focal * points[..., 0] / denom + float(resolution) / 2.0
    v = -focal * points[..., 1] / denom + float(resolution) / 2.0
    uv = torch.stack([u, v], dim=-1)
    valid = (
        (depth > 0)
        & torch.isfinite(uv).all(dim=-1)
        & (u >= 0)
        & (u < resolution)
        & (v >= 0)
        & (v < resolution)
    )
    return uv, depth, valid


def _tile_layout(
    canonical_size: int,
    tile_size: int,
    tile_stride: int,
) -> List[Tuple[int, int, int, int]]:
    starts = list(range(0, canonical_size - tile_size + 1, tile_stride))
    if not starts or starts[-1] != canonical_size - tile_size:
        raise ValueError("tile layout does not land on the final image edge")
    return [
        (x0, y0, x0 + tile_size, y0 + tile_size)
        for y0 in starts
        for x0 in starts
    ]


def _rows_inside_tile(
    uv_full: torch.Tensor,
    valid: torch.Tensor,
    box: Sequence[int],
) -> torch.Tensor:
    x0, y0, x1, y1 = (float(value) for value in box)
    mask = (
        valid
        & (uv_full[:, 0] >= x0)
        & (uv_full[:, 0] < x1)
        & (uv_full[:, 1] >= y0)
        & (uv_full[:, 1] < y1)
    )
    return torch.where(mask)[0]


def _build_tile_camera_transform(
    *,
    tile_id: int,
    box: Sequence[int],
    global_camera: Mapping[str, float],
    canonical_size: int,
    output_size: int,
) -> TileCameraTransform:
    x0, y0, x1, y1 = (int(v) for v in box)
    crop_w = x1 - x0
    crop_h = y1 - y0
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"invalid tile box {tuple(box)}")

    scale_x = float(output_size) / float(crop_w)
    scale_y = float(output_size) / float(crop_h)
    full_focal = _focal_pixels(
        float(global_camera["camera_angle_x"]),
        int(canonical_size),
    )
    tile_fx = full_focal * scale_x
    tile_fy = full_focal * scale_y
    if not math.isclose(tile_fx, tile_fy, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"tile focal mismatch fx={tile_fx}, fy={tile_fy}")

    tile_fov = 2.0 * math.atan(float(output_size) / (2.0 * tile_fx))
    return TileCameraTransform(
        tile_id=int(tile_id),
        box=(x0, y0, x1, y1),
        output_size=int(output_size),
        camera_angle_x=float(tile_fov),
        distance=float(global_camera["distance"]),
        mesh_scale=float(global_camera["mesh_scale"]),
        full_focal_pixels=float(full_focal),
        tile_focal_pixels=float(tile_fx),
        crop_scale_x=float(scale_x),
        crop_scale_y=float(scale_y),
    )


def _global_q_to_tile_q(
    q_global: torch.Tensor,
    *,
    global_camera: Mapping[str, float],
    transform: TileCameraTransform,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Re-express selected global canonical points in the crop camera frame."""
    if q_global.ndim != 2 or q_global.shape[1] != 3:
        raise ValueError("q_global must be [N,3]")

    center = torch.tensor(
        [0.0, 0.0, -float(global_camera["distance"])],
        device=q_global.device,
        dtype=q_global.dtype,
    )
    global_points = center[None] + q_global / (
        2.0 * float(global_camera["mesh_scale"])
    )
    uv_full, _, valid = _project_camera_points(
        global_points,
        float(global_camera["camera_angle_x"]),
        IMAGE_CANONICAL,
    )
    uv_full = uv_full[0]
    valid = valid[0]
    if not bool(valid.all().item()):
        raise RuntimeError("selected rows contain invalid global camera projections")

    x0, y0, _, _ = transform.box
    uv_tile = torch.stack(
        [
            (uv_full[:, 0] - float(x0)) * float(transform.crop_scale_x),
            (uv_full[:, 1] - float(y0)) * float(transform.crop_scale_y),
        ],
        dim=1,
    )

    qz = q_global[:, 2]
    tile_depth = float(transform.distance) - qz / (
        2.0 * float(transform.mesh_scale)
    )
    if bool((tile_depth <= 0).any().item()):
        raise RuntimeError("tile canonical depth became non-positive")

    xt = (
        (uv_tile[:, 0] - float(transform.output_size) / 2.0)
        * tile_depth
        / float(transform.tile_focal_pixels)
    )
    yt = -(
        (uv_tile[:, 1] - float(transform.output_size) / 2.0)
        * tile_depth
        / float(transform.tile_focal_pixels)
    )
    q_raw = torch.stack(
        [
            2.0 * float(transform.mesh_scale) * xt,
            2.0 * float(transform.mesh_scale) * yt,
            qz,
        ],
        dim=1,
    )

    overflow = (q_raw.abs() - 1.0).clamp_min(0.0)
    overflow_rows = (overflow > 0).any(dim=1)
    stats = {
        "rows": int(q_raw.shape[0]),
        "raw_outside_rows": int(overflow_rows.sum().item()),
        "raw_outside_fraction": float(overflow_rows.float().mean().item()),
        "raw_max_overflow": float(overflow.max().item()),
        "q_raw_min": [float(v) for v in q_raw.amin(dim=0).detach().cpu().tolist()],
        "q_raw_max": [float(v) for v in q_raw.amax(dim=0).detach().cpu().tolist()],
    }
    return q_raw, uv_tile, stats


def _tile_coords_to_camera(
    coords: torch.Tensor,
    *,
    transform: TileCameraTransform,
) -> torch.Tensor:
    q = _endpoint_indices_to_q(coords[:, 1:4], GRID_SHAPE_1024).to(coords.device)
    center = torch.tensor(
        [0.0, 0.0, -float(transform.distance)],
        device=coords.device,
        dtype=q.dtype,
    )
    return center[None] + q / (2.0 * float(transform.mesh_scale))


def _composite_on_black(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", int(size))
    except Exception:
        return ImageFont.load_default()


def _depth_color(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=1)


def _draw_uv_points(
    image: Image.Image,
    uv: torch.Tensor,
    qz: torch.Tensor,
    output: Path,
    title: str,
    max_points: int = 16000,
) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    uv_cpu = uv.detach().cpu().float().numpy()
    qz_cpu = qz.detach().cpu().float().numpy()
    original_count = uv_cpu.shape[0]
    if original_count > max_points:
        ids = np.linspace(0, original_count - 1, max_points).round().astype(np.int64)
        uv_cpu = uv_cpu[ids]
        qz_cpu = qz_cpu[ids]
    colors = (_depth_color((qz_cpu + 1.0) * 0.5) * 255.0).astype(np.uint8)
    for (u, v), color in zip(uv_cpu, colors):
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        x, y = int(round(float(u))), int(round(float(v)))
        if 0 <= x < canvas.width and 0 <= y < canvas.height:
            draw.ellipse(
                (x - 2, y - 2, x + 2, y + 2),
                fill=tuple(color.tolist()) + (190,),
            )
    draw.rectangle((0, 0, canvas.width, 62), fill=(0, 0, 0, 205))
    draw.text(
        (12, 15),
        f"{title} | points={original_count:,}",
        fill=(255, 255, 255, 255),
        font=_font(24),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _save_density_image(
    uv: torch.Tensor,
    output: Path,
    *,
    resolution: int,
    bins: int = 128,
) -> None:
    array = uv.detach().cpu().float().numpy()
    array = array[np.isfinite(array).all(axis=1)]
    hist, _, _ = np.histogram2d(
        array[:, 1] if len(array) else np.empty(0),
        array[:, 0] if len(array) else np.empty(0),
        bins=bins,
        range=[[0, resolution], [0, resolution]],
    )
    hist = np.log1p(hist)
    if hist.max() > 0:
        hist /= hist.max()
    rgb = (
        _depth_color(hist.reshape(-1)).reshape(bins, bins, 3) * 255.0
    ).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB").resize(
        (resolution, resolution),
        Image.Resampling.NEAREST,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _save_quantization_error_image(
    reference: Image.Image,
    uv_source: torch.Tensor,
    uv_quantized: torch.Tensor,
    output: Path,
    max_lines: int = 1000,
) -> None:
    canvas = reference.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    source = uv_source.detach().cpu().float().numpy()
    target = uv_quantized.detach().cpu().float().numpy()
    count = source.shape[0]
    if count > max_lines:
        ids = np.linspace(0, count - 1, max_lines).round().astype(np.int64)
        source = source[ids]
        target = target[ids]
    for src, dst in zip(source, target):
        if not np.isfinite(src).all() or not np.isfinite(dst).all():
            continue
        x0, y0 = float(src[0]), float(src[1])
        x1, y1 = float(dst[0]), float(dst[1])
        if max(abs(x1 - x0), abs(y1 - y0)) > 0.2:
            draw.line((x0, y0, x1, y1), fill=(255, 230, 0, 150), width=1)
        draw.ellipse((x0 - 1, y0 - 1, x0 + 1, y0 + 1), fill=(255, 40, 40, 210))
        draw.ellipse((x1 - 1, y1 - 1, x1 + 1, y1 + 1), fill=(40, 255, 255, 210))
    draw.rectangle((0, 0, canvas.width, 62), fill=(0, 0, 0, 205))
    draw.text(
        (12, 15),
        "red=continuous geometry1024, cyan=final tile C64 projection",
        fill=(255, 255, 255, 255),
        font=_font(22),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _save_extra_comparisons(
    reference_path: Path,
    render_path: Path,
    output_dir: Path,
) -> Dict[str, str]:
    reference = _composite_on_black(Image.open(reference_path))
    rendered = _composite_on_black(Image.open(render_path))
    if rendered.size != reference.size:
        rendered = rendered.resize(reference.size, Image.Resampling.LANCZOS)

    ref = np.asarray(reference, dtype=np.float32)
    pred = np.asarray(rendered, dtype=np.float32)
    overlay = Image.blend(reference, rendered, 0.5)
    diff = np.abs(ref - pred).mean(axis=2) / 255.0
    heat = (
        _depth_color(diff.reshape(-1)).reshape(diff.shape[0], diff.shape[1], 3)
        * 255.0
    ).astype(np.uint8)
    diff_image = Image.fromarray(heat, mode="RGB")

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "overlay_50.png"
    diff_path = output_dir / "abs_diff_heatmap.png"
    triptych_path = output_dir / "triptych_reference_render_diff.png"
    overlay.save(overlay_path)
    diff_image.save(diff_path)

    w, h = reference.size
    triptych = Image.new("RGB", (w * 3, h + 64), (18, 18, 18))
    triptych.paste(reference, (0, 64))
    triptych.paste(rendered, (w, 64))
    triptych.paste(diff_image, (w * 2, 64))
    draw = ImageDraw.Draw(triptych)
    draw.text((12, 15), "reference", fill=(255, 255, 255), font=_font(28))
    draw.text((w + 12, 15), "render", fill=(255, 255, 255), font=_font(28))
    draw.text((w * 2 + 12, 15), "absolute RGB error", fill=(255, 255, 255), font=_font(28))
    triptych.save(triptych_path)
    return {
        "overlay_png": str(overlay_path),
        "diff_heatmap_png": str(diff_path),
        "triptych_png": str(triptych_path),
    }


def _estimate_camera(
    *,
    image_1024: Image.Image,
    output_dir: Path,
    manual_fov: float,
    mesh_scale: float,
    extend_pixel: int,
    image_resolution: int,
) -> Dict[str, float]:
    if manual_fov > 0:
        distance = distance_from_fov(
            float(manual_fov),
            torch.tensor([-1.0, 0.0, 0.0]),
            torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
            float(mesh_scale),
            int(image_resolution),
        )["distance_from_x"]
        return {
            "camera_angle_x": float(manual_fov),
            "distance": float(distance),
            "mesh_scale": float(mesh_scale),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / f"_global_shape_prior_moge_{time.time_ns()}.png"
    image_1024.save(temporary)
    model = load_moge_model(device="cuda")
    try:
        params = get_camera_params_wild_moge(
            str(temporary),
            model,
            device="cuda",
            mesh_scale=float(mesh_scale),
            extend_pixel=int(extend_pixel),
            image_resolution=int(image_resolution),
        )
    finally:
        model.cpu()
        del model
        temporary.unlink(missing_ok=True)
        _empty_cuda_cache()
    return {
        "camera_angle_x": float(params["camera_angle_x"]),
        "distance": float(params["distance"]),
        "mesh_scale": float(params["mesh_scale"]),
    }


def _sampler_params(
    args: argparse.Namespace,
    pipeline: Any,
) -> Dict[str, Dict[str, Any]]:
    return {
        "ss": {
            **pipeline.sparse_structure_sampler_params,
            "steps": 12,
            "guidance_strength": float(args.ss_guidance_strength),
            "guidance_rescale": float(args.ss_guidance_rescale),
            "rescale_t": float(args.ss_rescale_t),
        },
        "shape": {
            **pipeline.shape_slat_sampler_params,
            "steps": 12,
            "guidance_strength": float(args.shape_guidance_strength),
            "guidance_rescale": float(args.shape_guidance_rescale),
            "rescale_t": float(args.shape_rescale_t),
        },
        "texture": {
            **pipeline.tex_slat_sampler_params,
            "steps": 12,
            "guidance_strength": float(args.texture_guidance_strength),
            "guidance_rescale": float(args.texture_guidance_rescale),
            "rescale_t": float(args.texture_rescale_t),
        },
    }


def _extract_samples(result: Any, description: str) -> SparseTensor:
    samples = getattr(result, "samples", result)
    if not isinstance(samples, SparseTensor):
        raise TypeError(f"{description}: sampler did not return SparseTensor samples")
    return samples


def _run_sampler_full(
    *,
    pipeline: Any,
    sampler: Any,
    model: torch.nn.Module,
    noise: SparseTensor,
    condition: Mapping[str, Any],
    params: Mapping[str, Any],
    description: str,
    concat_cond: Optional[SparseTensor] = None,
) -> Tuple[SparseTensor, float]:
    if concat_cond is not None and not torch.equal(noise.coords, concat_cond.coords):
        raise RuntimeError(f"{description}: noise and concat coordinates differ")
    if pipeline.low_vram:
        model.to(pipeline.device)

    call: Dict[str, Any] = {
        **condition,
        **dict(params),
        "verbose": True,
        "tqdm_desc": description,
        "record_trajectory": False,
        "return_model_history": False,
    }
    if concat_cond is not None:
        call["concat_cond"] = concat_cond

    started = time.perf_counter()
    result = sampler.sample(model, noise, **call)
    _sync_cuda()
    elapsed = time.perf_counter() - started
    samples = _extract_samples(result, description)
    if not torch.equal(samples.coords, noise.coords):
        raise RuntimeError(f"{description}: sampler changed sparse coordinates")

    if pipeline.low_vram:
        model.cpu()
        _empty_cuda_cache()
    print(
        f"[flow] {description}: tokens={noise.feats.shape[0]:,} "
        f"channels={noise.feats.shape[1]} seconds={elapsed:.3f}"
    )
    return samples, elapsed


def _run_shape512_and_upsample_c64(
    *,
    pipeline: Any,
    image_512: Image.Image,
    coords32: torch.Tensor,
    camera: Mapping[str, float],
    params: Mapping[str, Mapping[str, Any]],
    seed: int,
) -> Tuple[torch.Tensor, float]:
    condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_512,
        [image_512],
        coords32,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
        grid_resolution_override=GRID_SS,
    )
    model = pipeline.models["shape_slat_flow_model_512"]
    noise = SparseTensor(
        feats=_randn(
            coords32.shape[0],
            int(model.in_channels),
            device=pipeline.device,
            seed=int(seed),
        ),
        coords=coords32,
    )
    shape512_norm, elapsed = _run_sampler_full(
        pipeline=pipeline,
        sampler=pipeline.shape_slat_sampler,
        model=model,
        noise=noise,
        condition=condition,
        params=params["shape"],
        description="Global official shape 512",
    )
    shape512_denorm = _denormalize_sparse(
        shape512_norm,
        pipeline.shape_slat_normalization,
    )
    coords64 = _learned_upsample_shape512_to_c64(pipeline, shape512_denorm)
    del condition, noise, shape512_norm, shape512_denorm
    _empty_cuda_cache()
    return coords64, elapsed


def _run_shape1024(
    *,
    pipeline: Any,
    image_1024: Image.Image,
    coords64: torch.Tensor,
    camera: Mapping[str, float],
    params: Mapping[str, Mapping[str, Any]],
    seed: int,
    description: str,
) -> Tuple[SparseTensor, SparseTensor, float]:
    condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [image_1024],
        coords64,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
        grid_resolution_override=GRID_SHAPE_1024,
    )
    model = pipeline.models["shape_slat_flow_model_1024"]
    noise = SparseTensor(
        feats=_randn(
            coords64.shape[0],
            int(model.in_channels),
            device=pipeline.device,
            seed=int(seed),
        ),
        coords=coords64,
    )
    shape_norm, elapsed = _run_sampler_full(
        pipeline=pipeline,
        sampler=pipeline.shape_slat_sampler,
        model=model,
        noise=noise,
        condition=condition,
        params=params["shape"],
        description=description,
    )
    shape_denorm = _denormalize_sparse(
        shape_norm,
        pipeline.shape_slat_normalization,
    )
    del condition, noise
    _empty_cuda_cache()
    return shape_norm, shape_denorm, elapsed


def _run_global_official_geometry_to_shape1024(
    *,
    pipeline: Any,
    image_512: Image.Image,
    image_1024: Image.Image,
    camera: Mapping[str, float],
    params: Mapping[str, Mapping[str, Any]],
    seed: int,
    max_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, ShapeResult]:
    """Official global C32 -> shape512 -> C64 -> shape1024; no texture/decode."""
    condition_ss = pipeline.get_proj_cond_ss(
        [image_512],
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
    )
    _seed_everything(seed)
    coords32 = pipeline.sample_sparse_structure(
        condition_ss,
        resolution=GRID_SS,
        sampler_params=dict(params["ss"]),
    )
    del condition_ss
    if coords32.shape[0] == 0:
        raise RuntimeError("global sparse structure is empty")
    print(f"[global] C32 tokens={coords32.shape[0]:,}")

    coords64, shape512_seconds = _run_shape512_and_upsample_c64(
        pipeline=pipeline,
        image_512=image_512,
        coords32=coords32,
        camera=camera,
        params=params,
        seed=seed + 101,
    )
    if coords64.shape[0] > max_tokens:
        raise RuntimeError(
            f"global C64 support has {coords64.shape[0]:,} tokens; "
            f"exceeds --max-num-tokens={max_tokens:,}"
        )
    print(f"[global] learned C64 tokens={coords64.shape[0]:,}")

    shape_norm, shape_denorm, shape1024_seconds = _run_shape1024(
        pipeline=pipeline,
        image_1024=image_1024,
        coords64=coords64,
        camera=camera,
        params=params,
        seed=seed + 201,
        description="Global official shape 1024",
    )
    return coords32, coords64, ShapeResult(
        shape_norm=shape_norm,
        shape_denorm=shape_denorm,
        shape512_seconds=shape512_seconds,
        shape1024_seconds=shape1024_seconds,
    )


def _run_tile_official_shape_texture1024(
    *,
    pipeline: Any,
    tile_image: Image.Image,
    coords64: torch.Tensor,
    tile_camera: Mapping[str, float],
    params: Mapping[str, Mapping[str, Any]],
    seed: int,
    label: str,
) -> ModelResult:
    """Official tile shape1024 -> texture1024 on the supplied C64 support."""
    shape_norm, shape_denorm, shape_seconds = _run_shape1024(
        pipeline=pipeline,
        image_1024=tile_image.convert("RGB"),
        coords64=coords64,
        camera=tile_camera,
        params=params,
        seed=seed + 201,
        description=f"{label} shape 1024",
    )

    texture_condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [tile_image.convert("RGB")],
        coords64,
        camera_angle_x=float(tile_camera["camera_angle_x"]),
        distance=float(tile_camera["distance"]),
        mesh_scale=float(tile_camera["mesh_scale"]),
        grid_resolution_override=GRID_SHAPE_1024,
    )
    texture_model = pipeline.models["tex_slat_flow_model_1024"]
    texture_channels = int(texture_model.in_channels) - int(shape_norm.feats.shape[1])
    if texture_channels <= 0:
        raise RuntimeError(f"invalid texture noise channel count {texture_channels}")
    texture_noise = SparseTensor(
        feats=_randn(
            coords64.shape[0],
            texture_channels,
            device=pipeline.device,
            seed=seed + 301,
        ),
        coords=coords64,
    )
    texture_norm, texture_seconds = _run_sampler_full(
        pipeline=pipeline,
        sampler=pipeline.tex_slat_sampler,
        model=texture_model,
        noise=texture_noise,
        condition=texture_condition,
        params=params["texture"],
        description=f"{label} texture 1024",
        concat_cond=shape_norm,
    )
    texture_denorm = _denormalize_sparse(
        texture_norm,
        pipeline.tex_slat_normalization,
    )
    del texture_condition, texture_noise
    _empty_cuda_cache()
    return ModelResult(
        shape_norm=shape_norm,
        shape_denorm=shape_denorm,
        texture_norm=texture_norm,
        texture_denorm=texture_denorm,
        shape_seconds=shape_seconds,
        texture_seconds=texture_seconds,
    )


def _save_postprocess_cache(
    *,
    pipeline: Any,
    result: ModelResult,
    output_dir: Path,
    camera: Mapping[str, float],
    seed: int,
    label: str,
    texture_size: int,
    decimation_target: int,
    export_glb: bool,
) -> Dict[str, Any]:
    meshes = pipeline.decode_latent(
        result.shape_denorm,
        result.texture_denorm,
        DECODE_TILE,
    )
    mesh = meshes[0]
    vertices = int(mesh.vertices.shape[0])
    faces = int(mesh.faces.shape[0])
    print(f"[decode] {label}: vertices={vertices:,} faces={faces:,}")

    effective_target = faces if decimation_target <= 0 else min(
        faces, int(decimation_target)
    )
    export_kwargs = {
        "vertices": mesh.vertices,
        "faces": mesh.faces,
        "attr_volume": mesh.attrs,
        "coords": mesh.coords,
        "attr_layout": pipeline.pbr_attr_layout,
        "grid_size": DECODE_TILE,
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "decimation_target": effective_target,
        "texture_size": int(texture_size),
        "remesh": False,
        "use_tqdm": True,
        "verbose": False,
    }
    try:
        from pixal3d_directory_texture_eval import save_to_glb_cache
    except Exception as exc:
        raise RuntimeError("cannot import save_to_glb_cache") from exc

    cache_dir = output_dir / "postprocess_cache"
    manifest = save_to_glb_cache(
        cache_dir,
        export_kwargs,
        extra_metadata={
            "camera_params": dict(camera),
            "pipeline_resolution": DECODE_TILE,
            "actual_grid_resolution": GRID_SHAPE_1024,
            "seed": int(seed),
            "decoder_vertices": vertices,
            "decoder_faces": faces,
            "experiment": "global_geometry1024_prior_tile_eval_v2",
            "label": label,
        },
        overwrite=True,
    )
    manifest_path = cache_dir / "manifest.json"
    published = json.loads(manifest_path.read_text(encoding="utf-8"))
    published["grid_size"] = DECODE_TILE
    published["aabb"] = export_kwargs["aabb"]
    _atomic_json(manifest_path, published)

    glb_path: Optional[Path] = None
    if export_glb:
        if o_voxel is None:
            raise RuntimeError(
                "o_voxel is unavailable; use --no-export-glb to disable GLB export"
            )
        glb = o_voxel.postprocess.to_glb(**export_kwargs)
        glb.apply_transform(PIXAL3D_EXPORT_ROTATION)
        glb_path = output_dir / "model.glb"
        glb.export(str(glb_path), extension_webp=False)
        print(f"[glb] {label}: {glb_path}")

    return {
        "cache_dir": str(cache_dir),
        "manifest": manifest,
        "decoder_vertices": vertices,
        "decoder_faces": faces,
        "glb": None if glb_path is None else str(glb_path),
    }

def _run_evaluator(
    *,
    repository_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    reference_image: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    script = repository_dir / "render_pixal3d_cache_no_uv.py"
    if not script.is_file():
        raise FileNotFoundError(script)

    command = [
        sys.executable,
        str(script),
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(output_dir),
        "--reference-image",
        str(reference_image),
        "--lights",
        str(args.light),
        "--engine",
        str(args.render_engine),
        "--material-mode",
        "pbr",
        "--render-resolution",
        str(int(args.render_resolution)),
        "--metric-resolution",
        str(int(args.metric_resolution)),
        "--samples",
        str(int(args.blender_samples)),
        "--lpips-net",
        str(args.lpips_net),
        "--metric-device",
        str(args.metric_device),
        "--blender",
        str(args.blender),
        "--overwrite-renders",
        "--overwrite-package",
    ]
    if int(args.render_max_faces) > 0:
        command.extend(["--max-faces", str(int(args.render_max_faces))])
    if bool(args.skip_lpips):
        command.append("--skip-lpips")

    print("[render-eval-command] " + shlex.join(command))
    process = subprocess.run(command, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"renderer/evaluator exited with code {process.returncode}")

    metrics_path = output_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload.get("metrics", []))
    success_rows = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("status") == "success"
    ]
    if not success_rows:
        raise RuntimeError(f"no successful metrics row in {metrics_path}")
    return {
        "metrics_payload": payload,
        "metrics_row": success_rows[0],
        "metrics_json": str(metrics_path),
    }


def _evaluate_tile_result(
    *,
    pipeline: Any,
    result: ModelResult,
    output_dir: Path,
    camera: Mapping[str, float],
    seed: int,
    label: str,
    reference_image: Path,
    args: argparse.Namespace,
    repository_dir: Path,
) -> Dict[str, Any]:
    cache = _save_postprocess_cache(
        pipeline=pipeline,
        result=result,
        output_dir=output_dir,
        camera=camera,
        seed=seed,
        label=label,
        texture_size=int(args.texture_size),
        decimation_target=int(args.decimation_target),
        export_glb=bool(args.export_glb),
    )
    eval_result = _run_evaluator(
        repository_dir=repository_dir,
        cache_dir=Path(cache["cache_dir"]),
        output_dir=output_dir / "aligned_eval",
        reference_image=reference_image,
        args=args,
    )
    metric_row = dict(eval_result["metrics_row"])
    extras = _save_extra_comparisons(
        Path(metric_row["original_png"]),
        Path(metric_row["render_png"]),
        output_dir / "comparisons",
    )
    return {**cache, **metric_row, **extras}


def _parse_tile_ids(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not value.strip():
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _prepare_tile_reference(
    image_4096: Image.Image,
    box: Tuple[int, int, int, int],
    tile_dir: Path,
) -> Image.Image:
    tile = image_4096.crop(box).convert("RGBA")
    if tile.size != (IMAGE_FLOW, IMAGE_FLOW):
        tile = tile.resize((IMAGE_FLOW, IMAGE_FLOW), Image.Resampling.LANCZOS)
    reference = _composite_on_black(tile)
    tile_dir.mkdir(parents=True, exist_ok=True)
    reference.save(tile_dir / "reference_tile.png")
    return reference


def _prepare_tile_c64_from_global_geometry1024(
    *,
    reference: Image.Image,
    global_geometry1024: torch.Tensor,
    rows: torch.Tensor,
    global_camera: Mapping[str, float],
    transform: TileCameraTransform,
    output_dir: Path,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Transfer true global geometry-1024 points into one tile."""
    selected = global_geometry1024.index_select(0, rows)
    q_global = _voxel_indices_to_q(
        selected[:, 1:4], GEOMETRY_RESOLUTION
    ).to(selected.device)
    q_tile_raw, uv_all, transform_stats = _global_q_to_tile_q(
        q_global,
        global_camera=global_camera,
        transform=transform,
    )

    inside = (q_tile_raw.abs() <= 1.0).all(dim=1)
    if not bool(inside.any().item()):
        raise RuntimeError("no projected geometry1024 point lies in tile canonical volume")
    q_tile = q_tile_raw[inside]
    uv_continuous = uv_all[inside]
    q_global_inside = q_global[inside]
    selected_inside = selected[inside]

    ids1024_source = _q_to_voxel_indices(q_tile, GEOMETRY_RESOLUTION)
    coords1024_source = torch.cat(
        [
            torch.zeros(
                (ids1024_source.shape[0], 1),
                device=ids1024_source.device,
                dtype=torch.int32,
            ),
            ids1024_source,
        ],
        dim=1,
    )
    coords1024 = torch.unique(coords1024_source, dim=0)
    if coords1024.numel() == 0:
        raise RuntimeError("tile has no unique geometry1024 point")

    ids64 = _geometry1024_to_c64_indices(coords1024[:, 1:4])
    coords64_all = torch.cat([coords1024[:, :1], ids64], dim=1).to(torch.int32)
    coords64 = torch.unique(coords64_all, dim=0)
    if coords64.numel() == 0:
        raise RuntimeError("tile geometry1024 prior produced no C64 support")

    center = torch.tensor(
        [0.0, 0.0, -float(transform.distance)],
        device=coords1024.device,
        dtype=torch.float32,
    )

    q_geometry = _voxel_indices_to_q(
        coords1024[:, 1:4], GEOMETRY_RESOLUTION
    ).to(coords1024.device)
    camera_geometry = center[None] + q_geometry / (2.0 * float(transform.mesh_scale))
    uv_geometry, _, visible_geometry = _project_camera_points(
        camera_geometry, float(transform.camera_angle_x), IMAGE_FLOW
    )
    uv_geometry = uv_geometry[0]
    visible_geometry = visible_geometry[0]

    camera_c64 = _tile_coords_to_camera(coords64, transform=transform)
    uv_c64, _, visible_c64 = _project_camera_points(
        camera_c64, float(transform.camera_angle_x), IMAGE_FLOW
    )
    uv_c64 = uv_c64[0]
    visible_c64 = visible_c64[0]
    qz_c64 = _endpoint_indices_to_q(coords64[:, 3:4], GRID_SHAPE_1024)[:, 0]

    ids64_source = _geometry1024_to_c64_indices(ids1024_source)
    q64_source = _endpoint_indices_to_q(ids64_source, GRID_SHAPE_1024).to(q_tile.device)
    camera64_source = center[None] + q64_source / (2.0 * float(transform.mesh_scale))
    uv64_source, _, _ = _project_camera_points(
        camera64_source, float(transform.camera_angle_x), IMAGE_FLOW
    )
    uv64_source = uv64_source[0]
    pixel_error = torch.linalg.vector_norm(uv64_source - uv_continuous, dim=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    _draw_uv_points(
        reference,
        uv_continuous,
        q_global_inside[:, 2],
        output_dir / "selected_global_geometry1024_projection.png",
        "selected global geometry1024 points",
    )
    _draw_uv_points(
        reference,
        uv_geometry[visible_geometry],
        q_geometry[visible_geometry, 2],
        output_dir / "tile_geometry1024_support_overlay.png",
        "unique tile geometry1024 prior",
    )
    _save_density_image(
        uv_geometry[visible_geometry],
        output_dir / "tile_geometry1024_support_density.png",
        resolution=IMAGE_FLOW,
    )
    _draw_uv_points(
        reference,
        uv_c64[visible_c64],
        qz_c64[visible_c64],
        output_dir / "tile_c64_support_overlay.png",
        "final official tile C64 support",
    )
    _save_density_image(
        uv_c64[visible_c64],
        output_dir / "tile_c64_support_density.png",
        resolution=IMAGE_FLOW,
    )
    _save_quantization_error_image(
        reference,
        uv_continuous,
        uv64_source,
        output_dir / "geometry1024_to_c64_quantization_error.png",
    )

    stats = {
        "selected_global_geometry1024_rows": int(rows.numel()),
        "tile_canonical_inside_rows": int(inside.sum().item()),
        "tile_canonical_outside_rows": int((~inside).sum().item()),
        "tile_geometry1024_rows_before_unique": int(coords1024_source.shape[0]),
        "tile_geometry1024_unique_points": int(coords1024.shape[0]),
        "tile_geometry1024_merge_rows": int(
            coords1024_source.shape[0] - coords1024.shape[0]
        ),
        "tile_c64_rows_before_unique": int(coords64_all.shape[0]),
        "tile_c64_unique_tokens": int(coords64.shape[0]),
        "geometry1024_to_c64_merge_rows": int(
            coords64_all.shape[0] - coords64.shape[0]
        ),
        "visible_tile_geometry1024_points": int(visible_geometry.sum().item()),
        "visible_tile_c64_tokens": int(visible_c64.sum().item()),
        "transform": asdict(transform),
        "transform_stats": transform_stats,
        "pixel_error_mean": float(pixel_error.mean().item()),
        "pixel_error_p95": float(torch.quantile(pixel_error, 0.95).item()),
        "pixel_error_max": float(pixel_error.max().item()),
    }
    _atomic_json(output_dir / "support_stats.json", stats)
    torch.save(
        {
            "global_rows": rows.detach().cpu(),
            "selected_global_geometry1024": selected.detach().cpu(),
            "selected_inside_geometry1024": selected_inside.detach().cpu(),
            "q_global": q_global.detach().cpu(),
            "q_tile_raw": q_tile_raw.detach().cpu(),
            "inside_tile_canonical": inside.detach().cpu(),
            "coords1024_per_source": coords1024_source.detach().cpu(),
            "coords1024_unique": coords1024.detach().cpu(),
            "coords64_unique": coords64.detach().cpu(),
            "uv_continuous": uv_continuous.detach().cpu(),
            "uv_c64_per_source": uv64_source.detach().cpu(),
            "pixel_error": pixel_error.detach().cpu(),
        },
        output_dir / "support_debug.pt",
    )
    return coords64, coords1024, stats

def _resize_panel(path: Optional[Path], size: int = 512) -> Image.Image:
    if path is None or not path.is_file():
        return Image.new("RGB", (size, size), (30, 30, 30))
    image = Image.open(path).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def _label_panel(image: Image.Image, text: str) -> Image.Image:
    canvas = image.convert("RGBA")
    lines = text.splitlines()
    bar_height = max(88, 16 + 38 * len(lines))
    overlay = Image.new("RGBA", (canvas.width, bar_height), (0, 0, 0, 205))
    canvas.alpha_composite(overlay, dest=(0, 0))
    draw = ImageDraw.Draw(canvas)
    font = _font(28)
    y = 10
    for line in lines:
        draw.text((12, y), line, fill=(255, 255, 255, 255), font=font)
        y += 38
    return canvas.convert("RGB")

def _save_tile_comparison_sheet(
    *,
    reference_path: Path,
    support_overlay_path: Path,
    route_summary: Optional[Mapping[str, Any]],
    geometry1024_count: int,
    c64_count: int,
    output_path: Path,
) -> str:
    reference = _label_panel(
        _resize_panel(reference_path),
        "Reference 4096 crop -> 1024",
    )
    support = _label_panel(
        _resize_panel(support_overlay_path),
        (
            "Global geometry1024 prior\n"
            f"geometry1024={geometry1024_count:,}  C64={c64_count:,}"
        ),
    )

    if route_summary is None:
        render = _label_panel(_resize_panel(None), "Tile generation\nfailed")
        difference = _label_panel(_resize_panel(None), "Absolute error\nunavailable")
    else:
        render_path = Path(str(route_summary["render_png"])) if route_summary.get("render_png") else None
        diff_path = Path(str(route_summary["diff_heatmap_png"])) if route_summary.get("diff_heatmap_png") else None
        render = _label_panel(
            _resize_panel(render_path),
            (
                "Tile official 1024 generation\n"
                f"PSNR={route_summary.get('psnr_db')} "
                f"SSIM={route_summary.get('ssim')} "
                f"LPIPS={route_summary.get('lpips')}"
            ),
        )
        difference = _label_panel(
            _resize_panel(diff_path),
            "Absolute RGB error heatmap",
        )

    panels = [reference, support, render, difference]
    size = panels[0].width
    canvas = Image.new("RGB", (size * 2, size * 2), (18, 18, 18))
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % 2) * size, (index // 2) * size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return str(output_path)

def _write_contact_sheets(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> List[str]:
    rows = [
        row
        for row in records
        if row.get("status") == "success" and row.get("comparison_png")
    ]
    outputs: List[str] = []
    per_page = 6
    for start in range(0, len(rows), per_page):
        page = rows[start : start + per_page]
        cell_w = 1024
        cell_h = 1024
        cols = 2
        row_count = math.ceil(len(page) / cols)
        canvas = Image.new(
            "RGB",
            (cell_w * cols, cell_h * row_count),
            (20, 20, 20),
        )
        for index, row in enumerate(page):
            image = Image.open(str(row["comparison_png"])).convert("RGB")
            image.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            x0 = (index % cols) * cell_w + (cell_w - image.width) // 2
            y0 = (index // cols) * cell_h + (cell_h - image.height) // 2
            canvas.paste(image, (x0, y0))
        path = output_dir / f"all_tiles_global_shape_prior_{start // per_page:02d}.png"
        canvas.save(path)
        outputs.append(str(path))
    return outputs


def run(args: argparse.Namespace) -> None:
    if int(args.tile_size) != DEFAULT_TILE_SIZE:
        raise ValueError(f"this test requires --tile-size={DEFAULT_TILE_SIZE}")
    if int(args.tile_stride) != DEFAULT_TILE_STRIDE:
        raise ValueError(f"this test requires --tile-stride={DEFAULT_TILE_STRIDE}")

    repository_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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

    global_camera = _estimate_camera(
        image_1024=image_1024,
        output_dir=output_dir,
        manual_fov=float(args.fov),
        mesh_scale=float(args.mesh_scale),
        extend_pixel=int(args.extend_pixel),
        image_resolution=int(args.camera_image_resolution),
    )
    _atomic_json(output_dir / "global_camera.json", global_camera)
    print(
        f"[global-camera] fov={global_camera['camera_angle_x']:.8f} "
        f"distance={global_camera['distance']:.8f} "
        f"mesh_scale={global_camera['mesh_scale']:.8f}"
    )

    params = _sampler_params(args, pipeline)
    print("[global] official SS -> shape512 -> learned C64 -> shape1024")
    coords32, coords64, global_shape = _run_global_official_geometry_to_shape1024(
        pipeline=pipeline,
        image_512=image_512,
        image_1024=image_1024,
        camera=global_camera,
        params=params,
        seed=int(args.seed),
        max_tokens=int(args.max_num_tokens),
    )

    print("[global] FOUR decoder subdivisions: C64 latent -> geometry1024")
    global_geometry1024, upsample_stats = _learned_upsample_shape1024_to_geometry1024(
        pipeline, global_shape.shape_denorm
    )
    if global_geometry1024.shape[0] > int(args.max_global_geometry_points):
        raise RuntimeError(
            f"global geometry1024 has {global_geometry1024.shape[0]:,} points; "
            f"exceeds --max-global-geometry-points="
            f"{int(args.max_global_geometry_points):,}"
        )
    print(f"[global] geometry1024 points={global_geometry1024.shape[0]:,}")

    global_dir = output_dir / "global_geometry1024_prior"
    global_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "coords32": coords32.detach().cpu(),
            "coords64": coords64.detach().cpu(),
            "global_geometry1024": global_geometry1024.detach().cpu(),
            "shape1024_norm_feats": global_shape.shape_norm.feats.detach().cpu(),
            "shape1024_denorm_feats": global_shape.shape_denorm.feats.detach().cpu(),
        },
        global_dir / "global_geometry1024_prior.pt",
    )
    global_summary = {
        "global_c32_tokens": int(coords32.shape[0]),
        "global_c64_tokens": int(coords64.shape[0]),
        "global_geometry1024_points": int(global_geometry1024.shape[0]),
        "shape512_seconds": float(global_shape.shape512_seconds),
        "shape1024_seconds": float(global_shape.shape1024_seconds),
        "four_step_upsample": upsample_stats,
        "texture_generated": False,
        "decoded": False,
    }
    _atomic_json(global_dir / "summary.json", global_summary)
    del global_shape
    _empty_cuda_cache()

    global_camera_points = _global_geometry1024_to_camera(
        global_geometry1024, camera=global_camera
    )
    uv_full, _, valid_full = _project_camera_points(
        global_camera_points,
        float(global_camera["camera_angle_x"]),
        IMAGE_CANONICAL,
    )
    uv_full = uv_full[0]
    valid_full = valid_full[0]

    boxes = _tile_layout(IMAGE_CANONICAL, int(args.tile_size), int(args.tile_stride))
    selected_ids = _parse_tile_ids(args.tile_ids)
    if selected_ids is not None:
        invalid = sorted(i for i in selected_ids if i < 0 or i >= len(boxes))
        if invalid:
            raise ValueError(f"invalid tile ids: {invalid}; valid range is 0-{len(boxes)-1}")

    records: List[Dict[str, Any]] = []
    attempted = 0
    for tile_id, box in enumerate(boxes):
        if selected_ids is not None and tile_id not in selected_ids:
            continue
        if args.max_tiles is not None and attempted >= int(args.max_tiles):
            break
        attempted += 1

        tile_dir = output_dir / "tiles" / f"tile_{tile_id:02d}"
        reference_tile = _prepare_tile_reference(image_4096, box, tile_dir)
        rows = _rows_inside_tile(uv_full, valid_full, box)
        transform = _build_tile_camera_transform(
            tile_id=tile_id,
            box=box,
            global_camera=global_camera,
            canonical_size=IMAGE_CANONICAL,
            output_size=IMAGE_FLOW,
        )
        tile_camera = {
            "camera_angle_x": float(transform.camera_angle_x),
            "distance": float(transform.distance),
            "mesh_scale": float(transform.mesh_scale),
        }
        print(
            f"[tile {tile_id:02d}] projected geometry1024 rows={rows.numel():,} "
            f"box={box} fov={tile_camera['camera_angle_x']:.8f}"
        )

        if rows.numel() == 0:
            record = {
                "status": "skipped",
                "tile_id": int(tile_id),
                "box": list(box),
                "reason": "no global geometry1024 projection inside tile",
                "selected_global_geometry1024_rows": 0,
                "tile_geometry1024_points": 0,
                "tile_c64_tokens": 0,
            }
            records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            continue

        support_dir = tile_dir / "global_geometry1024_prior_support"
        try:
            coords64_tile, geometry1024_tile, support_stats = (
                _prepare_tile_c64_from_global_geometry1024(
                    reference=reference_tile,
                    global_geometry1024=global_geometry1024,
                    rows=rows,
                    global_camera=global_camera,
                    transform=transform,
                    output_dir=support_dir,
                )
            )
        except Exception as exc:
            record = {
                "status": "failed",
                "tile_id": int(tile_id),
                "box": list(box),
                "reason": f"support preparation failed: {type(exc).__name__}: {exc}",
                "selected_global_geometry1024_rows": int(rows.numel()),
                "tile_geometry1024_points": 0,
                "tile_c64_tokens": 0,
            }
            records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            print(f"[tile-support-error] tile={tile_id:02d}: {record['reason']}")
            _empty_cuda_cache()
            continue

        geometry_count = int(geometry1024_tile.shape[0])
        tile_tokens = int(coords64_tile.shape[0])
        if tile_tokens < int(args.min_tile_tokens):
            record = {
                "status": "skipped",
                "tile_id": int(tile_id),
                "box": list(box),
                "reason": (
                    f"tile C64 tokens {tile_tokens} < "
                    f"--min-tile-tokens={int(args.min_tile_tokens)}"
                ),
                "selected_global_geometry1024_rows": int(rows.numel()),
                "tile_geometry1024_points": geometry_count,
                "tile_c64_tokens": tile_tokens,
                "support_stats": support_stats,
            }
            records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            print(
                f"[tile-skip] tile={tile_id:02d} "
                f"geometry1024={geometry_count:,} C64={tile_tokens:,}"
            )
            continue
        if tile_tokens > int(args.max_num_tokens):
            record = {
                "status": "failed",
                "tile_id": int(tile_id),
                "box": list(box),
                "reason": (
                    f"tile C64 tokens {tile_tokens} exceed "
                    f"--max-num-tokens={int(args.max_num_tokens)}"
                ),
                "selected_global_geometry1024_rows": int(rows.numel()),
                "tile_geometry1024_points": geometry_count,
                "tile_c64_tokens": tile_tokens,
                "support_stats": support_stats,
            }
            records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            continue

        route_summary: Optional[Dict[str, Any]] = None
        failure: Optional[str] = None
        seed_tile = int(args.seed) + tile_id * 1000 + 1
        try:
            result = _run_tile_official_shape_texture1024(
                pipeline=pipeline,
                tile_image=reference_tile,
                coords64=coords64_tile,
                tile_camera=tile_camera,
                params=params,
                seed=seed_tile,
                label=f"Tile {tile_id:02d} global-geometry1024-prior",
            )
            route_dir = tile_dir / "local_official_1024_generation"
            route_summary = _evaluate_tile_result(
                pipeline=pipeline,
                result=result,
                output_dir=route_dir,
                camera=tile_camera,
                seed=seed_tile,
                label=f"Tile {tile_id:02d} global-geometry1024-prior",
                reference_image=tile_dir / "reference_tile.png",
                args=args,
                repository_dir=repository_dir,
            )
            route_summary.update(
                {
                    "tile_geometry1024_points": geometry_count,
                    "tile_c64_tokens": tile_tokens,
                    "shape1024_seconds": float(result.shape_seconds),
                    "texture1024_seconds": float(result.texture_seconds),
                }
            )
            _atomic_json(route_dir / "summary.json", route_summary)
            del result
            _empty_cuda_cache()
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            print(f"[tile-generation-error] tile={tile_id:02d}: {failure}")
            _empty_cuda_cache()

        comparison_path = _save_tile_comparison_sheet(
            reference_path=tile_dir / "reference_tile.png",
            support_overlay_path=support_dir / "tile_geometry1024_support_overlay.png",
            route_summary=route_summary,
            geometry1024_count=geometry_count,
            c64_count=tile_tokens,
            output_path=tile_dir / "comparison_reference_support_render_diff.png",
        )

        record = {
            "status": "success" if route_summary is not None else "failed",
            "tile_id": int(tile_id),
            "box": list(box),
            "selected_global_geometry1024_rows": int(rows.numel()),
            "tile_geometry1024_points": geometry_count,
            "tile_c64_tokens": tile_tokens,
            "tile_camera": tile_camera,
            "support_stats": support_stats,
            "comparison_png": comparison_path,
            "psnr_db": None if route_summary is None else route_summary.get("psnr_db"),
            "ssim": None if route_summary is None else route_summary.get("ssim"),
            "lpips": None if route_summary is None else route_summary.get("lpips"),
            "render_png": None if route_summary is None else route_summary.get("render_png"),
            "triptych_png": None if route_summary is None else route_summary.get("triptych_png"),
            "glb": None if route_summary is None else route_summary.get("glb"),
            "shape1024_seconds": None if route_summary is None else route_summary.get("shape1024_seconds"),
            "texture1024_seconds": None if route_summary is None else route_summary.get("texture1024_seconds"),
            "failure": failure,
        }
        records.append(record)
        _atomic_json(tile_dir / "summary.json", record)
        print(
            f"[tile-summary] tile={tile_id:02d} "
            f"geometry1024={geometry_count:,} C64={tile_tokens:,} "
            f"PSNR={record['psnr_db']} SSIM={record['ssim']} "
            f"LPIPS={record['lpips']}"
        )

    aggregate_csv = output_dir / "aggregate_metrics.csv"
    _write_csv(aggregate_csv, records)
    contact_sheets = _write_contact_sheets(records, output_dir)
    summary = {
        "format": "pixal3d_global_geometry1024_prior_tile_eval_v2",
        "image": str(Path(args.image).expanduser().resolve()),
        "global_camera": global_camera,
        "global_geometry": global_summary,
        "tile_size": int(args.tile_size),
        "tile_stride": int(args.tile_stride),
        "min_tile_tokens": int(args.min_tile_tokens),
        "max_num_tokens": int(args.max_num_tokens),
        "max_global_geometry_points": int(args.max_global_geometry_points),
        "export_glb": bool(args.export_glb),
        "attempted_tiles": attempted,
        "recorded_tiles": len(records),
        "successful_tiles": sum(row.get("status") == "success" for row in records),
        "skipped_tiles": sum(row.get("status") == "skipped" for row in records),
        "failed_tiles": sum(row.get("status") == "failed" for row in records),
        "aggregate_csv": str(aggregate_csv),
        "contact_sheets": contact_sheets,
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(f"[summary] {output_dir / 'summary.json'}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate official tile 1024 generation from a global geometry-1024 "
            "prior produced by four decoder subdivisions"
        )
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--tile-stride", type=int, default=DEFAULT_TILE_STRIDE)
    parser.add_argument(
        "--tile-ids",
        default=None,
        help="comma-separated tile ids; omitted means all 49 tiles",
    )
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--min-tile-tokens", type=int, default=1000)
    parser.add_argument("--max-num-tokens", type=int, default=160000)
    parser.add_argument(
        "--max-global-geometry-points",
        type=int,
        default=5000000,
        help="safety cap after the four global decoder subdivisions",
    )

    parser.add_argument("--fov", type=float, default=-1.0)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument("--camera-image-resolution", type=int, default=1024)

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
        "--low-vram", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--export-glb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="export model.glb for each successful tile",
    )
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--decimation-target", type=int, default=0)

    parser.add_argument("--light", default="studio")
    parser.add_argument(
        "--render-engine", choices=("cycles", "eevee"), default="cycles"
    )
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--metric-resolution", type=int, default=1024)
    parser.add_argument("--blender-samples", type=int, default=64)
    parser.add_argument("--render-max-faces", type=int, default=0)
    parser.add_argument(
        "--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg"
    )
    parser.add_argument("--metric-device", default="cuda")
    parser.add_argument(
        "--skip-lpips", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--blender", default="blender")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if int(args.min_tile_tokens) < 1:
        raise ValueError("--min-tile-tokens must be positive")
    if int(args.max_num_tokens) < int(args.min_tile_tokens):
        raise ValueError("--max-num-tokens must be >= --min-tile-tokens")
    if int(args.max_global_geometry_points) < 1:
        raise ValueError("--max-global-geometry-points must be positive")
    if args.max_tiles is not None and int(args.max_tiles) < 1:
        raise ValueError("--max-tiles must be positive")
    if int(args.render_resolution) < 1 or int(args.metric_resolution) < 1:
        raise ValueError("render and metric resolutions must be positive")
    run(args)


if __name__ == "__main__":
    main()