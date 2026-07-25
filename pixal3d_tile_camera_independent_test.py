#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent Pixal3D tile-camera diagnostic.

Purpose
-------
This script isolates the global-camera -> tile-camera canonical-space mapping.
It deliberately does NOT perform global/tile velocity fusion and does NOT run a
separate tile 512 support-regeneration stage.

Pipeline
--------
1. Canonically preprocess one input image to 4096/1024/512.
2. Estimate the global Pixal3D camera once from the complete image.
3. Run the complete-image sparse-structure and 512 shape stages, then learned
   upsample to obtain a complete-object global C128 support.
4. For every 4096->1024 tile:
   a. Select every global C128 point whose complete-image projection is inside
      that crop. No visibility/depth filtering is used.
   b. Projectively recanonicalize those points into the centered tile camera,
      preserving normalized z.
   c. Quantize directly to tile C64. This fixed support is the diagnostic input.
   d. Skip the tile when the unique C64 support has fewer than --min-tile-tokens.
   e. Independently run all 12 shape-flow steps and all 12 texture-flow steps.
   f. Decode at 1024, save a raw postprocess cache, render with the tile camera,
      compare against the 1024 crop, and compute PSNR/SSIM/LPIPS.
5. Optionally run an independent global C128/2048 baseline.
6. Save coordinate overlays, quantization-error diagnostics, render comparison,
   overlay, difference heatmap, per-tile summaries, aggregate metrics, and
   contact sheets.

The test answers a narrow question: if complete-object global coordinates are
re-expressed in the tile camera's own [-1,1]^3 and the tile model is allowed to
run independently for 12+12 steps, does the camera-facing side reconstruct the
high-resolution crop well?
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
import warnings
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


GRID_GLOBAL_LR = 32
GRID_TILE = 64
GRID_GLOBAL = 128
IMAGE_LR = 512
IMAGE_TILE = 1024
IMAGE_GLOBAL_FLOW = 1024
IMAGE_CANONICAL = 4096
DECODE_TILE = 1024
DECODE_GLOBAL = 2048
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
    return torch.round((q + 1.0) * (float(resolution - 1) / 2.0)).to(torch.int32)


def _quantize_decoder_candidates(
    candidates: torch.Tensor,
    *,
    target_grid: int,
    source_resolution: int = IMAGE_LR,
) -> torch.Tensor:
    if candidates.ndim != 2 or candidates.shape[1] != 4:
        raise ValueError(f"decoder candidates must be [N,4], got {tuple(candidates.shape)}")
    xyz = torch.round(
        (candidates[:, 1:].to(torch.float32) + 0.5)
        / float(source_resolution)
        * float(target_grid - 1)
    ).to(torch.int32)
    quantized = torch.cat([candidates[:, :1].to(torch.int32), xyz], dim=1)
    valid = ((quantized[:, 1:] >= 0) & (quantized[:, 1:] < target_grid)).all(dim=1)
    quantized = torch.unique(quantized[valid], dim=0)
    if quantized.numel() == 0:
        raise RuntimeError(f"decoder upsample produced no C{target_grid} coordinates")
    return quantized


def _learned_upsample(
    pipeline: Any,
    shape_denormalized: SparseTensor,
    *,
    target_grid: int,
) -> torch.Tensor:
    decoder = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        decoder.to(pipeline.device)
        decoder.low_vram = True
    try:
        candidates = decoder.upsample(shape_denormalized, upsample_times=4)
    finally:
        if pipeline.low_vram:
            decoder.cpu()
            decoder.low_vram = False
            _empty_cuda_cache()
    return _quantize_decoder_candidates(candidates, target_grid=target_grid)


def _focal_pixels(camera_angle_x: float, resolution: int) -> float:
    return float(resolution) / (2.0 * math.tan(float(camera_angle_x) / 2.0))


def _global_coords_to_camera(
    coords: torch.Tensor,
    *,
    grid_resolution: int,
    camera: Mapping[str, float],
) -> torch.Tensor:
    q = _endpoint_indices_to_q(coords[:, 1:4], grid_resolution).to(coords.device)
    center = torch.tensor(
        [0.0, 0.0, -float(camera["distance"])],
        device=coords.device,
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
    focal = float(resolution) / (2.0 * math.tan(float(camera_angle_x) / 2.0))
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
    extend_pixel: int,
) -> TileCameraTransform:
    x0, y0, x1, y1 = (int(v) for v in box)
    crop_w = x1 - x0
    crop_h = y1 - y0
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"invalid tile box {tuple(box)}")
    scale_x = float(output_size) / float(crop_w)
    scale_y = float(output_size) / float(crop_h)
    full_focal = _focal_pixels(float(global_camera["camera_angle_x"]), canonical_size)
    tile_fx = full_focal * scale_x
    tile_fy = full_focal * scale_y
    if not math.isclose(tile_fx, tile_fy, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"tile focal mismatch fx={tile_fx}, fy={tile_fy}")
    tile_fov = 2.0 * math.atan(float(output_size) / (2.0 * tile_fx))
    tile_mesh_scale = float(global_camera["mesh_scale"])
    tile_distance = distance_from_fov(
        tile_fov,
        torch.tensor([-1.0, 0.0, 0.0]),
        torch.tensor([0 - int(extend_pixel), output_size - 1 + int(extend_pixel)]),
        tile_mesh_scale,
        output_size,
    )["distance_from_x"]
    return TileCameraTransform(
        tile_id=int(tile_id),
        box=(x0, y0, x1, y1),
        output_size=int(output_size),
        camera_angle_x=float(tile_fov),
        distance=float(tile_distance),
        mesh_scale=float(tile_mesh_scale),
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
    clamp: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if q_global.ndim != 2 or q_global.shape[1] != 3:
        raise ValueError("q_global must be [N,3]")
    center = torch.tensor(
        [0.0, 0.0, -float(global_camera["distance"])],
        device=q_global.device,
        dtype=q_global.dtype,
    )
    global_points = center[None] + q_global / (2.0 * float(global_camera["mesh_scale"]))
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
    tile_depth = float(transform.distance) - qz / (2.0 * float(transform.mesh_scale))
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
    rows_overflow = (overflow > 0).any(dim=1)
    stats = {
        "rows": int(q_raw.shape[0]),
        "clamped_rows": int(rows_overflow.sum().item()),
        "clamped_fraction": float(rows_overflow.float().mean().item()) if q_raw.shape[0] else 0.0,
        "max_overflow": float(overflow.max().item()) if overflow.numel() else 0.0,
        "q_raw_min": [float(v) for v in q_raw.amin(dim=0).detach().cpu().tolist()],
        "q_raw_max": [float(v) for v in q_raw.amax(dim=0).detach().cpu().tolist()],
    }
    return (q_raw.clamp(-1.0, 1.0) if clamp else q_raw), uv_tile, stats


def _tile_q_to_global_q(
    q_tile: torch.Tensor,
    *,
    transform: TileCameraTransform,
    global_camera: Mapping[str, float],
    clamp: bool,
) -> torch.Tensor:
    st = float(transform.mesh_scale)
    qz = q_tile[:, 2]
    tile_points = torch.stack(
        [
            q_tile[:, 0] / (2.0 * st),
            q_tile[:, 1] / (2.0 * st),
            qz / (2.0 * st) - float(transform.distance),
        ],
        dim=1,
    )
    uv_tile, _, _ = _project_camera_points(
        tile_points,
        float(transform.camera_angle_x),
        int(transform.output_size),
    )
    uv_tile = uv_tile[0]
    x0, y0, _, _ = transform.box
    u_full = uv_tile[:, 0] / float(transform.crop_scale_x) + float(x0)
    v_full = uv_tile[:, 1] / float(transform.crop_scale_y) + float(y0)
    sg = float(global_camera["mesh_scale"])
    depth_global = float(global_camera["distance"]) - qz / (2.0 * sg)
    xg = (
        (u_full - IMAGE_CANONICAL / 2.0)
        * depth_global
        / float(transform.full_focal_pixels)
    )
    yg = -(
        (v_full - IMAGE_CANONICAL / 2.0)
        * depth_global
        / float(transform.full_focal_pixels)
    )
    q_global = torch.stack([2.0 * sg * xg, 2.0 * sg * yg, qz], dim=1)
    return q_global.clamp(-1.0, 1.0) if clamp else q_global


def _tile_coords_to_camera(
    coords: torch.Tensor,
    *,
    grid_resolution: int,
    transform: TileCameraTransform,
) -> torch.Tensor:
    q = _endpoint_indices_to_q(coords[:, 1:4], grid_resolution).to(coords.device)
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
    max_points: int = 12000,
) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    uv_cpu = uv.detach().cpu().float().numpy()
    qz_cpu = qz.detach().cpu().float().numpy()
    if uv_cpu.shape[0] > max_points:
        ids = np.linspace(0, uv_cpu.shape[0] - 1, max_points).round().astype(np.int64)
        uv_cpu = uv_cpu[ids]
        qz_cpu = qz_cpu[ids]
    colors = (_depth_color((qz_cpu + 1.0) * 0.5) * 255.0).astype(np.uint8)
    for (u, v), color in zip(uv_cpu, colors):
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        x, y = int(round(float(u))), int(round(float(v)))
        if 0 <= x < canvas.width and 0 <= y < canvas.height:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=tuple(color.tolist()) + (190,))
    draw.rectangle((0, 0, canvas.width, 30), fill=(0, 0, 0, 190))
    draw.text((8, 8), f"{title} | points={uv_cpu.shape[0]:,}", fill=(255, 255, 255, 255))
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
    valid = np.isfinite(array).all(axis=1)
    array = array[valid]
    hist, _, _ = np.histogram2d(
        array[:, 1] if len(array) else np.empty(0),
        array[:, 0] if len(array) else np.empty(0),
        bins=bins,
        range=[[0, resolution], [0, resolution]],
    )
    hist = np.log1p(hist)
    if hist.max() > 0:
        hist /= hist.max()
    rgb = (_depth_color(hist.reshape(-1)).reshape(bins, bins, 3) * 255.0).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB").resize((resolution, resolution), Image.Resampling.NEAREST)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _save_quantization_error_image(
    reference: Image.Image,
    uv_source: torch.Tensor,
    uv_quantized: torch.Tensor,
    output: Path,
    max_lines: int = 800,
) -> None:
    canvas = reference.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    source = uv_source.detach().cpu().numpy()
    target = uv_quantized.detach().cpu().numpy()
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
    draw.rectangle((0, 0, canvas.width, 38), fill=(0, 0, 0, 200))
    draw.text((8, 7), "red=exact crop ray, cyan=quantized tile C64 projection", fill=(255, 255, 255, 255))
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
    heat = (_depth_color(diff.reshape(-1)).reshape(diff.shape[0], diff.shape[1], 3) * 255.0).astype(np.uint8)
    diff_image = Image.fromarray(heat, mode="RGB")

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "overlay_50.png"
    diff_path = output_dir / "abs_diff_heatmap.png"
    triptych_path = output_dir / "triptych_reference_render_diff.png"
    overlay.save(overlay_path)
    diff_image.save(diff_path)

    w, h = reference.size
    triptych = Image.new("RGB", (w * 3, h + 34), (18, 18, 18))
    triptych.paste(reference, (0, 34))
    triptych.paste(rendered, (w, 34))
    triptych.paste(diff_image, (w * 2, 34))
    draw = ImageDraw.Draw(triptych)
    draw.text((8, 10), "reference", fill=(255, 255, 255))
    draw.text((w + 8, 10), "render", fill=(255, 255, 255))
    draw.text((w * 2 + 8, 10), "absolute RGB error", fill=(255, 255, 255))
    triptych.save(triptych_path)
    return {
        "overlay_png": str(overlay_path),
        "diff_heatmap_png": str(diff_path),
        "triptych_png": str(triptych_path),
    }



def _simple_psnr_ssim(reference: Image.Image, prediction: Image.Image) -> Tuple[float, float]:
    import torch.nn.functional as F

    ref = np.asarray(reference.convert("RGB"), dtype=np.float32) / 255.0
    pred = np.asarray(prediction.convert("RGB").resize(reference.size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    x = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0)
    y = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0)
    mse = float(F.mse_loss(y, x).item())
    psnr = float("inf") if mse <= 0.0 else float(10.0 * math.log10(1.0 / mse))
    window_size, sigma = 11, 1.5
    coordinates = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    kernel1 = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel1 = kernel1 / kernel1.sum()
    kernel2 = (kernel1[:, None] * kernel1[None, :])[None, None].expand(3, 1, window_size, window_size)
    padding = window_size // 2
    mu_x = F.conv2d(x, kernel2, padding=padding, groups=3)
    mu_y = F.conv2d(y, kernel2, padding=padding, groups=3)
    mu_x_sq, mu_y_sq, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sigma_x_sq = F.conv2d(x * x, kernel2, padding=padding, groups=3) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel2, padding=padding, groups=3) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel2, padding=padding, groups=3) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    value = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2) + 1e-12
    )
    return psnr, float(value.mean().item())


def _save_global_crop_comparison(
    *,
    global_render_path: Path,
    box_4096: Sequence[int],
    reference_path: Path,
    tile_render_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    full_render = _composite_on_black(Image.open(global_render_path))
    reference = _composite_on_black(Image.open(reference_path))
    tile_render = _composite_on_black(Image.open(tile_render_path)).resize(reference.size, Image.Resampling.LANCZOS)
    x0, y0, x1, y1 = (float(v) for v in box_4096)
    sx = full_render.width / float(IMAGE_CANONICAL)
    sy = full_render.height / float(IMAGE_CANONICAL)
    crop_box = (
        int(round(x0 * sx)),
        int(round(y0 * sy)),
        int(round(x1 * sx)),
        int(round(y1 * sy)),
    )
    global_crop = full_render.crop(crop_box).resize(reference.size, Image.Resampling.LANCZOS)
    output_dir.mkdir(parents=True, exist_ok=True)
    global_crop_path = output_dir / "global_baseline_crop.png"
    global_crop.save(global_crop_path)
    global_psnr, global_ssim = _simple_psnr_ssim(reference, global_crop)

    ref_arr = np.asarray(reference, dtype=np.float32)
    tile_arr = np.asarray(tile_render, dtype=np.float32)
    tile_diff = np.abs(ref_arr - tile_arr).mean(axis=2) / 255.0
    tile_heat = Image.fromarray(
        (_depth_color(tile_diff.reshape(-1)).reshape(tile_diff.shape[0], tile_diff.shape[1], 3) * 255).astype(np.uint8),
        mode="RGB",
    )
    w, h = reference.size
    fourway = Image.new("RGB", (w * 4, h + 34), (18, 18, 18))
    panels = [reference, global_crop, tile_render, tile_heat]
    labels = ["reference crop", "global baseline crop", "independent tile render", "tile absolute error"]
    draw = ImageDraw.Draw(fourway)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        fourway.paste(panel, (index * w, 34))
        draw.text((index * w + 8, 10), label, fill=(255, 255, 255))
    fourway_path = output_dir / "fourway_reference_global_tile_diff.png"
    fourway.save(fourway_path)
    return {
        "global_baseline_crop_png": str(global_crop_path),
        "global_baseline_crop_psnr_db": global_psnr,
        "global_baseline_crop_ssim": global_ssim,
        "fourway_png": str(fourway_path),
    }

def _save_diagnostic_sheet(tile_dir: Path) -> Optional[Path]:
    candidates = [
        tile_dir / "reference_tile.png",
        tile_dir / "coords_global_selected.png",
        tile_dir / "coords_tile_c64.png",
        tile_dir / "coord_quantization_error.png",
        tile_dir / "aligned_eval" / "studio",
        tile_dir / "comparisons" / "abs_diff_heatmap.png",
    ]
    images: List[Tuple[str, Image.Image]] = []
    labels = ["reference", "global selected", "tile C64", "quantization error"]
    for label, path in zip(labels, candidates[:4]):
        if path.is_file():
            images.append((label, Image.open(path).convert("RGB")))
    studio_dir = candidates[4]
    if studio_dir.is_dir():
        render_paths = sorted(studio_dir.glob("*__render.png"))
        if render_paths:
            images.append(("independent render", Image.open(render_paths[0]).convert("RGB")))
    if candidates[5].is_file():
        images.append(("absolute difference", Image.open(candidates[5]).convert("RGB")))
    if not images:
        return None

    cell = 384
    cols = 3
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 30)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(images):
        image.thumbnail((cell, cell), Image.Resampling.LANCZOS)
        x = (index % cols) * cell + (cell - image.width) // 2
        y0 = (index // cols) * (cell + 30)
        y = y0 + 30 + (cell - image.height) // 2
        draw.text((index % cols * cell + 8, y0 + 8), label, fill=(255, 255, 255))
        sheet.paste(image, (x, y))
    output = tile_dir / "diagnostic_sheet.png"
    sheet.save(output)
    return output


def _make_contact_sheets(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> List[str]:
    successful = [row for row in rows if row.get("status") == "success" and row.get("triptych_png")]
    paths: List[str] = []
    per_page = 9
    for page_index in range(0, len(successful), per_page):
        page_rows = successful[page_index : page_index + per_page]
        cell_w, cell_h = 768, 300
        cols = 3
        rows_count = math.ceil(len(page_rows) / cols)
        canvas = Image.new("RGB", (cols * cell_w, rows_count * (cell_h + 40)), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(page_rows):
            image = Image.open(str(row["triptych_png"])).convert("RGB")
            image.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            col = index % cols
            rr = index // cols
            x0 = col * cell_w
            y0 = rr * (cell_h + 40)
            x = x0 + (cell_w - image.width) // 2
            y = y0 + 40 + (cell_h - image.height) // 2
            title = (
                f"tile {int(row['tile_id']):02d} | K={int(row['tile_tokens'])} | "
                f"PSNR={row.get('psnr_db')} SSIM={row.get('ssim')} LPIPS={row.get('lpips')}"
            )
            draw.text((x0 + 8, y0 + 10), title, fill=(255, 255, 255))
            canvas.paste(image, (x, y))
        output = output_dir / f"all_tiles_contact_sheet_{page_index // per_page:02d}.png"
        canvas.save(output)
        paths.append(str(output))
    return paths


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
    temporary = output_dir / f"_tile_camera_test_moge_{time.time_ns()}.png"
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


def _sampler_params(args: argparse.Namespace, pipeline: Any) -> Dict[str, Dict[str, Any]]:
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


def _generate_global_support(
    *,
    pipeline: Any,
    image_512: Image.Image,
    camera: Mapping[str, float],
    params: Mapping[str, Mapping[str, Any]],
    seed: int,
    max_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, SparseTensor]:
    cond_ss = pipeline.get_proj_cond_ss(
        [image_512],
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
    )
    _seed_everything(seed)
    coords32 = pipeline.sample_sparse_structure(
        cond_ss,
        resolution=GRID_GLOBAL_LR,
        sampler_params=dict(params["ss"]),
    )
    del cond_ss
    if coords32.shape[0] == 0:
        raise RuntimeError("global sparse structure is empty")
    print(f"[global-support] C32={coords32.shape[0]:,}")

    cond_shape = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_512,
        [image_512],
        coords32,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
        grid_resolution_override=GRID_GLOBAL_LR,
    )
    model = pipeline.models["shape_slat_flow_model_512"]
    noise = SparseTensor(
        feats=_randn(
            coords32.shape[0],
            int(model.in_channels),
            device=pipeline.device,
            seed=seed + 101,
        ),
        coords=coords32,
    )
    shape512_norm, _ = _run_sampler_full(
        pipeline=pipeline,
        sampler=pipeline.shape_slat_sampler,
        model=model,
        noise=noise,
        condition=cond_shape,
        params=params["shape"],
        description="Global shape SLat 512 for support",
    )
    shape512_denorm = _denormalize_sparse(shape512_norm, pipeline.shape_slat_normalization)
    coords128 = _learned_upsample(pipeline, shape512_denorm, target_grid=GRID_GLOBAL)
    if coords128.shape[0] > max_tokens:
        raise RuntimeError(
            f"global C128 support has {coords128.shape[0]:,} tokens; "
            f"exceeds --max-num-tokens={max_tokens:,}"
        )
    print(f"[global-support] C128={coords128.shape[0]:,}")
    return coords32, coords128, shape512_norm


def _independent_shape_texture(
    *,
    pipeline: Any,
    image: Image.Image,
    coords: torch.Tensor,
    camera: Mapping[str, float],
    grid_resolution: int,
    params: Mapping[str, Mapping[str, Any]],
    seed: int,
    label: str,
) -> ModelResult:
    shape_condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [image],
        coords,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
        grid_resolution_override=int(grid_resolution),
    )
    shape_model = pipeline.models["shape_slat_flow_model_1024"]
    shape_noise = SparseTensor(
        feats=_randn(
            coords.shape[0],
            int(shape_model.in_channels),
            device=pipeline.device,
            seed=seed + 201,
        ),
        coords=coords,
    )
    shape_norm, shape_seconds = _run_sampler_full(
        pipeline=pipeline,
        sampler=pipeline.shape_slat_sampler,
        model=shape_model,
        noise=shape_noise,
        condition=shape_condition,
        params=params["shape"],
        description=f"{label} independent shape 1024",
    )
    shape_denorm = _denormalize_sparse(shape_norm, pipeline.shape_slat_normalization)
    del shape_condition, shape_noise
    _empty_cuda_cache()

    texture_condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [image],
        coords,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
        grid_resolution_override=int(grid_resolution),
    )
    texture_model = pipeline.models["tex_slat_flow_model_1024"]
    texture_channels = int(texture_model.in_channels) - int(shape_norm.feats.shape[1])
    if texture_channels <= 0:
        raise RuntimeError(f"invalid texture noise channel count {texture_channels}")
    texture_noise = SparseTensor(
        feats=_randn(
            coords.shape[0],
            texture_channels,
            device=pipeline.device,
            seed=seed + 301,
        ),
        coords=coords,
    )
    texture_norm, texture_seconds = _run_sampler_full(
        pipeline=pipeline,
        sampler=pipeline.tex_slat_sampler,
        model=texture_model,
        noise=texture_noise,
        condition=texture_condition,
        params=params["texture"],
        description=f"{label} independent texture 1024",
        concat_cond=shape_norm,
    )
    texture_denorm = _denormalize_sparse(texture_norm, pipeline.tex_slat_normalization)
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
    decode_resolution: int,
    grid_resolution: int,
    output_dir: Path,
    camera: Mapping[str, float],
    seed: int,
    label: str,
    export_glb: bool,
    texture_size: int,
    decimation_target: int,
) -> Dict[str, Any]:
    meshes = pipeline.decode_latent(
        result.shape_denorm,
        result.texture_denorm,
        int(decode_resolution),
    )
    mesh = meshes[0]
    vertices = int(mesh.vertices.shape[0])
    faces = int(mesh.faces.shape[0])
    print(f"[decode] {label}: vertices={vertices:,} faces={faces:,}")
    effective_target = faces if decimation_target <= 0 else min(faces, int(decimation_target))
    export_kwargs = {
        "vertices": mesh.vertices,
        "faces": mesh.faces,
        "attr_volume": mesh.attrs,
        "coords": mesh.coords,
        "attr_layout": pipeline.pbr_attr_layout,
        "grid_size": int(decode_resolution),
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
            "pipeline_resolution": int(decode_resolution),
            "actual_grid_resolution": int(grid_resolution),
            "seed": int(seed),
            "decoder_vertices": vertices,
            "decoder_faces": faces,
            "experiment": "independent_tile_camera_diagnostic",
            "label": label,
        },
        overwrite=True,
    )
    manifest_path = cache_dir / "manifest.json"
    published = json.loads(manifest_path.read_text(encoding="utf-8"))
    published["grid_size"] = int(decode_resolution)
    published["aabb"] = export_kwargs["aabb"]
    _atomic_json(manifest_path, published)

    glb_path: Optional[Path] = None
    if export_glb:
        if o_voxel is None:
            raise RuntimeError("o_voxel is unavailable; disable --export-glb")
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
        "--cache-dir", str(cache_dir),
        "--output-dir", str(output_dir),
        "--reference-image", str(reference_image),
        "--lights", str(args.light),
        "--engine", str(args.render_engine),
        "--material-mode", "pbr",
        "--render-resolution", str(int(args.render_resolution)),
        "--metric-resolution", str(int(args.metric_resolution)),
        "--samples", str(int(args.blender_samples)),
        "--lpips-net", str(args.lpips_net),
        "--metric-device", str(args.metric_device),
        "--blender", str(args.blender),
        "--overwrite-renders",
        "--overwrite-package",
    ]
    if args.render_max_faces > 0:
        command.extend(["--max-faces", str(int(args.render_max_faces))])
    if args.skip_lpips:
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
    success_rows = [row for row in rows if isinstance(row, dict) and row.get("status") == "success"]
    if not success_rows:
        raise RuntimeError(f"no successful metrics row in {metrics_path}")
    row = success_rows[0]
    return {"metrics_payload": payload, "metrics_row": row, "metrics_json": str(metrics_path)}


def _parse_tile_ids(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not value.strip():
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _prepare_tile_support_and_diagnostics(
    *,
    tile_id: int,
    box: Tuple[int, int, int, int],
    image_4096: Image.Image,
    base_coords128: torch.Tensor,
    base_uv_full: torch.Tensor,
    rows: torch.Tensor,
    global_camera: Mapping[str, float],
    transform: TileCameraTransform,
    tile_dir: Path,
) -> Tuple[torch.Tensor, Dict[str, Any], Image.Image]:
    tile_rgba = image_4096.crop(box).convert("RGBA")
    if tile_rgba.size != (IMAGE_TILE, IMAGE_TILE):
        tile_rgba = tile_rgba.resize((IMAGE_TILE, IMAGE_TILE), Image.Resampling.LANCZOS)
    reference = _composite_on_black(tile_rgba)
    reference_path = tile_dir / "reference_tile.png"
    tile_dir.mkdir(parents=True, exist_ok=True)
    reference.save(reference_path)

    selected = base_coords128.index_select(0, rows)
    q_global = _endpoint_indices_to_q(selected[:, 1:4], GRID_GLOBAL).to(selected.device)
    q_tile, uv_source_tile, transform_stats = _global_q_to_tile_q(
        q_global,
        global_camera=global_camera,
        transform=transform,
        clamp=True,
    )
    ids64_per_source = _q_to_endpoint_indices(q_tile, GRID_TILE).clamp(0, GRID_TILE - 1)
    coords64_per_source = torch.cat(
        [
            torch.zeros((ids64_per_source.shape[0], 1), device=ids64_per_source.device, dtype=torch.int32),
            ids64_per_source,
        ],
        dim=1,
    )
    coords64 = torch.unique(coords64_per_source, dim=0)

    q_tile_quant_per_source = _endpoint_indices_to_q(ids64_per_source, GRID_TILE).to(q_tile.device)
    tile_points_quant = torch.stack(
        [
            q_tile_quant_per_source[:, 0] / (2.0 * float(transform.mesh_scale)),
            q_tile_quant_per_source[:, 1] / (2.0 * float(transform.mesh_scale)),
            q_tile_quant_per_source[:, 2] / (2.0 * float(transform.mesh_scale)) - float(transform.distance),
        ],
        dim=1,
    )
    uv_quantized, _, _ = _project_camera_points(
        tile_points_quant,
        float(transform.camera_angle_x),
        IMAGE_TILE,
    )
    uv_quantized = uv_quantized[0]
    pixel_error = torch.linalg.vector_norm(uv_quantized - uv_source_tile, dim=1)
    q_global_roundtrip = _tile_q_to_global_q(
        q_tile_quant_per_source,
        transform=transform,
        global_camera=global_camera,
        clamp=True,
    )
    q_error = torch.linalg.vector_norm(q_global_roundtrip - q_global, dim=1)

    unique_points = _tile_coords_to_camera(
        coords64,
        grid_resolution=GRID_TILE,
        transform=transform,
    )
    unique_uv, _, _ = _project_camera_points(
        unique_points,
        float(transform.camera_angle_x),
        IMAGE_TILE,
    )
    unique_uv = unique_uv[0]
    unique_qz = _endpoint_indices_to_q(coords64[:, 3:4], GRID_TILE)[:, 0]

    _draw_uv_points(
        reference,
        uv_source_tile,
        q_global[:, 2],
        tile_dir / "coords_global_selected.png",
        "global C128 rows projected into crop",
    )
    _draw_uv_points(
        reference,
        unique_uv,
        unique_qz,
        tile_dir / "coords_tile_c64.png",
        "recanonicalized unique tile C64 support",
    )
    _save_density_image(
        unique_uv,
        tile_dir / "coords_tile_c64_density.png",
        resolution=IMAGE_TILE,
    )
    _save_quantization_error_image(
        reference,
        uv_source_tile,
        uv_quantized,
        tile_dir / "coord_quantization_error.png",
    )

    stats = {
        "tile_id": int(tile_id),
        "box": list(box),
        "selected_global_rows": int(rows.numel()),
        "tile_c64_rows_before_unique": int(coords64_per_source.shape[0]),
        "tile_c64_unique_tokens": int(coords64.shape[0]),
        "quantization_merge_rows": int(coords64_per_source.shape[0] - coords64.shape[0]),
        "transform": asdict(transform),
        "transform_stats": transform_stats,
        "pixel_error_mean": float(pixel_error.mean().item()),
        "pixel_error_p95": float(torch.quantile(pixel_error, 0.95).item()),
        "pixel_error_max": float(pixel_error.max().item()),
        "global_q_roundtrip_error_mean": float(q_error.mean().item()),
        "global_q_roundtrip_error_p95": float(torch.quantile(q_error, 0.95).item()),
        "global_q_roundtrip_error_max": float(q_error.max().item()),
        "reference_image": str(reference_path),
    }
    torch.save(
        {
            "base_rows": rows.detach().cpu(),
            "base_coords128": selected.detach().cpu(),
            "q_global": q_global.detach().cpu(),
            "q_tile": q_tile.detach().cpu(),
            "coords64_per_source": coords64_per_source.detach().cpu(),
            "coords64_unique": coords64.detach().cpu(),
            "uv_source_tile": uv_source_tile.detach().cpu(),
            "uv_quantized": uv_quantized.detach().cpu(),
            "pixel_error": pixel_error.detach().cpu(),
            "q_global_roundtrip": q_global_roundtrip.detach().cpu(),
            "q_roundtrip_error": q_error.detach().cpu(),
        },
        tile_dir / "coordinate_diagnostic.pt",
    )
    _atomic_json(tile_dir / "coordinate_stats.json", stats)
    return coords64, stats, reference


def _run_global_baseline(
    *,
    pipeline: Any,
    image_1024: Image.Image,
    coords128: torch.Tensor,
    camera: Mapping[str, float],
    params: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
    repository_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = _composite_on_black(image_1024)
    reference_path = output_dir / "reference_global.png"
    reference.save(reference_path)
    result = _independent_shape_texture(
        pipeline=pipeline,
        image=image_1024.convert("RGB"),
        coords=coords128,
        camera=camera,
        grid_resolution=GRID_GLOBAL,
        params=params,
        seed=int(args.seed) + 50000,
        label="Global C128",
    )
    cache = _save_postprocess_cache(
        pipeline=pipeline,
        result=result,
        decode_resolution=DECODE_GLOBAL,
        grid_resolution=GRID_GLOBAL,
        output_dir=output_dir,
        camera=camera,
        seed=int(args.seed) + 50000,
        label="Global C128",
        export_glb=bool(args.export_glb),
        texture_size=int(args.texture_size),
        decimation_target=int(args.decimation_target),
    )
    eval_result = _run_evaluator(
        repository_dir=repository_dir,
        cache_dir=Path(cache["cache_dir"]),
        output_dir=output_dir / "aligned_eval",
        reference_image=reference_path,
        args=args,
    )
    row = dict(eval_result["metrics_row"])
    extras = _save_extra_comparisons(
        Path(row["original_png"]),
        Path(row["render_png"]),
        output_dir / "comparisons",
    )
    summary = {
        "status": "success",
        "kind": "global_baseline",
        "tokens": int(coords128.shape[0]),
        "camera": dict(camera),
        "shape_seconds": result.shape_seconds,
        "texture_seconds": result.texture_seconds,
        **cache,
        **row,
        **extras,
    }
    _atomic_json(output_dir / "summary.json", summary)
    del result
    _empty_cuda_cache()
    return summary


def run(args: argparse.Namespace) -> None:
    if args.tile_size != 1024 or args.tile_stride != 512:
        raise ValueError("this diagnostic currently requires tile-size=1024 and tile-stride=512")
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

    camera = _estimate_camera(
        image_1024=image_1024,
        output_dir=output_dir,
        manual_fov=float(args.fov),
        mesh_scale=float(args.mesh_scale),
        extend_pixel=int(args.extend_pixel),
        image_resolution=int(args.camera_image_resolution),
    )
    print(
        f"[global-camera] fov={camera['camera_angle_x']:.8f} "
        f"distance={camera['distance']:.8f} mesh_scale={camera['mesh_scale']:.8f}"
    )
    params = _sampler_params(args, pipeline)
    coords32, coords128, shape512_norm = _generate_global_support(
        pipeline=pipeline,
        image_512=image_512,
        camera=camera,
        params=params,
        seed=int(args.seed),
        max_tokens=int(args.max_num_tokens),
    )
    torch.save(
        {
            "camera": camera,
            "coords32": coords32.detach().cpu(),
            "coords128": coords128.detach().cpu(),
            "shape512_final_normalized": shape512_norm.feats.detach().cpu(),
        },
        output_dir / "global_support.pt",
    )
    del shape512_norm
    _empty_cuda_cache()

    camera_points = _global_coords_to_camera(
        coords128,
        grid_resolution=GRID_GLOBAL,
        camera=camera,
    )
    uv_full, _, valid_full = _project_camera_points(
        camera_points,
        float(camera["camera_angle_x"]),
        IMAGE_CANONICAL,
    )
    uv_full = uv_full[0]
    valid_full = valid_full[0]
    q_global_all = _endpoint_indices_to_q(coords128[:, 1:4], GRID_GLOBAL)
    _draw_uv_points(
        _composite_on_black(image_4096),
        uv_full[valid_full],
        q_global_all[valid_full, 2],
        output_dir / "global_c128_projection_4096.png",
        "complete-object global C128 support",
    )
    _save_density_image(
        uv_full[valid_full],
        output_dir / "global_c128_projection_density.png",
        resolution=IMAGE_CANONICAL,
    )

    aggregate: List[Dict[str, Any]] = []
    global_summary: Optional[Dict[str, Any]] = None
    if args.run_global_baseline:
        global_summary = _run_global_baseline(
            pipeline=pipeline,
            image_1024=image_1024,
            coords128=coords128,
            camera=camera,
            params=params,
            output_dir=output_dir / "global_baseline",
            args=args,
            repository_dir=repository_dir,
        )

    boxes = _tile_layout(IMAGE_CANONICAL, int(args.tile_size), int(args.tile_stride))
    selected_ids = _parse_tile_ids(args.tile_ids)
    completed = 0
    for tile_id, box in enumerate(boxes):
        if selected_ids is not None and tile_id not in selected_ids:
            continue
        if args.max_tiles is not None and completed >= int(args.max_tiles):
            break
        rows = _rows_inside_tile(uv_full, valid_full, box)
        tile_dir = output_dir / "tiles" / f"tile_{tile_id:02d}"
        transform = _build_tile_camera_transform(
            tile_id=tile_id,
            box=box,
            global_camera=camera,
            canonical_size=IMAGE_CANONICAL,
            output_size=IMAGE_TILE,
            extend_pixel=int(args.extend_pixel),
        )
        if rows.numel() == 0:
            record = {
                "status": "skipped",
                "tile_id": tile_id,
                "reason": "no projected global C128 rows",
                "selected_global_rows": 0,
                "tile_tokens": 0,
            }
            aggregate.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            continue
        print(
            f"[tile {tile_id:02d}] rows={rows.numel():,} box={box} "
            f"fov={transform.camera_angle_x:.8f} distance={transform.distance:.8f}"
        )
        try:
            coords64, coordinate_stats, tile_reference = _prepare_tile_support_and_diagnostics(
                tile_id=tile_id,
                box=box,
                image_4096=image_4096,
                base_coords128=coords128,
                base_uv_full=uv_full,
                rows=rows,
                global_camera=camera,
                transform=transform,
                tile_dir=tile_dir,
            )
            token_count = int(coords64.shape[0])
            if token_count < int(args.min_tile_tokens):
                record = {
                    "status": "skipped",
                    "tile_id": tile_id,
                    "reason": f"tile C64 tokens {token_count} < {args.min_tile_tokens}",
                    "selected_global_rows": int(rows.numel()),
                    "tile_tokens": token_count,
                    **coordinate_stats,
                }
                aggregate.append(record)
                _atomic_json(tile_dir / "summary.json", record)
                print(f"[tile-skip] tile={tile_id:02d} C64={token_count} < {args.min_tile_tokens}")
                continue

            tile_camera = {
                "camera_angle_x": float(transform.camera_angle_x),
                "distance": float(transform.distance),
                "mesh_scale": float(transform.mesh_scale),
            }
            model_result = _independent_shape_texture(
                pipeline=pipeline,
                image=tile_reference.convert("RGB"),
                coords=coords64,
                camera=tile_camera,
                grid_resolution=GRID_TILE,
                params=params,
                seed=int(args.seed) + tile_id * 1000,
                label=f"Tile {tile_id:02d}",
            )
            if args.save_latents:
                torch.save(
                    {
                        "coords64": coords64.detach().cpu(),
                        "shape_normalized": model_result.shape_norm.feats.detach().cpu(),
                        "texture_normalized": model_result.texture_norm.feats.detach().cpu(),
                    },
                    tile_dir / "final_latents.pt",
                )
            cache = _save_postprocess_cache(
                pipeline=pipeline,
                result=model_result,
                decode_resolution=DECODE_TILE,
                grid_resolution=GRID_TILE,
                output_dir=tile_dir,
                camera=tile_camera,
                seed=int(args.seed) + tile_id * 1000,
                label=f"Tile {tile_id:02d}",
                export_glb=bool(args.export_glb),
                texture_size=int(args.texture_size),
                decimation_target=int(args.decimation_target),
            )
            eval_result = _run_evaluator(
                repository_dir=repository_dir,
                cache_dir=Path(cache["cache_dir"]),
                output_dir=tile_dir / "aligned_eval",
                reference_image=tile_dir / "reference_tile.png",
                args=args,
            )
            metric_row = dict(eval_result["metrics_row"])
            extras = _save_extra_comparisons(
                Path(metric_row["original_png"]),
                Path(metric_row["render_png"]),
                tile_dir / "comparisons",
            )
            global_crop_extras: Dict[str, Any] = {}
            if global_summary is not None and global_summary.get("render_png"):
                global_crop_extras = _save_global_crop_comparison(
                    global_render_path=Path(str(global_summary["render_png"])),
                    box_4096=box,
                    reference_path=tile_dir / "reference_tile.png",
                    tile_render_path=Path(str(metric_row["render_png"])),
                    output_dir=tile_dir / "comparisons",
                )
            _save_diagnostic_sheet(tile_dir)
            record = {
                "status": "success",
                "tile_id": int(tile_id),
                "box": list(box),
                "selected_global_rows": int(rows.numel()),
                "tile_tokens": token_count,
                "shape_seconds": model_result.shape_seconds,
                "texture_seconds": model_result.texture_seconds,
                "camera_angle_x": tile_camera["camera_angle_x"],
                "distance": tile_camera["distance"],
                "mesh_scale": tile_camera["mesh_scale"],
                "pixel_error_mean": coordinate_stats["pixel_error_mean"],
                "pixel_error_p95": coordinate_stats["pixel_error_p95"],
                "pixel_error_max": coordinate_stats["pixel_error_max"],
                "q_roundtrip_error_mean": coordinate_stats["global_q_roundtrip_error_mean"],
                "q_roundtrip_error_max": coordinate_stats["global_q_roundtrip_error_max"],
                "clamped_rows": coordinate_stats["transform_stats"]["clamped_rows"],
                "clamped_fraction": coordinate_stats["transform_stats"]["clamped_fraction"],
                "psnr_db": metric_row.get("psnr_db"),
                "ssim": metric_row.get("ssim"),
                "lpips": metric_row.get("lpips"),
                "render_png": metric_row.get("render_png"),
                "comparison_png": metric_row.get("comparison_png"),
                **cache,
                **extras,
                **global_crop_extras,
            }
            _atomic_json(tile_dir / "summary.json", record)
            aggregate.append(record)
            completed += 1
            print(
                f"[tile-result] tile={tile_id:02d} tokens={token_count:,} "
                f"PSNR={record['psnr_db']} SSIM={record['ssim']} LPIPS={record['lpips']}"
            )
            del model_result
            _empty_cuda_cache()
        except Exception as exc:
            record = {
                "status": "failed",
                "tile_id": int(tile_id),
                "box": list(box),
                "selected_global_rows": int(rows.numel()),
                "error": f"{type(exc).__name__}: {exc}",
            }
            aggregate.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            print(f"[tile-error] tile={tile_id:02d}: {type(exc).__name__}: {exc}")
            _empty_cuda_cache()
            if args.strict_tiles:
                raise

    _write_csv(output_dir / "aggregate_metrics.csv", aggregate)
    contact_sheets = _make_contact_sheets(aggregate, output_dir)
    summary = {
        "format": "pixal3d_independent_tile_camera_diagnostic_v1",
        "image": str(Path(args.image).expanduser().resolve()),
        "global_camera": camera,
        "global_c32_tokens": int(coords32.shape[0]),
        "global_c128_tokens": int(coords128.shape[0]),
        "min_tile_tokens": int(args.min_tile_tokens),
        "successful_tiles": sum(row.get("status") == "success" for row in aggregate),
        "skipped_tiles": sum(row.get("status") == "skipped" for row in aggregate),
        "failed_tiles": sum(row.get("status") == "failed" for row in aggregate),
        "global_baseline": global_summary,
        "contact_sheets": contact_sheets,
        "tiles": aggregate,
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(f"[done] summary={output_dir / 'summary.json'}")
    print(f"[done] metrics={output_dir / 'aggregate_metrics.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--tile-stride", type=int, default=DEFAULT_TILE_STRIDE)
    parser.add_argument("--tile-ids", default=None, help="comma-separated tile ids; omitted means all 49")
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--min-tile-tokens", type=int, default=100)
    parser.add_argument("--max-num-tokens", type=int, default=160000)
    parser.add_argument("--strict-tiles", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-global-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-latents", action=argparse.BooleanOptionalAction, default=False)

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

    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-glb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--decimation-target", type=int, default=0)

    parser.add_argument("--light", default="studio")
    parser.add_argument("--render-engine", choices=("cycles", "eevee"), default="cycles")
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--metric-resolution", type=int, default=1024)
    parser.add_argument("--blender-samples", type=int, default=64)
    parser.add_argument("--render-max-faces", type=int, default=0)
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--metric-device", default="cuda")
    parser.add_argument("--skip-lpips", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--blender", default="blender")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.min_tile_tokens < 1:
        raise ValueError("--min-tile-tokens must be positive")
    run(args)


if __name__ == "__main__":
    main()
