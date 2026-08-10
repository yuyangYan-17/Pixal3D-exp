#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate one coherent Pixal3D 2048 model with 4096 image-tile residuals.

The geometry support follows the normal global cascade first:

    global SS C32 -> shape512 -> C64 -> shape1024
    -> shape-decoder subdivision -> dense global C1024
    -> quantized unique global C128 support

Shape and texture then each run one global C128 flow.  A tile never samples
support, noise, or its own trajectory.  At every active local-guidance step the
current global state is transported temporarily to the tile's unique local C64
support.  The tile model predicts a velocity from the corresponding 1024 crop;
the transported global velocity is subtracted, and the local-only residual is
transported back through the normalized global-C128/local-C64 correspondence.
Overlapping tile residuals are tent-weighted and the global state receives
exactly one Euler update:

    v = v_global + lambda(t) * transport(local_v - transport(v_global))

Shape uses a late, small residual schedule.  Texture starts earlier and uses a
stronger schedule.  The final global C128 shape and texture latents are decoded
once at resolution 2048, rendered globally with Pixal3D's official renderer,
and evaluated against the complete canonical image.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

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
from render_pixal3d_raw_ovoxel import (  # noqa: E402
    load_envmap,
    render_and_evaluate_mesh,
)


GRID_SS = 32
GRID_SHAPE_1024 = 64
GRID_GLOBAL_UPSAMPLED = 1024
GRID_FINAL_2048 = 128
IMAGE_LR = 512
GLOBAL_CAMERA_IMAGE_SIZE = 1024
IMAGE_CANONICAL = 4096
IMAGE_FLOW = 1024
DECODE_TILE = 1024
DECODE_GLOBAL = 2048
DEFAULT_TILE_SIZE = 1024
DEFAULT_TILE_STRIDE = 512


@dataclass(frozen=True)
class TileCameraTransform:
    tile_id: int
    box: Tuple[int, int, int, int]
    output_width: int
    output_height: int

    # Centered camera consumed by Pixal3D and used to render transformed local geometry.
    camera_angle_x: float
    camera_angle_y: float
    distance: float
    mesh_scale: float
    global_distance: float
    global_mesh_scale: float
    local_recanonicalization_scale: float
    fx: float
    fy: float
    cx: float
    cy: float

    # Exact off-axis crop camera for rendering untransformed global geometry.
    offaxis_cx: float
    offaxis_cy: float
    offaxis_shift_x: float
    offaxis_shift_y: float

    # Resolution/crop bookkeeping.
    global_fx_1024: float
    global_fy_1024: float
    full_fx_4096: float
    full_fy_4096: float
    global_to_full_scale_x: float
    global_to_full_scale_y: float
    crop_to_output_scale_x: float
    crop_to_output_scale_y: float
    tile_center_full_x: float
    tile_center_full_y: float


@dataclass
class ShapeResult:
    shape_norm: SparseTensor
    shape_denorm: SparseTensor
    shape512_seconds: float
    shape1024_seconds: float


@dataclass
class GlobalBaselineResult:
    shape_norm: SparseTensor
    shape_denorm: SparseTensor
    texture_norm: SparseTensor
    texture_denorm: SparseTensor
    global_c32_tokens: int
    global_c64_tokens: int
    shape512_seconds: float
    shape1024_seconds: float
    texture1024_seconds: float


@dataclass
class ModelResult:
    shape_norm: SparseTensor
    shape_denorm: SparseTensor
    texture_norm: SparseTensor
    texture_denorm: SparseTensor
    tile_projective_c32_tokens: int
    tile_ss_c32_tokens: int
    tile_c32_overlap_tokens: int
    tile_c32_tokens: int
    tile_projective_c64_tokens: int
    tile_native_c64_tokens: int
    tile_c64_overlap_tokens: int
    tile_c64_tokens: int
    tile_ss_seconds: float
    shape512_seconds: float
    shape1024_seconds: float
    texture1024_seconds: float


@dataclass
class TileTransport:
    """Sparse transport obtained by projecting global C128 into local C64."""

    tile_id: int
    box: Tuple[int, int, int, int]
    transform: TileCameraTransform
    local_coords: torch.Tensor
    edge_global: torch.Tensor
    edge_local: torch.Tensor
    edge_weight: torch.Tensor
    edge_forward_weight: torch.Tensor
    global_token_rows: torch.Tensor
    global_token_to_local: torch.Tensor
    stats: Dict[str, Any]
    condition_cpu: Optional[Dict[str, Dict[str, torch.Tensor]]] = None


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


def _focal_pixels(camera_angle: float, resolution: int) -> float:
    return float(resolution) / (2.0 * math.tan(float(camera_angle) / 2.0))


def _camera_q_to_points(
    q: torch.Tensor,
    *,
    distance: float,
    mesh_scale: float,
) -> torch.Tensor:
    """Pixal3D q coordinates -> camera-space points; camera looks along -Z."""
    center = torch.tensor(
        [0.0, 0.0, -float(distance)],
        device=q.device,
        dtype=q.dtype,
    )
    return center[None] + q / (2.0 * float(mesh_scale))


def _camera_points_to_q(
    points: torch.Tensor,
    *,
    distance: float,
    mesh_scale: float,
) -> torch.Tensor:
    center = torch.tensor(
        [0.0, 0.0, -float(distance)],
        device=points.device,
        dtype=points.dtype,
    )
    return (points - center[None]) * (2.0 * float(mesh_scale))


def _project_points_with_intrinsics(
    camera_points: torch.Tensor,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points = camera_points
    if points.ndim == 2:
        points = points.unsqueeze(0)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("camera_points must be [K,3] or [B,K,3]")

    depth = -points[..., 2]
    denom = depth.clamp_min(1e-8)
    u = float(fx) * points[..., 0] / denom + float(cx)
    v = -float(fy) * points[..., 1] / denom + float(cy)
    uv = torch.stack([u, v], dim=-1)
    finite = (depth > 0) & torch.isfinite(uv).all(dim=-1)
    return uv, depth, finite


def _backproject_pixels_with_depth(
    uv: torch.Tensor,
    depth: torch.Tensor,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> torch.Tensor:
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must be [N,2]")
    if depth.ndim != 1 or depth.shape[0] != uv.shape[0]:
        raise ValueError("depth must be [N] and match uv")
    x = (uv[:, 0] - float(cx)) * depth / float(fx)
    y = -(uv[:, 1] - float(cy)) * depth / float(fy)
    z = -depth
    return torch.stack([x, y, z], dim=1)


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


def _derive_tile_camera(
    *,
    tile_id: int,
    box: Sequence[int],
    global_camera: Mapping[str, float],
    global_image_width: int = GLOBAL_CAMERA_IMAGE_SIZE,
    global_image_height: int = GLOBAL_CAMERA_IMAGE_SIZE,
    full_width: int = IMAGE_CANONICAL,
    full_height: int = IMAGE_CANONICAL,
    output_width: int = IMAGE_FLOW,
    output_height: int = IMAGE_FLOW,
    extend_pixel: int = 0,
    offaxis_shift_y_sign: int = 1,
) -> TileCameraTransform:
    """Derive both exact off-axis crop intrinsics and centered local intrinsics.

    The global camera was estimated on the 1024 canonical image. Coordinates are
    first scaled continuously to the 4096 canonical grid. A 4096 crop is then
    mapped to the 1024 tile input.

    The centered local camera keeps the crop focal length, puts its principal
    point at the tile center, and recomputes the Pixal3D distance for a complete
    local normalized cube. Global/local conversion preserves normalized q_z.
    """
    if offaxis_shift_y_sign not in (-1, 1):
        raise ValueError("offaxis_shift_y_sign must be +1 or -1")

    x0, y0, x1, y1 = (int(v) for v in box)
    crop_w = x1 - x0
    crop_h = y1 - y0
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"invalid tile box {tuple(box)}")

    global_to_full_x = float(full_width) / float(global_image_width)
    global_to_full_y = float(full_height) / float(global_image_height)
    crop_to_output_x = float(output_width) / float(crop_w)
    crop_to_output_y = float(output_height) / float(crop_h)

    global_fx_1024 = _focal_pixels(
        float(global_camera["camera_angle_x"]),
        int(global_image_width),
    )
    global_fy_1024 = global_fx_1024
    full_fx = global_fx_1024 * global_to_full_x
    full_fy = global_fy_1024 * global_to_full_y

    tile_fx = full_fx * crop_to_output_x
    tile_fy = full_fy * crop_to_output_y
    tile_fov_x = 2.0 * math.atan(float(output_width) / (2.0 * tile_fx))
    tile_fov_y = 2.0 * math.atan(float(output_height) / (2.0 * tile_fy))
    tile_mesh_scale = float(global_camera["mesh_scale"])
    tile_distance = distance_from_fov(
        float(tile_fov_x),
        torch.tensor([-1.0, 0.0, 0.0]),
        torch.tensor(
            [
                0 - int(extend_pixel),
                int(output_width) - 1 + int(extend_pixel),
            ]
        ),
        float(tile_mesh_scale),
        int(output_width),
    )["distance_from_x"]
    global_effective_distance = (
        float(global_camera["distance"])
        * float(global_camera["mesh_scale"])
    )
    tile_effective_distance = float(tile_distance) * float(tile_mesh_scale)
    if global_effective_distance <= 0.0:
        raise ValueError("global camera distance * mesh_scale must be positive")
    local_recanonicalization_scale = (
        tile_effective_distance / global_effective_distance
    )

    # Exact off-axis crop principal point for untransformed global geometry.
    full_cx = float(full_width) / 2.0
    full_cy = float(full_height) / 2.0
    offaxis_cx = (full_cx - float(x0)) * crop_to_output_x
    offaxis_cy = (full_cy - float(y0)) * crop_to_output_y
    shift_x = (float(output_width) / 2.0 - offaxis_cx) / float(output_width)
    shift_y = (
        (offaxis_cy - float(output_height) / 2.0)
        / float(output_width)
        * float(offaxis_shift_y_sign)
    )

    return TileCameraTransform(
        tile_id=int(tile_id),
        box=(x0, y0, x1, y1),
        output_width=int(output_width),
        output_height=int(output_height),
        camera_angle_x=float(tile_fov_x),
        camera_angle_y=float(tile_fov_y),
        distance=float(tile_distance),
        mesh_scale=float(tile_mesh_scale),
        global_distance=float(global_camera["distance"]),
        global_mesh_scale=float(global_camera["mesh_scale"]),
        local_recanonicalization_scale=float(local_recanonicalization_scale),
        fx=float(tile_fx),
        fy=float(tile_fy),
        cx=float(output_width) / 2.0,
        cy=float(output_height) / 2.0,
        offaxis_cx=float(offaxis_cx),
        offaxis_cy=float(offaxis_cy),
        offaxis_shift_x=float(shift_x),
        offaxis_shift_y=float(shift_y),
        global_fx_1024=float(global_fx_1024),
        global_fy_1024=float(global_fy_1024),
        full_fx_4096=float(full_fx),
        full_fy_4096=float(full_fy),
        global_to_full_scale_x=float(global_to_full_x),
        global_to_full_scale_y=float(global_to_full_y),
        crop_to_output_scale_x=float(crop_to_output_x),
        crop_to_output_scale_y=float(crop_to_output_y),
        tile_center_full_x=(float(x0) + float(x1)) / 2.0,
        tile_center_full_y=(float(y0) + float(y1)) / 2.0,
    )


def _project_global_q_to_1024_and_4096(
    q_global: torch.Tensor,
    *,
    global_camera: Mapping[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    points = _camera_q_to_points(
        q_global,
        distance=float(global_camera["distance"]),
        mesh_scale=float(global_camera["mesh_scale"]),
    )
    fx = _focal_pixels(
        float(global_camera["camera_angle_x"]),
        GLOBAL_CAMERA_IMAGE_SIZE,
    )
    uv_1024, depth, finite = _project_points_with_intrinsics(
        points,
        fx=fx,
        fy=fx,
        cx=GLOBAL_CAMERA_IMAGE_SIZE / 2.0,
        cy=GLOBAL_CAMERA_IMAGE_SIZE / 2.0,
    )
    uv_1024 = uv_1024[0]
    depth = depth[0]
    finite = finite[0]
    scale = torch.tensor(
        [
            IMAGE_CANONICAL / GLOBAL_CAMERA_IMAGE_SIZE,
            IMAGE_CANONICAL / GLOBAL_CAMERA_IMAGE_SIZE,
        ],
        device=uv_1024.device,
        dtype=uv_1024.dtype,
    )
    uv_4096 = uv_1024 * scale[None]
    return points, uv_1024, uv_4096, depth, finite


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


def _global_q_to_centered_tile_q(
    q_global: torch.Tensor,
    *,
    global_camera: Mapping[str, float],
    transform: TileCameraTransform,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Exact project/crop/back-project mapping into the centered tile camera.

    This is deliberately not a point-cloud normalization. The translation in X/Y
    is depth dependent and is fixed by the tile center ray; the scale is fixed by
    the focal/FOV ratio.
    """
    if q_global.ndim != 2 or q_global.shape[1] != 3:
        raise ValueError("q_global must be [N,3]")

    global_points, uv_1024, uv_4096, depth, finite = (
        _project_global_q_to_1024_and_4096(
            q_global,
            global_camera=global_camera,
        )
    )
    if not bool(finite.all().item()):
        raise RuntimeError("selected rows contain invalid global camera projections")

    x0, y0, _, _ = transform.box
    uv_tile = torch.stack(
        [
            (uv_4096[:, 0] - float(x0))
            * float(transform.crop_to_output_scale_x),
            (uv_4096[:, 1] - float(y0))
            * float(transform.crop_to_output_scale_y),
        ],
        dim=1,
    )

    qz = q_global[:, 2]
    local_depth = float(transform.distance) - (
        qz / (2.0 * float(transform.mesh_scale))
    )
    if bool((local_depth <= 0).any().item()):
        raise RuntimeError("local recanonicalized depth became non-positive")
    local_points = _backproject_pixels_with_depth(
        uv_tile,
        local_depth,
        fx=float(transform.fx),
        fy=float(transform.fy),
        cx=float(transform.cx),
        cy=float(transform.cy),
    )
    q_local = _camera_points_to_q(
        local_points,
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
    )

    # Exact pixel-domain closure test.
    uv_reproject, depth_reproject, finite_reproject = _project_points_with_intrinsics(
        local_points,
        fx=float(transform.fx),
        fy=float(transform.fy),
        cx=float(transform.cx),
        cy=float(transform.cy),
    )
    uv_reproject = uv_reproject[0]
    depth_reproject = depth_reproject[0]
    finite_reproject = finite_reproject[0]
    pixel_error = torch.linalg.vector_norm(uv_reproject - uv_tile, dim=1)
    depth_error = (depth_reproject - local_depth).abs()

    # Closed form: q_z is preserved, while X/Y use the ratio between local
    # and global ray depths plus the tile-center principal-point offset.
    full_cx = IMAGE_CANONICAL / 2.0
    full_cy = IMAGE_CANONICAL / 2.0
    dx_camera = (
        (full_cx - float(transform.tile_center_full_x))
        / float(transform.full_fx_4096)
    )
    dy_camera = (
        (float(transform.tile_center_full_y) - full_cy)
        / float(transform.full_fy_4096)
    )
    global_scale = float(global_camera["mesh_scale"])
    local_scale = float(transform.mesh_scale)
    qx_closed = 2.0 * local_scale * local_depth * (
        q_global[:, 0] / (2.0 * global_scale * depth)
        + float(dx_camera)
    )
    qy_closed = 2.0 * local_scale * local_depth * (
        q_global[:, 1] / (2.0 * global_scale * depth)
        + float(dy_camera)
    )
    q_closed = torch.stack([qx_closed, qy_closed, qz], dim=1)
    q_closed_error = (q_closed - q_local).abs()
    normalized_depth_error = (q_local[:, 2] - qz).abs()

    overflow = (q_local.abs() - 1.0).clamp_min(0.0)
    outside = (overflow > 0).any(dim=1)
    stats = {
        "rows": int(q_local.shape[0]),
        "raw_outside_rows": int(outside.sum().item()),
        "raw_outside_fraction": float(outside.float().mean().item()),
        "raw_max_overflow": float(overflow.max().item()),
        "q_global_min": [
            float(v) for v in q_global.amin(dim=0).detach().cpu().tolist()
        ],
        "q_global_max": [
            float(v) for v in q_global.amax(dim=0).detach().cpu().tolist()
        ],
        "q_local_min": [
            float(v) for v in q_local.amin(dim=0).detach().cpu().tolist()
        ],
        "q_local_max": [
            float(v) for v in q_local.amax(dim=0).detach().cpu().tolist()
        ],
        "pixel_roundtrip_mean": float(pixel_error.mean().item()),
        "pixel_roundtrip_max": float(pixel_error.max().item()),
        "local_depth_roundtrip_max": float(depth_error.max().item()),
        "normalized_depth_q_error_max": float(
            normalized_depth_error.max().item()
        ),
        "closed_form_q_error_max": float(q_closed_error.max().item()),
        "tile_center_camera_slope_x": float(dx_camera),
        "tile_center_camera_slope_y": float(dy_camera),
        "finite_reprojection_rows": int(finite_reproject.sum().item()),
        "local_recanonicalization_scale": float(
            transform.local_recanonicalization_scale
        ),
    }
    return q_local, uv_tile, uv_reproject, stats


def _centered_tile_q_to_global_q(
    q_local: torch.Tensor,
    *,
    global_camera: Mapping[str, float],
    transform: TileCameraTransform,
    validate_roundtrip: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Invert the centered local recanonicalization point by point.

    The local point supplies the tile pixel and preserved normalized q_z.
    Global physical depth is reconstructed from q_z and the MoGe global
    camera, then the full-image pixel is back-projected with global K.
    """
    if q_local.ndim != 2 or q_local.shape[1] != 3:
        raise ValueError("q_local must be [N,3]")
    local_points = _camera_q_to_points(
        q_local,
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
    )
    uv_tile, _, finite_local = _project_points_with_intrinsics(
        local_points,
        fx=float(transform.fx),
        fy=float(transform.fy),
        cx=float(transform.cx),
        cy=float(transform.cy),
    )
    uv_tile = uv_tile[0]
    finite_local = finite_local[0]
    if not bool(finite_local.all().item()):
        raise RuntimeError("local q contains invalid camera projections")

    x0, y0, _, _ = transform.box
    uv_full = torch.stack(
        [
            uv_tile[:, 0] / float(transform.crop_to_output_scale_x)
            + float(x0),
            uv_tile[:, 1] / float(transform.crop_to_output_scale_y)
            + float(y0),
        ],
        dim=1,
    )
    qz = q_local[:, 2]
    global_scale = float(global_camera["mesh_scale"])
    global_depth = float(global_camera["distance"]) - (
        qz / (2.0 * global_scale)
    )
    if bool((global_depth <= 0).any().item()):
        raise RuntimeError("inverse global depth became non-positive")
    global_points = _backproject_pixels_with_depth(
        uv_full,
        global_depth,
        fx=float(transform.full_fx_4096),
        fy=float(transform.full_fy_4096),
        cx=IMAGE_CANONICAL / 2.0,
        cy=IMAGE_CANONICAL / 2.0,
    )
    q_global = _camera_points_to_q(
        global_points,
        distance=float(global_camera["distance"]),
        mesh_scale=global_scale,
    )

    stats: Dict[str, Any] = {"rows": int(q_local.shape[0])}
    if validate_roundtrip:
        q_local_check, uv_tile_check, _, _ = (
            _global_q_to_centered_tile_q(
                q_global,
                global_camera=global_camera,
                transform=transform,
            )
        )
        stats.update(
            {
                "q_roundtrip_max_abs": float(
                    (q_local_check - q_local).abs().max().item()
                ),
                "pixel_roundtrip_max": float(
                    torch.linalg.vector_norm(
                        uv_tile_check - uv_tile,
                        dim=1,
                    ).max().item()
                ),
                "normalized_depth_q_error_max": float(
                    (q_global[:, 2] - qz).abs().max().item()
                ),
            }
        )
    return q_global, global_points, uv_full, stats



def _quantize_local_q_without_geometry_clip(
    q_local: torch.Tensor,
    *,
    resolution: int,
    lattice_name: str,
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Quantize only points already inside the requested local lattice.

    Both hard overflow and epsilon-sized numerical overflow are dropped. No
    point is clipped or compressed onto a lattice boundary.
    """
    if resolution <= 1:
        raise ValueError("resolution must exceed one")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if q_local.ndim != 2 or q_local.shape[1] != 3:
        raise ValueError("q_local must be [N,3]")

    overflow = (q_local.abs() - 1.0).clamp_min(0.0)
    strict_inside = (q_local.abs() <= 1.0).all(dim=1)
    hard_outside = (overflow > float(epsilon)).any(dim=1)
    numeric_outside = (~strict_inside) & (~hard_outside)
    kept = strict_inside
    if not bool(kept.any().item()):
        raise RuntimeError(
            f"all projected rows lie outside the local {lattice_name} canonical cube"
        )

    q_kept = q_local[kept]
    ids = _q_to_endpoint_indices(q_kept, int(resolution))
    if bool(((ids < 0) | (ids >= int(resolution))).any().item()):
        raise RuntimeError(f"{lattice_name} quantization produced out-of-range indices")

    coords_per_source = torch.cat(
        [
            torch.zeros(
                (ids.shape[0], 1),
                device=ids.device,
                dtype=torch.int32,
            ),
            ids,
        ],
        dim=1,
    )
    coords_unique = torch.unique(coords_per_source, dim=0)
    stats = {
        "lattice_name": str(lattice_name),
        "lattice_resolution": int(resolution),
        "input_rows": int(q_local.shape[0]),
        "hard_outside_rows_dropped": int(hard_outside.sum().item()),
        "hard_outside_fraction": float(hard_outside.float().mean().item()),
        "numeric_boundary_rows_dropped": int(numeric_outside.sum().item()),
        "boundary_epsilon_for_hard_outside_reporting": float(epsilon),
        "rows_before_unique": int(coords_per_source.shape[0]),
        "unique_tokens": int(coords_unique.shape[0]),
        "quantization_merge_rows": int(
            coords_per_source.shape[0] - coords_unique.shape[0]
        ),
    }
    return coords_unique, kept, stats

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
    draw.rectangle((0, 0, canvas.width, 38), fill=(0, 0, 0, 195))
    draw.text(
        (8, 9),
        f"{title} | points={original_count:,}",
        fill=(255, 255, 255, 255),
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
    draw.rectangle((0, 0, canvas.width, 38), fill=(0, 0, 0, 200))
    draw.text(
        (8, 8),
        "red=continuous local projection, cyan=quantized tile C64 projection",
        fill=(255, 255, 255, 255),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _composite_on_black(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


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


def _image_to_metric_tensor(
    path: Path,
    *,
    resolution: int,
) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = _composite_on_black(image)
    target = (int(resolution), int(resolution))
    if rgb.size != target:
        rgb = rgb.resize(target, Image.Resampling.LANCZOS)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _psnr_metric(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    mse = float(F.mse_loss(prediction, reference).item())
    return float("inf") if mse <= 0.0 else float(10.0 * math.log10(1.0 / mse))


def _gaussian_kernel(
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinates = torch.arange(window_size, device=device, dtype=dtype)
    coordinates = coordinates - (window_size - 1) / 2.0
    kernel = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel[:, None] * kernel[None, :]


def _ssim_metric(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    x = reference.unsqueeze(0).float()
    y = prediction.unsqueeze(0).float()
    channels = int(x.shape[1])
    kernel = _gaussian_kernel(
        window_size, sigma, x.device, x.dtype
    )[None, None].expand(channels, 1, window_size, window_size)
    padding = window_size // 2
    mu_x = F.conv2d(x, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=channels)
    mu_x_sq, mu_y_sq, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sigma_x_sq = (
        F.conv2d(x * x, kernel, padding=padding, groups=channels) - mu_x_sq
    )
    sigma_y_sq = (
        F.conv2d(y * y, kernel, padding=padding, groups=channels) - mu_y_sq
    )
    sigma_xy = (
        F.conv2d(x * y, kernel, padding=padding, groups=channels) - mu_xy
    )
    c1, c2 = 0.01**2, 0.03**2
    value = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1)
        * (sigma_x_sq + sigma_y_sq + c2)
        + 1e-12
    )
    return float(value.mean().item())


class _LPIPSEvaluator:
    def __init__(self, network: str, device: torch.device):
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError("The lpips package is required: pip install lpips") from exc
        self.device = device
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = lpips.LPIPS(net=network).eval().to(device)

    @torch.inference_mode()
    def evaluate(
        self,
        reference: torch.Tensor,
        prediction: torch.Tensor,
    ) -> float:
        x = reference.unsqueeze(0).to(self.device)
        y = prediction.unsqueeze(0).to(self.device)
        return float(
            self.model(x * 2.0 - 1.0, y * 2.0 - 1.0).mean().item()
        )


def _prepare_global_baseline_tile_crop(
    *,
    global_render_path: Path,
    reference_path: Path,
    box: Sequence[int],
    output_dir: Path,
) -> Dict[str, Any]:
    with Image.open(global_render_path) as image:
        global_render = _composite_on_black(image)

    x0, y0, x1, y1 = (int(value) for value in box)
    scale_x = float(global_render.width) / float(IMAGE_CANONICAL)
    scale_y = float(global_render.height) / float(IMAGE_CANONICAL)
    crop_box = (
        int(round(x0 * scale_x)),
        int(round(y0 * scale_y)),
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
    )
    crop_box = (
        max(0, min(global_render.width - 1, crop_box[0])),
        max(0, min(global_render.height - 1, crop_box[1])),
        max(1, min(global_render.width, crop_box[2])),
        max(1, min(global_render.height, crop_box[3])),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        raise RuntimeError(f"invalid baseline crop box {crop_box} for {global_render.size}")

    crop = global_render.crop(crop_box).resize(
        (IMAGE_FLOW, IMAGE_FLOW),
        Image.Resampling.LANCZOS,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_path = output_dir / "global_baseline_1024_crop.png"
    crop.save(crop_path)
    extras = _save_extra_comparisons(
        reference_path,
        crop_path,
        output_dir / "comparisons",
    )
    metadata = {
        "baseline_render_png": str(crop_path),
        "baseline_overlay_png": extras["overlay_png"],
        "baseline_diff_heatmap_png": extras["diff_heatmap_png"],
        "baseline_triptych_png": extras["triptych_png"],
        "baseline_source_render_png": str(global_render_path),
        "baseline_source_size": list(global_render.size),
        "baseline_crop_box_pixels": list(crop_box),
        "canonical_tile_box": [x0, y0, x1, y1],
        "baseline_view_method": (
            "crop the full global-camera render by the canonical tile box; "
            "this is ray-equivalent to the corresponding off-axis crop camera"
        ),
    }
    _atomic_json(output_dir / "baseline_crop.json", metadata)
    return metadata


def _estimate_camera(
    *,
    image_1024: Image.Image,
    output_dir: Path,
    manual_fov: float,
    mesh_scale: float,
    extend_pixel: int,
    image_resolution: int,
    moge_model_path: Optional[str],
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
    temporary = output_dir / f"_projective_tile_moge_{time.time_ns()}.png"
    image_1024.save(temporary)
    load_kwargs: Dict[str, Any] = {"device": "cuda"}
    if moge_model_path:
        load_kwargs["model_name"] = str(moge_model_path)
    model = load_moge_model(**load_kwargs)
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
            "steps": int(args.ss_steps),
            "guidance_strength": float(args.ss_guidance_strength),
            "guidance_rescale": float(args.ss_guidance_rescale),
            "rescale_t": float(args.ss_rescale_t),
        },
        "shape": {
            **pipeline.shape_slat_sampler_params,
            "steps": int(args.shape_steps),
            "guidance_strength": float(args.shape_guidance_strength),
            "guidance_rescale": float(args.shape_guidance_rescale),
            "rescale_t": float(args.shape_rescale_t),
        },
        "texture": {
            **pipeline.tex_slat_sampler_params,
            "steps": int(args.texture_steps),
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


def _quantize_shape512_candidates_to_c64(candidates: torch.Tensor) -> torch.Tensor:
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


def _learned_subdivide_shape1024_to_c1024(
    pipeline: Any,
    shape1024_denorm: SparseTensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    decoder = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        decoder.to(pipeline.device)
        decoder.low_vram = True
    try:
        decoder.set_resolution(DECODE_TILE)
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
    if not torch.isfinite(candidates.to(torch.float32)).all():
        raise RuntimeError("one-step shape1024 upsample returned non-finite coordinates")

    rounded = torch.round(candidates[:, 1:].to(torch.float32))
    max_fractional_error = float(
        (candidates[:, 1:].to(torch.float32) - rounded).abs().max().item()
    )
    xyz = rounded.to(torch.int32)
    coords1024_all = torch.cat([candidates[:, :1].to(torch.int32), xyz], dim=1)
    valid = (
        (coords1024_all[:, 1:] >= 0)
        & (coords1024_all[:, 1:] < GRID_GLOBAL_UPSAMPLED)
    ).all(dim=1)
    coords1024 = torch.unique(coords1024_all[valid], dim=0)
    if coords1024.numel() == 0:
        raise RuntimeError(
            "shape1024 decoder subdivision produced no valid C1024 coordinates"
        )

    stats = {
        "source_c64_tokens": int(shape1024_denorm.coords.shape[0]),
        "candidate_rows": int(candidates.shape[0]),
        "valid_candidate_rows": int(valid.sum().item()),
        "discarded_out_of_range_rows": int((~valid).sum().item()),
        "unique_c1024_points": int(coords1024.shape[0]),
        "unique_merge_rows": int(valid.sum().item() - coords1024.shape[0]),
        "max_fractional_coordinate_error": max_fractional_error,
        "min_xyz": [
            int(v) for v in coords1024[:, 1:].amin(dim=0).detach().cpu().tolist()
        ],
        "max_xyz": [
            int(v) for v in coords1024[:, 1:].amax(dim=0).detach().cpu().tolist()
        ],
    }
    return coords1024, stats



def _run_shape512_and_upsample_c64(
    *,
    pipeline: Any,
    image_512: Image.Image,
    coords32: torch.Tensor,
    camera: Mapping[str, float],
    params: Mapping[str, Mapping[str, Any]],
    seed: int,
    description: str,
) -> Tuple[torch.Tensor, float]:
    condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_512,
        [image_512.convert("RGB")],
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
        description=description,
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


def _run_texture1024(
    *,
    pipeline: Any,
    image_1024: Image.Image,
    coords64: torch.Tensor,
    camera: Mapping[str, float],
    shape_norm: SparseTensor,
    params: Mapping[str, Mapping[str, Any]],
    seed: int,
    description: str,
) -> Tuple[SparseTensor, SparseTensor, float]:
    texture_condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [image_1024.convert("RGB")],
        coords64,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
        grid_resolution_override=GRID_SHAPE_1024,
    )
    texture_model = pipeline.models["tex_slat_flow_model_1024"]
    texture_channels = int(texture_model.in_channels) - int(
        shape_norm.feats.shape[1]
    )
    if texture_channels <= 0:
        raise RuntimeError(f"invalid texture noise channel count {texture_channels}")
    texture_noise = SparseTensor(
        feats=_randn(
            coords64.shape[0],
            texture_channels,
            device=pipeline.device,
            seed=int(seed),
        ),
        coords=coords64,
    )
    texture_norm, elapsed = _run_sampler_full(
        pipeline=pipeline,
        sampler=pipeline.tex_slat_sampler,
        model=texture_model,
        noise=texture_noise,
        condition=texture_condition,
        params=params["texture"],
        description=description,
        concat_cond=shape_norm,
    )
    texture_denorm = _denormalize_sparse(
        texture_norm,
        pipeline.tex_slat_normalization,
    )
    del texture_condition, texture_noise
    _empty_cuda_cache()
    return texture_norm, texture_denorm, elapsed


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
        description="Global official shape 512",
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




def _run_tile_projected_c64_only(
    *,
    pipeline: Any,
    tile_image: Image.Image,
    projected_coords64: torch.Tensor,
    tile_camera: Mapping[str, float],
    params: Mapping[str, Mapping[str, Any]],
    seed: int,
    label: str,
    max_tokens: int,
) -> ModelResult:
    """Run tile shape1024/texture1024 directly on projected-global C64 support.

    No tile-native sparse-structure sampling, C32 filtering, shape512 flow,
    learned native C64 upsampling, or C64 fusion is executed here.
    """
    tile_1024 = tile_image.convert("RGB")
    coords64, filter_stats = _filter_unique_grid_coords(
        projected_coords64.to(device=pipeline.device, dtype=torch.int32),
        resolution=GRID_SHAPE_1024,
        source_name="projected_global_c1024_quantized_local_c64",
    )
    if coords64.numel() == 0:
        raise RuntimeError(f"{label}: projected-global C64 support is empty")
    if coords64.shape[0] > int(max_tokens):
        raise RuntimeError(
            f"{label}: projected-global C64 support has {coords64.shape[0]:,} "
            f"tokens; exceeds --max-num-tokens={int(max_tokens):,}"
        )

    print(
        f"[tile-projected-c64-only] {label}: "
        f"input_C64={projected_coords64.shape[0]:,} "
        f"unique_C64={coords64.shape[0]:,} "
        f"duplicates_removed={filter_stats['duplicate_rows_removed']:,}"
    )

    shape_norm, shape_denorm, shape1024_seconds = _run_shape1024(
        pipeline=pipeline,
        image_1024=tile_1024,
        coords64=coords64,
        camera=tile_camera,
        params=params,
        seed=seed + 201,
        description=f"{label} shape 1024",
    )

    texture_norm, texture_denorm, texture1024_seconds = _run_texture1024(
        pipeline=pipeline,
        image_1024=tile_1024,
        coords64=coords64,
        camera=tile_camera,
        shape_norm=shape_norm,
        params=params,
        seed=seed + 301,
        description=f"{label} texture 1024",
    )
    _empty_cuda_cache()
    return ModelResult(
        shape_norm=shape_norm,
        shape_denorm=shape_denorm,
        texture_norm=texture_norm,
        texture_denorm=texture_denorm,
        tile_projective_c32_tokens=0,
        tile_ss_c32_tokens=0,
        tile_c32_overlap_tokens=0,
        tile_c32_tokens=0,
        tile_projective_c64_tokens=int(coords64.shape[0]),
        tile_native_c64_tokens=0,
        tile_c64_overlap_tokens=0,
        tile_c64_tokens=int(coords64.shape[0]),
        tile_ss_seconds=0.0,
        shape512_seconds=0.0,
        shape1024_seconds=float(shape1024_seconds),
        texture1024_seconds=float(texture1024_seconds),
    )

def _decode_normal_mesh_with_ovoxel(
    *,
    pipeline: Any,
    shape_latent: SparseTensor,
    texture_latent: SparseTensor,
    label: str,
    resolution: int = DECODE_TILE,
) -> Tuple[List[Any], Any]:
    """Run the normal decoder once and retain its native ``MeshWithVoxel``."""
    decoded = pipeline.decode_latent(
        shape_latent,
        texture_latent,
        int(resolution),
    )
    if len(decoded) != 1:
        raise RuntimeError(
            f"{label}: normal decoder returned {len(decoded)} meshes; expected one"
        )
    mesh_with_ovoxel = decoded[0]
    required = (
        "vertices",
        "faces",
        "coords",
        "attrs",
        "origin",
        "voxel_size",
        "voxel_shape",
    )
    missing = [
        name for name in required if not hasattr(mesh_with_ovoxel, name)
    ]
    if missing:
        raise RuntimeError(
            f"{label}: normal decoder output is missing {', '.join(missing)}"
        )
    if mesh_with_ovoxel.vertices.ndim != 2 or mesh_with_ovoxel.vertices.shape[1] != 3:
        raise ValueError(f"{label}: decoded vertices must be [N,3]")
    if mesh_with_ovoxel.faces.ndim != 2 or mesh_with_ovoxel.faces.shape[1] != 3:
        raise ValueError(f"{label}: decoded faces must be [M,3]")
    if mesh_with_ovoxel.coords.ndim != 2 or mesh_with_ovoxel.coords.shape[1] != 3:
        raise ValueError(f"{label}: decoded O-Voxel coords must be [L,3]")
    if (
        mesh_with_ovoxel.attrs.ndim != 2
        or mesh_with_ovoxel.attrs.shape[0] != mesh_with_ovoxel.coords.shape[0]
    ):
        raise ValueError(
            f"{label}: decoded O-Voxel attrs must be [L,C] and align with coords"
        )
    print(
        f"[normal-decoder] {label}: "
        f"vertices={mesh_with_ovoxel.vertices.shape[0]:,} "
        f"faces={mesh_with_ovoxel.faces.shape[0]:,} "
        f"ovoxel_entries={mesh_with_ovoxel.coords.shape[0]:,}"
    )
    return decoded, mesh_with_ovoxel


def _evaluate_tile_result(
    *,
    pipeline: Any,
    result: ModelResult,
    output_dir: Path,
    camera: Mapping[str, float],
    global_camera: Mapping[str, float],
    transform: TileCameraTransform,
    seed: int,
    label: str,
    reference_image: Path,
    args: argparse.Namespace,
    envmap: Any,
) -> Dict[str, Any]:
    meshes, mesh = _decode_normal_mesh_with_ovoxel(
        pipeline=pipeline,
        shape_latent=result.shape_denorm,
        texture_latent=result.texture_denorm,
        label=label,
    )
    decoder_metadata = {
        "decoder_vertices": int(mesh.vertices.shape[0]),
        "decoder_faces": int(mesh.faces.shape[0]),
        "active_voxels": int(mesh.coords.shape[0]),
        "sample_type": type(mesh).__name__,
        "renderer": "pixal3d.utils.render_utils.render_frames",
        "camera_params": dict(camera),
        "global_camera_params": dict(global_camera),
        "tile_camera_transform": asdict(transform),
        "seed": int(seed),
    }

    # The decoded MeshWithVoxel owns the tensors needed by the official renderer.
    result.shape_norm = None  # type: ignore[assignment]
    result.shape_denorm = None  # type: ignore[assignment]
    result.texture_norm = None  # type: ignore[assignment]
    result.texture_denorm = None  # type: ignore[assignment]
    _empty_cuda_cache()

    metric_row = render_and_evaluate_mesh(
        mesh,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        output_dir=output_dir / "aligned_eval",
        reference_image=reference_image,
        resolution=int(args.render_resolution),
        metric_resolution=int(args.metric_resolution),
        envmap=envmap,
        envmap_name=str(args.envmap),
        ssaa=int(args.render_ssaa),
        peel_layers=int(args.render_peel_layers),
        use_envmap_bg=bool(args.use_envmap_bg),
        lpips_net=str(args.lpips_net),
        metric_device=str(args.metric_device),
        skip_lpips=bool(args.skip_lpips),
    )
    extras = _save_extra_comparisons(
        Path(metric_row["original_png"]),
        Path(metric_row["render_png"]),
        output_dir / "comparisons",
    )
    del meshes, mesh
    _empty_cuda_cache()
    return {**decoder_metadata, **metric_row, **extras}


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


def _project_grid_coords_to_tile_uv(
    coords: torch.Tensor,
    *,
    resolution: int,
    transform: TileCameraTransform,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = _endpoint_indices_to_q(coords[:, 1:4], int(resolution)).to(coords.device)
    points = _camera_q_to_points(
        q,
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
    )
    uv, _, valid = _project_points_with_intrinsics(
        points,
        fx=float(transform.fx),
        fy=float(transform.fy),
        cx=float(transform.cx),
        cy=float(transform.cy),
    )
    return uv[0], q, valid[0]




def _prepare_projective_tile_supports(
    *,
    reference: Image.Image,
    selected_coords128: torch.Tensor,
    global_camera: Mapping[str, float],
    transform: TileCameraTransform,
    output_dir: Path,
    boundary_epsilon: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Transform selected global C1024 support and quantize only local C64.

    The returned coordinates are the complete and only coordinate support used
    by the tile shape1024 and texture1024 flows.
    """
    q_global = _endpoint_indices_to_q(
        selected_coords128[:, 1:4],
        GRID_GLOBAL_UPSAMPLED,
    ).to(selected_coords128.device)
    q_local, uv_continuous, _, transform_stats = (
        _global_q_to_centered_tile_q(
            q_global,
            global_camera=global_camera,
            transform=transform,
        )
    )

    coords64, kept_mask64, quant_stats64 = (
        _quantize_local_q_without_geometry_clip(
            q_local,
            resolution=GRID_SHAPE_1024,
            lattice_name="projected-global local C64",
            epsilon=float(boundary_epsilon),
        )
    )
    q_local_kept = q_local[kept_mask64]
    uv_continuous_kept = uv_continuous[kept_mask64]
    q_global_kept = q_global[kept_mask64]

    ids_per_source = _q_to_endpoint_indices(
        q_local_kept,
        GRID_SHAPE_1024,
    )
    q_quantized = _endpoint_indices_to_q(
        ids_per_source,
        GRID_SHAPE_1024,
    ).to(q_local_kept.device)
    quantized_points = _camera_q_to_points(
        q_quantized,
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
    )
    uv_quantized, _, quantized_valid = _project_points_with_intrinsics(
        quantized_points,
        fx=float(transform.fx),
        fy=float(transform.fy),
        cx=float(transform.cx),
        cy=float(transform.cy),
    )
    uv_quantized = uv_quantized[0]
    quantized_valid = quantized_valid[0]
    pixel_error64 = torch.linalg.vector_norm(
        uv_quantized - uv_continuous_kept,
        dim=1,
    )

    unique_uv, unique_q, unique_valid = _project_grid_coords_to_tile_uv(
        coords64,
        resolution=GRID_SHAPE_1024,
        transform=transform,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _draw_uv_points(
        reference,
        uv_continuous_kept,
        q_global_kept[:, 2],
        output_dir / "continuous_local_projection.png",
        "selected global C1024 transformed into centered tile coordinates",
    )
    _draw_uv_points(
        reference,
        unique_uv[unique_valid],
        unique_q[:, 2][unique_valid],
        output_dir / "projected_global_c64_support_overlay.png",
        "projected-global C1024 quantized directly to unique local C64",
    )
    _save_density_image(
        unique_uv[unique_valid],
        output_dir / "projected_global_c64_support_density.png",
        resolution=IMAGE_FLOW,
    )
    _save_quantization_error_image(
        reference,
        uv_continuous_kept[quantized_valid],
        uv_quantized[quantized_valid],
        output_dir / "projected_global_c64_quantization_error.png",
    )

    stats = {
        "support_mode": "projected_global_c64_only",
        "selected_global_c1024_rows": int(selected_coords128.shape[0]),
        "tile_camera": asdict(transform),
        "transform_stats": transform_stats,
        "c64_quantization_stats": quant_stats64,
        "projected_c64_tokens": int(coords64.shape[0]),
        "visible_unique_projected_c64_tokens": int(unique_valid.sum().item()),
        "pixel_error_mean": float(pixel_error64.mean().item()),
        "pixel_error_p95": float(torch.quantile(pixel_error64, 0.95).item()),
        "pixel_error_max": float(pixel_error64.max().item()),
        "c64_pixel_error_mean": float(pixel_error64.mean().item()),
        "c64_pixel_error_p95": float(torch.quantile(pixel_error64, 0.95).item()),
        "c64_pixel_error_max": float(pixel_error64.max().item()),
        "tile_native_ss_enabled": False,
        "tile_shape512_enabled": False,
        "native_c64_enabled": False,
        "c64_fusion_enabled": False,
    }
    _atomic_json(output_dir / "support_stats.json", stats)
    torch.save(
        {
            "selected_global_c1024": selected_coords128.detach().cpu(),
            "q_global": q_global.detach().cpu(),
            "q_local": q_local.detach().cpu(),
            "kept_mask_c64": kept_mask64.detach().cpu(),
            "coords64_projected_unique": coords64.detach().cpu(),
            "uv_continuous": uv_continuous.detach().cpu(),
            "uv_continuous_kept": uv_continuous_kept.detach().cpu(),
            "uv_quantized_c64": uv_quantized.detach().cpu(),
            "pixel_error_c64": pixel_error64.detach().cpu(),
            "tile_camera": asdict(transform),
        },
        output_dir / "support_debug.pt",
    )
    print(
        f"[tile-projected-c64] tile={transform.tile_id:02d} "
        f"selected_global_C1024={selected_coords128.shape[0]:,} "
        f"projected_unique_C64={coords64.shape[0]:,} "
        f"outside={quant_stats64['hard_outside_fraction']:.6f}"
    )
    return coords64, stats

def _filter_unique_grid_coords(
    coords: torch.Tensor,
    *,
    resolution: int,
    source_name: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Keep only batch-0 integer coordinates inside a sparse cubic grid."""
    if int(resolution) <= 1:
        raise ValueError("resolution must exceed one")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(
            f"{source_name}: expected sparse coordinates [N,4], got "
            f"{tuple(coords.shape)}"
        )

    coords_i32 = coords.to(dtype=torch.int32)
    batch_valid = coords_i32[:, 0] == 0
    xyz_valid = (
        (coords_i32[:, 1:] >= 0)
        & (coords_i32[:, 1:] < int(resolution))
    ).all(dim=1)
    valid = batch_valid & xyz_valid
    filtered = coords_i32[valid]
    unique = torch.unique(filtered, dim=0)
    stats = {
        "source": str(source_name),
        "resolution": int(resolution),
        "input_rows": int(coords_i32.shape[0]),
        "invalid_batch_rows_dropped": int((~batch_valid).sum().item()),
        "out_of_grid_rows_dropped": int((batch_valid & ~xyz_valid).sum().item()),
        "valid_rows_before_unique": int(filtered.shape[0]),
        "unique_rows": int(unique.shape[0]),
        "duplicate_rows_removed": int(filtered.shape[0] - unique.shape[0]),
    }
    return unique, stats


def _resize_panel(path: Optional[Path], size: int = 512) -> Image.Image:
    if path is None or not path.is_file():
        return Image.new("RGB", (size, size), (30, 30, 30))
    image = Image.open(path).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def _label_panel(image: Image.Image, text: str) -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", (canvas.width, 76), (0, 0, 0, 185))
    canvas.alpha_composite(overlay, dest=(0, 0))
    draw = ImageDraw.Draw(canvas)
    y = 8
    for line in text.splitlines():
        draw.text((9, y), line, fill=(255, 255, 255, 255))
        y += 16
    return canvas.convert("RGB")




def _save_tile_comparison_sheet(
    *,
    reference_path: Path,
    support_overlay_path: Path,
    route_summary: Optional[Mapping[str, Any]],
    baseline_summary: Optional[Mapping[str, Any]],
    projected_c64_token_count: int,
    outside_fraction: float,
    output_path: Path,
) -> str:
    reference = _label_panel(
        _resize_panel(reference_path),
        "Reference 4096 crop -> 1024",
    )
    support = _label_panel(
        _resize_panel(support_overlay_path),
        (
            "Global C1024 -> centered tile -> quantized C64 only\n"
            f"projected C64={projected_c64_token_count:,} "
            f"outside={outside_fraction:.4%}"
        ),
    )

    if baseline_summary is None:
        baseline_render = _label_panel(
            _resize_panel(None),
            "Global ordinary 1024 baseline crop\nunavailable",
        )
        baseline_diff = _label_panel(
            _resize_panel(None),
            "Baseline absolute error\nunavailable",
        )
    else:
        baseline_render_path = (
            Path(str(baseline_summary["baseline_render_png"]))
            if baseline_summary.get("baseline_render_png")
            else None
        )
        baseline_diff_path = (
            Path(str(baseline_summary["baseline_diff_heatmap_png"]))
            if baseline_summary.get("baseline_diff_heatmap_png")
            else None
        )
        baseline_render = _label_panel(
            _resize_panel(baseline_render_path),
            (
                "Global 1024 model, ray-equivalent tile-view crop\n"
                f"PSNR={baseline_summary.get('baseline_psnr_db')} "
                f"SSIM={baseline_summary.get('baseline_ssim')} "
                f"LPIPS={baseline_summary.get('baseline_lpips')}"
            ),
        )
        baseline_diff = _label_panel(
            _resize_panel(baseline_diff_path),
            "Global baseline absolute RGB error",
        )

    if route_summary is None:
        tile_render = _label_panel(
            _resize_panel(None),
            "Projected-C64-only tile flow\nfailed",
        )
        tile_diff = _label_panel(
            _resize_panel(None),
            "Tile absolute error\nunavailable",
        )
    else:
        render_path = (
            Path(str(route_summary["render_png"]))
            if route_summary.get("render_png")
            else None
        )
        diff_path = (
            Path(str(route_summary["diff_heatmap_png"]))
            if route_summary.get("diff_heatmap_png")
            else None
        )
        tile_render = _label_panel(
            _resize_panel(render_path),
            (
                "Projected-global C64 only -> shape1024 -> texture1024\n"
                f"C64={route_summary.get('tile_c64_tokens')}\n"
                f"PSNR={route_summary.get('psnr_db')} "
                f"SSIM={route_summary.get('ssim')} "
                f"LPIPS={route_summary.get('lpips')}\n"
                f"gain: dPSNR={route_summary.get('psnr_gain_db')} "
                f"dSSIM={route_summary.get('ssim_gain')} "
                f"LPIPS-reduction={route_summary.get('lpips_reduction')}"
            ),
        )
        tile_diff = _label_panel(
            _resize_panel(diff_path),
            "Projected-C64-only tile absolute RGB error",
        )

    panels = [
        reference,
        baseline_render,
        tile_render,
        support,
        baseline_diff,
        tile_diff,
    ]
    size = panels[0].width
    canvas = Image.new("RGB", (size * 3, size * 2), (18, 18, 18))
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % 3) * size, (index // 3) * size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return str(output_path)

def _evaluate_baseline_tile_records(
    *,
    records: List[Dict[str, Any]],
    args: argparse.Namespace,
    lpips_evaluator: Optional[_LPIPSEvaluator] = None,
) -> None:
    comparable = [
        row
        for row in records
        if row.get("reference_png")
        and row.get("support_overlay_png")
        and row.get("comparison_png")
    ]
    if not comparable:
        return

    metric_candidates = [
        row for row in comparable if row.get("baseline_render_png")
    ]
    owned_lpips_evaluator: Optional[_LPIPSEvaluator] = None
    if (
        metric_candidates
        and not bool(args.skip_lpips)
        and lpips_evaluator is None
    ):
        device_name = str(args.metric_device)
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        owned_lpips_evaluator = _LPIPSEvaluator(
            str(args.lpips_net),
            torch.device(device_name),
        )
        lpips_evaluator = owned_lpips_evaluator

    try:
        for row in metric_candidates:
            reference = _image_to_metric_tensor(
                Path(str(row["reference_png"])),
                resolution=int(args.metric_resolution),
            )
            baseline = _image_to_metric_tensor(
                Path(str(row["baseline_render_png"])),
                resolution=int(args.metric_resolution),
            )
            row["baseline_psnr_db"] = _psnr_metric(reference, baseline)
            row["baseline_ssim"] = _ssim_metric(reference, baseline)
            row["baseline_lpips"] = (
                None
                if lpips_evaluator is None
                else lpips_evaluator.evaluate(reference, baseline)
            )

            row["psnr_gain_db"] = (
                None
                if row.get("psnr_db") is None
                else float(row["psnr_db"]) - float(row["baseline_psnr_db"])
            )
            row["ssim_gain"] = (
                None
                if row.get("ssim") is None
                else float(row["ssim"]) - float(row["baseline_ssim"])
            )
            row["lpips_reduction"] = (
                None
                if row.get("lpips") is None or row.get("baseline_lpips") is None
                else float(row["baseline_lpips"]) - float(row["lpips"])
            )
            print(
                f"[baseline-tile] tile={int(row['tile_id']):02d} "
                f"baseline_PSNR={row['baseline_psnr_db']:.4f} "
                f"tile_PSNR={row.get('psnr_db')} "
                f"gain={row.get('psnr_gain_db')}"
            )

        for row in comparable:
            route_summary = row if row.get("render_png") else None
            baseline_summary = row if row.get("baseline_render_png") else None
            row["comparison_png"] = _save_tile_comparison_sheet(
                reference_path=Path(str(row["reference_png"])),
                support_overlay_path=Path(str(row["support_overlay_png"])),
                route_summary=route_summary,
                baseline_summary=baseline_summary,
                projected_c64_token_count=int(row.get("tile_projective_c64_tokens") or 0),
                outside_fraction=float(row.get("hard_outside_fraction") or 0.0),
                output_path=Path(str(row["comparison_png"])),
            )
            _atomic_json(Path(str(row["tile_dir"])) / "summary.json", row)
    finally:
        if owned_lpips_evaluator is not None:
            owned_lpips_evaluator.model.cpu()
            del owned_lpips_evaluator
            _empty_cuda_cache()


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
        cell_w = 760
        cell_h = 760
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
        path = output_dir / f"all_tiles_baseline_vs_tile_{start // per_page:02d}.png"
        canvas.save(path)
        outputs.append(str(path))
    return outputs


def _quantize_global_c1024_to_c128(
    coords1024: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Quantize dense decoder points to the fixed global C128 support.

    Pixal3D's 2048 route uses half-voxel binning onto the 128 lattice.  The
    returned inverse index preserves the C1024-source -> C128-token relation.
    """
    if coords1024.ndim != 2 or coords1024.shape[1] != 4:
        raise ValueError(
            f"global C1024 coordinates must be [N,4], got {tuple(coords1024.shape)}"
        )
    if torch.any(coords1024[:, 0] != 0):
        raise ValueError("only batch zero is supported")
    xyz1024 = coords1024[:, 1:4].to(torch.float32)
    valid = (
        (xyz1024 >= 0)
        & (xyz1024 < GRID_GLOBAL_UPSAMPLED)
    ).all(dim=1)
    if not bool(valid.all().item()):
        raise RuntimeError("global C1024 support contains out-of-range coordinates")
    xyz128 = torch.floor(
        (xyz1024 + 0.5)
        / float(GRID_GLOBAL_UPSAMPLED)
        * float(GRID_FINAL_2048)
    ).to(torch.int32)
    coords128_per_source = torch.cat(
        [coords1024[:, :1].to(torch.int32), xyz128],
        dim=1,
    )
    if bool(
        (
            (coords128_per_source[:, 1:] < 0)
            | (coords128_per_source[:, 1:] >= GRID_FINAL_2048)
        ).any().item()
    ):
        raise RuntimeError("C1024 -> C128 quantization produced invalid indices")
    coords128, source_to_global = torch.unique(
        coords128_per_source,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    source_counts = torch.bincount(
        source_to_global,
        minlength=coords128.shape[0],
    )
    stats = {
        "quantization": (
            "floor((c1024_xyz + 0.5) / 1024 * 128), followed by unique"
        ),
        "source_c1024_points": int(coords1024.shape[0]),
        "global_c128_tokens": int(coords128.shape[0]),
        "merged_source_rows": int(coords1024.shape[0] - coords128.shape[0]),
        "sources_per_token_min": int(source_counts.min().item()),
        "sources_per_token_mean": float(source_counts.float().mean().item()),
        "sources_per_token_max": int(source_counts.max().item()),
    }
    return coords128, source_to_global.to(torch.long), stats


def _pack_proj_condition_cpu(
    condition: Mapping[str, Mapping[str, Any]],
    *,
    expected_coords: torch.Tensor,
    name: str,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Offload one token-aligned projected condition between flow steps."""
    packed: Dict[str, Dict[str, torch.Tensor]] = {}
    for branch_name in ("cond", "neg_cond"):
        if branch_name not in condition:
            raise KeyError(f"{name}: condition is missing {branch_name!r}")
        branch = condition[branch_name]
        global_value = branch.get("global")
        projected = branch.get("proj")
        if not isinstance(global_value, torch.Tensor):
            raise TypeError(f"{name}.{branch_name}.global must be a Tensor")
        if not isinstance(projected, SparseTensor):
            raise TypeError(f"{name}.{branch_name}.proj must be a SparseTensor")
        if not torch.equal(projected.coords, expected_coords):
            raise RuntimeError(f"{name}.{branch_name}: projected coords changed order")
        packed[branch_name] = {
            "global": global_value.detach().to(device="cpu", copy=True),
            "proj": projected.feats.detach().to(device="cpu", copy=True),
        }
    return packed


def _materialize_proj_condition(
    packed: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    coords: torch.Tensor,
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for branch_name in ("cond", "neg_cond"):
        branch = packed[branch_name]
        projected = branch["proj"].to(device=device)
        if projected.shape[0] != coords.shape[0]:
            raise RuntimeError(
                f"{branch_name} projected feature/token count mismatch"
            )
        output[branch_name] = {
            "global": branch["global"].to(device=device),
            "proj": SparseTensor(feats=projected, coords=coords),
        }
    return output


def _tent_weights_from_tile_uv(
    uv_tile: torch.Tensor,
    *,
    width: int,
    height: int,
    mode: str,
) -> torch.Tensor:
    if mode == "uniform":
        return torch.ones(
            uv_tile.shape[0],
            device=uv_tile.device,
            dtype=torch.float32,
        )
    if mode != "tent":
        raise ValueError("tile weight mode must be 'tent' or 'uniform'")
    u = uv_tile[:, 0].to(torch.float32) / float(width)
    v = uv_tile[:, 1].to(torch.float32) / float(height)
    wx = (1.0 - (2.0 * u - 1.0).abs()).clamp_min(1e-3)
    wy = (1.0 - (2.0 * v - 1.0).abs()).clamp_min(1e-3)
    return wx * wy


def _build_tile_transport(
    *,
    reference: Image.Image,
    global_coords128: torch.Tensor,
    selected_global_token_rows: torch.Tensor,
    global_camera: Mapping[str, float],
    transform: TileCameraTransform,
    output_dir: Path,
    boundary_epsilon: float,
    tile_weight_mode: str,
) -> TileTransport:
    """Project selected global C128 tokens and quantize them to local C64."""
    if global_coords128.ndim != 2 or global_coords128.shape[1] != 4:
        raise ValueError(
            "global C128 coordinates must be [N,4], got "
            f"{tuple(global_coords128.shape)}"
        )
    if bool((global_coords128[:, 0] != 0).any().item()):
        raise ValueError("tile transport supports only batch-zero global C128")
    if bool(
        (
            (global_coords128[:, 1:] < 0)
            | (global_coords128[:, 1:] >= GRID_FINAL_2048)
        ).any().item()
    ):
        raise ValueError("global C128 coordinates are outside the 128 lattice")
    selected_global_token_rows = selected_global_token_rows.to(
        device=global_coords128.device,
        dtype=torch.long,
    )
    if selected_global_token_rows.numel() == 0:
        raise ValueError("selected global C128 token rows are empty")
    if (
        int(selected_global_token_rows.min().item()) < 0
        or int(selected_global_token_rows.max().item())
        >= int(global_coords128.shape[0])
    ):
        raise IndexError("selected global C128 token row is out of range")
    if torch.unique(selected_global_token_rows).numel() != (
        selected_global_token_rows.numel()
    ):
        raise ValueError("selected global C128 token rows must be unique")
    selected_coords128 = global_coords128.index_select(
        0,
        selected_global_token_rows,
    )
    q_global = _endpoint_indices_to_q(
        selected_coords128[:, 1:4],
        GRID_FINAL_2048,
    ).to(global_coords128.device)
    q_local, uv_tile, _, transform_stats = _global_q_to_centered_tile_q(
        q_global,
        global_camera=global_camera,
        transform=transform,
    )
    _, kept, quant_stats = _quantize_local_q_without_geometry_clip(
        q_local,
        resolution=GRID_SHAPE_1024,
        lattice_name="projected-global local C64",
        epsilon=float(boundary_epsilon),
    )
    kept_global_token_rows = selected_global_token_rows[kept]
    q_global_kept = q_global[kept]
    q_local_kept = q_local[kept]
    uv_kept = uv_tile[kept]

    local_xyz = _q_to_endpoint_indices(q_local_kept, GRID_SHAPE_1024)
    local_per_global_token_coords = torch.cat(
        [
            torch.zeros(
                (local_xyz.shape[0], 1),
                device=local_xyz.device,
                dtype=torch.int32,
            ),
            local_xyz,
        ],
        dim=1,
    )
    local_coords, global_token_to_local = torch.unique(
        local_per_global_token_coords,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    global_token_to_local = global_token_to_local.to(torch.long)

    # Each selected source is already one unique global C128 flow token.  Keep
    # an explicit unique-pair step so both transport directions share the same
    # edge representation even if the upstream support contract later changes.
    pairs_per_global_token = torch.stack(
        [kept_global_token_rows, global_token_to_local],
        dim=1,
    )
    unique_pairs, global_token_to_edge = torch.unique(
        pairs_per_global_token,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    if unique_pairs.shape[0] != kept_global_token_rows.shape[0]:
        raise RuntimeError(
            "direct global-C128 transport produced duplicate token/local pairs"
        )
    projected_token_weight = _tent_weights_from_tile_uv(
        uv_kept,
        width=transform.output_width,
        height=transform.output_height,
        mode=tile_weight_mode,
    )
    edge_weight_sum = torch.zeros(
        unique_pairs.shape[0],
        device=global_coords128.device,
        dtype=torch.float32,
    )
    edge_token_count = torch.zeros_like(edge_weight_sum)
    edge_weight_sum.index_add_(
        0,
        global_token_to_edge,
        projected_token_weight,
    )
    edge_token_count.index_add_(
        0,
        global_token_to_edge,
        torch.ones_like(projected_token_weight),
    )
    edge_weight = edge_weight_sum / edge_token_count.clamp_min(1.0)
    edge_global = unique_pairs[:, 0].to(torch.long)
    edge_local = unique_pairs[:, 1].to(torch.long)

    local_weight_sum = torch.zeros(
        local_coords.shape[0],
        device=global_coords128.device,
        dtype=torch.float32,
    )
    local_weight_sum.index_add_(0, edge_local, edge_weight)
    edge_forward_weight = edge_weight / local_weight_sum.index_select(
        0, edge_local
    ).clamp_min(torch.finfo(torch.float32).eps)
    forward_check = torch.zeros_like(local_weight_sum)
    forward_check.index_add_(0, edge_local, edge_forward_weight)
    forward_normalization_error = float(
        (forward_check - 1.0).abs().max().item()
    )
    if forward_normalization_error > 1e-5:
        raise RuntimeError(
            "global-to-local transport weights do not sum to one: "
            f"max error={forward_normalization_error:.8e}"
        )

    q_quantized = _endpoint_indices_to_q(
        local_coords[:, 1:4],
        GRID_SHAPE_1024,
    ).to(global_coords128.device)
    unique_uv, _, unique_valid = _project_grid_coords_to_tile_uv(
        local_coords,
        resolution=GRID_SHAPE_1024,
        transform=transform,
    )
    quantized_points_per_global_token = _camera_q_to_points(
        _endpoint_indices_to_q(local_xyz, GRID_SHAPE_1024).to(
            global_coords128.device
        ),
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
    )
    uv_quantized_per_global_token, _, valid_quantized = (
        _project_points_with_intrinsics(
            quantized_points_per_global_token,
            fx=float(transform.fx),
            fy=float(transform.fy),
            cx=float(transform.cx),
            cy=float(transform.cy),
        )
    )
    uv_quantized_per_global_token = uv_quantized_per_global_token[0]
    valid_quantized = valid_quantized[0]
    pixel_error = torch.linalg.vector_norm(
        uv_quantized_per_global_token - uv_kept,
        dim=1,
    )

    _, global_degree = torch.unique(edge_global, return_counts=True)
    _, local_degree = torch.unique(edge_local, return_counts=True)
    stats = {
        "tile_id": int(transform.tile_id),
        "box": list(transform.box),
        "support_mode": "project_global_c128_tokens_then_quantize_local_c64",
        "selected_global_c128_tokens": int(
            selected_global_token_rows.shape[0]
        ),
        "kept_global_c128_tokens": int(kept_global_token_rows.shape[0]),
        "global_c128_tokens_touched": int(torch.unique(edge_global).shape[0]),
        "local_c64_tokens": int(local_coords.shape[0]),
        "deduplicated_transport_edges": int(unique_pairs.shape[0]),
        "duplicate_global_token_pairs_removed": int(
            kept_global_token_rows.shape[0] - unique_pairs.shape[0]
        ),
        "transport_normalization": {
            "global_to_local": (
                "projected global C128 edge tent weights normalized per "
                "local C64 token"
            ),
            "local_to_global": (
                "edge tent weights normalized after accumulation over all "
                "overlapping tiles"
            ),
            "global_to_local_max_sum_error": forward_normalization_error,
        },
        "tile_weight_mode": tile_weight_mode,
        "global_edge_degree_mean": float(global_degree.float().mean().item()),
        "global_edge_degree_max": int(global_degree.max().item()),
        "local_edge_degree_mean": float(local_degree.float().mean().item()),
        "local_edge_degree_max": int(local_degree.max().item()),
        "pixel_quantization_error_mean": float(pixel_error.mean().item()),
        "pixel_quantization_error_p95": float(
            torch.quantile(pixel_error, 0.95).item()
        ),
        "pixel_quantization_error_max": float(pixel_error.max().item()),
        "transform_stats": transform_stats,
        "local_quantization_stats": quant_stats,
        "tile_camera": asdict(transform),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _draw_uv_points(
        reference,
        uv_kept,
        q_global_kept[:, 2],
        output_dir / "continuous_global_c128_projection.png",
        "global C128 tokens transformed to the centered tile camera",
    )
    _draw_uv_points(
        reference,
        unique_uv[unique_valid],
        q_quantized[:, 2][unique_valid],
        output_dir / "local_c64_support_overlay.png",
        "unique local C64 support used only for temporary velocity prediction",
    )
    _save_density_image(
        unique_uv[unique_valid],
        output_dir / "local_c64_support_density.png",
        resolution=IMAGE_FLOW,
    )
    _save_quantization_error_image(
        reference,
        uv_kept[valid_quantized],
        uv_quantized_per_global_token[valid_quantized],
        output_dir / "local_c64_quantization_error.png",
    )
    _atomic_json(output_dir / "transport_stats.json", stats)
    torch.save(
        {
            "selected_global_c128_token_rows": (
                kept_global_token_rows.detach().cpu()
            ),
            "selected_coords_global_c128": global_coords128.index_select(
                0, kept_global_token_rows
            ).detach().cpu(),
            "global_c128_token_to_local_c64_token": (
                global_token_to_local.detach().cpu()
            ),
            "local_coords_c64": local_coords.detach().cpu(),
            "edge_global_c128": edge_global.detach().cpu(),
            "edge_local_c64": edge_local.detach().cpu(),
            "edge_weight": edge_weight.detach().cpu(),
            "edge_forward_weight": edge_forward_weight.detach().cpu(),
            "q_global": q_global_kept.detach().cpu(),
            "q_local": q_local_kept.detach().cpu(),
            "uv_tile": uv_kept.detach().cpu(),
            "tile_camera": asdict(transform),
        },
        output_dir / "transport_correspondence.pt",
    )
    return TileTransport(
        tile_id=int(transform.tile_id),
        box=tuple(int(v) for v in transform.box),
        transform=transform,
        local_coords=local_coords.detach().cpu(),
        edge_global=edge_global.detach().cpu(),
        edge_local=edge_local.detach().cpu(),
        edge_weight=edge_weight.detach().cpu(),
        edge_forward_weight=edge_forward_weight.detach().cpu(),
        global_token_rows=kept_global_token_rows.detach().cpu(),
        global_token_to_local=global_token_to_local.detach().cpu(),
        stats=stats,
    )


def _prepare_all_tile_transports(
    *,
    args: argparse.Namespace,
    image_4096: Image.Image,
    global_coords128: torch.Tensor,
    uv_full_4096: torch.Tensor,
    finite_global: torch.Tensor,
    global_camera: Mapping[str, float],
    output_dir: Path,
) -> Tuple[List[TileTransport], List[Dict[str, Any]]]:
    boxes = _tile_layout(
        IMAGE_CANONICAL,
        int(args.tile_size),
        int(args.tile_stride),
    )
    selected_ids = _parse_tile_ids(args.tile_ids)
    if selected_ids is not None:
        invalid = sorted(value for value in selected_ids if value not in range(len(boxes)))
        if invalid:
            raise ValueError(
                f"invalid tile ids {invalid}; valid range is 0-{len(boxes) - 1}"
            )

    transports: List[TileTransport] = []
    records: List[Dict[str, Any]] = []
    attempted = 0
    for tile_id, box in enumerate(boxes):
        if selected_ids is not None and tile_id not in selected_ids:
            continue
        if args.max_tiles is not None and attempted >= int(args.max_tiles):
            break
        attempted += 1
        tile_dir = output_dir / "tiles" / f"tile_{tile_id:02d}"
        reference = _prepare_tile_reference(image_4096, box, tile_dir)
        global_token_rows = _rows_inside_tile(
            uv_full_4096,
            finite_global,
            box,
        )
        transform = _derive_tile_camera(
            tile_id=tile_id,
            box=box,
            global_camera=global_camera,
            extend_pixel=int(args.extend_pixel),
            offaxis_shift_y_sign=int(args.offaxis_shift_y_sign),
        )
        _atomic_json(tile_dir / "tile_camera.json", asdict(transform))
        if global_token_rows.numel() == 0:
            record = {
                "status": "skipped",
                "tile_id": tile_id,
                "box": list(box),
                "reason": "no projected global C128 token in tile",
                "selected_global_c128_tokens": 0,
                "local_c64_tokens": 0,
            }
            records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            continue
        try:
            transport = _build_tile_transport(
                reference=reference,
                global_coords128=global_coords128,
                selected_global_token_rows=global_token_rows,
                global_camera=global_camera,
                transform=transform,
                output_dir=tile_dir / "transport",
                boundary_epsilon=float(args.boundary_epsilon),
                tile_weight_mode=str(args.tile_weight),
            )
            outside_fraction = float(
                transport.stats["local_quantization_stats"][
                    "hard_outside_fraction"
                ]
            )
            if outside_fraction > float(args.max_outside_fraction):
                raise RuntimeError(
                    f"outside fraction {outside_fraction:.6f} exceeds "
                    f"{float(args.max_outside_fraction):.6f}"
                )
            if transport.local_coords.shape[0] < int(args.min_tile_tokens):
                record = {
                    "status": "skipped",
                    "tile_id": tile_id,
                    "box": list(box),
                    "reason": "local C64 support below --min-tile-tokens",
                    **transport.stats,
                }
                records.append(record)
                _atomic_json(tile_dir / "summary.json", record)
                continue
            transports.append(transport)
            record = {
                "status": "active",
                **transport.stats,
                "reference_png": str(tile_dir / "reference_tile.png"),
                "transport_path": str(
                    tile_dir / "transport" / "transport_correspondence.pt"
                ),
            }
            records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            print(
                f"[transport] tile={tile_id:02d} "
                f"projected_global_C128={transport.global_token_rows.shape[0]:,} "
                f"local_C64={transport.local_coords.shape[0]:,} "
                f"edges={transport.edge_global.shape[0]:,}"
            )
        except Exception as exc:
            record = {
                "status": "failed",
                "tile_id": tile_id,
                "box": list(box),
                "selected_global_c128_tokens": int(
                    global_token_rows.shape[0]
                ),
                "reason": f"{type(exc).__name__}: {exc}",
            }
            records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            print(f"[transport-error] tile={tile_id:02d}: {record['reason']}")
    if not transports:
        raise RuntimeError("no tile produced a usable local C64 transport")
    return transports, records


def _prepare_stage_conditions(
    *,
    pipeline: Any,
    stage_name: str,
    image_1024: Image.Image,
    image_4096: Image.Image,
    global_coords: torch.Tensor,
    global_camera: Mapping[str, float],
    transports: Sequence[TileTransport],
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Any]]:
    if stage_name == "shape":
        image_cond_model = pipeline.image_cond_model_shape_1024
    elif stage_name == "texture":
        image_cond_model = pipeline.image_cond_model_tex_1024
    else:
        raise ValueError("stage_name must be 'shape' or 'texture'")

    started = time.perf_counter()
    global_condition = pipeline.get_proj_cond_shape(
        image_cond_model,
        [image_1024.convert("RGB")],
        global_coords,
        camera_angle_x=float(global_camera["camera_angle_x"]),
        distance=float(global_camera["distance"]),
        mesh_scale=float(global_camera["mesh_scale"]),
        grid_resolution_override=GRID_FINAL_2048,
    )
    global_packed = _pack_proj_condition_cpu(
        global_condition,
        expected_coords=global_coords,
        name=f"global_{stage_name}_C128",
    )
    del global_condition
    _empty_cuda_cache()

    tile_records: List[Dict[str, Any]] = []
    for transport in transports:
        tile_started = time.perf_counter()
        local_coords = transport.local_coords.to(
            device=pipeline.device,
            dtype=torch.int32,
        )
        tile_image = _composite_on_black(
            image_4096.crop(transport.box)
        )
        if tile_image.size != (IMAGE_FLOW, IMAGE_FLOW):
            tile_image = tile_image.resize(
                (IMAGE_FLOW, IMAGE_FLOW),
                Image.Resampling.LANCZOS,
            )
        camera = transport.transform
        local_condition = pipeline.get_proj_cond_shape(
            image_cond_model,
            [tile_image],
            local_coords,
            camera_angle_x=float(camera.camera_angle_x),
            distance=float(camera.distance),
            mesh_scale=float(camera.mesh_scale),
            grid_resolution_override=GRID_SHAPE_1024,
        )
        transport.condition_cpu = _pack_proj_condition_cpu(
            local_condition,
            expected_coords=local_coords,
            name=f"tile_{transport.tile_id:02d}_{stage_name}_C64",
        )
        elapsed = time.perf_counter() - tile_started
        tile_records.append(
            {
                "tile_id": int(transport.tile_id),
                "local_c64_tokens": int(local_coords.shape[0]),
                "seconds": float(elapsed),
            }
        )
        del local_coords, local_condition, tile_image
        _empty_cuda_cache()
        print(
            f"[condition-{stage_name}] tile={transport.tile_id:02d} "
            f"local_C64={transport.local_coords.shape[0]:,} seconds={elapsed:.3f}"
        )
    return global_packed, {
        "stage": stage_name,
        "global_grid": GRID_FINAL_2048,
        "local_grid": GRID_SHAPE_1024,
        "active_tiles": len(transports),
        "total_seconds": float(time.perf_counter() - started),
        "tiles": tile_records,
    }


def _transport_global_features_to_local(
    global_features: torch.Tensor,
    transport: TileTransport,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    device = global_features.device
    edge_global = transport.edge_global.to(device=device, dtype=torch.long)
    edge_local = transport.edge_local.to(device=device, dtype=torch.long)
    forward_weight = transport.edge_forward_weight.to(
        device=device,
        dtype=torch.float32,
    )
    local = torch.zeros(
        (transport.local_coords.shape[0], global_features.shape[1]),
        device=device,
        dtype=torch.float32,
    )
    local.index_add_(
        0,
        edge_local,
        global_features.index_select(0, edge_global).to(torch.float32)
        * forward_weight[:, None],
    )
    return local, (edge_global, edge_local, forward_weight)


def _local_residual_lambda(
    step_index: int,
    steps: int,
    *,
    start_fraction: float,
    maximum: float,
) -> float:
    if maximum == 0.0:
        return 0.0
    progress = (
        1.0
        if steps <= 1
        else float(step_index) / float(steps - 1)
    )
    if progress < start_fraction:
        return 0.0
    ramp = (progress - start_fraction) / max(1.0 - start_fraction, 1e-8)
    ramp = min(max(ramp, 0.0), 1.0)
    return float(maximum) * (0.5 - 0.5 * math.cos(math.pi * ramp))


@torch.no_grad()
def _run_single_global_c128_flow(
    *,
    pipeline: Any,
    stage_name: str,
    model: torch.nn.Module,
    sampler: Any,
    global_coords: torch.Tensor,
    global_condition_cpu: Mapping[str, Mapping[str, torch.Tensor]],
    transports: Sequence[TileTransport],
    params: Mapping[str, Any],
    seed: int,
    local_start_fraction: float,
    local_max_weight: float,
    concat_global: Optional[SparseTensor] = None,
) -> Tuple[SparseTensor, float, Dict[str, Any]]:
    """Run one state and one Euler update per step on global C128 only."""
    if concat_global is not None and not torch.equal(
        concat_global.coords, global_coords
    ):
        raise RuntimeError(f"{stage_name}: global concat coords are misaligned")
    input_channels = int(model.in_channels)
    latent_channels = input_channels
    if concat_global is not None:
        latent_channels -= int(concat_global.feats.shape[1])
    if latent_channels <= 0:
        raise RuntimeError(
            f"{stage_name}: invalid latent channel count {latent_channels}"
        )
    state = SparseTensor(
        feats=_randn(
            global_coords.shape[0],
            latent_channels,
            device=pipeline.device,
            seed=int(seed),
        ),
        coords=global_coords,
    )
    # This is the only noise/state initialization for the stage.  Tile states
    # below are deterministic transports of this tensor.
    t_seq = sampler.timestep_schedule(
        int(params["steps"]),
        float(params.get("rescale_t", 1.0)),
    )
    prediction_kwargs = {
        key: value
        for key, value in params.items()
        if key
        not in {
            "steps",
            "rescale_t",
            "verbose",
            "tqdm_desc",
            "record_trajectory",
            "trajectory_device",
            "return_model_history",
        }
    }
    global_condition = _materialize_proj_condition(
        global_condition_cpu,
        coords=global_coords,
        device=pipeline.device,
    )
    local_concat_cpu: Dict[int, torch.Tensor] = {}
    if concat_global is not None:
        for transport in transports:
            local_concat, _ = _transport_global_features_to_local(
                concat_global.feats,
                transport,
            )
            local_concat_cpu[transport.tile_id] = local_concat.detach().cpu()
            del local_concat

    # Static normalized back-transport denominator, including overlap weights.
    transport_weight_sum = torch.zeros(
        (global_coords.shape[0], 1),
        device=pipeline.device,
        dtype=torch.float32,
    )
    tile_coverage_count = torch.zeros(
        global_coords.shape[0],
        device=pipeline.device,
        dtype=torch.int32,
    )
    for transport in transports:
        edge_global = transport.edge_global.to(
            device=pipeline.device,
            dtype=torch.long,
        )
        edge_weight = transport.edge_weight.to(
            device=pipeline.device,
            dtype=torch.float32,
        )
        transport_weight_sum.index_add_(0, edge_global, edge_weight[:, None])
        unique_global = torch.unique(edge_global)
        tile_coverage_count.index_add_(
            0,
            unique_global,
            torch.ones_like(unique_global, dtype=torch.int32),
        )
        del edge_global, edge_weight, unique_global
    covered = transport_weight_sum[:, 0] > 0
    covered_count = int(covered.sum().item())
    overlap_count = int((tile_coverage_count > 1).sum().item())

    if pipeline.low_vram:
        model.to(pipeline.device)
    step_records: List[Dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for step_index, (t_value, t_next) in enumerate(zip(t_seq[:-1], t_seq[1:])):
            step_started = time.perf_counter()
            dt = float(t_value - t_next)
            _, _, global_velocity = sampler._get_model_prediction(
                model,
                state,
                float(t_value),
                global_condition["cond"],
                neg_cond=global_condition["neg_cond"],
                concat_cond=concat_global,
                **prediction_kwargs,
            )
            if not torch.equal(global_velocity.coords, global_coords):
                raise RuntimeError(f"{stage_name}: global velocity changed coords")

            local_weight = _local_residual_lambda(
                step_index,
                len(t_seq) - 1,
                start_fraction=float(local_start_fraction),
                maximum=float(local_max_weight),
            )
            residual_sum = torch.zeros(
                state.feats.shape,
                device=pipeline.device,
                dtype=torch.float32,
            )
            tile_calls = 0
            local_velocity_norm_sum = 0.0
            local_residual_norm_sum = 0.0
            if local_weight > 0.0:
                for transport in transports:
                    if transport.condition_cpu is None:
                        raise RuntimeError(
                            f"tile {transport.tile_id} has no {stage_name} condition"
                        )
                    local_coords = transport.local_coords.to(
                        device=pipeline.device,
                        dtype=torch.int32,
                    )
                    local_state_feats, indices = (
                        _transport_global_features_to_local(
                            state.feats,
                            transport,
                        )
                    )
                    edge_global, edge_local, _ = indices
                    local_state = SparseTensor(
                        feats=local_state_feats.to(dtype=state.feats.dtype),
                        coords=local_coords,
                    )
                    local_condition = _materialize_proj_condition(
                        transport.condition_cpu,
                        coords=local_coords,
                        device=pipeline.device,
                    )
                    local_concat = None
                    if concat_global is not None:
                        local_concat = SparseTensor(
                            feats=local_concat_cpu[transport.tile_id].to(
                                device=pipeline.device,
                                dtype=concat_global.feats.dtype,
                            ),
                            coords=local_coords,
                        )
                    _, _, local_velocity = sampler._get_model_prediction(
                        model,
                        local_state,
                        float(t_value),
                        local_condition["cond"],
                        neg_cond=local_condition["neg_cond"],
                        concat_cond=local_concat,
                        **prediction_kwargs,
                    )
                    mapped_global_velocity, _ = (
                        _transport_global_features_to_local(
                            global_velocity.feats,
                            transport,
                        )
                    )
                    local_residual = (
                        local_velocity.feats.to(torch.float32)
                        - mapped_global_velocity
                    )
                    edge_weight = transport.edge_weight.to(
                        device=pipeline.device,
                        dtype=torch.float32,
                    )
                    residual_sum.index_add_(
                        0,
                        edge_global,
                        local_residual.index_select(0, edge_local)
                        * edge_weight[:, None],
                    )
                    tile_calls += 1
                    local_velocity_norm_sum += float(
                        local_velocity.feats.float().norm().item()
                    )
                    local_residual_norm_sum += float(
                        local_residual.norm().item()
                    )
                    del (
                        local_coords,
                        local_state_feats,
                        indices,
                        edge_global,
                        edge_local,
                        local_state,
                        local_condition,
                        local_concat,
                        local_velocity,
                        mapped_global_velocity,
                        local_residual,
                        edge_weight,
                    )
                residual_global = (
                    residual_sum
                    / transport_weight_sum.clamp_min(
                        torch.finfo(torch.float32).eps
                    )
                )
                residual_global[~covered] = 0.0
            else:
                residual_global = residual_sum

            fused_velocity = (
                global_velocity.feats.to(torch.float32)
                + float(local_weight) * residual_global
            )
            state = state.replace(
                state.feats - dt * fused_velocity.to(dtype=state.feats.dtype)
            )
            if not torch.equal(state.coords, global_coords):
                raise RuntimeError(f"{stage_name}: Euler update changed coords")
            _sync_cuda()
            record = {
                "step_index": int(step_index),
                "t": float(t_value),
                "t_next": float(t_next),
                "dt": dt,
                "lambda_local_residual": float(local_weight),
                "global_velocity_norm": float(
                    global_velocity.feats.float().norm().item()
                ),
                "transported_residual_norm": float(
                    residual_global.norm().item()
                ),
                "fused_velocity_norm": float(fused_velocity.norm().item()),
                "local_model_calls": int(tile_calls),
                "mean_local_velocity_norm": (
                    None
                    if tile_calls == 0
                    else float(local_velocity_norm_sum / tile_calls)
                ),
                "mean_local_residual_norm": (
                    None
                    if tile_calls == 0
                    else float(local_residual_norm_sum / tile_calls)
                ),
                "seconds": float(time.perf_counter() - step_started),
            }
            step_records.append(record)
            print(
                f"[global-C128-{stage_name}] step={step_index:02d} "
                f"t={t_value:.8f}->{t_next:.8f} lambda={local_weight:.6f} "
                f"tiles={tile_calls} global_v={record['global_velocity_norm']:.4f} "
                f"residual={record['transported_residual_norm']:.4f} "
                f"seconds={record['seconds']:.3f}"
            )
            del (
                global_velocity,
                residual_sum,
                residual_global,
                fused_velocity,
            )
    finally:
        if pipeline.low_vram:
            model.cpu()
        _empty_cuda_cache()

    elapsed = time.perf_counter() - started
    diagnostics = {
        "stage": stage_name,
        "state_coordinate_system": "one fixed global C128 sparse support",
        "initial_noise_count": 1,
        "tile_noise_count": 0,
        "tile_trajectory_count": 0,
        "global_updates_per_step": 1,
        "global_tokens": int(global_coords.shape[0]),
        "active_tiles": int(len(transports)),
        "covered_global_tokens": covered_count,
        "covered_global_fraction": float(covered.float().mean().item()),
        "overlap_global_tokens": overlap_count,
        "overlap_global_fraction": float(
            (tile_coverage_count > 1).float().mean().item()
        ),
        "local_schedule": {
            "start_fraction": float(local_start_fraction),
            "maximum": float(local_max_weight),
            "ramp": "half-cosine",
        },
        "steps": step_records,
        "elapsed_seconds": float(elapsed),
    }
    for transport in transports:
        transport.condition_cpu = None
    del global_condition, local_concat_cpu
    _empty_cuda_cache()
    return state, elapsed, diagnostics


def _load_sparse_final_latents(
    path: Path,
    *,
    device: torch.device,
) -> Tuple[SparseTensor, SparseTensor, Dict[str, Any]]:
    """Restore final denormalized C128 latents after a render-only restart."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    required = ("coords", "shape_denorm_feats", "texture_denorm_feats")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(
            f"{path} is missing final latent fields: {', '.join(missing)}"
        )
    coords = torch.as_tensor(
        payload["coords"],
        dtype=torch.int32,
        device=device,
    )
    shape_feats = torch.as_tensor(
        payload["shape_denorm_feats"],
        device=device,
    )
    texture_feats = torch.as_tensor(
        payload["texture_denorm_feats"],
        device=device,
    )
    if (
        coords.ndim != 2
        or coords.shape[1] != 4
        or shape_feats.ndim != 2
        or texture_feats.ndim != 2
        or shape_feats.shape[0] != coords.shape[0]
        or texture_feats.shape[0] != coords.shape[0]
    ):
        raise ValueError(f"{path}: malformed or misaligned sparse latent tensors")
    if not torch.isfinite(shape_feats).all() or not torch.isfinite(texture_feats).all():
        raise ValueError(f"{path}: final latent features contain non-finite values")
    return (
        SparseTensor(feats=shape_feats, coords=coords),
        SparseTensor(feats=texture_feats, coords=coords),
        dict(payload),
    )


def _decode_save_and_render_final(
    *,
    pipeline: Any,
    shape_denorm: SparseTensor,
    texture_denorm: SparseTensor,
    output_dir: Path,
    global_camera: Mapping[str, float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Decode, persist the full model, release generation VRAM, then render."""
    final_dir = output_dir / "final_global_2048"
    final_dir.mkdir(parents=True, exist_ok=True)
    original_point_exporter = getattr(
        pipeline,
        "export_tex_voxel_point_cloud",
        None,
    )
    if original_point_exporter is not None and not bool(
        args.export_tex_point_cloud
    ):
        pipeline.export_tex_voxel_point_cloud = lambda *unused_args, **unused_kwargs: None
    try:
        decoded, mesh = _decode_normal_mesh_with_ovoxel(
            pipeline=pipeline,
            shape_latent=shape_denorm,
            texture_latent=texture_denorm,
            label="Single-global-state 2048",
            resolution=DECODE_GLOBAL,
        )
    finally:
        if original_point_exporter is not None:
            pipeline.export_tex_voxel_point_cloud = original_point_exporter
    decoder_summary = {
        "resolution": DECODE_GLOBAL,
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "active_voxels": int(mesh.coords.shape[0]),
        "sample_type": type(mesh).__name__,
    }

    # Persist before any renderer call.  A CUDA rasterizer failure must never
    # discard the expensive generated model.
    mesh_checkpoint = final_dir / "mesh_with_ovoxel.pt"
    mesh_cpu = mesh.to("cpu")
    torch.save(mesh_cpu, mesh_checkpoint)
    print(
        f"[checkpoint] full decoded MeshWithVoxel saved before rendering: "
        f"{mesh_checkpoint}"
    )

    # Standard mode otherwise leaves all generation/conditioning models on
    # CUDA.  Move them off-device before allocating nvdiffrast buffers.
    pipeline.to(torch.device("cpu"))
    del decoded, mesh
    _empty_cuda_cache()
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(
            f"[render-vram] after generation unload: "
            f"free={free_bytes / 2**30:.2f} GiB "
            f"total={total_bytes / 2**30:.2f} GiB"
        )

    render_mesh = mesh_cpu.to("cuda")
    del mesh_cpu
    _empty_cuda_cache()
    face_chunk_size = int(args.render_face_chunk_size)
    face_count = int(render_mesh.faces.shape[0])
    face_chunk_count = (
        1
        if face_chunk_size <= 0
        else (face_count + face_chunk_size - 1) // face_chunk_size
    )
    render_safety = {
        "generation_models_unloaded_before_render": True,
        "geometry_modified": False,
        "full_mesh_vertices": int(render_mesh.vertices.shape[0]),
        "full_mesh_faces": face_count,
        "face_chunk_size": face_chunk_size,
        "face_chunk_count": int(face_chunk_count),
        "chunk_merge": (
            "each face chunk emits K depth-peel geometry layers; per-pixel "
            "global top-K depth merge precedes O-Voxel lookup, PBR shading, "
            "alpha compositing, and SSAO"
        ),
        "ssaa": int(args.render_ssaa),
        "peel_layers": int(args.render_peel_layers),
    }
    _atomic_json(final_dir / "render_safety.json", render_safety)

    envmap = load_envmap(str(args.envmap), device="cuda")
    try:
        metric_row = render_and_evaluate_mesh(
            render_mesh,
            camera_angle_x=float(global_camera["camera_angle_x"]),
            distance=float(global_camera["distance"]),
            output_dir=final_dir / "aligned_eval",
            reference_image=output_dir / "canonical_4096.png",
            resolution=int(args.render_resolution),
            metric_resolution=int(args.metric_resolution),
            envmap=envmap,
            envmap_name=str(args.envmap),
            ssaa=int(args.render_ssaa),
            peel_layers=int(args.render_peel_layers),
            face_chunk_size=face_chunk_size,
            use_envmap_bg=bool(args.use_envmap_bg),
            lpips_net=str(args.lpips_net),
            metric_device=str(args.metric_device),
            skip_lpips=bool(args.skip_lpips),
        )
    except Exception as exc:
        failure = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "mesh_checkpoint": str(mesh_checkpoint),
            "render_safety": render_safety,
            "recovery": (
                "the complete model is already saved; rerun with "
                "--resume-final-latents and a lower "
                "--render-face-chunk-size/--render-ssaa/--render-peel-layers"
            ),
        }
        _atomic_json(final_dir / "render_failure.json", failure)
        raise RuntimeError(
            "final rendering failed after the full model was safely saved at "
            f"{mesh_checkpoint}; see {final_dir / 'render_failure.json'}"
        ) from exc
    comparison = _save_extra_comparisons(
        Path(metric_row["original_png"]),
        Path(metric_row["render_png"]),
        final_dir / "comparisons",
    )
    del render_mesh, envmap
    _empty_cuda_cache()
    return {
        "decoder": decoder_summary,
        "mesh_checkpoint": str(mesh_checkpoint),
        "render_safety": render_safety,
        "render_and_metrics": {**metric_row, **comparison},
    }



def _run_legacy_independent_tile_baseline(args: argparse.Namespace) -> None:
    if int(args.tile_size) != DEFAULT_TILE_SIZE:
        raise ValueError(f"this test requires --tile-size={DEFAULT_TILE_SIZE}")
    if int(args.tile_stride) != DEFAULT_TILE_STRIDE:
        raise ValueError(f"this test requires --tile-stride={DEFAULT_TILE_STRIDE}")
    if args.cuda_device is not None:
        torch.cuda.set_device(int(args.cuda_device))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = init_pipeline(
        args.model_path,
        device="cuda",
        low_vram=bool(args.low_vram),
    )
    envmap = load_envmap(str(args.envmap), device="cuda")
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
    _atomic_json(output_dir / "canonical_metadata.json", canonical["metadata"])

    saved_camera_path = output_dir / "global_camera.json"
    if bool(args.resume_final_latents) and saved_camera_path.is_file():
        global_camera = json.loads(saved_camera_path.read_text(encoding="utf-8"))
        global_camera = {
            "camera_angle_x": float(global_camera["camera_angle_x"]),
            "distance": float(global_camera["distance"]),
            "mesh_scale": float(global_camera["mesh_scale"]),
        }
        print(f"[resume] reused saved global camera: {saved_camera_path}")
    else:
        global_camera = _estimate_camera(
            image_1024=image_1024,
            output_dir=output_dir,
            manual_fov=float(args.fov),
            mesh_scale=float(args.mesh_scale),
            extend_pixel=int(args.extend_pixel),
            image_resolution=int(args.camera_image_resolution),
            moge_model_path=args.moge_model_path,
        )
    _atomic_json(output_dir / "global_camera.json", global_camera)
    print(
        f"[global-camera] fov={global_camera['camera_angle_x']:.8f} "
        f"distance={global_camera['distance']:.8f} "
        f"mesh_scale={global_camera['mesh_scale']:.8f}"
    )

    params = _sampler_params(args, pipeline)
    print(
        "[global-baseline] ordinary 1024 route: "
        "SS -> shape512 -> learned C64 -> shape1024 -> texture1024 -> decode"
    )
    coords32, coords64, global_shape = _run_global_official_geometry_to_shape1024(
        pipeline=pipeline,
        image_512=image_512,
        image_1024=image_1024,
        camera=global_camera,
        params=params,
        seed=int(args.seed),
        max_tokens=int(args.max_num_tokens),
    )
    global_shape512_seconds = float(global_shape.shape512_seconds)
    global_shape1024_seconds = float(global_shape.shape1024_seconds)
    global_texture_norm, global_texture_denorm, global_texture_seconds = (
        _run_texture1024(
            pipeline=pipeline,
            image_1024=image_1024,
            coords64=coords64,
            camera=global_camera,
            shape_norm=global_shape.shape_norm,
            params=params,
            seed=int(args.seed) + 301,
            description="Global ordinary texture 1024",
        )
    )
    global_baseline_result = GlobalBaselineResult(
        shape_norm=global_shape.shape_norm,
        shape_denorm=global_shape.shape_denorm,
        texture_norm=global_texture_norm,
        texture_denorm=global_texture_denorm,
        global_c32_tokens=int(coords32.shape[0]),
        global_c64_tokens=int(coords64.shape[0]),
        shape512_seconds=float(global_shape.shape512_seconds),
        shape1024_seconds=float(global_shape.shape1024_seconds),
        texture1024_seconds=float(global_texture_seconds),
    )
    global_baseline_dir = output_dir / "global_baseline_1024"
    global_baseline_dir.mkdir(parents=True, exist_ok=True)
    global_meshes, global_mesh = _decode_normal_mesh_with_ovoxel(
        pipeline=pipeline,
        shape_latent=global_baseline_result.shape_denorm,
        texture_latent=global_baseline_result.texture_denorm,
        label="Global ordinary 1024 baseline",
    )
    global_decoder_metadata = {
        "decoder_vertices": int(global_mesh.vertices.shape[0]),
        "decoder_faces": int(global_mesh.faces.shape[0]),
        "active_voxels": int(global_mesh.coords.shape[0]),
        "sample_type": type(global_mesh).__name__,
        "renderer": "pixal3d.utils.render_utils.render_frames",
    }

    print("[global] one learned decoder subdivision: C64 latent -> dense C1024 support")
    coords128, upsample_stats = _learned_subdivide_shape1024_to_c1024(
        pipeline,
        global_shape.shape_denorm,
    )
    if coords128.shape[0] > int(args.max_num_tokens):
        raise RuntimeError(
            f"global upsampled support has {coords128.shape[0]:,} tokens; "
            f"exceeds --max-num-tokens={int(args.max_num_tokens):,}"
        )
    print(f"[global] one-step-upsampled support tokens={coords128.shape[0]:,}")

    global_dir = output_dir / "global_geometry_prior"
    global_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "coords32": coords32.detach().cpu(),
            "coords64": coords64.detach().cpu(),
            "coords128": coords128.detach().cpu(),
            "shape1024_norm_feats": global_shape.shape_norm.feats.detach().cpu(),
            "shape1024_denorm_feats": global_shape.shape_denorm.feats.detach().cpu(),
        },
        global_dir / "global_geometry_prior.pt",
    )

    # The live MeshWithVoxel and dense C1024 support now own downstream data.
    del global_baseline_result
    del global_texture_norm, global_texture_denorm
    del global_shape
    _empty_cuda_cache()

    global_baseline_metric = render_and_evaluate_mesh(
        global_mesh,
        camera_angle_x=float(global_camera["camera_angle_x"]),
        distance=float(global_camera["distance"]),
        output_dir=global_baseline_dir / "aligned_eval",
        reference_image=output_dir / "canonical_1024.png",
        resolution=int(args.baseline_render_resolution),
        metric_resolution=int(args.metric_resolution),
        envmap=envmap,
        envmap_name=str(args.envmap),
        ssaa=int(args.render_ssaa),
        peel_layers=int(args.render_peel_layers),
        use_envmap_bg=bool(args.use_envmap_bg),
        lpips_net=str(args.lpips_net),
        metric_device=str(args.metric_device),
        skip_lpips=bool(args.skip_lpips),
    )
    global_baseline_extras = _save_extra_comparisons(
        Path(global_baseline_metric["original_png"]),
        Path(global_baseline_metric["render_png"]),
        global_baseline_dir / "comparisons",
    )
    global_baseline_summary = {
        **global_decoder_metadata,
        **global_baseline_metric,
        **global_baseline_extras,
        "route": (
            "ordinary global SS C32 -> shape512 -> learned C64 -> "
            "shape1024 -> texture1024 -> decode"
        ),
        "global_c32_tokens": int(coords32.shape[0]),
        "global_c64_tokens": int(coords64.shape[0]),
        "global_c1024_support_tokens": int(coords128.shape[0]),
        "shape512_seconds": global_shape512_seconds,
        "shape1024_seconds": global_shape1024_seconds,
        "texture1024_seconds": float(global_texture_seconds),
    }
    _atomic_json(global_baseline_dir / "summary.json", global_baseline_summary)
    global_baseline_render_path = Path(str(global_baseline_metric["render_png"]))
    del global_meshes, global_mesh
    _empty_cuda_cache()

    global_summary = {
        "global_c32_tokens": int(coords32.shape[0]),
        "global_c64_tokens": int(coords64.shape[0]),
        "global_upsampled_tokens": int(coords128.shape[0]),
        "shape512_seconds": global_shape512_seconds,
        "shape1024_seconds": global_shape1024_seconds,
        "texture1024_seconds": float(global_texture_seconds),
        "one_step_upsample": upsample_stats,
        "texture_generated": True,
        "decoded": True,
        "baseline_full_image_metrics": {
            "psnr_db": global_baseline_metric.get("psnr_db"),
            "ssim": global_baseline_metric.get("ssim"),
            "lpips": global_baseline_metric.get("lpips"),
        },
        "baseline_summary_json": str(global_baseline_dir / "summary.json"),
        "baseline_render_png": str(global_baseline_render_path),
        "baseline_render_resolution": int(args.baseline_render_resolution),
    }
    _atomic_json(global_dir / "summary.json", global_summary)

    q128_global = _endpoint_indices_to_q(
        coords128[:, 1:4],
        GRID_GLOBAL_UPSAMPLED,
    ).to(coords128.device)
    _, uv_global_1024, uv_full_4096, _, finite_global = (
        _project_global_q_to_1024_and_4096(
            q128_global,
            global_camera=global_camera,
        )
    )
    global_projection_stats = {
        "global_1024_uv_min": [
            float(v) for v in uv_global_1024.amin(dim=0).detach().cpu().tolist()
        ],
        "global_1024_uv_max": [
            float(v) for v in uv_global_1024.amax(dim=0).detach().cpu().tolist()
        ],
        "full_4096_uv_min": [
            float(v) for v in uv_full_4096.amin(dim=0).detach().cpu().tolist()
        ],
        "full_4096_uv_max": [
            float(v) for v in uv_full_4096.amax(dim=0).detach().cpu().tolist()
        ],
        "global_to_full_scale": IMAGE_CANONICAL / GLOBAL_CAMERA_IMAGE_SIZE,
        "finite_rows": int(finite_global.sum().item()),
    }
    _atomic_json(global_dir / "global_projection_stats.json", global_projection_stats)

    boxes = _tile_layout(
        IMAGE_CANONICAL,
        int(args.tile_size),
        int(args.tile_stride),
    )
    selected_ids = _parse_tile_ids(args.tile_ids)
    if selected_ids is not None:
        invalid_ids = sorted(
            tile_id
            for tile_id in selected_ids
            if tile_id >= len(boxes) or tile_id < 0
        )
        if invalid_ids:
            raise ValueError(
                f"invalid tile ids: {invalid_ids}; valid range is 0-{len(boxes)-1}"
            )

    records: List[Dict[str, Any]] = []
    comparison_lpips_evaluator: Optional[_LPIPSEvaluator] = None
    attempted = 0
    for tile_id, box in enumerate(boxes):
        if selected_ids is not None and tile_id not in selected_ids:
            continue
        if args.max_tiles is not None and attempted >= int(args.max_tiles):
            break
        attempted += 1

        tile_dir = output_dir / "tiles" / f"tile_{tile_id:02d}"
        reference_tile = _prepare_tile_reference(image_4096, box, tile_dir)
        rows = _rows_inside_tile(uv_full_4096, finite_global, box)
        transform = _derive_tile_camera(
            tile_id=tile_id,
            box=box,
            global_camera=global_camera,
            extend_pixel=int(args.extend_pixel),
            offaxis_shift_y_sign=int(args.offaxis_shift_y_sign),
        )
        tile_camera = {
            "camera_angle_x": float(transform.camera_angle_x),
            "distance": float(transform.distance),
            "mesh_scale": float(transform.mesh_scale),
        }
        _atomic_json(tile_dir / "tile_camera.json", asdict(transform))
        _atomic_json(
            tile_dir / "decoder_to_global_camera.json",
            {
                "mapping": (
                    "per-vertex local projection -> inverse crop -> "
                    "global-depth back-projection"
                ),
                "normalized_depth_policy": "q_global.z = q_local.z",
                "coordinate_system": "global camera XYZ; camera looks along -Z",
            },
        )
        print(
            f"[tile {tile_id:02d}] selected global rows={rows.numel():,} "
            f"box={box} centered_fov={math.degrees(transform.camera_angle_x):.6f}deg "
            f"offaxis_shift=({transform.offaxis_shift_x:.6f},"
            f"{transform.offaxis_shift_y:.6f})"
        )
        seed_tile = int(args.seed) + tile_id * 1000 + 1

        if rows.numel() == 0:
            record = {
                "status": "skipped",
                "tile_id": int(tile_id),
                "box": list(box),
                "reason": "no global support projection inside tile",
                "selected_global_c128_rows": 0,
                "tile_projective_c32_tokens": 0,
                "tile_ss_c32_tokens": 0,
                "tile_c32_overlap_tokens": 0,
                "tile_c32_tokens": 0,
                "tile_projective_c64_tokens": 0,
                "tile_native_c64_tokens": 0,
                "tile_c64_overlap_tokens": 0,
                "tile_c64_tokens": 0,
            }
            records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            continue

        support_dir = tile_dir / "projected_global_c64_support"
        selected_coords128 = coords128.index_select(0, rows)
        try:
            projected_coords64, support_stats = _prepare_projective_tile_supports(
                reference=reference_tile,
                selected_coords128=selected_coords128,
                global_camera=global_camera,
                transform=transform,
                output_dir=support_dir,
                boundary_epsilon=float(args.boundary_epsilon),
            )

            projected_c64_tokens = int(projected_coords64.shape[0])
            if projected_c64_tokens < int(args.min_tile_tokens):
                record = {
                    "status": "skipped",
                    "tile_id": int(tile_id),
                    "box": list(box),
                    "reason": (
                        "projected global-C1024-derived C64 below "
                        "min_tile_tokens"
                    ),
                    "selected_global_c128_rows": int(rows.numel()),
                    "tile_projective_c32_tokens": 0,
                    "tile_ss_c32_tokens": 0,
                    "tile_c32_overlap_tokens": 0,
                    "tile_c32_tokens": 0,
                    "tile_projective_c64_tokens": projected_c64_tokens,
                    "tile_native_c64_tokens": 0,
                    "tile_c64_overlap_tokens": 0,
                    "tile_c64_tokens": 0,
                    "tile_ss_seconds": 0.0,
                    "shape512_seconds": 0.0,
                }
                records.append(record)
                _atomic_json(tile_dir / "summary.json", record)
                print(
                    f"[tile-skip] tile={tile_id:02d} "
                    f"projected_C64={projected_c64_tokens:,} "
                    f"(< min_tile_tokens={int(args.min_tile_tokens):,})"
                )
                continue
        except Exception as exc:
            record = {
                "status": "failed",
                "tile_id": int(tile_id),
                "box": list(box),
                "reason": (
                    "projected C64 support preparation failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "selected_global_c128_rows": int(rows.numel()),
                "tile_projective_c32_tokens": 0,
                "tile_ss_c32_tokens": 0,
                "tile_c32_overlap_tokens": 0,
                "tile_c32_tokens": 0,
                "tile_projective_c64_tokens": 0,
                "tile_native_c64_tokens": 0,
                "tile_c64_overlap_tokens": 0,
                "tile_c64_tokens": 0,
                "tile_ss_seconds": 0.0,
                "shape512_seconds": 0.0,
            }
            records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            print(f"[tile-support-error] tile={tile_id:02d}: {record['reason']}")
            _empty_cuda_cache()
            continue

        outside_fraction = float(
            support_stats["c64_quantization_stats"]["hard_outside_fraction"]
        )
        if outside_fraction > float(args.max_outside_fraction):
            record = {
                "status": "failed",
                "tile_id": int(tile_id),
                "box": list(box),
                "reason": (
                    f"hard outside fraction {outside_fraction:.6f} exceeds "
                    f"--max-outside-fraction="
                    f"{float(args.max_outside_fraction):.6f}"
                ),
                "selected_global_c128_rows": int(rows.numel()),
                "tile_projective_c32_tokens": 0,
                "tile_ss_c32_tokens": 0,
                "tile_c32_overlap_tokens": 0,
                "tile_c32_tokens": 0,
                "tile_projective_c64_tokens": projected_c64_tokens,
                "tile_native_c64_tokens": 0,
                "tile_c64_overlap_tokens": 0,
                "tile_c64_tokens": 0,
                "tile_ss_seconds": 0.0,
                "shape512_seconds": 0.0,
                "support_stats": support_stats,
            }
            records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            print(f"[tile-domain-error] tile={tile_id:02d}: {record['reason']}")
            continue

        if projected_c64_tokens > int(args.max_num_tokens):
            record = {
                "status": "failed",
                "tile_id": int(tile_id),
                "box": list(box),
                "reason": (
                    f"projected tile C64 tokens {projected_c64_tokens} exceed "
                    f"--max-num-tokens={int(args.max_num_tokens)}"
                ),
                "selected_global_c128_rows": int(rows.numel()),
                "tile_projective_c32_tokens": 0,
                "tile_ss_c32_tokens": 0,
                "tile_c32_overlap_tokens": 0,
                "tile_c32_tokens": 0,
                "tile_projective_c64_tokens": projected_c64_tokens,
                "tile_native_c64_tokens": 0,
                "tile_c64_overlap_tokens": 0,
                "tile_c64_tokens": 0,
                "tile_ss_seconds": 0.0,
                "shape512_seconds": 0.0,
                "support_stats": support_stats,
            }
            records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            print(
                f"[tile-token-error] tile={tile_id:02d} "
                f"projected_C64={projected_c64_tokens:,}"
            )
            continue

        route_summary: Optional[Dict[str, Any]] = None
        failure: Optional[str] = None
        try:
            result = _run_tile_projected_c64_only(
                pipeline=pipeline,
                tile_image=reference_tile,
                projected_coords64=projected_coords64,
                tile_camera=tile_camera,
                params=params,
                seed=seed_tile,
                label=f"Tile {tile_id:02d} projected-global-C64-only",
                max_tokens=int(args.max_num_tokens),
            )
            route_dir = tile_dir / "projected_global_c64_shape1024_texture"
            route_summary = _evaluate_tile_result(
                pipeline=pipeline,
                result=result,
                output_dir=route_dir,
                camera=tile_camera,
                global_camera=global_camera,
                transform=transform,
                seed=seed_tile,
                label=f"Tile {tile_id:02d} projected-global-C64-only",
                reference_image=tile_dir / "reference_tile.png",
                args=args,
                envmap=envmap,
            )
            route_summary.update(
                {
                    "route": (
                        "global C1024 projection -> local C64 quantization -> "
                        "shape1024 -> texture1024 -> decode"
                    ),
                    "tile_projective_c32_tokens": 0,
                    "tile_ss_c32_tokens": 0,
                    "tile_c32_overlap_tokens": 0,
                    "tile_c32_tokens": 0,
                    "tile_projective_c64_tokens": int(
                        result.tile_projective_c64_tokens
                    ),
                    "tile_native_c64_tokens": 0,
                    "tile_c64_overlap_tokens": 0,
                    "tile_c64_tokens": int(result.tile_c64_tokens),
                    "tile_ss_seconds": 0.0,
                    "shape512_seconds": 0.0,
                    "shape1024_seconds": float(result.shape1024_seconds),
                    "texture1024_seconds": float(
                        result.texture1024_seconds
                    ),
                    "tile_camera": asdict(transform),
                    "tile_native_support_enabled": False,
                }
            )
            _atomic_json(route_dir / "summary.json", route_summary)
            del result
            _empty_cuda_cache()
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            print(f"[tile-generation-error] tile={tile_id:02d}: {failure}")
            _empty_cuda_cache()

        baseline_tile_summary: Dict[str, Any] = {}
        try:
            baseline_tile_summary = _prepare_global_baseline_tile_crop(
                global_render_path=global_baseline_render_path,
                reference_path=tile_dir / "reference_tile.png",
                box=box,
                output_dir=tile_dir / "global_baseline_1024_crop",
            )
        except Exception as exc:
            print(
                f"[baseline-crop-error] tile={tile_id:02d}: "
                f"{type(exc).__name__}: {exc}"
            )

        comparison_path = (
            tile_dir / "comparison_reference_baseline_projected_c64_diffs.png"
        )
        record = {
            "status": "success" if route_summary is not None else "failed",
            "tile_id": int(tile_id),
            "box": list(box),
            "selected_global_c128_rows": int(rows.numel()),
            "tile_projective_c32_tokens": 0,
            "tile_ss_c32_tokens": 0,
            "tile_c32_overlap_tokens": 0,
            "tile_c32_tokens": 0,
            "tile_projective_c64_tokens": projected_c64_tokens,
            "tile_native_c64_tokens": 0,
            "tile_c64_overlap_tokens": 0,
            "tile_c64_tokens": (
                None if route_summary is None
                else route_summary.get("tile_c64_tokens")
            ),
            "centered_tile_fov_deg": math.degrees(transform.camera_angle_x),
            "centered_tile_fx": transform.fx,
            "offaxis_cx": transform.offaxis_cx,
            "offaxis_cy": transform.offaxis_cy,
            "offaxis_shift_x": transform.offaxis_shift_x,
            "offaxis_shift_y": transform.offaxis_shift_y,
            "hard_outside_fraction": outside_fraction,
            "support_pixel_roundtrip_max": support_stats["transform_stats"][
                "pixel_roundtrip_max"
            ],
            "closed_form_q_error_max": support_stats["transform_stats"][
                "closed_form_q_error_max"
            ],
            "quantization_pixel_error_mean": support_stats[
                "c64_pixel_error_mean"
            ],
            "quantization_pixel_error_p95": support_stats[
                "c64_pixel_error_p95"
            ],
            "quantization_pixel_error_max": support_stats[
                "c64_pixel_error_max"
            ],
            "tile_dir": str(tile_dir),
            "reference_png": str(tile_dir / "reference_tile.png"),
            "support_overlay_png": str(
                support_dir / "projected_global_c64_support_overlay.png"
            ),
            "comparison_png": str(comparison_path),
            **baseline_tile_summary,
            "baseline_psnr_db": None,
            "baseline_ssim": None,
            "baseline_lpips": None,
            "psnr_gain_db": None,
            "ssim_gain": None,
            "lpips_reduction": None,
            "psnr_db": (
                None if route_summary is None
                else route_summary.get("psnr_db")
            ),
            "ssim": (
                None if route_summary is None
                else route_summary.get("ssim")
            ),
            "lpips": (
                None if route_summary is None
                else route_summary.get("lpips")
            ),
            "render_png": (
                None if route_summary is None
                else route_summary.get("render_png")
            ),
            "triptych_png": (
                None if route_summary is None
                else route_summary.get("triptych_png")
            ),
            "diff_heatmap_png": (
                None if route_summary is None
                else route_summary.get("diff_heatmap_png")
            ),
            "tile_ss_seconds": 0.0,
            "shape512_seconds": 0.0,
            "shape1024_seconds": (
                None if route_summary is None
                else route_summary.get("shape1024_seconds")
            ),
            "texture1024_seconds": (
                None if route_summary is None
                else route_summary.get("texture1024_seconds")
            ),
            "renderer": (
                None if route_summary is None
                else route_summary.get("renderer")
            ),
            "failure": failure,
            "tile_native_support_enabled": False,
        }
        records.append(record)
        if (
            comparison_lpips_evaluator is None
            and not bool(args.skip_lpips)
            and record.get("baseline_render_png")
        ):
            metric_device_name = str(args.metric_device)
            if (
                metric_device_name.startswith("cuda")
                and not torch.cuda.is_available()
            ):
                metric_device_name = "cpu"
            comparison_lpips_evaluator = _LPIPSEvaluator(
                str(args.lpips_net),
                torch.device(metric_device_name),
            )
        _evaluate_baseline_tile_records(
            records=[record],
            args=args,
            lpips_evaluator=comparison_lpips_evaluator,
        )
        _atomic_json(tile_dir / "summary.json", record)
        print(
            f"[tile-summary] tile={tile_id:02d} "
            f"projected_C64={projected_c64_tokens:,} "
            f"native_support=disabled "
            f"outside={outside_fraction:.6f} "
            f"PSNR={record['psnr_db']} SSIM={record['ssim']} "
            f"LPIPS={record['lpips']} renderer={record['renderer']}"
        )

    if comparison_lpips_evaluator is not None:
        comparison_lpips_evaluator.model.cpu()
        del comparison_lpips_evaluator
        _empty_cuda_cache()

    aggregate_csv = output_dir / "aggregate_metrics.csv"
    _write_csv(aggregate_csv, records)
    contact_sheets = _write_contact_sheets(records, output_dir)
    success_rows = [row for row in records if row.get("status") == "success"]

    def _mean(key: str) -> Optional[float]:
        values = [
            float(row[key])
            for row in success_rows
            if row.get(key) is not None and math.isfinite(float(row[key]))
        ]
        return None if not values else float(np.mean(values))

    summary = {
        "format": "pixal3d_global_baseline_vs_projected_c64_only_tile_v1",
        "image": str(Path(args.image).expanduser().resolve()),
        "global_camera": global_camera,
        "global_geometry": global_summary,
        "global_baseline_1024": global_baseline_summary,
        "comparison_protocol": (
            "render native MeshWithVoxel outputs with Pixal3D PbrMeshRenderer; "
            "render the single full global 1024-model baseline at "
            "baseline_render_resolution, crop it by each canonical 4096 tile "
            "box, resize only when necessary, then compare both baseline crop "
            "and tile render against the same reference tile"
        ),
        "tile_cascade": [
            "select global C1024 support whose projection lies inside the tile",
            "transform selected global support to centered tile local q",
            "drop transformed points outside the local canonical cube",
            "quantize transformed support directly to unique local C64",
            "run projected-global C64 through tile shape1024 flow",
            "run texture1024 flow conditioned on tile shape1024",
            "shape and texture decode",
            "compare tile render with the ray-equivalent global baseline crop",
        ],
        "coordinate_transform": {
            "global_projection_grid": GLOBAL_CAMERA_IMAGE_SIZE,
            "canonical_crop_grid": IMAGE_CANONICAL,
            "tile_output_grid": IMAGE_FLOW,
            "point_centroid_used": False,
            "point_bbox_normalization_used": False,
            "hard_geometry_clipping_used": False,
            "true_outside_rows_are_dropped": True,
            "outside_reporting_epsilon": float(args.boundary_epsilon),
            "seed_support_description": (
                "global C1024 q -> global camera point -> 1024 projection -> "
                "4096 crop -> centered tile back-projection with recomputed "
                "local distance and preserved normalized q_z -> local q -> "
                "direct C64 quantization and unique; no tile-native SS, C32, "
                "shape512, learned native C64, or C64 fusion"
            ),
            "tile_native_sparse_structure_enabled": False,
            "tile_shape512_enabled": False,
            "tile_native_c64_enabled": False,
            "tile_c64_fusion_enabled": False,
            "decoded_mesh_description": (
                "the native MeshWithVoxel returned by decode_latent is passed "
                "directly to Pixal3D render_utils.render_frames; PbrMeshRenderer "
                "queries its sparse O-Voxel at every visible surface point"
            ),
        },
        "raw_glb_export": False,
        "uv_or_atlas_export": False,
        "material_sampling": (
            "official PbrMeshRenderer grid_sample_3d sparse trilinear lookup"
        ),
        "official_renderer": {
            "entrypoint": "pixal3d.utils.render_utils.render_frames",
            "renderer": "PbrMeshRenderer",
            "rasterizer": "nvdiffrast",
            "envmap": str(args.envmap),
            "ssaa": int(args.render_ssaa),
            "peel_layers": int(args.render_peel_layers),
            "use_envmap_bg": bool(args.use_envmap_bg),
            "cuda_device": torch.cuda.current_device(),
        },
        "tile_size": int(args.tile_size),
        "tile_stride": int(args.tile_stride),
        "min_tile_tokens_c64_projected_from_global_c1024": int(args.min_tile_tokens),
        "min_tile_tokens_applies_to": "tile_projective_c64_tokens",
        "max_num_tokens": int(args.max_num_tokens),
        "max_outside_fraction": float(args.max_outside_fraction),
        "attempted_tiles": attempted,
        "recorded_tiles": len(records),
        "successful_tiles": len(success_rows),
        "skipped_tiles": sum(row.get("status") == "skipped" for row in records),
        "failed_tiles": sum(row.get("status") == "failed" for row in records),
        "mean_successful_metrics": {
            "psnr_db": _mean("psnr_db"),
            "ssim": _mean("ssim"),
            "lpips": _mean("lpips"),
            "baseline_psnr_db": _mean("baseline_psnr_db"),
            "baseline_ssim": _mean("baseline_ssim"),
            "baseline_lpips": _mean("baseline_lpips"),
            "psnr_gain_db": _mean("psnr_gain_db"),
            "ssim_gain": _mean("ssim_gain"),
            "lpips_reduction": _mean("lpips_reduction"),
            "hard_outside_fraction": _mean("hard_outside_fraction"),
            "quantization_pixel_error_mean": _mean(
                "quantization_pixel_error_mean"
            ),
            "tile_ss_seconds": _mean("tile_ss_seconds"),
            "shape512_seconds": _mean("shape512_seconds"),
            "shape1024_seconds": _mean("shape1024_seconds"),
            "texture1024_seconds": _mean("texture1024_seconds"),
        },
        "aggregate_csv": str(aggregate_csv),
        "contact_sheets": contact_sheets,
        "tiles": records,
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(f"[summary] {output_dir / 'summary.json'}")


def run(args: argparse.Namespace) -> None:
    """Execute the single-global-state 2048 generation and evaluation route."""
    if int(args.tile_size) != DEFAULT_TILE_SIZE:
        raise ValueError(f"this route requires --tile-size={DEFAULT_TILE_SIZE}")
    if int(args.tile_stride) != DEFAULT_TILE_STRIDE:
        raise ValueError(f"this route requires --tile-stride={DEFAULT_TILE_STRIDE}")
    if args.cuda_device is not None:
        torch.cuda.set_device(int(args.cuda_device))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(int(args.seed))
    pipeline = init_pipeline(
        args.model_path,
        device="cuda",
        low_vram=bool(args.low_vram),
    )

    canonical = pipeline.preprocess_canonical_images(Image.open(args.image))
    image_4096: Image.Image = canonical["image_4096"]
    image_1024: Image.Image = canonical["image_1024"]
    image_512: Image.Image = canonical["image_512"]
    foreground_mask_4096: Image.Image = canonical[
        "foreground_mask_4096"
    ].convert("L")
    if image_4096.size != (IMAGE_CANONICAL, IMAGE_CANONICAL):
        raise RuntimeError(
            f"canonical 4096 image has unexpected size {image_4096.size}"
        )
    image_4096.save(output_dir / "canonical_4096.png")
    image_1024.save(output_dir / "canonical_1024.png")
    image_512.save(output_dir / "canonical_512.png")
    foreground_mask_4096.save(
        output_dir / "canonical_foreground_mask_4096.png"
    )
    _atomic_json(output_dir / "canonical_metadata.json", canonical["metadata"])

    global_camera = _estimate_camera(
        image_1024=image_1024,
        output_dir=output_dir,
        manual_fov=float(args.fov),
        mesh_scale=float(args.mesh_scale),
        extend_pixel=int(args.extend_pixel),
        image_resolution=int(args.camera_image_resolution),
        moge_model_path=args.moge_model_path,
    )
    _atomic_json(output_dir / "global_camera.json", global_camera)
    print(
        f"[global-camera] fov={global_camera['camera_angle_x']:.8f} "
        f"distance={global_camera['distance']:.8f} "
        f"mesh_scale={global_camera['mesh_scale']:.8f}"
    )
    params = _sampler_params(args, pipeline)

    latent_path = output_dir / "global_c128_latents.pt"
    if bool(args.resume_final_latents):
        shape_denorm, texture_denorm, latent_payload = (
            _load_sparse_final_latents(
                latent_path,
                device=pipeline.device,
            )
        )
        print(
            f"[resume] loaded final global C128 latents: {latent_path} "
            f"tokens={shape_denorm.coords.shape[0]:,}; generation is skipped"
        )
        final_result = _decode_save_and_render_final(
            pipeline=pipeline,
            shape_denorm=shape_denorm,
            texture_denorm=texture_denorm,
            output_dir=output_dir,
            global_camera=global_camera,
            args=args,
        )
        metric_row = final_result["render_and_metrics"]

        def _optional_json(path: Path) -> Optional[Dict[str, Any]]:
            if not path.is_file():
                return None
            return json.loads(path.read_text(encoding="utf-8"))

        resume_summary = {
            "format": (
                "pixal3d_single_global_c128_direct_tile_c64_velocity_residual_v2"
            ),
            "resumed_from_final_latents": True,
            "image": str(Path(args.image).expanduser().resolve()),
            "global_camera": global_camera,
            "support": _optional_json(
                output_dir / "global_support" / "summary.json"
            ),
            "shape_flow": _optional_json(output_dir / "shape_flow.json"),
            "texture_flow": _optional_json(output_dir / "texture_flow.json"),
            "latents": str(latent_path),
            "latent_resolution": int(
                latent_payload.get("resolution", DECODE_GLOBAL)
            ),
            "decoder": final_result["decoder"],
            "mesh_checkpoint": final_result["mesh_checkpoint"],
            "render_safety": final_result["render_safety"],
            "render_and_metrics": metric_row,
            "visual_metrics": {
                "psnr_db": metric_row.get("psnr_db"),
                "ssim": metric_row.get("ssim"),
                "lpips": metric_row.get("lpips"),
            },
        }
        final_dir = output_dir / "final_global_2048"
        _atomic_json(final_dir / "summary.json", resume_summary)
        _atomic_json(output_dir / "summary.json", resume_summary)
        print(
            f"[done-resume] mesh={resume_summary['mesh_checkpoint']} "
            f"render={metric_row['render_png']} "
            f"PSNR={metric_row.get('psnr_db')} "
            f"SSIM={metric_row.get('ssim')} LPIPS={metric_row.get('lpips')}"
        )
        return

    # Standard global prior.  It supplies topology/support only; no coarse
    # texture is generated because the final texture lives on global C128.
    print(
        "[global-prior] SS C32 -> shape512 -> learned C64 -> shape1024"
    )
    coords32, coords64, coarse_shape = (
        _run_global_official_geometry_to_shape1024(
            pipeline=pipeline,
            image_512=image_512,
            image_1024=image_1024,
            camera=global_camera,
            params=params,
            seed=int(args.seed),
            max_tokens=int(args.max_num_tokens),
        )
    )
    coarse_timings = {
        "shape512_seconds": float(coarse_shape.shape512_seconds),
        "shape1024_seconds": float(coarse_shape.shape1024_seconds),
    }
    print(
        "[global-support] shape decoder subdivision: global C64 latent "
        "-> dense C1024"
    )
    coords1024, subdivision_stats = (
        _learned_subdivide_shape1024_to_c1024(
            pipeline,
            coarse_shape.shape_denorm,
        )
    )
    coords128, source_to_global, quantization_stats = (
        _quantize_global_c1024_to_c128(coords1024)
    )
    if coords128.shape[0] > int(args.max_num_tokens):
        raise RuntimeError(
            f"global C128 support has {coords128.shape[0]:,} tokens; exceeds "
            f"--max-num-tokens={int(args.max_num_tokens):,}"
        )
    print(
        f"[global-support] dense_C1024={coords1024.shape[0]:,} "
        f"final_global_C128={coords128.shape[0]:,}"
    )

    support_dir = output_dir / "global_support"
    support_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "coords_c32": coords32.detach().cpu(),
            "coords_c64": coords64.detach().cpu(),
            "coords_c1024": coords1024.detach().cpu(),
            "coords_global_c128": coords128.detach().cpu(),
            "c1024_source_to_global_c128_token": (
                source_to_global.detach().cpu()
            ),
        },
        support_dir / "global_support_and_source_mapping.pt",
    )
    support_summary = {
        "route": (
            "SS C32 -> shape512 -> learned C64 -> shape1024 -> "
            "shape decoder subdivision C1024 (global-only intermediate) -> "
            "fixed quantized C128"
        ),
        "global_c32_tokens": int(coords32.shape[0]),
        "global_c64_tokens": int(coords64.shape[0]),
        "dense_global_c1024_points": int(coords1024.shape[0]),
        "global_c128_tokens": int(coords128.shape[0]),
        "subdivision": subdivision_stats,
        "c1024_to_c128": quantization_stats,
        **coarse_timings,
    }
    _atomic_json(support_dir / "summary.json", support_summary)
    # From this point onward C1024 and its inverse mapping are deliberately
    # destroyed.  No tile-selection, camera transform, condition extraction,
    # or flow call can consume them.
    del coarse_shape, coords1024, source_to_global
    _empty_cuda_cache()

    # C1024 is used only to construct the fixed global C128 support above.
    # Tile selection and local flow transport start from global C128 itself.
    q128_global = _endpoint_indices_to_q(
        coords128[:, 1:4],
        GRID_FINAL_2048,
    ).to(coords128.device)
    _, uv_global_1024, uv_full_4096, _, finite_global = (
        _project_global_q_to_1024_and_4096(
            q128_global,
            global_camera=global_camera,
        )
    )
    projection_summary = {
        "projection_source": "unique global C128 flow tokens",
        "finite_global_c128_tokens": int(finite_global.sum().item()),
        "total_global_c128_tokens": int(coords128.shape[0]),
        "uv_global_1024_min": [
            float(v) for v in uv_global_1024.amin(dim=0).cpu().tolist()
        ],
        "uv_global_1024_max": [
            float(v) for v in uv_global_1024.amax(dim=0).cpu().tolist()
        ],
        "uv_full_4096_min": [
            float(v) for v in uv_full_4096.amin(dim=0).cpu().tolist()
        ],
        "uv_full_4096_max": [
            float(v) for v in uv_full_4096.amax(dim=0).cpu().tolist()
        ],
    }
    _atomic_json(support_dir / "projection_summary.json", projection_summary)

    transports, tile_records = _prepare_all_tile_transports(
        args=args,
        image_4096=image_4096,
        global_coords128=coords128,
        uv_full_4096=uv_full_4096,
        finite_global=finite_global,
        global_camera=global_camera,
        output_dir=output_dir,
    )
    _write_csv(output_dir / "tile_transport_summary.csv", tile_records)
    del (
        q128_global,
        uv_global_1024,
        uv_full_4096,
        finite_global,
        coords32,
        coords64,
    )
    _empty_cuda_cache()

    # Shape: late and deliberately small local residual.
    shape_condition_cpu, shape_condition_stats = _prepare_stage_conditions(
        pipeline=pipeline,
        stage_name="shape",
        image_1024=image_1024,
        image_4096=image_4096,
        global_coords=coords128,
        global_camera=global_camera,
        transports=transports,
    )
    shape_norm, shape_seconds, shape_flow = _run_single_global_c128_flow(
        pipeline=pipeline,
        stage_name="shape",
        model=pipeline.models["shape_slat_flow_model_1024"],
        sampler=pipeline.shape_slat_sampler,
        global_coords=coords128,
        global_condition_cpu=shape_condition_cpu,
        transports=transports,
        params=params["shape"],
        seed=int(args.seed) + 401,
        local_start_fraction=float(args.shape_local_start),
        local_max_weight=float(args.shape_local_weight),
    )
    shape_denorm = _denormalize_sparse(
        shape_norm,
        pipeline.shape_slat_normalization,
    )
    _atomic_json(output_dir / "shape_flow.json", shape_flow)
    del shape_condition_cpu
    _empty_cuda_cache()

    # Texture: the fixed final global shape is the concat condition everywhere.
    # Its tile form is transported from global C128; it is never regenerated.
    texture_condition_cpu, texture_condition_stats = _prepare_stage_conditions(
        pipeline=pipeline,
        stage_name="texture",
        image_1024=image_1024,
        image_4096=image_4096,
        global_coords=coords128,
        global_camera=global_camera,
        transports=transports,
    )
    texture_norm, texture_seconds, texture_flow = (
        _run_single_global_c128_flow(
            pipeline=pipeline,
            stage_name="texture",
            model=pipeline.models["tex_slat_flow_model_1024"],
            sampler=pipeline.tex_slat_sampler,
            global_coords=coords128,
            global_condition_cpu=texture_condition_cpu,
            transports=transports,
            params=params["texture"],
            seed=int(args.seed) + 501,
            local_start_fraction=float(args.texture_local_start),
            local_max_weight=float(args.texture_local_weight),
            concat_global=shape_norm,
        )
    )
    texture_denorm = _denormalize_sparse(
        texture_norm,
        pipeline.tex_slat_normalization,
    )
    _atomic_json(output_dir / "texture_flow.json", texture_flow)
    del texture_condition_cpu
    _empty_cuda_cache()

    torch.save(
        {
            "resolution": DECODE_GLOBAL,
            "grid_resolution": GRID_FINAL_2048,
            "coords": coords128.detach().cpu(),
            "shape_norm_feats": shape_norm.feats.detach().cpu(),
            "shape_denorm_feats": shape_denorm.feats.detach().cpu(),
            "texture_norm_feats": texture_norm.feats.detach().cpu(),
            "texture_denorm_feats": texture_denorm.feats.detach().cpu(),
            "global_camera": dict(global_camera),
        },
        latent_path,
    )

    final_result = _decode_save_and_render_final(
        pipeline=pipeline,
        shape_denorm=shape_denorm,
        texture_denorm=texture_denorm,
        output_dir=output_dir,
        global_camera=global_camera,
        args=args,
    )
    final_dir = output_dir / "final_global_2048"
    metric_row = final_result["render_and_metrics"]

    summary = {
        "format": (
            "pixal3d_single_global_c128_direct_tile_c64_velocity_residual_v2"
        ),
        "image": str(Path(args.image).expanduser().resolve()),
        "global_camera": global_camera,
        "support": support_summary,
        "projection": projection_summary,
        "transport": {
            "tile_size": int(args.tile_size),
            "tile_stride": int(args.tile_stride),
            "weight": str(args.tile_weight),
            "active_tiles": int(len(transports)),
            "tile_native_support_generation": False,
            "tile_noise_initializations": 0,
            "tile_flow_trajectories": 0,
            "source_correspondence": (
                "each kept projected global C128 flow token stores its direct "
                "local C64 token; C1024 is not projected into any tile"
            ),
            "tiles": tile_records,
        },
        "shape": {
            "condition_extraction": shape_condition_stats,
            "flow": shape_flow,
            "seconds": float(shape_seconds),
        },
        "texture": {
            "condition_extraction": texture_condition_stats,
            "flow": texture_flow,
            "seconds": float(texture_seconds),
        },
        "state_invariant": (
            "shape and texture each initialize one Gaussian global C128 state; "
            "tiles are temporary views and only one global Euler update occurs "
            "per step"
        ),
        "decoder": final_result["decoder"],
        "latents": str(latent_path),
        "mesh_checkpoint": final_result["mesh_checkpoint"],
        "render_safety": final_result["render_safety"],
        "render_and_metrics": metric_row,
        "visual_metrics": {
            "psnr_db": metric_row.get("psnr_db"),
            "ssim": metric_row.get("ssim"),
            "lpips": metric_row.get("lpips"),
        },
    }
    _atomic_json(final_dir / "summary.json", summary)
    _atomic_json(output_dir / "summary.json", summary)
    del final_result
    _empty_cuda_cache()
    print(
        f"[done] mesh={summary['mesh_checkpoint']} render={metric_row['render_png']} "
        f"PSNR={metric_row.get('psnr_db')} SSIM={metric_row.get('ssim')} "
        f"LPIPS={metric_row.get('lpips')}"
    )
    print(f"[summary] {output_dir / 'summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--moge-model-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-final-latents",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "skip support/flow generation and decode+render the existing "
            "OUTPUT_DIR/global_c128_latents.pt"
        ),
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="visible CUDA device index; omitted respects the current CUDA environment",
    )

    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--tile-stride", type=int, default=DEFAULT_TILE_STRIDE)
    parser.add_argument(
        "--tile-ids",
        default=None,
        help="comma-separated tile ids; omitted means all 49 tiles",
    )
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument(
        "--tile-weight",
        choices=("tent", "uniform"),
        default="tent",
        help=(
            "image-space weighting used when local residual transports overlap; "
            "tent suppresses tile-boundary seams"
        ),
    )
    parser.add_argument(
        "--min-tile-tokens",
        type=int,
        default=1000,
        help=(
            "minimum unique local C64 tokens required for a tile velocity expert"
        ),
    )
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=100000000,
        help="hard token limit for learned C64 and final global C128 supports",
    )
    parser.add_argument(
        "--boundary-epsilon",
        type=float,
        default=1e-5,
        help=(
            "Separates tiny numerical overflow from hard overflow in diagnostics. "
            "Both are dropped; neither is clipped to [-1,1]."
        ),
    )
    parser.add_argument(
        "--max-outside-fraction",
        type=float,
        default=0.10,
        help=(
            "Fail a tile if too many transformed points leave the local cube. "
            "A small nonzero fraction is expected at perspective far-plane edges."
        ),
    )
    parser.add_argument(
        "--offaxis-shift-y-sign",
        type=int,
        choices=(-1, 1),
        default=1,
        help="Vertical sign convention for saved off-axis crop diagnostics.",
    )

    parser.add_argument("--fov", type=float, default=-1.0)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument("--camera-image-resolution", type=int, default=1024)

    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--ss-guidance-strength", type=float, default=7.5)
    parser.add_argument("--ss-guidance-rescale", type=float, default=0.7)
    parser.add_argument("--ss-rescale-t", type=float, default=5.0)
    parser.add_argument("--shape-steps", type=int, default=12)
    parser.add_argument("--shape-guidance-strength", type=float, default=7.5)
    parser.add_argument("--shape-guidance-rescale", type=float, default=0.5)
    parser.add_argument("--shape-rescale-t", type=float, default=3.0)
    parser.add_argument(
        "--shape-local-start",
        type=float,
        default=0.65,
        help=(
            "denoising progress at which shape tile residuals begin; the "
            "default confines them to late steps"
        ),
    )
    parser.add_argument(
        "--shape-local-weight",
        type=float,
        default=0.20,
        help="maximum late shape velocity-residual weight",
    )
    parser.add_argument("--texture-steps", type=int, default=12)
    parser.add_argument("--texture-guidance-strength", type=float, default=1.0)
    parser.add_argument("--texture-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--texture-rescale-t", type=float, default=3.0)
    parser.add_argument(
        "--texture-local-start",
        type=float,
        default=0.10,
        help="denoising progress at which texture tile residuals begin",
    )
    parser.add_argument(
        "--texture-local-weight",
        type=float,
        default=0.80,
        help="maximum texture velocity-residual weight",
    )

    parser.add_argument(
        "--low-vram",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--envmap",
        default="studio",
        help="bundled Pixal3D HDRI name (for example studio) or an EXR path",
    )
    parser.add_argument("--render-resolution", type=int, default=DECODE_GLOBAL)
    parser.add_argument(
        "--render-face-chunk-size",
        type=int,
        default=4_000_000,
        help=(
            "maximum faces in one unchanged-mesh nvdiffrast call; chunk "
            "layers are merged exactly in screen-space depth order, and zero "
            "disables chunking"
        ),
    )
    parser.add_argument(
        "--export-tex-point-cloud",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "also export the decoder's very large debug texture point cloud "
            "to the working directory"
        ),
    )
    parser.add_argument("--metric-resolution", type=int, default=1024)
    parser.add_argument("--render-ssaa", type=int, default=1)
    parser.add_argument("--render-peel-layers", type=int, default=4)
    parser.add_argument(
        "--use-envmap-bg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the HDRI as the background; default keeps a black background.",
    )
    parser.add_argument(
        "--lpips-net",
        choices=("alex", "vgg", "squeeze"),
        default="vgg",
    )
    parser.add_argument("--metric-device", default="cuda")
    parser.add_argument(
        "--skip-lpips",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if int(args.min_tile_tokens) < 1:
        raise ValueError("--min-tile-tokens must be positive")
    if int(args.max_num_tokens) < int(args.min_tile_tokens):
        raise ValueError("--max-num-tokens must be >= --min-tile-tokens")
    if args.max_tiles is not None and int(args.max_tiles) < 1:
        raise ValueError("--max-tiles must be positive")
    if args.cuda_device is not None and int(args.cuda_device) < 0:
        raise ValueError("--cuda-device must be non-negative")
    if int(args.render_face_chunk_size) < 0:
        raise ValueError("--render-face-chunk-size must be non-negative")
    if (
        int(args.ss_steps) < 1
        or int(args.shape_steps) < 1
        or int(args.texture_steps) < 1
    ):
        raise ValueError("all flow step counts must be positive")
    if (
        int(args.render_resolution) < 1
        or int(args.metric_resolution) < 1
        or int(args.render_ssaa) < 1
        or int(args.render_peel_layers) < 1
    ):
        raise ValueError(
            "render resolutions, metric resolution, SSAA, and peel layers "
            "must be positive"
        )
    if float(args.boundary_epsilon) < 0:
        raise ValueError("--boundary-epsilon must be non-negative")
    if not (0.0 <= float(args.max_outside_fraction) <= 1.0):
        raise ValueError("--max-outside-fraction must be in [0,1]")
    for name in ("shape_local_start", "texture_local_start"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1]")
    for name in ("shape_local_weight", "texture_local_weight"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and non-negative"
            )
    run(args)


if __name__ == "__main__":
    main()
