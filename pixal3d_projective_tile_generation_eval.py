#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a normal global Pixal3D 1024 baseline, then projectively generate tiles.

The script first runs the complete ordinary global 1024 route:

    global SS C32 -> shape512 -> learned C64 -> shape1024 -> texture1024
    -> shape/texture decode -> full-image render and metrics

It then runs the tile cascade:

    global C1024 support -> local C1024 + projected C32 + projected C64
    tile-native sparse-structure C32 sampled from the tile image
    -> retain alpha-foreground tokens within one C32 cell of projected C32
    -> only constrained tile-native C32 enters tile shape512 flow
    -> learned native C64
    -> fuse native C64 with projected-global C64 and unique
    -> tile shape1024 flow
    -> tile texture1024 flow
    -> shape/texture decode

The global support is transformed into each centered tile camera by exact
projection/crop/back-projection. Each tile also independently runs the official
sparse-structure stage with ``get_proj_cond_ss`` and ``sample_sparse_structure``.
Tile-native C32 coordinates projected outside that tile's retained alpha
foreground or outside the projected-global C32 one-cell neighborhood are
discarded. Only those constrained native C32 coordinates enter shape512. Its
learned native C64 support is fused with the projected-global C64 support before
shape1024 and texture1024. No point-cloud centroid normalization, bounding-box
normalization, or boundary clipping is used. Projected points outside the local
canonical cube are dropped.

For every successful tile, the full global baseline render is cropped using the
corresponding 4096 tile box, resized to the tile evaluation resolution, and scored
against exactly the same tile reference. Final comparison sheets report both the
global-baseline crop quality and the tile-cascade quality, including metric gains.

For evaluation, the decoded tile remains a native ``MeshWithVoxel`` in the
centered tile canonical frame.  The aligned target-view camera is passed
directly to Pixal3D's official ``render_utils.render_frames`` API.  Its
``PbrMeshRenderer`` interpolates each visible surface position and performs the
native sparse O-Voxel trilinear query.  No UVs, GLB, atlas, Blender process, or
Cycles shader reconstruction are involved.

Expected repository layout
--------------------------
Place this script in the Pixal3D repository root beside ``inference.py`` and
``render_pixal3d_raw_ovoxel.py``.
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
IMAGE_LR = 512
GLOBAL_CAMERA_IMAGE_SIZE = 1024
IMAGE_CANONICAL = 4096
IMAGE_FLOW = 1024
DECODE_TILE = 1024
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


def _learned_upsample_shape1024_to_c1024(
    pipeline: Any,
    shape1024_denorm: SparseTensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
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
    coords128 = torch.unique(coords1024_all[valid], dim=0)
    if coords128.numel() == 0:
        raise RuntimeError("shape1024 one-step upsample produced no valid C128 coordinates")

    stats = {
        "source_c64_tokens": int(shape1024_denorm.coords.shape[0]),
        "candidate_rows": int(candidates.shape[0]),
        "valid_candidate_rows": int(valid.sum().item()),
        "discarded_out_of_range_rows": int((~valid).sum().item()),
        "unique_c128_tokens": int(coords128.shape[0]),
        "unique_merge_rows": int(valid.sum().item() - coords128.shape[0]),
        "max_fractional_coordinate_error": max_fractional_error,
        "min_xyz": [
            int(v) for v in coords128[:, 1:].amin(dim=0).detach().cpu().tolist()
        ],
        "max_xyz": [
            int(v) for v in coords128[:, 1:].amax(dim=0).detach().cpu().tolist()
        ],
    }
    return coords128, stats



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



def _run_tile_official_cascade(
    *,
    pipeline: Any,
    tile_image: Image.Image,
    native_coords32: torch.Tensor,
    projected_coords64: torch.Tensor,
    tile_camera: Mapping[str, float],
    transform: TileCameraTransform,
    reference: Image.Image,
    support_output_dir: Path,
    params: Mapping[str, Mapping[str, Any]],
    seed: int,
    label: str,
    max_tokens: int,
    projective_c32_tokens: int,
    sampled_ss_c32_tokens: int,
    c32_overlap_tokens: int,
    tile_ss_seconds: float,
) -> ModelResult:
    tile_1024 = tile_image.convert("RGB")
    tile_512 = tile_1024.resize(
        (IMAGE_LR, IMAGE_LR),
        Image.Resampling.LANCZOS,
    )

    native_coords64, shape512_seconds = _run_shape512_and_upsample_c64(
        pipeline=pipeline,
        image_512=tile_512,
        coords32=native_coords32,
        camera=tile_camera,
        params=params,
        seed=seed + 101,
        description=f"{label} native C32 shape 512",
    )
    coords64, c64_fusion_stats = _merge_projected_and_native_c64(
        projected_coords64=projected_coords64,
        native_coords64=native_coords64,
        transform=transform,
        reference=reference,
        output_dir=support_output_dir,
    )
    if coords64.shape[0] > int(max_tokens):
        raise RuntimeError(
            f"{label}: fused C64 support has {coords64.shape[0]:,} tokens; "
            f"exceeds --max-num-tokens={int(max_tokens):,}"
        )
    print(
        f"[tile-cascade] {label}: native_C32={native_coords32.shape[0]:,} "
        f"-> native_C64={native_coords64.shape[0]:,}; "
        f"projected_C64={projected_coords64.shape[0]:,} "
        f"-> fused_C64={coords64.shape[0]:,}"
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
    del tile_512
    _empty_cuda_cache()
    return ModelResult(
        shape_norm=shape_norm,
        shape_denorm=shape_denorm,
        texture_norm=texture_norm,
        texture_denorm=texture_denorm,
        tile_projective_c32_tokens=int(projective_c32_tokens),
        tile_ss_c32_tokens=int(sampled_ss_c32_tokens),
        tile_c32_overlap_tokens=int(c32_overlap_tokens),
        tile_c32_tokens=int(native_coords32.shape[0]),
        tile_projective_c64_tokens=int(
            c64_fusion_stats["projected_c64_tokens"]
        ),
        tile_native_c64_tokens=int(c64_fusion_stats["native_c64_tokens"]),
        tile_c64_overlap_tokens=int(c64_fusion_stats["overlap_c64_tokens"]),
        tile_c64_tokens=int(coords64.shape[0]),
        tile_ss_seconds=float(tile_ss_seconds),
        shape512_seconds=float(shape512_seconds),
        shape1024_seconds=float(shape1024_seconds),
        texture1024_seconds=float(texture1024_seconds),
    )


def _decode_normal_mesh_with_ovoxel(
    *,
    pipeline: Any,
    shape_latent: SparseTensor,
    texture_latent: SparseTensor,
    label: str,
) -> Tuple[List[Any], Any]:
    """Run the normal decoder once and retain its native ``MeshWithVoxel``."""
    decoded = pipeline.decode_latent(
        shape_latent,
        texture_latent,
        DECODE_TILE,
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


def _save_decoded_support_checkpoint(
    mesh: Any,
    output_path: Path,
    *,
    include_faces: bool,
) -> str:
    """Persist the tensors needed for support analysis without GLB/PLY export."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layout = {
        str(name): {
            "start": value.start,
            "stop": value.stop,
            "step": value.step,
        }
        for name, value in dict(mesh.layout).items()
    }
    payload: Dict[str, Any] = {
        "format": "pixal3d_decoded_support_v1",
        "vertices": mesh.vertices.detach().cpu().contiguous(),
        "ovoxel_coords_c1024": mesh.coords.detach().cpu().contiguous(),
        "ovoxel_attrs": mesh.attrs.detach().cpu().contiguous(),
        "origin": mesh.origin.detach().cpu().tolist(),
        "voxel_size": float(mesh.voxel_size),
        "voxel_shape": tuple(int(value) for value in mesh.voxel_shape),
        "layout": layout,
        "coordinate_conventions": {
            "vertices": (
                "decoder world coordinates in approximately [-0.5,0.5]^3; "
                "camera-normalized q = 2 * vertices"
            ),
            "ovoxel_coords_c1024": (
                "integer sparse material-voxel indices; center q is "
                "2 * (coord + 0.5) / 1024 - 1"
            ),
        },
    }
    if include_faces:
        payload["faces"] = mesh.faces.detach().cpu().contiguous()
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    return str(output_path)


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
    if bool(args.save_decoded_support):
        decoder_metadata["decoded_support_checkpoint"] = (
            _save_decoded_support_checkpoint(
                mesh,
                output_dir / "decoded_support.pt",
                include_faces=False,
            )
        )

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


def _prepare_tile_foreground_mask(
    foreground_mask_4096: Image.Image,
    box: Tuple[int, int, int, int],
    tile_dir: Path,
) -> Image.Image:
    """Crop the canonical alpha mask with exactly the RGB tile transform."""
    tile_mask = foreground_mask_4096.convert("L").crop(box)
    if tile_mask.size != (IMAGE_FLOW, IMAGE_FLOW):
        tile_mask = tile_mask.resize(
            (IMAGE_FLOW, IMAGE_FLOW),
            Image.Resampling.LANCZOS,
        )
    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_mask.save(tile_dir / "foreground_mask.png")
    return tile_mask


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
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
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

    (
        coords1024_local,
        kept_mask1024,
        quant_stats1024,
    ) = _quantize_local_q_without_geometry_clip(
        q_local,
        resolution=GRID_GLOBAL_UPSAMPLED,
        lattice_name="local C1024",
        epsilon=float(boundary_epsilon),
    )
    coords32, kept_mask32, quant_stats32 = _quantize_local_q_without_geometry_clip(
        q_local,
        resolution=GRID_SS,
        lattice_name="C32",
        epsilon=float(boundary_epsilon),
    )
    coords64, kept_mask64, quant_stats64 = _quantize_local_q_without_geometry_clip(
        q_local,
        resolution=GRID_SHAPE_1024,
        lattice_name="C64",
        epsilon=float(boundary_epsilon),
    )
    if not (
        torch.equal(kept_mask1024, kept_mask32)
        and torch.equal(kept_mask32, kept_mask64)
    ):
        raise RuntimeError("local C1024, C32, and C64 domain masks disagree")
    kept_mask = kept_mask32
    q_local_kept = q_local[kept_mask]
    uv_continuous_kept = uv_continuous[kept_mask]
    q_global_kept = q_global[kept_mask]

    def _quantization_diagnostics(
        coords: torch.Tensor,
        resolution: int,
    ) -> Dict[str, torch.Tensor]:
        ids_per_source = _q_to_endpoint_indices(q_local_kept, int(resolution))
        q_quantized = _endpoint_indices_to_q(
            ids_per_source,
            int(resolution),
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
        pixel_error = torch.linalg.vector_norm(
            uv_quantized - uv_continuous_kept,
            dim=1,
        )
        unique_uv, unique_q, unique_valid = _project_grid_coords_to_tile_uv(
            coords,
            resolution=int(resolution),
            transform=transform,
        )
        return {
            "uv_quantized": uv_quantized,
            "quantized_valid": quantized_valid,
            "pixel_error": pixel_error,
            "unique_uv": unique_uv,
            "unique_q": unique_q,
            "unique_valid": unique_valid,
        }

    diagnostics32 = _quantization_diagnostics(coords32, GRID_SS)
    diagnostics64 = _quantization_diagnostics(coords64, GRID_SHAPE_1024)

    output_dir.mkdir(parents=True, exist_ok=True)
    _draw_uv_points(
        reference,
        uv_continuous_kept,
        q_global_kept[:, 2],
        output_dir / "continuous_local_projection.png",
        "global support transformed by tile camera",
    )
    _draw_uv_points(
        reference,
        diagnostics32["unique_uv"][diagnostics32["unique_valid"]],
        diagnostics32["unique_q"][:, 2][diagnostics32["unique_valid"]],
        output_dir / "tile_c32_support_overlay.png",
        "unique projective tile C32 seed support",
    )
    _save_density_image(
        diagnostics32["unique_uv"][diagnostics32["unique_valid"]],
        output_dir / "tile_c32_support_density.png",
        resolution=IMAGE_FLOW,
    )
    _save_quantization_error_image(
        reference,
        uv_continuous_kept[diagnostics32["quantized_valid"]],
        diagnostics32["uv_quantized"][diagnostics32["quantized_valid"]],
        output_dir / "tile_c32_quantization_error.png",
    )
    _draw_uv_points(
        reference,
        diagnostics64["unique_uv"][diagnostics64["unique_valid"]],
        diagnostics64["unique_q"][:, 2][diagnostics64["unique_valid"]],
        output_dir / "projected_global_c64_support_overlay.png",
        "unique projected-global tile C64 support",
    )
    _save_density_image(
        diagnostics64["unique_uv"][diagnostics64["unique_valid"]],
        output_dir / "projected_global_c64_support_density.png",
        resolution=IMAGE_FLOW,
    )
    _save_quantization_error_image(
        reference,
        uv_continuous_kept[diagnostics64["quantized_valid"]],
        diagnostics64["uv_quantized"][diagnostics64["quantized_valid"]],
        output_dir / "tile_c64_quantization_error.png",
    )

    pixel_error32 = diagnostics32["pixel_error"]
    pixel_error64 = diagnostics64["pixel_error"]

    stats = {
        "selected_global_c128_rows": int(selected_coords128.shape[0]),
        "tile_camera": asdict(transform),
        "transform_stats": transform_stats,
        "local_c1024_quantization_stats": quant_stats1024,
        "quantization_stats": quant_stats32,
        "c32_quantization_stats": quant_stats32,
        "c64_quantization_stats": quant_stats64,
        "visible_unique_tile_c32_tokens": int(
            diagnostics32["unique_valid"].sum().item()
        ),
        "local_c1024_tokens": int(coords1024_local.shape[0]),
        "visible_unique_projected_c64_tokens": int(
            diagnostics64["unique_valid"].sum().item()
        ),
        "pixel_error_mean": float(pixel_error32.mean().item()),
        "pixel_error_p95": float(torch.quantile(pixel_error32, 0.95).item()),
        "pixel_error_max": float(pixel_error32.max().item()),
        "c32_pixel_error_mean": float(pixel_error32.mean().item()),
        "c32_pixel_error_p95": float(torch.quantile(pixel_error32, 0.95).item()),
        "c32_pixel_error_max": float(pixel_error32.max().item()),
        "c64_pixel_error_mean": float(pixel_error64.mean().item()),
        "c64_pixel_error_p95": float(torch.quantile(pixel_error64, 0.95).item()),
        "c64_pixel_error_max": float(pixel_error64.max().item()),
    }
    _atomic_json(output_dir / "support_stats.json", stats)
    torch.save(
        {
            "selected_global_c128": selected_coords128.detach().cpu(),
            "q_global": q_global.detach().cpu(),
            "q_local": q_local.detach().cpu(),
            "kept_mask": kept_mask.detach().cpu(),
            "coords1024_local_unique": coords1024_local.detach().cpu(),
            "coords32_unique": coords32.detach().cpu(),
            "coords64_projected_unique": coords64.detach().cpu(),
            "uv_continuous": uv_continuous.detach().cpu(),
            "uv_continuous_kept": uv_continuous_kept.detach().cpu(),
            "uv_quantized": diagnostics32["uv_quantized"].detach().cpu(),
            "pixel_error": pixel_error32.detach().cpu(),
            "uv_quantized_c32": diagnostics32["uv_quantized"].detach().cpu(),
            "pixel_error_c32": pixel_error32.detach().cpu(),
            "uv_quantized_c64": diagnostics64["uv_quantized"].detach().cpu(),
            "pixel_error_c64": pixel_error64.detach().cpu(),
            "tile_camera": asdict(transform),
        },
        output_dir / "support_debug.pt",
    )
    return coords32, coords64, stats


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


def _filter_unique_c32_coords(
    coords: torch.Tensor,
    *,
    source_name: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    return _filter_unique_grid_coords(
        coords,
        resolution=GRID_SS,
        source_name=source_name,
    )


def _project_c32_to_tile_uv(
    coords32: torch.Tensor,
    *,
    transform: TileCameraTransform,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _project_grid_coords_to_tile_uv(
        coords32,
        resolution=GRID_SS,
        transform=transform,
    )


def _filter_tile_native_coords_by_foreground(
    coords32: torch.Tensor,
    *,
    transform: TileCameraTransform,
    foreground_mask: Image.Image,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Remove tile-native tokens that project outside the alpha foreground."""
    expected_size = (
        int(transform.output_width),
        int(transform.output_height),
    )
    if foreground_mask.size != expected_size:
        raise ValueError(
            "tile foreground mask must match the tile camera output: "
            f"{foreground_mask.size} vs {expected_size}"
        )

    uv, q, projection_valid = _project_c32_to_tile_uv(
        coords32,
        transform=transform,
    )
    finite = torch.isfinite(uv).all(dim=1)

    # Match ProjGrid's align_corners=False pixel-center convention:
    # normalized = (projected_pixel + 0.5) / image_size.
    mask_width, mask_height = foreground_mask.size
    pixel_x = torch.floor(
        (uv[:, 0] + 0.5)
        / float(transform.output_width)
        * float(mask_width)
    ).to(torch.long)
    pixel_y = torch.floor(
        (uv[:, 1] + 0.5)
        / float(transform.output_height)
        * float(mask_height)
    ).to(torch.long)
    in_bounds = (
        finite
        & (pixel_x >= 0)
        & (pixel_x < mask_width)
        & (pixel_y >= 0)
        & (pixel_y < mask_height)
    )

    mask_array = np.array(
        foreground_mask.convert("L"),
        dtype=np.uint8,
        copy=True,
    )
    mask_tensor = torch.from_numpy(mask_array > 0)
    foreground = torch.zeros(
        coords32.shape[0],
        dtype=torch.bool,
        device=coords32.device,
    )
    bounded_rows = torch.where(in_bounds)[0]
    if bounded_rows.numel() > 0:
        sampled_foreground = mask_tensor[
            pixel_y[bounded_rows].detach().cpu(),
            pixel_x[bounded_rows].detach().cpu(),
        ].to(device=coords32.device)
        foreground[bounded_rows] = sampled_foreground

    keep = projection_valid & finite & in_bounds & foreground
    filtered = coords32[keep]
    stats = {
        "source": "tile_sample_sparse_structure",
        "input_rows": int(coords32.shape[0]),
        "invalid_projection_rows_dropped": int(
            ((~projection_valid) | (~finite)).sum().item()
        ),
        "out_of_image_rows_dropped": int(
            (projection_valid & finite & ~in_bounds).sum().item()
        ),
        "background_rows_dropped": int(
            (projection_valid & finite & in_bounds & ~foreground).sum().item()
        ),
        "foreground_rows_kept": int(keep.sum().item()),
        "foreground_fraction_of_input": (
            float(keep.float().mean().item()) if keep.numel() else 0.0
        ),
        "mask_rule": "canonical alpha > 0",
        "mask_size": [int(mask_width), int(mask_height)],
    }
    return filtered, keep, uv, q, stats


def _filter_tile_native_by_projected_c32_neighborhood(
    native_coords32: torch.Tensor,
    *,
    projected_coords32: torch.Tensor,
    radius: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Keep native C32 in projected C32 cells or their Chebyshev neighborhood."""
    if int(radius) < 0:
        raise ValueError("C32 neighborhood radius must be non-negative")
    projected_unique, _ = _filter_unique_c32_coords(
        projected_coords32,
        source_name="projected_global_c1024_c32_neighborhood_anchor",
    )
    native_unique, _ = _filter_unique_c32_coords(
        native_coords32,
        source_name="tile_native_c32_neighborhood_candidate",
    )
    if projected_unique.numel() == 0:
        raise RuntimeError("projected-global C32 neighborhood anchor is empty")

    occupancy = torch.zeros(
        (1, 1, GRID_SS, GRID_SS, GRID_SS),
        device=projected_unique.device,
        dtype=torch.float32,
    )
    anchor_xyz = projected_unique[:, 1:4].to(torch.long)
    occupancy[
        0,
        0,
        anchor_xyz[:, 0],
        anchor_xyz[:, 1],
        anchor_xyz[:, 2],
    ] = 1.0
    kernel_size = 2 * int(radius) + 1
    allowed = F.max_pool3d(
        occupancy,
        kernel_size=kernel_size,
        stride=1,
        padding=int(radius),
    )[0, 0] > 0

    native_xyz = native_unique[:, 1:4].to(torch.long)
    exact_match = occupancy[
        0,
        0,
        native_xyz[:, 0],
        native_xyz[:, 1],
        native_xyz[:, 2],
    ] > 0
    keep = allowed[
        native_xyz[:, 0],
        native_xyz[:, 1],
        native_xyz[:, 2],
    ]
    filtered = native_unique[keep]
    stats = {
        "source": "tile_native_c32_projected_global_neighborhood",
        "radius_cells": int(radius),
        "metric": "Chebyshev/L-infinity",
        "projected_anchor_tokens": int(projected_unique.shape[0]),
        "native_input_tokens": int(native_unique.shape[0]),
        "exact_projected_cell_tokens_kept": int(exact_match.sum().item()),
        "neighbor_only_tokens_kept": int((keep & ~exact_match).sum().item()),
        "outside_neighborhood_tokens_dropped": int((~keep).sum().item()),
        "native_tokens_kept": int(filtered.shape[0]),
    }
    return filtered, keep, stats


def _sample_and_constrain_tile_native_c32(
    *,
    pipeline: Any,
    tile_image: Image.Image,
    foreground_mask: Image.Image,
    projected_coords32: torch.Tensor,
    tile_camera: Mapping[str, float],
    transform: TileCameraTransform,
    params: Mapping[str, Mapping[str, Any]],
    seed: int,
    reference: Image.Image,
    output_dir: Path,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Sample tile-native SS and constrain it with foreground + projected C32.

    Only the returned tile-native coordinates are sent through shape512:

        tile-native SS C32
        -> alpha foreground
        -> projected-global C32 same cell or one-cell neighborhood
        -> shape512
    """
    tile_1024 = tile_image.convert("RGB")
    tile_512 = tile_1024.resize((IMAGE_LR, IMAGE_LR), Image.Resampling.LANCZOS)

    cond_ss = pipeline.get_proj_cond_ss(
        [tile_512],
        camera_angle_x=float(tile_camera["camera_angle_x"]),
        distance=float(tile_camera["distance"]),
        mesh_scale=float(tile_camera["mesh_scale"]),
    )
    _seed_everything(int(seed))
    started = time.perf_counter()
    sampled_raw = pipeline.sample_sparse_structure(
        cond_ss,
        resolution=GRID_SS,
        sampler_params=dict(params["ss"]),
    )
    _sync_cuda()
    tile_ss_seconds = time.perf_counter() - started
    del cond_ss, tile_512
    _empty_cuda_cache()

    # Only tile-native support is filtered and later sent through shape512.
    # Projected-global C32 is used solely as a one-cell neighborhood anchor.
    target_device = projected_coords32.device
    sampled_raw_unfiltered = sampled_raw.to(
        device=target_device,
        dtype=torch.int32,
    )
    projected_raw = projected_coords32.to(device=target_device, dtype=torch.int32)
    (
        sampled_raw,
        sampled_foreground_keep,
        sampled_uv_unfiltered,
        sampled_q_unfiltered,
        sampled_foreground_stats,
    ) = _filter_tile_native_coords_by_foreground(
        sampled_raw_unfiltered,
        transform=transform,
        foreground_mask=foreground_mask,
    )

    projected_unique, projected_filter_stats = _filter_unique_c32_coords(
        projected_raw,
        source_name="projected_global_c1024",
    )
    sampled_foreground_unique, sampled_filter_stats = _filter_unique_c32_coords(
        sampled_raw,
        source_name="tile_sample_sparse_structure_after_foreground",
    )
    (
        sampled_constrained,
        sampled_neighborhood_keep,
        neighborhood_stats,
    ) = _filter_tile_native_by_projected_c32_neighborhood(
        sampled_foreground_unique,
        projected_coords32=projected_unique,
        radius=1,
    )
    if sampled_constrained.numel() == 0:
        raise RuntimeError(
            "foreground-filtered tile-native SS has no token inside the "
            "projected-global C32 one-cell neighborhood"
        )

    diagnostic_union = torch.unique(
        torch.cat([projected_unique, sampled_constrained], dim=0),
        dim=0,
    )
    overlap_tokens = int(
        projected_unique.shape[0]
        + sampled_constrained.shape[0]
        - diagnostic_union.shape[0]
    )
    if overlap_tokens < 0:
        raise RuntimeError("negative C32 overlap indicates inconsistent union accounting")

    projected_uv, projected_q, projected_valid = _project_c32_to_tile_uv(
        projected_unique,
        transform=transform,
    )
    sampled_foreground_uv, sampled_foreground_q, sampled_foreground_valid = (
        _project_c32_to_tile_uv(
            sampled_foreground_unique,
            transform=transform,
        )
    )
    sampled_uv, sampled_q, sampled_valid = _project_c32_to_tile_uv(
        sampled_constrained,
        transform=transform,
    )
    diagnostic_union_uv, diagnostic_union_q, diagnostic_union_valid = (
        _project_c32_to_tile_uv(
            diagnostic_union,
            transform=transform,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _draw_uv_points(
        reference,
        sampled_uv_unfiltered,
        sampled_q_unfiltered[:, 2],
        output_dir / "tile_ss_c32_unfiltered_overlay.png",
        "tile-native SS C32 before foreground/neighborhood filters",
    )
    _draw_uv_points(
        reference,
        sampled_foreground_uv[sampled_foreground_valid],
        sampled_foreground_q[:, 2][sampled_foreground_valid],
        output_dir / "tile_ss_c32_foreground_overlay.png",
        "tile-native SS C32 after alpha-foreground filter",
    )
    _draw_uv_points(
        reference,
        sampled_uv[sampled_valid],
        sampled_q[:, 2][sampled_valid],
        output_dir / "tile_ss_c32_overlay.png",
        "tile-native SS C32 constrained by projected C32 radius=1",
    )
    _draw_uv_points(
        reference,
        diagnostic_union_uv[diagnostic_union_valid],
        diagnostic_union_q[:, 2][diagnostic_union_valid],
        output_dir / "tile_c32_union_overlay.png",
        "diagnostic projected C32 + constrained native C32 (shape512 uses native only)",
    )
    _save_density_image(
        sampled_uv[sampled_valid],
        output_dir / "tile_native_c32_constrained_density.png",
        resolution=IMAGE_FLOW,
    )

    stats = {
        "projected_c32_tokens": int(projected_unique.shape[0]),
        "sampled_ss_c32_tokens_before_foreground_filter": int(
            sampled_raw_unfiltered.shape[0]
        ),
        "sampled_ss_c32_tokens_after_foreground_filter": int(
            sampled_foreground_unique.shape[0]
        ),
        "sampled_ss_c32_tokens": int(sampled_constrained.shape[0]),
        "native_c32_shape512_input_tokens": int(sampled_constrained.shape[0]),
        "diagnostic_projected_plus_native_c32_tokens": int(
            diagnostic_union.shape[0]
        ),
        "overlap_c32_tokens": int(overlap_tokens),
        "tile_ss_seconds": float(tile_ss_seconds),
        "projected_filter": projected_filter_stats,
        "sampled_ss_foreground_filter": sampled_foreground_stats,
        "sampled_ss_filter": sampled_filter_stats,
        "sampled_ss_projected_c32_neighborhood_filter": neighborhood_stats,
        "visible_projected_c32_tokens": int(projected_valid.sum().item()),
        "visible_sampled_ss_c32_tokens": int(sampled_valid.sum().item()),
        "visible_diagnostic_union_c32_tokens": int(
            diagnostic_union_valid.sum().item()
        ),
    }
    _atomic_json(output_dir / "tile_native_c32_constraint_stats.json", stats)
    _atomic_json(output_dir / "tile_c32_union_stats.json", stats)
    debug_payload = {
        "projected_coords32_raw": projected_raw.detach().cpu(),
        "projected_coords32_unique": projected_unique.detach().cpu(),
        "sampled_ss_coords32_before_foreground_filter": (
            sampled_raw_unfiltered.detach().cpu()
        ),
        "sampled_ss_foreground_keep_mask": (
            sampled_foreground_keep.detach().cpu()
        ),
        "sampled_ss_uv_before_foreground_filter": (
            sampled_uv_unfiltered.detach().cpu()
        ),
        "sampled_ss_coords32_after_foreground": (
            sampled_foreground_unique.detach().cpu()
        ),
        "sampled_ss_neighborhood_keep_mask": (
            sampled_neighborhood_keep.detach().cpu()
        ),
        "sampled_ss_coords32_constrained": sampled_constrained.detach().cpu(),
        "diagnostic_projected_plus_native_c32": (
            diagnostic_union.detach().cpu()
        ),
        "tile_camera": asdict(transform),
        "seed": int(seed),
    }
    torch.save(
        debug_payload,
        output_dir / "tile_native_c32_constraint_debug.pt",
    )
    torch.save(
        debug_payload,
        output_dir / "tile_c32_union_debug.pt",
    )
    print(
        f"[tile-native-c32] tile={transform.tile_id:02d} "
        f"projected_anchor={projected_unique.shape[0]:,} "
        f"native={sampled_raw_unfiltered.shape[0]:,}"
        f"->{sampled_foreground_unique.shape[0]:,} foreground"
        f"->{sampled_constrained.shape[0]:,} radius1 "
        f"exact_overlap={overlap_tokens:,} "
        f"ss_seconds={tile_ss_seconds:.3f}"
    )
    return sampled_constrained, stats


def _merge_projected_and_native_c64(
    *,
    projected_coords64: torch.Tensor,
    native_coords64: torch.Tensor,
    transform: TileCameraTransform,
    reference: Image.Image,
    output_dir: Path,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Deduplicate projected-global C64 with shape512-learned native C64."""
    target_device = native_coords64.device
    projected_unique, projected_filter_stats = _filter_unique_grid_coords(
        projected_coords64.to(device=target_device, dtype=torch.int32),
        resolution=GRID_SHAPE_1024,
        source_name="projected_global_c1024_local_c64",
    )
    native_unique, native_filter_stats = _filter_unique_grid_coords(
        native_coords64.to(device=target_device, dtype=torch.int32),
        resolution=GRID_SHAPE_1024,
        source_name="tile_native_c32_shape512_learned_c64",
    )
    if native_unique.numel() == 0:
        raise RuntimeError("shape512 produced no valid tile-native C64 token")

    fused = torch.unique(
        torch.cat([projected_unique, native_unique], dim=0),
        dim=0,
    )
    overlap_tokens = int(
        projected_unique.shape[0] + native_unique.shape[0] - fused.shape[0]
    )
    if overlap_tokens < 0:
        raise RuntimeError("negative C64 overlap indicates inconsistent union accounting")

    projected_uv, projected_q, projected_valid = _project_grid_coords_to_tile_uv(
        projected_unique,
        resolution=GRID_SHAPE_1024,
        transform=transform,
    )
    native_uv, native_q, native_valid = _project_grid_coords_to_tile_uv(
        native_unique,
        resolution=GRID_SHAPE_1024,
        transform=transform,
    )
    fused_uv, fused_q, fused_valid = _project_grid_coords_to_tile_uv(
        fused,
        resolution=GRID_SHAPE_1024,
        transform=transform,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _draw_uv_points(
        reference,
        native_uv[native_valid],
        native_q[:, 2][native_valid],
        output_dir / "native_shape512_c64_overlay.png",
        "tile-native C32 -> shape512 -> learned native C64",
    )
    _draw_uv_points(
        reference,
        fused_uv[fused_valid],
        fused_q[:, 2][fused_valid],
        output_dir / "tile_c64_fusion_overlay.png",
        "projected-global C64 + learned native C64 -> deduplicated C64",
    )
    _save_density_image(
        fused_uv[fused_valid],
        output_dir / "tile_c64_fusion_density.png",
        resolution=IMAGE_FLOW,
    )

    stats = {
        "projected_c64_tokens": int(projected_unique.shape[0]),
        "native_c64_tokens": int(native_unique.shape[0]),
        "overlap_c64_tokens": int(overlap_tokens),
        "fused_c64_tokens": int(fused.shape[0]),
        "visible_projected_c64_tokens": int(projected_valid.sum().item()),
        "visible_native_c64_tokens": int(native_valid.sum().item()),
        "visible_fused_c64_tokens": int(fused_valid.sum().item()),
        "projected_filter": projected_filter_stats,
        "native_filter": native_filter_stats,
        "fusion_rule": "exact integer C64 coordinate union followed by unique",
    }
    _atomic_json(output_dir / "tile_c64_fusion_stats.json", stats)
    torch.save(
        {
            "projected_coords64_unique": projected_unique.detach().cpu(),
            "native_coords64_unique": native_unique.detach().cpu(),
            "fused_coords64_unique": fused.detach().cpu(),
            "projected_uv": projected_uv.detach().cpu(),
            "native_uv": native_uv.detach().cpu(),
            "fused_uv": fused_uv.detach().cpu(),
            "tile_camera": asdict(transform),
        },
        output_dir / "tile_c64_fusion_debug.pt",
    )
    print(
        f"[tile-c64-fusion] tile={transform.tile_id:02d} "
        f"projected={projected_unique.shape[0]:,} "
        f"native={native_unique.shape[0]:,} "
        f"overlap={overlap_tokens:,} fused={fused.shape[0]:,}"
    )
    return fused, stats


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
    native_c32_token_count: int,
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
            "Projected-global C64 + learned native C64 -> fused C64\n"
            f"native C32={native_c32_token_count:,} "
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
        tile_render = _label_panel(_resize_panel(None), "Tile cascade\nfailed")
        tile_diff = _label_panel(
            _resize_panel(None), "Tile absolute error\nunavailable"
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
                "Native C32 -> shape512; native+projected C64 -> shape1024+tex\n"
                f"C64 {route_summary.get('tile_projective_c64_tokens')}"
                f"+{route_summary.get('tile_native_c64_tokens')}"
                f"-{route_summary.get('tile_c64_overlap_tokens')}"
                f"={route_summary.get('tile_c64_tokens')}\n"
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
            "Tile cascade absolute RGB error",
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
                native_c32_token_count=int(row.get("tile_c32_tokens") or 0),
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



def run(args: argparse.Namespace) -> None:
    if int(args.tile_size) != DEFAULT_TILE_SIZE:
        raise ValueError(f"this test requires --tile-size={DEFAULT_TILE_SIZE}")
    if int(args.tile_stride) != DEFAULT_TILE_STRIDE:
        raise ValueError(f"this test requires --tile-stride={DEFAULT_TILE_STRIDE}")
    if args.cuda_device is not None:
        torch.cuda.set_device(int(args.cuda_device))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_config = {
        **vars(args),
        "image": str(Path(args.image).expanduser().resolve()),
        "output_dir": str(output_dir),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "moge_model_path": (
            None
            if args.moge_model_path is None
            else str(Path(args.moge_model_path).expanduser().resolve())
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_current_device": (
            int(torch.cuda.current_device()) if torch.cuda.is_available() else None
        ),
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
        "route_modes": {
            "global_geometry": "ordinary_pixal3d_1024",
            "global_crop": "canonical_4096_exact_crop",
            "global_guided_local": (
                "projected_global_c1024_plus_constrained_native_c64_union"
            ),
        },
        "renderer": "Pixal3D PbrMeshRenderer via render_utils.render_frames",
        "metric_protocol": "PSNR+SSIM+LPIPS on shared metric resolution",
    }
    _atomic_json(output_dir / "effective_run_config.json", effective_config)
    print(
        "[effective-config]\n"
        + json.dumps(effective_config, indent=2, sort_keys=True)
    )

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
    if bool(args.save_decoded_support):
        global_decoder_metadata["decoded_support_checkpoint"] = (
            _save_decoded_support_checkpoint(
                global_mesh,
                global_baseline_dir / "decoded_support.pt",
                include_faces=True,
            )
        )

    print("[global] one learned decoder subdivision: C64 latent -> dense C1024 support")
    coords128, upsample_stats = _learned_upsample_shape1024_to_c1024(
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
        tile_foreground_mask = _prepare_tile_foreground_mask(
            foreground_mask_4096,
            box,
            tile_dir,
        )
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

        support_dir = tile_dir / "projective_camera_support_c32"
        selected_coords128 = coords128.index_select(0, rows)
        try:
            (
                projected_coords32,
                projected_coords64,
                support_stats,
            ) = _prepare_projective_tile_supports(
                reference=reference_tile,
                selected_coords128=selected_coords128,
                global_camera=global_camera,
                transform=transform,
                output_dir=support_dir,
                boundary_epsilon=float(args.boundary_epsilon),
            )


            projected_tokens = int(projected_coords32.shape[0])
            if projected_tokens < int(args.min_tile_tokens):
                record = {
                    "status": "skipped",
                    "tile_id": int(tile_id),
                    "box": list(box),
                    "reason": "projected global-C1024-derived C32 below min_tile_tokens",
                    "tile_projective_c32_tokens": projected_tokens,
                    "tile_ss_c32_tokens": 0,
                    "tile_c32_overlap_tokens": 0,
                    "tile_c32_tokens": 0,
                    "tile_projective_c64_tokens": int(
                        projected_coords64.shape[0]
                    ),
                    "tile_native_c64_tokens": 0,
                    "tile_c64_overlap_tokens": 0,
                    "tile_c64_tokens": 0,
                }

                records.append(record)
                _atomic_json(tile_dir / "summary.json", record)

                print(
                    f"[tile-skip] tile={tile_id:02d} "
                    f"projective_C32={projected_tokens:,} "
                    f"(< min_tile_tokens={int(args.min_tile_tokens):,})"
                )

                continue

        except Exception as exc:
            record = {
                "status": "failed",
                "tile_id": int(tile_id),
                "box": list(box),
                "reason": f"C32 support preparation failed: {type(exc).__name__}: {exc}",
                "selected_global_c128_rows": int(rows.numel()),
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
            print(f"[tile-support-error] tile={tile_id:02d}: {record['reason']}")
            _empty_cuda_cache()
            continue

        projected_c32_tokens = int(projected_coords32.shape[0])
        projected_c64_tokens = int(projected_coords64.shape[0])
        outside_fraction = float(
            support_stats["c32_quantization_stats"]["hard_outside_fraction"]
        )
        try:
            native_coords32, native_constraint_stats = (
                _sample_and_constrain_tile_native_c32(
                pipeline=pipeline,
                tile_image=reference_tile,
                foreground_mask=tile_foreground_mask,
                projected_coords32=projected_coords32,
                tile_camera=tile_camera,
                transform=transform,
                params=params,
                seed=seed_tile + 11,
                reference=reference_tile,
                output_dir=support_dir,
                )
            )
            support_stats["tile_native_c32_constraint"] = (
                native_constraint_stats
            )
            _atomic_json(support_dir / "support_stats.json", support_stats)
        except Exception as exc:
            record = {
                "status": "failed",
                "tile_id": int(tile_id),
                "box": list(box),
                "reason": (
                    "tile-native sparse-structure C32 constraint failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "selected_global_c128_rows": int(rows.numel()),
                "tile_projective_c32_tokens": projected_c32_tokens,
                "tile_ss_c32_tokens": 0,
                "tile_c32_overlap_tokens": 0,
                "tile_c32_tokens": 0,
                "tile_projective_c64_tokens": projected_c64_tokens,
                "tile_native_c64_tokens": 0,
                "tile_c64_overlap_tokens": 0,
                "tile_c64_tokens": 0,
                "support_stats": support_stats,
            }
            records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            print(
                f"[tile-native-c32-error] tile={tile_id:02d}: "
                f"{record['reason']}"
            )
            _empty_cuda_cache()
            continue

        sampled_ss_c32_tokens = int(
            native_constraint_stats["sampled_ss_c32_tokens"]
        )
        c32_overlap_tokens = int(
            native_constraint_stats["overlap_c32_tokens"]
        )
        tile_ss_seconds = float(native_constraint_stats["tile_ss_seconds"])
        c32_tokens = int(native_coords32.shape[0])
        if outside_fraction > float(args.max_outside_fraction):
            record = {
                "status": "failed",
                "tile_id": int(tile_id),
                "box": list(box),
                "reason": (
                    f"hard outside fraction {outside_fraction:.6f} exceeds "
                    f"--max-outside-fraction={float(args.max_outside_fraction):.6f}"
                ),
                "selected_global_c128_rows": int(rows.numel()),
                "tile_projective_c32_tokens": projected_c32_tokens,
                "tile_ss_c32_tokens": sampled_ss_c32_tokens,
                "tile_c32_overlap_tokens": c32_overlap_tokens,
                "tile_c32_tokens": c32_tokens,
                "tile_projective_c64_tokens": projected_c64_tokens,
                "tile_native_c64_tokens": 0,
                "tile_c64_overlap_tokens": 0,
                "tile_c64_tokens": 0,
                "tile_ss_seconds": tile_ss_seconds,
                "support_stats": support_stats,
            }
            records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            print(f"[tile-domain-error] tile={tile_id:02d}: {record['reason']}")
            continue
        if c32_tokens > int(args.max_num_tokens):
            record = {
                "status": "failed",
                "tile_id": int(tile_id),
                "box": list(box),
                "reason": (
                    f"tile C32 tokens {c32_tokens} exceed "
                    f"--max-num-tokens={int(args.max_num_tokens)}"
                ),
                "selected_global_c128_rows": int(rows.numel()),
                "tile_projective_c32_tokens": projected_c32_tokens,
                "tile_ss_c32_tokens": sampled_ss_c32_tokens,
                "tile_c32_overlap_tokens": c32_overlap_tokens,
                "tile_c32_tokens": c32_tokens,
                "tile_projective_c64_tokens": projected_c64_tokens,
                "tile_native_c64_tokens": 0,
                "tile_c64_overlap_tokens": 0,
                "tile_c64_tokens": 0,
                "tile_ss_seconds": tile_ss_seconds,
                "support_stats": support_stats,
            }
            records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            print(f"[tile-token-error] tile={tile_id:02d} C32={c32_tokens:,}")
            continue

        route_summary: Optional[Dict[str, Any]] = None
        failure: Optional[str] = None
        try:
            result = _run_tile_official_cascade(
                pipeline=pipeline,
                tile_image=reference_tile,
                native_coords32=native_coords32,
                projected_coords64=projected_coords64,
                tile_camera=tile_camera,
                transform=transform,
                reference=reference_tile,
                support_output_dir=support_dir,
                params=params,
                seed=seed_tile,
                label=f"Tile {tile_id:02d} native-C32/projected-C64 cascade",
                max_tokens=int(args.max_num_tokens),
                projective_c32_tokens=projected_c32_tokens,
                sampled_ss_c32_tokens=sampled_ss_c32_tokens,
                c32_overlap_tokens=c32_overlap_tokens,
                tile_ss_seconds=tile_ss_seconds,
            )
            c64_fusion_stats = json.loads(
                (support_dir / "tile_c64_fusion_stats.json").read_text(
                    encoding="utf-8"
                )
            )
            support_stats["tile_c64_fusion"] = c64_fusion_stats
            _atomic_json(support_dir / "support_stats.json", support_stats)
            route_dir = (
                tile_dir
                / "native_c32_shape512_projected_plus_native_c64_shape1024_texture"
            )
            route_summary = _evaluate_tile_result(
                pipeline=pipeline,
                result=result,
                output_dir=route_dir,
                camera=tile_camera,
                global_camera=global_camera,
                transform=transform,
                seed=seed_tile,
                label=f"Tile {tile_id:02d} native-C32/projected-C64 cascade",
                reference_image=tile_dir / "reference_tile.png",
                args=args,
                envmap=envmap,
            )
            route_summary.update(
                {
                    "tile_projective_c32_tokens": int(
                        result.tile_projective_c32_tokens
                    ),
                    "tile_ss_c32_tokens": int(result.tile_ss_c32_tokens),
                    "tile_c32_overlap_tokens": int(
                        result.tile_c32_overlap_tokens
                    ),
                    "tile_c32_tokens": int(result.tile_c32_tokens),
                    "tile_projective_c64_tokens": int(
                        result.tile_projective_c64_tokens
                    ),
                    "tile_native_c64_tokens": int(
                        result.tile_native_c64_tokens
                    ),
                    "tile_c64_overlap_tokens": int(
                        result.tile_c64_overlap_tokens
                    ),
                    "tile_c64_tokens": int(result.tile_c64_tokens),
                    "tile_ss_seconds": float(result.tile_ss_seconds),
                    "shape512_seconds": float(result.shape512_seconds),
                    "shape1024_seconds": float(result.shape1024_seconds),
                    "texture1024_seconds": float(result.texture1024_seconds),
                    "tile_camera": asdict(transform),
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
            tile_dir / "comparison_reference_baseline_tile_support_diffs.png"
        )
        record = {
            "status": "success" if route_summary is not None else "failed",
            "tile_id": int(tile_id),
            "box": list(box),
            "selected_global_c128_rows": int(rows.numel()),
            "tile_projective_c32_tokens": projected_c32_tokens,
            "tile_ss_c32_tokens": sampled_ss_c32_tokens,
            "tile_c32_overlap_tokens": c32_overlap_tokens,
            "tile_c32_tokens": c32_tokens,
            "tile_projective_c64_tokens": projected_c64_tokens,
            "tile_native_c64_tokens": (
                None
                if route_summary is None
                else route_summary.get("tile_native_c64_tokens")
            ),
            "tile_c64_overlap_tokens": (
                None
                if route_summary is None
                else route_summary.get("tile_c64_overlap_tokens")
            ),
            "tile_c64_tokens": (
                None if route_summary is None else route_summary.get("tile_c64_tokens")
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
            "quantization_pixel_error_mean": support_stats["pixel_error_mean"],
            "quantization_pixel_error_p95": support_stats["pixel_error_p95"],
            "quantization_pixel_error_max": support_stats["pixel_error_max"],
            "tile_dir": str(tile_dir),
            "reference_png": str(tile_dir / "reference_tile.png"),
            "support_overlay_png": str(support_dir / "tile_c64_fusion_overlay.png"),
            "comparison_png": str(comparison_path),
            **baseline_tile_summary,
            "baseline_psnr_db": None,
            "baseline_ssim": None,
            "baseline_lpips": None,
            "psnr_gain_db": None,
            "ssim_gain": None,
            "lpips_reduction": None,
            "psnr_db": None if route_summary is None else route_summary.get("psnr_db"),
            "ssim": None if route_summary is None else route_summary.get("ssim"),
            "lpips": None if route_summary is None else route_summary.get("lpips"),
            "render_png": None if route_summary is None else route_summary.get("render_png"),
            "triptych_png": None if route_summary is None else route_summary.get("triptych_png"),
            "diff_heatmap_png": (
                None
                if route_summary is None
                else route_summary.get("diff_heatmap_png")
            ),
            "tile_ss_seconds": tile_ss_seconds,
            "shape512_seconds": (
                None if route_summary is None else route_summary.get("shape512_seconds")
            ),
            "shape1024_seconds": (
                None if route_summary is None else route_summary.get("shape1024_seconds")
            ),
            "texture1024_seconds": (
                None if route_summary is None else route_summary.get("texture1024_seconds")
            ),
            "renderer": (
                None if route_summary is None else route_summary.get("renderer")
            ),
            "failure": failure,
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
            f"projected_C32={projected_c32_tokens:,} "
            f"native_C32={sampled_ss_c32_tokens:,} "
            f"projected_C64={projected_c64_tokens:,} "
            f"native_C64={record['tile_native_c64_tokens']} "
            f"fused_C64={record['tile_c64_tokens']} "
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
        "format": "pixal3d_global_baseline_vs_tile_cascade_v7_official_renderer",
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
            "transform filtered global C1024 support to centered tile local q",
            "retain local C1024 and quantize the same support independently to projected C32 and C64",
            "tile get_proj_cond_ss + sample_sparse_structure at C32",
            "drop only tile-native C32 projected outside the tile alpha foreground",
            "retain tile-native C32 only in a projected C32 cell or its one-cell 3D neighborhood",
            "run only constrained tile-native C32 through shape512",
            "learned native upsample to C64",
            "union projected-global C64 with learned native C64 and deduplicate",
            "run fused C64 through shape1024 flow",
            "texture1024 flow conditioned on shape1024",
            "shape and texture decode",
            "immediately compare the tile render with the ray-equivalent global baseline crop",
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
                "global q -> global camera point -> 1024 projection -> 4096 crop "
                "-> centered tile back-projection with recomputed local distance "
                "and preserved normalized q_z -> local q -> projected C32 and C64; "
                "tile image independently runs get_proj_cond_ss and "
                "sample_sparse_structure; tile-native C32 is alpha-foreground "
                "filtered and constrained to projected-C32 radius one; only native "
                "C32 runs shape512; its learned native C64 is then unioned and "
                "deduplicated with projected-global C64"
            ),
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
        "min_tile_tokens_c32_projected_from_global_c1024": int(args.min_tile_tokens),
        "min_tile_tokens_applies_to": "tile_projective_c32_tokens",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--moge-model-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
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
        "--min-tile-tokens",
        type=int,
        default=1000,
        help=(
            "minimum projected C32 tokens simplified from the global C1024 "
            "support before tile-native sample_sparse_structure is constrained"
        ),
    )
    parser.add_argument("--max-num-tokens", type=int, default=100000000)
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
    parser.add_argument("--texture-steps", type=int, default=12)
    parser.add_argument("--texture-guidance-strength", type=float, default=1.0)
    parser.add_argument("--texture-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--texture-rescale-t", type=float, default=3.0)

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
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument(
        "--baseline-render-resolution",
        type=int,
        default=IMAGE_CANONICAL,
        help=(
            "Full-frame render resolution for the ordinary global 1024 baseline. "
            "The default 4096 makes each 1024 canonical tile crop retain 1024 "
            "rendered pixels instead of upsampling a 256-pixel crop."
        ),
    )
    parser.add_argument("--metric-resolution", type=int, default=1024)
    parser.add_argument("--render-ssaa", type=int, default=2)
    parser.add_argument("--render-peel-layers", type=int, default=8)
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
    parser.add_argument(
        "--save-decoded-support",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save global and per-tile decoded C1024 O-Voxel support/material "
            "tensors for coordinate-correspondence analysis. No PLY/GLB is made."
        ),
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
    if (
        int(args.render_resolution) < 1
        or int(args.baseline_render_resolution) < 1
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
    run(args)


if __name__ == "__main__":
    main()
