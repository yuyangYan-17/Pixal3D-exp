#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate an ordinary global Pixal3D 1024 baseline and two tile-to-global routes.

Global baseline:

    global SS C32 -> shape512 -> learned C64 -> shape1024 -> texture1024
    -> global decode -> full-image render and metrics

Route A: local SLAT returned to one global decode

    global shape1024 latent
    -> decoder learned subdivision to dense global C1024 support
    -> project global C1024 support to the canonical 4096 image
    -> select C1024 source rows inside each overlapping 1024 tile
    -> exact global-camera / crop / centered-tile-camera recanonicalization
    -> quantize only after the transform, directly to unique local C64
    -> tile shape1024 flow and tile texture1024 flow
    -> gather each local SLAT feature back to its source global C1024 rows
    -> quantize those source rows to global C64 only at return time
    -> keep one candidate per global C64 token by closest tile center
    -> assemble global shape/texture SLAT and decode once globally

Route B: full local decode returned to global object space

    use the same complete local C64 shape/texture SLAT from every successful tile
    -> decode the complete local latent without deleting losing/edge tokens first
    -> convert decoder object coordinates to local q
    -> invert the tile camera mapping to global q and global object coordinates
    -> assign triangles by nearest-tile-center ownership regions
    -> retain an O-Voxel ownership halo, requantize on a global 1024^3 lattice
    -> weld geometry, resolve O-Voxel conflicts by closest tile center
    -> render the merged MeshWithVoxel with the global camera

The important ordering follows the working projected-global tile experiment:
C1024 is transformed continuously into the tile camera before the only local C64
quantization. The code never uses C1024 -> global C64 -> local C64 as tile input.

The tile route does not run tile sparse-structure sampling, tile shape512 flow,
tile-native learned C64 upsampling, or projected/native support fusion. Points
outside the centered local canonical cube are dropped rather than clipped.

Decoded outputs remain native MeshWithVoxel objects and are rendered with
Pixal3D's official render_utils.render_frames / PbrMeshRenderer path. No UVs,
GLB, atlas, Blender process, or Cycles shader reconstruction are involved.

Place this script in the Pixal3D repository root beside inference.py and
render_pixal3d_raw_ovoxel.py.
"""

from __future__ import annotations

import argparse
import copy
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


@dataclass
class TileSupportMapping:
    local_coords64: torch.Tensor
    source_global_coords1024: torch.Tensor
    source_global_coords64: torch.Tensor
    source_to_local_index: torch.Tensor
    tile_center_distance_pixels: torch.Tensor
    stats: Dict[str, Any]


@dataclass
class TileLocalDecodePayload:
    tile_id: int
    transform: TileCameraTransform
    local_coords64: torch.Tensor
    local_shape_denorm_feats: torch.Tensor
    local_texture_denorm_feats: torch.Tensor


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
    coords1024 = torch.unique(coords1024_all[valid], dim=0)
    if coords1024.numel() == 0:
        raise RuntimeError("shape1024 one-step upsample produced no valid C1024 coordinates")

    stats = {
        "source_c64_tokens": int(shape1024_denorm.coords.shape[0]),
        "candidate_rows": int(candidates.shape[0]),
        "valid_candidate_rows": int(valid.sum().item()),
        "discarded_out_of_range_rows": int((~valid).sum().item()),
        "unique_c1024_tokens": int(coords1024.shape[0]),
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
    """Run tile shape1024/texture1024 on mapping-aligned local C64 support.

    The coordinate order is preserved exactly because ``source_to_local_index``
    refers to this order when tile SLAT features are returned to global tokens.
    """
    tile_1024 = tile_image.convert("RGB")
    coords64 = projected_coords64.to(
        device=pipeline.device,
        dtype=torch.int32,
    ).contiguous()
    if coords64.ndim != 2 or coords64.shape[1] != 4:
        raise ValueError(f"{label}: local C64 coordinates must be [N,4]")
    if coords64.numel() == 0:
        raise RuntimeError(f"{label}: local C64 support is empty")
    if bool((coords64[:, 0] != 0).any().item()):
        raise RuntimeError(f"{label}: local C64 support contains nonzero batch ids")
    if bool(
        (
            (coords64[:, 1:] < 0)
            | (coords64[:, 1:] >= GRID_SHAPE_1024)
        ).any().item()
    ):
        raise RuntimeError(f"{label}: local C64 support contains out-of-grid rows")
    unique_count = int(torch.unique(coords64, dim=0).shape[0])
    if unique_count != int(coords64.shape[0]):
        raise RuntimeError(
            f"{label}: local C64 support must already be unique; "
            f"rows={coords64.shape[0]:,}, unique={unique_count:,}"
        )
    if coords64.shape[0] > int(max_tokens):
        raise RuntimeError(
            f"{label}: local C64 support has {coords64.shape[0]:,} tokens; "
            f"exceeds --max-num-tokens={int(max_tokens):,}"
        )

    print(
        f"[tile-local-c64-flow] {label}: "
        f"local_C64={coords64.shape[0]:,} order_preserved=true"
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




def _quantize_global_c1024_support_to_c64(
    coords1024: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Quantize dense global C1024 support to unique global C64 support.

    The returned inverse index maps every C1024 source row to its unique global
    C64 token. Only coordinates are transferred; no global SLAT feature is used
    as a tile feature.
    """
    if coords1024.ndim != 2 or coords1024.shape[1] != 4:
        raise ValueError(
            f"global C1024 support must be [N,4], got {tuple(coords1024.shape)}"
        )
    q1024 = _endpoint_indices_to_q(
        coords1024[:, 1:4],
        GRID_GLOBAL_UPSAMPLED,
    ).to(coords1024.device)
    ids64 = _q_to_endpoint_indices(q1024, GRID_SHAPE_1024)
    coords64_per_source = torch.cat(
        [coords1024[:, :1].to(torch.int32), ids64],
        dim=1,
    )
    valid = (
        (coords64_per_source[:, 0] == 0)
        & (coords64_per_source[:, 1:] >= 0).all(dim=1)
        & (coords64_per_source[:, 1:] < GRID_SHAPE_1024).all(dim=1)
    )
    if not bool(valid.all().item()):
        coords64_per_source = coords64_per_source[valid]
    coords64_unique, inverse = torch.unique(
        coords64_per_source,
        dim=0,
        return_inverse=True,
    )
    if coords64_unique.numel() == 0:
        raise RuntimeError("global C1024 -> global C64 quantization is empty")
    stats = {
        "input_global_c1024_rows": int(coords1024.shape[0]),
        "valid_global_c1024_rows": int(coords64_per_source.shape[0]),
        "unique_global_c64_tokens": int(coords64_unique.shape[0]),
        "c1024_rows_merged_by_global_c64_quantization": int(
            coords64_per_source.shape[0] - coords64_unique.shape[0]
        ),
        "global_c64_resolution": int(GRID_SHAPE_1024),
    }
    return coords64_unique, inverse, stats


def _prepare_global_c1024_tile_mapping(
    *,
    reference: Image.Image,
    selected_global_coords1024: torch.Tensor,
    selected_global_uv_4096: torch.Tensor,
    global_camera: Mapping[str, float],
    transform: TileCameraTransform,
    output_dir: Path,
    boundary_epsilon: float,
) -> TileSupportMapping:
    """Transform dense global C1024 source rows directly to unique local C64.

    This preserves the ordering used by the successful single-tile experiment:

        global C1024 -> continuous camera transform -> local C64 quantization.

    ``source_to_local_index`` maps every retained global C1024 source row to the
    unique local C64 token whose flow feature must be returned to that source.
    A global C64 coordinate is computed only for the return/global-decode route;
    it is never used to construct the local support.
    """
    if selected_global_coords1024.ndim != 2 or selected_global_coords1024.shape[1] != 4:
        raise ValueError("selected_global_coords1024 must be [N,4]")
    if (
        selected_global_uv_4096.ndim != 2
        or selected_global_uv_4096.shape[1] != 2
        or selected_global_uv_4096.shape[0] != selected_global_coords1024.shape[0]
    ):
        raise ValueError("selected_global_uv_4096 must be [N,2] and align with C1024")

    q_global = _endpoint_indices_to_q(
        selected_global_coords1024[:, 1:4],
        GRID_GLOBAL_UPSAMPLED,
    ).to(selected_global_coords1024.device)
    q_local, uv_tile_continuous, _, transform_stats = (
        _global_q_to_centered_tile_q(
            q_global,
            global_camera=global_camera,
            transform=transform,
        )
    )

    overflow = (q_local.abs() - 1.0).clamp_min(0.0)
    strict_inside = (q_local.abs() <= 1.0).all(dim=1)
    hard_outside = (overflow > float(boundary_epsilon)).any(dim=1)
    numeric_outside = (~strict_inside) & (~hard_outside)
    kept = strict_inside
    if not bool(kept.any().item()):
        raise RuntimeError("all selected global C1024 rows leave the local cube")

    q_global_kept = q_global[kept]
    q_local_kept = q_local[kept]
    coords1024_kept = selected_global_coords1024[kept].to(torch.int32)
    global_uv_kept = selected_global_uv_4096[kept]
    uv_tile_kept = uv_tile_continuous[kept]

    local_ids_per_source = _q_to_endpoint_indices(
        q_local_kept,
        GRID_SHAPE_1024,
    )
    local_coords_per_source = torch.cat(
        [
            torch.zeros(
                (local_ids_per_source.shape[0], 1),
                device=local_ids_per_source.device,
                dtype=torch.int32,
            ),
            local_ids_per_source,
        ],
        dim=1,
    )
    local_coords_unique, source_to_local = torch.unique(
        local_coords_per_source,
        dim=0,
        return_inverse=True,
    )
    if local_coords_unique.numel() == 0:
        raise RuntimeError("local C64 support is empty after direct C1024 transform")

    global_ids64_per_source = _q_to_endpoint_indices(
        q_global_kept,
        GRID_SHAPE_1024,
    )
    source_global_coords64 = torch.cat(
        [coords1024_kept[:, :1], global_ids64_per_source],
        dim=1,
    ).to(torch.int32)
    valid_global64 = (
        (source_global_coords64[:, 0] == 0)
        & (source_global_coords64[:, 1:] >= 0).all(dim=1)
        & (source_global_coords64[:, 1:] < GRID_SHAPE_1024).all(dim=1)
    )
    if not bool(valid_global64.all().item()):
        raise RuntimeError("C1024 return mapping produced invalid global C64 coordinates")

    tile_center = torch.tensor(
        [transform.tile_center_full_x, transform.tile_center_full_y],
        device=global_uv_kept.device,
        dtype=global_uv_kept.dtype,
    )
    center_distance = torch.linalg.vector_norm(
        global_uv_kept - tile_center[None],
        dim=1,
    )

    q_quantized_per_source = _endpoint_indices_to_q(
        local_ids_per_source,
        GRID_SHAPE_1024,
    ).to(q_local_kept.device)
    quantized_points = _camera_q_to_points(
        q_quantized_per_source,
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
        uv_quantized - uv_tile_kept,
        dim=1,
    )

    unique_uv, unique_q, unique_valid = _project_grid_coords_to_tile_uv(
        local_coords_unique,
        resolution=GRID_SHAPE_1024,
        transform=transform,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _draw_uv_points(
        reference,
        uv_tile_kept,
        q_global_kept[:, 2],
        output_dir / "continuous_global_c1024_to_local_projection.png",
        "selected global C1024 transformed into centered tile coordinates",
    )
    _draw_uv_points(
        reference,
        unique_uv[unique_valid],
        unique_q[:, 2][unique_valid],
        output_dir / "local_c64_support_overlay.png",
        "global C1024 transformed first, then quantized to unique local C64",
    )
    _save_density_image(
        unique_uv[unique_valid],
        output_dir / "local_c64_support_density.png",
        resolution=IMAGE_FLOW,
    )
    _save_quantization_error_image(
        reference,
        uv_tile_kept[quantized_valid],
        uv_quantized[quantized_valid],
        output_dir / "global_c1024_to_local_c64_quantization_error.png",
    )

    stats = {
        "support_mode": "global_c1024_direct_to_local_c64",
        "selected_global_c1024_rows": int(selected_global_coords1024.shape[0]),
        "kept_source_global_c1024_rows": int(coords1024_kept.shape[0]),
        "unique_return_global_c64_tokens": int(
            torch.unique(source_global_coords64, dim=0).shape[0]
        ),
        "unique_local_c64_tokens": int(local_coords_unique.shape[0]),
        "source_rows_merged_by_local_quantization": int(
            coords1024_kept.shape[0] - local_coords_unique.shape[0]
        ),
        "hard_outside_rows_dropped": int(hard_outside.sum().item()),
        "hard_outside_fraction": float(hard_outside.float().mean().item()),
        "numeric_boundary_rows_dropped": int(numeric_outside.sum().item()),
        "boundary_epsilon": float(boundary_epsilon),
        "tile_center_distance_pixels_min": float(center_distance.min().item()),
        "tile_center_distance_pixels_mean": float(center_distance.mean().item()),
        "tile_center_distance_pixels_max": float(center_distance.max().item()),
        "quantization_pixel_error_mean": float(pixel_error.mean().item()),
        "quantization_pixel_error_p95": float(torch.quantile(pixel_error, 0.95).item()),
        "quantization_pixel_error_max": float(pixel_error.max().item()),
        "transform_stats": transform_stats,
        "tile_camera": asdict(transform),
    }
    _atomic_json(output_dir / "support_stats.json", stats)
    torch.save(
        {
            "selected_global_coords1024": selected_global_coords1024.detach().cpu(),
            "selected_global_uv_4096": selected_global_uv_4096.detach().cpu(),
            "kept_mask": kept.detach().cpu(),
            "source_global_coords1024": coords1024_kept.detach().cpu(),
            "source_global_coords64_return": source_global_coords64.detach().cpu(),
            "local_coords64_unique": local_coords_unique.detach().cpu(),
            "source_to_local_index": source_to_local.detach().cpu(),
            "tile_center_distance_pixels": center_distance.detach().cpu(),
            "q_global": q_global.detach().cpu(),
            "q_local": q_local.detach().cpu(),
            "uv_tile_continuous": uv_tile_continuous.detach().cpu(),
            "uv_tile_quantized_per_source": uv_quantized.detach().cpu(),
            "tile_camera": asdict(transform),
        },
        output_dir / "support_mapping_debug.pt",
    )
    print(
        f"[tile-map] tile={transform.tile_id:02d} "
        f"selected_global_C1024={selected_global_coords1024.shape[0]:,} "
        f"kept_sources={coords1024_kept.shape[0]:,} "
        f"unique_local_C64={local_coords_unique.shape[0]:,} "
        f"outside={stats['hard_outside_fraction']:.6f}"
    )
    return TileSupportMapping(
        local_coords64=local_coords_unique,
        source_global_coords1024=coords1024_kept,
        source_global_coords64=source_global_coords64,
        source_to_local_index=source_to_local.to(torch.long),
        tile_center_distance_pixels=center_distance,
        stats=stats,
    )

def _select_closest_tile_candidate_per_global_token(
    *,
    global_coords: torch.Tensor,
    shape_feats: torch.Tensor,
    texture_feats: torch.Tensor,
    distances: torch.Tensor,
    tile_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Keep exactly one candidate for each global C64 token.

    Ranking is lexicographic: global coordinate, distance to tile center,
    tile id, original candidate order. Therefore the closest tile center wins;
    ties are deterministic and no feature averaging is performed.
    """
    tensors = (global_coords, shape_feats, texture_feats, distances, tile_ids)
    rows = int(global_coords.shape[0])
    if rows == 0:
        raise RuntimeError("no tile SLAT candidates were produced")
    if any(int(t.shape[0]) != rows for t in tensors):
        raise ValueError("candidate tensors must have the same first dimension")
    if global_coords.ndim != 2 or global_coords.shape[1] != 4:
        raise ValueError("candidate global coordinates must be [N,4]")
    if not torch.isfinite(distances).all():
        raise RuntimeError("candidate center distances contain non-finite values")

    coords_cpu = global_coords.to(device="cpu", dtype=torch.int64).contiguous()
    distances_cpu = distances.to(device="cpu", dtype=torch.float64).contiguous()
    tile_ids_cpu = tile_ids.to(device="cpu", dtype=torch.int64).contiguous()
    xyz = coords_cpu[:, 1:4]
    keys = (
        xyz[:, 0] * (GRID_SHAPE_1024 * GRID_SHAPE_1024)
        + xyz[:, 1] * GRID_SHAPE_1024
        + xyz[:, 2]
    ).numpy()
    distance_np = distances_cpu.numpy()
    tile_np = tile_ids_cpu.numpy()
    original_order = np.arange(rows, dtype=np.int64)
    order = np.lexsort((original_order, tile_np, distance_np, keys))
    sorted_keys = keys[order]
    first = np.ones(rows, dtype=bool)
    first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    selected_np = order[first]
    selected = torch.from_numpy(selected_np).to(torch.long)

    counts = np.unique(keys, return_counts=True)[1]
    selected_coords = coords_cpu.index_select(0, selected).to(torch.int32)
    selected_shape = shape_feats.to("cpu").index_select(0, selected).contiguous()
    selected_texture = texture_feats.to("cpu").index_select(0, selected).contiguous()
    selected_distances = distances_cpu.index_select(0, selected).to(torch.float32)
    selected_tile_ids = tile_ids_cpu.index_select(0, selected).to(torch.int32)
    stats = {
        "candidate_rows": rows,
        "unique_global_c64_tokens_kept": int(selected.shape[0]),
        "duplicate_candidates_discarded": int(rows - selected.shape[0]),
        "global_tokens_with_multiple_candidates": int((counts > 1).sum()),
        "maximum_candidates_for_one_global_token": int(counts.max()),
        "selection_rule": (
            "keep one candidate per global C64 token; minimum projected "
            "distance to tile center wins; deterministic tile-id tie break"
        ),
        "feature_averaging_used": False,
        "selected_distance_pixels_min": float(selected_distances.min().item()),
        "selected_distance_pixels_mean": float(selected_distances.mean().item()),
        "selected_distance_pixels_max": float(selected_distances.max().item()),
    }
    return (
        selected_coords,
        selected_shape,
        selected_texture,
        selected_distances,
        selected_tile_ids,
        stats,
    )

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





def _to_3vector(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if tensor.numel() == 1:
        tensor = tensor.repeat(3)
    elif tensor.numel() != 3:
        raise ValueError(f"expected scalar or 3-vector, got shape {tuple(tensor.shape)}")
    return tensor


def _coord4_linear_keys(coords: torch.Tensor, resolution: int = GRID_SHAPE_1024) -> torch.Tensor:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("coords must be [N,4]")
    xyz = coords.to(torch.int64)[:, 1:4]
    return (
        xyz[:, 0] * (resolution * resolution)
        + xyz[:, 1] * resolution
        + xyz[:, 2]
    )


def _coord3_linear_keys(coords: torch.Tensor, shape3: torch.Tensor) -> torch.Tensor:
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be [N,3]")
    shape3 = torch.as_tensor(shape3, dtype=torch.int64).reshape(-1)
    if shape3.numel() == 1:
        shape3 = shape3.repeat(3)
    elif shape3.numel() != 3:
        raise ValueError("shape3 must be scalar or length 3")
    xyz = coords.to(torch.int64)
    return xyz[:, 0] * (int(shape3[1]) * int(shape3[2])) + xyz[:, 1] * int(shape3[2]) + xyz[:, 2]


def _transform_local_object_to_global_object(
    local_object: torch.Tensor,
    *,
    global_camera: Mapping[str, float],
    transform: TileCameraTransform,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Convert decoder local object coordinates to global object coordinates.

    Pixal3D's decoder emits mesh vertices and O-Voxel centers in object space,
    normally about ``[-0.5, 0.5]``. Camera recanonicalization functions operate
    on q coordinates in about ``[-1, 1]``. The correct conversion is:

        q_local = local_object * (2 * tile_mesh_scale)
        q_global = inverse_camera_mapping(q_local)
        global_object = q_global / (2 * global_mesh_scale)

    Treating decoder object coordinates directly as q introduces an exact
    factor-of-two placement error when mesh_scale=1.
    """
    if local_object.ndim != 2 or local_object.shape[1] != 3:
        raise ValueError("local_object must be [N,3]")
    if not torch.isfinite(local_object).all():
        raise RuntimeError("local decoder object coordinates contain non-finite values")

    tile_scale = float(transform.mesh_scale)
    global_scale = float(global_camera["mesh_scale"])
    if tile_scale <= 0.0 or global_scale <= 0.0:
        raise ValueError("mesh scales must be positive")

    local_q = local_object * (2.0 * tile_scale)
    global_q, _, _, stats = _centered_tile_q_to_global_q(
        local_q,
        global_camera=global_camera,
        transform=transform,
        validate_roundtrip=False,
    )
    global_object = global_q / (2.0 * global_scale)
    if not torch.isfinite(global_object).all():
        raise RuntimeError("transformed global object coordinates contain non-finite values")

    stats = {
        **stats,
        "local_object_min": [
            float(v) for v in local_object.amin(dim=0).detach().cpu().tolist()
        ],
        "local_object_max": [
            float(v) for v in local_object.amax(dim=0).detach().cpu().tolist()
        ],
        "local_q_min": [
            float(v) for v in local_q.amin(dim=0).detach().cpu().tolist()
        ],
        "local_q_max": [
            float(v) for v in local_q.amax(dim=0).detach().cpu().tolist()
        ],
        "global_object_min": [
            float(v) for v in global_object.amin(dim=0).detach().cpu().tolist()
        ],
        "global_object_max": [
            float(v) for v in global_object.amax(dim=0).detach().cpu().tolist()
        ],
        "tile_mesh_scale": tile_scale,
        "global_mesh_scale": global_scale,
        "object_q_conversion": (
            "local_q=local_object*(2*tile_mesh_scale); "
            "global_object=global_q/(2*global_mesh_scale)"
        ),
    }
    return global_object, global_q, stats


def _to_voxel_shape3(
    value: Any,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, List[int]]:
    """Normalize decoder voxel-shape metadata to spatial ``[X, Y, Z]``.

    Pixal3D decoder metadata may expose the lattice shape as:
      - a scalar;
      - ``[X, Y, Z]``;
      - ``[1, X, Y, Z]`` with a leading batch dimension;
      - another tensor shape whose final three entries are the spatial axes.

    Sparse O-Voxel coordinates contain only XYZ, so the final three entries
    are the relevant spatial lattice dimensions.
    """
    raw = torch.as_tensor(value, device=device, dtype=torch.int64).reshape(-1)
    raw_values = [int(v) for v in raw.detach().cpu().tolist()]

    if raw.numel() == 0:
        raise ValueError("voxel_shape is empty")
    if raw.numel() == 1:
        shape3 = raw.repeat(3)
    elif raw.numel() >= 3:
        shape3 = raw[-3:]
    else:
        raise ValueError(
            f"voxel_shape must contain 1 or at least 3 entries, got {raw_values}"
        )

    if bool((shape3 <= 0).any().item()):
        raise ValueError(
            f"voxel_shape spatial dimensions must be positive, got {raw_values}"
        )
    return shape3, raw_values


def _quantize_points_to_sparse_voxel_coords(
    points: torch.Tensor,
    *,
    origin: Any,
    voxel_size: Any,
    voxel_shape: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be [N,3]")
    device = points.device
    origin3 = _to_3vector(origin, device=device, dtype=points.dtype)
    voxel_size3 = _to_3vector(voxel_size, device=device, dtype=points.dtype)
    shape3, raw_voxel_shape = _to_voxel_shape3(
        voxel_shape,
        device=device,
    )
    if len(raw_voxel_shape) != 3:
        print(
            "[ovoxel-grid] normalized voxel_shape "
            f"{raw_voxel_shape} -> {[int(v) for v in shape3.detach().cpu().tolist()]}"
        )
    coords = torch.round(
        (points - origin3[None]) / voxel_size3[None] - 0.5
    ).to(torch.int64)
    valid = ((coords >= 0) & (coords < shape3[None])).all(dim=1)
    return coords[valid].to(torch.int32), valid, origin3.to(torch.float32), voxel_size3.to(torch.float32), shape3.to(torch.int32)


def _inside_tile_mask(
    uv_4096: torch.Tensor,
    finite: torch.Tensor,
    box: Sequence[int],
) -> torch.Tensor:
    x0, y0, x1, y1 = (float(v) for v in box)
    return (
        finite
        & (uv_4096[:, 0] >= x0)
        & (uv_4096[:, 0] < x1)
        & (uv_4096[:, 1] >= y0)
        & (uv_4096[:, 1] < y1)
    )


def _filter_faces_by_tile_projection(
    *,
    global_vertices_object: torch.Tensor,
    global_vertices_q: torch.Tensor,
    faces: torch.Tensor,
    global_camera: Mapping[str, float],
    box: Sequence[int],
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Keep triangles whose projected centroid belongs to this tile.

    The local decoder can generate surfaces outside the support-owned image
    region. Centroid ownership makes non-overlapping tiles a deterministic
    partition of image-space geometry and prevents unconditional concatenation
    of every locally decoded surface.
    """
    if int(chunk_size) <= 0:
        raise ValueError("face projection chunk_size must be positive")
    faces = faces.to(device=global_vertices_q.device, dtype=torch.int64)
    _, _, uv_4096, _, finite_vertices = _project_global_q_to_1024_and_4096(
        global_vertices_q,
        global_camera=global_camera,
    )

    kept_chunks: List[torch.Tensor] = []
    for start in range(0, int(faces.shape[0]), int(chunk_size)):
        face_chunk = faces[start : start + int(chunk_size)]
        tri_uv = uv_4096.index_select(0, face_chunk.reshape(-1)).reshape(-1, 3, 2)
        tri_finite = finite_vertices.index_select(
            0, face_chunk.reshape(-1)
        ).reshape(-1, 3).all(dim=1)
        centroid_uv = tri_uv.mean(dim=1)
        keep = _inside_tile_mask(centroid_uv, tri_finite, box)
        if bool(keep.any().item()):
            kept_chunks.append(face_chunk[keep])

    if not kept_chunks:
        return (
            torch.empty((0, 3), device=faces.device, dtype=torch.int64),
            torch.empty((0, 3), device=faces.device, dtype=global_vertices_object.dtype),
            {
                "input_vertices": int(global_vertices_object.shape[0]),
                "input_faces": int(faces.shape[0]),
                "owned_faces": 0,
                "compact_owned_vertices": 0,
                "face_ownership_rule": "projected triangle centroid inside tile box",
            },
        )

    owned_faces_source = torch.cat(kept_chunks, dim=0)
    used_vertices, inverse = torch.unique(
        owned_faces_source.reshape(-1),
        sorted=True,
        return_inverse=True,
    )
    compact_faces = inverse.reshape(-1, 3).to(torch.int64)
    compact_vertices = global_vertices_object.index_select(0, used_vertices)
    stats = {
        "input_vertices": int(global_vertices_object.shape[0]),
        "input_faces": int(faces.shape[0]),
        "owned_faces": int(compact_faces.shape[0]),
        "faces_dropped_outside_tile_ownership": int(
            faces.shape[0] - compact_faces.shape[0]
        ),
        "compact_owned_vertices": int(compact_vertices.shape[0]),
        "unused_vertices_dropped": int(
            global_vertices_object.shape[0] - compact_vertices.shape[0]
        ),
        "face_ownership_rule": "projected triangle centroid inside tile box",
    }
    return compact_faces, compact_vertices, stats


def _canonical_ovoxel_metadata(
    *,
    template_voxel_shape: Any,
    attr_channels: int,
    resolution: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if int(resolution) <= 1:
        raise ValueError("canonical O-Voxel resolution must exceed one")
    raw = torch.as_tensor(template_voxel_shape, dtype=torch.int64).reshape(-1)
    raw_values = [int(v) for v in raw.tolist()]
    if raw.numel() >= 5:
        shape_values = raw_values[:-3] + [int(resolution)] * 3
        if len(shape_values) >= 2:
            shape_values[-4] = int(attr_channels)
    elif raw.numel() == 4:
        shape_values = [int(attr_channels)] + [int(resolution)] * 3
    else:
        # Renderer metadata from Pixal3D normally includes batch and channel
        # dimensions. Build the official dense shape convention explicitly.
        shape_values = [1, int(attr_channels)] + [int(resolution)] * 3

    origin = torch.tensor(
        [-0.5, -0.5, -0.5],
        device=device,
        dtype=torch.float32,
    )
    voxel_size = torch.tensor(
        1.0 / float(resolution),
        device=device,
        dtype=torch.float32,
    )
    voxel_shape = torch.tensor(
        shape_values,
        device=device,
        dtype=torch.int64,
    )
    stats = {
        "template_voxel_shape": raw_values,
        "canonical_voxel_shape": shape_values,
        "canonical_spatial_resolution": int(resolution),
        "canonical_origin": [-0.5, -0.5, -0.5],
        "canonical_voxel_size": 1.0 / float(resolution),
        "canonical_object_bounds": [-0.5, 0.5],
    }
    return origin, voxel_size, voxel_shape, stats


def _select_closest_tile_ovoxel_candidates(
    *,
    coords: torch.Tensor,
    attrs: torch.Tensor,
    distances: torch.Tensor,
    tile_ids: torch.Tensor,
    resolution: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    rows = int(coords.shape[0])
    if rows == 0:
        raise RuntimeError("no O-Voxel candidates were produced")
    if any(int(t.shape[0]) != rows for t in (attrs, distances, tile_ids)):
        raise ValueError("O-Voxel candidate tensors must align")

    coords_cpu = coords.to(device="cpu", dtype=torch.int64).contiguous()
    distances_cpu = distances.to(device="cpu", dtype=torch.float64).contiguous()
    tile_ids_cpu = tile_ids.to(device="cpu", dtype=torch.int64).contiguous()
    keys = _coord3_linear_keys(
        coords_cpu,
        torch.tensor([resolution, resolution, resolution], dtype=torch.int64),
    ).numpy()
    distance_np = distances_cpu.numpy()
    tile_np = tile_ids_cpu.numpy()
    original_order = np.arange(rows, dtype=np.int64)
    order = np.lexsort((original_order, tile_np, distance_np, keys))
    sorted_keys = keys[order]
    first = np.ones(rows, dtype=bool)
    first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    selected_np = order[first]
    selected = torch.from_numpy(selected_np).to(torch.long)

    counts = np.unique(keys, return_counts=True)[1]
    selected_coords = coords_cpu.index_select(0, selected).to(torch.int32)
    selected_attrs = attrs.to("cpu").index_select(0, selected).contiguous()
    selected_distances = distances_cpu.index_select(0, selected).to(torch.float32)
    selected_tile_ids = tile_ids_cpu.index_select(0, selected).to(torch.int32)
    stats = {
        "candidate_rows": rows,
        "unique_global_ovoxels_kept": int(selected.shape[0]),
        "ovoxel_conflicts_discarded": int(rows - selected.shape[0]),
        "ovoxels_with_multiple_candidates": int((counts > 1).sum()),
        "maximum_candidates_for_one_ovoxel": int(counts.max()),
        "selection_rule": (
            "minimum projected distance to tile center; deterministic tile-id tie break"
        ),
        "feature_averaging_used": False,
    }
    return (
        selected_coords,
        selected_attrs,
        selected_distances,
        selected_tile_ids,
        stats,
    )


def _weld_vertices_and_remap_faces(
    *,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    tolerance: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Spatially weld near-identical vertices and remove degenerate faces."""
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be [N,3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be [M,3]")
    if float(tolerance) <= 0.0 or vertices.shape[0] == 0:
        return vertices, faces, {
            "enabled": False,
            "tolerance": float(tolerance),
            "input_vertices": int(vertices.shape[0]),
            "output_vertices": int(vertices.shape[0]),
            "input_faces": int(faces.shape[0]),
            "output_faces": int(faces.shape[0]),
            "degenerate_faces_removed": 0,
        }

    vertices_np = vertices.detach().cpu().numpy().astype(np.float64, copy=False)
    faces_np = faces.detach().cpu().numpy().astype(np.int64, copy=False)
    quantized = np.rint(vertices_np / float(tolerance)).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
    cluster_count = int(inverse.max()) + 1
    sums = np.zeros((cluster_count, 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=cluster_count).astype(np.float64)
    np.add.at(sums, inverse, vertices_np)
    welded_np = sums / counts[:, None]
    remapped_faces_np = inverse[faces_np]
    nondegenerate = (
        (remapped_faces_np[:, 0] != remapped_faces_np[:, 1])
        & (remapped_faces_np[:, 1] != remapped_faces_np[:, 2])
        & (remapped_faces_np[:, 0] != remapped_faces_np[:, 2])
    )
    remapped_faces_np = remapped_faces_np[nondegenerate]
    welded = torch.from_numpy(welded_np.astype(np.float32, copy=False))
    remapped_faces = torch.from_numpy(remapped_faces_np.astype(np.int64, copy=False))
    stats = {
        "enabled": True,
        "tolerance": float(tolerance),
        "input_vertices": int(vertices.shape[0]),
        "output_vertices": int(welded.shape[0]),
        "vertices_welded": int(vertices.shape[0] - welded.shape[0]),
        "input_faces": int(faces.shape[0]),
        "output_faces": int(remapped_faces.shape[0]),
        "degenerate_faces_removed": int(faces.shape[0] - remapped_faces.shape[0]),
    }
    return welded, remapped_faces, stats


def _assign_mesh_field(mesh: Any, name: str, value: Any) -> None:
    """Assign a field on an actual Pixal3D MeshWithVoxel instance."""
    try:
        setattr(mesh, name, value)
    except Exception:
        # Supports frozen dataclasses or slots-backed representation classes.
        object.__setattr__(mesh, name, value)


def _clear_mesh_template_storage(mesh: Any) -> Any:
    """Keep the official runtime type without retaining one tile's large tensors."""
    template = copy.copy(mesh)
    device = torch.device("cpu")
    vertex_dtype = mesh.vertices.dtype
    face_dtype = mesh.faces.dtype
    coord_dtype = mesh.coords.dtype
    attr_dtype = mesh.attrs.dtype
    attr_channels = int(mesh.attrs.shape[1])

    _assign_mesh_field(
        template,
        "vertices",
        torch.empty((0, 3), device=device, dtype=vertex_dtype),
    )
    _assign_mesh_field(
        template,
        "faces",
        torch.empty((0, 3), device=device, dtype=face_dtype),
    )
    _assign_mesh_field(
        template,
        "coords",
        torch.empty((0, 3), device=device, dtype=coord_dtype),
    )
    _assign_mesh_field(
        template,
        "attrs",
        torch.empty((0, attr_channels), device=device, dtype=attr_dtype),
    )
    return template


def _tile_center_ownership_box(
    transform: TileCameraTransform,
    *,
    tile_stride: int,
    canonical_size: int = IMAGE_CANONICAL,
    halo_pixels: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Voronoi ownership box induced by the regular tile-center grid.

    Interior tiles own one stride-sized center region. Boundary tiles extend to
    the canonical image edge. A positive halo is intended for O-Voxel lookup
    support, not for triangle ownership.
    """
    if int(tile_stride) <= 0:
        raise ValueError("tile_stride must be positive")
    if float(halo_pixels) < 0.0:
        raise ValueError("halo_pixels must be non-negative")
    x0, y0, x1, y1 = transform.box
    cx = float(transform.tile_center_full_x)
    cy = float(transform.tile_center_full_y)
    half = float(tile_stride) / 2.0
    left = 0.0 if x0 == 0 else cx - half
    right = float(canonical_size) if x1 == canonical_size else cx + half
    top = 0.0 if y0 == 0 else cy - half
    bottom = float(canonical_size) if y1 == canonical_size else cy + half
    halo = float(halo_pixels)
    return (
        max(0.0, left - halo),
        max(0.0, top - halo),
        min(float(canonical_size), right + halo),
        min(float(canonical_size), bottom + halo),
    )

def _decode_full_tile_payloads_and_merge_to_global(
    *,
    pipeline: Any,
    payloads: Sequence[TileLocalDecodePayload],
    global_camera: Mapping[str, float],
    template_voxel_shape: Any,
    canonical_voxel_resolution: int,
    vertex_weld_tolerance: float,
    face_projection_chunk_size: int,
    tile_stride: int,
    ovoxel_ownership_halo_pixels: int,
) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any]]:
    """Decode every complete local SLAT, then return decoder outputs globally.

    No local C64 token is deleted according to the Route-A global candidate
    selection. The local decoder therefore receives exactly the complete sparse
    latent produced by that tile's shape1024 and texture1024 flows.
    """
    if not payloads:
        raise RuntimeError("no successful tile payloads are available")

    vertex_chunks: List[torch.Tensor] = []
    face_chunks: List[torch.Tensor] = []
    ovoxel_coord_chunks: List[torch.Tensor] = []
    ovoxel_attr_chunks: List[torch.Tensor] = []
    ovoxel_distance_chunks: List[torch.Tensor] = []
    ovoxel_tile_id_chunks: List[torch.Tensor] = []
    tile_decoder_rows: List[Dict[str, Any]] = []
    official_mesh_template: Optional[Any] = None
    official_vertex_dtype: Optional[torch.dtype] = None
    official_face_dtype: Optional[torch.dtype] = None
    official_coord_dtype: Optional[torch.dtype] = None
    official_attr_dtype: Optional[torch.dtype] = None
    canonical_origin: Optional[torch.Tensor] = None
    canonical_voxel_size: Optional[torch.Tensor] = None
    canonical_voxel_shape: Optional[torch.Tensor] = None
    canonical_metadata_stats: Optional[Dict[str, Any]] = None
    vertex_offset = 0

    for payload in sorted(payloads, key=lambda x: int(x.tile_id)):
        local_coords = payload.local_coords64.to(
            device=pipeline.device,
            dtype=torch.int32,
        )
        local_shape_feats = payload.local_shape_denorm_feats.to(pipeline.device)
        local_texture_feats = payload.local_texture_denorm_feats.to(pipeline.device)
        shape_sparse = SparseTensor(feats=local_shape_feats, coords=local_coords)
        texture_sparse = SparseTensor(feats=local_texture_feats, coords=local_coords)
        decoded, mesh = _decode_normal_mesh_with_ovoxel(
            pipeline=pipeline,
            shape_latent=shape_sparse,
            texture_latent=texture_sparse,
            label=f"Tile {int(payload.tile_id):02d} complete local decode",
        )
        del decoded, shape_sparse, texture_sparse, local_shape_feats, local_texture_feats

        if official_mesh_template is None:
            official_mesh_template = _clear_mesh_template_storage(mesh)
            official_vertex_dtype = mesh.vertices.dtype
            official_face_dtype = mesh.faces.dtype
            official_coord_dtype = mesh.coords.dtype
            official_attr_dtype = mesh.attrs.dtype
            (
                canonical_origin,
                canonical_voxel_size,
                canonical_voxel_shape,
                canonical_metadata_stats,
            ) = _canonical_ovoxel_metadata(
                template_voxel_shape=template_voxel_shape,
                attr_channels=int(mesh.attrs.shape[1]),
                resolution=int(canonical_voxel_resolution),
                device=pipeline.device,
            )
            print(
                "[local-decode-merge] captured official mesh template: "
                f"{type(mesh).__module__}.{type(mesh).__name__}"
            )

        if canonical_origin is None or canonical_voxel_size is None or canonical_voxel_shape is None:
            raise RuntimeError("canonical O-Voxel metadata was not initialized")

        ownership_box = _tile_center_ownership_box(
            payload.transform,
            tile_stride=int(tile_stride),
            halo_pixels=0.0,
        )
        ovoxel_box = _tile_center_ownership_box(
            payload.transform,
            tile_stride=int(tile_stride),
            halo_pixels=float(ovoxel_ownership_halo_pixels),
        )

        local_vertices_object = mesh.vertices.to(
            device=pipeline.device,
            dtype=torch.float32,
        )
        (
            global_vertices_object,
            global_vertices_q,
            vertex_transform_stats,
        ) = _transform_local_object_to_global_object(
            local_vertices_object,
            global_camera=global_camera,
            transform=payload.transform,
        )
        compact_faces, compact_vertices, face_stats = _filter_faces_by_tile_projection(
            global_vertices_object=global_vertices_object,
            global_vertices_q=global_vertices_q,
            faces=mesh.faces,
            global_camera=global_camera,
            box=ownership_box,
            chunk_size=int(face_projection_chunk_size),
        )

        coords_local = mesh.coords.to(pipeline.device, dtype=torch.float32)
        attrs_local = mesh.attrs.to(pipeline.device)
        local_origin3 = _to_3vector(
            mesh.origin,
            device=pipeline.device,
            dtype=torch.float32,
        )
        local_voxel_size3 = _to_3vector(
            mesh.voxel_size,
            device=pipeline.device,
            dtype=torch.float32,
        )
        local_voxel_object = (
            local_origin3[None]
            + (coords_local + 0.5) * local_voxel_size3[None]
        )
        (
            global_voxel_object,
            global_voxel_q,
            voxel_transform_stats,
        ) = _transform_local_object_to_global_object(
            local_voxel_object,
            global_camera=global_camera,
            transform=payload.transform,
        )
        _, _, voxel_uv_4096, _, voxel_finite = _project_global_q_to_1024_and_4096(
            global_voxel_q,
            global_camera=global_camera,
        )
        voxel_owned = _inside_tile_mask(
            voxel_uv_4096,
            voxel_finite,
            ovoxel_box,
        )
        global_voxel_object_owned = global_voxel_object[voxel_owned]
        attrs_owned = attrs_local[voxel_owned]
        uv_owned = voxel_uv_4096[voxel_owned]

        quantized_coords3, valid_canonical, _, _, _ = (
            _quantize_points_to_sparse_voxel_coords(
                global_voxel_object_owned,
                origin=canonical_origin,
                voxel_size=canonical_voxel_size,
                voxel_shape=canonical_voxel_shape,
            )
        )
        quantized_attrs = attrs_owned[valid_canonical]
        uv_quantized_candidates = uv_owned[valid_canonical]

        if quantized_coords3.shape[0] == 0 or compact_faces.shape[0] == 0:
            tile_decoder_rows.append({
                "tile_id": int(payload.tile_id),
                "status": "skipped_after_decode",
                "reason": "no owned triangles or no canonical O-Voxels",
                "complete_local_c64_tokens": int(payload.local_coords64.shape[0]),
                "decoder_vertices": int(mesh.vertices.shape[0]),
                "decoder_faces": int(mesh.faces.shape[0]),
                "decoder_active_voxels": int(mesh.coords.shape[0]),
                "owned_faces": int(compact_faces.shape[0]),
                "owned_canonical_ovoxels": int(quantized_coords3.shape[0]),
                "triangle_ownership_box": list(ownership_box),
                "ovoxel_ownership_box": list(ovoxel_box),
            })
            del mesh
            _empty_cuda_cache()
            continue

        tile_center = torch.tensor(
            [
                payload.transform.tile_center_full_x,
                payload.transform.tile_center_full_y,
            ],
            device=uv_quantized_candidates.device,
            dtype=uv_quantized_candidates.dtype,
        )
        quantized_distances = torch.linalg.vector_norm(
            uv_quantized_candidates - tile_center[None],
            dim=1,
        )

        compact_faces = compact_faces + int(vertex_offset)
        vertex_chunks.append(compact_vertices.detach().cpu())
        face_chunks.append(compact_faces.detach().cpu())
        vertex_offset += int(compact_vertices.shape[0])
        ovoxel_coord_chunks.append(quantized_coords3.detach().cpu())
        ovoxel_attr_chunks.append(quantized_attrs.detach().cpu())
        ovoxel_distance_chunks.append(quantized_distances.detach().cpu())
        ovoxel_tile_id_chunks.append(
            torch.full(
                (int(quantized_coords3.shape[0]),),
                int(payload.tile_id),
                dtype=torch.int32,
            )
        )

        tile_decoder_rows.append({
            "tile_id": int(payload.tile_id),
            "status": "success",
            "complete_local_c64_tokens": int(payload.local_coords64.shape[0]),
            "decoder_vertices": int(mesh.vertices.shape[0]),
            "decoder_faces": int(mesh.faces.shape[0]),
            "decoder_active_voxels": int(mesh.coords.shape[0]),
            **face_stats,
            "triangle_ownership_box": list(ownership_box),
            "ovoxel_ownership_box": list(ovoxel_box),
            "ovoxel_ownership_halo_pixels": int(ovoxel_ownership_halo_pixels),
            "owned_ovoxel_centers_with_halo": int(voxel_owned.sum().item()),
            "owned_ovoxels_outside_canonical_cube": int(
                voxel_owned.sum().item() - valid_canonical.sum().item()
            ),
            "reprojected_global_active_voxels": int(quantized_coords3.shape[0]),
            "vertex_object_q_conversion": vertex_transform_stats.get(
                "object_q_conversion"
            ),
            "vertex_global_object_min": vertex_transform_stats.get(
                "global_object_min"
            ),
            "vertex_global_object_max": vertex_transform_stats.get(
                "global_object_max"
            ),
            "ovoxel_global_object_min": voxel_transform_stats.get(
                "global_object_min"
            ),
            "ovoxel_global_object_max": voxel_transform_stats.get(
                "global_object_max"
            ),
        })
        del (
            mesh,
            local_vertices_object,
            global_vertices_object,
            global_vertices_q,
            compact_faces,
            compact_vertices,
            coords_local,
            attrs_local,
            local_voxel_object,
            global_voxel_object,
            global_voxel_q,
            quantized_coords3,
            quantized_attrs,
            quantized_distances,
        )
        _empty_cuda_cache()

    success_rows = [row for row in tile_decoder_rows if row.get("status") == "success"]
    if not success_rows:
        raise RuntimeError("no complete tile local decode produced mergeable metadata")
    if not vertex_chunks or not face_chunks or not ovoxel_coord_chunks:
        raise RuntimeError("complete tile local decode produced no mergeable geometry")

    concatenated_vertices = torch.cat(vertex_chunks, dim=0).to(torch.float32)
    concatenated_faces = torch.cat(face_chunks, dim=0).to(torch.int64)
    merged_vertices, merged_faces, weld_stats = _weld_vertices_and_remap_faces(
        vertices=concatenated_vertices,
        faces=concatenated_faces,
        tolerance=float(vertex_weld_tolerance),
    )

    all_ovoxel_coords = torch.cat(ovoxel_coord_chunks, dim=0).to(torch.int32)
    all_ovoxel_attrs = torch.cat(ovoxel_attr_chunks, dim=0)
    all_ovoxel_distances = torch.cat(ovoxel_distance_chunks, dim=0).to(torch.float32)
    all_ovoxel_tile_ids = torch.cat(ovoxel_tile_id_chunks, dim=0).to(torch.int32)
    (
        merged_ovoxel_coords,
        merged_ovoxel_attrs,
        merged_ovoxel_distances,
        merged_ovoxel_tile_ids,
        ovoxel_selection_stats,
    ) = _select_closest_tile_ovoxel_candidates(
        coords=all_ovoxel_coords,
        attrs=all_ovoxel_attrs,
        distances=all_ovoxel_distances,
        tile_ids=all_ovoxel_tile_ids,
        resolution=int(canonical_voxel_resolution),
    )

    if official_mesh_template is None:
        raise RuntimeError("no official MeshWithVoxel template was captured")
    if (
        official_vertex_dtype is None
        or official_face_dtype is None
        or official_coord_dtype is None
        or official_attr_dtype is None
        or canonical_origin is None
        or canonical_voxel_size is None
        or canonical_voxel_shape is None
        or canonical_metadata_stats is None
    ):
        raise RuntimeError("official MeshWithVoxel metadata is incomplete")

    render_device = pipeline.device
    merged_mesh = official_mesh_template
    _assign_mesh_field(
        merged_mesh,
        "vertices",
        merged_vertices.to(
            device=render_device,
            dtype=official_vertex_dtype,
            non_blocking=True,
        ),
    )
    _assign_mesh_field(
        merged_mesh,
        "faces",
        merged_faces.to(
            device=render_device,
            dtype=official_face_dtype,
            non_blocking=True,
        ),
    )
    _assign_mesh_field(
        merged_mesh,
        "coords",
        merged_ovoxel_coords.to(
            device=render_device,
            dtype=official_coord_dtype,
            non_blocking=True,
        ),
    )
    _assign_mesh_field(
        merged_mesh,
        "attrs",
        merged_ovoxel_attrs.to(
            device=render_device,
            dtype=official_attr_dtype,
            non_blocking=True,
        ),
    )
    _assign_mesh_field(merged_mesh, "origin", canonical_origin)
    _assign_mesh_field(merged_mesh, "voxel_size", canonical_voxel_size)
    _assign_mesh_field(merged_mesh, "voxel_shape", canonical_voxel_shape)

    summary = {
        "attempted_complete_tile_local_decodes": len(payloads),
        "successful_complete_tile_local_decodes": len(success_rows),
        "concatenated_owned_vertices_before_weld": int(
            concatenated_vertices.shape[0]
        ),
        "concatenated_owned_faces_before_weld": int(
            concatenated_faces.shape[0]
        ),
        "merged_vertices": int(merged_vertices.shape[0]),
        "merged_faces": int(merged_faces.shape[0]),
        "vertex_welding": weld_stats,
        "merged_ovoxel_rows_before_conflict_selection": int(
            all_ovoxel_coords.shape[0]
        ),
        "merged_ovoxel_rows_after_conflict_selection": int(
            merged_ovoxel_coords.shape[0]
        ),
        "ovoxel_candidate_selection": ovoxel_selection_stats,
        "selected_ovoxel_tile_ids_min": int(merged_ovoxel_tile_ids.min().item()),
        "selected_ovoxel_tile_ids_max": int(merged_ovoxel_tile_ids.max().item()),
        "selected_ovoxel_center_distance_mean": float(
            merged_ovoxel_distances.mean().item()
        ),
        "canonical_ovoxel_metadata": canonical_metadata_stats,
        "tile_stride": int(tile_stride),
        "ovoxel_ownership_halo_pixels": int(ovoxel_ownership_halo_pixels),
        "merged_mesh_runtime_type": (
            f"{type(merged_mesh).__module__}.{type(merged_mesh).__name__}"
        ),
        "merged_mesh_device": str(merged_mesh.vertices.device),
        "merge_route": (
            "complete local SLAT decode -> local object to local q -> inverse tile "
            "camera mapping -> global object; nearest-center triangle ownership; "
            "O-Voxel halo and global requantization; weld and conflict selection"
        ),
    }
    return merged_mesh, tile_decoder_rows, summary

def _save_local_decode_merge_comparison_sheet(
    *,
    reference_path: Path,
    baseline_render_path: Path,
    merged_render_path: Path,
    covered_support_overlay_path: Path,
    baseline_diff_path: Path,
    merged_diff_path: Path,
    baseline_metrics: Mapping[str, Any],
    merged_metrics: Mapping[str, Any],
    coverage_fraction: float,
    output_path: Path,
) -> str:
    panels = [
        _label_panel(
            _resize_panel(reference_path),
            "Reference canonical 1024",
        ),
        _label_panel(
            _resize_panel(baseline_render_path),
            (
                "Ordinary global Pixal3D baseline\n"
                f"PSNR={baseline_metrics.get('psnr_db')} "
                f"SSIM={baseline_metrics.get('ssim')} "
                f"LPIPS={baseline_metrics.get('lpips')}"
            ),
        ),
        _label_panel(
            _resize_panel(merged_render_path),
            (
                "Tile-local decode, decoder-metadata merged globally\n"
                f"coverage={coverage_fraction:.4%}\n"
                f"PSNR={merged_metrics.get('psnr_db')} "
                f"SSIM={merged_metrics.get('ssim')} "
                f"LPIPS={merged_metrics.get('lpips')}"
            ),
        ),
        _label_panel(
            _resize_panel(covered_support_overlay_path),
            "Final selected global C64 support",
        ),
        _label_panel(
            _resize_panel(baseline_diff_path),
            "Baseline absolute RGB error",
        ),
        _label_panel(
            _resize_panel(merged_diff_path),
            "Merged tile-local-decode absolute RGB error",
        ),
    ]
    size = panels[0].width
    canvas = Image.new("RGB", (size * 3, size * 2), (18, 18, 18))
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % 3) * size, (index // 3) * size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return str(output_path)


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



def _save_global_merge_comparison_sheet(
    *,
    reference_path: Path,
    baseline_render_path: Path,
    merged_render_path: Path,
    covered_support_overlay_path: Path,
    baseline_diff_path: Path,
    merged_diff_path: Path,
    baseline_metrics: Mapping[str, Any],
    merged_metrics: Mapping[str, Any],
    coverage_fraction: float,
    output_path: Path,
) -> str:
    panels = [
        _label_panel(
            _resize_panel(reference_path),
            "Reference canonical 1024",
        ),
        _label_panel(
            _resize_panel(baseline_render_path),
            (
                "Ordinary global Pixal3D baseline\n"
                f"PSNR={baseline_metrics.get('psnr_db')} "
                f"SSIM={baseline_metrics.get('ssim')} "
                f"LPIPS={baseline_metrics.get('lpips')}"
            ),
        ),
        _label_panel(
            _resize_panel(merged_render_path),
            (
                "Tile SLAT returned to global; one global decode\n"
                f"coverage={coverage_fraction:.4%}\n"
                f"PSNR={merged_metrics.get('psnr_db')} "
                f"SSIM={merged_metrics.get('ssim')} "
                f"LPIPS={merged_metrics.get('lpips')}"
            ),
        ),
        _label_panel(
            _resize_panel(covered_support_overlay_path),
            "Final covered global C64 support",
        ),
        _label_panel(
            _resize_panel(baseline_diff_path),
            "Baseline absolute RGB error",
        ),
        _label_panel(
            _resize_panel(merged_diff_path),
            "Merged tile-SLAT absolute RGB error",
        ),
    ]
    size = panels[0].width
    canvas = Image.new("RGB", (size * 3, size * 2), (18, 18, 18))
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % 3) * size, (index // 3) * size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return str(output_path)


def run(args: argparse.Namespace) -> None:
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
        "[global-baseline] SS C32 -> shape512 -> learned C64 -> "
        "shape1024 -> texture1024 -> decode"
    )
    coords32, official_coords64, global_shape = (
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
    global_texture_norm, global_texture_denorm, global_texture_seconds = (
        _run_texture1024(
            pipeline=pipeline,
            image_1024=image_1024,
            coords64=official_coords64,
            camera=global_camera,
            shape_norm=global_shape.shape_norm,
            params=params,
            seed=int(args.seed) + 301,
            description="Global ordinary texture 1024",
        )
    )

    baseline_dir = output_dir / "global_baseline_1024"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_meshes, baseline_mesh = _decode_normal_mesh_with_ovoxel(
        pipeline=pipeline,
        shape_latent=global_shape.shape_denorm,
        texture_latent=global_texture_denorm,
        label="Global ordinary 1024 baseline",
    )
    baseline_decoder_metadata = {
        "decoder_vertices": int(baseline_mesh.vertices.shape[0]),
        "decoder_faces": int(baseline_mesh.faces.shape[0]),
        "active_voxels": int(baseline_mesh.coords.shape[0]),
        "sample_type": type(baseline_mesh).__name__,
        "renderer": "pixal3d.utils.render_utils.render_frames",
    }
    baseline_decoder_grid = {
        "origin": torch.as_tensor(baseline_mesh.origin).detach().cpu(),
        "voxel_size": torch.as_tensor(baseline_mesh.voxel_size).detach().cpu(),
        "voxel_shape": torch.as_tensor(baseline_mesh.voxel_shape).detach().cpu(),
    }

    print("[global-support] shape1024 latent -> dense global C1024 support")
    coords1024, upsample_stats = _learned_upsample_shape1024_to_c1024(
        pipeline,
        global_shape.shape_denorm,
    )
    if coords1024.shape[0] > int(args.max_num_tokens):
        raise RuntimeError(
            f"global C1024 support has {coords1024.shape[0]:,} tokens; "
            f"exceeds --max-num-tokens={int(args.max_num_tokens):,}"
        )
    global_coords64, c1024_to_c64_inverse, global_c64_stats = (
        _quantize_global_c1024_support_to_c64(coords1024)
    )
    print(
        f"[global-support] dense C1024={coords1024.shape[0]:,}; "
        f"return/decode global C64 domain={global_coords64.shape[0]:,}"
    )

    prior_dir = output_dir / "global_geometry_prior"
    prior_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "coords32_official": coords32.detach().cpu(),
            "coords64_official": official_coords64.detach().cpu(),
            "coords1024_dense": coords1024.detach().cpu(),
            "coords64_return_domain": global_coords64.detach().cpu(),
            "c1024_to_global_c64_inverse": c1024_to_c64_inverse.detach().cpu(),
            "global_shape1024_norm_feats": global_shape.shape_norm.feats.detach().cpu(),
            "global_shape1024_denorm_feats": global_shape.shape_denorm.feats.detach().cpu(),
        },
        prior_dir / "global_geometry_prior.pt",
    )

    baseline_metric = render_and_evaluate_mesh(
        baseline_mesh,
        camera_angle_x=float(global_camera["camera_angle_x"]),
        distance=float(global_camera["distance"]),
        output_dir=baseline_dir / "aligned_eval",
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
    baseline_extras = _save_extra_comparisons(
        Path(str(baseline_metric["original_png"])),
        Path(str(baseline_metric["render_png"])),
        baseline_dir / "comparisons",
    )
    baseline_summary = {
        **baseline_decoder_metadata,
        **baseline_metric,
        **baseline_extras,
        "route": (
            "ordinary global SS C32 -> shape512 -> learned C64 -> "
            "shape1024 -> texture1024 -> global decode"
        ),
        "global_c32_tokens": int(coords32.shape[0]),
        "official_global_c64_tokens": int(official_coords64.shape[0]),
        "dense_global_c1024_tokens": int(coords1024.shape[0]),
        "return_domain_global_c64_tokens": int(global_coords64.shape[0]),
        "shape512_seconds": float(global_shape.shape512_seconds),
        "shape1024_seconds": float(global_shape.shape1024_seconds),
        "texture1024_seconds": float(global_texture_seconds),
    }
    _atomic_json(baseline_dir / "summary.json", baseline_summary)
    baseline_render_path = Path(str(baseline_metric["render_png"]))

    del baseline_meshes, baseline_mesh
    del global_texture_norm, global_texture_denorm
    del global_shape
    _empty_cuda_cache()

    q_global1024 = _endpoint_indices_to_q(
        coords1024[:, 1:4],
        GRID_GLOBAL_UPSAMPLED,
    ).to(coords1024.device)
    _, _, uv_global1024_4096, _, finite_global1024 = (
        _project_global_q_to_1024_and_4096(
            q_global1024,
            global_camera=global_camera,
        )
    )
    if not bool(finite_global1024.any().item()):
        raise RuntimeError("dense global C1024 support has no finite projection")

    q_global64 = _endpoint_indices_to_q(
        global_coords64[:, 1:4],
        GRID_SHAPE_1024,
    ).to(global_coords64.device)
    _, _, uv_global64_4096, _, finite_global64 = (
        _project_global_q_to_1024_and_4096(
            q_global64,
            global_camera=global_camera,
        )
    )
    dense_support_overlay_path = prior_dir / "all_dense_global_c1024_support.png"
    return_domain_overlay_path = prior_dir / "all_return_domain_global_c64_support.png"
    _draw_uv_points(
        image_4096,
        uv_global1024_4096[finite_global1024],
        q_global1024[:, 2][finite_global1024],
        dense_support_overlay_path,
        "dense global C1024 support used to construct every tile",
    )
    _draw_uv_points(
        image_4096,
        uv_global64_4096[finite_global64],
        q_global64[:, 2][finite_global64],
        return_domain_overlay_path,
        "global C64 domain used only after local SLAT return",
    )
    _atomic_json(
        prior_dir / "summary.json",
        {
            "one_step_upsample": upsample_stats,
            "c1024_to_global_c64_return_domain": global_c64_stats,
            "dense_global_c1024_support_overlay": str(dense_support_overlay_path),
            "return_domain_global_c64_support_overlay": str(return_domain_overlay_path),
            "finite_projected_global_c1024_rows": int(finite_global1024.sum().item()),
            "finite_projected_return_domain_global_c64_rows": int(finite_global64.sum().item()),
        },
    )

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

    candidate_global_coords: List[torch.Tensor] = []
    candidate_shape_feats: List[torch.Tensor] = []
    candidate_texture_feats: List[torch.Tensor] = []
    candidate_distances: List[torch.Tensor] = []
    candidate_tile_ids: List[torch.Tensor] = []
    records: List[Dict[str, Any]] = []
    tile_local_decode_payloads: List[TileLocalDecodePayload] = []
    attempted = 0

    for tile_id, box in enumerate(boxes):
        if selected_ids is not None and tile_id not in selected_ids:
            continue
        if args.max_tiles is not None and attempted >= int(args.max_tiles):
            break
        attempted += 1

        tile_dir = output_dir / "tiles" / f"tile_{tile_id:02d}"
        reference_tile = _prepare_tile_reference(image_4096, box, tile_dir)
        rows = _rows_inside_tile(uv_global1024_4096, finite_global1024, box)
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
        print(
            f"[tile {tile_id:02d}] selected global C1024={rows.numel():,} "
            f"box={box} centered_fov={math.degrees(transform.camera_angle_x):.6f}deg"
        )

        if rows.numel() == 0:
            record = {
                "status": "skipped",
                "tile_id": int(tile_id),
                "box": list(box),
                "reason": "no dense global C1024 projection inside tile",
                "selected_global_c1024_rows": 0,
                "kept_source_global_c1024_rows": 0,
                "unique_local_c64_tokens": 0,
                "returned_candidate_rows": 0,
            }
            records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            continue

        selected_global_coords1024 = coords1024.index_select(0, rows)
        selected_global_uv_4096 = uv_global1024_4096.index_select(0, rows)
        support_dir = tile_dir / "global_c1024_direct_to_local_c64_support"
        try:
            mapping = _prepare_global_c1024_tile_mapping(
                reference=reference_tile,
                selected_global_coords1024=selected_global_coords1024,
                selected_global_uv_4096=selected_global_uv_4096,
                global_camera=global_camera,
                transform=transform,
                output_dir=support_dir,
                boundary_epsilon=float(args.boundary_epsilon),
            )
        except Exception as exc:
            record = {
                "status": "failed",
                "tile_id": int(tile_id),
                "box": list(box),
                "reason": f"support mapping failed: {type(exc).__name__}: {exc}",
                "selected_global_c1024_rows": int(rows.numel()),
                "kept_source_global_c1024_rows": 0,
                "unique_local_c64_tokens": 0,
                "returned_candidate_rows": 0,
            }
            records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            print(f"[tile-map-error] tile={tile_id:02d}: {record['reason']}")
            _empty_cuda_cache()
            continue

        local_tokens = int(mapping.local_coords64.shape[0])
        outside_fraction = float(mapping.stats["hard_outside_fraction"])
        if local_tokens < int(args.min_tile_tokens):
            record = {
                "status": "skipped",
                "tile_id": int(tile_id),
                "box": list(box),
                "reason": "unique local C64 below min_tile_tokens",
                "selected_global_c1024_rows": int(rows.numel()),
                "kept_source_global_c1024_rows": int(
                    mapping.source_global_coords1024.shape[0]
                ),
                "unique_local_c64_tokens": local_tokens,
                "returned_candidate_rows": 0,
                "hard_outside_fraction": outside_fraction,
            }
            records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            del mapping
            _empty_cuda_cache()
            continue
        if outside_fraction > float(args.max_outside_fraction):
            record = {
                "status": "failed",
                "tile_id": int(tile_id),
                "box": list(box),
                "reason": (
                    f"hard outside fraction {outside_fraction:.6f} exceeds "
                    f"{float(args.max_outside_fraction):.6f}"
                ),
                "selected_global_c1024_rows": int(rows.numel()),
                "kept_source_global_c1024_rows": int(
                    mapping.source_global_coords1024.shape[0]
                ),
                "unique_local_c64_tokens": local_tokens,
                "returned_candidate_rows": 0,
                "hard_outside_fraction": outside_fraction,
            }
            records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            del mapping
            _empty_cuda_cache()
            continue
        if local_tokens > int(args.max_num_tokens):
            raise RuntimeError(
                f"tile {tile_id:02d} local C64 has {local_tokens:,} tokens; "
                f"exceeds --max-num-tokens={int(args.max_num_tokens):,}"
            )

        seed_tile = int(args.seed) + tile_id * 1000 + 1
        try:
            result = _run_tile_projected_c64_only(
                pipeline=pipeline,
                tile_image=reference_tile,
                projected_coords64=mapping.local_coords64,
                tile_camera=tile_camera,
                params=params,
                seed=seed_tile,
                label=f"Tile {tile_id:02d} global-C1024-direct-local-C64",
                max_tokens=int(args.max_num_tokens),
            )
            expected_local_coords = mapping.local_coords64.to(
                device=result.shape_denorm.coords.device,
                dtype=torch.int32,
            )
            if not torch.equal(result.shape_denorm.coords, expected_local_coords):
                raise RuntimeError("tile shape SLAT coordinate order changed")
            if not torch.equal(result.texture_denorm.coords, expected_local_coords):
                raise RuntimeError("tile texture SLAT coordinates changed")

            source_to_local = mapping.source_to_local_index.to(
                device=result.shape_denorm.feats.device,
                dtype=torch.long,
            )
            returned_shape = result.shape_denorm.feats.index_select(
                0,
                source_to_local,
            )
            returned_texture = result.texture_denorm.feats.index_select(
                0,
                source_to_local,
            )
            returned_rows = int(mapping.source_global_coords1024.shape[0])
            if returned_shape.shape[0] != returned_rows:
                raise RuntimeError("returned shape candidate rows do not align")
            if returned_texture.shape[0] != returned_rows:
                raise RuntimeError("returned texture candidate rows do not align")

            candidate_global_coords.append(
                mapping.source_global_coords64.detach().cpu().to(torch.int32)
            )
            candidate_shape_feats.append(returned_shape.detach().cpu())
            candidate_texture_feats.append(returned_texture.detach().cpu())
            candidate_distances.append(
                mapping.tile_center_distance_pixels.detach().cpu().to(torch.float32)
            )
            candidate_tile_ids.append(
                torch.full(
                    (returned_rows,),
                    int(tile_id),
                    dtype=torch.int32,
                )
            )

            local_slat_path = tile_dir / "local_shape_texture_slat.pt"
            torch.save(
                {
                    "tile_id": int(tile_id),
                    "tile_camera": asdict(transform),
                    "local_coords64": result.shape_denorm.coords.detach().cpu(),
                    "local_shape_denorm_feats": result.shape_denorm.feats.detach().cpu(),
                    "local_texture_denorm_feats": result.texture_denorm.feats.detach().cpu(),
                    "source_global_coords1024": mapping.source_global_coords1024.detach().cpu(),
                    "source_global_coords64_return": mapping.source_global_coords64.detach().cpu(),
                    "source_to_local_index": mapping.source_to_local_index.detach().cpu(),
                    "tile_center_distance_pixels": mapping.tile_center_distance_pixels.detach().cpu(),
                },
                local_slat_path,
            )
            tile_local_decode_payloads.append(
                TileLocalDecodePayload(
                    tile_id=int(tile_id),
                    transform=transform,
                    local_coords64=result.shape_denorm.coords.detach().cpu().to(torch.int32),
                    local_shape_denorm_feats=result.shape_denorm.feats.detach().cpu(),
                    local_texture_denorm_feats=result.texture_denorm.feats.detach().cpu(),
                )
            )

            record = {
                "status": "success",
                "tile_id": int(tile_id),
                "box": list(box),
                "selected_global_c1024_rows": int(rows.numel()),
                "kept_source_global_c1024_rows": returned_rows,
                "unique_return_global_c64_tokens": int(
                    torch.unique(mapping.source_global_coords64, dim=0).shape[0]
                ),
                "unique_local_c64_tokens": local_tokens,
                "source_rows_merged_by_local_quantization": int(
                    mapping.stats["source_rows_merged_by_local_quantization"]
                ),
                "returned_candidate_rows": returned_rows,
                "hard_outside_fraction": outside_fraction,
                "tile_center_distance_pixels_mean": mapping.stats[
                    "tile_center_distance_pixels_mean"
                ],
                "quantization_pixel_error_mean": mapping.stats[
                    "quantization_pixel_error_mean"
                ],
                "shape1024_seconds": float(result.shape1024_seconds),
                "texture1024_seconds": float(result.texture1024_seconds),
                "complete_local_decode_deferred": True,
                "slat_returned_to_global": True,
                "local_slat_pt": str(local_slat_path),
                "support_overlay_png": str(
                    support_dir / "local_c64_support_overlay.png"
                ),
            }
            records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            print(
                f"[tile-return] tile={tile_id:02d} local_C64={local_tokens:,} "
                f"C1024_source_candidates={returned_rows:,}"
            )
            result.shape_norm = None  # type: ignore[assignment]
            result.shape_denorm = None  # type: ignore[assignment]
            result.texture_norm = None  # type: ignore[assignment]
            result.texture_denorm = None  # type: ignore[assignment]
            del returned_shape, returned_texture, source_to_local, result, mapping
            _empty_cuda_cache()
        except Exception as exc:
            record = {
                "status": "failed",
                "tile_id": int(tile_id),
                "box": list(box),
                "reason": f"tile flow/return failed: {type(exc).__name__}: {exc}",
                "selected_global_c1024_rows": int(rows.numel()),
                "kept_source_global_c1024_rows": int(
                    mapping.source_global_coords1024.shape[0]
                ),
                "unique_local_c64_tokens": local_tokens,
                "returned_candidate_rows": 0,
                "hard_outside_fraction": outside_fraction,
            }
            records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            print(f"[tile-flow-error] tile={tile_id:02d}: {record['reason']}")
            del mapping
            _empty_cuda_cache()

    successful_records = [row for row in records if row.get("status") == "success"]
    if not candidate_global_coords:
        raise RuntimeError("no successful tile returned SLAT candidates")

    all_candidate_coords = torch.cat(candidate_global_coords, dim=0)
    all_candidate_shape = torch.cat(candidate_shape_feats, dim=0)
    all_candidate_texture = torch.cat(candidate_texture_feats, dim=0)
    all_candidate_distance = torch.cat(candidate_distances, dim=0)
    all_candidate_tile_ids = torch.cat(candidate_tile_ids, dim=0)

    (
        merged_coords64_cpu,
        merged_shape_feats_cpu,
        merged_texture_feats_cpu,
        selected_distances_cpu,
        selected_tile_ids_cpu,
        selection_stats,
    ) = _select_closest_tile_candidate_per_global_token(
        global_coords=all_candidate_coords,
        shape_feats=all_candidate_shape,
        texture_feats=all_candidate_texture,
        distances=all_candidate_distance,
        tile_ids=all_candidate_tile_ids,
    )
    total_global_c64 = int(global_coords64.shape[0])
    covered_global_c64 = int(merged_coords64_cpu.shape[0])
    uncovered_global_c64 = int(total_global_c64 - covered_global_c64)
    coverage_fraction = float(covered_global_c64 / max(total_global_c64, 1))
    print(
        f"[global-slat-merge] source_candidates={all_candidate_coords.shape[0]:,} "
        f"kept_global_C64={covered_global_c64:,} coverage={coverage_fraction:.6f}"
    )

    merge_dir = output_dir / "merged_global_tile_slat"
    merge_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "coords64_global": merged_coords64_cpu,
            "shape_denorm_feats": merged_shape_feats_cpu,
            "texture_denorm_feats": merged_texture_feats_cpu,
            "selected_tile_ids": selected_tile_ids_cpu,
            "selected_tile_center_distance_pixels": selected_distances_cpu,
            "selection_stats": selection_stats,
            "coverage": {
                "all_return_domain_global_c64_tokens": total_global_c64,
                "covered_global_c64_tokens": covered_global_c64,
                "uncovered_global_c64_tokens_discarded": uncovered_global_c64,
                "coverage_fraction": coverage_fraction,
            },
        },
        merge_dir / "merged_global_slat.pt",
    )

    merged_coords64 = merged_coords64_cpu.to(
        device=pipeline.device,
        dtype=torch.int32,
    )
    merged_shape = SparseTensor(
        feats=merged_shape_feats_cpu.to(pipeline.device),
        coords=merged_coords64,
    )
    merged_texture = SparseTensor(
        feats=merged_texture_feats_cpu.to(pipeline.device),
        coords=merged_coords64,
    )
    merged_meshes, merged_mesh = _decode_normal_mesh_with_ovoxel(
        pipeline=pipeline,
        shape_latent=merged_shape,
        texture_latent=merged_texture,
        label="Merged tile SLAT returned to global C64",
    )
    merged_decoder_metadata = {
        "decoder_vertices": int(merged_mesh.vertices.shape[0]),
        "decoder_faces": int(merged_mesh.faces.shape[0]),
        "active_voxels": int(merged_mesh.coords.shape[0]),
        "sample_type": type(merged_mesh).__name__,
        "renderer": "pixal3d.utils.render_utils.render_frames",
    }
    del merged_shape, merged_texture
    _empty_cuda_cache()

    merged_metric = render_and_evaluate_mesh(
        merged_mesh,
        camera_angle_x=float(global_camera["camera_angle_x"]),
        distance=float(global_camera["distance"]),
        output_dir=merge_dir / "aligned_eval",
        reference_image=output_dir / "canonical_1024.png",
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
    merged_extras = _save_extra_comparisons(
        Path(str(merged_metric["original_png"])),
        Path(str(merged_metric["render_png"])),
        merge_dir / "comparisons",
    )
    del merged_meshes, merged_mesh
    _empty_cuda_cache()

    merged_q_global = _endpoint_indices_to_q(
        merged_coords64_cpu[:, 1:4],
        GRID_SHAPE_1024,
    ).to(pipeline.device)
    _, _, merged_uv_4096, _, merged_finite = _project_global_q_to_1024_and_4096(
        merged_q_global,
        global_camera=global_camera,
    )
    covered_support_overlay_path = merge_dir / "covered_global_c64_support.png"
    _draw_uv_points(
        image_4096,
        merged_uv_4096[merged_finite],
        merged_q_global[:, 2][merged_finite],
        covered_support_overlay_path,
        "global C64 tokens carrying returned local shape/texture SLAT",
    )

    baseline_psnr = baseline_metric.get("psnr_db")
    baseline_ssim = baseline_metric.get("ssim")
    baseline_lpips = baseline_metric.get("lpips")
    merged_psnr = merged_metric.get("psnr_db")
    merged_ssim = merged_metric.get("ssim")
    merged_lpips = merged_metric.get("lpips")
    metric_gains = {
        "psnr_gain_db": (
            None
            if baseline_psnr is None or merged_psnr is None
            else float(merged_psnr) - float(baseline_psnr)
        ),
        "ssim_gain": (
            None
            if baseline_ssim is None or merged_ssim is None
            else float(merged_ssim) - float(baseline_ssim)
        ),
        "lpips_reduction": (
            None
            if baseline_lpips is None or merged_lpips is None
            else float(baseline_lpips) - float(merged_lpips)
        ),
    }

    comparison_path = output_dir / "comparison_global_baseline_vs_merged_tile_slat.png"
    _save_global_merge_comparison_sheet(
        reference_path=output_dir / "canonical_1024.png",
        baseline_render_path=baseline_render_path,
        merged_render_path=Path(str(merged_metric["render_png"])),
        covered_support_overlay_path=covered_support_overlay_path,
        baseline_diff_path=Path(str(baseline_extras["diff_heatmap_png"])),
        merged_diff_path=Path(str(merged_extras["diff_heatmap_png"])),
        baseline_metrics=baseline_metric,
        merged_metrics=merged_metric,
        coverage_fraction=coverage_fraction,
        output_path=comparison_path,
    )

    merged_summary = {
        **merged_decoder_metadata,
        **merged_metric,
        **merged_extras,
        **metric_gains,
        "route": (
            "global C1024 direct transform to each local C64 -> local shape/texture "
            "flow -> gather features to source C1024 -> return-time global C64 "
            "quantization -> closest-center candidate -> one global decode"
        ),
        "candidate_selection": selection_stats,
        "all_return_domain_global_c64_tokens": total_global_c64,
        "covered_global_c64_tokens": covered_global_c64,
        "uncovered_global_c64_tokens_discarded": uncovered_global_c64,
        "coverage_fraction": coverage_fraction,
        "covered_support_overlay_png": str(covered_support_overlay_path),
        "merged_global_slat_pt": str(merge_dir / "merged_global_slat.pt"),
        "comparison_png": str(comparison_path),
    }
    _atomic_json(merge_dir / "summary.json", merged_summary)

    local_decode_merge_dir = output_dir / "merged_complete_local_tile_decodes"
    local_decode_merge_dir.mkdir(parents=True, exist_ok=True)
    (
        local_decoded_merged_mesh,
        local_decode_tile_rows,
        local_decode_merge_stats,
    ) = _decode_full_tile_payloads_and_merge_to_global(
        pipeline=pipeline,
        payloads=tile_local_decode_payloads,
        global_camera=global_camera,
        template_voxel_shape=baseline_decoder_grid["voxel_shape"],
        canonical_voxel_resolution=int(
            args.local_decode_canonical_voxel_resolution
        ),
        vertex_weld_tolerance=float(args.vertex_weld_tolerance),
        face_projection_chunk_size=int(args.face_projection_chunk_size),
        tile_stride=int(args.tile_stride),
        ovoxel_ownership_halo_pixels=int(args.ovoxel_ownership_halo_pixels),
    )
    local_decode_pre_render = {
        **local_decode_merge_stats,
        "render_face_chunk_size": int(
            args.local_decode_render_face_chunk_size
        ),
        "tile_decodes": local_decode_tile_rows,
    }
    _atomic_json(
        local_decode_merge_dir / "merge_stats_pre_render.json",
        local_decode_pre_render,
    )
    local_decode_face_count = int(local_decoded_merged_mesh.faces.shape[0])
    local_decode_face_chunk_size = int(
        args.local_decode_render_face_chunk_size
    )
    local_decode_render_chunks = (
        1
        if local_decode_face_chunk_size <= 0
        else (
            local_decode_face_count + local_decode_face_chunk_size - 1
        )
        // local_decode_face_chunk_size
    )
    print(
        "[local-decode-render] "
        f"vertices={int(local_decoded_merged_mesh.vertices.shape[0]):,} "
        f"faces={local_decode_face_count:,} "
        f"face_chunk_size={local_decode_face_chunk_size:,} "
        f"chunks={local_decode_render_chunks:,}"
    )
    local_decode_metric = render_and_evaluate_mesh(
        local_decoded_merged_mesh,
        camera_angle_x=float(global_camera["camera_angle_x"]),
        distance=float(global_camera["distance"]),
        output_dir=local_decode_merge_dir / "aligned_eval",
        reference_image=output_dir / "canonical_1024.png",
        resolution=int(args.render_resolution),
        metric_resolution=int(args.metric_resolution),
        envmap=envmap,
        envmap_name=str(args.envmap),
        ssaa=int(args.render_ssaa),
        peel_layers=int(args.render_peel_layers),
        face_chunk_size=local_decode_face_chunk_size,
        use_envmap_bg=bool(args.use_envmap_bg),
        lpips_net=str(args.lpips_net),
        metric_device=str(args.metric_device),
        skip_lpips=bool(args.skip_lpips),
    )
    local_decode_extras = _save_extra_comparisons(
        Path(str(local_decode_metric["original_png"])),
        Path(str(local_decode_metric["render_png"])),
        local_decode_merge_dir / "comparisons",
    )
    del local_decoded_merged_mesh
    _empty_cuda_cache()

    local_decode_metric_gains = {
        "psnr_gain_db": (
            None
            if baseline_psnr is None or local_decode_metric.get("psnr_db") is None
            else float(local_decode_metric["psnr_db"]) - float(baseline_psnr)
        ),
        "ssim_gain": (
            None
            if baseline_ssim is None or local_decode_metric.get("ssim") is None
            else float(local_decode_metric["ssim"]) - float(baseline_ssim)
        ),
        "lpips_reduction": (
            None
            if baseline_lpips is None or local_decode_metric.get("lpips") is None
            else float(baseline_lpips) - float(local_decode_metric["lpips"])
        ),
    }
    local_decode_compare_png = (
        output_dir / "comparison_global_baseline_vs_complete_local_decode_merge.png"
    )
    _save_local_decode_merge_comparison_sheet(
        reference_path=output_dir / "canonical_1024.png",
        baseline_render_path=baseline_render_path,
        merged_render_path=Path(str(local_decode_metric["render_png"])),
        covered_support_overlay_path=covered_support_overlay_path,
        baseline_diff_path=Path(str(baseline_extras["diff_heatmap_png"])),
        merged_diff_path=Path(str(local_decode_extras["diff_heatmap_png"])),
        baseline_metrics=baseline_metric,
        merged_metrics=local_decode_metric,
        coverage_fraction=coverage_fraction,
        output_path=local_decode_compare_png,
    )
    local_decode_summary = {
        **local_decode_metric,
        **local_decode_extras,
        **local_decode_metric_gains,
        **local_decode_merge_stats,
        "route": (
            "complete local shape/texture SLAT -> complete local decode -> "
            "object/q inverse camera mapping -> nearest-center spatial ownership -> "
            "global O-Voxel requantization and global-camera render"
        ),
        "comparison_png": str(local_decode_compare_png),
        "covered_support_overlay_png": str(covered_support_overlay_path),
        "tile_local_decode_rows_csv": str(
            local_decode_merge_dir / "tile_local_decode_records.csv"
        ),
        "tile_local_decode_records": local_decode_tile_rows,
    }
    _write_csv(
        local_decode_merge_dir / "tile_local_decode_records.csv",
        local_decode_tile_rows,
    )
    _atomic_json(local_decode_merge_dir / "summary.json", local_decode_summary)

    aggregate_csv = output_dir / "tile_flow_records.csv"
    _write_csv(aggregate_csv, records)
    summary = {
        "format": "pixal3d_c1024_direct_local_slat_and_complete_local_decode_return_global_v1",
        "image": str(Path(args.image).expanduser().resolve()),
        "global_camera": global_camera,
        "tile_layout": {
            "canonical_size": IMAGE_CANONICAL,
            "tile_size": int(args.tile_size),
            "tile_stride": int(args.tile_stride),
            "tiles_total": len(boxes),
            "overlap_pixels": max(0, int(args.tile_size) - int(args.tile_stride)),
        },
        "global_support_route": {
            "official_global_c64_tokens": int(official_coords64.shape[0]),
            "dense_global_c1024_tokens": int(coords1024.shape[0]),
            "return_domain_global_c64_tokens": total_global_c64,
            "c1024_to_global_c64_return_domain": global_c64_stats,
            "one_step_upsample": upsample_stats,
        },
        "tile_route": [
            "select dense global C1024 support by projection inside tile",
            "recanonicalize global C1024 q to centered local camera q",
            "quantize only after the transform to unique local C64",
            "run complete tile shape1024 and texture1024 flows",
            "save one complete local shape/texture SLAT per tile",
            "gather local features back to source C1024 rows",
            "quantize source rows to global C64 only for return/global decode",
            "retain closest-center candidate per global C64 and decode once globally",
            "independently decode each complete local SLAT and merge decoder outputs globally",
        ],
        "duplicate_policy": {
            "feature_averaging": False,
            "kept_candidates_per_global_token": 1,
            "ranking": "minimum projected source-C1024 distance to tile center",
            "tie_break": "smaller tile id, then original candidate order",
        },
        "local_decode_policy": {
            "decode_complete_local_latent": True,
            "predecode_global_winner_filtering": False,
            "triangle_ownership": "nearest regular tile center",
            "ovoxel_ownership_halo_pixels": int(args.ovoxel_ownership_halo_pixels),
        },
        "attempted_tiles": attempted,
        "successful_tiles": len(successful_records),
        "skipped_tiles": sum(row.get("status") == "skipped" for row in records),
        "failed_tiles": sum(row.get("status") == "failed" for row in records),
        "candidate_selection": selection_stats,
        "coverage": {
            "all_return_domain_global_c64_tokens": total_global_c64,
            "covered_global_c64_tokens": covered_global_c64,
            "uncovered_global_c64_tokens_discarded": uncovered_global_c64,
            "coverage_fraction": coverage_fraction,
        },
        "global_baseline_1024": baseline_summary,
        "merged_global_tile_slat": merged_summary,
        "merged_complete_local_tile_decodes": local_decode_summary,
        "metric_gains_over_baseline": {
            "merged_global_tile_slat": metric_gains,
            "merged_complete_local_tile_decodes": local_decode_metric_gains,
        },
        "comparison_png": str(comparison_path),
        "tile_flow_records_csv": str(aggregate_csv),
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
        help="comma-separated tile ids; omitted means all 49 overlapping tiles",
    )
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument(
        "--min-tile-tokens",
        type=int,
        default=1000,
        help=(
            "minimum unique local C64 tokens obtained after transforming "
            "dense global C1024 support into a tile and quantizing local C64"
        ),
    )
    parser.add_argument("--max-num-tokens", type=int, default=100000000)
    parser.add_argument(
        "--local-decode-canonical-voxel-resolution",
        type=int,
        default=1024,
        help=(
            "Full canonical O-Voxel lattice resolution for merged tile-local "
            "decoder metadata. Uses origin=-0.5 and voxel_size=1/resolution."
        ),
    )
    parser.add_argument(
        "--face-projection-chunk-size",
        type=int,
        default=500000,
        help="Triangle chunk size for projected tile-ownership filtering.",
    )
    parser.add_argument(
        "--ovoxel-ownership-halo-pixels",
        type=int,
        default=32,
        help=(
            "Canonical-4096 image-space halo retained around each nearest-center "
            "triangle ownership region for O-Voxel material lookup support."
        ),
    )
    parser.add_argument(
        "--vertex-weld-tolerance",
        type=float,
        default=1e-6,
        help=(
            "Global object-space tolerance for welding near-identical vertices "
            "after tile-owned triangle cropping; <=0 disables welding."
        ),
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
        "--local-decode-render-face-chunk-size",
        type=int,
        default=4_000_000,
        help=(
            "Maximum faces per nvdiffrast call when rendering the merged "
            "complete local decodes. The default stays below nvdiffrast's "
            "fixed 2^24-subtriangle limit; zero disables chunking."
        ),
    )
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
    if (
        int(args.render_resolution) < 1
        or int(args.baseline_render_resolution) < 1
        or int(args.metric_resolution) < 1
        or int(args.render_ssaa) < 1
        or int(args.render_peel_layers) < 1
        or int(args.local_decode_render_face_chunk_size) < 0
    ):
        raise ValueError(
            "render resolutions, metric resolution, SSAA, and peel layers "
            "must be positive; local decode render face chunk size must be "
            "non-negative"
        )
    if float(args.boundary_epsilon) < 0:
        raise ValueError("--boundary-epsilon must be non-negative")
    if int(args.ovoxel_ownership_halo_pixels) < 0:
        raise ValueError("--ovoxel-ownership-halo-pixels must be non-negative")
    if not (0.0 <= float(args.max_outside_fraction) <= 1.0):
        raise ValueError("--max-outside-fraction must be in [0,1]")
    run(args)


if __name__ == "__main__":
    main()
