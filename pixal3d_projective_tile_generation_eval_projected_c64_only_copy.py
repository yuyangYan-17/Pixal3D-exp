#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a normal global Pixal3D 1024 baseline, then compare three tile routes:

1. projected-global C64 support -> tile shape/texture generation;
2. the tile image directly -> the complete ordinary Pixal3D tile cascade;
3. a deliberately modified route that keeps global geometry, discards global
   material, and decodes tile texture with every projected global subdivision.

The global baseline keeps the complete ordinary Pixal3D 1024 route:

    global SS C32 -> shape512 -> learned C64 -> shape1024 -> texture1024
    -> shape/texture decode -> full-image render and metrics

The projected-C64 tile route deliberately disables every tile-native
support-generation stage:

    global shape1024 -> decoder learned subdivision -> dense global C1024 support
    -> select points whose global projection lies inside the tile
    -> exact global-camera / crop / centered-tile-camera recanonicalization
    -> quantize the transformed points directly to unique local C64 coordinates
    -> tile shape1024 flow
    -> tile texture1024 flow conditioned on tile shape1024
    -> shape/texture decode

The projected-C64 tile route does NOT run:

    tile get_proj_cond_ss
    tile sample_sparse_structure
    tile-native C32 foreground filtering
    tile-native C32 neighborhood filtering
    tile shape512 flow
    tile learned C32-to-C64 upsampling
    projected/native C64 fusion

Thus the only sparse coordinates entering that route's shape1024 and
texture1024 flows are C64 coordinates obtained by quantizing transformed global
C1024 support. The tile image is still used as the image condition for
shape1024 and texture1024. The separate direct-tile comparison does run the
ordinary tile-native sparse-structure and shape512 stages.

No point-cloud centroid normalization, bounding-box normalization, or boundary
clipping is used. Transformed points outside the centered tile canonical cube
are dropped before C64 quantization.

For evaluation, decoded outputs remain native MeshWithVoxel objects and are
rendered with Pixal3D's official render_utils.render_frames / PbrMeshRenderer
path. No UVs, GLB, atlas, Blender process, or Cycles shader reconstruction are
involved.

The modified material-render route is explicitly separated from the ordinary
decode/render routes. It projects the global shape decoder's valid C1024
subdivision leaves with the exact O-Voxel coordinate convention and coarsens
them to tile-local C64. A new tile shape SLat is generated from the tile image
on that C64 support, and the tile image texture flow uses that regenerated shape
SLat. The four texture-decoder guides are coarsened from the projected global
C1024 leaves so their parent/child rows remain exact after crop magnification.
Decoded attributes are assigned back to their source global C1024 leaves. The
subimage observation keeps global geometry and uses the exact off-axis
global-camera crop rays; the full view uses the complete global camera and
displays geometry without decoded tile material using camera-space normal
color.

Expected repository layout
--------------------------
Place this script in the Pixal3D repository root beside inference.py and
render_pixal3d_raw_ovoxel.py.
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
from pixal3d.representations import MeshWithVoxel  # noqa: E402
from pixal3d.utils import render_utils  # noqa: E402
from render_pixal3d_raw_ovoxel import (  # noqa: E402
    load_envmap,
    render_and_evaluate_mesh,
    save_render_outputs,
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
    ss_seconds: float
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
class ProjectedGlobalGuideHierarchy:
    guide_subs: List[SparseTensor]
    final_local_coords: torch.Tensor
    final_global_coords: torch.Tensor
    final_global_to_local_rows: torch.Tensor
    layer_stats: List[Dict[str, Any]]


@dataclass
class ProjectedGlobalTextureSupport:
    coords64: torch.Tensor
    final_local_coords: torch.Tensor
    final_global_coords: torch.Tensor
    source_to_final_local_rows: torch.Tensor
    projection_stats: Dict[str, Any]


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


@torch.no_grad()
def _render_and_evaluate_global_camera_tile_crop(
    mesh: MeshWithVoxel,
    *,
    global_camera: Mapping[str, float],
    transform: TileCameraTransform,
    output_dir: Path,
    reference_image: Path,
    resolution: int,
    metric_resolution: int,
    envmap: Any,
    envmap_name: str,
    ssaa: int,
    peel_layers: int,
    use_envmap_bg: bool,
    lpips_net: str,
    metric_device: str,
    skip_lpips: bool,
) -> Dict[str, Any]:
    """Render the tile's exact off-axis crop rays in the global camera frame."""
    if mesh.device.type != "cuda":
        mesh = mesh.cuda()
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.cuda.device(mesh.device):
        extrinsics, _ = render_utils.proj_camera_to_render_params(
            camera_angle_x=float(global_camera["camera_angle_x"]),
            distance=float(global_camera["distance"]),
        )
        intrinsics = torch.tensor(
            [
                [
                    float(transform.fx) / float(transform.output_width),
                    0.0,
                    float(transform.offaxis_cx)
                    / float(transform.output_width),
                ],
                [
                    0.0,
                    float(transform.fy) / float(transform.output_height),
                    float(transform.offaxis_cy)
                    / float(transform.output_height),
                ],
                [0.0, 0.0, 1.0],
            ],
            device=mesh.device,
            dtype=torch.float32,
        )
        near = max(0.01, float(global_camera["distance"]) - 2.0)
        far = float(global_camera["distance"]) + 10.0
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        renders = render_utils.render_frames(
            mesh,
            extrinsics=[extrinsics],
            intrinsics=[intrinsics],
            options={
                "resolution": int(resolution),
                "near": float(near),
                "far": float(far),
                "ssaa": int(ssaa),
                "peel_layers": int(peel_layers),
                "face_chunk_size": 0,
            },
            verbose=True,
            envmap=envmap,
            use_envmap_bg=bool(use_envmap_bg),
        )
        finished.record()
        torch.cuda.synchronize()
        render_seconds = float(started.elapsed_time(finished) / 1000.0)
    output_paths = save_render_outputs(renders, output_dir)

    with Image.open(reference_image) as image:
        original = _composite_on_black(image)
    target_size = (int(resolution), int(resolution))
    if original.size != target_size:
        original = original.resize(target_size, Image.Resampling.LANCZOS)
    original_path = output_dir / "original.png"
    original.save(original_path)
    rendered_path = Path(output_paths["render"])
    reference_tensor = _image_to_metric_tensor(
        original_path,
        resolution=int(metric_resolution),
    )
    rendered_tensor = _image_to_metric_tensor(
        rendered_path,
        resolution=int(metric_resolution),
    )
    psnr = _psnr_metric(reference_tensor, rendered_tensor)
    ssim = _ssim_metric(reference_tensor, rendered_tensor)
    lpips_value: Optional[float] = None
    if not bool(skip_lpips):
        device_name = str(metric_device)
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        evaluator = _LPIPSEvaluator(
            str(lpips_net),
            torch.device(device_name),
        )
        try:
            lpips_value = evaluator.evaluate(
                reference_tensor,
                rendered_tensor,
            )
        finally:
            evaluator.model.cpu()
            del evaluator
            _empty_cuda_cache()
    row = {
        "status": "success",
        "renderer": "pixal3d.utils.render_utils.render_frames",
        "sample_type": "MeshWithVoxel",
        "material_source": "official sparse O-Voxel surface-position lookup",
        "sampling_mode": "grid_sample_3d trilinear",
        "view_method": "exact off-axis global-camera tile crop",
        "envmap": str(envmap_name),
        "render_resolution": int(resolution),
        "metric_resolution": int(metric_resolution),
        "camera_angle_x": float(global_camera["camera_angle_x"]),
        "distance": float(global_camera["distance"]),
        "offaxis_fx_normalized": float(transform.fx)
        / float(transform.output_width),
        "offaxis_fy_normalized": float(transform.fy)
        / float(transform.output_height),
        "offaxis_cx_normalized": float(transform.offaxis_cx)
        / float(transform.output_width),
        "offaxis_cy_normalized": float(transform.offaxis_cy)
        / float(transform.output_height),
        "near": float(near),
        "far": float(far),
        "ssaa": int(ssaa),
        "peel_layers": int(peel_layers),
        "use_envmap_bg": bool(use_envmap_bg),
        "decoder_vertices": int(mesh.vertices.shape[0]),
        "decoder_faces": int(mesh.faces.shape[0]),
        "active_voxels": int(mesh.coords.shape[0]),
        "render_seconds": float(render_seconds),
        "render_png": str(rendered_path),
        "render_outputs": output_paths,
        "original_png": str(original_path),
        "comparison_png": None,
        "psnr_db": float(psnr),
        "ssim": float(ssim),
        "lpips": lpips_value,
        "error": None,
    }
    _atomic_json(output_dir / "metrics.json", row)
    return row


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
    label: str = "Global official",
    log_tag: str = "global",
) -> Tuple[torch.Tensor, torch.Tensor, ShapeResult]:
    condition_ss = pipeline.get_proj_cond_ss(
        [image_512],
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
    )
    _seed_everything(seed)
    _sync_cuda()
    ss_started = time.perf_counter()
    coords32 = pipeline.sample_sparse_structure(
        condition_ss,
        resolution=GRID_SS,
        sampler_params=dict(params["ss"]),
    )
    _sync_cuda()
    ss_seconds = time.perf_counter() - ss_started
    del condition_ss
    if coords32.shape[0] == 0:
        raise RuntimeError(f"{label}: sparse structure is empty")
    print(
        f"[{log_tag}] C32 tokens={coords32.shape[0]:,} "
        f"ss_seconds={ss_seconds:.3f}"
    )

    coords64, shape512_seconds = _run_shape512_and_upsample_c64(
        pipeline=pipeline,
        image_512=image_512,
        coords32=coords32,
        camera=camera,
        params=params,
        seed=seed + 101,
        description=f"{label} shape 512",
    )
    if coords64.shape[0] > max_tokens:
        raise RuntimeError(
            f"{label}: C64 support has {coords64.shape[0]:,} tokens; "
            f"exceeds --max-num-tokens={max_tokens:,}"
        )
    print(f"[{log_tag}] learned C64 tokens={coords64.shape[0]:,}")

    shape_norm, shape_denorm, shape1024_seconds = _run_shape1024(
        pipeline=pipeline,
        image_1024=image_1024,
        coords64=coords64,
        camera=camera,
        params=params,
        seed=seed + 201,
        description=f"{label} shape 1024",
    )
    return coords32, coords64, ShapeResult(
        shape_norm=shape_norm,
        shape_denorm=shape_denorm,
        ss_seconds=float(ss_seconds),
        shape512_seconds=shape512_seconds,
        shape1024_seconds=shape1024_seconds,
    )


def _run_tile_native_full_cascade(
    *,
    pipeline: Any,
    tile_image: Image.Image,
    tile_camera: Mapping[str, float],
    params: Mapping[str, Mapping[str, Any]],
    seed: int,
    label: str,
    max_tokens: int,
) -> ModelResult:
    """Run the ordinary Pixal3D cascade with the tile image as direct input."""
    tile_1024 = tile_image.convert("RGB")
    tile_512 = tile_1024.resize((IMAGE_LR, IMAGE_LR), Image.Resampling.LANCZOS)
    coords32, coords64, shape = _run_global_official_geometry_to_shape1024(
        pipeline=pipeline,
        image_512=tile_512,
        image_1024=tile_1024,
        camera=tile_camera,
        params=params,
        seed=int(seed),
        max_tokens=int(max_tokens),
        label=label,
        log_tag="tile-native-full",
    )
    texture_norm, texture_denorm, texture1024_seconds = _run_texture1024(
        pipeline=pipeline,
        image_1024=tile_1024,
        coords64=coords64,
        camera=tile_camera,
        shape_norm=shape.shape_norm,
        params=params,
        seed=int(seed) + 301,
        description=f"{label} texture 1024",
    )
    return ModelResult(
        shape_norm=shape.shape_norm,
        shape_denorm=shape.shape_denorm,
        texture_norm=texture_norm,
        texture_denorm=texture_denorm,
        tile_projective_c32_tokens=0,
        tile_ss_c32_tokens=int(coords32.shape[0]),
        tile_c32_overlap_tokens=0,
        tile_c32_tokens=int(coords32.shape[0]),
        tile_projective_c64_tokens=0,
        tile_native_c64_tokens=int(coords64.shape[0]),
        tile_c64_overlap_tokens=0,
        tile_c64_tokens=int(coords64.shape[0]),
        tile_ss_seconds=float(shape.ss_seconds),
        shape512_seconds=float(shape.shape512_seconds),
        shape1024_seconds=float(shape.shape1024_seconds),
        texture1024_seconds=float(texture1024_seconds),
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

@torch.no_grad()
def _decode_global_mesh_with_subdivision_guides(
    *,
    pipeline: Any,
    shape_latent: SparseTensor,
    texture_latent: SparseTensor,
    label: str,
) -> Tuple[List[MeshWithVoxel], MeshWithVoxel, List[SparseTensor]]:
    """Decode the global baseline once and retain shape decoder guide subs."""
    meshes, subs = pipeline.decode_shape_slat(shape_latent, DECODE_TILE)
    texture_voxels = pipeline.decode_tex_slat(texture_latent, subs)
    if len(meshes) != 1 or len(texture_voxels) != 1:
        raise RuntimeError(
            f"{label}: expected one shape and texture sample, got "
            f"{len(meshes)} and {len(texture_voxels)}"
        )
    shape_mesh = meshes[0]
    texture_voxel = texture_voxels[0]
    mesh = MeshWithVoxel(
        vertices=shape_mesh.vertices,
        faces=shape_mesh.faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / DECODE_TILE,
        coords=texture_voxel.coords[:, 1:],
        attrs=texture_voxel.feats,
        voxel_shape=torch.Size(
            [*texture_voxel.shape, *texture_voxel.spatial_shape]
        ),
        layout=pipeline.pbr_attr_layout,
    )
    if len(subs) == 0:
        raise RuntimeError(f"{label}: shape decoder returned no subdivision guides")
    print(
        f"[global-decoder-with-subs] {label}: "
        f"vertices={mesh.vertices.shape[0]:,} "
        f"faces={mesh.faces.shape[0]:,} "
        f"ovoxel_entries={mesh.coords.shape[0]:,} "
        f"subdivision_layers={len(subs)}"
    )
    return [mesh], mesh, subs


def _coordinate_codes(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    """Encode batched integer xyz coordinates into collision-free int64 keys."""
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"coordinates must be [N,4], got {tuple(coords.shape)}")
    values = coords.to(torch.int64)
    if values.shape[0] == 0:
        return values.new_empty((0,))
    if bool(
        (
            (values[:, 0] < 0)
            | (values[:, 1:] < 0).any(dim=1)
            | (values[:, 1:] >= int(resolution)).any(dim=1)
        ).any().item()
    ):
        raise ValueError(f"coordinate outside C{int(resolution)} lattice")
    code = values[:, 0]
    for dimension in range(1, 4):
        code = code * int(resolution) + values[:, dimension]
    return code


def _lookup_coordinate_rows(
    query: torch.Tensor,
    table: torch.Tensor,
    *,
    resolution: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return table row for each query coordinate and an exact-match mask."""
    if table.shape[0] == 0:
        return (
            torch.zeros(query.shape[0], device=query.device, dtype=torch.long),
            torch.zeros(query.shape[0], device=query.device, dtype=torch.bool),
        )
    if query.device != table.device:
        raise ValueError("coordinate lookup tensors must be on the same device")
    table_codes = _coordinate_codes(table, resolution)
    query_codes = _coordinate_codes(query, resolution)
    sorted_codes, sorted_rows = torch.sort(table_codes)
    positions = torch.searchsorted(sorted_codes, query_codes)
    safe_positions = positions.clamp(max=sorted_codes.shape[0] - 1)
    found = (
        (positions < sorted_codes.shape[0])
        & (sorted_codes[safe_positions] == query_codes)
    )
    rows = sorted_rows[safe_positions]
    return rows, found


def _expand_global_subdivision_children(
    subdivision: SparseTensor,
    *,
    parent_resolution: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Expand one global decoder subdivision mask into active child coords."""
    if subdivision.coords.ndim != 2 or subdivision.coords.shape[1] != 4:
        raise ValueError("global subdivision coordinates must be [N,4]")
    if subdivision.feats.ndim != 2 or subdivision.feats.shape[1] != 8:
        raise ValueError("global subdivision features must be [N,8]")
    if subdivision.feats.shape[0] != subdivision.coords.shape[0]:
        raise ValueError("global subdivision features/coordinates are not aligned")

    active_parent_rows, child_slots = (subdivision.feats > 0).nonzero(
        as_tuple=True
    )
    if active_parent_rows.numel() == 0:
        raise RuntimeError(
            f"global C{parent_resolution} subdivision has no active children"
        )
    child_coords = subdivision.coords[active_parent_rows].to(torch.int32).clone()
    child_coords[:, 1:] *= 2
    for dimension in range(3):
        child_coords[:, dimension + 1] += (
            child_slots // (2**dimension) % 2
        ).to(torch.int32)

    child_resolution = int(parent_resolution) * 2
    stats = {
        "parent_resolution": int(parent_resolution),
        "child_resolution": int(child_resolution),
        "global_parent_rows": int(subdivision.coords.shape[0]),
        "global_active_children": int(child_coords.shape[0]),
        "global_unique_children": int(child_coords.shape[0]),
        "uniqueness_guarantee": (
            "unique parent coords plus unique active child slot per parent"
        ),
    }
    # Keep the persistent hierarchy on CPU; each tile moves only one projection
    # chunk at a time back to the texture latent's device.
    return child_coords.detach().cpu(), stats


def _prepare_global_subdivision_children(
    subdivisions: Sequence[SparseTensor],
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    """Materialize the global C64->C1024 guide hierarchy once."""
    if len(subdivisions) == 0:
        raise RuntimeError("global shape decoder returned no subdivision layers")
    children_by_layer: List[torch.Tensor] = []
    stats: List[Dict[str, Any]] = []
    parent_resolution = GRID_SHAPE_1024
    for layer_index, subdivision in enumerate(subdivisions):
        children, layer_stats = _expand_global_subdivision_children(
            subdivision,
            parent_resolution=parent_resolution,
        )
        layer_stats["layer_index"] = int(layer_index)
        children_by_layer.append(children)
        stats.append(layer_stats)
        parent_resolution *= 2
    if parent_resolution != GRID_GLOBAL_UPSAMPLED:
        raise RuntimeError(
            "global subdivision hierarchy must end at C1024, got "
            f"C{parent_resolution}"
        )
    return children_by_layer, stats


@torch.no_grad()
def _project_global_c1024_subs_to_tile(
    *,
    global_final_children_cpu: torch.Tensor,
    global_camera: Mapping[str, float],
    transform: TileCameraTransform,
    chunk_size: int,
    device: torch.device,
) -> ProjectedGlobalTextureSupport:
    """Project global C1024 subdivision leaves and coarsen them to local C64.

    C1024 coordinates use the renderer's actual O-Voxel convention
    ``q = -1 + 2 * coord / 1024``. The projected leaves are then coarsened to
    local C64. Only coordinates are projected: the modified route regenerates
    the shape SLat from the tile image on this local C64 support.
    """
    if int(chunk_size) < 1:
        raise ValueError("guide projection chunk size must be positive")
    local_chunks: List[torch.Tensor] = []
    global_chunks: List[torch.Tensor] = []
    projected_rows = 0
    valid_rows = 0
    pixel_roundtrip_max = 0.0
    for start in range(
        0,
        global_final_children_cpu.shape[0],
        int(chunk_size),
    ):
        global_chunk = global_final_children_cpu[
            start : start + int(chunk_size)
        ].to(device=device, non_blocking=True)
        q_global = (
            global_chunk[:, 1:].to(torch.float32)
            * (2.0 / float(GRID_GLOBAL_UPSAMPLED))
            - 1.0
        )
        q_local, _, _, transform_stats = _global_q_to_centered_tile_q(
            q_global,
            global_camera=global_camera,
            transform=transform,
        )
        local_ids = torch.round(
            (q_local + 1.0)
            * (float(GRID_GLOBAL_UPSAMPLED) / 2.0)
        ).to(torch.int32)
        finite = torch.isfinite(q_local).all(dim=1)
        inside_lattice = (
            (local_ids >= 0)
            & (local_ids < GRID_GLOBAL_UPSAMPLED)
        ).all(dim=1)
        valid = finite & inside_lattice
        projected_rows += int(global_chunk.shape[0])
        valid_rows += int(valid.sum().item())
        pixel_roundtrip_max = max(
            pixel_roundtrip_max,
            float(transform_stats["pixel_roundtrip_max"]),
        )
        if bool(valid.any().item()):
            local_chunks.append(
                torch.cat(
                    [
                        global_chunk[valid, :1].to(torch.int32),
                        local_ids[valid],
                    ],
                    dim=1,
                )
            )
            global_chunks.append(global_chunk[valid].to(torch.int32))
    if not local_chunks:
        raise RuntimeError(
            f"no global C1024 subdivision leaves project into tile "
            f"{transform.tile_id}"
        )

    local_per_source = torch.cat(local_chunks, dim=0)
    global_per_source = torch.cat(global_chunks, dim=0)
    final_local_coords, source_to_final = torch.unique(
        local_per_source,
        dim=0,
        return_inverse=True,
    )
    local_c64_per_source = local_per_source.clone()
    local_c64_per_source[:, 1:] //= (
        GRID_GLOBAL_UPSAMPLED // GRID_SHAPE_1024
    )
    local_c64_coords = torch.unique(local_c64_per_source, dim=0)
    stats = {
        "projection_source": (
            "final global subdivision C1024 leaves; coordinates only"
        ),
        "coordinate_convention": (
            "O-Voxel origin=-0.5, voxel_size=1/1024; no endpoint res-1 mapping"
        ),
        "global_c1024_subdivision_leaves": int(
            global_final_children_cpu.shape[0]
        ),
        "projected_rows": int(projected_rows),
        "valid_tile_source_rows": int(valid_rows),
        "dropped_outside_tile_lattice_rows": int(projected_rows - valid_rows),
        "unique_local_c1024_leaves": int(final_local_coords.shape[0]),
        "unique_local_c64_shape_tokens": int(local_c64_coords.shape[0]),
        "local_c64_source_rows": int(local_c64_per_source.shape[0]),
        "projection_collision_rows": int(
            local_per_source.shape[0] - final_local_coords.shape[0]
        ),
        "pixel_roundtrip_max": float(pixel_roundtrip_max),
    }
    print(
        "[MODIFIED-TILE-SHAPE-FLOW] "
        f"tile={transform.tile_id:02d} "
        f"global_C1024={global_final_children_cpu.shape[0]:,} "
        f"valid_sources={valid_rows:,} "
        f"local_C1024={final_local_coords.shape[0]:,} "
        f"local_C64={local_c64_coords.shape[0]:,}"
    )
    return ProjectedGlobalTextureSupport(
        coords64=local_c64_coords,
        final_local_coords=final_local_coords,
        final_global_coords=global_per_source,
        source_to_final_local_rows=source_to_final,
        projection_stats=stats,
    )


def _build_guide_from_required_children(
    *,
    current_local_coords: torch.Tensor,
    required_local_children: torch.Tensor,
    parent_resolution: int,
) -> Tuple[SparseTensor, torch.Tensor, Dict[str, Any]]:
    """Build one exact decoder guide from a consistent projected leaf tree."""
    child_resolution = int(parent_resolution) * 2
    local_parents = required_local_children.clone()
    local_parents[:, 1:] //= 2
    parent_rows, parent_found = _lookup_coordinate_rows(
        local_parents,
        current_local_coords,
        resolution=int(parent_resolution),
    )
    if not bool(parent_found.all().item()):
        raise RuntimeError(
            f"projected leaf tree lost {(~parent_found).sum().item():,} "
            f"C{parent_resolution} parents while building C{child_resolution}"
        )
    child_bits = required_local_children[:, 1:].to(torch.long) % 2
    child_slots = (
        child_bits[:, 0] + 2 * child_bits[:, 1] + 4 * child_bits[:, 2]
    )
    guide_mask = torch.zeros(
        (current_local_coords.shape[0], 8),
        device=current_local_coords.device,
        dtype=torch.bool,
    )
    guide_mask[parent_rows, child_slots] = True
    active_parent_rows, active_slots = guide_mask.nonzero(as_tuple=True)
    next_local_coords = current_local_coords[active_parent_rows].clone()
    next_local_coords[:, 1:] *= 2
    for dimension in range(3):
        next_local_coords[:, dimension + 1] += (
            active_slots // (2**dimension) % 2
        ).to(next_local_coords.dtype)
    if next_local_coords.shape[0] != required_local_children.shape[0]:
        raise RuntimeError(
            f"C{child_resolution} guide produced "
            f"{next_local_coords.shape[0]:,} rows for "
            f"{required_local_children.shape[0]:,} required children"
        )
    guide = SparseTensor(
        feats=guide_mask,
        coords=current_local_coords,
    )
    stats = {
        "parent_resolution": int(parent_resolution),
        "child_resolution": int(child_resolution),
        "current_local_parent_rows": int(current_local_coords.shape[0]),
        "active_local_parent_rows": int(guide_mask.any(dim=1).sum().item()),
        "required_local_children": int(required_local_children.shape[0]),
        "next_local_unique_children": int(next_local_coords.shape[0]),
        "parent_mismatch_rows": 0,
        "construction": (
            "coarsen projected valid global C1024 subdivision leaves"
        ),
    }
    return guide, next_local_coords, stats


def _build_projected_global_guide_hierarchy(
    *,
    tile_texture_latent: SparseTensor,
    support: ProjectedGlobalTextureSupport,
) -> ProjectedGlobalGuideHierarchy:
    """Build all four local guides from projected global C1024 sub leaves."""
    current_local_coords = tile_texture_latent.coords.to(torch.int32)
    if bool((current_local_coords[:, 0] != 0).any().item()):
        raise ValueError("modified texture route currently expects batch size one")
    required_c64 = support.final_local_coords.clone()
    required_c64[:, 1:] //= (
        GRID_GLOBAL_UPSAMPLED // GRID_SHAPE_1024
    )
    required_c64 = torch.unique(required_c64, dim=0)
    current_codes = torch.sort(
        _coordinate_codes(current_local_coords, GRID_SHAPE_1024)
    ).values
    required_codes = torch.sort(
        _coordinate_codes(required_c64, GRID_SHAPE_1024)
    ).values
    if not torch.equal(current_codes, required_codes):
        raise RuntimeError(
            "tile texture-flow C64 coordinates differ from projected global "
            "shape-SLat C64 coordinates"
        )
    del current_codes, required_codes, required_c64

    guide_subs: List[SparseTensor] = []
    layer_stats: List[Dict[str, Any]] = []
    parent_resolution = GRID_SHAPE_1024
    for layer_index in range(4):
        child_resolution = int(parent_resolution) * 2
        divisor = GRID_GLOBAL_UPSAMPLED // child_resolution
        required_children = support.final_local_coords.clone()
        required_children[:, 1:] //= divisor
        required_children = torch.unique(required_children, dim=0)
        guide, next_local_coords, stats = (
            _build_guide_from_required_children(
                current_local_coords=current_local_coords,
                required_local_children=required_children,
                parent_resolution=parent_resolution,
            )
        )
        stats["layer_index"] = int(layer_index)
        layer_stats.append(stats)
        guide_subs.append(guide)
        current_local_coords = next_local_coords
        print(
            "[MODIFIED-PROJECTED-GLOBAL-SUBS] "
            f"C{parent_resolution}->C{child_resolution}: "
            f"parents={stats['current_local_parent_rows']:,} "
            f"children={stats['next_local_unique_children']:,} "
            "mismatch=0"
        )
        parent_resolution = child_resolution
    if parent_resolution != GRID_GLOBAL_UPSAMPLED:
        raise RuntimeError("projected guide hierarchy did not end at C1024")

    unique_final_to_decoder_rows, final_found = _lookup_coordinate_rows(
        support.final_local_coords,
        current_local_coords,
        resolution=GRID_GLOBAL_UPSAMPLED,
    )
    if not bool(final_found.all().item()):
        raise RuntimeError("final projected C1024 support was lost in guide tree")
    source_to_decoder_rows = unique_final_to_decoder_rows[
        support.source_to_final_local_rows
    ]
    return ProjectedGlobalGuideHierarchy(
        guide_subs=guide_subs,
        final_local_coords=current_local_coords,
        final_global_coords=support.final_global_coords,
        final_global_to_local_rows=source_to_decoder_rows,
        layer_stats=layer_stats,
    )

@torch.no_grad()
def _decode_tile_texture_with_projected_global_guides(
    *,
    pipeline: Any,
    tile_texture_latent: SparseTensor,
    hierarchy: ProjectedGlobalGuideHierarchy,
) -> Tuple[SparseTensor, torch.Tensor, float]:
    """Decode tile texture using guides coarsened from projected global subs."""
    decoder = pipeline.models["tex_slat_decoder"]
    if len(hierarchy.guide_subs) != len(hierarchy.layer_stats):
        raise RuntimeError("projected guide metadata is inconsistent")
    # Construct a fresh SparseTensor deliberately: a latent may carry a cached
    # upsample map, in which case SparseUpsample ignores guide_subs entirely.
    clean_texture_latent = SparseTensor(
        feats=tile_texture_latent.feats,
        coords=tile_texture_latent.coords,
    )
    if pipeline.low_vram:
        decoder.to(pipeline.device)
    started = time.perf_counter()
    try:
        texture_voxel = (
            decoder(
                clean_texture_latent,
                guide_subs=hierarchy.guide_subs,
            )
            * 0.5
            + 0.5
        )
        _sync_cuda()
    finally:
        if pipeline.low_vram:
            decoder.cpu()
            _empty_cuda_cache()
    elapsed = time.perf_counter() - started
    if not isinstance(texture_voxel, SparseTensor):
        raise TypeError("texture decoder did not return a SparseTensor")

    hierarchy_to_output_rows, found = _lookup_coordinate_rows(
        hierarchy.final_local_coords,
        texture_voxel.coords,
        resolution=GRID_GLOBAL_UPSAMPLED,
    )
    if not bool(found.all().item()):
        raise RuntimeError(
            "texture decoder output support differs from projected guide hierarchy"
        )
    global_source_to_output_rows = hierarchy_to_output_rows[
        hierarchy.final_global_to_local_rows
    ]
    print(
        "[MODIFIED-PROJECTED-GLOBAL-SUBS] texture decode: "
        f"local_C1024={texture_voxel.coords.shape[0]:,} "
        f"global_correspondences={global_source_to_output_rows.shape[0]:,} "
        f"seconds={elapsed:.3f}"
    )
    return texture_voxel, global_source_to_output_rows, float(elapsed)


def _save_global_normal_fallback_view(
    *,
    render_outputs: Mapping[str, Any],
    output_dir: Path,
) -> Tuple[str, str, Dict[str, Any]]:
    """Fill geometry pixels without decoded tile alpha using normal color."""
    required = ("shaded", "normal", "alpha")
    missing = [name for name in required if not render_outputs.get(name)]
    if missing:
        raise KeyError(
            "global render is missing maps required for normal fallback: "
            + ", ".join(missing)
        )
    shaded = np.asarray(
        Image.open(str(render_outputs["shaded"])).convert("RGB"),
        dtype=np.float32,
    )
    normal = np.asarray(
        Image.open(str(render_outputs["normal"])).convert("RGB"),
        dtype=np.float32,
    )
    alpha = np.asarray(
        Image.open(str(render_outputs["alpha"])).convert("L"),
        dtype=np.float32,
    )[..., None] / 255.0
    if shaded.shape != normal.shape or shaded.shape[:2] != alpha.shape[:2]:
        raise ValueError("global shaded/normal/alpha render sizes do not match")

    # The official renderer writes invalid-background normals as approximately
    # (0.5, 0.5, 0.5); a real unit normal cannot be close to that zero vector.
    normal_vector = normal / 255.0 - 0.5
    geometry_mask = (
        np.linalg.norm(normal_vector, axis=2, keepdims=True) > 0.10
    )
    material_alpha = np.clip(alpha, 0.0, 1.0)
    filled = shaded + (1.0 - material_alpha) * normal
    filled = np.where(geometry_mask, filled, 0.0)
    filled_u8 = np.clip(np.rint(filled), 0, 255).astype(np.uint8)

    coverage = np.zeros_like(filled_u8)
    colored = geometry_mask & (material_alpha > (1.0 / 255.0))
    normal_only = geometry_mask & ~colored
    coverage[colored[..., 0]] = np.array([60, 220, 80], dtype=np.uint8)
    coverage[normal_only[..., 0]] = np.array([220, 145, 45], dtype=np.uint8)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Replace the helper's black/transparent partial-material render alias so
    # the folder's canonical render.png is the requested normal-filled view.
    # The untouched material-only result remains available as shaded.png.
    display_path = output_dir / "render.png"
    coverage_path = output_dir / "global_view_material_coverage.png"
    Image.fromarray(filled_u8, mode="RGB").save(display_path)
    Image.fromarray(coverage, mode="RGB").save(coverage_path)

    geometry_pixels = int(geometry_mask[..., 0].sum())
    colored_pixels = int(colored[..., 0].sum())
    normal_only_pixels = int(normal_only[..., 0].sum())
    stats = {
        "geometry_pixels": geometry_pixels,
        "tile_material_pixels": colored_pixels,
        "normal_fallback_pixels": normal_only_pixels,
        "tile_material_geometry_fraction": float(
            colored_pixels / max(1, geometry_pixels)
        ),
        "normal_fallback_geometry_fraction": float(
            normal_only_pixels / max(1, geometry_pixels)
        ),
        "composition": (
            "official shaded tile material + (1-alpha)*camera-space normal; "
            "outside geometry is black"
        ),
    }
    return str(display_path), str(coverage_path), stats


@torch.no_grad()
def _render_modified_projected_global_subs_texture(
    *,
    pipeline: Any,
    global_vertices: torch.Tensor,
    global_faces: torch.Tensor,
    global_children_by_layer: Sequence[torch.Tensor],
    tile_image: Image.Image,
    params: Mapping[str, Mapping[str, Any]],
    seed: int,
    global_camera: Mapping[str, float],
    tile_camera: Mapping[str, float],
    transform: TileCameraTransform,
    reference_image: Path,
    output_dir: Path,
    args: argparse.Namespace,
    envmap: Any,
) -> Dict[str, Any]:
    """Regenerate tile shape/texture SLat on projected global C64 support."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(global_children_by_layer) != 4:
        raise RuntimeError(
            "modified route requires the complete global C64->C1024 "
            "subdivision hierarchy"
        )
    support = _project_global_c1024_subs_to_tile(
        global_final_children_cpu=global_children_by_layer[-1],
        global_camera=global_camera,
        transform=transform,
        chunk_size=int(args.material_mapping_chunk_size),
        device=pipeline.device,
    )
    tile_shape_norm, tile_shape_denorm, shape_flow_seconds = _run_shape1024(
        pipeline=pipeline,
        image_1024=tile_image,
        coords64=support.coords64,
        camera=tile_camera,
        params=params,
        seed=int(seed) + 201,
        description=(
            f"Tile {transform.tile_id:02d} projected-global-C1024-C64 "
            "modified shape 1024"
        ),
    )
    tile_texture_norm, tile_texture_latent, texture_flow_seconds = (
        _run_texture1024(
            pipeline=pipeline,
            image_1024=tile_image,
            coords64=support.coords64,
            camera=tile_camera,
            shape_norm=tile_shape_norm,
            params=params,
            seed=int(seed) + 301,
            description=(
                f"Tile {transform.tile_id:02d} regenerated-tile-shape "
                "modified texture 1024"
            ),
        )
    )
    del tile_shape_norm, tile_shape_denorm
    _empty_cuda_cache()
    hierarchy = _build_projected_global_guide_hierarchy(
        tile_texture_latent=tile_texture_latent,
        support=support,
    )
    texture_voxel, global_source_to_output_rows, decode_seconds = (
        _decode_tile_texture_with_projected_global_guides(
            pipeline=pipeline,
            tile_texture_latent=tile_texture_latent,
            hierarchy=hierarchy,
        )
    )
    alpha_values = texture_voxel.feats[
        ...,
        pipeline.pbr_attr_layout["alpha"],
    ].to(torch.float32)
    decoded_material_stats = {
        "decoded_entries": int(texture_voxel.coords.shape[0]),
        "alpha_min": float(alpha_values.amin().item()),
        "alpha_max": float(alpha_values.amax().item()),
        "alpha_mean": float(alpha_values.mean().item()),
        "alpha_positive_entries": int((alpha_values > 0.0).sum().item()),
        "alpha_above_0_01_entries": int(
            (alpha_values > 0.01).sum().item()
        ),
        "alpha_above_0_5_entries": int(
            (alpha_values > 0.5).sum().item()
        ),
    }
    # Decoder no longer needs the per-layer boolean guide tensors. Retain only
    # final local/global correspondence and lightweight statistics for renders.
    hierarchy.guide_subs.clear()
    del tile_texture_norm, tile_texture_latent
    _empty_cuda_cache()
    _atomic_json(
        output_dir / "projected_global_subdivision_hierarchy.json",
        {
            "source": (
                "tile-image-regenerated shape SLat on projected-global-C1024 "
                "local C64 support plus projected global C1024 subdivision leaves"
            ),
            "projection": (
                "project valid global C1024 O-Voxel leaves into tile C1024, "
                "then coarsen them to an exactly consistent C64->C1024 tree"
            ),
            "shape_slat_source": (
                "tile image shape1024 flow on projected global C1024-derived "
                "local C64 coordinates"
            ),
            "global_material_used": False,
            "projection_stats": support.projection_stats,
            "shape_flow_seconds": float(shape_flow_seconds),
            "texture_flow_seconds": float(texture_flow_seconds),
            "texture_decode_seconds": float(decode_seconds),
            "decoded_material_stats": decoded_material_stats,
            "layers": hierarchy.layer_stats,
            "final_local_c1024_entries": int(texture_voxel.coords.shape[0]),
            "final_global_correspondences": int(
                global_source_to_output_rows.shape[0]
            ),
        },
    )

    # Both observation views use the original global geometry and the material
    # assigned to its exact source global C1024 leaves. The subimage view
    # changes only the camera intrinsics to the tile's off-axis crop rays.
    global_coords = hierarchy.final_global_coords[:, 1:].to(
        device=texture_voxel.device,
        dtype=texture_voxel.coords.dtype,
    )
    global_attrs = texture_voxel.feats[global_source_to_output_rows]
    global_partial_mesh = MeshWithVoxel(
        vertices=global_vertices,
        faces=global_faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / GRID_GLOBAL_UPSAMPLED,
        coords=global_coords,
        attrs=global_attrs,
        voxel_shape=torch.Size(
            [
                1,
                global_attrs.shape[1],
                GRID_GLOBAL_UPSAMPLED,
                GRID_GLOBAL_UPSAMPLED,
                GRID_GLOBAL_UPSAMPLED,
            ]
        ),
        layout=pipeline.pbr_attr_layout,
    )
    print(
        "[MODIFIED-TILE-SHAPE-FLOW] subimage observation: unchanged global "
        "geometry/material lattice + exact global-camera off-axis tile rays"
    )
    subimage_output_dir = (
        output_dir / "subimage_view_global_camera_offaxis"
    )
    local_metric = _render_and_evaluate_global_camera_tile_crop(
        global_partial_mesh,
        global_camera=global_camera,
        transform=transform,
        output_dir=subimage_output_dir,
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
    local_extras = _save_extra_comparisons(
        Path(local_metric["original_png"]),
        Path(local_metric["render_png"]),
        output_dir / "subimage_view_comparisons",
    )
    local_summary = {
        **local_metric,
        **local_extras,
        "route_family": (
            "MODIFIED_TILE_SHAPE_FLOW_PROJECTED_GLOBAL_SUBS_TEXTURE_GLOBAL_CAMERA"
        ),
        "view": "global_camera_corresponding_subimage",
        "geometry_source": "unchanged global decoded mesh",
        "material_source": (
            "tile-image texture flow conditioned by a regenerated tile shape "
            "SLat on projected-global-C1024-derived local C64 coordinates; "
            "assigned back to source global C1024 subdivision leaves"
        ),
        "global_material_used": False,
        "geometry_coordinate_transform_used": False,
        "camera_protocol": (
            "global extrinsics and distance with exact off-axis intrinsics "
            "corresponding to the canonical tile crop"
        ),
    }
    _atomic_json(
        subimage_output_dir / "summary.json",
        local_summary,
    )
    _empty_cuda_cache()

    print(
        "[MODIFIED-PROJECTED-GLOBAL-SUBS] global observation: original global "
        "geometry + direct global-source correspondence; missing material uses "
        "normal color"
    )
    global_output_dir = output_dir / "global_view_with_normal_fallback"
    global_metric = render_and_evaluate_mesh(
        global_partial_mesh,
        camera_angle_x=float(global_camera["camera_angle_x"]),
        distance=float(global_camera["distance"]),
        output_dir=global_output_dir,
        reference_image=None,
        resolution=int(args.modified_global_render_resolution),
        metric_resolution=int(args.metric_resolution),
        envmap=envmap,
        envmap_name=str(args.envmap),
        ssaa=int(args.render_ssaa),
        peel_layers=int(args.render_peel_layers),
        # Normal fallback is defined against a black renderer background.
        use_envmap_bg=False,
        lpips_net=str(args.lpips_net),
        metric_device=str(args.metric_device),
        skip_lpips=True,
    )
    display_png, coverage_png, fallback_stats = (
        _save_global_normal_fallback_view(
            render_outputs=global_metric["render_outputs"],
            output_dir=global_output_dir,
        )
    )
    global_summary = {
        **global_metric,
        "route_family": (
            "MODIFIED_TILE_SHAPE_FLOW_PROJECTED_GLOBAL_SUBS_TEXTURE_GLOBAL_CAMERA"
        ),
        "view": "global_camera",
        "raw_partial_material_png": global_metric["render_outputs"]["shaded"],
        "render_png": display_png,
        "display_png": display_png,
        "material_coverage_png": coverage_png,
        "geometry_source": "unchanged global decoded mesh",
        "material_source": (
            "tile-image texture flow conditioned by a regenerated tile shape "
            "SLat on projected-global-C1024-derived local C64 coordinates; "
            "decoded attrs assigned through exact global-sub leaf correspondence"
        ),
        "global_material_used": False,
        "inverse_coordinate_remap_used": False,
        "uncolored_geometry_behavior": "camera-space normal-vector color",
        "global_c1024_material_entries": int(global_coords.shape[0]),
        "normal_fallback_stats": fallback_stats,
    }
    _atomic_json(global_output_dir / "summary.json", global_summary)
    summary = {
        "route_family": (
            "MODIFIED_TILE_SHAPE_FLOW_PROJECTED_GLOBAL_SUBS_TEXTURE_GLOBAL_CAMERA"
        ),
        "explicitly_separate_from_original_route": True,
        "global_material_used": False,
        "tile_shape_decoder_subs_used": False,
        "global_shape_slat_projected_for_texture_flow": False,
        "tile_shape_slat_regenerated_from_tile_image": True,
        "tile_shape_slat_support": (
            "global C1024 subdivision leaves projected to tile and coarsened "
            "to local C64"
        ),
        "global_subdivision_c1024_leaves_projected": True,
        "local_guide_layers_coarsened_from_projected_c1024": True,
        "inverse_tile_to_global_route_used": False,
        "projection_stats": support.projection_stats,
        "projected_guide_layers": hierarchy.layer_stats,
        "shape_flow_seconds": float(shape_flow_seconds),
        "texture_flow_seconds": float(texture_flow_seconds),
        "texture_decode_seconds": float(decode_seconds),
        "decoded_material_stats": decoded_material_stats,
        "local_view": local_summary,
        "global_view": global_summary,
    }
    _atomic_json(output_dir / "summary.json", summary)
    del (
        global_partial_mesh,
        global_coords,
        global_attrs,
        texture_voxel,
        hierarchy,
        support,
    )
    _empty_cuda_cache()
    return summary


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
    retain_decoded_mesh: bool = False,
) -> Tuple[Dict[str, Any], Optional[MeshWithVoxel]]:
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
        "route_family": "ORIGINAL_DECODE_RENDER",
        "material_or_geometry_replaced_after_decode": False,
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
    summary = {**decoder_metadata, **metric_row, **extras}
    del meshes
    if not retain_decoded_mesh:
        del mesh
        mesh = None
    _empty_cuda_cache()
    return summary, mesh


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
    native_tile_summary: Optional[Mapping[str, Any]],
    modified_material_summary: Optional[Mapping[str, Any]],
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

    if native_tile_summary is None:
        native_render = _label_panel(
            _resize_panel(None),
            "Direct tile -> ordinary full cascade\nfailed or disabled",
        )
        native_diff = _label_panel(
            _resize_panel(None),
            "Direct-tile absolute error\nunavailable",
        )
    else:
        native_render = _label_panel(
            _resize_panel(
                Path(str(native_tile_summary["render_png"]))
                if native_tile_summary.get("render_png")
                else None
            ),
            (
                "Direct tile -> SS -> shape512/1024 -> texture\n"
                f"C32={native_tile_summary.get('tile_ss_c32_tokens')} "
                f"C64={native_tile_summary.get('tile_native_c64_tokens')}\n"
                f"PSNR={native_tile_summary.get('psnr_db')} "
                f"SSIM={native_tile_summary.get('ssim')} "
                f"LPIPS={native_tile_summary.get('lpips')}"
            ),
        )
        native_diff = _label_panel(
            _resize_panel(
                Path(str(native_tile_summary["diff_heatmap_png"]))
                if native_tile_summary.get("diff_heatmap_png")
                else None
            ),
            "Direct-tile ordinary cascade absolute RGB error",
        )

    modified_local: Optional[Mapping[str, Any]] = None
    modified_global: Optional[Mapping[str, Any]] = None
    if modified_material_summary is not None:
        local_candidate = modified_material_summary.get("local_view")
        global_candidate = modified_material_summary.get("global_view")
        if isinstance(local_candidate, Mapping):
            modified_local = local_candidate
        if isinstance(global_candidate, Mapping):
            modified_global = global_candidate

    modified_local_render = _label_panel(
        _resize_panel(
            Path(str(modified_local["render_png"]))
            if modified_local is not None and modified_local.get("render_png")
            else None
        ),
        (
            "MODIFIED: global geometry + projected-global-subs texture\n"
            "global-camera corresponding subimage (off-axis)\n"
            + (
                f"PSNR={modified_local.get('psnr_db')} "
                f"SSIM={modified_local.get('ssim')} "
                f"LPIPS={modified_local.get('lpips')}"
                if modified_local is not None
                else "failed or disabled"
            )
        ),
    )
    modified_local_diff = _label_panel(
        _resize_panel(
            Path(str(modified_local["diff_heatmap_png"]))
            if (
                modified_local is not None
                and modified_local.get("diff_heatmap_png")
            )
            else None
        ),
        "MODIFIED global-camera subimage absolute RGB error",
    )
    modified_global_render = _label_panel(
        _resize_panel(
            Path(
                str(
                    modified_global.get("display_png")
                    or modified_global.get("render_png")
                )
            )
            if (
                modified_global is not None
                and (
                    modified_global.get("display_png")
                    or modified_global.get("render_png")
                )
            )
            else None
        ),
        (
            "MODIFIED: original global geometry, global camera\n"
            "tile material colored; missing material = normal color"
            if modified_global is not None
            else "MODIFIED global observation\nfailed or disabled"
        ),
    )
    global_coverage_path = (
        Path(str(modified_global["material_coverage_png"]))
        if (
            modified_global is not None
            and modified_global.get("material_coverage_png")
        )
        else None
    )
    modified_global_coverage = _label_panel(
        _resize_panel(global_coverage_path),
        "MODIFIED global material coverage\n"
        "green=tile texture, orange=normal fallback",
    )

    panels = [
        reference,
        baseline_render,
        native_render,
        tile_render,
        support,
        modified_local_render,
        modified_global_render,
        modified_global_coverage,
        baseline_diff,
        native_diff,
        tile_diff,
        modified_local_diff,
    ]
    size = panels[0].width
    canvas = Image.new("RGB", (size * 4, size * 3), (18, 18, 18))
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % 4) * size, (index // 4) * size))
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
            native_tile_summary = (
                {
                    "render_png": row.get("native_tile_render_png"),
                    "diff_heatmap_png": row.get(
                        "native_tile_diff_heatmap_png"
                    ),
                    "tile_ss_c32_tokens": row.get("native_tile_c32_tokens"),
                    "tile_native_c64_tokens": row.get(
                        "native_tile_c64_tokens"
                    ),
                    "psnr_db": row.get("native_tile_psnr_db"),
                    "ssim": row.get("native_tile_ssim"),
                    "lpips": row.get("native_tile_lpips"),
                }
                if row.get("native_tile_render_png")
                else None
            )
            modified_material_summary = (
                {
                    "local_view": {
                        "render_png": row.get("modified_local_render_png"),
                        "diff_heatmap_png": row.get(
                            "modified_local_diff_heatmap_png"
                        ),
                        "psnr_db": row.get("modified_local_psnr_db"),
                        "ssim": row.get("modified_local_ssim"),
                        "lpips": row.get("modified_local_lpips"),
                    },
                    "global_view": {
                        "display_png": row.get(
                            "modified_global_partial_render_png"
                        ),
                        "material_coverage_png": row.get(
                            "modified_global_material_coverage_png"
                        ),
                    },
                }
                if row.get("modified_local_render_png")
                or row.get("modified_global_partial_render_png")
                else None
            )
            row["comparison_png"] = _save_tile_comparison_sheet(
                reference_path=Path(str(row["reference_png"])),
                support_overlay_path=Path(str(row["support_overlay_png"])),
                route_summary=route_summary,
                baseline_summary=baseline_summary,
                native_tile_summary=native_tile_summary,
                modified_material_summary=modified_material_summary,
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
        path = output_dir / f"all_tiles_all_comparison_routes_{start // per_page:02d}.png"
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
                "projected_global_c1024_to_local_c64_shape1024_texture1024"
            ),
            "local_only": (
                "ordinary_tile_full_cascade"
                if bool(args.enable_direct_tile_comparison)
                else "disabled"
            ),
            "modified_material": (
                "tile_shape_flow_projected_global_subs"
                if bool(args.enable_modified_material_comparison)
                else "disabled"
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
    global_meshes, global_mesh, global_subs = (
        _decode_global_mesh_with_subdivision_guides(
            pipeline=pipeline,
            shape_latent=global_baseline_result.shape_denorm,
            texture_latent=global_baseline_result.texture_denorm,
            label="Global ordinary 1024 baseline",
        )
    )
    (
        global_subdivision_children,
        global_subdivision_stats,
    ) = _prepare_global_subdivision_children(global_subs)
    final_global_children = global_subdivision_children[-1]
    decoded_global_coords = torch.cat(
        [
            torch.zeros(
                (global_mesh.coords.shape[0], 1),
                dtype=torch.int32,
            ),
            global_mesh.coords.detach().to(
                device="cpu",
                dtype=torch.int32,
            ),
        ],
        dim=1,
    )
    if final_global_children.shape[0] != decoded_global_coords.shape[0]:
        raise RuntimeError(
            "global final subdivision support and decoded O-Voxel support have "
            f"different sizes: {final_global_children.shape[0]:,} vs "
            f"{decoded_global_coords.shape[0]:,}"
        )
    final_codes = torch.sort(
        _coordinate_codes(
            final_global_children,
            GRID_GLOBAL_UPSAMPLED,
        )
    ).values
    decoded_codes = torch.sort(
        _coordinate_codes(
            decoded_global_coords,
            GRID_GLOBAL_UPSAMPLED,
        )
    ).values
    if not torch.equal(final_codes, decoded_codes):
        raise RuntimeError(
            "global final subdivision support does not match decoded O-Voxel "
            "support; material/geometry correspondence is invalid"
        )
    del final_codes, decoded_codes, decoded_global_coords
    _atomic_json(
        global_baseline_dir / "global_shape_subdivision_hierarchy.json",
        {
            "source": "global shape_slat_decoder(..., return_subs=True)",
            "layers": global_subdivision_stats,
            "final_resolution": GRID_GLOBAL_UPSAMPLED,
        },
    )
    global_decoder_metadata = {
        "decoder_vertices": int(global_mesh.vertices.shape[0]),
        "decoder_faces": int(global_mesh.faces.shape[0]),
        "active_voxels": int(global_mesh.coords.shape[0]),
        "sample_type": type(global_mesh).__name__,
        "renderer": "pixal3d.utils.render_utils.render_frames",
    }

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

    # The live MeshWithVoxel and global C1024 subdivision leaves now own all
    # downstream global data. The modified route regenerates its shape SLat
    # from each tile image instead of retaining global shape-SLat features.
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
        "route_family": "ORIGINAL_DECODE_RENDER",
        "global_c32_tokens": int(coords32.shape[0]),
        "global_c64_tokens": int(coords64.shape[0]),
        "global_c1024_support_tokens": int(coords128.shape[0]),
        "global_shape_subdivision_layers": global_subdivision_stats,
        "shape512_seconds": global_shape512_seconds,
        "shape1024_seconds": global_shape1024_seconds,
        "texture1024_seconds": float(global_texture_seconds),
    }
    _atomic_json(global_baseline_dir / "summary.json", global_baseline_summary)
    global_baseline_render_path = Path(str(global_baseline_metric["render_png"]))
    # Retain geometry tensors only.  The original global O-Voxel material is
    # deliberately released and never used by the modified material route.
    global_geometry_vertices = global_mesh.vertices
    global_geometry_faces = global_mesh.faces
    del global_meshes, global_mesh, global_subs
    _empty_cuda_cache()

    global_summary = {
        "global_c32_tokens": int(coords32.shape[0]),
        "global_c64_tokens": int(coords64.shape[0]),
        "global_upsampled_tokens": int(coords128.shape[0]),
        "shape512_seconds": global_shape512_seconds,
        "shape1024_seconds": global_shape1024_seconds,
        "texture1024_seconds": float(global_texture_seconds),
        "one_step_upsample": upsample_stats,
        "shape_decoder_subdivision_layers": global_subdivision_stats,
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
        "modified_route_global_geometry_retained": True,
        "modified_route_global_material_retained": False,
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
            tile_dir / "global_to_tile_camera.json",
            {
                "mapping": (
                    "global camera projection -> tile crop coordinates -> "
                    "centered local-camera back-projection"
                ),
                "normalized_depth_policy": "q_local.z = q_global.z",
                "direction": "global_to_tile_only",
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
        modified_material_summary: Optional[Dict[str, Any]] = None
        native_tile_summary: Optional[Dict[str, Any]] = None
        failure: Optional[str] = None
        modified_material_failure: Optional[str] = None
        native_tile_failure: Optional[str] = None
        result: Optional[ModelResult] = None
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
            if bool(args.enable_modified_material_comparison):
                modified_dir = (
                    tile_dir
                    / (
                        "MODIFIED_TILE_SHAPE_FLOW_PROJECTED_GLOBAL_SUBS_TEXTURE_"
                        "GLOBAL_CAMERA"
                    )
                )
                try:
                    modified_material_summary = (
                        _render_modified_projected_global_subs_texture(
                            pipeline=pipeline,
                            global_vertices=global_geometry_vertices,
                            global_faces=global_geometry_faces,
                            global_children_by_layer=global_subdivision_children,
                            tile_image=reference_tile,
                            params=params,
                            seed=seed_tile + 1301,
                            global_camera=global_camera,
                            tile_camera=tile_camera,
                            transform=transform,
                            reference_image=tile_dir / "reference_tile.png",
                            output_dir=modified_dir,
                            args=args,
                            envmap=envmap,
                        )
                    )
                except Exception as exc:
                    modified_material_failure = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    print(
                        "[MODIFIED-PROJECTED-GLOBAL-SUBS-ERROR] "
                        f"tile={tile_id:02d}: {modified_material_failure}"
                    )
                    _atomic_json(
                        modified_dir / "failure.json",
                        {
                            "status": "failed",
                            "route_family": (
                                "MODIFIED_TILE_SHAPE_FLOW_PROJECTED_GLOBAL_SUBS_"
                                "TEXTURE_GLOBAL_CAMERA"
                            ),
                            "tile_id": int(tile_id),
                            "error": modified_material_failure,
                        },
                    )
                    _empty_cuda_cache()
            route_dir = tile_dir / "projected_global_c64_shape1024_texture"
            route_summary, _ = _evaluate_tile_result(
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
                retain_decoded_mesh=False,
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
                    "route_family": "ORIGINAL_DECODE_RENDER",
                }
            )
            _atomic_json(route_dir / "summary.json", route_summary)
            del result
            result = None
            _empty_cuda_cache()
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            print(f"[tile-generation-error] tile={tile_id:02d}: {failure}")
            if result is not None:
                del result
                result = None
            _empty_cuda_cache()

        if bool(args.enable_direct_tile_comparison):
            native_seed = seed_tile + 100_000
            native_dir = tile_dir / "direct_tile_ordinary_full_cascade"
            native_result: Optional[ModelResult] = None
            try:
                native_result = _run_tile_native_full_cascade(
                    pipeline=pipeline,
                    tile_image=reference_tile,
                    tile_camera=tile_camera,
                    params=params,
                    seed=native_seed,
                    label=f"Tile {tile_id:02d} direct-input ordinary",
                    max_tokens=int(args.max_num_tokens),
                )
                native_tile_summary, _ = _evaluate_tile_result(
                    pipeline=pipeline,
                    result=native_result,
                    output_dir=native_dir,
                    camera=tile_camera,
                    global_camera=global_camera,
                    transform=transform,
                    seed=native_seed,
                    label=f"Tile {tile_id:02d} direct-input ordinary",
                    reference_image=tile_dir / "reference_tile.png",
                    args=args,
                    envmap=envmap,
                    retain_decoded_mesh=False,
                )
                native_tile_summary.update(
                    {
                        "route": (
                            "direct tile image -> tile-native SS C32 -> "
                            "shape512 -> learned native C64 -> shape1024 -> "
                            "texture1024 -> decode"
                        ),
                        "route_family": "ORIGINAL_DECODE_RENDER",
                        "direct_tile_image_input": True,
                        "camera_protocol": (
                            "same centered tile camera as projected-C64 route "
                            "for an aligned image/geometry comparison"
                        ),
                        "tile_ss_c32_tokens": int(
                            native_result.tile_ss_c32_tokens
                        ),
                        "tile_native_c64_tokens": int(
                            native_result.tile_native_c64_tokens
                        ),
                        "tile_c64_tokens": int(native_result.tile_c64_tokens),
                        "tile_ss_seconds": float(
                            native_result.tile_ss_seconds
                        ),
                        "shape512_seconds": float(
                            native_result.shape512_seconds
                        ),
                        "shape1024_seconds": float(
                            native_result.shape1024_seconds
                        ),
                        "texture1024_seconds": float(
                            native_result.texture1024_seconds
                        ),
                    }
                )
                _atomic_json(native_dir / "summary.json", native_tile_summary)
                del native_result
                native_result = None
                _empty_cuda_cache()
            except Exception as exc:
                native_tile_failure = f"{type(exc).__name__}: {exc}"
                print(
                    f"[direct-tile-generation-error] tile={tile_id:02d}: "
                    f"{native_tile_failure}"
                )
                _atomic_json(
                    native_dir / "failure.json",
                    {
                        "status": "failed",
                        "route_family": "ORIGINAL_DECODE_RENDER",
                        "tile_id": int(tile_id),
                        "error": native_tile_failure,
                    },
                )
                if native_result is not None:
                    del native_result
                    native_result = None
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
            tile_dir
            / "comparison_all_original_and_MODIFIED_material_routes.png"
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
            "direct_tile_comparison_requested": bool(
                args.enable_direct_tile_comparison
            ),
            "modified_material_comparison_requested": bool(
                args.enable_modified_material_comparison
            ),
            "native_tile_failure": native_tile_failure,
            "modified_material_failure": modified_material_failure,
            "native_tile_psnr_db": (
                None
                if native_tile_summary is None
                else native_tile_summary.get("psnr_db")
            ),
            "native_tile_ssim": (
                None
                if native_tile_summary is None
                else native_tile_summary.get("ssim")
            ),
            "native_tile_lpips": (
                None
                if native_tile_summary is None
                else native_tile_summary.get("lpips")
            ),
            "native_tile_c32_tokens": (
                None
                if native_tile_summary is None
                else native_tile_summary.get("tile_ss_c32_tokens")
            ),
            "native_tile_c64_tokens": (
                None
                if native_tile_summary is None
                else native_tile_summary.get("tile_native_c64_tokens")
            ),
            "native_tile_render_png": (
                None
                if native_tile_summary is None
                else native_tile_summary.get("render_png")
            ),
            "native_tile_diff_heatmap_png": (
                None
                if native_tile_summary is None
                else native_tile_summary.get("diff_heatmap_png")
            ),
            "modified_local_psnr_db": (
                None
                if modified_material_summary is None
                else modified_material_summary["local_view"].get("psnr_db")
            ),
            "modified_local_ssim": (
                None
                if modified_material_summary is None
                else modified_material_summary["local_view"].get("ssim")
            ),
            "modified_local_lpips": (
                None
                if modified_material_summary is None
                else modified_material_summary["local_view"].get("lpips")
            ),
            "modified_local_render_png": (
                None
                if modified_material_summary is None
                else modified_material_summary["local_view"].get("render_png")
            ),
            "modified_local_diff_heatmap_png": (
                None
                if modified_material_summary is None
                else modified_material_summary["local_view"].get(
                    "diff_heatmap_png"
                )
            ),
            "modified_global_partial_render_png": (
                None
                if modified_material_summary is None
                else modified_material_summary["global_view"].get("display_png")
            ),
            "modified_global_material_coverage_png": (
                None
                if modified_material_summary is None
                else modified_material_summary["global_view"].get(
                    "material_coverage_png"
                )
            ),
            "modified_global_c1024_material_entries": (
                None
                if modified_material_summary is None
                else modified_material_summary["global_view"].get(
                    "global_c1024_material_entries"
                )
            ),
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
            f"projected_route_native_support=disabled "
            f"outside={outside_fraction:.6f} "
            f"PSNR={record['psnr_db']} SSIM={record['ssim']} "
            f"LPIPS={record['lpips']} "
            f"direct_tile={'ok' if native_tile_summary else 'unavailable'} "
            f"modified_material="
            f"{'ok' if modified_material_summary else 'unavailable'} "
            f"renderer={record['renderer']}"
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
        "format": (
            "pixal3d_tile_shape_flow_projected_global_subs_global_camera_"
            "comparison_v6"
        ),
        "image": str(Path(args.image).expanduser().resolve()),
        "global_camera": global_camera,
        "global_geometry": global_summary,
        "global_baseline_1024": global_baseline_summary,
        "comparison_protocol": (
            "render native MeshWithVoxel outputs with Pixal3D PbrMeshRenderer; "
            "render the single full global 1024-model baseline at "
            "baseline_render_resolution, crop it by each canonical 4096 tile "
            "box, and compare it with (a) projected-C64 tile generation, "
            "(b) direct-tile ordinary generation, and (c) an explicitly "
            "modified route that projects global C1024 subdivision leaves to "
            "local C64, regenerates tile shape SLat from the tile image, runs "
            "tile-image texture flow with that shape SLat, and builds all "
            "texture-decoder guides from the projected global C1024 leaves. "
            "The modified route emits an exact off-axis global-camera subimage "
            "and a full global-camera observation."
        ),
        "comparison_routes": {
            "global_baseline": {
                "route_family": "ORIGINAL_DECODE_RENDER",
                "description": "ordinary full-image Pixal3D model",
            },
            "projected_c64_tile": {
                "route_family": "ORIGINAL_DECODE_RENDER",
                "description": (
                    "projected global C1024 support quantized to local C64; "
                    "tile shape1024 and texture1024 generation"
                ),
            },
            "direct_tile_ordinary": {
                "enabled": bool(args.enable_direct_tile_comparison),
                "route_family": "ORIGINAL_DECODE_RENDER",
                "description": (
                    "tile image directly drives tile-native SS C32, shape512, "
                    "learned C64, shape1024, texture1024, and normal decode"
                ),
            },
            "modified_global_geometry_tile_material": {
                "enabled": bool(args.enable_modified_material_comparison),
                "route_family": (
                    "MODIFIED_TILE_SHAPE_FLOW_PROJECTED_GLOBAL_SUBS_TEXTURE_"
                    "GLOBAL_CAMERA"
                ),
                "explicitly_separate_from_original_route": True,
                "global_geometry_retained": True,
                "global_material_used": False,
                "tile_material_source": (
                    "tile-image texture flow conditioned on tile shape SLat "
                    "regenerated from the tile image on projected-global-"
                    "C1024-derived local C64 coordinates"
                ),
                "local_view": (
                    "unchanged global topology and C1024 material lattice; "
                    "global extrinsics with exact tile off-axis intrinsics"
                ),
                "guide_projection": (
                    "project valid global C1024 subdivision leaves with the "
                    "O-Voxel lattice convention, then coarsen them into an "
                    "exact local C64->C1024 parent/child tree"
                ),
                "global_view": (
                    "unchanged global topology; decoded tile attrs are assigned "
                    "to their exact source global C1024 children; geometry "
                    "without material is displayed using normal-vector color"
                ),
                "inverse_tile_to_global_route_used": False,
                "global_material_lattice_resolution": GRID_GLOBAL_UPSAMPLED,
                "global_render_resolution": int(
                    args.modified_global_render_resolution
                ),
            },
        },
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
            "scope": "projected-C64 tile route only",
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
        "material_mapping_chunk_size": int(args.material_mapping_chunk_size),
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
            "native_tile_psnr_db": _mean("native_tile_psnr_db"),
            "native_tile_ssim": _mean("native_tile_ssim"),
            "native_tile_lpips": _mean("native_tile_lpips"),
            "modified_local_psnr_db": _mean("modified_local_psnr_db"),
            "modified_local_ssim": _mean("modified_local_ssim"),
            "modified_local_lpips": _mean("modified_local_lpips"),
            "modified_global_c1024_material_entries": _mean(
                "modified_global_c1024_material_entries"
            ),
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
            "minimum unique local C64 tokens obtained by directly quantizing "
            "the transformed global C1024 support"
        ),
    )
    parser.add_argument("--max-num-tokens", type=int, default=100000000)
    parser.add_argument(
        "--enable-direct-tile-comparison",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also run the tile image through the ordinary tile-native "
            "SS/shape512/C64/shape1024/texture1024 cascade."
        ),
    )
    parser.add_argument(
        "--enable-modified-material-comparison",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Project global C1024 subdivision leaves into tile-local C64, "
            "regenerate tile shape SLat from the tile image on that support, "
            "run tile-image texture flow, and decode on guides coarsened from "
            "the projected global C1024 subdivision leaves."
        ),
    )
    parser.add_argument(
        "--material-mapping-chunk-size",
        type=int,
        default=262144,
        help=(
            "Chunk size for global-subdivision projection and global-geometry "
            "vertex transforms in the modified route."
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
    parser.add_argument("--render-resolution", type=int, default=2048)
    parser.add_argument(
        "--modified-global-render-resolution",
        type=int,
        default=2048,
        help=(
            "Image resolution of the modified global-camera observation. "
            "Missing material is displayed using normal-vector color."
        ),
    )
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
    parser.add_argument("--metric-resolution", type=int, default=2048)
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if int(args.min_tile_tokens) < 1:
        raise ValueError("--min-tile-tokens must be positive")
    if int(args.max_num_tokens) < int(args.min_tile_tokens):
        raise ValueError("--max-num-tokens must be >= --min-tile-tokens")
    if args.max_tiles is not None and int(args.max_tiles) < 1:
        raise ValueError("--max-tiles must be positive")
    if int(args.material_mapping_chunk_size) < 1:
        raise ValueError("--material-mapping-chunk-size must be positive")
    if args.cuda_device is not None and int(args.cuda_device) < 0:
        raise ValueError("--cuda-device must be non-negative")
    if (
        int(args.render_resolution) < 1
        or int(args.modified_global_render_resolution) < 1
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
