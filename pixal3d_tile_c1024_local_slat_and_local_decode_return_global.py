#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global Pixal3D-1024 baseline and an O-Voxel tile encode/decode test.

This script intentionally contains only two routes:

1. The ordinary Pixal3D ``1024_cascade`` image-to-3D route.
2. A reconstruction test driven by the O-Voxels decoded by route 1:

   global O-Voxel index
   -> global object-space cell center
   -> global q
   -> project to a canonical-4096 image tile
   -> exact centered local-camera q
   -> local object-space cell center / local O-Voxel index
   -> shape and PBR VAE encoders
   -> common C64 latent support
   -> ordinary Pixal3D shape/PBR decoders
   -> exact local-to-global camera transform
   -> comparison with the corresponding part of the original global mesh.

The camera mapping follows ``GLOBAL_MOGE_TO_LOCAL_TILE_CAMERA.md``.  No point
cloud centering, bounding-box normalization, clipping, or affine shortcut is
used.  A tile is reconstructed only when more than 1000 decoded global
O-Voxel centers project into it (the default threshold is therefore 1001).

The shape encoder cannot infer geometry from PBR O-Voxel attributes alone.
Consequently it encodes the part of the original global mesh owned by the
same tile, while the PBR encoder encodes the transformed O-Voxels.  Their
encoded C64 supports are intersected and put in the same deterministic order
before the normal joint decoder is called.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
)
os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "1")

import numpy as np
import o_voxel
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import pixal3d.models as pixal3d_models
from inference import (
    MODEL_PATH,
    distance_from_fov,
    get_camera_params_wild_moge,
    init_pipeline,
    load_moge_model,
)
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVoxel
from render_pixal3d_raw_ovoxel import (
    image_to_tensor,
    load_envmap,
    psnr_metric,
    render_and_evaluate_mesh,
    ssim_metric,
)


GLOBAL_IMAGE_SIZE = 1024
CANONICAL_IMAGE_SIZE = 4096
TILE_SIZE = 1024
TILE_STRIDE = 512
OVOXEL_RESOLUTION = 1024
LATENT_RESOLUTION = 64
DEFAULT_ENCODER_ROOT = Path(
    "/home/nvme04/yyyan/download/model/TRELLIS.2-4B/"
    "microsoft/TRELLIS___2-4B/ckpts"
)


@dataclass(frozen=True)
class TileCameraTransform:
    tile_id: int
    box: Tuple[int, int, int, int]
    output_width: int
    output_height: int
    camera_angle_x: float
    camera_angle_y: float
    distance: float
    mesh_scale: float
    global_distance: float
    global_mesh_scale: float
    global_fx_1024: float
    global_fy_1024: float
    full_fx_4096: float
    full_fy_4096: float
    full_cx_4096: float
    full_cy_4096: float
    crop_to_output_scale_x: float
    crop_to_output_scale_y: float
    fx: float
    fy: float
    cx: float
    cy: float
    offaxis_cx: float
    offaxis_cy: float
    tile_center_full_x: float
    tile_center_full_y: float


@dataclass
class LocalOVoxelMapping:
    local_coords: torch.Tensor
    local_attrs: torch.Tensor
    source_global_rows: torch.Tensor
    local_q: torch.Tensor
    stats: Dict[str, Any]


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
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _tensor_range(value: torch.Tensor) -> Dict[str, List[float]]:
    if value.numel() == 0:
        return {"min": [], "max": []}
    return {
        "min": [float(v) for v in value.amin(dim=0).detach().cpu().tolist()],
        "max": [float(v) for v in value.amax(dim=0).detach().cpu().tolist()],
    }


def _tile_layout(
    canonical_size: int = CANONICAL_IMAGE_SIZE,
    tile_size: int = TILE_SIZE,
    stride: int = TILE_STRIDE,
) -> List[Tuple[int, int, int, int]]:
    if canonical_size <= 0 or tile_size <= 0 or stride <= 0:
        raise ValueError("canonical size, tile size, and stride must be positive")
    starts = list(range(0, canonical_size - tile_size + 1, stride))
    if not starts or starts[-1] != canonical_size - tile_size:
        raise ValueError("tile layout does not land exactly on the canonical edge")
    return [
        (x0, y0, x0 + tile_size, y0 + tile_size)
        for y0 in starts
        for x0 in starts
    ]


def _parse_tile_ids(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not value.strip():
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _focal_pixels(camera_angle: float, resolution: int) -> float:
    return float(resolution) / (2.0 * math.tan(float(camera_angle) / 2.0))


def _derive_tile_camera(
    *,
    tile_id: int,
    box: Sequence[int],
    global_camera: Mapping[str, float],
    extend_pixel: int,
) -> TileCameraTransform:
    x0, y0, x1, y1 = (int(v) for v in box)
    crop_width = x1 - x0
    crop_height = y1 - y0
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("tile crop has non-positive dimensions")

    rx = TILE_SIZE / float(crop_width)
    ry = TILE_SIZE / float(crop_height)
    global_fx = _focal_pixels(
        float(global_camera["camera_angle_x"]), GLOBAL_IMAGE_SIZE
    )
    global_fy = global_fx
    full_fx = global_fx * CANONICAL_IMAGE_SIZE / GLOBAL_IMAGE_SIZE
    full_fy = global_fy * CANONICAL_IMAGE_SIZE / GLOBAL_IMAGE_SIZE
    full_cx = CANONICAL_IMAGE_SIZE / 2.0
    full_cy = CANONICAL_IMAGE_SIZE / 2.0

    local_fx = full_fx * rx
    local_fy = full_fy * ry
    local_cx = TILE_SIZE / 2.0
    local_cy = TILE_SIZE / 2.0
    angle_x = 2.0 * math.atan(TILE_SIZE / (2.0 * local_fx))
    angle_y = 2.0 * math.atan(TILE_SIZE / (2.0 * local_fy))
    mesh_scale = float(global_camera["mesh_scale"])
    distance = distance_from_fov(
        angle_x,
        torch.tensor([-1.0, 0.0, 0.0]),
        torch.tensor(
            [
                0.0 - float(extend_pixel),
                TILE_SIZE - 1.0 + float(extend_pixel),
            ]
        ),
        mesh_scale,
        TILE_SIZE,
    )["distance_from_x"]

    return TileCameraTransform(
        tile_id=int(tile_id),
        box=(x0, y0, x1, y1),
        output_width=TILE_SIZE,
        output_height=TILE_SIZE,
        camera_angle_x=float(angle_x),
        camera_angle_y=float(angle_y),
        distance=float(distance),
        mesh_scale=mesh_scale,
        global_distance=float(global_camera["distance"]),
        global_mesh_scale=mesh_scale,
        global_fx_1024=float(global_fx),
        global_fy_1024=float(global_fy),
        full_fx_4096=float(full_fx),
        full_fy_4096=float(full_fy),
        full_cx_4096=float(full_cx),
        full_cy_4096=float(full_cy),
        crop_to_output_scale_x=float(rx),
        crop_to_output_scale_y=float(ry),
        fx=float(local_fx),
        fy=float(local_fy),
        cx=float(local_cx),
        cy=float(local_cy),
        offaxis_cx=float((full_cx - x0) * rx),
        offaxis_cy=float((full_cy - y0) * ry),
        tile_center_full_x=float(x0 + local_cx / rx),
        tile_center_full_y=float(y0 + local_cy / ry),
    )


def _camera_q_to_points(
    q: torch.Tensor,
    *,
    distance: float,
    mesh_scale: float,
) -> torch.Tensor:
    center = q.new_tensor([0.0, 0.0, -float(distance)])
    return center[None] + q / (2.0 * float(mesh_scale))


def _camera_points_to_q(
    points: torch.Tensor,
    *,
    distance: float,
    mesh_scale: float,
) -> torch.Tensor:
    center = points.new_tensor([0.0, 0.0, -float(distance)])
    return (points - center[None]) * (2.0 * float(mesh_scale))


def _project_points(
    points: torch.Tensor,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    depth = -points[:, 2]
    finite = torch.isfinite(points).all(dim=1) & torch.isfinite(depth) & (depth > 0)
    safe_depth = torch.where(finite, depth, torch.ones_like(depth))
    u = float(fx) * points[:, 0] / safe_depth + float(cx)
    v = -float(fy) * points[:, 1] / safe_depth + float(cy)
    uv = torch.stack((u, v), dim=1)
    finite &= torch.isfinite(uv).all(dim=1)
    return uv, depth, finite


def _project_global_q_to_4096(
    q_global: torch.Tensor,
    *,
    global_camera: Mapping[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points = _camera_q_to_points(
        q_global,
        distance=float(global_camera["distance"]),
        mesh_scale=float(global_camera["mesh_scale"]),
    )
    focal = _focal_pixels(
        float(global_camera["camera_angle_x"]), GLOBAL_IMAGE_SIZE
    ) * (CANONICAL_IMAGE_SIZE / GLOBAL_IMAGE_SIZE)
    return _project_points(
        points,
        fx=focal,
        fy=focal,
        cx=CANONICAL_IMAGE_SIZE / 2.0,
        cy=CANONICAL_IMAGE_SIZE / 2.0,
    )


def _global_q_to_local_q(
    q_global: torch.Tensor,
    *,
    global_camera: Mapping[str, float],
    transform: TileCameraTransform,
) -> Tuple[torch.Tensor, torch.Tensor]:
    uv_full, _, finite = _project_global_q_to_4096(
        q_global, global_camera=global_camera
    )
    if not bool(finite.all().item()):
        raise RuntimeError("global-to-local input contains invalid camera projections")

    x0, y0, _, _ = transform.box
    uv_tile = torch.empty_like(uv_full)
    uv_tile[:, 0] = (
        uv_full[:, 0] - float(x0)
    ) * transform.crop_to_output_scale_x
    uv_tile[:, 1] = (
        uv_full[:, 1] - float(y0)
    ) * transform.crop_to_output_scale_y

    qz = q_global[:, 2]
    local_depth = float(transform.distance) - qz / (
        2.0 * float(transform.mesh_scale)
    )
    x = (uv_tile[:, 0] - float(transform.cx)) * local_depth / float(
        transform.fx
    )
    y = -(uv_tile[:, 1] - float(transform.cy)) * local_depth / float(
        transform.fy
    )
    local_points = torch.stack((x, y, -local_depth), dim=1)
    q_local = _camera_points_to_q(
        local_points,
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
    )
    return q_local, uv_tile


def _local_q_to_global_q(
    q_local: torch.Tensor,
    *,
    global_camera: Mapping[str, float],
    transform: TileCameraTransform,
) -> Tuple[torch.Tensor, torch.Tensor]:
    local_points = _camera_q_to_points(
        q_local,
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
    )
    uv_tile, _, finite = _project_points(
        local_points,
        fx=float(transform.fx),
        fy=float(transform.fy),
        cx=float(transform.cx),
        cy=float(transform.cy),
    )
    if not bool(finite.all().item()):
        raise RuntimeError("local-to-global input contains invalid camera projections")

    x0, y0, _, _ = transform.box
    uv_full = torch.empty_like(uv_tile)
    uv_full[:, 0] = (
        uv_tile[:, 0] / transform.crop_to_output_scale_x + float(x0)
    )
    uv_full[:, 1] = (
        uv_tile[:, 1] / transform.crop_to_output_scale_y + float(y0)
    )

    qz = q_local[:, 2]
    global_scale = float(global_camera["mesh_scale"])
    global_depth = float(global_camera["distance"]) - qz / (2.0 * global_scale)
    focal = transform.full_fx_4096
    x = (uv_full[:, 0] - transform.full_cx_4096) * global_depth / focal
    y = -(uv_full[:, 1] - transform.full_cy_4096) * global_depth / (
        transform.full_fy_4096
    )
    global_points = torch.stack((x, y, -global_depth), dim=1)
    q_global = _camera_points_to_q(
        global_points,
        distance=float(global_camera["distance"]),
        mesh_scale=global_scale,
    )
    return q_global, uv_full


def _inside_tile(
    uv: torch.Tensor,
    finite: torch.Tensor,
    box: Sequence[int],
) -> torch.Tensor:
    x0, y0, x1, y1 = (float(v) for v in box)
    return (
        finite
        & (uv[:, 0] >= x0)
        & (uv[:, 0] < x1)
        & (uv[:, 1] >= y0)
        & (uv[:, 1] < y1)
    )


def _to_vec3(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if tensor.numel() == 1:
        tensor = tensor.repeat(3)
    if tensor.numel() != 3:
        raise ValueError(f"expected scalar or length-3 value, got {tensor.tolist()}")
    return tensor


def _spatial_shape3(value: Any, *, device: torch.device) -> torch.Tensor:
    shape = torch.as_tensor(value, device=device, dtype=torch.int64).reshape(-1)
    if shape.numel() == 1:
        shape = shape.repeat(3)
    elif shape.numel() >= 3:
        shape = shape[-3:]
    else:
        raise ValueError("voxel shape must contain one or at least three entries")
    if bool((shape <= 0).any().item()):
        raise ValueError("voxel shape dimensions must be positive")
    return shape


def _ovoxel_indices_to_object(
    coords: torch.Tensor,
    *,
    origin: Any,
    voxel_size: Any,
) -> torch.Tensor:
    origin3 = _to_vec3(origin, device=coords.device, dtype=torch.float32)
    size3 = _to_vec3(voxel_size, device=coords.device, dtype=torch.float32)
    return origin3[None] + (coords.to(torch.float32) + 0.5) * size3[None]


def _object_to_nearest_ovoxel_indices(
    points: torch.Tensor,
    *,
    origin: Any,
    voxel_size: Any,
    voxel_shape: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    origin3 = _to_vec3(origin, device=points.device, dtype=points.dtype)
    size3 = _to_vec3(voxel_size, device=points.device, dtype=points.dtype)
    shape3 = _spatial_shape3(voxel_shape, device=points.device)
    continuous = (points - origin3[None]) / size3[None] - 0.5
    coords = torch.round(continuous).to(torch.int64)
    valid = (
        torch.isfinite(points).all(dim=1)
        & ((coords >= 0) & (coords < shape3[None])).all(dim=1)
    )
    centers = origin3[None] + (coords.to(points.dtype) + 0.5) * size3[None]
    error = torch.linalg.vector_norm(points - centers, dim=1)
    return coords.to(torch.int32), valid, error


def _linear_keys(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    xyz = coords.to(torch.int64)
    return (
        (xyz[:, 0] * int(resolution) + xyz[:, 1]) * int(resolution)
        + xyz[:, 2]
    )


def _unique_by_key_then_error(
    coords: torch.Tensor,
    error: torch.Tensor,
) -> torch.Tensor:
    if coords.shape[0] == 0:
        return torch.empty(0, device=coords.device, dtype=torch.long)
    resolution = int(coords.to(torch.int64).amax().item()) + 1
    keys = _linear_keys(coords, max(resolution, 1))
    by_error = torch.argsort(error, stable=True)
    by_key_relative = torch.argsort(keys[by_error], stable=True)
    order = by_error[by_key_relative]
    sorted_keys = keys[order]
    keep = torch.ones(order.shape[0], dtype=torch.bool, device=coords.device)
    keep[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return order[keep]


def _map_global_ovoxels_to_local(
    *,
    global_mesh: MeshWithVoxel,
    global_q: torch.Tensor,
    global_uv_4096: torch.Tensor,
    finite_projection: torch.Tensor,
    global_camera: Mapping[str, float],
    transform: TileCameraTransform,
) -> LocalOVoxelMapping:
    selected_mask = _inside_tile(
        global_uv_4096, finite_projection, transform.box
    )
    selected_rows = torch.where(selected_mask)[0]
    if selected_rows.numel() == 0:
        return LocalOVoxelMapping(
            local_coords=torch.empty((0, 3), dtype=torch.int32),
            local_attrs=torch.empty((0, global_mesh.attrs.shape[1])),
            source_global_rows=torch.empty(0, dtype=torch.long),
            local_q=torch.empty((0, 3)),
            stats={"projected_global_ovoxels": 0},
        )

    q_selected = global_q.index_select(0, selected_rows)
    q_local, uv_tile = _global_q_to_local_q(
        q_selected,
        global_camera=global_camera,
        transform=transform,
    )
    local_object = q_local / (2.0 * float(transform.mesh_scale))
    local_coords, valid_grid, quantization_error = (
        _object_to_nearest_ovoxel_indices(
            local_object,
            origin=[-0.5, -0.5, -0.5],
            voxel_size=1.0 / OVOXEL_RESOLUTION,
            voxel_shape=[OVOXEL_RESOLUTION] * 3,
        )
    )
    inside_cube = ((q_local >= -1.0) & (q_local <= 1.0)).all(dim=1)
    valid = (
        valid_grid
        & inside_cube
        & torch.isfinite(q_local).all(dim=1)
        & torch.isfinite(quantization_error)
    )
    valid_rows = torch.where(valid)[0]
    coords_valid = local_coords.index_select(0, valid_rows)
    errors_valid = quantization_error.index_select(0, valid_rows)
    keep_relative = _unique_by_key_then_error(coords_valid, errors_valid)
    kept_selected_rows = valid_rows.index_select(0, keep_relative)
    source_rows = selected_rows.index_select(0, kept_selected_rows)
    local_coords_unique = local_coords.index_select(0, kept_selected_rows)
    local_attrs = global_mesh.attrs.index_select(0, source_rows).to(torch.float32)
    local_q_unique = q_local.index_select(0, kept_selected_rows)

    roundtrip, _ = _local_q_to_global_q(
        local_q_unique,
        global_camera=global_camera,
        transform=transform,
    )
    roundtrip_error = (
        roundtrip
        - global_q.index_select(0, source_rows).to(roundtrip.device)
    ).abs()
    stats = {
        "projected_global_ovoxels": int(selected_rows.numel()),
        "local_cube_valid_rows": int(valid_rows.numel()),
        "local_cube_dropped_rows": int(selected_rows.numel() - valid_rows.numel()),
        "unique_local_ovoxels": int(local_coords_unique.shape[0]),
        "local_quantization_collisions": int(
            valid_rows.numel() - local_coords_unique.shape[0]
        ),
        "local_quantization_error_object_mean": float(
            errors_valid.index_select(0, keep_relative).mean().item()
        ),
        "local_quantization_error_object_max": float(
            errors_valid.index_select(0, keep_relative).max().item()
        ),
        "global_local_global_q_max_abs_error": float(roundtrip_error.max().item()),
        "global_local_global_q_mean_abs_error": float(roundtrip_error.mean().item()),
        "selected_tile_uv_range": _tensor_range(
            uv_tile.index_select(0, kept_selected_rows)
        ),
        "local_q_range": _tensor_range(local_q_unique),
    }
    return LocalOVoxelMapping(
        local_coords=local_coords_unique.to(device="cpu", dtype=torch.int32),
        local_attrs=local_attrs.to(device="cpu", dtype=torch.float32),
        source_global_rows=source_rows.to(device="cpu", dtype=torch.long),
        local_q=local_q_unique.to(device="cpu", dtype=torch.float32),
        stats=stats,
    )


def _project_face_centers(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    mesh_scale: float,
    global_camera: Mapping[str, float],
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if chunk_size <= 0:
        raise ValueError("face projection chunk size must be positive")
    uv_chunks: List[torch.Tensor] = []
    valid_chunks: List[torch.Tensor] = []
    for start in range(0, int(faces.shape[0]), int(chunk_size)):
        chunk = faces[start : start + chunk_size].to(torch.long)
        centers = vertices.index_select(0, chunk.reshape(-1)).reshape(
            -1, 3, 3
        ).mean(dim=1)
        q = centers * (2.0 * float(mesh_scale))
        uv, _, valid = _project_global_q_to_4096(
            q, global_camera=global_camera
        )
        uv_chunks.append(uv.to(device="cpu", dtype=torch.float32))
        valid_chunks.append(valid.to(device="cpu"))
    return torch.cat(uv_chunks, dim=0), torch.cat(valid_chunks, dim=0)


def _compact_submesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if faces.numel() == 0:
        raise RuntimeError("cannot compact an empty face set")
    vertex_ids, inverse = torch.unique(
        faces.reshape(-1).to(torch.long),
        sorted=True,
        return_inverse=True,
    )
    compact_vertices = vertices.index_select(0, vertex_ids)
    compact_faces = inverse.reshape(-1, 3).to(torch.int32)
    return compact_vertices, compact_faces, vertex_ids


def _prepare_tile_geometry(
    *,
    global_vertices: torch.Tensor,
    global_faces: torch.Tensor,
    global_face_uv: torch.Tensor,
    global_face_finite: torch.Tensor,
    global_camera: Mapping[str, float],
    transform: TileCameraTransform,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    face_mask = _inside_tile(
        global_face_uv, global_face_finite, transform.box
    )
    projected_face_ids = torch.where(face_mask)[0]
    if projected_face_ids.numel() == 0:
        raise RuntimeError("no global mesh face centroid projects into the tile")
    projected_faces = global_faces.index_select(0, projected_face_ids).to(torch.long)

    _, face_local0, vertex_ids0 = _compact_submesh(
        global_vertices, projected_faces
    )
    q_global0 = (
        global_vertices.index_select(0, vertex_ids0)
        * (2.0 * float(global_camera["mesh_scale"]))
    )
    q_local0, _ = _global_q_to_local_q(
        q_global0,
        global_camera=global_camera,
        transform=transform,
    )
    vertex_inside = (
        torch.isfinite(q_local0).all(dim=1)
        & ((q_local0 >= -1.0) & (q_local0 <= 1.0)).all(dim=1)
    )
    face_encodable = vertex_inside.index_select(
        0, face_local0.to(torch.long).reshape(-1)
    ).reshape(-1, 3).all(dim=1)
    encodable_faces = projected_faces.index_select(
        0, torch.where(face_encodable)[0]
    )
    if encodable_faces.numel() == 0:
        raise RuntimeError(
            "no projected global faces have all vertices inside the local cube"
        )

    reference_vertices, reference_faces, final_vertex_ids = _compact_submesh(
        global_vertices, encodable_faces
    )
    q_global = reference_vertices * (
        2.0 * float(global_camera["mesh_scale"])
    )
    q_local, _ = _global_q_to_local_q(
        q_global,
        global_camera=global_camera,
        transform=transform,
    )
    local_vertices = q_local / (2.0 * float(transform.mesh_scale))

    triangles = local_vertices.index_select(
        0, reference_faces.to(torch.long).reshape(-1)
    ).reshape(-1, 3, 3)
    twice_area = torch.linalg.vector_norm(
        torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=1,
        ),
        dim=1,
    )
    nondegenerate = torch.isfinite(twice_area) & (twice_area > 1e-12)
    if not bool(nondegenerate.all().item()):
        surviving = reference_faces.index_select(
            0, torch.where(nondegenerate)[0]
        )
        reference_vertices, reference_faces, relative_ids = _compact_submesh(
            reference_vertices, surviving
        )
        final_vertex_ids = final_vertex_ids.index_select(0, relative_ids)
        q_global = reference_vertices * (
            2.0 * float(global_camera["mesh_scale"])
        )
        q_local, _ = _global_q_to_local_q(
            q_global,
            global_camera=global_camera,
            transform=transform,
        )
        local_vertices = q_local / (2.0 * float(transform.mesh_scale))

    stats = {
        "projected_global_faces": int(projected_faces.shape[0]),
        "local_cube_encodable_faces": int(reference_faces.shape[0]),
        "local_cube_dropped_faces": int(
            projected_faces.shape[0] - reference_faces.shape[0]
        ),
        "reference_vertices": int(reference_vertices.shape[0]),
        "local_vertex_range": _tensor_range(local_vertices),
        "source_global_vertex_ids": int(final_vertex_ids.shape[0]),
    }
    return (
        local_vertices.to(device="cpu", dtype=torch.float32),
        reference_faces.to(device="cpu", dtype=torch.int64),
        reference_vertices.to(device="cpu", dtype=torch.float32),
        reference_faces.to(device="cpu", dtype=torch.int32),
        stats,
    )


def _encode_local_shape(
    *,
    encoder: torch.nn.Module,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    device: torch.device,
    low_vram: bool,
) -> Tuple[SparseTensor, Dict[str, Any]]:
    started = time.perf_counter()
    voxel_indices, dual_vertices, intersected = (
        o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices=vertices.to(device="cpu", dtype=torch.float32),
            faces=faces.to(device="cpu", dtype=torch.long),
            grid_size=OVOXEL_RESOLUTION,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=True,
        )
    )
    if voxel_indices.shape[0] == 0:
        raise RuntimeError("local mesh produced an empty flexible dual grid")
    dual_features = (
        dual_vertices.to(torch.float32) * OVOXEL_RESOLUTION
        - voxel_indices.to(torch.float32)
    )
    coords = torch.cat(
        [
            torch.zeros_like(voxel_indices[:, :1]),
            voxel_indices,
        ],
        dim=1,
    ).to(torch.int32)
    vertex_sparse = SparseTensor(dual_features, coords).to(device)
    intersected_sparse = vertex_sparse.replace(intersected).to(device)
    if low_vram:
        encoder.to(device)
    with torch.no_grad():
        latent = encoder(
            vertex_sparse,
            intersected_sparse,
            sample_posterior=False,
        )
    _sync_cuda()
    if low_vram:
        encoder.cpu()
    if not torch.isfinite(latent.feats).all():
        raise RuntimeError("shape encoder produced non-finite latent features")
    stats = {
        "dual_grid_entries": int(voxel_indices.shape[0]),
        "shape_latent_tokens_before_alignment": int(latent.feats.shape[0]),
        "shape_encoder_seconds": float(time.perf_counter() - started),
    }
    del vertex_sparse, intersected_sparse, dual_vertices, intersected
    _empty_cuda_cache()
    return latent, stats


def _encode_local_pbr(
    *,
    encoder: torch.nn.Module,
    coords: torch.Tensor,
    attrs: torch.Tensor,
    device: torch.device,
    low_vram: bool,
) -> Tuple[SparseTensor, Dict[str, Any]]:
    started = time.perf_counter()
    coords4 = torch.cat(
        [torch.zeros_like(coords[:, :1]), coords], dim=1
    ).to(device=device, dtype=torch.int32)
    # The released PBR encoder was trained on attributes in [-1, 1].
    feats = (attrs.to(device=device, dtype=torch.float32) * 2.0 - 1.0).clamp(
        -1.0, 1.0
    )
    sparse = SparseTensor(feats, coords4)
    if low_vram:
        encoder.to(device)
    with torch.no_grad():
        latent = encoder(sparse, sample_posterior=False)
    _sync_cuda()
    if low_vram:
        encoder.cpu()
    if not torch.isfinite(latent.feats).all():
        raise RuntimeError("PBR encoder produced non-finite latent features")
    stats = {
        "pbr_encoder_input_ovoxels": int(coords.shape[0]),
        "pbr_latent_tokens_before_alignment": int(latent.feats.shape[0]),
        "pbr_encoder_seconds": float(time.perf_counter() - started),
    }
    del sparse, feats
    _empty_cuda_cache()
    return latent, stats


def _align_latent_supports(
    shape_latent: SparseTensor,
    pbr_latent: SparseTensor,
) -> Tuple[SparseTensor, SparseTensor, Dict[str, Any]]:
    shape_coords = shape_latent.coords.to(torch.int64)
    pbr_coords = pbr_latent.coords.to(torch.int64)
    if shape_coords.shape[1] != 4 or pbr_coords.shape[1] != 4:
        raise ValueError("encoded latent coordinates must have shape [N,4]")
    if bool((shape_coords[:, 0] != 0).any().item()) or bool(
        (pbr_coords[:, 0] != 0).any().item()
    ):
        raise ValueError("only batch zero is supported")

    shape_keys = _linear_keys(shape_coords[:, 1:], LATENT_RESOLUTION)
    pbr_keys = _linear_keys(pbr_coords[:, 1:], LATENT_RESOLUTION)
    shape_order = torch.argsort(shape_keys)
    pbr_order = torch.argsort(pbr_keys)
    shape_sorted = shape_keys[shape_order]
    pbr_sorted = pbr_keys[pbr_order]
    positions = torch.searchsorted(pbr_sorted, shape_sorted)
    in_bounds = positions < pbr_sorted.shape[0]
    safe_positions = positions.clamp_max(max(0, pbr_sorted.shape[0] - 1))
    common = in_bounds & (
        pbr_sorted.index_select(0, safe_positions) == shape_sorted
    )
    shape_rows = shape_order.index_select(0, torch.where(common)[0])
    pbr_rows = pbr_order.index_select(
        0, positions.index_select(0, torch.where(common)[0])
    )
    if shape_rows.numel() == 0:
        raise RuntimeError("shape and PBR encoders have no common C64 support")

    aligned_coords = shape_coords.index_select(0, shape_rows).to(torch.int32)
    aligned_shape = SparseTensor(
        shape_latent.feats.index_select(0, shape_rows),
        aligned_coords,
    )
    aligned_pbr = SparseTensor(
        pbr_latent.feats.index_select(0, pbr_rows),
        aligned_coords,
    )
    if not torch.equal(aligned_shape.coords, aligned_pbr.coords):
        raise RuntimeError("aligned shape/PBR latent coordinates differ")
    stats = {
        "shape_latent_tokens_before_alignment": int(shape_coords.shape[0]),
        "pbr_latent_tokens_before_alignment": int(pbr_coords.shape[0]),
        "common_latent_tokens": int(aligned_coords.shape[0]),
        "shape_only_tokens_dropped": int(
            shape_coords.shape[0] - aligned_coords.shape[0]
        ),
        "pbr_only_tokens_dropped": int(
            pbr_coords.shape[0] - aligned_coords.shape[0]
        ),
        "latent_resolution": LATENT_RESOLUTION,
        "alignment": "exact C64 coordinate intersection; common sorted order",
    }
    return aligned_shape, aligned_pbr, stats


def _validate_mesh(mesh: Any, label: str) -> MeshWithVoxel:
    if not isinstance(mesh, MeshWithVoxel):
        raise TypeError(f"{label}: expected MeshWithVoxel, got {type(mesh)!r}")
    if mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3:
        raise ValueError(f"{label}: vertices must have shape [N,3]")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError(f"{label}: faces must have shape [M,3]")
    if mesh.coords.ndim != 2 or mesh.coords.shape[1] != 3:
        raise ValueError(f"{label}: O-Voxel coords must have shape [L,3]")
    if mesh.attrs.ndim != 2 or mesh.attrs.shape[0] != mesh.coords.shape[0]:
        raise ValueError(f"{label}: O-Voxel attrs and coords are not aligned")
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise RuntimeError(f"{label}: decoded mesh is empty")
    if not torch.isfinite(mesh.vertices).all() or not torch.isfinite(
        mesh.attrs
    ).all():
        raise RuntimeError(f"{label}: mesh contains non-finite tensors")
    return mesh


def _make_mesh_with_voxel(
    *,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    coords: torch.Tensor,
    attrs: torch.Tensor,
    template: MeshWithVoxel,
) -> MeshWithVoxel:
    return MeshWithVoxel(
        vertices=vertices.to(torch.float32),
        faces=faces.to(torch.int32),
        origin=torch.as_tensor(template.origin).detach().cpu().tolist(),
        voxel_size=float(torch.as_tensor(template.voxel_size).item()),
        coords=coords.to(torch.int32),
        attrs=attrs.to(torch.float32),
        voxel_shape=torch.Size(template.voxel_shape),
        layout=dict(template.layout),
    )


def _make_reference_tile_mesh(
    *,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    mapping: LocalOVoxelMapping,
    global_mesh: MeshWithVoxel,
) -> MeshWithVoxel:
    source_rows = mapping.source_global_rows.to(torch.long)
    coords = global_mesh.coords.index_select(0, source_rows)
    attrs = global_mesh.attrs.index_select(0, source_rows)
    return _make_mesh_with_voxel(
        vertices=vertices,
        faces=faces,
        coords=coords,
        attrs=attrs,
        template=global_mesh,
    )


def _return_decoded_mesh_to_global(
    *,
    local_mesh: MeshWithVoxel,
    global_template: MeshWithVoxel,
    global_camera: Mapping[str, float],
    transform: TileCameraTransform,
    face_chunk_size: int,
) -> Tuple[MeshWithVoxel, Dict[str, Any]]:
    local_vertices = local_mesh.vertices.to(torch.float32)
    q_local_vertices = local_vertices * (2.0 * float(transform.mesh_scale))
    q_global_vertices, _ = _local_q_to_global_q(
        q_local_vertices,
        global_camera=global_camera,
        transform=transform,
    )
    global_vertices = q_global_vertices / (
        2.0 * float(global_camera["mesh_scale"])
    )

    uv_faces, valid_faces = _project_face_centers(
        global_vertices,
        local_mesh.faces,
        mesh_scale=float(global_camera["mesh_scale"]),
        global_camera=global_camera,
        chunk_size=face_chunk_size,
    )
    owned_face_mask = _inside_tile(uv_faces, valid_faces, transform.box)
    owned_faces = local_mesh.faces.to(device="cpu").index_select(
        0, torch.where(owned_face_mask)[0]
    )
    if owned_faces.numel() == 0:
        raise RuntimeError("returned local decode has no face owned by the tile")
    compact_vertices, compact_faces, _ = _compact_submesh(
        global_vertices.to(device="cpu"), owned_faces
    )

    local_ovoxel_object = _ovoxel_indices_to_object(
        local_mesh.coords,
        origin=local_mesh.origin,
        voxel_size=local_mesh.voxel_size,
    )
    q_local_ovoxel = local_ovoxel_object * (
        2.0 * float(transform.mesh_scale)
    )
    q_global_ovoxel, uv_global_ovoxel = _local_q_to_global_q(
        q_local_ovoxel,
        global_camera=global_camera,
        transform=transform,
    )
    global_ovoxel_object = q_global_ovoxel / (
        2.0 * float(global_camera["mesh_scale"])
    )
    global_coords, valid_grid, quant_error = _object_to_nearest_ovoxel_indices(
        global_ovoxel_object,
        origin=global_template.origin,
        voxel_size=global_template.voxel_size,
        voxel_shape=global_template.voxel_shape,
    )
    finite_projection = torch.isfinite(uv_global_ovoxel).all(dim=1)
    owned_voxel = valid_grid & _inside_tile(
        uv_global_ovoxel, finite_projection, transform.box
    )
    owned_rows = torch.where(owned_voxel)[0]
    if owned_rows.numel() == 0:
        raise RuntimeError("returned local decode has no global O-Voxel in the tile")
    coords_owned = global_coords.index_select(0, owned_rows)
    error_owned = quant_error.index_select(0, owned_rows)
    keep_relative = _unique_by_key_then_error(coords_owned, error_owned)
    kept_rows = owned_rows.index_select(0, keep_relative)
    coords_unique = global_coords.index_select(0, kept_rows)
    attrs_unique = local_mesh.attrs.index_select(0, kept_rows)

    returned = _make_mesh_with_voxel(
        vertices=compact_vertices,
        faces=compact_faces,
        coords=coords_unique.to(device="cpu"),
        attrs=attrs_unique.to(device="cpu"),
        template=global_template,
    )
    stats = {
        "local_decoder_vertices": int(local_mesh.vertices.shape[0]),
        "local_decoder_faces": int(local_mesh.faces.shape[0]),
        "local_decoder_ovoxels": int(local_mesh.coords.shape[0]),
        "returned_global_vertices": int(returned.vertices.shape[0]),
        "returned_global_faces": int(returned.faces.shape[0]),
        "returned_global_ovoxels": int(returned.coords.shape[0]),
        "faces_outside_tile_dropped": int(
            local_mesh.faces.shape[0] - returned.faces.shape[0]
        ),
        "returned_ovoxel_quantization_collisions": int(
            owned_rows.numel() - coords_unique.shape[0]
        ),
        "returned_ovoxel_quantization_error_object_mean": float(
            error_owned.index_select(0, keep_relative).mean().item()
        ),
        "returned_ovoxel_quantization_error_object_max": float(
            error_owned.index_select(0, keep_relative).max().item()
        ),
        "returned_global_vertex_range": _tensor_range(returned.vertices),
    }
    return returned, stats


def _sample_mesh_surface(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    samples: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if samples <= 0:
        raise ValueError("surface sample count must be positive")
    triangles = vertices.index_select(
        0, faces.to(torch.long).reshape(-1)
    ).reshape(-1, 3, 3)
    cross = torch.linalg.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=1,
    )
    twice_area = torch.linalg.vector_norm(cross, dim=1)
    valid = torch.isfinite(twice_area) & (twice_area > 1e-14)
    triangles = triangles[valid]
    cross = cross[valid]
    twice_area = twice_area[valid]
    if triangles.shape[0] == 0:
        raise RuntimeError("mesh has no non-degenerate triangle for sampling")
    generator = torch.Generator(device=vertices.device)
    generator.manual_seed(int(seed))
    # torch.multinomial rejects more than 2^24 categories.  A low-step smoke
    # generation can legitimately produce that many decoded triangles, so
    # sample the same area distribution through its CDF without that limit.
    area_cdf = torch.cumsum(twice_area.to(torch.float32), dim=0)
    total_area = area_cdf[-1]
    if not torch.isfinite(total_area) or float(total_area.item()) <= 0.0:
        raise RuntimeError("mesh has invalid total surface area")
    draws = torch.rand(
        samples,
        device=vertices.device,
        generator=generator,
        dtype=area_cdf.dtype,
    ) * total_area
    face_rows = torch.searchsorted(area_cdf, draws, right=False).clamp_max(
        triangles.shape[0] - 1
    )
    selected = triangles.index_select(0, face_rows)
    u = torch.rand(
        (samples, 1), device=vertices.device, generator=generator
    )
    v = torch.rand(
        (samples, 1), device=vertices.device, generator=generator
    )
    sqrt_u = torch.sqrt(u)
    bary0 = 1.0 - sqrt_u
    bary1 = sqrt_u * (1.0 - v)
    bary2 = sqrt_u * v
    points = (
        bary0 * selected[:, 0]
        + bary1 * selected[:, 1]
        + bary2 * selected[:, 2]
    )
    normals = F.normalize(cross.index_select(0, face_rows), dim=1)
    return points, normals


def _nearest_distances(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if chunk_size <= 0:
        raise ValueError("nearest-neighbor chunk size must be positive")
    distances: List[torch.Tensor] = []
    indices: List[torch.Tensor] = []
    for start in range(0, int(source.shape[0]), int(chunk_size)):
        matrix = torch.cdist(
            source[start : start + chunk_size].to(torch.float32),
            target.to(torch.float32),
        )
        values, rows = matrix.min(dim=1)
        distances.append(values)
        indices.append(rows)
        del matrix
    return torch.cat(distances), torch.cat(indices)


def _mesh_surface_similarity(
    reference: MeshWithVoxel,
    prediction: MeshWithVoxel,
    *,
    samples: int,
    chunk_size: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    ref_points, ref_normals = _sample_mesh_surface(
        reference.vertices.to(device),
        reference.faces.to(device),
        samples=samples,
        seed=seed,
    )
    pred_points, pred_normals = _sample_mesh_surface(
        prediction.vertices.to(device),
        prediction.faces.to(device),
        samples=samples,
        seed=seed + 1,
    )
    ref_to_pred, ref_nn = _nearest_distances(
        ref_points, pred_points, chunk_size=chunk_size
    )
    pred_to_ref, pred_nn = _nearest_distances(
        pred_points, ref_points, chunk_size=chunk_size
    )
    ref_normal_cos = (
        ref_normals
        * pred_normals.index_select(0, ref_nn)
    ).sum(dim=1)
    pred_normal_cos = (
        pred_normals
        * ref_normals.index_select(0, pred_nn)
    ).sum(dim=1)
    voxel_size = float(torch.as_tensor(reference.voxel_size).item())
    metrics: Dict[str, Any] = {
        "surface_samples_per_mesh": int(samples),
        "chamfer_l1_object": float(
            0.5 * (ref_to_pred.mean() + pred_to_ref.mean()).item()
        ),
        "chamfer_l2_object": float(
            0.5
            * (
                ref_to_pred.square().mean()
                + pred_to_ref.square().mean()
            ).item()
        ),
        "reference_to_reconstruction_mean_object": float(
            ref_to_pred.mean().item()
        ),
        "reconstruction_to_reference_mean_object": float(
            pred_to_ref.mean().item()
        ),
        "symmetric_p95_object": float(
            torch.cat((ref_to_pred, pred_to_ref)).quantile(0.95).item()
        ),
        "symmetric_hausdorff_object": float(
            torch.maximum(ref_to_pred.max(), pred_to_ref.max()).item()
        ),
        "normal_cosine_oriented_mean": float(
            0.5
            * (ref_normal_cos.mean() + pred_normal_cos.mean()).item()
        ),
        "normal_cosine_absolute_mean": float(
            0.5
            * (
                ref_normal_cos.abs().mean()
                + pred_normal_cos.abs().mean()
            ).item()
        ),
    }
    metrics["chamfer_l1_in_global_voxels"] = (
        metrics["chamfer_l1_object"] / voxel_size
    )
    metrics["symmetric_p95_in_global_voxels"] = (
        metrics["symmetric_p95_object"] / voxel_size
    )
    for cells in (1, 2, 4, 8):
        threshold = cells * voxel_size
        recall = float((ref_to_pred <= threshold).float().mean().item())
        precision = float((pred_to_ref <= threshold).float().mean().item())
        fscore = (
            0.0
            if precision + recall == 0.0
            else 2.0 * precision * recall / (precision + recall)
        )
        metrics[f"precision_at_{cells}_global_voxels"] = precision
        metrics[f"recall_at_{cells}_global_voxels"] = recall
        metrics[f"fscore_at_{cells}_global_voxels"] = fscore
    del ref_points, pred_points, ref_normals, pred_normals
    _empty_cuda_cache()
    return metrics


def _ovoxel_support_similarity(
    reference_coords: torch.Tensor,
    prediction_coords: torch.Tensor,
) -> Dict[str, Any]:
    ref_keys = torch.unique(
        _linear_keys(reference_coords.to(torch.int64), OVOXEL_RESOLUTION)
    )
    pred_keys = torch.unique(
        _linear_keys(prediction_coords.to(torch.int64), OVOXEL_RESOLUTION)
    )
    ref_sorted = ref_keys.sort().values
    pred_sorted = pred_keys.sort().values
    positions = torch.searchsorted(pred_sorted, ref_sorted)
    valid = positions < pred_sorted.shape[0]
    safe = positions.clamp_max(max(0, pred_sorted.shape[0] - 1))
    intersection = int(
        (valid & (pred_sorted.index_select(0, safe) == ref_sorted)).sum().item()
    )
    ref_count = int(ref_sorted.shape[0])
    pred_count = int(pred_sorted.shape[0])
    union = ref_count + pred_count - intersection
    precision = 0.0 if pred_count == 0 else intersection / pred_count
    recall = 0.0 if ref_count == 0 else intersection / ref_count
    return {
        "reference_global_ovoxels": ref_count,
        "reconstructed_global_ovoxels": pred_count,
        "global_ovoxel_intersection": intersection,
        "global_ovoxel_union": union,
        "global_ovoxel_iou": 0.0 if union == 0 else intersection / union,
        "global_ovoxel_precision": precision,
        "global_ovoxel_recall": recall,
        "global_ovoxel_fscore": (
            0.0
            if precision + recall == 0.0
            else 2.0 * precision * recall / (precision + recall)
        ),
    }


def _save_projection_overlay(
    image_4096: Image.Image,
    uv: torch.Tensor,
    output_path: Path,
    title: str,
) -> None:
    canvas = image_4096.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    points = uv.detach().to(device="cpu", dtype=torch.float32).numpy()
    for x, y in points:
        if np.isfinite(x) and np.isfinite(y):
            draw.ellipse(
                (float(x) - 2, float(y) - 2, float(x) + 2, float(y) + 2),
                fill=(0, 255, 255),
            )
    draw.rectangle((0, 0, canvas.width, 34), fill=(0, 0, 0))
    draw.text((8, 9), title, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _depth_gradient_rgb(
    depth: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """Map positive camera depth to a near-warm / far-cool color gradient."""
    finite = np.isfinite(depth)
    if not finite.any():
        raise RuntimeError("depth visualization has no finite values")
    finite_depth = depth[finite].astype(np.float32, copy=False)
    depth_near = float(np.quantile(finite_depth, 0.01))
    depth_far = float(np.quantile(finite_depth, 0.99))
    if not math.isfinite(depth_near) or not math.isfinite(depth_far):
        raise RuntimeError("depth visualization range is non-finite")
    if depth_far <= depth_near:
        depth_far = depth_near + 1e-6
    normalized = np.clip(
        (depth.astype(np.float32, copy=False) - depth_near)
        / (depth_far - depth_near),
        0.0,
        1.0,
    )

    # near -> far: yellow/red -> green -> cyan -> blue/purple
    stops = np.asarray(
        [
            [255, 240, 40],
            [255, 70, 30],
            [60, 210, 80],
            [35, 210, 230],
            [50, 90, 255],
            [155, 60, 220],
        ],
        dtype=np.float32,
    )
    scaled = normalized * float(stops.shape[0] - 1)
    left = np.floor(scaled).astype(np.int64)
    right = np.minimum(left + 1, stops.shape[0] - 1)
    alpha = (scaled - left.astype(np.float32))[:, None]
    colors = stops[left] * (1.0 - alpha) + stops[right] * alpha
    return np.clip(colors, 0, 255).astype(np.uint8), depth_near, depth_far


def _rasterize_points(
    canvas: np.ndarray,
    *,
    uv: np.ndarray,
    colors: np.ndarray,
    depth: np.ndarray,
) -> Tuple[int, int]:
    """Rasterize every record as one pixel; nearer rows win pixel collisions."""
    height, width = canvas.shape[:2]
    x = np.rint(uv[:, 0]).astype(np.int64)
    y = np.rint(uv[:, 1]).astype(np.int64)
    valid = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(depth)
        & (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )
    x = x[valid]
    y = y[valid]
    colors = colors[valid]
    valid_depth = depth[valid]
    # Paint far-to-near so the physically nearest O-Voxel remains visible
    # when several projections round to the same output pixel.
    order = np.argsort(valid_depth)[::-1]
    canvas[y[order], x[order]] = colors[order]
    occupied = np.unique(y * width + x).shape[0]
    return int(valid.sum()), int(occupied)


def _draw_crosses(
    canvas: np.ndarray,
    *,
    uv: np.ndarray,
    valid: np.ndarray,
) -> int:
    """Draw small outlined × marks for projected C64 latent activations."""
    height, width = canvas.shape[:2]
    x = np.rint(uv[:, 0]).astype(np.int64)
    y = np.rint(uv[:, 1]).astype(np.int64)
    valid = (
        valid.astype(bool, copy=False)
        & np.isfinite(uv).all(axis=1)
        & (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )
    x = x[valid]
    y = y[valid]

    def paint(offsets: Sequence[Tuple[int, int]], color: Sequence[int]) -> None:
        for dx, dy in offsets:
            xx = x + int(dx)
            yy = y + int(dy)
            inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            canvas[yy[inside], xx[inside]] = np.asarray(color, dtype=np.uint8)

    # A black 5x5 diagonal outline keeps the marker readable over every
    # O-Voxel depth color; the inner 3x3 × is white.
    paint(
        (
            (-2, -2),
            (-1, -1),
            (0, 0),
            (1, 1),
            (2, 2),
            (-2, 2),
            (-1, 1),
            (1, -1),
            (2, -2),
        ),
        (0, 0, 0),
    )
    paint(
        ((-1, -1), (0, 0), (1, 1), (-1, 1), (1, -1)),
        (255, 255, 255),
    )
    return int(valid.sum())


def _project_local_latent_to_tile(
    coords: torch.Tensor,
    *,
    transform: TileCameraTransform,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("encoded latent coordinates must have shape [N,4]")
    if bool((coords[:, 0] != 0).any().item()):
        raise ValueError("only batch-zero encoded latent visualization is supported")
    local_object = (
        coords[:, 1:4].to(torch.float32) + 0.5
    ) / float(LATENT_RESOLUTION) - 0.5
    local_q = local_object * (2.0 * float(transform.mesh_scale))
    local_points = _camera_q_to_points(
        local_q,
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
    )
    return _project_points(
        local_points,
        fx=float(transform.fx),
        fy=float(transform.fy),
        cx=float(transform.cx),
        cy=float(transform.cy),
    )


def _save_selected_ovoxel_visualizations(
    *,
    tile_image: Image.Image,
    ovoxel_uv: torch.Tensor,
    ovoxel_depth: torch.Tensor,
    output_dir: Path,
    tile_id: int,
    latent_coords: Optional[torch.Tensor] = None,
    transform: Optional[TileCameraTransform] = None,
) -> Dict[str, Any]:
    """Save depth-colored O-Voxel points and an optional encoded-latent overlay."""
    uv_np = ovoxel_uv.detach().to(device="cpu", dtype=torch.float32).numpy()
    depth_np = (
        ovoxel_depth.detach().to(device="cpu", dtype=torch.float32).numpy()
    )
    colors, depth_near, depth_far = _depth_gradient_rgb(depth_np)
    base = np.asarray(tile_image.convert("RGB"), dtype=np.uint8).copy()
    rasterized_rows, occupied_pixels = _rasterize_points(
        base,
        uv=uv_np,
        colors=colors,
        depth=depth_np,
    )
    selected_path = output_dir / "selected_global_ovoxels.png"
    selected_projection = Image.fromarray(base)
    selected_image = Image.new(
        "RGB",
        (selected_projection.width, selected_projection.height + 50),
        (0, 0, 0),
    )
    selected_image.paste(selected_projection, (0, 50))
    selected_draw = ImageDraw.Draw(selected_image)
    selected_draw.text(
        (8, 7),
        (
            f"tile {tile_id:02d}: global O-Voxel projection "
            "(one record -> one pixel; near warm, far cool)"
        ),
        fill=(255, 255, 255),
    )
    selected_draw.text(
        (8, 27),
        f"near D={depth_near:.5f}    far D={depth_far:.5f}",
        fill=(220, 220, 220),
    )
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_image.save(selected_path)

    stats: Dict[str, Any] = {
        "selected_global_ovoxels_png": str(selected_path),
        "ovoxel_records": int(ovoxel_uv.shape[0]),
        "ovoxel_records_rasterized": rasterized_rows,
        "ovoxel_occupied_pixels": occupied_pixels,
        "ovoxel_projection_pixel_collisions": int(
            rasterized_rows - occupied_pixels
        ),
        "depth_convention": "positive global camera depth D=-Pz",
        "depth_near_p01": depth_near,
        "depth_far_p99": depth_far,
        "depth_color": "near warm (yellow/red), far cool (blue/purple)",
    }
    if latent_coords is None:
        return stats
    if transform is None:
        raise ValueError("transform is required when latent_coords are provided")

    latent_uv, _, latent_valid = _project_local_latent_to_tile(
        latent_coords,
        transform=transform,
    )
    overlay = base.copy()
    latent_drawn = _draw_crosses(
        overlay,
        uv=latent_uv.detach().to(device="cpu", dtype=torch.float32).numpy(),
        valid=latent_valid.detach().to(device="cpu").numpy(),
    )
    overlay_path = output_dir / "selected_global_ovoxels_with_encoded_latent.png"
    overlay_projection = Image.fromarray(overlay)
    overlay_image = Image.new(
        "RGB",
        (overlay_projection.width, overlay_projection.height + 50),
        (0, 0, 0),
    )
    overlay_image.paste(overlay_projection, (0, 50))
    overlay_draw = ImageDraw.Draw(overlay_image)
    overlay_draw.text(
        (8, 7),
        (
            f"tile {tile_id:02d}: depth-colored O-Voxels + "
            "common encoded C64 latent"
        ),
        fill=(255, 255, 255),
    )
    overlay_draw.text(
        (8, 27),
        "colored pixel = global O-Voxel    white outlined × = active C64 latent",
        fill=(220, 220, 220),
    )
    overlay_image.save(overlay_path)
    stats.update(
        {
            "selected_global_ovoxels_with_encoded_latent_png": str(
                overlay_path
            ),
            "encoded_common_c64_latent_points": int(latent_coords.shape[0]),
            "encoded_common_c64_latent_points_projected": latent_drawn,
            "latent_marker": "small white × with black outline",
            "latent_projection": (
                "C64 cell center -> local object -> local q -> centered "
                "tile-camera projection"
            ),
        }
    )
    return stats


def _tile_crop_render_metrics(
    *,
    reference_render: Path,
    prediction_render: Path,
    box: Sequence[int],
    output_dir: Path,
    metric_resolution: int,
) -> Dict[str, Any]:
    with Image.open(reference_render) as image:
        reference = image.convert("RGB")
    with Image.open(prediction_render) as image:
        prediction = image.convert("RGB")
    if reference.size != prediction.size:
        prediction = prediction.resize(reference.size, Image.Resampling.LANCZOS)
    width, height = reference.size
    x0, y0, x1, y1 = (float(v) for v in box)
    crop = (
        int(round(x0 * width / CANONICAL_IMAGE_SIZE)),
        int(round(y0 * height / CANONICAL_IMAGE_SIZE)),
        int(round(x1 * width / CANONICAL_IMAGE_SIZE)),
        int(round(y1 * height / CANONICAL_IMAGE_SIZE)),
    )
    reference_crop = reference.crop(crop)
    prediction_crop = prediction.crop(crop)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = output_dir / "reference_tile_render.png"
    prediction_path = output_dir / "reconstructed_tile_render.png"
    comparison_path = output_dir / "comparison.png"
    reference_crop.save(reference_path)
    prediction_crop.save(prediction_path)
    comparison = Image.new(
        "RGB",
        (reference_crop.width * 2, reference_crop.height),
        (0, 0, 0),
    )
    comparison.paste(reference_crop, (0, 0))
    comparison.paste(prediction_crop, (reference_crop.width, 0))
    comparison.save(comparison_path)

    size = (int(metric_resolution), int(metric_resolution))
    reference_tensor = image_to_tensor(reference_crop, size)
    prediction_tensor = image_to_tensor(prediction_crop, size)
    return {
        "tile_render_psnr_db": psnr_metric(reference_tensor, prediction_tensor),
        "tile_render_ssim": ssim_metric(reference_tensor, prediction_tensor),
        "reference_tile_render_png": str(reference_path),
        "reconstructed_tile_render_png": str(prediction_path),
        "tile_render_comparison_png": str(comparison_path),
        "render_crop_pixels": list(crop),
    }


def _estimate_camera(
    *,
    image_1024: Image.Image,
    output_dir: Path,
    manual_fov: float,
    mesh_scale: float,
    extend_pixel: int,
    moge_model_path: Optional[str],
) -> Dict[str, float]:
    if manual_fov > 0.0:
        distance = distance_from_fov(
            float(manual_fov),
            torch.tensor([-1.0, 0.0, 0.0]),
            torch.tensor(
                [
                    0.0 - float(extend_pixel),
                    GLOBAL_IMAGE_SIZE - 1.0 + float(extend_pixel),
                ]
            ),
            float(mesh_scale),
            GLOBAL_IMAGE_SIZE,
        )["distance_from_x"]
        return {
            "camera_angle_x": float(manual_fov),
            "distance": float(distance),
            "mesh_scale": float(mesh_scale),
            "source": "manual_fov",
        }

    temporary = output_dir / f"_moge_input_{time.time_ns()}.png"
    image_1024.save(temporary)
    try:
        if moge_model_path:
            model = load_moge_model(device="cuda", model_name=moge_model_path)
        else:
            model = load_moge_model(device="cuda")
        params = get_camera_params_wild_moge(
            str(temporary),
            model,
            device="cuda",
            mesh_scale=float(mesh_scale),
            extend_pixel=int(extend_pixel),
            image_resolution=GLOBAL_IMAGE_SIZE,
        )
        model.cpu()
        del model
        _empty_cuda_cache()
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "camera_angle_x": float(params["camera_angle_x"]),
        "distance": float(params["distance"]),
        "mesh_scale": float(params["mesh_scale"]),
        "source": "moge2_global_canonical_1024",
    }


def _sampler_overrides(args: argparse.Namespace) -> Tuple[Dict[str, Any], ...]:
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


def _render(
    mesh: MeshWithVoxel,
    *,
    output_dir: Path,
    camera: Mapping[str, float],
    reference_image: Optional[Path],
    args: argparse.Namespace,
    envmap: Any,
) -> Dict[str, Any]:
    device = torch.device("cuda")
    live_mesh = mesh.to(device)
    result = render_and_evaluate_mesh(
        live_mesh,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        output_dir=output_dir,
        reference_image=reference_image,
        resolution=int(args.render_resolution),
        metric_resolution=int(args.metric_resolution),
        envmap=envmap,
        envmap_name=str(args.envmap),
        ssaa=int(args.render_ssaa),
        peel_layers=int(args.render_peel_layers),
        face_chunk_size=int(args.render_face_chunk_size),
        use_envmap_bg=bool(args.use_envmap_bg),
        lpips_net=str(args.lpips_net),
        metric_device="cuda",
        skip_lpips=bool(args.skip_lpips),
    )
    del live_mesh
    _empty_cuda_cache()
    return result


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(int(args.cuda_device))
    device = torch.device("cuda")
    print(
        f"[cuda] requested physical/current index={int(args.cuda_device)} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())}"
    )

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
    canonical["foreground_mask_4096"].save(
        output_dir / "canonical_foreground_mask_4096.png"
    )
    _atomic_json(output_dir / "canonical_metadata.json", canonical["metadata"])

    global_camera = _estimate_camera(
        image_1024=image_1024,
        output_dir=output_dir,
        manual_fov=float(args.fov),
        mesh_scale=float(args.mesh_scale),
        extend_pixel=int(args.extend_pixel),
        moge_model_path=args.moge_model_path,
    )
    _atomic_json(output_dir / "global_camera.json", global_camera)
    print(
        "[global-camera] "
        f"fov={global_camera['camera_angle_x']:.8f} "
        f"distance={global_camera['distance']:.8f} "
        f"mesh_scale={global_camera['mesh_scale']:.8f}"
    )

    ss_params, shape_params, texture_params = _sampler_overrides(args)
    print("[global-baseline] running ordinary Pixal3D 1024_cascade")
    _seed_everything(int(args.seed))
    baseline_started = time.perf_counter()
    baseline_output, baseline_latents = pipeline.run(
        image_1024,
        camera_params=global_camera,
        seed=int(args.seed),
        sparse_structure_sampler_params=ss_params,
        shape_slat_sampler_params=shape_params,
        tex_slat_sampler_params=texture_params,
        preprocess_image=False,
        return_latent=True,
        pipeline_type="1024_cascade",
        max_num_tokens=int(args.max_num_tokens),
    )
    baseline_seconds = time.perf_counter() - baseline_started
    if len(baseline_output) != 1:
        raise RuntimeError(
            f"global baseline returned {len(baseline_output)} meshes, expected one"
        )
    baseline_live = _validate_mesh(
        baseline_output[0], "global ordinary Pixal3D-1024 baseline"
    )
    shape_latent, texture_latent, decoded_resolution = baseline_latents
    if int(decoded_resolution) != OVOXEL_RESOLUTION:
        raise RuntimeError(
            f"baseline decoder resolution is {decoded_resolution}, expected 1024"
        )
    baseline_dir = output_dir / "global_baseline_1024"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    envmap = (
        load_envmap(str(args.envmap), device="cuda") if args.render else None
    )
    baseline_render: Optional[Dict[str, Any]] = None
    if args.render:
        baseline_render = _render(
            baseline_live,
            output_dir=baseline_dir / "aligned_eval",
            camera=global_camera,
            reference_image=output_dir / "canonical_1024.png",
            args=args,
            envmap=envmap,
        )

    baseline_mesh = baseline_live.to("cpu")
    baseline_summary: Dict[str, Any] = {
        "route": "ordinary pipeline.run(..., pipeline_type='1024_cascade')",
        "generation_seconds": float(baseline_seconds),
        "decoder_resolution": int(decoded_resolution),
        "vertices": int(baseline_mesh.vertices.shape[0]),
        "faces": int(baseline_mesh.faces.shape[0]),
        "active_ovoxels": int(baseline_mesh.coords.shape[0]),
        "shape_latent_tokens": int(shape_latent.feats.shape[0]),
        "texture_latent_tokens": int(texture_latent.feats.shape[0]),
        "render": baseline_render,
    }
    _atomic_json(baseline_dir / "summary.json", baseline_summary)
    if args.save_mesh_checkpoints:
        torch.save(baseline_mesh, baseline_dir / "mesh_with_ovoxel.pt")
    del baseline_output, baseline_live, baseline_latents
    del shape_latent, texture_latent
    _empty_cuda_cache()

    print("[global-analysis] projecting decoded O-Voxel cell centers and faces")
    global_ovoxel_object = _ovoxel_indices_to_object(
        baseline_mesh.coords,
        origin=baseline_mesh.origin,
        voxel_size=baseline_mesh.voxel_size,
    )
    global_ovoxel_q = global_ovoxel_object * (
        2.0 * float(global_camera["mesh_scale"])
    )
    global_ovoxel_uv, _, global_ovoxel_finite = (
        _project_global_q_to_4096(
            global_ovoxel_q, global_camera=global_camera
        )
    )
    _save_projection_overlay(
        image_4096,
        global_ovoxel_uv[global_ovoxel_finite],
        output_dir / "global_baseline_1024" / "ovoxel_projection_4096.png",
        "ordinary global-1024 decoder O-Voxel cell centers",
    )
    global_face_uv, global_face_finite = _project_face_centers(
        baseline_mesh.vertices,
        baseline_mesh.faces,
        mesh_scale=float(global_camera["mesh_scale"]),
        global_camera=global_camera,
        chunk_size=int(args.face_projection_chunk_size),
    )

    print(f"[encoder] loading shape encoder: {args.shape_encoder}")
    shape_encoder = pixal3d_models.from_pretrained(
        str(Path(args.shape_encoder).expanduser())
    ).eval()
    print(f"[encoder] loading PBR encoder: {args.pbr_encoder}")
    pbr_encoder = pixal3d_models.from_pretrained(
        str(Path(args.pbr_encoder).expanduser())
    ).eval()
    if not args.low_vram:
        shape_encoder.to(device)
        pbr_encoder.to(device)

    boxes = _tile_layout()
    requested_ids = _parse_tile_ids(args.tile_ids)
    if requested_ids is not None:
        invalid = sorted(tile_id for tile_id in requested_ids if tile_id not in range(49))
        if invalid:
            raise ValueError(f"invalid tile ids {invalid}; valid ids are 0..48")

    tile_records: List[Dict[str, Any]] = []
    reconstructed_tiles = 0
    for tile_id, box in enumerate(boxes):
        if requested_ids is not None and tile_id not in requested_ids:
            continue
        transform = _derive_tile_camera(
            tile_id=tile_id,
            box=box,
            global_camera=global_camera,
            extend_pixel=int(args.extend_pixel),
        )
        projected_count = int(
            _inside_tile(
                global_ovoxel_uv, global_ovoxel_finite, box
            ).sum().item()
        )
        tile_dir = output_dir / "tiles" / f"tile_{tile_id:02d}"
        tile_dir.mkdir(parents=True, exist_ok=True)
        image_4096.crop(box).save(tile_dir / "tile_reference.png")
        _atomic_json(tile_dir / "tile_camera.json", asdict(transform))
        print(
            f"[tile {tile_id:02d}] projected_global_ovoxels={projected_count:,} "
            f"box={box}"
        )
        if projected_count < int(args.min_tile_ovoxels):
            record = {
                "status": "skipped",
                "tile_id": tile_id,
                "box": list(box),
                "projected_global_ovoxels": projected_count,
                "reason": (
                    f"requires at least {int(args.min_tile_ovoxels)} projected "
                    "global O-Voxels"
                ),
            }
            tile_records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            continue
        if args.max_tiles is not None and reconstructed_tiles >= int(args.max_tiles):
            break
        reconstructed_tiles += 1

        tile_started = time.perf_counter()
        try:
            mapping = _map_global_ovoxels_to_local(
                global_mesh=baseline_mesh,
                global_q=global_ovoxel_q,
                global_uv_4096=global_ovoxel_uv,
                finite_projection=global_ovoxel_finite,
                global_camera=global_camera,
                transform=transform,
            )
            if mapping.local_coords.shape[0] == 0:
                raise RuntimeError("global-to-local mapping produced no O-Voxel")
            if (
                mapping.stats["global_local_global_q_max_abs_error"]
                > float(args.roundtrip_tolerance)
            ):
                raise RuntimeError(
                    "global/local camera round-trip exceeded tolerance: "
                    f"{mapping.stats['global_local_global_q_max_abs_error']:.3e}"
                )
            selected_uv = global_ovoxel_uv.index_select(
                0, mapping.source_global_rows
            )
            selected_tile_uv = torch.stack(
                (
                    selected_uv[:, 0] - float(box[0]),
                    selected_uv[:, 1] - float(box[1]),
                ),
                dim=1,
            )
            selected_global_q = global_ovoxel_q.index_select(
                0, mapping.source_global_rows
            )
            selected_global_depth = float(global_camera["distance"]) - (
                selected_global_q[:, 2]
                / (2.0 * float(global_camera["mesh_scale"]))
            )

            (
                local_geometry_vertices,
                local_geometry_faces,
                reference_vertices,
                reference_faces,
                geometry_stats,
            ) = _prepare_tile_geometry(
                global_vertices=baseline_mesh.vertices,
                global_faces=baseline_mesh.faces,
                global_face_uv=global_face_uv,
                global_face_finite=global_face_finite,
                global_camera=global_camera,
                transform=transform,
            )
            reference_mesh = _make_reference_tile_mesh(
                vertices=reference_vertices,
                faces=reference_faces,
                mapping=mapping,
                global_mesh=baseline_mesh,
            )

            shape_z, shape_encoder_stats = _encode_local_shape(
                encoder=shape_encoder,
                vertices=local_geometry_vertices,
                faces=local_geometry_faces,
                device=device,
                low_vram=bool(args.low_vram),
            )
            pbr_z, pbr_encoder_stats = _encode_local_pbr(
                encoder=pbr_encoder,
                coords=mapping.local_coords,
                attrs=mapping.local_attrs,
                device=device,
                low_vram=bool(args.low_vram),
            )
            shape_z, pbr_z, alignment_stats = _align_latent_supports(
                shape_z, pbr_z
            )
            visualization_stats = _save_selected_ovoxel_visualizations(
                tile_image=image_4096.crop(box),
                ovoxel_uv=selected_tile_uv,
                ovoxel_depth=selected_global_depth,
                output_dir=tile_dir,
                tile_id=tile_id,
                latent_coords=shape_z.coords,
                transform=transform,
            )
            if shape_z.feats.shape[0] > int(args.max_num_tokens):
                raise RuntimeError(
                    f"common local latent has {shape_z.feats.shape[0]:,} tokens, "
                    f"exceeding --max-num-tokens={int(args.max_num_tokens):,}"
                )

            decode_started = time.perf_counter()
            with torch.no_grad():
                decoded = pipeline.decode_latent(
                    shape_z,
                    pbr_z,
                    OVOXEL_RESOLUTION,
                )
            _sync_cuda()
            decode_seconds = time.perf_counter() - decode_started
            if len(decoded) != 1:
                raise RuntimeError(
                    f"local latent decoder returned {len(decoded)} meshes"
                )
            local_decoded = _validate_mesh(
                decoded[0], f"tile {tile_id:02d} local encoded reconstruction"
            )
            returned_mesh, return_stats = _return_decoded_mesh_to_global(
                local_mesh=local_decoded,
                global_template=baseline_mesh,
                global_camera=global_camera,
                transform=transform,
                face_chunk_size=int(args.face_projection_chunk_size),
            )

            surface_metrics = _mesh_surface_similarity(
                reference_mesh,
                returned_mesh,
                samples=int(args.surface_samples),
                chunk_size=int(args.nearest_chunk_size),
                seed=int(args.seed) + tile_id * 100,
                device=device,
            )
            ov_stats = _ovoxel_support_similarity(
                reference_mesh.coords, returned_mesh.coords
            )

            render_stats: Optional[Dict[str, Any]] = None
            if args.render:
                reference_render = _render(
                    reference_mesh,
                    output_dir=tile_dir / "reference_global_part_render",
                    camera=global_camera,
                    reference_image=None,
                    args=args,
                    envmap=envmap,
                )
                reconstruction_render = _render(
                    returned_mesh,
                    output_dir=tile_dir / "reconstructed_returned_global_render",
                    camera=global_camera,
                    reference_image=None,
                    args=args,
                    envmap=envmap,
                )
                crop_metrics = _tile_crop_render_metrics(
                    reference_render=Path(reference_render["render_png"]),
                    prediction_render=Path(reconstruction_render["render_png"]),
                    box=box,
                    output_dir=tile_dir / "tile_render_comparison",
                    metric_resolution=int(args.metric_resolution),
                )
                render_stats = {
                    "reference": reference_render,
                    "reconstruction": reconstruction_render,
                    **crop_metrics,
                }

            if args.save_mesh_checkpoints:
                torch.save(
                    {
                        "local_coords": mapping.local_coords,
                        "local_attrs": mapping.local_attrs,
                        "source_global_rows": mapping.source_global_rows,
                        "shape_latent": shape_z.to("cpu"),
                        "pbr_latent": pbr_z.to("cpu"),
                    },
                    tile_dir / "local_ovoxel_and_latents.pt",
                )
                torch.save(reference_mesh, tile_dir / "reference_global_part_mesh.pt")
                torch.save(returned_mesh, tile_dir / "returned_global_mesh.pt")

            record = {
                "status": "success",
                "tile_id": tile_id,
                "box": list(box),
                "tile_seconds": float(time.perf_counter() - tile_started),
                "decode_seconds": float(decode_seconds),
                "mapping": mapping.stats,
                "geometry_input": geometry_stats,
                "shape_encoder": shape_encoder_stats,
                "pbr_encoder": pbr_encoder_stats,
                "latent_alignment": alignment_stats,
                "visualizations": visualization_stats,
                "returned_mesh": return_stats,
                "surface_similarity": surface_metrics,
                "ovoxel_support_similarity": ov_stats,
                "render_similarity": render_stats,
            }
            tile_records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            print(
                f"[tile {tile_id:02d}] success "
                f"common_C64={alignment_stats['common_latent_tokens']:,} "
                f"chamfer={surface_metrics['chamfer_l1_in_global_voxels']:.4f}vox "
                f"F@2={surface_metrics['fscore_at_2_global_voxels']:.4f} "
                f"O-IoU={ov_stats['global_ovoxel_iou']:.4f}"
            )
            del (
                mapping,
                reference_mesh,
                returned_mesh,
                local_decoded,
                decoded,
                shape_z,
                pbr_z,
            )
            _empty_cuda_cache()
        except Exception as exc:
            record = {
                "status": "failed",
                "tile_id": tile_id,
                "box": list(box),
                "projected_global_ovoxels": projected_count,
                "tile_seconds": float(time.perf_counter() - tile_started),
                "reason": f"{type(exc).__name__}: {exc}",
            }
            tile_records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            print(f"[tile {tile_id:02d}] FAILED: {record['reason']}")
            _empty_cuda_cache()

    del shape_encoder, pbr_encoder
    _empty_cuda_cache()
    success_rows = [row for row in tile_records if row["status"] == "success"]
    failed_rows = [row for row in tile_records if row["status"] == "failed"]
    skipped_rows = [row for row in tile_records if row["status"] == "skipped"]
    aggregate: Dict[str, Any] = {}
    if success_rows:
        aggregate = {
            "mean_chamfer_l1_in_global_voxels": float(
                np.mean(
                    [
                        row["surface_similarity"][
                            "chamfer_l1_in_global_voxels"
                        ]
                        for row in success_rows
                    ]
                )
            ),
            "mean_fscore_at_2_global_voxels": float(
                np.mean(
                    [
                        row["surface_similarity"][
                            "fscore_at_2_global_voxels"
                        ]
                        for row in success_rows
                    ]
                )
            ),
            "mean_global_ovoxel_iou": float(
                np.mean(
                    [
                        row["ovoxel_support_similarity"]["global_ovoxel_iou"]
                        for row in success_rows
                    ]
                )
            ),
            "mean_tile_render_psnr_db": (
                None
                if not args.render
                else float(
                    np.mean(
                        [
                            row["render_similarity"]["tile_render_psnr_db"]
                            for row in success_rows
                        ]
                    )
                )
            ),
            "mean_tile_render_ssim": (
                None
                if not args.render
                else float(
                    np.mean(
                        [
                            row["render_similarity"]["tile_render_ssim"]
                            for row in success_rows
                        ]
                    )
                )
            ),
        }

    summary = {
        "format": "pixal3d_global_1024_ovoxel_tile_encode_decode_v1",
        "image": str(Path(args.image).expanduser().resolve()),
        "cuda_device": int(args.cuda_device),
        "global_camera": global_camera,
        "global_baseline_1024": baseline_summary,
        "tile_policy": {
            "canonical_image_size": CANONICAL_IMAGE_SIZE,
            "tile_size": TILE_SIZE,
            "tile_stride": TILE_STRIDE,
            "minimum_projected_global_ovoxels": int(args.min_tile_ovoxels),
            "strictly_more_than_1000_by_default": int(args.min_tile_ovoxels)
            == 1001,
            "global_index_to_absolute": (
                "global_object = origin + (global_index + 0.5) * voxel_size"
            ),
            "global_to_local": (
                "global_object -> global_q -> global camera projection -> "
                "tile pixels -> local camera backprojection -> local_q"
            ),
            "local_quantization": (
                "nearest local O-Voxel cell center; out-of-cube rows dropped; "
                "no clamp"
            ),
            "encoder_inputs": (
                "shape encoder: same tile-owned global mesh transformed local; "
                "PBR encoder: transformed local O-Voxels"
            ),
            "decoder_input": "exact intersection of encoded shape/PBR C64 supports",
            "return_to_global": (
                "decoder local object -> local_q -> exact inverse camera mapping "
                "-> global_q -> global object"
            ),
        },
        "successful_tiles": len(success_rows),
        "failed_tiles": len(failed_rows),
        "skipped_tiles": len(skipped_rows),
        "aggregate_similarity": aggregate,
        "tiles": tile_records,
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(
        f"[done] success={len(success_rows)} failed={len(failed_rows)} "
        f"skipped={len(skipped_rows)} summary={output_dir / 'summary.json'}"
    )


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
        default=4,
        help="physical CUDA index; defaults to GPU 4 as requested",
    )
    parser.add_argument(
        "--low-vram",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="move the large flow/encoder/decoder models on demand",
    )
    parser.add_argument(
        "--shape-encoder",
        default=str(DEFAULT_ENCODER_ROOT / "shape_enc_next_dc_f16c32_fp16"),
    )
    parser.add_argument(
        "--pbr-encoder",
        default=str(DEFAULT_ENCODER_ROOT / "tex_enc_next_dc_f16c32_fp16"),
    )

    parser.add_argument("--tile-ids", default=None)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument(
        "--min-tile-ovoxels",
        type=int,
        default=1001,
        help="minimum projected global O-Voxels; 1001 means strictly >1000",
    )
    parser.add_argument("--max-num-tokens", type=int, default=1_000_000)
    parser.add_argument("--roundtrip-tolerance", type=float, default=2e-5)
    parser.add_argument("--face-projection-chunk-size", type=int, default=250_000)
    parser.add_argument("--surface-samples", type=int, default=10_000)
    parser.add_argument("--nearest-chunk-size", type=int, default=1_024)
    parser.add_argument(
        "--save-mesh-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

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

    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--metric-resolution", type=int, default=512)
    parser.add_argument("--render-ssaa", type=int, default=2)
    parser.add_argument("--render-peel-layers", type=int, default=8)
    parser.add_argument("--render-face-chunk-size", type=int, default=4_000_000)
    parser.add_argument(
        "--use-envmap-bg",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--skip-lpips",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--lpips-net",
        choices=("alex", "vgg", "squeeze"),
        default="vgg",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if int(args.cuda_device) < 0:
        raise ValueError("--cuda-device must be non-negative")
    if int(args.min_tile_ovoxels) < 1001:
        raise ValueError("--min-tile-ovoxels must be >=1001 for this experiment")
    if args.max_tiles is not None and int(args.max_tiles) < 1:
        raise ValueError("--max-tiles must be positive")
    if int(args.max_num_tokens) < 1:
        raise ValueError("--max-num-tokens must be positive")
    if float(args.roundtrip_tolerance) <= 0.0:
        raise ValueError("--roundtrip-tolerance must be positive")
    if (
        int(args.face_projection_chunk_size) < 1
        or int(args.surface_samples) < 1
        or int(args.nearest_chunk_size) < 1
    ):
        raise ValueError("projection/sample/chunk sizes must be positive")
    if (
        int(args.render_resolution) < 1
        or int(args.metric_resolution) < 1
        or int(args.render_ssaa) < 1
        or int(args.render_peel_layers) < 1
        or int(args.render_face_chunk_size) < 0
    ):
        raise ValueError("invalid render configuration")
    for encoder_path in (args.shape_encoder, args.pbr_encoder):
        base = Path(encoder_path).expanduser()
        if not Path(f"{base}.json").is_file() or not Path(
            f"{base}.safetensors"
        ).is_file():
            raise FileNotFoundError(
                f"encoder checkpoint pair not found for base path {base}"
            )
    run(args)


if __name__ == "__main__":
    main()
