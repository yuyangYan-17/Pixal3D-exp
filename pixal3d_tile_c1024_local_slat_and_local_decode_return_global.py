#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pixal3D 1024 local dual-grid tile encode/decode experiment.

The route in this file is deliberately local from the first tile operation
onward.  A normal ``1024_cascade`` run supplies the global baseline mesh and
its continuous PBR field.  Each 1024 crop of the canonical 4096 image then
does the following:

    projected global triangle bbox
      -> exact global/local camera round trip
      -> local mesh_to_flexible_dual_grid (C1024)
      -> local shape/PBR VAE encoders on the same support
      -> native Pixal3D flow sampler with the local image condition
      -> local MeshWithVoxel decode
      -> face-corner PBR query and exact return to global object space

The final global object is intentionally an unrepaired concatenation of all
successful local patches.  It never welds vertices, merges overlap, removes
faces, remeshes, or converts local O-Voxel coordinates back to global voxel
indices.  Final PBR rendering uses the repository's nvdiffrast path through
``MeshWithVertexPbr``; every face corner is a separate vertex, so the renderer
performs barycentric interpolation of independent corner attributes.  The
renderer already performs exact depth-layer merging when face chunking is
enabled.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
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
import trimesh
import utils3d
from PIL import Image, ImageDraw, ImageOps

import pixal3d.models as pixal3d_models
from inference import (
    MODEL_PATH,
    distance_from_fov,
    get_camera_params_wild_moge,
    init_pipeline,
    load_moge_model,
)
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVertexPbr, MeshWithVoxel
from pixal3d.utils import render_utils
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
PBR_LAYOUT = {
    "base_color": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}

# GLB bufferView byte offsets are encoded as uint32 by the glTF/GLB format
# used by trimesh.  Leave headroom for JSON, alignment, and accessor tables;
# a scene above this limit is exported as independent tile-part GLBs instead
# of attempting a monolithic file that cannot be represented by GLB.
GLB_SAFE_BUFFER_BYTES = 3_500_000_000
GLB_SHARD_FACE_LIMIT = 16_000_000


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
class LocalGeometry:
    vertices: torch.Tensor
    faces: torch.Tensor
    coords: torch.Tensor
    dual_vertices: torch.Tensor
    dual_vertices_world: torch.Tensor
    intersected: torch.Tensor
    selected_global_face_ids: torch.Tensor
    stats: Dict[str, Any]


@dataclass
class TileFlowLatents:
    reference_shape: SparseTensor
    reference_texture: SparseTensor
    shape_norm: SparseTensor
    shape_denorm: SparseTensor
    texture_norm: SparseTensor
    texture_denorm: SparseTensor
    stats: Dict[str, Any]


@dataclass
class ReturnedTilePatch:
    tile_id: int
    box: Tuple[int, int, int, int]
    vertices: torch.Tensor
    faces: torch.Tensor
    vertex_attrs: torch.Tensor
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
    value = value.detach().to(torch.float32)
    return {
        "min": [float(v) for v in value.amin(dim=0).cpu().tolist()],
        "max": [float(v) for v in value.amax(dim=0).cpu().tolist()],
    }


def _tile_layout(
    canonical_size: int = CANONICAL_IMAGE_SIZE,
    tile_size: int = TILE_SIZE,
    stride: int = TILE_STRIDE,
) -> List[Tuple[int, int, int, int]]:
    if canonical_size <= 0 or tile_size <= 0 or stride <= 0:
        raise ValueError("canonical size, tile size, and stride must be positive")
    if tile_size > canonical_size:
        raise ValueError("tile size cannot exceed canonical size")
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
            [0.0 - float(extend_pixel), TILE_SIZE - 1.0 + float(extend_pixel)]
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
    uv_tile[:, 0] = (uv_full[:, 0] - float(x0)) * transform.crop_to_output_scale_x
    uv_tile[:, 1] = (uv_full[:, 1] - float(y0)) * transform.crop_to_output_scale_y

    qz = q_global[:, 2]
    local_depth = float(transform.distance) - qz / (2.0 * float(transform.mesh_scale))
    x = (uv_tile[:, 0] - float(transform.cx)) * local_depth / float(transform.fx)
    y = -(uv_tile[:, 1] - float(transform.cy)) * local_depth / float(transform.fy)
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
    uv_full[:, 0] = uv_tile[:, 0] / transform.crop_to_output_scale_x + float(x0)
    uv_full[:, 1] = uv_tile[:, 1] / transform.crop_to_output_scale_y + float(y0)

    qz = q_local[:, 2]
    global_scale = float(global_camera["mesh_scale"])
    global_depth = float(global_camera["distance"]) - qz / (2.0 * global_scale)
    x = (uv_full[:, 0] - transform.full_cx_4096) * global_depth / transform.full_fx_4096
    y = -(uv_full[:, 1] - transform.full_cy_4096) * global_depth / transform.full_fy_4096
    global_points = torch.stack((x, y, -global_depth), dim=1)
    q_global = _camera_points_to_q(
        global_points,
        distance=float(global_camera["distance"]),
        mesh_scale=global_scale,
    )
    return q_global, uv_full


def _linear_keys(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    xyz = coords.to(torch.int64)
    return (xyz[:, 0] * int(resolution) + xyz[:, 1]) * int(resolution) + xyz[:, 2]


def _compact_submesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if faces.numel() == 0:
        raise RuntimeError("cannot compact an empty face set")
    vertex_ids, inverse = torch.unique(
        faces.reshape(-1).to(torch.long), sorted=True, return_inverse=True
    )
    compact_vertices = vertices.index_select(0, vertex_ids)
    compact_faces = inverse.reshape(-1, 3).to(torch.int64)
    return compact_vertices, compact_faces, vertex_ids


def _project_face_bboxes(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    mesh_scale: float,
    global_camera: Mapping[str, float],
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project all triangle corners and return their conservative 2-D bboxes.

    This is intentionally a triangle-bbox test.  No face centroid or
    visibility test is used, so front, back, and occluded triangles entering a
    tile are retained.
    """
    if chunk_size <= 0:
        raise ValueError("face projection chunk size must be positive")
    mins: List[torch.Tensor] = []
    maxs: List[torch.Tensor] = []
    finite_rows: List[torch.Tensor] = []
    faces_long = faces.to(torch.long)
    for start in range(0, int(faces.shape[0]), int(chunk_size)):
        face_chunk = faces_long[start : start + chunk_size]
        q = vertices.index_select(0, face_chunk.reshape(-1)).reshape(-1, 3, 3)
        uv, _, finite = _project_global_q_to_4096(
            q.reshape(-1, 3) * (2.0 * float(mesh_scale)),
            global_camera=global_camera,
        )
        uv = uv.reshape(-1, 3, 2)
        finite = finite.reshape(-1, 3)
        safe_min = torch.where(
            finite[..., None], uv, torch.full_like(uv, float("inf"))
        ).amin(dim=1)
        safe_max = torch.where(
            finite[..., None], uv, torch.full_like(uv, float("-inf"))
        ).amax(dim=1)
        mins.append(safe_min.cpu().to(torch.float32))
        maxs.append(safe_max.cpu().to(torch.float32))
        finite_rows.append(finite.all(dim=1).cpu())
    return torch.cat(mins), torch.cat(maxs), torch.cat(finite_rows)


def _tile_face_ids_from_bbox(
    face_min: torch.Tensor,
    face_max: torch.Tensor,
    face_finite: torch.Tensor,
    box: Sequence[int],
) -> torch.Tensor:
    x0, y0, x1, y1 = (float(v) for v in box)
    intersects = (
        face_finite
        & (face_max[:, 0] >= x0)
        & (face_min[:, 0] < x1)
        & (face_max[:, 1] >= y0)
        & (face_min[:, 1] < y1)
    )
    return torch.where(intersects)[0]


def _camera_roundtrip_stats(
    q_global: torch.Tensor,
    *,
    global_camera: Mapping[str, float],
    transform: TileCameraTransform,
) -> Dict[str, float]:
    q_local, _ = _global_q_to_local_q(
        q_global, global_camera=global_camera, transform=transform
    )
    roundtrip, _ = _local_q_to_global_q(
        q_local, global_camera=global_camera, transform=transform
    )
    error = (roundtrip - q_global).abs()
    return {
        "global_local_global_q_max_abs_error": float(error.max().item()),
        "global_local_global_q_mean_abs_error": float(error.mean().item()),
    }


def _build_local_dual_grid(
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the local C1024 geometry support and the encoder features."""
    coords, dual_vertices_world, intersected = (
        o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices=vertices.to(device="cpu", dtype=torch.float32),
            faces=faces.to(device="cpu", dtype=torch.int32),
            grid_size=OVOXEL_RESOLUTION,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=True,
        )
    )
    coords = coords.to(torch.int32).cpu()
    dual_vertices_world = dual_vertices_world.to(torch.float32).cpu()
    intersected = intersected.cpu()
    if coords.shape[0] == 0:
        raise RuntimeError("local mesh produced an empty flexible dual grid")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise RuntimeError(f"local dual-grid coords have invalid shape {coords.shape}")
    if bool(((coords < 0) | (coords >= OVOXEL_RESOLUTION)).any().item()):
        raise RuntimeError("local dual-grid coordinates lie outside C1024")

    # This is the exact representation used by FlexiDualGridVaeEncoder and by
    # data_toolkit/dual_grid.py: absolute dual vertices are expressed inside
    # their local voxel cell, not obtained from any global PBR coordinate.
    dual_vertices = dual_vertices_world * OVOXEL_RESOLUTION - coords.to(torch.float32)
    if not torch.isfinite(dual_vertices).all():
        raise RuntimeError("local dual vertices are non-finite")
    if float(dual_vertices.amin().item()) < -1e-3 or float(dual_vertices.amax().item()) > 1.001:
        raise RuntimeError(
            "local dual vertices are outside their voxel cells: "
            f"range={_tensor_range(dual_vertices)}"
        )
    dual_vertices = dual_vertices.clamp(0.0, 1.0)
    return coords, dual_vertices, dual_vertices_world, intersected


def _prepare_tile_geometry(
    *,
    global_vertices: torch.Tensor,
    global_faces: torch.Tensor,
    global_face_min: torch.Tensor,
    global_face_max: torch.Tensor,
    global_face_finite: torch.Tensor,
    global_camera: Mapping[str, float],
    transform: TileCameraTransform,
) -> LocalGeometry:
    face_ids = _tile_face_ids_from_bbox(
        global_face_min, global_face_max, global_face_finite, transform.box
    )
    if face_ids.numel() == 0:
        raise RuntimeError("no global triangle projection bbox intersects the tile")
    selected_faces = global_faces.index_select(0, face_ids).to(torch.int64)
    global_local_vertices, local_faces, global_vertex_ids = _compact_submesh(
        global_vertices, selected_faces
    )
    q_global = global_local_vertices.to(torch.float32) * (
        2.0 * float(global_camera["mesh_scale"])
    )
    q_local, local_uv = _global_q_to_local_q(
        q_global, global_camera=global_camera, transform=transform
    )
    if not torch.isfinite(q_local).all():
        raise RuntimeError("selected tile triangle vertices have invalid local q")
    local_vertices = q_local / (2.0 * float(transform.mesh_scale))
    roundtrip_stats = _camera_roundtrip_stats(
        q_global, global_camera=global_camera, transform=transform
    )
    coords, dual_vertices, dual_vertices_world, intersected = _build_local_dual_grid(
        local_vertices, local_faces
    )
    stats = {
        "projected_bbox_faces": int(face_ids.shape[0]),
        "selected_global_face_ids": int(face_ids.shape[0]),
        "local_mesh_faces": int(local_faces.shape[0]),
        "local_mesh_vertices": int(local_vertices.shape[0]),
        "local_dual_grid_entries": int(coords.shape[0]),
        "local_vertex_range": _tensor_range(local_vertices),
        "local_dual_vertices_world_range": _tensor_range(dual_vertices_world),
        "local_dual_vertices_cell_range": _tensor_range(dual_vertices),
        "local_intersected_shape": list(intersected.shape),
        "source_global_vertex_ids": int(global_vertex_ids.shape[0]),
        "selected_local_uv_range": _tensor_range(local_uv),
        **roundtrip_stats,
        "face_selection": (
            "triangle projected bbox intersects tile rectangle; no centroid, "
            "z-buffer, front/back, occlusion, or face-visibility filtering"
        ),
    }
    return LocalGeometry(
        vertices=local_vertices.to(device="cpu", dtype=torch.float32),
        faces=local_faces.to(device="cpu", dtype=torch.int64),
        coords=coords,
        dual_vertices=dual_vertices,
        dual_vertices_world=dual_vertices_world,
        intersected=intersected,
        selected_global_face_ids=face_ids.to(device="cpu", dtype=torch.long),
        stats=stats,
    )


def _make_local_reference_mesh(
    geometry: LocalGeometry,
    attrs: torch.Tensor,
    template: MeshWithVoxel,
) -> MeshWithVoxel:
    return MeshWithVoxel(
        vertices=geometry.vertices.to(torch.float32),
        faces=geometry.faces.to(torch.int32),
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / OVOXEL_RESOLUTION,
        coords=geometry.coords.to(torch.int32),
        attrs=attrs.to(torch.float32),
        voxel_shape=torch.Size(
            [1, int(attrs.shape[1]), OVOXEL_RESOLUTION, OVOXEL_RESOLUTION, OVOXEL_RESOLUTION]
        ),
        layout=dict(template.layout) if isinstance(template.layout, Mapping) else dict(PBR_LAYOUT),
    )


def _make_attribute_query_mesh(
    global_mesh: MeshWithVoxel,
    device: torch.device,
) -> MeshWithVoxel:
    """Keep only the baseline field needed by MeshWithVoxel.query_attrs()."""
    return MeshWithVoxel(
        vertices=torch.empty((1, 3), dtype=torch.float32, device=device),
        faces=torch.empty((0, 3), dtype=torch.int32, device=device),
        origin=torch.as_tensor(global_mesh.origin).cpu().tolist(),
        voxel_size=float(torch.as_tensor(global_mesh.voxel_size).item()),
        coords=global_mesh.coords.to(device=device, dtype=torch.int32),
        attrs=global_mesh.attrs.to(device=device, dtype=torch.float32),
        voxel_shape=torch.Size(global_mesh.voxel_shape),
        layout=dict(global_mesh.layout),
    )


@torch.no_grad()
def _query_mesh_attrs_chunked(
    mesh: MeshWithVoxel,
    points: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError("attribute query chunk size must be positive")
    if points.shape[0] == 0:
        return torch.empty((0, mesh.attrs.shape[1]), device=points.device)
    rows: List[torch.Tensor] = []
    for start in range(0, int(points.shape[0]), int(chunk_size)):
        rows.append(mesh.query_attrs(points[start : start + chunk_size]).to(torch.float32))
    return torch.cat(rows, dim=0)


def _closest_points_on_triangles(
    points: torch.Tensor,
    triangles: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return closest point, barycentric coordinates, and Euclidean distance."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab = b - a
    ac = c - a
    ap = points - a
    d00 = (ab * ab).sum(dim=1)
    d01 = (ab * ac).sum(dim=1)
    d11 = (ac * ac).sum(dim=1)
    d20 = (ap * ab).sum(dim=1)
    d21 = (ap * ac).sum(dim=1)
    denom = d00 * d11 - d01 * d01
    safe_denom = denom.clamp_min(1e-20)
    v = (d11 * d20 - d01 * d21) / safe_denom
    w = (d00 * d21 - d01 * d20) / safe_denom
    u = 1.0 - v - w
    plane_bary = torch.stack((u, v, w), dim=1)
    plane_point = (
        plane_bary[:, 0:1] * a
        + plane_bary[:, 1:2] * b
        + plane_bary[:, 2:3] * c
    )
    inside = (
        (denom > 1e-20)
        & (plane_bary >= 0.0).all(dim=1)
        & (plane_bary <= 1.0).all(dim=1)
    )

    def segment_candidate(
        p0: torch.Tensor,
        p1: torch.Tensor,
        first: int,
        second: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edge = p1 - p0
        t = ((points - p0) * edge).sum(dim=1) / (edge * edge).sum(dim=1).clamp_min(1e-20)
        t = t.clamp(0.0, 1.0)
        candidate = p0 + t[:, None] * edge
        bary = torch.zeros((points.shape[0], 3), device=points.device, dtype=points.dtype)
        bary[:, first] = 1.0 - t
        bary[:, second] = t
        return candidate, bary

    ab_point, ab_bary = segment_candidate(a, b, 0, 1)
    ac_point, ac_bary = segment_candidate(a, c, 0, 2)
    bc_point, bc_bary = segment_candidate(b, c, 1, 2)
    candidates = torch.stack((a, b, c, ab_point, ac_point, bc_point), dim=1)
    candidate_bary = torch.stack(
        (
            torch.tensor([1.0, 0.0, 0.0], device=points.device, dtype=points.dtype).expand(points.shape[0], -1),
            torch.tensor([0.0, 1.0, 0.0], device=points.device, dtype=points.dtype).expand(points.shape[0], -1),
            torch.tensor([0.0, 0.0, 1.0], device=points.device, dtype=points.dtype).expand(points.shape[0], -1),
            ab_bary,
            ac_bary,
            bc_bary,
        ),
        dim=1,
    )
    distance2 = (candidates - points[:, None]).square().sum(dim=2)
    best = distance2.argmin(dim=1)
    nearest = candidates.gather(1, best[:, None, None].expand(-1, 1, 3)).squeeze(1)
    nearest_bary = candidate_bary.gather(
        1, best[:, None, None].expand(-1, 1, 3)
    ).squeeze(1)
    nearest_distance = distance2.gather(1, best[:, None]).squeeze(1).clamp_min(0).sqrt()
    plane_distance = (plane_point - points).square().sum(dim=1).clamp_min(0).sqrt()
    closest = torch.where(inside[:, None], plane_point, nearest)
    bary = torch.where(inside[:, None], plane_bary, nearest_bary)
    distance = torch.where(inside, plane_distance, nearest_distance)
    return closest, bary, distance


def _nearest_faces_by_surface_distance(
    points: torch.Tensor,
    triangles: torch.Tensor,
    *,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find the exact closest triangle for numerical support fallbacks."""
    if chunk_size <= 0:
        raise ValueError("triangle fallback chunk size must be positive")
    best_distance = torch.full(
        (points.shape[0],), float("inf"), device=points.device, dtype=torch.float32
    )
    best_face = torch.full(
        (points.shape[0],), -1, device=points.device, dtype=torch.long
    )
    best_point = torch.zeros(
        (points.shape[0], 3), device=points.device, dtype=torch.float32
    )
    best_bary = torch.zeros(
        (points.shape[0], 3), device=points.device, dtype=torch.float32
    )
    point_chunk_size = 4096
    for face_start in range(0, int(triangles.shape[0]), int(chunk_size)):
        triangle_chunk = triangles[face_start : face_start + chunk_size].to(torch.float32)
        for point_start in range(0, int(points.shape[0]), point_chunk_size):
            point_chunk = points[point_start : point_start + point_chunk_size].to(torch.float32)
            point_count = int(point_chunk.shape[0])
            triangle_count = int(triangle_chunk.shape[0])
            expanded_points = point_chunk[:, None, :].expand(-1, triangle_count, -1)
            expanded_triangles = triangle_chunk[None, :, :, :].expand(point_count, -1, -1, -1)
            _, _, distances_flat = _closest_points_on_triangles(
                expanded_points.reshape(-1, 3),
                expanded_triangles.reshape(-1, 3, 3),
            )
            distances = distances_flat.reshape(point_count, triangle_count)
            values, rows = distances.min(dim=1)
            replace = values < best_distance[point_start : point_start + point_count]
            if bool(replace.any().item()):
                local_points, local_bary, _ = _closest_points_on_triangles(
                    point_chunk,
                    triangle_chunk.index_select(0, rows),
                )
                best_distance[point_start : point_start + point_count] = torch.where(
                    replace,
                    values,
                    best_distance[point_start : point_start + point_count],
                )
                best_face[point_start : point_start + point_count] = torch.where(
                    replace,
                    rows + int(face_start),
                    best_face[point_start : point_start + point_count],
                )
                best_point[point_start : point_start + point_count] = torch.where(
                    replace[:, None],
                    local_points,
                    best_point[point_start : point_start + point_count],
                )
                best_bary[point_start : point_start + point_count] = torch.where(
                    replace[:, None],
                    local_bary,
                    best_bary[point_start : point_start + point_count],
                )
            del expanded_points, expanded_triangles, distances_flat, distances
        del triangle_chunk
    if bool((best_face < 0).any().item()):
        raise RuntimeError("nearest surface fallback found no triangle")
    return best_face, best_point, best_bary, best_distance


@torch.no_grad()
def _resample_local_attrs_from_global(
    *,
    geometry: LocalGeometry,
    global_attr_field: MeshWithVoxel,
    global_camera: Mapping[str, float],
    transform: TileCameraTransform,
    query_chunk_size: int,
    face_chunk_size: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Sample the baseline PBR field at local surface correspondences.

    A local dual-grid cell is assigned every local triangle whose triangle AABB
    overlaps that cell.  The voxel center is projected to each candidate using
    the exact closest-point/barycentric operation.  Candidates that can reach
    the cell are queried in global object space and averaged with inverse
    distance weights.  A nearest-candidate/nearest-triangle fallback only
    handles numerical or degenerate input and never changes the support.
    """
    if query_chunk_size <= 0 or face_chunk_size <= 0:
        raise ValueError("material query and face chunk sizes must be positive")
    triangles_cpu = geometry.vertices.index_select(
        0, geometry.faces.to(torch.long).reshape(-1)
    ).reshape(-1, 3, 3)
    if triangles_cpu.shape[0] == 0:
        raise RuntimeError("local geometry contains no triangle for material sampling")
    twice_area = torch.linalg.vector_norm(
        torch.linalg.cross(
            triangles_cpu[:, 1] - triangles_cpu[:, 0],
            triangles_cpu[:, 2] - triangles_cpu[:, 0],
            dim=1,
        ),
        dim=1,
    )
    valid_triangles = torch.isfinite(twice_area) & (twice_area > 1e-12)
    if not bool(valid_triangles.any().item()):
        raise RuntimeError("local geometry contains no non-degenerate triangle")
    triangles_cpu = triangles_cpu[valid_triangles]

    coords_cpu = geometry.coords.to(device="cpu", dtype=torch.int64)
    local_size = 1.0 / float(OVOXEL_RESOLUTION)
    local_origin = -0.5
    support_by_key: Dict[int, int] = {}
    for row, coord in enumerate(coords_cpu.tolist()):
        key = (int(coord[0]) * OVOXEL_RESOLUTION + int(coord[1])) * OVOXEL_RESOLUTION + int(coord[2])
        support_by_key[key] = int(row)

    candidate_lists: List[List[int]] = [[] for _ in range(int(coords_cpu.shape[0]))]
    tri_min = torch.floor(
        (triangles_cpu.amin(dim=1) - local_origin) / local_size
    ).to(torch.int64)
    tri_max = torch.floor(
        (triangles_cpu.amax(dim=1) - local_origin) / local_size
    ).to(torch.int64)
    tri_min = tri_min.clamp(0, OVOXEL_RESOLUTION - 1)
    tri_max = tri_max.clamp(0, OVOXEL_RESOLUTION - 1)
    for face_id, (lo, hi) in enumerate(zip(tri_min.tolist(), tri_max.tolist())):
        if any(int(lo[i]) > int(hi[i]) for i in range(3)):
            continue
        for ix in range(int(lo[0]), int(hi[0]) + 1):
            for iy in range(int(lo[1]), int(hi[1]) + 1):
                for iz in range(int(lo[2]), int(hi[2]) + 1):
                    key = (ix * OVOXEL_RESOLUTION + iy) * OVOXEL_RESOLUTION + iz
                    row = support_by_key.get(key)
                    if row is not None:
                        candidate_lists[row].append(int(face_id))

    device = global_attr_field.device
    triangles = triangles_cpu.to(device=device, dtype=torch.float32)
    centers = (
        torch.tensor([-0.5, -0.5, -0.5], device=device)[None]
        + (coords_cpu.to(device=device, dtype=torch.float32) + 0.5) * local_size
    )
    pair_rows_cpu: List[int] = []
    pair_faces_cpu: List[int] = []
    for row, candidates in enumerate(candidate_lists):
        pair_rows_cpu.extend([row] * len(candidates))
        pair_faces_cpu.extend(candidates)

    accepted_rows = torch.empty((0,), device=device, dtype=torch.long)
    accepted_faces = torch.empty((0,), device=device, dtype=torch.long)
    accepted_points = torch.empty((0, 3), device=device, dtype=torch.float32)
    accepted_bary = torch.empty((0, 3), device=device, dtype=torch.float32)
    accepted_distances = torch.empty((0,), device=device, dtype=torch.float32)
    candidate_fallback_rows = torch.empty((0,), device=device, dtype=torch.long)
    candidate_fallback_faces = torch.empty((0,), device=device, dtype=torch.long)
    candidate_fallback_points = torch.empty((0, 3), device=device, dtype=torch.float32)
    candidate_fallback_bary = torch.empty((0, 3), device=device, dtype=torch.float32)
    candidate_fallback_distances = torch.empty((0,), device=device, dtype=torch.float32)

    if pair_rows_cpu:
        pair_rows = torch.tensor(pair_rows_cpu, device=device, dtype=torch.long)
        pair_faces = torch.tensor(pair_faces_cpu, device=device, dtype=torch.long)
        pair_points: List[torch.Tensor] = []
        pair_barycentric: List[torch.Tensor] = []
        pair_distances: List[torch.Tensor] = []
        for start in range(0, int(pair_rows.shape[0]), int(query_chunk_size)):
            point, barycentric, distance = _closest_points_on_triangles(
                centers.index_select(0, pair_rows[start : start + query_chunk_size]),
                triangles.index_select(0, pair_faces[start : start + query_chunk_size]),
            )
            pair_points.append(point)
            pair_barycentric.append(barycentric)
            pair_distances.append(distance)
        all_points = torch.cat(pair_points, dim=0)
        all_barycentric = torch.cat(pair_barycentric, dim=0)
        all_distances = torch.cat(pair_distances, dim=0)
        max_distance = math.sqrt(3.0) * 0.5 * local_size + 1e-7
        accepted_mask = torch.isfinite(all_distances) & (all_distances <= max_distance)
        accepted_rows = pair_rows[accepted_mask]
        accepted_faces = pair_faces[accepted_mask]
        accepted_points = all_points[accepted_mask]
        accepted_bary = all_barycentric[accepted_mask]
        accepted_distances = all_distances[accepted_mask]

        covered = torch.zeros(centers.shape[0], device=device, dtype=torch.bool)
        if accepted_rows.numel():
            covered[accepted_rows] = True
        candidate_present = torch.tensor(
            [bool(value) for value in candidate_lists], device=device, dtype=torch.bool
        )
        fallback_needed = (~covered) & candidate_present
        if bool(fallback_needed.any().item()):
            sortable_distances = torch.where(
                torch.isfinite(all_distances),
                all_distances,
                torch.full_like(all_distances, float("inf")),
            )
            order = torch.argsort(sortable_distances, stable=True)
            seen = torch.zeros(centers.shape[0], device=device, dtype=torch.bool)
            sorted_rows = pair_rows[order]
            first = torch.isfinite(all_distances[order]) & ~seen[sorted_rows]
            seen[sorted_rows[first]] = True
            best_positions = order[first]
            best_rows = pair_rows[best_positions]
            use = fallback_needed[best_rows]
            candidate_fallback_rows = best_rows[use]
            candidate_fallback_faces = pair_faces[best_positions[use]]
            candidate_fallback_points = all_points[best_positions[use]]
            candidate_fallback_bary = all_barycentric[best_positions[use]]
            candidate_fallback_distances = all_distances[best_positions[use]]
    else:
        covered = torch.zeros(centers.shape[0], device=device, dtype=torch.bool)
        candidate_present = torch.zeros(centers.shape[0], device=device, dtype=torch.bool)

    assigned = torch.zeros(centers.shape[0], device=device, dtype=torch.bool)
    if accepted_rows.numel():
        assigned[accepted_rows] = True
    if candidate_fallback_rows.numel():
        assigned[candidate_fallback_rows] = True
    missing_rows = torch.where(~assigned)[0]
    nearest_fallback_faces = torch.empty((0,), device=device, dtype=torch.long)
    nearest_fallback_points = torch.empty((0, 3), device=device, dtype=torch.float32)
    nearest_fallback_bary = torch.empty((0, 3), device=device, dtype=torch.float32)
    nearest_fallback_distances = torch.empty((0,), device=device, dtype=torch.float32)
    if missing_rows.numel():
        (
            nearest_fallback_faces,
            nearest_fallback_points,
            nearest_fallback_bary,
            nearest_fallback_distances,
        ) = _nearest_faces_by_surface_distance(
            centers.index_select(0, missing_rows), triangles, chunk_size=face_chunk_size
        )

    final_rows = torch.cat(
        (
            accepted_rows,
            candidate_fallback_rows,
            missing_rows,
        ),
        dim=0,
    )
    final_faces = torch.cat(
        (
            accepted_faces,
            candidate_fallback_faces,
            nearest_fallback_faces,
        ),
        dim=0,
    )
    final_points = torch.cat(
        (
            accepted_points,
            candidate_fallback_points,
            nearest_fallback_points,
        ),
        dim=0,
    )
    final_barycentric = torch.cat(
        (
            accepted_bary,
            candidate_fallback_bary,
            nearest_fallback_bary,
        ),
        dim=0,
    )
    final_distances = torch.cat(
        (
            accepted_distances,
            candidate_fallback_distances,
            nearest_fallback_distances,
        ),
        dim=0,
    )
    if final_rows.numel() == 0:
        raise RuntimeError("local material resampling produced no triangle correspondences")
    if not torch.isfinite(final_barycentric).all():
        raise RuntimeError("local material resampling produced non-finite barycentric coordinates")
    final_triangles = triangles.index_select(0, final_faces)
    # Keep the queried surface point explicitly coupled to the computed
    # triangle barycentric coordinates.
    final_points = (final_triangles * final_barycentric[:, :, None]).sum(dim=1)

    # The closest point is local object space.  This is not an affine vertex
    # transform: every surface point goes through the exact inverse camera map
    # before the baseline field is queried.
    q_local_surface = final_points * (2.0 * float(transform.mesh_scale))
    q_global_surface, _ = _local_q_to_global_q(
        q_local_surface, global_camera=global_camera, transform=transform
    )
    global_surface = q_global_surface / (2.0 * float(global_camera["mesh_scale"]))
    queried_attrs = _query_mesh_attrs_chunked(
        global_attr_field, global_surface, chunk_size=query_chunk_size
    )
    eps = max(local_size * 1e-3, 1e-8)
    weights = 1.0 / (final_distances.to(torch.float32).clamp_min(0.0) + eps)
    attr_sum = torch.zeros(
        (centers.shape[0], queried_attrs.shape[1]), device=device, dtype=torch.float32
    )
    weight_sum = torch.zeros((centers.shape[0],), device=device, dtype=torch.float32)
    attr_sum.index_add_(0, final_rows, queried_attrs * weights[:, None])
    weight_sum.index_add_(0, final_rows, weights)
    if bool((weight_sum <= 0).any().item()) or not torch.isfinite(attr_sum).all():
        raise RuntimeError("local material resampling left unsupported local cells")
    local_attrs = attr_sum / weight_sum[:, None]
    stats = {
        "local_attrs_support_tokens": int(local_attrs.shape[0]),
        "local_attrs_channels": int(local_attrs.shape[1]),
        "local_attrs_range": _tensor_range(local_attrs),
        "triangle_candidate_pairs": int(len(pair_rows_cpu)),
        "accepted_triangle_contributions": int(accepted_rows.shape[0]),
        "candidate_nearest_fallback_cells": int(candidate_fallback_rows.shape[0]),
        "nearest_triangle_fallback_cells": int(missing_rows.shape[0]),
        "mean_surface_projection_distance_local": float(final_distances.mean().item()),
        "max_surface_projection_distance_local": float(final_distances.max().item()),
        "barycentric_coordinates": True,
        "distance_weight": "1 / (distance + max(local_voxel_size*1e-3, 1e-8))",
        "triangle_membership": (
            "triangle AABB cell candidates, closest-point distance gate at half "
            "voxel diagonal, then nearest candidate fallback"
        ),
        "global_query": (
            "local closest surface point -> exact local_q_to_global_q -> "
            "global MeshWithVoxel.query_attrs trilinear query"
        ),
    }
    return local_attrs.to(device="cpu", dtype=torch.float32), stats


def _encode_local_shape(
    *,
    encoder: torch.nn.Module,
    local_coords: Optional[torch.Tensor] = None,
    local_dual_vertices: Optional[torch.Tensor] = None,
    local_intersected: Optional[torch.Tensor] = None,
    # Compatibility inputs are accepted only to build a fresh local dual grid;
    # no global support or material coordinates are consulted.
    vertices: Optional[torch.Tensor] = None,
    faces: Optional[torch.Tensor] = None,
    device: torch.device,
    low_vram: bool,
) -> Tuple[SparseTensor, Dict[str, Any]]:
    started = time.perf_counter()
    if local_coords is None or local_dual_vertices is None or local_intersected is None:
        if vertices is None or faces is None:
            raise ValueError("local dual-grid inputs or vertices/faces are required")
        local_coords, local_dual_vertices, _, local_intersected = _build_local_dual_grid(
            vertices, faces
        )
    coords4 = torch.cat(
        [torch.zeros_like(local_coords[:, :1]), local_coords], dim=1
    ).to(device=device, dtype=torch.int32)
    vertex_sparse = SparseTensor(
        local_dual_vertices.to(device=device, dtype=torch.float32), coords4
    )
    intersected_sparse = vertex_sparse.replace(
        local_intersected.to(device=device)
    )
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
    if not isinstance(latent, SparseTensor) or not torch.isfinite(latent.feats).all():
        raise RuntimeError("shape encoder produced an invalid latent")
    stats = {
        "input_coords": int(local_coords.shape[0]),
        "input_dual_vertices_range": _tensor_range(local_dual_vertices),
        "input_intersected_shape": list(local_intersected.shape),
        "shape_latent_tokens": int(latent.feats.shape[0]),
        "shape_latent_channels": int(latent.feats.shape[1]),
        "shape_latent_coords_range": _tensor_range(latent.coords[:, 1:].to(torch.float32)),
        "shape_encoder_seconds": float(time.perf_counter() - started),
    }
    del vertex_sparse, intersected_sparse
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
    coords4 = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1).to(
        device=device, dtype=torch.int32
    )
    attrs = attrs.to(device=device, dtype=torch.float32)
    # The released PBR encoder is trained on [0,1] attributes mapped to [-1,1].
    encoder_input = (attrs * 2.0 - 1.0).clamp(-1.0, 1.0)
    sparse = SparseTensor(encoder_input, coords4)
    if low_vram:
        encoder.to(device)
    with torch.no_grad():
        latent = encoder(sparse, sample_posterior=False)
    _sync_cuda()
    if low_vram:
        encoder.cpu()
    if not isinstance(latent, SparseTensor) or not torch.isfinite(latent.feats).all():
        raise RuntimeError("PBR encoder produced an invalid latent")
    stats = {
        "input_coords": int(coords.shape[0]),
        "input_attrs_range": _tensor_range(attrs),
        "pbr_latent_tokens": int(latent.feats.shape[0]),
        "pbr_latent_channels": int(latent.feats.shape[1]),
        "pbr_latent_coords_range": _tensor_range(latent.coords[:, 1:].to(torch.float32)),
        "pbr_encoder_seconds": float(time.perf_counter() - started),
    }
    del sparse, encoder_input
    _empty_cuda_cache()
    return latent, stats


def _latent_support_diagnostics(
    shape_latent: SparseTensor,
    texture_latent: SparseTensor,
) -> Dict[str, Any]:
    shape_coords = shape_latent.coords.to(torch.int64)
    texture_coords = texture_latent.coords.to(torch.int64)
    exact = tuple(shape_coords.shape) == tuple(texture_coords.shape) and torch.equal(
        shape_coords, texture_coords
    )
    shape_keys = _linear_keys(shape_coords[:, 1:], LATENT_RESOLUTION)
    texture_keys = _linear_keys(texture_coords[:, 1:], LATENT_RESOLUTION)
    shape_unique = torch.unique(shape_keys)
    texture_unique = torch.unique(texture_keys)
    set_equal = shape_unique.shape == texture_unique.shape and torch.equal(
        shape_unique.sort().values, texture_unique.sort().values
    )
    return {
        "coordinates_exactly_equal": bool(exact),
        "coordinate_sets_equal": bool(set_equal),
        "shape_tokens": int(shape_coords.shape[0]),
        "texture_tokens": int(texture_coords.shape[0]),
        "shape_only_tokens": int(torch.isin(shape_unique, texture_unique).logical_not().sum().item()),
        "texture_only_tokens": int(torch.isin(texture_unique, shape_unique).logical_not().sum().item()),
        "support_action": "use exact encoder output support; no intersection or silent dropping",
    }


def _normalize_slat(
    value: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
) -> SparseTensor:
    mean = torch.as_tensor(normalization["mean"], device=value.device, dtype=value.feats.dtype)[None]
    std = torch.as_tensor(normalization["std"], device=value.device, dtype=value.feats.dtype)[None]
    if mean.shape[1] != value.feats.shape[1] or std.shape[1] != value.feats.shape[1]:
        raise ValueError(
            "latent normalization channel count does not match encoder output: "
            f"stats={mean.shape[1]} latent={value.feats.shape[1]}"
        )
    if bool((std == 0).any().item()):
        raise ValueError("latent normalization contains zero standard deviation")
    return value.replace((value.feats - mean) / std)


def _denormalize_slat(
    value: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
) -> SparseTensor:
    mean = torch.as_tensor(normalization["mean"], device=value.device, dtype=value.feats.dtype)[None]
    std = torch.as_tensor(normalization["std"], device=value.device, dtype=value.feats.dtype)[None]
    if mean.shape[1] != value.feats.shape[1] or std.shape[1] != value.feats.shape[1]:
        raise ValueError("latent denormalization channel count mismatch")
    return value.replace(value.feats * std + mean)


def _native_noised_endpoint(
    clean_endpoint: SparseTensor,
    noise: SparseTensor,
    sampler: Any,
    timestep: float,
) -> SparseTensor:
    """Use FlowEulerSampler's x0/epsilon convention for a noised endpoint.

    ``FlowEulerSampler._v_to_xstart_eps`` defines
    ``x_t = (1-t) * x_0 + sigma(t) * epsilon`` with
    ``sigma(t)=sigma_min+(1-sigma_min)t``.  This helper mirrors that native
    convention; it is not a DDIM or ad-hoc ``x0 + noise`` operation.
    """
    if not torch.equal(clean_endpoint.coords, noise.coords):
        raise RuntimeError("reference endpoint and flow noise coordinates differ")
    if clean_endpoint.feats.shape != noise.feats.shape:
        raise RuntimeError("reference endpoint and flow noise feature shapes differ")
    t = float(timestep)
    if not 0.0 <= t <= 1.0:
        raise ValueError("native flow timestep must lie in [0,1]")
    sigma = float(sampler.sigma_min) + (1.0 - float(sampler.sigma_min)) * t
    return clean_endpoint.replace(
        (1.0 - t) * clean_endpoint.feats + sigma * noise.feats
    )


def _run_native_tile_flow(
    *,
    pipeline: Any,
    tile_image: Image.Image,
    transform: TileCameraTransform,
    shape_reference: SparseTensor,
    texture_reference: SparseTensor,
    shape_params: Mapping[str, Any],
    texture_params: Mapping[str, Any],
    seed: int,
    tile_id: int,
) -> TileFlowLatents:
    """Run shape and texture flow on the exact local encoder support."""
    device = torch.device(pipeline.device)
    shape_reference = shape_reference.to(device)
    texture_reference = texture_reference.to(device)
    if not torch.equal(shape_reference.coords, texture_reference.coords):
        raise RuntimeError("shape/PBR reference supports differ before flow")
    coords = shape_reference.coords.to(torch.int32)
    if coords.shape[0] > 0 and bool(((coords[:, 1:] < 0) | (coords[:, 1:] >= LATENT_RESOLUTION)).any().item()):
        raise RuntimeError("local latent coordinates lie outside C64")

    shape_condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [tile_image.convert("RGB")],
        coords,
        camera_angle_x=float(transform.camera_angle_x),
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
        grid_resolution_override=LATENT_RESOLUTION,
    )
    texture_condition = None
    shape_model = pipeline.models["shape_slat_flow_model_1024"]
    texture_model = pipeline.models["tex_slat_flow_model_1024"]
    merged_shape_params = {**pipeline.shape_slat_sampler_params, **dict(shape_params)}
    merged_texture_params = {**pipeline.tex_slat_sampler_params, **dict(texture_params)}
    shape_steps = int(merged_shape_params["steps"])
    texture_steps = int(merged_texture_params["steps"])
    shape_times = pipeline.shape_slat_sampler.timestep_schedule(
        shape_steps, float(merged_shape_params["rescale_t"])
    )
    texture_times = pipeline.tex_slat_sampler.timestep_schedule(
        texture_steps, float(merged_texture_params["rescale_t"])
    )
    if shape_times != texture_times and shape_steps == texture_steps:
        raise RuntimeError("shape and texture native timestep schedules differ")

    _seed_everything(int(seed))
    shape_clean = _normalize_slat(shape_reference, pipeline.shape_slat_normalization)
    if int(shape_model.in_channels) != int(shape_clean.feats.shape[1]):
        raise RuntimeError(
            "shape flow input channels do not match encoded reference: "
            f"flow={shape_model.in_channels} reference={shape_clean.feats.shape[1]}"
        )
    shape_noise = SparseTensor(
        torch.randn(
            coords.shape[0], int(shape_model.in_channels), device=device, dtype=torch.float32
        ),
        coords,
    )
    shape_noised = _native_noised_endpoint(
        shape_clean, shape_noise, pipeline.shape_slat_sampler, shape_times[0]
    )
    if pipeline.low_vram:
        shape_model.to(device)
    shape_started = time.perf_counter()
    try:
        shape_result = pipeline.shape_slat_sampler.sample(
            shape_model,
            shape_noised,
            **shape_condition,
            **merged_shape_params,
            verbose=True,
            tqdm_desc=f"Tile {tile_id:02d} shape SLat flow",
            return_model_history=False,
        )
    finally:
        if pipeline.low_vram:
            shape_model.cpu()
    _sync_cuda()
    shape_norm = getattr(shape_result, "samples", shape_result)
    if not isinstance(shape_norm, SparseTensor) or not torch.equal(shape_norm.coords, coords):
        raise RuntimeError("shape flow changed local latent coordinates")
    shape_seconds = time.perf_counter() - shape_started
    shape_denorm = _denormalize_slat(shape_norm, pipeline.shape_slat_normalization)
    shape_noise_range = _tensor_range(shape_noise.feats)
    shape_noised_reference_range = _tensor_range(shape_noised.feats)
    del shape_condition, shape_result, shape_noised, shape_noise, shape_clean
    _empty_cuda_cache()

    texture_condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [tile_image.convert("RGB")],
        coords,
        camera_angle_x=float(transform.camera_angle_x),
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
        grid_resolution_override=LATENT_RESOLUTION,
    )
    shape_cond_norm = _normalize_slat(shape_denorm, pipeline.shape_slat_normalization)
    texture_channels = int(texture_model.in_channels) - int(shape_cond_norm.feats.shape[1])
    if texture_channels <= 0:
        raise RuntimeError(
            "texture flow input channels must exceed generated shape condition "
            f"channels, got flow={texture_model.in_channels} "
            f"shape={shape_cond_norm.feats.shape[1]}"
        )
    texture_clean = _normalize_slat(texture_reference, pipeline.tex_slat_normalization)
    if int(texture_clean.feats.shape[1]) != texture_channels:
        raise RuntimeError(
            "texture flow noise channels do not match encoded PBR reference: "
            f"flow_noise={texture_channels} reference={texture_clean.feats.shape[1]}"
        )
    texture_noise = SparseTensor(
        torch.randn(
            coords.shape[0], texture_channels, device=device, dtype=torch.float32
        ),
        coords,
    )
    texture_noised = _native_noised_endpoint(
        texture_clean, texture_noise, pipeline.tex_slat_sampler, texture_times[0]
    )
    if pipeline.low_vram:
        texture_model.to(device)
    texture_started = time.perf_counter()
    try:
        texture_result = pipeline.tex_slat_sampler.sample(
            texture_model,
            texture_noised,
            concat_cond=shape_cond_norm,
            **texture_condition,
            **merged_texture_params,
            verbose=True,
            tqdm_desc=f"Tile {tile_id:02d} texture SLat flow",
            return_model_history=False,
        )
    finally:
        if pipeline.low_vram:
            texture_model.cpu()
    _sync_cuda()
    texture_norm = getattr(texture_result, "samples", texture_result)
    if not isinstance(texture_norm, SparseTensor) or not torch.equal(texture_norm.coords, coords):
        raise RuntimeError("texture flow changed local latent coordinates")
    texture_seconds = time.perf_counter() - texture_started
    texture_denorm = _denormalize_slat(texture_norm, pipeline.tex_slat_normalization)
    stats = {
        "seed": int(seed),
        "shape_tokens": int(shape_norm.feats.shape[0]),
        "texture_tokens": int(texture_norm.feats.shape[0]),
        "shape_reference_range": _tensor_range(shape_reference.feats),
        "texture_reference_range": _tensor_range(texture_reference.feats),
        "shape_noise_range": shape_noise_range,
        "texture_noise_range": _tensor_range(texture_noise.feats),
        "shape_noised_reference_range": shape_noised_reference_range,
        "texture_noised_reference_range": _tensor_range(texture_noised.feats),
        "shape_flow_seconds": float(shape_seconds),
        "texture_flow_seconds": float(texture_seconds),
        "shape_sampler": dict(merged_shape_params),
        "texture_sampler": dict(merged_texture_params),
        "shape_timestep_schedule": [float(v) for v in shape_times],
        "texture_timestep_schedule": [float(v) for v in texture_times],
        "noise_timestep_used": {
            "shape": float(shape_times[0]),
            "texture": float(texture_times[0]),
        },
        "flow_matching_convention": (
            "native FlowEulerSampler endpoint bridge: x_t=(1-t)x0+sigma(t)eps; "
            "sigma(t)=sigma_min+(1-sigma_min)t; first native t is used before "
            "the unchanged Pixal3D Euler/CFG sampler"
        ),
        "tile_condition": "fresh shape_1024 and tex_1024 DINO projection for the 1024 tile",
        "texture_condition": "generated shape SLat normalized support passed as concat_cond",
        "support_action": "exact local encoder support retained through both flows; no intersection",
    }
    del texture_condition, texture_result, texture_noised, texture_noise, texture_clean, shape_cond_norm
    _empty_cuda_cache()
    return TileFlowLatents(
        reference_shape=shape_reference,
        reference_texture=texture_reference,
        shape_norm=shape_norm,
        shape_denorm=shape_denorm,
        texture_norm=texture_norm,
        texture_denorm=texture_denorm,
        stats=stats,
    )


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
    if not torch.isfinite(mesh.vertices).all() or not torch.isfinite(mesh.attrs).all():
        raise RuntimeError(f"{label}: mesh contains non-finite tensors")
    return mesh


def _sample_mesh_surface(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    samples: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if samples <= 0:
        raise ValueError("surface sample count must be positive")
    triangles = vertices.index_select(0, faces.to(torch.long).reshape(-1)).reshape(-1, 3, 3)
    cross = torch.linalg.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=1
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
    cdf = torch.cumsum(twice_area.to(torch.float32), dim=0)
    total = cdf[-1]
    draws = torch.rand(samples, device=vertices.device, generator=generator) * total
    rows = torch.searchsorted(cdf, draws).clamp_max(triangles.shape[0] - 1)
    selected = triangles.index_select(0, rows)
    u = torch.rand((samples, 1), device=vertices.device, generator=generator)
    v = torch.rand((samples, 1), device=vertices.device, generator=generator)
    sqrt_u = torch.sqrt(u)
    b0 = 1.0 - sqrt_u
    b1 = sqrt_u * (1.0 - v)
    b2 = sqrt_u * v
    points = b0 * selected[:, 0] + b1 * selected[:, 1] + b2 * selected[:, 2]
    normals = F.normalize(cross.index_select(0, rows), dim=1)
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
        values, rows = torch.cdist(
            source[start : start + chunk_size].to(torch.float32),
            target.to(torch.float32),
        ).min(dim=1)
        distances.append(values)
        indices.append(rows)
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
        reference.vertices.to(device), reference.faces.to(device), samples=samples, seed=seed
    )
    pred_points, pred_normals = _sample_mesh_surface(
        prediction.vertices.to(device), prediction.faces.to(device), samples=samples, seed=seed + 1
    )
    ref_to_pred, ref_nn = _nearest_distances(ref_points, pred_points, chunk_size=chunk_size)
    pred_to_ref, pred_nn = _nearest_distances(pred_points, ref_points, chunk_size=chunk_size)
    ref_cos = (ref_normals * pred_normals.index_select(0, ref_nn)).sum(dim=1)
    pred_cos = (pred_normals * ref_normals.index_select(0, pred_nn)).sum(dim=1)
    voxel_size = float(torch.as_tensor(reference.voxel_size).item())
    metrics: Dict[str, Any] = {
        "surface_samples_per_mesh": int(samples),
        "chamfer_l1_object": float(0.5 * (ref_to_pred.mean() + pred_to_ref.mean()).item()),
        "chamfer_l2_object": float(0.5 * (ref_to_pred.square().mean() + pred_to_ref.square().mean()).item()),
        "symmetric_p95_object": float(torch.cat((ref_to_pred, pred_to_ref)).quantile(0.95).item()),
        "symmetric_hausdorff_object": float(torch.maximum(ref_to_pred.max(), pred_to_ref.max()).item()),
        "normal_cosine_oriented_mean": float(0.5 * (ref_cos.mean() + pred_cos.mean()).item()),
        "normal_cosine_absolute_mean": float(0.5 * (ref_cos.abs().mean() + pred_cos.abs().mean()).item()),
    }
    metrics["chamfer_l1_in_local_voxels"] = metrics["chamfer_l1_object"] / voxel_size
    metrics["symmetric_p95_in_local_voxels"] = metrics["symmetric_p95_object"] / voxel_size
    for cells in (1, 2, 4, 8):
        threshold = cells * voxel_size
        recall = float((ref_to_pred <= threshold).float().mean().item())
        precision = float((pred_to_ref <= threshold).float().mean().item())
        fscore = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        metrics[f"precision_at_{cells}_local_voxels"] = precision
        metrics[f"recall_at_{cells}_local_voxels"] = recall
        metrics[f"fscore_at_{cells}_local_voxels"] = fscore
    del ref_points, pred_points, ref_normals, pred_normals
    _empty_cuda_cache()
    return metrics


def _support_similarity(
    reference_coords: torch.Tensor,
    prediction_coords: torch.Tensor,
    resolution: int = OVOXEL_RESOLUTION,
) -> Dict[str, Any]:
    # Reference meshes are intentionally kept on CPU while decoded flow
    # meshes normally remain on CUDA.  Compare compact integer keys on one
    # device so support diagnostics never perform a cross-device searchsorted.
    ref = torch.unique(
        _linear_keys(reference_coords.to(device="cpu", dtype=torch.int64), resolution)
    ).sort().values
    pred = torch.unique(
        _linear_keys(prediction_coords.to(device="cpu", dtype=torch.int64), resolution)
    ).sort().values
    positions = torch.searchsorted(pred, ref)
    valid = positions < pred.shape[0]
    safe = positions.clamp_max(max(0, pred.shape[0] - 1))
    intersection = int((valid & (pred.index_select(0, safe) == ref)).sum().item())
    union = int(ref.shape[0] + pred.shape[0] - intersection)
    precision = 0.0 if pred.shape[0] == 0 else intersection / int(pred.shape[0])
    recall = 0.0 if ref.shape[0] == 0 else intersection / int(ref.shape[0])
    return {
        "reference_tokens": int(ref.shape[0]),
        "prediction_tokens": int(pred.shape[0]),
        "intersection": intersection,
        "union": union,
        "iou": 0.0 if union == 0 else intersection / union,
        "precision": precision,
        "recall": recall,
    }


def _local_mesh_to_global_patch(
    *,
    tile_id: int,
    box: Sequence[int],
    local_mesh: MeshWithVoxel,
    global_camera: Mapping[str, float],
    transform: TileCameraTransform,
    query_chunk_size: int,
) -> ReturnedTilePatch:
    """Return every decoded local face as independent global face corners."""
    if query_chunk_size <= 0:
        raise ValueError("face-corner query chunk size must be positive")
    mesh = local_mesh
    vertices = mesh.vertices.to(torch.float32)
    faces = mesh.faces.to(torch.long)
    if faces.shape[0] == 0:
        raise RuntimeError("local flow mesh has no faces")
    corner_local = vertices.index_select(0, faces.reshape(-1)).reshape(-1, 3)
    corner_attrs = _query_mesh_attrs_chunked(
        mesh, corner_local, chunk_size=query_chunk_size
    )
    q_local = corner_local * (2.0 * float(transform.mesh_scale))
    q_global, _ = _local_q_to_global_q(
        q_local, global_camera=global_camera, transform=transform
    )
    global_corners = q_global / (2.0 * float(global_camera["mesh_scale"]))
    q_local_roundtrip, _ = _global_q_to_local_q(
        q_global, global_camera=global_camera, transform=transform
    )
    roundtrip_error = (q_local_roundtrip - q_local).abs()
    corner_faces = torch.arange(
        global_corners.shape[0], device=global_corners.device, dtype=torch.int32
    ).reshape(-1, 3)
    stats = {
        "local_decoder_vertices": int(vertices.shape[0]),
        "local_decoder_faces": int(faces.shape[0]),
        "returned_global_corner_vertices": int(global_corners.shape[0]),
        "returned_global_faces": int(corner_faces.shape[0]),
        "face_corner_attrs": int(corner_attrs.shape[0]),
        "face_corner_attr_range": _tensor_range(corner_attrs),
        "local_to_global_local_q_max_abs_error": float(roundtrip_error.max().item()),
        "local_to_global_local_q_mean_abs_error": float(roundtrip_error.mean().item()),
        "face_policy": "all local decoded faces copied; no face deletion, ownership, welding, or overlap fusion",
        "pbr_policy": "local MeshWithVoxel.query_attrs at every face corner; independent corner attributes",
    }
    return ReturnedTilePatch(
        tile_id=int(tile_id),
        box=tuple(int(v) for v in box),
        vertices=global_corners.detach().cpu().to(torch.float32),
        faces=corner_faces.detach().cpu().to(torch.int32),
        vertex_attrs=corner_attrs.detach().cpu().to(torch.float32),
        stats=stats,
    )


def _stitch_tile_patches(
    patches: Sequence[ReturnedTilePatch],
    *,
    layout: Mapping[str, slice],
) -> Tuple[MeshWithVertexPbr, Dict[str, Any]]:
    if not patches:
        raise RuntimeError("cannot stitch an empty successful tile list")
    vertices: List[torch.Tensor] = []
    faces: List[torch.Tensor] = []
    attrs: List[torch.Tensor] = []
    offset = 0
    for patch in patches:
        vertices.append(patch.vertices)
        attrs.append(patch.vertex_attrs)
        faces.append(patch.faces.to(torch.long) + int(offset))
        offset += int(patch.vertices.shape[0])
    merged_vertices = torch.cat(vertices, dim=0)
    merged_faces = torch.cat(faces, dim=0).to(torch.int32)
    merged_attrs = torch.cat(attrs, dim=0).to(torch.float32)
    if merged_faces.numel() and int(merged_faces.max().item()) >= int(merged_vertices.shape[0]):
        raise RuntimeError("stitched face indices are out of bounds")
    mesh = MeshWithVertexPbr(
        vertices=merged_vertices,
        faces=merged_faces,
        vertex_attrs=merged_attrs,
        layout=dict(layout),
    )
    stats = {
        "successful_tile_patches": int(len(patches)),
        "stitched_vertices": int(merged_vertices.shape[0]),
        "stitched_faces": int(merged_faces.shape[0]),
        "stitched_face_corner_attrs": int(merged_attrs.shape[0]),
        "expected_vertices_from_direct_concat": int(sum(p.vertices.shape[0] for p in patches)),
        "expected_faces_from_direct_concat": int(sum(p.faces.shape[0] for p in patches)),
        "vertex_attrs_range": _tensor_range(merged_attrs),
        "operation": "direct concatenation of every successful tile patch",
        "welding": False,
        "deduplication": False,
        "remesh": False,
        "seam_repair": False,
        "overlap_fusion": False,
        "face_deletion": False,
    }
    return mesh, stats


def _stitch_tile_patches_nearest(
    patches: Sequence[ReturnedTilePatch],
    *,
    layout: Mapping[str, slice],
    global_camera: Mapping[str, float],
    face_chunk_size: int,
    weld_tolerance: float,
) -> Tuple[MeshWithVertexPbr, Dict[str, Any]]:
    """Remove tile overlap, then weld nearby mesh vertices.

    This is intentionally a small deterministic stitcher rather than a
    topology-aware remesher:

    1. A decoded face is owned by the successful tile whose projected tile
       center is nearest to that face centroid.  Thus a face in a 50% crop
       overlap is retained once, while faces outside the canonical camera
       projection stay with their source tile.
    2. The remaining face-corner vertices are merged by a spatial nearest
       voxel hash at ``weld_tolerance``.  Positions and PBR attributes are
       averaged inside each hash cell, and degenerate faces are removed.

    No face is selected by z-buffer visibility, and no surface is remeshed.
    The operation is deliberately simple and its counts are recorded so the
    result remains auditable against the original direct concatenation.
    """
    if not patches:
        raise RuntimeError("cannot nearest-stitch an empty tile list")
    if face_chunk_size <= 0:
        raise ValueError("stitch face chunk size must be positive")
    if weld_tolerance <= 0.0:
        raise ValueError("stitch weld tolerance must be positive")

    patch_ids = torch.tensor(
        [int(patch.tile_id) for patch in patches], dtype=torch.long
    )
    patch_boxes = torch.tensor(
        [list(patch.box) for patch in patches], dtype=torch.float32
    )
    patch_centers = torch.stack(
        (
            (patch_boxes[:, 0] + patch_boxes[:, 2]) * 0.5,
            (patch_boxes[:, 1] + patch_boxes[:, 3]) * 0.5,
        ),
        dim=1,
    )
    global_scale = float(global_camera["mesh_scale"])
    kept_vertices: List[torch.Tensor] = []
    kept_attrs: List[torch.Tensor] = []
    kept_faces: List[torch.Tensor] = []
    vertex_offset = 0
    total_input_faces = 0
    total_kept_faces = 0
    total_overlap_faces = 0
    total_invalid_projection_faces = 0
    per_tile: List[Dict[str, Any]] = []

    for patch in patches:
        vertices = patch.vertices.to(device="cpu", dtype=torch.float32)
        faces = patch.faces.to(device="cpu", dtype=torch.long)
        attrs = patch.vertex_attrs.to(device="cpu", dtype=torch.float32)
        face_count = int(faces.shape[0])
        tile_kept = 0
        tile_invalid = 0
        for face_start in range(0, face_count, int(face_chunk_size)):
            face_end = min(face_start + int(face_chunk_size), face_count)
            face_chunk = faces[face_start:face_end]
            corner_ids = face_chunk.reshape(-1)
            corner_vertices = vertices.index_select(0, corner_ids).reshape(-1, 3, 3)
            corner_attrs = attrs.index_select(0, corner_ids).reshape(
                -1, 3, attrs.shape[1]
            )
            centroids = corner_vertices.mean(dim=1)
            uv, _, finite = _project_global_q_to_4096(
                centroids * (2.0 * global_scale), global_camera=global_camera
            )
            inside = (
                finite[:, None]
                & (uv[:, None, 0] >= patch_boxes[None, :, 0])
                & (uv[:, None, 0] < patch_boxes[None, :, 2])
                & (uv[:, None, 1] >= patch_boxes[None, :, 1])
                & (uv[:, None, 1] < patch_boxes[None, :, 3])
            )
            distance2 = (
                uv[:, None, :] - patch_centers[None, :, :]
            ).square().sum(dim=2)
            distance2 = torch.where(
                inside, distance2, torch.full_like(distance2, float("inf"))
            )
            nearest_distance2, nearest_patch = distance2.min(dim=1)
            has_owner = finite & torch.isfinite(nearest_distance2)
            owner_ids = patch_ids.index_select(0, nearest_patch)
            keep = (~has_owner) | (owner_ids == int(patch.tile_id))
            invalid_count = int((~finite).sum().item())
            tile_invalid += invalid_count
            total_invalid_projection_faces += invalid_count
            kept_count = int(keep.sum().item())
            tile_kept += kept_count
            if kept_count == 0:
                continue
            selected_vertices = corner_vertices[keep].reshape(-1, 3)
            selected_attrs = corner_attrs[keep].reshape(-1, attrs.shape[1])
            selected_faces = torch.arange(
                selected_vertices.shape[0], dtype=torch.int32
            ).reshape(-1, 3) + int(vertex_offset)
            kept_vertices.append(selected_vertices)
            kept_attrs.append(selected_attrs)
            kept_faces.append(selected_faces)
            vertex_offset += int(selected_vertices.shape[0])

        overlap_faces = face_count - tile_kept
        total_input_faces += face_count
        total_kept_faces += tile_kept
        total_overlap_faces += overlap_faces
        per_tile.append(
            {
                "tile_id": int(patch.tile_id),
                "input_faces": int(face_count),
                "kept_faces": int(tile_kept),
                "overlap_faces_removed": int(overlap_faces),
                "invalid_projection_faces_kept": int(tile_invalid),
            }
        )

    if not kept_vertices:
        raise RuntimeError("nearest overlap stitch removed every face")
    raw_vertices = torch.cat(kept_vertices, dim=0)
    raw_attrs = torch.cat(kept_attrs, dim=0)
    raw_faces = torch.cat(kept_faces, dim=0).to(torch.long)
    del kept_vertices, kept_attrs, kept_faces

    # Spatial nearest-neighbor weld.  Quantized cells are a bounded and
    # memory-predictable approximation to a radius search; averaging keeps
    # geometry and continuous PBR values instead of snapping them to a grid.
    quantized = torch.round(raw_vertices / float(weld_tolerance)).to(torch.int64)
    _, inverse = torch.unique(
        quantized, dim=0, sorted=True, return_inverse=True
    )
    welded_count = int(inverse.max().item()) + 1 if inverse.numel() else 0
    welded_vertices = torch.zeros(
        (welded_count, 3), dtype=torch.float32
    )
    welded_attrs = torch.zeros(
        (welded_count, raw_attrs.shape[1]), dtype=torch.float32
    )
    counts = torch.bincount(
        inverse, minlength=welded_count
    ).to(torch.float32)
    welded_vertices.index_add_(0, inverse, raw_vertices)
    welded_attrs.index_add_(0, inverse, raw_attrs)
    welded_vertices = welded_vertices / counts[:, None].clamp_min(1.0)
    welded_attrs = welded_attrs / counts[:, None].clamp_min(1.0)
    welded_faces = inverse.index_select(0, raw_faces.reshape(-1)).reshape(-1, 3)
    nondegenerate = (
        (welded_faces[:, 0] != welded_faces[:, 1])
        & (welded_faces[:, 0] != welded_faces[:, 2])
        & (welded_faces[:, 1] != welded_faces[:, 2])
    )
    degenerate_faces_removed = int((~nondegenerate).sum().item())
    welded_faces = welded_faces[nondegenerate].to(torch.int32)
    if welded_faces.shape[0] == 0:
        raise RuntimeError("nearest overlap stitch produced only degenerate faces")

    mesh = MeshWithVertexPbr(
        vertices=welded_vertices,
        faces=welded_faces,
        vertex_attrs=welded_attrs,
        layout=dict(layout),
    )
    stats = {
        "operation": "projected tile-center nearest-owner overlap removal followed by spatial nearest-neighbor weld",
        "input_tiles": int(len(patches)),
        "input_faces": int(total_input_faces),
        "overlap_faces_removed": int(total_overlap_faces),
        "kept_faces_before_weld": int(total_kept_faces),
        "degenerate_faces_removed_after_weld": int(degenerate_faces_removed),
        "raw_face_corner_vertices": int(raw_vertices.shape[0]),
        "welded_vertices": int(welded_vertices.shape[0]),
        "welded_faces": int(welded_faces.shape[0]),
        "vertices_welded": int(raw_vertices.shape[0] - welded_vertices.shape[0]),
        "weld_tolerance_object": float(weld_tolerance),
        "invalid_projection_faces_kept": int(total_invalid_projection_faces),
        "face_chunk_size": int(face_chunk_size),
        "nearest_owner": "successful tile center in projected 4096 image space",
        "weld": "round(object_xyz / tolerance) spatial hash; average positions and PBR attrs per cell",
        "face_policy": "overlap owner deletion and degenerate-face deletion only; no remesh or seam optimization",
        "per_tile": per_tile,
        "vertex_attrs_range": _tensor_range(welded_attrs),
    }
    return mesh, stats


def _export_tiled_glb(
    patches: Sequence[ReturnedTilePatch],
    output_path: Path,
) -> Dict[str, Any]:
    """Export independent tile primitives with at least base-color vertex data.

    A direct concatenation can legitimately contain hundreds of millions of
    face-corner vertices.  Such a scene is still valid for the in-memory
    nvdiffrast path, but a single GLB cannot exceed the uint32 buffer-offset
    limit.  Small scenes use the requested monolithic path; large scenes are
    written as tile/part GLBs plus a JSON manifest, without welding or
    dropping any faces.
    """
    if not patches:
        raise RuntimeError("cannot export an empty tiled GLB")

    def _estimate_patch_bytes(patch: ReturnedTilePatch) -> int:
        # The GLB export stores float32 positions, uint8 RGBA colors, and
        # uint32 triangle indices.  This intentionally overestimates only the
        # data buffer and adds a fixed table/alignment margin below.
        return int(
            patch.vertices.numel() * 4
            + patch.vertices.shape[0] * 4
            + patch.faces.numel() * 4
        )

    estimated_total_bytes = int(sum(_estimate_patch_bytes(patch) for patch in patches))
    estimated_total_bytes += 16 * 1024 * 1024

    def _arrays_for_face_range(
        patch: ReturnedTilePatch,
        face_start: int,
        face_end: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        faces = patch.faces.to(device="cpu", dtype=torch.long)[face_start:face_end]
        # Preserve welded vertices within each exported part while making the
        # part self-contained.  Face-corner patches naturally keep all their
        # corners; a nearest-stitched mesh keeps its shared vertex topology.
        vertex_ids, inverse = torch.unique(
            faces.reshape(-1), sorted=True, return_inverse=True
        )
        vertices = patch.vertices.to(device="cpu", dtype=torch.float32).index_select(0, vertex_ids)
        attrs = patch.vertex_attrs.to(device="cpu", dtype=torch.float32).index_select(0, vertex_ids)
        local_faces = inverse.reshape(-1).to(torch.int32)
        base = attrs[:, 0:3].clamp(0.0, 1.0)
        alpha = (
            attrs[:, 5:6].clamp(0.0, 1.0)
            if attrs.shape[1] >= 6
            else torch.ones((attrs.shape[0], 1), dtype=torch.float32)
        )
        colors = torch.cat((base, alpha), dim=1)
        return (
            vertices.numpy(),
            local_faces.numpy(),
            (colors * 255.0).round().to(torch.uint8).numpy(),
        )

    def _export_scene_parts(
        parts: Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
        path: Path,
    ) -> None:
        scene = trimesh.Scene()
        for name, vertices, faces, colors in parts:
            tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            tri.visual.vertex_colors = colors
            scene.add_geometry(tri, geom_name=name, node_name=name)
        path.parent.mkdir(parents=True, exist_ok=True)
        scene.export(str(path), file_type="glb")

    primitive_stats = []

    def _patch_label(patch: ReturnedTilePatch) -> str:
        return "stitched_global" if int(patch.tile_id) < 0 else f"tile_{patch.tile_id:02d}"

    # Keep the original single-scene behavior when the data can fit in GLB.
    if estimated_total_bytes <= GLB_SAFE_BUFFER_BYTES:
        parts = []
        for patch in patches:
            vertices, faces, colors = _arrays_for_face_range(
                patch, 0, int(patch.faces.shape[0])
            )
            parts.append((_patch_label(patch), vertices, faces, colors))
            primitive_stats.append(
                {
                    "tile_id": int(patch.tile_id),
                    "part": 0,
                    "vertices": int(vertices.shape[0]),
                    "faces": int(faces.shape[0]),
                }
            )
        try:
            _export_scene_parts(parts, output_path)
        except OverflowError:
            # A conservative estimate can still miss exporter-specific
            # padding.  Fall through to the shard path if trimesh rejects it.
            output_path.unlink(missing_ok=True)
        else:
            return {
                "path": str(output_path),
                "format": "GLB",
                "primitive_count": int(len(primitive_stats)),
                "primitives": primitive_stats,
                "stored_channels": "geometry + base_color vertex colors (+ alpha); full PBR evaluation uses nvdiffrast",
                "process": False,
                "estimated_buffer_bytes": int(estimated_total_bytes),
                "sharded": False,
            }

    # Do not leave a stale monolithic artifact next to the manifest after a
    # previous run or after the conservative small-scene attempt overflowed.
    output_path.unlink(missing_ok=True)
    primitive_stats = []
    shard_dir = output_path.parent / f"{output_path.stem}_parts"
    manifest_path = output_path.parent / f"{output_path.stem}_manifest.json"
    shard_records = []
    for patch in patches:
        face_count = int(patch.faces.shape[0])
        part_count = max(1, (face_count + GLB_SHARD_FACE_LIMIT - 1) // GLB_SHARD_FACE_LIMIT)
        patch_label = _patch_label(patch)
        for part_index, face_start in enumerate(
            range(0, face_count, GLB_SHARD_FACE_LIMIT)
        ):
            face_end = min(face_start + GLB_SHARD_FACE_LIMIT, face_count)
            vertices, faces, colors = _arrays_for_face_range(
                patch, face_start, face_end
            )
            part_name = f"{patch_label}_part_{part_index:03d}.glb"
            part_path = shard_dir / part_name
            _export_scene_parts(
                [(f"{patch_label}_part_{part_index:03d}", vertices, faces, colors)],
                part_path,
            )
            record = {
                "tile_id": int(patch.tile_id),
                "part": int(part_index),
                "part_count": int(part_count),
                "face_start": int(face_start),
                "face_end": int(face_end),
                "vertices": int(vertices.shape[0]),
                "faces": int(faces.shape[0]),
                "path": str(part_path),
                "relative_path": str(part_path.relative_to(output_path.parent)),
            }
            shard_records.append(record)
            primitive_stats.append(record.copy())

    manifest = {
        "format": "GLB_SHARDED",
        "source_requested_path": str(output_path),
        "parts_directory": str(shard_dir),
        "combined_glb_limit_bytes": int(GLB_SAFE_BUFFER_BYTES),
        "shard_face_limit": int(GLB_SHARD_FACE_LIMIT),
        "estimated_combined_buffer_bytes": int(estimated_total_bytes),
        "successful_tile_count": int(len(patches)),
        "primitive_count": int(len(shard_records)),
        "primitives": shard_records,
        "stored_channels": "geometry + base_color vertex colors (+ alpha); full PBR evaluation uses nvdiffrast",
        "operation": "direct tile/face-part export; no welding, deduplication, remesh, seam repair, overlap fusion, or face deletion",
    }
    _atomic_json(manifest_path, manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return {
        "path": str(manifest_path),
        "manifest_path": str(manifest_path),
        "parts_directory": str(shard_dir),
        "format": "GLB_SHARDED",
        "primitive_count": int(len(primitive_stats)),
        "primitives": primitive_stats,
        "stored_channels": "geometry + base_color vertex colors (+ alpha); full PBR evaluation uses nvdiffrast",
        "process": False,
        "estimated_buffer_bytes": int(estimated_total_bytes),
        "sharded": True,
    }


def _render(
    mesh: Any,
    *,
    output_dir: Path,
    camera: Mapping[str, float],
    reference_image: Optional[Path],
    args: argparse.Namespace,
    envmap: Any,
) -> Dict[str, Any]:
    live_mesh = mesh.to("cuda")
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


def _metric_subset(row: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if row is None:
        return {"psnr_db": None, "ssim": None, "lpips": None}
    return {
        "psnr_db": row.get("psnr_db"),
        "ssim": row.get("ssim"),
        "lpips": row.get("lpips"),
    }


def _save_three_way_comparison(
    *,
    original_path: Path,
    baseline_path: Path,
    local_path: Path,
    baseline_metrics: Mapping[str, Any],
    local_metrics: Mapping[str, Any],
    output_path: Path,
) -> None:
    panel_size = 512
    header = 62
    entries = [
        (original_path, "input image", None),
        (baseline_path, "ordinary global 1024", baseline_metrics),
        (local_path, "stitched local tiles", local_metrics),
    ]
    canvas = Image.new("RGB", (panel_size * 3, panel_size + header), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (path, title, metrics) in enumerate(entries):
        with Image.open(path) as image:
            panel = ImageOps.contain(image.convert("RGB"), (panel_size, panel_size))
        x = index * panel_size + (panel_size - panel.width) // 2
        canvas.paste(panel, (x, header + (panel_size - panel.height) // 2))
        draw.text((index * panel_size + 8, 8), title, fill=(255, 255, 255))
        if metrics is not None:
            draw.text(
                (index * panel_size + 8, 28),
                f"PSNR {metrics.get('psnr_db')}  SSIM {metrics.get('ssim')}",
                fill=(220, 220, 220),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _save_tile_overlap_visualization(
    *,
    image_4096: Image.Image,
    boxes: Sequence[Sequence[int]],
    successful_ids: Sequence[int],
    output_path: Path,
) -> Dict[str, Any]:
    scale = 4
    width = CANONICAL_IMAGE_SIZE // scale
    height = CANONICAL_IMAGE_SIZE // scale
    base = image_4096.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    coverage = np.zeros((height, width), dtype=np.int32)
    success = set(int(v) for v in successful_ids)
    palette = np.asarray(
        [
            [239, 83, 80], [255, 167, 38], [253, 216, 53], [102, 187, 106],
            [38, 198, 218], [66, 165, 245], [126, 87, 194], [236, 64, 122],
        ],
        dtype=np.uint8,
    )
    color = np.zeros((height, width, 3), dtype=np.uint8)
    for tile_id, box in enumerate(boxes):
        if tile_id not in success:
            continue
        x0, y0, x1, y1 = [int(v) // scale for v in box]
        coverage[y0:y1, x0:x1] += 1
        color[y0:y1, x0:x1] = palette[tile_id % len(palette)]
    overlap_alpha = np.clip(0.18 + 0.16 * coverage[..., None], 0.0, 0.75)
    overlay = (
        np.asarray(base, dtype=np.float32) * (1.0 - overlap_alpha)
        + color.astype(np.float32) * overlap_alpha
    ).clip(0, 255).astype(np.uint8)
    canvas = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    for tile_id, box in enumerate(boxes):
        x0, y0, x1, y1 = [int(v) // scale for v in box]
        outline = tuple(int(v) for v in palette[tile_id % len(palette)]) if tile_id in success else (100, 100, 100)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=outline, width=2)
        draw.text((x0 + 5, y0 + 5), f"{tile_id:02d}", fill=outline)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {
        "path": str(output_path),
        "resolution": [width, height],
        "successful_tile_ids": sorted(success),
        "coverage_min": int(coverage.min()),
        "coverage_max": int(coverage.max()),
        "overlap_pixels": int((coverage > 1).sum()),
        "visualization": "successful tile rectangles colored by tile id; overlap opacity encodes coverage count",
    }


def _render_multiview_comparison(
    baseline_mesh: MeshWithVoxel,
    stitched_mesh: MeshWithVertexPbr,
    *,
    output_dir: Path,
    camera: Mapping[str, float],
    args: argparse.Namespace,
    envmap: Any,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed = [
        ("front", 0.0, 0.0),
        ("right", 90.0, 0.0),
        ("left", -90.0, 0.0),
        ("back", 180.0, 0.0),
        ("top", 0.0, 75.0),
        ("bottom", 0.0, -75.0),
    ]
    turntable_count = int(args.multiview_turntable_frames)
    turntable = [
        (f"turntable_{index:02d}", 360.0 * index / turntable_count, 0.0)
        for index in range(turntable_count)
    ]
    specs = fixed + turntable
    device = torch.device("cuda")
    radius = float(camera["distance"]) * float(args.multiview_radius_scale)
    fov = torch.tensor(float(camera["camera_angle_x"]), device=device)
    intrinsic = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
    target = torch.zeros(3, device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    extrinsics = []
    intrinsics = []
    labels = []
    for label, yaw_degrees, pitch_degrees in specs:
        yaw = math.radians(yaw_degrees)
        pitch = math.radians(pitch_degrees)
        direction = torch.tensor(
            [
                math.sin(yaw) * math.cos(pitch),
                math.sin(pitch),
                math.cos(yaw) * math.cos(pitch),
            ],
            device=device,
            dtype=torch.float32,
        )
        position = target + direction * radius
        extrinsics.append(utils3d.torch.extrinsics_look_at(position, target, up))
        intrinsics.append(intrinsic)
        labels.append(f"{label} yaw={yaw_degrees:g} pitch={pitch_degrees:g}")
    render_options = {
        "resolution": int(args.multiview_resolution),
        "near": max(0.01, radius - 2.0),
        "far": radius + 10.0,
        "ssaa": int(args.multiview_ssaa),
        "peel_layers": int(args.multiview_peel_layers),
        "face_chunk_size": int(args.render_face_chunk_size),
    }
    renderer = render_utils.get_renderer(baseline_mesh, **render_options)

    def render_mesh(mesh: Any) -> List[np.ndarray]:
        live = mesh.to(device)
        frames = render_utils.render_frames(
            live,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            options=render_options,
            verbose=True,
            renderer=renderer,
            envmap=envmap,
            use_envmap_bg=bool(args.use_envmap_bg),
        ).get("shaded")
        del live
        _empty_cuda_cache()
        if frames is None or len(frames) != len(labels):
            raise RuntimeError("nvdiffrast multi-view render returned incomplete frames")
        return [np.asarray(frame) for frame in frames]

    baseline_frames = render_mesh(baseline_mesh)
    local_frames = render_mesh(stitched_mesh)
    baseline_paths: List[Path] = []
    local_paths: List[Path] = []
    pair_paths: List[Path] = []
    pair_metrics: List[Dict[str, Any]] = []
    all_pair_images: List[Image.Image] = []
    turntable_pair_images: List[Image.Image] = []
    for index, (baseline_frame, local_frame, label) in enumerate(
        zip(baseline_frames, local_frames, labels)
    ):
        baseline_path = output_dir / f"view_{index:03d}_baseline.png"
        local_path = output_dir / f"view_{index:03d}_stitched_local.png"
        pair_path = output_dir / f"view_{index:03d}_baseline_vs_local.png"
        baseline_image = Image.fromarray(baseline_frame).convert("RGB")
        local_image = Image.fromarray(local_frame).convert("RGB")
        baseline_image.save(baseline_path)
        local_image.save(local_path)
        pair = Image.new("RGB", (baseline_image.width * 2, baseline_image.height))
        pair.paste(baseline_image, (0, 0))
        pair.paste(local_image, (baseline_image.width, 0))
        pair.save(pair_path)
        baseline_paths.append(baseline_path)
        local_paths.append(local_path)
        pair_paths.append(pair_path)
        all_pair_images.append(pair.copy())
        if index >= len(fixed):
            turntable_pair_images.append(pair.copy())
        ref_tensor = image_to_tensor(baseline_image, baseline_image.size)
        local_tensor = image_to_tensor(local_image, local_image.size)
        pair_metrics.append(
            {
                "view": int(index),
                "label": label,
                "baseline_vs_stitched_psnr_db": psnr_metric(ref_tensor, local_tensor),
                "baseline_vs_stitched_ssim": ssim_metric(ref_tensor, local_tensor),
            }
        )
    def save_sheet(paths: Sequence[Path], path: Path, labels_: Sequence[str]) -> None:
        panel = 384
        header = 42
        columns = 3
        rows = (len(paths) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * panel, rows * (panel + header)), "black")
        draw = ImageDraw.Draw(sheet)
        for index, (frame_path, label) in enumerate(zip(paths, labels_)):
            with Image.open(frame_path) as source:
                image = ImageOps.contain(source.convert("RGB"), (panel, panel))
            x = (index % columns) * panel
            y = (index // columns) * (panel + header)
            sheet.paste(image, (x + (panel - image.width) // 2, y + header))
            draw.text((x + 5, y + 10), label, fill=(255, 255, 255))
        sheet.save(path)

    baseline_sheet = output_dir / "baseline_multiview_sheet.png"
    local_sheet = output_dir / "stitched_local_multiview_sheet.png"
    comparison_sheet = output_dir / "baseline_vs_stitched_multiview_sheet.png"
    save_sheet(baseline_paths, baseline_sheet, labels)
    save_sheet(local_paths, local_sheet, labels)
    save_sheet(pair_paths, comparison_sheet, labels)
    gif_path = output_dir / "turntable_baseline_vs_stitched.gif"
    if turntable_pair_images:
        turntable_pair_images[0].save(
            gif_path,
            save_all=True,
            append_images=turntable_pair_images[1:],
            duration=100,
            loop=0,
        )
    _atomic_json(output_dir / "multiview_metrics.json", {"views": pair_metrics})
    return {
        "enabled": True,
        "renderer": "pixal3d.utils.render_utils -> PbrMeshRenderer -> nvdiffrast",
        "camera_policy": "fixed global camera trajectory shared byte-for-byte by baseline and stitched local",
        "resolution": int(args.multiview_resolution),
        "ssaa": int(args.multiview_ssaa),
        "peel_layers": int(args.multiview_peel_layers),
        "face_chunk_size": int(args.render_face_chunk_size),
        "background_envmap": str(args.envmap),
        "fixed_views": [label for label, _, _ in fixed],
        "turntable_frames": int(turntable_count),
        "baseline_frame_pngs": [str(path) for path in baseline_paths],
        "stitched_local_frame_pngs": [str(path) for path in local_paths],
        "pair_frame_pngs": [str(path) for path in pair_paths],
        "baseline_sheet_png": str(baseline_sheet),
        "stitched_local_sheet_png": str(local_sheet),
        "comparison_sheet_png": str(comparison_sheet),
        "turntable_gif": str(gif_path) if gif_path.is_file() else None,
        "metrics_json": str(output_dir / "multiview_metrics.json"),
        "pair_metrics": pair_metrics,
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
            torch.tensor([0.0 - float(extend_pixel), GLOBAL_IMAGE_SIZE - 1.0 + float(extend_pixel)]),
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
    model = load_moge_model(
        device="cuda", model_name=moge_model_path
    ) if moge_model_path else load_moge_model(device="cuda")
    try:
        params = get_camera_params_wild_moge(
            str(temporary), model, device="cuda", mesh_scale=float(mesh_scale),
            extend_pixel=int(extend_pixel), image_resolution=GLOBAL_IMAGE_SIZE,
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


def _write_tile_summary(tile_dir: Path, record: Mapping[str, Any]) -> None:
    _atomic_json(tile_dir / "summary.json", record)


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(int(args.cuda_device))
    device = torch.device("cuda")
    print(
        f"[cuda] requested/current index={int(args.cuda_device)} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())}"
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_image = Image.open(args.image).convert("RGB")
    source_image.save(output_dir / "input_original.png")

    pipeline = init_pipeline(
        args.model_path, device="cuda", low_vram=bool(args.low_vram)
    )
    canonical = pipeline.preprocess_canonical_images(Image.open(args.image))
    image_4096: Image.Image = canonical["image_4096"]
    image_1024: Image.Image = canonical["image_1024"]
    image_512: Image.Image = canonical["image_512"]
    image_4096.save(output_dir / "canonical_4096.png")
    image_1024.save(output_dir / "canonical_1024.png")
    image_512.save(output_dir / "canonical_512.png")
    canonical["foreground_mask_4096"].save(output_dir / "canonical_foreground_mask_4096.png")
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
        raise RuntimeError(f"global baseline returned {len(baseline_output)} meshes, expected one")
    baseline_live = _validate_mesh(baseline_output[0], "global ordinary Pixal3D-1024 baseline")
    baseline_shape_slat, baseline_texture_slat, decoded_resolution = baseline_latents
    if int(decoded_resolution) != OVOXEL_RESOLUTION:
        raise RuntimeError(f"baseline decoder resolution is {decoded_resolution}, expected 1024")
    baseline_dir = output_dir / "global_baseline_1024"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    envmap = load_envmap(str(args.envmap), device="cuda") if args.render else None
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
        "shape_slat_tokens": int(baseline_shape_slat.feats.shape[0]),
        "texture_slat_tokens": int(baseline_texture_slat.feats.shape[0]),
        "shape_slat_range": _tensor_range(baseline_shape_slat.feats),
        "texture_slat_range": _tensor_range(baseline_texture_slat.feats),
        "baseline_pbr_attr_range": _tensor_range(baseline_mesh.attrs),
        "render": baseline_render,
    }
    _atomic_json(baseline_dir / "summary.json", baseline_summary)
    if args.save_mesh_checkpoints:
        torch.save(
            {
                "mesh": baseline_mesh,
                "shape_slat": baseline_shape_slat.to("cpu"),
                "texture_slat": baseline_texture_slat.to("cpu"),
                "resolution": int(decoded_resolution),
            },
            baseline_dir / "baseline_mesh_and_slats.pt",
        )
    del baseline_output, baseline_live, baseline_latents
    del baseline_shape_slat, baseline_texture_slat
    _empty_cuda_cache()

    print("[global-analysis] projecting triangle corners for tile bbox selection")
    face_min, face_max, face_finite = _project_face_bboxes(
        baseline_mesh.vertices,
        baseline_mesh.faces,
        mesh_scale=float(global_camera["mesh_scale"]),
        global_camera=global_camera,
        chunk_size=int(args.face_projection_chunk_size),
    )
    global_attr_field = _make_attribute_query_mesh(baseline_mesh, device)

    print(f"[encoder] loading shape encoder: {args.shape_encoder}")
    shape_encoder = pixal3d_models.from_pretrained(str(Path(args.shape_encoder).expanduser())).eval()
    print(f"[encoder] loading PBR encoder: {args.pbr_encoder}")
    pbr_encoder = pixal3d_models.from_pretrained(str(Path(args.pbr_encoder).expanduser())).eval()
    if not args.low_vram:
        shape_encoder.to(device)
        pbr_encoder.to(device)

    boxes = _tile_layout()
    requested_ids = _parse_tile_ids(args.tile_ids)
    if requested_ids is not None:
        invalid = sorted(tile_id for tile_id in requested_ids if tile_id not in range(len(boxes)))
        if invalid:
            raise ValueError(f"invalid tile ids {invalid}; valid ids are 0..{len(boxes)-1}")

    tile_records: List[Dict[str, Any]] = []
    returned_patches: List[ReturnedTilePatch] = []
    attempted_tiles = 0
    for tile_id, box in enumerate(boxes):
        if requested_ids is not None and tile_id not in requested_ids:
            continue
        tile_dir = output_dir / "tiles" / f"tile_{tile_id:02d}"
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_image = image_4096.crop(box).convert("RGB")
        tile_image.save(tile_dir / "tile_reference.png")
        transform = _derive_tile_camera(
            tile_id=tile_id,
            box=box,
            global_camera=global_camera,
            extend_pixel=int(args.extend_pixel),
        )
        _atomic_json(tile_dir / "tile_camera.json", asdict(transform))
        selected_face_ids = _tile_face_ids_from_bbox(face_min, face_max, face_finite, box)
        selected_face_count = int(selected_face_ids.shape[0])
        print(f"[tile {tile_id:02d}] bbox_faces={selected_face_count:,} box={box}")
        if selected_face_count == 0:
            record = {
                "status": "skipped",
                "tile_id": int(tile_id),
                "box": list(box),
                "projected_bbox_faces": 0,
                "reason": "no triangle projection bbox intersects tile",
            }
            tile_records.append(record)
            _write_tile_summary(tile_dir, record)
            continue
        if args.max_tiles is not None and attempted_tiles >= int(args.max_tiles):
            break
        attempted_tiles += 1
        started = time.perf_counter()
        geometry: Optional[LocalGeometry] = None
        alignment_stats: Optional[Dict[str, Any]] = None
        try:
            geometry = _prepare_tile_geometry(
                global_vertices=baseline_mesh.vertices,
                global_faces=baseline_mesh.faces,
                global_face_min=face_min,
                global_face_max=face_max,
                global_face_finite=face_finite,
                global_camera=global_camera,
                transform=transform,
            )
            if geometry.stats["global_local_global_q_max_abs_error"] > float(args.roundtrip_tolerance):
                raise RuntimeError(
                    "global/local camera round-trip exceeded tolerance: "
                    f"{geometry.stats['global_local_global_q_max_abs_error']:.3e}"
                )
            local_attrs, material_stats = _resample_local_attrs_from_global(
                geometry=geometry,
                global_attr_field=global_attr_field,
                global_camera=global_camera,
                transform=transform,
                query_chunk_size=int(args.material_query_chunk_size),
                face_chunk_size=int(args.material_face_chunk_size),
            )
            reference_mesh = _make_local_reference_mesh(geometry, local_attrs, baseline_mesh)
            shape_reference, shape_encoder_stats = _encode_local_shape(
                encoder=shape_encoder,
                local_coords=geometry.coords,
                local_dual_vertices=geometry.dual_vertices,
                local_intersected=geometry.intersected,
                device=device,
                low_vram=bool(args.low_vram),
            )
            texture_reference, pbr_encoder_stats = _encode_local_pbr(
                encoder=pbr_encoder,
                coords=geometry.coords,
                attrs=local_attrs,
                device=device,
                low_vram=bool(args.low_vram),
            )
            alignment_stats = _latent_support_diagnostics(shape_reference, texture_reference)
            if not alignment_stats["coordinates_exactly_equal"]:
                raise RuntimeError(
                    "shape/PBR encoder output coordinates differ; "
                    + json.dumps(alignment_stats, ensure_ascii=False)
                )
            if int(shape_reference.feats.shape[0]) > int(args.max_num_tokens):
                raise RuntimeError(
                    f"local latent has {shape_reference.feats.shape[0]:,} tokens, "
                    f"exceeding --max-num-tokens={int(args.max_num_tokens):,}"
                )

            reference_decode_started = time.perf_counter()
            with torch.no_grad():
                reference_decoded = pipeline.decode_latent(
                    shape_reference,
                    texture_reference,
                    OVOXEL_RESOLUTION,
                )
            _sync_cuda()
            reference_decode_seconds = time.perf_counter() - reference_decode_started
            if len(reference_decoded) != 1:
                raise RuntimeError("reference SLat decode returned more than one mesh")
            reference_slat_mesh = _validate_mesh(
                reference_decoded[0], f"tile {tile_id:02d} reference SLat decode"
            )

            tile_seed = int(args.seed)
            flow_latents = _run_native_tile_flow(
                pipeline=pipeline,
                tile_image=tile_image,
                transform=transform,
                shape_reference=shape_reference,
                texture_reference=texture_reference,
                shape_params=shape_params,
                texture_params=texture_params,
                seed=tile_seed,
                tile_id=tile_id,
            )
            decode_started = time.perf_counter()
            with torch.no_grad():
                decoded = pipeline.decode_latent(
                    flow_latents.shape_denorm,
                    flow_latents.texture_denorm,
                    OVOXEL_RESOLUTION,
                )
            _sync_cuda()
            decode_seconds = time.perf_counter() - decode_started
            if len(decoded) != 1:
                raise RuntimeError("tile flow decode returned more than one mesh")
            flow_mesh = _validate_mesh(decoded[0], f"tile {tile_id:02d} flow decode")
            returned_patch = _local_mesh_to_global_patch(
                tile_id=tile_id,
                box=box,
                local_mesh=flow_mesh,
                global_camera=global_camera,
                transform=transform,
                query_chunk_size=int(args.material_query_chunk_size),
            )

            local_surface = _mesh_surface_similarity(
                reference_mesh,
                flow_mesh,
                samples=int(args.surface_samples),
                chunk_size=int(args.nearest_chunk_size),
                seed=int(args.seed) + tile_id * 17,
                device=device,
            )
            local_support = _support_similarity(reference_mesh.coords, flow_mesh.coords)
            tile_render_stats: Optional[Dict[str, Any]] = None
            if args.render:
                tile_camera = {
                    "camera_angle_x": float(transform.camera_angle_x),
                    "distance": float(transform.distance),
                    "mesh_scale": float(transform.mesh_scale),
                }
                reference_render = _render(
                    reference_mesh,
                    output_dir=tile_dir / "local_reference_mesh_render",
                    camera=tile_camera,
                    reference_image=tile_dir / "tile_reference.png",
                    args=args,
                    envmap=envmap,
                )
                reference_decode_render = _render(
                    reference_slat_mesh,
                    output_dir=tile_dir / "reference_slat_decode_render",
                    camera=tile_camera,
                    reference_image=tile_dir / "tile_reference.png",
                    args=args,
                    envmap=envmap,
                )
                flow_render = _render(
                    flow_mesh,
                    output_dir=tile_dir / "tile_flow_local_mesh_render",
                    camera=tile_camera,
                    reference_image=tile_dir / "tile_reference.png",
                    args=args,
                    envmap=envmap,
                )
                tile_render_stats = {
                    "local_reference_mesh": reference_render,
                    "reference_slat_decode": reference_decode_render,
                    "tile_flow_local_mesh": flow_render,
                    "tile_original": str(tile_dir / "tile_reference.png"),
                }

            if args.save_mesh_checkpoints:
                torch.save(
                    {
                        "geometry": geometry,
                        "local_attrs": local_attrs,
                        "shape_reference": shape_reference.to("cpu"),
                        "texture_reference": texture_reference.to("cpu"),
                        "flow_shape_slat": flow_latents.shape_denorm.to("cpu"),
                        "flow_texture_slat": flow_latents.texture_denorm.to("cpu"),
                    },
                    tile_dir / "local_geometry_attrs_slats.pt",
                )
                torch.save(reference_mesh, tile_dir / "local_reference_mesh.pt")
                torch.save(reference_slat_mesh, tile_dir / "reference_slat_decode_mesh.pt")
                torch.save(flow_mesh, tile_dir / "tile_flow_local_mesh.pt")

            record = {
                "status": "success",
                "tile_id": int(tile_id),
                "box": list(box),
                "tile_seconds": float(time.perf_counter() - started),
                "projected_bbox_faces": selected_face_count,
                "geometry": geometry.stats,
                "material_resampling": material_stats,
                "shape_encoder": shape_encoder_stats,
                "pbr_encoder": pbr_encoder_stats,
                "latent_support": alignment_stats,
                "reference_slat_decode_seconds": float(reference_decode_seconds),
                "flow": flow_latents.stats,
                "flow_decode_seconds": float(decode_seconds),
                "flow_mesh_pbr_range": _tensor_range(flow_mesh.attrs),
                "local_surface_similarity": local_surface,
                "local_support_similarity": local_support,
                "returned_global_patch": returned_patch.stats,
                "tile_renders": tile_render_stats,
            }
            tile_records.append(record)
            _write_tile_summary(tile_dir, record)
            # A patch becomes part of the final direct concatenation only
            # after the complete tile flow, decode, PBR query, metrics,
            # optional render/checkpoint path, and tile summary have succeeded.
            returned_patches.append(returned_patch)
            print(
                f"[tile {tile_id:02d}] success "
                f"faces={selected_face_count:,} local_C64={alignment_stats['shape_tokens']:,} "
                f"flow_faces={returned_patch.faces.shape[0]:,} "
                f"seconds={record['tile_seconds']:.2f}"
            )
            del (
                geometry,
                local_attrs,
                reference_mesh,
                reference_slat_mesh,
                flow_mesh,
                decoded,
                reference_decoded,
                shape_reference,
                texture_reference,
                flow_latents,
            )
            _empty_cuda_cache()
        except Exception as exc:
            # Keep error isolation meaningful in low-VRAM mode: a failed tile
            # must not retain its partially decoded CUDA tensors into the next
            # tile.  The successful patch list is CPU-only and is untouched.
            geometry = None
            local_attrs = None
            reference_mesh = None
            reference_slat_mesh = None
            flow_mesh = None
            decoded = None
            reference_decoded = None
            shape_reference = None
            texture_reference = None
            flow_latents = None
            record = {
                "status": "failed",
                "tile_id": int(tile_id),
                "box": list(box),
                "projected_bbox_faces": selected_face_count,
                "tile_seconds": float(time.perf_counter() - started),
                "geometry": None if geometry is None else geometry.stats,
                "latent_support": alignment_stats,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            tile_records.append(record)
            _write_tile_summary(tile_dir, record)
            print(f"[tile {tile_id:02d}] FAILED: {record['reason']}")
            _empty_cuda_cache()

    del shape_encoder, pbr_encoder, global_attr_field
    _empty_cuda_cache()
    successful_rows = [row for row in tile_records if row["status"] == "success"]
    failed_rows = [row for row in tile_records if row["status"] == "failed"]
    skipped_rows = [row for row in tile_records if row["status"] == "skipped"]

    final_render: Dict[str, Any] = {}
    stitched_mesh: Optional[MeshWithVertexPbr] = None
    if returned_patches:
        stitched_mesh, stitch_stats = _stitch_tile_patches_nearest(
            returned_patches,
            layout=baseline_mesh.layout,
            global_camera=global_camera,
            face_chunk_size=int(args.face_projection_chunk_size),
            weld_tolerance=float(args.stitch_tolerance),
        )
        stitched_dir = output_dir / "stitched_global_tiled_mesh"
        stitched_dir.mkdir(parents=True, exist_ok=True)
        stitched_export_patch = ReturnedTilePatch(
            tile_id=-1,
            box=(0, 0, CANONICAL_IMAGE_SIZE, CANONICAL_IMAGE_SIZE),
            vertices=stitched_mesh.vertices,
            faces=stitched_mesh.faces,
            vertex_attrs=stitched_mesh.vertex_attrs,
            stats=stitch_stats,
        )
        glb_stats = (
            _export_tiled_glb(
                [stitched_export_patch],
                stitched_dir / "stitched_local_tiles.glb",
            )
            if args.export_glb
            else {"enabled": False}
        )
        overlap_stats = _save_tile_overlap_visualization(
            image_4096=image_4096,
            boxes=boxes,
            successful_ids=[patch.tile_id for patch in returned_patches],
            output_path=stitched_dir / "tile_overlap_coverage.png",
        )
        if args.save_mesh_checkpoints:
            torch.save(
                {
                    "mesh": stitched_mesh,
                    "patches": returned_patches,
                    "stitch_stats": stitch_stats,
                },
                stitched_dir / "stitched_global_tiled_mesh.pt",
            )
        if args.render:
            stitched_render = _render(
                stitched_mesh,
                output_dir=stitched_dir / "aligned_eval",
                camera=global_camera,
                reference_image=output_dir / "canonical_1024.png",
                args=args,
                envmap=envmap,
            )
            stitched_baseline_render = None
            if baseline_render is not None:
                stitched_baseline_render = _render(
                    stitched_mesh,
                    output_dir=stitched_dir / "against_global_baseline",
                    camera=global_camera,
                    reference_image=Path(baseline_render["render_png"]),
                    args=args,
                    envmap=envmap,
                )
            comparison_path = output_dir / "comparison_input_global_baseline_stitched_local.png"
            _save_three_way_comparison(
                original_path=output_dir / "input_original.png",
                baseline_path=Path(str(baseline_render["render_png"])),
                local_path=Path(str(stitched_render["render_png"])),
                baseline_metrics=_metric_subset(baseline_render),
                local_metrics=_metric_subset(stitched_render),
                output_path=comparison_path,
            )
            multiview = (
                _render_multiview_comparison(
                    baseline_mesh,
                    stitched_mesh,
                    output_dir=stitched_dir / "multiview",
                    camera=global_camera,
                    args=args,
                    envmap=envmap,
                )
                if args.render_multiview
                else {"enabled": False}
            )
            final_render = {
                "input_reference": str(output_dir / "canonical_1024.png"),
                "baseline_against_input": _metric_subset(baseline_render),
                "stitched_local_against_input": _metric_subset(stitched_render),
                "stitched_local_against_baseline": _metric_subset(stitched_baseline_render),
                "baseline_render": baseline_render,
                "stitched_local_render": stitched_render,
                "stitched_local_against_baseline_render": stitched_baseline_render,
                "three_way_comparison_png": str(comparison_path),
                "overlap_visualization": overlap_stats,
                "multiview": multiview,
                "same_render_settings": {
                    "camera": global_camera,
                    "envmap": str(args.envmap),
                    "resolution": int(args.render_resolution),
                    "ssaa": int(args.render_ssaa),
                    "peel_layers": int(args.render_peel_layers),
                    "face_chunk_size": int(args.render_face_chunk_size),
                    "use_envmap_bg": bool(args.use_envmap_bg),
                },
            }
        else:
            final_render = {
                "enabled": False,
                "overlap_visualization": overlap_stats,
            }
        _atomic_json(
            stitched_dir / "summary.json",
            {
                "stitch": stitch_stats,
                "glb": glb_stats,
                "render": final_render,
            },
        )

    aggregate: Dict[str, Any] = {}
    if successful_rows:
        aggregate = {
            "mean_local_chamfer_l1_in_voxels": float(
                np.mean([row["local_surface_similarity"]["chamfer_l1_in_local_voxels"] for row in successful_rows])
            ),
            "mean_local_support_iou": float(
                np.mean([row["local_support_similarity"]["iou"] for row in successful_rows])
            ),
            "mean_tile_flow_seconds": float(np.mean([row["flow"]["shape_flow_seconds"] + row["flow"]["texture_flow_seconds"] for row in successful_rows])),
        }
    summary = {
        "format": "pixal3d_local_c1024_dual_grid_reference_flow_stitched_v1",
        "image": str(Path(args.image).expanduser().resolve()),
        "cuda_device": int(args.cuda_device),
        "global_camera": global_camera,
        "global_baseline_1024": baseline_summary,
        "tile_policy": {
            "canonical_image_size": CANONICAL_IMAGE_SIZE,
            "tile_size": TILE_SIZE,
            "tile_stride": TILE_STRIDE,
            "tile_count": len(boxes),
            "halo": False,
            "overlap": "natural 50% overlap from adjacent 1024 crops",
            "face_selection": "projected triangle bbox intersects tile rectangle",
            "material": "local dual-grid support -> closest local triangle point -> exact inverse camera -> baseline MeshWithVoxel.query_attrs",
            "local_geometry": "fresh mesh_to_flexible_dual_grid at 1024 for every tile",
            "latent_support": "shape and PBR encoder coordinates must be exactly equal; mismatch fails the tile",
            "flow": "native FlowEulerSampler x0/epsilon convention, same CLI sampler parameters and seed for every tile",
            "return_to_global": "all local face corners query local MeshWithVoxel attrs, then exact local-to-global camera inverse",
            "stitch": "projected tile-center nearest-owner overlap deletion followed by spatial nearest-neighbor vertex weld; degenerate faces are removed",
        },
        "compatibility_parameters": {
            "min_tile_ovoxels": int(args.min_tile_ovoxels),
            "min_tile_ovoxels_behavior": "retained CLI parameter; intentionally not used for global O-Voxel remapping or tile selection",
        },
        "successful_tiles": int(len(successful_rows)),
        "failed_tiles": int(len(failed_rows)),
        "skipped_tiles": int(len(skipped_rows)),
        "aggregate": aggregate,
        "final_stitched": final_render,
        "tiles": tile_records,
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(
        f"[done] success={len(successful_rows)} failed={len(failed_rows)} "
        f"skipped={len(skipped_rows)} summary={output_dir / 'summary.json'}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--moge-model-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument(
        "--low-vram",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="move large flow/encoder/decoder models on demand",
    )
    parser.add_argument("--shape-encoder", default=str(DEFAULT_ENCODER_ROOT / "shape_enc_next_dc_f16c32_fp16"))
    parser.add_argument("--pbr-encoder", default=str(DEFAULT_ENCODER_ROOT / "tex_enc_next_dc_f16c32_fp16"))
    parser.add_argument("--tile-ids", default=None)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument(
        "--min-tile-ovoxels",
        type=int,
        default=1001,
        help="retained compatibility parameter; local route uses projected triangle bboxes instead",
    )
    parser.add_argument("--max-num-tokens", type=int, default=1_000_000)
    parser.add_argument("--roundtrip-tolerance", type=float, default=2e-5)
    parser.add_argument("--face-projection-chunk-size", type=int, default=250_000)
    parser.add_argument("--material-query-chunk-size", type=int, default=65_536)
    parser.add_argument("--material-face-chunk-size", type=int, default=16_384)
    parser.add_argument("--surface-samples", type=int, default=10_000)
    parser.add_argument("--nearest-chunk-size", type=int, default=1_024)
    parser.add_argument(
        "--stitch-tolerance",
        type=float,
        default=1.0 / OVOXEL_RESOLUTION,
        help="object-space radius of the simple nearest-neighbor vertex weld",
    )
    parser.add_argument("--save-mesh-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--export-glb", action=argparse.BooleanOptionalAction, default=True)

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

    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--metric-resolution", type=int, default=512)
    parser.add_argument("--render-ssaa", type=int, default=2)
    parser.add_argument("--render-peel-layers", type=int, default=8)
    parser.add_argument("--render-face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--use-envmap-bg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--skip-lpips",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="skip optional LPIPS metrics even when the package is installed",
    )
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--render-multiview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multiview-resolution", type=int, default=512)
    parser.add_argument("--multiview-ssaa", type=int, default=1)
    parser.add_argument("--multiview-peel-layers", type=int, default=4)
    parser.add_argument("--multiview-radius-scale", type=float, default=1.0)
    parser.add_argument("--multiview-turntable-frames", type=int, default=24)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if int(args.cuda_device) < 0:
        raise ValueError("--cuda-device must be non-negative")
    if int(args.min_tile_ovoxels) < 0:
        raise ValueError("--min-tile-ovoxels must be non-negative")
    if args.max_tiles is not None and int(args.max_tiles) < 1:
        raise ValueError("--max-tiles must be positive")
    if int(args.max_num_tokens) < 1:
        raise ValueError("--max-num-tokens must be positive")
    if float(args.roundtrip_tolerance) <= 0.0:
        raise ValueError("--roundtrip-tolerance must be positive")
    if (
        int(args.face_projection_chunk_size) < 1
        or int(args.material_query_chunk_size) < 1
        or int(args.material_face_chunk_size) < 1
        or int(args.surface_samples) < 1
        or int(args.nearest_chunk_size) < 1
        or float(args.stitch_tolerance) <= 0.0
    ):
        raise ValueError("projection/material/sample/chunk sizes must be positive")
    if (
        int(args.render_resolution) < 1
        or int(args.metric_resolution) < 1
        or int(args.render_ssaa) < 1
        or int(args.render_peel_layers) < 1
        or int(args.render_face_chunk_size) < 0
        or int(args.multiview_resolution) < 1
        or int(args.multiview_ssaa) < 1
        or int(args.multiview_peel_layers) < 1
        or int(args.multiview_turntable_frames) < 1
        or float(args.multiview_radius_scale) <= 0.0
    ):
        raise ValueError("invalid render or multiview configuration")
    for encoder_path in (args.shape_encoder, args.pbr_encoder):
        base = Path(encoder_path).expanduser()
        if not Path(f"{base}.json").is_file() or not Path(f"{base}.safetensors").is_file():
            raise FileNotFoundError(f"encoder checkpoint pair not found for base path {base}")
    if not args.skip_lpips and importlib.util.find_spec("lpips") is None:
        print("[metrics] lpips package unavailable; continuing without LPIPS")
        args.skip_lpips = True
    run(args)


if __name__ == "__main__":
    main()
