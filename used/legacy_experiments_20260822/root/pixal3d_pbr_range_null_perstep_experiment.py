#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training-free PBR field range--null projection experiment.

This is an independent experiment built on the successful fixed-shape
cross-tile texture-flow route.  The official route is kept intact:

    x_t -> official prediction -> pred_x0 -> decode PBR -> field projection
    -> official PBR encode -> cycle-cancelled x0 correction -> _xstart_to_pred
    -> Euler

Only the hidden-field projection is changed.  For a hidden local support, the
global baseline supplies the mean on a quantised global 1024 parent cell and
the current HR tile keeps its complete within-parent detail:

    Y+ = A^dagger A Y_G + (I - A^dagger A) Y_H

The implementation deliberately avoids latent mixing, global velocity, a
global timestep trajectory, or a weighted G/HR blend.  CUDA 4, seed 42, the
native 12-step texture schedule, fixed local shape, and Jacobi/barrier order
are the defaults required by Codex.md.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
)

import numpy as np
import torch
import utils3d
from PIL import Image, ImageDraw, ImageOps

import pixal3d.models as pixal3d_models
import pixal3d_cross_tile_pbr_perstep as base
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
from inference import MODEL_PATH
from pixal3d.modules.sparse import SparseTensor
from pixal3d.renderers.mesh_renderer import intrinsics_to_projection
from pixal3d.representations import Mesh, MeshWithVertexPbr, MeshWithVoxel
from pixal3d.utils import render_utils


FORMAT = "pixal3d_pbr_range_null_perstep_v1"
CANONICAL_IMAGE_SIZE = 4096
TILE_SIZE = 1024
TILE_STRIDE = 512
OVOXEL_RESOLUTION = 1024
LATENT_RESOLUTION = 64
PBR_CHANNELS = {
    "RGB": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}
PBR_CHANNEL_NAMES = ("RGB", "metallic", "roughness", "alpha")
REFERENCE_VARIANT_NAMES = {
    "pure_HR": "pure_HR",
    "current_gaussian_guided": "cross_tile_pbr_perstep_guided",
}


@dataclass
class TileContext(base.TileContext):
    """Base tile context plus the immutable projection support.

    ``global_pbr_reference`` is the exact local-resampled global baseline
    field and is row-aligned with ``geometry.coords``/``target_points``.
    ``observed_mask`` and ``hidden_mask`` are strict boolean complements, and
    ``global_parent_id`` is the quantised global 1024 O-voxel parent.
    """

    global_pbr_reference: Optional[torch.Tensor] = None
    observed_mask: Optional[torch.Tensor] = None
    hidden_mask: Optional[torch.Tensor] = None
    global_parent_id: Optional[torch.Tensor] = None
    range_null_endpoint: Optional[SparseTensor] = None


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    base._atomic_json(path, _jsonable(payload))


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _save_tensor(path: Path, value: torch.Tensor) -> None:
    _atomic_torch_save(path, {"tensor": value.detach().cpu()})


def _load_tensor(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and "tensor" in payload:
        value = payload["tensor"]
    else:
        value = payload
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"cached tensor is invalid: {path}")
    return value


def _load_mesh_any(path: Path) -> Any:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mesh = payload.get("mesh", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(mesh, (MeshWithVoxel, MeshWithVertexPbr)):
        raise RuntimeError(f"cached mesh is invalid: {path}")
    return mesh


def _context_from_base(
    context: base.TileContext,
    *,
    global_pbr_reference: Optional[torch.Tensor] = None,
    observed_mask: Optional[torch.Tensor] = None,
    hidden_mask: Optional[torch.Tensor] = None,
    global_parent_id: Optional[torch.Tensor] = None,
) -> TileContext:
    return TileContext(
        tile_id=context.tile_id,
        box=context.box,
        transform=context.transform,
        image=context.image,
        tile_dir=context.tile_dir,
        geometry=context.geometry,
        shape_reference=context.shape_reference,
        shape_norm=context.shape_norm,
        shape_denorm=context.shape_denorm,
        texture_reference=context.texture_reference,
        texture_norm=context.texture_norm,
        noise=context.noise,
        initial_state=context.initial_state,
        condition=context.condition,
        target_coords=context.target_coords,
        target_points=context.target_points,
        static_stats=context.static_stats,
        pure_endpoint=context.pure_endpoint,
        guided_endpoint=context.guided_endpoint,
        global_pbr_reference=global_pbr_reference,
        observed_mask=observed_mask,
        hidden_mask=hidden_mask,
        global_parent_id=global_parent_id,
    )


def _stage_context(context: TileContext, device: torch.device, low_vram: bool) -> None:
    """Put immutable context tensors on the normal or explicit low-VRAM path."""
    if low_vram:
        device = torch.device("cpu")
    sparse_names = (
        "shape_reference",
        "shape_norm",
        "shape_denorm",
        "texture_reference",
        "texture_norm",
        "noise",
        "initial_state",
    )
    for name in sparse_names:
        setattr(context, name, base._sparse_to_device(getattr(context, name), device))
    context.target_coords = context.target_coords.detach().to(device=device, dtype=torch.int32)
    context.target_points = context.target_points.detach().to(device=device, dtype=torch.float32)
    if context.global_pbr_reference is not None:
        context.global_pbr_reference = context.global_pbr_reference.detach().to(device=device)
    if context.observed_mask is not None:
        context.observed_mask = context.observed_mask.detach().to(device=device, dtype=torch.bool)
    if context.hidden_mask is not None:
        context.hidden_mask = context.hidden_mask.detach().to(device=device, dtype=torch.bool)
    if context.global_parent_id is not None:
        context.global_parent_id = context.global_parent_id.detach().to(device=device, dtype=torch.int64)
    context.condition = base._move_condition(context.condition, device)


def _channel_mean_abs_dict(left: torch.Tensor, right: torch.Tensor) -> Dict[str, float]:
    return {
        name: base._safe_mean((left[:, sl] - right[:, sl]).abs())
        for name, sl in PBR_CHANNELS.items()
    }


def _channel_norm_dict(value: torch.Tensor) -> Dict[str, float]:
    return {
        name: float(torch.linalg.vector_norm(value[:, sl].detach().to(torch.float64)).item())
        if value[:, sl].numel()
        else 0.0
        for name, sl in PBR_CHANNELS.items()
    }


def _channel_range_stats(value: torch.Tensor) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for name, sl in PBR_CHANNELS.items():
        field = value[:, sl]
        result[name] = {
            "min": float(field.min().item()) if field.numel() else 0.0,
            "max": float(field.max().item()) if field.numel() else 0.0,
            "out_of_range_ratio": float(
                ((field < 0.0) | (field > 1.0)).to(torch.float32).mean().item()
            )
            if field.numel()
            else 0.0,
        }
    return result


def _group_rows(
    parent_id: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if parent_id.ndim != 1 or mask.ndim != 1 or parent_id.shape != mask.shape:
        raise ValueError("parent_id and mask must be aligned one-dimensional tensors")
    valid = mask & (parent_id >= 0)
    rows = torch.where(valid)[0]
    if rows.numel() == 0:
        return rows, torch.empty((0,), device=parent_id.device, dtype=torch.int64), rows
    unique_parent, inverse = torch.unique(
        parent_id.index_select(0, rows).to(torch.int64),
        sorted=True,
        return_inverse=True,
    )
    return rows, inverse.to(torch.int64), unique_parent


def _group_means(
    value: torch.Tensor,
    parent_id: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    rows, inverse, unique_parent = _group_rows(parent_id, mask)
    channels = int(value.shape[1])
    if rows.numel() == 0:
        empty = torch.empty((0, channels), device=value.device, dtype=torch.float32)
        return {
            "rows": rows,
            "inverse": inverse,
            "unique_parent": unique_parent,
            "count": torch.empty((0,), device=value.device, dtype=torch.float32),
            "mean": empty,
        }
    selected = value.index_select(0, rows).to(torch.float32)
    count = torch.bincount(inverse, minlength=int(unique_parent.numel())).to(torch.float32)
    sums = torch.zeros(
        (int(unique_parent.numel()), channels), device=value.device, dtype=torch.float32
    )
    sums.index_add_(0, inverse, selected)
    return {
        "rows": rows,
        "inverse": inverse,
        "unique_parent": unique_parent,
        "count": count,
        "mean": sums / count[:, None].clamp_min(1.0),
    }


def _projection_invariants(
    projected: torch.Tensor,
    self_field: torch.Tensor,
    global_field: torch.Tensor,
    groups: Mapping[str, torch.Tensor],
    *,
    tolerance: float,
) -> Dict[str, Any]:
    rows = groups["rows"]
    inverse = groups["inverse"]
    if rows.numel() == 0:
        result = {
            "max_parent_mean_error": 0.0,
            "mean_parent_mean_error": 0.0,
            "max_null_preservation_error": 0.0,
            "mean_null_preservation_error": 0.0,
            "parent_count": 0,
            "hidden_count": 0,
            "passed": True,
        }
        return result
    means_projected = groups["mean_projected"]
    mean_global = groups["mean_global"]
    mean_self = groups["mean_self"]
    parent_error = (means_projected - mean_global).abs()
    projected_null = projected.index_select(0, rows).to(torch.float32) - means_projected.index_select(0, inverse)
    self_null = self_field.index_select(0, rows).to(torch.float32) - mean_self.index_select(0, inverse)
    null_error = (projected_null - self_null).abs()
    max_parent = float(parent_error.max().item())
    max_null = float(null_error.max().item())
    mean_parent = float(parent_error.mean().item())
    mean_null = float(null_error.mean().item())
    passed = max(max_parent, max_null) <= float(tolerance)
    result = {
        "max_parent_mean_error": max_parent,
        "mean_parent_mean_error": mean_parent,
        "max_null_preservation_error": max_null,
        "mean_null_preservation_error": mean_null,
        "parent_count": int(groups["unique_parent"].numel()),
        "hidden_count": int(rows.numel()),
        "passed": bool(passed),
    }
    if not passed:
        raise AssertionError(
            "range-null invariant failed: "
            f"max_parent_mean_error={max_parent:.9g}, "
            f"max_null_preservation_error={max_null:.9g}, "
            f"tolerance={float(tolerance):.9g}"
        )
    return result


def range_null_project(
    self_field: torch.Tensor,
    global_pbr_reference: torch.Tensor,
    global_parent_id: torch.Tensor,
    hidden_mask: torch.Tensor,
    *,
    tolerance: float = 5e-5,
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, torch.Tensor]]:
    """Apply ``A^dagger A Y_G + (I-A^dagger A)Y_H`` on hidden rows.

    ``A`` is the mean-reduction operator over rows sharing one valid global
    parent id.  ``A^dagger`` broadcasts a parent vector back to those rows.
    The only reductions are GPU ``unique``, ``bincount`` and ``index_add_``;
    there is no Python loop over voxels or parent groups.
    """
    if self_field.ndim != 2 or self_field.shape[1] != 6:
        raise ValueError(f"self_field must be [N,6], got {tuple(self_field.shape)}")
    if global_pbr_reference.shape != self_field.shape:
        raise ValueError("global PBR reference and self field must have identical support/shape")
    if global_parent_id.shape != hidden_mask.shape or hidden_mask.shape[0] != self_field.shape[0]:
        raise ValueError("range-null support tensors are not aligned")
    if not torch.isfinite(self_field).all() or not torch.isfinite(global_pbr_reference).all():
        raise ValueError("range-null input contains non-finite PBR values")
    groups_self = _group_means(self_field, global_parent_id, hidden_mask)
    groups_global = _group_means(global_pbr_reference, global_parent_id, hidden_mask)
    if not torch.equal(groups_self["rows"], groups_global["rows"]) or not torch.equal(
        groups_self["inverse"], groups_global["inverse"]
    ):
        raise RuntimeError("range-null group construction changed between self and global fields")
    rows = groups_self["rows"]
    inverse = groups_self["inverse"]
    means_self = groups_self["mean"]
    means_global = groups_global["mean"]
    projected = self_field.clone()
    if rows.numel():
        correction = means_global - means_self
        selected = self_field.index_select(0, rows).to(torch.float32)
        projected_rows = selected + correction.index_select(0, inverse)
        projected.index_copy_(0, rows, projected_rows.to(projected.dtype))
    else:
        correction = torch.empty((0, 6), device=self_field.device, dtype=torch.float32)
    groups: Dict[str, torch.Tensor] = {
        "rows": rows,
        "inverse": inverse,
        "unique_parent": groups_self["unique_parent"],
        "count": groups_self["count"],
        "mean_self": means_self,
        "mean_global": means_global,
        "mean_projected": _group_means(projected, global_parent_id, hidden_mask)["mean"],
        "correction": correction,
    }
    invariants = _projection_invariants(
        projected,
        self_field,
        global_pbr_reference,
        groups,
        tolerance=float(tolerance),
    )
    stats = {
        **invariants,
        "projection": "global parent mean range + HR within-parent null/detail",
        "channels": list(PBR_CHANNEL_NAMES),
        "correction_mean_abs": {
            name: float(correction[:, sl].abs().mean().item()) if correction.numel() else 0.0
            for name, sl in PBR_CHANNELS.items()
        },
        "self_vs_global_mean_abs": _channel_mean_abs_dict(self_field, global_pbr_reference),
        "projected_vs_self_mean_abs_hidden": _channel_mean_abs_dict(
            projected.index_select(0, rows), self_field.index_select(0, rows)
        )
        if rows.numel()
        else {name: 0.0 for name in PBR_CHANNEL_NAMES},
        "self_range": _channel_range_stats(self_field),
        "global_range": _channel_range_stats(global_pbr_reference),
        "projected_range": _channel_range_stats(projected),
    }
    details = {
        "hidden_rows": rows.detach(),
        "hidden_parent_inverse": inverse.detach(),
        "global_parent_unique": groups["unique_parent"].detach(),
        "mean_self": means_self.detach(),
        "mean_global": means_global.detach(),
        "mean_projected": groups["mean_projected"].detach(),
        "correction": correction.detach(),
    }
    return projected, stats, details


def _field_projection_metrics(
    output_field: torch.Tensor,
    self_field: torch.Tensor,
    global_field: torch.Tensor,
    parent_id: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, Any]:
    """Compute E_R, E_N, and detail energy for one observed/hidden subset."""
    groups_out = _group_means(output_field, parent_id, mask)
    groups_self = _group_means(self_field, parent_id, mask)
    groups_global = _group_means(global_field, parent_id, mask)
    rows = groups_out["rows"]
    inverse = groups_out["inverse"]
    if rows.numel() == 0:
        zeros = {name: 0.0 for name in PBR_CHANNEL_NAMES}
        return {
            "voxel_count": 0,
            "parent_count": 0,
            "E_R": zeros,
            "E_N": zeros,
            "E_detail": {"output": zeros, "self_HR": zeros, "global_G": zeros},
        }
    mean_out = groups_out["mean"]
    mean_self = groups_self["mean"]
    mean_global = groups_global["mean"]
    out_rows = output_field.index_select(0, rows).to(torch.float32)
    self_rows = self_field.index_select(0, rows).to(torch.float32)
    global_rows = global_field.index_select(0, rows).to(torch.float32)
    out_null = out_rows - mean_out.index_select(0, inverse)
    self_null = self_rows - mean_self.index_select(0, inverse)
    global_null = global_rows - mean_global.index_select(0, inverse)
    er = (mean_out - mean_global).abs()
    en = (out_null - self_null).abs()
    detail_out = out_null.square().mean(dim=0).sqrt()
    detail_self = self_null.square().mean(dim=0).sqrt()
    detail_global = global_null.square().mean(dim=0).sqrt()

    def _mean_channels(values: torch.Tensor) -> Dict[str, float]:
        return {
            name: float(values[:, sl].mean().item())
            for name, sl in PBR_CHANNELS.items()
        }

    return {
        "voxel_count": int(rows.numel()),
        "parent_count": int(groups_out["unique_parent"].numel()),
        "E_R": _mean_channels(er),
        "E_N": _mean_channels(en),
        "E_detail": {
            "output": _mean_channels(detail_out[None]),
            "self_HR": _mean_channels(detail_self[None]),
            "global_G": _mean_channels(detail_global[None]),
        },
    }


def _decompose_hidden_field(
    field: torch.Tensor,
    parent_id: torch.Tensor,
    hidden_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return full-support coarse and null/detail components for snapshots."""
    groups = _group_means(field, parent_id, hidden_mask)
    coarse = torch.zeros_like(field)
    detail = torch.zeros_like(field)
    rows = groups["rows"]
    inverse = groups["inverse"]
    if rows.numel():
        means = groups["mean"]
        selected = field.index_select(0, rows).to(torch.float32)
        coarse.index_copy_(0, rows, means.index_select(0, inverse).to(coarse.dtype))
        detail.index_copy_(0, rows, (selected - means.index_select(0, inverse)).to(detail.dtype))
    return coarse, detail


@torch.no_grad()
def _rasterize_canonical_front_surface(
    baseline_mesh: MeshWithVoxel,
    global_camera: Mapping[str, float],
    output_dir: Path,
    *,
    face_chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Rasterize the nearest baseline surface at the mandated 4096 resolution.

    The rasterizer stores an object-space position for the nearest triangle at
    every pixel.  Faces are processed in chunks only to stay below
    nvdiffrast's face-count limit; the per-chunk depth buffer is merged with a
    pixelwise nearest-depth comparison, so no faces are dropped.
    """
    if int(face_chunk_size) <= 0:
        raise ValueError("visibility face chunk size must be positive")
    position_path = output_dir / "canonical_front_surface_position_4096.pt"
    mask_path = output_dir / "canonical_front_surface_mask_4096.png"
    if position_path.is_file() and mask_path.is_file():
        payload = torch.load(position_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or "position" not in payload or "valid" not in payload:
            raise RuntimeError(f"invalid cached canonical front surface: {position_path}")
        position = payload["position"].to(torch.float32).contiguous()
        valid = payload["valid"].to(torch.bool).contiguous()
        if tuple(position.shape) != (CANONICAL_IMAGE_SIZE, CANONICAL_IMAGE_SIZE, 3):
            raise RuntimeError("cached canonical front position has the wrong shape")
        return position, valid, {
            "resolution": CANONICAL_IMAGE_SIZE,
            "face_count": int(baseline_mesh.faces.shape[0]),
            "face_chunk_size": int(face_chunk_size),
            "cached": True,
            "valid_pixel_count": int(valid.sum().item()),
            "mask_png": str(mask_path.resolve()),
            "position_tensor": str(position_path.resolve()),
        }

    import nvdiffrast.torch as dr

    device = torch.device("cuda")
    vertices = baseline_mesh.vertices.to(device=device, dtype=torch.float32).contiguous()
    faces = baseline_mesh.faces.to(device=device, dtype=torch.int32).contiguous()
    extrinsics, intrinsics = render_utils.proj_camera_to_render_params(
        camera_angle_x=float(global_camera["camera_angle_x"]),
        distance=float(global_camera["distance"]),
    )
    near = max(0.01, float(global_camera["distance"]) - 2.0)
    far = float(global_camera["distance"]) + 10.0
    perspective = intrinsics_to_projection(intrinsics, near, far)
    vertices_homo = torch.cat(
        [vertices, torch.ones_like(vertices[..., :1])], dim=-1
    ).unsqueeze(0)
    full_proj = (perspective @ extrinsics).unsqueeze(0)
    vertices_clip = torch.bmm(vertices_homo, full_proj.transpose(-1, -2)).contiguous()
    vertices_batched = vertices.unsqueeze(0)
    depth = torch.full(
        (CANONICAL_IMAGE_SIZE, CANONICAL_IMAGE_SIZE),
        float("inf"),
        device=device,
        dtype=torch.float32,
    )
    position = torch.zeros(
        (CANONICAL_IMAGE_SIZE, CANONICAL_IMAGE_SIZE, 3),
        device=device,
        dtype=torch.float32,
    )
    raster_context = dr.RasterizeCudaContext(device="cuda")
    face_count = int(faces.shape[0])
    for start in range(0, face_count, int(face_chunk_size)):
        face_chunk = faces[start : start + int(face_chunk_size)]
        rast, _ = dr.rasterize(
            raster_context,
            vertices_clip,
            face_chunk,
            (CANONICAL_IMAGE_SIZE, CANONICAL_IMAGE_SIZE),
        )
        valid = rast[0, ..., 3] > 0.0
        replace = valid & (rast[0, ..., 2] < depth)
        if bool(replace.any().item()):
            chunk_position = dr.interpolate(
                vertices_batched,
                rast,
                face_chunk,
            )[0][0]
            depth = torch.where(replace, rast[0, ..., 2], depth)
            position = torch.where(replace[..., None], chunk_position, position)
            del chunk_position
        del rast, face_chunk, valid, replace
        if (start // int(face_chunk_size)) % 4 == 0:
            print(
                "[visibility] canonical raster faces "
                f"[{start:,},{min(start + int(face_chunk_size), face_count):,})/{face_count:,}"
            )
    valid = torch.isfinite(depth)
    position_cpu = position.detach().cpu().to(torch.float32)
    valid_cpu = valid.detach().cpu().to(torch.bool)
    _atomic_torch_save(
        position_path,
        {"position": position_cpu.to(torch.float16), "valid": valid_cpu},
    )
    Image.fromarray((valid_cpu.numpy().astype(np.uint8) * 255), mode="L").save(mask_path)
    del vertices, faces, vertices_homo, vertices_clip, vertices_batched, depth, position, valid
    _empty_cuda_cache()
    return position_cpu, valid_cpu, {
        "resolution": CANONICAL_IMAGE_SIZE,
        "face_count": face_count,
        "face_chunk_size": int(face_chunk_size),
        "cached": False,
        "valid_pixel_count": int(valid_cpu.sum().item()),
        "mask_png": str(mask_path),
        "position_tensor": str(position_path),
    }


def _global_parent_ids(global_points: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Quantise normalized object coordinates onto the global C1024 grid."""
    if global_points.ndim != 2 or global_points.shape[1] != 3:
        raise ValueError("global points must have shape [N,3]")
    finite = torch.isfinite(global_points).all(dim=1)
    quantized = torch.floor(
        float(OVOXEL_RESOLUTION) * (global_points.to(torch.float32) + 0.5)
    ).to(torch.int64)
    valid = finite & (quantized >= 0).all(dim=1) & (quantized < OVOXEL_RESOLUTION).all(dim=1)
    parent = torch.full(
        (global_points.shape[0],), -1, device=global_points.device, dtype=torch.int64
    )
    if bool(valid.any().item()):
        xyz = quantized[valid]
        parent[valid] = (
            (xyz[:, 0] * OVOXEL_RESOLUTION + xyz[:, 1]) * OVOXEL_RESOLUTION
            + xyz[:, 2]
        )
    return parent, {
        "quantization": "floor(1024 * (global_normalized_coordinate + 0.5))",
        "resolution": OVOXEL_RESOLUTION,
        "valid_count": int(valid.sum().item()),
        "invalid_count": int((~valid).sum().item()),
        "unique_parent_count": int(torch.unique(parent[valid]).numel()) if bool(valid.any().item()) else 0,
    }


def _visibility_for_context(
    context: TileContext,
    global_camera: Mapping[str, float],
    front_position: torch.Tensor,
    front_valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Classify every local support point against the canonical front surface."""
    _, uv_full, global_points = base._local_to_global(
        context.target_points,
        transform=context.transform,
        global_camera=global_camera,
    )
    _, depth, finite_projection = core._project_global_q_to_4096(
        global_points * (2.0 * float(global_camera["mesh_scale"])),
        global_camera=global_camera,
    )
    points_cpu = global_points.detach().cpu().to(torch.float32)
    uv_cpu = uv_full.detach().cpu().to(torch.float32)
    depth_cpu = depth.detach().cpu().to(torch.float32)
    finite_cpu = finite_projection.detach().cpu().to(torch.bool)
    height, width = front_valid.shape
    pixel = torch.floor(uv_cpu).to(torch.int64)
    inside = (
        finite_cpu
        & (pixel[:, 0] >= 0)
        & (pixel[:, 0] < width)
        & (pixel[:, 1] >= 0)
        & (pixel[:, 1] < height)
    )
    safe_pixel = pixel.clamp(
        min=0,
        max=max(width, height) - 1,
    )
    flat = safe_pixel[:, 1] * width + safe_pixel[:, 0]
    front_valid_flat = front_valid.reshape(-1).index_select(0, flat)
    front_position_flat = front_position.reshape(-1, 3).index_select(0, flat)
    distance = torch.linalg.vector_norm(front_position_flat - points_cpu, dim=1)
    focal_4096 = float(core._focal_pixels(float(global_camera["camera_angle_x"]), 1024)) * 4.0
    voxel_half_diagonal = math.sqrt(3.0) / (2.0 * float(OVOXEL_RESOLUTION))
    pixel_half_diagonal = (
        torch.clamp(depth_cpu, min=0.0) / float(focal_4096) * (math.sqrt(2.0) / 2.0)
    )
    threshold = voxel_half_diagonal + pixel_half_diagonal
    observed_cpu = inside & front_valid_flat & torch.isfinite(distance) & (distance <= threshold)
    hidden_cpu = ~observed_cpu
    if not bool((observed_cpu ^ hidden_cpu).all().item()):
        raise AssertionError("observed/hidden masks are not a strict binary complement")
    device = context.target_points.device
    observed = observed_cpu.to(device=device)
    hidden = hidden_cpu.to(device=device)
    parent_id, parent_stats = _global_parent_ids(global_points)
    if bool((parent_id < 0).any().item()):
        invalid_count = int((parent_id < 0).sum().item())
        raise RuntimeError(
            f"tile {context.tile_id}: {invalid_count} local support points do not map to global C1024"
        )
    tile_uv = base._tile_uv(uv_full, context.transform).detach().cpu()
    canvas = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    tile_pixel = torch.floor(tile_uv).to(torch.int64)
    tile_inside = (
        (tile_pixel[:, 0] >= 0)
        & (tile_pixel[:, 0] < TILE_SIZE)
        & (tile_pixel[:, 1] >= 0)
        & (tile_pixel[:, 1] < TILE_SIZE)
    )
    rows_inside = torch.where(tile_inside)[0]
    if rows_inside.numel():
        tx = tile_pixel.index_select(0, rows_inside)[:, 0].numpy()
        ty = tile_pixel.index_select(0, rows_inside)[:, 1].numpy()
        obs_values = observed_cpu.index_select(0, rows_inside).numpy()
        canvas[ty[obs_values], tx[obs_values]] = (40, 220, 70)
        canvas[ty[~obs_values], tx[~obs_values]] = (230, 50, 50)
    Image.fromarray(canvas, mode="RGB").save(context.tile_dir / "observed_hidden_mask.png")
    _save_tensor(context.tile_dir / "global_parent_id.pt", parent_id)
    _save_tensor(context.tile_dir / "observed_mask.pt", observed)
    _save_tensor(context.tile_dir / "hidden_mask.pt", hidden)
    stats = {
        "tile_id": int(context.tile_id),
        "observed_voxel_count": int(observed.sum().item()),
        "hidden_voxel_count": int(hidden.sum().item()),
        "observed_fraction": float(observed.to(torch.float32).mean().item()),
        "hidden_fraction": float(hidden.to(torch.float32).mean().item()),
        "front_surface_match_threshold": {
            "voxel_half_diagonal": voxel_half_diagonal,
            "pixel_half_diagonal_formula": "depth / focal_4096 * sqrt(2)/2",
            "threshold_is_blend_weight": False,
        },
        "front_surface_valid_at_projected_pixel_count": int(
            (inside & front_valid_flat).sum().item()
        ),
        "front_surface_distance_statistics": base._tensor_stats(distance[inside & front_valid_flat]),
        "global_parent": parent_stats,
        "mask_png": str((context.tile_dir / "observed_hidden_mask.png").resolve()),
        "mask_policy": "canonical 4096 baseline front position map; strict binary observed/hidden",
    }
    return global_points, uv_full, observed, hidden, stats


def _save_visibility_overview(
    image_4096: Image.Image,
    contexts: Sequence[TileContext],
    global_camera: Mapping[str, float],
    output_path: Path,
) -> None:
    """Save a compact manual-inspection view of observed (green)/hidden (red)."""
    scale = 4
    canvas = np.asarray(
        image_4096.convert("RGB").resize(
            (CANONICAL_IMAGE_SIZE // scale, CANONICAL_IMAGE_SIZE // scale),
            Image.Resampling.LANCZOS,
        ),
        dtype=np.float32,
    )
    color = np.zeros_like(canvas)
    coverage = np.zeros(canvas.shape[:2], dtype=np.float32)
    for context in contexts:
        _, uv_full, _ = base._local_to_global(
            context.target_points,
            transform=context.transform,
            global_camera=global_camera,
        )
        uv = torch.floor(uv_full.detach().cpu() / float(scale)).to(torch.int64)
        inside = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < canvas.shape[1])
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < canvas.shape[0])
        )
        rows = torch.where(inside)[0]
        if not rows.numel():
            continue
        x = uv.index_select(0, rows)[:, 0].numpy()
        y = uv.index_select(0, rows)[:, 1].numpy()
        observed = context.observed_mask.detach().cpu().index_select(0, rows).numpy()
        color[y[observed], x[observed]] = (40, 220, 70)
        color[y[~observed], x[~observed]] = (230, 50, 50)
        coverage[y, x] += 1.0
    alpha = np.clip(0.3 + 0.35 * coverage[..., None], 0.0, 0.85)
    output = (canvas * (1.0 - alpha) + color * alpha).clip(0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="RGB").save(output_path)


def _tile_image(image_4096: Image.Image, box: Sequence[int]) -> Image.Image:
    image = image_4096.crop(tuple(int(v) for v in box)).convert("RGB")
    if image.size != (TILE_SIZE, TILE_SIZE):
        image = image.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS)
    return image


def _make_condition(
    pipeline: Any,
    tile_image: Image.Image,
    shape_norm: SparseTensor,
    transform: Any,
    *,
    low_vram: bool,
) -> Mapping[str, Any]:
    condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [tile_image],
        shape_norm.coords.to(device=torch.device("cuda"), dtype=torch.int32),
        camera_angle_x=float(transform.camera_angle_x),
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
        grid_resolution_override=LATENT_RESOLUTION,
    )
    return base._move_condition(condition, torch.device("cpu" if low_vram else "cuda"))


def _load_reusable_contexts(
    *,
    source_dir: Path,
    args: argparse.Namespace,
    pipeline: Any,
    baseline_mesh: MeshWithVoxel,
    global_camera: Mapping[str, float],
    image_4096: Image.Image,
    output_dir: Path,
    boxes: Sequence[Tuple[int, int, int, int]],
) -> List[TileContext]:
    """Load fixed-shape/initial-state caches from the previous guided run."""
    device = torch.device("cuda")
    face_min, face_max, face_finite = core._project_face_bboxes(
        baseline_mesh.vertices,
        baseline_mesh.faces,
        mesh_scale=float(global_camera["mesh_scale"]),
        global_camera=global_camera,
        chunk_size=int(args.face_projection_chunk_size),
    )
    requested = base._parse_ids(args.tile_ids)
    source_summary_path = source_dir / "tile_preparation_summary.json"
    source_summary = (
        json.loads(source_summary_path.read_text(encoding="utf-8"))
        if source_summary_path.is_file()
        else {}
    )
    source_active = set(int(v) for v in source_summary.get("prepared_tile_ids", []))
    contexts: List[TileContext] = []
    for tile_id, box in enumerate(boxes):
        if requested is not None and int(tile_id) not in requested:
            continue
        source_tile = source_dir / "tiles" / f"tile_{tile_id:02d}"
        source_files = [
            source_tile / "fixed_shape_norm.pt",
            source_tile / "texture_reference_norm.pt",
            source_tile / "texture_initial_state.pt",
            source_tile / "fixed_shape_summary.json",
        ]
        if source_active and tile_id not in source_active:
            continue
        if not all(path.is_file() for path in source_files):
            continue
        tile_dir = output_dir / "tiles" / f"tile_{tile_id:02d}"
        tile_dir.mkdir(parents=True, exist_ok=True)
        transform = core._derive_tile_camera(
            tile_id=int(tile_id),
            box=box,
            global_camera=global_camera,
            extend_pixel=int(args.extend_pixel),
        )
        _atomic_json(tile_dir / "tile_camera.json", transform.__dict__)
        tile_image = _tile_image(image_4096, box)
        tile_image.save(tile_dir / "hr_tile_1024_condition.png")
        geometry = core._prepare_tile_geometry(
            global_vertices=baseline_mesh.vertices,
            global_faces=baseline_mesh.faces,
            global_face_min=face_min,
            global_face_max=face_max,
            global_face_finite=face_finite,
            global_camera=global_camera,
            transform=transform,
        )
        shape_norm = base._load_sparse_payload(source_files[0])
        texture_norm = base._load_sparse_payload(source_files[1])
        initial_state = base._load_sparse_payload(source_files[2])
        alignment = core._latent_support_diagnostics(shape_norm, texture_norm)
        if not alignment["coordinates_exactly_equal"]:
            raise RuntimeError(f"tile {tile_id}: reusable shape/texture support mismatch: {alignment}")
        static_stats = json.loads(source_files[3].read_text(encoding="utf-8"))
        shape_reference = base._fresh_sparse(shape_norm)
        texture_reference = base._fresh_sparse(texture_norm)
        shape_denorm = base._denormalize_slat(shape_norm, pipeline.shape_slat_normalization)
        noise = SparseTensor(torch.zeros_like(texture_norm.feats), texture_norm.coords.detach().clone())
        context = TileContext(
            tile_id=int(tile_id),
            box=tuple(int(v) for v in box),
            transform=transform,
            image=tile_image,
            tile_dir=tile_dir,
            geometry=geometry,
            shape_reference=shape_reference,
            shape_norm=shape_norm,
            shape_denorm=shape_denorm,
            texture_reference=texture_reference,
            texture_norm=texture_norm,
            noise=noise,
            initial_state=initial_state,
            condition=_make_condition(
                pipeline,
                tile_image,
                shape_norm,
                transform,
                low_vram=bool(args.low_vram),
            ),
            target_coords=geometry.coords.to(device=device, dtype=torch.int32),
            target_points=(
                geometry.coords.to(device=device, dtype=torch.float32) + 0.5
            )
            / float(OVOXEL_RESOLUTION)
            - 0.5,
            static_stats={**static_stats, "reused_from": str(source_dir.resolve())},
        )
        _stage_context(context, device, bool(args.low_vram))
        contexts.append(context)
        print(
            f"[prepare reusable tile {tile_id:02d}] "
            f"tokens={context.target_coords.shape[0]:,} source={source_dir}"
        )
    return contexts


def _prepare_range_contexts(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    baseline_mesh: MeshWithVoxel,
    global_camera: Mapping[str, float],
    image_4096: Image.Image,
    output_dir: Path,
    global_attr_field: MeshWithVoxel,
    shape_encoder: Optional[torch.nn.Module],
    pbr_encoder: torch.nn.Module,
    boxes: Sequence[Tuple[int, int, int, int]],
    reference_dir: Optional[Path],
) -> Tuple[List[TileContext], Dict[str, Any]]:
    """Prepare fixed support, global reference, parent ids, and masks."""
    device = torch.device("cuda")
    contexts: List[TileContext]
    reused = reference_dir is not None and (reference_dir / "tiles").is_dir()
    if reused:
        contexts = _load_reusable_contexts(
            source_dir=reference_dir,
            args=args,
            pipeline=pipeline,
            baseline_mesh=baseline_mesh,
            global_camera=global_camera,
            image_4096=image_4096,
            output_dir=output_dir,
            boxes=boxes,
        )
        if not contexts:
            raise RuntimeError(f"reference experiment has no reusable tile contexts: {reference_dir}")
    else:
        if shape_encoder is None:
            raise RuntimeError("fresh context preparation requires the shape encoder")
        raw_contexts = base._prepare_tile_contexts(
            args=args,
            pipeline=pipeline,
            baseline_mesh=baseline_mesh,
            global_camera=global_camera,
            image_4096=image_4096,
            output_dir=output_dir,
            global_attr_field=global_attr_field,
            shape_encoder=shape_encoder,
            pbr_encoder=pbr_encoder,
            boxes=boxes,
        )
        contexts = [_context_from_base(context) for context in raw_contexts]
        if bool(args.low_vram):
            for context in contexts:
                _stage_context(context, device, True)

    front_position, front_valid, raster_stats = _rasterize_canonical_front_surface(
        baseline_mesh,
        global_camera,
        output_dir,
        face_chunk_size=int(args.visibility_face_chunk_size),
    )
    visibility_records: List[Dict[str, Any]] = []
    for context in contexts:
        reference_path = context.tile_dir / "global_pbr_reference.pt"
        if bool(args.resume) and reference_path.is_file():
            reference = _load_tensor(reference_path).to(
                device=context.target_points.device,
                dtype=torch.float32,
            )
            material_stats = {"source": "resumed exact local_attrs cache"}
        else:
            local_attrs, material_stats = core._resample_local_attrs_from_global(
                geometry=context.geometry,
                global_attr_field=global_attr_field,
                global_camera=global_camera,
                transform=context.transform,
                query_chunk_size=int(args.material_query_chunk_size),
                face_chunk_size=int(args.material_face_chunk_size),
            )
            reference = local_attrs.to(
                device=context.target_points.device,
                dtype=torch.float32,
            )
            _save_tensor(reference_path, reference)
        if reference.ndim != 2 or reference.shape != (context.target_coords.shape[0], 6):
            raise RuntimeError(
                f"tile {context.tile_id}: global PBR reference is not aligned with geometry.coords: "
                f"reference={tuple(reference.shape)} coords={tuple(context.target_coords.shape)}"
            )
        if not torch.isfinite(reference).all():
            raise RuntimeError(f"tile {context.tile_id}: global PBR reference contains non-finite values")
        global_points, uv_full, observed, hidden, visibility_stats = _visibility_for_context(
            context,
            global_camera,
            front_position,
            front_valid,
        )
        parent_id, parent_stats = _global_parent_ids(global_points)
        if not torch.equal(observed, ~hidden):
            raise AssertionError(f"tile {context.tile_id}: observed/hidden mask complement failed")
        if bool((parent_id < 0).any().item()):
            raise RuntimeError(f"tile {context.tile_id}: invalid global parent ids")
        context.global_pbr_reference = reference
        context.observed_mask = observed
        context.hidden_mask = hidden
        context.global_parent_id = parent_id
        context.static_stats["global_pbr_reference"] = {
            "field": "local_attrs from _resample_local_attrs_from_global",
            "aligned_with": ["geometry.coords", "target_points"],
            "material_stats": material_stats,
            "range": _channel_range_stats(reference),
        }
        context.static_stats["visibility"] = visibility_stats
        context.static_stats["global_parent"] = parent_stats
        _atomic_json(context.tile_dir / "range_null_static.json", context.static_stats)
        visibility_records.append(visibility_stats)
    _save_visibility_overview(
        image_4096,
        contexts,
        global_camera,
        output_dir / "canonical_observed_hidden_overview.png",
    )
    _atomic_json(
        output_dir / "visibility_summary.json",
        {
            "format": f"{FORMAT}_visibility_v1",
            "canonical_raster": raster_stats,
            "tiles": visibility_records,
            "observed_voxel_count": int(sum(row["observed_voxel_count"] for row in visibility_records)),
            "hidden_voxel_count": int(sum(row["hidden_voxel_count"] for row in visibility_records)),
            "manual_check_artifacts": {
                "front_surface_mask": str((output_dir / "canonical_front_surface_mask_4096.png").resolve()),
                "observed_hidden_overview": str((output_dir / "canonical_observed_hidden_overview.png").resolve()),
            },
        },
    )
    return contexts, {
        "reused_contexts": bool(reused),
        "reference_dir": str(reference_dir.resolve()) if reference_dir is not None else None,
        "raster": raster_stats,
        "tiles": visibility_records,
        "observed_voxel_count": int(sum(row["observed_voxel_count"] for row in visibility_records)),
        "hidden_voxel_count": int(sum(row["hidden_voxel_count"] for row in visibility_records)),
        "manual_check_artifacts": {
            "front_surface_mask": str((output_dir / "canonical_front_surface_mask_4096.png").resolve()),
            "observed_hidden_overview": str((output_dir / "canonical_observed_hidden_overview.png").resolve()),
        },
    }


def _save_projection_snapshot(
    context: TileContext,
    output_root: Path,
    step_index: int,
    self_field: torch.Tensor,
    current_cross_tile_fused: torch.Tensor,
    projected_field: torch.Tensor,
) -> str:
    """Persist the requested self/projected/coarse/null diagnostic fields."""
    if context.global_pbr_reference is None or context.hidden_mask is None:
        raise RuntimeError("projection snapshot requires global reference and hidden mask")
    coarse_global, _ = _decompose_hidden_field(
        context.global_pbr_reference,
        context.global_parent_id,
        context.hidden_mask,
    )
    _, null_self = _decompose_hidden_field(
        self_field,
        context.global_parent_id,
        context.hidden_mask,
    )
    coarse_projected, null_projected = _decompose_hidden_field(
        projected_field,
        context.global_parent_id,
        context.hidden_mask,
    )
    correction = projected_field - self_field
    path = (
        output_root
        / "tiles"
        / f"tile_{int(context.tile_id):02d}"
        / f"step_{int(step_index):02d}_projection.pt"
    )
    _atomic_torch_save(
        path,
        {
            "format": f"{FORMAT}_projection_snapshot_v1",
            "tile_id": int(context.tile_id),
            "step": int(step_index),
            "field_dtype": "float16; source tensors remain float32 in the flow",
            "self_pbr": self_field.detach().cpu().to(torch.float16),
            "projected_pbr": projected_field.detach().cpu().to(torch.float16),
            "coarse_component_global": coarse_global.detach().cpu().to(torch.float16),
            "null_detail_component_self_HR": null_self.detach().cpu().to(torch.float16),
            "correction_field": correction.detach().cpu().to(torch.float16),
            "global_reference_path": str(
                (context.tile_dir / "global_pbr_reference.pt").resolve()
            ),
            "observed_mask_path": str((context.tile_dir / "observed_mask.pt").resolve()),
            "hidden_mask_path": str((context.tile_dir / "hidden_mask.pt").resolve()),
            "global_parent_id_path": str((context.tile_dir / "global_parent_id.pt").resolve()),
        },
    )
    return str(path.resolve())


def _fuse_observed_field_only(
    *,
    target: TileContext,
    contexts: Sequence[TileContext],
    decoded: Mapping[int, MeshWithVoxel],
    self_field: torch.Tensor,
    global_camera: Mapping[str, float],
    sigma_pixels: float,
    query_chunk_size: int,
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, torch.Tensor]]:
    """Run the existing Gaussian helper only on canonical-observed rows.

    Passing a compact target view makes the helper issue no donor queries for
    hidden rows.  The returned full field is identity on hidden rows and is
    replaced only on ``target.observed_mask`` by the caller.
    """
    if target.observed_mask is None:
        raise RuntimeError("observed-only fusion requires target.observed_mask")
    observed_rows = torch.where(target.observed_mask)[0]
    hidden_count = int(target.observed_mask.numel() - observed_rows.numel())
    if observed_rows.numel() == 0:
        empty_stats = {
            "target_tile": int(target.tile_id),
            "active_ovoxel_count": 0,
            "observed_ovoxel_count": 0,
            "hidden_ovoxel_count": hidden_count,
            "overlap_ovoxel_count": 0,
            "non_overlap_ovoxel_count": 0,
            "query_valid_donor_count": {"min": 0, "mean": 0.0, "max": 0},
            "covered_donor_count": {"min": 0, "mean": 0.0, "max": 0},
            "distance_to_center_pixels": base._tensor_stats(torch.empty((0,), device=self_field.device)),
            "normalized_fusion_weight": base._tensor_stats(torch.empty((0,), device=self_field.device)),
            "raw_fusion_weight": base._tensor_stats(torch.empty((0,), device=self_field.device)),
            "gaussian_sigma_pixels": float(sigma_pixels),
            "fusion_region": "observed_only",
            "pbr_self_vs_fused_mean_abs_all": {name: 0.0 for name in PBR_CHANNEL_NAMES},
            "pbr_self_vs_fused_mean_abs_overlap": {name: 0.0 for name in PBR_CHANNEL_NAMES},
        }
        return self_field.clone(), empty_stats, {
            "target_tile": torch.tensor(int(target.tile_id), dtype=torch.int64),
            "capacity": torch.tensor(base._candidate_capacity(), dtype=torch.int64),
        }
    target_view = SimpleNamespace(
        tile_id=int(target.tile_id),
        transform=target.transform,
        target_points=target.target_points.index_select(0, observed_rows),
        target_coords=target.target_coords.index_select(0, observed_rows),
    )
    observed_field, stats, details = base._fuse_tile_field(
        target=target_view,
        contexts=contexts,
        decoded=decoded,
        self_field=self_field.index_select(0, observed_rows),
        global_camera=global_camera,
        sigma_pixels=float(sigma_pixels),
        query_chunk_size=int(query_chunk_size),
    )
    full_field = self_field.clone()
    full_field.index_copy_(0, observed_rows, observed_field)
    stats = dict(stats)
    stats.update(
        {
            "active_ovoxel_count": int(target.target_points.shape[0]),
            "observed_ovoxel_count": int(observed_rows.numel()),
            "hidden_ovoxel_count": hidden_count,
            "fusion_region": "observed_only",
            "hidden_cross_tile_queries": 0,
        }
    )
    return full_field, stats, details


def _projection_step_record(
    *,
    context: TileContext,
    self_field: torch.Tensor,
    global_field: torch.Tensor,
    current_cross_tile_fused: torch.Tensor,
    projected_field: torch.Tensor,
    range_null_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    if context.observed_mask is None or context.hidden_mask is None or context.global_parent_id is None:
        raise RuntimeError("projection diagnostics require complete context masks/parents")
    hidden_before = _field_projection_metrics(
        self_field,
        self_field,
        global_field,
        context.global_parent_id,
        context.hidden_mask,
    )
    hidden_after = _field_projection_metrics(
        projected_field,
        self_field,
        global_field,
        context.global_parent_id,
        context.hidden_mask,
    )
    observed_after = _field_projection_metrics(
        projected_field,
        self_field,
        global_field,
        context.global_parent_id,
        context.observed_mask,
    )
    all_difference = (projected_field - self_field).abs()
    hidden_rows = context.hidden_mask
    observed_rows = context.observed_mask
    return {
        "observed_voxel_count": int(observed_rows.sum().item()),
        "hidden_voxel_count": int(hidden_rows.sum().item()),
        "number_of_global_parents": int(
            torch.unique(context.global_parent_id[hidden_rows]).numel()
        ),
        "mean_abs_Y_H_minus_Y_G": {
            "all": _channel_mean_abs_dict(self_field, global_field),
            "observed": _channel_mean_abs_dict(
                self_field[observed_rows], global_field[observed_rows]
            )
            if bool(observed_rows.any().item())
            else {name: 0.0 for name in PBR_CHANNEL_NAMES},
            "hidden": _channel_mean_abs_dict(
                self_field[hidden_rows], global_field[hidden_rows]
            )
            if bool(hidden_rows.any().item())
            else {name: 0.0 for name in PBR_CHANNEL_NAMES},
        },
        "mean_abs_Y_projected_minus_Y_H": {
            "all": _channel_mean_abs_dict(projected_field, self_field),
            "observed": _channel_mean_abs_dict(
                projected_field[observed_rows], self_field[observed_rows]
            )
            if bool(observed_rows.any().item())
            else {name: 0.0 for name in PBR_CHANNEL_NAMES},
            "hidden": _channel_mean_abs_dict(
                projected_field[hidden_rows], self_field[hidden_rows]
            )
            if bool(hidden_rows.any().item())
            else {name: 0.0 for name in PBR_CHANNEL_NAMES},
        },
        "coarse_residual_before": hidden_before["E_R"],
        "coarse_residual_after": hidden_after["E_R"],
        "null_preservation_error": hidden_after["E_N"],
        "E_R": {
            "observed": observed_after["E_R"],
            "hidden": hidden_after["E_R"],
        },
        "E_N": {
            "observed": observed_after["E_N"],
            "hidden": hidden_after["E_N"],
        },
        "E_detail": {
            "observed": observed_after["E_detail"],
            "hidden": hidden_after["E_detail"],
        },
        "per_channel_correction_magnitude": _channel_mean_abs_dict(
            projected_field, self_field
        ),
        "pbr_range": {
            "self_HR": _channel_range_stats(self_field),
            "global_G": _channel_range_stats(global_field),
            "current_cross_tile_fused": _channel_range_stats(current_cross_tile_fused),
            "projected": _channel_range_stats(projected_field),
        },
        "range_null": dict(range_null_stats),
    }


@torch.no_grad()
def _run_range_null_guided_flow(
    *,
    contexts: Sequence[TileContext],
    pipeline: Any,
    global_camera: Mapping[str, float],
    texture_params: Mapping[str, Any],
    pbr_encoder: torch.nn.Module,
    args: argparse.Namespace,
    output_root: Path,
    step_limit: Optional[int] = None,
    write_endpoint: bool = True,
) -> Dict[str, Any]:
    """Run the official texture sampler with hidden range-null projection."""
    if not contexts:
        raise RuntimeError("range-null flow requires at least one tile")
    sampler = pipeline.tex_slat_sampler
    model = pipeline.models["tex_slat_flow_model_1024"]
    merged = {**pipeline.tex_slat_sampler_params, **dict(texture_params)}
    schedule = base._native_schedule(sampler, merged)
    start_index = base._schedule_start(schedule, float(args.noise_timestep))
    step_kwargs = base._sampler_step_kwargs(merged)
    pairs = list(zip(schedule[start_index:-1], schedule[start_index + 1 :]))
    if step_limit is not None:
        if int(step_limit) <= 0:
            raise ValueError("step_limit must be positive")
        pairs = pairs[: int(step_limit)]
    states: Dict[int, SparseTensor] = {
        int(context.tile_id): base._fresh_sparse(context.initial_state)
        for context in contexts
    }
    fixed_shape_digest = {
        int(context.tile_id): base._coordinate_digest(context.shape_norm)
        for context in contexts
    }
    output_root.mkdir(parents=True, exist_ok=True)
    per_step: List[Dict[str, Any]] = []
    started = time.perf_counter()
    low_vram = bool(args.low_vram)
    if low_vram:
        model.to(torch.device("cuda"))
    try:
        for local_step, (t, t_next) in enumerate(pairs):
            step_index = int(start_index + local_step)
            step_started = time.perf_counter()
            print(
                f"[range-null step {step_index:02d}] t={float(t):.9f} "
                f"t_next={float(t_next):.9f} tiles={len(contexts)}"
            )

            # Phase A: official pred_x0/pred_v for every tile before any decode.
            prediction_started = time.perf_counter()
            predictions: Dict[int, Dict[str, Any]] = {}
            for context in contexts:
                tile_id = int(context.tile_id)
                state = states[tile_id]
                model_state = base._sparse_to_device(state, torch.device("cuda")) if low_vram else state
                shape_condition = (
                    base._sparse_to_device(context.shape_norm, torch.device("cuda"))
                    if low_vram
                    else context.shape_norm
                )
                condition = base._move_condition(context.condition, torch.device("cuda"))
                try:
                    pred_x0, _, pred_v = sampler._get_model_prediction(
                        model,
                        model_state,
                        float(t),
                        cond=condition["cond"],
                        neg_cond=condition["neg_cond"],
                        concat_cond=shape_condition,
                        **step_kwargs,
                    )
                finally:
                    del condition, shape_condition
                if not isinstance(pred_x0, SparseTensor) or not isinstance(pred_v, SparseTensor):
                    raise RuntimeError(f"tile {tile_id}: official prediction is not SparseTensor")
                pred_check = base._strict_sparse_check(
                    model_state, pred_x0, f"tile {tile_id} step {step_index} pred_x0"
                )
                velocity_check = base._strict_sparse_check(
                    model_state, pred_v, f"tile {tile_id} step {step_index} pred_v"
                )
                predictions[tile_id] = {
                    "pred_x0": base._sparse_to_cpu(pred_x0) if low_vram else pred_x0,
                    "pred_v": base._sparse_to_cpu(pred_v) if low_vram else pred_v,
                    "pred_check": pred_check,
                    "velocity_check": velocity_check,
                }
                del model_state, pred_x0, pred_v
                if low_vram:
                    _empty_cuda_cache()
            prediction_barrier = True

            # Phase B: decode the frozen pred_x0 endpoint for every tile.
            decode_started = time.perf_counter()
            decoded: Dict[int, MeshWithVoxel] = {}
            decoded_fields: Dict[int, torch.Tensor] = {}
            decode_stats: Dict[int, Dict[str, Any]] = {}
            for context in contexts:
                tile_id = int(context.tile_id)
                decode_shape = (
                    base._sparse_to_device(context.shape_denorm, torch.device("cuda"))
                    if low_vram
                    else context.shape_denorm
                )
                decode_texture = (
                    base._sparse_to_device(predictions[tile_id]["pred_x0"], torch.device("cuda"))
                    if low_vram
                    else predictions[tile_id]["pred_x0"]
                )
                decode_points = context.target_points.to(device="cuda") if low_vram else context.target_points
                mesh, field, stats = base._decode_endpoint(
                    pipeline=pipeline,
                    shape_denorm=decode_shape,
                    texture_norm=decode_texture,
                    query_points=decode_points,
                    query_chunk_size=int(args.query_chunk_size),
                    label=f"tile {tile_id:02d} step {step_index:02d} pred_x0",
                )
                if low_vram:
                    mesh_cpu = mesh.to("cpu")
                    field_cpu = field.detach().cpu().clone()
                    del mesh, field
                    mesh, field = mesh_cpu, field_cpu
                if not torch.isfinite(field).all():
                    raise RuntimeError(f"tile {tile_id} step {step_index}: decoded field is non-finite")
                decoded[tile_id] = mesh
                decoded_fields[tile_id] = field
                decode_stats[tile_id] = stats
                del decode_shape, decode_texture, decode_points
                if low_vram:
                    _empty_cuda_cache()
            decode_barrier = True

            # Phase C: current Gaussian fusion is used only on observed rows;
            # hidden rows receive the range-null projection from the immutable
            # global baseline reference.  No global endpoint or velocity enters.
            fusion_started = time.perf_counter()
            fused_fields: Dict[int, torch.Tensor] = {}
            projected_fields: Dict[int, torch.Tensor] = {}
            fusion_stats_by_tile: Dict[int, Dict[str, Any]] = {}
            projection_records: Dict[int, Dict[str, Any]] = {}
            snapshot_paths: Dict[int, str] = {}
            for context in contexts:
                tile_id = int(context.tile_id)
                self_field = decoded_fields[tile_id]
                fused_field, fusion_stats, donor_details = _fuse_observed_field_only(
                    target=context,
                    contexts=contexts,
                    decoded=decoded,
                    self_field=self_field,
                    global_camera=global_camera,
                    sigma_pixels=float(args.fusion_sigma_pixels),
                    query_chunk_size=int(args.query_chunk_size),
                )
                if bool(args.save_donor_details):
                    base._save_fusion_details(
                        output_root
                        / "tiles"
                        / f"tile_{tile_id:02d}"
                        / "steps"
                        / f"step_{step_index:02d}_observed_donors.pt",
                        donor_details,
                        {
                            "format": FORMAT,
                            "step": step_index,
                            "target_tile": tile_id,
                            "projection_region": "observed_only_after_projection",
                            "G_guidance_used": False,
                        },
                    )
                if (
                    context.global_pbr_reference is None
                    or context.hidden_mask is None
                    or context.observed_mask is None
                    or context.global_parent_id is None
                ):
                    raise RuntimeError(f"tile {tile_id}: incomplete range-null context")
                hidden_projected, range_stats, _ = range_null_project(
                    self_field,
                    context.global_pbr_reference,
                    context.global_parent_id,
                    context.hidden_mask,
                    tolerance=float(args.invariant_tolerance),
                )
                projected_field = torch.where(
                    context.observed_mask[:, None],
                    fused_field,
                    hidden_projected,
                )
                if not torch.isfinite(projected_field).all():
                    raise RuntimeError(f"tile {tile_id} step {step_index}: projected field is non-finite")
                fused_fields[tile_id] = fused_field
                projected_fields[tile_id] = projected_field
                fusion_stats_by_tile[tile_id] = fusion_stats
                projection_records[tile_id] = _projection_step_record(
                    context=context,
                    self_field=self_field,
                    global_field=context.global_pbr_reference,
                    current_cross_tile_fused=fused_field,
                    projected_field=projected_field,
                    range_null_stats=range_stats,
                )
                if step_index in {0, 6, 11} or step_limit is not None:
                    snapshot_paths[tile_id] = _save_projection_snapshot(
                        context,
                        output_root,
                        step_index,
                        self_field,
                        fused_field,
                        projected_field,
                    )
            fusion_barrier = True

            # Phase D: official self/projected PBR encodes on the exact local
            # geometry support.  The latent residual is the only flow guidance.
            encode_started = time.perf_counter()
            encoded: Dict[int, Dict[str, Any]] = {}
            for context in contexts:
                tile_id = int(context.tile_id)
                cycle_raw, cycle_stats = core._encode_local_pbr(
                    encoder=pbr_encoder,
                    coords=context.geometry.coords,
                    attrs=decoded_fields[tile_id],
                    device=torch.device("cuda"),
                    low_vram=low_vram,
                )
                projected_raw, projected_stats = core._encode_local_pbr(
                    encoder=pbr_encoder,
                    coords=context.geometry.coords,
                    attrs=projected_fields[tile_id],
                    device=torch.device("cuda"),
                    low_vram=low_vram,
                )
                cycle_norm = base._normalize_slat(cycle_raw, pipeline.tex_slat_normalization)
                projected_norm = base._normalize_slat(projected_raw, pipeline.tex_slat_normalization)
                if low_vram:
                    cycle_norm = base._sparse_to_cpu(cycle_norm)
                    projected_norm = base._sparse_to_cpu(projected_norm)
                pred_x0 = predictions[tile_id]["pred_x0"]
                cycle_check = base._strict_sparse_check(
                    pred_x0, cycle_norm, f"tile {tile_id} step {step_index} x0_cycle"
                )
                projected_check = base._strict_sparse_check(
                    pred_x0, projected_norm, f"tile {tile_id} step {step_index} x0_projected"
                )
                encoded[tile_id] = {
                    "cycle_norm": cycle_norm,
                    "projected_norm": projected_norm,
                    "cycle_stats": cycle_stats,
                    "projected_stats": projected_stats,
                    "cycle_check": cycle_check,
                    "projected_check": projected_check,
                }
                del cycle_raw, projected_raw
            encode_barrier = True

            # Phase E: cycle-cancelled endpoint correction and official velocity.
            correction_started = time.perf_counter()
            corrected: Dict[int, Dict[str, Any]] = {}
            for context in contexts:
                tile_id = int(context.tile_id)
                pred_x0 = predictions[tile_id]["pred_x0"]
                pred_v = predictions[tile_id]["pred_v"]
                cycle_norm = encoded[tile_id]["cycle_norm"]
                projected_norm = encoded[tile_id]["projected_norm"]
                model_state = (
                    base._sparse_to_device(states[tile_id], torch.device("cuda"))
                    if low_vram
                    else states[tile_id]
                )
                if low_vram:
                    pred_x0 = base._sparse_to_device(pred_x0, torch.device("cuda"))
                    pred_v = base._sparse_to_device(pred_v, torch.device("cuda"))
                    cycle_norm = base._sparse_to_device(cycle_norm, torch.device("cuda"))
                    projected_norm = base._sparse_to_device(projected_norm, torch.device("cuda"))
                delta = projected_norm.feats - cycle_norm.feats
                guided_x0 = SparseTensor(
                    pred_x0.feats + float(args.eta) * delta,
                    pred_x0.coords.detach().clone(),
                )
                guided_x0_check = base._strict_sparse_check(
                    pred_x0, guided_x0, f"tile {tile_id} step {step_index} x0_guided"
                )
                guided_v = sampler._xstart_to_pred(model_state, float(t), guided_x0)
                guided_v_check = base._strict_sparse_check(
                    pred_v, guided_v, f"tile {tile_id} step {step_index} guided_v"
                )
                next_state = SparseTensor(
                    model_state.feats - float(t - t_next) * guided_v.feats,
                    model_state.coords.detach().clone(),
                )
                next_check = base._strict_sparse_check(
                    model_state, next_state, f"tile {tile_id} step {step_index} x_t_next"
                )
                corrected[tile_id] = {
                    "next_state": base._sparse_to_cpu(next_state) if low_vram else next_state,
                    "pred_x0": base._sparse_to_cpu(pred_x0) if low_vram else pred_x0,
                    "pred_v": base._sparse_to_cpu(pred_v) if low_vram else pred_v,
                    "cycle_norm": base._sparse_to_cpu(cycle_norm) if low_vram else cycle_norm,
                    "projected_norm": base._sparse_to_cpu(projected_norm) if low_vram else projected_norm,
                    "guided_x0": base._sparse_to_cpu(guided_x0) if low_vram else guided_x0,
                    "guided_v": base._sparse_to_cpu(guided_v) if low_vram else guided_v,
                    "guided_x0_check": guided_x0_check,
                    "guided_v_check": guided_v_check,
                    "next_check": next_check,
                }
                del model_state, pred_x0, pred_v, cycle_norm, projected_norm, delta, guided_x0, guided_v, next_state
                if low_vram:
                    _empty_cuda_cache()
            correction_barrier = True

            # Phase F: synchronized Jacobi Euler update.
            for context in contexts:
                states[int(context.tile_id)] = corrected[int(context.tile_id)]["next_state"]
            update_barrier = True
            for context in contexts:
                tile_id = int(context.tile_id)
                if not torch.equal(states[tile_id].coords, context.initial_state.coords):
                    raise RuntimeError(f"tile {tile_id}: state support changed after Euler update")

            step_tile_records: List[Dict[str, Any]] = []
            for context in contexts:
                tile_id = int(context.tile_id)
                row = corrected[tile_id]
                record = {
                    "tile_id": tile_id,
                    "step": step_index,
                    "t": float(t),
                    "t_next": float(t_next),
                    "active_ovoxel_count": int(context.target_coords.shape[0]),
                    "projection": projection_records[tile_id],
                    "snapshot": snapshot_paths.get(tile_id),
                    "gaussian_fusion": fusion_stats_by_tile[tile_id],
                    "norm_pred_x0": base._norm(row["pred_x0"].feats),
                    "norm_x0_projected_minus_x0_cycle": base._norm(
                        row["projected_norm"].feats - row["cycle_norm"].feats
                    ),
                    "norm_x0_guided_minus_pred_x0": base._norm(
                        row["guided_x0"].feats - row["pred_x0"].feats
                    ),
                    "norm_guided_v_minus_pred_v": base._norm(
                        row["guided_v"].feats - row["pred_v"].feats
                    ),
                    "support_checks": {
                        "pred_x0": predictions[tile_id]["pred_check"],
                        "pred_v": predictions[tile_id]["velocity_check"],
                        "x0_cycle": encoded[tile_id]["cycle_check"],
                        "x0_projected": encoded[tile_id]["projected_check"],
                        "x0_guided": row["guided_x0_check"],
                        "guided_v": row["guided_v_check"],
                        "x_t_next": row["next_check"],
                    },
                    "fixed_shape_coord_digest": fixed_shape_digest[tile_id],
                    "fixed_shape_unchanged": fixed_shape_digest[tile_id]
                    == base._coordinate_digest(context.shape_norm),
                    "decode": decode_stats[tile_id],
                    "encode": {
                        "cycle": encoded[tile_id]["cycle_stats"],
                        "projected": encoded[tile_id]["projected_stats"],
                    },
                }
                _atomic_json(
                    output_root
                    / "tiles"
                    / f"tile_{tile_id:02d}"
                    / "steps"
                    / f"step_{step_index:02d}_diagnostics.json",
                    record,
                )
                step_tile_records.append(record)
            step_summary = {
                "step": step_index,
                "t": float(t),
                "t_next": float(t_next),
                "tile_count": int(len(contexts)),
                "step_seconds": float(time.perf_counter() - step_started),
                "barriers": {
                    "prediction_barrier": prediction_barrier,
                    "decoded_field_barrier": decode_barrier,
                    "fusion_barrier": fusion_barrier,
                    "encode_barrier": encode_barrier,
                    "endpoint_correction_barrier": correction_barrier,
                    "euler_update_barrier": update_barrier,
                    "all_tiles_synchronized": all(
                        [prediction_barrier, decode_barrier, fusion_barrier, encode_barrier, correction_barrier, update_barrier]
                    ),
                },
                "phase_seconds": {
                    "prediction": float(decode_started - prediction_started),
                    "decode": float(fusion_started - decode_started),
                    "fusion": float(encode_started - fusion_started),
                    "encode": float(correction_started - encode_started),
                    "correction_and_euler": float(time.perf_counter() - correction_started),
                },
                "tiles": step_tile_records,
            }
            _atomic_json(output_root / "steps" / f"step_{step_index:02d}_summary.json", step_summary)
            per_step.append(step_summary)
            del predictions, decoded, decoded_fields, fused_fields, projected_fields
            del fusion_stats_by_tile, projection_records, encoded, corrected
            _empty_cuda_cache()
    finally:
        if low_vram:
            model.cpu()
    if write_endpoint:
        for context in contexts:
            context.range_null_endpoint = states[int(context.tile_id)]
            _save_sparse_payload = base._save_sparse_payload
            _save_sparse_payload(
                output_root
                / "tiles"
                / f"tile_{int(context.tile_id):02d}"
                / "range_null_guided_endpoint.pt",
                context.range_null_endpoint,
                pipeline.tex_slat_normalization,
            )
    return {
        "route": (
            "official pred_x0/pred_v -> frozen HR decode barrier -> observed-only Gaussian fusion + "
            "hidden global-parent range-null projection -> official self/projected PBR encode -> "
            "cycle-cancelled x0 residual -> official _xstart_to_pred -> synchronized Euler"
        ),
        "native_schedule": schedule,
        "schedule_start_index": int(start_index),
        "noise_timestep": float(args.noise_timestep),
        "noise_strength": float(args.noise_strength),
        "flow_steps": int(len(pairs)),
        "model_forward_count": int(len(pairs) * len(contexts)),
        "tile_count": int(len(contexts)),
        "eta": float(args.eta),
        "fusion_sigma_pixels": float(args.fusion_sigma_pixels),
        "flow_seconds": float(time.perf_counter() - started),
        "all_tiles_synchronized_per_step": all(
            bool(step["barriers"]["all_tiles_synchronized"]) for step in per_step
        ),
        "steps": per_step,
        "shape_flow_called": False,
        "shape_sampler_called": False,
        "G_guidance_used": False,
        "G_velocity_used": False,
        "G_timestep_trajectory_used": False,
        "latent_low_high_frequency_mixing_used": False,
        "weighted_G_HR_blend_used": False,
        "velocity_averaging_used": False,
        "preflight_step_limit": int(step_limit) if step_limit is not None else None,
    }


def _variant_patch_and_stitch(
    *,
    variant: str,
    endpoint_attr: str,
    contexts: Sequence[TileContext],
    pipeline: Any,
    global_camera: Mapping[str, float],
    baseline_mesh: MeshWithVoxel,
    args: argparse.Namespace,
    output_root: Path,
) -> Tuple[MeshWithVertexPbr, Dict[str, Any]]:
    """Decode range-null endpoints and use the established stitcher unchanged."""
    patches: List[core.ReturnedTilePatch] = []
    tile_records: List[Dict[str, Any]] = []
    for index, context in enumerate(contexts):
        endpoint = getattr(context, endpoint_attr, None)
        if endpoint is None:
            raise RuntimeError(f"tile {context.tile_id}: missing endpoint {endpoint_attr}")
        print(
            f"[{variant}] final decode tile {context.tile_id:02d} "
            f"({index + 1}/{len(contexts)})"
        )
        low_vram = bool(args.low_vram)
        decode_shape = (
            base._sparse_to_device(context.shape_denorm, torch.device("cuda"))
            if low_vram
            else context.shape_denorm
        )
        decode_texture = (
            base._sparse_to_device(endpoint, torch.device("cuda"))
            if low_vram
            else endpoint
        )
        decode_points = (
            context.target_points[:0].to(device="cuda")
            if low_vram
            else context.target_points[:0]
        )
        mesh, _, decode_stats = base._decode_endpoint(
            pipeline=pipeline,
            shape_denorm=decode_shape,
            texture_norm=decode_texture,
            query_points=decode_points,
            query_chunk_size=int(args.query_chunk_size),
            label=f"tile {context.tile_id:02d} final {variant}",
        )
        patch = core._local_mesh_to_global_patch(
            tile_id=int(context.tile_id),
            box=context.box,
            local_mesh=mesh,
            global_camera=global_camera,
            transform=context.transform,
            query_chunk_size=int(args.query_chunk_size),
        )
        patches.append(patch)
        tile_records.append(
            {
                "tile_id": int(context.tile_id),
                "box": list(context.box),
                "decode": decode_stats,
                "returned_global_patch": patch.stats,
            }
        )
        del mesh, decode_shape, decode_texture, decode_points
        _empty_cuda_cache()
    stitched, stitch_stats = core._stitch_tile_patches_nearest(
        patches,
        layout=dict(baseline_mesh.layout),
        global_camera=global_camera,
        face_chunk_size=int(args.face_projection_chunk_size),
        weld_tolerance=float(args.stitch_tolerance),
    )
    variant_dir = output_root / "variants" / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
        variant_dir / "global_merged_mesh.pt",
        {
            "format": f"{FORMAT}_{variant}_global_mesh",
            "variant": variant,
            "mesh": stitched,
            "stitch_stats": stitch_stats,
            "tile_records": tile_records,
        },
    )
    exported_patch = core.ReturnedTilePatch(
        tile_id=-1,
        box=(0, 0, CANONICAL_IMAGE_SIZE, CANONICAL_IMAGE_SIZE),
        vertices=stitched.vertices,
        faces=stitched.faces,
        vertex_attrs=stitched.vertex_attrs,
        stats=stitch_stats,
    )
    glb_stats = core._export_tiled_glb(
        [exported_patch], variant_dir / "global_merged_mesh.glb"
    )
    summary = {
        "variant": variant,
        "vertices": int(stitched.vertices.shape[0]),
        "faces": int(stitched.faces.shape[0]),
        "tile_count": int(len(patches)),
        "tile_records": tile_records,
        "stitch": stitch_stats,
        "glb": glb_stats,
        "mesh_pt": str((variant_dir / "global_merged_mesh.pt").resolve()),
        "mesh_glb": str((variant_dir / "global_merged_mesh.glb").resolve()),
    }
    _atomic_json(variant_dir / "global_variant_summary.json", summary)
    return stitched, summary


def _reference_variant_path(reference_dir: Optional[Path], variant: str) -> Optional[Path]:
    if reference_dir is None:
        return None
    source_name = REFERENCE_VARIANT_NAMES.get(variant, variant)
    path = reference_dir / "variants" / source_name / "global_merged_mesh.pt"
    return path if path.is_file() else None


def _load_or_make_reference_variant(
    *,
    variant: str,
    endpoint_attr: str,
    contexts: Sequence[TileContext],
    pipeline: Any,
    global_camera: Mapping[str, float],
    baseline_mesh: MeshWithVoxel,
    args: argparse.Namespace,
    output_root: Path,
    reference_dir: Optional[Path],
) -> Tuple[Any, Dict[str, Any]]:
    source_path = _reference_variant_path(reference_dir, variant)
    if source_path is not None:
        mesh = _load_mesh_any(source_path)
        source_summary_path = source_path.with_name("global_variant_summary.json")
        source_summary = (
            json.loads(source_summary_path.read_text(encoding="utf-8"))
            if source_summary_path.is_file()
            else {}
        )
        return mesh, {
            "variant": variant,
            "source": "existing successful variant read without rerun",
            "source_path": str(source_path.resolve()),
            "source_summary": source_summary,
            "vertices": int(mesh.vertices.shape[0]),
            "faces": int(mesh.faces.shape[0]),
        }
    mesh, summary = _variant_patch_and_stitch(
        variant=variant,
        endpoint_attr=endpoint_attr,
        contexts=contexts,
        pipeline=pipeline,
        global_camera=global_camera,
        baseline_mesh=baseline_mesh,
        args=args,
        output_root=output_root,
    )
    summary["source"] = "generated in this experiment because no reusable variant was found"
    return mesh, summary


def _frame_to_image(frame: Any) -> Image.Image:
    array = np.asarray(frame)
    if array.dtype.kind == "f":
        if array.size and float(array.max()) <= 1.0:
            array = array * 255.0
        array = array.clip(0.0, 255.0).astype(np.uint8)
    else:
        array = array.clip(0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] >= 3:
        array = array[..., :3]
    return Image.fromarray(array, mode="RGB")


def _make_contact_sheet(
    paths: Sequence[Path],
    labels: Sequence[str],
    output_path: Path,
    *,
    panel: int = 320,
    columns: int = 3,
) -> None:
    if not paths:
        return
    header = 40
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * panel, rows * (panel + header)), "black")
    draw = ImageDraw.Draw(sheet)
    for index, (path, label) in enumerate(zip(paths, labels)):
        with Image.open(path) as source:
            image = ImageOps.contain(source.convert("RGB"), (panel - 8, panel - 8))
        x0 = (index % columns) * panel
        y0 = (index // columns) * (panel + header)
        sheet.paste(image, (x0 + (panel - image.width) // 2, y0 + header + (panel - image.height) // 2))
        draw.text((x0 + 5, y0 + 10), label, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _render_aligned_variants(
    *,
    meshes: Mapping[str, Any],
    baseline_mesh: MeshWithVoxel,
    global_camera: Mapping[str, float],
    output_root: Path,
    canonical_4096_path: Path,
    args: argparse.Namespace,
    envmap: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    records: Dict[str, Any] = {}
    table: List[Dict[str, Any]] = []
    for variant, mesh in meshes.items():
        render_dir = output_root / "variants" / variant / "aligned_eval_4096"
        print(f"[render aligned] {variant} resolution={int(args.render_resolution)}")
        render_record = core._render(
            mesh,
            output_dir=render_dir,
            camera=global_camera,
            reference_image=canonical_4096_path,
            args=args,
            envmap=envmap,
        )
        records[variant] = render_record
        metrics = core._metric_subset(render_record)
        table.append(
            {
                "variant": variant,
                "vertices": int(mesh.vertices.shape[0]),
                "faces": int(mesh.faces.shape[0]),
                "PSNR": metrics["psnr_db"],
                "SSIM": metrics["ssim"],
                "LPIPS": metrics["lpips"],
                "render_resolution": int(args.render_resolution),
            }
        )
    comparison_paths: List[Path] = []
    comparison_labels: List[str] = []
    for variant in meshes:
        path = Path(str(records[variant]["render_png"]))
        comparison_paths.append(path)
        comparison_labels.append(variant)
    comparison_path = output_root / "aligned_4096_four_variant_comparison.png"
    _make_contact_sheet(comparison_paths, comparison_labels, comparison_path, panel=512, columns=2)
    records["four_variant_contact_sheet"] = str(comparison_path.resolve())
    return records, table


def _render_multiview_variants(
    *,
    meshes: Mapping[str, Any],
    baseline_mesh: MeshWithVoxel,
    global_camera: Mapping[str, float],
    output_root: Path,
    args: argparse.Namespace,
    envmap: Any,
) -> Dict[str, Any]:
    """Render front/right/left/back/top/bottom and a 24-frame turntable."""
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
    radius = float(global_camera["distance"]) * float(args.multiview_radius_scale)
    fov = torch.tensor(float(global_camera["camera_angle_x"]), device=device)
    intrinsic = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
    target = torch.zeros(3, device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    extrinsics: List[torch.Tensor] = []
    intrinsics: List[torch.Tensor] = []
    labels: List[str] = []
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
    options = {
        "resolution": int(args.multiview_resolution),
        "near": max(0.01, radius - 2.0),
        "far": radius + 10.0,
        "ssaa": int(args.multiview_ssaa),
        "peel_layers": int(args.multiview_peel_layers),
        "face_chunk_size": int(args.render_face_chunk_size),
    }
    renderer = render_utils.get_renderer(baseline_mesh, **options)
    root = output_root / "multiview_4variants"
    root.mkdir(parents=True, exist_ok=True)
    records: Dict[str, Any] = {
        "enabled": True,
        "renderer": "Pixal3D render_utils / PbrMeshRenderer / nvdiffrast",
        "fixed_views": [item[0] for item in fixed],
        "turntable_frames": turntable_count,
        "resolution": int(args.multiview_resolution),
        "variants": {},
    }
    shaded_back_paths: List[Path] = []
    shaded_back_labels: List[str] = []
    pbr_contact_paths: Dict[str, List[Path]] = {name: [] for name in PBR_CHANNEL_NAMES}
    pbr_contact_labels: Dict[str, List[str]] = {name: [] for name in PBR_CHANNEL_NAMES}
    for variant, mesh in meshes.items():
        print(f"[render multiview] {variant} frames={len(specs)}")
        live = mesh.to(device)
        rendered = render_utils.render_frames(
            live,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            options=options,
            verbose=True,
            renderer=renderer,
            envmap=envmap,
            use_envmap_bg=bool(args.use_envmap_bg),
        )
        del live
        _empty_cuda_cache()
        variant_dir = root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_record: Dict[str, Any] = {"frames": {}}
        shaded_paths: List[Path] = []
        for index, label in enumerate(labels):
            frame_record: Dict[str, str] = {}
            for channel in ("shaded", "base_color", "metallic", "roughness", "alpha"):
                if channel not in rendered:
                    continue
                path = variant_dir / f"view_{index:03d}_{channel}.png"
                _frame_to_image(rendered[channel][index]).save(path)
                frame_record[channel] = str(path.resolve())
                contact_channel = "RGB" if channel == "base_color" else channel
                if index in (0, 3) and contact_channel in pbr_contact_paths:
                    pbr_contact_paths[contact_channel].append(path)
                    pbr_contact_labels[contact_channel].append(f"{variant}/{label.split()[0]}")
            if "shaded" in rendered:
                shaded_paths.append(Path(frame_record["shaded"]))
            variant_record["frames"][str(index)] = frame_record
        sheet = variant_dir / "shaded_multiview_sheet.png"
        _make_contact_sheet(shaded_paths, labels, sheet, panel=320, columns=3)
        turntable_paths = shaded_paths[len(fixed) :]
        gif_path = variant_dir / "shaded_turntable_24.gif"
        if turntable_paths:
            with Image.open(turntable_paths[0]) as first:
                first.copy().save(
                    gif_path,
                    save_all=True,
                    append_images=[Image.open(path).copy() for path in turntable_paths[1:]],
                    duration=100,
                    loop=0,
                )
        variant_record["shaded_sheet"] = str(sheet.resolve())
        variant_record["turntable_gif"] = str(gif_path.resolve()) if gif_path.is_file() else None
        records["variants"][variant] = variant_record
        if len(shaded_paths) > 3:
            shaded_back_paths.append(shaded_paths[3])
            shaded_back_labels.append(variant)
    back_sheet = root / "back_view_contact_sheet_four_variants.png"
    _make_contact_sheet(shaded_back_paths, shaded_back_labels, back_sheet, panel=512, columns=2)
    records["back_view_contact_sheet"] = str(back_sheet.resolve())
    pbr_sheets: Dict[str, str] = {}
    for channel, paths in pbr_contact_paths.items():
        if not paths:
            continue
        path = root / f"{channel}_front_back_contact_sheet.png"
        _make_contact_sheet(paths, pbr_contact_labels[channel], path, panel=320, columns=2)
        pbr_sheets[channel] = str(path.resolve())
    records["pbr_front_back_contact_sheets"] = pbr_sheets
    _atomic_json(root / "multiview_summary.json", records)
    return records


def _write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "vertices", "faces", "PSNR", "SSIM", "LPIPS", "render_resolution"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _aggregate_projection_metrics(flow_stats: Mapping[str, Any]) -> Dict[str, Any]:
    """Aggregate per-tile E_R/E_N/detail curves without hiding region labels."""
    by_step: Dict[int, List[Mapping[str, Any]]] = {}
    for step in flow_stats.get("steps", []):
        by_step[int(step["step"])] = list(step.get("tiles", []))

    def mean_channel(rows: Sequence[Mapping[str, Any]], path: Sequence[str], channel: str) -> float:
        values: List[float] = []
        for row in rows:
            value: Any = row["projection"]
            for key in path:
                value = value[key]
            if channel in value:
                values.append(float(value[channel]))
        return float(np.mean(values)) if values else 0.0

    curves: List[Dict[str, Any]] = []
    for step_index in sorted(by_step):
        rows = by_step[step_index]
        regions: Dict[str, Any] = {}
        for region in ("observed", "hidden"):
            regions[region] = {
                "E_R": {
                    channel: mean_channel(rows, ("E_R", region), channel)
                    for channel in PBR_CHANNEL_NAMES
                },
                "E_N": {
                    channel: mean_channel(rows, ("E_N", region), channel)
                    for channel in PBR_CHANNEL_NAMES
                },
                "E_detail_output": {
                    channel: mean_channel(rows, ("E_detail", region, "output"), channel)
                    for channel in PBR_CHANNEL_NAMES
                },
                "E_detail_self_HR": {
                    channel: mean_channel(rows, ("E_detail", region, "self_HR"), channel)
                    for channel in PBR_CHANNEL_NAMES
                },
                "E_detail_global_G": {
                    channel: mean_channel(rows, ("E_detail", region, "global_G"), channel)
                    for channel in PBR_CHANNEL_NAMES
                },
            }
        curves.append({"step": int(step_index), "tile_count": len(rows), "regions": regions})
    return {
        "steps": curves,
        "definition": {
            "E_R": "abs(A Y_out - A Y_G), parent means",
            "E_N": "abs((I-A^dagger A)Y_out - (I-A^dagger A)Y_H)",
            "E_detail": "RMS per channel of (I-A^dagger A)Y",
            "regions": ["observed", "hidden"],
            "channels": list(PBR_CHANNEL_NAMES),
        },
    }


def _write_range_report(output_dir: Path, summary: Mapping[str, Any]) -> Path:
    correctness = summary.get("correctness_test", {})
    guidance = summary.get("guidance", {})
    evaluation = summary.get("evaluation", {})
    table = evaluation.get("table", []) if isinstance(evaluation, Mapping) else []
    visibility = summary.get("visibility", {})
    projection = summary.get("new_metrics", {})
    lines = [
        "# PBR Range--Null Per-Step Experiment",
        "",
        "## 结论",
        "",
        "本实验保持 official texture prediction / decoder / PBR encoder / "
        "`_xstart_to_pred` / Euler 路线，只替换 hidden PBR field projection。",
        "",
        f"- CUDA: `{summary.get('cuda_device')}` / `{summary.get('cuda_name')}`；seed `{summary.get('seed')}`；texture steps `{summary.get('sampler', {}).get('texture_steps')}`。",
        f"- correctness preflight: `{correctness.get('passed')}`；flow invariant failure count: `{guidance.get('invariant_failure_count', 0)}`。",
        f"- observed voxels: `{visibility.get('observed_voxel_count')}`；hidden voxels: `{visibility.get('hidden_voxel_count')}`。",
        "- 背面/hidden 的视觉判断以 `back_view_contact_sheet_four_variants.png` 和各通道 front/back contact sheet 为准；脚本不会把视觉改善未经指标验证地写成成功。",
        "",
        "## 1. 实际的 A 与 A†",
        "",
        "`A` 是在 hidden mask 内按 `global_parent_id` 分组的 parent-mean reduction："
        "每个 parent 的六个 PBR 通道分别取均值。`A†` 是把 parent 向量广播回该 parent 的所有 hidden rows。",
        "代码中的实现是 `torch.unique(return_inverse=True)` + `torch.bincount` + "
        "`index_add_`，没有 Python 逐 voxel/逐 parent loop。",
        "",
        "## 2. Projector 性质与数值 invariant",
        "",
        "对同一 hidden support，`A†A` 是“组内均值后广播”的幂等 projector；"
        "因此 `A†A Y_G + (I-A†A)Y_H` 的 parent mean 等于 `Y_G`，且 HR 的组内 centered detail 不变。",
        f"correctness 结果：`{correctness.get('invariants')}`。",
        "",
        "## 3. Visibility 与 global parent",
        "",
        "canonical 4096 baseline mesh rasterization 生成 front position map；"
        "local support 通过既有 local→global camera round-trip 投影，使用由 C1024 voxel 半对角线和 canonical pixel footprint 推导的距离阈值。"
        "阈值是二值 visibility 判定，不是 foreground/background blending weight。",
        f"- raster artifact: `{visibility.get('raster', {}).get('mask_png')}`",
        f"- manual overview: `{visibility.get('manual_check_artifacts', {}).get('observed_hidden_overview')}`",
        "",
        "## 4. Variant 对比",
        "",
        "| variant | vertices | faces | PSNR | SSIM | LPIPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row.get('variant')} | {row.get('vertices')} | {row.get('faces')} | "
            f"{row.get('PSNR')} | {row.get('SSIM')} | {row.get('LPIPS')} |"
        )
    lines.extend(
        [
            "",
            "## 5. 新指标",
            "",
            "`E_R = |A Y_out - A Y_G|`，`E_N = |(I-A†A)Y_out - "
            "(I-A†A)Y_H|`，`E_detail = ||(I-A†A)Y||_2`；按 RGB / metallic / "
            "roughness / alpha 与 observed / hidden 分开记录。",
            f"完整曲线 JSON 位于 `{summary.get('artifacts', {}).get('projection_metrics')}`。",
            "",
            "## 6. 失败定位",
            "",
            f"当前自动诊断：`{summary.get('diagnosis', {})}`。若最终 hidden 高频仍不一致，"
            "应归入 G（coarse consistency 正确但 hidden tiles 高频仍不一致），下一阶段再研究 canonical stochastic coupling / atlas consensus；"
            "本实验不会自行重新加入 hidden Gaussian averaging。",
            "",
            "## 7. Artifacts",
            "",
            f"- summary: `{output_dir / 'summary.json'}`",
            f"- metrics: `{output_dir / 'metrics.csv'}`",
            f"- projection snapshots: `{output_dir / 'tiles'}`（step 0/6/11）",
            f"- report: `{output_dir / 'RANGE_NULL_PBR_EXPERIMENT.md'}`",
        ]
    )
    path = output_dir / "RANGE_NULL_PBR_EXPERIMENT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="assets/choose/0_img.png")
    parser.add_argument("--output-dir", default="outputs/pbr_range_null_perstep_cuda4")
    parser.add_argument("--reference-experiment-dir", default=None)
    parser.add_argument("--baseline-dir", default=None)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-ids", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--shape-encoder",
        default=str(core.DEFAULT_ENCODER_ROOT / "shape_enc_next_dc_f16c32_fp16"),
    )
    parser.add_argument(
        "--pbr-encoder",
        default=str(core.DEFAULT_ENCODER_ROOT / "tex_enc_next_dc_f16c32_fp16"),
    )
    parser.add_argument("--fov", type=float, default=-1.0)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument("--max-num-tokens", type=int, default=1_000_000)
    parser.add_argument("--face-projection-chunk-size", type=int, default=250_000)
    parser.add_argument("--visibility-face-chunk-size", type=int, default=250_000)
    parser.add_argument("--material-query-chunk-size", type=int, default=65_536)
    parser.add_argument("--material-face-chunk-size", type=int, default=16_384)
    parser.add_argument("--query-chunk-size", type=int, default=65_536)
    parser.add_argument("--roundtrip-tolerance", type=float, default=2e-5)
    parser.add_argument("--noise-timestep", type=float, default=1.0)
    parser.add_argument("--noise-strength", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--invariant-tolerance", type=float, default=5e-5)
    parser.add_argument("--fusion-sigma-pixels", type=float, default=256.0)
    parser.add_argument("--save-donor-details", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stitch-tolerance", type=float, default=1.0 / OVOXEL_RESOLUTION)

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

    parser.add_argument("--correctness-tile-count", type=int, default=4)
    parser.add_argument("--correctness-step-count", type=int, default=1)
    parser.add_argument("--skip-correctness", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--render-resolution", type=int, default=4096)
    parser.add_argument("--metric-resolution", type=int, default=1024)
    parser.add_argument("--render-ssaa", type=int, default=1)
    parser.add_argument("--render-peel-layers", type=int, default=8)
    parser.add_argument("--render-face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--use-envmap-bg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-lpips", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--render-multiview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multiview-resolution", type=int, default=512)
    parser.add_argument("--multiview-ssaa", type=int, default=1)
    parser.add_argument("--multiview-peel-layers", type=int, default=4)
    parser.add_argument("--multiview-radius-scale", type=float, default=1.0)
    parser.add_argument("--multiview-turntable-frames", type=int, default=24)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the range-null experiment")
    if int(args.cuda_device) < 0 or int(args.cuda_device) >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {args.cuda_device} is unavailable")
    if not Path(args.image).expanduser().is_file():
        raise FileNotFoundError(args.image)
    for encoder in (args.shape_encoder, args.pbr_encoder):
        base_path = Path(encoder).expanduser()
        if not Path(f"{base_path}.json").is_file() or not Path(f"{base_path}.safetensors").is_file():
            raise FileNotFoundError(f"encoder checkpoint pair not found for {base_path}")
    positive_names = (
        "max_num_tokens",
        "face_projection_chunk_size",
        "visibility_face_chunk_size",
        "material_query_chunk_size",
        "material_face_chunk_size",
        "query_chunk_size",
        "render_resolution",
        "metric_resolution",
        "render_ssaa",
        "render_peel_layers",
        "render_face_chunk_size",
        "multiview_resolution",
        "multiview_ssaa",
        "multiview_peel_layers",
        "multiview_turntable_frames",
        "correctness_tile_count",
        "correctness_step_count",
    )
    for name in positive_names:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args.texture_steps) != 12:
        raise ValueError("Codex.md fixes texture steps at 12")
    if not 0.0 <= float(args.noise_timestep) <= 1.0:
        raise ValueError("--noise-timestep must lie in [0,1]")
    if float(args.noise_strength) <= 0.0 or float(args.fusion_sigma_pixels) <= 0.0:
        raise ValueError("noise strength and fusion sigma must be positive")
    if float(args.invariant_tolerance) <= 0.0 or float(args.stitch_tolerance) <= 0.0:
        raise ValueError("invariant and stitch tolerances must be positive")
    if float(args.multiview_radius_scale) <= 0.0:
        raise ValueError("multiview radius scale must be positive")
    if not bool(args.skip_lpips) and importlib.util.find_spec("lpips") is None:
        print("[metrics] lpips is unavailable; continuing without LPIPS")
        args.skip_lpips = True
    if bool(args.low_vram):
        print("[warning] explicit --low-vram enabled; normal/full-VRAM mode is the default")


def _resolve_reference_dir(args: argparse.Namespace) -> Optional[Path]:
    if args.reference_experiment_dir:
        path = Path(args.reference_experiment_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"reference experiment directory not found: {path}")
        return path
    candidate = Path("outputs/cross_tile_pbr_perstep_guided_cuda4_full_staged").resolve()
    return candidate if candidate.is_dir() else None


def _check_correctness(stats: Mapping[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    if not bool(stats.get("all_tiles_synchronized_per_step")):
        errors.append("barrier synchronization failed")
    for step in stats.get("steps", []):
        for tile in step.get("tiles", []):
            projection = tile.get("projection", {})
            if not bool(projection.get("range_null", {}).get("passed", False)):
                errors.append(f"tile {tile.get('tile_id')} projection invariant failed")
            checks = tile.get("support_checks", {})
            for name in ("pred_x0", "pred_v", "x0_cycle", "x0_projected", "x0_guided", "guided_v", "x_t_next"):
                if checks.get(name, {}).get("coords_exact") is not True:
                    errors.append(f"tile {tile.get('tile_id')} support check {name} failed")
            if not math.isfinite(float(tile.get("norm_x0_projected_minus_x0_cycle", 0.0))):
                errors.append(f"tile {tile.get('tile_id')} non-finite encode delta")
    return {
        "passed": not errors,
        "errors": errors,
        "tile_count": int(stats.get("tile_count", 0)),
        "flow_steps": int(stats.get("flow_steps", 0)),
        "invariants": "coarse and null/detail invariants asserted per tile/step",
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _validate_args(args)
    torch.cuda.set_device(int(args.cuda_device))
    device = torch.device("cuda")
    print(
        f"[cuda] requested/current index={int(args.cuda_device)}/{torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())} "
        f"low_vram={bool(args.low_vram)}"
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(args.resume):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory {output_dir}; use a new directory or --resume"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    # Reused base helpers write step artifacts relative to args.output_dir;
    # make that path identical to the guarded absolute output directory.
    args.output_dir = str(output_dir)
    reference_dir = _resolve_reference_dir(args)
    if reference_dir is not None:
        print(f"[reference] reusing existing variants/context caches from {reference_dir}")
        if args.baseline_dir is None:
            args.baseline_dir = str(reference_dir)

    source_path = Path(args.image).expanduser().resolve()
    with Image.open(source_path) as source:
        source_rgb = source.convert("RGB")
    source_rgb.save(output_dir / "input_original.png")

    pipeline = core.init_pipeline(
        args.model_path,
        device="cuda",
        low_vram=bool(args.low_vram),
    )
    canonical = pipeline.preprocess_canonical_images(source_rgb)
    image_4096: Image.Image = canonical["image_4096"]
    image_1024: Image.Image = canonical["image_1024"]
    image_512: Image.Image = canonical["image_512"]
    image_4096.save(output_dir / "canonical_4096.png")
    image_1024.save(output_dir / "canonical_1024.png")
    image_512.save(output_dir / "canonical_512.png")
    canonical["foreground_mask_4096"].save(output_dir / "canonical_foreground_mask_4096.png")
    _atomic_json(output_dir / "canonical_metadata.json", canonical.get("metadata", {}))

    camera_path = output_dir / "global_camera.json"
    reference_camera_path = reference_dir / "global_camera.json" if reference_dir is not None else None
    if bool(args.resume) and camera_path.is_file():
        global_camera = json.loads(camera_path.read_text(encoding="utf-8"))
    elif reference_camera_path is not None and reference_camera_path.is_file():
        global_camera = json.loads(reference_camera_path.read_text(encoding="utf-8"))
        _atomic_json(camera_path, global_camera)
    else:
        global_camera = core._estimate_camera(
            image_1024=image_1024,
            output_dir=output_dir,
            manual_fov=float(args.fov),
            mesh_scale=float(args.mesh_scale),
            extend_pixel=int(args.extend_pixel),
            moge_model_path=None,
        )
        _atomic_json(camera_path, global_camera)

    baseline_mesh, baseline_summary = base._load_or_run_global_baseline(
        args=args,
        pipeline=pipeline,
        image_1024=image_1024,
        global_camera=global_camera,
        output_dir=output_dir,
    )
    baseline_mesh = baseline_mesh.to("cpu")
    _atomic_json(output_dir / "global_baseline_summary.json", baseline_summary)
    global_attr_field = core._make_attribute_query_mesh(baseline_mesh, device)

    boxes = core._tile_layout(
        canonical_size=CANONICAL_IMAGE_SIZE,
        tile_size=TILE_SIZE,
        stride=TILE_STRIDE,
    )
    if base._parse_ids(args.tile_ids) is None and len(boxes) != 49:
        raise RuntimeError(f"fixed 4096/1024/512 layout must contain 49 tiles, got {len(boxes)}")
    _atomic_json(
        output_dir / "tile_layout.json",
        {
            "canonical_image_size": CANONICAL_IMAGE_SIZE,
            "tile_size": TILE_SIZE,
            "stride": TILE_STRIDE,
            "tile_count": len(boxes),
            "boxes": boxes,
        },
    )

    reusable_contexts = reference_dir is not None and (reference_dir / "tiles").is_dir()
    print("[prepare] loading official PBR encoder")
    pbr_encoder = pixal3d_models.from_pretrained(str(Path(args.pbr_encoder).expanduser())).eval()
    if not bool(args.low_vram):
        pbr_encoder.to(device)
    shape_encoder: Optional[torch.nn.Module] = None
    if not reusable_contexts:
        print("[prepare] loading official fixed-shape encoder for fresh contexts")
        shape_encoder = pixal3d_models.from_pretrained(str(Path(args.shape_encoder).expanduser())).eval()
        if not bool(args.low_vram):
            shape_encoder.to(device)
    contexts, visibility_summary = _prepare_range_contexts(
        args=args,
        pipeline=pipeline,
        baseline_mesh=baseline_mesh,
        global_camera=global_camera,
        image_4096=image_4096,
        output_dir=output_dir,
        global_attr_field=global_attr_field,
        shape_encoder=shape_encoder,
        pbr_encoder=pbr_encoder,
        boxes=boxes,
        reference_dir=reference_dir,
    )
    del global_attr_field, shape_encoder
    _empty_cuda_cache()
    if not contexts:
        raise RuntimeError("range-null preparation produced no active tile")
    selected_ids = sorted(int(context.tile_id) for context in contexts)
    _atomic_json(
        output_dir / "tile_preparation_summary.json",
        {
            "prepared_tile_ids": selected_ids,
            "active_tile_count": len(selected_ids),
            "layout_tile_count": len(boxes),
            "reused_contexts": reusable_contexts,
        },
    )

    texture_params = core._sampler_overrides(args)[2]
    correctness_stats: Dict[str, Any]
    if bool(args.skip_correctness):
        correctness_stats = {
            "passed": False,
            "skipped": True,
            "reason": "explicit --skip-correctness",
        }
    else:
        correctness_contexts = list(contexts[: min(int(args.correctness_tile_count), len(contexts))])
        correctness_flow = _run_range_null_guided_flow(
            contexts=correctness_contexts,
            pipeline=pipeline,
            global_camera=global_camera,
            texture_params=texture_params,
            pbr_encoder=pbr_encoder,
            args=args,
            output_root=output_dir / "correctness_test",
            step_limit=int(args.correctness_step_count),
            write_endpoint=False,
        )
        correctness_stats = _check_correctness(correctness_flow)
        correctness_stats["flow"] = correctness_flow
        _atomic_json(output_dir / "correctness_test" / "correctness_summary.json", correctness_stats)
        if not bool(correctness_stats["passed"]):
            raise RuntimeError(f"correctness preflight failed: {correctness_stats['errors']}")
        print(
            f"[correctness] passed tiles={correctness_stats['tile_count']} "
            f"steps={correctness_stats['flow_steps']}; continuing to full experiment"
        )

    # Existing pure-HR and current Gaussian-guided variants are read directly
    # when available.  If no reference run exists, generate those established
    # routes with their original implementation before running the new route.
    pure_source = _reference_variant_path(reference_dir, "pure_HR")
    current_source = _reference_variant_path(reference_dir, "current_gaussian_guided")
    pure_flow: Optional[Dict[str, Any]] = None
    current_flow: Optional[Dict[str, Any]] = None
    if pure_source is None or current_source is None:
        pure_flow = base._run_pure_hr_flow(
            contexts=contexts,
            pipeline=pipeline,
            texture_params=texture_params,
            args=args,
        )
        current_flow = base._run_cross_tile_guided_flow(
            contexts=contexts,
            pipeline=pipeline,
            global_camera=global_camera,
            texture_params=texture_params,
            pbr_encoder=pbr_encoder,
            args=args,
        )
    pure_mesh, pure_summary = _load_or_make_reference_variant(
        variant="pure_HR",
        endpoint_attr="pure_endpoint",
        contexts=contexts,
        pipeline=pipeline,
        global_camera=global_camera,
        baseline_mesh=baseline_mesh,
        args=args,
        output_root=output_dir,
        reference_dir=reference_dir,
    )
    current_mesh, current_summary = _load_or_make_reference_variant(
        variant="current_gaussian_guided",
        endpoint_attr="guided_endpoint",
        contexts=contexts,
        pipeline=pipeline,
        global_camera=global_camera,
        baseline_mesh=baseline_mesh,
        args=args,
        output_root=output_dir,
        reference_dir=reference_dir,
    )

    range_flow = _run_range_null_guided_flow(
        contexts=contexts,
        pipeline=pipeline,
        global_camera=global_camera,
        texture_params=texture_params,
        pbr_encoder=pbr_encoder,
        args=args,
        output_root=output_dir,
        step_limit=None,
        write_endpoint=True,
    )
    range_mesh, range_summary = _variant_patch_and_stitch(
        variant="range_null_guided",
        endpoint_attr="range_null_endpoint",
        contexts=contexts,
        pipeline=pipeline,
        global_camera=global_camera,
        baseline_mesh=baseline_mesh,
        args=args,
        output_root=output_dir,
    )
    del pbr_encoder
    _empty_cuda_cache()

    meshes: Dict[str, Any] = {
        "global_baseline": baseline_mesh,
        "pure_HR": pure_mesh,
        "current_gaussian_guided": current_mesh,
        "range_null_guided": range_mesh,
    }
    render_records: Dict[str, Any] = {}
    evaluation_table: List[Dict[str, Any]] = []
    multiview_record: Dict[str, Any] = {"enabled": False}
    if bool(args.render):
        envmap = core.load_envmap(str(args.envmap), device="cuda")
        render_records, evaluation_table = _render_aligned_variants(
            meshes=meshes,
            baseline_mesh=baseline_mesh,
            global_camera=global_camera,
            output_root=output_dir,
            canonical_4096_path=output_dir / "canonical_4096.png",
            args=args,
            envmap=envmap,
        )
        if bool(args.render_multiview):
            multiview_record = _render_multiview_variants(
                meshes=meshes,
                baseline_mesh=baseline_mesh,
                global_camera=global_camera,
                output_root=output_dir,
                args=args,
                envmap=envmap,
            )
        del envmap
        _empty_cuda_cache()
    else:
        evaluation_table = [
            {
                "variant": variant,
                "vertices": int(mesh.vertices.shape[0]),
                "faces": int(mesh.faces.shape[0]),
                "PSNR": None,
                "SSIM": None,
                "LPIPS": None,
                "render_resolution": None,
            }
            for variant, mesh in meshes.items()
        ]
    _write_metrics_csv(output_dir / "metrics.csv", evaluation_table)
    projection_metrics = _aggregate_projection_metrics(range_flow)
    _atomic_json(output_dir / "projection_metrics.json", projection_metrics)

    # Keep convenient root-level names without duplicating the potentially
    # multi-gigabyte stitched tensors/GLBs.
    root_mesh = output_dir / "global_merged_mesh.pt"
    range_mesh_path = output_dir / "variants" / "range_null_guided" / "global_merged_mesh.pt"
    if not root_mesh.exists():
        root_mesh.symlink_to(Path("variants") / "range_null_guided" / "global_merged_mesh.pt")
    root_glb = output_dir / "global_merged_mesh.glb"
    range_glb = output_dir / "variants" / "range_null_guided" / "global_merged_mesh.glb"
    if range_glb.is_file() and not root_glb.exists():
        root_glb.symlink_to(Path("variants") / "range_null_guided" / "global_merged_mesh.glb")

    hidden_count = 0
    invariant_failure_count = 0
    for step in range_flow.get("steps", []):
        for row in step.get("tiles", []):
            hidden_count += int(row["projection"].get("hidden_voxel_count", 0))
            if not bool(row["projection"].get("range_null", {}).get("passed", False)):
                invariant_failure_count += 1
    guidance = {
        "formula": "A^dagger A Y_G + (I-A^dagger A)Y_H",
        "A": "hidden global_parent_id group mean",
        "A_dagger": "broadcast parent mean to hidden rows",
        "global_pbr_reference": "local_attrs from _resample_local_attrs_from_global, aligned with geometry.coords and target_points",
        "hidden_projection": "range-null only; no cross-tile Gaussian averaging",
        "observed_projection": "current _fuse_tile_field Gaussian result",
        "pbr_channels": list(PBR_CHANNEL_NAMES),
        "eta": float(args.eta),
        "fusion_sigma_pixels": float(args.fusion_sigma_pixels),
        "G_guidance_used": False,
        "G_velocity_used": False,
        "G_timestep_trajectory_used": False,
        "latent_low_high_frequency_mixing_used": False,
        "weighted_G_HR_blend_used": False,
        "invariant_tolerance": float(args.invariant_tolerance),
        "invariant_failure_count": invariant_failure_count,
        "hidden_voxel_count_across_steps": hidden_count,
    }
    route_checks = {
        "shape_flow_called": False,
        "shape_sampler_called": False,
        "fixed_shape_unchanged": all(
            bool(context.static_stats.get("fixed_shape", {}).get("support_unchanged", True))
            for context in contexts
        ),
        "all_tiles_synchronized_per_step": bool(range_flow["all_tiles_synchronized_per_step"]),
        "official_texture_sampler": True,
        "official_texture_decoder": True,
        "official_texture_encoder": True,
        "official_meshwithvoxel_query": True,
        "cycle_cancelled_residual_used": True,
        "observed_current_gaussian_preserved": True,
        "hidden_gaussian_disabled": True,
        "no_training": True,
    }
    diagnosis = {
        "A_visibility_mask": "passed binary/canonical raster preflight",
        "B_global_parent_mapping": "passed valid C1024 quantisation preflight",
        "C_range_null_invariant": "passed" if invariant_failure_count == 0 else "failed",
        "D_encoder_transport": "recorded by x0_cycle/x0_projected support and delta diagnostics",
        "E_flow_dynamics": "recorded by guided_v and per-step endpoint diagnostics",
        "F_final_patch_stitch": "recorded by range_null_guided stitch summary",
        "G_hidden_high_frequency": "inspect E_detail and back-view contact sheet",
    }
    summary: Dict[str, Any] = {
        "format": FORMAT,
        "image": str(source_path),
        "cuda_device": int(args.cuda_device),
        "cuda_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "low_vram": bool(args.low_vram),
        "seed": int(args.seed),
        "reference_experiment_dir": str(reference_dir) if reference_dir is not None else None,
        "global_camera": global_camera,
        "global_baseline": baseline_summary,
        "tile_layout": {
            "canonical_image_size": CANONICAL_IMAGE_SIZE,
            "tile_size": TILE_SIZE,
            "stride": TILE_STRIDE,
            "tile_count": len(boxes),
            "participating_tile_ids": selected_ids,
            "boxes": boxes,
        },
        "visibility": visibility_summary,
        "correctness_test": correctness_stats,
        "guidance": guidance,
        "sampler": {
            "texture_steps": int(args.texture_steps),
            "noise_timestep": float(args.noise_timestep),
            "noise_strength": float(args.noise_strength),
            "seed": int(args.seed),
            "fixed_shape": True,
            "tile_shape_flow": False,
            "route": "official texture CFG/timestep/Euler; _get_model_prediction -> _xstart_to_pred",
        },
        "pure_HR": {"flow": pure_flow, "variant": pure_summary},
        "current_gaussian_guided": {"flow": current_flow, "variant": current_summary},
        "range_null_guided": {"flow": range_flow, "variant": range_summary},
        "new_metrics": projection_metrics,
        "route_checks": route_checks,
        "diagnosis": diagnosis,
        "evaluation": {
            "reference": str((output_dir / "canonical_4096.png").resolve()),
            "table": evaluation_table,
            "renders": render_records,
            "multiview": multiview_record,
        },
        "variants": {
            "global_baseline": baseline_summary,
            "pure_HR": pure_summary,
            "current_gaussian_guided": current_summary,
            "range_null_guided": range_summary,
        },
        "artifacts": {
            "global_baseline_mesh": str((output_dir / "global_baseline_mesh.pt").resolve()),
            "global_merged_mesh_pt": str(root_mesh.resolve()),
            "global_merged_mesh_glb": str(root_glb.resolve()) if root_glb.exists() else None,
            "metrics_csv": str((output_dir / "metrics.csv").resolve()),
            "projection_metrics": str((output_dir / "projection_metrics.json").resolve()),
            "steps_directory": str((output_dir / "steps").resolve()),
            "back_view_contact_sheet": multiview_record.get("back_view_contact_sheet"),
        },
    }
    report_path = _write_range_report(output_dir, summary)
    summary["report_markdown"] = str(report_path.resolve())
    _atomic_json(output_dir / "summary.json", summary)
    print(
        f"[done] tiles={len(contexts)} range_null_steps={range_flow['flow_steps']} "
        f"range_null_mesh_vertices={range_summary['vertices']:,} "
        f"range_null_mesh_faces={range_summary['faces']:,} "
        f"summary={output_dir / 'summary.json'}"
    )
    return summary


def main() -> None:
    run(_build_parser().parse_args())


if __name__ == "__main__":
    main()
