#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase-1 local C256/C1024 sparse MRA verification.

This is an independent, field-space experiment following ``Codex.md``.  It
does not alter Pixal3D's flow sampler.  The existing CUDA4 texture-only
cache supplies the global baseline and PureHR endpoints; this script
re-decodes PureHR, constructs a local C256/C1024 support from the same local
mesh, builds the exact sparse trilinear prolongation used by
``MeshWithVoxel.query_attrs``, solves the least-squares restriction, and
applies the correction only on hidden fine rows.

The query points used by the operator are O-voxel cell centers derived from
active sparse coordinates.  Decoder mesh vertices are used only for the final
renderer, never as the definition of P/A.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import tempfile
import time
import traceback
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
import o_voxel
import torch
from PIL import Image, ImageDraw, ImageOps
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr, splu

import pixal3d_texture_pbr_degradation_experiment as experiment
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
import pixal3d_texture_pbr_global_stitch as global_stitch
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVertexPbr, MeshWithVoxel


FORMAT = "pixal3d_sparse_mra_hidden_phase1_v1"
GLOBAL_RESOLUTION = 1024
COARSE_RESOLUTION = 256
CANONICAL_RESOLUTION = 4096
PBR_CHANNELS = 6
CHANNEL_NAMES = ("base_color_r", "base_color_g", "base_color_b", "metallic", "roughness", "alpha")
SUMMARY_CHANNELS = {
    "rgb": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
    "joint": slice(0, 6),
}


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        return torch.load(path, map_location="cpu")


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
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: _jsonable(row.get(key)) for key in keys} for row in rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _parse_ids(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not str(value).strip():
        return None
    return {int(part.strip()) for part in str(value).split(",") if part.strip()}


def _norm(value: np.ndarray | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        return float(torch.linalg.vector_norm(value.detach().to(torch.float64)).item())
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


def _relative_error(value: np.ndarray | torch.Tensor, reference: np.ndarray | torch.Tensor, eps: float = 1e-8) -> float:
    return _norm(value) / (_norm(reference) + float(eps))


def _tensor_range(value: torch.Tensor) -> Dict[str, List[float]]:
    value = value.detach().to(torch.float32)
    if value.numel() == 0:
        return {"min": [], "max": []}
    return {"min": value.amin(dim=0).cpu().tolist(), "max": value.amax(dim=0).cpu().tolist()}


def _metric_stats(values: Iterable[float]) -> Dict[str, Optional[float]]:
    array = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _field_error(left: torch.Tensor, right: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError(f"field shapes differ: {tuple(left.shape)} vs {tuple(right.shape)}")
    if mask is None:
        mask = torch.ones(left.shape[0], dtype=torch.bool)
    mask = mask.detach().cpu().bool()
    delta = (left.detach().cpu().to(torch.float64) - right.detach().cpu().to(torch.float64))[mask]
    reference = right.detach().cpu().to(torch.float64)[mask]
    if delta.numel() == 0:
        return {"count": 0, "mean_abs": None, "max_abs": None, "relative_l2": None, "per_channel": {}}
    per_channel: Dict[str, Any] = {}
    for index, name in enumerate(CHANNEL_NAMES):
        d = delta[:, index]
        r = reference[:, index]
        per_channel[name] = {
            "mean_abs": float(d.abs().mean().item()),
            "max_abs": float(d.abs().max().item()),
            "relative_l2": _relative_error(d.numpy(), r.numpy()),
        }
    return {
        "count": int(delta.shape[0]),
        "mean_abs": float(delta.abs().mean().item()),
        "max_abs": float(delta.abs().max().item()),
        "relative_l2": _relative_error(delta.numpy(), reference.numpy()),
        "per_channel": per_channel,
    }


def _detail_stats(detail: torch.Tensor, field: torch.Tensor) -> Dict[str, Any]:
    detail64 = detail.detach().cpu().to(torch.float64)
    field64 = field.detach().cpu().to(torch.float64)
    result: Dict[str, Any] = {
        "relative_l2": _relative_error(detail64.numpy(), field64.numpy()),
        "l2": _norm(detail64),
        "per_channel": {},
    }
    for index, name in enumerate(CHANNEL_NAMES):
        result["per_channel"][name] = {
            "l2": _norm(detail64[:, index]),
            "field_l2": _norm(field64[:, index]),
            "relative_l2": _relative_error(detail64[:, index].numpy(), field64[:, index].numpy()),
            "mean_abs": float(detail64[:, index].abs().mean().item()),
            "p95_abs": float(detail64[:, index].abs().quantile(0.95).item()),
        }
    return result


def _channel_view(value: torch.Tensor, name: str) -> torch.Tensor:
    if name == "rgb":
        return value[:, 0:3].mean(dim=1)
    if name == "metallic":
        return value[:, 3]
    if name == "roughness":
        return value[:, 4]
    if name == "alpha":
        return value[:, 5]
    raise KeyError(name)


def _linear_keys(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    coords = coords.to(torch.int64)
    return (coords[:, 0] * int(resolution) + coords[:, 1]) * int(resolution) + coords[:, 2]


def _cell_centers(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    return -0.5 + (coords.to(torch.float32) + 0.5) / float(resolution)


def _build_prolongation(
    coarse_coords: torch.Tensor,
    fine_points: torch.Tensor,
    *,
    coarse_resolution: int = COARSE_RESOLUTION,
) -> Tuple[csr_matrix, Dict[str, Any]]:
    """Build the exact sparse linear map used by ``grid_sample_3d``.

    For a sparse input, Pixal3D's trilinear kernel drops missing neighbors and
    renormalizes by the remaining interpolation weight.  This normalization
    is important: treating missing support as zero changes P and makes the
    ``query_attrs`` test fail near sparse boundaries.
    """
    coarse_coords = coarse_coords.detach().cpu().to(torch.int64)
    fine_points = fine_points.detach().cpu().to(torch.float32)
    coarse_count = int(coarse_coords.shape[0])
    fine_count = int(fine_points.shape[0])
    if coarse_coords.ndim != 2 or coarse_coords.shape[1] != 3:
        raise ValueError(f"coarse coords must be [N,3], got {tuple(coarse_coords.shape)}")
    if fine_points.ndim != 2 or fine_points.shape[1] != 3:
        raise ValueError(f"fine points must be [N,3], got {tuple(fine_points.shape)}")
    if coarse_count == 0:
        return csr_matrix((fine_count, 0), dtype=np.float32), {
            "fine_rows": fine_count,
            "coarse_columns": 0,
            "nnz": 0,
            "coverage_ratio": 0.0,
            "row_nnz": {"mean": 0.0, "max": 0, "min": 0},
            "support_rule": "sparse trilinear query with valid-neighbor weight renormalization",
        }

    support_keys = _linear_keys(coarse_coords, coarse_resolution)
    order = torch.argsort(support_keys, stable=True)
    sorted_keys = support_keys.index_select(0, order)
    if sorted_keys.numel() > 1 and bool((sorted_keys[1:] == sorted_keys[:-1]).any().item()):
        raise RuntimeError("coarse support contains duplicate coordinates")

    # grid_sample receives ((x - origin) / voxel_size), while its neighboring
    # integer locations are centered at query_pts +/- 0.5.  The base integer
    # therefore is floor(grid - 0.5), and the fractional part is measured from
    # the cell center (base + 0.5).
    grid = (fine_points + 0.5) * float(coarse_resolution)
    base = torch.floor(grid - 0.5).to(torch.int64)
    frac = grid - (base.to(torch.float32) + 0.5)
    row_parts: List[torch.Tensor] = []
    col_parts: List[torch.Tensor] = []
    weight_parts: List[torch.Tensor] = []
    row_weight_sum = torch.zeros(fine_count, dtype=torch.float32)

    for bits in range(8):
        bit = torch.tensor(
            [(bits >> 0) & 1, (bits >> 1) & 1, (bits >> 2) & 1],
            dtype=torch.int64,
        )
        neighbor = base + bit
        weight = torch.where(bit.bool(), frac, 1.0 - frac).prod(dim=1)
        valid = ((neighbor >= 0) & (neighbor < int(coarse_resolution))).all(dim=1)
        neighbor_key = _linear_keys(neighbor, coarse_resolution)
        positions = torch.searchsorted(sorted_keys, neighbor_key)
        valid &= positions < sorted_keys.numel()
        safe_positions = positions.clamp_max(max(0, int(sorted_keys.numel()) - 1))
        if sorted_keys.numel():
            valid &= sorted_keys.index_select(0, safe_positions) == neighbor_key
        rows = torch.where(valid)[0]
        if rows.numel():
            columns = order.index_select(0, safe_positions.index_select(0, rows))
            values = weight.index_select(0, rows)
            row_parts.append(rows)
            col_parts.append(columns)
            weight_parts.append(values)
            row_weight_sum.index_add_(0, rows, values)

    if not row_parts:
        return csr_matrix((fine_count, coarse_count), dtype=np.float32), {
            "fine_rows": fine_count,
            "coarse_columns": coarse_count,
            "nnz": 0,
            "coverage_ratio": 0.0,
            "row_nnz": {"mean": 0.0, "max": 0, "min": 0},
            "support_rule": "sparse trilinear query with valid-neighbor weight renormalization",
        }

    rows = torch.cat(row_parts, dim=0)
    cols = torch.cat(col_parts, dim=0)
    values = torch.cat(weight_parts, dim=0)
    values = values / row_weight_sum.index_select(0, rows).clamp_min(1e-12)
    matrix = csr_matrix(
        (
            values.numpy().astype(np.float32, copy=False),
            (rows.numpy(), cols.numpy()),
        ),
        shape=(fine_count, coarse_count),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    row_nnz = np.diff(matrix.indptr)
    coverage = row_nnz > 0
    info = {
        "fine_rows": fine_count,
        "coarse_columns": coarse_count,
        "nnz": int(matrix.nnz),
        "coverage_ratio": float(coverage.mean()) if coverage.size else 0.0,
        "uncovered_rows": int((~coverage).sum()),
        "row_nnz": {
            "mean": float(row_nnz.mean()) if row_nnz.size else 0.0,
            "max": int(row_nnz.max()) if row_nnz.size else 0,
            "min": int(row_nnz.min()) if row_nnz.size else 0,
            "p50": float(np.median(row_nnz)) if row_nnz.size else 0.0,
            "p90": float(np.quantile(row_nnz, 0.90)) if row_nnz.size else 0.0,
        },
        "support_rule": "sparse trilinear query with valid-neighbor weight renormalization",
    }
    return matrix, info


def _apply_operator(operator: csr_matrix, value: torch.Tensor) -> torch.Tensor:
    array = value.detach().cpu().to(torch.float32).numpy()
    result = operator.dot(array)
    return torch.from_numpy(np.asarray(result, dtype=np.float32))


def _solve_restriction(
    operator: csr_matrix,
    fine_field: torch.Tensor,
    *,
    label: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve (P^T P)c=P^T Y without forming an inverse.

    A sparse LU factorization is a direct linear-system solve, not an explicit
    inverse.  If the normal matrix is singular, LSQR is used as the Moore-
    Penrose-compatible fallback on the original sparse P.
    """
    field = fine_field.detach().cpu().to(torch.float32).numpy()
    if field.ndim == 1:
        field = field[:, None]
    if field.shape[0] != operator.shape[0]:
        raise ValueError(f"{label}: P rows {operator.shape[0]} != Y rows {field.shape[0]}")
    active = np.asarray(operator.getnnz(axis=0)).reshape(-1) > 0
    active_ids = np.where(active)[0]
    result = np.zeros((operator.shape[1], field.shape[1]), dtype=np.float32)
    info: Dict[str, Any] = {
        "label": label,
        "input_rows": int(operator.shape[0]),
        "input_columns": int(operator.shape[1]),
        "active_columns": int(active_ids.size),
        "inactive_columns": int((~active).sum()),
        "method": "sparse_normal_equation_lu",
        "fallback_channels": [],
    }
    if active_ids.size == 0:
        info["method"] = "empty"
        return result, info
    reduced = operator[:, active_ids].tocsr()
    gram = (reduced.T @ reduced).tocsc()
    gram.eliminate_zeros()
    info["normal_matrix_nnz"] = int(gram.nnz)
    rhs = reduced.T.dot(field)
    solution: Optional[np.ndarray] = None
    try:
        factor = splu(gram)
        solution = factor.solve(np.asarray(rhs, dtype=np.float32)).astype(np.float32, copy=False)
        del factor
    except Exception as exc:
        info["method"] = "lsqr_pseudoinverse_fallback"
        info["factorization_error"] = f"{type(exc).__name__}: {exc}"
        solution = np.zeros((active_ids.size, field.shape[1]), dtype=np.float32)
        for channel in range(field.shape[1]):
            solved = lsqr(
                reduced,
                field[:, channel].astype(np.float64),
                atol=1e-7,
                btol=1e-7,
                iter_lim=300,
            )
            solution[:, channel] = solved[0].astype(np.float32)
            info["fallback_channels"].append({"channel": int(channel), "iterations": int(solved[2]), "istop": int(solved[1])})
    result[active_ids] = solution
    del reduced, gram, rhs, solution
    return result, info


def _basis_partition(operator: csr_matrix, observed: torch.Tensor) -> Dict[str, Any]:
    observed_np = observed.detach().cpu().bool().numpy()
    column_has_observed = np.zeros(operator.shape[1], dtype=bool)
    if observed_np.any():
        observed_operator = operator[observed_np]
        if observed_operator.nnz:
            column_has_observed[np.unique(observed_operator.indices)] = True
        del observed_operator
    column_has_any = np.asarray(operator.getnnz(axis=0)).reshape(-1) > 0
    pure_hidden = column_has_any & ~column_has_observed
    mixed = column_has_any & column_has_observed
    return {
        "hidden_basis_count": int(pure_hidden.sum()),
        "pure_hidden_basis_count": int(pure_hidden.sum()),
        "mixed_basis_count": int(mixed.sum()),
        "uncovered_basis_count": int((~column_has_any).sum()),
        "pure_hidden_ids": pure_hidden,
    }


@torch.no_grad()
def _query_mesh_field(mesh: MeshWithVoxel, points: torch.Tensor, chunk_size: int) -> torch.Tensor:
    points = points.detach().cpu().to(torch.float32)
    device = mesh.attrs.device
    outputs: List[torch.Tensor] = []
    for start in range(0, int(points.shape[0]), int(chunk_size)):
        chunk = points[start : start + int(chunk_size)].to(device=device)
        outputs.append(mesh.query_attrs(chunk).detach().cpu().to(torch.float32))
    if not outputs:
        return torch.empty((0, int(mesh.attrs.shape[1])), dtype=torch.float32)
    value = torch.cat(outputs, dim=0)
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError("decoder/global PBR query produced non-finite values")
    return value


@torch.no_grad()
def _query_global_at_local_points(
    global_field: MeshWithVoxel,
    local_points: torch.Tensor,
    *,
    transform: Any,
    global_camera: Mapping[str, float],
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map local normalized points to global normalized coordinates and query G."""
    local_points = local_points.detach().cpu().to(torch.float32)
    device = global_field.attrs.device
    fields: List[torch.Tensor] = []
    global_points: List[torch.Tensor] = []
    uv_points: List[torch.Tensor] = []
    for start in range(0, int(local_points.shape[0]), int(chunk_size)):
        local_chunk = local_points[start : start + int(chunk_size)].to(device=device)
        q_local = local_chunk * (2.0 * float(transform.mesh_scale))
        q_global, uv = core._local_q_to_global_q(
            q_local,
            global_camera=global_camera,
            transform=transform,
        )
        normalized_global = q_global / (2.0 * float(global_camera["mesh_scale"]))
        fields.append(global_field.query_attrs(normalized_global).detach().cpu().to(torch.float32))
        global_points.append(normalized_global.detach().cpu().to(torch.float32))
        uv_points.append(uv.detach().cpu().to(torch.float32))
    if not fields:
        empty_field = torch.empty((0, int(global_field.attrs.shape[1])), dtype=torch.float32)
        empty_points = torch.empty((0, 3), dtype=torch.float32)
        empty_uv = torch.empty((0, 2), dtype=torch.float32)
        return empty_field, empty_points, empty_uv
    field = torch.cat(fields, dim=0)
    points = torch.cat(global_points, dim=0)
    uv = torch.cat(uv_points, dim=0)
    if not bool(torch.isfinite(field).all().item()):
        raise RuntimeError("global baseline PBR query produced non-finite values")
    return field, points, uv


@torch.no_grad()
def _project_local_points(
    local_points: torch.Tensor,
    *,
    transform: Any,
    global_camera: Mapping[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local_points = local_points.detach().cpu().to(torch.float32)
    q_local = local_points * (2.0 * float(transform.mesh_scale))
    q_global, uv = core._local_q_to_global_q(
        q_local,
        global_camera=global_camera,
        transform=transform,
    )
    _, depth, finite = core._project_global_q_to_4096(q_global, global_camera=global_camera)
    return uv.detach().cpu().to(torch.float32), depth.detach().cpu().to(torch.float32), finite.detach().cpu().bool()


def _local_to_global_normalized(
    local_points: torch.Tensor,
    *,
    transform: Any,
    global_camera: Mapping[str, float],
    chunk_size: int,
) -> torch.Tensor:
    local_points = local_points.detach().cpu().to(torch.float32)
    pieces: List[torch.Tensor] = []
    for start in range(0, int(local_points.shape[0]), int(chunk_size)):
        q_local = local_points[start : start + int(chunk_size)] * (2.0 * float(transform.mesh_scale))
        q_global, _ = core._local_q_to_global_q(
            q_local,
            global_camera=global_camera,
            transform=transform,
        )
        pieces.append((q_global / (2.0 * float(global_camera["mesh_scale"]))).cpu().to(torch.float32))
    return torch.cat(pieces, dim=0) if pieces else torch.empty((0, 3), dtype=torch.float32)


def _load_visibility(visibility_dir: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"source": str(visibility_dir), "rule": "fallback_foreground"}
    triangle_path = visibility_dir / "triangle_id.pt"
    depth_path = visibility_dir / "depth.pt"
    foreground_path = visibility_dir / "foreground.pt"
    if triangle_path.is_file() and depth_path.is_file():
        triangle_payload = _load_torch(triangle_path)
        depth_payload = _load_torch(depth_path)
        result["triangle_id"] = triangle_payload["triangle_id"].detach().cpu().to(torch.int32)
        result["depth"] = depth_payload["depth"].detach().cpu().to(torch.float32)
        result["rule"] = "global triangle-id plus depth agreement"
        result["near"] = depth_payload.get("near")
        result["far"] = depth_payload.get("far")
    if foreground_path.is_file():
        foreground_payload = _load_torch(foreground_path)
        result["foreground"] = foreground_payload["foreground"].detach().cpu().bool()
    return result


def _sample_visibility(
    uv: torch.Tensor,
    depth: torch.Tensor,
    finite: torch.Tensor,
    visibility: Mapping[str, Any],
    *,
    depth_tolerance_pixels: float,
    focal_pixels: float,
) -> torch.Tensor:
    uv = uv.detach().cpu().to(torch.float32)
    depth = depth.detach().cpu().to(torch.float32)
    finite = finite.detach().cpu().bool()
    if "triangle_id" not in visibility or "depth" not in visibility:
        foreground = visibility.get("foreground")
        if foreground is None:
            return torch.zeros(uv.shape[0], dtype=torch.bool)
        height, width = foreground.shape
        x = torch.round(uv[:, 0]).to(torch.long)
        y = torch.round(uv[:, 1]).to(torch.long)
        inside = finite & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        output = torch.zeros(uv.shape[0], dtype=torch.bool)
        if bool(inside.any().item()):
            output[inside] = foreground[y[inside], x[inside]]
        return output
    triangle_id = visibility["triangle_id"]
    depth_map = visibility["depth"]
    height, width = triangle_id.shape
    x = torch.round(uv[:, 0]).to(torch.long)
    y = torch.round(uv[:, 1]).to(torch.long)
    inside = finite & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    output = torch.zeros(uv.shape[0], dtype=torch.bool)
    if bool(inside.any().item()):
        local_depth = depth[inside]
        raster_depth = depth_map[y[inside], x[inside]]
        raster_tri = triangle_id[y[inside], x[inside]]
        tolerance = float(depth_tolerance_pixels) * local_depth.abs().clamp_min(1e-5) / float(focal_pixels)
        output[inside] = (
            (raster_tri >= 0)
            & torch.isfinite(raster_depth)
            & torch.isfinite(local_depth)
            & ((local_depth - raster_depth).abs() <= tolerance)
        )
    return output


def _voxelize_support(vertices: torch.Tensor, faces: torch.Tensor, resolution: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords, dual_vertices_world, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices.detach().cpu().to(torch.float32),
        faces=faces.detach().cpu().to(torch.int32),
        grid_size=int(resolution),
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
        timing=False,
    )
    coords = coords.detach().cpu().to(torch.int32)
    dual_vertices_world = dual_vertices_world.detach().cpu().to(torch.float32)
    intersected = intersected.detach().cpu()
    if coords.shape[0] == 0:
        raise RuntimeError(f"C{resolution} voxelization returned empty support")
    return coords, dual_vertices_world, intersected


@torch.no_grad()
def _query_attrs_linearity(
    coarse_coords: torch.Tensor,
    fine_points: torch.Tensor,
    *,
    sample_rows: int,
    seed: int,
) -> Dict[str, Any]:
    """Numerically test Q(a c1+b c2)=aQ(c1)+bQ(c2)."""
    if fine_points.shape[0] == 0:
        return {"sample_rows": 0, "max_abs_error": 0.0, "mean_abs_error": 0.0, "relative_l2": 0.0, "explicit_P_match": 0.0}
    count = min(int(sample_rows), int(fine_points.shape[0]))
    indices = torch.linspace(0, int(fine_points.shape[0]) - 1, count).round().to(torch.long)
    points = fine_points.index_select(0, indices)
    device = torch.device("cuda")
    coords = coarse_coords.to(device=device, dtype=torch.int32)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    c1 = torch.randn((coords.shape[0], PBR_CHANNELS), generator=generator, device=device)
    c2 = torch.randn((coords.shape[0], PBR_CHANNELS), generator=generator, device=device)
    alpha, beta = 1.371, -0.417
    query_mesh = MeshWithVoxel(
        vertices=torch.empty((1, 3), device=device),
        faces=torch.empty((0, 3), dtype=torch.int32, device=device),
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / float(COARSE_RESOLUTION),
        coords=coords,
        attrs=c1,
        voxel_shape=torch.Size([1, PBR_CHANNELS, COARSE_RESOLUTION, COARSE_RESOLUTION, COARSE_RESOLUTION]),
    )
    points_device = points.to(device=device)
    q1 = query_mesh.query_attrs(points_device)
    query_mesh.attrs = c2
    q2 = query_mesh.query_attrs(points_device)
    query_mesh.attrs = alpha * c1 + beta * c2
    q_combo = query_mesh.query_attrs(points_device)
    rhs = alpha * q1 + beta * q2
    error = (q_combo - rhs).detach().to(torch.float64)
    # The sparse matrix is constructed from the same kernel rule; compare it
    # independently so an implementation mismatch cannot hide behind Q's own
    # linearity.
    operator, _ = _build_prolongation(coarse_coords, points, coarse_resolution=COARSE_RESOLUTION)
    explicit = torch.from_numpy(operator.dot(c1.detach().cpu().numpy()).astype(np.float32))
    explicit_error = explicit - q1.detach().cpu().to(torch.float32)
    return {
        "sample_rows": int(count),
        "max_abs_error": float(error.abs().max().item()),
        "mean_abs_error": float(error.abs().mean().item()),
        "relative_l2": _relative_error(error.cpu().numpy(), rhs.detach().cpu().numpy()),
        "explicit_P_match_max_abs_error": float(explicit_error.abs().max().item()),
        "explicit_P_match_mean_abs_error": float(explicit_error.abs().mean().item()),
        "coefficients": {"alpha": alpha, "beta": beta},
        "operator": "MeshWithVoxel.query_attrs trilinear with sparse support renormalization",
    }


def _make_panel(
    uv: torch.Tensor,
    values: torch.Tensor,
    *,
    box: Sequence[int],
    size: int,
    signed: bool = False,
    mask: Optional[torch.Tensor] = None,
    sample_limit: int = 150_000,
) -> Image.Image:
    """Rasterize sparse O-voxel values into a compact diagnostic panel."""
    uv = uv.detach().cpu().to(torch.float32)
    values = values.detach().cpu().to(torch.float32).reshape(-1)
    if mask is None:
        mask = torch.ones(values.shape[0], dtype=torch.bool)
    else:
        mask = mask.detach().cpu().bool()
    finite = torch.isfinite(values) & torch.isfinite(uv).all(dim=1) & mask
    ids = torch.where(finite)[0]
    if ids.numel() > int(sample_limit):
        stride = max(1, int(ids.numel()) // int(sample_limit))
        ids = ids[::stride][: int(sample_limit)]
    canvas = torch.zeros((size, size), dtype=torch.float32)
    counts = torch.zeros((size, size), dtype=torch.float32)
    if ids.numel():
        x0, y0, x1, y1 = (float(v) for v in box)
        x = ((uv.index_select(0, ids)[:, 0] - x0) / max(1.0, x1 - x0) * size).floor().to(torch.long)
        y = ((uv.index_select(0, ids)[:, 1] - y0) / max(1.0, y1 - y0) * size).floor().to(torch.long)
        valid = (x >= 0) & (x < size) & (y >= 0) & (y < size)
        x, y = x[valid], y[valid]
        val = values.index_select(0, ids)[valid]
        flat = y * size + x
        canvas.view(-1).index_add_(0, flat, val)
        counts.view(-1).index_add_(0, flat, torch.ones_like(val))
    canvas = canvas / counts.clamp_min(1.0)
    if signed:
        scale = float(canvas.abs().max().item())
        if scale < 1e-8:
            scale = 1e-8
        norm = (canvas / scale).clamp(-1.0, 1.0)
        red = (norm.clamp_min(0.0) * 255.0 + (1.0 - norm.abs()) * 255.0).clamp(0, 255)
        blue = ((-norm).clamp_min(0.0) * 255.0 + (1.0 - norm.abs()) * 255.0).clamp(0, 255)
        green = ((1.0 - norm.abs()) * 255.0).clamp(0, 255)
        array = torch.stack((red, green, blue), dim=-1).to(torch.uint8).numpy()
    else:
        norm = canvas
        finite_values = norm[counts > 0]
        if finite_values.numel():
            lo = float(finite_values.quantile(0.01).item())
            hi = float(finite_values.quantile(0.99).item())
            if hi - lo < 1e-8:
                hi = lo + 1.0
            norm = ((norm - lo) / (hi - lo)).clamp(0.0, 1.0)
        array = (norm[..., None].repeat(1, 1, 3) * 255.0).to(torch.uint8).numpy()
    # Make uncovered pixels neutral black, preserving the fact that the panel
    # is a sparse projection rather than a dense image prediction.
    array[counts.numpy() == 0] = 0
    return Image.fromarray(array, mode="RGB")


def _save_decomposition_visual(
    path: Path,
    *,
    uv_coarse: torch.Tensor,
    uv_fine: torch.Tensor,
    fields: Mapping[str, torch.Tensor],
    box: Sequence[int],
    hidden: torch.Tensor,
    size: int = 192,
) -> None:
    columns = ("G256", "P G256", "G1024", "D_G", "H1024", "P A H", "D_H", "hidden")
    rows = ("rgb", "metallic", "roughness")
    header = 36
    label_width = 74
    sheet = Image.new("RGB", (label_width + len(columns) * size, header + len(rows) * (size + 22)), "black")
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(columns):
        draw.text((label_width + col * size + 4, 10), label, fill=(255, 255, 255))
    for row_index, row_name in enumerate(rows):
        y = header + row_index * (size + 22)
        draw.text((7, y + size // 2 - 7), row_name, fill=(255, 255, 255))
        for col, label in enumerate(columns):
            if label == "G256":
                panel = _make_panel(uv_coarse, _channel_view(fields["G256"], row_name), box=box, size=size)
            elif label == "P G256":
                panel = _make_panel(uv_fine, _channel_view(fields["P_G256"], row_name), box=box, size=size)
            elif label == "G1024":
                panel = _make_panel(uv_fine, _channel_view(fields["G1024"], row_name), box=box, size=size)
            elif label == "D_G":
                panel = _make_panel(uv_fine, _channel_view(fields["D_G"], row_name), box=box, size=size, signed=True)
            elif label == "H1024":
                panel = _make_panel(uv_fine, _channel_view(fields["H1024"], row_name), box=box, size=size)
            elif label == "P A H":
                panel = _make_panel(uv_fine, _channel_view(fields["PA_H"], row_name), box=box, size=size)
            elif label == "D_H":
                panel = _make_panel(uv_fine, _channel_view(fields["D_H"], row_name), box=box, size=size, signed=True)
            else:
                panel = _make_panel(uv_fine, hidden.to(torch.float32), box=box, size=size)
            sheet.paste(panel, (label_width + col * size, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


@torch.no_grad()
def _decode_purehr_mesh(
    pipeline: Any,
    endpoint: Mapping[str, Any],
    *,
    label: str,
) -> Tuple[MeshWithVoxel, Dict[str, Any]]:
    device = torch.device("cuda")
    shape = SparseTensor(
        endpoint["shape_norm"].to(device=device, dtype=torch.float32),
        endpoint["shape_coords"].to(device=device, dtype=torch.int32),
    )
    texture = SparseTensor(
        endpoint["hr_tex_norm"].to(device=device, dtype=torch.float32),
        endpoint["hr_tex_coords"].to(device=device, dtype=torch.int32),
    )
    shape_denorm = experiment._denormalize_slat(shape, pipeline.shape_slat_normalization)
    mesh, _, stats = experiment._decode_and_query(
        pipeline=pipeline,
        shape_denorm=shape_denorm,
        texture_latent_norm=texture,
        normalization=pipeline.tex_slat_normalization,
        query_points_device=torch.empty((0, 3), device=device, dtype=torch.float32),
        resolution=GLOBAL_RESOLUTION,
        query_chunk_size=65_536,
        label=label,
    )
    del shape, texture, shape_denorm
    return mesh, stats


def _process_tile(
    *,
    args: argparse.Namespace,
    source_dir: Path,
    output_dir: Path,
    tile_id: int,
    row: Mapping[str, Any],
    pipeline: Any,
    global_field: MeshWithVoxel,
    global_camera: Mapping[str, float],
    visibility: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tile_dir = output_dir / "tiles" / f"tile_{int(tile_id):02d}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = tile_dir / "operator_metrics.json"
    payload_path = tile_dir / "phase1_tile.pt"
    if bool(args.resume) and metrics_path.is_file() and payload_path.is_file():
        metrics = _load_torch(metrics_path) if metrics_path.suffix == ".pt" else json.loads(metrics_path.read_text(encoding="utf-8"))
        payload = _load_torch(payload_path)
        print(f"[tile {tile_id:02d}] reused phase1 cache")
        return metrics, payload

    started = time.perf_counter()
    cache_path = source_dir / "global_stitched_quality" / "decoded_global_tiles" / f"tile_{int(tile_id):02d}.pt"
    endpoint_path = source_dir / "tiles" / f"tile_{int(tile_id):02d}" / "endpoints.pt"
    if not cache_path.is_file() or not endpoint_path.is_file():
        raise FileNotFoundError(f"tile {tile_id}: missing global cache or endpoint")
    cached = _load_torch(cache_path)
    endpoint = _load_torch(endpoint_path)
    transform = core.TileCameraTransform(**endpoint["transform"])

    mesh_hr, decode_stats = _decode_purehr_mesh(
        pipeline,
        endpoint,
        label=f"phase1 tile {tile_id:02d} PureHR",
    )
    local_vertices = mesh_hr.vertices.detach().cpu().to(torch.float32)
    local_faces = mesh_hr.faces.detach().cpu().to(torch.int32)
    if local_vertices.shape[0] != cached["global_vertices"].shape[0]:
        raise RuntimeError(
            f"tile {tile_id}: decoded vertex count differs from cached global mesh: "
            f"{local_vertices.shape[0]} vs {cached['global_vertices'].shape[0]}"
        )
    reconstructed_global = _local_to_global_normalized(
        local_vertices,
        transform=transform,
        global_camera=global_camera,
        chunk_size=int(args.query_chunk_size),
    )
    geometry_roundtrip = (reconstructed_global - cached["global_vertices"].to(torch.float32)).abs()
    geometry_roundtrip_max = float(geometry_roundtrip.max().item()) if geometry_roundtrip.numel() else 0.0
    if geometry_roundtrip_max > float(args.geometry_tolerance):
        raise RuntimeError(f"tile {tile_id}: cache/decode geometry mismatch {geometry_roundtrip_max:.3e}")

    coarse_coords, _, _ = _voxelize_support(local_vertices, local_faces, COARSE_RESOLUTION)
    fine_coords, _, _ = _voxelize_support(local_vertices, local_faces, GLOBAL_RESOLUTION)
    coarse_points = _cell_centers(coarse_coords, COARSE_RESOLUTION)
    fine_points = _cell_centers(fine_coords, GLOBAL_RESOLUTION)
    print(
        f"[tile {tile_id:02d}] mesh={local_vertices.shape[0]:,} "
        f"C256={coarse_coords.shape[0]:,} C1024={fine_coords.shape[0]:,}",
        flush=True,
    )

    g_coarse, global_coarse_points, uv_coarse = _query_global_at_local_points(
        global_field,
        coarse_points,
        transform=transform,
        global_camera=global_camera,
        chunk_size=int(args.query_chunk_size),
    )
    g_fine, global_fine_points, uv_fine = _query_global_at_local_points(
        global_field,
        fine_points,
        transform=transform,
        global_camera=global_camera,
        chunk_size=int(args.query_chunk_size),
    )
    h_fine = _query_mesh_field(mesh_hr, fine_points, int(args.query_chunk_size))
    q_global_fine = global_fine_points * (2.0 * float(global_camera["mesh_scale"]))
    _, depth_fine, finite_fine = core._project_global_q_to_4096(q_global_fine, global_camera=global_camera)
    observed_fine = _sample_visibility(
        uv_fine,
        depth_fine,
        finite_fine,
        visibility,
        depth_tolerance_pixels=float(args.depth_tolerance_pixels),
        focal_pixels=float(core._focal_pixels(float(global_camera["camera_angle_x"]), CANONICAL_RESOLUTION)),
    )
    hidden_fine = ~observed_fine

    operator, operator_info = _build_prolongation(coarse_coords, fine_points)
    basis = _basis_partition(operator, observed_fine)
    pure_hidden_ids = basis["pure_hidden_ids"]
    active_columns = np.asarray(operator.getnnz(axis=0)).reshape(-1) > 0
    linearity = _query_attrs_linearity(
        coarse_coords,
        fine_points,
        sample_rows=int(args.operator_test_rows),
        seed=int(args.seed) + int(tile_id) * 1009,
    )

    rng = np.random.default_rng(int(args.seed) + int(tile_id) * 17)
    random_coarse = torch.from_numpy(rng.standard_normal((coarse_coords.shape[0], 3)).astype(np.float32))
    random_projection = _apply_operator(operator, random_coarse)
    # The full P/A operator is shared by G, H, and the AP identity probe.
    # Solving the three RHS blocks together avoids refactorizing P^T P.
    joint_rhs = torch.cat((g_fine, h_fine, random_projection), dim=1)
    a_joint, solve_full = _solve_restriction(operator, joint_rhs, label=f"tile_{tile_id:02d}_G_H_AP")
    a_full = a_joint[:, :PBR_CHANNELS]
    a_full_h = a_joint[:, PBR_CHANNELS : 2 * PBR_CHANNELS]
    random_recovered = a_joint[:, 2 * PBR_CHANNELS :]
    solve_full_h = dict(solve_full)
    solve_full_h["label"] = f"tile_{tile_id:02d}_H_shared_factorization"
    pa_g = _apply_operator(operator, torch.from_numpy(a_full))
    pa_h = _apply_operator(operator, torch.from_numpy(a_full_h))
    d_g = g_fine - pa_g
    d_h = h_fine - pa_h

    random_reference = random_coarse.numpy()
    ap_delta = random_recovered[active_columns] - random_reference[active_columns]
    ap_identity = {
        "max_abs_error": float(np.abs(ap_delta).max()) if ap_delta.size else 0.0,
        "mean_abs_error": float(np.abs(ap_delta).mean()) if ap_delta.size else 0.0,
        "relative_l2": _relative_error(ap_delta, random_reference[active_columns]) if ap_delta.size else 0.0,
        "tested_columns": int(active_columns.sum()),
    }

    ag_vs_gc = _field_error(
        torch.from_numpy(a_full),
        g_coarse,
        mask=torch.from_numpy(active_columns),
    )
    null_joint, solve_null = _solve_restriction(
        operator,
        torch.cat((d_g, d_h), dim=1),
        label=f"tile_{tile_id:02d}_nullspace",
    )
    null_g = null_joint[:, :PBR_CHANNELS]
    null_h = null_joint[:, PBR_CHANNELS : 2 * PBR_CHANNELS]
    nullspace = {
        "G": {
            "max_abs": float(np.abs(null_g).max()) if null_g.size else 0.0,
            "mean_abs": float(np.abs(null_g).mean()) if null_g.size else 0.0,
            "relative_l2": _relative_error(null_g, d_g.numpy()) if null_g.size else 0.0,
        },
        "H": {
            "max_abs": float(np.abs(null_h).max()) if null_h.size else 0.0,
            "mean_abs": float(np.abs(null_h).mean()) if null_h.size else 0.0,
            "relative_l2": _relative_error(null_h, d_h.numpy()) if null_h.size else 0.0,
        },
    }

    corrected_fine = h_fine.clone()
    hidden_solve: Dict[str, Any] = {"method": "empty", "active_columns": 0}
    hidden_correction = torch.zeros_like(h_fine)
    hidden_np = hidden_fine.numpy()
    if bool(hidden_np.any()) and bool(pure_hidden_ids.any()):
        p_hidden = operator[hidden_np][:, pure_hidden_ids].tocsr()
        a_hidden, hidden_solve = _solve_restriction(
            p_hidden,
            h_fine[hidden_fine],
            label=f"tile_{tile_id:02d}_hidden",
        )
        coarse_delta = g_coarse[pure_hidden_ids].numpy() - a_hidden
        hidden_delta = p_hidden.dot(coarse_delta).astype(np.float32)
        corrected_np = corrected_fine.numpy()
        corrected_np[hidden_np] += hidden_delta
        corrected_fine = torch.from_numpy(corrected_np)
        hidden_correction[hidden_fine] = torch.from_numpy(hidden_delta)
        del p_hidden, a_hidden, coarse_delta, hidden_delta, corrected_np

    # The observed branch is intentionally copied verbatim.  This is an
    # explicit invariant, not a post-hoc tolerance check.
    observed_identity = _field_error(corrected_fine, h_fine, observed_fine)
    hidden_hr_vs_g = _field_error(h_fine, g_fine, hidden_fine)
    hidden_phase_vs_g = _field_error(corrected_fine, g_fine, hidden_fine)
    hidden_phase_vs_hr = _field_error(corrected_fine, h_fine, hidden_fine)

    # Evaluate the same hidden correction at decoder mesh vertices for the
    # renderer.  The operator metrics above remain defined only on X_f.
    render_uv, render_depth, render_finite = _project_local_points(
        local_vertices,
        transform=transform,
        global_camera=global_camera,
    )
    render_observed = _sample_visibility(
        render_uv,
        render_depth,
        render_finite,
        visibility,
        depth_tolerance_pixels=float(args.depth_tolerance_pixels),
        focal_pixels=float(core._focal_pixels(float(global_camera["camera_angle_x"]), CANONICAL_RESOLUTION)),
    )
    p_render, p_render_info = _build_prolongation(coarse_coords, local_vertices)
    cached_hr = cached["attrs"]["HR"].detach().cpu().to(torch.float32)
    if cached_hr.shape[0] != local_vertices.shape[0]:
        raise RuntimeError(f"tile {tile_id}: cached HR attrs and decoded mesh vertices differ")
    phase_render = cached_hr.clone()
    if bool(pure_hidden_ids.any()):
        # The correction vector is recoverable from the already computed
        # hidden field correction on any non-empty hidden row.  Recompute the
        # small hidden solve only when needed for the vertex evaluation so the
        # saved phase1 field is unambiguous.
        if bool(hidden_np.any()):
            p_hidden = operator[hidden_np][:, pure_hidden_ids].tocsr()
            a_hidden, _ = _solve_restriction(p_hidden, h_fine[hidden_fine], label=f"tile_{tile_id:02d}_hidden_render")
            coarse_delta = g_coarse[pure_hidden_ids].numpy() - a_hidden
            render_delta = p_render[:, pure_hidden_ids].dot(coarse_delta).astype(np.float32)
            phase_np = phase_render.numpy()
            phase_np[~render_observed] += render_delta[~render_observed]
            phase_render = torch.from_numpy(phase_np)
            del p_hidden, a_hidden, coarse_delta, render_delta, phase_np

    visual_fields = {
        "G256": g_coarse,
        "P_G256": _apply_operator(operator, g_coarse),
        "G1024": g_fine,
        "D_G": d_g,
        "H1024": h_fine,
        "PA_H": pa_h,
        "D_H": d_h,
    }
    if not bool(args.skip_visualization):
        _save_decomposition_visual(
            output_dir / "visualizations" / f"tile_{int(tile_id):02d}_decomposition.png",
            uv_coarse=uv_coarse,
            uv_fine=uv_fine,
            fields=visual_fields,
            box=row["box"],
            hidden=hidden_fine,
        )

    operator_metrics = {
        "tile_id": int(tile_id),
        "box": [int(v) for v in row["box"]],
        "status": "success",
        "decode": decode_stats,
        "geometry": {
            "local_vertices": int(local_vertices.shape[0]),
            "local_faces": int(local_faces.shape[0]),
            "cache_geometry_roundtrip_max_abs": geometry_roundtrip_max,
            "C256_active_support": int(coarse_coords.shape[0]),
            "C1024_active_support": int(fine_coords.shape[0]),
            "query_definition": "active O-voxel cell centers from sparse coordinates; not decoder mesh vertices",
        },
        "mask": {
            "observed_count": int(observed_fine.sum()),
            "hidden_count": int(hidden_fine.sum()),
            "observed_ratio": float(observed_fine.float().mean().item()),
            "rule": visibility.get("rule"),
            "depth_tolerance_pixels": float(args.depth_tolerance_pixels),
        },
        "P": operator_info,
        "linearity_error_query": linearity,
        "basis": {key: value for key, value in basis.items() if key != "pure_hidden_ids"},
        "AP_identity_error": ap_identity,
        "AG1024_vs_G256_error": ag_vs_gc,
        "nullspace_error": nullspace,
        "detail_G": _detail_stats(d_g, g_fine),
        "detail_H": _detail_stats(d_h, h_fine),
        "solvers": {"G": solve_full, "H": solve_full_h, "nullspace": solve_null, "hidden": hidden_solve},
        "hidden_correction": {
            "observed_identity_error": observed_identity,
            "HR_vs_G_hidden": hidden_hr_vs_g,
            "phase1_vs_G_hidden": hidden_phase_vs_g,
            "phase1_vs_H_hidden": hidden_phase_vs_hr,
            "render_observed_count": int(render_observed.sum()),
            "render_hidden_count": int((~render_observed).sum()),
            "render_operator": p_render_info,
        },
        "timing_seconds": float(time.perf_counter() - started),
    }
    _write_json(metrics_path, operator_metrics)
    phase_payload = {
        "format": f"{FORMAT}_tile",
        "tile_id": int(tile_id),
        "box": [int(v) for v in row["box"]],
        "phase1_attrs": phase_render.detach().cpu().to(torch.float32),
        "geometry_vertices": int(local_vertices.shape[0]),
    }
    temporary = tile_dir / f".phase1_tile.{time.time_ns()}.tmp"
    torch.save(phase_payload, temporary)
    os.replace(temporary, payload_path)

    del mesh_hr, local_vertices, local_faces, coarse_coords, fine_coords, coarse_points, fine_points
    del g_coarse, g_fine, h_fine, pa_g, pa_h, d_g, d_h, corrected_fine, hidden_correction
    del operator, visual_fields, cached, endpoint, joint_rhs, a_joint, random_projection, random_coarse, null_joint
    _empty_cuda_cache()
    return operator_metrics, phase_payload


def _render_global_variants(
    *,
    args: argparse.Namespace,
    source_dir: Path,
    output_dir: Path,
    tile_records: Sequence[Mapping[str, Any]],
    global_camera: Mapping[str, float],
) -> Dict[str, Any]:
    successful = [row for row in tile_records if row.get("status") == "success"]
    if not successful:
        return {"status": "skipped_no_successful_tiles"}
    payloads: Dict[int, Dict[str, Any]] = {}
    for row in successful:
        tile_id = int(row["tile_id"])
        source_payload = _load_torch(
            source_dir / "global_stitched_quality" / "decoded_global_tiles" / f"tile_{tile_id:02d}.pt"
        )
        phase_payload = _load_torch(output_dir / "tiles" / f"tile_{tile_id:02d}" / "phase1_tile.pt")
        phase_attrs = phase_payload["phase1_attrs"].detach().cpu().to(torch.float32)
        if phase_attrs.shape != source_payload["attrs"]["HR"].shape:
            raise RuntimeError(f"tile {tile_id}: phase1 attrs do not match cached global HR attrs")
        payloads[tile_id] = {
            "box": list(source_payload["box"]),
            "global_vertices": source_payload["global_vertices"].detach().cpu().to(torch.float32),
            "faces": source_payload["faces"].detach().cpu().to(torch.int32),
            "attrs": {
                "G": source_payload["attrs"]["G"].detach().cpu().to(torch.float32),
                "HR": source_payload["attrs"]["HR"].detach().cpu().to(torch.float32),
                "phase1_hidden_mra": phase_attrs,
            },
        }

    geometry_patches = global_stitch._patches_from_payloads(payloads, "G")
    shared_geometry = global_stitch._stitch_global_geometry(
        geometry_patches,
        global_camera=global_camera,
        face_chunk_size=int(args.stitch_face_chunk_size),
        weld_tolerance=float(args.stitch_tolerance),
    )
    render_args = SimpleNamespace(
        aligned_resolution=int(args.aligned_resolution),
        metric_resolution=int(args.metric_resolution),
        aligned_ssaa=int(args.aligned_ssaa),
        multiview_resolution=int(args.multiview_resolution),
        multiview_ssaa=int(args.multiview_ssaa),
        peel_layers=int(args.peel_layers),
        face_chunk_size=int(args.face_chunk_size),
        radius_scale=float(args.radius_scale),
        use_envmap_bg=bool(args.use_envmap_bg),
        envmap=str(args.envmap),
    )
    render_root = output_dir / "renders"
    render_root.mkdir(parents=True, exist_ok=True)
    envmap = core.load_envmap(str(args.envmap), device="cuda")
    variants: Dict[str, Any] = {}
    for variant in ("G", "HR", "phase1_hidden_mra"):
        merged_attrs = global_stitch._apply_stitched_attrs(shared_geometry, payloads, variant)
        merged = MeshWithVertexPbr(
            vertices=shared_geometry["welded_vertices"],
            faces=shared_geometry["welded_faces"],
            vertex_attrs=merged_attrs,
            layout=dict(core.PBR_LAYOUT),
        )
        print(f"[render] global {variant} vertices={merged.vertices.shape[0]:,} faces={merged.faces.shape[0]:,}", flush=True)
        record = global_stitch._render_variant(
            variant=variant,
            mesh=merged,
            output_root=render_root,
            reference_path=source_dir / "canonical_1024.png",
            global_camera=global_camera,
            args=render_args,
            envmap=envmap,
        )
        variants[variant] = record
        if variant == "phase1_hidden_mra":
            torch.save(
                {
                    "format": f"{FORMAT}_global_mesh",
                    "variant": variant,
                    "mesh": merged,
                    "stitch_stats": shared_geometry["stats"],
                },
                render_root / variant / "global_merged_mesh.pt",
            )
        del merged_attrs, merged
        _empty_cuda_cache()
    del envmap

    # Compact final comparison sheets: aligned front, then six fixed views.
    view_labels = ("aligned_front", "front", "right", "back", "left", "top", "bottom")
    sheet = Image.new("RGB", (128 + len(view_labels) * 192, 42 + 3 * 220), "black")
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(view_labels):
        draw.text((128 + col * 192 + 4, 14), label, fill=(255, 255, 255))
    for row_index, variant in enumerate(("G", "HR", "phase1_hidden_mra")):
        y = 42 + row_index * 220
        draw.text((8, y + 90), variant, fill=(255, 255, 255))
        paths = [Path(variants[variant]["aligned_view"])] + [Path(variants[variant]["multiview"][label]) for label in ("front", "right", "back", "left", "top", "bottom")]
        for col, image_path in enumerate(paths):
            with Image.open(image_path) as source:
                image = ImageOps.contain(source.convert("RGB"), (192, 192))
            sheet.paste(image, (128 + col * 192 + (192 - image.width) // 2, y + (192 - image.height) // 2))
    sheet_path = render_root / "global_phase1_comparison_sheet.png"
    sheet.save(sheet_path)

    channel_sheet = Image.new("RGB", (128 + 3 * 192, 42 + 3 * 220), "black")
    draw = ImageDraw.Draw(channel_sheet)
    for col, label in enumerate(("base_color", "metallic", "roughness")):
        draw.text((128 + col * 192 + 4, 14), label, fill=(255, 255, 255))
    for row_index, variant in enumerate(("G", "HR", "phase1_hidden_mra")):
        y = 42 + row_index * 220
        draw.text((8, y + 90), variant, fill=(255, 255, 255))
        aligned_dir = Path(variants[variant]["aligned_view"]).parent
        for col, channel in enumerate(("base_color", "metallic", "roughness")):
            image_path = aligned_dir / f"{channel}.png"
            if image_path.is_file():
                with Image.open(image_path) as source:
                    image = ImageOps.contain(source.convert("RGB"), (192, 192))
                channel_sheet.paste(image, (128 + col * 192 + (192 - image.width) // 2, y + (192 - image.height) // 2))
    channel_path = render_root / "global_pbr_channel_comparison.png"
    channel_sheet.save(channel_path)
    _write_json(
        render_root / "render_summary.json",
        {
            "format": f"{FORMAT}_render",
            "source_tiles": sorted(payloads),
            "stitch_stats": shared_geometry["stats"],
            "variants": variants,
            "comparison_sheet": str(sheet_path),
            "pbr_channel_sheet": str(channel_path),
        },
    )
    source_tiles = sorted(payloads)
    del geometry_patches, shared_geometry, payloads
    _empty_cuda_cache()
    return {
        "status": "success",
        "source_tiles": source_tiles,
        "variants": variants,
        "comparison_sheet": str(sheet_path),
        "pbr_channel_sheet": str(channel_path),
    }


def _get_nested(row: Mapping[str, Any], path: Sequence[str], default: Optional[float] = None) -> Optional[float]:
    value: Any = row
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _write_reports(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    source_dir: Path,
    global_camera: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    render_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    successful = [row for row in records if row.get("status") == "success"]
    failed = [row for row in records if row.get("status") == "failed"]
    operator_keys = {
        "linearity_query_max": ("linearity_error_query", "max_abs_error"),
        "linearity_query_explicit_P_max": ("linearity_error_query", "explicit_P_match_max_abs_error"),
        "AP_identity_relative": ("AP_identity_error", "relative_l2"),
        "AG1024_vs_G256_relative": ("AG1024_vs_G256_error", "relative_l2"),
        "nullspace_G_relative": ("nullspace_error", "G", "relative_l2"),
        "nullspace_H_relative": ("nullspace_error", "H", "relative_l2"),
        "detail_G_relative": ("detail_G", "relative_l2"),
        "detail_H_relative": ("detail_H", "relative_l2"),
        "observed_identity_max": ("hidden_correction", "observed_identity_error", "max_abs"),
        "hidden_HR_vs_G_relative": ("hidden_correction", "HR_vs_G_hidden", "relative_l2"),
        "hidden_phase_vs_G_relative": ("hidden_correction", "phase1_vs_G_hidden", "relative_l2"),
        "hidden_phase_vs_H_relative": ("hidden_correction", "phase1_vs_H_hidden", "relative_l2"),
    }
    aggregate = {
        name: _metric_stats(_get_nested(row, path) for row in successful)
        for name, path in operator_keys.items()
    }
    hidden_hr = [v for v in (_get_nested(row, operator_keys["hidden_HR_vs_G_relative"]) for row in successful) if v is not None]
    hidden_phase = [v for v in (_get_nested(row, operator_keys["hidden_phase_vs_G_relative"]) for row in successful) if v is not None]
    aggregate["hidden_proxy_improvement_ratio_phase_over_hr"] = _metric_stats(
        (p / h if h > 1e-12 else None) for p, h in zip(hidden_phase, hidden_hr)
    )
    aggregate["observed_ratio"] = _metric_stats(_get_nested(row, ("mask", "observed_ratio")) for row in successful)
    aggregate["pure_hidden_basis_count"] = _metric_stats(_get_nested(row, ("basis", "pure_hidden_basis_count")) for row in successful)
    aggregate["mixed_basis_count"] = _metric_stats(_get_nested(row, ("basis", "mixed_basis_count")) for row in successful)
    aggregate["coverage_ratio"] = _metric_stats(_get_nested(row, ("P", "coverage_ratio")) for row in successful)

    tile_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    for row in records:
        tile_id = row.get("tile_id")
        if row.get("status") != "success":
            tile_rows.append({"tile_id": tile_id, "status": row.get("status"), "reason": row.get("reason")})
            continue
        tile = {
            "tile_id": tile_id,
            "status": row.get("status"),
            "C256_active": _get_nested(row, ("geometry", "C256_active_support")),
            "C1024_active": _get_nested(row, ("geometry", "C1024_active_support")),
            "coverage_ratio": _get_nested(row, ("P", "coverage_ratio")),
            "observed_ratio": _get_nested(row, ("mask", "observed_ratio")),
            "hidden_basis_count": _get_nested(row, ("basis", "hidden_basis_count")),
            "pure_hidden_basis_count": _get_nested(row, ("basis", "pure_hidden_basis_count")),
            "mixed_basis_count": _get_nested(row, ("basis", "mixed_basis_count")),
            "linearity_error_query_max": _get_nested(row, ("linearity_error_query", "max_abs_error")),
            "AP_identity_relative": _get_nested(row, ("AP_identity_error", "relative_l2")),
            "AG1024_vs_G256_relative": _get_nested(row, ("AG1024_vs_G256_error", "relative_l2")),
            "nullspace_G_relative": _get_nested(row, ("nullspace_error", "G", "relative_l2")),
            "nullspace_H_relative": _get_nested(row, ("nullspace_error", "H", "relative_l2")),
            "detail_G_relative": _get_nested(row, ("detail_G", "relative_l2")),
            "detail_H_relative": _get_nested(row, ("detail_H", "relative_l2")),
            "hidden_HR_vs_G_relative": _get_nested(row, ("hidden_correction", "HR_vs_G_hidden", "relative_l2")),
            "hidden_phase_vs_G_relative": _get_nested(row, ("hidden_correction", "phase1_vs_G_hidden", "relative_l2")),
            "hidden_phase_vs_H_relative": _get_nested(row, ("hidden_correction", "phase1_vs_H_hidden", "relative_l2")),
            "observed_identity_max": _get_nested(row, ("hidden_correction", "observed_identity_error", "max_abs")),
            "visualization": str(output_dir / "visualizations" / f"tile_{int(tile_id):02d}_decomposition.png"),
        }
        tile_rows.append(tile)
        metric_rows.extend(
            [
                {"scope": "tile", "tile_id": tile_id, "variant": "HR", "region": "hidden", "metric": "relative_l2_vs_G", "value": tile["hidden_HR_vs_G_relative"]},
                {"scope": "tile", "tile_id": tile_id, "variant": "phase1_hidden_mra", "region": "hidden", "metric": "relative_l2_vs_G", "value": tile["hidden_phase_vs_G_relative"]},
                {"scope": "tile", "tile_id": tile_id, "variant": "phase1_hidden_mra", "region": "hidden", "metric": "relative_l2_vs_HR", "value": tile["hidden_phase_vs_H_relative"]},
                {"scope": "tile", "tile_id": tile_id, "variant": "phase1_hidden_mra", "region": "observed", "metric": "max_abs_vs_HR", "value": tile["observed_identity_max"]},
            ]
        )

    render_variants = render_summary.get("variants", {}) if isinstance(render_summary, Mapping) else {}
    for variant, data in render_variants.items():
        aligned_metrics_path = Path(data["aligned_metrics"])
        if aligned_metrics_path.is_file():
            aligned_metrics = json.loads(aligned_metrics_path.read_text(encoding="utf-8"))
            for metric_name in ("psnr_db", "ssim", "lpips"):
                if aligned_metrics.get(metric_name) is not None:
                    metric_rows.append({"scope": "global_aligned", "tile_id": "all", "variant": variant, "region": "front", "metric": metric_name, "value": aligned_metrics[metric_name]})

    _write_csv(output_dir / "tile_stats.csv", tile_rows)
    _write_csv(output_dir / "metrics.csv", metric_rows)
    operator_payload = {
        "format": f"{FORMAT}_operator_metrics",
        "source_dir": str(source_dir),
        "cuda_device": int(args.cuda_device),
        "successful_tiles": [int(row["tile_id"]) for row in successful],
        "failed_tiles": [int(row["tile_id"]) for row in failed],
        "aggregate": aggregate,
        "tiles": list(records),
    }
    _write_json(output_dir / "operator_metrics.json", operator_payload)

    worst = sorted(successful, key=lambda row: _get_nested(row, ("AG1024_vs_G256_error", "relative_l2"), 0.0) or 0.0, reverse=True)
    mean_ag = aggregate["AG1024_vs_G256_relative"].get("mean")
    mean_linearity = aggregate["linearity_query_max"].get("max")
    mean_ap = aggregate["AP_identity_relative"].get("mean")
    mean_null = max(
        aggregate["nullspace_G_relative"].get("max") or 0.0,
        aggregate["nullspace_H_relative"].get("max") or 0.0,
    )
    mean_hidden_ratio = aggregate["hidden_proxy_improvement_ratio_phase_over_hr"].get("mean")
    p_linear = bool((mean_linearity or 0.0) < 1e-5 and (aggregate["linearity_query_explicit_P_max"].get("max") or 0.0) < 1e-5)
    p_identity = bool((mean_ap or 0.0) < 1e-3)
    commute = bool((mean_ag is not None) and mean_ag < 0.1)
    next_stage = bool(p_linear and p_identity and commute and mean_null < 1e-3)
    report_lines = [
        "# Phase 1：local C256/C1024 sparse MRA hidden-only 验证",
        "",
        "## 实验范围",
        "",
        f"- source cache: `{source_dir}`",
        f"- CUDA device: `{args.cuda_device}`",
        f"- successful tiles: `{len(successful)}`; failed tiles: `{len(failed)}`",
        "- geometry fixed；PureHR 最终 tile endpoint 只用于得到 H1024，不调用 per-step guidance、不改主 flow。",
        "- P/A 的 query rows 是同一 local mesh 产生的 active O-voxel cell centers，不是 decoder mesh vertices。",
        "",
        "## 汇总数值",
        "",
        f"- query_attrs 线性误差 max：`{mean_linearity}`；显式 P 与 query_attrs 误差 max：`{aggregate['linearity_query_explicit_P_max'].get('max')}`",
        f"- A(Pc)≈c relative：mean=`{mean_ap}`",
        f"- A G1024≈G256 relative：mean=`{mean_ag}`",
        f"- null-space relative max（G/H）：`{mean_null}`",
        f"- hidden proxy phase1/G 相对误差 ÷ PureHR/G 相对误差：mean=`{mean_hidden_ratio}`（<1 表示更接近 baseline coarse field）",
        f"- observed branch max abs 改动：`{aggregate['observed_identity_max'].get('max')}`",
        "",
        "## 最差 tile",
        "",
    ]
    for row in worst[:5]:
        report_lines.append(
            f"- tile {int(row['tile_id']):02d}: AG1024/G256=`{_get_nested(row, ('AG1024_vs_G256_error', 'relative_l2'))}`, "
            f"linearity=`{_get_nested(row, ('linearity_error_query', 'max_abs_error'))}`, "
            f"visual=`visualizations/tile_{int(row['tile_id']):02d}_decomposition.png`"
        )
    report_lines.extend(
        [
            "",
            "## 必须问题的结论",
            "",
            f"1. `query_attrs` 能否作为线性 P：**{'可以' if p_linear else '数值线性成立，但显式 P 对 query_attrs 不达标'}**。判据为测试误差 <1e-5；具体数值见 `operator_metrics.json`。",
            f"2. A_kP_k≈I：**{'成立' if p_identity else '不成立/数值误差偏大'}**。",
            f"3. A_kG1024≈G256：**{'成立' if commute else '不成立'}**；这是 local degradation 的关键 commutativity 检验，mean relative=`{mean_ag}`。",
            "4. detail 是否主要是细节：D_G 与 D_H 的数值能量和 RGB/metallic/roughness 分通道统计已写入 JSON；请结合 `visualizations/` 看 residual 是否集中在局部纹理/边缘。该脚本不把 null-space 资格自动等同于语义细节。",
            f"5. hidden-only 结果是否优于 PureHR：按无 GT 的 baseline-coherence proxy，phase1/PureHR relative ratio mean=`{mean_hidden_ratio}`；{'更接近 G' if mean_hidden_ratio is not None and mean_hidden_ratio < 1.0 else '没有显示出改善'}。这不是 hidden GT 指标。",
            f"6. 前景/observed 是否保持不变：**{'是' if (aggregate['observed_identity_max'].get('max') or 0.0) < 1e-7 else '否'}**，fine query rows 的 observed max abs=`{aggregate['observed_identity_max'].get('max')}`。",
            f"7. 是否值得进入 per-step flow guidance：**{'可以进入下一阶段' if next_stage else '暂不建议'}**；需要先解决 P/A commutativity 或 detail 语义问题。",
            "",
            "## 产物",
            "",
            "- `operator_metrics.json`：逐 tile operator/solver/mask/detail 指标。",
            "- `tile_stats.csv`、`metrics.csv`：表格化结果。",
            "- `visualizations/`：G256、P G256、G1024、D_G、H1024、P A H、D_H、hidden mask。",
            "- `renders/`：global baseline、PureHR、phase1 的 aligned/front/back 等视图及 PBR channel 对比。",
        ]
    )
    report_path = output_dir / "PHASE1_REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    summary = {
        "format": FORMAT,
        "source_dir": str(source_dir),
        "cuda_device": int(args.cuda_device),
        "global_camera": dict(global_camera),
        "successful_tiles": [int(row["tile_id"]) for row in successful],
        "failed_tiles": [int(row["tile_id"]) for row in failed],
        "aggregate": aggregate,
        "conclusions": {
            "query_attrs_is_linear_P": p_linear,
            "AP_identity_pass": p_identity,
            "AG1024_vs_G256_pass": commute,
            "hidden_proxy_phase_better_than_HR": bool(mean_hidden_ratio is not None and mean_hidden_ratio < 1.0),
            "observed_unchanged": bool((aggregate["observed_identity_max"].get("max") or 0.0) < 1e-7),
            "recommend_per_step_next_stage": next_stage,
        },
        "render": dict(render_summary),
        "artifacts": {
            "report": str(report_path),
            "operator_metrics": str(output_dir / "operator_metrics.json"),
            "tile_stats": str(output_dir / "tile_stats.csv"),
            "metrics": str(output_dir / "metrics.csv"),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default="outputs/codex_texture_pbr_degradation_cuda4_all_tiles",
        help="CUDA4 cache containing global baseline, endpoints and decoded global tile meshes",
    )
    parser.add_argument("--output-dir", default="outputs/sparse_mra_hidden_phase1")
    parser.add_argument("--visibility-dir", default="outputs/visibility_guided_pbr_flow_cuda4_mesh_ovoxel_slat/visibility_4096")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--tile-ids", default=None, help="comma-separated successful tile ids")
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-render", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-visualization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--query-chunk-size", type=int, default=65_536)
    parser.add_argument("--operator-test-rows", type=int, default=8_192)
    parser.add_argument("--depth-tolerance-pixels", type=float, default=4.0)
    parser.add_argument("--geometry-tolerance", type=float, default=1e-4)
    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--aligned-resolution", type=int, default=1024)
    parser.add_argument("--metric-resolution", type=int, default=512)
    parser.add_argument("--aligned-ssaa", type=int, default=2)
    parser.add_argument("--multiview-resolution", type=int, default=512)
    parser.add_argument("--multiview-ssaa", type=int, default=1)
    parser.add_argument("--peel-layers", type=int, default=8)
    parser.add_argument("--face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--stitch-face-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--stitch-tolerance", type=float, default=1.0 / 1024.0)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--use-envmap-bg", action=argparse.BooleanOptionalAction, default=False)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if int(args.cuda_device) < 0 or int(args.cuda_device) >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {args.cuda_device} is unavailable")
    if int(args.query_chunk_size) <= 0 or int(args.operator_test_rows) <= 0:
        raise ValueError("query/operator chunk sizes must be positive")
    if float(args.depth_tolerance_pixels) <= 0.0:
        raise ValueError("--depth-tolerance-pixels must be positive")
    if float(args.stitch_tolerance) <= 0.0:
        raise ValueError("--stitch-tolerance must be positive")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _validate_args(args)
    torch.cuda.set_device(int(args.cuda_device))
    print(
        f"[cuda] requested/current={int(args.cuda_device)}/{torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())}",
        flush=True,
    )
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    visibility_dir = Path(args.visibility_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    global_camera = json.loads((source_dir / "global_camera.json").read_text(encoding="utf-8"))
    selected = _parse_ids(args.tile_ids)
    rows: List[Dict[str, Any]] = []
    for row in source_summary.get("tiles", []):
        if row.get("status") != "success":
            continue
        tile_id = int(row["tile_id"])
        if selected is not None and tile_id not in selected:
            continue
        rows.append(dict(row))
    rows.sort(key=lambda row: int(row["tile_id"]))
    if args.max_tiles is not None:
        rows = rows[: int(args.max_tiles)]
    if not rows:
        raise RuntimeError("no successful source tiles selected")

    visibility = _load_visibility(visibility_dir)
    print(f"[visibility] {visibility.get('rule')} source={visibility_dir}", flush=True)
    baseline_payload = _load_torch(source_dir / "global_baseline_mesh.pt")
    baseline_mesh = baseline_payload["mesh"] if isinstance(baseline_payload, Mapping) else baseline_payload
    if not isinstance(baseline_mesh, MeshWithVoxel):
        raise RuntimeError(f"expected MeshWithVoxel global baseline, got {type(baseline_mesh)!r}")
    global_field = core._make_attribute_query_mesh(baseline_mesh, torch.device("cuda"))
    del baseline_payload, baseline_mesh
    _empty_cuda_cache()

    need_processing = []
    for row in rows:
        tile_dir = output_dir / "tiles" / f"tile_{int(row['tile_id']):02d}"
        if not (bool(args.resume) and (tile_dir / "operator_metrics.json").is_file() and (tile_dir / "phase1_tile.pt").is_file()):
            need_processing.append(row)
    pipeline = None
    if need_processing:
        pipeline = init_pipeline(args.model_path, device="cuda", low_vram=True)

    records: List[Dict[str, Any]] = []
    for row in rows:
        tile_id = int(row["tile_id"])
        try:
            metrics, _ = _process_tile(
                args=args,
                source_dir=source_dir,
                output_dir=output_dir,
                tile_id=tile_id,
                row=row,
                pipeline=pipeline,
                global_field=global_field,
                global_camera=global_camera,
                visibility=visibility,
            )
            records.append(metrics)
            print(
                f"[tile {tile_id:02d}] done AG/G={_get_nested(metrics, ('AG1024_vs_G256_error', 'relative_l2'))} "
                f"hidden={_get_nested(metrics, ('mask', 'observed_ratio'))}",
                flush=True,
            )
        except Exception as exc:
            failed = {
                "tile_id": tile_id,
                "box": row.get("box"),
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            records.append(failed)
            _write_json(output_dir / "tiles" / f"tile_{tile_id:02d}" / "operator_metrics.json", failed)
            print(f"[tile {tile_id:02d}] FAILED: {failed['reason']}", flush=True)
            traceback.print_exc()
        finally:
            _empty_cuda_cache()

    del pipeline, global_field
    _empty_cuda_cache()
    render_summary: Dict[str, Any]
    if bool(args.skip_render):
        render_summary = {"status": "skipped_by_cli"}
    else:
        try:
            render_summary = _render_global_variants(
                args=args,
                source_dir=source_dir,
                output_dir=output_dir,
                tile_records=records,
                global_camera=global_camera,
            )
        except Exception as exc:
            render_summary = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            print(f"[render] FAILED: {render_summary['reason']}", flush=True)
            traceback.print_exc()
    summary = _write_reports(
        args=args,
        output_dir=output_dir,
        source_dir=source_dir,
        global_camera=global_camera,
        records=records,
        render_summary=render_summary,
    )
    print(f"[done] report={summary['artifacts']['report']}", flush=True)
    return summary


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
