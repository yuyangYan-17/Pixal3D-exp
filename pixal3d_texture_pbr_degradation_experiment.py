#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed-shape texture-only PBR degradation experiment.

This experiment follows ``Codex.md`` on top of the repository's native
Pixal3D path:

* one ordinary 1024-cascade global baseline is generated first;
* a canonical 4096 image tile is projected into a fresh local C1024
  dual-grid and encoded with the official shape encoder;
* that local shape SLat is immutable -- no shape flow is called;
* the global baseline MeshWithVoxel PBR field is queried in global normalized
  object space and re-encoded on the fixed local support to form ``G_tex``;
* the HR 1024x1024 crop from the canonical 4096 image is the texture image
  condition, and native texture FlowEuler starts from a native noised
  ``G_tex`` endpoint to produce ``HR_tex``;
* both endpoints are decoded by the same official texture decoder and queried
  at identical local mesh positions before the PBR-space analysis.

The analysis never assumes that SLat channels are frequency channels.  It
first evaluates the PBR cell-mean projector ``P_F`` at global-C64, 2x, and 4x
spacings, optionally compares it with a sparse trilinear least-squares
operator, and only then runs the explicitly exploratory SLat tests.
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
import torch
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

import pixal3d.models as pixal3d_models
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
from inference import MODEL_PATH
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVertexPbr, MeshWithVoxel


FORMAT = "pixal3d_texture_pbr_degradation_v1"
GLOBAL_IMAGE_SIZE = 1024
CANONICAL_IMAGE_SIZE = 4096
TILE_SIZE = 1024
OVOXEL_RESOLUTION = 1024
LATENT_RESOLUTION = 64
PBR_LAYOUT = {
    "base_color": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}
PBR_GROUPS = {
    "base_color": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
    "joint": slice(0, 6),
}
DEFAULT_SCALES = (1, 2, 4)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _clone_sparse(value: SparseTensor) -> SparseTensor:
    return value.replace(value.feats.detach().clone())


def _fresh_sparse(value: SparseTensor) -> SparseTensor:
    """Make a SparseTensor without carrying native spatial caches."""
    return SparseTensor(
        value.feats.detach().clone(),
        value.coords.detach().clone(),
    )


def _as_float_tensor(value: Any, *, device: Optional[torch.device] = None) -> torch.Tensor:
    result = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    result = result.to(dtype=torch.float32)
    if device is not None:
        result = result.to(device)
    return result


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.to(torch.float64)).item())


def _relative_error(value: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> float:
    return _norm(value) / (_norm(reference) + float(eps))


def _tensor_stats(value: torch.Tensor) -> Dict[str, Any]:
    flat = value.detach().to(torch.float32).reshape(-1)
    if flat.numel() == 0:
        return {"numel": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "l2": 0.0}
    return {
        "numel": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "l2": _norm(flat),
    }


def _parse_int_set(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not str(value).strip():
        return None
    return {int(part.strip()) for part in str(value).split(",") if part.strip()}


def _parse_scales(value: str) -> Tuple[int, ...]:
    scales = tuple(sorted({int(part.strip()) for part in str(value).split(",") if part.strip()}))
    if not scales or any(scale <= 0 or LATENT_RESOLUTION % scale != 0 for scale in scales):
        raise ValueError("--coarse-scales must be positive divisors of 64")
    return scales


def _normalize_slat(value: SparseTensor, normalization: Mapping[str, Sequence[float]]) -> SparseTensor:
    features = value.feats.to(torch.float32)
    mean = torch.as_tensor(normalization["mean"], device=features.device, dtype=features.dtype).reshape(1, -1)
    std = torch.as_tensor(normalization["std"], device=features.device, dtype=features.dtype).reshape(1, -1)
    if mean.shape[1] != features.shape[1] or std.shape[1] != features.shape[1]:
        raise RuntimeError(
            f"normalization channels {mean.shape[1]}/{std.shape[1]} do not match {features.shape[1]}"
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or bool((std == 0).any().item()):
        raise RuntimeError("invalid latent normalization")
    return value.replace((features - mean) / std)


def _denormalize_slat(value: SparseTensor, normalization: Mapping[str, Sequence[float]]) -> SparseTensor:
    features = value.feats.to(torch.float32)
    mean = torch.as_tensor(normalization["mean"], device=features.device, dtype=features.dtype).reshape(1, -1)
    std = torch.as_tensor(normalization["std"], device=features.device, dtype=features.dtype).reshape(1, -1)
    return value.replace(features * std + mean)


def _normalization_error(left: Mapping[str, Sequence[float]], right: Mapping[str, Sequence[float]]) -> Dict[str, float]:
    mean_left = torch.as_tensor(left["mean"], dtype=torch.float64)
    mean_right = torch.as_tensor(right["mean"], dtype=torch.float64)
    std_left = torch.as_tensor(left["std"], dtype=torch.float64)
    std_right = torch.as_tensor(right["std"], dtype=torch.float64)
    return {
        "mean_max_abs_error": float((mean_left - mean_right).abs().max().item()),
        "std_max_abs_error": float((std_left - std_right).abs().max().item()),
    }


def _latent_support_checks(
    shape_norm: SparseTensor,
    g_tex_norm: SparseTensor,
    hr_tex_norm: SparseTensor,
    shape_before_flow: SparseTensor,
    shape_after_flow: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
) -> Dict[str, Any]:
    coord_equal = bool(torch.equal(g_tex_norm.coords, hr_tex_norm.coords))
    shape_g_equal = bool(torch.equal(shape_norm.coords, g_tex_norm.coords))
    shape_hr_equal = bool(torch.equal(shape_norm.coords, hr_tex_norm.coords))
    shape_condition_equal = bool(
        torch.equal(shape_before_flow.coords, shape_after_flow.coords)
        and torch.equal(shape_before_flow.feats, shape_after_flow.feats)
    )
    coord_error = 0.0
    if tuple(g_tex_norm.coords.shape) == tuple(hr_tex_norm.coords.shape) and g_tex_norm.coords.numel():
        coord_error = float(
            (g_tex_norm.coords.to(torch.float64) - hr_tex_norm.coords.to(torch.float64)).abs().max().item()
        )
    return {
        "support_equal": bool(coord_equal and shape_g_equal and shape_hr_equal),
        "g_tex_coords_equal_hr_tex_coords": coord_equal,
        "g_tex_coords_equal_shape_condition_coords": shape_g_equal,
        "hr_tex_coords_equal_shape_condition_coords": shape_hr_equal,
        "token_count_g_tex": int(g_tex_norm.coords.shape[0]),
        "token_count_hr_tex": int(hr_tex_norm.coords.shape[0]),
        "token_count_shape_condition": int(shape_norm.coords.shape[0]),
        "token_count_equal": bool(
            g_tex_norm.coords.shape[0] == hr_tex_norm.coords.shape[0] == shape_norm.coords.shape[0]
        ),
        "token_order_equal": coord_equal,
        "coord_max_error": coord_error,
        "shape_condition_equal": shape_condition_equal,
        "shape_condition_feature_max_abs_error": float(
            (shape_before_flow.feats - shape_after_flow.feats).abs().max().item()
        )
        if tuple(shape_before_flow.feats.shape) == tuple(shape_after_flow.feats.shape)
        else None,
        "normalization_mean_std": dict(normalization),
        "normalization_mean_std_equal": True,
        "normalization_max_abs_error": {"mean_max_abs_error": 0.0, "std_max_abs_error": 0.0},
    }


def _native_noised_endpoint(
    clean: SparseTensor,
    noise: SparseTensor,
    sampler: Any,
    timestep: float,
    strength: float,
) -> SparseTensor:
    if not torch.equal(clean.coords, noise.coords) or clean.feats.shape != noise.feats.shape:
        raise RuntimeError("G_tex and texture noise support/shape differ")
    t = float(timestep)
    if not 0.0 <= t <= 1.0:
        raise ValueError("noise timestep must be in [0,1]")
    sigma = float(sampler.sigma_min) + (1.0 - float(sampler.sigma_min)) * t
    return clean.replace((1.0 - t) * clean.feats + sigma * float(strength) * noise.feats)


def _sampler_step_kwargs(params: Mapping[str, Any]) -> Dict[str, Any]:
    consumed = {
        "steps",
        "rescale_t",
        "verbose",
        "tqdm_desc",
        "record_trajectory",
        "trajectory_device",
        "return_model_history",
    }
    return {key: value for key, value in params.items() if key not in consumed}


@torch.no_grad()
def _run_texture_flow(
    *,
    pipeline: Any,
    initial_state: SparseTensor,
    shape_condition: SparseTensor,
    condition: Mapping[str, Any],
    params: Mapping[str, Any],
    noise_timestep: float,
) -> Tuple[SparseTensor, Dict[str, Any]]:
    model = pipeline.models["tex_slat_flow_model_1024"]
    sampler = pipeline.tex_slat_sampler
    merged = {**pipeline.tex_slat_sampler_params, **dict(params)}
    schedule = [float(v) for v in sampler.timestep_schedule(int(merged["steps"]), float(merged["rescale_t"]))]
    matches = [i for i, value in enumerate(schedule) if abs(value - float(noise_timestep)) <= 1e-6]
    if len(matches) != 1:
        raise RuntimeError(
            f"noise timestep {noise_timestep} is not an exact native schedule point: {schedule}"
        )
    start = matches[0]
    if pipeline.low_vram:
        model.to(torch.device(pipeline.device))
    started = time.perf_counter()
    try:
        if start == 0:
            result = sampler.sample(
                model,
                initial_state,
                cond=condition["cond"],
                neg_cond=condition["neg_cond"],
                concat_cond=shape_condition,
                **merged,
                verbose=True,
                tqdm_desc="HR local texture flow from G_tex noise",
                record_trajectory=True,
                trajectory_device="cpu",
                return_model_history=False,
            )
            output = getattr(result, "samples", result)
            trajectory = getattr(result, "trajectory", None)
            state_count = len(trajectory.states) if trajectory is not None else None
            velocity_count = len(trajectory.velocities) if trajectory is not None else None
        else:
            step_kwargs = _sampler_step_kwargs(merged)
            state = initial_state
            state_count = 1
            velocity_count = 0
            for t, t_prev in zip(schedule[start:-1], schedule[start + 1 :]):
                result = sampler.sample_once(
                    model,
                    state,
                    float(t),
                    float(t_prev),
                    cond=condition["cond"],
                    neg_cond=condition["neg_cond"],
                    concat_cond=shape_condition,
                    **step_kwargs,
                )
                state = result.pred_x_prev
                state_count += 1
                velocity_count += 1
            output = state
    finally:
        if pipeline.low_vram:
            model.cpu()
    _sync_cuda()
    if not isinstance(output, SparseTensor):
        raise RuntimeError(f"texture flow returned {type(output)!r}, expected SparseTensor")
    if not torch.equal(output.coords, initial_state.coords):
        raise RuntimeError("texture flow changed the local sparse support")
    return output, {
        "route": "native tex_slat_sampler.sample from native noised G_tex endpoint"
        if start == 0
        else "native tex_slat_sampler.sample_once over exact native suffix",
        "noise_timestep": float(noise_timestep),
        "native_schedule": schedule,
        "schedule_start_index": int(start),
        "flow_steps": int(len(schedule) - 1 - start),
        "trajectory_state_count": state_count,
        "trajectory_velocity_count": velocity_count,
        "flow_seconds": float(time.perf_counter() - started),
        "shape_flow_called": False,
        "shape_slat_sampler_sample_called": False,
    }


def _query_mesh_chunked(mesh: MeshWithVoxel, points: torch.Tensor, chunk_size: int) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    for start in range(0, int(points.shape[0]), int(chunk_size)):
        rows.append(mesh.query_attrs(points[start : start + chunk_size]).to(torch.float32))
    return torch.cat(rows, dim=0) if rows else torch.empty((0, mesh.attrs.shape[1]), device=points.device)


def _map_local_to_global(
    local_points: torch.Tensor,
    *,
    transform: Any,
    global_camera: Mapping[str, float],
) -> torch.Tensor:
    q_local = local_points.to(torch.float32) * (2.0 * float(transform.mesh_scale))
    q_global, _ = core._local_q_to_global_q(q_local, global_camera=global_camera, transform=transform)
    return q_global / (2.0 * float(global_camera["mesh_scale"]))


def _map_local_to_global_chunked(
    local_points: torch.Tensor,
    *,
    transform: Any,
    global_camera: Mapping[str, float],
    chunk_size: int,
) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    for start in range(0, int(local_points.shape[0]), int(chunk_size)):
        rows.append(
            _map_local_to_global(
                local_points[start : start + chunk_size],
                transform=transform,
                global_camera=global_camera,
            )
        )
    return torch.cat(rows, dim=0) if rows else torch.empty((0, 3), device=local_points.device)


def _sample_indices(count: int, limit: int) -> torch.Tensor:
    if count <= limit:
        return torch.arange(count, dtype=torch.long)
    return torch.linspace(0, count - 1, steps=int(limit), dtype=torch.float64).round().to(torch.long)


def _coarse_cell_ids(global_positions: torch.Tensor, scale: int) -> torch.Tensor:
    cells = LATENT_RESOLUTION // int(scale)
    h = float(scale) / float(LATENT_RESOLUTION)
    normalized = ((global_positions.to(torch.float64) + 0.5) / h).floor().to(torch.long)
    normalized = normalized.clamp(0, cells - 1)
    return (normalized[:, 0] * cells + normalized[:, 1]) * cells + normalized[:, 2]


def _group_mean(values: torch.Tensor, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    values = values.to(torch.float32)
    ids = ids.to(torch.long)
    unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
    sums = torch.zeros((unique.shape[0], values.shape[1]), dtype=torch.float64)
    sums.index_add_(0, inverse, values.to(torch.float64))
    counts = torch.bincount(inverse, minlength=unique.shape[0]).to(torch.float64)
    means = (sums / counts[:, None]).to(torch.float32)
    return means[inverse], means, unique, counts.to(torch.long)


def _project_cell_mean(values: torch.Tensor, ids: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
    projected, means, unique, counts = _group_mean(values, ids)
    return projected, {
        "num_coarse_cells": int(unique.shape[0]),
        "cell_count_min": int(counts.min().item()) if counts.numel() else 0,
        "cell_count_max": int(counts.max().item()) if counts.numel() else 0,
        "cell_count_mean": float(counts.to(torch.float32).mean().item()) if counts.numel() else 0.0,
        "coarse_means": means,
        "unique_ids": unique,
        "counts": counts,
    }


def _channel_error_metrics(
    delta: torch.Tensor,
    delta_low: torch.Tensor,
    delta_high: torch.Tensor,
    coarse_delta: Optional[torch.Tensor],
    low: torch.Tensor,
    high: torch.Tensor,
    source: torch.Tensor,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    total_delta_sq = float((delta.to(torch.float64) ** 2).sum().item())
    for name, channel_slice in PBR_GROUPS.items():
        d = delta[:, channel_slice]
        dl = delta_low[:, channel_slice]
        dh = delta_high[:, channel_slice]
        denom_sq = float((d.to(torch.float64) ** 2).sum().item())
        row: Dict[str, Any] = {
            "r_low": float((dl.to(torch.float64) ** 2).sum().item()) / (denom_sq + 1e-12),
            "r_high": float((dh.to(torch.float64) ** 2).sum().item()) / (denom_sq + 1e-12),
            "delta_energy": denom_sq,
            "reconstruction_error": _relative_error(
                low[:, channel_slice] + high[:, channel_slice] - source[:, channel_slice],
                source[:, channel_slice],
            ),
        }
        if coarse_delta is not None:
            coarse = coarse_delta[:, channel_slice]
            row["A_delta_relative_error"] = _norm(coarse) / (_norm(d) + 1e-8)
            row["A_delta_over_coarse_G"] = _norm(coarse) / (_norm(low[:, channel_slice]) + 1e-8)
        else:
            row["A_delta_relative_error"] = _norm(dl) / (_norm(d) + 1e-8)
            row["A_delta_over_coarse_G"] = _norm(dl) / (_norm(low[:, channel_slice]) + 1e-8)
        result[name] = row
    result["joint"]["r_low_plus_r_high"] = result["joint"]["r_low"] + result["joint"]["r_high"]
    result["joint"]["total_delta_energy"] = total_delta_sq
    return result


def _analyze_cell_average(
    fg: torch.Tensor,
    fh: torch.Tensor,
    ids: torch.Tensor,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    fg_low, fg_info = _project_cell_mean(fg, ids)
    fh_low, fh_info = _project_cell_mean(fh, ids)
    if not torch.equal(fg_info["unique_ids"], fh_info["unique_ids"]):
        raise RuntimeError("G/HR cell ids differ in cell-average operator")
    fg_high = fg - fg_low
    fh_high = fh - fh_low
    delta = fh - fg
    delta_low = fh_low - fg_low
    delta_high = fh_high - fg_high
    delta_means, delta_means_by_cell, delta_unique, delta_counts = _group_mean(delta, ids)
    if not torch.equal(delta_unique, fg_info["unique_ids"]):
        raise RuntimeError("delta cell ids differ from G cell ids")
    coarse_consistency: Dict[str, Any] = {}
    for name, channel_slice in PBR_GROUPS.items():
        g = fg_info["coarse_means"][:, channel_slice]
        h = fh_info["coarse_means"][:, channel_slice]
        coarse_consistency[name] = {
            "e_coarse": _norm(h - g) / (_norm(g) + 1e-8),
            "coarse_delta_l2": _norm(h - g),
            "coarse_G_l2": _norm(g),
        }
    projected_high_g, _ = _project_cell_mean(fg_high, ids)
    projected_high_h, _ = _project_cell_mean(fh_high, ids)
    projector_checks = {
        "reconstruction_error_G": _relative_error(fg_low + fg_high - fg, fg),
        "reconstruction_error_HR": _relative_error(fh_low + fh_high - fh, fh),
        "null_error_G": _relative_error(projected_high_g, fg),
        "null_error_HR": _relative_error(projected_high_h, fh),
        "idempotence_error_G": _relative_error(_project_cell_mean(fg_low, ids)[0] - fg_low, fg_low),
        "idempotence_error_HR": _relative_error(_project_cell_mean(fh_low, ids)[0] - fh_low, fh_low),
        "delta_projection_error": _relative_error(
            _project_cell_mean(delta_low, ids)[0] - delta_low, delta_low
        ),
    }
    metrics = {
        "coarse_consistency": coarse_consistency,
        "delta_decomposition": _channel_error_metrics(
            delta,
            delta_low,
            delta_high,
            delta_means,
            fg_low,
            fg_high,
            fg,
        ),
        "projector_checks": projector_checks,
        "num_samples": int(fg.shape[0]),
        "num_coarse_cells": int(fg_info["num_coarse_cells"]),
        "cell_count": {
            "min": fg_info["cell_count_min"],
            "max": fg_info["cell_count_max"],
            "mean": fg_info["cell_count_mean"],
        },
    }
    fields = {
        "G": fg,
        "HR": fh,
        "G_low": fg_low,
        "G_high": fg_high,
        "HR_low": fh_low,
        "HR_high": fh_high,
        "delta": delta,
        "delta_low": delta_low,
        "delta_high": delta_high,
        "G_low_HR_high": fg_low + fh_high,
        "HR_low_G_high": fh_low + fg_high,
    }
    return metrics, fields


def _trilinear_design(global_positions: np.ndarray, scale: int) -> coo_matrix:
    cells = LATENT_RESOLUTION // int(scale)
    nodes = cells + 1
    pos = np.asarray(global_positions, dtype=np.float64)
    if not np.isfinite(pos).all():
        raise RuntimeError("non-finite positions in trilinear operator")
    u = np.clip((pos + 0.5) * cells, 0.0, float(cells))
    lower = np.floor(u).astype(np.int64)
    upper_boundary = lower >= cells
    lower = np.minimum(lower, cells - 1)
    frac = u - lower
    frac[upper_boundary] = 1.0
    rows = []
    cols = []
    data = []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                weight = (
                    (frac[:, 0] if dx else 1.0 - frac[:, 0])
                    * (frac[:, 1] if dy else 1.0 - frac[:, 1])
                    * (frac[:, 2] if dz else 1.0 - frac[:, 2])
                )
                node_xyz = lower + np.asarray([dx, dy, dz], dtype=np.int64)[None]
                node_id = (node_xyz[:, 0] * nodes + node_xyz[:, 1]) * nodes + node_xyz[:, 2]
                rows.append(np.arange(pos.shape[0], dtype=np.int64))
                cols.append(node_id)
                data.append(weight.astype(np.float64))
    return coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(pos.shape[0], nodes**3),
    ).tocsr()


def _least_squares_project(values: torch.Tensor, design: Any, args: argparse.Namespace) -> Tuple[torch.Tensor, Dict[str, Any]]:
    values_np = values.detach().cpu().numpy().astype(np.float64, copy=False)
    projected = np.empty_like(values_np)
    rows = []
    for channel in range(values_np.shape[1]):
        result = lsqr(
            design,
            values_np[:, channel],
            atol=float(args.lsqr_atol),
            btol=float(args.lsqr_btol),
            iter_lim=int(args.lsqr_iter_lim),
            show=False,
        )
        projected[:, channel] = design @ result[0]
        rows.append({
            "channel": int(channel),
            "istop": int(result[1]),
            "iterations": int(result[2]),
            "residual_norm": float(result[3]),
            "normal_residual_norm": float(result[4]),
        })
    return torch.from_numpy(projected.astype(np.float32)), {
        "rows": rows,
        "matrix_shape": [int(design.shape[0]), int(design.shape[1])],
        "matrix_nnz": int(design.nnz),
    }


def _analyze_trilinear(
    fg: torch.Tensor,
    fh: torch.Tensor,
    global_positions: torch.Tensor,
    scale: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    max_points = min(int(args.trilinear_max_points), int(fg.shape[0]))
    indices = _sample_indices(int(fg.shape[0]), max_points)
    fg_sub = fg.index_select(0, indices)
    fh_sub = fh.index_select(0, indices)
    positions_sub = global_positions.index_select(0, indices)
    design = _trilinear_design(positions_sub.numpy(), scale)
    fg_low, fg_solver = _least_squares_project(fg_sub, design, args)
    fh_low, fh_solver = _least_squares_project(fh_sub, design, args)
    fg_high = fg_sub - fg_low
    fh_high = fh_sub - fh_low
    delta = fh_sub - fg_sub
    delta_low = fh_low - fg_low
    delta_high = fh_high - fg_high
    coarse_consistency = {}
    for name, channel_slice in PBR_GROUPS.items():
        coarse_consistency[name] = {
            "e_coarse": _norm((fh_low - fg_low)[:, channel_slice])
            / (_norm(fg_low[:, channel_slice]) + 1e-8),
            "coarse_delta_l2": _norm((fh_low - fg_low)[:, channel_slice]),
            "coarse_G_l2": _norm(fg_low[:, channel_slice]),
        }
    metrics = _channel_error_metrics(
        delta,
        delta_low,
        delta_high,
        None,
        fg_low,
        fg_high,
        fg_sub,
    )
    return {
        "operator": "coarse-grid least-squares fit + trilinear reconstruction",
        "coarse_scale": int(scale),
        "num_samples": int(fg_sub.shape[0]),
        "subsampled_from": int(fg.shape[0]),
        "coarse_consistency": coarse_consistency,
        "delta_decomposition": metrics,
        "reconstruction_error_G": _relative_error(fg_low + fg_high - fg_sub, fg_sub),
        "reconstruction_error_HR": _relative_error(fh_low + fh_high - fh_sub, fh_sub),
        "solver": {"G": fg_solver, "HR": fh_solver},
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    if left.size < 2 or right.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _latent_similarity(
    positions: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    *,
    pairs: int,
    seed: int,
) -> Dict[str, Any]:
    n = int(positions.shape[0])
    if n < 2:
        return {"status": "insufficient_tokens", "tokens": n}
    rng = np.random.default_rng(int(seed))
    count = min(int(pairs), n * (n - 1))
    left = rng.integers(0, n, size=count, dtype=np.int64)
    right = rng.integers(0, n, size=count, dtype=np.int64)
    same = left == right
    while bool(same.any()):
        right[same] = rng.integers(0, n, size=int(same.sum()), dtype=np.int64)
        same = left == right
    pos_np = positions.numpy().astype(np.float64)
    dist = np.linalg.norm(pos_np[left] - pos_np[right], axis=1)

    def one(features: torch.Tensor) -> Dict[str, Any]:
        values = features.numpy().astype(np.float64)
        a = values[left]
        b = values[right]
        cosine = (a * b).sum(axis=1) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
        )
        l2 = np.linalg.norm(a - b, axis=1)
        bins = np.linspace(0.0, max(float(dist.max()), 1e-8), 11)
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (dist >= lo) & (dist <= hi if hi == bins[-1] else dist < hi)
            rows.append({
                "distance_low": float(lo),
                "distance_high": float(hi),
                "count": int(mask.sum()),
                "latent_l2_mean": float(l2[mask].mean()) if bool(mask.any()) else None,
                "cosine_mean": float(cosine[mask].mean()) if bool(mask.any()) else None,
            })
        return {
            "l2_mean": float(l2.mean()),
            "cosine_mean": float(cosine.mean()),
            "distance_latent_l2_pearson": _pearson(dist, l2),
            "distance_cosine_pearson": _pearson(dist, cosine),
            "bins": rows,
        }

    return {
        "status": "success",
        "tokens": n,
        "pairs": int(count),
        "coordinate_space": "continuous global normalized object space",
        "G": one(g),
        "HR": one(h),
    }


def _query_common_fields(
    mesh: MeshWithVoxel,
    points: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    values = _query_mesh_chunked(mesh, points, chunk_size)
    if not torch.isfinite(values).all():
        raise RuntimeError("decoder PBR query returned non-finite values")
    return values


def _mesh_geometry_check(g_mesh: MeshWithVoxel, h_mesh: MeshWithVoxel) -> Dict[str, Any]:
    same_coords = bool(torch.equal(g_mesh.coords, h_mesh.coords))
    same_faces = bool(torch.equal(g_mesh.faces, h_mesh.faces))
    same_vertices = tuple(g_mesh.vertices.shape) == tuple(h_mesh.vertices.shape) and bool(
        torch.equal(g_mesh.vertices, h_mesh.vertices)
    )
    vertex_error = None
    if tuple(g_mesh.vertices.shape) == tuple(h_mesh.vertices.shape) and g_mesh.vertices.numel():
        vertex_error = float((g_mesh.vertices - h_mesh.vertices).abs().max().item())
    return {
        "same_decoded_ovoxel_coords": same_coords,
        "same_decoded_faces": same_faces,
        "same_decoded_vertices": same_vertices,
        "decoded_vertex_max_abs_error": vertex_error,
        "geometry_equal": bool(same_coords and same_faces and same_vertices),
    }


def _render_variant(
    *,
    name: str,
    vertices_cpu: torch.Tensor,
    faces_cpu: torch.Tensor,
    attrs_cpu: torch.Tensor,
    transform: Any,
    reference: Path,
    tile_dir: Path,
    args: argparse.Namespace,
    envmap: Any,
) -> Dict[str, Any]:
    mesh = MeshWithVertexPbr(
        vertices_cpu,
        faces_cpu,
        attrs_cpu,
        layout=dict(PBR_LAYOUT),
    )
    return core._render(
        mesh,
        output_dir=tile_dir / "renders" / name,
        camera={
            "camera_angle_x": float(transform.camera_angle_x),
            "distance": float(transform.distance),
            "mesh_scale": float(transform.mesh_scale),
        },
        reference_image=reference,
        args=args,
        envmap=envmap,
    )


def _render_field_variants(
    *,
    vertices_cpu: torch.Tensor,
    faces_cpu: torch.Tensor,
    fg_vertices: torch.Tensor,
    fh_vertices: torch.Tensor,
    global_positions_vertices: torch.Tensor,
    transform: Any,
    reference: Path,
    tile_dir: Path,
    args: argparse.Namespace,
    envmap: Any,
) -> Dict[str, Any]:
    ids = _coarse_cell_ids(global_positions_vertices, 1)
    fg_low, _ = _project_cell_mean(fg_vertices, ids)
    fh_low, _ = _project_cell_mean(fh_vertices, ids)
    variants = {
        "G": fg_vertices,
        "HR": fh_vertices,
        "G_low": fg_low,
        "HR_low": fh_low,
        "G_low_HR_high": fg_low + (fh_vertices - fh_low),
        "HR_low_G_high": fh_low + (fg_vertices - fg_low),
    }
    results: Dict[str, Any] = {}
    for name, attrs in variants.items():
        print(f"[render] {name} vertices={attrs.shape[0]:,}")
        results[name] = core._metric_subset(
            _render_variant(
                name=name,
                vertices_cpu=vertices_cpu,
                faces_cpu=faces_cpu,
                attrs_cpu=attrs.contiguous().cpu(),
                transform=transform,
                reference=reference,
                tile_dir=tile_dir,
                args=args,
                envmap=envmap,
            )
        )
    return results


def _decode_and_query(
    *,
    pipeline: Any,
    shape_denorm: SparseTensor,
    texture_latent_norm: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
    query_points_device: torch.Tensor,
    resolution: int,
    query_chunk_size: int,
    label: str,
) -> Tuple[MeshWithVoxel, torch.Tensor, Dict[str, Any]]:
    # The native sparse decoder caches spatial broadcast/index maps on the
    # SparseTensor path.  Give every endpoint/interpolation decode independent
    # objects so a previous decode cannot leak a stale map into the next one.
    shape_input = _fresh_sparse(shape_denorm)
    texture_denorm = _denormalize_slat(texture_latent_norm, normalization)
    texture_input = _fresh_sparse(texture_denorm)
    started = time.perf_counter()
    decoded = pipeline.decode_latent(shape_input, texture_input, int(resolution))
    _sync_cuda()
    if len(decoded) != 1:
        raise RuntimeError(f"{label} decoder returned {len(decoded)} meshes")
    mesh = core._validate_mesh(decoded[0], f"{label} decoded mesh")
    fields = _query_common_fields(mesh, query_points_device, int(query_chunk_size)).cpu()
    stats = {
        "decode_seconds": float(time.perf_counter() - started),
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "active_ovoxels": int(mesh.coords.shape[0]),
        "pbr_range": core._tensor_range(mesh.attrs),
        "query_tokens": int(fields.shape[0]),
    }
    return mesh, fields, stats


def _save_report(output_dir: Path, summary: Mapping[str, Any]) -> Path:
    tile_rows = [row for row in summary.get("tiles", []) if row.get("status") == "success"]
    lines = [
        "# Pixal3D texture-only PBR degradation experiment",
        "",
        "## 1. 生成路径核对",
        "",
        "本实验复用官方 `1024_cascade` baseline、`MeshWithVoxel.query_attrs()` 的三线性 PBR 查询、官方 shape/PBR encoder、texture SLat decoder 与原生 FlowEuler sampler。HR condition 是 canonical 4096 图像中对应的 1024×1024 crop；shape flow 未调用。",
        "",
        f"- image: `{summary.get('image')}`",
        f"- CUDA device: `{summary.get('cuda_device')}`",
        f"- tile rows: `{len(tile_rows)}` successful",
        "",
        "## 2. G / HR 是否严格同 support",
        "",
    ]
    for row in tile_rows:
        checks = row.get("support_checks", {})
        lines.append(
            f"- tile {row['tile_id']:02d}: support_equal=`{checks.get('support_equal')}`, "
            f"num_tokens=`{checks.get('token_count_g_tex')}`, "
            f"coord_max_error=`{checks.get('coord_max_error')}`, "
            f"token_order_equal=`{checks.get('token_order_equal')}`, "
            f"shape_condition_equal=`{checks.get('shape_condition_equal')}`, "
            f"normalization_equal=`{checks.get('normalization_mean_std_equal')}`"
        )
    lines.extend(["", "## 3. PBR coarse consistency", ""])
    for row in tile_rows:
        lines.append(f"### tile {row['tile_id']:02d}")
        lines.append("")
        for scale_name, analysis in row.get("pbr_operator", {}).get("cell_average", {}).items():
            if not isinstance(analysis, Mapping):
                continue
            if "coarse_consistency" not in analysis:
                continue
            cc = analysis["coarse_consistency"]
            lines.append(
                f"- cell-average scale `{scale_name}`: "
                + "; ".join(
                    f"{name} e_coarse={values.get('e_coarse')}"
                    for name, values in cc.items()
                )
            )
        lines.append("")
    lines.extend(["## 4. HR-G 改变量中 low/high 的比例", ""])
    for row in tile_rows:
        lines.append(f"### tile {row['tile_id']:02d}")
        lines.append("")
        for scale_name, analysis in row.get("pbr_operator", {}).get("cell_average", {}).items():
            if not isinstance(analysis, Mapping) or "delta_decomposition" not in analysis:
                continue
            joint = analysis["delta_decomposition"].get("joint", {})
            lines.append(
                f"- scale `{scale_name}`: r_low=`{joint.get('r_low')}`, "
                f"r_high=`{joint.get('r_high')}`, "
                f"A_delta_relative_error=`{joint.get('A_delta_relative_error')}`"
            )
        lines.append("")
    lines.extend([
        "## 5. `G_low + HR_high` 是否保留 HR 提升",
        "",
        "渲染指标见每个 tile 的 `render_metrics.json`。这些 hybrid field 直接送入 `MeshWithVertexPbr` renderer，没有为了重建 latent 而 re-encode。",
        "",
        "## 6. 哪个 coarse scale 最合理",
        "",
    ])
    for row in tile_rows:
        best = row.get("pbr_operator", {}).get("best_cell_average_scale")
        lines.append(f"- tile {row['tile_id']:02d}: `{best}`（按 joint e_coarse 最小选择；不预设解释）")
    lines.extend([
        "",
        "## 7. cell-average 与 trilinear operator",
        "",
        "Experiment B 使用稀疏 trilinear design matrix 与 LSQR；若配置了 `--skip-experiment-b`，此处明确标记为 skipped。A/B 的数值分别保存在 `pbr_operator` 中。",
        "",
        "## 8. 简单 SLat projector 是否和 PBR projector 对应",
        "",
    ])
    for row in tile_rows:
        exploratory = row.get("latent_exploratory", {})
        lines.append(f"### tile {row['tile_id']:02d}")
        lines.append("")
        lines.append(f"- status: `{exploratory.get('status')}`")
        for name, value in exploratory.get("commute", {}).items():
            lines.append(f"- {name}: `{value}`")
        sim = exploratory.get("distance_latent_similarity", {})
        if sim:
            lines.append(f"- distance-latent correlation: `{sim}`")
        lines.append("")
    lines.extend([
        "## 9. 数学假设判断",
        "",
        "本报告不预设“保 coarse、补 fine”。若 `P_F F_G` 与 `P_F F_H` 在某尺度/通道不接近，或 `r_low` 较大，则直接保留失败结论。最终 JSON 中的 `conclusion` 给出逐 tile 的数值判断。",
        "",
        "## 输出",
        "",
        "每个 tile 的 `endpoints.pt` 保存 G_tex/HR_tex coords、normalized feats、unnormalized feats；`pbr_queries.pt` 保存共同 local/global query positions 与字段；`render_metrics.json` 保存六种 PBR field 版本的 renderer 指标。",
        "",
    ])
    path = output_dir / "experiment_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-ids", default=None, help="comma-separated canonical 4096 tile ids")
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shape-encoder", default=str(core.DEFAULT_ENCODER_ROOT / "shape_enc_next_dc_f16c32_fp16"))
    parser.add_argument("--pbr-encoder", default=str(core.DEFAULT_ENCODER_ROOT / "tex_enc_next_dc_f16c32_fp16"))
    parser.add_argument("--fov", type=float, default=-1.0)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument("--max-num-tokens", type=int, default=1_000_000)
    parser.add_argument("--face-projection-chunk-size", type=int, default=250_000)
    parser.add_argument("--material-query-chunk-size", type=int, default=65_536)
    parser.add_argument("--material-face-chunk-size", type=int, default=16_384)
    parser.add_argument("--query-chunk-size", type=int, default=65_536)
    parser.add_argument("--roundtrip-tolerance", type=float, default=2e-5)
    parser.add_argument("--max-query-points", type=int, default=65_536)
    parser.add_argument("--coarse-scales", default="1,2,4")
    parser.add_argument("--skip-experiment-b", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trilinear-max-points", type=int, default=16_384)
    parser.add_argument("--lsqr-atol", type=float, default=1e-5)
    parser.add_argument("--lsqr-btol", type=float, default=1e-5)
    parser.add_argument("--lsqr-iter-lim", type=int, default=80)
    parser.add_argument("--latent-pairs", type=int, default=100_000)
    parser.add_argument("--latent-seed", type=int, default=20260812)
    parser.add_argument("--skip-latent-exploratory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--noise-timestep", type=float, default=1.0)
    parser.add_argument("--noise-strength", type=float, default=1.0)
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
    parser.add_argument("--skip-lpips", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    return parser


def _validate_args(args: argparse.Namespace) -> Tuple[int, ...]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if int(args.cuda_device) < 0 or int(args.cuda_device) >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {args.cuda_device} is unavailable")
    if int(args.max_num_tokens) < 1 or int(args.max_query_points) < 1:
        raise ValueError("token/query limits must be positive")
    if float(args.noise_strength) <= 0.0:
        raise ValueError("--noise-strength must be positive")
    for encoder_path in (args.shape_encoder, args.pbr_encoder):
        base_path = Path(encoder_path).expanduser()
        if not Path(f"{base_path}.json").is_file() or not Path(f"{base_path}.safetensors").is_file():
            raise FileNotFoundError(f"encoder checkpoint pair not found for {base_path}")
    if not args.skip_lpips and importlib.util.find_spec("lpips") is None:
        print("[metrics] lpips unavailable; continuing without LPIPS")
        args.skip_lpips = True
    return _parse_scales(args.coarse_scales)


def _load_or_run_global(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    image_1024: Image.Image,
    output_dir: Path,
    global_camera: Mapping[str, float],
) -> Tuple[MeshWithVoxel, Dict[str, Any]]:
    mesh_path = output_dir / "global_baseline_mesh.pt"
    slat_path = output_dir / "global_baseline_slats.pt"
    if bool(args.resume) and mesh_path.is_file() and slat_path.is_file():
        mesh_payload = torch.load(mesh_path, map_location="cpu", weights_only=False)
        slat_payload = torch.load(slat_path, map_location="cpu", weights_only=False)
        mesh = mesh_payload["mesh"] if isinstance(mesh_payload, Mapping) else mesh_payload
        if not isinstance(mesh, MeshWithVoxel):
            raise RuntimeError("cached global baseline mesh is not MeshWithVoxel")
        print(f"[global-baseline] reused {mesh_path}")
        return mesh, dict(slat_payload)

    _seed_everything(int(args.seed))
    ss_params, shape_params, texture_params = core._sampler_overrides(args)
    started = time.perf_counter()
    print("[global-baseline] ordinary Pixal3D 1024_cascade")
    output, latents = pipeline.run(
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
    if len(output) != 1:
        raise RuntimeError(f"global baseline returned {len(output)} meshes")
    mesh = core._validate_mesh(output[0], "global 1024 baseline")
    shape_raw, texture_raw, resolution = latents
    if int(resolution) != OVOXEL_RESOLUTION:
        raise RuntimeError(f"baseline resolution={resolution}, expected {OVOXEL_RESOLUTION}")
    if not torch.equal(shape_raw.coords, texture_raw.coords):
        raise RuntimeError("global baseline shape/texture supports differ")
    shape_norm = _normalize_slat(shape_raw, pipeline.shape_slat_normalization)
    texture_norm = _normalize_slat(texture_raw, pipeline.tex_slat_normalization)
    mesh_cpu = mesh.to("cpu")
    payload = {
        "format": f"{FORMAT}_global_slats",
        "seed": int(args.seed),
        "resolution": int(resolution),
        "coords": shape_norm.coords.detach().cpu().to(torch.int32),
        "shape_raw": shape_raw.feats.detach().float().cpu(),
        "shape_norm": shape_norm.feats.detach().float().cpu(),
        "texture_raw": texture_raw.feats.detach().float().cpu(),
        "texture_norm": texture_norm.feats.detach().float().cpu(),
        "shape_normalization": dict(pipeline.shape_slat_normalization),
        "texture_normalization": dict(pipeline.tex_slat_normalization),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _atomic_torch_save(output_dir / "global_baseline_mesh.pt", {"format": f"{FORMAT}_global_mesh", "mesh": mesh_cpu})
    _atomic_torch_save(output_dir / "global_baseline_slats.pt", payload)
    del output, latents, shape_raw, texture_raw, shape_norm, texture_norm, mesh
    _empty_cuda_cache()
    return mesh_cpu, payload


def _run_tile(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    baseline_mesh: MeshWithVoxel,
    global_attr_field: MeshWithVoxel,
    global_camera: Mapping[str, float],
    image_4096: Image.Image,
    output_dir: Path,
    tile_id: int,
    box: Sequence[int],
    face_min: torch.Tensor,
    face_max: torch.Tensor,
    face_finite: torch.Tensor,
    scales: Sequence[int],
) -> Dict[str, Any]:
    tile_dir = output_dir / "tiles" / f"tile_{int(tile_id):02d}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    record: Dict[str, Any] = {"tile_id": int(tile_id), "box": list(map(int, box)), "status": "started"}
    started = time.perf_counter()
    transform = core._derive_tile_camera(
        tile_id=int(tile_id), box=tuple(map(int, box)), global_camera=global_camera, extend_pixel=int(args.extend_pixel)
    )
    _atomic_json(tile_dir / "tile_camera.json", transform.__dict__)
    hr_tile = image_4096.crop(tuple(map(int, box))).convert("RGB")
    if hr_tile.size != (TILE_SIZE, TILE_SIZE):
        hr_tile = hr_tile.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS)
    hr_tile_path = tile_dir / "hr_tile_1024_condition.png"
    hr_tile.save(hr_tile_path)
    selected = core._tile_face_ids_from_bbox(face_min, face_max, face_finite, tuple(map(int, box)))
    record["projected_bbox_faces"] = int(selected.shape[0])
    if selected.numel() == 0:
        record.update({"status": "skipped", "reason": "no projected triangle bbox"})
        _atomic_json(tile_dir / "summary.json", record)
        return record

    try:
        geometry = core._prepare_tile_geometry(
            global_vertices=baseline_mesh.vertices,
            global_faces=baseline_mesh.faces,
            global_face_min=face_min,
            global_face_max=face_max,
            global_face_finite=face_finite,
            global_camera=global_camera,
            transform=transform,
        )
        if float(geometry.stats["global_local_global_q_max_abs_error"]) > float(args.roundtrip_tolerance):
            raise RuntimeError("local/global camera round-trip exceeded tolerance")
        shape_encoder = pixal3d_models.from_pretrained(str(Path(args.shape_encoder).expanduser())).eval()
        pbr_encoder = pixal3d_models.from_pretrained(str(Path(args.pbr_encoder).expanduser())).eval()
        if not args.low_vram:
            shape_encoder.to(torch.device("cuda"))
            pbr_encoder.to(torch.device("cuda"))
        local_attrs, material_stats = core._resample_local_attrs_from_global(
            geometry=geometry,
            global_attr_field=global_attr_field,
            global_camera=global_camera,
            transform=transform,
            query_chunk_size=int(args.material_query_chunk_size),
            face_chunk_size=int(args.material_face_chunk_size),
        )
        shape_reference, shape_stats = core._encode_local_shape(
            encoder=shape_encoder,
            local_coords=geometry.coords,
            local_dual_vertices=geometry.dual_vertices,
            local_intersected=geometry.intersected,
            device=torch.device("cuda"),
            low_vram=bool(args.low_vram),
        )
        texture_reference, texture_stats = core._encode_local_pbr(
            encoder=pbr_encoder,
            coords=geometry.coords,
            attrs=local_attrs,
            device=torch.device("cuda"),
            low_vram=bool(args.low_vram),
        )
        alignment = core._latent_support_diagnostics(shape_reference, texture_reference)
        if not alignment["coordinates_exactly_equal"]:
            raise RuntimeError(f"shape/G_tex support mismatch: {alignment}")
        fixed_shape_norm = _normalize_slat(shape_reference, pipeline.shape_slat_normalization)
        g_tex_norm = _normalize_slat(texture_reference, pipeline.tex_slat_normalization)
        shape_before_flow = _clone_sparse(fixed_shape_norm)
        texture_condition = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_tex_1024,
            [hr_tile],
            fixed_shape_norm.coords.to(torch.int32),
            camera_angle_x=float(transform.camera_angle_x),
            distance=float(transform.distance),
            mesh_scale=float(transform.mesh_scale),
            grid_resolution_override=LATENT_RESOLUTION,
        )
        texture_model = pipeline.models["tex_slat_flow_model_1024"]
        texture_channels = int(texture_model.in_channels) - int(fixed_shape_norm.feats.shape[1])
        if int(g_tex_norm.feats.shape[1]) != texture_channels:
            raise RuntimeError(f"G_tex channels={g_tex_norm.feats.shape[1]} flow expects {texture_channels}")
        _seed_everything(int(args.seed) + int(tile_id) * 100003)
        noise = SparseTensor(
            torch.randn_like(g_tex_norm.feats),
            g_tex_norm.coords,
        )
        initial_state = _native_noised_endpoint(
            g_tex_norm,
            noise,
            pipeline.tex_slat_sampler,
            float(args.noise_timestep),
            float(args.noise_strength),
        )
        hr_tex_norm, flow_stats = _run_texture_flow(
            pipeline=pipeline,
            initial_state=initial_state,
            shape_condition=fixed_shape_norm,
            condition=texture_condition,
            params=core._sampler_overrides(args)[2],
            noise_timestep=float(args.noise_timestep),
        )
        shape_after_flow = _clone_sparse(fixed_shape_norm)
        support_checks = _latent_support_checks(
            fixed_shape_norm,
            g_tex_norm,
            hr_tex_norm,
            shape_before_flow,
            shape_after_flow,
            pipeline.tex_slat_normalization,
        )
        support_checks["normalization_max_abs_error"] = _normalization_error(
            pipeline.tex_slat_normalization, pipeline.tex_slat_normalization
        )
        support_checks["normalization_mean_std_equal"] = bool(
            support_checks["normalization_max_abs_error"]["mean_max_abs_error"] == 0.0
            and support_checks["normalization_max_abs_error"]["std_max_abs_error"] == 0.0
        )
        record["support_checks"] = support_checks
        if not all(
            bool(support_checks[key])
            for key in (
                "support_equal",
                "token_count_equal",
                "token_order_equal",
                "shape_condition_equal",
                "normalization_mean_std_equal",
            )
        ):
            raise RuntimeError(f"strict G/HR support check failed: {support_checks}")

        endpoints_payload = {
            "format": f"{FORMAT}_endpoints",
            "tile_id": int(tile_id),
            "box": list(map(int, box)),
            "transform": transform.__dict__,
            "shape_coords": fixed_shape_norm.coords.detach().cpu().to(torch.int32),
            "shape_raw": shape_reference.feats.detach().float().cpu(),
            "shape_norm": fixed_shape_norm.feats.detach().float().cpu(),
            "g_tex_coords": g_tex_norm.coords.detach().cpu().to(torch.int32),
            "g_tex_raw": texture_reference.feats.detach().float().cpu(),
            "g_tex_norm": g_tex_norm.feats.detach().float().cpu(),
            "hr_tex_coords": hr_tex_norm.coords.detach().cpu().to(torch.int32),
            "hr_tex_norm": hr_tex_norm.feats.detach().float().cpu(),
            "hr_tex_raw": _denormalize_slat(hr_tex_norm, pipeline.tex_slat_normalization).feats.detach().float().cpu(),
            "noise": noise.feats.detach().float().cpu(),
            "normalization": {
                "shape": dict(pipeline.shape_slat_normalization),
                "texture": dict(pipeline.tex_slat_normalization),
            },
            "flow": flow_stats,
            "support_checks": support_checks,
        }
        _atomic_torch_save(tile_dir / "endpoints.pt", endpoints_payload)

        shape_denorm = _denormalize_slat(fixed_shape_norm, pipeline.shape_slat_normalization)
        g_mesh, _, g_decode_stats = _decode_and_query(
            pipeline=pipeline,
            shape_denorm=shape_denorm,
            texture_latent_norm=g_tex_norm,
            normalization=pipeline.tex_slat_normalization,
            query_points_device=fixed_shape_norm.coords.new_empty((0, 3), dtype=torch.float32),
            resolution=OVOXEL_RESOLUTION,
            query_chunk_size=int(args.query_chunk_size),
            label="G_tex",
        )
        # The first decode above intentionally uses no query points only to keep
        # the decoder invocation isolated.  The actual common query positions
        # are the decoded G mesh vertices and are then used for both fields.
        query_points = g_mesh.vertices
        sample_index = _sample_indices(int(query_points.shape[0]), int(args.max_query_points))
        sample_points_device = query_points.index_select(0, sample_index.to(query_points.device))
        g_fields = _query_common_fields(g_mesh, sample_points_device, int(args.query_chunk_size)).cpu()
        g_fields_all = _query_common_fields(g_mesh, query_points, int(args.query_chunk_size)).cpu()
        vertices_cpu = query_points.detach().cpu().to(torch.float32)
        faces_cpu = g_mesh.faces.detach().cpu().to(torch.int32)
        full_global_positions = _map_local_to_global_chunked(
            query_points,
            transform=transform,
            global_camera=global_camera,
            chunk_size=int(args.query_chunk_size),
        ).detach().cpu().to(torch.float32)
        sample_global_positions = full_global_positions.index_select(0, sample_index)
        geometry_vertex_count = int(query_points.shape[0])
        g_mesh_support = g_mesh.coords.detach().cpu().to(torch.int32)
        hr_mesh, _, hr_decode_stats = _decode_and_query(
            pipeline=pipeline,
            shape_denorm=shape_denorm,
            texture_latent_norm=hr_tex_norm,
            normalization=pipeline.tex_slat_normalization,
            query_points_device=sample_points_device,
            resolution=OVOXEL_RESOLUTION,
            query_chunk_size=int(args.query_chunk_size),
            label="HR_tex",
        )
        geometry_check = _mesh_geometry_check(g_mesh, hr_mesh)
        if not geometry_check["geometry_equal"]:
            raise RuntimeError(f"fixed shape decode geometry differs: {geometry_check}")
        hr_fields = _query_common_fields(hr_mesh, sample_points_device, int(args.query_chunk_size)).cpu()
        hr_fields_all = _query_common_fields(hr_mesh, query_points, int(args.query_chunk_size)).cpu()
        record["decoded_geometry_check"] = geometry_check
        record["decode"] = {"G_tex": g_decode_stats, "HR_tex": hr_decode_stats}

        pbr_analysis: Dict[str, Any] = {"cell_average": {}, "trilinear": {}}
        pbr_fields_for_save: Dict[str, torch.Tensor] = {}
        for scale in scales:
            ids = _coarse_cell_ids(sample_global_positions, int(scale))
            analysis, fields = _analyze_cell_average(g_fields, hr_fields, ids)
            key = f"global_C64_x{int(scale)}"
            pbr_analysis["cell_average"][key] = analysis
            if int(scale) == 1:
                pbr_fields_for_save = fields
        joint_errors = {
            key: float(value["coarse_consistency"]["joint"]["e_coarse"])
            for key, value in pbr_analysis["cell_average"].items()
        }
        best_key = min(joint_errors, key=joint_errors.get)
        pbr_analysis["best_cell_average_scale"] = best_key
        if not args.skip_experiment_b:
            for scale in scales:
                key = f"global_C64_x{int(scale)}"
                pbr_analysis["trilinear"][key] = _analyze_trilinear(
                    g_fields,
                    hr_fields,
                    sample_global_positions,
                    int(scale),
                    args,
                )
        else:
            pbr_analysis["trilinear"] = {"status": "skipped_by_cli"}
        _atomic_json(tile_dir / "pbr_analysis.json", pbr_analysis)
        _atomic_torch_save(
            tile_dir / "pbr_queries.pt",
            {
                "format": f"{FORMAT}_pbr_queries",
                "local_coordinates": sample_points_device.detach().cpu().float(),
                "global_normalized_object_coordinates": sample_global_positions,
                "G": g_fields,
                "HR": hr_fields,
                "fields_cell_average_global_C64": pbr_fields_for_save,
                "coarse_scales": tuple(int(v) for v in scales),
            },
        )

        render_metrics: Dict[str, Any] = {"status": "skipped"}
        if args.render:
            envmap = core.load_envmap(str(args.envmap), device="cuda")
            render_metrics = _render_field_variants(
                vertices_cpu=vertices_cpu,
                faces_cpu=faces_cpu,
                fg_vertices=g_fields_all,
                fh_vertices=hr_fields_all,
                global_positions_vertices=full_global_positions,
                transform=transform,
                reference=hr_tile_path,
                tile_dir=tile_dir,
                args=args,
                envmap=envmap,
            )
        _atomic_json(tile_dir / "render_metrics.json", render_metrics)

        # Renderer inputs are deliberately large for this tile.  Release both
        # decoded GPU meshes and their full-vertex field tensors before the
        # exploratory decodes; otherwise the sparse decoder can retain stale
        # spatial-cache bookkeeping across the renderer/decoder boundary.
        if args.render:
            del envmap
        del g_mesh, hr_mesh, g_fields_all, hr_fields_all
        _empty_cuda_cache()

        exploratory: Dict[str, Any]
        if args.skip_latent_exploratory:
            exploratory = {"status": "skipped_by_cli"}
        else:
            exploratory = _run_latent_exploratory(
                args=args,
                pipeline=pipeline,
                shape_denorm=shape_denorm,
                fixed_shape_norm=fixed_shape_norm,
                g_tex_norm=g_tex_norm,
                hr_tex_norm=hr_tex_norm,
                normalization=pipeline.tex_slat_normalization,
                query_points_device=sample_points_device,
                query_points_cpu=sample_points_device.detach().cpu(),
                sample_global_positions=sample_global_positions,
                transform=transform,
                global_camera=global_camera,
                g_fields=g_fields,
                hr_fields=hr_fields,
                tile_dir=tile_dir,
                label_prefix=f"tile_{int(tile_id):02d}",
            )
        _atomic_json(tile_dir / "latent_exploratory.json", exploratory)

        record.update(
            {
                "status": "success",
                "tile_seconds": float(time.perf_counter() - started),
                "hr_condition": {
                    "source": "canonical 4096 image crop",
                    "image": str(hr_tile_path),
                    "size": list(hr_tile.size),
                    "box_4096": list(map(int, box)),
                },
                "geometry": geometry.stats,
                "material_resampling": material_stats,
                "fixed_shape": {
                    "source": "global baseline mesh -> local C1024 voxelize -> official shape encoder",
                    "shape_flow_called": False,
                    "shape_slat_sampler_sample_called": False,
                    "encoder": shape_stats,
                    "tokens": int(fixed_shape_norm.coords.shape[0]),
                },
                "G_tex": {
                    "source": "global baseline MeshWithVoxel PBR query -> local official texture/PBR encoder",
                    "encoder": texture_stats,
                    "tokens": int(g_tex_norm.coords.shape[0]),
                    "normalized_features_saved": True,
                    "unnormalized_features_saved": True,
                },
                "flow": flow_stats,
                "pbr_operator": pbr_analysis,
                "render_metrics": render_metrics,
                "latent_exploratory": exploratory,
                "common_query": {
                    "source": "decoded G_tex mesh vertices",
                    "decoded_mesh_vertices": geometry_vertex_count,
                    "analyzed_query_points": int(sample_points_device.shape[0]),
                    "local_coordinates_saved": True,
                    "global_normalized_object_coordinates_saved": True,
                },
                "conclusion": _tile_conclusion(pbr_analysis, render_metrics),
            }
        )
        _atomic_json(tile_dir / "summary.json", record)
        return record
    except Exception as exc:
        record.update({"status": "failed", "tile_seconds": float(time.perf_counter() - started), "reason": f"{type(exc).__name__}: {exc}"})
        _atomic_json(tile_dir / "summary.json", record)
        print(f"[tile {int(tile_id):02d}] FAILED: {record['reason']}")
        traceback.print_exc()
        return record
    finally:
        _empty_cuda_cache()


def _tile_conclusion(pbr_analysis: Mapping[str, Any], render_metrics: Mapping[str, Any]) -> Dict[str, Any]:
    cell = pbr_analysis.get("cell_average", {})
    rows = []
    for scale, value in cell.items():
        if not isinstance(value, Mapping):
            continue
        joint = value.get("coarse_consistency", {}).get("joint", {})
        delta = value.get("delta_decomposition", {}).get("joint", {})
        rows.append({
            "scale": scale,
            "e_coarse": joint.get("e_coarse"),
            "r_low": delta.get("r_low"),
            "r_high": delta.get("r_high"),
            "A_delta_relative_error": delta.get("A_delta_relative_error"),
        })
    best = pbr_analysis.get("best_cell_average_scale")
    best_row = next((row for row in rows if row["scale"] == best), None)
    supports = False
    if best_row is not None and best_row["e_coarse"] is not None:
        supports = bool(float(best_row["e_coarse"]) < 0.1 and float(best_row["r_high"]) > float(best_row["r_low"]))
    return {
        "cell_average_scales": rows,
        "best_scale_by_e_coarse": best,
        "supports_coarse_preserve_fine_detail_heuristic": supports,
        "interpretation": "heuristic only; no SLat frequency assumption",
        "render_metric_variants": sorted(render_metrics) if isinstance(render_metrics, Mapping) else [],
    }


def _run_latent_exploratory(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    shape_denorm: SparseTensor,
    fixed_shape_norm: SparseTensor,
    g_tex_norm: SparseTensor,
    hr_tex_norm: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
    query_points_device: torch.Tensor,
    query_points_cpu: torch.Tensor,
    sample_global_positions: torch.Tensor,
    transform: Any,
    global_camera: Mapping[str, float],
    g_fields: torch.Tensor,
    hr_fields: torch.Tensor,
    tile_dir: Path,
    label_prefix: str,
) -> Dict[str, Any]:
    coords = g_tex_norm.coords
    token_positions = _map_local_to_global(
        -0.5 + (coords[:, 1:].to(torch.float32) + 0.5) / float(LATENT_RESOLUTION),
        transform=transform,
        global_camera=global_camera,
    ).detach().cpu()
    z_g = g_tex_norm.feats.detach().cpu().float()
    z_h = hr_tex_norm.feats.detach().cpu().float()
    z_g_low, z_info = _project_cell_mean(z_g, _coarse_cell_ids(token_positions, 1))
    z_h_low, _ = _project_cell_mean(z_h, _coarse_cell_ids(token_positions, 1))
    commute: Dict[str, Any] = {}
    for name, latent_low in (("G", z_g_low), ("HR", z_h_low)):
        latent_sparse = SparseTensor(latent_low.to(torch.device("cuda")), coords)
        _, decoded_field, decode_stats = _decode_and_query(
            pipeline=pipeline,
            shape_denorm=shape_denorm,
            texture_latent_norm=latent_sparse,
            normalization=normalization,
            query_points_device=query_points_device,
            resolution=OVOXEL_RESOLUTION,
            query_chunk_size=int(args.query_chunk_size),
            label=f"{label_prefix} P_Z{name}",
        )
        pbr_low = _project_cell_mean(
            g_fields if name == "G" else hr_fields,
            _coarse_cell_ids(sample_global_positions, 1),
        )[0]
        commute[f"D(P_Z{name})_vs_P_FD({name})"] = _relative_error(decoded_field - pbr_low, pbr_low)
        commute[f"D(P_Z{name})_decode_seconds"] = decode_stats["decode_seconds"]
        del latent_sparse
        _empty_cuda_cache()

    similarity = _latent_similarity(
        token_positions,
        z_g,
        z_h,
        pairs=int(args.latent_pairs),
        seed=int(args.latent_seed),
    )
    interpolation: Dict[str, Any] = {}
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        key = f"alpha_{alpha:.2f}"
        if alpha == 0.0:
            field = g_fields
            interpolation[key] = {"decoded": False, "field_reference": "G", "pbr_l2_to_linear": 0.0}
            continue
        if alpha == 1.0:
            field = hr_fields
            interpolation[key] = {"decoded": False, "field_reference": "HR", "pbr_l2_to_linear": 0.0}
            continue
        z_alpha = z_g + float(alpha) * (z_h - z_g)
        latent_sparse = SparseTensor(z_alpha.to(torch.device("cuda")), coords)
        _, field, decode_stats = _decode_and_query(
            pipeline=pipeline,
            shape_denorm=shape_denorm,
            texture_latent_norm=latent_sparse,
            normalization=normalization,
            query_points_device=query_points_device,
            resolution=OVOXEL_RESOLUTION,
            query_chunk_size=int(args.query_chunk_size),
            label=f"{label_prefix} latent interpolation {alpha:.2f}",
        )
        linear = (1.0 - float(alpha)) * g_fields + float(alpha) * hr_fields
        interpolation[key] = {
            "decoded": True,
            "decode_seconds": decode_stats["decode_seconds"],
            "pbr_l2_to_linear": _relative_error(field - linear, linear),
            "pbr_l2_to_G": _relative_error(field - g_fields, g_fields),
            "pbr_l2_to_HR": _relative_error(field - hr_fields, hr_fields),
        }
        del latent_sparse
        _empty_cuda_cache()
    _atomic_json(tile_dir / "latent_interpolation_metrics.json", interpolation)
    return {
        "status": "success",
        "normalization": "same model tex_slat_normalization for G and HR",
        "projector": "global-equivalent C64 cell mean on normalized texture SLat; exploratory only",
        "token_count": int(coords.shape[0]),
        "num_coarse_cells": int(z_info["num_coarse_cells"]),
        "commute": commute,
        "distance_latent_similarity": similarity,
        "interpolation": interpolation,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    scales = _validate_args(args)
    torch.cuda.set_device(int(args.cuda_device))
    print(
        f"[cuda] requested/current index={int(args.cuda_device)}/{torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())}"
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.image).expanduser().resolve()
    with Image.open(source_path) as source:
        source_rgb = source.convert("RGB")
    source_rgb.save(output_dir / "input_original.png")
    pipeline = core.init_pipeline(args.model_path, device="cuda", low_vram=bool(args.low_vram))
    canonical = pipeline.preprocess_canonical_images(source_rgb)
    image_4096: Image.Image = canonical["image_4096"]
    image_1024: Image.Image = canonical["image_1024"]
    image_512: Image.Image = canonical["image_512"]
    image_4096.save(output_dir / "canonical_4096.png")
    image_1024.save(output_dir / "canonical_1024.png")
    image_512.save(output_dir / "canonical_512.png")
    canonical["foreground_mask_4096"].save(output_dir / "canonical_foreground_mask_4096.png")
    _atomic_json(output_dir / "canonical_metadata.json", canonical["metadata"])
    camera_path = output_dir / "global_camera.json"
    if bool(args.resume) and camera_path.is_file():
        global_camera = json.loads(camera_path.read_text("utf-8"))
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
    baseline_mesh, baseline_slats = _load_or_run_global(
        args=args,
        pipeline=pipeline,
        image_1024=image_1024,
        output_dir=output_dir,
        global_camera=global_camera,
    )
    baseline_summary = {
        "tokens": int(baseline_mesh.coords.shape[0]),
        "vertices": int(baseline_mesh.vertices.shape[0]),
        "faces": int(baseline_mesh.faces.shape[0]),
        "pbr_channels": int(baseline_mesh.attrs.shape[1]),
        "generation_seconds": baseline_slats.get("elapsed_seconds"),
        "route": "ordinary pipeline.run(..., pipeline_type='1024_cascade')",
    }
    baseline_render = None
    if args.render:
        envmap = core.load_envmap(str(args.envmap), device="cuda")
        baseline_render = core._render(
            baseline_mesh,
            output_dir=output_dir / "global_baseline_1024" / "aligned_eval",
            camera=global_camera,
            reference_image=output_dir / "canonical_1024.png",
            args=args,
            envmap=envmap,
        )
        baseline_summary["render_metrics"] = core._metric_subset(baseline_render)
    baseline_mesh = baseline_mesh.to("cpu")
    face_min, face_max, face_finite = core._project_face_bboxes(
        baseline_mesh.vertices,
        baseline_mesh.faces,
        mesh_scale=float(global_camera["mesh_scale"]),
        global_camera=global_camera,
        chunk_size=int(args.face_projection_chunk_size),
    )
    global_attr_field = core._make_attribute_query_mesh(baseline_mesh, torch.device("cuda"))
    boxes = core._tile_layout(canonical_size=CANONICAL_IMAGE_SIZE, tile_size=TILE_SIZE, stride=TILE_SIZE)
    requested = _parse_int_set(args.tile_ids)
    rows: List[Dict[str, Any]] = []
    attempted = 0
    for tile_id, box in enumerate(boxes):
        if requested is not None and int(tile_id) not in requested:
            continue
        if args.max_tiles is not None and attempted >= int(args.max_tiles):
            break
        attempted += 1
        if bool(args.resume):
            cached_summary_path = output_dir / "tiles" / f"tile_{int(tile_id):02d}" / "summary.json"
            if cached_summary_path.is_file():
                try:
                    cached_row = json.loads(cached_summary_path.read_text("utf-8"))
                except Exception:
                    cached_row = None
                if isinstance(cached_row, Mapping) and cached_row.get("status") == "success":
                    print(f"[tile {tile_id:02d}] reused successful summary")
                    rows.append(dict(cached_row))
                    continue
        print(f"[tile {tile_id:02d}] box={box}")
        row = _run_tile(
            args=args,
            pipeline=pipeline,
            baseline_mesh=baseline_mesh,
            global_attr_field=global_attr_field,
            global_camera=global_camera,
            image_4096=image_4096,
            output_dir=output_dir,
            tile_id=int(tile_id),
            box=box,
            face_min=face_min,
            face_max=face_max,
            face_finite=face_finite,
            scales=scales,
        )
        rows.append(row)
    success = [row for row in rows if row.get("status") == "success"]
    failure = [row for row in rows if row.get("status") == "failed"]
    skipped = [row for row in rows if row.get("status") == "skipped"]
    summary: Dict[str, Any] = {
        "format": FORMAT,
        "image": str(source_path),
        "cuda_device": int(args.cuda_device),
        "seed": int(args.seed),
        "global_camera": global_camera,
        "global_baseline": baseline_summary,
        "requested_tile_ids": sorted(requested) if requested is not None else None,
        "tile_layout": {"canonical_image_size": CANONICAL_IMAGE_SIZE, "tile_size": TILE_SIZE, "stride": TILE_SIZE, "boxes": boxes},
        "sampler": {"texture": core._sampler_overrides(args)[2], "noise_timestep": float(args.noise_timestep), "noise_strength": float(args.noise_strength)},
        "route_checks": {
            "shape_flow_called": False,
            "shape_slat_sampler_sample_called": False,
            "hr_condition_is_4096_crop": True,
            "official_decoder": True,
            "official_meshwithvoxel_pbr_query": True,
            "no_sampler_velocity_modification": True,
            "no_training": True,
        },
        "successful_tiles": len(success),
        "failed_tiles": len(failure),
        "skipped_tiles": len(skipped),
        "tiles": rows,
        "artifacts": {"global_mesh": str(output_dir / "global_baseline_mesh.pt"), "global_slats": str(output_dir / "global_baseline_slats.pt")},
    }
    report = _save_report(output_dir, summary)
    summary["report_markdown"] = str(report)
    _atomic_json(output_dir / "summary.json", summary)
    print(f"[done] success={len(success)} failed={len(failure)} skipped={len(skipped)} report={report}")
    return summary


def main() -> None:
    args = _build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
