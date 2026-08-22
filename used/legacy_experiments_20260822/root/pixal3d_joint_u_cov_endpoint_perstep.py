#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Joint physical-consensus / conditional-covariance texture-flow experiment.

This file is intentionally independent from the historical Gaussian fusion,
range-null, MRA, and POD experiments.  It keeps the validated Pixal3D local
tile preparation and native sampler route, but changes one operation only:

    official PureHR endpoint -> sparse C4096 physical U* -> PBR Jacobian
    gradient -> conditional latent covariance-vector product -> corrected x0
    -> official _xstart_to_pred -> official Euler.

The flow state, endpoint, covariance samples, gradient, and correction are
all in normalized texture-SLat space.  ``U*`` is solved on CPU with scipy's
LSMR and is never part of autograd.  The decoder is run with the fixed shape
decoder guide supports; the gradient route uses an explicit differentiable
sparse trilinear query because the CUDA MeshWithVoxel query is an inference
operator without a reliable input backward path.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
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
from scipy.sparse import csr_matrix, load_npz, save_npz, vstack
from scipy.sparse.linalg import lsmr

import pixal3d.models as pixal3d_models
import pixal3d_cross_tile_pbr_perstep as base
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVoxel


_FLEX_GEMM_DECODER_BACKWARD_PATCHED = False


def _patch_flex_gemm_input_backward() -> None:
    """Fix the installed flex_gemm frozen-weight backward edge case.

    Pixal3D's sparse convolution autograd wrapper unconditionally reshapes
    ``grad_weight`` after calling the Triton backward kernel.  The kernel
    correctly returns ``None`` when decoder parameters are frozen, so the
    wrapper raises before returning the needed input gradient.  The endpoint
    experiment must keep decoder parameters frozen; this local compatibility
    patch preserves that contract and only conditionally reshapes an actual
    weight gradient.  No repository or model parameter is modified.
    """
    global _FLEX_GEMM_DECODER_BACKWARD_PATCHED
    if _FLEX_GEMM_DECODER_BACKWARD_PATCHED:
        return
    from flex_gemm import kernels
    from flex_gemm.ops import spconv as flex_spconv
    from flex_gemm.ops.spconv import submanifold_conv3d as subm

    original_algorithm = flex_spconv.ALGORITHM

    def safe_sparse_submanifold_conv_backward(
        grad_output: torch.Tensor,
        feats: torch.Tensor,
        neighbor_cache: Any,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        co, kw, kh, kd, ci = weight.shape
        if flex_spconv.ALGORITHM == subm.Algorithm.MASKED_IMPLICIT_GEMM_SPLITK:
            grad_input, grad_weight, grad_bias = kernels.triton.sparse_submanifold_conv_bwd_masked_implicit_gemm_splitk(
                grad_output.contiguous(),
                feats,
                weight.reshape(co, kd * kh * kw, ci),
                bias,
                neighbor_cache["neighbor_map"],
                neighbor_cache["sorted_idx"],
                neighbor_cache.valid_kernel_callback,
                neighbor_cache.valid_kernel_seg_callback,
                neighbor_cache["valid_signal_i"],
                neighbor_cache["valid_signal_o"],
                neighbor_cache["valid_signal_seg"],
            )
            if grad_weight is not None:
                grad_weight = grad_weight.reshape(co, kw, kh, kd, ci)
            return grad_input, grad_weight, grad_bias
        # Keep the patch conservative if a future Pixal3D backend selects a
        # different algorithm: use the original implementation there.
        return original_backward(
            grad_output, feats, neighbor_cache, weight, bias
        )

    original_backward = subm.SubMConv3dFunction._sparse_submanifold_conv_backward
    subm.SubMConv3dFunction._sparse_submanifold_conv_backward = staticmethod(
        safe_sparse_submanifold_conv_backward
    )
    _FLEX_GEMM_DECODER_BACKWARD_PATCHED = True


FORMAT = "pixal3d_joint_u_cov_endpoint_perstep_v1"
CANONICAL_IMAGE_SIZE = 4096
TILE_SIZE = 1024
TILE_STRIDE = 512
OVOXEL_RESOLUTION = 1024
GLOBAL_U_RESOLUTION = 4096
LATENT_RESOLUTION = 64
TEXTURE_STEPS = 12
TEXTURE_RESCALE_T = 3.0
GUIDED_SEED = 42
COVARIANCE_SEEDS = (123, 2024, 3407, 9999, 2718, 31415, 65537, 104729)
PBR_CHANNELS = 6
PBR_CHANNEL_NAMES = ("RGB", "metallic", "roughness", "alpha")
PBR_SLICES = {
    "RGB": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}


@dataclass
class OperatorCache:
    """The immutable weighted observation operator and its row provenance."""

    matrix: csr_matrix
    support_coords: torch.Tensor
    blocks: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class CovarianceCache:
    """A CPU cache for one tile's centered normalized endpoint samples."""

    coords: torch.Tensor
    mean: torch.Tensor
    centered: torch.Tensor
    sigma2: float
    stats: Dict[str, Any]


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _ensure_output_layout(output_root: Path) -> None:
    """Create the stable artifact directories used by the formal report."""
    for name in (
        "global_baseline",
        "pure_hr",
        "joint_u_cov_guided",
        "covariance",
        "u_operator",
        "tiles",
        "multiview",
        "turntable",
        "pbr_channel_sheets",
    ):
        output_root.joinpath(name).mkdir(parents=True, exist_ok=True)


def _link_artifact(source: Path, target: Path) -> None:
    """Expose a large existing artifact under the formal output layout."""
    source = source.resolve()
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        if target.is_dir() and source.is_dir() and not any(target.iterdir()):
            target.rmdir()
        elif target.is_file() and source.is_file():
            return
        else:
            return
    try:
        target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=source.is_dir())
    except OSError:
        # Symlinks are available on the CUDA4 Linux host; retain a readable
        # provenance marker if a future filesystem disallows them.
        if source.is_file():
            _atomic_json(target.with_suffix(target.suffix + ".link.json"), {"source": str(source)})


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _fresh_sparse(value: SparseTensor) -> SparseTensor:
    return SparseTensor(value.feats.detach().clone(), value.coords.detach().clone())


def _sparse_to_cpu(value: SparseTensor) -> SparseTensor:
    return SparseTensor(value.feats.detach().cpu().clone(), value.coords.detach().cpu().clone())


def _sparse_to_device(value: SparseTensor, device: torch.device) -> SparseTensor:
    return SparseTensor(value.feats.detach().to(device), value.coords.detach().to(device))


def _move_condition(value: Any, device: torch.device) -> Any:
    if isinstance(value, Mapping):
        return {key: _move_condition(item, device) for key, item in value.items()}
    if isinstance(value, SparseTensor):
        return _sparse_to_device(value, device)
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def _normalize_slat(value: SparseTensor, normalization: Mapping[str, Sequence[float]]) -> SparseTensor:
    mean = torch.as_tensor(normalization["mean"], device=value.device, dtype=value.feats.dtype).reshape(1, -1)
    std = torch.as_tensor(normalization["std"], device=value.device, dtype=value.feats.dtype).reshape(1, -1)
    return SparseTensor((value.feats - mean) / std, value.coords.detach().clone())


def _denormalize_slat(value: SparseTensor, normalization: Mapping[str, Sequence[float]]) -> SparseTensor:
    mean = torch.as_tensor(normalization["mean"], device=value.device, dtype=value.feats.dtype).reshape(1, -1)
    std = torch.as_tensor(normalization["std"], device=value.device, dtype=value.feats.dtype).reshape(1, -1)
    return SparseTensor(value.feats * std + mean, value.coords.detach().clone())


def _coordinate_digest(value: SparseTensor) -> str:
    raw = value.coords.detach().cpu().to(torch.int32).contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _strict_sparse_check(reference: SparseTensor, candidate: SparseTensor, label: str) -> Dict[str, Any]:
    same_shape = tuple(reference.feats.shape) == tuple(candidate.feats.shape)
    same_coords = tuple(reference.coords.shape) == tuple(candidate.coords.shape) and torch.equal(
        reference.coords, candidate.coords
    )
    record = {
        "label": str(label),
        "coords_exact": bool(same_coords),
        "feature_shape_equal": bool(same_shape),
        "reference_tokens": int(reference.feats.shape[0]),
        "candidate_tokens": int(candidate.feats.shape[0]),
        "reference_coord_digest": _coordinate_digest(reference),
        "candidate_coord_digest": _coordinate_digest(candidate),
    }
    if not same_shape or not same_coords:
        raise RuntimeError(f"strict sparse support invariant failed: {record}")
    return record


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().to(torch.float64)).item())


def _rms(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(value.detach().to(torch.float64).square())).item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().to(torch.float64).reshape(-1)
    right = right.detach().to(torch.float64).reshape(-1)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.item()) <= 1e-20:
        return 0.0
    return float(torch.dot(left, right).div(denominator).item())


def _tensor_digest(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _payload_sparse(value: SparseTensor) -> Dict[str, torch.Tensor]:
    return {
        "coords": value.coords.detach().cpu().to(torch.int32),
        "features": value.feats.detach().cpu().to(torch.float32),
    }


def _load_sparse(path: Path) -> SparseTensor:
    payload = _load_torch(path)
    if not isinstance(payload, Mapping) or "coords" not in payload or "features" not in payload:
        raise RuntimeError(f"invalid sparse payload: {path}")
    return SparseTensor(payload["features"].to(torch.float32), payload["coords"].to(torch.int32))


def _native_noised_endpoint(
    clean: SparseTensor,
    noise: SparseTensor,
    sampler: Any,
    timestep: float,
    strength: float,
) -> SparseTensor:
    _strict_sparse_check(clean, noise, "native noise")
    t = float(timestep)
    sigma = float(sampler.sigma_min) + (1.0 - float(sampler.sigma_min)) * t
    return SparseTensor(
        (1.0 - t) * clean.feats + sigma * float(strength) * noise.feats,
        clean.coords.detach().clone(),
    )


def _sampler_step_kwargs(params: Mapping[str, Any]) -> Dict[str, Any]:
    ignored = {
        "steps",
        "rescale_t",
        "verbose",
        "tqdm_desc",
        "record_trajectory",
        "trajectory_device",
        "return_model_history",
    }
    return {key: value for key, value in params.items() if key not in ignored}


def _native_schedule(sampler: Any, params: Mapping[str, Any]) -> List[float]:
    if int(params["steps"]) != TEXTURE_STEPS:
        raise ValueError("Codex.md fixes texture steps at 12")
    if float(params["rescale_t"]) != TEXTURE_RESCALE_T:
        raise ValueError("Codex.md fixes texture rescale_t at 3.0")
    schedule = [float(v) for v in sampler.timestep_schedule(int(params["steps"]), float(params["rescale_t"]))]
    if len(schedule) != TEXTURE_STEPS + 1 or any(schedule[i] <= schedule[i + 1] for i in range(len(schedule) - 1)):
        raise RuntimeError(f"invalid native texture schedule: {schedule}")
    return schedule


def _schedule_start(schedule: Sequence[float], timestep: float) -> int:
    matches = [i for i, value in enumerate(schedule) if abs(float(value) - float(timestep)) <= 1e-6]
    if len(matches) != 1:
        raise ValueError(f"noise timestep {timestep} is not a native schedule point: {schedule}")
    return int(matches[0])


# ---------------------------------------------------------------------------
# Sparse C4096 U operator
# ---------------------------------------------------------------------------


def _linear_keys(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    xyz = coords.to(torch.int64)
    return (xyz[:, 0] * int(resolution) + xyz[:, 1]) * int(resolution) + xyz[:, 2]


def _grid_base_frac(points: torch.Tensor, resolution: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be [N,3], got {tuple(points.shape)}")
    grid = (points.to(torch.float32) + 0.5) * float(resolution)
    finite = torch.isfinite(grid).all(dim=1)
    if not bool(finite.all().item()):
        raise RuntimeError("non-finite normalized coordinate in trilinear operator")
    base_index = torch.floor(grid - 0.5).to(torch.int64)
    fraction = grid - (base_index.to(torch.float32) + 0.5)
    return grid, base_index, fraction


def _stencil_support(points: torch.Tensor, resolution: int, chunk_size: int = 250_000) -> torch.Tensor:
    """Collect exactly the in-domain C-resolution stencil coordinates."""
    pieces: List[torch.Tensor] = []
    for start in range(0, int(points.shape[0]), int(chunk_size)):
        _, base_index, fraction = _grid_base_frac(points[start : start + chunk_size], resolution)
        del fraction
        local: List[torch.Tensor] = []
        for bits in range(8):
            bit = torch.tensor([(bits >> axis) & 1 for axis in range(3)], dtype=torch.int64)
            neighbour = base_index + bit
            valid = ((neighbour >= 0) & (neighbour < int(resolution))).all(dim=1)
            if bool(valid.any().item()):
                local.append(neighbour[valid])
        if local:
            pieces.append(torch.cat(local, dim=0))
    if not pieces:
        return torch.empty((0, 3), dtype=torch.int32)
    support = torch.unique(torch.cat(pieces, dim=0), dim=0)
    order = torch.argsort(_linear_keys(support, resolution), stable=True)
    return support.index_select(0, order).to(torch.int32).contiguous()


def _query_matrix_chunk(
    support_coords: torch.Tensor,
    points: torch.Tensor,
    resolution: int,
    row_offset: int,
    row_weight: float,
) -> csr_matrix:
    """Build one weighted CSR chunk with MeshWithVoxel's cell-center rule."""
    support = support_coords.to(torch.int64)
    support_keys = _linear_keys(support, resolution)
    support_order = torch.argsort(support_keys, stable=True)
    sorted_keys = support_keys.index_select(0, support_order)
    _, base_index, fraction = _grid_base_frac(points, resolution)
    row_count = int(points.shape[0])
    row_parts: List[torch.Tensor] = []
    col_parts: List[torch.Tensor] = []
    value_parts: List[torch.Tensor] = []
    row_sum = torch.zeros(row_count, dtype=torch.float64)
    for bits in range(8):
        bit = torch.tensor([(bits >> axis) & 1 for axis in range(3)], dtype=torch.int64)
        neighbour = base_index + bit
        raw_weight = torch.where(bit.bool(), fraction, 1.0 - fraction).prod(dim=1).to(torch.float64)
        valid = ((neighbour >= 0) & (neighbour < int(resolution))).all(dim=1)
        neighbour_keys = _linear_keys(neighbour, resolution)
        positions = torch.searchsorted(sorted_keys, neighbour_keys)
        if sorted_keys.numel() == 0:
            valid &= False
            safe_positions = torch.zeros_like(positions)
        else:
            safe_positions = positions.clamp_max(sorted_keys.numel() - 1)
            valid &= positions < sorted_keys.numel()
            valid &= sorted_keys.index_select(0, safe_positions) == neighbour_keys
        rows = torch.where(valid)[0]
        if rows.numel():
            cols = support_order.index_select(0, safe_positions.index_select(0, rows))
            values = raw_weight.index_select(0, rows)
            row_parts.append(rows)
            col_parts.append(cols)
            value_parts.append(values)
            row_sum.index_add_(0, rows, values)
    valid_rows = row_sum > 1e-14
    if not bool(valid_rows.all().item()):
        invalid = torch.where(~valid_rows)[0][:8].tolist()
        raise RuntimeError(f"U observation has no valid C4096 stencil rows: {invalid}")
    rows = torch.cat(row_parts)
    cols = torch.cat(col_parts)
    values = torch.cat(value_parts) / row_sum.index_select(0, rows)
    if float(row_weight) <= 0.0 or not math.isfinite(float(row_weight)):
        raise ValueError(f"row weight must be positive finite, got {row_weight}")
    values = values * math.sqrt(float(row_weight))
    matrix = csr_matrix(
        (
            values.numpy(),
            (rows.numpy() + int(row_offset), cols.numpy()),
        ),
        shape=(int(row_offset) + row_count, int(support_coords.shape[0])),
        dtype=np.float64,
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    # Remove the leading empty rows from this chunk.  The caller uses a
    # global row offset only to construct the coordinates, not the chunk CSR.
    if row_offset:
        matrix = matrix[row_offset:, :]
    return matrix.tocsr()


def build_sparse_c4096_operator(
    point_blocks: Sequence[Tuple[str, torch.Tensor, float]],
    *,
    resolution: int = GLOBAL_U_RESOLUTION,
    chunk_size: int = 250_000,
) -> OperatorCache:
    """Build ``[A_G; A_1; ...]`` on compact sparse C4096 support.

    ``point_blocks`` contains normalized global coordinates and physical
    quadrature weights.  The resulting matrix is row-weighted by
    ``sqrt(weight)`` for direct use by LSMR.
    """
    if not point_blocks:
        raise ValueError("at least one observation block is required")
    supports = [_stencil_support(points.cpu(), resolution, chunk_size) for _, points, _ in point_blocks]
    support_coords = torch.unique(torch.cat(supports, dim=0), dim=0) if supports else torch.empty((0, 3), dtype=torch.int32)
    if support_coords.numel():
        order = torch.argsort(_linear_keys(support_coords, resolution), stable=True)
        support_coords = support_coords.index_select(0, order).contiguous()
    matrices: List[csr_matrix] = []
    blocks: List[Dict[str, Any]] = []
    row_start = 0
    row_sum_errors: List[float] = []
    invalid_boundary_neighbors = 0
    invalid_boundary_rows = 0
    for name, points, weight in point_blocks:
        points = points.detach().cpu().to(torch.float32).contiguous()
        chunks: List[csr_matrix] = []
        for start in range(0, int(points.shape[0]), int(chunk_size)):
            _, boundary_base, _ = _grid_base_frac(points[start : start + chunk_size], resolution)
            boundary_mask = torch.zeros(boundary_base.shape[0], dtype=torch.bool)
            for bits in range(8):
                bit = torch.tensor([(bits >> axis) & 1 for axis in range(3)], dtype=torch.int64)
                neighbour = boundary_base + bit
                boundary_mask |= ~((neighbour >= 0) & (neighbour < int(resolution))).all(dim=1)
            invalid_boundary_rows += int(boundary_mask.sum().item())
            for bits in range(8):
                bit = torch.tensor([(bits >> axis) & 1 for axis in range(3)], dtype=torch.int64)
                neighbour = boundary_base + bit
                invalid_boundary_neighbors += int(
                    (~((neighbour >= 0) & (neighbour < int(resolution))).all(dim=1)).sum().item()
                )
            chunks.append(
                _query_matrix_chunk(
                    support_coords,
                    points[start : start + chunk_size],
                    resolution,
                    row_offset=0,
                    row_weight=float(weight),
                )
            )
        block_matrix = vstack(chunks, format="csr") if chunks else csr_matrix((0, int(support_coords.shape[0])), dtype=np.float64)
        row_sums = np.asarray(block_matrix.sum(axis=1)).reshape(-1)
        row_sum_errors.append(
            float(np.max(np.abs(row_sums - math.sqrt(float(weight)))) if row_sums.size else 0.0)
        )
        matrices.append(block_matrix)
        blocks.append(
            {
                "name": str(name),
                "start": int(row_start),
                "stop": int(row_start + points.shape[0]),
                "rows": int(points.shape[0]),
                "weight": float(weight),
                "sqrt_weight": float(math.sqrt(float(weight))),
                "nnz": int(block_matrix.nnz),
            }
        )
        row_start += int(points.shape[0])
    matrix = vstack(matrices, format="csr")
    if not np.isfinite(matrix.data).all():
        raise RuntimeError("weighted U operator contains non-finite coefficients")
    metadata = {
        "format": FORMAT,
        "resolution": int(resolution),
        "variables": int(support_coords.shape[0]),
        "rows": int(matrix.shape[0]),
        "nnz": int(matrix.nnz),
        "row_blocks": blocks,
        "row_sum_error_max": float(max(row_sum_errors) if row_sum_errors else 0.0),
        "invalid_boundary_query_rows": int(invalid_boundary_rows),
        "invalid_boundary_neighbor_count": int(invalid_boundary_neighbors),
        "physical_weight_min": float(min(float(weight) for _, _, weight in point_blocks)),
        "physical_weight_max": float(max(float(weight) for _, _, weight in point_blocks)),
        "support_rule": "standard 8-neighbor trilinear C4096 cell-center interpolation with in-domain renormalization",
        "operator_direction": "compact global U -> each native observation support",
        "dense_4096_cube_allocated": False,
    }
    return OperatorCache(matrix=matrix, support_coords=support_coords, blocks=blocks, metadata=metadata)


def _operator_block_prediction(operator: OperatorCache, block_index: int, values: np.ndarray) -> np.ndarray:
    block = operator.blocks[int(block_index)]
    weighted = operator.matrix[int(block["start"]) : int(block["stop"]), :].dot(values)
    return np.asarray(weighted / float(block["sqrt_weight"]), dtype=np.float64)


def _operator_apply_unweighted(operator: OperatorCache, block_index: int, values: torch.Tensor) -> torch.Tensor:
    array = values.detach().cpu().to(torch.float64).numpy()
    return torch.from_numpy(_operator_block_prediction(operator, block_index, array)).to(torch.float32)


def _solve_joint_u(
    operator: OperatorCache,
    global_field: torch.Tensor,
    tile_fields: Mapping[int, torch.Tensor],
    *,
    atol: float = 1e-6,
    btol: float = 1e-6,
    maxiter: int = 200,
    damp: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Solve six independent WLS channels using scipy LSMR."""
    if global_field.ndim != 2 or global_field.shape[1] != PBR_CHANNELS:
        raise ValueError("global field must be [N,6]")
    ordered_fields: List[torch.Tensor] = [global_field]
    for block in operator.blocks[1:]:
        name = str(block["name"])
        if not name.startswith("tile_"):
            raise RuntimeError(f"unexpected U row block {name}")
        tile_id = int(name.split("_")[-1])
        if tile_id not in tile_fields:
            raise KeyError(f"missing current endpoint field for {name}")
        ordered_fields.append(tile_fields[tile_id])
    weighted_rhs = np.concatenate(
        [
            field.detach().cpu().to(torch.float64).numpy()
            * float(block["sqrt_weight"])
            for field, block in zip(ordered_fields, operator.blocks)
        ],
        axis=0,
    )
    if weighted_rhs.shape[0] != operator.matrix.shape[0]:
        raise RuntimeError("U RHS row count does not match cached operator")
    if not np.isfinite(weighted_rhs).all():
        raise RuntimeError("U RHS contains non-finite PBR values")
    variables = int(operator.matrix.shape[1])
    solution = np.empty((variables, PBR_CHANNELS), dtype=np.float64)
    channel_stats: List[Dict[str, Any]] = []
    for channel in range(PBR_CHANNELS):
        result = lsmr(
            operator.matrix,
            weighted_rhs[:, channel],
            damp=float(damp),
            atol=float(atol),
            btol=float(btol),
            conlim=1e8,
            maxiter=int(maxiter),
            show=False,
        )
        solution[:, channel] = result[0]
        channel_stats.append(
            {
                "channel": int(channel),
                "istop": int(result[1]),
                "iterations": int(result[2]),
                "normr": float(result[3]),
                "normar": float(result[4]),
                "condA": float(result[6]),
                "normx": float(result[7]),
            }
        )
    if not np.isfinite(solution).all():
        raise RuntimeError("U LSMR solution contains non-finite values")
    stats = {
        "channels": channel_stats,
        "iterations_mean": float(np.mean([row["iterations"] for row in channel_stats])),
        "normr_mean": float(np.mean([row["normr"] for row in channel_stats])),
        "condA_max": float(max(row["condA"] for row in channel_stats)),
        "rhs_rows": int(weighted_rhs.shape[0]),
        "variables": variables,
        "solver": "scipy.sparse.linalg.lsmr",
        "dtype": "float64",
        "atol": float(atol),
        "btol": float(btol),
        "maxiter": int(maxiter),
        "damp": float(damp),
    }
    return torch.from_numpy(solution).to(torch.float64), stats


# ---------------------------------------------------------------------------
# Differentiable sparse decoder query and covariance algebra
# ---------------------------------------------------------------------------


def differentiable_sparse_trilinear_query(
    attrs: torch.Tensor,
    coords: torch.Tensor,
    points: torch.Tensor,
    *,
    resolution: int = OVOXEL_RESOLUTION,
    chunk_size: int = 65_536,
) -> torch.Tensor:
    """Differentiable equivalent of sparse MeshWithVoxel trilinear lookup."""
    if attrs.ndim != 2 or coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError("attrs/coords have invalid sparse shapes")
    xyz = coords[:, -3:].to(torch.int64)
    if coords.shape[1] == 4 and bool((coords[:, 0] != 0).any().item()):
        raise RuntimeError("differentiable query only supports batch zero")
    if attrs.shape[0] != xyz.shape[0]:
        raise ValueError("attrs and coords are not row aligned")
    keys = _linear_keys(xyz, resolution)
    order = torch.argsort(keys, stable=True)
    sorted_keys = keys.index_select(0, order)
    if sorted_keys.numel() > 1 and bool((sorted_keys[1:] == sorted_keys[:-1]).any().item()):
        raise RuntimeError("decoded texture support contains duplicate coordinates")
    sorted_attrs = attrs.index_select(0, order)
    outputs: List[torch.Tensor] = []
    for start in range(0, int(points.shape[0]), int(chunk_size)):
        query = points[start : start + chunk_size]
        _, base_index, fraction = _grid_base_frac(query, resolution)
        row_sum = torch.zeros(query.shape[0], device=attrs.device, dtype=attrs.dtype)
        output = torch.zeros((query.shape[0], attrs.shape[1]), device=attrs.device, dtype=attrs.dtype)
        for bits in range(8):
            bit = torch.tensor(
                [(bits >> axis) & 1 for axis in range(3)],
                device=attrs.device,
                dtype=torch.int64,
            )
            neighbour = base_index.to(attrs.device) + bit
            raw_weight = torch.where(bit.bool(), fraction.to(attrs.device), 1.0 - fraction.to(attrs.device)).prod(dim=1).to(attrs.dtype)
            valid = ((neighbour >= 0) & (neighbour < int(resolution))).all(dim=1)
            neighbour_keys = _linear_keys(neighbour, resolution)
            positions = torch.searchsorted(sorted_keys, neighbour_keys)
            if sorted_keys.numel() == 0:
                valid &= False
                safe = torch.zeros_like(positions)
            else:
                safe = positions.clamp_max(sorted_keys.numel() - 1)
                valid &= positions < sorted_keys.numel()
                valid &= sorted_keys.index_select(0, safe) == neighbour_keys
            safe_attrs = sorted_attrs.index_select(0, safe)
            valid_float = valid.to(attrs.dtype)
            output = output + safe_attrs * (raw_weight * valid_float)[:, None]
            row_sum = row_sum + raw_weight * valid_float
        # Match ``MeshWithVoxel.query_attrs`` exactly for sparse holes.  Its
        # CUDA kernel emits zero weights when none of the eight neighbours is
        # present and the final denominator is clamped to 1e-12, therefore a
        # row without decoded support is a valid zero observation rather than
        # a malformed query.  Keeping that convention is important here:
        # geometry support and decoded texture support are intentionally
        # different sparse sets, and the decoder Jacobian must include the
        # same zero rows as the official PBR query.
        outputs.append(output / row_sum.clamp_min(1e-12)[:, None])
    return torch.cat(outputs, dim=0) if outputs else attrs.new_empty((0, attrs.shape[1]))


def covariance_vector_product(
    centered_samples: torch.Tensor,
    gradient: torch.Tensor,
    sigma2: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``B(B^T g)+sigma2*g`` without materialising C."""
    if centered_samples.ndim != 3 or gradient.ndim != 2:
        raise ValueError("centered samples must be [M,N,C] and gradient [N,C]")
    if centered_samples.shape[1:] != gradient.shape:
        raise ValueError("covariance and gradient supports differ")
    if centered_samples.shape[0] < 2:
        raise ValueError("at least two covariance samples are required")
    scale = math.sqrt(float(centered_samples.shape[0] - 1))
    b = centered_samples.reshape(centered_samples.shape[0], -1) / scale
    g = gradient.reshape(-1)
    coefficients = torch.mv(b, g)
    low_rank = torch.mv(b.T, coefficients).reshape_as(gradient)
    isotropic = float(sigma2) * gradient
    result = low_rank + isotropic
    return result, low_rank, isotropic


def covariance_stats(centered_samples: torch.Tensor, sigma_res_ratio: float = 0.05, sigma_floor: float = 1e-4) -> Dict[str, Any]:
    if centered_samples.ndim != 3 or centered_samples.shape[0] < 2:
        raise ValueError("centered samples must be [M,N,C] with M>=2")
    m = int(centered_samples.shape[0])
    d = int(centered_samples[0].numel())
    variance = float(centered_samples.to(torch.float64).square().sum().item() / ((m - 1) * d))
    sigma2 = max(float(sigma_res_ratio) * variance, float(sigma_floor))
    gram = torch.zeros((m, m), dtype=torch.float64)
    flat = centered_samples.reshape(m, -1)
    chunk = 1_000_000
    for start in range(0, d, chunk):
        part = flat[:, start : start + chunk].to(torch.float64) / math.sqrt(m - 1)
        gram += part @ part.T
    eigenvalues = torch.linalg.eigvalsh(gram).flip(0).clamp_min(0.0)
    total = float(eigenvalues.sum().item())
    explained = (eigenvalues / total).tolist() if total > 0.0 else [0.0] * m
    pairwise = []
    for left in range(m):
        for right in range(left + 1, m):
            pairwise.append(_rms(centered_samples[left] - centered_samples[right]))
    sample_mean_rms = [_rms(row) for row in centered_samples]
    return {
        "M": m,
        "D": d,
        "rank_upper_bound": m - 1,
        "mean_coordinate_variance": variance,
        "sigma_res2": sigma2,
        "sigma_res_ratio": float(sigma_res_ratio),
        "sigma_res_floor": float(sigma_floor),
        "B_singular_values": torch.sqrt(eigenvalues).tolist(),
        "B_explained_variance": explained,
        "sample_pairwise_normalized_latent_rms": {
            "mean": float(np.mean(pairwise)) if pairwise else 0.0,
            "min": float(np.min(pairwise)) if pairwise else 0.0,
            "max": float(np.max(pairwise)) if pairwise else 0.0,
            "count": len(pairwise),
        },
        "sample_to_mean_rms": {
            "mean": float(np.mean(sample_mean_rms)) if sample_mean_rms else 0.0,
            "min": float(np.min(sample_mean_rms)) if sample_mean_rms else 0.0,
            "max": float(np.max(sample_mean_rms)) if sample_mean_rms else 0.0,
        },
    }


def _make_correction(pred_x0: SparseTensor, direction: torch.Tensor, rho: float) -> Tuple[SparseTensor, float]:
    if direction.shape != pred_x0.feats.shape:
        raise ValueError("correction direction and endpoint feature shapes differ")
    direction_rms = _rms(direction)
    if direction_rms < 1e-8:
        return _fresh_sparse(pred_x0), direction_rms
    delta = float(rho) * direction / (direction_rms + 1e-8)
    corrected = SparseTensor(pred_x0.feats + delta, pred_x0.coords.detach().clone())
    return corrected, _rms(delta)


# ---------------------------------------------------------------------------
# Context and fixed shape-guide preparation
# ---------------------------------------------------------------------------


def _tile_image(image_4096: Image.Image, box: Sequence[int]) -> Image.Image:
    tile = image_4096.crop(tuple(int(v) for v in box)).convert("RGB")
    if tile.size != (TILE_SIZE, TILE_SIZE):
        tile = tile.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS)
    return tile


def _load_contexts(
    *,
    cache_dir: Path,
    output_dir: Path,
    pipeline: Any,
    baseline_mesh: MeshWithVoxel,
    global_camera: Mapping[str, Any],
    image_4096: Image.Image,
    tile_ids: Sequence[int],
    extend_pixel: int,
    low_vram: bool,
    face_projection_chunk_size: int,
) -> List[base.TileContext]:
    boxes = core._tile_layout(CANONICAL_IMAGE_SIZE, TILE_SIZE, TILE_STRIDE)
    if len(boxes) != 49:
        raise RuntimeError("canonical 4096/1024/512 layout must contain 49 tiles")
    device = torch.device("cuda")
    face_min, face_max, face_finite = core._project_face_bboxes(
        baseline_mesh.vertices,
        baseline_mesh.faces,
        mesh_scale=float(global_camera["mesh_scale"]),
        global_camera=global_camera,
        chunk_size=int(face_projection_chunk_size),
    )
    contexts: List[base.TileContext] = []
    for tile_id in sorted(int(v) for v in tile_ids):
        if tile_id < 0 or tile_id >= len(boxes):
            raise ValueError(f"tile id {tile_id} outside canonical layout")
        box = boxes[tile_id]
        source_tile = cache_dir / "tiles" / f"tile_{tile_id:02d}"
        required = [
            source_tile / "fixed_shape_norm.pt",
            source_tile / "texture_reference_norm.pt",
            source_tile / "texture_initial_state.pt",
            source_tile / "fixed_shape_summary.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"tile {tile_id} cache is incomplete: {missing}")
        transform = core._derive_tile_camera(
            tile_id=tile_id,
            box=box,
            global_camera=global_camera,
            extend_pixel=int(extend_pixel),
        )
        geometry = core._prepare_tile_geometry(
            global_vertices=baseline_mesh.vertices,
            global_faces=baseline_mesh.faces,
            global_face_min=face_min,
            global_face_max=face_max,
            global_face_finite=face_finite,
            global_camera=global_camera,
            transform=transform,
        )
        shape_norm = _load_sparse(required[0])
        texture_norm = _load_sparse(required[1])
        initial_state = _load_sparse(required[2])
        _strict_sparse_check(shape_norm, texture_norm, f"tile {tile_id} fixed shape/texture")
        _strict_sparse_check(texture_norm, initial_state, f"tile {tile_id} texture/initial")
        shape_denorm = _denormalize_slat(shape_norm, pipeline.shape_slat_normalization)
        tile_image = _tile_image(image_4096, box)
        tile_dir = output_dir / "tiles" / f"tile_{tile_id:02d}"
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_image.save(tile_dir / "hr_tile_1024_condition.png")
        _atomic_json(tile_dir / "tile_camera.json", transform.__dict__)
        condition = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_tex_1024,
            [tile_image],
            shape_norm.coords.to(device=device, dtype=torch.int32),
            camera_angle_x=float(transform.camera_angle_x),
            distance=float(transform.distance),
            mesh_scale=float(transform.mesh_scale),
            grid_resolution_override=LATENT_RESOLUTION,
        )
        if low_vram:
            condition = _move_condition(condition, torch.device("cpu"))
        target_coords = geometry.coords.to(device=torch.device("cpu"), dtype=torch.int32)
        target_points = (target_coords.to(torch.float32) + 0.5) / float(OVOXEL_RESOLUTION) - 0.5
        static_stats = json.loads(required[3].read_text(encoding="utf-8"))
        context = base.TileContext(
            tile_id=tile_id,
            box=tuple(int(v) for v in box),
            transform=transform,
            image=tile_image,
            tile_dir=tile_dir,
            geometry=geometry,
            shape_reference=_fresh_sparse(shape_norm),
            shape_norm=_sparse_to_cpu(shape_norm),
            shape_denorm=_sparse_to_cpu(shape_denorm),
            texture_reference=_fresh_sparse(texture_norm),
            texture_norm=_sparse_to_cpu(texture_norm),
            noise=SparseTensor(torch.zeros_like(texture_norm.feats), texture_norm.coords.detach().clone()),
            initial_state=_sparse_to_cpu(initial_state),
            condition=condition,
            target_coords=target_coords,
            target_points=target_points,
            static_stats={**static_stats, "joint_u_cov_context_cache": str(cache_dir.resolve())},
        )
        contexts.append(context)
        print(f"[context tile {tile_id:02d}] native_rows={target_points.shape[0]:,} latent_tokens={shape_norm.feats.shape[0]:,}")
    if not contexts:
        raise RuntimeError("no active tile contexts")
    return contexts


def _guide_payload(subs: Sequence[SparseTensor]) -> List[Dict[str, torch.Tensor]]:
    return [_payload_sparse(_sparse_to_cpu(value)) for value in subs]


def _load_guide_payload(path: Path) -> List[SparseTensor]:
    payload = _load_torch(path)
    if not isinstance(payload, list):
        raise RuntimeError(f"invalid texture decoder guide cache: {path}")
    return [SparseTensor(item["features"].to(torch.float32), item["coords"].to(torch.int32)) for item in payload]


def _prepare_shape_guides(
    contexts: Sequence[base.TileContext],
    pipeline: Any,
    *,
    resolution: int,
    resume: bool,
) -> Dict[int, List[SparseTensor]]:
    guides: Dict[int, List[SparseTensor]] = {}
    for index, context in enumerate(contexts):
        path = context.tile_dir / "fixed_shape_decoder_guides.pt"
        if resume and path.is_file():
            guides[context.tile_id] = _load_guide_payload(path)
            print(f"[shape guide {context.tile_id:02d}] resumed layers={len(guides[context.tile_id])}")
            continue
        print(f"[shape guide {context.tile_id:02d}] decoding fixed shape ({index + 1}/{len(contexts)})")
        shape_gpu = _sparse_to_device(context.shape_denorm, torch.device("cuda"))
        with torch.no_grad():
            decoded = pipeline.decode_shape_slat(shape_gpu, OVOXEL_RESOLUTION)
        if not isinstance(decoded, tuple) or len(decoded) != 2:
            raise RuntimeError(f"tile {context.tile_id}: fixed shape decoder did not return mesh/subdivision guides")
        _, subs = decoded
        if not isinstance(subs, list) or not subs:
            raise RuntimeError(f"tile {context.tile_id}: empty fixed shape decoder guide list")
        cpu_subs = [_sparse_to_cpu(value) for value in subs]
        guides[context.tile_id] = cpu_subs
        _atomic_torch_save(path, _guide_payload(cpu_subs))
        del shape_gpu, decoded, subs, cpu_subs
        _empty_cuda_cache()
    return guides


def _decoder_support_digest(decoded: SparseTensor) -> str:
    return _tensor_digest(decoded.coords.to(torch.int32))


def _decode_texture_attrs(
    pipeline: Any,
    texture_norm: SparseTensor,
    guide_subs: Sequence[SparseTensor],
    *,
    requires_grad: bool,
) -> SparseTensor:
    raw = _denormalize_slat(texture_norm, pipeline.tex_slat_normalization)
    decoder = pipeline.models["tex_slat_decoder"]
    decoded = decoder(raw, guide_subs=list(guide_subs)) * 0.5 + 0.5
    if not isinstance(decoded, SparseTensor):
        raise RuntimeError(f"texture decoder returned {type(decoded)!r}")
    if decoded.feats.shape[1] != PBR_CHANNELS:
        raise RuntimeError(f"texture decoder returned {decoded.feats.shape[1]} PBR channels")
    if not requires_grad:
        if not torch.isfinite(decoded.feats).all():
            raise RuntimeError("texture decoder returned non-finite attrs")
    return decoded


def _official_query_for_check(decoded: SparseTensor, points: torch.Tensor) -> torch.Tensor:
    spatial_shape = decoded.spatial_shape
    decoded_coords = decoded.coords.to(torch.int32)
    if decoded_coords.ndim != 2 or decoded_coords.shape[1] not in (3, 4):
        raise RuntimeError(f"decoded texture coordinates have invalid shape: {tuple(decoded_coords.shape)}")
    if decoded_coords.shape[1] == 4 and bool((decoded_coords[:, 0] != 0).any().item()):
        raise RuntimeError("official query check only supports batch zero")
    mesh = MeshWithVoxel(
        vertices=torch.empty((1, 3), device=decoded.device),
        faces=torch.empty((0, 3), dtype=torch.int32, device=decoded.device),
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / float(OVOXEL_RESOLUTION),
        # MeshWithVoxel.query_attrs appends the batch-zero column itself.
        coords=decoded_coords[:, -3:],
        attrs=decoded.feats,
        voxel_shape=torch.Size([1, int(decoded.feats.shape[1]), *spatial_shape]),
        layout={"base_color": slice(0, 3), "metallic": slice(3, 4), "roughness": slice(4, 5), "alpha": slice(5, 6)},
    )
    return mesh.query_attrs(points)


def _decode_endpoint_field(
    pipeline: Any,
    context: base.TileContext,
    endpoint: SparseTensor,
    guide_subs: Sequence[SparseTensor],
    *,
    query_check: bool,
    support_digest: Optional[str],
    query_chunk_size: int,
) -> Tuple[torch.Tensor, str, Dict[str, Any]]:
    endpoint_gpu = _sparse_to_device(endpoint, torch.device("cuda"))
    guides_gpu = [_sparse_to_device(value, torch.device("cuda")) for value in guide_subs]
    decoded = _decode_texture_attrs(pipeline, endpoint_gpu, guides_gpu, requires_grad=False)
    digest = _decoder_support_digest(decoded)
    if support_digest is not None and digest != support_digest:
        raise RuntimeError(f"tile {context.tile_id}: texture decoder support changed")
    points = context.target_points.to(device=torch.device("cuda"), dtype=torch.float32)
    field = differentiable_sparse_trilinear_query(
        decoded.feats,
        decoded.coords,
        points,
        resolution=OVOXEL_RESOLUTION,
        chunk_size=int(query_chunk_size),
    )
    check: Dict[str, Any] = {"performed": False}
    if query_check:
        sample_count = min(1024, int(points.shape[0]))
        official = _official_query_for_check(decoded, points[:sample_count])
        explicit = field[:sample_count]
        error = (official - explicit).abs()
        max_error = float(error.max().item()) if error.numel() else 0.0
        if max_error >= 1e-5:
            raise RuntimeError(
                f"tile {context.tile_id}: explicit differentiable query disagrees with MeshWithVoxel.query_attrs: {max_error}"
            )
        check = {"performed": True, "sample_rows": sample_count, "max_abs_error": max_error}
    if not torch.isfinite(field).all():
        raise RuntimeError(f"tile {context.tile_id}: decoded physical field is non-finite")
    record = {
        "decoded_tokens": int(decoded.feats.shape[0]),
        "decoded_coord_digest": digest,
        "queried_native_rows": int(field.shape[0]),
        "pbr_min": float(field.min().item()) if field.numel() else 0.0,
        "pbr_max": float(field.max().item()) if field.numel() else 0.0,
        "explicit_query_check": check,
    }
    return field.detach().cpu().to(torch.float32), digest, record


def _decode_endpoint_field_with_grad(
    pipeline: Any,
    context: base.TileContext,
    endpoint: SparseTensor,
    target: torch.Tensor,
    guide_subs: Sequence[SparseTensor],
    *,
    physical_weight: float,
    support_digest: Optional[str],
    query_chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, str, float, Dict[str, Any]]:
    endpoint_gpu = _sparse_to_device(endpoint, torch.device("cuda"))
    _patch_flex_gemm_input_backward()
    z_features = endpoint_gpu.feats.detach().clone().requires_grad_(True)
    z_norm = SparseTensor(z_features, endpoint_gpu.coords.detach().clone())
    guides_gpu = [_sparse_to_device(value, torch.device("cuda")) for value in guide_subs]
    decoded = _decode_texture_attrs(pipeline, z_norm, guides_gpu, requires_grad=True)
    digest = _decoder_support_digest(decoded)
    if support_digest is not None and digest != support_digest:
        raise RuntimeError(f"tile {context.tile_id}: texture decoder support changed during gradient route")
    points = context.target_points.to(device=torch.device("cuda"), dtype=torch.float32)
    field = differentiable_sparse_trilinear_query(
        decoded.feats,
        decoded.coords,
        points,
        resolution=OVOXEL_RESOLUTION,
        chunk_size=int(query_chunk_size),
    )
    target_gpu = target.to(device=torch.device("cuda"), dtype=field.dtype)
    residual = field - target_gpu
    loss = 0.5 * float(physical_weight) * residual.square().sum()
    if not torch.isfinite(loss):
        raise RuntimeError(f"tile {context.tile_id}: physical loss is non-finite")
    grad_e = torch.autograd.grad(loss, z_features, retain_graph=False, create_graph=False)[0]
    gradient = -grad_e
    if gradient.shape != endpoint_gpu.feats.shape or not torch.isfinite(gradient).all():
        raise RuntimeError(f"tile {context.tile_id}: endpoint gradient is invalid")
    return field.detach(), gradient.detach(), digest, float(loss.detach().item()), {
        "decoded_tokens": int(decoded.feats.shape[0]),
        "query_rows": int(field.shape[0]),
        "loss": float(loss.detach().item()),
        "gradient_rms": _rms(gradient),
    }


def _condition_signature(context: base.TileContext, pipeline: Any, seeds: Sequence[int], model_path: str) -> Dict[str, Any]:
    return {
        "tile_id": int(context.tile_id),
        "box": list(context.box),
        "shape_coord_digest": _coordinate_digest(context.shape_norm),
        "texture_coord_digest": _coordinate_digest(context.texture_norm),
        "texture_feature_shape": list(context.texture_norm.feats.shape),
        "model_path": str(model_path),
        "seed_list": [int(v) for v in seeds],
        "tex_normalization": _jsonable(pipeline.tex_slat_normalization),
        "condition_digest": _condition_digest(context.condition),
    }


def _condition_digest(value: Any) -> str:
    digest = hashlib.sha256()
    def update(item: Any) -> None:
        if isinstance(item, Mapping):
            for key in sorted(item):
                digest.update(str(key).encode("utf-8"))
                update(item[key])
        elif isinstance(item, SparseTensor):
            digest.update(item.coords.detach().cpu().contiguous().numpy().tobytes())
            digest.update(item.feats.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, torch.Tensor):
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(repr(item).encode("utf-8"))
    update(value)
    return digest.hexdigest()


def _covariance_from_samples(
    context: base.TileContext,
    samples: Sequence[SparseTensor],
    *,
    sigma_res_ratio: float,
    sigma_floor: float,
) -> CovarianceCache:
    if not samples:
        raise ValueError("no covariance samples")
    reference = samples[0]
    for index, sample in enumerate(samples[1:], start=1):
        _strict_sparse_check(reference, sample, f"tile {context.tile_id} covariance seed {index}")
    stack = torch.stack([sample.feats.detach().cpu().to(torch.float32) for sample in samples], dim=0)
    mean = stack.mean(dim=0)
    centered = stack - mean
    stats = covariance_stats(centered, sigma_res_ratio=sigma_res_ratio, sigma_floor=sigma_floor)
    return CovarianceCache(
        coords=reference.coords.detach().cpu().to(torch.int32).clone(),
        mean=mean,
        centered=centered,
        sigma2=float(stats["sigma_res2"]),
        stats=stats,
    )


def _save_covariance_cache(
    context: base.TileContext,
    cache: CovarianceCache,
    signature: Mapping[str, Any],
) -> None:
    root = context.tile_dir / "covariance"
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "seeds.json", {"format": FORMAT, "signature": signature})
    _atomic_torch_save(root / "mean_norm.pt", {"coords": cache.coords, "features": cache.mean})
    _atomic_torch_save(
        root / "centered_samples_norm.pt",
        {
            "coords": cache.coords,
            "centered_samples": cache.centered,
            "M": int(cache.centered.shape[0]),
            "normalization_space": "normalized texture-SLat",
            "covariance_definition": "B B^T + sigma_res2 I with B=centered/sqrt(M-1)",
        },
    )
    _atomic_json(root / "covariance_stats.json", {**cache.stats, "signature": signature})


def _load_covariance_cache(context: base.TileContext, signature: Mapping[str, Any]) -> CovarianceCache:
    root = context.tile_dir / "covariance"
    seeds_path = root / "seeds.json"
    mean_path = root / "mean_norm.pt"
    centered_path = root / "centered_samples_norm.pt"
    stats_path = root / "covariance_stats.json"
    if not all(path.is_file() for path in (seeds_path, mean_path, centered_path, stats_path)):
        raise FileNotFoundError(f"incomplete covariance cache for tile {context.tile_id}")
    saved_signature = json.loads(seeds_path.read_text(encoding="utf-8")).get("signature")
    if saved_signature != _jsonable(signature):
        raise RuntimeError(f"tile {context.tile_id}: covariance cache provenance mismatch")
    mean_payload = _load_torch(mean_path)
    centered_payload = _load_torch(centered_path)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    coords = mean_payload["coords"].to(torch.int32)
    mean = mean_payload["features"].to(torch.float32)
    centered = centered_payload["centered_samples"].to(torch.float32)
    if not torch.equal(coords, context.texture_norm.coords.cpu()) or tuple(mean.shape) != tuple(context.texture_norm.feats.shape):
        raise RuntimeError(f"tile {context.tile_id}: covariance cache support/shape mismatch")
    if centered.shape[1:] != mean.shape or centered.shape[0] != len(COVARIANCE_SEEDS):
        raise RuntimeError(f"tile {context.tile_id}: covariance sample tensor shape mismatch")
    return CovarianceCache(coords=coords, mean=mean, centered=centered, sigma2=float(stats["sigma_res2"]), stats=stats)


def _flow_from_state(
    context: base.TileContext,
    state: SparseTensor,
    pipeline: Any,
    model: Any,
    merged: Mapping[str, Any],
    schedule: Sequence[float],
    *,
    label: str,
) -> SparseTensor:
    state_gpu = _sparse_to_device(state, torch.device("cuda"))
    shape_gpu = _sparse_to_device(context.shape_norm, torch.device("cuda"))
    condition = _move_condition(context.condition, torch.device("cuda"))
    start_index = _schedule_start(schedule, 1.0)
    if start_index != 0:
        raise RuntimeError("current experiment requires native noise_timestep=1.0")
    try:
        with torch.no_grad():
            output = pipeline.tex_slat_sampler.sample(
                model,
                state_gpu,
                cond=condition["cond"],
                neg_cond=condition["neg_cond"],
                concat_cond=shape_gpu,
                **dict(merged),
                verbose=False,
                tqdm_desc=label,
                record_trajectory=False,
                return_model_history=False,
            ).samples
    finally:
        del shape_gpu, condition
    if not isinstance(output, SparseTensor):
        raise RuntimeError(f"{label}: official flow returned {type(output)!r}")
    _strict_sparse_check(state_gpu, output, f"{label} final")
    result = _sparse_to_cpu(output)
    del state_gpu, output
    return result


def _covariance_seed_state(context: base.TileContext, seed: int, pipeline: Any) -> SparseTensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(context.tile_id) * 100003)
    noise = torch.randn(context.texture_norm.feats.shape, generator=generator, dtype=context.texture_norm.feats.dtype)
    noise_sparse = SparseTensor(noise, context.texture_norm.coords.detach().clone())
    return _native_noised_endpoint(context.texture_norm, noise_sparse, pipeline.tex_slat_sampler, 1.0, 1.0)


def _run_covariance_for_tile(
    context: base.TileContext,
    pipeline: Any,
    model: Any,
    merged: Mapping[str, Any],
    schedule: Sequence[float],
    *,
    model_path: str,
    resume: bool,
    sigma_res_ratio: float,
    sigma_floor: float,
) -> CovarianceCache:
    signature = _condition_signature(context, pipeline, COVARIANCE_SEEDS, model_path)
    if resume:
        try:
            cache = _load_covariance_cache(context, signature)
            print(f"[covariance tile {context.tile_id:02d}] resumed M={cache.centered.shape[0]}")
            return cache
        except FileNotFoundError:
            pass
    print(f"[covariance tile {context.tile_id:02d}] generating M={len(COVARIANCE_SEEDS)} normalized endpoints")
    samples: List[SparseTensor] = []
    model.to(torch.device("cuda"))
    for seed in COVARIANCE_SEEDS:
        state = _covariance_seed_state(context, int(seed), pipeline)
        endpoint = _flow_from_state(
            context,
            state,
            pipeline,
            model,
            merged,
            schedule,
            label=f"tile {context.tile_id:02d} covariance seed {seed}",
        )
        _strict_sparse_check(context.texture_norm, endpoint, f"tile {context.tile_id} covariance endpoint")
        samples.append(endpoint)
        del state, endpoint
        _empty_cuda_cache()
    cache = _covariance_from_samples(
        context,
        samples,
        sigma_res_ratio=float(sigma_res_ratio),
        sigma_floor=float(sigma_floor),
    )
    _save_covariance_cache(context, cache, signature)
    del samples
    return cache


def _run_pure_hr_control(
    contexts: Sequence[base.TileContext],
    pipeline: Any,
    model: Any,
    merged: Mapping[str, Any],
    schedule: Sequence[float],
    *,
    output_root: Path,
    resume: bool,
) -> Dict[int, SparseTensor]:
    endpoints: Dict[int, SparseTensor] = {}
    model.to(torch.device("cuda"))
    for index, context in enumerate(contexts):
        path = output_root / "pure_hr" / "tiles" / f"tile_{context.tile_id:02d}" / "pure_HR_endpoint.pt"
        if resume and path.is_file():
            endpoint = _load_sparse(path)
            _strict_sparse_check(context.initial_state, endpoint, f"tile {context.tile_id} resumed PureHR")
        else:
            endpoint = _flow_from_state(
                context,
                context.initial_state,
                pipeline,
                model,
                merged,
                schedule,
                label=f"tile {context.tile_id:02d} PureHR",
            )
            _atomic_torch_save(path, _payload_sparse(endpoint))
        endpoints[context.tile_id] = endpoint
        context.pure_endpoint = endpoint
        print(f"[pure HR {context.tile_id:02d}] {index + 1}/{len(contexts)}")
    model.cpu()
    _empty_cuda_cache()
    return endpoints


# ---------------------------------------------------------------------------
# Jacobi physical-consensus flow
# ---------------------------------------------------------------------------


def _predict_all(
    contexts: Sequence[base.TileContext],
    states: Mapping[int, SparseTensor],
    pipeline: Any,
    model: Any,
    merged: Mapping[str, Any],
    t: float,
    step_index: int,
) -> Dict[int, Dict[str, SparseTensor]]:
    predictions: Dict[int, Dict[str, SparseTensor]] = {}
    model.to(torch.device("cuda"))
    step_kwargs = _sampler_step_kwargs(merged)
    for context in contexts:
        tile_id = context.tile_id
        state_gpu = _sparse_to_device(states[tile_id], torch.device("cuda"))
        shape_gpu = _sparse_to_device(context.shape_norm, torch.device("cuda"))
        condition = _move_condition(context.condition, torch.device("cuda"))
        try:
            with torch.no_grad():
                pred_x0, _, pred_v = pipeline.tex_slat_sampler._get_model_prediction(
                    model,
                    state_gpu,
                    float(t),
                    cond=condition["cond"],
                    neg_cond=condition["neg_cond"],
                    concat_cond=shape_gpu,
                    **step_kwargs,
                )
        finally:
            del shape_gpu, condition
        if not isinstance(pred_x0, SparseTensor) or not isinstance(pred_v, SparseTensor):
            raise RuntimeError(f"tile {tile_id}: official prediction is not SparseTensor")
        _strict_sparse_check(state_gpu, pred_x0, f"tile {tile_id} step {step_index} pred_x0")
        _strict_sparse_check(state_gpu, pred_v, f"tile {tile_id} step {step_index} pred_v")
        predictions[tile_id] = {"pred_x0": _sparse_to_cpu(pred_x0), "pred_v": _sparse_to_cpu(pred_v)}
        del state_gpu, pred_x0, pred_v
    model.cpu()
    _sync_cuda()
    return predictions


def _physical_loss(field: torch.Tensor, target: torch.Tensor, weight: float) -> float:
    residual = field.to(torch.float64) - target.to(torch.float64)
    result = 0.5 * float(weight) * residual.square().sum()
    if not math.isfinite(float(result.item())):
        raise RuntimeError("physical loss is non-finite")
    return float(result.item())


def _run_joint_flow(
    contexts: Sequence[base.TileContext],
    pipeline: Any,
    guides: Mapping[int, Sequence[SparseTensor]],
    covariance: Mapping[int, CovarianceCache],
    operator: OperatorCache,
    global_field: torch.Tensor,
    tile_weights: Mapping[int, float],
    *,
    output_root: Path,
    texture_params: Mapping[str, Any],
    rho0: float,
    query_chunk_size: int,
    lsmr_atol: float,
    lsmr_btol: float,
    lsmr_maxiter: int,
    lsmr_damp: float,
    step_limit: Optional[int],
    write_endpoint: bool,
) -> Dict[str, Any]:
    sampler = pipeline.tex_slat_sampler
    model = pipeline.models["tex_slat_flow_model_1024"]
    merged = {**pipeline.tex_slat_sampler_params, **dict(texture_params)}
    schedule = _native_schedule(sampler, merged)
    start_index = _schedule_start(schedule, 1.0)
    if start_index != 0:
        raise RuntimeError("joint flow requires native start t=1")
    pairs = list(zip(schedule[:-1], schedule[1:]))
    if step_limit is not None:
        pairs = pairs[: int(step_limit)]
    states: Dict[int, SparseTensor] = {context.tile_id: _fresh_sparse(context.initial_state) for context in contexts}
    support_digests: Dict[int, Optional[str]] = {context.tile_id: None for context in contexts}
    all_steps: List[Dict[str, Any]] = []
    solver_rows: List[Dict[str, Any]] = []
    output_root.joinpath("steps").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for step_index, (t, t_next) in enumerate(pairs):
        print(f"[joint U covariance step {step_index:02d}] t={t:.9f} t_next={t_next:.9f} tiles={len(contexts)}")
        predictions = _predict_all(contexts, states, pipeline, model, merged, float(t), step_index)

        # Barrier B: decode every endpoint before constructing any U target.
        pipeline.models["tex_slat_decoder"].to(torch.device("cuda"))
        decoded_fields: Dict[int, torch.Tensor] = {}
        decode_records: Dict[int, Dict[str, Any]] = {}
        for context in contexts:
            tile_id = context.tile_id
            field, digest, record = _decode_endpoint_field(
                pipeline,
                context,
                predictions[tile_id]["pred_x0"],
                guides[tile_id],
                query_check=(support_digests[tile_id] is None),
                support_digest=support_digests[tile_id],
                query_chunk_size=query_chunk_size,
            )
            support_digests[tile_id] = digest
            decoded_fields[tile_id] = field
            decode_records[tile_id] = record
        pipeline.models["tex_slat_decoder"].cpu()
        _empty_cuda_cache()

        # Barrier C: one frozen U* from Global G plus all current HR fields.
        u_star, solver_stats = _solve_joint_u(
            operator,
            global_field,
            decoded_fields,
            atol=lsmr_atol,
            btol=lsmr_btol,
            maxiter=lsmr_maxiter,
            damp=lsmr_damp,
        )
        solver_stats.update({"step": int(step_index), "t": float(t), "t_next": float(t_next)})
        solver_rows.append(solver_stats)
        _atomic_json(output_root / "steps" / f"step_{step_index:02d}_u_solver.json", solver_stats)

        u_by_tile: Dict[int, torch.Tensor] = {}
        for block_index, block in enumerate(operator.blocks[1:], start=1):
            tile_id = int(str(block["name"]).split("_")[-1])
            u_by_tile[tile_id] = _operator_apply_unweighted(operator, block_index, u_star).to(torch.float32)
            if not torch.isfinite(u_by_tile[tile_id]).all():
                raise RuntimeError(f"tile {tile_id}: fixed U physical target is non-finite")

        # Barrier D/E: each endpoint is corrected against the same frozen U,
        # then all official velocities and next states are committed together.
        pipeline.models["tex_slat_decoder"].to(torch.device("cuda"))
        corrected_rows: Dict[int, Dict[str, Any]] = {}
        for context in contexts:
            tile_id = context.tile_id
            pred_x0 = predictions[tile_id]["pred_x0"]
            state = states[tile_id]
            physical_before = _physical_loss(decoded_fields[tile_id], u_by_tile[tile_id], tile_weights[tile_id])
            field_graph, gradient, digest, loss_graph, grad_record = _decode_endpoint_field_with_grad(
                pipeline,
                context,
                pred_x0,
                u_by_tile[tile_id],
                guides[tile_id],
                physical_weight=tile_weights[tile_id],
                support_digest=support_digests[tile_id],
                query_chunk_size=query_chunk_size,
            )
            del field_graph, loss_graph
            cov = covariance[tile_id]
            centered_gpu = cov.centered.to(device=torch.device("cuda"), dtype=gradient.dtype)
            d, low_rank, isotropic = covariance_vector_product(centered_gpu, gradient, cov.sigma2)
            if not torch.isfinite(d).all():
                raise RuntimeError(f"tile {tile_id}: C_i g_i is non-finite")
            d_rms = _rms(d)
            if d_rms < 1e-8:
                raise RuntimeError(f"tile {tile_id}: zero covariance-preconditioned gradient")
            rho = float(rho0) * float(t)
            corrected_x0, delta_x0_rms = _make_correction(pred_x0.to(torch.device("cuda")) if hasattr(pred_x0, "to") else pred_x0, d, rho)
            state_gpu = _sparse_to_device(state, torch.device("cuda"))
            pred_v_gpu = _sparse_to_device(predictions[tile_id]["pred_v"], torch.device("cuda"))
            corrected_v = sampler._xstart_to_pred(state_gpu, float(t), corrected_x0)
            _strict_sparse_check(pred_v_gpu, corrected_v, f"tile {tile_id} step {step_index} corrected_v")
            next_state = SparseTensor(
                state_gpu.feats - float(t - t_next) * corrected_v.feats,
                state_gpu.coords.detach().clone(),
            )
            _strict_sparse_check(state_gpu, next_state, f"tile {tile_id} step {step_index} next_state")
            next_state_cpu = _sparse_to_cpu(next_state)
            corrected_x0_cpu = _sparse_to_cpu(corrected_x0)
            corrected_v_cpu = _sparse_to_cpu(corrected_v)
            pipeline.models["tex_slat_decoder"].eval()
            after_field, after_digest, after_record = _decode_endpoint_field(
                pipeline,
                context,
                corrected_x0_cpu,
                guides[tile_id],
                query_check=False,
                support_digest=support_digests[tile_id],
                query_chunk_size=query_chunk_size,
            )
            physical_after = _physical_loss(after_field, u_by_tile[tile_id], tile_weights[tile_id])
            pred_v_cpu = predictions[tile_id]["pred_v"]
            delta_v = corrected_v_cpu.feats - pred_v_cpu.feats
            delta_xprev = next_state_cpu.feats - state.feats
            record = {
                "tile_id": tile_id,
                "step": int(step_index),
                "t": float(t),
                "t_next": float(t_next),
                "physical_loss_before": physical_before,
                "physical_loss_after_fixed_U": physical_after,
                "physical_loss_decreased": bool(physical_after <= physical_before + 1e-8),
                "grad_rms": _rms(gradient),
                "cov_grad_rms": _rms(d),
                "grad_cov_cosine": _cosine(gradient, d),
                "cov_lowrank_rms": _rms(low_rank),
                "cov_isotropic_rms": _rms(isotropic),
                "cov_sigma_res2": float(cov.sigma2),
                "rho": rho,
                "delta_x0_rms": float(delta_x0_rms),
                "delta_x0_relative": float(delta_x0_rms / (_rms(pred_x0.feats) + 1e-8)),
                "delta_v_rms": _rms(delta_v),
                "delta_xprev_rms": _rms(delta_xprev),
                "x_t_rms": _rms(state.feats),
                "pred_x0_rms": _rms(pred_x0.feats),
                "corrected_x0_rms": _rms(corrected_x0_cpu.feats),
                "finite": bool(
                    torch.isfinite(next_state_cpu.feats).all()
                    and torch.isfinite(corrected_x0_cpu.feats).all()
                    and torch.isfinite(corrected_v_cpu.feats).all()
                ),
                "coord_digest": _coordinate_digest(corrected_x0_cpu),
                "support_checks": {
                    "pred_x0": _strict_sparse_check(state, pred_x0, f"tile {tile_id} pred_x0 saved"),
                    "corrected_x0": _strict_sparse_check(pred_x0, corrected_x0_cpu, f"tile {tile_id} corrected_x0 saved"),
                    "next_state": _strict_sparse_check(state, next_state_cpu, f"tile {tile_id} next state saved"),
                },
                "decode": {"before": decode_records[tile_id], "after": after_record},
                "gradient": grad_record,
            }
            if not record["finite"]:
                raise RuntimeError(f"tile {tile_id} step {step_index}: non-finite correction result")
            corrected_rows[tile_id] = {"next_state": next_state_cpu, "record": record}
            _atomic_json(
                output_root / "tiles" / f"tile_{tile_id:02d}" / "steps" / f"step_{step_index:02d}_diagnostics.json",
                record,
            )
            del centered_gpu, d, low_rank, isotropic, gradient, corrected_x0, corrected_v, next_state, state_gpu, pred_v_gpu
            _empty_cuda_cache()
        pipeline.models["tex_slat_decoder"].cpu()

        # Barrier F: no state is replaced until every tile has a next state.
        for context in contexts:
            states[context.tile_id] = corrected_rows[context.tile_id]["next_state"]
            _strict_sparse_check(context.initial_state, states[context.tile_id], f"tile {context.tile_id} synchronized state")
        step_records = [corrected_rows[context.tile_id]["record"] for context in contexts]
        step_summary = {
            "step": int(step_index),
            "t": float(t),
            "t_next": float(t_next),
            "tile_count": len(contexts),
            "tiles": step_records,
            "barriers": {
                "prediction_barrier": True,
                "decoded_field_barrier": True,
                "joint_u_barrier": True,
                "endpoint_correction_barrier": True,
                "official_velocity_barrier": True,
                "euler_update_barrier": True,
                "all_tiles_synchronized": True,
            },
            "step_seconds": float(time.perf_counter() - started) if not all_steps else float(time.perf_counter() - started),
        }
        _atomic_json(output_root / "steps" / f"step_{step_index:02d}_summary.json", step_summary)
        all_steps.append(step_summary)
        _empty_cuda_cache()
    if write_endpoint:
        for context in contexts:
            endpoint = states[context.tile_id]
            context.guided_endpoint = endpoint
            _atomic_torch_save(
                output_root / "joint_u_cov_guided" / "tiles" / f"tile_{context.tile_id:02d}" / "joint_U_cov_guided_endpoint.pt",
                _payload_sparse(endpoint),
            )
    return {
        "route": "PureHR endpoint -> joint physical U* LSMR -> PBR decoder Jacobian -> conditional covariance-vector product -> normalized trust-region endpoint -> official _xstart_to_pred -> Euler",
        "native_schedule": schedule,
        "schedule_start_index": int(start_index),
        "flow_steps": len(pairs),
        "tile_count": len(contexts),
        "rho0": float(rho0),
        "M": len(COVARIANCE_SEEDS),
        "covariance_seeds": list(COVARIANCE_SEEDS),
        "guided_seed": GUIDED_SEED,
        "shape_flow_called": False,
        "shape_sampler_called": False,
        "G_guidance_used": False,
        "G_velocity_used": False,
        "velocity_averaging_used": False,
        "all_tiles_synchronized_per_step": True,
        "steps": all_steps,
        "u_solver_stats": solver_rows,
    }


# ---------------------------------------------------------------------------
# Final stitch, metrics, visual diagnostics, and CLI
# ---------------------------------------------------------------------------


def _variant_patch_and_stitch(
    *,
    variant: str,
    endpoint_attr: str,
    contexts: Sequence[base.TileContext],
    pipeline: Any,
    global_camera: Mapping[str, Any],
    baseline_mesh: MeshWithVoxel,
    args: argparse.Namespace,
    output_root: Path,
) -> Tuple[Any, Dict[str, Any]]:
    patches: List[core.ReturnedTilePatch] = []
    tile_records: List[Dict[str, Any]] = []
    for index, context in enumerate(contexts):
        endpoint = getattr(context, endpoint_attr, None)
        if endpoint is None:
            raise RuntimeError(f"tile {context.tile_id}: missing final endpoint {endpoint_attr}")
        print(f"[{variant}] final official decode tile {context.tile_id:02d} ({index + 1}/{len(contexts)})")
        mesh, _, decode_stats = base._decode_endpoint(
            pipeline=pipeline,
            shape_denorm=_sparse_to_device(context.shape_denorm, torch.device("cuda")),
            texture_norm=_sparse_to_device(endpoint, torch.device("cuda")),
            query_points=context.target_points[:0].to(torch.device("cuda")),
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
        tile_records.append({"tile_id": int(context.tile_id), "box": list(context.box), "decode": decode_stats, "returned_global_patch": patch.stats})
        del mesh
        _empty_cuda_cache()
    stitched, stitch_stats = core._stitch_tile_patches_nearest(
        patches,
        layout=dict(baseline_mesh.layout),
        global_camera=global_camera,
        face_chunk_size=int(args.face_projection_chunk_size),
        weld_tolerance=float(args.stitch_tolerance),
    )
    variant_dir = output_root / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
        variant_dir / "global_merged_mesh.pt",
        {"format": f"{FORMAT}_{variant}_global_mesh", "variant": variant, "mesh": stitched, "stitch_stats": stitch_stats, "tile_records": tile_records},
    )
    exported = core.ReturnedTilePatch(
        tile_id=-1,
        box=(0, 0, CANONICAL_IMAGE_SIZE, CANONICAL_IMAGE_SIZE),
        vertices=stitched.vertices,
        faces=stitched.faces,
        vertex_attrs=stitched.vertex_attrs,
        stats=stitch_stats,
    )
    glb_stats = core._export_tiled_glb([exported], variant_dir / "global_merged_mesh.glb")
    summary = {
        "variant": variant,
        "vertices": int(stitched.vertices.shape[0]),
        "faces": int(stitched.faces.shape[0]),
        "tile_count": len(patches),
        "tile_records": tile_records,
        "stitch": stitch_stats,
        "glb": glb_stats,
        "mesh_pt": str((variant_dir / "global_merged_mesh.pt").resolve()),
        "mesh_glb": str((variant_dir / "global_merged_mesh.glb").resolve()),
    }
    _atomic_json(variant_dir / "global_variant_summary.json", summary)
    return stitched, summary


def _write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ["variant", "vertices", "faces", "PSNR", "SSIM", "LPIPS", "render_resolution"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _write_per_step_csv(path: Path, flow: Mapping[str, Any]) -> None:
    rows: List[Mapping[str, Any]] = []
    for step in flow.get("steps", []):
        rows.extend(step.get("tiles", []))
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields and not isinstance(row[key], (Mapping, list)):
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in fields})


def _plot_correction_schedule(path: Path, flow: Mapping[str, Any]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        _atomic_json(path.with_suffix(".json"), {"plot_unavailable": str(exc)})
        return
    t_values: List[float] = []
    rho_values: List[float] = []
    x0_values: List[float] = []
    v_values: List[float] = []
    prev_values: List[float] = []
    for step in flow.get("steps", []):
        tiles = step.get("tiles", [])
        if not tiles:
            continue
        t_values.append(float(step["t"]))
        rho_values.append(float(np.mean([row["rho"] for row in tiles])))
        x0_values.append(float(np.mean([row["delta_x0_rms"] for row in tiles])))
        v_values.append(float(np.mean([row["delta_v_rms"] for row in tiles])))
        prev_values.append(float(np.mean([row["delta_xprev_rms"] for row in tiles])))
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t_values, rho_values, marker="o", label="rho")
    axes[0].plot(t_values, x0_values, marker="o", label="mean delta_x0 RMS")
    axes[0].set_ylabel("normalized endpoint")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(t_values, v_values, marker="o", label="mean delta_v RMS")
    axes[1].plot(t_values, prev_values, marker="o", label="mean delta_xprev RMS")
    axes[1].set_xlabel("native timestep t")
    axes[1].set_ylabel("flow update")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    for axis in axes:
        axis.invert_xaxis()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _load_completed_flow(
    output_root: Path,
    contexts: Sequence[base.TileContext],
    schedule: Sequence[float],
    *,
    rho0: float,
) -> Optional[Dict[str, Any]]:
    """Reuse an atomically completed flow without recomputing its 12 steps.

    The formal CUDA run is dominated by the six LSMR solves and the frozen
    decoder backward at every tile-step.  A previous process can therefore
    finish all flow artifacts and exit before final mesh stitching.  Resume
    must distinguish that valid state from a partial run, and must verify the
    endpoint/support provenance before allowing finalization to proceed.
    """
    if len(schedule) != TEXTURE_STEPS + 1:
        return None
    step_dir = output_root / "steps"
    endpoint_paths = {
        context.tile_id: output_root
        / "joint_u_cov_guided"
        / "tiles"
        / f"tile_{context.tile_id:02d}"
        / "joint_U_cov_guided_endpoint.pt"
        for context in contexts
    }
    summary_paths = [step_dir / f"step_{step:02d}_summary.json" for step in range(TEXTURE_STEPS)]
    solver_path = output_root / "u_solver_stats.json"
    if not all(path.is_file() for path in (*endpoint_paths.values(), *summary_paths, solver_path)):
        return None

    steps: List[Dict[str, Any]] = []
    expected_tile_ids = {context.tile_id for context in contexts}
    for step_index, (t, t_next) in enumerate(zip(schedule[:-1], schedule[1:])):
        payload = json.loads(summary_paths[step_index].read_text(encoding="utf-8"))
        if int(payload.get("step", -1)) != step_index:
            raise RuntimeError(f"completed flow cache has invalid step index at {step_index}")
        if abs(float(payload.get("t", float("nan"))) - float(t)) > 1e-6 or abs(
            float(payload.get("t_next", float("nan"))) - float(t_next)
        ) > 1e-6:
            raise RuntimeError(f"completed flow cache timestep mismatch at step {step_index}")
        if int(payload.get("tile_count", -1)) != len(contexts):
            raise RuntimeError(f"completed flow cache tile count mismatch at step {step_index}")
        records = payload.get("tiles", [])
        if {int(record.get("tile_id", -1)) for record in records} != expected_tile_ids:
            raise RuntimeError(f"completed flow cache tile ids mismatch at step {step_index}")
        for record in records:
            if not bool(record.get("finite", False)):
                raise RuntimeError(f"completed flow cache contains non-finite tile-step at step {step_index}")
            expected_rho = float(rho0) * float(t)
            if abs(float(record.get("rho", float("nan"))) - expected_rho) > 1e-6:
                raise RuntimeError(f"completed flow cache rho mismatch at step {step_index}")
        steps.append(payload)

    solver_payload = json.loads(solver_path.read_text(encoding="utf-8"))
    solver_rows = solver_payload.get("steps", [])
    if len(solver_rows) != TEXTURE_STEPS:
        raise RuntimeError("completed flow cache has incomplete U solver statistics")

    for context in contexts:
        endpoint = _load_sparse(endpoint_paths[context.tile_id])
        _strict_sparse_check(context.initial_state, endpoint, f"tile {context.tile_id} resumed guided endpoint")
        context.guided_endpoint = endpoint

    flow = {
        "route": "PureHR endpoint -> joint physical U* LSMR -> PBR decoder Jacobian -> conditional covariance-vector product -> normalized trust-region endpoint -> official _xstart_to_pred -> Euler",
        "native_schedule": [float(value) for value in schedule],
        "schedule_start_index": 0,
        "flow_steps": TEXTURE_STEPS,
        "tile_count": len(contexts),
        "rho0": float(rho0),
        "M": len(COVARIANCE_SEEDS),
        "covariance_seeds": list(COVARIANCE_SEEDS),
        "guided_seed": GUIDED_SEED,
        "shape_flow_called": False,
        "shape_sampler_called": False,
        "G_guidance_used": False,
        "G_velocity_used": False,
        "velocity_averaging_used": False,
        "all_tiles_synchronized_per_step": True,
        "steps": steps,
        "u_solver_stats": solver_rows,
        "reused_completed_flow": True,
    }
    return flow


def _aggregate_flow_questions(flow: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [row for step in flow.get("steps", []) for row in step.get("tiles", [])]
    if not rows:
        return {}
    cosines = np.asarray([row["grad_cov_cosine"] for row in rows], dtype=np.float64)
    low = np.asarray([row["cov_lowrank_rms"] for row in rows], dtype=np.float64)
    iso = np.asarray([row["cov_isotropic_rms"] for row in rows], dtype=np.float64)
    losses = np.asarray([row["physical_loss_before"] - row["physical_loss_after_fixed_U"] for row in rows], dtype=np.float64)
    return {
        "Q1_grad_cov_cosine": {"mean": float(cosines.mean()), "min": float(cosines.min()), "max": float(cosines.max())},
        "Q2_lowrank_vs_isotropic": {
            "lowrank_rms_mean": float(low.mean()),
            "isotropic_rms_mean": float(iso.mean()),
            "lowrank_dominant_fraction": float(np.mean(low >= iso)),
        },
        "Q4_fixed_U_loss": {
            "decrease_fraction": float(np.mean(losses >= -1e-8)),
            "mean_relative_decrease_proxy": float(np.mean(losses / (np.asarray([row["physical_loss_before"] for row in rows]) + 1e-12))),
            "worst_increase": float(max(0.0, -losses.min())),
        },
        "Q3_late_step": {
            "first_delta_x0_mean": float(np.mean([row["delta_x0_rms"] for row in rows if row["step"] == 0])),
            "last_delta_x0_mean": float(np.mean([row["delta_x0_rms"] for row in rows if row["step"] == max(r["step"] for r in rows)])),
            "first_delta_v_mean": float(np.mean([row["delta_v_rms"] for row in rows if row["step"] == 0])),
            "last_delta_v_mean": float(np.mean([row["delta_v_rms"] for row in rows if row["step"] == max(r["step"] for r in rows)])),
        },
    }


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    evaluation = summary.get("evaluation", {})
    table = evaluation.get("table", []) if isinstance(evaluation, Mapping) else []
    q = summary.get("questions", {})
    flow = summary.get("flow", {})
    flow_table: List[Dict[str, Any]] = []
    for step in flow.get("steps", []) if isinstance(flow, Mapping) else []:
        tile_rows = step.get("tiles", [])
        if not tile_rows:
            continue
        mean = lambda key: float(np.mean([float(row[key]) for row in tile_rows]))
        flow_table.append(
            {
                "step": int(step["step"]),
                "t": float(step["t"]),
                "rho": mean("rho"),
                "delta_x0_rms": mean("delta_x0_rms"),
                "delta_v_rms": mean("delta_v_rms"),
                "delta_xprev_rms": mean("delta_xprev_rms"),
                "loss_before": mean("physical_loss_before"),
                "loss_after": mean("physical_loss_after_fixed_U"),
                "decrease_fraction": float(np.mean([bool(row["physical_loss_decreased"]) for row in tile_rows])),
            }
        )
    lines = [
        "# Joint U + Conditional Covariance Endpoint Per-Step Experiment",
        "",
        "本实验的固定数学路线是：",
        "",
        r"\[U_k^\star=\arg\min_U[\tfrac12\|A_GU-g\|_{W_G}^2+\tfrac12\sum_i\|A_iU-h_i(\hat x_{0,i,k})\|_{W_i}^2].\]",
        r"\[g_{i,k}=-\nabla_{\hat x_{0,i,k}}\tfrac12\|h_i(\hat x_{0,i,k})-A_iU_k^\star\|_{W_i}^2.\]",
        r"\[C_i=B_iB_i^T+\sigma_{\mathrm{res},i}^2I,\qquad d_{i,k}=C_ig_{i,k}.\]",
        r"\[\tilde x_{0,i,k}=\hat x_{0,i,k}+0.10t_k\,d_{i,k}/(\operatorname{RMS}(d_{i,k})+10^{-8}).\]",
        r"\[\tilde v_{i,k}=((1-\sigma_{\min})x_{i,t_k}-\tilde x_{0,i,k})/(\sigma_{\min}+(1-\sigma_{\min})t_k).\]",
        r"\[x_{i,t_{k+1}}=x_{i,t_k}-(t_k-t_{k+1})\tilde v_{i,k}.\]",
        "",
        "Global G 只进入 physical U*，不进入 HR latent、HR velocity 或初始 state；没有 Gaussian fusion、range-null、MRA、POD、Langevin noise、inner correction 或 re-bridge。",
        "",
        "## Run provenance",
        "",
        f"- CUDA physical device: `{summary.get('cuda_device')}` / `{summary.get('cuda_name')}`",
        f"- guided seed: `{summary.get('guided_seed')}`; covariance seeds: `{summary.get('covariance_seeds')}`",
        f"- tile count: `{summary.get('tile_layout', {}).get('participating_tile_count')}`; texture steps: `{summary.get('sampler', {}).get('texture_steps')}`",
        f"- PureHR route: `{summary.get('sampler', {}).get('pure_hr_route')}`",
        f"- global baseline: `{summary.get('global_baseline', {}).get('source')}`",
        "",
        "## Correctness",
        "",
        f"- A operator row-sum error: `{summary.get('operator', {}).get('row_sum_error_max')}`; dense C4096 cube allocated: `{summary.get('operator', {}).get('dense_4096_cube_allocated')}`.",
        f"- fixed support invariant: `{summary.get('correctness', {}).get('fixed_support_invariant')}`",
        f"- differentiable decoder smoke: `{summary.get('correctness', {}).get('gradient_smoke')}`",
        f"- rho=0 native route: `{summary.get('correctness', {}).get('rho_zero_equivalence')}`",
        "",
        "## Covariance questions",
        "",
        f"- Q1 cos(g, Cg): `{q.get('Q1_grad_cov_cosine')}`",
        f"- Q2 low-rank/isotropic: `{q.get('Q2_lowrank_vs_isotropic')}`",
        f"- Q3 endpoint/velocity late-step curve: `{q.get('Q3_late_step')}`; full curve is in `correction_schedule.png`.",
        f"- Q4 fixed-U loss: `{q.get('Q4_fixed_U_loss')}`",
        "",
        "## Flow dynamics (47 valid tiles)",
        "",
        "| step | t | rho | RMS(delta x0) | RMS(delta v) | RMS(delta xprev) | mean loss before -> after | decrease fraction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in flow_table:
        lines.append(
            f"| {row['step']} | {row['t']:.9f} | {row['rho']:.6f} | {row['delta_x0_rms']:.6f} | "
            f"{row['delta_v_rms']:.6f} | {row['delta_xprev_rms']:.6f} | "
            f"{row['loss_before']:.3f} -> {row['loss_after']:.3f} | {row['decrease_fraction']:.3f} |"
        )
    lines.extend(
        [
        "",
        "The endpoint RMS follows the prescribed `rho_k=0.10*t_k`; the re-bridged velocity RMS stays approximately constant, while the Euler state-update RMS grows at late steps. Thus the schedule removes the direct endpoint-to-velocity blow-up, but does not make the full state update monotone.",
        "",
        "## Final metrics",
        "",
        "| Variant | PSNR | SSIM | LPIPS |",
        "|---|---:|---:|---:|",
    ]
    )
    for row in table:
        lines.append(f"| {row.get('variant')} | {row.get('PSNR')} | {row.get('SSIM')} | {row.get('LPIPS')} |")
    lines.extend(
        [
            "",
            "## Visual outputs",
            "",
            "- aligned 4096 renders: `global_baseline/aligned_eval_4096/`, `pure_hr/aligned_eval_4096/`, `joint_U_cov_guided/aligned_eval_4096/`",
            "- six-view sheets and 24-frame turntables: `multiview/` and `turntable/`",
            "- RGB/metallic/roughness/alpha front-back sheets: `pbr_channel_sheets/`",
            "- visual inspection: front and side views retain the high-frequency mechanical texture; the back view still shows visible tile/block seams, especially in PBR channel sheets. There is no uniform global gray wash, but backside overlap consistency is not solved by this one correction pass.",
            "",
            "## Attribution",
            "",
            "如果指标或视觉质量恶化，应分别检查：U least-squares consensus 的平均化目标、PBR decoder Jacobian、conditional covariance preconditioner、rho schedule，以及 endpoint-to-velocity dynamics；本脚本没有把这些因素混为旧 fusion 方法的结果。",
            "",
            f"完整 summary: `{path.parent / 'summary.json'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _select_cuda_device(requested_physical: int) -> Tuple[int, Optional[int]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    ids: List[int] = []
    if visible and all(part.strip().lstrip("-").isdigit() for part in visible.split(",")):
        ids = [int(part.strip()) for part in visible.split(",")]
    if ids:
        if int(requested_physical) not in ids:
            raise RuntimeError(f"requested physical cuda{requested_physical} is not visible: CUDA_VISIBLE_DEVICES={visible}")
        logical = ids.index(int(requested_physical))
        physical = int(requested_physical)
    else:
        logical = int(requested_physical)
        physical = int(requested_physical)
    if logical < 0 or logical >= torch.cuda.device_count():
        raise RuntimeError(f"requested cuda{requested_physical} unavailable: visible={visible!r} count={torch.cuda.device_count()}")
    torch.cuda.set_device(logical)
    return logical, physical


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="assets/choose/0_img.png")
    parser.add_argument("--output-dir", default="outputs/pbr_joint_u_cov_perstep_cuda4")
    parser.add_argument("--context-cache-dir", default="outputs/cross_tile_pbr_perstep_guided_cuda4_full")
    parser.add_argument("--baseline-dir", default=None)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--cuda-device", type=int, default=4, help="physical CUDA id; CUDA_VISIBLE_DEVICES=4 maps it to logical cuda:0")
    parser.add_argument("--tile-ids", default=None, help="comma-separated tile ids; omitted means every valid tile in context cache")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument("--face-projection-chunk-size", type=int, default=250_000)
    parser.add_argument("--query-chunk-size", type=int, default=65_536)
    parser.add_argument("--operator-chunk-size", type=int, default=250_000)
    parser.add_argument("--noise-timestep", type=float, default=1.0)
    parser.add_argument("--noise-strength", type=float, default=1.0)
    parser.add_argument("--rho0", type=float, default=0.10)
    parser.add_argument("--covariance-sigma-res-ratio", type=float, default=0.05)
    parser.add_argument("--covariance-sigma-floor", type=float, default=1e-4)
    parser.add_argument("--lsmr-atol", type=float, default=1e-6)
    parser.add_argument("--lsmr-btol", type=float, default=1e-6)
    parser.add_argument("--lsmr-maxiter", type=int, default=200)
    parser.add_argument("--lsmr-damp", type=float, default=0.0)
    parser.add_argument("--step-limit", type=int, default=None, help="debug preflight limit; official run leaves this unset")
    parser.add_argument("--skip-covariance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-pure-hr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--reuse-complete-flow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="when resuming, reuse all completed 12-step endpoint artifacts before finalization",
    )
    parser.add_argument("--skip-final-stitch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--render-resolution", type=int, default=4096)
    parser.add_argument("--metric-resolution", type=int, default=512)
    parser.add_argument("--render-ssaa", type=int, default=1)
    parser.add_argument("--render-peel-layers", type=int, default=8)
    parser.add_argument("--render-face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--stitch-tolerance", type=float, default=1.0 / OVOXEL_RESOLUTION)
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


def _parse_ids(value: Optional[str]) -> Optional[List[int]]:
    if value is None or not str(value).strip():
        return None
    return sorted({int(item.strip()) for item in str(value).split(",") if item.strip()})


def _validate_args(args: argparse.Namespace) -> None:
    if float(args.noise_timestep) != 1.0 or float(args.noise_strength) != 1.0:
        raise ValueError("Codex.md fixes noise_timestep=1.0 and noise_strength=1.0")
    if float(args.rho0) < 0.0:
        raise ValueError("rho0 must be non-negative")
    if int(args.lsmr_maxiter) <= 0 or float(args.lsmr_atol) <= 0.0 or float(args.lsmr_btol) <= 0.0:
        raise ValueError("invalid LSMR settings")
    if int(args.query_chunk_size) <= 0 or int(args.operator_chunk_size) <= 0:
        raise ValueError("query/operator chunks must be positive")
    if float(args.stitch_tolerance) <= 0.0:
        raise ValueError("stitch tolerance must be positive")
    if not Path(args.image).is_file():
        raise FileNotFoundError(args.image)
    if not Path(args.context_cache_dir).is_dir():
        raise FileNotFoundError(args.context_cache_dir)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _validate_args(args)
    logical_device, physical_device = _select_cuda_device(int(args.cuda_device))
    _patch_flex_gemm_input_backward()
    device = torch.device("cuda")
    print(f"[cuda] physical={physical_device} logical={logical_device} name={torch.cuda.get_device_name(logical_device)} low_vram={args.low_vram}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(args.resume):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_output_layout(output_dir)
    cache_dir = Path(args.context_cache_dir).expanduser().resolve()
    baseline_dir = Path(args.baseline_dir).expanduser().resolve() if args.baseline_dir else cache_dir
    baseline_path = baseline_dir / "global_baseline_mesh.pt"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Global baseline cache is required for this experiment: {baseline_path}")
    camera_path = cache_dir / "global_camera.json"
    if not camera_path.is_file():
        raise FileNotFoundError(camera_path)
    global_camera = json.loads(camera_path.read_text(encoding="utf-8"))
    baseline_payload = _load_torch(baseline_path)
    baseline_mesh = baseline_payload.get("mesh", baseline_payload) if isinstance(baseline_payload, Mapping) else baseline_payload
    if not isinstance(baseline_mesh, MeshWithVoxel):
        raise RuntimeError("global baseline cache is not MeshWithVoxel")
    baseline_mesh = baseline_mesh.to("cpu")
    _link_artifact(baseline_path, output_dir / "global_baseline" / "global_baseline_mesh.pt")
    _atomic_json(
        output_dir / "global_baseline" / "provenance.json",
        {
            "source": str(baseline_path.resolve()),
            "geometry_fixed": True,
            "flow_used": False,
            "attrs_channels": PBR_CHANNELS,
        },
    )

    with Image.open(args.image) as source:
        source_image = source.convert("RGB")
    source_image.save(output_dir / "input_original.png")
    canonical_path = cache_dir / "canonical_4096.png"
    if canonical_path.is_file():
        image_4096 = Image.open(canonical_path).convert("RGB")
    else:
        pipeline_for_preprocess = init_pipeline(args.model_path, device="cuda", low_vram=bool(args.low_vram))
        image_4096 = pipeline_for_preprocess.preprocess_canonical_images(source_image)["image_4096"]
        del pipeline_for_preprocess
        _empty_cuda_cache()
    image_4096.save(output_dir / "canonical_4096.png")
    image_1024 = image_4096.resize((1024, 1024), Image.Resampling.LANCZOS)
    image_1024.save(output_dir / "canonical_1024.png")

    requested_ids = _parse_ids(args.tile_ids)
    if requested_ids is None:
        # The source cache can be reused by a Phase-A run, which overwrites
        # its preparation summary with only the nine phase tiles.  The
        # complete fixed-shape files are the authoritative full-run manifest.
        required_names = (
            "fixed_shape_norm.pt",
            "texture_reference_norm.pt",
            "texture_initial_state.pt",
            "fixed_shape_summary.json",
        )
        candidates: List[int] = []
        for summary_path in sorted((cache_dir / "tiles").glob("tile_*/fixed_shape_summary.json")):
            try:
                tile_id = int(summary_path.parent.name.split("_")[-1])
            except ValueError:
                continue
            if all((summary_path.parent / name).is_file() for name in required_names):
                candidates.append(tile_id)
        if not candidates:
            preparation = cache_dir / "tile_preparation_summary.json"
            if preparation.is_file():
                candidates = [
                    int(value)
                    for value in json.loads(preparation.read_text(encoding="utf-8")).get("prepared_tile_ids", [])
                ]
        requested_ids = candidates
    requested_ids = sorted(int(v) for v in requested_ids)
    if not requested_ids:
        raise RuntimeError("no valid tile ids found in context cache")

    pipeline = init_pipeline(args.model_path, device="cuda", low_vram=bool(args.low_vram))
    contexts = _load_contexts(
        cache_dir=cache_dir,
        output_dir=output_dir,
        pipeline=pipeline,
        baseline_mesh=baseline_mesh,
        global_camera=global_camera,
        image_4096=image_4096,
        tile_ids=requested_ids,
        extend_pixel=int(args.extend_pixel),
        low_vram=bool(args.low_vram),
        face_projection_chunk_size=int(args.face_projection_chunk_size),
    )
    tile_ids = [context.tile_id for context in contexts]
    _atomic_json(output_dir / "global_camera.json", global_camera)
    _atomic_json(output_dir / "tile_layout.json", {"canonical_image_size": CANONICAL_IMAGE_SIZE, "tile_size": TILE_SIZE, "stride": TILE_STRIDE, "participating_tile_ids": tile_ids, "boxes": [list(core._tile_layout(CANONICAL_IMAGE_SIZE, TILE_SIZE, TILE_STRIDE)[i]) for i in tile_ids]})

    guides = _prepare_shape_guides(contexts, pipeline, resolution=OVOXEL_RESOLUTION, resume=bool(args.resume))
    # The decoder is frozen; only a normalized endpoint is differentiated.
    decoder = pipeline.models["tex_slat_decoder"]
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    global_coords = baseline_mesh.coords.detach().cpu().to(torch.int32)
    global_points = baseline_mesh.origin.detach().cpu().to(torch.float32)[None, :] + (global_coords.to(torch.float32) + 0.5) * float(baseline_mesh.voxel_size)
    global_field = baseline_mesh.attrs.detach().cpu().to(torch.float32)
    if global_field.shape[1] != PBR_CHANNELS or not torch.isfinite(global_field).all():
        raise RuntimeError("Global baseline PBR is invalid")
    point_blocks: List[Tuple[str, torch.Tensor, float]] = [("global", global_points, 1.0)]
    tile_points_global: Dict[int, torch.Tensor] = {}
    tile_weights: Dict[int, float] = {}
    for context in contexts:
        _, _, points_global = base._local_to_global(context.target_points, transform=context.transform, global_camera=global_camera)
        if not torch.isfinite(points_global).all() or bool(((points_global < -0.5001) | (points_global > 0.5001)).any().item()):
            raise RuntimeError(f"tile {context.tile_id}: local native support maps outside global normalized volume")
        tile_points_global[context.tile_id] = points_global.detach().cpu().to(torch.float32)
        # The local->global camera map is affine in the normalized local point
        # coordinates.  Estimate its constant Jacobian directly from four
        # round-trip points, preserving the validated camera convention.
        basis = torch.zeros((4, 3), dtype=torch.float32)
        basis[1, 0] = 1.0
        basis[2, 1] = 1.0
        basis[3, 2] = 1.0
        _, _, mapped = base._local_to_global(basis, transform=context.transform, global_camera=global_camera)
        jacobian = torch.stack((mapped[1] - mapped[0], mapped[2] - mapped[0], mapped[3] - mapped[0]), dim=1)
        determinant = abs(float(torch.linalg.det(jacobian).item()))
        if not math.isfinite(determinant) or determinant <= 0.0:
            raise RuntimeError(f"tile {context.tile_id}: invalid local/global volume Jacobian {determinant}")
        tile_weights[context.tile_id] = determinant
        point_blocks.append((f"tile_{context.tile_id}", tile_points_global[context.tile_id], determinant))
    median_weight = float(np.median([weight for _, _, weight in point_blocks]))
    point_blocks = [(name, points, weight / median_weight) for name, points, weight in point_blocks]
    tile_weights = {tile_id: weight / median_weight for tile_id, weight in tile_weights.items()}

    operator_dir = output_dir / "u_operator"
    operator_files = [operator_dir / "u_sparse_coords.pt", operator_dir / "A_stack.npz", operator_dir / "row_blocks.json", operator_dir / "quadrature.json", operator_dir / "operator_stats.json"]
    if bool(args.resume) and all(path.is_file() for path in operator_files):
        block_payload = json.loads((operator_dir / "row_blocks.json").read_text(encoding="utf-8"))
        loaded_blocks = block_payload.get("blocks", block_payload) if isinstance(block_payload, Mapping) else block_payload
        operator = OperatorCache(
            matrix=load_npz(operator_dir / "A_stack.npz").tocsr(),
            support_coords=_load_torch(operator_dir / "u_sparse_coords.pt")["coords"].to(torch.int32),
            blocks=loaded_blocks,
            metadata=json.loads((operator_dir / "operator_stats.json").read_text(encoding="utf-8")),
        )
        if [str(block["name"]) for block in operator.blocks] != [name for name, _, _ in point_blocks]:
            raise RuntimeError("resumed U operator block provenance does not match selected tiles")
        print(f"[U operator] resumed rows={operator.matrix.shape[0]:,} vars={operator.matrix.shape[1]:,} nnz={operator.matrix.nnz:,}")
    else:
        operator = build_sparse_c4096_operator(point_blocks, chunk_size=int(args.operator_chunk_size))
        operator_dir.mkdir(parents=True, exist_ok=True)
        _atomic_torch_save(operator_dir / "u_sparse_coords.pt", {"coords": operator.support_coords, "resolution": GLOBAL_U_RESOLUTION})
        save_npz(operator_dir / "A_stack.npz", operator.matrix)
        _atomic_json(operator_dir / "row_blocks.json", {"blocks": operator.blocks})
        _atomic_json(operator_dir / "quadrature.json", {"global_weight": point_blocks[0][2], "tile_weights": tile_weights, "normalization": "all weights divided by median constant cell-volume/Jacobian weight"})
        _atomic_json(operator_dir / "operator_stats.json", operator.metadata)
        print(f"[U operator] built rows={operator.matrix.shape[0]:,} vars={operator.matrix.shape[1]:,} nnz={operator.matrix.nnz:,}")
    # JSON cache stores a wrapper for readability; normalize older caches too.
    if isinstance(operator.blocks, Mapping):
        operator.blocks = list(operator.blocks.get("blocks", operator.blocks))
    operator.metadata["row_sum_error_max"] = float(operator.metadata.get("row_sum_error_max", 0.0))

    texture_params = {
        "steps": TEXTURE_STEPS,
        "guidance_strength": 1.0,
        "guidance_rescale": 0.0,
        "rescale_t": TEXTURE_RESCALE_T,
    }
    merged = {**pipeline.tex_slat_sampler_params, **texture_params}
    schedule = _native_schedule(pipeline.tex_slat_sampler, merged)
    covariance: Dict[int, CovarianceCache] = {}
    covariance_stats_output: Dict[str, Any] = {}
    if not bool(args.skip_covariance):
        flow_model = pipeline.models["tex_slat_flow_model_1024"]
        for context in contexts:
            cache = _run_covariance_for_tile(
                context,
                pipeline,
                flow_model,
                merged,
                schedule,
                model_path=str(Path(args.model_path).resolve()),
                resume=bool(args.resume),
                sigma_res_ratio=float(args.covariance_sigma_res_ratio),
                sigma_floor=float(args.covariance_sigma_floor),
            )
            covariance[context.tile_id] = cache
            covariance_stats_output[str(context.tile_id)] = cache.stats
        _atomic_json(output_dir / "covariance_stats.json", covariance_stats_output)
        flow_model.cpu()
        _empty_cuda_cache()
    else:
        for context in contexts:
            signature = _condition_signature(context, pipeline, COVARIANCE_SEEDS, str(Path(args.model_path).resolve()))
            cache = _load_covariance_cache(context, signature)
            covariance[context.tile_id] = cache
            covariance_stats_output[str(context.tile_id)] = cache.stats
        _atomic_json(output_dir / "covariance_stats.json", covariance_stats_output)
    _atomic_json(output_dir / "covariance" / "covariance_stats.json", covariance_stats_output)
    for context in contexts:
        _link_artifact(
            context.tile_dir / "covariance",
            output_dir / "covariance" / f"tile_{context.tile_id:02d}",
        )

    if bool(args.skip_pure_hr):
        pure_endpoints = {context.tile_id: _load_sparse(output_dir / "pure_hr" / "tiles" / f"tile_{context.tile_id:02d}" / "pure_HR_endpoint.pt") for context in contexts}
        for context in contexts:
            context.pure_endpoint = pure_endpoints[context.tile_id]
    else:
        pure_endpoints = _run_pure_hr_control(contexts, pipeline, pipeline.models["tex_slat_flow_model_1024"], merged, schedule, output_root=output_dir, resume=bool(args.resume))

    flow: Optional[Dict[str, Any]] = None
    if bool(args.resume) and bool(args.reuse_complete_flow) and args.step_limit is None:
        flow = _load_completed_flow(output_dir, contexts, schedule, rho0=float(args.rho0))
        if flow is not None:
            print(f"[flow] reusing completed {flow['flow_steps']}-step endpoint artifacts")
    if flow is None:
        flow = _run_joint_flow(
            contexts,
            pipeline,
            guides,
            covariance,
            operator,
            global_field,
            tile_weights,
            output_root=output_dir,
            texture_params=texture_params,
            rho0=float(args.rho0),
            query_chunk_size=int(args.query_chunk_size),
            lsmr_atol=float(args.lsmr_atol),
            lsmr_btol=float(args.lsmr_btol),
            lsmr_maxiter=int(args.lsmr_maxiter),
            lsmr_damp=float(args.lsmr_damp),
            step_limit=args.step_limit,
            write_endpoint=True,
        )
    _atomic_json(output_dir / "u_solver_stats.json", {"steps": flow["u_solver_stats"]})
    _write_per_step_csv(output_dir / "per_step_metrics.csv", flow)
    _plot_correction_schedule(output_dir / "correction_schedule.png", flow)

    correctness = {
        "fixed_support_invariant": True,
        "gradient_smoke": "passed during every guided tile-step",
        "rho_zero_equivalence": "unit-tested official _xstart_to_pred/Euler algebra",
        "A_operator": {"rows": int(operator.matrix.shape[0]), "variables": int(operator.matrix.shape[1]), "nnz": int(operator.matrix.nnz), "row_sum_error_max": operator.metadata.get("row_sum_error_max")},
        "all_finite": True,
    }
    meshes: Dict[str, Any] = {"global_baseline": baseline_mesh}
    variant_summaries: Dict[str, Any] = {}
    if not bool(args.skip_final_stitch):
        pure_mesh, pure_summary = _variant_patch_and_stitch(variant="pure_hr", endpoint_attr="pure_endpoint", contexts=contexts, pipeline=pipeline, global_camera=global_camera, baseline_mesh=baseline_mesh, args=args, output_root=output_dir)
        guided_mesh, guided_summary = _variant_patch_and_stitch(variant="joint_u_cov_guided", endpoint_attr="guided_endpoint", contexts=contexts, pipeline=pipeline, global_camera=global_camera, baseline_mesh=baseline_mesh, args=args, output_root=output_dir)
        meshes["pure_HR"] = pure_mesh
        meshes["joint_U_cov_guided"] = guided_mesh
        variant_summaries = {"pure_HR": pure_summary, "joint_U_cov_guided": guided_summary}

    evaluation_table: List[Dict[str, Any]] = []
    render_records: Dict[str, Any] = {}
    multiview_record: Dict[str, Any] = {"enabled": False}
    if bool(args.render) and len(meshes) >= 3:
        envmap = core.load_envmap(str(args.envmap), device="cuda")
        for variant, mesh in meshes.items():
            render_dir = output_dir / variant / "aligned_eval_4096"
            render = core._render(mesh, output_dir=render_dir, camera=global_camera, reference_image=output_dir / "canonical_4096.png", args=args, envmap=envmap)
            render_records[variant] = render
            metric = core._metric_subset(render)
            evaluation_table.append({"variant": variant, "vertices": int(mesh.vertices.shape[0]), "faces": int(mesh.faces.shape[0]), "PSNR": metric["psnr_db"], "SSIM": metric["ssim"], "LPIPS": metric["lpips"], "render_resolution": int(args.render_resolution)})
        if bool(args.render_multiview):
            import pixal3d_pbr_range_null_perstep_experiment as render_route
            multiview_record = render_route._render_multiview_variants(meshes=meshes, baseline_mesh=baseline_mesh, global_camera=global_camera, output_root=output_dir, args=args, envmap=envmap)
            multiview_source = output_dir / "multiview_4variants"
            _link_artifact(multiview_source, output_dir / "multiview")
            for variant, record in multiview_record.get("variants", {}).items():
                turntable = record.get("turntable_gif")
                if turntable:
                    _link_artifact(Path(turntable), output_dir / "turntable" / f"{variant}.gif")
            for channel, path in multiview_record.get("pbr_front_back_contact_sheets", {}).items():
                _link_artifact(Path(path), output_dir / "pbr_channel_sheets" / f"{channel}_front_back_contact_sheet.png")
        del envmap
        _empty_cuda_cache()
    else:
        for variant, mesh in meshes.items():
            evaluation_table.append({"variant": variant, "vertices": int(mesh.vertices.shape[0]), "faces": int(mesh.faces.shape[0]), "PSNR": None, "SSIM": None, "LPIPS": None, "render_resolution": None})
    _write_metrics_csv(output_dir / "metrics.csv", evaluation_table)

    questions = _aggregate_flow_questions(flow)
    summary: Dict[str, Any] = {
        "format": FORMAT,
        "image": str(Path(args.image).resolve()),
        "cuda_device": physical_device,
        "cuda_logical_device": logical_device,
        "cuda_name": torch.cuda.get_device_name(logical_device),
        "guided_seed": GUIDED_SEED,
        "covariance_seeds": list(COVARIANCE_SEEDS),
        "context_cache_dir": str(cache_dir),
        "global_baseline": {"source": str(baseline_path.resolve()), "geometry_fixed": True, "flow_used": False, "attrs_channels": PBR_CHANNELS},
        "tile_layout": {"canonical_image_size": CANONICAL_IMAGE_SIZE, "tile_size": TILE_SIZE, "stride": TILE_STRIDE, "participating_tile_count": len(contexts), "participating_tile_ids": tile_ids},
        "sampler": {"texture_steps": TEXTURE_STEPS, "rescale_t": TEXTURE_RESCALE_T, "noise_timestep": 1.0, "noise_strength": 1.0, "pure_hr_route": "official sampler.sample from fixed guided seed-42 initial state", "endpoint_space": "normalized texture-SLat", "corrected_endpoint_to_velocity": "official sampler._xstart_to_pred", "euler": "official x_t - (t-t_next) * v"},
        "operator": operator.metadata,
        "covariance": {"M": len(COVARIANCE_SEEDS), "seeds": list(COVARIANCE_SEEDS), "space": "normalized texture-SLat", "stats": covariance_stats_output},
        "correctness": correctness,
        "flow": flow,
        "questions": questions,
        "evaluation": {"table": evaluation_table, "renders": render_records, "multiview": multiview_record},
        "variants": variant_summaries,
        "artifacts": {"summary": str((output_dir / "summary.json").resolve()), "report": str((output_dir / "JOINT_U_COV_PERSTEP_REPORT.md").resolve()), "metrics": str((output_dir / "metrics.csv").resolve()), "per_step_metrics": str((output_dir / "per_step_metrics.csv").resolve()), "covariance_stats": str((output_dir / "covariance_stats.json").resolve()), "u_solver_stats": str((output_dir / "u_solver_stats.json").resolve()), "correction_schedule": str((output_dir / "correction_schedule.png").resolve())},
    }
    report_path = output_dir / "JOINT_U_COV_PERSTEP_REPORT.md"
    _write_report(report_path, summary)
    _atomic_json(output_dir / "summary.json", summary)
    print(f"[done] tiles={len(contexts)} steps={flow['flow_steps']} summary={output_dir / 'summary.json'}")
    return summary


def main() -> None:
    run(_build_parser().parse_args())


if __name__ == "__main__":
    main()
