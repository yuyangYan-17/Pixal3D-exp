#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed-shape cross-tile PBR consensus guidance for Pixal3D.

This experiment keeps one local shape SLat fixed for every canonical 4096
tile.  Only the texture SLat is flowed.  At every native texture-flow Euler
step all tiles first predict a clean endpoint, then all endpoints are decoded
before any tile is queried or updated.  PBR fields are fused in object space
through the exact local/global camera transforms and continuous
``MeshWithVoxel.query_attrs`` queries.  The fused field is sent back through
the official PBR encoder using a cycle-cancelled latent residual:

    x0_guided = pred_x0 + eta * (encode(F_fused) - encode(F_self))

The default route is the requested 49 tiles (1024 tile, 512 stride), CUDA 4,
normal/full-VRAM mode, and eta=1.0.  ``--low-vram`` is available only as an
explicit recovery option; it is deliberately disabled by default.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import shutil
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
from PIL import Image, ImageDraw, ImageOps

import pixal3d.models as pixal3d_models
import pixal3d_texture_pbr_degradation_experiment as degradation
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
from inference import MODEL_PATH
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVertexPbr, MeshWithVoxel
from pixal3d.utils import render_utils
import utils3d


FORMAT = "pixal3d_cross_tile_pbr_perstep_v1"
CANONICAL_IMAGE_SIZE = 4096
TILE_SIZE = 1024
TILE_STRIDE = 512
OVOXEL_RESOLUTION = 1024
LATENT_RESOLUTION = 64
PBR_LAYOUT = dict(core.PBR_LAYOUT)
PBR_CHANNELS = {
    "RGB": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}


@dataclass
class TileContext:
    tile_id: int
    box: Tuple[int, int, int, int]
    transform: Any
    image: Image.Image
    tile_dir: Path
    geometry: core.LocalGeometry
    shape_reference: SparseTensor
    shape_norm: SparseTensor
    shape_denorm: SparseTensor
    texture_reference: SparseTensor
    texture_norm: SparseTensor
    noise: SparseTensor
    initial_state: SparseTensor
    condition: Mapping[str, Any]
    target_coords: torch.Tensor
    target_points: torch.Tensor
    static_stats: Dict[str, Any]
    pure_endpoint: Optional[SparseTensor] = None
    guided_endpoint: Optional[SparseTensor] = None


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
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _fresh_sparse(value: SparseTensor) -> SparseTensor:
    # SparseTensor.replace intentionally shares the source spatial cache.  A
    # long multi-tile run would therefore retain every encoder/decoder cache
    # on CUDA.  Reconstruct both arrays to create a genuinely cache-free value.
    return SparseTensor(
        value.feats.detach().clone(),
        value.coords.detach().clone(),
    )


def _move_condition(value: Any, device: torch.device) -> Any:
    """Recursively move a projection condition without changing its schema."""
    if isinstance(value, Mapping):
        return {key: _move_condition(item, device) for key, item in value.items()}
    if isinstance(value, SparseTensor):
        return SparseTensor(
            value.feats.detach().to(device),
            value.coords.detach().to(device),
        )
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def _sparse_to_device(value: SparseTensor, device: torch.device) -> SparseTensor:
    """Move a sparse value without inheriting its source spatial cache."""
    return SparseTensor(
        value.feats.detach().to(device),
        value.coords.detach().to(device),
    )


def _sparse_to_cpu(value: SparseTensor) -> SparseTensor:
    return _sparse_to_device(value, torch.device("cpu"))


def _offload_contexts_to_cpu(contexts: Sequence[TileContext]) -> None:
    """Release retained CUDA tensors while preserving the exact tile route."""
    sparse_names = (
        "shape_reference",
        "shape_norm",
        "shape_denorm",
        "texture_reference",
        "texture_norm",
        "noise",
        "initial_state",
    )
    for context in contexts:
        for name in sparse_names:
            value = getattr(context, name)
            setattr(context, name, _sparse_to_cpu(value))
        context.target_coords = context.target_coords.detach().to("cpu").clone()
        context.target_points = context.target_points.detach().to("cpu").clone()
        context.condition = _move_condition(context.condition, torch.device("cpu"))


def _normalize_slat(value: SparseTensor, normalization: Mapping[str, Sequence[float]]) -> SparseTensor:
    mean = torch.as_tensor(normalization["mean"], device=value.device, dtype=value.feats.dtype).reshape(1, -1)
    std = torch.as_tensor(normalization["std"], device=value.device, dtype=value.feats.dtype).reshape(1, -1)
    return SparseTensor((value.feats - mean) / std, value.coords.detach().clone())


def _denormalize_slat(value: SparseTensor, normalization: Mapping[str, Sequence[float]]) -> SparseTensor:
    mean = torch.as_tensor(normalization["mean"], device=value.device, dtype=value.feats.dtype).reshape(1, -1)
    std = torch.as_tensor(normalization["std"], device=value.device, dtype=value.feats.dtype).reshape(1, -1)
    return SparseTensor(value.feats * std + mean, value.coords.detach().clone())


def _native_noised_endpoint(
    clean: SparseTensor,
    noise: SparseTensor,
    sampler: Any,
    timestep: float,
    strength: float,
) -> SparseTensor:
    if not torch.equal(clean.coords, noise.coords) or clean.feats.shape != noise.feats.shape:
        raise RuntimeError("texture reference and noise support/shape differ")
    t = float(timestep)
    if not 0.0 <= t <= 1.0:
        raise ValueError("noise timestep must lie in [0,1]")
    sigma = float(sampler.sigma_min) + (1.0 - float(sampler.sigma_min)) * t
    return SparseTensor(
        (1.0 - t) * clean.feats + sigma * float(strength) * noise.feats,
        clean.coords.detach().clone(),
    )


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


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().to(torch.float64)).item())


def _safe_mean(value: torch.Tensor) -> float:
    return float(value.detach().to(torch.float64).mean().item()) if value.numel() else 0.0


def _relative(value: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> float:
    return _norm(value) / (_norm(reference) + float(eps))


def _tensor_stats(value: torch.Tensor) -> Dict[str, Any]:
    flat = value.detach().to(torch.float32).reshape(-1)
    if flat.numel() == 0:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    # torch.quantile has a practical input-size limit in this environment;
    # keep exact count/min/max/mean and use a deterministic bounded sample for
    # order statistics so large overlap diagnostics remain serializable.
    quantile_input = flat
    max_quantile_values = 1_000_000
    if quantile_input.numel() > max_quantile_values:
        stride = max(1, int(math.ceil(quantile_input.numel() / max_quantile_values)))
        quantile_input = quantile_input[::stride][:max_quantile_values]
    return {
        "count": int(flat.numel()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "median": float(quantile_input.median().item()),
        "q10": float(torch.quantile(quantile_input, 0.10).item()),
        "q50": float(torch.quantile(quantile_input, 0.50).item()),
        "q90": float(torch.quantile(quantile_input, 0.90).item()),
        "quantile_sample_count": int(quantile_input.numel()),
    }


def _channel_mean_abs(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    if left.shape != right.shape:
        raise ValueError(f"PBR comparison shape mismatch: {left.shape} vs {right.shape}")
    result: Dict[str, float] = {}
    for name, channel_slice in PBR_CHANNELS.items():
        values = (left[:, channel_slice] - right[:, channel_slice]).abs()
        if mask is not None:
            values = values[mask]
        result[name] = _safe_mean(values)
    return result


def _coordinate_digest(value: SparseTensor) -> str:
    coords = value.coords.detach().to(device="cpu", dtype=torch.int32).contiguous()
    return hashlib.sha256(coords.numpy().tobytes()).hexdigest()


def _strict_sparse_check(reference: SparseTensor, candidate: SparseTensor, label: str) -> Dict[str, Any]:
    same_shape = tuple(reference.feats.shape) == tuple(candidate.feats.shape)
    same_coords = tuple(reference.coords.shape) == tuple(candidate.coords.shape) and torch.equal(
        reference.coords, candidate.coords
    )
    result = {
        "label": str(label),
        "coords_exact": bool(same_coords),
        "feature_shape_equal": bool(same_shape),
        "reference_tokens": int(reference.feats.shape[0]),
        "candidate_tokens": int(candidate.feats.shape[0]),
        "reference_coord_digest": _coordinate_digest(reference),
        "candidate_coord_digest": _coordinate_digest(candidate),
    }
    if not same_coords or not same_shape:
        raise RuntimeError(f"strict sparse endpoint check failed: {result}")
    return result


def _query_mesh_chunked(mesh: MeshWithVoxel, points: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if int(chunk_size) <= 0:
        raise ValueError("query chunk size must be positive")
    # flex_gemm's MeshWithVoxel.query_attrs implementation is CUDA-backed.
    # In low-VRAM mode decoded donor meshes are frozen on CPU between barrier
    # phases; materialize only the current donor for the official query and
    # immediately return the sampled field to the caller's device.
    temporary_cuda_mesh = mesh.device.type != "cuda"
    query_mesh = mesh.to("cuda") if temporary_cuda_mesh else mesh
    query_points = points.to(device=query_mesh.device)
    rows: List[torch.Tensor] = []
    for start in range(0, int(query_points.shape[0]), int(chunk_size)):
        rows.append(
            query_mesh.query_attrs(query_points[start : start + int(chunk_size)]).to(torch.float32)
        )
    if not rows:
        result = torch.empty(
            (0, int(mesh.attrs.shape[1])), device=query_points.device, dtype=torch.float32
        )
    else:
        result = torch.cat(rows, dim=0)
    if temporary_cuda_mesh:
        result = result.to(device=points.device)
        del query_mesh, query_points
        _empty_cuda_cache()
    return result


def _validate_decoded_mesh(mesh: Any, label: str) -> MeshWithVoxel:
    mesh = core._validate_mesh(mesh, label)
    if mesh.attrs.shape[1] != 6:
        raise RuntimeError(f"{label}: expected six PBR channels, got {mesh.attrs.shape[1]}")
    return mesh


def _decode_endpoint(
    *,
    pipeline: Any,
    shape_denorm: SparseTensor,
    texture_norm: SparseTensor,
    query_points: torch.Tensor,
    query_chunk_size: int,
    label: str,
) -> Tuple[MeshWithVoxel, torch.Tensor, Dict[str, Any]]:
    started = time.perf_counter()
    decoded = pipeline.decode_latent(
        _fresh_sparse(shape_denorm),
        _fresh_sparse(_denormalize_slat(texture_norm, pipeline.tex_slat_normalization)),
        OVOXEL_RESOLUTION,
    )
    _sync_cuda()
    if len(decoded) != 1:
        raise RuntimeError(f"{label}: decoder returned {len(decoded)} meshes")
    mesh = _validate_decoded_mesh(decoded[0], label)
    field = _query_mesh_chunked(mesh, query_points, int(query_chunk_size))
    if not torch.isfinite(field).all():
        raise RuntimeError(f"{label}: decoded PBR query is non-finite")
    stats = {
        "decode_seconds": float(time.perf_counter() - started),
        "decoded_vertices": int(mesh.vertices.shape[0]),
        "decoded_faces": int(mesh.faces.shape[0]),
        "decoded_active_ovoxels": int(mesh.coords.shape[0]),
        "queried_fixed_support_tokens": int(field.shape[0]),
        "decoded_pbr_range": core._tensor_range(mesh.attrs),
        "decoded_support_coord_digest": hashlib.sha256(
            mesh.coords.detach().to(device="cpu", dtype=torch.int32).contiguous().numpy().tobytes()
        ).hexdigest(),
    }
    return mesh, field, stats


def _tile_uv(uv_full: torch.Tensor, transform: Any) -> torch.Tensor:
    x0, y0, _, _ = transform.box
    uv_tile = torch.empty_like(uv_full)
    uv_tile[:, 0] = (uv_full[:, 0] - float(x0)) * float(transform.crop_to_output_scale_x)
    uv_tile[:, 1] = (uv_full[:, 1] - float(y0)) * float(transform.crop_to_output_scale_y)
    return uv_tile


def _local_to_global(
    points_local: torch.Tensor,
    *,
    transform: Any,
    global_camera: Mapping[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_local = points_local * (2.0 * float(transform.mesh_scale))
    q_global, uv_full = core._local_q_to_global_q(
        q_local,
        global_camera=global_camera,
        transform=transform,
    )
    global_points = q_global / (2.0 * float(global_camera["mesh_scale"]))
    return q_global, uv_full, global_points


def _global_to_local(
    q_global: torch.Tensor,
    *,
    transform: Any,
    global_camera: Mapping[str, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_local, uv_tile = core._global_q_to_local_q(
        q_global,
        global_camera=global_camera,
        transform=transform,
    )
    return q_local / (2.0 * float(transform.mesh_scale)), uv_tile


def _inside_tile(uv_tile: torch.Tensor) -> torch.Tensor:
    return (
        torch.isfinite(uv_tile).all(dim=1)
        & (uv_tile[:, 0] >= 0.0)
        & (uv_tile[:, 0] < float(TILE_SIZE))
        & (uv_tile[:, 1] >= 0.0)
        & (uv_tile[:, 1] < float(TILE_SIZE))
    )


def _gaussian_fuse_candidates(
    candidate_pbr: torch.Tensor,
    candidate_valid: torch.Tensor,
    candidate_distance: torch.Tensor,
    sigma_pixels: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse [N,K,6] PBR candidates with only Gaussian center weights."""
    if candidate_pbr.ndim != 3 or candidate_pbr.shape[-1] != 6:
        raise ValueError(f"candidate PBR must be [N,K,6], got {candidate_pbr.shape}")
    if candidate_valid.shape != candidate_distance.shape or candidate_valid.shape[:2] != candidate_pbr.shape[:2]:
        raise ValueError("candidate validity/distance shapes do not match candidate PBR")
    if float(sigma_pixels) <= 0.0:
        raise ValueError("fusion sigma must be positive")
    raw = torch.exp(-candidate_distance.square() / (2.0 * float(sigma_pixels) ** 2))
    raw = torch.where(candidate_valid, raw, torch.zeros_like(raw))
    denominator = raw.sum(dim=1, keepdim=True)
    if bool((denominator <= 0.0).any().item()):
        raise RuntimeError("PBR fusion found a target without a valid self candidate")
    normalized = raw / denominator
    fused = (normalized[..., None] * candidate_pbr).sum(dim=1)
    return fused, normalized, raw


def _candidate_capacity() -> int:
    # With 1024 tiles and 512 stride, a point is covered by at most four image
    # tiles.  One extra slot is reserved for the mandatory self candidate when
    # its projection lies just outside its own crop.
    return 5


def _fuse_tile_field(
    *,
    target: TileContext,
    contexts: Sequence[TileContext],
    decoded: Mapping[int, MeshWithVoxel],
    self_field: torch.Tensor,
    global_camera: Mapping[str, float],
    sigma_pixels: float,
    query_chunk_size: int,
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, torch.Tensor]]:
    """Cross-tile query/fusion for one target against a frozen decode barrier."""
    points_local = target.target_points
    device = points_local.device
    count = int(points_local.shape[0])
    capacity = _candidate_capacity()
    candidate_tile_ids = torch.full((count, capacity), -1, device=device, dtype=torch.int64)
    candidate_local_points = torch.zeros((count, capacity, 3), device=device, dtype=torch.float32)
    candidate_global_points = torch.zeros((count, 3), device=device, dtype=torch.float32)
    candidate_uv = torch.zeros((count, capacity, 2), device=device, dtype=torch.float32)
    candidate_distance = torch.full((count, capacity), float("inf"), device=device, dtype=torch.float32)
    candidate_pbr = torch.zeros((count, capacity, 6), device=device, dtype=torch.float32)
    candidate_valid = torch.zeros((count, capacity), device=device, dtype=torch.bool)
    candidate_covered = torch.zeros((count, capacity), device=device, dtype=torch.bool)

    q_global, uv_full, global_points = _local_to_global(
        points_local,
        transform=target.transform,
        global_camera=global_camera,
    )
    candidate_global_points.copy_(global_points)
    self_uv = _tile_uv(uv_full, target.transform)
    center = self_uv.new_tensor([511.5, 511.5])
    candidate_tile_ids[:, 0] = int(target.tile_id)
    candidate_local_points[:, 0] = points_local
    candidate_uv[:, 0] = self_uv
    candidate_distance[:, 0] = torch.linalg.vector_norm(self_uv - center[None], dim=1)
    candidate_pbr[:, 0] = self_field
    candidate_valid[:, 0] = True
    candidate_covered[:, 0] = True

    other_slot_count = torch.zeros((count,), device=device, dtype=torch.int64)
    coverage_count = torch.zeros((count,), device=device, dtype=torch.int64)
    target_bounds = torch.tensor(
        [-0.5 - 1e-5, -0.5 - 1e-5, -0.5 - 1e-5], device=device, dtype=torch.float32
    )
    target_upper = torch.tensor(
        [0.5 + 1e-5, 0.5 + 1e-5, 0.5 + 1e-5], device=device, dtype=torch.float32
    )

    for donor in contexts:
        if int(donor.tile_id) == int(target.tile_id):
            continue
        donor_points, donor_uv = _global_to_local(
            q_global,
            transform=donor.transform,
            global_camera=global_camera,
        )
        inside = _inside_tile(donor_uv)
        available = inside & (other_slot_count < capacity - 1)
        rows = torch.where(available)[0]
        if rows.numel() == 0:
            continue
        slots = 1 + other_slot_count.index_select(0, rows)
        candidate_tile_ids[rows, slots] = int(donor.tile_id)
        candidate_local_points[rows, slots] = donor_points.index_select(0, rows)
        candidate_uv[rows, slots] = donor_uv.index_select(0, rows)
        candidate_distance[rows, slots] = torch.linalg.vector_norm(
            donor_uv.index_select(0, rows) - center[None], dim=1
        )
        candidate_covered[rows, slots] = True
        coverage_count[rows] += 1

        local_finite = torch.isfinite(donor_points.index_select(0, rows)).all(dim=1)
        in_volume = (
            (donor_points.index_select(0, rows) >= target_bounds[None]).all(dim=1)
            & (donor_points.index_select(0, rows) <= target_upper[None]).all(dim=1)
        )
        query_points = donor_points.index_select(0, rows)
        queried = _query_mesh_chunked(decoded[int(donor.tile_id)], query_points, int(query_chunk_size))
        queried_finite = torch.isfinite(queried).all(dim=1)
        query_valid = local_finite & in_volume & queried_finite
        candidate_pbr[rows, slots] = torch.where(
            query_valid[:, None], queried, torch.zeros_like(queried)
        )
        candidate_valid[rows, slots] = query_valid
        other_slot_count[rows] += 1

    fused, normalized_weights, raw_weights = _gaussian_fuse_candidates(
        candidate_pbr,
        candidate_valid,
        candidate_distance,
        float(sigma_pixels),
    )
    valid_other_count = candidate_valid[:, 1:].sum(dim=1).to(torch.int64)
    covered_other_count = candidate_covered[:, 1:].sum(dim=1).to(torch.int64)
    overlap = valid_other_count > 0
    fused = torch.where(overlap[:, None], fused, self_field)

    valid_distances = candidate_distance[candidate_valid]
    valid_normalized_weights = normalized_weights[candidate_valid]
    overlap_pbr_abs = (fused - self_field).abs()
    stats = {
        "target_tile": int(target.tile_id),
        "active_ovoxel_count": int(count),
        "overlap_ovoxel_count": int(overlap.sum().item()),
        "non_overlap_ovoxel_count": int((~overlap).sum().item()),
        "covered_donor_count": {
            "min": int(covered_other_count.min().item()) if count else 0,
            "mean": float(covered_other_count.float().mean().item()) if count else 0.0,
            "max": int(covered_other_count.max().item()) if count else 0,
        },
        "query_valid_donor_count": {
            "min": int(valid_other_count.min().item()) if count else 0,
            "mean": float(valid_other_count.float().mean().item()) if count else 0.0,
            "max": int(valid_other_count.max().item()) if count else 0,
        },
        "distance_to_center_pixels": _tensor_stats(valid_distances),
        "normalized_fusion_weight": _tensor_stats(valid_normalized_weights),
        "raw_fusion_weight": _tensor_stats(raw_weights[candidate_valid]),
        "gaussian_sigma_pixels": float(sigma_pixels),
        "self_candidate_always_present": True,
        "fusion_channels": ["RGB", "metallic", "roughness", "alpha"],
        "pbr_self_vs_fused_mean_abs_all": _channel_mean_abs(self_field, fused),
        "pbr_self_vs_fused_mean_abs_overlap": _channel_mean_abs(self_field, fused, overlap),
        "pbr_self_vs_fused_mean_abs_overlap_joint": _safe_mean(overlap_pbr_abs[overlap]),
    }
    details = {
        "target_tile": torch.tensor(int(target.tile_id), dtype=torch.int64),
        "target_local_ovoxel": target.target_coords.detach().index_select(
            0, torch.where(overlap)[0]
        ).cpu(),
        "target_local_query_position": points_local.detach().index_select(
            0, torch.where(overlap)[0]
        ).cpu(),
        "global_query_position": candidate_global_points.detach().index_select(
            0, torch.where(overlap)[0]
        ).cpu(),
        "donor_tile_ids": candidate_tile_ids.detach().index_select(
            0, torch.where(overlap)[0]
        ).cpu(),
        "donor_local_query_positions": candidate_local_points.detach().index_select(
            0, torch.where(overlap)[0]
        ).cpu(),
        "donor_projected_uv": candidate_uv.detach().index_select(
            0, torch.where(overlap)[0]
        ).cpu(),
        "distance_to_tile_center": candidate_distance.detach().index_select(
            0, torch.where(overlap)[0]
        ).cpu(),
        "normalized_weights": normalized_weights.detach().index_select(
            0, torch.where(overlap)[0]
        ).cpu(),
        "query_valid": candidate_valid.detach().index_select(
            0, torch.where(overlap)[0]
        ).cpu(),
        "covered": candidate_covered.detach().index_select(
            0, torch.where(overlap)[0]
        ).cpu(),
        "queried_pbr": candidate_pbr.detach().index_select(
            0, torch.where(overlap)[0]
        ).cpu(),
        "final_fused_pbr": fused.detach().index_select(0, torch.where(overlap)[0]).cpu(),
        "self_pbr": self_field.detach().index_select(0, torch.where(overlap)[0]).cpu(),
        "capacity": torch.tensor(capacity, dtype=torch.int64),
    }
    if not torch.isfinite(fused).all():
        raise RuntimeError(f"tile {target.tile_id}: cross-tile PBR fusion produced non-finite values")
    return fused, stats, details


def _save_sparse_payload(path: Path, value: SparseTensor, normalization: Optional[Mapping[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {
        "coords": value.coords.detach().cpu().to(torch.int32),
        "features": value.feats.detach().cpu().to(torch.float32),
    }
    if normalization is not None:
        payload["normalization"] = dict(normalization)
    _atomic_torch_save(path, payload)


def _save_fusion_details(path: Path, details: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]) -> None:
    payload = {str(k): v for k, v in details.items()}
    payload["metadata"] = dict(metadata)
    _atomic_torch_save(path, payload)


def _parse_ids(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not str(value).strip():
        return None
    return {int(item.strip()) for item in str(value).split(",") if item.strip()}


def _load_mesh(path: Path) -> MeshWithVoxel:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mesh = payload.get("mesh", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(mesh, MeshWithVoxel):
        raise RuntimeError(f"cached baseline is not MeshWithVoxel: {path}")
    return mesh


def _load_sparse_payload(path: Path) -> SparseTensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "coords" not in payload or "features" not in payload:
        raise RuntimeError(f"cached sparse payload has invalid schema: {path}")
    return SparseTensor(
        payload["features"].to(torch.float32).contiguous(),
        payload["coords"].to(torch.int32).contiguous(),
    )


def _load_or_run_global_baseline(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    image_1024: Image.Image,
    global_camera: Mapping[str, float],
    output_dir: Path,
) -> Tuple[MeshWithVoxel, Dict[str, Any]]:
    output_path = output_dir / "global_baseline_mesh.pt"
    source_path = Path(args.baseline_dir).expanduser() / "global_baseline_mesh.pt" if args.baseline_dir else None
    if source_path is not None and source_path.is_file():
        mesh = _load_mesh(source_path)
        _atomic_torch_save(output_path, {"format": f"{FORMAT}_global_mesh", "mesh": mesh})
        return mesh, {
            "source": "cached baseline mesh",
            "source_path": str(source_path.resolve()),
            "generated_once": False,
            "vertices": int(mesh.vertices.shape[0]),
            "faces": int(mesh.faces.shape[0]),
            "active_ovoxels": int(mesh.coords.shape[0]),
        }
    if bool(args.resume) and output_path.is_file():
        mesh = _load_mesh(output_path)
        return mesh, {
            "source": "resumed baseline mesh",
            "source_path": str(output_path.resolve()),
            "generated_once": False,
            "vertices": int(mesh.vertices.shape[0]),
            "faces": int(mesh.faces.shape[0]),
            "active_ovoxels": int(mesh.coords.shape[0]),
        }

    _seed_everything(int(args.seed))
    ss_params, shape_params, texture_params = core._sampler_overrides(args)
    started = time.perf_counter()
    print("[global-baseline] running ordinary Pixal3D 1024_cascade once")
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
    mesh = core._validate_mesh(output[0], "global 1024 baseline").to("cpu")
    _atomic_torch_save(output_path, {"format": f"{FORMAT}_global_mesh", "mesh": mesh})
    record = {
        "source": "ordinary pipeline.run pipeline_type=1024_cascade",
        "generated_once": True,
        "generation_seconds": float(time.perf_counter() - started),
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "active_ovoxels": int(mesh.coords.shape[0]),
        "baseline_shape_sampler_used_once_for_global_baseline": True,
        "tile_shape_flow_used": False,
    }
    del output, latents
    _empty_cuda_cache()
    return mesh, record


def _prepare_tile_contexts(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    baseline_mesh: MeshWithVoxel,
    global_camera: Mapping[str, float],
    image_4096: Image.Image,
    output_dir: Path,
    global_attr_field: MeshWithVoxel,
    shape_encoder: torch.nn.Module,
    pbr_encoder: torch.nn.Module,
    boxes: Sequence[Tuple[int, int, int, int]],
) -> List[TileContext]:
    device = torch.device("cuda")
    face_min, face_max, face_finite = core._project_face_bboxes(
        baseline_mesh.vertices,
        baseline_mesh.faces,
        mesh_scale=float(global_camera["mesh_scale"]),
        global_camera=global_camera,
        chunk_size=int(args.face_projection_chunk_size),
    )
    requested = _parse_ids(args.tile_ids)
    contexts: List[TileContext] = []
    tex_params = core._sampler_overrides(args)[2]
    texture_model = pipeline.models["tex_slat_flow_model_1024"]
    for tile_id, box in enumerate(boxes):
        if requested is not None and int(tile_id) not in requested:
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
        tile_image = image_4096.crop(box).convert("RGB")
        if tile_image.size != (TILE_SIZE, TILE_SIZE):
            tile_image = tile_image.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS)
        tile_image.save(tile_dir / "hr_tile_1024_condition.png")
        selected = core._tile_face_ids_from_bbox(face_min, face_max, face_finite, box)
        if selected.numel() == 0:
            _atomic_json(
                tile_dir / "summary.json",
                {
                    "status": "skipped_invalid_empty",
                    "tile_id": int(tile_id),
                    "box": list(map(int, box)),
                    "reason": "no projected global triangle bbox intersects tile",
                    "participates_in_flow": False,
                },
            )
            print(f"[prepare tile {tile_id:02d}] skipped: no projected global triangle bbox")
            continue
        print(f"[prepare tile {tile_id:02d}] box={box} projected_faces={int(selected.numel()):,}")

        cached_shape_path = tile_dir / "fixed_shape_norm.pt"
        cached_texture_path = tile_dir / "texture_reference_norm.pt"
        cached_initial_path = tile_dir / "texture_initial_state.pt"
        cached_summary_path = tile_dir / "fixed_shape_summary.json"
        if bool(args.resume) and all(
            path.is_file()
            for path in (
                cached_shape_path,
                cached_texture_path,
                cached_initial_path,
                cached_summary_path,
            )
        ):
            print(f"[prepare tile {tile_id:02d}] resuming cached fixed latent/support")
            geometry = core._prepare_tile_geometry(
                global_vertices=baseline_mesh.vertices,
                global_faces=baseline_mesh.faces,
                global_face_min=face_min,
                global_face_max=face_max,
                global_face_finite=face_finite,
                global_camera=global_camera,
                transform=transform,
            )
            shape_norm = _load_sparse_payload(cached_shape_path)
            texture_norm = _load_sparse_payload(cached_texture_path)
            initial_state = _load_sparse_payload(cached_initial_path)
            alignment = core._latent_support_diagnostics(shape_norm, texture_norm)
            if not alignment["coordinates_exactly_equal"]:
                raise RuntimeError(f"tile {tile_id}: resumed shape/texture support mismatch: {alignment}")
            shape_denorm = _denormalize_slat(shape_norm, pipeline.shape_slat_normalization)
            shape_reference = _fresh_sparse(shape_norm)
            texture_reference = _fresh_sparse(texture_norm)
            noise = SparseTensor(
                torch.zeros_like(texture_norm.feats),
                texture_norm.coords.detach().clone(),
            )
            condition = pipeline.get_proj_cond_shape(
                pipeline.image_cond_model_tex_1024,
                [tile_image],
                shape_norm.coords.to(device=device, dtype=torch.int32),
                camera_angle_x=float(transform.camera_angle_x),
                distance=float(transform.distance),
                mesh_scale=float(transform.mesh_scale),
                grid_resolution_override=LATENT_RESOLUTION,
            )
            if bool(args.low_vram):
                condition = _move_condition(condition, torch.device("cpu"))
                _empty_cuda_cache()
            target_coords = geometry.coords.to(device=device, dtype=torch.int32)
            target_points = (target_coords.to(torch.float32) + 0.5) / float(OVOXEL_RESOLUTION) - 0.5
            static_stats = json.loads(cached_summary_path.read_text(encoding="utf-8"))
            static_stats["resumed_from_cache"] = True
            contexts.append(
                TileContext(
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
                    condition=condition,
                    target_coords=target_coords,
                    target_points=target_points,
                    static_stats=static_stats,
                )
            )
            continue
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
            raise RuntimeError(
                f"tile {tile_id}: camera round-trip exceeded tolerance: {geometry.stats}"
            )
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
            device=device,
            low_vram=bool(args.low_vram),
        )
        texture_reference, texture_stats = core._encode_local_pbr(
            encoder=pbr_encoder,
            coords=geometry.coords,
            attrs=local_attrs,
            device=device,
            low_vram=bool(args.low_vram),
        )
        # The released encoders may return values carrying large attention /
        # spatial caches.  Keep only the actual latent arrays in the tile
        # context; this is important even in normal mode and is mandatory for
        # the CPU-staged recovery path.
        shape_reference = _fresh_sparse(shape_reference)
        texture_reference = _fresh_sparse(texture_reference)
        alignment = core._latent_support_diagnostics(shape_reference, texture_reference)
        if not alignment["coordinates_exactly_equal"]:
            raise RuntimeError(f"tile {tile_id}: fixed shape/PBR support mismatch: {alignment}")
        shape_norm = _normalize_slat(shape_reference, pipeline.shape_slat_normalization)
        texture_norm = _normalize_slat(texture_reference, pipeline.tex_slat_normalization)
        if not torch.equal(shape_norm.coords, texture_norm.coords):
            raise RuntimeError(f"tile {tile_id}: normalized fixed shape/texture coordinates differ")
        condition = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_tex_1024,
            [tile_image],
            shape_norm.coords.to(torch.int32),
            camera_angle_x=float(transform.camera_angle_x),
            distance=float(transform.distance),
            mesh_scale=float(transform.mesh_scale),
            grid_resolution_override=LATENT_RESOLUTION,
        )
        if bool(args.low_vram):
            # A 49-tile run otherwise retains every DINO/NAF projection
            # condition on CUDA.  Conditions are immutable and are restored
            # per model call below, so CPU storage preserves the exact route
            # while bounding peak VRAM.
            condition = _move_condition(condition, torch.device("cpu"))
            _empty_cuda_cache()
        shape_cond_norm = shape_norm
        texture_channels = int(texture_model.in_channels) - int(shape_cond_norm.feats.shape[1])
        if int(texture_norm.feats.shape[1]) != texture_channels:
            raise RuntimeError(
                f"tile {tile_id}: texture latent channels={texture_norm.feats.shape[1]} "
                f"but sampler expects {texture_channels}"
            )
        _seed_everything(int(args.seed) + int(tile_id) * 100003)
        noise = SparseTensor(torch.randn_like(texture_norm.feats), texture_norm.coords)
        initial_state = _native_noised_endpoint(
            texture_norm,
            noise,
            pipeline.tex_slat_sampler,
            float(args.noise_timestep),
            float(args.noise_strength),
        )
        target_coords = geometry.coords.to(device=device, dtype=torch.int32)
        target_points = (
            target_coords.to(torch.float32) + 0.5
        ) / float(OVOXEL_RESOLUTION) - 0.5
        shape_denorm = _denormalize_slat(shape_norm, pipeline.shape_slat_normalization)
        static_stats = {
            "tile_id": int(tile_id),
            "box": list(map(int, box)),
            "geometry": geometry.stats,
            "material_resampling": material_stats,
            "shape_encoder": shape_stats,
            "pbr_encoder": texture_stats,
            "fixed_shape": {
                "shape_flow_called": False,
                "shape_sampler_called": False,
                "shape_tokens": int(shape_norm.feats.shape[0]),
                "coord_digest": _coordinate_digest(shape_norm),
                "support_unchanged": True,
            },
            "fixed_texture_reference": {
                "tokens": int(texture_norm.feats.shape[0]),
                "coord_digest": _coordinate_digest(texture_norm),
                "source": "global baseline PBR field queried on local C1024 support then official PBR encoder",
            },
            "active_local_c1024_ovoxels": int(target_coords.shape[0]),
            "texture_condition": {
                "source": "canonical 4096 image crop",
                "path": str(tile_dir / "hr_tile_1024_condition.png"),
                "size": [TILE_SIZE, TILE_SIZE],
                "box_4096": list(map(int, box)),
            },
        }
        _atomic_json(tile_dir / "fixed_shape_summary.json", static_stats)
        _save_sparse_payload(
            tile_dir / "fixed_shape_norm.pt",
            shape_norm,
            pipeline.shape_slat_normalization,
        )
        _save_sparse_payload(
            tile_dir / "texture_reference_norm.pt",
            texture_norm,
            pipeline.tex_slat_normalization,
        )
        _save_sparse_payload(
            tile_dir / "texture_initial_state.pt",
            initial_state,
            pipeline.tex_slat_normalization,
        )
        contexts.append(
            TileContext(
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
                condition=condition,
                target_coords=target_coords,
                target_points=target_points,
                static_stats=static_stats,
            )
        )
    if not contexts:
        raise RuntimeError("no tile contexts were prepared")
    _atomic_json(
        output_dir / "tile_preparation_summary.json",
        {
            "prepared_tile_ids": [int(context.tile_id) for context in contexts],
            "skipped_tile_ids": [
                int(tile_id)
                for tile_id in range(len(boxes))
                if not (output_dir / "tiles" / f"tile_{tile_id:02d}" / "fixed_shape_summary.json").is_file()
            ],
            "active_tile_count": int(len(contexts)),
            "all_requested_tiles_processed": _parse_ids(args.tile_ids) is None,
        },
    )
    return contexts


def _native_schedule(
    sampler: Any,
    params: Mapping[str, Any],
) -> List[float]:
    schedule = [
        float(value)
        for value in sampler.timestep_schedule(
            int(params["steps"]), float(params["rescale_t"])
        )
    ]
    if len(schedule) < 2 or any(schedule[i] <= schedule[i + 1] for i in range(len(schedule) - 1)):
        raise RuntimeError(f"invalid native texture timestep schedule: {schedule}")
    return schedule


def _schedule_start(schedule: Sequence[float], timestep: float) -> int:
    matches = [i for i, value in enumerate(schedule) if abs(float(value) - float(timestep)) <= 1e-6]
    if len(matches) != 1:
        raise RuntimeError(
            f"noise timestep {timestep} is not an exact native texture schedule point: {list(schedule)}"
        )
    return int(matches[0])


@torch.no_grad()
def _run_pure_hr_flow(
    *,
    contexts: Sequence[TileContext],
    pipeline: Any,
    texture_params: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run the official texture sampler without endpoint guidance."""
    sampler = pipeline.tex_slat_sampler
    model = pipeline.models["tex_slat_flow_model_1024"]
    merged = {**pipeline.tex_slat_sampler_params, **dict(texture_params)}
    schedule = _native_schedule(sampler, merged)
    start_index = _schedule_start(schedule, float(args.noise_timestep))
    started = time.perf_counter()
    route = "official tex_slat_sampler.sample"
    for context_index, context in enumerate(contexts):
        print(
            f"[pure_HR] tile {context.tile_id:02d} "
            f"({context_index + 1}/{len(contexts)})"
        )
        cached_endpoint = context.tile_dir / "pure_HR_endpoint.pt"
        if bool(args.resume) and cached_endpoint.is_file():
            output = _load_sparse_payload(cached_endpoint)
            _strict_sparse_check(context.initial_state, output, f"tile {context.tile_id} resumed pure_HR output")
            context.pure_endpoint = output
            route = "resumed cached pure_HR endpoint"
            continue
        low_vram = bool(args.low_vram)
        state = (
            _sparse_to_device(context.initial_state, torch.device("cuda"))
            if low_vram
            else _fresh_sparse(context.initial_state)
        )
        shape_condition = (
            _sparse_to_device(context.shape_norm, torch.device("cuda"))
            if low_vram
            else context.shape_norm
        )
        condition = _move_condition(context.condition, torch.device("cuda"))
        if low_vram:
            model.to(torch.device("cuda"))
        try:
            step_kwargs = _sampler_step_kwargs(merged)
            if start_index == 0:
                result = sampler.sample(
                    model,
                    state,
                    cond=condition["cond"],
                    neg_cond=condition["neg_cond"],
                    concat_cond=shape_condition,
                    **merged,
                    verbose=True,
                    tqdm_desc=f"Tile {context.tile_id:02d} pure_HR texture flow",
                    record_trajectory=False,
                    return_model_history=False,
                )
                output = getattr(result, "samples", result)
                route = "official tex_slat_sampler.sample"
            else:
                for t, t_next in zip(schedule[start_index:-1], schedule[start_index + 1 :]):
                    out = sampler.sample_once(
                        model,
                        state,
                        float(t),
                        float(t_next),
                        cond=condition["cond"],
                        neg_cond=condition["neg_cond"],
                        concat_cond=shape_condition,
                        **step_kwargs,
                    )
                    state = out.pred_x_prev
                output = state
                route = "official tex_slat_sampler.sample_once exact suffix"
        finally:
            if low_vram:
                model.cpu()
        if not isinstance(output, SparseTensor):
            raise RuntimeError(f"tile {context.tile_id}: pure_HR flow returned {type(output)!r}")
        _strict_sparse_check(state, output, f"tile {context.tile_id} pure_HR output")
        context.pure_endpoint = _sparse_to_cpu(output) if low_vram else output
        _save_sparse_payload(
            context.tile_dir / "pure_HR_endpoint.pt",
            context.pure_endpoint,
            pipeline.tex_slat_normalization,
        )
        del condition, shape_condition, state, output
        _empty_cuda_cache()
    _sync_cuda()
    return {
        "route": route,
        "native_schedule": schedule,
        "schedule_start_index": int(start_index),
        "noise_timestep": float(args.noise_timestep),
        "flow_steps": int(len(schedule) - 1 - start_index),
        "tile_count": int(len(contexts)),
        "flow_seconds": float(time.perf_counter() - started),
        "shape_flow_called": False,
        "shape_sampler_called": False,
    }


@torch.no_grad()
def _run_cross_tile_guided_flow_legacy(
    *,
    contexts: Sequence[TileContext],
    pipeline: Any,
    global_camera: Mapping[str, float],
    texture_params: Mapping[str, Any],
    pbr_encoder: torch.nn.Module,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run all texture flows in strict per-step Jacobi/barrier order."""
    if not contexts:
        raise RuntimeError("guided flow requires at least one tile")
    sampler = pipeline.tex_slat_sampler
    model = pipeline.models["tex_slat_flow_model_1024"]
    merged = {**pipeline.tex_slat_sampler_params, **dict(texture_params)}
    schedule = _native_schedule(sampler, merged)
    start_index = _schedule_start(schedule, float(args.noise_timestep))
    step_kwargs = _sampler_step_kwargs(merged)
    states: Dict[int, SparseTensor] = {
        int(context.tile_id): _fresh_sparse(context.initial_state) for context in contexts
    }
    fixed_shape_digest = {
        int(context.tile_id): _coordinate_digest(context.shape_norm) for context in contexts
    }
    started = time.perf_counter()
    per_step: List[Dict[str, Any]] = []

    if bool(args.low_vram):
        model.to(torch.device("cuda"))
    try:
        for step_index, (t, t_next) in enumerate(
            zip(schedule[start_index:-1], schedule[start_index + 1 :]),
            start=start_index,
        ):
            step_started = time.perf_counter()
            print(
                f"[cross-tile step {step_index:02d}] t={float(t):.9f} "
                f"t_next={float(t_next):.9f} tiles={len(contexts)}"
            )

            # Barrier 1: every tile predicts its clean endpoint and velocity
            # from the same frozen state/time before any decode starts.
            predictions: Dict[int, Dict[str, SparseTensor]] = {}
            prediction_started = time.perf_counter()
            for context in contexts:
                state = states[int(context.tile_id)]
                pred_x0, _, pred_v = sampler._get_model_prediction(
                    model,
                    state,
                    float(t),
                    cond=context.condition["cond"],
                    neg_cond=context.condition["neg_cond"],
                    concat_cond=context.shape_norm,
                    **step_kwargs,
                )
                if not isinstance(pred_x0, SparseTensor) or not isinstance(pred_v, SparseTensor):
                    raise RuntimeError(
                        f"tile {context.tile_id}: official sampler prediction is not SparseTensor"
                    )
                _strict_sparse_check(state, pred_x0, f"tile {context.tile_id} pred_x0")
                _strict_sparse_check(state, pred_v, f"tile {context.tile_id} pred_v")
                predictions[int(context.tile_id)] = {"pred_x0": pred_x0, "pred_v": pred_v}
            prediction_barrier = True

            # Barrier 2: decode every clean endpoint while states and decoded
            # fields are still untouched.  The dict is the frozen Jacobi read
            # set used by all target tiles below.
            decoded: Dict[int, MeshWithVoxel] = {}
            decoded_fields: Dict[int, torch.Tensor] = {}
            decode_stats: Dict[int, Dict[str, Any]] = {}
            decode_started = time.perf_counter()
            for context in contexts:
                tile_id = int(context.tile_id)
                mesh, field, stats = _decode_endpoint(
                    pipeline=pipeline,
                    shape_denorm=context.shape_denorm,
                    texture_norm=predictions[tile_id]["pred_x0"],
                    query_points=context.target_points,
                    query_chunk_size=int(args.query_chunk_size),
                    label=f"tile {tile_id:02d} step {step_index:02d} pred_x0",
                )
                decoded[tile_id] = mesh
                decoded_fields[tile_id] = field
                decode_stats[tile_id] = stats
            decode_barrier = True

            # Barrier 3/4: all fields above remain frozen while every target
            # fuses and re-encodes on its original C1024 support.  No state is
            # changed until the complete endpoint map has been built.
            fused_endpoints: Dict[int, SparseTensor] = {}
            step_tile_records: List[Dict[str, Any]] = []
            fusion_started = time.perf_counter()
            encode_started = time.perf_counter()
            for context in contexts:
                tile_id = int(context.tile_id)
                fusion_started_tile = time.perf_counter()
                self_field = decoded_fields[tile_id]
                fused_field, fusion_stats, donor_details = _fuse_tile_field(
                    target=context,
                    contexts=contexts,
                    decoded=decoded,
                    self_field=self_field,
                    global_camera=global_camera,
                    sigma_pixels=float(args.fusion_sigma_pixels),
                    query_chunk_size=int(args.query_chunk_size),
                )
                donor_metadata = {
                    "format": FORMAT,
                    "step": int(step_index),
                    "t": float(t),
                    "t_next": float(t_next),
                    "target_tile": tile_id,
                    "tile_center": [511.5, 511.5],
                    "weight_rule": "exp(-distance_to_tile_center_pixels^2/(2*sigma_pixels^2))",
                    "sigma_pixels": float(args.fusion_sigma_pixels),
                    "visibility_weight_used": False,
                    "foreground_weight_used": False,
                    "facing_weight_used": False,
                    "G_guidance_used": False,
                }
                if bool(args.save_donor_details):
                    _save_fusion_details(
                        context.tile_dir / "steps" / f"step_{step_index:02d}_donors.pt",
                        donor_details,
                        donor_metadata,
                    )
                cycle_raw, cycle_encode_stats = core._encode_local_pbr(
                    encoder=pbr_encoder,
                    coords=context.geometry.coords,
                    attrs=self_field,
                    device=torch.device("cuda"),
                    low_vram=bool(args.low_vram),
                )
                fused_raw, fused_encode_stats = core._encode_local_pbr(
                    encoder=pbr_encoder,
                    coords=context.geometry.coords,
                    attrs=fused_field,
                    device=torch.device("cuda"),
                    low_vram=bool(args.low_vram),
                )
                cycle_norm = _normalize_slat(cycle_raw, pipeline.tex_slat_normalization)
                fused_norm = _normalize_slat(fused_raw, pipeline.tex_slat_normalization)
                pred_x0 = predictions[tile_id]["pred_x0"]
                pred_v = predictions[tile_id]["pred_v"]
                cycle_check = _strict_sparse_check(
                    pred_x0, cycle_norm, f"tile {tile_id} step {step_index} x0_cycle"
                )
                fused_check = _strict_sparse_check(
                    pred_x0, fused_norm, f"tile {tile_id} step {step_index} x0_fused"
                )
                delta = fused_norm.feats - cycle_norm.feats
                guided_x0 = pred_x0.replace(
                    pred_x0.feats + float(args.eta) * delta
                )
                guided_x0_check = _strict_sparse_check(
                    pred_x0, guided_x0, f"tile {tile_id} step {step_index} x0_guided"
                )
                guided_v = sampler._xstart_to_pred(
                    states[tile_id],
                    float(t),
                    guided_x0,
                )
                guided_v_check = _strict_sparse_check(
                    pred_v, guided_v, f"tile {tile_id} step {step_index} guided_v"
                )
                next_state = states[tile_id].replace(
                    states[tile_id].feats - float(t - t_next) * guided_v.feats
                )
                next_state_check = _strict_sparse_check(
                    states[tile_id], next_state, f"tile {tile_id} step {step_index} x_t_next"
                )
                fused_endpoints[tile_id] = next_state
                overlap_mask = torch.zeros(
                    int(context.target_coords.shape[0]),
                    device=self_field.device,
                    dtype=torch.bool,
                )
                overlap_count = int(fusion_stats["overlap_ovoxel_count"])
                if overlap_count:
                    # Reconstructing the mask from saved detail is unnecessary
                    # for the required global channel statistics; the fusion
                    # helper already reports the overlap subset.  Keep this
                    # mask all-false here and use the explicit overlap values.
                    pass
                record = {
                    "tile_id": tile_id,
                    "step": int(step_index),
                    "t": float(t),
                    "t_next": float(t_next),
                    "active_ovoxel_count": int(context.target_coords.shape[0]),
                    "overlap_ovoxel_count": overlap_count,
                    "non_overlap_ovoxel_count": int(fusion_stats["non_overlap_ovoxel_count"]),
                    "donor_count": fusion_stats["query_valid_donor_count"],
                    "covered_donor_count": fusion_stats["covered_donor_count"],
                    "distance_to_center_statistics": fusion_stats["distance_to_center_pixels"],
                    "fusion_weight_statistics": fusion_stats["normalized_fusion_weight"],
                    "pbr_self_vs_fused_mean_abs": fusion_stats["pbr_self_vs_fused_mean_abs_overlap"],
                    "pbr_self_vs_fused_mean_abs_all": fusion_stats["pbr_self_vs_fused_mean_abs_all"],
                    "norm_pred_x0": _norm(pred_x0.feats),
                    "norm_x0_cycle_minus_pred_x0": _norm(cycle_norm.feats - pred_x0.feats),
                    "norm_x0_fused_minus_x0_cycle": _norm(fused_norm.feats - cycle_norm.feats),
                    "norm_x0_guided_minus_pred_x0": _norm(guided_x0.feats - pred_x0.feats),
                    "norm_guided_v_minus_pred_v": _norm(guided_v.feats - pred_v.feats),
                    "relative_x0_cycle_minus_pred_x0": _relative(cycle_norm.feats - pred_x0.feats, pred_x0.feats),
                    "relative_x0_fused_minus_x0_cycle": _relative(fused_norm.feats - cycle_norm.feats, cycle_norm.feats),
                    "relative_x0_guided_minus_pred_x0": _relative(guided_x0.feats - pred_x0.feats, pred_x0.feats),
                    "support_checks": {
                        "pred_x0": _strict_sparse_check(states[tile_id], pred_x0, f"tile {tile_id} pred_x0 saved"),
                        "x0_cycle": cycle_check,
                        "x0_fused": fused_check,
                        "x0_guided": guided_x0_check,
                        "guided_v": guided_v_check,
                        "x_t_next": next_state_check,
                    },
                    "fixed_shape_coord_digest": fixed_shape_digest[tile_id],
                    "fixed_shape_unchanged": fixed_shape_digest[tile_id] == _coordinate_digest(context.shape_norm),
                    "decode": decode_stats[tile_id],
                    "encode": {
                        "cycle": cycle_encode_stats,
                        "fused": fused_encode_stats,
                    },
                    "fusion_seconds": float(time.perf_counter() - fusion_started_tile),
                }
                _atomic_json(
                    context.tile_dir / "steps" / f"step_{step_index:02d}_diagnostics.json",
                    record,
                )
                step_tile_records.append(record)
                del cycle_raw, fused_raw, cycle_norm, fused_norm, delta, guided_x0, guided_v
            fusion_barrier = True
            encode_barrier = True

            # Barrier 5: update all tile states only after every tile's guided
            # endpoint and velocity have been computed.
            for context in contexts:
                states[int(context.tile_id)] = fused_endpoints[int(context.tile_id)]
            update_barrier = True
            for context in contexts:
                if not torch.equal(states[int(context.tile_id)].coords, context.initial_state.coords):
                    raise RuntimeError(f"tile {context.tile_id}: state support changed after Euler update")
            step_summary = {
                "step": int(step_index),
                "t": float(t),
                "t_next": float(t_next),
                "tile_count": int(len(contexts)),
                "prediction_seconds": float(decode_started - prediction_started),
                "decode_seconds": float(fusion_started - decode_started),
                "fusion_encode_seconds": float(time.perf_counter() - fusion_started),
                "step_seconds": float(time.perf_counter() - step_started),
                "barriers": {
                    "prediction_barrier": bool(prediction_barrier),
                    "decoded_field_barrier": bool(decode_barrier),
                    "fusion_barrier": bool(fusion_barrier),
                    "encode_barrier": bool(encode_barrier),
                    "euler_update_barrier": bool(update_barrier),
                    "all_tiles_synchronized": bool(
                        prediction_barrier
                        and decode_barrier
                        and fusion_barrier
                        and encode_barrier
                        and update_barrier
                    ),
                },
                "tiles": step_tile_records,
            }
            _atomic_json(
                Path(args.output_dir).expanduser().resolve() / "steps" / f"step_{step_index:02d}_summary.json",
                step_summary,
            )
            per_step.append(step_summary)
            del predictions, decoded_fields, decoded, fused_endpoints
            _empty_cuda_cache()
    finally:
        if bool(args.low_vram):
            model.cpu()
    for context in contexts:
        context.guided_endpoint = states[int(context.tile_id)]
        _save_sparse_payload(
            context.tile_dir / "cross_tile_pbr_perstep_guided_endpoint.pt",
            context.guided_endpoint,
            pipeline.tex_slat_normalization,
        )
    return {
        "route": (
            "all tiles predict pred_x0/pred_v -> all tiles decode pred_x0 -> frozen decoded fields -> "
            "cross-tile MeshWithVoxel.query_attrs Gaussian PBR fusion -> official self/fused PBR encode -> "
            "cycle-cancelled x0 correction -> official _xstart_to_pred -> synchronized Euler"
        ),
        "native_schedule": schedule,
        "schedule_start_index": int(start_index),
        "noise_timestep": float(args.noise_timestep),
        "flow_steps": int(len(schedule) - 1 - start_index),
        "model_forward_count": int((len(schedule) - 1 - start_index) * len(contexts)),
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
        "visibility_weight_used": False,
        "foreground_weight_used": False,
        "facing_weight_used": False,
        "velocity_averaging_used": False,
    }


@torch.no_grad()
def _run_cross_tile_guided_flow(
    *,
    contexts: Sequence[TileContext],
    pipeline: Any,
    global_camera: Mapping[str, float],
    texture_params: Mapping[str, Any],
    pbr_encoder: torch.nn.Module,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Strict Jacobi/barrier implementation of the requested guided flow.

    The older implementation above is retained as a debugging reference.  In
    this production route, fusion, encoding, endpoint correction, and Euler
    update are distinct phases, each with a barrier across all active tiles.
    """
    if not contexts:
        raise RuntimeError("guided flow requires at least one tile")
    sampler = pipeline.tex_slat_sampler
    model = pipeline.models["tex_slat_flow_model_1024"]
    merged = {**pipeline.tex_slat_sampler_params, **dict(texture_params)}
    schedule = _native_schedule(sampler, merged)
    start_index = _schedule_start(schedule, float(args.noise_timestep))
    step_kwargs = _sampler_step_kwargs(merged)
    states: Dict[int, SparseTensor] = {
        int(context.tile_id): _fresh_sparse(context.initial_state) for context in contexts
    }
    fixed_shape_digest = {
        int(context.tile_id): _coordinate_digest(context.shape_norm) for context in contexts
    }
    per_step: List[Dict[str, Any]] = []
    started = time.perf_counter()
    low_vram = bool(args.low_vram)
    if low_vram:
        model.to(torch.device("cuda"))
    try:
        for step_index, (t, t_next) in enumerate(
            zip(schedule[start_index:-1], schedule[start_index + 1 :]),
            start=start_index,
        ):
            step_started = time.perf_counter()
            print(
                f"[cross-tile step {step_index:02d}] t={float(t):.9f} "
                f"t_next={float(t_next):.9f} tiles={len(contexts)}"
            )

            # Phase A + barrier: predict all clean endpoints and velocities.
            prediction_started = time.perf_counter()
            predictions: Dict[int, Dict[str, SparseTensor]] = {}
            for context in contexts:
                tile_id = int(context.tile_id)
                state = states[tile_id]
                model_state = _sparse_to_device(state, torch.device("cuda")) if low_vram else state
                shape_condition = (
                    _sparse_to_device(context.shape_norm, torch.device("cuda"))
                    if low_vram
                    else context.shape_norm
                )
                condition = _move_condition(context.condition, torch.device("cuda"))
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
                pred_check = _strict_sparse_check(model_state, pred_x0, f"tile {tile_id} step {step_index} pred_x0")
                velocity_check = _strict_sparse_check(model_state, pred_v, f"tile {tile_id} step {step_index} pred_v")
                predictions[tile_id] = {
                    "pred_x0": _sparse_to_cpu(pred_x0) if low_vram else pred_x0,
                    "pred_v": _sparse_to_cpu(pred_v) if low_vram else pred_v,
                    "pred_check": pred_check,
                    "velocity_check": velocity_check,
                }
                del model_state, pred_x0, pred_v
                if low_vram:
                    _empty_cuda_cache()
            prediction_barrier = True

            # Phase B + barrier: decode every pred_x0.  These meshes are the
            # immutable read set for all cross-tile queries in this step.
            decode_started = time.perf_counter()
            decoded: Dict[int, MeshWithVoxel] = {}
            decoded_fields: Dict[int, torch.Tensor] = {}
            decode_stats: Dict[int, Dict[str, Any]] = {}
            for context in contexts:
                tile_id = int(context.tile_id)
                decode_shape = (
                    _sparse_to_device(context.shape_denorm, torch.device("cuda"))
                    if low_vram
                    else context.shape_denorm
                )
                decode_texture = (
                    _sparse_to_device(predictions[tile_id]["pred_x0"], torch.device("cuda"))
                    if low_vram
                    else predictions[tile_id]["pred_x0"]
                )
                decode_points = (
                    context.target_points.to(device="cuda")
                    if low_vram
                    else context.target_points
                )
                mesh, field, stats = _decode_endpoint(
                    pipeline=pipeline,
                    shape_denorm=decode_shape,
                    texture_norm=decode_texture,
                    query_points=decode_points,
                    query_chunk_size=int(args.query_chunk_size),
                    label=f"tile {tile_id:02d} step {step_index:02d} pred_x0",
                )
                if low_vram:
                    mesh_cpu = mesh.to("cpu")
                    field_cpu = field.detach().to("cpu").clone()
                    del mesh, field
                    mesh, field = mesh_cpu, field_cpu
                decoded[tile_id] = mesh
                decoded_fields[tile_id] = field
                decode_stats[tile_id] = stats
                del decode_shape, decode_texture, decode_points
                if low_vram:
                    _empty_cuda_cache()
            decode_barrier = True

            # Phase C + barrier: fuse every target against only the frozen
            # decoded fields.  No endpoint is encoded in this phase.
            fusion_started = time.perf_counter()
            fused_fields: Dict[int, torch.Tensor] = {}
            fusion_stats_by_tile: Dict[int, Dict[str, Any]] = {}
            for context in contexts:
                tile_id = int(context.tile_id)
                fused_field, fusion_stats, donor_details = _fuse_tile_field(
                    target=context,
                    contexts=contexts,
                    decoded=decoded,
                    self_field=decoded_fields[tile_id],
                    global_camera=global_camera,
                    sigma_pixels=float(args.fusion_sigma_pixels),
                    query_chunk_size=int(args.query_chunk_size),
                )
                fused_fields[tile_id] = fused_field
                fusion_stats_by_tile[tile_id] = fusion_stats
                if bool(args.save_donor_details):
                    _save_fusion_details(
                        context.tile_dir / "steps" / f"step_{step_index:02d}_donors.pt",
                        donor_details,
                        {
                            "format": FORMAT,
                            "step": int(step_index),
                            "t": float(t),
                            "t_next": float(t_next),
                            "target_tile": tile_id,
                            "tile_center": [511.5, 511.5],
                            "weight_rule": "exp(-distance_to_tile_center_pixels^2/(2*sigma_pixels^2))",
                            "sigma_pixels": float(args.fusion_sigma_pixels),
                            "visibility_weight_used": False,
                            "foreground_weight_used": False,
                            "facing_weight_used": False,
                            "G_guidance_used": False,
                        },
                    )
            fusion_barrier = True

            # Phase D + barrier: encode self and fused PBR fields for every
            # tile on exactly the original local C1024 support.
            encode_started = time.perf_counter()
            encoded: Dict[int, Dict[str, Any]] = {}
            for context in contexts:
                tile_id = int(context.tile_id)
                cycle_raw, cycle_stats = core._encode_local_pbr(
                    encoder=pbr_encoder,
                    coords=context.geometry.coords,
                    attrs=decoded_fields[tile_id],
                    device=torch.device("cuda"),
                    low_vram=bool(args.low_vram),
                )
                fused_raw, fused_stats = core._encode_local_pbr(
                    encoder=pbr_encoder,
                    coords=context.geometry.coords,
                    attrs=fused_fields[tile_id],
                    device=torch.device("cuda"),
                    low_vram=bool(args.low_vram),
                )
                cycle_norm = _normalize_slat(cycle_raw, pipeline.tex_slat_normalization)
                fused_norm = _normalize_slat(fused_raw, pipeline.tex_slat_normalization)
                if low_vram:
                    cycle_norm = _sparse_to_cpu(cycle_norm)
                    fused_norm = _sparse_to_cpu(fused_norm)
                pred_x0 = predictions[tile_id]["pred_x0"]
                cycle_check = _strict_sparse_check(pred_x0, cycle_norm, f"tile {tile_id} step {step_index} x0_cycle")
                fused_check = _strict_sparse_check(pred_x0, fused_norm, f"tile {tile_id} step {step_index} x0_fused")
                encoded[tile_id] = {
                    "cycle_norm": cycle_norm,
                    "fused_norm": fused_norm,
                    "cycle_stats": cycle_stats,
                    "fused_stats": fused_stats,
                    "cycle_check": cycle_check,
                    "fused_check": fused_check,
                }
                del cycle_raw, fused_raw
            encode_barrier = True

            # Phase E + barrier: correct every endpoint and calculate every
            # next state before any tile state is replaced.
            correction_started = time.perf_counter()
            corrected: Dict[int, Dict[str, Any]] = {}
            for context in contexts:
                tile_id = int(context.tile_id)
                pred_x0 = predictions[tile_id]["pred_x0"]
                pred_v = predictions[tile_id]["pred_v"]
                cycle_norm = encoded[tile_id]["cycle_norm"]
                fused_norm = encoded[tile_id]["fused_norm"]
                model_state = _sparse_to_device(states[tile_id], torch.device("cuda")) if low_vram else states[tile_id]
                if low_vram:
                    pred_x0 = _sparse_to_device(pred_x0, torch.device("cuda"))
                    pred_v = _sparse_to_device(pred_v, torch.device("cuda"))
                    cycle_norm = _sparse_to_device(cycle_norm, torch.device("cuda"))
                    fused_norm = _sparse_to_device(fused_norm, torch.device("cuda"))
                guided_x0 = SparseTensor(
                    pred_x0.feats + float(args.eta) * (fused_norm.feats - cycle_norm.feats),
                    pred_x0.coords.detach().clone(),
                )
                guided_x0_check = _strict_sparse_check(pred_x0, guided_x0, f"tile {tile_id} step {step_index} x0_guided")
                guided_v = sampler._xstart_to_pred(model_state, float(t), guided_x0)
                guided_v_check = _strict_sparse_check(pred_v, guided_v, f"tile {tile_id} step {step_index} guided_v")
                next_state = SparseTensor(
                    model_state.feats - float(t - t_next) * guided_v.feats,
                    model_state.coords.detach().clone(),
                )
                next_check = _strict_sparse_check(model_state, next_state, f"tile {tile_id} step {step_index} x_t_next")
                corrected[tile_id] = {
                    "next_state": _sparse_to_cpu(next_state) if low_vram else next_state,
                    "pred_x0": _sparse_to_cpu(pred_x0) if low_vram else pred_x0,
                    "pred_v": _sparse_to_cpu(pred_v) if low_vram else pred_v,
                    "cycle_norm": _sparse_to_cpu(cycle_norm) if low_vram else cycle_norm,
                    "fused_norm": _sparse_to_cpu(fused_norm) if low_vram else fused_norm,
                    "guided_x0": _sparse_to_cpu(guided_x0) if low_vram else guided_x0,
                    "guided_v": _sparse_to_cpu(guided_v) if low_vram else guided_v,
                    "guided_x0_check": guided_x0_check,
                    "guided_v_check": guided_v_check,
                    "next_check": next_check,
                }
                del model_state, pred_x0, pred_v, cycle_norm, fused_norm, guided_x0, guided_v, next_state
                if low_vram:
                    _empty_cuda_cache()
            correction_barrier = True

            # Phase F + barrier: synchronized Jacobi Euler update.
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
                stats = fusion_stats_by_tile[tile_id]
                pred_x0 = row["pred_x0"]
                pred_v = row["pred_v"]
                cycle_norm = row["cycle_norm"]
                fused_norm = row["fused_norm"]
                guided_x0 = row["guided_x0"]
                guided_v = row["guided_v"]
                record = {
                    "tile_id": tile_id,
                    "step": int(step_index),
                    "t": float(t),
                    "t_next": float(t_next),
                    "active_ovoxel_count": int(context.target_coords.shape[0]),
                    "overlap_ovoxel_count": int(stats["overlap_ovoxel_count"]),
                    "non_overlap_ovoxel_count": int(stats["non_overlap_ovoxel_count"]),
                    "donor_count": stats["query_valid_donor_count"],
                    "covered_donor_count": stats["covered_donor_count"],
                    "distance_to_center_statistics": stats["distance_to_center_pixels"],
                    "fusion_weight_statistics": stats["normalized_fusion_weight"],
                    "pbr_self_vs_fused_mean_abs": stats["pbr_self_vs_fused_mean_abs_overlap"],
                    "pbr_self_vs_fused_mean_abs_all": stats["pbr_self_vs_fused_mean_abs_all"],
                    "norm_pred_x0": _norm(pred_x0.feats),
                    "norm_x0_cycle_minus_pred_x0": _norm(cycle_norm.feats - pred_x0.feats),
                    "norm_x0_fused_minus_x0_cycle": _norm(fused_norm.feats - cycle_norm.feats),
                    "norm_x0_guided_minus_pred_x0": _norm(guided_x0.feats - pred_x0.feats),
                    "norm_guided_v_minus_pred_v": _norm(guided_v.feats - pred_v.feats),
                    "relative_x0_cycle_minus_pred_x0": _relative(cycle_norm.feats - pred_x0.feats, pred_x0.feats),
                    "relative_x0_fused_minus_x0_cycle": _relative(fused_norm.feats - cycle_norm.feats, cycle_norm.feats),
                    "relative_x0_guided_minus_pred_x0": _relative(guided_x0.feats - pred_x0.feats, pred_x0.feats),
                    "support_checks": {
                        "pred_x0": predictions[tile_id]["pred_check"],
                        "pred_v": predictions[tile_id]["velocity_check"],
                        "x0_cycle": encoded[tile_id]["cycle_check"],
                        "x0_fused": encoded[tile_id]["fused_check"],
                        "x0_guided": row["guided_x0_check"],
                        "guided_v": row["guided_v_check"],
                        "x_t_next": row["next_check"],
                    },
                    "fixed_shape_coord_digest": fixed_shape_digest[tile_id],
                    "fixed_shape_unchanged": fixed_shape_digest[tile_id] == _coordinate_digest(context.shape_norm),
                    "decode": decode_stats[tile_id],
                    "encode": {
                        "cycle": encoded[tile_id]["cycle_stats"],
                        "fused": encoded[tile_id]["fused_stats"],
                    },
                    "phase_seconds": {
                        "prediction": float(decode_started - prediction_started),
                        "decode": float(fusion_started - decode_started),
                        "fusion": float(encode_started - fusion_started),
                        "encode": float(correction_started - encode_started),
                        "correction_and_euler": float(time.perf_counter() - correction_started),
                    },
                }
                _atomic_json(context.tile_dir / "steps" / f"step_{step_index:02d}_diagnostics.json", record)
                step_tile_records.append(record)

            step_summary = {
                "step": int(step_index),
                "t": float(t),
                "t_next": float(t_next),
                "tile_count": int(len(contexts)),
                "step_seconds": float(time.perf_counter() - step_started),
                "barriers": {
                    "prediction_barrier": bool(prediction_barrier),
                    "decoded_field_barrier": bool(decode_barrier),
                    "fusion_barrier": bool(fusion_barrier),
                    "encode_barrier": bool(encode_barrier),
                    "endpoint_correction_barrier": bool(correction_barrier),
                    "euler_update_barrier": bool(update_barrier),
                    "all_tiles_synchronized": bool(
                        prediction_barrier
                        and decode_barrier
                        and fusion_barrier
                        and encode_barrier
                        and correction_barrier
                        and update_barrier
                    ),
                },
                "tiles": step_tile_records,
            }
            _atomic_json(
                Path(args.output_dir).expanduser().resolve() / "steps" / f"step_{step_index:02d}_summary.json",
                step_summary,
            )
            per_step.append(step_summary)
            del predictions, decoded, decoded_fields, fused_fields, fusion_stats_by_tile, encoded, corrected
            _empty_cuda_cache()
    finally:
        if bool(args.low_vram):
            model.cpu()
    for context in contexts:
        context.guided_endpoint = states[int(context.tile_id)]
        _save_sparse_payload(
            context.tile_dir / "cross_tile_pbr_perstep_guided_endpoint.pt",
            context.guided_endpoint,
            pipeline.tex_slat_normalization,
        )
    return {
        "route": (
            "all tiles predict endpoint -> all decode pred_x0 PBR -> frozen field barrier -> all PBR fusion -> "
            "all self/fused encode -> all cycle-cancelled endpoint correction -> all _xstart_to_pred -> Euler"
        ),
        "native_schedule": schedule,
        "schedule_start_index": int(start_index),
        "noise_timestep": float(args.noise_timestep),
        "flow_steps": int(len(schedule) - 1 - start_index),
        "model_forward_count": int((len(schedule) - 1 - start_index) * len(contexts)),
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
        "visibility_weight_used": False,
        "foreground_weight_used": False,
        "facing_weight_used": False,
        "velocity_averaging_used": False,
    }


def _variant_patch_and_stitch(
    *,
    variant: str,
    contexts: Sequence[TileContext],
    pipeline: Any,
    global_camera: Mapping[str, float],
    baseline_mesh: MeshWithVoxel,
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[MeshWithVertexPbr, Dict[str, Any], Dict[str, Any]]:
    """Decode one final endpoint per tile and apply the established stitcher."""
    if variant not in {"pure_HR", "cross_tile_pbr_perstep_guided"}:
        raise ValueError(f"unknown final variant: {variant}")
    endpoint_name = "pure_endpoint" if variant == "pure_HR" else "guided_endpoint"
    patches: List[core.ReturnedTilePatch] = []
    tile_records: List[Dict[str, Any]] = []
    for index, context in enumerate(contexts):
        endpoint = getattr(context, endpoint_name)
        if endpoint is None:
            raise RuntimeError(f"tile {context.tile_id}: missing {variant} final endpoint")
        print(
            f"[{variant}] final decode tile {context.tile_id:02d} "
            f"({index + 1}/{len(contexts)})"
        )
        low_vram = bool(args.low_vram)
        decode_shape = (
            _sparse_to_device(context.shape_denorm, torch.device("cuda"))
            if low_vram
            else context.shape_denorm
        )
        decode_texture = (
            _sparse_to_device(endpoint, torch.device("cuda"))
            if low_vram
            else endpoint
        )
        decode_points = (
            context.target_points[:0].to(device="cuda")
            if low_vram
            else context.target_points[:0]
        )
        mesh, _, decode_stats = _decode_endpoint(
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
    variant_dir = output_dir / "variants" / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": f"{FORMAT}_{variant}_global_mesh",
            "variant": variant,
            "mesh": stitched,
            "stitch_stats": stitch_stats,
            "tile_records": tile_records,
        },
        variant_dir / "global_merged_mesh.pt",
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
        [exported_patch],
        variant_dir / "global_merged_mesh.glb",
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
    return stitched, summary, {"patches": patches}


def _save_variant_comparison(
    *,
    input_path: Path,
    baseline_path: Optional[Path],
    pure_path: Path,
    guided_path: Path,
    metrics: Mapping[str, Mapping[str, Any]],
    output_path: Path,
) -> None:
    entries = [(input_path, "input")]
    if baseline_path is not None:
        entries.append((baseline_path, "global_baseline"))
    entries.extend([(pure_path, "pure_HR"), (guided_path, "cross_tile_pbr_perstep_guided")])
    panel_size = 420
    header = 72
    canvas = Image.new("RGB", (panel_size * len(entries), panel_size + header), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (path, label) in enumerate(entries):
        with Image.open(path) as image:
            panel = ImageOps.contain(image.convert("RGB"), (panel_size, panel_size))
        x = index * panel_size + (panel_size - panel.width) // 2
        canvas.paste(panel, (x, header + (panel_size - panel.height) // 2))
        draw.text((index * panel_size + 8, 8), label, fill=(255, 255, 255))
        row = metrics.get(label)
        if row is not None:
            draw.text(
                (index * panel_size + 8, 31),
                f"PSNR {row.get('psnr_db')}  SSIM {row.get('ssim')}",
                fill=(220, 220, 220),
            )
            draw.text(
                (index * panel_size + 8, 49),
                f"LPIPS {row.get('lpips')}",
                fill=(190, 190, 190),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


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


def _make_contact_sheet(paths: Sequence[Path], labels: Sequence[str], output_path: Path) -> None:
    if not paths:
        return
    panel = 320
    header = 38
    columns = 3
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * panel, rows * (panel + header)), "black")
    draw = ImageDraw.Draw(sheet)
    for index, (path, label) in enumerate(zip(paths, labels)):
        with Image.open(path) as image:
            image = ImageOps.contain(image.convert("RGB"), (panel - 8, panel - 8))
        x = (index % columns) * panel + (panel - image.width) // 2
        y = (index // columns) * (panel + header)
        sheet.paste(image, (x, y + header + (panel - image.height) // 2))
        draw.text(((index % columns) * panel + 5, y + 10), label, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _render_multiview_pair(
    *,
    pure_mesh: MeshWithVertexPbr,
    guided_mesh: MeshWithVertexPbr,
    output_dir: Path,
    camera: Mapping[str, float],
    args: argparse.Namespace,
    envmap: Any,
) -> Dict[str, Any]:
    """Render fixed views plus turntable comparison for pure_HR vs guided."""
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
    options = {
        "resolution": int(args.multiview_resolution),
        "near": max(0.01, radius - 2.0),
        "far": radius + 10.0,
        "ssaa": int(args.multiview_ssaa),
        "peel_layers": int(args.multiview_peel_layers),
        "face_chunk_size": int(args.render_face_chunk_size),
    }
    renderer = render_utils.get_renderer(pure_mesh, **options)

    def render(mesh: MeshWithVertexPbr) -> List[Image.Image]:
        live = mesh.to(device)
        frames = render_utils.render_frames(
            live,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            options=options,
            verbose=True,
            renderer=renderer,
            envmap=envmap,
            use_envmap_bg=bool(args.use_envmap_bg),
        ).get("shaded")
        del live
        _empty_cuda_cache()
        if frames is None or len(frames) != len(labels):
            raise RuntimeError("multiview renderer returned an incomplete frame list")
        return [_frame_to_image(frame) for frame in frames]

    pure_frames = render(pure_mesh)
    guided_frames = render(guided_mesh)
    pure_paths: List[Path] = []
    guided_paths: List[Path] = []
    pair_paths: List[Path] = []
    pair_images: List[Image.Image] = []
    for index, (pure, guided, label) in enumerate(zip(pure_frames, guided_frames, labels)):
        pure_path = output_dir / f"view_{index:03d}_pure_HR.png"
        guided_path = output_dir / f"view_{index:03d}_cross_tile_pbr_perstep_guided.png"
        pair_path = output_dir / f"view_{index:03d}_pure_HR_vs_cross_tile_pbr_perstep_guided.png"
        pure.save(pure_path)
        guided.save(guided_path)
        pair = Image.new("RGB", (pure.width * 2, pure.height), "black")
        pair.paste(pure, (0, 0))
        pair.paste(guided, (pure.width, 0))
        pair.save(pair_path)
        pure_paths.append(pure_path)
        guided_paths.append(guided_path)
        pair_paths.append(pair_path)
        pair_images.append(pair)
    pure_sheet = output_dir / "pure_HR_multiview_sheet.png"
    guided_sheet = output_dir / "cross_tile_pbr_perstep_guided_multiview_sheet.png"
    comparison_sheet = output_dir / "pure_HR_vs_cross_tile_pbr_perstep_guided_sheet.png"
    _make_contact_sheet(pure_paths, labels, pure_sheet)
    _make_contact_sheet(guided_paths, labels, guided_sheet)
    _make_contact_sheet(pair_paths, labels, comparison_sheet)
    gif_path = output_dir / "pure_HR_vs_cross_tile_pbr_perstep_guided_turntable.gif"
    turntable_pairs = pair_images[len(fixed) :]
    if turntable_pairs:
        turntable_pairs[0].save(
            gif_path,
            save_all=True,
            append_images=turntable_pairs[1:],
            duration=100,
            loop=0,
        )
    return {
        "enabled": True,
        "renderer": "Pixal3D render_utils / nvdiffrast",
        "camera_policy": "fixed global camera trajectory shared by pure_HR and guided",
        "fixed_views": [name for name, _, _ in fixed],
        "turntable_frames": int(turntable_count),
        "pure_HR_frame_pngs": [str(path) for path in pure_paths],
        "cross_tile_pbr_perstep_guided_frame_pngs": [str(path) for path in guided_paths],
        "comparison_frame_pngs": [str(path) for path in pair_paths],
        "pure_HR_sheet": str(pure_sheet),
        "cross_tile_pbr_perstep_guided_sheet": str(guided_sheet),
        "comparison_sheet": str(comparison_sheet),
        "turntable_gif": str(gif_path) if gif_path.is_file() else None,
    }


def _write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "vertices", "faces", "PSNR", "SSIM", "LPIPS"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_report(output_dir: Path, summary: Mapping[str, Any]) -> None:
    evaluation = summary.get("evaluation", {})
    rows = evaluation.get("table", []) if isinstance(evaluation, Mapping) else []
    lines = [
        "# Pixal3D Cross-Tile PBR Per-Step Guidance",
        "",
        "## Route",
        "",
        "- canonical image: 4096x4096; tile: 1024x1024; stride: 512; active tiles: 49",
        "- fixed local shape SLat; no tile shape flow or tile shape sampler",
        "- Jacobi order: predict clean endpoint -> decode all PBR fields -> freeze -> cross-tile query/fusion -> self/fused encode -> cycle-cancelled endpoint -> Euler",
        "- Gaussian center weight only; no G, visibility, foreground, background, facing or velocity averaging weight",
        "",
        "## Evaluation",
        "",
        "| variant | vertices | faces | PSNR | SSIM | LPIPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('variant')} | {row.get('vertices')} | {row.get('faces')} | "
            f"{row.get('PSNR')} | {row.get('SSIM')} | {row.get('LPIPS')} |"
        )
    lines.extend(
        [
            "",
            "## Strict route checks",
            "",
        ]
    )
    for key, value in summary.get("route_checks", {}).items():
        lines.append(f"- `{key}` = `{value}`")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="assets/choose/0_img.png")
    parser.add_argument(
        "--output-dir",
        default="outputs/cross_tile_pbr_perstep_guided_cuda4",
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--baseline-dir", default=None)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-ids", default=None, help="debug-only comma-separated tile ids")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--low-vram",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="explicit recovery mode; omitted by default to keep normal/full VRAM mode",
    )
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
    parser.add_argument("--material-query-chunk-size", type=int, default=65_536)
    parser.add_argument("--material-face-chunk-size", type=int, default=16_384)
    parser.add_argument("--query-chunk-size", type=int, default=65_536)
    parser.add_argument("--roundtrip-tolerance", type=float, default=2e-5)
    parser.add_argument("--noise-timestep", type=float, default=1.0)
    parser.add_argument("--noise-strength", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument(
        "--fusion-sigma-pixels",
        type=float,
        default=256.0,
        help="fixed Gaussian tile-center sigma in 1024-tile pixels",
    )
    parser.add_argument(
        "--save-donor-details",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--stitch-tolerance", type=float, default=1.0 / OVOXEL_RESOLUTION)

    # Keep the official baseline/sampler defaults exposed for reproducibility.
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
    parser.add_argument("--render-multiview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multiview-resolution", type=int, default=512)
    parser.add_argument("--multiview-ssaa", type=int, default=1)
    parser.add_argument("--multiview-peel-layers", type=int, default=4)
    parser.add_argument("--multiview-radius-scale", type=float, default=1.0)
    parser.add_argument("--multiview-turntable-frames", type=int, default=24)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    if int(args.cuda_device) < 0 or int(args.cuda_device) >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {args.cuda_device} is unavailable")
    if not Path(args.image).expanduser().is_file():
        raise FileNotFoundError(args.image)
    for encoder in (args.shape_encoder, args.pbr_encoder):
        base = Path(encoder).expanduser()
        if not Path(f"{base}.json").is_file() or not Path(f"{base}.safetensors").is_file():
            raise FileNotFoundError(f"encoder checkpoint pair not found for {base}")
    for name in (
        "face_projection_chunk_size",
        "material_query_chunk_size",
        "material_face_chunk_size",
        "query_chunk_size",
        "max_num_tokens",
        "render_resolution",
        "metric_resolution",
        "render_ssaa",
        "render_peel_layers",
        "render_face_chunk_size",
        "multiview_resolution",
        "multiview_ssaa",
        "multiview_peel_layers",
        "multiview_turntable_frames",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 <= float(args.noise_timestep) <= 1.0:
        raise ValueError("--noise-timestep must lie in [0,1]")
    if float(args.noise_strength) <= 0.0:
        raise ValueError("--noise-strength must be positive")
    if float(args.eta) < 0.0:
        raise ValueError("--eta must be non-negative")
    if float(args.fusion_sigma_pixels) <= 0.0:
        raise ValueError("--fusion-sigma-pixels must be positive")
    if float(args.stitch_tolerance) <= 0.0:
        raise ValueError("--stitch-tolerance must be positive")
    if float(args.multiview_radius_scale) <= 0.0:
        raise ValueError("--multiview-radius-scale must be positive")
    if not bool(args.skip_lpips) and importlib.util.find_spec("lpips") is None:
        print("[metrics] lpips is unavailable; continuing without LPIPS")
        args.skip_lpips = True
    if bool(args.low_vram):
        print("[warning] explicit --low-vram enabled; normal/full mode is the default")


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
    output_dir.mkdir(parents=True, exist_ok=True)
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
    baseline_camera_path = (
        Path(args.baseline_dir).expanduser() / "global_camera.json"
        if args.baseline_dir
        else None
    )
    if bool(args.resume) and camera_path.is_file():
        global_camera = json.loads(camera_path.read_text(encoding="utf-8"))
    elif baseline_camera_path is not None and baseline_camera_path.is_file():
        global_camera = json.loads(baseline_camera_path.read_text(encoding="utf-8"))
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

    baseline_mesh, baseline_summary = _load_or_run_global_baseline(
        args=args,
        pipeline=pipeline,
        image_1024=image_1024,
        global_camera=global_camera,
        output_dir=output_dir,
    )
    baseline_mesh = baseline_mesh.to("cpu")
    _atomic_json(output_dir / "global_baseline_summary.json", baseline_summary)
    global_attr_field = core._make_attribute_query_mesh(
        baseline_mesh,
        device,
    )

    boxes = core._tile_layout(
        canonical_size=CANONICAL_IMAGE_SIZE,
        tile_size=TILE_SIZE,
        stride=TILE_STRIDE,
    )
    if _parse_ids(args.tile_ids) is None and len(boxes) != 49:
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

    print("[prepare] loading official fixed-shape and PBR encoders")
    shape_encoder = pixal3d_models.from_pretrained(str(Path(args.shape_encoder).expanduser())).eval()
    pbr_encoder = pixal3d_models.from_pretrained(str(Path(args.pbr_encoder).expanduser())).eval()
    if not bool(args.low_vram):
        shape_encoder.to(device)
        pbr_encoder.to(device)
    contexts = _prepare_tile_contexts(
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
    if bool(args.low_vram):
        _offload_contexts_to_cpu(contexts)
        print(f"[prepare] CPU-staged {len(contexts)} active tile contexts for low-VRAM barriers")
    selected_ids = sorted(int(context.tile_id) for context in contexts)
    requested_ids = _parse_ids(args.tile_ids)
    preparation_summary_path = output_dir / "tile_preparation_summary.json"
    preparation_summary = (
        json.loads(preparation_summary_path.read_text(encoding="utf-8"))
        if preparation_summary_path.is_file()
        else {}
    )
    skipped_empty_ids = [int(v) for v in preparation_summary.get("skipped_tile_ids", [])]
    # "All valid tiles" means every requested/default layout tile was checked;
    # empty background crops are recorded as invalid and do not enter flow.
    expected_all_tiles = requested_ids is None
    del global_attr_field, shape_encoder
    _empty_cuda_cache()

    texture_params = core._sampler_overrides(args)[2]
    pure_stats = _run_pure_hr_flow(
        contexts=contexts,
        pipeline=pipeline,
        texture_params=texture_params,
        args=args,
    )
    guided_stats = _run_cross_tile_guided_flow(
        contexts=contexts,
        pipeline=pipeline,
        global_camera=global_camera,
        texture_params=texture_params,
        pbr_encoder=pbr_encoder,
        args=args,
    )
    del pbr_encoder
    _empty_cuda_cache()

    pure_mesh, pure_summary, _ = _variant_patch_and_stitch(
        variant="pure_HR",
        contexts=contexts,
        pipeline=pipeline,
        global_camera=global_camera,
        baseline_mesh=baseline_mesh,
        args=args,
        output_dir=output_dir,
    )
    guided_mesh, guided_summary, _ = _variant_patch_and_stitch(
        variant="cross_tile_pbr_perstep_guided",
        contexts=contexts,
        pipeline=pipeline,
        global_camera=global_camera,
        baseline_mesh=baseline_mesh,
        args=args,
        output_dir=output_dir,
    )

    # The requested root-level artifacts are the guided final mesh; variant
    # directories retain both pure_HR and guided files for direct comparison.
    _atomic_torch_save(
        output_dir / "global_merged_mesh.pt",
        {
            "format": f"{FORMAT}_guided_global_mesh",
            "variant": "cross_tile_pbr_perstep_guided",
            "mesh": guided_mesh,
            "stitch_stats": guided_summary["stitch"],
        },
    )
    guided_glb = output_dir / "variants" / "cross_tile_pbr_perstep_guided" / "global_merged_mesh.glb"
    root_glb = output_dir / "global_merged_mesh.glb"
    if guided_glb.is_file():
        shutil.copy2(guided_glb, root_glb)
    else:
        manifest = guided_glb.with_name(f"{guided_glb.stem}_manifest.json")
        if manifest.is_file():
            shutil.copy2(manifest, output_dir / "global_merged_mesh_manifest.json")

    evaluation_table: List[Dict[str, Any]] = []
    render_records: Dict[str, Any] = {}
    multiview_record: Dict[str, Any] = {"enabled": False}
    comparison_path = output_dir / "input_baseline_pure_HR_guided_comparison.png"
    if bool(args.render):
        envmap = core.load_envmap(str(args.envmap), device="cuda")
        baseline_render = core._render(
            baseline_mesh,
            output_dir=output_dir / "global_baseline_1024" / "aligned_eval",
            camera=global_camera,
            reference_image=output_dir / "canonical_1024.png",
            args=args,
            envmap=envmap,
        )
        pure_render = core._render(
            pure_mesh,
            output_dir=output_dir / "variants" / "pure_HR" / "aligned_eval",
            camera=global_camera,
            reference_image=output_dir / "canonical_1024.png",
            args=args,
            envmap=envmap,
        )
        guided_render = core._render(
            guided_mesh,
            output_dir=output_dir / "variants" / "cross_tile_pbr_perstep_guided" / "aligned_eval",
            camera=global_camera,
            reference_image=output_dir / "canonical_1024.png",
            args=args,
            envmap=envmap,
        )
        render_records = {
            "global_baseline": baseline_render,
            "pure_HR": pure_render,
            "cross_tile_pbr_perstep_guided": guided_render,
        }
        evaluation_table = [
            {
                "variant": "global_baseline",
                "vertices": int(baseline_mesh.vertices.shape[0]),
                "faces": int(baseline_mesh.faces.shape[0]),
                "PSNR": core._metric_subset(baseline_render)["psnr_db"],
                "SSIM": core._metric_subset(baseline_render)["ssim"],
                "LPIPS": core._metric_subset(baseline_render)["lpips"],
            },
            {
                "variant": "pure_HR",
                "vertices": int(pure_mesh.vertices.shape[0]),
                "faces": int(pure_mesh.faces.shape[0]),
                "PSNR": core._metric_subset(pure_render)["psnr_db"],
                "SSIM": core._metric_subset(pure_render)["ssim"],
                "LPIPS": core._metric_subset(pure_render)["lpips"],
            },
            {
                "variant": "cross_tile_pbr_perstep_guided",
                "vertices": int(guided_mesh.vertices.shape[0]),
                "faces": int(guided_mesh.faces.shape[0]),
                "PSNR": core._metric_subset(guided_render)["psnr_db"],
                "SSIM": core._metric_subset(guided_render)["ssim"],
                "LPIPS": core._metric_subset(guided_render)["lpips"],
            },
        ]
        metric_lookup = {
            "global_baseline": core._metric_subset(baseline_render),
            "pure_HR": core._metric_subset(pure_render),
            "cross_tile_pbr_perstep_guided": core._metric_subset(guided_render),
        }
        _save_variant_comparison(
            input_path=output_dir / "canonical_1024.png",
            baseline_path=Path(str(baseline_render["render_png"])),
            pure_path=Path(str(pure_render["render_png"])),
            guided_path=Path(str(guided_render["render_png"])),
            metrics=metric_lookup,
            output_path=comparison_path,
        )
        if bool(args.render_multiview):
            multiview_record = _render_multiview_pair(
                pure_mesh=pure_mesh,
                guided_mesh=guided_mesh,
                output_dir=output_dir / "multiview",
                camera=global_camera,
                args=args,
                envmap=envmap,
            )
        del envmap
        _empty_cuda_cache()
    else:
        evaluation_table = [
            {
                "variant": "pure_HR",
                "vertices": int(pure_mesh.vertices.shape[0]),
                "faces": int(pure_mesh.faces.shape[0]),
                "PSNR": None,
                "SSIM": None,
                "LPIPS": None,
            },
            {
                "variant": "cross_tile_pbr_perstep_guided",
                "vertices": int(guided_mesh.vertices.shape[0]),
                "faces": int(guided_mesh.faces.shape[0]),
                "PSNR": None,
                "SSIM": None,
                "LPIPS": None,
            },
        ]

    _write_metrics_csv(output_dir / "metrics.csv", evaluation_table)
    _atomic_json(output_dir / "metrics.json", {"table": evaluation_table, "renders": render_records})
    route_checks = {
        "shape_flow_called": False,
        "shape_sampler_called": False,
        "fixed_shape_unchanged": all(
            bool(context.static_stats["fixed_shape"]["support_unchanged"])
            for context in contexts
        ),
        "G_guidance_used": False,
        "visibility_weight_used": False,
        "foreground_weight_used": False,
        "background_weight_used": False,
        "facing_weight_used": False,
        "velocity_averaging_used": False,
        "all_tiles_synchronized_per_step": bool(
            guided_stats["all_tiles_synchronized_per_step"]
        ),
        "all_active_tiles_participated": bool(expected_all_tiles),
        "official_texture_sampler": True,
        "official_texture_decoder": True,
        "official_texture_encoder": True,
        "official_meshwithvoxel_query": True,
        "cycle_cancelled_residual_used": True,
        "no_training": True,
    }
    summary: Dict[str, Any] = {
        "format": FORMAT,
        "image": str(source_path),
        "cuda_device": int(args.cuda_device),
        "cuda_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "low_vram": bool(args.low_vram),
        "seed": int(args.seed),
        "global_camera": global_camera,
        "global_baseline": baseline_summary,
        "tile_layout": {
            "canonical_image_size": CANONICAL_IMAGE_SIZE,
            "tile_size": TILE_SIZE,
            "stride": TILE_STRIDE,
            "tile_count": len(boxes),
            "participating_tile_ids": selected_ids,
            "skipped_invalid_empty_tile_ids": skipped_empty_ids,
            "all_tiles_participated": bool(expected_all_tiles),
            "boxes": boxes,
        },
        "fixed_shape": {
            "source": "global baseline mesh projection -> exact global/local camera -> local C1024 dual-grid -> official shape encoder",
            "shape_flow_called": False,
            "shape_sampler_called": False,
            "per_tile_supports_may_differ": True,
            "token_matching_across_tiles": False,
        },
        "guidance": {
            "eta": float(args.eta),
            "fusion_weight": "Gaussian distance to tile center",
            "fusion_sigma_pixels": float(args.fusion_sigma_pixels),
            "tile_center_pixels": [511.5, 511.5],
            "pbr_channels": ["RGB", "metallic", "roughness", "alpha"],
            "overlap_only": True,
            "non_overlap_identity": True,
            "donor_query": "exact global->donor local transform followed by MeshWithVoxel.query_attrs trilinear query",
            "self_candidate_included": True,
            "G_guidance_used": False,
            "visibility_weight_used": False,
            "foreground_weight_used": False,
            "background_weight_used": False,
            "facing_weight_used": False,
            "velocity_averaging_used": False,
            "cycle_cancelled": "x0_guided=pred_x0+eta*(x0_fused-x0_cycle)",
        },
        "sampler": {
            "texture": texture_params,
            "noise_timestep": float(args.noise_timestep),
            "noise_strength": float(args.noise_strength),
            "route": "official texture CFG/timestep/Euler; _get_model_prediction -> _xstart_to_pred",
        },
        "pure_HR": pure_stats,
        "cross_tile_pbr_perstep_guided": guided_stats,
        "route_checks": route_checks,
        "evaluation": {
            "reference": str((output_dir / "canonical_1024.png").resolve()),
            "table": evaluation_table,
            "renders": render_records,
            "comparison": str(comparison_path) if comparison_path.is_file() else None,
            "multiview": multiview_record,
        },
        "variants": {
            "pure_HR": pure_summary,
            "cross_tile_pbr_perstep_guided": guided_summary,
        },
        "artifacts": {
            "global_baseline_mesh": str((output_dir / "global_baseline_mesh.pt").resolve()),
            "global_merged_mesh_pt": str((output_dir / "global_merged_mesh.pt").resolve()),
            "global_merged_mesh_glb": str(root_glb.resolve()) if root_glb.is_file() else None,
            "metrics_csv": str((output_dir / "metrics.csv").resolve()),
            "steps_directory": str((output_dir / "steps").resolve()),
        },
    }
    _write_report(output_dir, summary)
    summary["report_markdown"] = str((output_dir / "summary.md").resolve())
    _atomic_json(output_dir / "summary.json", summary)
    print(
        f"[done] tiles={len(contexts)} guided_steps={guided_stats['flow_steps']} "
        f"guided_mesh_vertices={guided_summary['vertices']:,} "
        f"guided_mesh_faces={guided_summary['faces']:,} "
        f"summary={output_dir / 'summary.json'}"
    )
    return summary


def main() -> None:
    run(_build_parser().parse_args())


if __name__ == "__main__":
    main()
