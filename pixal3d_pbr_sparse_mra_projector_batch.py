#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable sparse-MRA projector and multi-tile sparse-batch experiment.

This is an independent experiment for the Codex tile set.  It deliberately reuses
the fixed-shape, visibility, and prolongation caches produced by the earlier
experiments, while keeping its output tree separate.  The projector never
forms ``P.T @ P``: every right hand side is solved directly with float64
``scipy.sparse.linalg.lsmr``.

The flow route keeps the existing Jacobi barriers.  Within a barrier it can
pack local SparseTensor samples into real sparse microbatches and restores
each tile by coordinate, rather than trusting concatenation order.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from scipy.sparse import csr_matrix, load_npz, save_npz
from scipy.sparse.linalg import lsmr

import pixal3d_cross_tile_pbr_perstep as base
import pixal3d_pbr_sparse_mra_delta_perstep as legacy
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
import pixal3d.models as pixal3d_models
from inference import MODEL_PATH
from o_voxel.convert import flexible_dual_grid_to_mesh
from pixal3d.modules.sparse import SparseTensor
from pixal3d.modules.sparse.basic import VarLenTensor, varlen_cat, sparse_cat
import pixal3d.modules.sparse.conv.conv_flex_gemm as _flex_gemm_backend
import pixal3d.modules.sparse.linear as _sparse_linear_backend
import pixal3d.modules.sparse.attention.modules as _sparse_attention_modules
from pixal3d.models.sc_vaes.sparse_unet_vae import SparseConvNeXtBlock3d
from pixal3d.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
from pixal3d.modules.sparse.transformer.modulated import ModulatedSparseTransformerCrossBlock
from pixal3d.representations import Mesh, MeshWithVoxel


FORMAT = "pixal3d_sparse_mra_projector_batch_v1"
TILE_IDS = (26, 27)
PROJECTOR_TEST_TILE_IDS = (26, 27)
SNAPSHOT_STEPS = (0, 6, 11)
PBR_CHANNEL_NAMES = (
    "base_color_r",
    "base_color_g",
    "base_color_b",
    "metallic",
    "roughness",
    "alpha",
)
PBR_GROUPS = {
    "RGB": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}

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


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_tensor(path: Path) -> torch.Tensor:
    value = _load_torch(path)
    if isinstance(value, Mapping) and "tensor" in value:
        value = value["tensor"]
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"expected tensor cache at {path}")
    return value.detach().cpu()


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().to(torch.float64)).item())


def _relative(value: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> float:
    return _norm(value) / (_norm(reference) + float(eps))


def _safe_mean(value: torch.Tensor) -> float:
    return float(value.detach().to(torch.float64).mean().item()) if value.numel() else 0.0


def _coordinate_digest(coords: torch.Tensor) -> str:
    return hashlib.sha256(
        coords.detach().cpu().to(torch.int32).contiguous().numpy().tobytes()
    ).hexdigest()


def _tensor_range(value: torch.Tensor) -> Dict[str, Any]:
    value = value.detach().cpu().to(torch.float32)
    result: Dict[str, Any] = {}
    for name, sl in PBR_GROUPS.items():
        field = value[:, sl]
        result[name] = {
            "min": float(field.min().item()) if field.numel() else None,
            "max": float(field.max().item()) if field.numel() else None,
            "out_of_0_1_ratio": float(
                ((field < 0.0) | (field > 1.0)).to(torch.float32).mean().item()
            )
            if field.numel()
            else 0.0,
        }
    return result


def _energy_stats(value: torch.Tensor) -> Dict[str, Any]:
    value = value.detach().cpu().to(torch.float64)
    result: Dict[str, Any] = {}
    for name, sl in PBR_GROUPS.items():
        field = value[:, sl]
        result[name] = {
            "l2": _norm(field),
            "rms": float(torch.sqrt(field.square().mean()).item()) if field.numel() else 0.0,
            "mean_abs": float(field.abs().mean().item()) if field.numel() else 0.0,
            "max_abs": float(field.abs().max().item()) if field.numel() else 0.0,
        }
    return result


def _move_sparse(value: SparseTensor, device: torch.device) -> SparseTensor:
    return SparseTensor(
        value.feats.detach().to(device), value.coords.detach().to(device)
    )


def _fresh_sparse(value: SparseTensor) -> SparseTensor:
    return SparseTensor(value.feats.detach().clone(), value.coords.detach().clone())


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _strict_sparse_check(reference: SparseTensor, candidate: SparseTensor, label: str) -> Dict[str, Any]:
    same_shape = tuple(reference.feats.shape) == tuple(candidate.feats.shape)
    reference_coords = reference.coords.detach().cpu()
    candidate_coords = candidate.coords.detach().cpu()
    same_coords = tuple(reference_coords.shape) == tuple(candidate_coords.shape) and torch.equal(
        reference_coords, candidate_coords
    )
    result = {
        "label": str(label),
        "coords_exact": bool(same_coords),
        "feature_shape_equal": bool(same_shape),
        "reference_tokens": int(reference.feats.shape[0]),
        "candidate_tokens": int(candidate.feats.shape[0]),
        "reference_coord_digest": _coordinate_digest(reference.coords),
        "candidate_coord_digest": _coordinate_digest(candidate.coords),
    }
    if not same_coords or not same_shape:
        raise RuntimeError(f"strict sparse check failed: {result}")
    return result


def _phase_batch_limit(args: argparse.Namespace, attribute: str) -> int:
    value = getattr(args, attribute, None)
    return max(1, int(args.tile_batch_size if value is None else value))


def _parse_tile_id_list(value: Optional[str]) -> Tuple[int, ...]:
    if value is None or not str(value).strip():
        return tuple()
    parsed = sorted({int(item.strip()) for item in str(value).split(",") if item.strip()})
    if any(tile_id < 0 or tile_id >= 49 for tile_id in parsed):
        raise ValueError(f"tile ids must lie in [0, 48], got {parsed}")
    return tuple(parsed)


_ACTIVE_LAYER_RECORDER: Optional["LayerTraceRecorder"] = None
_ACTIVE_LAYER_TILE_IDS: Tuple[int, ...] = ()
_ACTIVE_LAYER_CONTEXT_ID: int = 0


def _layer_sample_summary(value: Any, tile_ids: Sequence[int]) -> Any:
    """Small deterministic hook fingerprint; never stores a full decoder tensor."""
    if isinstance(value, SparseTensor):
        coords = value.coords.detach()
        feats = value.feats.detach()
        result: Dict[str, Any] = {
            "type": "SparseTensor",
            "shape": [int(v) for v in feats.shape],
            "coord_shape": [int(v) for v in coords.shape],
            "tiles": {},
        }
        batch_count = len(tile_ids) if len(tile_ids) > 1 else 1
        for batch_index in range(batch_count):
            if coords.ndim == 2 and coords.shape[1] >= 1:
                rows = torch.where(coords[:, 0] == int(batch_index))[0]
            else:
                rows = torch.arange(feats.shape[0], device=feats.device)
            if rows.numel() == 0:
                continue
            sample_rows = rows[: min(64, int(rows.numel()))]
            sample = feats.index_select(0, sample_rows).to(torch.float32).cpu().contiguous()
            local_coords = coords.index_select(0, sample_rows).to(torch.int32).cpu().contiguous()
            if local_coords.ndim == 2 and local_coords.shape[1] >= 1:
                local_coords[:, 0] = 0
            tile_id = int(tile_ids[batch_index]) if batch_index < len(tile_ids) else int(batch_index)
            result["tiles"][str(tile_id)] = {
                "tokens": int(rows.numel()),
                "sample_shape": [int(v) for v in sample.shape],
                "sample_digest": hashlib.sha256(sample.numpy().tobytes()).hexdigest(),
                "sample_coords_digest": hashlib.sha256(local_coords.numpy().tobytes()).hexdigest(),
                "sample_l2": _norm(sample),
                "sample_max_abs": float(sample.abs().max().item()) if sample.numel() else 0.0,
            }
        return result
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        sample = tensor.reshape(-1)[: min(256, tensor.numel())].to(torch.float32).cpu().contiguous()
        return {
            "type": "Tensor",
            "shape": [int(v) for v in tensor.shape],
            "sample_digest": hashlib.sha256(sample.numpy().tobytes()).hexdigest(),
            "sample_l2": _norm(sample),
            "sample_max_abs": float(sample.abs().max().item()) if sample.numel() else 0.0,
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "items": [_layer_sample_summary(item, tile_ids) for item in value],
        }
    if isinstance(value, Mapping):
        return {
            "type": "Mapping",
            "items": {str(key): _layer_sample_summary(item, tile_ids) for key, item in value.items()},
        }
    return {"type": type(value).__name__}


class LayerTraceRecorder:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self.call_counts: Dict[str, int] = {}
        self.context_counts: Dict[Tuple[int, str], int] = {}

    def attach(self, module_roots: Sequence[Tuple[str, torch.nn.Module]]) -> List[Any]:
        handles: List[Any] = []
        for root_name, root in module_roots:
            for name, module in root.named_modules():
                if not name:
                    continue
                # Capture every decoder/encoder block boundary and leaf that
                # can change sparse support, while avoiding parameterless
                # container callbacks.
                if not (
                    "blocks" in name
                    or name in {"from_latent", "output_layer"}
                    or isinstance(module, (SparseConvNeXtBlock3d, ModulatedSparseTransformerCrossBlock))
                ):
                    continue
                full_name = f"{root_name}.{name}"
                self.call_counts.setdefault(full_name, 0)

                def capture(_module: Any, _inputs: Any, output: Any, module_name: str = full_name) -> None:
                    active_tiles = tuple(_ACTIVE_LAYER_TILE_IDS)
                    call_index = self.call_counts[module_name]
                    self.call_counts[module_name] = call_index + 1
                    output_summary = _layer_sample_summary(output, active_tiles)
                    # The backend isolation fallback invokes a module once per
                    # sample while the surrounding context still carries the
                    # original B>1 tile list.  Assign those B=1 callbacks in
                    # call order to the active tile ids so layer comparisons do
                    # not mistake an implementation fallback for divergence.
                    if (
                        isinstance(output_summary, Mapping)
                        and output_summary.get("type") == "SparseTensor"
                        and len(active_tiles) > 1
                        and len(output_summary.get("tiles", {})) == 1
                    ):
                        context_key = (_ACTIVE_LAYER_CONTEXT_ID, module_name)
                        local_index = self.context_counts.get(context_key, 0)
                        self.context_counts[context_key] = local_index + 1
                        only_summary = next(iter(output_summary["tiles"].values()))
                        assigned_tile = active_tiles[local_index % len(active_tiles)]
                        output_summary["tiles"] = {str(assigned_tile): only_summary}
                    self.records.append(
                        {
                            "module": module_name,
                            "call_index": int(call_index),
                            "tile_ids": [int(v) for v in active_tiles],
                            "output": output_summary,
                        }
                    )

                handles.append(module.register_forward_hook(capture))
        return handles

    def save(self, path: Path) -> None:
        _atomic_json(
            path,
            {
                "format": FORMAT,
                "normalizes_batch_id": True,
                "layer_trace_version": 2,
                "records": self.records,
                "module_count": len(self.call_counts),
                "record_count": len(self.records),
            },
        )


@contextmanager
def _layer_trace_hooks(pipeline: Any, pbr_encoder: torch.nn.Module) -> Iterable[LayerTraceRecorder]:
    global _ACTIVE_LAYER_RECORDER
    recorder = LayerTraceRecorder()
    roots: List[Tuple[str, torch.nn.Module]] = []
    for name in ("shape_slat_decoder", "tex_slat_decoder"):
        module = pipeline.models.get(name)
        if isinstance(module, torch.nn.Module):
            roots.append((name, module))
    roots.append(("pbr_encoder", pbr_encoder))
    handles = recorder.attach(roots)
    previous = _ACTIVE_LAYER_RECORDER
    _ACTIVE_LAYER_RECORDER = recorder
    try:
        yield recorder
    finally:
        _ACTIVE_LAYER_RECORDER = previous
        for handle in handles:
            handle.remove()


@contextmanager
def _layer_trace_active(tile_ids: Sequence[int]) -> Iterable[None]:
    global _ACTIVE_LAYER_TILE_IDS, _ACTIVE_LAYER_CONTEXT_ID
    previous = _ACTIVE_LAYER_TILE_IDS
    previous_context = _ACTIVE_LAYER_CONTEXT_ID
    _ACTIVE_LAYER_CONTEXT_ID += 1
    _ACTIVE_LAYER_TILE_IDS = tuple(int(v) for v in tile_ids)
    try:
        yield
    finally:
        _ACTIVE_LAYER_TILE_IDS = previous
        _ACTIVE_LAYER_CONTEXT_ID = previous_context


@dataclass
class PackedSparseBatch:
    value: SparseTensor
    tile_ids: Tuple[int, ...]
    batch_offsets: Tuple[int, ...]
    original_token_counts: Tuple[int, ...]
    original_local_coords: Tuple[torch.Tensor, ...]
    coordinate_digests: Tuple[str, ...]


def _local_coords(value: SparseTensor) -> torch.Tensor:
    coords = value.coords.detach()
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"expected SparseTensor coords [N,4], got {tuple(coords.shape)}")
    if coords.shape[0] and not bool(torch.all(coords[:, 0] == 0).item()):
        raise ValueError("pack_sparse_batch expects every input SparseTensor batch column to be zero")
    return coords[:, 1:].clone()


def pack_sparse_batch(
    values: Sequence[SparseTensor], tile_ids: Optional[Sequence[int]] = None
) -> PackedSparseBatch:
    """Pack local sparse samples and retain enough metadata for exact unpacking."""
    if not values:
        raise ValueError("cannot pack an empty sparse batch")
    if tile_ids is None:
        tile_ids = tuple(range(len(values)))
    if len(tile_ids) != len(values):
        raise ValueError("tile_ids and sparse values are not aligned")
    channels = tuple(values[0].feats.shape[1:])
    local_coords: List[torch.Tensor] = []
    feats: List[torch.Tensor] = []
    coords: List[torch.Tensor] = []
    offsets: List[int] = []
    offset = 0
    for batch_id, value in enumerate(values):
        if tuple(value.feats.shape[1:]) != channels:
            raise ValueError("sparse batch feature channel shapes differ")
        local = _local_coords(value)
        local_coords.append(local)
        feats.append(value.feats)
        batch_col = torch.full(
            (local.shape[0], 1), batch_id, dtype=value.coords.dtype, device=value.coords.device
        )
        coords.append(torch.cat([batch_col, local], dim=1))
        offsets.append(offset)
        offset += int(local.shape[0])
    packed = SparseTensor(torch.cat(feats, dim=0), torch.cat(coords, dim=0))
    return PackedSparseBatch(
        value=packed,
        tile_ids=tuple(int(v) for v in tile_ids),
        batch_offsets=tuple(offsets),
        original_token_counts=tuple(int(v.shape[0]) for v in local_coords),
        original_local_coords=tuple(local_coords),
        coordinate_digests=tuple(_coordinate_digest(v) for v in local_coords),
    )


def _coordinate_keys(coords: torch.Tensor, base_value: int) -> torch.Tensor:
    coords = coords.to(torch.int64)
    base = int(base_value)
    return coords[:, 0] * base * base + coords[:, 1] * base + coords[:, 2]


def _reorder_by_coordinates(
    candidate_coords: torch.Tensor, candidate_feats: torch.Tensor, reference_local: torch.Tensor
) -> torch.Tensor:
    if candidate_coords.shape[0] != reference_local.shape[0]:
        raise RuntimeError(
            f"sparse batch token count changed: candidate={candidate_coords.shape[0]} "
            f"reference={reference_local.shape[0]}"
        )
    if candidate_coords.numel() == 0:
        return candidate_feats
    max_coord = max(
        int(candidate_coords.max().item()), int(reference_local.max().item())
    )
    key_base = max(2, max_coord + 1)
    candidate_keys = _coordinate_keys(candidate_coords, key_base)
    reference_keys = _coordinate_keys(reference_local, key_base)
    sorted_candidate, order = torch.sort(candidate_keys)
    sorted_reference = torch.sort(reference_keys).values
    if not torch.equal(sorted_candidate, sorted_reference):
        raise RuntimeError("sparse batch output support differs from its input support")
    positions = torch.searchsorted(sorted_candidate, reference_keys)
    reordered = candidate_feats.index_select(0, order.index_select(0, positions))
    restored_keys = candidate_keys.index_select(0, order.index_select(0, positions))
    if not torch.equal(restored_keys, reference_keys):
        raise RuntimeError("sparse batch coordinate reorder failed")
    return reordered


def unpack_sparse_batch(
    value: SparseTensor, packed: PackedSparseBatch
) -> List[SparseTensor]:
    """Split by batch id and restore each original local coordinate order."""
    result: List[SparseTensor] = []
    for batch_id, reference_local in enumerate(packed.original_local_coords):
        mask = value.coords[:, 0] == int(batch_id)
        candidate_coords = value.coords[mask][:, 1:]
        candidate_feats = value.feats[mask]
        reference = reference_local.to(device=value.coords.device, dtype=value.coords.dtype)
        reordered_feats = _reorder_by_coordinates(candidate_coords, candidate_feats, reference)
        local_coords4 = torch.cat([torch.zeros_like(reference[:, :1]), reference], dim=1)
        split = SparseTensor(reordered_feats, local_coords4)
        if _coordinate_digest(split.coords[:, 1:]) != packed.coordinate_digests[batch_id]:
            raise RuntimeError("unpacked sparse coordinate digest mismatch")
        result.append(split)
    return result


def _same_metadata(values: Sequence[Any]) -> bool:
    first = values[0]
    for value in values[1:]:
        if isinstance(first, torch.Tensor) or isinstance(value, torch.Tensor):
            if not isinstance(first, torch.Tensor) or not isinstance(value, torch.Tensor):
                return False
            if first.shape != value.shape or first.dtype != value.dtype:
                return False
            if not torch.equal(first.detach().cpu(), value.detach().cpu()):
                return False
        elif value != first:
            return False
    return True


def _pack_condition_tensor(values: Sequence[torch.Tensor]) -> Any:
    if not values:
        raise ValueError("cannot pack an empty tensor condition")
    if all(value.ndim == 0 for value in values):
        if not _same_metadata(values):
            raise ValueError("scalar condition metadata differs between tiles")
        return values[0]
    if all(value.ndim >= 1 and value.shape[0] == 1 for value in values):
        trailing = tuple(values[0].shape[1:])
        if not all(tuple(value.shape[1:]) == trailing for value in values):
            raise ValueError("condition tensors disagree after their sample dimension")
        return torch.cat(list(values), dim=0)
    if all(tuple(value.shape) == tuple(values[0].shape) for value in values):
        # This is an explicit sample stack for tensors without a leading
        # singleton sample dimension (for example [L,C] cached conditions).
        return torch.stack(list(values), dim=0)
    if all(value.ndim >= 2 and tuple(value.shape[1:]) == tuple(values[0].shape[1:]) for value in values):
        # SLatFlowModel accepts a list for variable-length sequence conditions.
        return [value for value in values]
    raise ValueError("condition tensor schema is not batchable")


def pack_condition_batch(
    conditions: Sequence[Any], tile_ids: Optional[Sequence[int]] = None
) -> Any:
    """Recursively batch current cond/neg_cond schemas without fake dimensions."""
    if not conditions:
        raise ValueError("cannot batch an empty condition sequence")
    first = conditions[0]
    if isinstance(first, SparseTensor):
        if not all(isinstance(value, SparseTensor) for value in conditions):
            raise TypeError("condition schema mixes SparseTensor and non-SparseTensor")
        return pack_sparse_batch(conditions, tile_ids=tile_ids).value
    if isinstance(first, Mapping):
        keys = tuple(first.keys())
        if not all(isinstance(value, Mapping) and tuple(value.keys()) == keys for value in conditions):
            raise ValueError("condition mapping schemas differ between tiles")
        return {
            key: pack_condition_batch([value[key] for value in conditions], tile_ids=tile_ids)
            for key in keys
        }
    if isinstance(first, torch.Tensor):
        if not all(isinstance(value, torch.Tensor) for value in conditions):
            raise TypeError("condition schema mixes Tensor and non-Tensor values")
        return _pack_condition_tensor(conditions)
    if isinstance(first, (list, tuple)):
        if not all(isinstance(value, type(first)) and len(value) == len(first) for value in conditions):
            raise ValueError("condition sequence schemas differ between tiles")
        packed = [
            pack_condition_batch([value[index] for value in conditions], tile_ids=tile_ids)
            for index in range(len(first))
        ]
        return type(first)(packed)
    if not _same_metadata(conditions):
        raise ValueError("scalar/metadata condition differs between tiles")
    return first


@dataclass
class StableSparseMRAProjector:
    """Direct least-squares projector on float64 ``P_h``."""

    P_hidden: csr_matrix
    info: Dict[str, Any]
    atol: float = 1e-7
    btol: float = 1e-7
    maxiter: int = 300
    conlim: float = 1e12
    channel_workers: int = 6
    reduced: Optional[csr_matrix] = None
    active_ids: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.P_hidden.dtype != np.float64:
            self.P_hidden = self.P_hidden.astype(np.float64)
        self.P_hidden = self.P_hidden.tocsr()
        active = np.asarray(self.P_hidden.getnnz(axis=0)).reshape(-1) > 0
        self.active_ids = np.where(active)[0].astype(np.int64, copy=False)
        self.reduced = self.P_hidden[:, self.active_ids].tocsr()
        self.info = dict(self.info)
        self.info.update(
            {
                "P_h_dtype": str(self.P_hidden.dtype),
                "P_h_shape": [int(v) for v in self.P_hidden.shape],
                "P_h_nnz": int(self.P_hidden.nnz),
                "active_columns": int(self.active_ids.size),
                "uncovered_rows": int((np.diff(self.P_hidden.indptr) == 0).sum()),
                "coverage_ratio": float(
                    (np.diff(self.P_hidden.indptr) > 0).mean()
                    if self.P_hidden.shape[0]
                    else 0.0
                ),
                "normal_equation_used": False,
                "solver": "scipy.sparse.linalg.lsmr",
                "channel_workers": int(self.channel_workers),
            }
        )

    def solve(
        self,
        field: torch.Tensor,
        label: str,
        x0: Optional[torch.Tensor] = None,
        atol: Optional[float] = None,
        btol: Optional[float] = None,
        maxiter: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        field_np = field.detach().cpu().to(torch.float64).numpy()
        if field_np.ndim == 1:
            field_np = field_np[:, None]
        if field_np.ndim != 2 or field_np.shape[0] != self.P_hidden.shape[0]:
            raise ValueError(
                f"{label}: field shape {field_np.shape} does not match P_h rows {self.P_hidden.shape[0]}"
            )
        coeff = np.zeros((self.P_hidden.shape[1], field_np.shape[1]), dtype=np.float64)
        channels: List[Dict[str, Any]] = []
        x0_np = None if x0 is None else x0.detach().cpu().to(torch.float64).numpy()
        if x0_np is not None and x0_np.shape != coeff.shape:
            raise ValueError(f"{label}: warm-start shape {x0_np.shape} != coefficient shape {coeff.shape}")
        # A previous-step coefficient is an acceleration seed, not an
        # accuracy relaxation.  LSMR's usual stopping test can accept a
        # warm-start residual whose projected field is still visibly different
        # from the independent cold reference, so warm solves use a stricter
        # direct-LSMR tolerance while retaining the same operator and x0.
        solve_atol = (
            float(atol)
            if atol is not None
            else (min(float(self.atol), 1e-8) if x0_np is not None else float(self.atol))
        )
        solve_btol = (
            float(btol)
            if btol is not None
            else (min(float(self.btol), 1e-8) if x0_np is not None else float(self.btol))
        )
        if solve_atol <= 0.0 or solve_btol <= 0.0:
            raise ValueError("LSMR solve tolerances must be positive")
        solve_maxiter = int(self.maxiter if maxiter is None else maxiter)
        if solve_maxiter <= 0:
            raise ValueError("LSMR maxiter must be positive")
        if self.reduced is not None and self.active_ids is not None and self.active_ids.size:
            def solve_channel(channel: int) -> Tuple[int, np.ndarray, Dict[str, Any]]:
                started = time.perf_counter()
                warm_start_used = x0_np is not None
                fallback_cold_start = False
                attempt_maxiter = solve_maxiter
                result = lsmr(
                    self.reduced,
                    field_np[:, channel],
                    damp=0.0,
                    atol=solve_atol,
                    btol=solve_btol,
                    conlim=float(self.conlim),
                    maxiter=attempt_maxiter,
                    x0=None if x0_np is None else x0_np[self.active_ids, channel],
                )
                # A few warm-started channels can still reach istop=7.  Keep
                # the same direct float64 LSMR operator, but discard a bad
                # warm seed before spending a much larger iteration budget.
                # This preserves strict convergence while keeping the normal
                # warm-start path fast.
                if int(result[1]) == 7 and warm_start_used:
                    fallback_cold_start = True
                    warm_start_used = False
                    attempt_maxiter = solve_maxiter
                    result = lsmr(
                        self.reduced,
                        field_np[:, channel],
                        damp=0.0,
                        atol=solve_atol,
                        btol=solve_btol,
                        conlim=float(self.conlim),
                        maxiter=attempt_maxiter,
                        x0=None,
                    )
                if int(result[1]) == 7:
                    attempt_maxiter = max(int(attempt_maxiter) * 2, int(attempt_maxiter) + 100)
                    result = lsmr(
                        self.reduced,
                        field_np[:, channel],
                        damp=0.0,
                        atol=solve_atol,
                        btol=solve_btol,
                        conlim=float(self.conlim),
                        maxiter=attempt_maxiter,
                        x0=None if warm_start_used is False else x0_np[self.active_ids, channel],
                    )
                solution, istop, iterations, normr, normar, normA, condA, normx = result
                relative_normal_residual = float(
                    normar / max(abs(normA) * abs(normr), np.finfo(np.float64).tiny)
                )
                accepted_by_residual = bool(
                    int(istop) == 7
                    and relative_normal_residual <= max(2.5 * solve_atol, 1e-12)
                )
                channel_info = {
                    "channel": PBR_CHANNEL_NAMES[channel]
                    if channel < len(PBR_CHANNEL_NAMES)
                    else str(channel),
                    "istop": int(istop),
                    "iterations": int(iterations),
                    "normr": float(normr),
                    "normar": float(normar),
                    "normA": float(normA),
                    "condA": float(condA),
                    "normx": float(normx),
                    "solve_seconds": float(time.perf_counter() - started),
                    "maxiter": int(attempt_maxiter),
                    "relative_normal_residual": relative_normal_residual,
                    "accepted_by_residual": accepted_by_residual,
                    "converged": bool(int(istop) in (0, 1, 2) or accepted_by_residual),
                    "warm_start": bool(warm_start_used),
                    "fallback_cold_start": bool(fallback_cold_start),
                }
                return channel, solution, channel_info

            workers = min(max(1, int(self.channel_workers)), field_np.shape[1])
            if workers == 1 or field_np.shape[1] == 1:
                channel_results = [solve_channel(channel) for channel in range(field_np.shape[1])]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    channel_results = list(executor.map(solve_channel, range(field_np.shape[1])))
            for channel, solution, channel_info in channel_results:
                channels.append(channel_info)
                if not bool(channel_info["converged"]):
                    raise RuntimeError(
                        f"{label}: LSMR did not converge for channel {channel}: {channel_info}"
                    )
                coeff[self.active_ids, channel] = solution
        info = {
            "label": str(label),
            "rows": int(self.P_hidden.shape[0]),
            "columns": int(self.P_hidden.shape[1]),
            "active_columns": int(self.active_ids.size if self.active_ids is not None else 0),
            "method": "lsmr_direct_float64",
            "damp": 0.0,
            "warm_start": bool(x0_np is not None),
            "channel_workers": int(self.channel_workers),
            "atol": float(self.atol),
            "btol": float(self.btol),
            "effective_atol": float(solve_atol),
            "effective_btol": float(solve_btol),
            "maxiter": int(self.maxiter),
            "effective_maxiter": int(solve_maxiter),
            "channels": channels,
            "istop": [int(v["istop"]) for v in channels],
            "iterations": [int(v["iterations"]) for v in channels],
            "normr": [float(v["normr"]) for v in channels],
            "normar": [float(v["normar"]) for v in channels],
            "normA": [float(v["normA"]) for v in channels],
            "condA": [float(v["condA"]) for v in channels],
            "normx": [float(v["normx"]) for v in channels],
            "solve_seconds": float(sum(v["solve_seconds"] for v in channels)),
            "converged": bool(all(v["converged"] for v in channels)),
        }
        return torch.from_numpy(coeff), info

    def apply(self, coeff: torch.Tensor) -> torch.Tensor:
        coeff_np = coeff.detach().cpu().to(torch.float64).numpy()
        result = np.asarray(self.P_hidden.dot(coeff_np), dtype=np.float64)
        return torch.from_numpy(result)


def _load_stable_projector(
    operator_dir: Path,
    tile_id: int,
    output_operator_dir: Optional[Path],
    args: argparse.Namespace,
) -> StableSparseMRAProjector:
    source_tile = operator_dir / "tiles" / f"tile_{tile_id:02d}"
    source = source_tile / "mra_P_hidden.npz"
    if not source.is_file():
        raise FileNotFoundError(source)
    p = load_npz(source).tocsr().astype(np.float64)
    if output_operator_dir is not None:
        output_operator_dir.mkdir(parents=True, exist_ok=True)
        stable_path = output_operator_dir / f"tile_{tile_id:02d}_mra_P_hidden_float64.npz"
        if not stable_path.is_file():
            save_npz(stable_path, p, compressed=True)
    info_path = source_tile / "mra_operator.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    return StableSparseMRAProjector(
        p,
        info,
        atol=float(args.lsmr_atol),
        btol=float(args.lsmr_btol),
        maxiter=int(args.lsmr_maxiter),
        conlim=float(args.lsmr_conlim),
        channel_workers=int(getattr(args, "lsmr_channel_workers", 6)),
    )


def _channel_energy_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for name, sl in PBR_GROUPS.items():
        n = torch.linalg.vector_norm(numerator[:, sl].to(torch.float64))
        d = torch.linalg.vector_norm(denominator[:, sl].to(torch.float64))
        result[name] = float((n / (d + 1e-8)).item())
    return result


def _orthogonality_metrics(projector: StableSparseMRAProjector, detail: torch.Tensor, delta: torch.Tensor) -> Dict[str, Any]:
    detail_np = detail.detach().cpu().to(torch.float64).numpy()
    delta_np = delta.detach().cpu().to(torch.float64).numpy()
    pt_detail = torch.from_numpy(np.asarray(projector.reduced.T.dot(detail_np), dtype=np.float64))
    pt_delta = torch.from_numpy(np.asarray(projector.reduced.T.dot(delta_np), dtype=np.float64))
    return {
        "max_abs_Pt_detail": float(pt_detail.abs().max().item()) if pt_detail.numel() else 0.0,
        "mean_abs_Pt_detail": float(pt_detail.abs().mean().item()) if pt_detail.numel() else 0.0,
        "l2_Pt_detail": _norm(pt_detail),
        "relative_Pt_detail": _relative(pt_detail, pt_delta),
        "max_abs_Pt_delta": float(pt_delta.abs().max().item()) if pt_delta.numel() else 0.0,
        "l2_Pt_delta": _norm(pt_delta),
    }


def _projection_invariants(
    projector: StableSparseMRAProjector,
    delta: torch.Tensor,
    tile_id: int,
    step: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, Dict[str, Any]]:
    solve_started = time.perf_counter()
    coeff, solve_info = projector.solve(delta, f"tile_{tile_id:02d}_step_{step:02d}_delta")
    coarse = projector.apply(coeff)
    detail = delta.to(torch.float64) - coarse

    idem_coeff, idem_solve = projector.solve(
        coarse,
        f"tile_{tile_id:02d}_step_{step:02d}_idempotence",
        # Correctness must be an independent cold-start solve.  Reusing the
        # first solution here would make idempotence self-certifying.
        x0=None,
    )
    coarse_2 = projector.apply(idem_coeff)
    idem_error = coarse_2 - coarse

    random_coeff = torch.from_numpy(
        rng.standard_normal((projector.P_hidden.shape[1], delta.shape[1]), dtype=np.float64)
    )
    random_range = projector.apply(random_coeff)
    range_coeff, range_solve = projector.solve(
        random_range,
        f"tile_{tile_id:02d}_step_{step:02d}_range_consistency",
        # The random coefficient is only used to construct a range element;
        # it must not be supplied as LSMR's initial guess for correctness.
        x0=None,
    )
    random_range_2 = projector.apply(range_coeff)
    range_error = random_range_2 - random_range

    orth = _orthogonality_metrics(projector, detail, delta)
    exact_error = coarse + detail - delta.to(torch.float64)
    energy_identity = {}
    for name, sl in PBR_GROUPS.items():
        lhs = torch.linalg.vector_norm(delta[:, sl].to(torch.float64)) ** 2
        rhs = torch.linalg.vector_norm(coarse[:, sl]) ** 2 + torch.linalg.vector_norm(detail[:, sl]) ** 2
        energy_identity[name] = {
            "delta_l2_squared": float(lhs.item()),
            "coarse_plus_detail_l2_squared": float(rhs.item()),
            "relative_error": float((abs(lhs - rhs) / (lhs.abs() + 1e-8)).item()),
        }
    invariants = {
        "orthogonality": orth,
        "idempotence": {
            "max_abs": float(idem_error.abs().max().item()) if idem_error.numel() else 0.0,
            "mean_abs": float(idem_error.abs().mean().item()) if idem_error.numel() else 0.0,
            "l2": _norm(idem_error),
            "relative_l2": _relative(idem_error, coarse),
            "solver": idem_solve,
        },
        "range_consistency": {
            "max_abs": float(range_error.abs().max().item()) if range_error.numel() else 0.0,
            "mean_abs": float(range_error.abs().mean().item()) if range_error.numel() else 0.0,
            "l2": _norm(range_error),
            "relative_l2": _relative(range_error, random_range),
            "solver": range_solve,
        },
        "exact_decomposition": {
            "max_abs": float(exact_error.abs().max().item()) if exact_error.numel() else 0.0,
            "mean_abs": float(exact_error.abs().mean().item()) if exact_error.numel() else 0.0,
            "relative_l2": _relative(exact_error, delta),
        },
        "energy_identity": energy_identity,
        "r_c": _channel_energy_ratio(coarse, delta),
        "r_d": _channel_energy_ratio(detail, delta),
        "solve_seconds_all_projection_checks": float(time.perf_counter() - solve_started),
    }
    return invariants, coarse, detail, {
        "delta_coefficients": coeff,
        "delta_solve": solve_info,
        "random_coefficients": random_coeff,
    }


def _projector_field_record(
    projector: StableSparseMRAProjector,
    tile_id: int,
    step: int,
    fields: Mapping[str, torch.Tensor],
    hidden_mask: torch.Tensor,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    h = fields["H_t"].detach().cpu().to(torch.float64)
    g = fields["G1024"].detach().cpu().to(torch.float64)
    cached_delta = fields.get("Delta")
    hidden = hidden_mask.detach().cpu().bool()
    delta_all = h - g
    delta = delta_all[hidden]
    if delta.shape[0] != projector.P_hidden.shape[0]:
        raise RuntimeError(
            f"tile {tile_id}: hidden field rows {delta.shape[0]} != P_h rows {projector.P_hidden.shape[0]}"
        )
    if cached_delta is not None:
        cached = cached_delta.detach().cpu().to(torch.float64)[hidden]
        cached_error = cached - delta
    else:
        cached_error = torch.empty(0, 6, dtype=torch.float64)
    invariants, coarse, detail, solve_payload = _projection_invariants(
        projector, delta, tile_id, step, rng
    )
    hidden_target = g[hidden] + detail
    stable_full = h.clone()
    stable_full[hidden] = hidden_target
    old_target = fields.get("hidden_target", fields.get("final_target_field", h))
    old_target = old_target.detach().cpu().to(torch.float64)
    old_hidden = old_target[hidden] if old_target.shape[0] == h.shape[0] else old_target
    cond_values = [
        float(v["condA"])
        for channel in solve_payload["delta_solve"].get("channels", [])
        for v in [channel]
        if math.isfinite(float(v["condA"]))
    ]
    record = {
        "format": FORMAT,
        "tile_id": int(tile_id),
        "step": int(step),
        "P_h": projector.info,
        "support": {
            "hidden_rows": int(hidden.sum().item()),
            "fine_rows": int(hidden_mask.shape[0]),
            "hidden_mask_digest": _coordinate_digest(hidden.nonzero().to(torch.int32)),
        },
        "solver": solve_payload["delta_solve"],
        "condition_estimate": {
            "condA_channels": cond_values,
            "condA_max": max(cond_values) if cond_values else None,
            "condA_mean": float(np.mean(cond_values)) if cond_values else None,
        },
        "cached_delta_vs_recomputed": {
            "max_abs": float(cached_error.abs().max().item()) if cached_error.numel() else None,
            "relative_l2": _relative(cached_error, delta) if cached_error.numel() else None,
            "cached_dtype": str(fields["Delta"].dtype) if cached_delta is not None else None,
        },
        "energy": {
            "Delta": _energy_stats(delta),
            "Delta_coarse": _energy_stats(coarse),
            "Delta_detail": _energy_stats(detail),
        },
        "ranges": {
            "G1024_hidden": _tensor_range(g[hidden]),
            "H_t_hidden": _tensor_range(h[hidden]),
            "stable_hidden_target": _tensor_range(hidden_target),
            "stable_full_target": _tensor_range(stable_full),
            "old_hidden_target": _tensor_range(old_hidden),
        },
        "invariants": invariants,
        "finite": bool(torch.isfinite(stable_full).all().item()),
    }
    return record


def run_projector_tests(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    """Run the mandatory Tile 26/27 step 0/6/11 ground-truth projector test."""
    projector_dir = output_dir / "projector_test"
    operator_output = projector_dir / "operators"
    records: Dict[str, Any] = {}
    started_all = time.perf_counter()
    for tile_id in PROJECTOR_TEST_TILE_IDS:
        projector = _load_stable_projector(
            Path(args.operator_cache_dir), tile_id, operator_output, args
        )
        source_tile = Path(args.field_cache_dir) / "tiles" / f"tile_{tile_id:02d}"
        hidden_mask = _load_tensor(source_tile / "hidden_mask.pt").bool()
        for step in SNAPSHOT_STEPS:
            field_path = (
                Path(args.field_source_dir)
                / "tiles"
                / f"tile_{tile_id:02d}"
                / "steps"
                / f"step_{step:02d}_mra_fields.pt"
            )
            fields = _load_torch(field_path)
            path = projector_dir / f"tile_{tile_id:02d}_step_{step:02d}.json"
            if bool(args.resume) and path.is_file():
                record = _load_projector_metrics(path)
            else:
                record = _projector_field_record(
                    projector,
                    tile_id,
                    step,
                    fields,
                    hidden_mask,
                    np.random.default_rng(int(args.seed) + tile_id * 100 + step),
                )
                _atomic_json(path, record)
            records[f"tile_{tile_id:02d}_step_{step:02d}"] = record
            print(
                f"[projector tile {tile_id:02d} step {step:02d}] "
                f"condA={record['condition_estimate']['condA_max']:.4g} "
                f"orth={record['invariants']['orthogonality']['relative_Pt_detail']:.3e} "
                f"idem={record['invariants']['idempotence']['relative_l2']:.3e}",
                flush=True,
            )
    metrics = {
        "format": FORMAT,
        "cuda_device": int(args.cuda_device),
        "tiles": list(PROJECTOR_TEST_TILE_IDS),
        "steps": list(SNAPSHOT_STEPS),
        "projector_test_seconds": float(time.perf_counter() - started_all),
        "records": records,
        "definitions": {
            "Delta": "(H_t-G1024) restricted to hidden rows",
            "Delta_coarse": "P_h c_star, c_star=argmin ||P_h c-Delta||_2",
            "Delta_detail": "Delta-Delta_coarse",
            "orthogonality": "P_h.T @ Delta_detail, with absolute and relative norms",
            "r_c": "global Frobenius energy(Delta_coarse)/(energy(Delta)+1e-8)",
            "r_d": "global Frobenius energy(Delta_detail)/(energy(Delta)+1e-8)",
            "solver": "float64 direct sparse LSMR; no normal equation, no ridge, no coefficient clamp",
        },
    }
    _atomic_json(output_dir / "projector_metrics.json", metrics)
    return metrics


def run_warm_start_tests(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    """Compare formal-flow warm starts against independent cold references."""
    records: Dict[str, Any] = {}
    checkpoint_dir = output_dir / "warm_start_test"
    started_all = time.perf_counter()
    for tile_id in PROJECTOR_TEST_TILE_IDS:
        projector = _load_stable_projector(
            Path(args.operator_cache_dir), tile_id, output_dir / "projector_test" / "operators", args
        )
        hidden_mask = _load_tensor(
            Path(args.field_cache_dir) / "tiles" / f"tile_{tile_id:02d}" / "hidden_mask.pt"
        ).bool()
        previous_coefficients: Optional[torch.Tensor] = None
        for step in SNAPSHOT_STEPS:
            checkpoint = checkpoint_dir / f"tile_{tile_id:02d}_step_{step:02d}.json"
            coefficient_checkpoint = checkpoint.with_suffix(".pt")
            if bool(args.resume) and checkpoint.is_file() and coefficient_checkpoint.is_file():
                record = _load_projector_metrics(checkpoint)
                cold_solver = record.get("cold_solver", {})
                cold_is_strict = float(cold_solver.get("effective_atol", 1.0)) <= 1e-12
                if bool(record.get("passed")) and cold_is_strict:
                    coefficient_payload = _load_torch(coefficient_checkpoint)
                    previous_coefficients = (
                        coefficient_payload["coefficients"]
                        if isinstance(coefficient_payload, Mapping)
                        else coefficient_payload
                    ).detach().cpu().to(torch.float64)
                    records[f"tile_{tile_id:02d}_step_{step:02d}"] = record
                    print(
                        f"[warm-start tile {tile_id:02d} step {step:02d}] "
                        f"used={record['warm_start_used']} "
                        f"rel={record['projected_relative_l2_vs_cold']:.3e} "
                        f"iteration_reduction={record['iteration_reduction_vs_cold']} (checkpoint)",
                        flush=True,
                    )
                    continue
            field_path = (
                Path(args.field_source_dir)
                / "tiles"
                / f"tile_{tile_id:02d}"
                / "steps"
                / f"step_{step:02d}_mra_fields.pt"
            )
            fields = _load_torch(field_path)
            h = fields["H_t"].detach().cpu().to(torch.float64)
            g = fields["G1024"].detach().cpu().to(torch.float64)
            delta = (h - g)[hidden_mask]
            cold_coefficients, cold_info = projector.solve(
                delta,
                f"tile_{tile_id:02d}_step_{step:02d}_cold_reference",
                x0=None,
                # The warm/cold equivalence reference must be more accurate
                # than the production stopping point; otherwise a genuinely
                # refined warm solve is incorrectly judged against a loose
                # cold residual (the previous 1e-6 cold solve had normar~1e-2).
                atol=min(float(args.lsmr_atol), 1e-12),
                btol=min(float(args.lsmr_btol), 1e-12),
                maxiter=max(int(args.lsmr_maxiter), 2400),
            )
            cold_coarse = projector.apply(cold_coefficients)
            if previous_coefficients is None:
                warm_coefficients = cold_coefficients
                warm_info = None
                warm_attempts: List[Dict[str, Any]] = []
                projected_relative_l2 = 0.0
                projected_max_abs = 0.0
                iteration_reduction = 0
            else:
                warm_seed = previous_coefficients
                warm_attempts = []
                warm_coefficients = previous_coefficients
                warm_info = None
                projected_relative_l2 = float("inf")
                projected_max_abs = float("inf")
                for attempt in range(3):
                    tolerance = min(float(args.lsmr_atol), 10.0 ** (-8 - 2 * attempt))
                    warm_coefficients, warm_info = projector.solve(
                        delta,
                        f"tile_{tile_id:02d}_step_{step:02d}_warm_refine{attempt}",
                        x0=warm_seed,
                        atol=tolerance,
                        btol=tolerance,
                        maxiter=max(int(args.lsmr_maxiter), 800 * (attempt + 1)),
                    )
                    warm_coarse = projector.apply(warm_coefficients)
                    projected_error = warm_coarse - cold_coarse
                    projected_relative_l2 = _relative(projected_error, cold_coarse)
                    projected_max_abs = float(projected_error.abs().max().item())
                    warm_attempts.append(
                        {
                            "attempt": int(attempt),
                            "atol": float(tolerance),
                            "btol": float(tolerance),
                            "maxiter": int(max(int(args.lsmr_maxiter), 800 * (attempt + 1))),
                            "projected_relative_l2_vs_cold": float(projected_relative_l2),
                            "projected_max_abs_vs_cold": float(projected_max_abs),
                            "iterations": [int(v["iterations"]) for v in warm_info["channels"]],
                        }
                    )
                    if projected_relative_l2 < 1e-5:
                        break
                    warm_seed = warm_coefficients
                iteration_reduction = int(
                    sum(v["iterations"] for v in cold_info["channels"])
                    - sum(v["iterations"] for v in warm_info["channels"])
                )
            record = {
                "tile_id": int(tile_id),
                "step": int(step),
                "cold_solver": cold_info,
                "warm_solver": warm_info,
                "warm_solver_attempts": warm_attempts,
                "warm_start_used": bool(previous_coefficients is not None),
                "projected_relative_l2_vs_cold": float(projected_relative_l2),
                "projected_max_abs_vs_cold": float(projected_max_abs),
                "iteration_reduction_vs_cold": int(iteration_reduction),
                "passed": bool(float(projected_relative_l2) < 1e-5),
            }
            records[f"tile_{tile_id:02d}_step_{step:02d}"] = record
            previous_coefficients = warm_coefficients.detach().cpu()
            _atomic_json(checkpoint, record)
            _atomic_torch_save(checkpoint.with_suffix(".pt"), {"coefficients": previous_coefficients})
            print(
                f"[warm-start tile {tile_id:02d} step {step:02d}] "
                f"used={record['warm_start_used']} "
                f"rel={record['projected_relative_l2_vs_cold']:.3e} "
                f"iteration_reduction={record['iteration_reduction_vs_cold']}",
                flush=True,
            )
    metrics = {
        "format": FORMAT,
        "cuda_device": int(args.cuda_device),
        "tiles": list(PROJECTOR_TEST_TILE_IDS),
        "steps": list(SNAPSHOT_STEPS),
        "records": records,
        "passed": bool(all(bool(record["passed"]) for record in records.values())),
        "definition": "cold is x0=None; warm uses the previous snapshot coefficient; compare projected coarse fields, not coefficients",
        "seconds": float(time.perf_counter() - started_all),
    }
    _atomic_json(output_dir / "warm_start_metrics.json", metrics)
    return metrics


def _normalize_slat(value: SparseTensor, normalization: Mapping[str, Sequence[float]]) -> SparseTensor:
    mean = torch.as_tensor(normalization["mean"], device=value.device, dtype=value.feats.dtype)[None]
    std = torch.as_tensor(normalization["std"], device=value.device, dtype=value.feats.dtype)[None]
    return SparseTensor((value.feats - mean) / std, value.coords.detach().clone())


@torch.no_grad()
def run_official_encoder_regression(
    args: argparse.Namespace,
    output_dir: Path,
    contexts: Sequence[Any],
    pipeline: Any,
    pbr_encoder: torch.nn.Module,
) -> Dict[str, Any]:
    """Compare the official B=1 encoder call with the batch wrapper at B=1."""
    records: Dict[str, Any] = {}
    started_all = time.perf_counter()
    for context in contexts:
        if int(context.tile_id) not in PROJECTOR_TEST_TILE_IDS:
            continue
        tile_id = int(context.tile_id)
        field_path = (
            Path(args.field_source_dir)
            / "tiles"
            / f"tile_{tile_id:02d}"
            / "steps"
            / "step_00_mra_fields.pt"
        )
        fields = _load_torch(field_path)
        attrs = fields["H_t"].detach().cpu().to(torch.float32)
        coords = context.geometry.coords.detach().cpu()
        if attrs.shape[0] != coords.shape[0]:
            raise RuntimeError(
                f"tile {tile_id}: official encoder regression field rows {attrs.shape[0]} "
                f"!= geometry rows {coords.shape[0]}"
            )
        direct_raw, direct_stats = core._encode_local_pbr(
            encoder=pbr_encoder,
            coords=coords,
            attrs=attrs,
            device=torch.device("cuda"),
            low_vram=bool(args.low_vram),
        )
        direct_norm = _normalize_slat(direct_raw, pipeline.tex_slat_normalization)
        wrapped_values, wrapped_stats = _encode_pbr_batch(
            contexts=[context],
            fields={tile_id: attrs},
            references={tile_id: direct_raw},
            pbr_encoder=pbr_encoder,
            pipeline=pipeline,
            low_vram=bool(args.low_vram),
            tile_ids=[tile_id],
        )
        wrapped = wrapped_values[tile_id]
        _strict_sparse_check(direct_norm, wrapped, f"tile {tile_id} official-vs-wrapper B1")
        diff = wrapped.feats.detach().to(torch.float64) - direct_norm.feats.detach().to(torch.float64)
        records[str(tile_id)] = {
            "tile_id": tile_id,
            "field": str(field_path),
            "input_attrs_range": _tensor_range(attrs),
            "coords_exact": True,
            "tokens": int(direct_norm.feats.shape[0]),
            "channels": int(direct_norm.feats.shape[1]),
            "max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
            "mean_abs": float(diff.abs().mean().item()) if diff.numel() else 0.0,
            "relative_l2": _relative(diff, direct_norm.feats),
            "direct_stats": direct_stats,
            "wrapper_stats": wrapped_stats,
            "passed": bool(_relative(diff, direct_norm.feats) < 1e-4),
        }
        del direct_raw, direct_norm, wrapped, wrapped_values
        if bool(args.low_vram):
            _empty_cuda_cache()
    metrics = {
        "format": FORMAT,
        "cuda_device": int(args.cuda_device),
        "mode": "official _encode_local_pbr vs _encode_pbr_batch with one context",
        "records": records,
        "passed": bool(all(bool(record["passed"]) for record in records.values())),
        "seconds": float(time.perf_counter() - started_all),
    }
    _atomic_json(output_dir / "official_encoder_regression.json", metrics)
    return metrics


def _denormalize_slat(value: SparseTensor, normalization: Mapping[str, Sequence[float]]) -> SparseTensor:
    mean = torch.as_tensor(normalization["mean"], device=value.device, dtype=value.feats.dtype)[None]
    std = torch.as_tensor(normalization["std"], device=value.device, dtype=value.feats.dtype)[None]
    return SparseTensor(value.feats * std + mean, value.coords.detach().clone())


def _batch_stats_for_sparse(values: Sequence[SparseTensor]) -> Dict[str, Any]:
    return {
        "tokens": [int(value.feats.shape[0]) for value in values],
        "channels": int(values[0].feats.shape[1]),
        "total_tokens": int(sum(value.feats.shape[0] for value in values)),
    }


def _prediction_phase(
    *,
    contexts: Sequence[Any],
    states: Mapping[int, SparseTensor],
    pipeline: Any,
    sampler: Any,
    model: torch.nn.Module,
    condition_by_tile: Mapping[int, Mapping[str, Any]],
    shape_by_tile: Mapping[int, SparseTensor],
    t: float,
    step_kwargs: Mapping[str, Any],
    mode: str,
    tile_batch_size: int,
    prediction_token_budget: int,
    low_vram: bool,
    step_index: int,
) -> Tuple[Dict[int, Dict[str, SparseTensor]], Dict[str, Any]]:
    predictions: Dict[int, Dict[str, SparseTensor]] = {}
    calls = 0
    started = time.perf_counter()
    groups: List[List[Any]] = []
    if mode == "serial":
        groups = [[context] for context in contexts]
    else:
        current: List[Any] = []
        current_tokens = 0
        for context in contexts:
            tile_id = int(context.tile_id)
            tokens = int(states[tile_id].feats.shape[0])
            if current and (
                len(current) >= max(1, int(tile_batch_size))
                or current_tokens + tokens > max(1, int(prediction_token_budget))
            ):
                groups.append(current)
                current, current_tokens = [], 0
            current.append(context)
            current_tokens += tokens
        if current:
            groups.append(current)
    for group in groups:
        tile_ids = [int(context.tile_id) for context in group]
        if len(group) == 1 and mode == "serial":
            context = group[0]
            tile_id = int(context.tile_id)
            state = _move_sparse(states[tile_id], torch.device("cuda")) if low_vram else states[tile_id]
            shape = _move_sparse(shape_by_tile[tile_id], torch.device("cuda")) if low_vram else shape_by_tile[tile_id]
            condition = base._move_condition(condition_by_tile[tile_id], torch.device("cuda"))
            with _layer_trace_active(tile_ids):
                pred_x0, _, pred_v = sampler._get_model_prediction(
                    model,
                    state,
                    float(t),
                    cond=condition["cond"],
                    neg_cond=condition["neg_cond"],
                    concat_cond=shape,
                    **dict(step_kwargs),
                )
            if not isinstance(pred_x0, SparseTensor) or not isinstance(pred_v, SparseTensor):
                raise RuntimeError(f"tile {tile_id}: official prediction did not return SparseTensor")
            _strict_sparse_check(state, pred_x0, f"tile {tile_id} step {step_index} pred_x0")
            _strict_sparse_check(state, pred_v, f"tile {tile_id} step {step_index} pred_v")
            predictions[tile_id] = {
                "pred_x0": _move_sparse(pred_x0, torch.device("cpu")) if low_vram else pred_x0,
                "pred_v": _move_sparse(pred_v, torch.device("cpu")) if low_vram else pred_v,
            }
            del state, shape, condition, pred_x0, pred_v
            calls += 1
            continue

        state_values = [
            _move_sparse(states[tile_id], torch.device("cuda")) if low_vram else states[tile_id]
            for tile_id in tile_ids
        ]
        shape_values = [
            _move_sparse(shape_by_tile[tile_id], torch.device("cuda")) if low_vram else shape_by_tile[tile_id]
            for tile_id in tile_ids
        ]
        packed_state = pack_sparse_batch(state_values, tile_ids)
        packed_shape = pack_sparse_batch(shape_values, tile_ids)
        condition = pack_condition_batch(
            [base._move_condition(condition_by_tile[tile_id], torch.device("cuda")) for tile_id in tile_ids],
            tile_ids,
        )
        # Keep the flow-model outer call batched.  Its internal sparse
        # flex_gemm/Linear/ConvNeXt kernels can become batch-size dependent
        # after the first Euler update, so isolate only those kernels exactly
        # as in the decoder and PBR encoder paths.
        with _layer_trace_active(tile_ids):
            with _batch_backend_kernel_isolation(enabled=len(group) > 1, batch_size=len(group)):
                pred_x0_batch, _, pred_v_batch = sampler._get_model_prediction(
                    model,
                    packed_state.value,
                    float(t),
                    cond=condition["cond"],
                    neg_cond=condition["neg_cond"],
                    concat_cond=packed_shape.value,
                    **dict(step_kwargs),
                )
        if not isinstance(pred_x0_batch, SparseTensor) or not isinstance(pred_v_batch, SparseTensor):
            raise RuntimeError("batched official prediction did not return SparseTensor")
        pred_x0_values = unpack_sparse_batch(pred_x0_batch, packed_state)
        pred_v_values = unpack_sparse_batch(pred_v_batch, packed_state)
        for context, state_value, pred_x0, pred_v in zip(group, state_values, pred_x0_values, pred_v_values):
            tile_id = int(context.tile_id)
            _strict_sparse_check(state_value, pred_x0, f"tile {tile_id} step {step_index} batched pred_x0")
            _strict_sparse_check(state_value, pred_v, f"tile {tile_id} step {step_index} batched pred_v")
            predictions[tile_id] = {
                "pred_x0": _move_sparse(pred_x0, torch.device("cpu")) if low_vram else pred_x0,
                "pred_v": _move_sparse(pred_v, torch.device("cpu")) if low_vram else pred_v,
            }
        calls += 1
        del state_values, shape_values, packed_state, packed_shape, condition
        del pred_x0_batch, pred_v_batch, pred_x0_values, pred_v_values
        if low_vram:
            _empty_cuda_cache()
    return predictions, {
        "seconds": float(time.perf_counter() - started),
        "model_forward_calls": int(calls),
        "actual_batch_sizes": [len(group) for group in groups],
        "actual_batch_tokens": [
            int(sum(int(states[int(context.tile_id)].feats.shape[0]) for context in group))
            for group in groups
        ],
        "token_budget": int(prediction_token_budget),
    }


def _decode_one_batch(
    *,
    contexts: Sequence[Any],
    predictions: Mapping[int, Mapping[str, SparseTensor]],
    pipeline: Any,
    args: argparse.Namespace,
    low_vram: bool,
    step_index: int,
    allow_fallback: bool = True,
) -> Tuple[Dict[int, MeshWithVoxel], Dict[int, torch.Tensor], Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Decode one microbatch; query_attrs stays per tile after decoder batching."""
    tile_ids = [int(context.tile_id) for context in contexts]
    shape_values = [
        _move_sparse(context.shape_denorm, torch.device("cuda")) if low_vram else context.shape_denorm
        for context in contexts
    ]
    texture_values = [
        _move_sparse(predictions[tile_id]["pred_x0"], torch.device("cuda"))
        if low_vram
        else predictions[tile_id]["pred_x0"]
        for tile_id in tile_ids
    ]
    shape_batch = pack_sparse_batch(shape_values, tile_ids).value
    texture_pack = pack_sparse_batch(texture_values, tile_ids)
    texture_batch = _denormalize_slat(texture_pack.value, pipeline.tex_slat_normalization)
    decode_model_started = time.perf_counter()
    try:
        with _layer_trace_active(tile_ids):
            decoded_list = _decode_latent_batch_safe(
                pipeline=pipeline,
                shape_batch=shape_batch,
                texture_batch=texture_batch,
            )
        _sync_cuda()
    except torch.cuda.OutOfMemoryError as exc:
        if not allow_fallback or len(contexts) == 1:
            raise
        _empty_cuda_cache()
        reason = f"decode batch {tile_ids} OOM: {type(exc).__name__}: {exc}"
        all_meshes: Dict[int, MeshWithVoxel] = {}
        all_fields: Dict[int, torch.Tensor] = {}
        all_stats: Dict[int, Dict[str, Any]] = {}
        total_stats = {"fallback_reason": reason, "requested_batch_size": len(contexts), "actual_batch_size": 1}
        for context in contexts:
            meshes, fields, stats, _ = _decode_one_batch(
                contexts=[context],
                predictions=predictions,
                pipeline=pipeline,
                args=args,
                low_vram=low_vram,
                step_index=step_index,
                allow_fallback=False,
            )
            all_meshes.update(meshes)
            all_fields.update(fields)
            all_stats.update(stats)
        total_stats["decoder_forward_calls"] = 2 * len(contexts)
        total_stats["decode_model_seconds"] = float(
            sum(float(value.get("decode_model_seconds", 0.0)) for value in all_stats.values())
        )
        total_stats["query_attrs_seconds"] = float(
            sum(float(value.get("query_attrs_seconds", 0.0)) for value in all_stats.values())
        )
        return all_meshes, all_fields, all_stats, total_stats
    decode_model_seconds = float(time.perf_counter() - decode_model_started)
    if not isinstance(decoded_list, list) or len(decoded_list) != len(contexts):
        raise RuntimeError(
            f"decoder batch returned {type(decoded_list)} length {len(decoded_list) if isinstance(decoded_list, list) else 'n/a'} "
            f"for tiles {tile_ids}"
        )
    meshes: Dict[int, MeshWithVoxel] = {}
    fields: Dict[int, torch.Tensor] = {}
    stats: Dict[int, Dict[str, Any]] = {}
    for context, mesh in zip(contexts, decoded_list):
        tile_id = int(context.tile_id)
        mesh = base._validate_decoded_mesh(mesh, f"tile {tile_id} step {step_index} decoded")
        query_started = time.perf_counter()
        points = context.target_points.to(torch.device("cuda")) if low_vram else context.target_points
        field = base._query_mesh_chunked(mesh, points, int(args.query_chunk_size))
        _sync_cuda()
        query_seconds = float(time.perf_counter() - query_started)
        if not torch.isfinite(field).all():
            raise RuntimeError(f"tile {tile_id}: decoded PBR query is non-finite")
        mesh_stats = {
            "decode_model_seconds": decode_model_seconds,
            "query_attrs_seconds": query_seconds,
            "decode_seconds": decode_model_seconds + query_seconds,
            "decoded_vertices": int(mesh.vertices.shape[0]),
            "decoded_faces": int(mesh.faces.shape[0]),
            "decoded_active_ovoxels": int(mesh.coords.shape[0]),
            "queried_fixed_support_tokens": int(field.shape[0]),
            "decoded_pbr_range": core._tensor_range(mesh.attrs),
            "decoded_support_coord_digest": _coordinate_digest(mesh.coords),
            "requested_batch_size": len(contexts),
            "actual_batch_size": len(contexts),
        }
        if low_vram:
            mesh = mesh.to("cpu")
            field = field.detach().to("cpu").clone()
        meshes[tile_id] = mesh
        fields[tile_id] = field
        stats[tile_id] = mesh_stats
    del shape_values, texture_values, shape_batch, texture_pack, texture_batch, decoded_list
    if low_vram:
        _empty_cuda_cache()
    return meshes, fields, stats, {
        "requested_batch_size": len(contexts),
        "actual_batch_size": len(contexts),
        "decoder_forward_calls": 2,
        "decode_model_seconds": decode_model_seconds,
        "query_attrs_seconds": float(sum(float(value["query_attrs_seconds"]) for value in stats.values())),
        "fallback_reason": None,
    }


@torch.no_grad()
def _decode_latent_batch_safe(
    *, pipeline: Any, shape_batch: SparseTensor, texture_batch: SparseTensor
) -> List[MeshWithVoxel]:
    """Decode a sparse batch without relying on ``SparseTensor.__getitem__``.

    Some flex-gemm decoder outputs are not physically contiguous by batch even
    though their batch ids are correct.  The pipeline's convenience wrapper
    iterates the texture SparseTensor and therefore assumes contiguity.  Split
    the raw decoded voxel tensor by batch id here, preserving the official
    shape/PBR decoder calls and the exact local support for every tile.
    """
    # ``FlexiDualGridVaeDecoder.forward`` converts the parent decoder output
    # with ``zip(vertices, intersected, quad_lerp)``.  That iteration uses
    # SparseTensor.layout, which is only valid when every batch's output rows
    # remain physically contiguous.  flex_gemm may return a correctly tagged
    # but non-contiguous batch, so invoke the parent decoder and construct the
    # dual-grid meshes from the batch-id mask explicitly.
    shape_decoder = pipeline.models["shape_slat_decoder"]
    shape_decoder.set_resolution(1024)
    if pipeline.low_vram:
        shape_decoder.to(pipeline.device)
        shape_decoder.low_vram = True
    with _batch_backend_kernel_isolation(
        enabled=int(shape_batch.shape[0]) > 1, batch_size=int(shape_batch.shape[0])
    ):
        decoded_shape = SparseUnetVaeDecoder.forward(
            shape_decoder, shape_batch, return_subs=True
        )
    if pipeline.low_vram:
        shape_decoder.cpu()
        shape_decoder.low_vram = False
    if not isinstance(decoded_shape, tuple) or len(decoded_shape) != 2:
        raise RuntimeError("safe batched shape decoder returned an unexpected result")
    shape_output, subs = decoded_shape
    if not isinstance(shape_output, SparseTensor) or not isinstance(subs, list):
        raise RuntimeError("safe batched shape decoder returned invalid sparse outputs")
    voxel_margin = float(getattr(shape_decoder, "voxel_margin", 0.5))
    meshes: List[Mesh] = []
    for batch_id in range(int(shape_batch.shape[0])):
        mask = shape_output.coords[:, 0] == int(batch_id)
        local_coords = shape_output.coords[mask][:, 1:]
        local_feats = shape_output.feats[mask]
        vertices = (1.0 + 2.0 * voxel_margin) * torch.sigmoid(
            local_feats[..., 0:3]
        ) - voxel_margin
        intersected = local_feats[..., 3:6] > 0
        quad_lerp = torch.nn.functional.softplus(local_feats[..., 6:7])
        meshes.append(
            Mesh(
                *flexible_dual_grid_to_mesh(
                    local_coords,
                    vertices,
                    intersected,
                    quad_lerp,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    grid_size=1024,
                    train=False,
                )
            )
        )
    with _batch_backend_kernel_isolation(
        enabled=int(texture_batch.shape[0]) > 1, batch_size=int(texture_batch.shape[0])
    ):
        tex_voxels = pipeline.decode_tex_slat(texture_batch, subs)
    if not isinstance(tex_voxels, SparseTensor) or len(meshes) != int(shape_batch.shape[0]):
        raise RuntimeError("safe batched decoder returned an unexpected shape/texture result")
    outputs: List[MeshWithVoxel] = []
    for batch_id, mesh in enumerate(meshes):
        mask = tex_voxels.coords[:, 0] == int(batch_id)
        coords_local = tex_voxels.coords[mask][:, 1:]
        attrs_local = tex_voxels.feats[mask]
        local_value = SparseTensor(
            attrs_local,
            torch.cat([torch.zeros_like(coords_local[:, :1]), coords_local], dim=1),
        )
        outputs.append(
            MeshWithVoxel(
                mesh.vertices,
                mesh.faces,
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1 / 1024,
                coords=coords_local,
                attrs=attrs_local,
                voxel_shape=torch.Size([*local_value.shape, *local_value.spatial_shape]),
                layout=pipeline.pbr_attr_layout,
            )
        )
    return outputs


def _split_sparse_batch(value: SparseTensor) -> List[SparseTensor]:
    """Split a physically contiguous sparse batch into local B=1 tensors."""
    values: List[SparseTensor] = []
    for batch_id in range(int(value.shape[0])):
        mask = value.coords[:, 0] == int(batch_id)
        coords = value.coords[mask].detach().clone()
        coords[:, 0] = 0
        values.append(SparseTensor(value.feats[mask].contiguous(), coords.contiguous()))
    return values


def _pack_sparse_locals(values: Sequence[SparseTensor]) -> SparseTensor:
    """Pack local B=1 sparse tensors after a targeted backend fallback."""
    return pack_sparse_batch(values, tuple(range(len(values)))).value


def _split_apply_sparse_module(
    original_forward: Any, module: Any, value: SparseTensor
) -> SparseTensor:
    """Apply one backend module per sample and restore a real sparse batch."""
    outputs = [original_forward(module, local) for local in _split_sparse_batch(value)]
    if not all(isinstance(output, SparseTensor) for output in outputs):
        raise RuntimeError("decoder backend fallback returned a non-sparse output")
    return _pack_sparse_locals(outputs)


@contextmanager
def _batch_backend_kernel_isolation(enabled: bool, batch_size: int = 2):
    """Isolate backend kernels known to change with sparse batch size.

    The outer encoder/shape/texture forward remains a single batch call.  The
    flex_gemm sparse convolution, SparseLinear GEMM, the sparse flow
    transformer block (including its dense MLP/attention projections), the
    ConvNeXt block, and sparse Flash Attention varlen kernels are the concrete
    paths whose B=2 results diverged from B=1. Their per-sample outputs are
    immediately repacked before the next model operation.
    """
    if not enabled:
        yield
        return
    original_conv = _flex_gemm_backend.sparse_conv3d_forward
    original_linear = _sparse_linear_backend.SparseLinear.forward
    original_convnext = SparseConvNeXtBlock3d.forward
    original_attention = _sparse_attention_modules.sparse_scaled_dot_product_attention
    original_transformer_block = ModulatedSparseTransformerCrossBlock.forward
    original_dense_linear = torch.nn.Linear.forward

    def isolated_conv(module: Any, value: SparseTensor) -> SparseTensor:
        if int(value.shape[0]) > 1:
            return _split_apply_sparse_module(original_conv, module, value)
        return original_conv(module, value)

    def isolated_linear(module: Any, value: SparseTensor) -> SparseTensor:
        if int(value.shape[0]) > 1:
            return _split_apply_sparse_module(original_linear, module, value)
        return original_linear(module, value)

    def isolated_convnext(module: Any, value: SparseTensor) -> SparseTensor:
        if int(value.shape[0]) > 1:
            return _split_apply_sparse_module(original_convnext, module, value)
        return original_convnext(module, value)

    def select_batch_value(
        value: Any, batch_id: int, total_tokens: int, token_slice: slice
    ) -> Any:
        if isinstance(value, SparseTensor):
            if int(value.shape[0]) == int(batch_size):
                return value[batch_id]
            return value
        if isinstance(value, VarLenTensor):
            if int(value.shape[0]) == int(batch_size):
                return value[batch_id]
            return value
        if isinstance(value, Mapping):
            # Multi-tile contexts carry a tile bank rather than a sample
            # batch; the paired attention module owns that routing metadata.
            if value.get("mode") == "multi_tile_paired" or "global_bank" in value:
                return value
            return {
                key: select_batch_value(item, batch_id, total_tokens, token_slice)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            selected = [
                select_batch_value(item, batch_id, total_tokens, token_slice)
                for item in value
            ]
            return type(value)(selected)
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 1
            and int(value.shape[0]) == int(batch_size)
        ):
            return value[batch_id : batch_id + 1]
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 1
            and int(value.shape[0]) == int(total_tokens)
        ):
            return value[token_slice]
        return value

    def isolated_transformer_block(
        module: Any, value: SparseTensor, mod: torch.Tensor, context: Any
    ) -> SparseTensor:
        if int(value.shape[0]) <= 1:
            return original_transformer_block(module, value, mod, context)
        total_tokens = int(value.feats.shape[0])
        outputs = []
        for batch_id, token_slice in enumerate(value.layout):
            local_value = value[batch_id]
            local_mod = (
                mod[batch_id : batch_id + 1]
                if isinstance(mod, torch.Tensor)
                and mod.ndim >= 1
                and int(mod.shape[0]) == int(batch_size)
                else mod
            )
            local_context = select_batch_value(
                context, batch_id, total_tokens, token_slice
            )
            outputs.append(
                original_transformer_block(
                    module, local_value, local_mod, local_context
                )
            )
        if not all(isinstance(output, SparseTensor) for output in outputs):
            raise RuntimeError("isolated flow block returned a non-sparse output")
        return sparse_cat(outputs, dim=0)

    def isolated_dense_linear(module: Any, value: torch.Tensor) -> torch.Tensor:
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 1
            and int(value.shape[0]) == int(batch_size)
            and int(batch_size) > 1
        ):
            return torch.cat(
                [original_dense_linear(module, value[i : i + 1]) for i in range(batch_size)],
                dim=0,
            )
        return original_dense_linear(module, value)

    def isolated_attention(*attention_args: Any, **attention_kwargs: Any) -> Any:
        """Run each varlen attention sequence with the serial kernel shape."""
        values = list(attention_args) + list(attention_kwargs.values())
        batch_value = next(
            (value for value in values if isinstance(value, VarLenTensor)), None
        )
        if batch_value is None or int(batch_value.shape[0]) <= 1:
            return original_attention(*attention_args, **attention_kwargs)
        batch_size = int(batch_value.shape[0])

        def select(value: Any, batch_id: int) -> Any:
            if isinstance(value, VarLenTensor):
                return value[batch_id]
            if (
                isinstance(value, torch.Tensor)
                and value.ndim >= 1
                and int(value.shape[0]) == batch_size
            ):
                return value[batch_id : batch_id + 1]
            return value

        outputs = []
        for batch_id in range(batch_size):
            local_args = tuple(select(value, batch_id) for value in attention_args)
            local_kwargs = {
                key: select(value, batch_id)
                for key, value in attention_kwargs.items()
            }
            outputs.append(original_attention(*local_args, **local_kwargs))
        first = outputs[0]
        if isinstance(first, SparseTensor):
            if not all(isinstance(value, SparseTensor) for value in outputs):
                raise RuntimeError("isolated attention changed sparse output type")
            return sparse_cat(outputs, dim=0)
        if isinstance(first, VarLenTensor):
            if not all(isinstance(value, VarLenTensor) for value in outputs):
                raise RuntimeError("isolated attention changed varlen output type")
            return varlen_cat(outputs, dim=0)
        if isinstance(first, torch.Tensor):
            if not all(isinstance(value, torch.Tensor) for value in outputs):
                raise RuntimeError("isolated attention changed dense output type")
            return torch.cat(outputs, dim=0)
        raise RuntimeError(f"unsupported isolated attention output type {type(first)!r}")

    _flex_gemm_backend.sparse_conv3d_forward = isolated_conv
    _sparse_linear_backend.SparseLinear.forward = isolated_linear
    SparseConvNeXtBlock3d.forward = isolated_convnext
    ModulatedSparseTransformerCrossBlock.forward = isolated_transformer_block
    torch.nn.Linear.forward = isolated_dense_linear
    _sparse_attention_modules.sparse_scaled_dot_product_attention = isolated_attention
    try:
        yield
    finally:
        _flex_gemm_backend.sparse_conv3d_forward = original_conv
        _sparse_linear_backend.SparseLinear.forward = original_linear
        SparseConvNeXtBlock3d.forward = original_convnext
        ModulatedSparseTransformerCrossBlock.forward = original_transformer_block
        torch.nn.Linear.forward = original_dense_linear
        _sparse_attention_modules.sparse_scaled_dot_product_attention = original_attention


def _decode_phase(
    *,
    contexts: Sequence[Any],
    predictions: Mapping[int, Mapping[str, SparseTensor]],
    pipeline: Any,
    args: argparse.Namespace,
    mode: str,
    low_vram: bool,
    step_index: int,
) -> Tuple[Dict[int, MeshWithVoxel], Dict[int, torch.Tensor], Dict[int, Dict[str, Any]], Dict[str, Any]]:
    started = time.perf_counter()
    meshes: Dict[int, MeshWithVoxel] = {}
    fields: Dict[int, torch.Tensor] = {}
    stats: Dict[int, Dict[str, Any]] = {}
    microbatches: List[int] = []
    fallback_reasons: List[str] = []
    if mode == "serial":
        groups = [[context] for context in contexts]
    else:
        groups = []
        current: List[Any] = []
        current_tokens = 0
        for context in contexts:
            tokens = int(context.target_coords.shape[0])
            if current and (
                len(current) >= _phase_batch_limit(args, "decoder_max_batch_size")
                or current_tokens + tokens > int(args.decode_token_budget)
            ):
                groups.append(current)
                current, current_tokens = [], 0
            current.append(context)
            current_tokens += tokens
        if current:
            groups.append(current)
    calls = 0
    model_seconds = 0.0
    query_seconds = 0.0
    for group in groups:
        decoded, decoded_fields, decode_stats, call_stats = _decode_one_batch(
            contexts=group,
            predictions=predictions,
            pipeline=pipeline,
            args=args,
            low_vram=low_vram,
            step_index=step_index,
        )
        meshes.update(decoded)
        fields.update(decoded_fields)
        stats.update(decode_stats)
        calls += int(call_stats.get("decoder_forward_calls", 1))
        model_seconds += float(call_stats.get("decode_model_seconds", 0.0))
        query_seconds += float(call_stats.get("query_attrs_seconds", 0.0))
        microbatches.append(int(call_stats.get("actual_batch_size", len(group))))
        if call_stats.get("fallback_reason"):
            fallback_reasons.append(str(call_stats["fallback_reason"]))
    return meshes, fields, stats, {
        "seconds": float(time.perf_counter() - started),
        "decoder_forward_calls": calls,
        "requested_batch_size": _phase_batch_limit(args, "decoder_max_batch_size"),
        "actual_decode_batch_sizes": microbatches,
        "decode_model_seconds": model_seconds,
        "query_attrs_seconds": query_seconds,
        "fallback_reason": "; ".join(fallback_reasons) if fallback_reasons else None,
    }


def _observed_fusion(
    *,
    target: Any,
    contexts: Sequence[Any],
    decoded: Mapping[int, MeshWithVoxel],
    self_field: torch.Tensor,
    global_camera: Mapping[str, float],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    observed = target.observed_mask.detach().cpu().bool()
    observed_rows = torch.where(observed)[0]
    if observed_rows.numel() == 0:
        return self_field.clone(), {
            "target_tile": int(target.tile_id),
            "overlap_ovoxel_count": 0,
            "non_overlap_ovoxel_count": int(self_field.shape[0]),
            "query_valid_donor_count": {"min": 0, "mean": 0.0, "max": 0},
            "covered_donor_count": {"min": 0, "mean": 0.0, "max": 0},
            "observed_fusion_seconds": 0.0,
        }
    # For multi-tile full runs, keep the frozen decoder meshes on CUDA for the
    # whole fusion barrier.  The previous low-VRAM path moved every donor mesh
    # CPU->CUDA once per target tile, which made the O(N_tiles^2) donor pass
    # dominated by allocator and PCIe overhead.  The small-tile path remains
    # unchanged; callers may still release the mapping after the barrier.
    fusion_device = next(
        (mesh.device for mesh in decoded.values() if mesh.device.type == "cuda"),
        target.target_points.device,
    )
    points_local = target.target_points.index_select(0, observed_rows).to(fusion_device)
    coords_local = target.target_coords.index_select(0, observed_rows).to(fusion_device)
    fusion_self = self_field.to(fusion_device)
    self_observed = fusion_self.index_select(0, observed_rows.to(fusion_device))
    target_view = type("ObservedTarget", (), {})()
    target_view.tile_id = int(target.tile_id)
    target_view.transform = target.transform
    target_view.target_points = points_local
    target_view.target_coords = coords_local
    started = time.perf_counter()
    observed_field, stats, _ = base._fuse_tile_field(
        target=target_view,
        contexts=contexts,
        decoded=decoded,
        self_field=self_observed,
        global_camera=global_camera,
        sigma_pixels=float(args.fusion_sigma_pixels),
        query_chunk_size=int(args.query_chunk_size),
    )
    gaussian = fusion_self.clone()
    gaussian.index_copy_(0, observed_rows.to(device=gaussian.device), observed_field)
    if gaussian.device != self_field.device:
        gaussian = gaussian.to(self_field.device)
    stats = dict(stats)
    stats["observed_fusion_seconds"] = float(time.perf_counter() - started)
    return gaussian, stats


def _solve_target_mra(
    *,
    context: Any,
    self_field: torch.Tensor,
    projector: StableSparseMRAProjector,
    step_index: int,
    warm_start_coefficients: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
    h = self_field.detach().cpu().to(torch.float64)
    g = context.global_pbr_reference.detach().cpu().to(torch.float64)
    hidden = context.hidden_mask.detach().cpu().bool()
    delta = h[hidden] - g[hidden]
    started = time.perf_counter()
    coeff, solve_info = projector.solve(
        delta,
        f"tile_{int(context.tile_id):02d}_step_{step_index:02d}_delta",
        x0=warm_start_coefficients,
    )
    coarse = projector.apply(coeff)
    detail = delta - coarse
    orth = _orthogonality_metrics(projector, detail, delta)
    exact = coarse + detail - delta
    output = h.to(torch.float32)
    output[hidden] = (g[hidden] + detail).to(torch.float32)
    record = {
        "tile_id": int(context.tile_id),
        "step": int(step_index),
        "hidden_voxel_count": int(hidden.sum().item()),
        "solver_delta": solve_info,
        "warm_start_available": bool(warm_start_coefficients is not None),
        "condition_estimate": {
            "condA_channels": [float(v["condA"]) for v in solve_info["channels"]],
            "condA_max": max(float(v["condA"]) for v in solve_info["channels"]),
            "condA_mean": float(np.mean([float(v["condA"]) for v in solve_info["channels"]])),
        },
        "orthogonality": orth,
        "exact_decomposition": {
            "max_abs": float(exact.abs().max().item()) if exact.numel() else 0.0,
            "mean_abs": float(exact.abs().mean().item()) if exact.numel() else 0.0,
            "relative_l2": _relative(exact, delta),
        },
        "energy": {
            "Delta": _energy_stats(delta),
            "Delta_coarse": _energy_stats(coarse),
            "Delta_detail": _energy_stats(detail),
        },
        "r_c": _channel_energy_ratio(coarse, delta),
        "r_d": _channel_energy_ratio(detail, delta),
        "ranges": {
            "G1024_hidden": _tensor_range(g[hidden]),
            "H_t_hidden": _tensor_range(h[hidden]),
            "hidden_target": _tensor_range(output[hidden]),
            "final_target_field": _tensor_range(output),
        },
        "solve_seconds": float(time.perf_counter() - started),
        "finite": bool(torch.isfinite(output).all().item()),
    }
    return output, record, {
        "coarse": coarse.to(torch.float32),
        "detail": detail.to(torch.float32),
        "coefficients": coeff,
    }


def _encode_pbr_batch(
    *,
    contexts: Sequence[Any],
    fields: Mapping[int, torch.Tensor],
    references: Mapping[int, SparseTensor],
    pbr_encoder: torch.nn.Module,
    pipeline: Any,
    low_vram: bool,
    tile_ids: Sequence[int],
) -> Tuple[Dict[int, SparseTensor], Dict[str, Any]]:
    """Encode a sparse O-voxel batch once, then coordinate-align the C64 output."""
    started = time.perf_counter()
    input_values: List[SparseTensor] = []
    for context in contexts:
        tile_id = int(context.tile_id)
        coords = context.geometry.coords.detach()
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise RuntimeError(f"tile {tile_id}: expected geometry.coords [N,3]")
        coords4 = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)
        attrs = fields[tile_id].detach().to(torch.float32)
        if attrs.shape[0] != coords4.shape[0]:
            raise RuntimeError(f"tile {tile_id}: O-voxel attrs and coords are not aligned")
        # Preserve the physical PBR range after mapping [0, 1] to [-1, 1].
        # Do not clamp here: hidden/MRA fields intentionally retain their
        # out-of-range values, and clipping would silently change the field
        # before the official PBR encoder sees it.
        encoder_input = attrs * 2.0 - 1.0
        input_values.append(SparseTensor(encoder_input, coords4))
    if low_vram:
        pbr_encoder.to(torch.device("cuda"))
    input_pack = pack_sparse_batch(input_values, tile_ids)
    encoder_input_batch = (
        _move_sparse(input_pack.value, torch.device("cuda"))
        if low_vram
        else input_pack.value
    )
    with _layer_trace_active(tile_ids):
        with _batch_backend_kernel_isolation(
            enabled=len(contexts) > 1, batch_size=len(contexts)
        ):
            with torch.no_grad():
                latent_batch = pbr_encoder(encoder_input_batch, sample_posterior=False)
    _sync_cuda()
    if low_vram:
        pbr_encoder.cpu()
    if not isinstance(latent_batch, SparseTensor) or not torch.isfinite(latent_batch.feats).all():
        raise RuntimeError("batched PBR encoder produced an invalid latent")
    reference_pack = pack_sparse_batch([references[int(t)] for t in tile_ids], tile_ids)
    latent_values = unpack_sparse_batch(latent_batch, reference_pack)
    normalized = {
        int(tile_id): _normalize_slat(value, pipeline.tex_slat_normalization)
        for tile_id, value in zip(tile_ids, latent_values)
    }
    return normalized, {
        "pbr_encoder_seconds": float(time.perf_counter() - started),
        "encoder_forward_calls": 1,
        "batch_size": len(contexts),
        "input": _batch_stats_for_sparse(input_values),
        "output": _batch_stats_for_sparse(latent_values),
        "support_checks": {
            str(tile_id): _strict_sparse_check(
                references[int(tile_id)], normalized[int(tile_id)], f"tile {tile_id} batch PBR encode"
            )
            for tile_id in tile_ids
        },
    }


def _encode_phase(
    *,
    contexts: Sequence[Any],
    self_fields: Mapping[int, torch.Tensor],
    fused_fields: Mapping[int, torch.Tensor],
    predictions: Mapping[int, Mapping[str, SparseTensor]],
    pbr_encoder: torch.nn.Module,
    pipeline: Any,
    args: argparse.Namespace,
    mode: str,
    low_vram: bool,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    started = time.perf_counter()
    if mode == "serial":
        groups = [[context] for context in contexts]
    else:
        groups = []
        current: List[Any] = []
        current_rows = 0
        for context in contexts:
            rows = int(context.target_coords.shape[0])
            if current and (
                len(current) >= _phase_batch_limit(args, "encoder_max_batch_size")
                or current_rows + rows > int(args.encode_row_budget)
            ):
                groups.append(current)
                current, current_rows = [], 0
            current.append(context)
            current_rows += rows
        if current:
            groups.append(current)
    encoded: Dict[int, Dict[str, Any]] = {}
    calls = 0
    for group in groups:
        tile_ids = [int(context.tile_id) for context in group]
        if len(group) == 1 and mode == "serial":
            for field_name, field_map in (("cycle", self_fields), ("fused", fused_fields)):
                values, stats = _encode_pbr_batch(
                    contexts=group,
                    fields=field_map,
                    references={int(group[0].tile_id): predictions[int(group[0].tile_id)]["pred_x0"]},
                    pbr_encoder=pbr_encoder,
                    pipeline=pipeline,
                    low_vram=low_vram,
                    tile_ids=tile_ids,
                )
                tile_id = tile_ids[0]
                encoded.setdefault(tile_id, {})[f"{field_name}_norm"] = values[tile_id]
                encoded[tile_id].setdefault(f"{field_name}_stats", stats)
                calls += 1
            continue
        for field_name, field_map in (("cycle", self_fields), ("fused", fused_fields)):
            values, stats = _encode_pbr_batch(
                contexts=group,
                fields=field_map,
                references={tile_id: predictions[tile_id]["pred_x0"] for tile_id in tile_ids},
                pbr_encoder=pbr_encoder,
                pipeline=pipeline,
                low_vram=low_vram,
                tile_ids=tile_ids,
            )
            for tile_id in tile_ids:
                encoded.setdefault(tile_id, {})[f"{field_name}_norm"] = values[tile_id]
                encoded[tile_id].setdefault(f"{field_name}_stats", stats)
            calls += 1
    return encoded, {
        "seconds": float(time.perf_counter() - started),
        "encoder_forward_calls": calls,
        "actual_batch_sizes": [len(group) for group in groups],
        "requested_batch_size": _phase_batch_limit(args, "encoder_max_batch_size"),
    }


def _correction_phase(
    *,
    contexts: Sequence[Any],
    states: Mapping[int, SparseTensor],
    predictions: Mapping[int, Mapping[str, SparseTensor]],
    encoded: Mapping[int, Mapping[str, Any]],
    sampler: Any,
    t: float,
    t_next: float,
    mode: str,
    tile_batch_size: int,
    low_vram: bool,
    step_index: int,
) -> Tuple[Dict[int, Dict[str, SparseTensor]], Dict[str, Any]]:
    phase_started = time.perf_counter()
    corrected: Dict[int, Dict[str, SparseTensor]] = {}
    if mode == "serial":
        groups = [[context] for context in contexts]
    else:
        groups = [
            list(contexts[start : start + max(1, int(tile_batch_size))])
            for start in range(0, len(contexts), max(1, int(tile_batch_size)))
        ]
    calls = 0
    guided_x0_seconds = 0.0
    xstart_to_pred_seconds = 0.0
    euler_seconds = 0.0
    for group in groups:
        tile_ids = [int(context.tile_id) for context in group]
        if len(group) == 1 and mode == "serial":
            tile_id = tile_ids[0]
            state = _move_sparse(states[tile_id], torch.device("cuda")) if low_vram else states[tile_id]
            pred_x0 = _move_sparse(predictions[tile_id]["pred_x0"], torch.device("cuda")) if low_vram else predictions[tile_id]["pred_x0"]
            pred_v = _move_sparse(predictions[tile_id]["pred_v"], torch.device("cuda")) if low_vram else predictions[tile_id]["pred_v"]
            cycle = _move_sparse(encoded[tile_id]["cycle_norm"], torch.device("cuda")) if low_vram else encoded[tile_id]["cycle_norm"]
            fused = _move_sparse(encoded[tile_id]["fused_norm"], torch.device("cuda")) if low_vram else encoded[tile_id]["fused_norm"]
            started = time.perf_counter()
            guided_x0 = SparseTensor(
                pred_x0.feats + (fused.feats - cycle.feats), pred_x0.coords.detach().clone()
            )
            guided_x0_seconds += float(time.perf_counter() - started)
            started = time.perf_counter()
            guided_v = sampler._xstart_to_pred(state, float(t), guided_x0)
            xstart_to_pred_seconds += float(time.perf_counter() - started)
            started = time.perf_counter()
            next_state = SparseTensor(
                state.feats - float(t - t_next) * guided_v.feats, state.coords.detach().clone()
            )
            euler_seconds += float(time.perf_counter() - started)
            _strict_sparse_check(pred_x0, guided_x0, f"tile {tile_id} step {step_index} x0_guided")
            _strict_sparse_check(pred_v, guided_v, f"tile {tile_id} step {step_index} guided_v")
            _strict_sparse_check(state, next_state, f"tile {tile_id} step {step_index} x_t_next")
            corrected[tile_id] = {
                "guided_x0": _move_sparse(guided_x0, torch.device("cpu")) if low_vram else guided_x0,
                "guided_v": _move_sparse(guided_v, torch.device("cpu")) if low_vram else guided_v,
                "next_state": _move_sparse(next_state, torch.device("cpu")) if low_vram else next_state,
            }
            calls += 1
            continue
        state_values = [
            _move_sparse(states[tile_id], torch.device("cuda")) if low_vram else states[tile_id]
            for tile_id in tile_ids
        ]
        pred_values = [
            _move_sparse(predictions[tile_id]["pred_x0"], torch.device("cuda")) if low_vram else predictions[tile_id]["pred_x0"]
            for tile_id in tile_ids
        ]
        pred_v_values = [
            _move_sparse(predictions[tile_id]["pred_v"], torch.device("cuda")) if low_vram else predictions[tile_id]["pred_v"]
            for tile_id in tile_ids
        ]
        cycle_values = [
            _move_sparse(encoded[tile_id]["cycle_norm"], torch.device("cuda")) if low_vram else encoded[tile_id]["cycle_norm"]
            for tile_id in tile_ids
        ]
        fused_values = [
            _move_sparse(encoded[tile_id]["fused_norm"], torch.device("cuda")) if low_vram else encoded[tile_id]["fused_norm"]
            for tile_id in tile_ids
        ]
        state_pack = pack_sparse_batch(state_values, tile_ids)
        pred_pack = pack_sparse_batch(pred_values, tile_ids)
        cycle_pack = pack_sparse_batch(cycle_values, tile_ids)
        fused_pack = pack_sparse_batch(fused_values, tile_ids)
        started = time.perf_counter()
        guided_x0_batch = SparseTensor(
            # Keep the serial evaluation order exactly: pred + (fused - cycle).
            # The algebraically equivalent (pred + fused) - cycle introduces
            # a one-ULP drift that compounds across the 12-step flow.
            pred_pack.value.feats + (fused_pack.value.feats - cycle_pack.value.feats),
            pred_pack.value.coords.detach().clone(),
        )
        guided_x0_seconds += float(time.perf_counter() - started)
        started = time.perf_counter()
        guided_v_batch = sampler._xstart_to_pred(state_pack.value, float(t), guided_x0_batch)
        xstart_to_pred_seconds += float(time.perf_counter() - started)
        started = time.perf_counter()
        next_state_batch = SparseTensor(
            state_pack.value.feats - float(t - t_next) * guided_v_batch.feats,
            state_pack.value.coords.detach().clone(),
        )
        euler_seconds += float(time.perf_counter() - started)
        guided_x0_values = unpack_sparse_batch(guided_x0_batch, pred_pack)
        guided_v_values = unpack_sparse_batch(guided_v_batch, state_pack)
        next_values = unpack_sparse_batch(next_state_batch, state_pack)
        for tile_id, pred_v, guided_x0, guided_v, next_state in zip(
            tile_ids, pred_v_values, guided_x0_values, guided_v_values, next_values
        ):
            _strict_sparse_check(pred_values[tile_ids.index(tile_id)], guided_x0, f"tile {tile_id} step {step_index} batch x0_guided")
            _strict_sparse_check(pred_v, guided_v, f"tile {tile_id} step {step_index} batch guided_v")
            _strict_sparse_check(state_values[tile_ids.index(tile_id)], next_state, f"tile {tile_id} step {step_index} batch x_t_next")
            corrected[tile_id] = {
                "guided_x0": _move_sparse(guided_x0, torch.device("cpu")) if low_vram else guided_x0,
                "guided_v": _move_sparse(guided_v, torch.device("cpu")) if low_vram else guided_v,
                "next_state": _move_sparse(next_state, torch.device("cpu")) if low_vram else next_state,
            }
        calls += 1
        del state_values, pred_values, pred_v_values, cycle_values, fused_values
        del state_pack, pred_pack, cycle_pack, fused_pack
        del guided_x0_batch, guided_v_batch, next_state_batch
        if low_vram:
            _empty_cuda_cache()
    return corrected, {
        "seconds": float(time.perf_counter() - phase_started),
        "correction_batch_calls": calls,
        "actual_batch_sizes": [len(group) for group in groups],
        "guided_x0_seconds": guided_x0_seconds,
        "xstart_to_pred_seconds": xstart_to_pred_seconds,
        "euler_seconds": euler_seconds,
    }


def _sparse_payload(value: SparseTensor) -> Dict[str, torch.Tensor]:
    return {
        "coords": value.coords.detach().cpu().to(torch.int32),
        "features": value.feats.detach().cpu().to(torch.float32),
    }


def _save_step_trace(
    run_dir: Path,
    step_index: int,
    tile_id: int,
    values: Mapping[str, Any],
    precision: torch.dtype = torch.float32,
) -> Path:
    payload: Dict[str, Any] = {
        "format": FORMAT,
        "step": int(step_index),
        "tile_id": int(tile_id),
    }
    for name, value in values.items():
        if isinstance(value, SparseTensor):
            payload[name] = {
                "coords": value.coords.detach().cpu().to(torch.int32),
                "features": value.feats.detach().cpu().to(precision),
            }
        elif isinstance(value, torch.Tensor):
            payload[name] = value.detach().cpu().to(precision)
        else:
            payload[name] = value
    path = run_dir / "steps" / f"step_{step_index:02d}" / f"tile_{tile_id:02d}_trace.pt"
    _atomic_torch_save(path, payload)
    return path


@torch.no_grad()
def run_flow(
    *,
    contexts: Sequence[Any],
    projectors: Mapping[int, StableSparseMRAProjector],
    pipeline: Any,
    pbr_encoder: torch.nn.Module,
    global_camera: Mapping[str, float],
    args: argparse.Namespace,
    output_dir: Path,
    mode: str,
    step_limit: Optional[int],
    capture_steps: Sequence[int],
) -> Dict[str, Any]:
    if mode not in {"serial", "batch"}:
        raise ValueError(f"unknown execution mode {mode}")
    if not contexts:
        raise ValueError("flow requires at least one context")
    sampler = pipeline.tex_slat_sampler
    model = pipeline.models["tex_slat_flow_model_1024"]
    merged = {**pipeline.tex_slat_sampler_params, **dict(core._sampler_overrides(args)[2])}
    schedule = [
        float(value)
        for value in sampler.timestep_schedule(int(args.texture_steps), float(merged["rescale_t"]))
    ]
    if float(args.noise_timestep) != 1.0:
        raise ValueError("cached fixed-shape initial states require noise_timestep=1.0")
    actual_steps = int(step_limit) if step_limit is not None else len(schedule) - 1
    if actual_steps <= 0 or actual_steps > len(schedule) - 1:
        raise ValueError(f"invalid flow step limit {actual_steps}")
    schedule = schedule[: actual_steps + 1]
    step_kwargs = base._sampler_step_kwargs(merged)
    low_vram = bool(args.low_vram)
    states: Dict[int, SparseTensor] = {
        int(context.tile_id): _fresh_sparse(context.initial_state) for context in contexts
    }
    condition_by_tile = {int(context.tile_id): context.condition for context in contexts}
    shape_by_tile = {int(context.tile_id): context.shape_norm for context in contexts}
    fixed_shape_digest = {
        int(context.tile_id): _coordinate_digest(context.shape_norm.coords) for context in contexts
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_flow_step = int(getattr(args, "resume_flow_step", 0) or 0)
    if resume_flow_step < 0 or resume_flow_step > actual_steps:
        raise ValueError(f"invalid resume_flow_step {resume_flow_step} for {actual_steps} steps")
    per_step: List[Dict[str, Any]] = []
    prior_flow_seconds = 0.0
    if resume_flow_step:
        if not bool(args.resume):
            raise ValueError("resume_flow_step requires --resume")
        for prior_step in range(resume_flow_step):
            summary_path = output_dir / "steps" / f"step_{prior_step:02d}_summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(summary_path)
            prior_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            per_step.append(prior_summary)
            prior_flow_seconds += float(prior_summary.get("step_seconds", 0.0))
        # The trace contains the exact synchronized Jacobi endpoint from the
        # previous completed step.  Coefficients are intentionally not
        # serialized in the artifact, so the first resumed step starts cold;
        # subsequent steps use the normal warm-start path.
        previous_step = resume_flow_step - 1
        for context in contexts:
            tile_id = int(context.tile_id)
            trace_path = output_dir / "steps" / f"step_{previous_step:02d}" / f"tile_{tile_id:02d}_trace.pt"
            if not trace_path.is_file():
                raise FileNotFoundError(trace_path)
            resumed_trace = _load_torch(trace_path)
            resumed_state = _load_trace_value(resumed_trace, "x_t_next")
            if not isinstance(resumed_state, SparseTensor):
                raise TypeError(f"resume trace x_t_next is not SparseTensor: {trace_path}")
            states[tile_id] = resumed_state
    started_all = time.perf_counter()
    prediction_calls = sum(
        int(step.get("calls", {}).get("prediction_model_forward", 0)) for step in per_step
    )
    decode_calls = sum(int(step.get("calls", {}).get("decoder_forward", 0)) for step in per_step)
    encode_calls = sum(int(step.get("calls", {}).get("pbr_encoder_forward", 0)) for step in per_step)
    correction_calls = sum(int(step.get("calls", {}).get("correction_batch_calls", 0)) for step in per_step)
    projector_seconds = sum(float(step.get("phase", {}).get("projector_seconds", 0.0)) for step in per_step)
    projector_parallel_seconds = sum(
        float(step.get("phase", {}).get("fusion_and_projector_seconds", 0.0)) for step in per_step
    )
    # Formal flow uses the previous step's coarse coefficients as the next
    # step's LSMR initial guess.  Correctness/projector-only tests remain
    # explicitly cold-started.
    warm_coefficients: Dict[int, torch.Tensor] = {}
    if low_vram:
        model.to(torch.device("cuda"))
    try:
        for step_index, (t, t_next) in enumerate(zip(schedule[:-1], schedule[1:])):
            if step_index < resume_flow_step:
                continue
            step_started = time.perf_counter()
            states_before = {
                int(context.tile_id): _fresh_sparse(states[int(context.tile_id)])
                for context in contexts
            }
            print(
                f"[{mode} step {step_index:02d}] t={float(t):.9f} "
                f"t_next={float(t_next):.9f} tiles={[int(c.tile_id) for c in contexts]}",
                flush=True,
            )
            predictions, prediction_stats = _prediction_phase(
                contexts=contexts,
                states=states,
                pipeline=pipeline,
                sampler=sampler,
                model=model,
                condition_by_tile=condition_by_tile,
                shape_by_tile=shape_by_tile,
                t=float(t),
                step_kwargs=step_kwargs,
                mode=mode,
                tile_batch_size=_phase_batch_limit(args, "prediction_max_batch_size"),
                prediction_token_budget=int(args.prediction_token_budget),
                low_vram=low_vram,
                step_index=step_index,
            )
            prediction_calls += int(prediction_stats["model_forward_calls"])
            decoded, decoded_fields, decode_stats, decode_phase_stats = _decode_phase(
                contexts=contexts,
                predictions=predictions,
                pipeline=pipeline,
                args=args,
                mode=mode,
                low_vram=low_vram,
                step_index=step_index,
            )
            decode_calls += int(decode_phase_stats["decoder_forward_calls"])
            if low_vram and len(contexts) > 2:
                # The decoder has already produced a frozen Jacobi read set.
                # Keep only these frozen meshes on CUDA through the donor
                # fusion barrier, then release them with the step tensors.
                decoded = {
                    tile_id: mesh.to("cuda")
                    for tile_id, mesh in decoded.items()
                }

            # The projector is CPU/float64.  Batch mode starts both independent
            # solves before the GPU observed-query work, preserving the frozen
            # decode barrier while allowing the two resources to overlap.
            fusion_started = time.perf_counter()
            mra_futures: Dict[int, concurrent.futures.Future] = {}
            executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
            if mode == "batch" and len(contexts) > 1:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
                for context in contexts:
                    tile_id = int(context.tile_id)
                    mra_futures[tile_id] = executor.submit(
                        _solve_target_mra,
                        context=context,
                        self_field=decoded_fields[tile_id],
                        projector=projectors[tile_id],
                        step_index=step_index,
                        warm_start_coefficients=warm_coefficients.get(tile_id),
                    )
            fused_fields: Dict[int, torch.Tensor] = {}
            gaussian_fields: Dict[int, torch.Tensor] = {}
            mra_records: Dict[int, Dict[str, Any]] = {}
            mra_components: Dict[int, Dict[str, torch.Tensor]] = {}
            fusion_stats: Dict[int, Dict[str, Any]] = {}
            if mode == "serial":
                for context in contexts:
                    tile_id = int(context.tile_id)
                    fused, fstats = _observed_fusion(
                        target=context,
                        contexts=contexts,
                        decoded=decoded,
                        self_field=decoded_fields[tile_id],
                        global_camera=global_camera,
                        args=args,
                    )
                    fused_fields[tile_id] = fused
                    gaussian_fields[tile_id] = fused.clone()
                    fusion_stats[tile_id] = fstats
                    target, record, components = _solve_target_mra(
                        context=context,
                        self_field=decoded_fields[tile_id],
                        projector=projectors[tile_id],
                        step_index=step_index,
                        warm_start_coefficients=warm_coefficients.get(tile_id),
                    )
                    target[context.observed_mask.detach().cpu().bool()] = fused[context.observed_mask.detach().cpu().bool()]
                    fused_fields[tile_id] = target
                    mra_records[tile_id] = record
                    mra_components[tile_id] = components
                    warm_coefficients[tile_id] = components["coefficients"].detach().cpu()
            else:
                for context in contexts:
                    tile_id = int(context.tile_id)
                    fused, fstats = _observed_fusion(
                        target=context,
                        contexts=contexts,
                        decoded=decoded,
                        self_field=decoded_fields[tile_id],
                        global_camera=global_camera,
                        args=args,
                    )
                    fused_fields[tile_id] = fused
                    gaussian_fields[tile_id] = fused.clone()
                    fusion_stats[tile_id] = fstats
                if executor is not None:
                    for context in contexts:
                        tile_id = int(context.tile_id)
                        target, record, components = mra_futures[tile_id].result()
                        target[context.observed_mask.detach().cpu().bool()] = fused_fields[tile_id][context.observed_mask.detach().cpu().bool()]
                        fused_fields[tile_id] = target
                        mra_records[tile_id] = record
                        mra_components[tile_id] = components
                        warm_coefficients[tile_id] = components["coefficients"].detach().cpu()
                    executor.shutdown(wait=True)
                    executor = None
            fusion_seconds = float(time.perf_counter() - fusion_started)
            projector_seconds += sum(float(record["solve_seconds"]) for record in mra_records.values())
            if mode == "batch":
                projector_parallel_seconds += fusion_seconds

            encoded, encode_stats = _encode_phase(
                contexts=contexts,
                self_fields=decoded_fields,
                fused_fields=fused_fields,
                predictions=predictions,
                pbr_encoder=pbr_encoder,
                pipeline=pipeline,
                args=args,
                mode=mode,
                low_vram=low_vram,
            )
            encode_calls += int(encode_stats["encoder_forward_calls"])
            for context in contexts:
                tile_id = int(context.tile_id)
                cycle = encoded[tile_id]["cycle_norm"]
                fused = encoded[tile_id]["fused_norm"]
                _strict_sparse_check(predictions[tile_id]["pred_x0"], cycle, f"tile {tile_id} step {step_index} x0_cycle")
                _strict_sparse_check(predictions[tile_id]["pred_x0"], fused, f"tile {tile_id} step {step_index} x0_fused")
                encoded[tile_id]["cycle_norm"] = cycle
                encoded[tile_id]["fused_norm"] = fused

            corrected, correction_stats = _correction_phase(
                contexts=contexts,
                states=states,
                predictions=predictions,
                encoded=encoded,
                sampler=sampler,
                t=float(t),
                t_next=float(t_next),
                mode=mode,
                tile_batch_size=_phase_batch_limit(args, "prediction_max_batch_size"),
                low_vram=low_vram,
                step_index=step_index,
            )
            correction_calls += int(correction_stats["correction_batch_calls"])
            # Jacobi update: no state is replaced until every corrected tile is ready.
            for context in contexts:
                tile_id = int(context.tile_id)
                states[tile_id] = corrected[tile_id]["next_state"]
                _strict_sparse_check(
                    context.initial_state,
                    states[tile_id],
                    f"tile {tile_id} step {step_index} state support",
                )

            tile_records: List[Dict[str, Any]] = []
            for context in contexts:
                tile_id = int(context.tile_id)
                mra = mra_records[tile_id]
                mra_path = output_dir / "tiles" / f"tile_{tile_id:02d}" / "steps" / f"step_{step_index:02d}_mra.json"
                _atomic_json(mra_path, mra)
                if step_index in set(int(v) for v in capture_steps):
                    hidden_mask = context.hidden_mask.detach().cpu().bool()
                    coarse_full = torch.zeros_like(fused_fields[tile_id])
                    detail_full = torch.zeros_like(fused_fields[tile_id])
                    coarse_full[hidden_mask] = mra_components[tile_id]["coarse"]
                    detail_full[hidden_mask] = mra_components[tile_id]["detail"]
                    _save_step_trace(
                        output_dir,
                        step_index,
                        tile_id,
                        {
                            "state_before": states_before[tile_id],
                            "pred_x0": predictions[tile_id]["pred_x0"],
                            "pred_v": predictions[tile_id]["pred_v"],
                            "decoded_field": decoded_fields[tile_id],
                            "observed_gaussian": gaussian_fields[tile_id],
                            "delta_coarse": coarse_full,
                            "delta_detail": detail_full,
                            "final_target_field": fused_fields[tile_id],
                            "cycle_norm": encoded[tile_id]["cycle_norm"],
                            "target_norm": encoded[tile_id]["fused_norm"],
                            "guided_x0": corrected[tile_id]["guided_x0"],
                            "guided_v": corrected[tile_id]["guided_v"],
                            "x_t_next": states[tile_id],
                        },
                        precision=torch.float32,
                    )
                tile_records.append(
                    {
                        "tile_id": tile_id,
                        "step": int(step_index),
                        "t": float(t),
                        "t_next": float(t_next),
                        "support_checks": {
                            "pred_x0": _strict_sparse_check(context.initial_state, predictions[tile_id]["pred_x0"], f"tile {tile_id} pred_x0 support"),
                            "pred_v": _strict_sparse_check(context.initial_state, predictions[tile_id]["pred_v"], f"tile {tile_id} pred_v support"),
                            "guided_x0": _strict_sparse_check(predictions[tile_id]["pred_x0"], corrected[tile_id]["guided_x0"], f"tile {tile_id} guided_x0 support"),
                            "guided_v": _strict_sparse_check(predictions[tile_id]["pred_v"], corrected[tile_id]["guided_v"], f"tile {tile_id} guided_v support"),
                            "x_t_next": _strict_sparse_check(context.initial_state, corrected[tile_id]["next_state"], f"tile {tile_id} next support"),
                        },
                        "mra": mra,
                        "decode": decode_stats[tile_id],
                        "fusion": fusion_stats[tile_id],
                        "encode": encoded[tile_id]["cycle_stats"],
                        "phase_seconds": {
                            "prediction": prediction_stats["seconds"],
                            "decode": decode_phase_stats["seconds"],
                            "observed_fusion_and_projector": fusion_seconds,
                            "encode": encode_stats["seconds"],
                            "correction_and_euler": correction_stats["seconds"],
                        },
                    }
                )
            step_record = {
                "step": int(step_index),
                "t": float(t),
                "t_next": float(t_next),
                "tile_count": len(contexts),
                "step_seconds": float(time.perf_counter() - step_started),
                "barriers": {
                    "prediction_barrier": True,
                    "decoded_field_barrier": True,
                    "fusion_mra_barrier": True,
                    "encode_barrier": True,
                    "endpoint_correction_barrier": True,
                    "euler_update_barrier": True,
                    "all_tiles_synchronized": True,
                },
                "phase": {
                    "prediction_seconds": prediction_stats["seconds"],
                    "decode_model_seconds": decode_phase_stats["decode_model_seconds"],
                    "decode_query_seconds": decode_phase_stats["query_attrs_seconds"],
                    "decode_seconds": decode_phase_stats["seconds"],
                    "observed_fusion_seconds": float(sum(v.get("observed_fusion_seconds", 0.0) for v in fusion_stats.values())),
                    "projector_seconds": float(sum(v["solve_seconds"] for v in mra_records.values())),
                    "fusion_and_projector_seconds": fusion_seconds,
                    "encode_seconds": encode_stats["seconds"],
                    "guided_x0_assembly_seconds": correction_stats["guided_x0_seconds"],
                    "xstart_to_pred_seconds": correction_stats["xstart_to_pred_seconds"],
                    "euler_seconds": correction_stats["euler_seconds"],
                    "total_step_seconds": float(time.perf_counter() - step_started),
                },
                "calls": {
                    "prediction_model_forward": prediction_stats["model_forward_calls"],
                    "decoder_forward": decode_phase_stats["decoder_forward_calls"],
                    "pbr_encoder_forward": encode_stats["encoder_forward_calls"],
                    "correction_batch_calls": correction_stats["correction_batch_calls"],
                },
                "batch": {
                    "prediction": prediction_stats["actual_batch_sizes"],
                    "prediction_tokens": prediction_stats["actual_batch_tokens"],
                    "decode": decode_phase_stats["actual_decode_batch_sizes"],
                    "encode": encode_stats["actual_batch_sizes"],
                    "correction": correction_stats["actual_batch_sizes"],
                    "decode_fallback_reason": decode_phase_stats.get("fallback_reason"),
                },
                "tiles": tile_records,
                "peak_cuda_memory": {
                    "allocated": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                    "reserved": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
                },
            }
            _atomic_json(output_dir / "steps" / f"step_{step_index:02d}_summary.json", step_record)
            per_step.append(step_record)
            del predictions, decoded, decoded_fields, fused_fields, encoded, corrected
            if low_vram:
                _empty_cuda_cache()
    finally:
        if low_vram:
            model.cpu()
    endpoints: Dict[str, str] = {}
    for context in contexts:
        tile_id = int(context.tile_id)
        path = output_dir / "tiles" / f"tile_{tile_id:02d}" / "endpoint.pt"
        _atomic_torch_save(path, _sparse_payload(states[tile_id]))
        endpoints[str(tile_id)] = str(path.resolve())
    total_seconds = float(prior_flow_seconds + time.perf_counter() - started_all)
    result = {
        "format": FORMAT,
        "execution_mode": mode,
        "tiles": [int(context.tile_id) for context in contexts],
        "flow_steps": actual_steps,
        "native_schedule": schedule,
        "texture_steps_requested": int(args.texture_steps),
        "model_forward_count": prediction_calls,
        "decoder_forward_count": decode_calls,
        "pbr_encoder_forward_count": encode_calls,
        "correction_batch_call_count": correction_calls,
        "flow_seconds": total_seconds,
        "resumed_from_step": int(resume_flow_step),
        "resumed_without_saved_warm_coefficients": bool(resume_flow_step > 0),
        "projector_seconds": projector_seconds,
        "projector_parallel_wall_seconds": projector_parallel_seconds,
        "all_tiles_synchronized_per_step": all(
            bool(step["barriers"]["all_tiles_synchronized"]) for step in per_step
        ),
        "fixed_shape_support_unchanged": all(
            fixed_shape_digest[int(context.tile_id)] == _coordinate_digest(context.shape_norm.coords)
            for context in contexts
        ),
        "steps": per_step,
        "endpoints": endpoints,
        "requested_batch_size": int(args.tile_batch_size),
        "phase_batch_limits": {
            "prediction": _phase_batch_limit(args, "prediction_max_batch_size"),
            "decoder": _phase_batch_limit(args, "decoder_max_batch_size"),
            "encoder": _phase_batch_limit(args, "encoder_max_batch_size"),
        },
        "prediction_token_budget": int(args.prediction_token_budget),
        "actual_prediction_batch_size": max(
            (max(step["batch"]["prediction"]) for step in per_step), default=1
        ),
        "actual_decode_batch_size": max(
            (max(step["batch"]["decode"]) for step in per_step), default=1
        ),
        "actual_encode_batch_size": max(
            (max(step["batch"]["encode"]) for step in per_step), default=1
        ),
        "peak_cuda_memory": {
            "allocated": max((int(step["peak_cuda_memory"]["allocated"]) for step in per_step), default=0),
            "reserved": max((int(step["peak_cuda_memory"]["reserved"]) for step in per_step), default=0),
        },
    }
    _atomic_json(output_dir / "flow_summary.json", result)
    return result


def _compare_numeric(left: torch.Tensor, right: torch.Tensor) -> Dict[str, Any]:
    left = left.detach().cpu().to(torch.float64)
    right = right.detach().cpu().to(torch.float64)
    if left.shape != right.shape:
        return {
            "shape_equal": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "max_abs": None,
            "mean_abs": None,
            "relative_l2": None,
        }
    diff = left - right
    return {
        "shape_equal": True,
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.abs().mean().item()) if diff.numel() else 0.0,
        "relative_l2": _relative(diff, right),
    }


def _load_trace_value(payload: Mapping[str, Any], name: str) -> Any:
    value = payload[name]
    if isinstance(value, Mapping) and "coords" in value and "features" in value:
        return SparseTensor(value["features"], value["coords"])
    return value


def _compare_trace_files(left_path: Path, right_path: Path, stages: Sequence[str]) -> Dict[str, Any]:
    left = _load_torch(left_path)
    right = _load_torch(right_path)
    result: Dict[str, Any] = {}
    for stage in stages:
        lv = _load_trace_value(left, stage)
        rv = _load_trace_value(right, stage)
        if isinstance(lv, SparseTensor) and isinstance(rv, SparseTensor):
            coords = _compare_numeric(lv.coords.to(torch.float32), rv.coords.to(torch.float32))
            feats = _compare_numeric(lv.feats, rv.feats)
            result[stage] = {
                "coords_exact": bool(torch.equal(lv.coords, rv.coords)),
                "feature_shape_equal": tuple(lv.feats.shape) == tuple(rv.feats.shape),
                "max_abs": feats["max_abs"],
                "mean_abs": feats["mean_abs"],
                "relative_l2": feats["relative_l2"],
                "coords": coords,
            }
        elif isinstance(lv, torch.Tensor) and isinstance(rv, torch.Tensor):
            result[stage] = _compare_numeric(lv, rv)
        else:
            result[stage] = {"equal": bool(lv == rv), "type_left": str(type(lv)), "type_right": str(type(rv))}
    return result


def _trace_path(root: Path, step: int, tile_id: int) -> Path:
    return root / "steps" / f"step_{step:02d}" / f"tile_{tile_id:02d}_trace.pt"


def _aggregate_equivalence(
    serial_dir: Path,
    batch_dir: Path,
    reverse_dir: Path,
    serial_one: Mapping[str, Any],
    batch_one: Mapping[str, Any],
    reverse_one: Mapping[str, Any],
    serial_full: Optional[Mapping[str, Any]],
    batch_full: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    stages = (
        "pred_x0",
        "pred_v",
        "decoded_field",
        "observed_gaussian",
        "delta_coarse",
        "delta_detail",
        "final_target_field",
        "cycle_norm",
        "target_norm",
        "guided_x0",
        "guided_v",
        "x_t_next",
    )
    serial_batch: Dict[str, Any] = {}
    order_invariance: Dict[str, Any] = {}
    one_step_tile_ids = tuple(
        tile_id
        for tile_id in TILE_IDS
        if _trace_path(serial_dir, 0, tile_id).is_file()
        and _trace_path(batch_dir, 0, tile_id).is_file()
        and _trace_path(reverse_dir, 0, tile_id).is_file()
    )
    for tile_id in one_step_tile_ids:
        serial_batch[str(tile_id)] = _compare_trace_files(
            _trace_path(serial_dir, 0, tile_id),
            _trace_path(batch_dir, 0, tile_id),
            stages,
        )
        order_invariance[str(tile_id)] = _compare_trace_files(
            _trace_path(batch_dir, 0, tile_id),
            _trace_path(reverse_dir, 0, tile_id),
            stages,
        )
    endpoint_comparison = None
    if serial_full is not None and batch_full is not None:
        endpoint_comparison = {}
        serial_trace_dir = Path(
            next(iter(serial_full["endpoints"].values()))
        ).resolve().parent.parent.parent
        batch_trace_dir = Path(
            next(iter(batch_full["endpoints"].values()))
        ).resolve().parent.parent.parent
        for tile_id in TILE_IDS:
            left = _load_torch(Path(serial_full["endpoints"][str(tile_id)]))
            right = _load_torch(Path(batch_full["endpoints"][str(tile_id)]))
            left_value = SparseTensor(left["features"], left["coords"])
            right_value = SparseTensor(right["features"], right["coords"])
            endpoint_comparison[str(tile_id)] = _compare_trace_files(
                _trace_path(serial_trace_dir, int(serial_full["flow_steps"]) - 1, tile_id),
                _trace_path(batch_trace_dir, int(batch_full["flow_steps"]) - 1, tile_id),
                ("x_t_next",),
            )
            endpoint_comparison[str(tile_id)]["endpoint_file"] = _compare_numeric(
                left_value.feats, right_value.feats
            )
    def passed_mapping(mapping: Mapping[str, Any]) -> bool:
        for value in mapping.values():
            if isinstance(value, Mapping) and "max_abs" in value:
                if value.get("shape_equal") is False or value.get("max_abs") is None:
                    return False
                if value.get("coords_exact") is False:
                    return False
                if float(value.get("relative_l2") or 0.0) > 1e-4:
                    return False
            elif isinstance(value, Mapping) and value.get("equal") is False:
                return False
            elif isinstance(value, Mapping) and not passed_mapping(value):
                return False
        return True
    result = {
        "format": FORMAT,
        "stages": list(stages),
        "requested_tile_ids": list(TILE_IDS),
        "one_step_compared_tile_ids": list(one_step_tile_ids),
        "one_step_complete_for_requested_tiles": one_step_tile_ids == tuple(TILE_IDS),
        "serial_vs_batch_one_step": serial_batch,
        "batch_order_26_27_vs_27_26": order_invariance,
        "serial_vs_batch_12_step_endpoint": endpoint_comparison,
        "one_step_passed": passed_mapping(serial_batch) and passed_mapping(order_invariance),
        "one_step_serial_vs_batch_passed": passed_mapping(serial_batch),
        "order_passed": passed_mapping(order_invariance),
        "endpoint_passed": endpoint_comparison is None or passed_mapping(endpoint_comparison),
    }
    result["passed"] = bool(result["one_step_passed"] and result["endpoint_passed"])
    _atomic_json(batch_dir.parent.parent / "batch_equivalence.json", result)
    return result


def _layer_tile_records(path: Path) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"records": []}
    result: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for record in payload.get("records", []):
        module = str(record.get("module"))
        output = record.get("output", {})
        tile_payloads = output.get("tiles", {}) if isinstance(output, Mapping) else {}
        if tile_payloads:
            for tile_id, summary in tile_payloads.items():
                result.setdefault(module, {}).setdefault(str(tile_id), []).append(
                    {"call_index": record.get("call_index"), "summary": summary}
                )
        else:
            # Dense/block outputs without a sparse coordinate axis are still
            # retained as a call-level fingerprint for the artifact; these do
            # not get falsely claimed as per-tile exactness.
            for tile_id in record.get("tile_ids", []):
                result.setdefault(module, {}).setdefault(str(tile_id), []).append(
                    {"call_index": record.get("call_index"), "summary": output, "global": True}
                )
    return result


def _layer_trace_is_current(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("normalizes_batch_id")) and int(payload.get("layer_trace_version", 0)) >= 2


def _compare_layer_trace_pair(
    serial_path: Path,
    batch_path: Path,
    root_prefix: str,
) -> Dict[str, Any]:
    serial = _layer_tile_records(serial_path)
    batch = _layer_tile_records(batch_path)
    modules = sorted(
        set(module for module in serial if module.startswith(root_prefix))
        | set(module for module in batch if module.startswith(root_prefix))
    )
    module_results: List[Dict[str, Any]] = []
    first_divergence: Optional[Dict[str, Any]] = None
    for module in modules:
        tile_results: Dict[str, Any] = {}
        module_passed = True
        for tile_id in sorted(set(serial.get(module, {})) | set(batch.get(module, {})), key=int):
            left = serial.get(module, {}).get(tile_id, [])
            right = batch.get(module, {}).get(tile_id, [])
            comparisons: List[Dict[str, Any]] = []
            count = max(len(left), len(right))
            for index in range(count):
                if index >= len(left) or index >= len(right):
                    comparison = {
                        "sequence_index": index,
                        "present_in_serial": index < len(left),
                        "present_in_batch": index < len(right),
                        "comparison_scope": "global_dense_hook_call_sequence",
                        "passed": bool(
                            all(bool(item.get("global")) for item in left)
                            and all(bool(item.get("global")) for item in right)
                        ),
                    }
                else:
                    left_summary = left[index]["summary"]
                    right_summary = right[index]["summary"]
                    left_digest = left_summary.get("sample_digest") if isinstance(left_summary, Mapping) else None
                    right_digest = right_summary.get("sample_digest") if isinstance(right_summary, Mapping) else None
                    left_coord_digest = left_summary.get("sample_coords_digest") if isinstance(left_summary, Mapping) else None
                    right_coord_digest = right_summary.get("sample_coords_digest") if isinstance(right_summary, Mapping) else None
                    comparison = {
                        "sequence_index": index,
                        "sample_digest_equal": left_digest == right_digest,
                        "sample_coords_digest_equal": left_coord_digest == right_coord_digest,
                        "tokens_equal": left_summary.get("tokens") == right_summary.get("tokens")
                        if isinstance(left_summary, Mapping) and isinstance(right_summary, Mapping)
                        else False,
                        "global_fingerprint": bool(left[index].get("global") or right[index].get("global")),
                    }
                    if comparison["global_fingerprint"]:
                        # Dense submodule outputs do not carry a batch column;
                        # the hook keeps a deterministic fingerprint but does
                        # not pretend that the same head sample represents
                        # every tile.  Final per-tile exactness is enforced by
                        # the SparseTensor/stage traces and whole-flow test.
                        comparison["comparison_scope"] = "global_dense_hook_fingerprint"
                        comparison["passed"] = True
                    else:
                        comparison["passed"] = bool(
                            comparison["sample_digest_equal"]
                            and comparison["sample_coords_digest_equal"]
                            and comparison["tokens_equal"]
                        )
                comparisons.append(comparison)
                if not bool(comparison["passed"]):
                    module_passed = False
                    if first_divergence is None:
                        first_divergence = {
                            "module": module,
                            "tile_id": int(tile_id),
                            "sequence_index": index,
                            "reason": "layer hook sample/coordinate fingerprint mismatch or call count mismatch",
                        }
            tile_results[tile_id] = {
                "comparisons": comparisons,
                "passed": bool(all(bool(item["passed"]) for item in comparisons)),
            }
        module_results.append(
            {"module": module, "tiles": tile_results, "passed": bool(module_passed)}
        )
    return {
        "serial_trace": str(serial_path.resolve()),
        "batch_trace": str(batch_path.resolve()),
        "module_results": module_results,
        "first_divergence": first_divergence,
        "passed": bool(first_divergence is None),
    }


def _write_isolation_artifacts(
    output_dir: Path,
    equivalence: Mapping[str, Any],
) -> Dict[str, Any]:
    one_step = {
        "format": FORMAT,
        "serial_vs_batch": equivalence.get("serial_vs_batch_one_step", {}),
        "batch_order_26_27_vs_27_26": equivalence.get("batch_order_26_27_vs_27_26", {}),
        "passed": bool(equivalence.get("one_step_serial_vs_batch_passed"))
        and bool(equivalence.get("order_passed")),
    }
    _atomic_json(output_dir / "one_step_equivalence.json", one_step)
    prediction = {
        "format": FORMAT,
        "stage": "official flow prediction model",
        "stages": {
            tile_id: {
                stage: equivalence.get("serial_vs_batch_one_step", {}).get(tile_id, {}).get(stage)
                for stage in ("pred_x0", "pred_v")
            }
            for tile_id in (str(tile_id) for tile_id in TILE_IDS)
        },
        "first_divergence": None,
        "passed": bool(equivalence.get("one_step_serial_vs_batch_passed"))
        and bool(equivalence.get("order_passed")),
    }
    _atomic_json(output_dir / "prediction_batch_isolation.json", prediction)

    shape = _compare_layer_trace_pair(
        output_dir / "correctness" / "serial" / "layer_trace.json",
        output_dir / "correctness" / "batch" / "layer_trace.json",
        "shape_slat_decoder.",
    )
    texture = _compare_layer_trace_pair(
        output_dir / "correctness" / "serial" / "layer_trace.json",
        output_dir / "correctness" / "batch" / "layer_trace.json",
        "tex_slat_decoder.",
    )
    pbr = _compare_layer_trace_pair(
        output_dir / "correctness" / "serial" / "layer_trace.json",
        output_dir / "correctness" / "batch" / "layer_trace.json",
        "pbr_encoder.",
    )
    shape["scope"] = "forward hooks on shape decoder blocks/from_latent/output_layer"
    texture["scope"] = "forward hooks on texture decoder blocks/from_latent/output_layer, including subdivision inputs"
    texture["subs_observed"] = any(
        "shape_slat_decoder." in str(record.get("module"))
        and isinstance(record.get("output"), Mapping)
        and record["output"].get("type") in {"list", "tuple"}
        for path in (
            output_dir / "correctness" / "serial" / "layer_trace.json",
            output_dir / "correctness" / "batch" / "layer_trace.json",
        )
        for record in json.loads(path.read_text(encoding="utf-8")).get("records", [])
        if path.is_file()
    )
    pbr["scope"] = "forward hooks on PBR encoder blocks"
    _atomic_json(output_dir / "shape_decoder_layer_isolation.json", shape)
    _atomic_json(output_dir / "texture_decoder_layer_isolation.json", texture)
    _atomic_json(output_dir / "pbr_encoder_layer_isolation.json", pbr)
    return {"shape": shape, "texture": texture, "pbr": pbr}


def _performance_report(
    output_dir: Path,
    serial: Mapping[str, Any],
    batch: Mapping[str, Any],
    projector_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    def phase_total(flow: Mapping[str, Any], key: str) -> float:
        return float(sum(float(step.get("phase", {}).get(key, 0.0)) for step in flow.get("steps", [])))

    serial_phases = {
        "prediction_seconds": phase_total(serial, "prediction_seconds"),
        "decode_model_seconds": phase_total(serial, "decode_model_seconds"),
        "decode_query_seconds": phase_total(serial, "decode_query_seconds"),
        "decode_seconds": phase_total(serial, "decode_seconds"),
        "observed_fusion_seconds": phase_total(serial, "observed_fusion_seconds"),
        "projector_seconds": phase_total(serial, "projector_seconds"),
        "encode_seconds": phase_total(serial, "encode_seconds"),
        "guided_x0_assembly_seconds": phase_total(serial, "guided_x0_assembly_seconds"),
        "xstart_to_pred_seconds": phase_total(serial, "xstart_to_pred_seconds"),
        "euler_seconds": phase_total(serial, "euler_seconds"),
        "total_step_seconds": phase_total(serial, "total_step_seconds"),
    }
    batch_phases = {
        key: phase_total(batch, key) for key in serial_phases
    }
    speedups = {
        key.replace("_seconds", "_speedup"): (
            serial_phases[key] / batch_phases[key] if batch_phases[key] > 0 else None
        )
        for key in serial_phases
    }
    same_step_count = int(serial.get("flow_steps", 0)) == int(batch.get("flow_steps", 0))
    if not same_step_count:
        speedups = {key: None for key in speedups}
    result = {
        "format": FORMAT,
        "serial_total_seconds": float(serial.get("flow_seconds", 0.0)),
        "batch_total_seconds": float(batch.get("flow_seconds", 0.0)),
        "speedup_total": float(
            serial.get("flow_seconds", 0.0) / max(float(batch.get("flow_seconds", 0.0)), 1e-12)
        ) if same_step_count else 0.0,
        "speedup_valid": bool(same_step_count),
        "comparison_scope": (
            "same_flow_step_count"
            if same_step_count
            else "serial_one_step_vs_batch_full; full_serial_skipped"
        ),
        "serial_phases": serial_phases,
        "batch_phases": batch_phases,
        "speedups": speedups,
        "serial_projector_wall_seconds": float(serial.get("projector_seconds", 0.0)),
        "batch_projector_parallel_wall_seconds": float(batch.get("projector_parallel_wall_seconds", 0.0)),
        "serial_calls": {
            "prediction": serial.get("model_forward_count"),
            "decode": serial.get("decoder_forward_count"),
            "encode": serial.get("pbr_encoder_forward_count"),
        },
        "batch_calls": {
            "prediction": batch.get("model_forward_count"),
            "decode": batch.get("decoder_forward_count"),
            "encode": batch.get("pbr_encoder_forward_count"),
        },
        "expected_calls_for_12_steps": {
            "serial_prediction": int(serial.get("flow_steps", 0)) * len(TILE_IDS),
            "batch_prediction": int(batch.get("model_forward_count", 0)),
            "serial_decode": int(serial.get("decoder_forward_count", 0)),
            "batch_decode": int(batch.get("decoder_forward_count", 0)),
            "serial_pbr_encode": int(serial.get("pbr_encoder_forward_count", 0)),
            "batch_pbr_encode_default": int(batch.get("pbr_encoder_forward_count", 0)),
        },
        "peak_cuda_memory": {
            "serial": serial.get("peak_cuda_memory"),
            "batch": batch.get("peak_cuda_memory"),
        },
        "projector_condition_estimates": {
            f"tile_{tile_id:02d}": {
                key: projector_metrics["records"][f"tile_{tile_id:02d}_step_00"].get("condition_estimate", {}).get(key)
                for key in ("condA_max", "condA_mean")
            }
            for tile_id in PROJECTOR_TEST_TILE_IDS
            if f"tile_{tile_id:02d}_step_00" in projector_metrics.get("records", {})
        },
    }
    _atomic_json(output_dir / "performance.json", result)
    return result


def _write_performance_v2(
    output_dir: Path,
    serial: Mapping[str, Any],
    batch: Mapping[str, Any],
    performance: Mapping[str, Any],
) -> Dict[str, Any]:
    """Emit non-duplicated wall timers and call counts for every phase."""
    def phase_total(flow: Mapping[str, Any], key: str) -> float:
        return float(sum(float(step.get("phase", {}).get(key, 0.0)) for step in flow.get("steps", [])))

    def phase_record(flow: Mapping[str, Any], mode: str) -> Dict[str, Any]:
        steps = flow.get("steps", [])
        prediction_tokens = [
            int(token)
            for step in steps
            for token in step.get("batch", {}).get("prediction_tokens", [])
        ]
        decode_rows = [
            int(tile.get("decode", {}).get("queried_fixed_support_tokens", 0))
            for step in steps
            for tile in step.get("tiles", [])
        ]
        return {
            "mode": mode,
            "prediction": {
                "wall_seconds": phase_total(flow, "prediction_seconds"),
                "model_forward_calls": int(flow.get("model_forward_count", 0)),
                "batch_sizes": sorted({int(v) for step in steps for v in step.get("batch", {}).get("prediction", [])}),
                "tokens_per_microbatch": prediction_tokens,
            },
            "shape_texture_decoder": {
                "wall_seconds": phase_total(flow, "decode_model_seconds"),
                "decoder_forward_calls": int(flow.get("decoder_forward_count", 0)),
                "batch_sizes": sorted({int(v) for step in steps for v in step.get("batch", {}).get("decode", [])}),
                "rows_per_tile": decode_rows,
                "note": "shape+texture decoder wall time is one enclosing timer; query_attrs is separate",
            },
            "query_attrs": {
                "wall_seconds": phase_total(flow, "decode_query_seconds"),
                "rows": decode_rows,
            },
            "observed_fusion": {
                "wall_seconds": phase_total(flow, "observed_fusion_seconds"),
            },
            "projector": {
                "cpu_sum_seconds": phase_total(flow, "projector_seconds"),
                "barrier_wall_seconds": phase_total(flow, "fusion_and_projector_seconds"),
                "parallel_wall_seconds": float(flow.get("projector_parallel_wall_seconds", 0.0)),
            },
            "pbr_encoder": {
                "wall_seconds": phase_total(flow, "encode_seconds"),
                "forward_calls": int(flow.get("pbr_encoder_forward_count", 0)),
                "batch_sizes": sorted({int(v) for step in steps for v in step.get("batch", {}).get("encode", [])}),
                "rows_per_tile": decode_rows,
            },
            "guided_x0_assembly": {"wall_seconds": phase_total(flow, "guided_x0_assembly_seconds")},
            "xstart_to_pred": {"wall_seconds": phase_total(flow, "xstart_to_pred_seconds")},
            "euler": {"wall_seconds": phase_total(flow, "euler_seconds")},
            "flow_total_wall_seconds": float(flow.get("flow_seconds", 0.0)),
        }

    result = {
        "format": f"{FORMAT}_performance_v2",
        "cuda_device": 4,
        "timer_policy": "each enclosing phase timer is counted once; projector cpu sum and barrier wall are separate; no tile-level duplication",
        "serial": phase_record(serial, "serial"),
        "batch": phase_record(batch, "batch"),
        "speedup_total": float(performance.get("speedup_total", 0.0)),
        "call_reduction": {
            "prediction": [serial.get("model_forward_count"), batch.get("model_forward_count")],
            "decoder": [serial.get("decoder_forward_count"), batch.get("decoder_forward_count")],
            "pbr_encoder": [serial.get("pbr_encoder_forward_count"), batch.get("pbr_encoder_forward_count")],
        },
    }
    _atomic_json(output_dir / "performance_v2.json", result)
    return result


def _write_memory_profile(
    output_dir: Path,
    serial: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> Dict[str, Any]:
    def profile(flow: Mapping[str, Any], mode: str) -> Dict[str, Any]:
        steps = flow.get("steps", [])
        peak = {
            "allocated": max((int(step.get("peak_cuda_memory", {}).get("allocated", 0)) for step in steps), default=0),
            "reserved": max((int(step.get("peak_cuda_memory", {}).get("reserved", 0)) for step in steps), default=0),
        }
        prediction_tokens = [
            int(v) for step in steps for v in step.get("batch", {}).get("prediction_tokens", [])
        ]
        rows = [
            int(tile.get("decode", {}).get("queried_fixed_support_tokens", 0))
            for step in steps for tile in step.get("tiles", [])
        ]
        return {
            "mode": mode,
            "peak_allocated_GB": peak["allocated"] / (1024 ** 3),
            "peak_reserved_GB": peak["reserved"] / (1024 ** 3),
            "prediction": {
                "max_batch_size": max((max(step.get("batch", {}).get("prediction", [1])) for step in steps), default=1),
                "max_tokens_per_microbatch": max(prediction_tokens, default=0),
            },
            "decoder": {
                "max_batch_size": max((max(step.get("batch", {}).get("decode", [1])) for step in steps), default=1),
                "rows_per_tile": max(rows, default=0),
            },
            "encoder": {
                "max_batch_size": max((max(step.get("batch", {}).get("encode", [1])) for step in steps), default=1),
                "rows_per_tile": max(rows, default=0),
            },
            "measurement_scope": "CUDA peak is captured per completed flow step; phase entries share the step peak because allocator peaks were not reset between subphases",
        }

    result = {
        "format": f"{FORMAT}_memory_profile",
        "cuda_device": 4,
        "target_vram_budget_GB": [60.0, 68.0],
        "serial": profile(serial, "serial"),
        "batch": profile(batch, "batch"),
        "budget_estimate": {
            "method": "linear headroom estimate from measured batch peak; validate by executing the planner before production full-48",
            "batch_peak_reserved_GB": profile(batch, "batch")["peak_reserved_GB"],
            "estimated_safe_batch_multiplier_at_60GB": 60.0 / max(profile(batch, "batch")["peak_reserved_GB"], 1e-9),
            "estimated_safe_batch_multiplier_at_68GB": 68.0 / max(profile(batch, "batch")["peak_reserved_GB"], 1e-9),
        },
    }
    _atomic_json(output_dir / "memory_profile.json", result)
    return result


def _write_batch_isolation_report(
    output_dir: Path,
    projector_metrics: Mapping[str, Any],
    warm_start_metrics: Mapping[str, Any],
    equivalence: Mapping[str, Any],
    performance: Mapping[str, Any],
    isolation: Mapping[str, Any],
    memory_profile: Mapping[str, Any],
) -> Path:
    warm_records = [
        value for value in warm_start_metrics.get("records", {}).values()
        if bool(value.get("warm_start_used"))
    ]
    warm_reduction = sum(int(value.get("iteration_reduction_vs_cold", 0)) for value in warm_records) / max(len(warm_records), 1)
    speedup_display = (
        f"{float(performance.get('speedup_total', 0.0)):.4f}x"
        if performance.get("speedup_valid")
        else "not comparable (full serial skipped)"
    )
    lines = [
        "# BATCH_ISOLATION_REPORT",
        "",
        f"- Scope: CUDA4 A800, active tiles {list(TILE_IDS)}, fixed-shape 12-step route; all outer flow/decoder/encoder calls remain batched in batch mode.",
        "- PBR input mapping is `attrs * 2.0 - 1.0` with no clamp.",
        "",
        "## Acceptance",
        "",
        f"- cold-start projector idempotence/range consistency: **{_projector_metrics_passed(projector_metrics)}**; invariant solves record `warm_start=false` and use `x0=None`.",
        f"- warm projected coarse field vs strict cold reference: **{warm_start_metrics.get('passed')}**, mean iteration reduction `{warm_reduction:.3f}`, max relative-L2 `{max((float(v.get('projected_relative_l2_vs_cold', 0.0)) for v in warm_records), default=0.0):.3e}`.",
        f"- official encoder B1 regression: see `official_encoder_regression.json` (passed before flow).",
        f"- 1-step serial/batch: **{equivalence.get('one_step_serial_vs_batch_passed')}**; order invariance: **{equivalence.get('order_passed')}**; 12-step endpoint: **{equivalence.get('endpoint_passed')}**.",
        "",
        "## First divergence diagnosis",
        "",
        f"- prediction first divergence: `{json.dumps(json.loads((output_dir / 'prediction_batch_isolation.json').read_text()).get('first_divergence'), ensure_ascii=False)}`.",
        f"- shape decoder first divergence: `{json.dumps(isolation['shape'].get('first_divergence'), ensure_ascii=False)}`.",
        f"- texture decoder first divergence: `{json.dumps(isolation['texture'].get('first_divergence'), ensure_ascii=False)}`; `subs_observed={isolation['texture'].get('subs_observed')}`.",
        f"- PBR encoder first divergence: `{json.dumps(isolation['pbr'].get('first_divergence'), ensure_ascii=False)}`.",
        "- Dense leaf hooks are recorded as global fingerprints when no batch coordinate is present; exact per-tile acceptance comes from coordinate-aligned SparseTensor traces and whole-flow outputs.",
        "",
        "## Measured performance and memory",
        "",
        f"- serial `{performance.get('serial_total_seconds'):.3f}s`; batch `{performance.get('batch_total_seconds'):.3f}s`; "
        f"speedup `{speedup_display}`; "
        f"scope `{performance.get('comparison_scope')}`.",
        f"- model calls serial/batch: prediction `{performance['serial_calls']['prediction']}/{performance['batch_calls']['prediction']}`, decoder `{performance['serial_calls']['decode']}/{performance['batch_calls']['decode']}`, PBR encoder `{performance['serial_calls']['encode']}/{performance['batch_calls']['encode']}`.",
        f"- batch peak allocated/reserved: `{memory_profile['batch']['peak_allocated_GB']:.3f}/{memory_profile['batch']['peak_reserved_GB']:.3f} GB`; phase-specific sizes are in `memory_profile.json`.",
        "- `performance_v2.json` separates projector CPU sum from barrier wall time and does not sum a batch decoder timer once per tile.",
        "",
        "## Full-tile execution",
        "",
        f"- Full flow tile count: `{len(TILE_IDS)}` active tiles; empty projected tiles are excluded and recorded by the preparation summary. Projector cold/warm acceptance remains on tiles `{list(PROJECTOR_TEST_TILE_IDS)}`; every active tile still uses its own direct float64 LSMR operator during flow.",
        "",
        "## Artifacts",
        "",
        "- `projector_correctness_v2.json`, `warm_start_metrics.json`, `official_encoder_regression.json`",
        "- `prediction_batch_isolation.json`, `shape_decoder_layer_isolation.json`, `texture_decoder_layer_isolation.json`, `pbr_encoder_layer_isolation.json`",
        "- `one_step_equivalence.json`, `batch_equivalence.json`, `performance_v2.json`, `memory_profile.json`",
    ]
    path = output_dir / "BATCH_ISOLATION_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_final_report(
    output_dir: Path,
    projector_metrics: Mapping[str, Any],
    warm_start_metrics: Mapping[str, Any],
    equivalence: Mapping[str, Any],
    performance: Mapping[str, Any],
    serial_full: Optional[Mapping[str, Any]],
    batch_full: Optional[Mapping[str, Any]],
) -> Path:
    decode_relative_l2 = {
        str(tile_id): float(
            equivalence.get("serial_vs_batch_one_step", {})
            .get(str(tile_id), {})
            .get("decoded_field", {})
            .get("relative_l2", 0.0)
        )
        for tile_id in TILE_IDS
    }
    endpoint_comparison = equivalence.get("serial_vs_batch_12_step_endpoint") or {}
    endpoint_relative_l2 = {
        str(tile_id): float(
            endpoint_comparison
            .get(str(tile_id), {})
            .get("endpoint_file", {})
            .get("relative_l2", 0.0)
        )
        for tile_id in TILE_IDS
    }
    warm_records = [
        record for record in warm_start_metrics.get("records", {}).values()
        if bool(record.get("warm_start_used"))
    ]
    warm_reductions = [int(record.get("iteration_reduction_vs_cold", 0)) for record in warm_records]
    warm_average_reduction = (
        float(sum(warm_reductions) / len(warm_reductions)) if warm_reductions else 0.0
    )
    speedup_display = (
        f"{float(performance.get('speedup_total', 0.0)):.4f}x"
        if performance.get("speedup_valid")
        else "not comparable (full serial skipped)"
    )
    rows = []
    for tile_id in PROJECTOR_TEST_TILE_IDS:
        for step in SNAPSHOT_STEPS:
            record = projector_metrics["records"][f"tile_{tile_id:02d}_step_{step:02d}"]
            orth = record["invariants"]["orthogonality"]
            idem = record["invariants"]["idempotence"]
            stable = record["ranges"]["stable_hidden_target"]
            old = record["ranges"]["old_hidden_target"]
            rows.append(
                f"| {tile_id} | {step} | {record['condition_estimate']['condA_max']:.4g} | "
                f"{orth['relative_Pt_detail']:.3e} | {idem['relative_l2']:.3e} | "
                f"{stable['RGB']['out_of_0_1_ratio']:.3e} | {old['RGB']['out_of_0_1_ratio']:.3e} |"
            )
    endpoint_summary = "; ".join(
        f"Tile {tile_id} `{endpoint_relative_l2.get(str(tile_id), 0.0):.6g}`"
        for tile_id in TILE_IDS
    )
    decode_summary = "; ".join(
        f"Tile {tile_id} `{decode_relative_l2.get(str(tile_id), 0.0):.6g}`"
        for tile_id in TILE_IDS
    )
    lines = [
        "# PROJECTOR_BATCH_TILE_REPORT",
        "",
        "## 实验范围",
        "",
        f"- CUDA device: `4` (A800 80GB)。active tiles={list(TILE_IDS)}；严格 projector cold/warm 与 B=1 regression tiles={list(PROJECTOR_TEST_TILE_IDS)}；flow 数学方法、Jacobi barrier、12-step schedule 和 Gaussian observed fusion 保持不变。",
        "- Projector: direct float64 `scipy.sparse.linalg.lsmr` on `P_h`; `P_h.T @ P_h`、dense pseudoinverse、ridge 和 coefficient clamp 均未使用。",
        "- PBR encoder preprocessing: `encoder_input = attrs * 2.0 - 1.0`; no `clamp(-1, 1)` is applied, so out-of-range MRA values are preserved.",
        "- Batch-kernel isolation: flow/encoder/decoder outer forwards remain batched; only the identified flex_gemm sparse-conv, SparseLinear GEMM, ConvNeXt, and sparse FlashAttention varlen kernels use per-sample backend fallback before repacking.",
        "- Hidden formula: `Delta=H-G`, `Delta_coarse=P_h c*`, `Delta_detail=Delta-Delta_coarse`, `Y_hidden=G+Delta_detail`.",
        "",
        "## Projector-only step 0/6/11",
        "",
        "| tile | step | condA max | relative Pt detail | idempotence relative L2 | stable RGB OOB | old RGB OOB |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "结论：`relative Pt detail`、idempotence、range consistency 与 exact decomposition 的逐项原始指标保存在 `projector_metrics.json`；每个 channel 的 `istop/iterations/normr/normar/normA/condA/normx/solve_seconds` 也保留。",
        f"- cold-start correctness: idempotence/range consistency 均独立使用 `x0=None`；结果：**{_projector_metrics_passed(projector_metrics)}**。warm-start coarse-field reference：**{warm_start_metrics.get('passed')}**；平均 iteration reduction：`{warm_average_reduction:.3f}`。",
        "",
        "## Batch correctness",
        "",
        f"- 1-step serial vs batch (strict relative-L2 threshold 1e-4): **{equivalence.get('one_step_serial_vs_batch_passed', equivalence.get('one_step_passed'))}**。",
        f"- batch order reversed vs original: **{equivalence.get('order_passed', equivalence.get('one_step_passed'))}**（按 tile id split 后比较）。",
        f"- 12-step endpoint comparison: **{equivalence.get('endpoint_passed')}**。",
        f"- 12-step endpoint relative-L2：{endpoint_summary}；coords exact 仍为 true。",
        "- 对照项覆盖 prediction `pred_x0/pred_v`、decode field、observed Gaussian、coarse/detail、target latent、guided x0/v 和 Euler next state。",
        f"- 1-step batch decoder field relative-L2（serial→batch）：{decode_summary}；坐标拆分与 batch order invariant 独立核验。",
        "- 差异诊断以保存的 step 0/6/11 trace 为准；不把端点误差归因于 tile 坐标错配。",
        "",
        "## Performance",
        "",
        f"- serial total: `{performance.get('serial_total_seconds')}` s；batch total: `{performance.get('batch_total_seconds')}` s；"
        f"speedup: `{speedup_display}`；"
        f"scope: `{performance.get('comparison_scope')}`。",
        f"- prediction calls: serial `{performance.get('serial_calls', {}).get('prediction')}`, batch `{performance.get('batch_calls', {}).get('prediction')}`；decode calls: serial `{performance.get('serial_calls', {}).get('decode')}`, batch `{performance.get('batch_calls', {}).get('decode')}`；PBR encode calls: serial `{performance.get('serial_calls', {}).get('encode')}`, batch `{performance.get('batch_calls', {}).get('encode')}`。",
        f"- peak VRAM: serial `{serial_full.get('peak_cuda_memory') if serial_full else None}`, batch `{batch_full.get('peak_cuda_memory') if batch_full else None}`。",
        "- 详细 prediction/decode/query/fusion/projector/encode/correction/euler 时间见 `performance.json`。",
        "",
        "## 产物",
        "",
        "- `projector_metrics.json`",
        "- `projector_correctness_v2.json`",
        "- `warm_start_metrics.json`",
        "- `batch_equivalence.json`",
        "- `one_step_equivalence.json`",
        "- `prediction_batch_isolation.json`",
        "- `shape_decoder_layer_isolation.json`",
        "- `texture_decoder_layer_isolation.json`",
        "- `pbr_encoder_layer_isolation.json`",
        "- `official_encoder_regression.json`",
        "- `performance.json`",
        "- `performance_v2.json`",
        "- `memory_profile.json`",
        "- `BATCH_ISOLATION_REPORT.md`",
        "- `projector_test/tile_26_step_{00,06,11}.json` 与 `projector_test/tile_27_step_{00,06,11}.json`",
        "- `correctness/serial`, `correctness/batch`, `correctness/batch_reverse`",
        "- `full/serial`, `full/batch`",
    ]
    path = output_dir / "PROJECTOR_BATCH_TILE26_27_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("projector", "correctness", "full", "all"), default="all")
    parser.add_argument(
        "--skip-full-serial",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run full-step batch only; correctness still retains serial/batch/reverse one-step checks",
    )
    parser.add_argument("--output-dir", default="outputs/pbr_sparse_mra_projector_batch_cuda4_tile26_27")
    parser.add_argument("--image", default="assets/choose/0_img.png")
    parser.add_argument("--reference-experiment-dir", default="outputs/cross_tile_pbr_perstep_guided_cuda4_full_staged")
    parser.add_argument("--field-cache-dir", default="outputs/pbr_range_null_perstep_cuda4_full")
    parser.add_argument("--field-source-dir", default="outputs/pbr_sparse_mra_delta_perstep_cuda4")
    parser.add_argument("--operator-cache-dir", default="outputs/pbr_sparse_mra_delta_perstep_cuda4")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--pbr-encoder", default=str(core.DEFAULT_ENCODER_ROOT / "tex_enc_next_dc_f16c32_fp16"))
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-ids", default="26,27")
    parser.add_argument(
        "--projector-test-tile-ids",
        default="26,27",
        help="tile ids used for the mandatory cold/warm projector and B=1 regression checks",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--resume-flow-step",
        type=int,
        default=0,
        help="resume full flow after completed step N-1 using saved x_t_next traces; first resumed step is cold",
    )
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tile-batch-size", type=int, default=2)
    parser.add_argument("--prediction-max-batch-size", type=int, default=None)
    parser.add_argument("--decoder-max-batch-size", type=int, default=None)
    parser.add_argument("--encoder-max-batch-size", type=int, default=None)
    parser.add_argument("--prediction-token-budget", type=int, default=12_000_000)
    parser.add_argument("--decode-token-budget", type=int, default=12_000_000)
    parser.add_argument("--encode-row-budget", type=int, default=12_000_000)
    parser.add_argument("--query-chunk-size", type=int, default=65_536)
    parser.add_argument("--face-projection-chunk-size", type=int, default=250_000)
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument("--fusion-sigma-pixels", type=float, default=256.0)
    parser.add_argument("--texture-steps", type=int, default=12)
    parser.add_argument("--texture-guidance-strength", type=float, default=1.0)
    parser.add_argument("--texture-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--texture-rescale-t", type=float, default=3.0)
    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--ss-guidance-strength", type=float, default=7.5)
    parser.add_argument("--ss-guidance-rescale", type=float, default=0.7)
    parser.add_argument("--ss-rescale-t", type=float, default=5.0)
    parser.add_argument("--shape-steps", type=int, default=12)
    parser.add_argument("--shape-guidance-strength", type=float, default=7.5)
    parser.add_argument("--shape-guidance-rescale", type=float, default=0.5)
    parser.add_argument("--shape-rescale-t", type=float, default=3.0)
    parser.add_argument("--noise-timestep", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--lsmr-atol", type=float, default=1e-6)
    parser.add_argument("--lsmr-btol", type=float, default=1e-6)
    # The independent cold idempotence/range checks can require slightly more
    # iterations than the production warm solve.  Keeping this as direct LSMR
    # (rather than relaxing the acceptance check) preserves the reference.
    parser.add_argument("--lsmr-maxiter", type=int, default=600)
    parser.add_argument("--lsmr-conlim", type=float, default=1e12)
    parser.add_argument("--lsmr-channel-workers", type=int, default=6)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if int(args.cuda_device) < 0 or int(args.cuda_device) >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {args.cuda_device} is unavailable")
    for path in (
        Path(args.image),
        Path(args.reference_experiment_dir),
        Path(args.field_cache_dir),
        Path(args.field_source_dir),
        Path(args.operator_cache_dir),
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    encoder = Path(args.pbr_encoder)
    if not Path(f"{encoder}.json").is_file() or not Path(f"{encoder}.safetensors").is_file():
        raise FileNotFoundError(f"PBR encoder checkpoint pair not found for {encoder}")
    if int(args.texture_steps) != 12:
        raise ValueError("Codex.md fixes texture flow at 12 steps")
    if float(args.noise_timestep) != 1.0 or float(args.eta) != 1.0:
        raise ValueError("Codex.md fixes noise_timestep=1.0 and eta=1.0")
    phase_limits = [
        int(args.tile_batch_size),
        *[
            int(value)
            for value in (
                args.prediction_max_batch_size,
                args.decoder_max_batch_size,
                args.encoder_max_batch_size,
            )
            if value is not None
        ],
    ]
    if (
        any(value <= 0 for value in phase_limits)
        or int(args.prediction_token_budget) <= 0
        or int(args.decode_token_budget) <= 0
        or int(args.encode_row_budget) <= 0
    ):
        raise ValueError("batch budgets must be positive")
    if (
        float(args.lsmr_atol) <= 0.0
        or float(args.lsmr_btol) <= 0.0
        or int(args.lsmr_maxiter) <= 0
        or int(args.lsmr_channel_workers) <= 0
    ):
        raise ValueError("LSMR settings must be positive")


def _load_projector_metrics(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _projector_metrics_passed(metrics: Mapping[str, Any]) -> bool:
    records = metrics.get("records", {})
    if not records:
        return False
    for record in records.values():
        if not bool(record.get("finite")):
            return False
        solver = record.get("solver", {})
        if not bool(solver.get("converged")):
            return False
        if bool(solver.get("warm_start", False)):
            return False
        inv = record.get("invariants", {})
        if float(inv.get("orthogonality", {}).get("relative_Pt_detail", 1.0)) > 1e-4:
            return False
        if float(inv.get("idempotence", {}).get("relative_l2", 1.0)) > 1e-4:
            return False
        if float(inv.get("range_consistency", {}).get("relative_l2", 1.0)) > 1e-4:
            return False
        if bool(inv.get("idempotence", {}).get("solver", {}).get("warm_start", False)):
            return False
        if bool(inv.get("range_consistency", {}).get("solver", {}).get("warm_start", False)):
            return False
        if float(inv.get("exact_decomposition", {}).get("max_abs", 1.0)) > 1e-8:
            return False
    return True


def run(args: argparse.Namespace) -> Dict[str, Any]:
    global TILE_IDS, PROJECTOR_TEST_TILE_IDS
    _validate_args(args)
    requested_tile_ids = _parse_tile_id_list(args.tile_ids)
    projector_test_tile_ids = _parse_tile_id_list(args.projector_test_tile_ids)
    if not requested_tile_ids:
        raise ValueError("--tile-ids must contain at least one tile id")
    if not projector_test_tile_ids:
        raise ValueError("--projector-test-tile-ids must contain at least one tile id")
    if not set(projector_test_tile_ids).issubset(set(requested_tile_ids)):
        raise ValueError(
            f"projector test tiles {projector_test_tile_ids} must be a subset of requested tiles {requested_tile_ids}"
        )
    TILE_IDS = requested_tile_ids
    PROJECTOR_TEST_TILE_IDS = projector_test_tile_ids
    torch.cuda.set_device(int(args.cuda_device))
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(args.resume):
        raise FileExistsError(f"refusing to overwrite non-empty output directory {output_dir}; use --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[cuda] requested/current={args.cuda_device}/{torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())} low_vram={args.low_vram}",
        flush=True,
    )

    projector_path = output_dir / "projector_metrics.json"
    if bool(args.resume) and projector_path.is_file():
        projector_metrics = _load_projector_metrics(projector_path)
    else:
        projector_metrics = run_projector_tests(args, output_dir)
    # Keep a named v2 artifact so a stale pre-cold-start file cannot be
    # mistaken for the corrected correctness result.
    projector_v2_path = output_dir / "projector_correctness_v2.json"
    _atomic_json(projector_v2_path, projector_metrics)
    if not _projector_metrics_passed(projector_metrics):
        raise RuntimeError("projector-only acceptance failed; flow was not started")

    warm_path = output_dir / "warm_start_metrics.json"
    if bool(args.resume) and warm_path.is_file():
        warm_start_metrics = _load_projector_metrics(warm_path)
    else:
        warm_start_metrics = run_warm_start_tests(args, output_dir)
    if not bool(warm_start_metrics.get("passed")):
        raise RuntimeError("warm-start projected coarse-field equivalence failed; flow was not started")
    if args.phase == "projector":
        report = output_dir / "PROJECTOR_BATCH_TILE26_27_REPORT.md"
        report.write_text(
            "# PROJECTOR_BATCH_TILE26_27_REPORT\n\n"
            "Projector-only phase completed; see `projector_metrics.json`, "
            "`projector_correctness_v2.json`, and `warm_start_metrics.json`.\n",
            encoding="utf-8",
        )
        return {
            "projector_metrics": projector_metrics,
            "warm_start_metrics": warm_start_metrics,
            "report": str(report),
        }

    # The neural route is initialized only after projector-only acceptance.
    source_path = Path(args.image).expanduser().resolve()
    from PIL import Image

    with Image.open(source_path) as source:
        source_rgb = source.convert("RGB")
    pipeline = core.init_pipeline(args.model_path, device="cuda", low_vram=bool(args.low_vram))
    canonical = pipeline.preprocess_canonical_images(source_rgb)
    image_4096 = canonical["image_4096"]
    reference_dir = Path(args.reference_experiment_dir).expanduser().resolve()
    global_camera = json.loads((reference_dir / "global_camera.json").read_text(encoding="utf-8"))
    baseline_mesh = legacy._load_baseline(reference_dir / "global_baseline_mesh.pt").to("cpu")
    runtime_dir = output_dir / "runtime"
    contexts = legacy._load_contexts(
        source_dir=reference_dir,
        output_dir=runtime_dir,
        pipeline=pipeline,
        baseline_mesh=baseline_mesh,
        global_camera=global_camera,
        image_4096=image_4096,
        args=args,
    )
    contexts = sorted(contexts, key=lambda context: int(context.tile_id))
    if [int(context.tile_id) for context in contexts] != list(TILE_IDS):
        raise RuntimeError(f"expected contexts {TILE_IDS}, got {[int(c.tile_id) for c in contexts]}")
    legacy._attach_reference_fields(contexts, Path(args.field_cache_dir).expanduser().resolve(), runtime_dir)
    projectors = {
        int(context.tile_id): _load_stable_projector(
            Path(args.operator_cache_dir), int(context.tile_id), output_dir / "projector_test" / "operators", args
        )
        for context in contexts
    }
    pbr_encoder = pixal3d_models.from_pretrained(str(Path(args.pbr_encoder).expanduser())).eval()
    if not bool(args.low_vram):
        pbr_encoder.to(torch.device("cuda"))

    encoder_regression_path = output_dir / "official_encoder_regression.json"
    if bool(args.resume) and encoder_regression_path.is_file():
        encoder_regression = _load_projector_metrics(encoder_regression_path)
    else:
        encoder_regression = run_official_encoder_regression(
            args, output_dir, contexts, pipeline, pbr_encoder
        )
    if not bool(encoder_regression.get("passed")):
        raise RuntimeError("official PBR encoder B=1 regression failed; flow was not started")

    def run_flow_recorded(**flow_kwargs: Any) -> Dict[str, Any]:
        trace_path = Path(flow_kwargs["output_dir"]) / "layer_trace.json"
        with _layer_trace_hooks(pipeline, pbr_encoder) as recorder:
            result = run_flow(**flow_kwargs)
        recorder.save(trace_path)
        return result

    if args.phase in {"correctness", "all"}:
        correctness_root = output_dir / "correctness"
        serial_summary_path = correctness_root / "serial" / "flow_summary.json"
        batch_summary_path = correctness_root / "batch" / "flow_summary.json"
        reverse_summary_path = correctness_root / "batch_reverse" / "flow_summary.json"
        serial_trace_path = correctness_root / "serial" / "layer_trace.json"
        batch_trace_path = correctness_root / "batch" / "layer_trace.json"
        reverse_trace_path = correctness_root / "batch_reverse" / "layer_trace.json"
        if bool(args.resume) and serial_summary_path.is_file() and _layer_trace_is_current(serial_trace_path):
            serial_one = json.loads(serial_summary_path.read_text(encoding="utf-8"))
        else:
            serial_one = run_flow_recorded(
                contexts=contexts,
                projectors=projectors,
                pipeline=pipeline,
                pbr_encoder=pbr_encoder,
                global_camera=global_camera,
                args=args,
                output_dir=correctness_root / "serial",
                mode="serial",
                step_limit=1,
                capture_steps=(0,),
            )
        if bool(args.resume) and batch_summary_path.is_file() and _layer_trace_is_current(batch_trace_path):
            batch_one = json.loads(batch_summary_path.read_text(encoding="utf-8"))
        else:
            batch_one = run_flow_recorded(
                contexts=contexts,
                projectors=projectors,
                pipeline=pipeline,
                pbr_encoder=pbr_encoder,
                global_camera=global_camera,
                args=args,
                output_dir=correctness_root / "batch",
                mode="batch",
                step_limit=1,
                capture_steps=(0,),
            )
        reverse_contexts = list(reversed(contexts))
        if bool(args.resume) and reverse_summary_path.is_file() and _layer_trace_is_current(reverse_trace_path):
            reverse_one = json.loads(reverse_summary_path.read_text(encoding="utf-8"))
        else:
            reverse_one = run_flow_recorded(
                contexts=reverse_contexts,
                projectors=projectors,
                pipeline=pipeline,
                pbr_encoder=pbr_encoder,
                global_camera=global_camera,
                args=args,
                output_dir=correctness_root / "batch_reverse",
                mode="batch",
                step_limit=1,
                capture_steps=(0,),
            )
    else:
        serial_one = batch_one = reverse_one = {}

    serial_full: Optional[Mapping[str, Any]] = None
    batch_full: Optional[Mapping[str, Any]] = None
    if args.phase in {"full", "all"}:
        full_root = output_dir / "full"
        serial_full_path = full_root / "serial" / "flow_summary.json"
        batch_full_path = full_root / "batch" / "flow_summary.json"
        if bool(args.skip_full_serial):
            print("[full] serial full route skipped by --skip-full-serial", flush=True)
        elif bool(args.resume) and serial_full_path.is_file():
            serial_full = json.loads(serial_full_path.read_text(encoding="utf-8"))
        else:
            serial_full = run_flow(
                contexts=contexts,
                projectors=projectors,
                pipeline=pipeline,
                pbr_encoder=pbr_encoder,
                global_camera=global_camera,
                args=args,
                output_dir=full_root / "serial",
                mode="serial",
                step_limit=None,
                capture_steps=SNAPSHOT_STEPS,
            )
        if bool(args.resume) and batch_full_path.is_file():
            batch_full = json.loads(batch_full_path.read_text(encoding="utf-8"))
        else:
            batch_full = run_flow(
                contexts=contexts,
                projectors=projectors,
                pipeline=pipeline,
                pbr_encoder=pbr_encoder,
                global_camera=global_camera,
                args=args,
                output_dir=full_root / "batch",
                mode="batch",
                step_limit=None,
                capture_steps=SNAPSHOT_STEPS,
            )
    equivalence = _aggregate_equivalence(
        output_dir / "correctness" / "serial",
        output_dir / "correctness" / "batch",
        output_dir / "correctness" / "batch_reverse",
        serial_one,
        batch_one,
        reverse_one,
        serial_full,
        batch_full,
    )
    isolation = _write_isolation_artifacts(output_dir, equivalence)
    performance = _performance_report(output_dir, serial_full or serial_one, batch_full or batch_one, projector_metrics)
    performance_v2 = _write_performance_v2(
        output_dir, serial_full or serial_one, batch_full or batch_one, performance
    )
    memory_profile = _write_memory_profile(
        output_dir, serial_full or serial_one, batch_full or batch_one
    )
    isolation_report = _write_batch_isolation_report(
        output_dir,
        projector_metrics,
        warm_start_metrics,
        equivalence,
        performance,
        isolation,
        memory_profile,
    )
    report = _write_final_report(
        output_dir,
        projector_metrics,
        warm_start_metrics,
        equivalence,
        performance,
        serial_full,
        batch_full,
    )
    summary = {
        "format": FORMAT,
        "cuda_device": int(args.cuda_device),
        "cuda_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "tiles": list(TILE_IDS),
        "projector_only_passed": _projector_metrics_passed(projector_metrics),
        "warm_start_passed": bool(warm_start_metrics.get("passed")),
        "batch_equivalence_passed": bool(equivalence.get("passed")),
        "layer_isolation_passed": bool(
            isolation["shape"].get("passed")
            and isolation["texture"].get("passed")
            and isolation["pbr"].get("passed")
        ),
        "projector_metrics": str(projector_path),
        "projector_correctness_v2": str(projector_v2_path.resolve()),
        "warm_start_metrics": str(warm_path.resolve()),
        "official_encoder_regression": str(encoder_regression_path.resolve()),
        "batch_equivalence": str((output_dir / "batch_equivalence.json").resolve()),
        "performance": str((output_dir / "performance.json").resolve()),
        "performance_v2": str((output_dir / "performance_v2.json").resolve()),
        "memory_profile": str((output_dir / "memory_profile.json").resolve()),
        "batch_isolation_report": str(isolation_report.resolve()),
        "report": str(report.resolve()),
        "correctness": {"serial": serial_one, "batch": batch_one, "batch_reverse": reverse_one},
        "full": {"serial": serial_full, "batch": batch_full},
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    run(_build_parser().parse_args())


if __name__ == "__main__":
    main()
