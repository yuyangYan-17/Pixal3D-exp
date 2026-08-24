#!/usr/bin/env python3
"""Global-master 4096 tile endpoint-rollout experiment.

This file is the executable implementation of ``Codex2.md``.  It intentionally
does not reuse the older global-C4096 carrier or velocity-averaging routes.
The important representation is a CPU-stable table of irregular master rows:

    master_id -> global q / global 4096 uv / first-owner tile

Each tile has an exact centered-local C64 view of those rows.  The encoder is
used only to obtain activation coordinates.  Shape and texture flow states are
created from shared master-keyed noise and are synchronized with a Jacobi
barrier: every tile in a real sparse batch rolls out the frozen state through
the remaining official schedule, endpoints are fused in FP32, and the global
state advances once through the official endpoint-to-velocity formula.

The default batch profile follows the CUDA4 reference naming convention:
``flow=44, decode=12, encode=13``.  A batch element is one tile; sparse rows
remain variable length and carry an explicit batch id.  No serial B=1 model
fallback is hidden behind the batching helpers.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# These must be set before importing torch/sparse backends.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    str(Path(__file__).with_name("autotune_cache.json")),
)

import numpy as np
import o_voxel
import torch
from PIL import Image, ImageDraw

import pixal3d.models as pixal3d_models
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithFacePbr, MeshWithVertexPbr, MeshWithVoxel


FORMAT = "pixal3d_global4096_tile_endpoint_rollout_sync_v1"
CANONICAL_SIZE = 4096
GLOBAL_SIZE = 1024
SHAPE_IMAGE_SIZE = 512
TILE_SIZE = 1024
TILE_STRIDE = 512
TILE_GRID = 7
TILE_COUNT = TILE_GRID * TILE_GRID
LOCAL_OVOXEL = 1024
LATENT_SIZE = 64
SIGMA_PIXELS = TILE_SIZE / 4.0

# The reference output is named ``batch44_12_13``.  Keep the mapping explicit
# so it cannot be accidentally reordered in a config or report.
FLOW_BATCH_SIZE = 44
DECODE_BATCH_SIZE = 12
ENCODE_BATCH_SIZE = 13
# The CUDA4 reference keeps the geometry/initial sparse encoder at B=1.  A
# local C1024 tile can contain several million O-Voxel points, so this is a
# separate memory-safe batch from the B=13 image-condition encoder below.
NATIVE_GEOMETRY_ENCODE_BATCH_SIZE = 1
# The texture condition's NAF branch targets 1024×1024 feature maps.  Keeping
# this model call at B=1 avoids stacking that high-resolution branch on top of
# the sparse-encoder workspace; the reference B=13 encode profile is still
# used by the shape/image-condition path and is recorded explicitly.
TEXTURE_CONDITION_BATCH_SIZE = 1

DEFAULT_IMAGE = Path("/home/nvme04/yyyan/Pixal3D/assets/images/0_img.png")
DEFAULT_BASELINE_DIR = Path("/home/nvme04/yyyan/Pixal3D/outputs/baseline1024_pbr_mesh_compare")
DEFAULT_OUTPUT_DIR = Path("/home/nvme04/yyyan/Pixal3D/outputs/global4096_tile_endpoint_rollout_sync_cuda5")
DEFAULT_ENCODER_ROOT = Path(
    "/home/nvme04/yyyan/download/model/TRELLIS.2-4B/"
    "microsoft/TRELLIS___2-4B/ckpts"
)
DEFAULT_SHAPE_ENCODER = DEFAULT_ENCODER_ROOT / "shape_enc_next_dc_f16c32_fp16"
DEFAULT_MOGE_MODEL = Path("/home/nvme04/yyyan/download/model/moge-2-vitl/model.pt")

PBR_LAYOUT = {
    "base_color": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}


class SupportCollisionError(RuntimeError):
    """Raised for a forbidden local-C64 collision in a tile view."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        super().__init__(
            f"tile {self.report.get('tile_id')} has {self.report.get('collision_count')} "
            "different master IDs quantized to the same local C64 coordinate"
        )


@dataclass
class TileView:
    tile_id: int
    box: Tuple[int, int, int, int]
    transform: Any
    master_ids: torch.Tensor          # [M], CPU int64
    local_coords: torch.Tensor        # [M,4], CPU int32, batch zero
    master_uv_4096: torch.Tensor      # [M,2], CPU float32
    gaussian_weight: torch.Tensor     # [M], CPU float32
    stats: Dict[str, Any]


@dataclass
class MasterSupport:
    master_q_global: torch.Tensor     # [N,3], CPU float32
    master_uv_4096: torch.Tensor      # [N,2], CPU float32
    owner_tile_id: torch.Tensor       # [N], CPU int16
    owner_local_coord_c64: torch.Tensor  # [N,3], CPU int32
    tile_views: Dict[int, TileView]
    tile_stats: Dict[int, Dict[str, Any]]
    collision_report: List[Dict[str, Any]]
    roundtrip_max_abs_error: float


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
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
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _capture_rng_state(device: torch.device) -> Dict[str, Any]:
    state: Dict[str, Any] = {"cpu": torch.get_rng_state().clone()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device=device).clone()
    return state


def _restore_rng_state(state: Mapping[str, Any], device: torch.device) -> None:
    if "cpu" in state:
        torch.set_rng_state(state["cpu"].cpu())
    if device.type == "cuda" and "cuda" in state:
        # torch.cuda.set_rng_state consumes a CPU ByteTensor even when the
        # target generator is a CUDA device.
        torch.cuda.set_rng_state(state["cuda"].cpu(), device=device)


def _sha256_tensor(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(tuple(value.shape)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def _git_metadata(root: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"commit": None, "dirty": None, "status": None}
    try:
        result["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True
        )
        result["status"] = status.splitlines()
        result["dirty"] = bool(status.strip())
    except Exception as exc:  # pragma: no cover - git is present in normal runs
        result["error"] = str(exc)
    return result


def _nvidia_smi() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"], text=True
        ).strip()
    except Exception:
        return None


def _runtime_metadata(physical_device: int, logical_device: torch.device) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this task must run on physical CUDA 5")
    return {
        "physical_cuda_device_requested": int(physical_device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": str(logical_device),
        "torch_current_device": int(torch.cuda.current_device()),
        "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "total_memory_bytes": int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory),
        "free_memory_bytes": int(torch.cuda.mem_get_info(torch.cuda.current_device())[0]),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "python": sys.version,
        "platform": platform.platform(),
        "nvidia_smi": _nvidia_smi(),
    }


def _tile_layout(
    canonical_size: int = CANONICAL_SIZE,
    tile_size: int = TILE_SIZE,
    stride: int = TILE_STRIDE,
) -> List[Tuple[int, int, int, int]]:
    if canonical_size <= 0 or tile_size <= 0 or stride <= 0:
        raise ValueError("canonical_size, tile_size and stride must be positive")
    starts = list(range(0, canonical_size - tile_size + 1, stride))
    if starts[-1] != canonical_size - tile_size:
        raise ValueError("tile layout does not land on the canonical edge")
    boxes = [(x, y, x + tile_size, y + tile_size) for y in starts for x in starts]
    if len(boxes) != TILE_COUNT:
        raise AssertionError(f"expected 49 tiles, got {len(boxes)}")
    return boxes


def _inside_box(uv: torch.Tensor, box: Sequence[int]) -> torch.Tensor:
    x0, y0, x1, y1 = (float(v) for v in box)
    # Pixel rectangles are half-open.  The canonical right/bottom edge is an
    # outer boundary and is not duplicated by a neighboring tile.
    return (
        (uv[:, 0] >= x0) & (uv[:, 0] < x1)
        & (uv[:, 1] >= y0) & (uv[:, 1] < y1)
    )


def _inside_any_box(uv: torch.Tensor, boxes: Sequence[Sequence[int]]) -> torch.Tensor:
    result = torch.zeros((uv.shape[0],), dtype=torch.bool, device=uv.device)
    for box in boxes:
        result |= _inside_box(uv, box)
    return result


def _c64_coords_from_q(q_local: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if q_local.ndim != 2 or q_local.shape[1] != 3:
        raise ValueError(f"q_local must be [N,3], got {tuple(q_local.shape)}")
    continuous = (q_local + 1.0) * ((LATENT_SIZE - 1) / 2.0)
    indices = torch.round(continuous).to(torch.int32)
    valid = torch.isfinite(continuous).all(dim=1) & (
        (indices >= 0).all(dim=1) & (indices < LATENT_SIZE).all(dim=1)
    )
    return indices, valid


def _coord_keys(coords: torch.Tensor, resolution: int = LATENT_SIZE) -> torch.Tensor:
    xyz = coords[:, -3:].to(torch.int64)
    return (xyz[:, 0] * resolution + xyz[:, 1]) * resolution + xyz[:, 2]


def _sort_coords(coords: torch.Tensor) -> torch.Tensor:
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"coords must be [N,3] or [N,4], got {tuple(coords.shape)}")
    key = _coord_keys(coords)
    return torch.argsort(key, stable=True)


def _assert_unique_coords(coords: torch.Tensor, label: str) -> None:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"{label}: expected [N,4] coordinates")
    if coords.shape[0] and torch.unique(_coord_keys(coords), sorted=True).numel() != coords.shape[0]:
        raise RuntimeError(f"{label}: duplicate sparse coordinates")


def _pack_sparse_batch(values: Sequence[SparseTensor], label: str) -> SparseTensor:
    """Pack variable-length local B=1 sparse tensors into a real sparse B batch."""
    if not values:
        raise ValueError(f"{label}: empty batch")
    features: List[torch.Tensor] = []
    coords: List[torch.Tensor] = []
    feature_shape = tuple(values[0].feats.shape[1:])
    for batch_id, value in enumerate(values):
        if tuple(value.feats.shape[1:]) != feature_shape:
            raise RuntimeError(f"{label}: feature shapes differ")
        local = value.coords.detach()
        if local.ndim != 2 or local.shape[1] != 4 or (local.shape[0] and not torch.all(local[:, 0] == 0)):
            raise RuntimeError(f"{label}: every item must have local batch-zero [N,4] coords")
        local = local.clone()
        local[:, 0] = int(batch_id)
        features.append(value.feats.detach())
        coords.append(local)
    packed = SparseTensor(torch.cat(features, dim=0), torch.cat(coords, dim=0))
    if len(packed) != len(values):
        raise RuntimeError(f"{label}: packed SparseTensor has batch {len(packed)}")
    return packed


def _split_sparse_batch(value: SparseTensor, batch_size: int, label: str) -> List[SparseTensor]:
    if not isinstance(value, SparseTensor):
        raise TypeError(f"{label}: expected SparseTensor, got {type(value)!r}")
    parts: List[SparseTensor] = []
    for batch_id in range(int(batch_size)):
        mask = value.coords[:, 0] == int(batch_id)
        if not bool(mask.any()):
            raise RuntimeError(f"{label}: batch {batch_id} has no rows")
        coords = value.coords[mask].detach().clone()
        coords[:, 0] = 0
        parts.append(SparseTensor(value.feats[mask].detach().clone(), coords))
    return parts


def _normalize_slat(value: SparseTensor, normalization: Mapping[str, Sequence[float]]) -> SparseTensor:
    mean = torch.as_tensor(normalization["mean"], dtype=value.feats.dtype, device=value.device)[None]
    std = torch.as_tensor(normalization["std"], dtype=value.feats.dtype, device=value.device)[None]
    if mean.shape[1] != value.feats.shape[1]:
        raise ValueError(f"normalization channels {mean.shape[1]} != latent channels {value.feats.shape[1]}")
    return SparseTensor((value.feats - mean) / std, value.coords.detach().clone())


def _denormalize_features(value: torch.Tensor, normalization: Mapping[str, Sequence[float]]) -> torch.Tensor:
    mean = torch.as_tensor(normalization["mean"], dtype=value.dtype, device=value.device)[None]
    std = torch.as_tensor(normalization["std"], dtype=value.dtype, device=value.device)[None]
    if mean.shape[1] != value.shape[1]:
        raise ValueError(f"denormalization channels {mean.shape[1]} != latent channels {value.shape[1]}")
    return value * std + mean


def _prediction_kwargs(params: Mapping[str, Any]) -> Dict[str, Any]:
    excluded = {"steps", "rescale_t", "verbose", "tqdm_desc", "record_trajectory", "trajectory_device", "return_model_history"}
    return {key: value for key, value in params.items() if key not in excluded}


def gaussian_weights(uv: torch.Tensor, box: Sequence[int], sigma: float = SIGMA_PIXELS) -> torch.Tensor:
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must be [N,2]")
    x0, y0, x1, y1 = (float(v) for v in box)
    center = uv.new_tensor([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
    delta = uv - center
    return torch.exp(-0.5 * delta.square().sum(dim=1) / float(sigma) ** 2).to(torch.float32)


def _tile_center_gaussian(uv: torch.Tensor, box: Sequence[int], sigma: float) -> torch.Tensor:
    return gaussian_weights(uv, box, sigma)


def _build_master_support(
    native_coords_by_tile: Mapping[int, torch.Tensor],
    transforms: Mapping[int, Any],
    global_camera: Mapping[str, float],
    *,
    tile_boxes: Optional[Sequence[Sequence[int]]] = None,
    sigma_pixels: float = SIGMA_PIXELS,
    collision_policy: str = "error",
) -> MasterSupport:
    """Build the irregular master using only 2-D first-owner rectangles.

    This function is deliberately independent of encoder feature values.  The
    input is activation coordinates only; a poisoned or absent feature tensor
    cannot affect master IDs, q positions, ownership, or tile views.
    """
    boxes = list(tile_boxes or _tile_layout())
    if len(boxes) != TILE_COUNT:
        raise ValueError("the master support requires the fixed 49-tile layout")
    if collision_policy not in {"error"}:
        raise ValueError("the main experiment supports collision_policy='error' only")

    master_q_parts: List[torch.Tensor] = []
    master_uv_parts: List[torch.Tensor] = []
    owner_parts: List[torch.Tensor] = []
    owner_coord_parts: List[torch.Tensor] = []
    tile_views: Dict[int, TileView] = {}
    tile_stats: Dict[int, Dict[str, Any]] = {}
    collision_reports: List[Dict[str, Any]] = []
    previous_boxes: List[Tuple[int, int, int, int]] = []
    next_master_id = 0

    for tile_id in range(TILE_COUNT):
        box = tuple(int(v) for v in boxes[tile_id])
        native = native_coords_by_tile.get(tile_id)
        if native is None or native.numel() == 0:
            tile_stats[tile_id] = {"status": "inactive", "reason": "no_encoder_activation"}
            continue
        native = native.detach().cpu().to(torch.int32).contiguous()
        if native.ndim != 2 or native.shape[1] != 4 or (native.shape[0] and not torch.all(native[:, 0] == 0)):
            raise ValueError(f"tile {tile_id}: native coords must be local [N,4] with batch zero")
        _assert_unique_coords(native, f"tile {tile_id} native support")

        native_xyz = native[:, 1:].to(torch.float32)
        native_q_local = native_xyz / ((LATENT_SIZE - 1) / 2.0) - 1.0
        native_q_global, native_uv = core._local_q_to_global_q(
            native_q_local,
            global_camera=global_camera,
            transform=transforms[tile_id],
        )
        finite = torch.isfinite(native_q_global).all(dim=1) & torch.isfinite(native_uv).all(dim=1)
        if not bool(finite.all()):
            raise RuntimeError(f"tile {tile_id}: non-finite native activation projection")
        in_tile = _inside_box(native_uv, box)
        in_previous = _inside_any_box(native_uv, previous_boxes)
        overlap_mask = in_tile & in_previous
        new_mask = in_tile & ~in_previous

        order = _sort_coords(native)
        new_rows = order[new_mask[order]]
        new_q = native_q_global.index_select(0, new_rows)
        new_uv = native_uv.index_select(0, new_rows)
        new_owner_coords = native.index_select(0, new_rows)[:, 1:].contiguous()
        new_ids = torch.arange(next_master_id, next_master_id + new_rows.numel(), dtype=torch.int64)
        next_master_id += int(new_rows.numel())
        if new_rows.numel():
            master_q_parts.append(new_q)
            master_uv_parts.append(new_uv)
            owner_parts.append(torch.full((new_rows.numel(),), tile_id, dtype=torch.int16))
            owner_coord_parts.append(new_owner_coords)

        if master_uv_parts:
            all_uv_before = torch.cat(master_uv_parts, dim=0)
            existing_ids = torch.where(_inside_box(all_uv_before, box))[0].to(torch.int64)
        else:
            existing_ids = torch.empty((0,), dtype=torch.int64)
        # At this point ``existing_ids`` includes newly-created rows.  Because
        # new rows lie in R_new, subtract their IDs and append them once below.
        if new_ids.numel():
            existing_ids = existing_ids[existing_ids < next_master_id - new_ids.numel()]
        selected_ids = torch.cat((existing_ids, new_ids), dim=0)

        all_q = torch.cat(master_q_parts, dim=0) if master_q_parts else torch.empty((0, 3), dtype=torch.float32)
        if selected_ids.numel():
            selected_q = all_q.index_select(0, selected_ids)
            selected_local_q, selected_uv = core._global_q_to_local_q(
                selected_q,
                global_camera=global_camera,
                transform=transforms[tile_id],
            )
            local_xyz, local_valid = _c64_coords_from_q(selected_local_q)
            if new_ids.numel():
                owner_part = local_valid[-new_ids.numel():]
                if not bool(owner_part.all()):
                    raise RuntimeError(f"tile {tile_id}: owner activation cannot be represented in local C64")
            kept = local_valid
            kept_ids = selected_ids[kept]
            kept_local = local_xyz[kept]
            kept_uv = all_uv_before.index_select(0, kept_ids)
        else:
            kept_ids = torch.empty((0,), dtype=torch.int64)
            kept_local = torch.empty((0, 3), dtype=torch.int32)
            kept_uv = torch.empty((0, 2), dtype=torch.float32)

        view_coords = torch.cat((torch.zeros((kept_local.shape[0], 1), dtype=torch.int32), kept_local), dim=1)
        key = _coord_keys(view_coords)
        unique_key, counts = torch.unique(key, return_counts=True)
        collision = counts > 1
        if bool(collision.any()):
            bad_keys = unique_key[collision]
            rows = torch.where(torch.isin(key, bad_keys))[0]
            report = {
                "tile_id": tile_id,
                "collision_count": int(bad_keys.numel()),
                "rows": rows.tolist()[:256],
                "master_ids": kept_ids.index_select(0, rows).tolist()[:256],
                "local_coords": view_coords.index_select(0, rows).tolist()[:256],
                "master_q_global": selected_q.index_select(0, rows).tolist()[:256],
                "master_uv_4096": kept_uv.index_select(0, rows).tolist()[:256],
                "owner_tile_id": (
                    torch.cat(owner_parts, dim=0).index_select(0, kept_ids.index_select(0, rows)).tolist()
                    if owner_parts else []
                ),
                "policy": collision_policy,
            }
            collision_reports.append(report)
            raise SupportCollisionError(report)
        _assert_unique_coords(view_coords, f"tile {tile_id} local C64 view")

        weights = _tile_center_gaussian(kept_uv, box, sigma_pixels)
        tile_views[tile_id] = TileView(
            tile_id=tile_id,
            box=box,
            transform=transforms[tile_id],
            master_ids=kept_ids.to(torch.int64).contiguous(),
            local_coords=view_coords.contiguous(),
            master_uv_4096=kept_uv.to(torch.float32).contiguous(),
            gaussian_weight=weights.contiguous(),
            stats={
                "status": "active",
                "native_activation_count": int(native.shape[0]),
                "native_in_tile_count": int(in_tile.sum()),
                "native_overlap_discard_count": int(overlap_mask.sum()),
                "native_new_region_count": int(new_mask.sum()),
                "new_master_count": int(new_ids.numel()),
                "reused_master_count": int(existing_ids.numel()),
                "out_of_local_range_count": int((~local_valid).sum()) if selected_ids.numel() else 0,
                "view_token_count": int(view_coords.shape[0]),
                "local_coord_unique": True,
                "ownership_rule": "2D first-owner half-open rectangle union",
                "native_feature_values_used": False,
            },
        )
        tile_stats[tile_id] = dict(tile_views[tile_id].stats)
        if view_coords.shape[0] == 0:
            tile_views.pop(tile_id)
            tile_stats[tile_id] = {"status": "inactive", "reason": "empty_view_after_exact_remap"}
        else:
            previous_boxes.append(box)

    if not master_q_parts:
        raise RuntimeError("no active tile created a master support")
    master_q = torch.cat(master_q_parts, dim=0).to(torch.float32).contiguous()
    master_uv = torch.cat(master_uv_parts, dim=0).to(torch.float32).contiguous()
    owner = torch.cat(owner_parts, dim=0).to(torch.int16).contiguous()
    owner_coord = torch.cat(owner_coord_parts, dim=0).to(torch.int32).contiguous()
    master_count = int(master_q.shape[0])
    if not (master_uv.shape[0] == owner.shape[0] == owner_coord.shape[0] == master_count):
        raise RuntimeError("master support tables have different row counts")
    coverage = torch.zeros((master_count,), dtype=torch.int32)
    for view in tile_views.values():
        coverage.index_add_(0, view.master_ids, torch.ones_like(view.master_ids, dtype=torch.int32))
        owner_ids = torch.where(owner == int(view.tile_id))[0]
        if owner_ids.numel() and not bool(torch.isin(owner_ids, view.master_ids).all()):
            raise RuntimeError(f"owner tile {view.tile_id} does not contain all created master IDs")
    if bool((coverage <= 0).any()):
        raise RuntimeError("some master IDs do not belong to a tile view")

    # Global/local/global is an exact camera identity check, not a voxel-index
    # check.  It is evaluated on all master rows before any feature flow.
    roundtrip_max = 0.0
    for tile_id, view in tile_views.items():
        q_local, _ = core._global_q_to_local_q(
            master_q.index_select(0, view.master_ids),
            global_camera=global_camera,
            transform=transforms[tile_id],
        )
        q_back, _ = core._local_q_to_global_q(
            q_local,
            global_camera=global_camera,
            transform=transforms[tile_id],
        )
        if q_back.numel():
            roundtrip_max = max(roundtrip_max, float((q_back - master_q.index_select(0, view.master_ids)).abs().max()))

    return MasterSupport(
        master_q_global=master_q,
        master_uv_4096=master_uv,
        owner_tile_id=owner,
        owner_local_coord_c64=owner_coord,
        tile_views=tile_views,
        tile_stats=tile_stats,
        collision_report=collision_reports,
        roundtrip_max_abs_error=roundtrip_max,
    )


def _owner_map_images(support: MasterSupport, output_dir: Path, active_tile_ids: Sequence[int]) -> None:
    """Save the rectangle owner map and sparse support density maps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    owner = np.full((CANONICAL_SIZE, CANONICAL_SIZE), -1, dtype=np.int16)
    overlap = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE), dtype=np.uint8)
    for tile_id in active_tile_ids:
        x0, y0, x1, y1 = _tile_layout()[tile_id]
        overlap[y0:y1, x0:x1] = np.minimum(255, overlap[y0:y1, x0:x1].astype(np.uint16) + 1).astype(np.uint8)
        block = owner[y0:y1, x0:x1]
        block[block < 0] = int(tile_id)
    density = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE), dtype=np.uint16)
    for view in support.tile_views.values():
        uv = torch.floor(view.master_uv_4096).to(torch.int64).numpy()
        valid = (uv[:, 0] >= 0) & (uv[:, 0] < CANONICAL_SIZE) & (uv[:, 1] >= 0) & (uv[:, 1] < CANONICAL_SIZE)
        np.add.at(density, (uv[valid, 1], uv[valid, 0]), 1)

    colors = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE, 3), dtype=np.uint8)
    palette = np.asarray([(37 * i % 256, 97 * i % 256, 181 * i % 256) for i in range(TILE_COUNT)], dtype=np.uint8)
    valid_owner = owner >= 0
    colors[valid_owner] = palette[owner[valid_owner]]
    Image.fromarray(colors, mode="RGB").save(output_dir / "support_owner_map_4096.png")
    Image.fromarray(np.minimum(255, density).astype(np.uint8), mode="L").save(output_dir / "support_density_map_4096.png")
    Image.fromarray((overlap.astype(np.float32) / max(1, int(overlap.max())) * 255).astype(np.uint8), mode="L").save(output_dir / "support_overlap_count_map_4096.png")
    _atomic_save(output_dir / "support_owner_map_4096.pt", {"owner_tile_id": torch.from_numpy(owner), "overlap_count": torch.from_numpy(overlap), "density": torch.from_numpy(density)})


def _save_master_support(support: MasterSupport, output_dir: Path, transforms: Mapping[int, Any]) -> None:
    support_dir = output_dir / "support"
    support_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save(
        support_dir / "master_support.pt",
        {
            "format": FORMAT,
            "master_id": torch.arange(support.master_q_global.shape[0], dtype=torch.int64),
            "master_q_global": support.master_q_global,
            "master_uv_4096": support.master_uv_4096,
            "owner_tile_id": support.owner_tile_id,
            "owner_local_coord_c64": support.owner_local_coord_c64,
            "encoder_feature_values_present": False,
        },
    )
    tile_dir = support_dir / "tile_views"
    for tile_id, view in sorted(support.tile_views.items()):
        _atomic_save(
            tile_dir / f"tile_{tile_id:02d}.pt",
            {
                "tile_id": tile_id,
                "box": list(view.box),
                "master_ids": view.master_ids,
                "local_coords_c64": view.local_coords,
                "master_uv_4096": view.master_uv_4096,
                "gaussian_weight": view.gaussian_weight,
                "tile_camera": _jsonable(transforms[tile_id].__dict__),
            },
        )
    _atomic_json(
        support_dir / "master_support.json",
        {
            "format": FORMAT,
            "master_token_count": int(support.master_q_global.shape[0]),
            "active_tile_ids": sorted(support.tile_views),
            "inactive_tile_ids": [i for i in range(TILE_COUNT) if i not in support.tile_views],
            "baseline_global_c64_comparison": {
                "baseline_c64_tokens": None,
                "master_density_ratio": None,
                "note": "irregular master; no regular global C1024 lattice is claimed",
            },
            "roundtrip_max_abs_error": support.roundtrip_max_abs_error,
            "tile_stats": support.tile_stats,
            "collision_report": support.collision_report,
            "ownership": "row-major tile order, half-open 2D image rectangles, first-owner union",
            "encoder_feature_values_used": False,
        },
    )
    _atomic_json(support_dir / "support_collision_report.json", {"collisions": support.collision_report, "policy": "error"})
    _owner_map_images(support, support_dir, sorted(support.tile_views))


def _load_mesh(path: Path) -> MeshWithVoxel:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    data = payload.get("mesh", payload) if isinstance(payload, Mapping) else payload
    if isinstance(data, MeshWithVoxel):
        return data.cpu()
    if not isinstance(data, Mapping):
        raise RuntimeError(f"baseline payload is not a MeshWithVoxel/mapping: {path}")
    return MeshWithVoxel(
        torch.as_tensor(data["vertices"]).to(torch.float32),
        torch.as_tensor(data["faces"]).to(torch.int32),
        torch.as_tensor(data["origin"]).tolist(),
        float(data["voxel_size"]),
        torch.as_tensor(data["coords"]).to(torch.int32),
        torch.as_tensor(data["attrs"]).to(torch.float32),
        torch.Size(data["voxel_shape"]),
        dict(data["layout"]),
    )


def _load_camera(baseline_dir: Path) -> Dict[str, float]:
    candidates = [baseline_dir / "global_camera.json", baseline_dir / "summary.json"]
    for path in candidates:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "camera" in payload and isinstance(payload["camera"], Mapping):
            payload = payload["camera"]
        if "camera_angle_x" in payload and "distance" in payload:
            return {
                "camera_angle_x": float(payload["camera_angle_x"]),
                "distance": float(payload["distance"]),
                "mesh_scale": float(payload.get("mesh_scale", 1.0)),
            }
    raise FileNotFoundError(f"no global camera JSON/summary found in {baseline_dir}")


def _save_canonical_images(canonical: Mapping[str, Any], output_dir: Path) -> Dict[str, Any]:
    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    source_square_rgba = canonical["source_square_rgba"]
    rgba_4096 = source_square_rgba.resize((CANONICAL_SIZE, CANONICAL_SIZE), Image.Resampling.LANCZOS)
    rgba_4096.save(input_dir / "canonical_foreground_rgba_4096.png")
    canonical["foreground_mask_4096"].save(input_dir / "canonical_foreground_mask_4096.png")
    canonical["image_4096"].save(input_dir / "canonical_foreground_rgb_4096.png")
    canonical["image_1024"].save(input_dir / "global_input_1024.png")
    canonical["image_512"].save(input_dir / "global_input_512.png")
    return dict(canonical.get("metadata", {}))


def _face_bounds(baseline: MeshWithVoxel, camera: Mapping[str, float], path: Path, chunk_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return payload["face_min"], payload["face_max"], payload["face_finite"]
    face_min, face_max, face_finite = core._project_face_bboxes(
        baseline.vertices.cpu(), baseline.faces.cpu(),
        mesh_scale=float(camera["mesh_scale"]), global_camera=camera,
        chunk_size=int(chunk_size), source_width=CANONICAL_SIZE, source_height=CANONICAL_SIZE,
    )
    _atomic_save(path, {"face_min": face_min, "face_max": face_max, "face_finite": face_finite})
    return face_min, face_max, face_finite


def _minimal_geometry_payload(geometry: Any) -> Dict[str, Any]:
    return {
        "coords": geometry.coords.detach().cpu().to(torch.int32),
        "dual_vertices": geometry.dual_vertices.detach().cpu().to(torch.float32),
        "intersected": geometry.intersected.detach().cpu(),
        "stats": dict(geometry.stats),
    }


def _prepare_native_geometries(
    baseline: MeshWithVoxel,
    camera: Mapping[str, float],
    transforms: Mapping[int, Any],
    face_bounds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output_dir: Path,
    tile_ids: Sequence[int],
    face_chunk_size: int,
) -> Dict[int, Dict[str, Any]]:
    geometry_dir = output_dir / "tile_geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    face_min, face_max, face_finite = face_bounds
    results: Dict[int, Dict[str, Any]] = {}
    for tile_id in tile_ids:
        path = geometry_dir / f"tile_{tile_id:02d}.pt"
        if path.is_file():
            results[tile_id] = torch.load(path, map_location="cpu", weights_only=False)
            continue
        try:
            geometry = core._prepare_tile_geometry(
                global_vertices=baseline.vertices.cpu(), global_faces=baseline.faces.cpu(),
                global_face_min=face_min, global_face_max=face_max, global_face_finite=face_finite,
                global_camera=camera, transform=transforms[tile_id],
            )
        except RuntimeError as exc:
            if "no global triangle projection bbox intersects" in str(exc):
                payload = {"status": "inactive", "reason": "no_projected_face_bbox"}
                _atomic_save(path, payload)
                results[tile_id] = payload
                continue
            raise
        payload = {"status": "active", **_minimal_geometry_payload(geometry)}
        _atomic_save(path, payload)
        results[tile_id] = payload
        del geometry
        _empty_cuda_cache()
    return results


def _encode_activation_batch(
    items: Sequence[Tuple[int, Mapping[str, Any]]],
    encoder: torch.nn.Module,
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """Run the local shape encoder as one physical sparse [B,...] call."""
    dual_values: List[SparseTensor] = []
    intersected_values: List[SparseTensor] = []
    for _, item in items:
        coords = item["coords"].to(torch.int32)
        coords4 = torch.cat((torch.zeros_like(coords[:, :1]), coords), dim=1)
        vertices = SparseTensor(item["dual_vertices"].to(torch.float32), coords4)
        dual_values.append(vertices)
        intersected_values.append(vertices.replace(item["intersected"]))
    dual_batch = _pack_sparse_batch(dual_values, "shape encoder input")
    intersected_batch = _pack_sparse_batch(intersected_values, "shape encoder intersected input")
    encoder.to(device)
    raw = encoder(dual_batch.to(device), intersected_batch.to(device), sample_posterior=False)
    if not isinstance(raw, SparseTensor) or not torch.isfinite(raw.feats).all():
        raise RuntimeError("shape encoder returned an invalid sparse output")
    parts = _split_sparse_batch(raw, len(items), "shape encoder output")
    result: Dict[int, torch.Tensor] = {}
    for (tile_id, _), part in zip(items, parts):
        coords = part.coords.detach().cpu().to(torch.int32)
        coords[:, 0] = 0
        if coords.shape[1] != 4 or coords.shape[0] == 0:
            raise RuntimeError(f"tile {tile_id}: encoder returned empty/invalid C64 support")
        if bool(((coords[:, 1:] < 0) | (coords[:, 1:] >= LATENT_SIZE)).any()):
            raise RuntimeError(f"tile {tile_id}: encoder returned out-of-range C64 coords")
        order = _sort_coords(coords)
        coords = coords.index_select(0, order)
        _assert_unique_coords(coords, f"tile {tile_id} encoder C64 support")
        result[tile_id] = coords
    del raw, dual_batch, intersected_batch, dual_values, intersected_values, parts
    _empty_cuda_cache()
    return result


def _encode_native_supports(
    geometries: Mapping[int, Mapping[str, Any]],
    encoder_path: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int = NATIVE_GEOMETRY_ENCODE_BATCH_SIZE,
) -> Dict[int, torch.Tensor]:
    support_dir = output_dir / "native_support"
    support_dir.mkdir(parents=True, exist_ok=True)
    native: Dict[int, torch.Tensor] = {}
    pending: List[Tuple[int, Mapping[str, Any]]] = []
    for tile_id in sorted(geometries):
        path = support_dir / f"tile_{tile_id:02d}.pt"
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("status") == "active":
                native[tile_id] = payload["coords"].to(torch.int32)
            continue
        item = geometries[tile_id]
        if item.get("status") != "active":
            _atomic_save(path, dict(item))
            continue
        pending.append((tile_id, item))

    if pending:
        encoder = pixal3d_models.from_pretrained(str(encoder_path)).eval()
        try:
            for start in range(0, len(pending), int(batch_size)):
                group = pending[start:start + int(batch_size)]
                encoded = _encode_activation_batch(group, encoder, device)
                for tile_id, coords in encoded.items():
                    # Only the coordinate field is retained.  This is the
                    # explicit feature-poison boundary required by Codex2.
                    payload = {
                        "status": "active",
                        "coords": coords,
                        "encoder_feature_values_used": False,
                        "encoder_batch_size": len(group),
                    }
                    _atomic_save(support_dir / f"tile_{tile_id:02d}.pt", payload)
                    native[tile_id] = coords
        finally:
            encoder.cpu()
            del encoder
            _empty_cuda_cache()
    return native


def _build_batched_image_conditions(
    pipeline: Any,
    image_model: torch.nn.Module,
    views: Mapping[int, TileView],
    tile_images: Mapping[int, Image.Image],
    output_dir: Path,
    stage: str,
    camera: Mapping[str, float],
    device: torch.device,
    batch_size: int = ENCODE_BATCH_SIZE,
) -> Dict[int, Dict[str, Any]]:
    """Extract tile conditions in physical image/feature batches.

    Tile supports have different lengths.  The image extractor therefore gets
    a padded ``[B,K,3]`` grid-index tensor, and padding is discarded before a
    condition is saved.  The valid rows never share a condition or sparse
    identity with another tile.
    """
    condition_root = output_dir / "conditions" / stage
    condition_root.mkdir(parents=True, exist_ok=True)
    conditions: Dict[int, Dict[str, Any]] = {}
    pending: List[TileView] = []
    for tile_id, view in sorted(views.items()):
        path = condition_root / f"tile_{tile_id:02d}.pt"
        if path.is_file():
            conditions[tile_id] = torch.load(path, map_location="cpu", weights_only=False)
        else:
            pending.append(view)
    if not pending:
        return conditions

    image_model.to(device)
    try:
        for start in range(0, len(pending), int(batch_size)):
            group = pending[start:start + int(batch_size)]
            max_tokens = max(int(v.local_coords.shape[0]) for v in group)
            grid_indices = torch.zeros((len(group), max_tokens, 3), dtype=torch.int64, device=device)
            for batch_id, view in enumerate(group):
                n = int(view.local_coords.shape[0])
                grid_indices[batch_id, :n] = view.local_coords[:, 1:].to(device=device, dtype=torch.int64)
            camera_angle = torch.tensor([float(v.transform.camera_angle_x) for v in group], dtype=torch.float32, device=device)
            distance = torch.tensor([float(v.transform.distance) for v in group], dtype=torch.float32, device=device)
            mesh_scale = torch.tensor([float(v.transform.mesh_scale) for v in group], dtype=torch.float32, device=device)
            images = [tile_images[v.tile_id].convert("RGB") for v in group]
            z_global, z_proj = image_model(
                images,
                camera_angle_x=camera_angle,
                distance=distance,
                mesh_scale=mesh_scale,
                grid_indices=grid_indices,
                grid_resolution=LATENT_SIZE,
            )
            if z_global.shape[0] != len(group) or z_proj.shape[0] != len(group):
                raise RuntimeError(f"{stage}: batched image condition returned wrong B")
            for batch_id, view in enumerate(group):
                n = int(view.local_coords.shape[0])
                proj = z_proj[batch_id, :n].detach().cpu().contiguous()
                glob = z_global[batch_id:batch_id + 1].detach().cpu().contiguous()
                conditions[view.tile_id] = {
                    "tile_id": int(view.tile_id),
                    "coords": view.local_coords.detach().cpu().clone(),
                    "cond": {"global": glob, "proj": proj},
                    "neg_cond": {"global": torch.zeros_like(glob), "proj": torch.zeros_like(proj)},
                    "image_batch_size": len(group),
                    "stage": stage,
                }
                _atomic_save(condition_root / f"tile_{view.tile_id:02d}.pt", conditions[view.tile_id])
            del z_global, z_proj, grid_indices
            _empty_cuda_cache()
    finally:
        image_model.cpu()
        _empty_cuda_cache()
    return conditions


def _pack_flow_condition(
    group: Sequence[TileView],
    conditions: Mapping[int, Mapping[str, Any]],
    coords: torch.Tensor,
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    globals_pos: List[torch.Tensor] = []
    proj_pos: List[torch.Tensor] = []
    for batch_id, view in enumerate(group):
        condition = conditions[view.tile_id]
        if not torch.equal(condition["coords"].to(torch.int32), view.local_coords):
            raise RuntimeError(f"tile {view.tile_id}: condition/local view rows differ")
        globals_pos.append(condition["cond"]["global"])
        proj_pos.append(condition["cond"]["proj"])
    proj = torch.cat(proj_pos, dim=0).to(device=device)
    neg_proj = torch.zeros_like(proj)
    glob = torch.cat(globals_pos, dim=0).to(device=device)
    neg_glob = torch.zeros_like(glob)
    return {
        "cond": {"global": glob, "proj": SparseTensor(proj, coords)},
        "neg_cond": {"global": neg_glob, "proj": SparseTensor(neg_proj, coords)},
    }


def _pack_state_batch(
    group: Sequence[TileView],
    master_features: torch.Tensor,
    device: torch.device,
) -> Tuple[SparseTensor, torch.Tensor]:
    values: List[SparseTensor] = []
    ids_parts: List[torch.Tensor] = []
    for view in group:
        ids = view.master_ids
        ids_parts.append(ids)
        values.append(SparseTensor(master_features.index_select(0, ids), view.local_coords))
    local = _pack_sparse_batch(values, "flow state batch")
    return local.to(device), torch.cat(ids_parts, dim=0)


def _unpack_state_parts(value: SparseTensor, group: Sequence[TileView], label: str) -> List[SparseTensor]:
    parts = _split_sparse_batch(value, len(group), label)
    for view, part in zip(group, parts):
        if not torch.equal(part.coords.cpu(), view.local_coords):
            raise RuntimeError(f"{label}: model changed tile {view.tile_id} sparse support/order")
    return parts


def _stats_tensor(value: torch.Tensor) -> Dict[str, Any]:
    value = value.detach().float()
    if value.numel() == 0:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "norm": 0.0}
    return {
        "count": int(value.numel()),
        "mean": float(value.mean()),
        "min": float(value.min()),
        "max": float(value.max()),
        "norm": float(value.norm()),
    }


@torch.no_grad()
def _run_synchronized_endpoint_flow(
    *,
    stage: str,
    initial_features: torch.Tensor,
    master_coords: torch.Tensor,
    views: Mapping[int, TileView],
    conditions: Mapping[int, Mapping[str, Any]],
    sampler: Any,
    model: torch.nn.Module,
    sampler_params: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    flow_batch_size: int = FLOW_BATCH_SIZE,
    concat_features: Optional[torch.Tensor] = None,
    resume: bool = False,
    save_step_tensors: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Run nested suffix rollouts with one all-tile endpoint barrier per step."""
    if initial_features.ndim != 2:
        raise ValueError("initial_features must be [N,C]")
    if master_coords.ndim != 2 or master_coords.shape[1] != 4:
        raise ValueError("master_coords must be [N,4]")
    if concat_features is not None and concat_features.shape[0] != initial_features.shape[0]:
        raise ValueError("concat features are not aligned with master rows")
    active = [views[key] for key in sorted(views)]
    if not active:
        raise RuntimeError(f"{stage}: no active tile views")
    steps = int(sampler_params["steps"])
    times = tuple(float(v) for v in sampler.timestep_schedule(steps, float(sampler_params.get("rescale_t", 1.0))))
    if len(times) != steps + 1:
        raise RuntimeError(f"{stage}: sampler schedule has wrong length")

    stage_dir = output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save(stage_dir / "initial_noise.pt", {"coords": master_coords.cpu(), "features": initial_features.cpu(), "stage": stage})
    support_hash = _sha256_tensor(master_coords)
    checkpoint = stage_dir / "checkpoint.pt"
    state = initial_features.detach().cpu().to(torch.float32).contiguous()
    start_step = 0
    records: List[Dict[str, Any]] = []
    if resume and checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("support_sha256") == support_hash and int(saved.get("steps", -1)) == steps:
            state = saved["state"].to(torch.float32)
            start_step = int(saved["next_step"])
            records = list(saved.get("records", []))
            if "rng_state" in saved:
                _restore_rng_state(saved["rng_state"], device)
        else:
            checkpoint.unlink()

    model.to(device)
    model.eval()
    inner_total = 0
    forward_total = 0
    try:
        for outer in range(start_step, steps):
            _sync_cuda()
            started = time.perf_counter()
            frozen = state.clone()
            endpoint_sum = torch.zeros_like(frozen, dtype=torch.float32)
            weight_sum = torch.zeros((frozen.shape[0], 1), dtype=torch.float32)
            tile_records: List[Dict[str, Any]] = []
            for group_start in range(0, len(active), int(flow_batch_size)):
                group = active[group_start:group_start + int(flow_batch_size)]
                local, ids_concat = _pack_state_batch(group, frozen, device)
                local_coords = local.coords
                condition = _pack_flow_condition(group, conditions, local_coords, device)
                concat = None
                if concat_features is not None:
                    concat, _ = _pack_state_batch(group, concat_features.cpu(), device)
                endpoint = local
                group_inner = 0
                for inner in range(outer, steps):
                    t = times[inner]
                    t_next = times[inner + 1]
                    out = sampler.sample_once(
                        model,
                        endpoint,
                        t,
                        t_next,
                        cond=condition["cond"],
                        neg_cond=condition["neg_cond"],
                        concat_cond=concat,
                        **_prediction_kwargs(sampler_params),
                    )
                    endpoint = out.pred_x_prev
                    if not isinstance(endpoint, SparseTensor):
                        raise RuntimeError(f"{stage}: sampler returned non-sparse endpoint")
                    group_inner += 1
                    guidance = float(sampler_params.get("guidance_strength", 1.0))
                    interval = sampler_params.get("guidance_interval")
                    active_cfg = guidance not in (0.0, 1.0)
                    if interval is not None:
                        active_cfg = active_cfg and float(interval[0]) <= t <= float(interval[1])
                    forward_total += 2 if active_cfg else 1
                # Report tile-inner steps, while ``forward_total`` remains the
                # number of physical sparse model calls made by this process.
                inner_total += group_inner * len(group)
                parts = _unpack_state_parts(endpoint, group, f"{stage} endpoint")
                for view, part in zip(group, parts):
                    ids = view.master_ids
                    values = part.feats.detach().cpu().float()
                    weights = view.gaussian_weight.float()
                    endpoint_sum.index_add_(0, ids, values * weights[:, None])
                    weight_sum.index_add_(0, ids, weights[:, None])
                    tile_records.append({
                        "tile_id": view.tile_id,
                        "inner_steps": group_inner,
                        "endpoint": _stats_tensor(values),
                        "weight": _stats_tensor(weights),
                    })
                del local, endpoint, condition, concat, parts
                _empty_cuda_cache()

            if bool((weight_sum <= 0).any()):
                bad = torch.where(weight_sum[:, 0] <= 0)[0][:16].tolist()
                raise RuntimeError(f"{stage} outer {outer}: uncovered master IDs {bad}")
            merged = endpoint_sum / weight_sum
            frozen_sparse = SparseTensor(frozen, master_coords)
            merged_sparse = SparseTensor(merged, master_coords)
            velocity = sampler._xstart_to_pred(frozen_sparse, times[outer], merged_sparse)
            if not isinstance(velocity, SparseTensor):
                raise RuntimeError(f"{stage}: _xstart_to_pred did not preserve SparseTensor")
            velocity_features = velocity.feats.detach().cpu().float()
            state = frozen - float(times[outer] - times[outer + 1]) * velocity_features
            if not torch.isfinite(state).all():
                raise RuntimeError(f"{stage} outer {outer}: non-finite global state")

            step_dir = stage_dir / f"step_{outer:02d}"
            if save_step_tensors:
                _atomic_save(step_dir / "global_state.pt", {"coords": master_coords.cpu(), "features": state})
                _atomic_save(step_dir / "merged_endpoint.pt", {"coords": master_coords.cpu(), "features": merged})
                _atomic_save(step_dir / "sum_weight.pt", weight_sum)
            step_record = {
                "step": outer,
                "t": times[outer],
                "t_next": times[outer + 1],
                "dt": float(times[outer] - times[outer + 1]),
                "inner_steps_per_tile": steps - outer,
                "active_tile_count": len(active),
                "flow_batch_size": int(flow_batch_size),
                "frozen_state": _stats_tensor(frozen),
                "merged_endpoint": _stats_tensor(merged),
                "effective_velocity": _stats_tensor(velocity_features),
                "sum_weight": _stats_tensor(weight_sum),
                "tile_endpoints": tile_records,
                "seconds": float(time.perf_counter() - started),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
            }
            _atomic_json(step_dir / "effective_velocity_stats.json", step_record["effective_velocity"])
            _atomic_json(step_dir / "per_tile_endpoint_stats.json", {"tiles": tile_records})
            _atomic_json(step_dir / "timing_memory.json", {"seconds": step_record["seconds"], "peak_allocated_bytes": step_record["peak_allocated_bytes"], "peak_reserved_bytes": step_record["peak_reserved_bytes"]})
            records.append(step_record)
            _atomic_save(
                checkpoint,
                {
                    "stage": stage,
                    "support_sha256": support_hash,
                    "master_coords": master_coords.cpu(),
                    "steps": steps,
                    "next_step": outer + 1,
                    "state": state,
                    "records": records,
                    "sampler_params": dict(sampler_params),
                    "rng_state": _capture_rng_state(device),
                },
            )
    finally:
        model.cpu()
        _empty_cuda_cache()

    # Reconstruct diagnostics from persisted records on a resume-only call.
    # The loop body is skipped when ``next_step == steps``; reporting zero in
    # that case would make a completed official run look like an empty flow.
    expected_inner = 0
    expected_forwards = 0
    for record in records:
        active_count = int(record.get("active_tile_count", len(active)))
        group_count = (active_count + int(flow_batch_size) - 1) // int(flow_batch_size)
        outer_step = int(record["step"])
        expected_inner += active_count * (steps - outer_step)
        for inner in range(outer_step, steps):
            t_inner = times[inner]
            guidance = float(sampler_params.get("guidance_strength", 1.0))
            interval = sampler_params.get("guidance_interval")
            cfg = guidance not in (0.0, 1.0)
            if interval is not None:
                cfg = cfg and float(interval[0]) <= t_inner <= float(interval[1])
            expected_forwards += group_count * (2 if cfg else 1)
    inner_total = expected_inner
    forward_total = expected_forwards

    result = {
        "stage": stage,
        "steps": steps,
        "schedule": list(times),
        "nested_inner_steps_per_tile": list(range(steps, 0, -1)),
        "total_inner_steps_per_tile": steps * (steps + 1) // 2,
        "active_tile_count": len(active),
        "flow_batch_size": int(flow_batch_size),
        "actual_inner_steps": inner_total,
        "actual_model_forwards": forward_total,
        "actual_model_forwards_estimate": forward_total,
        "records": records,
        "support_sha256": support_hash,
        "endpoint_fusion": "FP32 two-dimensional Gaussian by master ID",
        "global_update": "sampler._xstart_to_pred(frozen, t_k, merged_endpoint); frozen-dt*v",
        "serial_tile_fallback": False,
    }
    _atomic_json(stage_dir / "flow_summary.json", result)
    _atomic_save(stage_dir / "final_state.pt", {"coords": master_coords.cpu(), "features": state})
    return state, result


def _pack_decoder_batch(group: Sequence[TileView], features: torch.Tensor, device: torch.device) -> SparseTensor:
    values = [SparseTensor(features.index_select(0, view.master_ids), view.local_coords) for view in group]
    return _pack_sparse_batch(values, "decoder input").to(device)


def _master_index_coords(count: int) -> torch.Tensor:
    """Make unique metadata coordinates for the non-lattice master table.

    These coordinates are never used as geometric identity or as a regular
    global SLat.  They only let SparseTensor arithmetic retain row alignment;
    the actual identity remains the stable ``master_id`` row.
    """
    if count < 0 or count >= 1024 ** 3:
        raise ValueError("master table is too large for metadata coordinates")
    ids = torch.arange(int(count), dtype=torch.int64)
    x = torch.div(ids, 1024 * 1024, rounding_mode="floor")
    y = torch.div(ids, 1024, rounding_mode="floor") % 1024
    z = ids % 1024
    return torch.stack((x, y, z), dim=1).to(torch.int32)


def _owner_for_uv(uv: torch.Tensor, active_tile_ids: Sequence[int], sigma: float = SIGMA_PIXELS) -> torch.Tensor:
    owner = torch.full((uv.shape[0],), -1, dtype=torch.int64, device=uv.device)
    best = torch.full((uv.shape[0],), -float("inf"), dtype=torch.float32, device=uv.device)
    boxes = _tile_layout()
    for tile_id in active_tile_ids:
        inside = _inside_box(uv, boxes[tile_id])
        weights = gaussian_weights(uv, boxes[tile_id], sigma)
        replace = inside & (weights > best)
        owner[replace] = int(tile_id)
        best[replace] = weights[replace]
    return owner


@torch.no_grad()
def _decode_and_merge(
    *,
    pipeline: Any,
    support: MasterSupport,
    shape_features: torch.Tensor,
    texture_features: torch.Tensor,
    camera: Mapping[str, float],
    output_dir: Path,
    device: torch.device,
    decode_batch_size: int = DECODE_BATCH_SIZE,
) -> Dict[str, Any]:
    """Decode local views in B=12 batches and merge faces in image space."""
    final_dir = output_dir / "final"
    cached_vertex_path = final_dir / "final_per_vertex_pbr_mesh.pt"
    cached_face_path = final_dir / "final_per_face_pbr_mesh.pt"
    if cached_vertex_path.is_file() and cached_face_path.is_file():
        vertex_payload = torch.load(cached_vertex_path, map_location="cpu", weights_only=False)
        face_payload = torch.load(cached_face_path, map_location="cpu", weights_only=False)
        vertex_mesh = vertex_payload.get("mesh", vertex_payload)
        face_mesh = face_payload.get("mesh", face_payload)
        if isinstance(vertex_mesh, MeshWithVertexPbr) and isinstance(face_mesh, MeshWithFacePbr):
            ownership_path = final_dir / "face_ownership.json"
            ownership = json.loads(ownership_path.read_text(encoding="utf-8")) if ownership_path.is_file() else {}
            return {
                "vertex_mesh": vertex_mesh,
                "face_mesh": face_mesh,
                "decoded_tiles": ownership.get("decoded_tiles", []),
                "vertices": int(vertex_mesh.vertices.shape[0]),
                "faces": int(vertex_mesh.faces.shape[0]),
                "cache_hit": True,
            }
    active = [support.tile_views[key] for key in sorted(support.tile_views)]
    vertex_parts: List[torch.Tensor] = []
    face_parts: List[torch.Tensor] = []
    vertex_attr_parts: List[torch.Tensor] = []
    face_attr_parts: List[torch.Tensor] = []
    owner_parts: List[torch.Tensor] = []
    decoded_rows: List[Dict[str, Any]] = []
    pipeline.low_vram = True
    for start in range(0, len(active), int(decode_batch_size)):
        group = active[start:start + int(decode_batch_size)]
        shape_batch = _pack_decoder_batch(group, _denormalize_features(shape_features, pipeline.shape_slat_normalization), device)
        texture_batch = _pack_decoder_batch(group, _denormalize_features(texture_features, pipeline.tex_slat_normalization), device)
        decoded = pipeline.decode_latent(shape_batch, texture_batch, LOCAL_OVOXEL)
        if len(decoded) != len(group):
            raise RuntimeError(f"decoder returned {len(decoded)} meshes for B={len(group)}")
        for view, mesh in zip(group, decoded):
            if not isinstance(mesh, MeshWithVoxel):
                raise RuntimeError(f"tile {view.tile_id}: decoder returned {type(mesh)!r}")
            mesh = mesh.to(device)
            q_local = mesh.vertices * (2.0 * float(view.transform.mesh_scale))
            q_global, _ = core._local_q_to_global_q(q_local, global_camera=camera, transform=view.transform)
            vertices_global = q_global / (2.0 * float(camera["mesh_scale"]))
            faces = mesh.faces.to(torch.long)
            face_local = mesh.vertices.index_select(0, faces.reshape(-1)).reshape(-1, 3, 3).mean(dim=1)
            face_q_local = face_local * (2.0 * float(view.transform.mesh_scale))
            face_q_global, face_uv = core._local_q_to_global_q(face_q_local, global_camera=camera, transform=view.transform)
            global_owner = _owner_for_uv(face_uv, sorted(support.tile_views))
            keep = global_owner == int(view.tile_id)
            keep_rows = torch.where(keep)[0]
            if keep_rows.numel() == 0:
                decoded_rows.append({"tile_id": view.tile_id, "decoded_vertices": int(vertices_global.shape[0]), "decoded_faces": int(faces.shape[0]), "owned_faces": 0})
                del mesh
                continue
            selected_faces = faces.index_select(0, keep_rows)
            used, inverse = torch.unique(selected_faces.reshape(-1), sorted=True, return_inverse=True)
            compact_vertices = vertices_global.index_select(0, used)
            compact_faces = inverse.reshape(-1, 3).to(torch.int32)
            vertex_attrs = mesh.query_vertex_attrs().index_select(0, used).detach().cpu().float()
            selected_face_local = face_local.index_select(0, keep_rows)
            face_attrs = mesh.query_attrs(selected_face_local).detach().cpu().float()
            offset = sum(int(v.shape[0]) for v in vertex_parts)
            vertex_parts.append(compact_vertices.detach().cpu().float())
            face_parts.append(compact_faces.cpu() + offset)
            vertex_attr_parts.append(vertex_attrs)
            face_attr_parts.append(face_attrs)
            owner_parts.append(torch.full((compact_faces.shape[0],), int(view.tile_id), dtype=torch.int16))
            decoded_rows.append({"tile_id": view.tile_id, "decoded_vertices": int(vertices_global.shape[0]), "decoded_faces": int(faces.shape[0]), "owned_faces": int(compact_faces.shape[0])})
            del mesh, q_local, q_global, vertices_global, faces, selected_faces, used, inverse, compact_vertices, compact_faces, vertex_attrs, face_attrs
        del shape_batch, texture_batch, decoded
        _empty_cuda_cache()
    if not face_parts:
        raise RuntimeError("all decoded tile faces were removed by image-space face ownership")
    vertices = torch.cat(vertex_parts, dim=0).contiguous()
    faces = torch.cat(face_parts, dim=0).contiguous()
    vertex_attrs = torch.cat(vertex_attr_parts, dim=0).contiguous()
    face_attrs = torch.cat(face_attr_parts, dim=0).contiguous()
    vertex_mesh = MeshWithVertexPbr(vertices, faces, vertex_attrs, layout=dict(PBR_LAYOUT))
    face_mesh = MeshWithFacePbr(vertices, faces, face_attrs, layout=dict(PBR_LAYOUT))
    _atomic_save(final_dir / "unwelded_tile_patches.pt", {"vertices": vertices, "faces": faces, "vertex_attrs": vertex_attrs, "face_attrs": face_attrs, "face_owner_tile_id": torch.cat(owner_parts)})
    _atomic_save(final_dir / "final_per_vertex_pbr_mesh.pt", {"format": FORMAT, "mesh": vertex_mesh})
    _atomic_save(final_dir / "final_per_face_pbr_mesh.pt", {"format": FORMAT, "mesh": face_mesh})
    _atomic_json(final_dir / "face_ownership.json", {"rule": "max 2D Gaussian weight among active tile boxes covering face centroid", "decoded_tiles": decoded_rows, "vertices": int(vertices.shape[0]), "faces": int(faces.shape[0])})
    return {"vertex_mesh": vertex_mesh, "face_mesh": face_mesh, "decoded_tiles": decoded_rows, "vertices": int(vertices.shape[0]), "faces": int(faces.shape[0])}


def _to_image(array: torch.Tensor, path: Path, channels: Optional[int] = None) -> np.ndarray:
    value = array.detach().float().cpu()
    if value.ndim == 3 and value.shape[0] in (1, 3):
        value = value.permute(1, 2, 0)
    if value.ndim == 2:
        value = value[..., None]
    data = np.nan_to_num(value.numpy(), nan=0.0, posinf=1.0, neginf=0.0)
    data = np.clip(data, 0.0, 1.0)
    if channels == 1 or data.shape[-1] == 1:
        image = Image.fromarray((data[..., 0] * 255.0 + 0.5).astype(np.uint8), mode="L")
    else:
        image = Image.fromarray((data[..., :3] * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return data.astype(np.float32)


def _final_render_aliases(output_dir: Path, prefix: str) -> None:
    """Expose the Codex2 canonical final/ filenames alongside prefixed files."""
    if prefix != "final":
        return
    aliases = {
        "final_render_rgb_4096.png": "render_rgb_4096.png",
        "final_render_alpha_4096.png": "render_alpha_4096.png",
        "final_render_normal_camera_4096.png": "render_normal_camera_4096.png",
        "final_render_normal_world_4096.png": "render_normal_world_4096.png",
        "final_pbr_base_color_4096.png": "pbr_base_color_4096.png",
        "final_pbr_metallic_4096.png": "pbr_metallic_4096.png",
        "final_pbr_roughness_4096.png": "pbr_roughness_4096.png",
        "final_pbr_alpha_4096.png": "pbr_alpha_4096.png",
        "final_depth_4096.pt": "depth_4096.pt",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for source_name, alias_name in aliases.items():
        source = output_dir / source_name
        alias = output_dir / alias_name
        if source.is_file() and not alias.is_file():
            shutil.copyfile(source, alias)


def _render_one(
    mesh: Any,
    camera: Mapping[str, float],
    output_dir: Path,
    device: torch.device,
    prefix: str,
    resolution: int,
    reference: Optional[np.ndarray] = None,
    foreground: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    cached_rgb = output_dir / f"{prefix}_render_rgb_4096.png"
    cached_alpha = output_dir / f"{prefix}_render_alpha_4096.png"
    cached_depth = output_dir / f"{prefix}_depth_4096.pt"
    if cached_rgb.is_file() and cached_alpha.is_file() and cached_depth.is_file():
        _final_render_aliases(output_dir, prefix)
        return {
            "resolution": int(resolution),
            "metrics": {},
            "rgb_path": str(cached_rgb),
            "alpha_path": str(cached_alpha),
            "depth_path": str(cached_depth),
            "cache_hit": True,
        }
    from pixal3d.renderers import PbrMeshRenderer
    from pixal3d.renderers import MeshRenderer
    import pixal3d_baseline1024_pbr_mesh_compare as baseline_render
    from render_pixal3d_raw_ovoxel import load_envmap

    extrinsics, intrinsics, _ = baseline_render._make_camera_views(float(camera["camera_angle_x"]), float(camera["distance"]), (0,))
    renderer = PbrMeshRenderer(
        rendering_options={"resolution": int(resolution), "near": max(0.01, float(camera["distance"]) - 2.0), "far": float(camera["distance"]) + 10.0, "ssaa": 1, "peel_layers": 8, "face_chunk_size": 4_000_000},
        device=str(device),
    )
    envmap = load_envmap("studio", device=device)
    live = mesh.to(device)
    result = renderer.render(live, extrinsics[0].to(device), intrinsics.to(device), envmap=envmap, use_envmap_bg=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: Dict[str, np.ndarray] = {}
    rendered["rgb"] = _to_image(result["shaded"], output_dir / f"{prefix}_render_rgb_4096.png")
    _to_image(result["mask"], output_dir / f"{prefix}_render_alpha_4096.png", channels=1)
    normal_camera = _to_image(result["normal"], output_dir / f"{prefix}_render_normal_camera_4096.png")
    # PbrMeshRenderer exposes the encoded camera-space normal.  Convert it
    # back through the exact world->camera rotation for the world-normal map.
    encoded = result["normal"].detach().float().to(device)
    normal_cam = -(encoded * 2.0 - 1.0)
    rotation = extrinsics[0].to(device)[:3, :3]
    normal_world = torch.matmul(rotation.transpose(0, 1), normal_cam.reshape(3, -1)).reshape_as(normal_cam)
    normal_world = -normal_world * 0.5 + 0.5
    _to_image(normal_world, output_dir / f"{prefix}_render_normal_world_4096.png")
    _to_image(result["base_color"], output_dir / f"{prefix}_pbr_base_color_4096.png")
    _to_image(result["metallic"], output_dir / f"{prefix}_pbr_metallic_4096.png", channels=1)
    _to_image(result["roughness"], output_dir / f"{prefix}_pbr_roughness_4096.png", channels=1)
    _to_image(result["alpha"], output_dir / f"{prefix}_pbr_alpha_4096.png", channels=1)
    # PbrMeshRenderer intentionally does not expose its internal z buffer.
    # Run the project's native mesh rasterizer once more for a true lossless
    # camera-space depth map; the mask fallback used by older experiments is
    # not a depth image and is therefore forbidden here.
    del result, envmap, renderer
    _empty_cuda_cache()
    depth_renderer = MeshRenderer(
        rendering_options={
            "resolution": int(resolution),
            "near": max(0.01, float(camera["distance"]) - 2.0),
            "far": float(camera["distance"]) + 10.0,
            "ssaa": 1,
            "chunk_size": 4_000_000,
            "antialias": False,
        },
        device=str(device),
    )
    depth_result = depth_renderer.render(
        live,
        extrinsics[0].to(device),
        intrinsics.to(device),
        return_types=["depth", "mask"],
    )
    depth = (-depth_result["depth"].detach().float()).to(torch.float32)
    depth = torch.where(depth_result["mask"].detach() > 0, depth, torch.zeros_like(depth))
    depth_path = output_dir / f"{prefix}_depth_4096.pt"
    _atomic_save(depth_path, {"depth_camera_positive": depth.cpu(), "resolution": [int(resolution), int(resolution)], "background": 0.0})
    _final_render_aliases(output_dir, prefix)
    del depth_result, depth_renderer, live
    _empty_cuda_cache()
    metrics = {}
    if reference is not None and foreground is not None:
        diff = rendered["rgb"] - reference
        mask = foreground > 0.5
        mse = float(np.square(diff).mean())
        fg_mse = float(np.square(diff[mask]).mean()) if bool(mask.any()) else float("nan")
        metrics = {"full_psnr_db": float(10.0 * math.log10(1.0 / max(mse, 1e-12))), "foreground_psnr_db": float(10.0 * math.log10(1.0 / max(fg_mse, 1e-12))) if math.isfinite(fg_mse) else None, "full_mae": float(np.abs(diff).mean())}
    return {
        "resolution": int(resolution),
        "metrics": metrics,
        "rgb_path": str(output_dir / f"{prefix}_render_rgb_4096.png"),
        "alpha_path": str(output_dir / f"{prefix}_render_alpha_4096.png"),
        "depth_path": str(depth_path),
        "normal_camera_range": [float(normal_camera.min()), float(normal_camera.max())],
    }


def _load_or_make_baseline(
    args: argparse.Namespace,
    pipeline: Any,
    image_1024: Image.Image,
    camera: Mapping[str, float],
    output_dir: Path,
) -> Tuple[MeshWithVoxel, Dict[str, Any]]:
    candidates = [
        Path(args.baseline_dir) / "raw_ovoxel_mesh.pt",
        Path(args.baseline_dir) / "global_baseline_mesh.pt",
        Path(args.baseline_dir) / "baseline_raw_ovoxel_mesh.pt",
    ]
    for path in candidates:
        if path.is_file():
            mesh = _load_mesh(path)
            _atomic_save(output_dir / "baseline" / "global_baseline_mesh.pt", {"format": FORMAT, "mesh": mesh})
            return mesh, {"source": str(path.resolve()), "generated_once": False, "vertices": int(mesh.vertices.shape[0]), "faces": int(mesh.faces.shape[0]), "active_ovoxels": int(mesh.coords.shape[0])}
    _seed(int(args.seed))
    sampler_shape = {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0}
    sampler_tex = {"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0}
    meshes, _ = pipeline.run(
        image_1024,
        camera_params=dict(camera),
        seed=int(args.seed),
        shape_slat_sampler_params=sampler_shape,
        tex_slat_sampler_params=sampler_tex,
        preprocess_image=False,
        return_latent=True,
        pipeline_type="1024_cascade",
        max_num_tokens=1_000_000,
    )
    if len(meshes) != 1 or not isinstance(meshes[0], MeshWithVoxel):
        raise RuntimeError("official global baseline did not return one MeshWithVoxel")
    mesh = meshes[0].cpu()
    _atomic_save(output_dir / "baseline" / "global_baseline_mesh.pt", {"format": FORMAT, "mesh": mesh})
    return mesh, {"source": "pipeline.run(1024_cascade)", "generated_once": True, "vertices": int(mesh.vertices.shape[0]), "faces": int(mesh.faces.shape[0]), "active_ovoxels": int(mesh.coords.shape[0])}


def _baseline_vertex_mesh(baseline: MeshWithVoxel, device: torch.device, cache_path: Path) -> MeshWithVertexPbr:
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        mesh = payload.get("mesh", payload) if isinstance(payload, Mapping) else payload
        if isinstance(mesh, MeshWithVertexPbr):
            return mesh
    live = baseline.to(device)
    attrs = live.query_vertex_attrs().detach().cpu().float()
    mesh = MeshWithVertexPbr(baseline.vertices.cpu(), baseline.faces.cpu(), attrs, layout=dict(PBR_LAYOUT))
    _atomic_save(cache_path, {"format": FORMAT, "mesh": mesh})
    del live
    _empty_cuda_cache()
    return mesh


def _lpips_native_patches(reference: np.ndarray, prediction: np.ndarray, patch_size: int = 512) -> Optional[float]:
    """Compute AlexNet LPIPS on native-resolution patches without upsampling."""
    try:
        import lpips
        metric = lpips.LPIPS(net="alex", verbose=False).eval()
    except Exception:
        return None
    values: List[float] = []
    with torch.no_grad():
        for y in range(0, reference.shape[0], patch_size):
            for x in range(0, reference.shape[1], patch_size):
                ref_patch = torch.from_numpy(reference[y:y + patch_size, x:x + patch_size]).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
                pred_patch = torch.from_numpy(prediction[y:y + patch_size, x:x + patch_size]).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
                values.append(float(metric(pred_patch, ref_patch).mean()))
    del metric
    return float(np.mean(values)) if values else None


def _compute_global_metrics(reference: Image.Image, mask: Image.Image, output_dir: Path, renders: Mapping[str, Mapping[str, Any]]) -> None:
    from scipy.ndimage import binary_dilation, binary_erosion
    from skimage.metrics import structural_similarity

    rows = []
    ref = np.asarray(reference.convert("RGB"), dtype=np.float32) / 255.0
    fg = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    fg_bool = fg > 0.5
    boundary = binary_dilation(fg_bool, iterations=8) ^ binary_erosion(fg_bool, iterations=8)
    if not np.any(boundary):
        boundary = fg_bool
    for name, item in renders.items():
        rgb_path = Path(item.get("rgb_path", output_dir / f"{name}_render_rgb_4096.png"))
        if not rgb_path.is_file():
            continue
        pred = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
        diff = pred - ref
        full_mse = float(np.mean(diff * diff))
        fg_mse = float(np.mean((diff[fg_bool]) ** 2)) if np.any(fg_bool) else float("nan")
        boundary_mse = float(np.mean((diff[boundary]) ** 2)) if np.any(boundary) else float("nan")
        _, ssim_map = structural_similarity(ref, pred, data_range=1.0, channel_axis=2, full=True)
        alpha_path = Path(item.get("alpha_path", output_dir / f"{name}_render_alpha_4096.png"))
        if alpha_path.is_file():
            pred_alpha = np.asarray(Image.open(alpha_path).convert("L"), dtype=np.float32) / 255.0
            pred_alpha_bool = pred_alpha > 0.5
            intersection = np.logical_and(pred_alpha_bool, fg_bool).sum()
            union = np.logical_or(pred_alpha_bool, fg_bool).sum()
            alpha_iou = float(intersection / max(1, union))
        else:
            alpha_iou = None
        rows.append({
            "variant": name,
            "resolution": [pred.shape[1], pred.shape[0]],
            "psnr_db": 10.0 * math.log10(1.0 / max(full_mse, 1e-12)),
            "foreground_psnr_db": 10.0 * math.log10(1.0 / max(fg_mse, 1e-12)) if math.isfinite(fg_mse) else None,
            "boundary_band_psnr_db": 10.0 * math.log10(1.0 / max(boundary_mse, 1e-12)) if math.isfinite(boundary_mse) else None,
            "ssim": float(np.mean(ssim_map)),
            "foreground_ssim": float(np.mean(ssim_map[fg_bool])) if np.any(fg_bool) else None,
            "boundary_band_ssim": float(np.mean(ssim_map[boundary])) if np.any(boundary) else None,
            "alpha_iou": alpha_iou,
            "lpips_alex_native_512_patch": _lpips_native_patches(ref, pred),
        })
    _atomic_json(output_dir / "metrics_4096.json", {
        "reference": "canonical_foreground_rgb_4096",
        "resolution_required": [4096, 4096],
        "rows": rows,
        "boundary_band": "8-pixel dilation XOR erosion of the 4096 foreground mask",
        "lpips": "AlexNet LPIPS averaged over native 512x512 patches; no resize/upscale",
    })


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if (CANONICAL_SIZE, TILE_SIZE, TILE_STRIDE, LOCAL_OVOXEL, LATENT_SIZE) != (4096, 1024, 512, 1024, 64):
        raise AssertionError("Codex2 fixed resolution configuration changed")
    if args.encode_batch_size != ENCODE_BATCH_SIZE or args.flow_batch_size != FLOW_BATCH_SIZE or args.decode_batch_size != DECODE_BATCH_SIZE:
        raise ValueError("main experiment uses the reference batch profile encode=13, flow=44, decode=12")
    if args.native_geometry_encode_batch_size <= 0:
        raise ValueError("native geometry encoder batch size must be positive")
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    # With CUDA_VISIBLE_DEVICES=5 the only visible device is logical cuda:0.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.split(",")[0].strip() != str(args.cuda_device):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r} does not expose requested physical CUDA {args.cuda_device}")
    if visible is not None and len([v for v in visible.split(",") if v.strip()]) != 1:
        raise RuntimeError("main run requires exactly one visible GPU so physical CUDA5 is logical cuda:0")
    if visible is None:
        if args.cuda_device >= torch.cuda.device_count():
            raise RuntimeError(f"physical CUDA {args.cuda_device} is unavailable")
        torch.cuda.set_device(args.cuda_device)
        device = torch.device("cuda", args.cuda_device)
    else:
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = _runtime_metadata(args.cuda_device, device)
    _seed(args.seed)

    # Pipeline is initialized once.  Its canonical preprocess is the sole
    # background-removal/foreground-mask operation in this experiment.
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=True)
    source_image = Image.open(args.image)
    canonical = pipeline.preprocess_canonical_images(source_image)
    preprocess_meta = _save_canonical_images(canonical, output_dir)
    camera = _load_camera(args.baseline_dir)
    _atomic_json(output_dir / "baseline" / "global_camera.json", camera)
    _atomic_json(output_dir / "global_camera.json", camera)
    baseline, baseline_meta = _load_or_make_baseline(args, pipeline, canonical["image_1024"], camera, output_dir)
    transforms = {
        tile_id: core._derive_tile_camera(
            tile_id=tile_id,
            box=box,
            global_camera=camera,
            extend_pixel=0,
            source_width=CANONICAL_SIZE,
            source_height=CANONICAL_SIZE,
            model_width=TILE_SIZE,
            model_height=TILE_SIZE,
        )
        for tile_id, box in enumerate(_tile_layout())
    }
    _atomic_json(output_dir / "tile_cameras.json", {str(k): transforms[k].__dict__ for k in transforms})

    face_bounds = _face_bounds(baseline, camera, output_dir / "face_projection_bounds.pt", args.face_projection_chunk_size)
    tile_images: Dict[int, Image.Image] = {}
    shape_tile_images: Dict[int, Image.Image] = {}
    (output_dir / "inputs" / "tiles_1024").mkdir(parents=True, exist_ok=True)
    (output_dir / "inputs" / "tiles_512").mkdir(parents=True, exist_ok=True)
    for tile_id, box in enumerate(_tile_layout()):
        crop = canonical["image_4096"].crop(box).convert("RGB")
        tile_images[tile_id] = crop
        shape_tile_images[tile_id] = crop.resize((SHAPE_IMAGE_SIZE, SHAPE_IMAGE_SIZE), Image.Resampling.LANCZOS)
        tile_images[tile_id].save(output_dir / "inputs" / "tiles_1024" / f"tile_{tile_id:02d}.png")
        shape_tile_images[tile_id].save(output_dir / "inputs" / "tiles_512" / f"tile_{tile_id:02d}.png")

    geometries = _prepare_native_geometries(
        baseline, camera, transforms, face_bounds, output_dir,
        list(range(TILE_COUNT)), args.face_projection_chunk_size,
    )
    native_coords = _encode_native_supports(
        geometries, args.shape_encoder, output_dir, device, args.native_geometry_encode_batch_size,
    )
    support = _build_master_support(native_coords, transforms, camera, sigma_pixels=SIGMA_PIXELS)
    _save_master_support(support, output_dir, transforms)
    master_coords = torch.cat(
        (
            torch.zeros((support.master_q_global.shape[0], 1), dtype=torch.int32),
            _master_index_coords(support.master_q_global.shape[0]),
        ),
        dim=1,
    )
    # master_coords is only a stable metadata index for the state table; flow
    # model calls always use each tile's exact local view coordinates.  It is
    # intentionally not a regular global C64/C128 lattice.

    shape_conditions = _build_batched_image_conditions(
        pipeline, pipeline.image_cond_model_shape_1024, support.tile_views,
        shape_tile_images, output_dir, "shape", camera, device, args.encode_batch_size,
    )
    texture_conditions = _build_batched_image_conditions(
        pipeline, pipeline.image_cond_model_tex_1024, support.tile_views,
        tile_images, output_dir, "texture", camera, device, TEXTURE_CONDITION_BATCH_SIZE,
    )

    shape_model = pipeline.models["shape_slat_flow_model_1024"]
    texture_model = pipeline.models["tex_slat_flow_model_1024"]
    shape_channels = int(shape_model.in_channels)
    texture_channels = int(texture_model.in_channels - shape_channels)
    shape_noise = torch.randn((support.master_q_global.shape[0], shape_channels), generator=torch.Generator(device="cpu").manual_seed(int(args.shape_seed)), dtype=torch.float32)
    texture_noise = torch.randn((support.master_q_global.shape[0], texture_channels), generator=torch.Generator(device="cpu").manual_seed(int(args.texture_seed)), dtype=torch.float32)
    _atomic_save(output_dir / "shape" / "shape_noise.pt", {"master_ids": torch.arange(shape_noise.shape[0]), "features": shape_noise})
    _atomic_save(output_dir / "texture" / "texture_noise.pt", {"master_ids": torch.arange(texture_noise.shape[0]), "features": texture_noise})

    shape_params = dict(pipeline.shape_slat_sampler_params)
    texture_params = dict(pipeline.tex_slat_sampler_params)
    if args.steps is not None:
        shape_params["steps"] = int(args.steps)
        texture_params["steps"] = int(args.steps)
    shape_final, shape_summary = _run_synchronized_endpoint_flow(
        stage="shape", initial_features=shape_noise, master_coords=master_coords,
        views=support.tile_views, conditions=shape_conditions,
        sampler=pipeline.shape_slat_sampler, model=shape_model, sampler_params=shape_params,
        output_dir=output_dir, device=device, flow_batch_size=args.flow_batch_size,
        resume=args.resume, save_step_tensors=args.save_step_tensors,
    )
    texture_final, texture_summary = _run_synchronized_endpoint_flow(
        stage="texture", initial_features=texture_noise, master_coords=master_coords,
        views=support.tile_views, conditions=texture_conditions,
        sampler=pipeline.tex_slat_sampler, model=texture_model, sampler_params=texture_params,
        output_dir=output_dir, device=device, flow_batch_size=args.flow_batch_size,
        concat_features=shape_final, resume=args.resume, save_step_tensors=args.save_step_tensors,
    )

    final = _decode_and_merge(
        pipeline=pipeline, support=support, shape_features=shape_final, texture_features=texture_final,
        camera=camera, output_dir=output_dir, device=device, decode_batch_size=args.decode_batch_size,
    )

    # Path A and Path B both render directly at native 4096 when requested.
    render_summary: Dict[str, Any] = {}
    if args.render:
        reference = np.asarray(canonical["image_4096"].convert("RGB"), dtype=np.float32) / 255.0
        foreground = np.asarray(canonical["foreground_mask_4096"].convert("L"), dtype=np.float32) / 255.0
        baseline_vertex = _baseline_vertex_mesh(baseline, device, output_dir / "baseline" / "baseline_per_vertex_pbr_mesh.pt")
        render_summary["B0_global_baseline"] = _render_one(baseline_vertex, camera, output_dir / "baseline", device, "baseline", CANONICAL_SIZE, reference, foreground)
        render_summary["E1_global_master"] = _render_one(final["vertex_mesh"], camera, output_dir / "final", device, "final", CANONICAL_SIZE, reference, foreground)
        _compute_global_metrics(canonical["image_4096"], canonical["foreground_mask_4096"], output_dir, {"baseline": render_summary["B0_global_baseline"], "final": render_summary["E1_global_master"]})

    summary = {
        "format": FORMAT,
        "status": "complete",
        "input": str(args.image.resolve()),
        "camera": camera,
        "runtime": runtime,
        "seed": {"global": args.seed, "shape": args.shape_seed, "texture": args.texture_seed},
        "git": _git_metadata(Path(__file__).resolve().parent),
        "preprocess": preprocess_meta,
        "baseline": baseline_meta,
        "layout": {"canonical": CANONICAL_SIZE, "global": GLOBAL_SIZE, "shape": SHAPE_IMAGE_SIZE, "tile": TILE_SIZE, "stride": TILE_STRIDE, "tile_count": TILE_COUNT, "tile_order": "row-major", "rectangle": "half-open"},
        "batch_profile": {
            "encode": args.encode_batch_size,
            "image_condition_encode": args.encode_batch_size,
            "texture_condition_encode": TEXTURE_CONDITION_BATCH_SIZE,
            "native_geometry_encode": args.native_geometry_encode_batch_size,
            "flow": args.flow_batch_size,
            "decode": args.decode_batch_size,
            "model_input": "[B,...] real sparse batch; one tile per B item",
            "reference_mapping": "flow=44, decode=12, pbr/image-condition encode=13; initial local geometry encode=1",
        },
        "support": {"master_tokens": int(support.master_q_global.shape[0]), "active_tiles": sorted(support.tile_views), "inactive_tiles": [i for i in range(TILE_COUNT) if i not in support.tile_views], "roundtrip_max_abs_error": support.roundtrip_max_abs_error, "feature_values_used": False},
        "shape_flow": shape_summary,
        "texture_flow": texture_summary,
        "decode": {key: value for key, value in final.items() if key not in {"vertex_mesh", "face_mesh"}},
        "render": render_summary,
        "artifacts": {"support": str(output_dir / "support" / "master_support.pt"), "shape_flow": str(output_dir / "shape" / "flow_summary.json"), "texture_flow": str(output_dir / "texture" / "flow_summary.json"), "final_vertex_mesh": str(output_dir / "final" / "final_per_vertex_pbr_mesh.pt"), "final_face_mesh": str(output_dir / "final" / "final_per_face_pbr_mesh.pt"), "metrics_4096": str(output_dir / "metrics_4096.json")},
    }
    _atomic_json(output_dir / "config.json", {"format": FORMAT, "args": vars(args), "runtime": runtime, "batch_profile": summary["batch_profile"]})
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--shape-encoder", type=Path, default=DEFAULT_SHAPE_ENCODER)
    parser.add_argument("--cuda-device", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--shape-seed", type=int, default=20260823)
    parser.add_argument("--texture-seed", type=int, default=20260824)
    parser.add_argument("--encode-batch-size", type=int, default=ENCODE_BATCH_SIZE)
    parser.add_argument(
        "--native-geometry-encode-batch-size", type=int,
        default=NATIVE_GEOMETRY_ENCODE_BATCH_SIZE,
        help="memory-safe local C1024 geometry encoder B; reference default is 1",
    )
    parser.add_argument("--flow-batch-size", type=int, default=FLOW_BATCH_SIZE)
    parser.add_argument("--decode-batch-size", type=int, default=DECODE_BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=None, help="debug override; main quality run keeps official 12")
    parser.add_argument("--face-projection-chunk-size", type=int, default=250_000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-step-tensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run(args)
    print(f"[done] {summary['status']} output={Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
