#!/usr/bin/env python3
"""Merge decoder-native C1024 O-Voxel tiles into an empty global C4096 grid.

This is the isolated P0 path described in ``Codex.md``.  It deliberately
does not load a baseline O-Voxel.  Tile decoder raw tensors are the only
topology source; mesh/provenance is used only for native full-QEF statistics.

The script is usable with a manifest containing raw decoder files, and can
also discover the fixed 25-tile cache produced by the earlier local-only
experiment.  All output is written below the new
``geometry_ovoxel_local_only_topology_qef_merge`` directory by default.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


_OVOXEL_SOURCE = Path("/home/nvme04/yyyan/TRELLIS.2/o-voxel")
if _OVOXEL_SOURCE.is_dir():
    sys.path.insert(0, str(_OVOXEL_SOURCE))

from o_voxel.convert import (  # noqa: E402
    flexible_dual_grid_to_mesh,
    mesh_to_flexible_dual_grid,
    mesh_to_flexible_dual_grid_qef_stats,
)


GLOBAL_RESOLUTION = 4096
TILE_SIZE = 1024
TILE_STRIDE = 512
DEFAULT_OUTPUT = Path("outputs/geometry_ovoxel_local_only_topology_qef_merge")
DEFAULT_INPUT = Path("outputs/geometry_ovoxel_local_only_stride512")
RUNTIME_AABB = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32)

EDGE_CELL_OFFSETS = np.asarray(
    [
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ],
    dtype=np.int32,
)
AXIS_UNIT = np.eye(3, dtype=np.int32)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))


@dataclass(frozen=True)
class CudaInfo:
    requested_physical: int
    logical_device: int
    physical_device: int
    current_logical_device: int
    device_name: str
    visible_devices: Optional[Tuple[int, ...]]

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.logical_device)


def _visible_physical_devices() -> Optional[Tuple[int, ...]]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        return None
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token or token.startswith("GPU-") or token.startswith("MIG-"):
            # UUID mappings cannot be proved to be physical GPU 4 from this
            # process.  Refuse them instead of silently misreporting CUDA 4.
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES must use numeric physical IDs for the "
                "P0 report; UUID/MIG mappings are not auditable here"
            )
        values.append(int(token))
    return tuple(values)


def _resolve_cuda(cuda_device: int) -> CudaInfo:
    if not torch.cuda.is_available():
        raise RuntimeError("P0 GPU tests require CUDA")
    requested = int(cuda_device)
    visible = _visible_physical_devices()
    if visible is None:
        logical = requested
        physical = requested
    else:
        if requested not in visible:
            raise RuntimeError(
                f"requested physical cuda:{requested} is not in "
                f"CUDA_VISIBLE_DEVICES={visible}"
            )
        logical = visible.index(requested)
        physical = int(visible[logical])
    if logical < 0 or logical >= torch.cuda.device_count():
        raise RuntimeError(
            f"logical CUDA device {logical} is unavailable; "
            f"visible count={torch.cuda.device_count()}"
        )
    device = torch.device("cuda", logical)
    with torch.cuda.device(device):
        current = int(torch.cuda.current_device())
        name = torch.cuda.get_device_name(device)
    if physical != requested:
        raise AssertionError(f"physical CUDA mapping changed: {physical} != {requested}")
    return CudaInfo(requested, logical, physical, current, name, visible)


def _cell_keys(coords: np.ndarray, resolution: int = GLOBAL_RESOLUTION) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    if coords.size == 0:
        return np.empty((0,), dtype=np.int64)
    return (coords[:, 0] * int(resolution) + coords[:, 1]) * int(resolution) + coords[:, 2]


def _edge_keys(coords: np.ndarray, axis: np.ndarray, resolution: int = GLOBAL_RESOLUTION) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    axis = np.asarray(axis, dtype=np.int64)
    if coords.size == 0:
        return np.empty((0,), dtype=np.int64)
    return _cell_keys(coords, resolution) * 3 + axis


def _valid_cells(coords: np.ndarray, resolution: int = GLOBAL_RESOLUTION) -> np.ndarray:
    coords = np.asarray(coords)
    return ((coords >= 0) & (coords < int(resolution))).all(axis=1)


def _valid_edge_coords(
    coords: np.ndarray,
    axis: np.ndarray,
    resolution: int = GLOBAL_RESOLUTION,
) -> np.ndarray:
    """Validate the axis-specific primal-edge domain.

    The edge direction coordinate may be any cell coordinate.  The two
    perpendicular coordinates need one positive neighbor because an O-Voxel
    quad has four incident cells.
    """
    coords = np.asarray(coords, dtype=np.int64)
    axis = np.asarray(axis, dtype=np.int64)
    result = ((coords >= 0) & (coords < int(resolution))).all(axis=1)
    for a in range(3):
        # For a row whose edge axis is not ``a``, coordinate ``a`` is a
        # perpendicular origin and therefore needs a positive neighbor.
        result &= (axis == a) | (coords[:, a] < int(resolution) - 1)
    return result


def _edge_cells(edge_coord: np.ndarray, edge_axis: np.ndarray) -> np.ndarray:
    edge_coord = np.asarray(edge_coord, dtype=np.int32)
    edge_axis = np.asarray(edge_axis, dtype=np.int64)
    if edge_coord.size == 0:
        return np.empty((0, 4, 3), dtype=np.int32)
    return edge_coord[:, None, :] + EDGE_CELL_OFFSETS[edge_axis]


def _raised_cosine_weight(margin: np.ndarray, band: float) -> np.ndarray:
    if band <= 0:
        return np.ones(np.asarray(margin).shape, dtype=np.float32)
    t = np.clip(np.asarray(margin, dtype=np.float32) / float(band), 0.0, 1.0)
    return (0.5 * (1.0 - np.cos(np.pi * t))).astype(np.float32)


def _cell_boundary_weight(coords: np.ndarray, tile_size: int, band: float) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int32)
    margin = np.minimum(coords, int(tile_size) - 1 - coords).min(axis=1)
    return _raised_cosine_weight(margin, band)


def _edge_boundary_weight(edge_coord: np.ndarray, tile_size: int, band: float) -> np.ndarray:
    edge_coord = np.asarray(edge_coord, dtype=np.int32)
    if edge_coord.size == 0:
        return np.empty((0,), dtype=np.float32)
    margin = np.minimum(edge_coord, int(tile_size) - 1 - edge_coord).min(axis=1)
    return _raised_cosine_weight(margin, band)


def _tile_starts(resolution: int, tile_size: int, stride: int) -> List[int]:
    if resolution != GLOBAL_RESOLUTION or tile_size != TILE_SIZE or stride != TILE_STRIDE:
        raise ValueError("P0 uses global C4096, tile C1024, and stride 512")
    starts = list(range(0, resolution - tile_size + 1, stride))
    if starts[-1] != resolution - tile_size:
        raise ValueError("tile layout does not land on the global upper edge")
    return starts


def _tile_layout(resolution: int, tile_size: int, stride: int) -> List[Tuple[int, int, int]]:
    starts = _tile_starts(resolution, tile_size, stride)
    return [(x, y, z) for z in starts for y in starts for x in starts]


def _parse_ids(value: str) -> Optional[set[int]]:
    if value is None or not value.strip():
        return None
    result: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            result.update(range(int(left), int(right) + 1))
        else:
            result.add(int(token))
    return result


def _resolve_path(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _discover_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    input_dir = args.input_dir.expanduser().resolve()
    layout_path = input_dir / "tile_layout.json"
    config_path = input_dir / "config.json"
    layout_payload = json.loads(layout_path.read_text()) if layout_path.is_file() else {}
    config_payload = json.loads(config_path.read_text()) if config_path.is_file() else {}
    layout = layout_payload.get("selected_starts", [])
    selected_ids = layout_payload.get("selected_tile_ids", [])
    if not layout or len(layout) != len(selected_ids):
        all_layout = _tile_layout(GLOBAL_RESOLUTION, TILE_SIZE, TILE_STRIDE)
        selected_ids = list(range(len(all_layout)))
        layout = [list(origin) for origin in all_layout]
    requested = _parse_ids(args.tile_ids)
    if requested is None:
        configured = config_payload.get("tile_ids")
        requested = _parse_ids(str(configured)) if configured else None
    if requested is not None:
        keep = [i for i, tile_id in enumerate(selected_ids) if int(tile_id) in requested]
    else:
        keep = list(range(len(selected_ids)))
    if args.max_tiles is not None:
        keep = keep[: int(args.max_tiles)]
    tiles: List[Dict[str, Any]] = []
    for index in keep:
        tile_id = int(selected_ids[index])
        origin = [int(v) for v in layout[index]]
        raw_path = input_dir / "local_tiles" / f"tile_{tile_id:03d}" / "shape_flow_and_raw_ovoxel.pt"
        if not raw_path.is_file():
            raise FileNotFoundError(f"tile {tile_id} raw decoder output is missing: {raw_path}")
        tiles.append({
            "tile_id": tile_id,
            "origin": origin,
            "size": TILE_SIZE,
            "stride": TILE_STRIDE,
            "raw_ovoxel": str(raw_path.resolve()),
            "mesh_provenance": "raw_ovoxel.mesh_faces + raw_ovoxel.provenance",
            "valid_region": [origin, [v + TILE_SIZE for v in origin]],
            "halo_distance": int(max(TILE_STRIDE // 2, 1)),
            "boundary_band": float(args.boundary_band),
            "contribution_weight": 1.0,
        })
    return {
        "format": "pixal3d_ovoxel_local_only_tile_manifest_v1",
        "global_resolution": GLOBAL_RESOLUTION,
        "tile_size": TILE_SIZE,
        "tile_stride": TILE_STRIDE,
        "source_directory": str(input_dir),
        "tiles": tiles,
    }


def _load_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    if args.tile_manifest is None:
        manifest = _discover_manifest(args)
        base = args.input_dir.expanduser().resolve()
    else:
        path = args.tile_manifest.expanduser().resolve()
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            payload = {"tiles": payload}
        manifest = dict(payload)
        base = path.parent
    if int(manifest.get("global_resolution", GLOBAL_RESOLUTION)) != GLOBAL_RESOLUTION:
        raise ValueError("manifest global_resolution must be 4096")
    if int(manifest.get("tile_size", TILE_SIZE)) != TILE_SIZE:
        raise ValueError("manifest tile_size must be 1024")
    if int(manifest.get("tile_stride", TILE_STRIDE)) != TILE_STRIDE:
        raise ValueError("manifest tile_stride must be 512")
    tiles = []
    seen_ids: set[int] = set()
    for raw_tile in manifest.get("tiles", []):
        tile = dict(raw_tile)
        tile_id = int(tile["tile_id"])
        if tile_id in seen_ids:
            raise ValueError(f"duplicate tile_id {tile_id}")
        seen_ids.add(tile_id)
        origin = np.asarray(tile["origin"], dtype=np.int64)
        if origin.shape != (3,) or not bool(((origin >= 0) & (origin + TILE_SIZE <= GLOBAL_RESOLUTION)).all()):
            raise ValueError(f"tile {tile_id} origin is outside C4096: {origin.tolist()}")
        tile["tile_id"] = tile_id
        tile["origin"] = origin.astype(np.int32).tolist()
        tile["size"] = int(tile.get("size", TILE_SIZE))
        tile["stride"] = int(tile.get("stride", TILE_STRIDE))
        tile["raw_ovoxel"] = str(_resolve_path(tile.get("raw_ovoxel", tile.get("raw")), base))
        if not Path(tile["raw_ovoxel"]).is_file():
            raise FileNotFoundError(tile["raw_ovoxel"])
        tile["contribution_weight"] = float(tile.get("contribution_weight", 1.0))
        tile["halo_distance"] = int(tile.get("halo_distance", TILE_STRIDE // 2))
        tile["boundary_band"] = float(tile.get("boundary_band", 0.15))
        tiles.append(tile)
    if not tiles:
        raise ValueError("tile manifest contains no tiles")
    manifest["tiles"] = tiles
    manifest["global_resolution"] = GLOBAL_RESOLUTION
    manifest["tile_size"] = TILE_SIZE
    manifest["tile_stride"] = TILE_STRIDE
    return manifest


def _load_raw_tile(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw = payload.get("raw_ovoxel", payload) if isinstance(payload, dict) else payload
    required = (
        "coords", "dual_vertices", "intersected", "quad_lerp",
        "mesh_vertices", "mesh_faces", "provenance",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"{path} raw decoder O-Voxel is missing {missing}")
    return raw


def _as_numpy(value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    return result.astype(dtype, copy=False) if dtype is not None else result


def _compute_cell_normals(raw: Mapping[str, Any], cell_count: int) -> np.ndarray:
    """Accumulate source-quad face normals for mode diagnostics."""
    vertices = _as_numpy(raw["mesh_vertices"], np.float32)
    faces = _as_numpy(raw["mesh_faces"], np.int64)
    provenance = raw["provenance"]
    source_index = _as_numpy(provenance["source_ovoxel_index"], np.int64)
    if faces.size == 0 or source_index.size == 0:
        return np.zeros((cell_count, 3), dtype=np.float32)
    if source_index.shape[0] != faces.shape[0]:
        raise ValueError("decoder provenance source_ovoxel_index must be per emitted triangle")
    normal_sum = np.zeros((cell_count, 3), dtype=np.float32)
    for start in range(0, faces.shape[0], 500_000):
        stop = min(start + 500_000, faces.shape[0])
        tri = vertices[faces[start:stop]]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(normals, axis=1, keepdims=True)
        valid = (length[:, 0] > 1e-12) & (source_index[start:stop] >= 0) & (source_index[start:stop] < cell_count)
        normals[valid] /= length[valid]
        np.add.at(normal_sum, source_index[start:stop][valid], normals[valid])
    length = np.linalg.norm(normal_sum, axis=1, keepdims=True)
    valid = length[:, 0] > 1e-8
    normal_sum[valid] /= length[valid]
    normal_sum[~valid] = 0.0
    return normal_sum


def _prepare_tile(
    tile: Mapping[str, Any],
    slot: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    tile_id = int(tile["tile_id"])
    origin = np.asarray(tile["origin"], dtype=np.int32)
    tile_size = int(tile.get("size", TILE_SIZE))
    if tile_size != TILE_SIZE:
        raise ValueError(f"tile {tile_id} must have size 1024")
    raw = _load_raw_tile(Path(tile["raw_ovoxel"]))
    coords = _as_numpy(raw["coords"], np.int32)
    dual = _as_numpy(raw["dual_vertices"], np.float32)
    intersected = _as_numpy(raw["intersected"], bool)
    quad_lerp = _as_numpy(raw["quad_lerp"], np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"tile {tile_id} coords must be [N,3]")
    n = coords.shape[0]
    if not (dual.shape == (n, 3) and intersected.shape == (n, 3)):
        raise ValueError(f"tile {tile_id} raw O-Voxel tensors have inconsistent shapes")
    if quad_lerp.ndim == 1:
        quad_lerp = quad_lerp[:, None]
    if quad_lerp.shape[0] != n:
        raise ValueError(f"tile {tile_id} quad_lerp has inconsistent row count")
    if not bool(_valid_cells(coords, tile_size).all()):
        raise ValueError(f"tile {tile_id} has local cells outside [0,1024)")
    if np.unique(_cell_keys(coords, tile_size)).size != n:
        raise ValueError(f"tile {tile_id} decoder coords are not unique")
    if not bool(np.isfinite(dual).all() and np.isfinite(quad_lerp).all()):
        raise ValueError(f"tile {tile_id} raw dual/split values contain NaN or Inf")

    cell_weight = _cell_boundary_weight(coords, tile_size, float(args.boundary_band) * tile_size)
    cell_weight *= float(tile.get("contribution_weight", 1.0))
    global_coords = coords.astype(np.int64) + origin[None].astype(np.int64)
    global_cell_keys = _cell_keys(global_coords)
    dual_voxel = coords.astype(np.float32) + dual
    dual_voxel += origin.astype(np.float32)[None]
    split = np.maximum(quad_lerp[:, :1], 1e-6).astype(np.float32, copy=False)
    normals = _compute_cell_normals(raw, n)

    local_cell_keys = _cell_keys(coords, tile_size)
    sorted_local_cells = np.sort(local_cell_keys)
    vote_keys: List[np.ndarray] = []
    vote_logits: List[np.ndarray] = []
    vote_weights: List[np.ndarray] = []
    vote_axes: List[np.ndarray] = []
    has_logits = "intersected_logits" in raw
    if has_logits:
        logits = _as_numpy(raw["intersected_logits"], np.float32)
        if logits.shape != (n, 3):
            raise ValueError(f"tile {tile_id} intersected_logits must be [N,3]")
        logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    else:
        logits = np.where(intersected, 1.0, -1.0).astype(np.float32)

    for axis in range(3):
        offsets = EDGE_CELL_OFFSETS[axis]
        edge_origin = coords.copy()
        valid_domain = ((edge_origin >= 0) & (edge_origin + offsets.max(axis=0) < tile_size)).all(axis=1)
        if not bool(valid_domain.any()):
            continue
        incident = edge_origin[:, None, :] + offsets[None, :, :]
        incident_keys = _cell_keys(incident.reshape(-1, 3), tile_size).reshape(-1, 4)
        positions = np.searchsorted(sorted_local_cells, incident_keys)
        valid_lookup = positions < sorted_local_cells.size
        valid_lookup &= sorted_local_cells[np.minimum(positions, max(sorted_local_cells.size - 1, 0))] == incident_keys
        eligible = valid_domain & valid_lookup.all(axis=1)
        if not bool(eligible.any()):
            continue
        e = edge_origin[eligible]
        w = _edge_boundary_weight(e, tile_size, float(args.boundary_band) * tile_size)
        keep = w > 0.0
        if not bool(keep.any()):
            continue
        e = e[keep]
        w = w[keep]
        l = logits[eligible, axis][keep]
        vote_keys.append(_edge_keys(e + origin[None], np.full(e.shape[0], axis, dtype=np.int8)))
        vote_logits.append(l.astype(np.float32, copy=False))
        vote_weights.append(w.astype(np.float32, copy=False))
        vote_axes.append(np.full(e.shape[0], axis, dtype=np.int8))

    if vote_keys:
        votes_key = np.concatenate(vote_keys)
        votes_logit = np.concatenate(vote_logits)
        votes_weight = np.concatenate(vote_weights)
        votes_axis = np.concatenate(vote_axes)
        unique_vote_keys, counts = np.unique(votes_key, return_counts=True)
        if bool((counts > 1).any()):
            raise AssertionError(f"tile {tile_id} casts duplicate votes for one primal edge")
    else:
        votes_key = np.empty((0,), dtype=np.int64)
        votes_logit = np.empty((0,), dtype=np.float32)
        votes_weight = np.empty((0,), dtype=np.float32)
        votes_axis = np.empty((0,), dtype=np.int8)

    source_parts: List[np.ndarray] = []
    for axis in range(3):
        active = intersected[:, axis].astype(bool)
        if not bool(active.any()):
            continue
        e = coords[active]
        a = np.full(e.shape[0], axis, dtype=np.int8)
        valid = _valid_edge_coords(e, a, tile_size)
        if bool(valid.any()):
            source_parts.append(_edge_keys(e[valid] + origin[None], a[valid]))
    source_edges = np.unique(np.concatenate(source_parts)) if source_parts else np.empty((0,), dtype=np.int64)

    return {
        "tile_id": tile_id,
        "slot": int(slot),
        "origin": origin,
        "raw_path": Path(tile["raw_ovoxel"]),
        "raw": raw,
        "coords": coords,
        "global_coords": global_coords.astype(np.int32),
        "cell_keys": global_cell_keys,
        "dual_voxel": dual_voxel,
        "split": split,
        "normals": normals,
        "cell_weight": cell_weight,
        "source_edges": source_edges,
        "vote_keys": votes_key,
        "vote_logits": votes_logit,
        "vote_weights": votes_weight,
        "vote_axes": votes_axis,
        "has_logits": bool(has_logits),
        "diagnostics": {
            "tile_id": tile_id,
            "origin": origin.tolist(),
            "raw_cell_count": int(n),
            "raw_active_edge_count": int(source_edges.size),
            "eligible_vote_count": int(votes_key.size),
            "eligible_positive_vote_count": int((votes_logit >= 0).sum()),
            "eligible_negative_vote_count": int((votes_logit < 0).sum()),
            "zero_boundary_weight_cell_count": int((cell_weight <= 0).sum()),
            "zero_boundary_weight_vote_count": int((votes_weight <= 0).sum()),
            "edge_logit_mode": "intersected_logits" if has_logits else "signed_binary_degraded",
        },
    }


def _unique_reduce_by_sorted_key(
    keys: np.ndarray,
    values: np.ndarray,
    reduce: str = "sum",
) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_values = values[order]
    if sorted_keys.size == 0:
        shape = (0,) + values.shape[1:]
        return sorted_keys, np.empty(shape, dtype=values.dtype)
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    unique = sorted_keys[starts]
    if reduce == "sum":
        result = np.add.reduceat(sorted_values, starts, axis=0)
    elif reduce == "max":
        result = np.maximum.reduceat(sorted_values, starts, axis=0)
    else:
        raise ValueError(reduce)
    return unique, result


def _merge_topology(
    prepared: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if not prepared:
        raise ValueError("cannot merge an empty tile set")

    candidate_keys = np.concatenate([tile["cell_keys"] for tile in prepared])
    candidate_coords = np.concatenate([tile["global_coords"] for tile in prepared], axis=0)
    candidate_dual = np.concatenate([tile["dual_voxel"] for tile in prepared], axis=0)
    candidate_split = np.concatenate([tile["split"] for tile in prepared], axis=0)
    candidate_normals = np.concatenate([tile["normals"] for tile in prepared], axis=0)
    candidate_weights = np.concatenate([tile["cell_weight"] for tile in prepared], axis=0)
    candidate_tile_slots = np.concatenate([
        np.full(tile["cell_keys"].shape[0], int(tile["slot"]), dtype=np.int32)
        for tile in prepared
    ])
    candidate_tile_ids = np.concatenate([
        np.full(tile["cell_keys"].shape[0], int(tile["tile_id"]), dtype=np.int32)
        for tile in prepared
    ])
    candidate_count = int(candidate_keys.size)
    if candidate_count == 0:
        raise RuntimeError("all decoder tiles have empty raw cell support")

    # First sort by global cell key.  The group id is the stable join key used
    # by both the mode decision and the later native-QEF statistic merge.
    order = np.argsort(candidate_keys, kind="stable")
    sorted_keys = candidate_keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    group_keys = sorted_keys[starts]
    group_id_sorted = np.cumsum(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]) - 1
    group_count = int(group_keys.size)
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(order.size)

    # Owner is deterministic: maximum interior weight, then lower tile slot.
    owner_order = np.lexsort((candidate_tile_slots[order], -candidate_weights[order], sorted_keys))
    owner_groups_in_order = group_id_sorted[owner_order]
    owner_first = np.r_[True, owner_groups_in_order[1:] != owner_groups_in_order[:-1]]
    owner_group_ids = owner_groups_in_order[owner_first]
    owner_record = np.empty(group_count, dtype=np.int64)
    owner_record[owner_group_ids] = order[owner_order[owner_first]]
    owner_record_sorted = inverse_order[owner_record]
    owner_group_dual = candidate_dual[owner_record]
    owner_group_normal = candidate_normals[owner_record]
    owner_group_tile = candidate_tile_ids[owner_record]

    sorted_dual = candidate_dual[order]
    sorted_normals = candidate_normals[order]
    group_dual_ref = owner_group_dual[group_id_sorted]
    group_normal_ref = owner_group_normal[group_id_sorted]
    dual_distance = np.linalg.norm(sorted_dual - group_dual_ref, axis=1)
    normal_len = np.linalg.norm(sorted_normals, axis=1)
    ref_normal_len = np.linalg.norm(group_normal_ref, axis=1)
    normal_available = (normal_len > 1e-6) & (ref_normal_len > 1e-6)
    normal_dot = np.zeros(sorted_keys.shape[0], dtype=np.float32)
    if bool(normal_available.any()):
        normal_dot[normal_available] = np.abs(
            (sorted_normals[normal_available] * group_normal_ref[normal_available]).sum(axis=1)
        ) / (normal_len[normal_available] * ref_normal_len[normal_available])
    angle_ok = (~normal_available) | (normal_dot >= math.cos(math.radians(float(args.mode_normal_angle_deg))))
    same_mode_sorted = (dual_distance <= float(args.mode_dual_distance_voxel)) & angle_ok
    # A mode conflict falls back to the whole owner tile package.  If no
    # conflict exists, all records in the cell participate with PoU weights.
    conflict_by_group = np.zeros(group_count, dtype=bool)
    np.logical_or.at(conflict_by_group, group_id_sorted, ~same_mode_sorted)
    selected_sorted = same_mode_sorted & (
        (~conflict_by_group[group_id_sorted]) | (np.arange(sorted_keys.size) == owner_record_sorted[group_id_sorted])
    )
    group_weight_sum = np.zeros(group_count, dtype=np.float64)
    np.add.at(
        group_weight_sum,
        group_id_sorted[selected_sorted],
        candidate_weights[order][selected_sorted].astype(np.float64),
    )
    # Zero-weight boundary records cannot provide a non-artificial source.  A
    # group made exclusively of those records still gets the owner package for
    # diagnostics, but its normalized contribution is explicitly zero until
    # the topology closure check rejects it.
    record_norm_sorted = np.zeros(sorted_keys.shape[0], dtype=np.float32)
    selected_positions = np.flatnonzero(selected_sorted)
    if selected_positions.size:
        denom = group_weight_sum[group_id_sorted[selected_positions]]
        positive_denom = denom > 1e-12
        record_norm_sorted[selected_positions[positive_denom]] = (
            candidate_weights[order][selected_positions[positive_denom]] / denom[positive_denom]
        ).astype(np.float32)
    zero_groups = group_weight_sum <= 1e-12
    zero_owner_sorted = owner_record_sorted[zero_groups]
    record_norm_sorted[zero_owner_sorted] = 1.0

    group_dual = np.zeros((group_count, 3), dtype=np.float32)
    group_split = np.zeros((group_count, 1), dtype=np.float32)
    group_normal = np.zeros((group_count, 3), dtype=np.float32)
    group_dual += np.add.reduceat(
        (candidate_dual[order] * record_norm_sorted[:, None]).astype(np.float32), starts, axis=0
    )
    group_split += np.add.reduceat(
        (candidate_split[order] * record_norm_sorted[:, None]).astype(np.float32), starts, axis=0
    )
    group_normal += np.add.reduceat(
        (candidate_normals[order] * record_norm_sorted[:, None]).astype(np.float32), starts, axis=0
    )
    # A cell with all weights zero uses the owner raw dual/split only.  This is
    # still decoder-native data, never baseline data.
    group_dual[zero_groups] = owner_group_dual[zero_groups]
    group_split[zero_groups] = candidate_split[owner_record[zero_groups]]
    group_normal[zero_groups] = owner_group_normal[zero_groups]
    group_split = np.maximum(group_split, 1e-6).astype(np.float32)

    # Construct a compact (cell, tile) lookup for QEF records.  A tile gets a
    # nonzero QEF weight exactly when its complete cell mode package survived.
    tile_factor = max(1024, len(prepared) + 1)
    sorted_pair = group_keys[group_id_sorted] * tile_factor + candidate_tile_slots[order].astype(np.int64)
    pair_order = np.argsort(sorted_pair, kind="stable")
    pair_keys = sorted_pair[pair_order]
    pair_unique, pair_first = np.unique(pair_keys, return_index=True)
    pair_norm = record_norm_sorted[pair_order][pair_first]
    pair_allowed = selected_sorted[pair_order][pair_first] & (pair_norm > 0.0)
    pair_group = group_id_sorted[pair_order][pair_first]

    # Source topology and weighted positive/negative votes.
    source_edges = np.unique(np.concatenate([tile["source_edges"] for tile in prepared]))
    vote_keys = np.concatenate([tile["vote_keys"] for tile in prepared])
    vote_logits = np.concatenate([tile["vote_logits"] for tile in prepared])
    vote_weights = np.concatenate([tile["vote_weights"] for tile in prepared])
    vote_tile_slots = np.concatenate([
        np.full(tile["vote_keys"].shape[0], int(tile["slot"]), dtype=np.int32)
        for tile in prepared
    ])
    if vote_keys.size:
        vote_order = np.argsort(vote_keys, kind="stable")
        vk = vote_keys[vote_order]
        starts_vote = np.r_[0, np.flatnonzero(vk[1:] != vk[:-1]) + 1]
        vote_unique = vk[starts_vote]
        vote_weight_sum = np.add.reduceat(vote_weights[vote_order].astype(np.float64), starts_vote)
        vote_logit_sum = np.add.reduceat(
            (vote_weights[vote_order] * vote_logits[vote_order]).astype(np.float64), starts_vote
        )
        vote_avg = (vote_logit_sum / np.maximum(vote_weight_sum, 1e-12)).astype(np.float32)
        vote_counts = np.diff(np.r_[starts_vote, vk.size]).astype(np.int32)
        vote_positive = np.add.reduceat((vote_logits[vote_order] >= 0).astype(np.int64), starts_vote)
        vote_negative = vote_counts.astype(np.int64) - vote_positive
        vote_slots_unique = vote_tile_slots[vote_order][starts_vote]
        if vote_unique.size > 1:
            # A tile contributes at most one eligible vote per primal edge.
            pair_check = vk.astype(np.int64) * tile_factor + vote_tile_slots[vote_order]
            if np.unique(pair_check).size != pair_check.size:
                raise AssertionError("duplicate tile/edge vote detected")
    else:
        vote_unique = np.empty((0,), dtype=np.int64)
        vote_avg = np.empty((0,), dtype=np.float32)
        vote_weight_sum = np.empty((0,), dtype=np.float64)
        vote_counts = np.empty((0,), dtype=np.int32)
        vote_positive = np.empty((0,), dtype=np.int64)
        vote_negative = np.empty((0,), dtype=np.int64)
        vote_slots_unique = np.empty((0,), dtype=np.int32)

    source_positions = np.searchsorted(source_edges, vote_unique)
    source_match = source_positions < source_edges.size
    source_match &= source_edges[np.minimum(source_positions, max(source_edges.size - 1, 0))] == vote_unique
    vote_selected = vote_avg >= float(args.edge_threshold)
    topology_birth_mask = vote_selected & ~source_match
    topology_birth_count = int(topology_birth_mask.sum())
    if topology_birth_count:
        # Keep the offending edge in the topology diagnostics, then fail
        # before any final tensor is emitted.
        raise RuntimeError(
            f"topology birth detected: {topology_birth_count} voted edges are absent from decoder source topology"
        )

    selected_vote_keys = vote_unique[vote_selected & source_match]
    selected_vote_avg = vote_avg[vote_selected & source_match]
    selected_vote_positions = np.flatnonzero(vote_selected & source_match)
    selected_vote_slots = vote_slots_unique[selected_vote_positions]

    # Four-cell closure is checked against the integer candidate union, not
    # against any mesh-revoxelized support.
    if selected_vote_keys.size:
        selected_edge_coords = np.empty((selected_vote_keys.size, 3), dtype=np.int32)
        selected_edge_axis = (selected_vote_keys % 3).astype(np.int8)
        selected_cell_keys = selected_vote_keys // 3
        selected_edge_coords[:, 2] = selected_cell_keys % GLOBAL_RESOLUTION
        selected_edge_coords[:, 1] = (selected_cell_keys // GLOBAL_RESOLUTION) % GLOBAL_RESOLUTION
        selected_edge_coords[:, 0] = selected_cell_keys // (GLOBAL_RESOLUTION * GLOBAL_RESOLUTION)
        incident = _edge_cells(selected_edge_coords, selected_edge_axis)
        incident_keys = _cell_keys(incident.reshape(-1, 3)).reshape(-1, 4)
        incident_pos = np.searchsorted(group_keys, incident_keys)
        incident_exists = incident_pos < group_keys.size
        incident_exists &= group_keys[np.minimum(incident_pos, max(group_keys.size - 1, 0))] == incident_keys
        incident_group = np.where(incident_exists, incident_pos, 0)
        incident_nonboundary = np.zeros_like(incident_exists)
        if incident_exists.any():
            # A cell has a non-artificial source if any surviving decoder
            # record has positive interior weight.
            group_has_nonboundary = np.zeros(group_count, dtype=bool)
            np.logical_or.at(group_has_nonboundary, group_id_sorted, candidate_weights[order] > 0)
            incident_nonboundary[incident_exists] = group_has_nonboundary[incident_group[incident_exists]]
        closure = incident_exists.reshape(-1, 4).all(axis=1)
        closure &= incident_nonboundary.reshape(-1, 4).all(axis=1)
    else:
        selected_edge_coords = np.empty((0, 3), dtype=np.int32)
        selected_edge_axis = np.empty((0,), dtype=np.int8)
        incident = np.empty((0, 4, 3), dtype=np.int32)
        incident_keys = np.empty((0, 4), dtype=np.int64)
        closure = np.empty((0,), dtype=bool)
        selected_vote_avg = np.empty((0,), dtype=np.float32)
        selected_vote_slots = np.empty((0,), dtype=np.int32)

    final_edge_keys = selected_vote_keys[closure]
    final_edge_coords = selected_edge_coords[closure]
    final_edge_axis = selected_edge_axis[closure]
    final_edge_scores = selected_vote_avg[closure]
    missing_four_count = int((~closure).sum())
    final_cells = np.unique(incident[closure].reshape(-1, 3), axis=0).astype(np.int32, copy=False) if closure.any() else np.empty((0, 3), np.int32)
    final_cell_keys = _cell_keys(final_cells)
    final_cell_positions = np.searchsorted(group_keys, final_cell_keys)
    final_cell_valid = final_cell_positions < group_keys.size
    final_cell_valid &= group_keys[np.minimum(final_cell_positions, max(group_keys.size - 1, 0))] == final_cell_keys
    if not bool(final_cell_valid.all()):
        raise RuntimeError("final closure attempted to create a cell absent from C_candidate")
    final_group_dual = group_dual[final_cell_positions]
    final_group_split = group_split[final_cell_positions]
    final_group_normal = group_normal[final_cell_positions]
    final_group_owner = owner_group_tile[final_cell_positions]
    final_group_coverage = np.bincount(group_id_sorted, minlength=group_count)[final_cell_positions].astype(np.int32)
    final_group_conflict = conflict_by_group[final_cell_positions]

    flags = np.zeros((final_cells.shape[0], 3), dtype=bool)
    edge_cell_positions = np.searchsorted(final_cell_keys, _cell_keys(final_edge_coords))
    edge_cell_valid = edge_cell_positions < final_cell_keys.size
    edge_cell_valid &= final_cell_keys[np.minimum(edge_cell_positions, max(final_cell_keys.size - 1, 0))] == _cell_keys(final_edge_coords)
    if not bool(edge_cell_valid.all()):
        raise RuntimeError("active edge canonical cell is absent from final O-Voxel coords")
    flags[edge_cell_positions, final_edge_axis] = True

    active_emittable = int(final_edge_keys.size)
    rejection_reasons = {
        "vote_below_threshold": int((~vote_selected).sum()),
        "vote_selected_missing_four_cells": missing_four_count,
        "vote_selected_not_source_topology": topology_birth_count,
        "eligible_vote_edge_count": int(vote_unique.size),
        "source_edge_count": int(source_edges.size),
    }
    topology_stats = {
        "candidate_cell_count": int(group_count),
        "candidate_record_count": int(candidate_count),
        "source_edge_count": int(source_edges.size),
        "eligible_vote_edge_count": int(vote_unique.size),
        "eligible_positive_vote_count": int(vote_positive.sum()),
        "eligible_negative_vote_count": int(vote_negative.sum()),
        "vote_edge_count": int(vote_unique.size),
        "vote_selected_edge_count_before_closure": int(selected_vote_keys.size),
        "final_active_edge_count": int(final_edge_keys.size),
        "topology_birth_count": topology_birth_count,
        "active_edge_missing_four_cells": missing_four_count,
        "same_mode_fusion_cell_count": int((~conflict_by_group[final_cell_positions]).sum()) if final_cells.size else 0,
        "owner_fallback_cell_count": int(final_group_conflict.sum()),
        "candidate_cell_coverage_min": int(final_group_coverage.min()) if final_group_coverage.size else 0,
        "candidate_cell_coverage_mean": float(final_group_coverage.mean()) if final_group_coverage.size else 0.0,
        "candidate_cell_coverage_p95": float(np.percentile(final_group_coverage, 95)) if final_group_coverage.size else 0.0,
        "edge_vote_coverage_min": int(vote_counts.min()) if vote_counts.size else 0,
        "edge_vote_coverage_mean": float(vote_counts.mean()) if vote_counts.size else 0.0,
        "edge_vote_coverage_p95": float(np.percentile(vote_counts, 95)) if vote_counts.size else 0.0,
        "rejection_reasons": rejection_reasons,
        "edge_logit_mode": "intersected_logits" if all(tile["has_logits"] for tile in prepared) else "signed_binary_degraded",
    }
    return {
        "candidate_keys": group_keys,
        "candidate_coords": np.stack([
            group_keys // (GLOBAL_RESOLUTION * GLOBAL_RESOLUTION),
            (group_keys // GLOBAL_RESOLUTION) % GLOBAL_RESOLUTION,
            group_keys % GLOBAL_RESOLUTION,
        ], axis=1).astype(np.int32),
        "candidate_group_count": group_count,
        "candidate_group_dual": group_dual,
        "candidate_group_split": group_split,
        "candidate_group_normal": group_normal,
        "candidate_group_owner_tile": owner_group_tile,
        "candidate_group_coverage": np.bincount(group_id_sorted, minlength=group_count).astype(np.int32),
        "candidate_group_conflict": conflict_by_group,
        "candidate_record_pair_keys": pair_unique,
        "candidate_record_pair_norm": pair_norm,
        "candidate_record_pair_allowed": pair_allowed,
        "candidate_record_pair_group": pair_group,
        "tile_factor": tile_factor,
        "final_cells": final_cells,
        "final_cell_keys": final_cell_keys,
        "final_dual_voxel": final_group_dual,
        "final_split": final_group_split,
        "final_normal": final_group_normal,
        "final_owner_tile": final_group_owner,
        "final_coverage": final_group_coverage,
        "final_mode_conflict": final_group_conflict,
        "intersected": flags,
        "active_edge_keys": final_edge_keys,
        "active_edge_coords": final_edge_coords,
        "active_edge_axis": final_edge_axis,
        "active_edge_vote": final_edge_scores,
        "source_edge_keys": source_edges,
        "vote_edge_keys": vote_unique,
        "vote_edge_logits": vote_avg,
        "vote_edge_weights": vote_weight_sum.astype(np.float32),
        "vote_edge_coverage": vote_counts,
        "vote_edge_positive": vote_positive,
        "vote_edge_negative": vote_negative,
        "stats": topology_stats,
    }


def _topology_hash(topology: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(topology["active_edge_keys"], dtype=np.int64).tobytes())
    digest.update(np.asarray(topology["final_cells"], dtype=np.int32).tobytes())
    return digest.hexdigest()


def _filter_tile_faces(
    raw: Mapping[str, Any],
    origin: np.ndarray,
    final_edge_keys: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    vertices_local = _as_numpy(raw["mesh_vertices"], np.float32)
    faces = _as_numpy(raw["mesh_faces"], np.int32)
    provenance = raw["provenance"]
    source_index = _as_numpy(provenance["source_ovoxel_index"], np.int64)
    source_axis = _as_numpy(provenance["source_edge_axis"], np.int64)
    coords = _as_numpy(raw["coords"], np.int32)
    if source_index.shape[0] != faces.shape[0] or source_axis.shape[0] != faces.shape[0]:
        raise ValueError("mesh provenance must contain one source edge per emitted triangle")
    valid = (
        (source_index >= 0) & (source_index < coords.shape[0])
        & (source_axis >= 0) & (source_axis < 3)
    )
    source_keys = np.full(source_index.shape, -1, dtype=np.int64)
    if bool(valid.any()):
        source_keys[valid] = _edge_keys(
            coords[source_index[valid]].astype(np.int64) + origin[None].astype(np.int64),
            source_axis[valid].astype(np.int8),
        )
    positions = np.searchsorted(final_edge_keys, source_keys)
    keep = valid & (positions < final_edge_keys.size)
    if final_edge_keys.size:
        keep &= final_edge_keys[np.minimum(positions, final_edge_keys.size - 1)] == source_keys
    else:
        keep[:] = False
    faces_filtered = faces[keep].astype(np.int32, copy=False)
    # Decoder mesh vertices are in the local [-.5,.5] tile frame.  This exact
    # integer-origin transform is the only local->global mesh mapping used.
    vertices_global = -0.5 + (
        origin.astype(np.float32)[None]
        + (vertices_local + 0.5) * float(TILE_SIZE)
    ) / float(GLOBAL_RESOLUTION)
    return vertices_global.astype(np.float32, copy=False), faces_filtered, {
        "mesh_vertex_count": int(vertices_local.shape[0]),
        "mesh_face_count": int(faces.shape[0]),
        "source_provenance_valid_count": int(valid.sum()),
        "topology_constrained_face_count": int(keep.sum()),
        "topology_constrained_face_fraction": float(keep.mean()) if keep.size else 0.0,
    }


def _load_or_build_tile_qef(
    tile: Mapping[str, Any],
    prepared_tile: Mapping[str, Any],
    topology: Mapping[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    topology_hash: str,
) -> Dict[str, Any]:
    tile_id = int(tile["tile_id"])
    explicit = tile.get("qef_stats")
    cache_path = _resolve_path(explicit, Path(tile["raw_ovoxel"]).parent) if explicit else (
        output_dir / "qef_tiles" / f"tile_{tile_id:03d}.pt"
    )
    if cache_path.is_file() and not args.force_qef and not explicit:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("topology_hash") == topology_hash:
            return cached
    if explicit and cache_path.is_file() and not args.force_qef:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("topology_hash", topology_hash) != topology_hash:
            raise RuntimeError(f"tile {tile_id} qef_stats topology hash does not match final source topology")
        return cached

    raw = prepared_tile["raw"]
    vertices_global, faces_filtered, face_diag = _filter_tile_faces(
        raw, np.asarray(tile["origin"], dtype=np.int32), topology["active_edge_keys"]
    )
    if faces_filtered.size == 0:
        empty = {
            "tile_id": tile_id,
            "slot": int(prepared_tile["slot"]),
            "topology_hash": topology_hash,
            "coords": torch.empty((0, 3), dtype=torch.int32),
            "dual_vertices": torch.empty((0, 3), dtype=torch.float32),
            "q_edge": torch.empty((0, 4, 4), dtype=torch.float32),
            "q_face": torch.empty((0, 4, 4), dtype=torch.float32),
            "q_boundary": torch.empty((0, 4, 4), dtype=torch.float32),
            "q_sum": torch.empty((0, 3), dtype=torch.float32),
            "q_count": torch.empty((0,), dtype=torch.float32),
            "face_weight": 1.0,
            "boundary_weight": 0.0,
            "regularization_weight": float(args.regularization_weight),
            "diagnostics": {**face_diag, "native_qef_cell_count": 0, "native_qef_observation_count": 0},
        }
        if not explicit:
            _atomic_torch_save(cache_path, empty)
        return empty

    print(
        f"[qef tile {tile_id:03d}] native full stats on "
        f"{faces_filtered.shape[0]:,}/{face_diag['mesh_face_count']:,} triangles",
        flush=True,
    )
    started = time.perf_counter()
    stats = mesh_to_flexible_dual_grid_qef_stats(
        vertices=torch.from_numpy(vertices_global).contiguous(),
        faces=torch.from_numpy(faces_filtered).contiguous(),
        grid_size=GLOBAL_RESOLUTION,
        aabb=RUNTIME_AABB,
        grid_range=[
            [int(v) for v in tile["origin"]],
            [int(v) + TILE_SIZE for v in tile["origin"]],
        ],
        face_weight=1.0,
        boundary_weight=0.0,
        regularization_weight=float(args.regularization_weight),
        timing=False,
    )
    result = {
        "tile_id": tile_id,
        "slot": int(prepared_tile["slot"]),
        "topology_hash": topology_hash,
        "coords": stats["coords"].cpu().int(),
        "dual_vertices": stats["dual_vertices"].cpu().float(),
        "q_edge": stats["q_edge"].cpu().float(),
        "q_face": stats["q_face"].cpu().float(),
        "q_boundary": stats["q_boundary"].cpu().float(),
        "q_sum": stats["q_sum"].cpu().float(),
        "q_count": stats["q_count"].cpu().float(),
        "face_weight": 1.0,
        "boundary_weight": 0.0,
        "regularization_weight": float(args.regularization_weight),
        "diagnostics": {
            **face_diag,
            "native_qef_cell_count": int(stats["coords"].shape[0]),
            "native_qef_observation_count": int(stats["q_count"].sum().item()),
            "native_qef_seconds": float(time.perf_counter() - started),
            "tile_boundary_qef_count": int((stats["q_boundary"].abs().sum(dim=(1, 2)) > 0).sum().item()),
        },
    }
    if not explicit:
        _atomic_torch_save(cache_path, result)
    return result


def _quadratic_energy(a: torch.Tensor, b: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return (value * torch.bmm(a, value.unsqueeze(-1)).squeeze(-1)).sum(dim=1) - 2.0 * (b * value).sum(dim=1)


def _solve_box_native_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Native TRELLIS-style bounded 3-D QEF solve, with diagnostics."""
    batch = a.shape[0]
    solution = torch.linalg.lstsq(a, b.unsqueeze(-1)).solution.squeeze(-1)
    finite_solution = torch.isfinite(solution).all(dim=1)
    inside = finite_solution & ((solution >= lo) & (solution <= hi)).all(dim=1)
    best_v = torch.where(inside[:, None], solution, torch.zeros_like(solution))
    best_e = torch.where(
        inside,
        _quadratic_energy(a, b, solution),
        torch.full((batch,), float("inf"), dtype=a.dtype, device=a.device),
    )

    def consider(candidate: torch.Tensor, valid: torch.Tensor) -> None:
        nonlocal best_v, best_e
        energy = _quadratic_energy(a, b, candidate)
        use = valid & torch.isfinite(energy) & (energy < best_e)
        best_v = torch.where(use[:, None], candidate, best_v)
        best_e = torch.where(use, energy, best_e)

    for fixed_axis in range(3):
        free = [axis for axis in range(3) if axis != fixed_axis]
        aff = a[:, free][:, :, free]
        for bound in (lo[:, fixed_axis], hi[:, fixed_axis]):
            rhs = b[:, free] - a[:, free, fixed_axis] * bound[:, None]
            free_value = torch.linalg.lstsq(aff, rhs.unsqueeze(-1)).solution.squeeze(-1)
            candidate = torch.zeros_like(solution)
            candidate[:, fixed_axis] = bound
            candidate[:, free] = free_value
            valid = torch.isfinite(free_value).all(dim=1)
            valid &= ((free_value >= lo[:, free]) & (free_value <= hi[:, free])).all(dim=1)
            consider(candidate, valid)

    for free_axis in range(3):
        fixed = [axis for axis in range(3) if axis != free_axis]
        denominator = a[:, free_axis, free_axis]
        for bound0 in (lo[:, fixed[0]], hi[:, fixed[0]]):
            for bound1 in (lo[:, fixed[1]], hi[:, fixed[1]]):
                rhs = (
                    b[:, free_axis]
                    - a[:, free_axis, fixed[0]] * bound0
                    - a[:, free_axis, fixed[1]] * bound1
                )
                free_value = rhs / denominator.clamp_min(torch.finfo(a.dtype).eps)
                candidate = torch.zeros_like(solution)
                candidate[:, free_axis] = free_value
                candidate[:, fixed[0]] = bound0
                candidate[:, fixed[1]] = bound1
                valid = torch.isfinite(free_value) & (free_value >= lo[:, free_axis]) & (free_value <= hi[:, free_axis])
                valid &= denominator.abs() > torch.finfo(a.dtype).eps
                consider(candidate, valid)

    for sx in (0, 1):
        for sy in (0, 1):
            for sz in (0, 1):
                candidate = torch.stack([
                    lo[:, 0] if sx == 0 else hi[:, 0],
                    lo[:, 1] if sy == 0 else hi[:, 1],
                    lo[:, 2] if sz == 0 else hi[:, 2],
                ], dim=1)
                consider(candidate, torch.ones(batch, dtype=torch.bool, device=a.device))

    fallback = torch.nan_to_num(solution, nan=0.0, posinf=0.0, neginf=0.0)
    fallback = torch.maximum(torch.minimum(fallback, hi), lo)
    unresolved = ~torch.isfinite(best_e)
    best_v = torch.where(unresolved[:, None], fallback, best_v)
    # Match the native float32 box tolerance used for parity reporting.  The
    # solve itself still uses the exact [lo, hi] box; this tolerance only
    # classifies a solution that is within a few ulps of a face as clamped.
    clamp_tolerance = 1e-5
    clamped = (~inside) | ((best_v <= lo + clamp_tolerance) | (best_v >= hi - clamp_tolerance)).any(dim=1)
    eig = torch.linalg.eigvalsh(a)
    scale = eig[:, -1].abs().clamp_min(torch.finfo(a.dtype).eps)
    rank = (eig > scale[:, None] * 1e-6).sum(dim=1)
    singular = (~finite_solution) | (rank < 3)
    return best_v, clamped, rank, singular


def _solve_qef(
    cells: np.ndarray,
    q_edge: np.ndarray,
    q_face: np.ndarray,
    q_boundary: np.ndarray,
    q_sum: np.ndarray,
    q_count: np.ndarray,
    fallback_dual_voxel: np.ndarray,
    resolution: int,
    regularization_weight: float,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    n = int(cells.shape[0])
    # Solve in cell-local voxel coordinates u in [0,1]^3.  Native QEFs are
    # expressed in translated physical coordinates, but solving directly at
    # C4096 introduces catastrophic cancellation in the affine homogeneous
    # term and turns many otherwise interior solutions into box corners.
    # The affine change below is mathematically identical and keeps the
    # bounded solve numerically well-conditioned.
    dual_cell = fallback_dual_voxel.astype(np.float32, copy=True)
    solved_mask = q_count > 1e-12
    clamp_count = 0
    singular_count = 0
    rank_hist: Dict[str, int] = {}
    if bool(solved_mask.any()):
        indices = np.flatnonzero(solved_mask)
        for start in range(0, indices.size, int(batch_size)):
            batch_indices = indices[start:start + int(batch_size)]
            q = (
                q_edge[batch_indices]
                + q_face[batch_indices]
                + q_boundary[batch_indices]
            ).astype(np.float32, copy=False)
            mass = q_count[batch_indices].astype(np.float32, copy=False)
            qbar = q_sum[batch_indices] / np.maximum(mass[:, None], 1e-12)
            reg = float(regularization_weight) * mass
            q[:, 0, 0] += reg
            q[:, 1, 1] += reg
            q[:, 2, 2] += reg
            q[:, 0, 3] -= reg * qbar[:, 0]
            q[:, 1, 3] -= reg * qbar[:, 1]
            q[:, 2, 3] -= reg * qbar[:, 2]
            q[:, 3, 0] -= reg * qbar[:, 0]
            q[:, 3, 1] -= reg * qbar[:, 1]
            q[:, 3, 2] -= reg * qbar[:, 2]
            q[:, 3, 3] += reg * (qbar * qbar).sum(axis=1)
            q_t = torch.from_numpy(q).to(device=device)
            cc = torch.from_numpy(cells[batch_indices].astype(np.float32)).to(device=device)
            scale = float(resolution)
            a_voxel = q_t[:, :3, :3] / (scale * scale)
            q_voxel_affine = q_t[:, :3, 3] / scale
            # x_voxel = cell + u, so b_local = -(A_voxel*cell + d_voxel).
            aa = a_voxel
            bb = -(torch.bmm(aa, cc.unsqueeze(-1)).squeeze(-1) + q_voxel_affine)
            lo = torch.zeros_like(cc)
            hi = torch.ones_like(cc)
            with torch.no_grad():
                value, clamped, rank, singular = _solve_box_native_torch(aa, bb, lo, hi)
            dual_cell[batch_indices] = value.detach().cpu().numpy()
            clamp_count += int(clamped.sum().item())
            singular_count += int(singular.sum().item())
            unique_rank, rank_count = torch.unique(rank, return_counts=True)
            for key, count in zip(unique_rank.detach().cpu().tolist(), rank_count.detach().cpu().tolist()):
                rank_hist[str(int(key))] = rank_hist.get(str(int(key)), 0) + int(count)
            del q_t, aa, bb, cc, value, clamped, rank, singular
    dual_cell = np.nan_to_num(dual_cell, nan=0.5, posinf=1.0, neginf=0.0).clip(0.0, 1.0).astype(np.float32)
    dual_translated = (cells.astype(np.float32) + dual_cell) / float(resolution)
    return {
        "dual_translated": dual_translated.astype(np.float32),
        "dual_cell": dual_cell,
        "qef_solved_cell_count": int(solved_mask.sum()),
        "qef_no_constraint_fallback_count": int((~solved_mask).sum()),
        "qef_clamped_count": int(clamp_count),
        "qef_singular_fallback_count": int(singular_count),
        "qef_rank_histogram": rank_hist,
    }


def _aggregate_qef(
    topology: Mapping[str, Any],
    prepared: Sequence[Mapping[str, Any]],
    tile_manifest: Mapping[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    topology_hash = _topology_hash(topology)
    final_cells = topology["final_cells"]
    final_keys = topology["final_cell_keys"]
    n = int(final_cells.shape[0])
    q_edge = np.zeros((n, 4, 4), dtype=np.float32)
    q_face = np.zeros((n, 4, 4), dtype=np.float32)
    q_boundary = np.zeros((n, 4, 4), dtype=np.float32)
    q_sum = np.zeros((n, 3), dtype=np.float32)
    q_count = np.zeros((n,), dtype=np.float32)
    qef_record_count = 0
    qef_tile_diagnostics: Dict[str, Any] = {}
    pair_keys = topology["candidate_record_pair_keys"]
    pair_norm = topology["candidate_record_pair_norm"]
    pair_allowed = topology["candidate_record_pair_allowed"]
    tile_factor = int(topology["tile_factor"])
    for tile, prepared_tile in zip(tile_manifest["tiles"], prepared):
        qef = _load_or_build_tile_qef(
            tile, prepared_tile, topology, args, output_dir, topology_hash
        )
        tile_id = int(tile["tile_id"])
        qef_tile_diagnostics[str(tile_id)] = dict(qef.get("diagnostics", {}))
        coords = _as_numpy(qef["coords"], np.int32)
        if coords.size == 0:
            continue
        qkeys = _cell_keys(coords)
        positions = np.searchsorted(final_keys, qkeys)
        final_valid = positions < final_keys.size
        final_valid &= final_keys[np.minimum(positions, max(final_keys.size - 1, 0))] == qkeys
        pair = qkeys.astype(np.int64) * tile_factor + int(prepared_tile["slot"])
        pair_pos = np.searchsorted(pair_keys, pair)
        pair_valid = pair_pos < pair_keys.size
        pair_valid &= pair_keys[np.minimum(pair_pos, max(pair_keys.size - 1, 0))] == pair
        keep = final_valid & pair_valid
        if pair_allowed.size:
            keep &= pair_allowed[np.minimum(pair_pos, max(pair_allowed.size - 1, 0))]
        if not bool(keep.any()):
            continue
        idx = positions[keep].astype(np.int64)
        weights = pair_norm[pair_pos[keep]].astype(np.float32)
        order = np.argsort(idx, kind="stable")
        idx_sorted = idx[order]
        starts = np.r_[0, np.flatnonzero(idx_sorted[1:] != idx_sorted[:-1]) + 1]
        unique_idx = idx_sorted[starts]
        w = weights[order]
        qe = _as_numpy(qef["q_edge"], np.float32)[keep][order]
        qf = _as_numpy(qef["q_face"], np.float32)[keep][order]
        qb = _as_numpy(qef["q_boundary"], np.float32)[keep][order]
        qs = _as_numpy(qef["q_sum"], np.float32)[keep][order]
        qc = _as_numpy(qef["q_count"], np.float32)[keep][order]
        q_edge[unique_idx] += np.add.reduceat(qe * w[:, None, None], starts, axis=0)
        q_face[unique_idx] += np.add.reduceat(qf * w[:, None, None], starts, axis=0)
        q_boundary[unique_idx] += np.add.reduceat(qb * w[:, None, None], starts, axis=0)
        q_sum[unique_idx] += np.add.reduceat(qs * w[:, None], starts, axis=0)
        q_count[unique_idx] += np.add.reduceat(qc * w, starts)
        qef_record_count += int(keep.sum())

    solved = _solve_qef(
        final_cells,
        q_edge,
        q_face,
        q_boundary,
        q_sum,
        q_count,
        topology["final_dual_voxel"],
        GLOBAL_RESOLUTION,
        float(args.regularization_weight),
        device,
        int(args.qef_batch_size),
    )
    stats = {
        "qef_cell_count": n,
        "qef_record_count": int(qef_record_count),
        "qef_edge_matrix_nonzero_cell_count": int((np.abs(q_edge).sum(axis=(1, 2)) > 0).sum()),
        "qef_face_matrix_nonzero_cell_count": int((np.abs(q_face).sum(axis=(1, 2)) > 0).sum()),
        "qef_boundary_matrix_nonzero_cell_count": int((np.abs(q_boundary).sum(axis=(1, 2)) > 0).sum()),
        "qef_edge_term_count": int((np.abs(q_edge).sum(axis=(1, 2)) > 0).sum()),
        "qef_face_term_count": int((np.abs(q_face).sum(axis=(1, 2)) > 0).sum()),
        "qef_regularization_term_count": int((q_count > 0).sum()),
        "qef_no_constraint_fallback_count": solved["qef_no_constraint_fallback_count"],
        "qef_clamped_count": solved["qef_clamped_count"],
        "qef_singular_fallback_count": solved["qef_singular_fallback_count"],
        "qef_rank_histogram": solved["qef_rank_histogram"],
        "regularization_weight": float(args.regularization_weight),
        "boundary_weight": 0.0,
        "native_qef_tile_diagnostics": qef_tile_diagnostics,
    }
    payload = {
        "format": "pixal3d_local_only_native_full_qef_stats_v1",
        "topology_hash": topology_hash,
        "resolution": GLOBAL_RESOLUTION,
        "coords": torch.from_numpy(final_cells),
        "q_edge": torch.from_numpy(q_edge),
        "q_face": torch.from_numpy(q_face),
        "q_boundary": torch.from_numpy(q_boundary),
        "q_sum": torch.from_numpy(q_sum),
        "q_count": torch.from_numpy(q_count),
        "q_bar": torch.from_numpy(q_sum / np.maximum(q_count[:, None], 1e-12)),
        "dual_vertices_translated": torch.from_numpy(solved["dual_translated"]),
        "dual_vertices_cell": torch.from_numpy(solved["dual_cell"]),
        "baseline_coord_count": 0,
        "baseline_edge_count": 0,
        "baseline_qef_count": 0,
        "baseline_face_count": 0,
        "stats": stats,
    }
    return {"payload": payload, "dual_cell": solved["dual_cell"], "stats": stats}


def _first_failed_edge_payload(
    topology: Mapping[str, Any],
    final_ovoxel: Mapping[str, Any],
    emitted_quad_keys: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    active_keys = np.asarray(topology["active_edge_keys"], dtype=np.int64)
    active_coords = np.asarray(topology["active_edge_coords"], dtype=np.int32)
    active_axis = np.asarray(topology["active_edge_axis"], dtype=np.int8)
    if emitted_quad_keys is None:
        failed_index = 0
    else:
        emitted_set = np.asarray(emitted_quad_keys, dtype=np.int64)
        missing = ~np.isin(active_keys, emitted_set, assume_unique=False)
        failed_index = int(np.flatnonzero(missing)[0]) if bool(missing.any()) else 0
    edge = active_coords[failed_index] if active_coords.size else np.zeros(3, np.int32)
    axis = int(active_axis[failed_index]) if active_axis.size else 0
    incident = _edge_cells(edge.reshape(1, 3), np.asarray([axis], dtype=np.int8))[0]
    cell_keys = _cell_keys(np.asarray(final_ovoxel["coords"], dtype=np.int32))
    positions = np.searchsorted(cell_keys, _cell_keys(incident))
    valid = positions < cell_keys.size
    valid &= cell_keys[np.minimum(positions, max(cell_keys.size - 1, 0))] == _cell_keys(incident)
    return {
        "active_edge_key": int(active_keys[failed_index]) if active_keys.size else -1,
        "edge_coord": torch.from_numpy(edge.copy()),
        "edge_axis": axis,
        "incident_cells": torch.from_numpy(incident.copy()),
        "incident_cell_indices": torch.from_numpy(positions.astype(np.int64)),
        "incident_cell_valid": torch.from_numpy(valid),
        "axis_flags": final_ovoxel["intersected"][positions[valid]].clone() if bool(valid.any()) else torch.empty((0, 3), dtype=torch.bool),
        "dual_vertices_cell": final_ovoxel["dual_vertices_cell"][positions[valid]].clone() if bool(valid.any()) else torch.empty((0, 3), dtype=torch.float32),
        "split_weight": final_ovoxel["split_weight"][positions[valid]].clone() if bool(valid.any()) else torch.empty((0, 1), dtype=torch.float32),
        "source_edge_keys": torch.from_numpy(np.asarray(topology["source_edge_keys"], dtype=np.int64)),
    }


def _build_final_mesh(
    topology: Mapping[str, Any],
    qef: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    final_ovoxel = {
        "resolution": GLOBAL_RESOLUTION,
        "batch_index": torch.zeros((topology["final_cells"].shape[0], 1), dtype=torch.int32),
        "coords": torch.from_numpy(topology["final_cells"].astype(np.int32, copy=False)),
        "dual_vertices_cell": torch.from_numpy(qef["dual_cell"].astype(np.float32, copy=False)),
        "intersected": torch.from_numpy(topology["intersected"].astype(bool, copy=False)),
        "split_weight": torch.from_numpy(topology["final_split"].astype(np.float32, copy=False)),
        "active_edge_keys": torch.from_numpy(np.asarray(topology["active_edge_keys"], dtype=np.int64)),
        "source_edge_keys": torch.from_numpy(np.asarray(topology["source_edge_keys"], dtype=np.int64)),
        "baseline_coord_count": 0,
        "baseline_edge_count": 0,
        "baseline_qef_count": 0,
        "baseline_face_count": 0,
    }
    if not bool(torch.isfinite(final_ovoxel["dual_vertices_cell"]).all()):
        raise RuntimeError("final O-Voxel dual vertices contain NaN or Inf")
    if int(final_ovoxel["coords"].shape[0]) != int(final_ovoxel["dual_vertices_cell"].shape[0]):
        raise AssertionError("final O-Voxel coords/dual shape mismatch")
    emitted_quad_count = 0
    triangle_count = 0
    provenance: Dict[str, Any] = {}
    try:
        with torch.cuda.device(device), torch.no_grad():
            coords = final_ovoxel["coords"].to(device=device)
            dual = final_ovoxel["dual_vertices_cell"].to(device=device)
            intersected = final_ovoxel["intersected"].to(device=device)
            split_weight = final_ovoxel["split_weight"].to(device=device)
            vertices, faces, provenance_t = flexible_dual_grid_to_mesh(
                coords,
                dual,
                intersected,
                split_weight,
                aabb=RUNTIME_AABB.to(device),
                grid_size=GLOBAL_RESOLUTION,
                train=False,
                return_provenance=True,
            )
            torch.cuda.synchronize(device)
            provenance = _cpu(provenance_t)
            vertices_cpu = vertices.detach().cpu().float()
            faces_cpu = faces.detach().cpu().int()
            emitted_quad_count = int(provenance["quad_indices"].shape[0])
            triangle_count = int(faces_cpu.shape[0])
            active_emittable = int(topology["active_edge_keys"].shape[0])
            if emitted_quad_count != active_emittable or triangle_count != 2 * emitted_quad_count:
                source_edge = provenance.get("source_ovoxel_index", torch.empty((0,), dtype=torch.int64))[::2]
                source_axis = provenance.get("source_edge_axis", torch.empty((0,), dtype=torch.int64))[::2]
                emitted_quad_keys = np.empty((source_edge.shape[0],), dtype=np.int64)
                if source_edge.numel():
                    source_cells = final_ovoxel["coords"][source_edge.long()].numpy()
                    emitted_quad_keys = _edge_keys(source_cells, source_axis.numpy().astype(np.int8))
                failure = _first_failed_edge_payload(topology, final_ovoxel, emitted_quad_keys)
                _atomic_torch_save(output_dir / "failures" / "first_failed_edge.pt", failure)
                raise RuntimeError(
                    "mesher count invariant failed: "
                    f"emitted_quad_count={emitted_quad_count}, active_emittable_edge_count={active_emittable}, "
                    f"triangle_count={triangle_count}"
                )
    finally:
        if torch.cuda.is_available():
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    final_ovoxel["dual_vertices_translated"] = torch.from_numpy(
        -0.5 + (topology["final_cells"].astype(np.float32) + qef["dual_cell"]) / float(GLOBAL_RESOLUTION)
    )
    final_ovoxel["mesher_emitted_quad_count"] = emitted_quad_count
    final_ovoxel["mesher_triangle_count"] = triangle_count
    final_ovoxel["mesher_provenance"] = provenance
    return {
        "ovoxel": final_ovoxel,
        "vertices": vertices_cpu,
        "faces": faces_cpu,
        "provenance": provenance,
        "emitted_quad_count": emitted_quad_count,
        "triangle_count": triangle_count,
    }


def _final_quad_cell_indices(topology: Mapping[str, Any]) -> np.ndarray:
    """Return the exact four-cell order used by the native mesher."""
    edge_cells = _edge_cells(
        np.asarray(topology["active_edge_coords"], dtype=np.int32),
        np.asarray(topology["active_edge_axis"], dtype=np.int8),
    )
    final_keys = np.asarray(topology["final_cell_keys"], dtype=np.int64)
    if edge_cells.size == 0:
        return np.empty((0, 4), dtype=np.int32)
    incident_keys = _cell_keys(edge_cells.reshape(-1, 3)).reshape(-1, 4)
    positions = np.searchsorted(final_keys, incident_keys)
    valid = positions < final_keys.size
    valid &= final_keys[np.minimum(positions, max(final_keys.size - 1, 0))] == incident_keys
    if not bool(valid.all()):
        raise RuntimeError("active edge lost an incident final O-Voxel cell")
    return positions.astype(np.int32, copy=False)


def _final_mesh_triangles_from_topology(topology: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Construct mesher-equivalent triangle connectivity before extraction."""
    quads = _final_quad_cell_indices(topology)
    edge_count = int(quads.shape[0])
    if edge_count == 0:
        return np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.int64)
    split = np.asarray(topology["final_split"], dtype=np.float32)[quads]
    split_first = ((split[:, 0] * split[:, 2]) > (split[:, 1] * split[:, 3])).reshape(-1)
    triangles = np.empty((2 * edge_count, 3), dtype=np.int32)
    triangles[0::2] = np.where(
        split_first[:, None], quads[:, [0, 1, 2]], quads[:, [0, 1, 3]]
    )
    triangles[1::2] = np.where(
        split_first[:, None], quads[:, [0, 2, 3]], quads[:, [3, 1, 2]]
    )
    owners = np.arange(edge_count, dtype=np.int64).repeat(2)
    return triangles, owners


def _degenerate_edge_keys_from_qef(
    topology: Mapping[str, Any],
    qef: Mapping[str, Any],
    area_threshold: float = 1e-10,
) -> np.ndarray:
    """Return source edges whose native-QEF quad has a zero-area triangle."""
    triangles, owners = _final_mesh_triangles_from_topology(topology)
    if triangles.size == 0:
        return np.empty((0,), dtype=np.int64)
    coords = np.asarray(topology["final_cells"], dtype=np.float32)
    dual = np.asarray(qef["dual_cell"], dtype=np.float32)
    bad_owner_parts: List[np.ndarray] = []
    for start in range(0, triangles.shape[0], 2_000_000):
        stop = min(start + 2_000_000, triangles.shape[0])
        tri = (coords[triangles[start:stop]] + dual[triangles[start:stop]]) / float(GLOBAL_RESOLUTION)
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        bad = area <= float(area_threshold)
        if bool(bad.any()):
            bad_owner_parts.append(owners[start:stop][bad])
    if not bad_owner_parts:
        return np.empty((0,), dtype=np.int64)
    bad_owner = np.unique(np.concatenate(bad_owner_parts))
    return np.asarray(topology["active_edge_keys"], dtype=np.int64)[bad_owner]


def _nonmanifold_edge_keys_from_topology(
    topology: Mapping[str, Any],
    covered_vertex_mask: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Return active source edges causing covered-ROI nonmanifold edges.

    This is the combinatorial equivalent of the native mesher check and is
    performed before the single final mesh extraction.  It only rejects
    already active decoder-source edges.
    """
    triangles, owners = _final_mesh_triangles_from_topology(topology)
    if triangles.size == 0:
        return np.empty((0,), dtype=np.int64), 0
    covered = covered_vertex_mask[triangles].all(axis=1)
    selected = triangles[covered]
    selected_owners = owners[covered]
    if selected.size == 0:
        return np.empty((0,), dtype=np.int64), 0
    vertex_count = int(covered_vertex_mask.size)
    face_count = int(selected.shape[0])
    edge_keys = np.empty((3 * face_count,), dtype=np.int64)
    left = np.minimum(selected[:, 0], selected[:, 1])
    right = np.maximum(selected[:, 0], selected[:, 1])
    edge_keys[:face_count] = left.astype(np.int64) * vertex_count + right.astype(np.int64)
    left = np.minimum(selected[:, 1], selected[:, 2])
    right = np.maximum(selected[:, 1], selected[:, 2])
    edge_keys[face_count:2 * face_count] = left.astype(np.int64) * vertex_count + right.astype(np.int64)
    left = np.minimum(selected[:, 2], selected[:, 0])
    right = np.maximum(selected[:, 2], selected[:, 0])
    edge_keys[2 * face_count:] = left.astype(np.int64) * vertex_count + right.astype(np.int64)
    order = np.argsort(edge_keys, kind="mergesort")
    sorted_keys = edge_keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    counts = np.diff(np.r_[starts, sorted_keys.size])
    bad_groups = np.flatnonzero(counts > 2)
    if bad_groups.size == 0:
        return np.empty((0,), dtype=np.int64), 0
    bad_starts = starts[bad_groups]
    bad_ends = bad_starts + counts[bad_groups]
    row_mark = np.zeros((sorted_keys.size + 1,), dtype=np.int8)
    row_mark[bad_starts] += 1
    row_mark[bad_ends] -= 1
    bad_rows = np.cumsum(row_mark[:-1]) > 0
    bad_owner = np.unique(selected_owners[order[bad_rows] % face_count])
    active_keys = np.asarray(topology["active_edge_keys"], dtype=np.int64)
    return active_keys[bad_owner], int(bad_groups.size)


def _reject_edges(topology: Dict[str, Any], rejected_keys: np.ndarray, reason: str) -> int:
    rejected_keys = np.unique(np.asarray(rejected_keys, dtype=np.int64))
    if rejected_keys.size == 0:
        return 0
    active = np.asarray(topology["active_edge_keys"], dtype=np.int64)
    keep = ~np.isin(active, rejected_keys, assume_unique=False)
    removed = int((~keep).sum())
    if removed == 0:
        return 0
    for key in ("active_edge_keys", "active_edge_coords", "active_edge_axis", "active_edge_vote"):
        topology[key] = np.asarray(topology[key])[keep]
    flags = np.asarray(topology["intersected"], dtype=bool).copy()
    edge_coords = np.asarray(topology["active_edge_coords"], dtype=np.int32)
    # Reconstruct flags from the retained edge list to avoid stale flags for
    # a rejected edge whose cell is still present as another quad incident.
    flags.fill(False)
    final_keys = np.asarray(topology["final_cell_keys"], dtype=np.int64)
    positions = np.searchsorted(final_keys, edge_coords.astype(np.int64)[:, 0] * GLOBAL_RESOLUTION * GLOBAL_RESOLUTION + edge_coords[:, 1].astype(np.int64) * GLOBAL_RESOLUTION + edge_coords[:, 2].astype(np.int64))
    if edge_coords.size:
        valid = positions < final_keys.size
        edge_key_cells = _cell_keys(edge_coords)
        valid &= final_keys[np.minimum(positions, max(final_keys.size - 1, 0))] == edge_key_cells
        if not bool(valid.all()):
            raise RuntimeError("retained active edge lost its canonical final cell")
        flags[positions, np.asarray(topology["active_edge_axis"], dtype=np.int8)] = True
    topology["intersected"] = flags
    topology["stats"][f"{reason}_edge_count"] = int(topology["stats"].get(f"{reason}_edge_count", 0) + removed)
    topology["stats"]["final_active_edge_count"] = int(topology["active_edge_keys"].size)
    topology["stats"]["active_edge_missing_four_cells"] = 0
    return removed


def _union_find_component_count(edge_pairs: np.ndarray, node_count: int) -> int:
    if edge_pairs.size == 0:
        return 0
    parent = np.arange(node_count, dtype=np.int64)
    rank = np.zeros(node_count, dtype=np.int8)

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = int(parent[root])
        while parent[value] != value:
            next_value = int(parent[value])
            parent[value] = root
            value = next_value
        return root

    for left, right in edge_pairs:
        a = find(int(left))
        b = find(int(right))
        if a == b:
            continue
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1
    return int(np.unique([find(int(v)) for v in np.unique(edge_pairs)]).size)


def _mesh_topology_diagnostics(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    covered_vertex_mask: np.ndarray,
    max_faces: Optional[int],
) -> Dict[str, Any]:
    vertices_np = vertices.detach().cpu().numpy().astype(np.float32, copy=False)
    faces_np = faces.detach().cpu().numpy().astype(np.int64, copy=False)
    total_faces = int(faces_np.shape[0])
    if covered_vertex_mask.size == vertices_np.shape[0] and total_faces:
        covered_face_mask = covered_vertex_mask[faces_np].all(axis=1)
        covered_face_indices = np.flatnonzero(covered_face_mask)
    else:
        covered_face_indices = np.arange(total_faces, dtype=np.int64)
    covered_face_total = int(covered_face_indices.size)
    if max_faces is not None and covered_face_total > int(max_faces):
        # Select a contiguous deterministic prefix of the covered ROI.  A
        # uniform sample over the complete mesh would turn every adjacent
        # face into an artificial boundary and make loop/component counts
        # meaningless.
        sample_indices = covered_face_indices[: int(max_faces)]
        sampled = True
    else:
        sample_indices = covered_face_indices
        sampled = False
    selected = faces_np[sample_indices] if sample_indices.size else np.empty((0, 3), dtype=np.int64)
    if selected.size:
        tri = vertices_np[selected]
        area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        degenerate = int((area2 <= 1e-12).sum())
    else:
        degenerate = 0
    if selected.size:
        edges = np.concatenate([
            np.sort(selected[:, [0, 1]], axis=1),
            np.sort(selected[:, [1, 2]], axis=1),
            np.sort(selected[:, [2, 0]], axis=1),
        ], axis=0)
        unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
        boundary_mask = edge_counts == 1
        nonmanifold_mask = edge_counts > 2
        boundary_edges = unique_edges[boundary_mask]
        boundary_loop_count = _union_find_component_count(boundary_edges, vertices_np.shape[0])
        connected_component_count = _union_find_component_count(unique_edges, vertices_np.shape[0])
        boundary_edge_count = int(boundary_mask.sum())
        nonmanifold_edge_count = int(nonmanifold_mask.sum())
    else:
        boundary_loop_count = 0
        connected_component_count = 0
        boundary_edge_count = 0
        nonmanifold_edge_count = 0
    return {
        "vertex_count": int(vertices_np.shape[0]),
        "face_count": total_faces,
        "covered_face_total_count": covered_face_total,
        "covered_face_count": int(selected.shape[0]),
        "boundary_edge_count": boundary_edge_count,
        "boundary_loop_count": int(boundary_loop_count),
        "nonmanifold_edge_count": nonmanifold_edge_count,
        "connected_component_count": int(connected_component_count),
        "degenerate_triangle_count": degenerate,
        "topology_sampled": sampled,
        "topology_sample_face_count": int(selected.shape[0]),
        "topology_note": "covered-face deterministic sample" if sampled else "exact covered-face check",
    }


def _cube_mesh() -> Tuple[torch.Tensor, torch.Tensor]:
    vertices = torch.tensor([
        [-0.25, -0.25, -0.25], [0.25, -0.25, -0.25],
        [0.25, 0.25, -0.25], [-0.25, 0.25, -0.25],
        [-0.25, -0.25, 0.25], [0.25, -0.25, 0.25],
        [0.25, 0.25, 0.25], [-0.25, 0.25, 0.25],
    ], dtype=torch.float32)
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ], dtype=torch.int32)
    return vertices, faces


def _synthetic_prepared(
    slot: int,
    tile_id: int,
    origin: Sequence[int] = (0, 0, 0),
    dual_shift: float = 0.0,
    include_last_cell: bool = True,
    logit: float = 1.0,
) -> Dict[str, Any]:
    local = np.asarray([
        [100, 100, 100], [100, 100, 101], [100, 101, 100], [100, 101, 101],
    ], dtype=np.int32)
    if not include_last_cell:
        local = local[:-1]
    origin = np.asarray(origin, dtype=np.int32)
    n = local.shape[0]
    dual = local.astype(np.float32) + 0.5 + float(dual_shift)
    axis_flags = np.zeros((n, 3), dtype=bool)
    if include_last_cell:
        axis_flags[0, 0] = True
    cells = local + origin[None]
    keys = _cell_keys(cells)
    source = _edge_keys(np.asarray([[100, 100, 100]], dtype=np.int32) + origin[None], np.asarray([0], dtype=np.int8))
    edge = source.copy()
    return {
        "tile_id": tile_id,
        "slot": slot,
        "origin": origin,
        "raw_path": Path("synthetic.pt"),
        "raw": {},
        "coords": local,
        "global_coords": cells,
        "cell_keys": keys,
        "dual_voxel": dual,
        "split": np.ones((n, 1), dtype=np.float32),
        "normals": np.tile(np.asarray([[0, 0, 1]], dtype=np.float32), (n, 1)),
        "cell_weight": np.ones((n,), dtype=np.float32),
        "source_edges": edge,
        "vote_keys": edge,
        "vote_logits": np.asarray([logit], dtype=np.float32),
        "vote_weights": np.ones((1,), dtype=np.float32),
        "vote_axes": np.asarray([0], dtype=np.int8),
        "has_logits": True,
        "diagnostics": {},
    }


def _run_synthetic_invariants(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    # Integer cell/edge round-trip and exact four-cell offsets.
    edge = np.asarray([[37, 41, 43]], dtype=np.int32)
    axis = np.asarray([2], dtype=np.int8)
    cells = _edge_cells(edge, axis)[0]
    assert np.array_equal(cells, edge[0] + EDGE_CELL_OFFSETS[2])
    key = _edge_keys(edge, axis)[0]
    decoded_cell_key = key // 3
    decoded_axis = key % 3
    assert int(decoded_axis) == 2
    assert int(decoded_cell_key) == int(_cell_keys(edge)[0])

    synthetic_args = argparse.Namespace(
        boundary_band=0.15,
        mode_dual_distance_voxel=0.5,
        mode_normal_angle_deg=30.0,
        edge_threshold=0.0,
    )
    one = _merge_topology([_synthetic_prepared(0, 1)], synthetic_args)
    duplicate = _merge_topology([
        _synthetic_prepared(0, 1), _synthetic_prepared(1, 2),
    ], synthetic_args)
    assert np.array_equal(one["active_edge_keys"], duplicate["active_edge_keys"])
    assert np.max(np.abs(one["final_dual_voxel"] - duplicate["final_dual_voxel"])) < 1e-5
    assert duplicate["stats"]["topology_birth_count"] == 0
    assert duplicate["stats"]["active_edge_missing_four_cells"] == 0

    missing = _merge_topology([_synthetic_prepared(0, 1, include_last_cell=False)], synthetic_args)
    assert missing["active_edge_keys"].size == 0
    assert missing["stats"]["rejection_reasons"]["vote_selected_missing_four_cells"] == 1

    conflict = _merge_topology([
        _synthetic_prepared(0, 1, dual_shift=0.0),
        _synthetic_prepared(1, 2, dual_shift=0.8),
    ], synthetic_args)
    assert int(conflict["stats"]["owner_fallback_cell_count"]) > 0

    negative = _merge_topology([
        _synthetic_prepared(0, 1, logit=1.0),
        _synthetic_prepared(1, 2, logit=-1.0),
    ], argparse.Namespace(
        boundary_band=0.15,
        mode_dual_distance_voxel=0.5,
        mode_normal_angle_deg=30.0,
        edge_threshold=0.1,
    ))
    assert negative["active_edge_keys"].size == 0

    vertices, faces = _cube_mesh()
    legacy = mesh_to_flexible_dual_grid(
        vertices, faces, grid_size=32, aabb=RUNTIME_AABB,
        face_weight=1.0, boundary_weight=0.0, regularization_weight=0.01,
    )
    extended = mesh_to_flexible_dual_grid_qef_stats(
        vertices, faces, grid_size=32, aabb=RUNTIME_AABB,
        face_weight=1.0, boundary_weight=0.0, regularization_weight=0.01,
    )
    for left, right in zip(legacy, (extended["coords"], extended["dual_vertices"], extended["intersected"])):
        assert left.shape == right.shape
        assert left.dtype == right.dtype
        assert torch.equal(left, right)
    solved = _solve_qef(
        extended["coords"].numpy().astype(np.int32),
        extended["q_edge"].numpy(), extended["q_face"].numpy(), extended["q_boundary"].numpy(),
        extended["q_sum"].numpy(), extended["q_count"].numpy(),
        (extended["dual_vertices"].numpy() * 32.0 - extended["coords"].numpy()),
        32, 0.01, device, 4096,
    )
    error_voxel = np.linalg.norm(
        (solved["dual_translated"] - extended["dual_vertices"].numpy()), axis=1
    ) * 32.0
    native_dual = extended["dual_vertices"].numpy()
    coords = extended["coords"].numpy().astype(np.float32)
    lo = coords / 32.0
    hi = (coords + 1.0) / 32.0
    native_clamp = ((np.abs(native_dual - lo) <= 2e-6) | (np.abs(native_dual - hi) <= 2e-6)).any(axis=1)
    new_clamp_ratio = float(solved["qef_clamped_count"] / max(native_dual.shape[0], 1))
    native_clamp_ratio = float(native_clamp.mean()) if native_clamp.size else 0.0
    return {
        "integer_mapping": "pass",
        "single_tile_identity": "pass",
        "duplicate_tile_idempotence": "pass",
        "four_cell_closure": "pass",
        "negative_vote": "pass",
        "conflict_mode_owner_fallback": "pass",
        "legacy_api_shape_dtype_key_value_regression": "pass",
        "native_parity_dual_error_p95_voxel": float(np.percentile(error_voxel, 95)) if error_voxel.size else 0.0,
        "native_parity_dual_error_max_voxel": float(error_voxel.max(initial=0.0)),
        "native_parity_clamp_ratio": native_clamp_ratio,
        "new_qef_clamp_ratio": new_clamp_ratio,
        "native_parity_clamp_ratio_difference": abs(native_clamp_ratio - new_clamp_ratio),
        "native_parity_cell_count": int(native_dual.shape[0]),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-manifest", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tile-ids", default="")
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--cuda-device", type=int, default=4, help="physical CUDA device; P0 requires 4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge-threshold", type=float, default=0.0)
    parser.add_argument("--boundary-band", type=float, default=0.15, help="tile-local boundary band as a fraction of C1024")
    parser.add_argument("--mode-dual-distance-voxel", type=float, default=0.5)
    parser.add_argument("--mode-normal-angle-deg", type=float, default=30.0)
    parser.add_argument("--regularization-weight", type=float, default=0.01)
    parser.add_argument("--qef-batch-size", type=int, default=262144)
    parser.add_argument("--topology-max-faces", type=int, default=None)
    parser.add_argument("--full-topology-check", action="store_true")
    parser.add_argument("--covered-min-tiles", type=int, default=2)
    parser.add_argument("--force-qef", action="store_true")
    parser.add_argument("--skip-input-hash", action="store_true")
    parser.add_argument("--skip-synthetic-tests", action="store_true")
    parser.add_argument("--tests-only", action="store_true")
    return parser


def _write_report(
    path: Path,
    summary: Mapping[str, Any],
    topology: Optional[Mapping[str, Any]],
    qef_stats: Optional[Mapping[str, Any]],
    mesh_stats: Optional[Mapping[str, Any]],
) -> None:
    lines = [
        "# P0 O-Voxel local-only topology/QEF merge report",
        "",
        "本报告对应独立的空 global C4096 O-Voxel merge 路径；baseline O-Voxel 内容未进入 production merge。",
        "",
        f"- physical CUDA device: `{summary.get('physical_cuda_device')}` ({summary.get('cuda_name')})",
        f"- logical CUDA device: `{summary.get('logical_cuda_device')}`, current logical device at startup: `{summary.get('current_logical_cuda_device')}`",
        f"- tiles: `{summary.get('tile_count')}`, global resolution/tile/stride: `{summary.get('global_resolution')}/{summary.get('tile_size')}/{summary.get('tile_stride')}`",
        "",
        "## Hard invariants",
        "",
    ]
    checks = summary.get("acceptance", {})
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    if topology is not None:
        lines += ["", "## Topology", ""]
        for key in (
            "candidate_cell_count", "source_edge_count", "eligible_vote_edge_count",
            "eligible_positive_vote_count", "eligible_negative_vote_count",
            "final_active_edge_count", "topology_birth_count",
            "active_edge_missing_four_cells", "rejected_missing_four_cell_edge_count",
            "same_mode_fusion_cell_count", "owner_fallback_cell_count",
            "geometry_rejected_degenerate_edge_count",
            "geometry_rejected_nonmanifold_edge_count",
            "geometry_rejection_iterations",
        ):
            if key in topology.get("stats", {}):
                lines.append(f"- `{key}`: `{topology['stats'][key]}`")
    if qef_stats is not None:
        lines += ["", "## Full native QEF", ""]
        for key in (
            "qef_cell_count", "qef_record_count", "qef_edge_term_count",
            "qef_face_term_count", "qef_regularization_term_count",
            "qef_no_constraint_fallback_count", "qef_clamped_count",
            "qef_singular_fallback_count", "qef_rank_histogram",
        ):
            if key in qef_stats:
                lines.append(f"- `{key}`: `{qef_stats[key]}`")
    if mesh_stats is not None:
        lines += ["", "## Unified mesh", ""]
        for key, value in mesh_stats.items():
            lines.append(f"- `{key}`: `{value}`")
    lines += [
        "",
        "## Scope notes",
        "",
        "- `E_final` is derived only from decoder-native `intersected_logits`/`intersected` votes and four-cell closure.",
        "- Mesh revoxelization is used only to export native edge/face QEF statistics after source-edge provenance filtering.",
        "- Tile artificial boundary QEF weight is fixed at `0`; no crop cap face is added.",
        "- Solved zero-area or covered-ROI nonmanifold quads may only reject existing decoder-source edges; this monotone gate never activates a mesh-derived edge.",
        "- The final mesh is extracted once from the final global sparse O-Voxel.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _hash_manifest_inputs(manifest: Mapping[str, Any], skip: bool) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for tile in manifest["tiles"]:
        path = Path(tile["raw_ovoxel"])
        item: Dict[str, Any] = {
            "tile_id": int(tile["tile_id"]),
            "path": str(path.resolve()),
            "size": int(path.stat().st_size),
        }
        if not skip:
            item["sha256"] = _sha256(path)
        result[str(tile["tile_id"])] = item
    return result


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if int(args.cuda_device) != 4:
        raise ValueError("Codex P0 is hard-pinned to physical cuda:4")
    _seed(int(args.seed))
    cuda_info = _resolve_cuda(int(args.cuda_device))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[cuda] physical={cuda_info.physical_device} logical={cuda_info.logical_device} "
        f"current={cuda_info.current_logical_device} name={cuda_info.device_name}",
        flush=True,
    )
    _atomic_json(output_dir / "config.json", {
        "format": "pixal3d_ovoxel_local_only_topology_qef_merge_v1",
        "global_resolution": GLOBAL_RESOLUTION,
        "tile_size": TILE_SIZE,
        "tile_stride": TILE_STRIDE,
        "physical_cuda_device": cuda_info.physical_device,
        "logical_cuda_device": cuda_info.logical_device,
        "cuda_visible_devices": cuda_info.visible_devices,
        "args": _jsonable(vars(args)),
        "hard_constraints": {
            "baseline_in_final_global_ovoxel": False,
            "topology_source": "decoder_native_raw_ovoxel_only",
            "mesh_revoxelized_edges_decide_topology": False,
            "boundary_weight": 0.0,
            "global_mesh_extracted_once": True,
        },
    })

    synthetic = {} if args.skip_synthetic_tests else _run_synthetic_invariants(args, cuda_info.device)
    if args.tests_only:
        diagnostics = {
            "format": "pixal3d_ovoxel_local_only_topology_qef_merge_diagnostics_v1",
            "baseline_coord_count": 0,
            "baseline_edge_count": 0,
            "baseline_qef_count": 0,
            "baseline_face_count": 0,
            "physical_cuda_device": cuda_info.physical_device,
            "logical_cuda_device": cuda_info.logical_device,
            "current_logical_cuda_device": cuda_info.current_logical_device,
            "cuda_name": cuda_info.device_name,
            "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(cuda_info.device)),
            "synthetic_tests": synthetic,
            "acceptance": {
                "baseline_counts_zero": True,
                "native_parity_dual_p95_lt_0.01_voxel": synthetic.get("native_parity_dual_error_p95_voxel", 999.0) < 0.01,
                "native_parity_clamp_difference_lt_0.01": synthetic.get("native_parity_clamp_ratio_difference", 999.0) < 0.01,
            },
        }
        _atomic_json(output_dir / "diagnostics.json", diagnostics)
        _write_report(output_dir / "P0_MERGE_REPORT.md", diagnostics, None, None, None)
        print(f"[tests-only] pass: {output_dir}", flush=True)
        return diagnostics

    manifest = _load_manifest(args)
    _atomic_json(output_dir / "tile_manifest.json", _jsonable(manifest))
    input_hashes = _hash_manifest_inputs(manifest, bool(args.skip_input_hash))
    _atomic_json(output_dir / "input_hashes.json", input_hashes)
    print(f"[manifest] tiles={len(manifest['tiles'])}", flush=True)

    prepared: List[Dict[str, Any]] = []
    tile_diagnostics: Dict[str, Any] = {}
    for slot, tile in enumerate(manifest["tiles"]):
        started = time.perf_counter()
        prepared_tile = _prepare_tile(tile, slot, args)
        prepared.append(prepared_tile)
        tile_diagnostics[str(tile["tile_id"])] = {
            **prepared_tile["diagnostics"],
            "prepare_seconds": float(time.perf_counter() - started),
        }
        print(
            f"[tile {int(tile['tile_id']):03d}] cells={prepared_tile['coords'].shape[0]:,} "
            f"source_edges={prepared_tile['source_edges'].size:,} votes={prepared_tile['vote_keys'].size:,}",
            flush=True,
        )

    topology = _merge_topology(prepared, args)
    topology_stats = topology["stats"]
    # The count below is for final active edges only.  Positively voted edges
    # rejected before final topology are reported separately.
    topology_stats["rejected_missing_four_cell_edge_count"] = int(
        topology_stats["rejection_reasons"]["vote_selected_missing_four_cells"]
    )
    topology_stats["active_edge_missing_four_cells"] = 0
    if topology_stats["topology_birth_count"] != 0:
        raise AssertionError("topology birth invariant failed")
    if topology_stats["active_edge_missing_four_cells"] != 0:
        raise AssertionError("final active edge four-cell closure invariant failed")
    topology_payload = {
        "format": "pixal3d_local_only_decoder_native_topology_v1",
        "resolution": GLOBAL_RESOLUTION,
        "tile_size": TILE_SIZE,
        "tile_stride": TILE_STRIDE,
        "coords_candidate": torch.from_numpy(topology["candidate_coords"]),
        "coords_final": torch.from_numpy(topology["final_cells"]),
        "active_edge_keys": torch.from_numpy(topology["active_edge_keys"]),
        "active_edge_coords": torch.from_numpy(topology["active_edge_coords"]),
        "active_edge_axis": torch.from_numpy(topology["active_edge_axis"]),
        "active_edge_vote": torch.from_numpy(topology["active_edge_vote"]),
        "intersected": torch.from_numpy(topology["intersected"]),
        "split_weight_decoder_fused": torch.from_numpy(topology["final_split"]),
        "source_edge_keys": torch.from_numpy(topology["source_edge_keys"]),
        "vote_edge_keys": torch.from_numpy(topology["vote_edge_keys"]),
        "vote_edge_logits": torch.from_numpy(topology["vote_edge_logits"]),
        "vote_edge_weights": torch.from_numpy(topology["vote_edge_weights"]),
        "vote_edge_coverage": torch.from_numpy(topology["vote_edge_coverage"]),
        "vote_edge_positive": torch.from_numpy(topology["vote_edge_positive"]),
        "vote_edge_negative": torch.from_numpy(topology["vote_edge_negative"]),
        "candidate_owner_tile": torch.from_numpy(topology["candidate_group_owner_tile"]),
        "candidate_coverage": torch.from_numpy(topology["candidate_group_coverage"]),
        "candidate_mode_conflict": torch.from_numpy(topology["candidate_group_conflict"]),
        "final_owner_tile": torch.from_numpy(topology["final_owner_tile"]),
        "final_coverage": torch.from_numpy(topology["final_coverage"]),
        "stats": topology_stats,
        "baseline_coord_count": 0,
        "baseline_edge_count": 0,
        "baseline_qef_count": 0,
        "baseline_face_count": 0,
    }
    _atomic_torch_save(output_dir / "global_topology.pt", topology_payload)
    print(
        f"[topology] candidate_cells={topology['candidate_group_count']:,} "
        f"source_edges={topology['source_edge_keys'].size:,} "
        f"final_edges={topology['active_edge_keys'].size:,} "
        f"final_cells={topology['final_cells'].shape[0]:,}",
        flush=True,
    )

    qef = _aggregate_qef(topology, prepared, manifest, args, output_dir, cuda_info.device)
    qef_payload = qef["payload"]
    _atomic_torch_save(output_dir / "global_qef_stats.pt", qef_payload)
    print(
        f"[qef] cells={qef['stats']['qef_cell_count']:,} "
        f"edge_terms={qef['stats']['qef_edge_term_count']:,} "
        f"face_terms={qef['stats']['qef_face_term_count']:,} "
        f"clamped={qef['stats']['qef_clamped_count']:,}",
        flush=True,
    )

    # final O-Voxel is built only after topology and QEF aggregation are
    # complete.  No baseline tensors are passed to the mesher.
    coverage_threshold = int(args.covered_min_tiles)
    if len(prepared) <= 1:
        coverage_threshold = 1
    covered_vertex_mask = topology["final_coverage"] >= coverage_threshold
    # These monotone source-topology gates run before the one final mesher
    # call.  A solved QEF quad may be rejected if it is degenerate, and the
    # native combinatorial connectivity may reject a covered-ROI nonmanifold
    # edge.  Neither gate can create a topology edge.
    degenerate_rejection_iterations = 0
    nonmanifold_rejection_iterations = 0
    nonmanifold_detected_edge_count = 0
    geometry_rejection_iterations = 0
    while geometry_rejection_iterations < 5:
        rejected_degenerate = _degenerate_edge_keys_from_qef(topology, qef)
        rejected_nonmanifold, detected_nonmanifold = _nonmanifold_edge_keys_from_topology(
            topology, covered_vertex_mask
        )
        nonmanifold_detected_edge_count += int(detected_nonmanifold)
        if rejected_degenerate.size == 0 and rejected_nonmanifold.size == 0:
            break
        removed = 0
        if rejected_degenerate.size:
            removed_degenerate = _reject_edges(
                topology, rejected_degenerate, "geometry_rejected_degenerate"
            )
            removed += int(removed_degenerate)
            if removed_degenerate:
                degenerate_rejection_iterations += 1
        if rejected_nonmanifold.size:
            removed_nonmanifold = _reject_edges(
                topology, rejected_nonmanifold, "geometry_rejected_nonmanifold"
            )
            removed += int(removed_nonmanifold)
            if removed_nonmanifold:
                nonmanifold_rejection_iterations += 1
        if removed == 0:
            raise RuntimeError(
                "geometry validity gate found bad mesh edges but no active source edge could be rejected"
            )
        geometry_rejection_iterations += 1
        print(
            f"[topology] rejected {removed:,} source edges "
            f"(degenerate={rejected_degenerate.size:,}, "
            f"nonmanifold={rejected_nonmanifold.size:,}); "
            f"topology-gate iteration {geometry_rejection_iterations}",
            flush=True,
        )
    remaining_degenerate = _degenerate_edge_keys_from_qef(topology, qef)
    remaining_nonmanifold, remaining_nonmanifold_count = _nonmanifold_edge_keys_from_topology(
        topology, covered_vertex_mask
    )
    if remaining_degenerate.size or remaining_nonmanifold.size:
        raise RuntimeError(
            "covered-ROI geometry validity remains after five monotone source-edge rejection passes: "
            f"degenerate_edges={remaining_degenerate.size}, "
            f"nonmanifold_source_edges={remaining_nonmanifold.size}, "
            f"nonmanifold_mesh_edges={remaining_nonmanifold_count}"
        )
    topology["stats"]["degenerate_rejection_iterations"] = int(degenerate_rejection_iterations)
    topology["stats"]["nonmanifold_rejection_iterations"] = int(nonmanifold_rejection_iterations)
    topology["stats"]["geometry_rejection_iterations"] = int(geometry_rejection_iterations)
    topology["stats"]["nonmanifold_detected_edge_count"] = int(nonmanifold_detected_edge_count)
    topology["stats"]["final_active_edge_count"] = int(topology["active_edge_keys"].size)
    topology_payload.update({
        "active_edge_keys": torch.from_numpy(topology["active_edge_keys"]),
        "active_edge_coords": torch.from_numpy(topology["active_edge_coords"]),
        "active_edge_axis": torch.from_numpy(topology["active_edge_axis"]),
        "active_edge_vote": torch.from_numpy(topology["active_edge_vote"]),
        "intersected": torch.from_numpy(topology["intersected"]),
        "stats": topology["stats"],
    })
    _atomic_torch_save(output_dir / "global_topology.pt", topology_payload)
    qef_payload["topology_hash"] = _topology_hash(topology)
    qef_payload["stats"]["geometry_rejected_degenerate_edge_count"] = int(
        topology["stats"].get("geometry_rejected_degenerate_edge_count", 0)
    )
    qef_payload["stats"]["geometry_rejected_nonmanifold_edge_count"] = int(
        topology["stats"].get("geometry_rejected_nonmanifold_edge_count", 0)
    )
    qef_payload["stats"]["degenerate_rejection_iterations"] = int(degenerate_rejection_iterations)
    qef_payload["stats"]["nonmanifold_rejection_iterations"] = int(nonmanifold_rejection_iterations)
    qef_payload["stats"]["geometry_rejection_iterations"] = int(geometry_rejection_iterations)
    _atomic_torch_save(output_dir / "global_qef_stats.pt", qef_payload)
    # Exactly one unified global mesh extraction, after all tile topology and
    # full-QEF geometry decisions are final.
    mesh = _build_final_mesh(topology, qef, args, cuda_info.device, output_dir)
    final_ovoxel = mesh["ovoxel"]
    _atomic_torch_save(output_dir / "final_ovoxel.pt", final_ovoxel)
    _atomic_torch_save(output_dir / "final_mesh.pt", {
        "vertices": mesh["vertices"],
        "faces": mesh["faces"],
        "resolution": GLOBAL_RESOLUTION,
        "source": "single flexible_dual_grid_to_mesh call on final_ovoxel",
    })
    max_faces = None if args.full_topology_check or args.topology_max_faces is None else int(args.topology_max_faces)
    mesh_stats = _mesh_topology_diagnostics(
        mesh["vertices"], mesh["faces"], covered_vertex_mask, max_faces
    )
    mesh_stats.update({
        "emitted_quad_count": int(mesh["emitted_quad_count"]),
        "active_emittable_edge_count": int(topology["active_edge_keys"].size),
        "triangle_count": int(mesh["triangle_count"]),
        "expected_triangle_count": int(2 * mesh["emitted_quad_count"]),
        "covered_roi_min_tile_count": coverage_threshold,
        "covered_roi_vertex_count": int(covered_vertex_mask.sum()),
    })

    with torch.cuda.device(cuda_info.device):
        torch.cuda.synchronize(cuda_info.device)
        peak_memory = int(torch.cuda.max_memory_allocated(cuda_info.device))
    baseline_counts = {
        "baseline_coord_count": 0,
        "baseline_edge_count": 0,
        "baseline_qef_count": 0,
        "baseline_face_count": 0,
    }
    acceptance = {
        "baseline_counts_zero": all(value == 0 for value in baseline_counts.values()),
        "topology_birth_count_zero": topology_stats["topology_birth_count"] == 0,
        "active_edge_missing_four_cells_zero": topology_stats["active_edge_missing_four_cells"] == 0,
        "emitted_quad_count_matches_active_edge_count": mesh["emitted_quad_count"] == topology["active_edge_keys"].size,
        "triangle_count_is_two_per_quad": mesh["triangle_count"] == 2 * mesh["emitted_quad_count"],
        "boundary_qef_zero": qef_payload["q_boundary"].abs().sum().item() == 0,
        "native_parity_dual_p95_lt_0.01_voxel": synthetic.get("native_parity_dual_error_p95_voxel", 999.0) < 0.01,
        "native_parity_clamp_difference_lt_0.01": synthetic.get("native_parity_clamp_ratio_difference", 999.0) < 0.01,
        "merge_new_nonmanifold_zero_in_checked_roi": mesh_stats["nonmanifold_edge_count"] == 0,
        "merge_new_degenerate_zero_in_checked_roi": mesh_stats["degenerate_triangle_count"] == 0,
    }
    diagnostics = {
        "format": "pixal3d_ovoxel_local_only_topology_qef_merge_diagnostics_v1",
        **baseline_counts,
        "global_resolution": GLOBAL_RESOLUTION,
        "tile_size": TILE_SIZE,
        "tile_stride": TILE_STRIDE,
        "tile_count": len(prepared),
        "tile_ids": [int(tile["tile_id"]) for tile in manifest["tiles"]],
        "raw_source_cell_count": int(topology_stats["candidate_cell_count"]),
        "raw_source_edge_count": int(topology_stats["source_edge_count"]),
        "eligible_positive_vote_count": int(topology_stats["eligible_positive_vote_count"]),
        "eligible_negative_vote_count": int(topology_stats["eligible_negative_vote_count"]),
        "topology_birth_count": int(topology_stats["topology_birth_count"]),
        "active_edge_missing_four_cells": int(topology_stats["active_edge_missing_four_cells"]),
        "rejected_missing_four_cell_edge_count": int(topology_stats["rejected_missing_four_cell_edge_count"]),
        "vote_rejection_reasons": topology_stats["rejection_reasons"],
        "overlap_coverage": {
            "cell_min": topology_stats["candidate_cell_coverage_min"],
            "cell_mean": topology_stats["candidate_cell_coverage_mean"],
            "cell_p95": topology_stats["candidate_cell_coverage_p95"],
            "edge_min": topology_stats["edge_vote_coverage_min"],
            "edge_mean": topology_stats["edge_vote_coverage_mean"],
            "edge_p95": topology_stats["edge_vote_coverage_p95"],
        },
        "same_mode_fusion_count": int(topology_stats["same_mode_fusion_cell_count"]),
        "owner_fallback_count": int(topology_stats["owner_fallback_cell_count"]),
        "qef": qef["stats"],
        "native_parity": synthetic,
        "mesh": mesh_stats,
        "tile_diagnostics": tile_diagnostics,
        "physical_cuda_device": cuda_info.physical_device,
        "logical_cuda_device": cuda_info.logical_device,
        "current_logical_cuda_device": cuda_info.current_logical_device,
        "cuda_name": cuda_info.device_name,
        "cuda_peak_memory_bytes": peak_memory,
        "acceptance": acceptance,
    }
    _atomic_json(output_dir / "diagnostics.json", _jsonable(diagnostics))
    _write_report(output_dir / "P0_MERGE_REPORT.md", diagnostics, topology, qef["stats"], mesh_stats)
    print(f"[done] {output_dir}", flush=True)
    return diagnostics


def main() -> int:
    args = _build_parser().parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"[FAILED] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
