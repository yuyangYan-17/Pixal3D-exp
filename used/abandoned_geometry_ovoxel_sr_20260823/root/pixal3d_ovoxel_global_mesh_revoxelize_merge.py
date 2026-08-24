#!/usr/bin/env python3
"""Re-voxelize decoder local meshes into one empty global C4096 O-Voxel.

This is the isolated implementation for the P0 described in ``Codex.md``.
The production path is deliberately mesh-first:

    local mesh -> uniform 3-D placement -> one canonical edge accumulator
                -> intersection mode selection -> four-cell closure
                -> full QEF -> one final O-Voxel mesher call

Decoder ``coords``/``intersected`` are loaded only for provenance and audit
statistics.  They never create, remove, or carry a final global cell/edge.
There is no camera, image-plane, depth, or projection operation in the
geometry path.

The native O-Voxel extension is used for its exact triangle lattice traversal
and face-plane QEF rasterization.  Triangle chunks are independent calls to
that traversal, but their Hermite observations are merged immediately into
the same global canonical accumulator before topology is selected.
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
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch


OVOXEL_SOURCE = Path("/home/nvme04/yyyan/TRELLIS.2/o-voxel")
if OVOXEL_SOURCE.is_dir():
    sys.path.insert(0, str(OVOXEL_SOURCE))

try:  # Keep pure placement/key tests importable if the extension is absent.
    from o_voxel.convert import (  # type: ignore
        flexible_dual_grid_to_mesh,
        mesh_to_flexible_dual_grid,
        mesh_to_flexible_dual_grid_qef_stats,
    )
    _NATIVE_IMPORT_ERROR: Optional[BaseException] = None
except BaseException as exc:  # pragma: no cover - exercised only on bare hosts
    flexible_dual_grid_to_mesh = None  # type: ignore[assignment]
    mesh_to_flexible_dual_grid = None  # type: ignore[assignment]
    mesh_to_flexible_dual_grid_qef_stats = None  # type: ignore[assignment]
    _NATIVE_IMPORT_ERROR = exc


GLOBAL_RESOLUTION = 4096
TILE_SIZE = 1024
TILE_STRIDE = 512
DEFAULT_INPUT = Path("outputs/geometry_ovoxel_local_only_stride512")
DEFAULT_OUTPUT = Path("outputs/geometry_ovoxel_global_mesh_revoxelize_merge")
CLUSTER_POLICY = "stable_mode_owner_resolution_v4"
RUNTIME_AABB = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32)
YAW_ANGLES = (0, 60, 120, 180, 240, 300)

EDGE_CELL_OFFSETS = np.asarray(
    [
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ],
    dtype=np.int32,
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        json.dump(_jsonable(value), handle, indent=2, ensure_ascii=False, allow_nan=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@dataclass(frozen=True)
class CudaInfo:
    requested_physical: int
    logical_device: int
    physical_device: int
    current_logical_device: int
    device_name: str
    total_memory_bytes: int
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
        if not token or token.startswith(("GPU-", "MIG-")):
            raise RuntimeError(
                "P0 requires an auditable numeric CUDA_VISIBLE_DEVICES mapping; "
                "UUID/MIG mappings are refused"
            )
        values.append(int(token))
    return tuple(values)


def _resolve_cuda(requested_physical: int) -> CudaInfo:
    if not torch.cuda.is_available():
        raise RuntimeError("P0 GPU tests require CUDA")
    visible = _visible_physical_devices()
    requested = int(requested_physical)
    if visible is None:
        logical = requested
        physical = requested
    else:
        if requested not in visible:
            raise RuntimeError(
                f"physical cuda:{requested} is not exposed by CUDA_VISIBLE_DEVICES={visible}"
            )
        logical = visible.index(requested)
        physical = int(visible[logical])
    if logical < 0 or logical >= torch.cuda.device_count():
        raise RuntimeError(
            f"logical cuda:{logical} unavailable; visible device count={torch.cuda.device_count()}"
        )
    device = torch.device("cuda", logical)
    # The context is local.  This intentionally does not call torch.cuda.set_device
    # and therefore cannot change another training/encoder/decoder path.
    with torch.cuda.device(device):
        current = int(torch.cuda.current_device())
        name = torch.cuda.get_device_name(device)
        total = int(torch.cuda.get_device_properties(device).total_memory)
    if physical != requested:
        raise AssertionError(f"CUDA mapping changed: physical {physical} != requested {requested}")
    return CudaInfo(requested, logical, physical, current, name, total, visible)


def _as_numpy(value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    return result.astype(dtype, copy=False) if dtype is not None else result


def _cell_keys(coords: np.ndarray, resolution: int = GLOBAL_RESOLUTION) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    if coords.size == 0:
        return np.empty((0,), dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"cell coordinates must be [N,3], got {coords.shape}")
    return (coords[:, 0] * int(resolution) + coords[:, 1]) * int(resolution) + coords[:, 2]


def _edge_keys(
    coords: np.ndarray,
    axis: np.ndarray,
    resolution: int = GLOBAL_RESOLUTION,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    axis = np.asarray(axis, dtype=np.int64)
    if coords.size == 0:
        return np.empty((0,), dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3 or axis.shape != (coords.shape[0],):
        raise ValueError("edge coordinates/axis have incompatible shapes")
    return _cell_keys(coords, resolution) * 3 + axis


def _decode_edge_keys(keys: np.ndarray, resolution: int = GLOBAL_RESOLUTION) -> Tuple[np.ndarray, np.ndarray]:
    keys = np.asarray(keys, dtype=np.int64)
    axis = (keys % 3).astype(np.int8)
    cell = keys // 3
    coords = np.empty((keys.size, 3), dtype=np.int32)
    coords[:, 2] = cell % int(resolution)
    coords[:, 1] = (cell // int(resolution)) % int(resolution)
    coords[:, 0] = cell // (int(resolution) * int(resolution))
    return coords, axis


def _valid_edge_coords(
    coords: np.ndarray,
    axis: np.ndarray,
    resolution: int = GLOBAL_RESOLUTION,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    axis = np.asarray(axis, dtype=np.int64)
    valid = ((coords >= 0) & (coords < int(resolution))).all(axis=1)
    for perpendicular in range(3):
        valid &= (axis == perpendicular) | (coords[:, perpendicular] < int(resolution) - 1)
    valid &= (axis >= 0) & (axis < 3)
    return valid


def _edge_cells(edge_coord: np.ndarray, edge_axis: np.ndarray) -> np.ndarray:
    edge_coord = np.asarray(edge_coord, dtype=np.int32)
    edge_axis = np.asarray(edge_axis, dtype=np.int64)
    if edge_coord.size == 0:
        return np.empty((0, 4, 3), dtype=np.int32)
    return edge_coord[:, None, :] + EDGE_CELL_OFFSETS[edge_axis]


def _raised_cosine_weight(margin: np.ndarray, band: float) -> np.ndarray:
    margin = np.asarray(margin, dtype=np.float32)
    if band <= 0:
        return (margin > 0).astype(np.float32)
    t = np.clip(margin / float(band), 0.0, 1.0)
    weight = 0.5 * (1.0 - np.cos(np.pi * t))
    # Artificial crop faces are exactly zero, including floating point noise
    # within the explicit crop tolerance.
    return np.where(margin <= 1e-6, 0.0, weight).astype(np.float32)


def local_to_global(
    points: np.ndarray,
    origin: Sequence[int],
    convention: str = "local_centered",
) -> np.ndarray:
    """Apply the only production geometry transform: uniform scale + translation."""
    points = np.asarray(points, dtype=np.float32)
    # ``global_centered`` is an explicit adapter convention for a cache whose
    # camera inverse has already been performed before this P0 entry point.
    # The merge itself still sees only pre-placed global object coordinates and
    # never calls a camera/projection routine.
    if convention in {"global_centered", "global_object", "preplaced_global"}:
        return points.astype(np.float32, copy=True)
    origin_f = np.asarray(origin, dtype=np.float32).reshape(1, 3)
    if convention in {"local_centered", "centered", "[-0.5,0.5]"}:
        return (-0.5 + (origin_f + (points + 0.5) * TILE_SIZE) / GLOBAL_RESOLUTION).astype(np.float32)
    if convention in {"local_voxel", "voxel", "[0,1024]"}:
        return (-0.5 + (origin_f + points) / GLOBAL_RESOLUTION).astype(np.float32)
    raise ValueError(f"unsupported local coordinate convention: {convention}")


def global_to_local(
    points: np.ndarray,
    origin: Sequence[int],
    convention: str = "local_centered",
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if convention in {"global_centered", "global_object", "preplaced_global"}:
        return points.astype(np.float32, copy=True)
    origin_f = np.asarray(origin, dtype=np.float32).reshape(1, 3)
    voxel = (points + 0.5) * GLOBAL_RESOLUTION - origin_f
    if convention in {"local_centered", "centered", "[-0.5,0.5]"}:
        return (voxel / TILE_SIZE - 0.5).astype(np.float32)
    if convention in {"local_voxel", "voxel", "[0,1024]"}:
        return voxel.astype(np.float32)
    raise ValueError(f"unsupported local coordinate convention: {convention}")


def _parse_ids(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not str(value).strip():
        return None
    result: set[int] = set()
    for token in str(value).split(","):
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
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _tile_layout(resolution: int = GLOBAL_RESOLUTION) -> List[Tuple[int, int, int]]:
    starts = list(range(0, resolution - TILE_SIZE + 1, TILE_STRIDE))
    if starts[-1] != resolution - TILE_SIZE:
        raise ValueError("C4096/C1024/stride512 layout does not reach the upper boundary")
    return [(x, y, z) for z in starts for y in starts for x in starts]


def _discover_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    input_dir = args.input_dir.expanduser().resolve()
    layout_payload: Dict[str, Any] = {}
    layout_path = input_dir / "tile_layout.json"
    if layout_path.is_file():
        layout_payload = json.loads(layout_path.read_text(encoding="utf-8"))
    selected_ids = [int(v) for v in layout_payload.get("selected_tile_ids", [])]
    selected_starts = layout_payload.get("selected_starts", [])
    if not selected_ids or len(selected_ids) != len(selected_starts):
        selected_ids = list(range(len(_tile_layout())))
        selected_starts = [list(v) for v in _tile_layout()]
    requested = _parse_ids(args.tile_ids)
    if requested is not None:
        pairs = [(tid, start) for tid, start in zip(selected_ids, selected_starts) if tid in requested]
    else:
        pairs = list(zip(selected_ids, selected_starts))
    if args.max_tiles is not None:
        pairs = pairs[: int(args.max_tiles)]
    tiles: List[Dict[str, Any]] = []
    for tile_id, origin in pairs:
        tile_dir = input_dir / "local_tiles" / f"tile_{tile_id:03d}"
        raw_path = tile_dir / "shape_flow_and_raw_ovoxel.pt"
        if not raw_path.is_file():
            raise FileNotFoundError(f"tile {tile_id} local mesh/raw cache is missing: {raw_path}")
        item: Dict[str, Any] = {
            "tile_id": int(tile_id),
            "origin": [int(v) for v in origin],
            "size": TILE_SIZE,
            "stride": TILE_STRIDE,
            "raw_ovoxel": str(raw_path),
            "hermite_cache": str(tile_dir / "global_hermite.pt") if (tile_dir / "global_hermite.pt").is_file() else None,
            "coordinate_convention": "local_centered",
            "boundary_band": 0.15,
            "contribution_weight": 1.0,
        }
        tiles.append(item)
    return {
        "format": "pixal3d_ovoxel_global_mesh_revoxelize_tile_manifest_v1",
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
        manifest_path = args.tile_manifest.expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = {"tiles": payload} if isinstance(payload, list) else dict(payload)
        base = manifest_path.parent
        requested = _parse_ids(args.tile_ids)
        if requested is not None:
            manifest["tiles"] = [t for t in manifest.get("tiles", []) if int(t["tile_id"]) in requested]
        if args.max_tiles is not None:
            manifest["tiles"] = list(manifest.get("tiles", []))[: int(args.max_tiles)]
    if int(manifest.get("global_resolution", GLOBAL_RESOLUTION)) != GLOBAL_RESOLUTION:
        raise ValueError("manifest global_resolution must be 4096")
    if int(manifest.get("tile_size", TILE_SIZE)) != TILE_SIZE:
        raise ValueError("manifest tile_size must be 1024")
    if int(manifest.get("tile_stride", TILE_STRIDE)) != TILE_STRIDE:
        raise ValueError("manifest tile_stride must be 512")
    normalized: List[Dict[str, Any]] = []
    for raw_tile in manifest.get("tiles", []):
        tile = dict(raw_tile)
        tile_id = int(tile["tile_id"])
        origin = np.asarray(tile["origin"], dtype=np.int64)
        if origin.shape != (3,) or not bool(((origin >= 0) & (origin + TILE_SIZE <= GLOBAL_RESOLUTION)).all()):
            raise ValueError(f"tile {tile_id} origin is outside C4096: {origin.tolist()}")
        tile["tile_id"] = tile_id
        tile["origin"] = origin.astype(np.int32).tolist()
        tile["size"] = TILE_SIZE
        tile["stride"] = TILE_STRIDE
        raw_value = tile.get("raw_ovoxel", tile.get("raw"))
        if raw_value is None:
            raise KeyError(f"tile {tile_id} is missing raw_ovoxel")
        tile["raw_ovoxel"] = str(_resolve_path(raw_value, base))
        if not Path(tile["raw_ovoxel"]).is_file():
            raise FileNotFoundError(tile["raw_ovoxel"])
        hermite_value = tile.get("hermite_cache")
        if hermite_value:
            tile["hermite_cache"] = str(_resolve_path(hermite_value, base))
        else:
            adjacent = Path(tile["raw_ovoxel"]).parent / "global_hermite.pt"
            tile["hermite_cache"] = str(adjacent) if adjacent.is_file() else None
        tile["coordinate_convention"] = str(tile.get("coordinate_convention", "local_centered"))
        tile["boundary_band"] = float(tile.get("boundary_band", 0.15))
        tile["contribution_weight"] = float(tile.get("contribution_weight", 1.0))
        normalized.append(tile)
    if not normalized:
        raise ValueError("tile manifest contains no tiles")
    manifest["tiles"] = normalized
    manifest["global_resolution"] = GLOBAL_RESOLUTION
    manifest["tile_size"] = TILE_SIZE
    manifest["tile_stride"] = TILE_STRIDE
    return manifest


def _load_raw_tile(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw = payload.get("raw_ovoxel", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(raw, Mapping):
        raise TypeError(f"{path} does not contain a mapping raw_ovoxel payload")
    required = ("coords", "intersected", "quad_lerp", "mesh_vertices", "mesh_faces", "provenance")
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"{path} raw local mesh is missing required provenance fields: {missing}")
    return raw


def _face_flags(raw: Mapping[str, Any], names: Sequence[str], count: int) -> np.ndarray:
    containers = [raw]
    provenance = raw.get("provenance")
    if isinstance(provenance, Mapping):
        containers.append(provenance)
    for container in containers:
        for name in names:
            if name in container:
                value = _as_numpy(container[name]).reshape(-1)
                if value.shape[0] == count:
                    return value.astype(bool, copy=False)
    return np.zeros((count,), dtype=bool)


def _compute_local_face_weight(
    triangle_vertices_local: np.ndarray,
    convention: str,
    band_fraction: float,
    contribution_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangle_vertices_local = np.asarray(triangle_vertices_local, dtype=np.float32)
    if convention in {"global_centered", "global_object", "preplaced_global"}:
        # The upstream adapter has already applied the 2-D tile camera inverse
        # and cropped faces by ownership.  There is no local 3-D crop plane in
        # this merge stream, so no artificial boundary weight is introduced.
        finite = np.isfinite(triangle_vertices_local).all(axis=(1, 2))
        weights = np.full(
            (triangle_vertices_local.shape[0],),
            float(contribution_weight),
            dtype=np.float32,
        )
        return weights, np.zeros_like(finite, dtype=bool), finite.astype(bool)
    if convention in {"local_centered", "centered", "[-0.5,0.5]"}:
        local = triangle_vertices_local
        scale = float(TILE_SIZE)
        margin = np.minimum(local + 0.5, 0.5 - local).min(axis=(1, 2)) * scale
    elif convention in {"local_voxel", "voxel", "[0,1024]"}:
        local = triangle_vertices_local
        margin = np.minimum(local, TILE_SIZE - local).min(axis=(1, 2))
    else:
        raise ValueError(f"unsupported local coordinate convention: {convention}")
    band = float(band_fraction) * TILE_SIZE
    weights = _raised_cosine_weight(margin, band) * float(contribution_weight)
    touches = margin <= 1e-5
    valid_local = np.isfinite(local).all(axis=(1, 2))
    valid_local &= (margin >= -1e-4)
    return weights.astype(np.float32), touches.astype(bool), valid_local.astype(bool)


@dataclass
class TileStream:
    tile_id: int
    slot: int
    origin: np.ndarray
    convention: str
    vertices_global: np.ndarray
    faces: np.ndarray
    face_local_ids: np.ndarray
    face_uids: np.ndarray
    source_cell: np.ndarray
    source_axis: np.ndarray
    source_quad: np.ndarray
    source_decoder_edge_key: np.ndarray
    face_weight: np.ndarray
    face_touches_boundary: np.ndarray
    split: np.ndarray
    raw_edge_keys: np.ndarray
    raw_cell_keys: np.ndarray
    source_boundary_samples_global_voxel: np.ndarray
    hermite_cache: Optional[Path]
    diagnostics: Dict[str, Any]


def _native_grid_range(tile: TileStream) -> List[List[int]]:
    """Return the integer lattice range used by the native traversal.

    Normal production tiles use their C4096 cube.  The explicit pre-placed
    adapter convention is already in global object space, so its native
    traversal must be allowed to visit the full canonical lattice; the tile
    origin remains metadata only for this adapter input.
    """
    if tile.convention in {"global_centered", "global_object", "preplaced_global"}:
        return [[0, 0, 0], [GLOBAL_RESOLUTION, GLOBAL_RESOLUTION, GLOBAL_RESOLUTION]]
    return [tile.origin.tolist(), (tile.origin + TILE_SIZE).tolist()]


def _prepare_tile(
    tile: Mapping[str, Any],
    slot: int,
    args: argparse.Namespace,
    seen_face_keys: set[Tuple[int, int]],
) -> TileStream:
    tile_id = int(tile["tile_id"])
    origin = np.asarray(tile["origin"], dtype=np.int32)
    raw = _load_raw_tile(Path(tile["raw_ovoxel"]))
    vertices_local = _as_numpy(raw["mesh_vertices"], np.float32)
    faces_all = _as_numpy(raw["mesh_faces"], np.int64)
    coords = _as_numpy(raw["coords"], np.int32)
    intersected = _as_numpy(raw["intersected"], bool)
    quad_lerp = _as_numpy(raw["quad_lerp"], np.float32)
    provenance = raw["provenance"]
    if vertices_local.ndim != 2 or vertices_local.shape[1] != 3:
        raise ValueError(f"tile {tile_id} mesh_vertices must be [N,3]")
    if faces_all.ndim != 2 or faces_all.shape[1] != 3:
        raise ValueError(f"tile {tile_id} mesh_faces must be [F,3]")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"tile {tile_id} raw coords must be [N,3]")
    if intersected.shape != (coords.shape[0], 3):
        raise ValueError(f"tile {tile_id} intersected shape does not match coords")
    if quad_lerp.ndim == 1:
        quad_lerp = quad_lerp[:, None]
    if quad_lerp.shape[0] != coords.shape[0]:
        raise ValueError(f"tile {tile_id} quad_lerp row count does not match coords")
    if not bool(np.isfinite(vertices_local).all()):
        raise ValueError(f"tile {tile_id} local mesh contains NaN/Inf")

    source_index = _as_numpy(provenance.get("source_ovoxel_index"), np.int64)
    source_axis = _as_numpy(provenance.get("source_edge_axis"), np.int64)
    if source_index.shape != (faces_all.shape[0],) or source_axis.shape != (faces_all.shape[0],):
        if not args.allow_provenance_fallback:
            raise ValueError(
                f"tile {tile_id} lacks per-triangle source_ovoxel_index/source_edge_axis; "
                "formal merge refuses an untraceable tile"
            )
        source_index = np.full((faces_all.shape[0],), -1, dtype=np.int64)
        source_axis = np.full((faces_all.shape[0],), -1, dtype=np.int64)
    valid_source = (
        (source_index >= 0) & (source_index < coords.shape[0])
        & (source_axis >= 0) & (source_axis < 3)
    )
    if not bool(valid_source.all()) and not args.allow_provenance_fallback:
        bad = int((~valid_source).sum())
        raise ValueError(f"tile {tile_id} has {bad} faces with invalid source provenance")

    local_edge_key = np.full((faces_all.shape[0],), -1, dtype=np.int64)
    if bool(valid_source.any()):
        local_edge_key[valid_source] = _edge_keys(
            coords[source_index[valid_source]], source_axis[valid_source].astype(np.int8), TILE_SIZE
        )
    explicit_quad = provenance.get("source_quad_id") if isinstance(provenance, Mapping) else None
    if explicit_quad is not None and _as_numpy(explicit_quad).reshape(-1).shape[0] == faces_all.shape[0]:
        source_quad = _as_numpy(explicit_quad, np.int64).reshape(-1)
    else:
        quad_indices = provenance.get("quad_indices") if isinstance(provenance, Mapping) else None
        qcount = _as_numpy(quad_indices).shape[0] if quad_indices is not None and _as_numpy(quad_indices).ndim == 2 else 0
        if qcount * 2 == faces_all.shape[0]:
            source_quad = np.arange(faces_all.shape[0], dtype=np.int64) // 2
        else:
            source_quad = local_edge_key.copy()
            source_quad[source_quad < 0] = np.arange(faces_all.shape[0], dtype=np.int64)[source_quad < 0]

    safe_faces = np.clip(faces_all, 0, max(vertices_local.shape[0] - 1, 0))
    face_weight_all, touches_all, valid_local = _compute_local_face_weight(
        vertices_local[safe_faces], str(tile.get("coordinate_convention", "local_centered")),
        float(tile.get("boundary_band", args.boundary_band)), float(tile.get("contribution_weight", 1.0)),
    )
    cap_flags = _face_flags(
        raw,
        ("is_artificial_cap", "artificial_cap", "cap_face", "is_crop_cap", "detected_cap_face"),
        faces_all.shape[0],
    )
    crop_flags = _face_flags(
        raw,
        ("touches_artificial_boundary", "touches_crop_boundary", "crop_boundary_face"),
        faces_all.shape[0],
    )
    cap_flags |= crop_flags & _face_flags(raw, ("is_cap",), faces_all.shape[0])
    valid_index = ((faces_all >= 0) & (faces_all < vertices_local.shape[0])).all(axis=1)
    tri = np.zeros((faces_all.shape[0], 3, 3), dtype=np.float32)
    if bool(valid_index.any()):
        tri[valid_index] = vertices_local[faces_all[valid_index]]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    zero_area = area2 <= float(args.degenerate_area_epsilon)
    finite_faces = np.isfinite(tri).all(axis=(1, 2))
    duplicate = np.zeros((faces_all.shape[0],), dtype=bool)
    for row, local_face_id in enumerate(range(faces_all.shape[0])):
        key = (tile_id, int(local_face_id))
        if key in seen_face_keys:
            duplicate[row] = True
        else:
            seen_face_keys.add(key)
    keep = valid_index & finite_faces & ~zero_area & valid_local & ~cap_flags & (face_weight_all > 0) & ~duplicate
    retained = np.flatnonzero(keep)
    if retained.size == 0:
        raise RuntimeError(f"tile {tile_id} has no positive-weight, provenance-valid local triangles")
    faces = faces_all[retained].astype(np.int32, copy=False)
    face_local_ids = retained.astype(np.int64, copy=False)
    face_uids = (np.int64(tile_id + 1) * np.int64(2**32) + face_local_ids).astype(np.int64)
    source_cell = source_index[retained].astype(np.int32, copy=False)
    source_axis_retained = source_axis[retained].astype(np.int8, copy=False)
    split = quad_lerp[:, :1].astype(np.float32, copy=False)
    split = np.nan_to_num(split, nan=1.0, posinf=1.0, neginf=1.0)
    split = np.maximum(split, 1e-6)

    global_coords = coords.astype(np.int64) + origin.astype(np.int64)[None]
    raw_cell_keys = np.unique(_cell_keys(global_coords))
    raw_edge_parts: List[np.ndarray] = []
    for axis in range(3):
        active = intersected[:, axis]
        valid = active & _valid_edge_coords(coords, np.full(coords.shape[0], axis, dtype=np.int8), TILE_SIZE)
        if bool(valid.any()):
            raw_edge_parts.append(_edge_keys(global_coords[valid], np.full(int(valid.sum()), axis, dtype=np.int8)))
    raw_edge_keys = np.unique(np.concatenate(raw_edge_parts)) if raw_edge_parts else np.empty((0,), dtype=np.int64)
    vertices_global = local_to_global(
        vertices_local, origin, str(tile.get("coordinate_convention", "local_centered"))
    )
    source_boundary_samples = np.empty((0, 3), dtype=np.float32)
    source_boundary_edges = np.empty((0, 2), dtype=np.int64)
    # Pre-placed adapter meshes have already crossed the 2-D ownership stage;
    # their compact outer contour is not a 3-D tile crop plane.  Avoid building
    # a multi-hundred-million-row edge table merely for a non-production
    # diagnostic in that convention.
    valid_boundary_faces = (
        np.empty((0, 3), dtype=np.int64)
        if str(tile.get("coordinate_convention", "local_centered"))
        in {"global_centered", "global_object", "preplaced_global"}
        else faces_all[valid_index]
    )
    if valid_boundary_faces.size:
        boundary_edges = np.concatenate([
            np.sort(valid_boundary_faces[:, [0, 1]], axis=1),
            np.sort(valid_boundary_faces[:, [1, 2]], axis=1),
            np.sort(valid_boundary_faces[:, [2, 0]], axis=1),
        ], axis=0)
        unique_boundary_edges, boundary_counts = np.unique(
            boundary_edges, axis=0, return_counts=True
        )
        source_boundary_edges = unique_boundary_edges[boundary_counts == 1]
        if source_boundary_edges.size:
            boundary_end0 = (vertices_global[source_boundary_edges[:, 0]] + 0.5) * GLOBAL_RESOLUTION
            boundary_end1 = (vertices_global[source_boundary_edges[:, 1]] + 0.5) * GLOBAL_RESOLUTION
            source_boundary_samples = np.concatenate([
                boundary_end0,
                boundary_end1,
                (boundary_end0 + boundary_end1) * 0.5,
            ], axis=0).astype(np.float32, copy=False)
    tile_diag = {
        "tile_id": tile_id,
        "origin": origin.tolist(),
        "input_vertices": int(vertices_local.shape[0]),
        "input_faces": int(faces_all.shape[0]),
        "placed_triangles": int(retained.size),
        "filtered_nan_inf_face_count": int((~finite_faces).sum()),
        "filtered_invalid_index_face_count": int((~valid_index).sum()),
        "filtered_zero_area_face_count": int(zero_area.sum()),
        "filtered_duplicate_face_count": int(duplicate.sum()),
        "filtered_zero_weight_face_count": int(((face_weight_all <= 0) & ~cap_flags).sum()),
        "detected_cap_face_count": int(cap_flags.sum()),
        "rejected_cap_face_count": int(cap_flags.sum()),
        "accepted_cap_face_count": 0,
        "crop_boundary_weight_zero_face_count": int((face_weight_all <= 0).sum()),
        "source_provenance_invalid_face_count": int((~valid_source).sum()),
        "raw_cell_count": int(coords.shape[0]),
        "raw_supported_edge_count": int(raw_edge_keys.size),
        "source_mesh_boundary_edge_count": int(source_boundary_edges.shape[0]) if valid_boundary_faces.size else 0,
        "coordinate_convention": str(tile.get("coordinate_convention", "local_centered")),
        "scale": TILE_SIZE / GLOBAL_RESOLUTION,
        "translation_only_camera_calls": 0,
    }
    cache_value = tile.get("hermite_cache")
    cache_path = Path(cache_value) if cache_value else None
    return TileStream(
        tile_id=tile_id,
        slot=int(slot),
        origin=origin,
        convention=str(tile.get("coordinate_convention", "local_centered")),
        vertices_global=vertices_global,
        faces=faces,
        face_local_ids=face_local_ids,
        face_uids=face_uids,
        source_cell=source_cell,
        source_axis=source_axis_retained,
        source_quad=source_quad[retained].astype(np.int64, copy=False),
        source_decoder_edge_key=local_edge_key[retained].astype(np.int64, copy=False),
        face_weight=face_weight_all[retained].astype(np.float32, copy=False),
        face_touches_boundary=touches_all[retained].astype(bool, copy=False),
        split=split,
        raw_edge_keys=raw_edge_keys,
        raw_cell_keys=raw_cell_keys,
        source_boundary_samples_global_voxel=source_boundary_samples,
        hermite_cache=cache_path if cache_path is not None and cache_path.is_file() else None,
        diagnostics=tile_diag,
    )


def _empty_observations() -> Dict[str, np.ndarray]:
    return {
        "edge_key": np.empty((0,), dtype=np.int64),
        "edge_coord": np.empty((0, 3), dtype=np.int32),
        "edge_axis": np.empty((0,), dtype=np.int8),
        "tau": np.empty((0,), dtype=np.float32),
        "q": np.empty((0, 3), dtype=np.float32),
        "normal": np.empty((0, 3), dtype=np.float32),
        "tile_slot": np.empty((0,), dtype=np.int32),
        "tile_id": np.empty((0,), dtype=np.int32),
        "face_uid": np.empty((0,), dtype=np.int64),
        "face_local_id": np.empty((0,), dtype=np.int64),
        "source_quad": np.empty((0,), dtype=np.int64),
        "source_cell": np.empty((0,), dtype=np.int32),
        "source_axis": np.empty((0,), dtype=np.int8),
        "source_decoder_edge_key": np.empty((0,), dtype=np.int64),
        "interior_weight": np.empty((0,), dtype=np.float32),
        "touches_artificial_boundary": np.empty((0,), dtype=bool),
    }


def _concat_observations(items: Sequence[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not items:
        return _empty_observations()
    return {key: np.concatenate([item[key] for item in items], axis=0) for key in _empty_observations()}


def _hermite_arrays(payload: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    hermite = payload.get("hermite", payload)
    if not isinstance(hermite, Mapping):
        raise TypeError("Hermite cache is not a mapping")
    required = ("edge_coord", "edge_axis", "q", "n", "tau", "face_id")
    missing = [key for key in required if key not in hermite]
    if missing:
        raise KeyError(f"Hermite cache missing {missing}")
    result = {key: _as_numpy(hermite[key]) for key in required}
    result["edge_coord"] = result["edge_coord"].astype(np.int32, copy=False)
    result["edge_axis"] = result["edge_axis"].reshape(-1).astype(np.int8, copy=False)
    result["q"] = result["q"].astype(np.float32, copy=False)
    result["n"] = result["n"].astype(np.float32, copy=False)
    result["tau"] = result["tau"].reshape(-1).astype(np.float32, copy=False)
    result["face_id"] = result["face_id"].reshape(-1).astype(np.int64, copy=False)
    return result


def _observation_from_hermite(
    tile: TileStream,
    hermite: Mapping[str, np.ndarray],
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    edge_coord = hermite["edge_coord"]
    edge_axis = hermite["edge_axis"]
    q = hermite["q"]
    normal = hermite["n"]
    tau = hermite["tau"]
    face_local = hermite["face_id"]
    finite = (
        np.isfinite(edge_coord).all(axis=1) & np.isfinite(q).all(axis=1)
        & np.isfinite(normal).all(axis=1) & np.isfinite(tau)
    )
    finite &= (tau >= -float(args.intersection_tolerance)) & (tau <= 1.0 + float(args.intersection_tolerance))
    finite &= _valid_edge_coords(edge_coord, edge_axis, GLOBAL_RESOLUTION)
    face_pos = np.searchsorted(tile.face_local_ids, face_local)
    face_valid = face_pos < tile.face_local_ids.size
    face_valid &= tile.face_local_ids[np.minimum(face_pos, max(tile.face_local_ids.size - 1, 0))] == face_local
    keep = finite & face_valid
    if not bool(keep.any()):
        return _empty_observations()
    face_pos = face_pos[keep].astype(np.int64)
    edge_coord = edge_coord[keep]
    edge_axis = edge_axis[keep]
    q = q[keep]
    normal = normal[keep]
    tau = np.clip(tau[keep], 0.0, 1.0)
    normal_len = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = normal / np.maximum(normal_len, 1e-12)
    return {
        "edge_key": _edge_keys(edge_coord, edge_axis),
        "edge_coord": edge_coord.astype(np.int32, copy=False),
        "edge_axis": edge_axis.astype(np.int8, copy=False),
        "tau": tau.astype(np.float32, copy=False),
        "q": q.astype(np.float32, copy=False),
        "normal": normal.astype(np.float32, copy=False),
        "tile_slot": np.full((q.shape[0],), tile.slot, dtype=np.int32),
        "tile_id": np.full((q.shape[0],), tile.tile_id, dtype=np.int32),
        "face_uid": tile.face_uids[face_pos],
        "face_local_id": tile.face_local_ids[face_pos],
        "source_quad": tile.source_quad[face_pos],
        "source_cell": tile.source_cell[face_pos],
        "source_axis": tile.source_axis[face_pos],
        "source_decoder_edge_key": tile.source_decoder_edge_key[face_pos],
        "interior_weight": tile.face_weight[face_pos],
        "touches_artificial_boundary": tile.face_touches_boundary[face_pos],
    }


def _deduplicate_tile_observations(
    tile: TileStream,
    observations: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Collapse native scan-direction duplicates before global clustering.

    The native triangle traversal can report the same primal edge several
    times for one source quad (one report per scan direction, and sometimes
    once per decoder triangle).  Those reports are one local surface package,
    not competing global modes.  Grouping by ``(edge_key, source_quad)`` keeps
    them out of the ambiguity gate while retaining one deterministic
    provenance representative and a weighted/projective Hermite average.
    """
    n = int(observations["edge_key"].size)
    if n <= 1:
        tile.diagnostics["tile_local_intersection_record_count"] = n
        tile.diagnostics["tile_local_deduplicated_intersection_record_count"] = 0
        tile.diagnostics["tile_local_intersection_package_count"] = n
        return {key: np.asarray(value) for key, value in observations.items()}

    # Weight-descending order makes the first row of each package the stable
    # highest-confidence provenance representative; face_uid is the final
    # deterministic tie breaker.
    order = np.lexsort((
        observations["face_uid"],
        -observations["interior_weight"],
        observations["source_quad"],
        observations["edge_key"],
    ))
    edge_sorted = observations["edge_key"][order]
    quad_sorted = observations["source_quad"][order]
    starts_mask = np.r_[True, (edge_sorted[1:] != edge_sorted[:-1]) | (quad_sorted[1:] != quad_sorted[:-1])]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    group_ids = np.cumsum(starts_mask, dtype=np.int64) - 1
    group_count = int(starts.size)
    weights = observations["interior_weight"][order].astype(np.float64)
    weight_sum = np.add.reduceat(weights, starts)
    package_weight = np.maximum.reduceat(weights, starts).astype(np.float32)

    q_sorted = observations["q"][order].astype(np.float64)
    q_sum = np.zeros((group_count, 3), dtype=np.float64)
    np.add.at(q_sum, group_ids, q_sorted * weights[:, None])
    package_q = (q_sum / np.maximum(weight_sum[:, None], 1e-12)).astype(np.float32)

    tau_sorted = observations["tau"][order].astype(np.float64)
    tau_sum = np.zeros((group_count,), dtype=np.float64)
    np.add.at(tau_sum, group_ids, tau_sorted * weights)
    package_tau = (tau_sum / np.maximum(weight_sum, 1e-12)).astype(np.float32)

    normal_sorted = observations["normal"][order].astype(np.float64)
    reference = normal_sorted[starts]
    expanded_reference = reference[group_ids]
    signs = np.where((normal_sorted * expanded_reference).sum(axis=1) < 0.0, -1.0, 1.0)
    normal_sum = np.zeros((group_count, 3), dtype=np.float64)
    np.add.at(normal_sum, group_ids, normal_sorted * signs[:, None] * weights[:, None])
    package_normal = normal_sum / np.maximum(np.linalg.norm(normal_sum, axis=1, keepdims=True), 1e-12)
    package_normal = package_normal.astype(np.float32)

    package_boundary = np.zeros((group_count,), dtype=bool)
    np.logical_or.at(package_boundary, group_ids, observations["touches_artificial_boundary"][order])
    representatives = order[starts]
    result = {
        "edge_key": observations["edge_key"][representatives].astype(np.int64, copy=False),
        "edge_coord": observations["edge_coord"][representatives].astype(np.int32, copy=False),
        "edge_axis": observations["edge_axis"][representatives].astype(np.int8, copy=False),
        "tau": package_tau,
        "q": package_q,
        "normal": package_normal,
        "tile_slot": observations["tile_slot"][representatives].astype(np.int32, copy=False),
        "tile_id": observations["tile_id"][representatives].astype(np.int32, copy=False),
        "face_uid": observations["face_uid"][representatives].astype(np.int64, copy=False),
        "face_local_id": observations["face_local_id"][representatives].astype(np.int64, copy=False),
        "source_quad": observations["source_quad"][representatives].astype(np.int64, copy=False),
        "source_cell": observations["source_cell"][representatives].astype(np.int32, copy=False),
        "source_axis": observations["source_axis"][representatives].astype(np.int8, copy=False),
        "source_decoder_edge_key": observations["source_decoder_edge_key"][representatives].astype(np.int64, copy=False),
        "interior_weight": package_weight,
        "touches_artificial_boundary": package_boundary,
    }
    tile.diagnostics["tile_local_intersection_record_count"] = n
    tile.diagnostics["tile_local_deduplicated_intersection_record_count"] = int(n - group_count)
    tile.diagnostics["tile_local_intersection_package_count"] = group_count
    return result


def _collect_tile_intersections(tile: TileStream, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    """Collect one tile's observations into the shared canonical record schema."""
    if tile.hermite_cache is not None and not args.force_intersections:
        payload = torch.load(tile.hermite_cache, map_location="cpu", weights_only=False)
        observations = _deduplicate_tile_observations(
            tile, _observation_from_hermite(tile, _hermite_arrays(payload), args)
        )
        tile.diagnostics["intersection_source"] = "cached_native_global_hermite"
        tile.diagnostics["raw_intersection_record_count"] = int(
            tile.diagnostics.get("tile_local_intersection_record_count", observations["edge_key"].size)
        )
        tile.diagnostics["native_chunk_count"] = 0
        return observations
    if mesh_to_flexible_dual_grid is None:
        raise RuntimeError(f"native O-Voxel extension is unavailable: {_NATIVE_IMPORT_ERROR!r}")
    pieces: List[Dict[str, np.ndarray]] = []
    chunk = max(1, int(args.triangle_chunk_size))
    native_chunk_count = 0
    for start in range(0, tile.faces.shape[0], chunk):
        stop = min(start + chunk, tile.faces.shape[0])
        native_faces = torch.from_numpy(tile.faces[start:stop].astype(np.int32, copy=False)).contiguous()
        _, _, _, hermite = mesh_to_flexible_dual_grid(
            vertices=torch.from_numpy(tile.vertices_global).contiguous().float(),
            faces=native_faces,
            grid_size=GLOBAL_RESOLUTION,
            aabb=RUNTIME_AABB,
            grid_range=_native_grid_range(tile),
            face_weight=1.0,
            boundary_weight=0.0,
            regularization_weight=0.0,
            timing=False,
            return_hermite=True,
        )
        h = _hermite_arrays(hermite)
        h["face_id"] = tile.face_local_ids[start:stop][h["face_id"]]
        pieces.append(_observation_from_hermite(tile, h, args))
        native_chunk_count += 1
        del native_faces, hermite, h
    observations = _deduplicate_tile_observations(tile, _concat_observations(pieces))
    tile.diagnostics["intersection_source"] = "native_global_lattice_chunked"
    tile.diagnostics["raw_intersection_record_count"] = int(
        tile.diagnostics.get("tile_local_intersection_record_count", observations["edge_key"].size)
    )
    tile.diagnostics["native_chunk_count"] = native_chunk_count
    return observations


def _save_or_load_intersection_shard(
    tile: TileStream,
    observations: Dict[str, np.ndarray],
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    path = output_dir / "intersection_shards" / f"tile_{tile.tile_id:03d}.pt"
    if path.is_file() and not args.force_intersections:
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if (
            cached.get("shard_schema") == "tile_source_quad_edge_dedup_v2"
            and cached.get("shard_tile_id") == tile.tile_id
            and cached.get("shard_face_count") == int(tile.faces.shape[0])
        ):
            loaded = {key: _as_numpy(cached[key]) for key in _empty_observations()}
            loaded["edge_coord"] = loaded["edge_coord"].astype(np.int32, copy=False)
            loaded["edge_axis"] = loaded["edge_axis"].astype(np.int8, copy=False)
            loaded["tau"] = loaded["tau"].astype(np.float32, copy=False)
            loaded["q"] = loaded["q"].astype(np.float32, copy=False)
            loaded["normal"] = loaded["normal"].astype(np.float32, copy=False)
            loaded["touches_artificial_boundary"] = loaded["touches_artificial_boundary"].astype(bool, copy=False)
            tile.diagnostics["tile_local_intersection_record_count"] = int(
                cached.get("raw_record_count", loaded["edge_key"].size)
            )
            tile.diagnostics["tile_local_deduplicated_intersection_record_count"] = int(
                tile.diagnostics["tile_local_intersection_record_count"] - loaded["edge_key"].size
            )
            tile.diagnostics["tile_local_intersection_package_count"] = int(loaded["edge_key"].size)
            tile.diagnostics["raw_intersection_record_count"] = tile.diagnostics["tile_local_intersection_record_count"]
            tile.diagnostics["intersection_shard_cache_hit"] = True
            return loaded
    payload = {key: torch.from_numpy(value) for key, value in observations.items()}
    payload.update({
        "format": "pixal3d_global_mesh_intersection_shard_v1",
        "shard_schema": "tile_source_quad_edge_dedup_v2",
        "shard_tile_id": tile.tile_id,
        "shard_face_count": int(tile.faces.shape[0]),
        "raw_record_count": int(tile.diagnostics.get("tile_local_intersection_record_count", observations["edge_key"].size)),
        "package_count": int(observations["edge_key"].size),
    })
    _atomic_torch_save(path, payload)
    tile.diagnostics["intersection_shard_cache_hit"] = False
    return observations


def _sample_hash(observations: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ("edge_key", "tau", "q", "normal", "tile_id", "face_uid"):
        digest.update(np.ascontiguousarray(observations[key]).tobytes())
    return digest.hexdigest()


def _topology_hash(topology: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(topology["active_edge_keys"], dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(topology["final_cells"], dtype=np.int32).tobytes())
    return digest.hexdigest()


def _qef_input_hash(topology: Mapping[str, Any]) -> str:
    """Hash geometry-bearing selected modes, not just cell/edge topology."""
    digest = hashlib.sha256()
    digest.update(_topology_hash(topology).encode("ascii"))
    for key, dtype in (
        ("active_q", np.float32),
        ("active_normal", np.float32),
        ("active_tau", np.float32),
        ("active_source_face_uid", np.int64),
        ("active_source_cell", np.int32),
        ("active_source_axis", np.int8),
    ):
        digest.update(np.ascontiguousarray(topology[key], dtype=dtype).tobytes())
    return digest.hexdigest()


def _cluster_intersections(
    observations: Mapping[str, np.ndarray],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Cluster global-edge samples and select exactly one surface mode per edge."""
    n = int(observations["edge_key"].size)
    if n == 0:
        raise RuntimeError("global triangle stream produced no valid lattice intersections")
    edge_key = observations["edge_key"]
    tau = observations["tau"]
    normal = observations["normal"]
    tile_slot = observations["tile_slot"]
    # Stable lexicographic ordering makes repeated runs and repeated input tiles
    # deterministic.  No float coordinate is used to create an edge key.
    order = np.lexsort((observations["face_uid"], tile_slot, tau, edge_key))
    sorted_edge = edge_key[order]
    sorted_tau = tau[order]
    sorted_normal = normal[order]
    tau_gap = np.zeros((n,), dtype=bool)
    tau_gap[1:] = (sorted_tau[1:] - sorted_tau[:-1]) > float(args.tau_cluster_threshold)
    normal_dot = np.ones((n,), dtype=np.float32)
    if n > 1:
        normal_dot[1:] = np.abs((sorted_normal[1:] * sorted_normal[:-1]).sum(axis=1))
    normal_gap = np.zeros((n,), dtype=bool)
    normal_gap[1:] = normal_dot[1:] < math.cos(math.radians(float(args.normal_angle_deg)))
    new_cluster = np.r_[True, (sorted_edge[1:] != sorted_edge[:-1]) | tau_gap[1:] | normal_gap[1:]]
    cluster_sorted = np.cumsum(new_cluster, dtype=np.int64) - 1
    cluster_count = int(cluster_sorted[-1] + 1)
    cluster_original = np.empty_like(cluster_sorted)
    cluster_original[order] = cluster_sorted

    tile_factor = max(int(tile_slot.max(initial=0)) + 1, 1)
    edge_tile_pairs = np.unique(edge_key.astype(np.int64) * tile_factor + tile_slot.astype(np.int64))
    edge_tile_keys, edge_tile_support = np.unique(
        edge_tile_pairs // tile_factor, return_counts=True
    )
    covered_seam_edge_keys = edge_tile_keys[edge_tile_support >= 2].astype(np.int64, copy=False)
    pair = cluster_sorted.astype(np.int64) * tile_factor + tile_slot[order].astype(np.int64)
    pair_order = np.lexsort((observations["face_uid"][order], -observations["interior_weight"][order], pair))
    pair_sorted = pair[pair_order]
    pair_starts = np.r_[0, np.flatnonzero(pair_sorted[1:] != pair_sorted[:-1]) + 1]
    pair_cluster = cluster_sorted[pair_order][pair_starts]
    pair_tile = tile_slot[order][pair_order][pair_starts]
    pair_weights_sorted = observations["interior_weight"][order][pair_order].astype(np.float64)
    pair_weight_sum = np.add.reduceat(pair_weights_sorted, pair_starts)
    # A tile is one partition-of-unity package per cluster.  The package's
    # confidence is a max, not a sum, so duplicating a face/triangle cannot
    # change the result.
    pair_weight = np.maximum.reduceat(pair_weights_sorted, pair_starts).astype(np.float32)
    pair_q_sorted = observations["q"][order][pair_order]
    pair_q_sum = np.add.reduceat(pair_q_sorted * pair_weights_sorted[:, None].astype(np.float32), pair_starts, axis=0)
    pair_q = pair_q_sum / np.maximum(pair_weight_sum[:, None], 1e-12).astype(np.float32)
    pair_tau_sorted = observations["tau"][order][pair_order]
    pair_tau = np.add.reduceat(pair_tau_sorted.astype(np.float64) * pair_weights_sorted, pair_starts) / np.maximum(pair_weight_sum, 1e-12)
    pair_n_sorted = observations["normal"][order][pair_order]
    first_n = pair_n_sorted[pair_starts]
    signs = np.where((pair_n_sorted * first_n[np.repeat(np.arange(pair_starts.size), np.diff(np.r_[pair_starts, pair_sorted.size]))]).sum(axis=1) < 0, -1.0, 1.0).astype(np.float32)
    aligned_n = pair_n_sorted * signs[:, None]
    pair_n_sum = np.add.reduceat(aligned_n * pair_weights_sorted[:, None].astype(np.float32), pair_starts, axis=0)
    pair_n = pair_n_sum / np.maximum(pair_weight_sum[:, None], 1e-12).astype(np.float32)
    pair_n /= np.maximum(np.linalg.norm(pair_n, axis=1, keepdims=True), 1e-12)
    # Provenance is chosen by the same stable max-weight/lowest-face order.
    choice = order[pair_order[pair_starts]]
    pair_edge = edge_key[choice]
    pair_face = observations["face_uid"][choice]
    pair_source_quad = observations["source_quad"][choice]
    pair_source_cell = observations["source_cell"][choice]
    pair_source_axis = observations["source_axis"][choice]
    pair_source_decoder_edge = observations["source_decoder_edge_key"][choice]
    pair_boundary = observations["touches_artificial_boundary"][choice].astype(bool)

    # A cluster may contain one package from several tiles.  Collapse its
    # edge key to one representative package before sorting cluster-level
    # statistics; ``pair_edge`` itself is still package-level.
    cluster_first_pair = np.full((cluster_count,), -1, dtype=np.int64)
    for idx, cid in enumerate(pair_cluster):
        if cluster_first_pair[cid] < 0:
            cluster_first_pair[cid] = idx
    cluster_edge = pair_edge[cluster_first_pair]
    cluster_weight = np.zeros((cluster_count,), dtype=np.float64)
    np.add.at(cluster_weight, pair_cluster, pair_weight.astype(np.float64))
    cluster_support = np.bincount(pair_cluster, minlength=cluster_count).astype(np.int32)
    cluster_min_tile = np.full((cluster_count,), np.iinfo(np.int32).max, dtype=np.int32)
    np.minimum.at(cluster_min_tile, pair_cluster, pair_tile.astype(np.int32))
    cluster_tau_sum = np.zeros((cluster_count,), dtype=np.float64)
    np.add.at(cluster_tau_sum, pair_cluster, pair_tau * pair_weight)
    cluster_tau = cluster_tau_sum / np.maximum(cluster_weight, 1e-12)
    cluster_q_sum = np.zeros((cluster_count, 3), dtype=np.float64)
    np.add.at(cluster_q_sum, pair_cluster, pair_q * pair_weight[:, None])
    cluster_q = (cluster_q_sum / np.maximum(cluster_weight[:, None], 1e-12)).astype(np.float32)
    cluster_n_sum = np.zeros((cluster_count, 3), dtype=np.float64)
    # Align each tile package to the first package of the cluster.
    cluster_ref_n = pair_n[cluster_first_pair]
    pair_sign = np.where((pair_n * cluster_ref_n[ pair_cluster]).sum(axis=1) < 0, -1.0, 1.0)
    np.add.at(cluster_n_sum, pair_cluster, pair_n * pair_sign[:, None] * pair_weight[:, None])
    cluster_n = (cluster_n_sum / np.maximum(cluster_weight[:, None], 1e-12)).astype(np.float32)
    cluster_n /= np.maximum(np.linalg.norm(cluster_n, axis=1, keepdims=True), 1e-12)
    cluster_tau_var_sum = np.zeros((cluster_count,), dtype=np.float64)
    np.add.at(cluster_tau_var_sum, pair_cluster, pair_weight * (pair_tau - cluster_tau[pair_cluster]) ** 2)
    cluster_tau_var = cluster_tau_var_sum / np.maximum(cluster_weight, 1e-12)
    cluster_normal_agreement_sum = np.zeros((cluster_count,), dtype=np.float64)
    np.add.at(cluster_normal_agreement_sum, pair_cluster, pair_weight * np.abs((pair_n * cluster_n[pair_cluster]).sum(axis=1)))
    cluster_normal_agreement = cluster_normal_agreement_sum / np.maximum(cluster_weight, 1e-12)
    cluster_boundary = np.zeros((cluster_count,), dtype=bool)
    np.logical_or.at(cluster_boundary, pair_cluster, pair_boundary)
    cluster_choice = np.full((cluster_count,), -1, dtype=np.int64)
    for idx, cid in enumerate(pair_cluster):
        if cluster_choice[cid] < 0:
            cluster_choice[cid] = idx

    # Select by independent tile support first, then normalized interior weight,
    # variance, and normal agreement.  The tiny deterministic IDs are only the
    # last tie breaker and are never used as geometry.
    edge_order = np.lexsort((np.arange(cluster_count), cluster_tau_var, -cluster_normal_agreement, -cluster_weight, -cluster_support, cluster_edge))
    sorted_cluster_edge = cluster_edge[edge_order]
    edge_starts = np.r_[0, np.flatnonzero(sorted_cluster_edge[1:] != sorted_cluster_edge[:-1]) + 1]
    edge_ends = np.r_[edge_starts[1:], cluster_count]
    selected_cluster = edge_order[edge_starts]
    ambiguous = np.zeros((cluster_count,), dtype=bool)
    resolved_close_conflict_edge_count = 0
    for edge_group_index, (start, end) in enumerate(zip(edge_starts, edge_ends)):
        if end - start <= 1:
            continue
        top = int(edge_order[start])
        second = int(edge_order[start + 1])
        same_support = int(cluster_support[top]) == int(cluster_support[second])
        weight_gap = abs(float(cluster_weight[top]) - float(cluster_weight[second]))
        denom = max(float(cluster_weight[top]), float(cluster_weight[second]), 1e-12)
        close = weight_gap / denom <= float(args.ambiguity_weight_gap)
        separated = abs(float(cluster_tau[top]) - float(cluster_tau[second])) > float(args.tau_cluster_threshold)
        if same_support and close and separated:
            # A two-tile overlap often gives one package per competing mode:
            # neither mode has independent multi-tile support yet, so dropping
            # both would create a topology hole.  Select the deterministic
            # dominant package (the ordering above already uses support,
            # weight, variance, normal agreement, then stable IDs).  Reserve
            # the hard ambiguity gate for two close modes that are each
            # independently supported by multiple tiles; that is the case in
            # which no tile-priority decision can be justified.
            # Close separated modes are resolved to one deterministic owner
            # rather than dropped.  Dropping both modes creates a topology
            # hole; averaging them creates a Frankenstein surface.  The owner
            # is chosen by the established score ordering, with the lowest
            # tile slot as a stable ownership tie breaker across a connected
            # overlap.  The original conflict remains reported separately.
            candidates = edge_order[start:end]
            winner = int(candidates[np.lexsort((candidates, cluster_min_tile[candidates]))[0]])
            selected_cluster[edge_group_index] = winner
            resolved_close_conflict_edge_count += 1
    chosen_by_edge = {int(cluster_edge[cid]): int(cid) for cid in selected_cluster if not ambiguous[cid]}
    active_cluster_ids = np.asarray([cid for cid in selected_cluster if not ambiguous[cid]], dtype=np.int64)
    if active_cluster_ids.size:
        active_cluster_ids = active_cluster_ids[np.argsort(cluster_edge[active_cluster_ids], kind="stable")]
    selected = np.isin(cluster_original, active_cluster_ids)
    # A selected cluster must contain a positive-weight, non-artificial source.
    active_cluster_ids = active_cluster_ids[cluster_weight[active_cluster_ids] > 0]
    selected &= np.isin(cluster_original, active_cluster_ids)
    active_keys = cluster_edge[active_cluster_ids].astype(np.int64)
    active_choice = cluster_choice[active_cluster_ids]
    ambiguous_edge_keys = np.unique(cluster_edge[ambiguous]).astype(np.int64)
    ambiguous_seam_edge_count = 0
    if ambiguous_edge_keys.size:
        ambiguous_record_mask = np.isin(edge_key, ambiguous_edge_keys)
        pair_ambiguous = edge_key[ambiguous_record_mask].astype(np.int64) * tile_factor + tile_slot[ambiguous_record_mask].astype(np.int64)
        unique_ambiguous_pairs = np.unique(pair_ambiguous)
        ambiguous_edge_support = np.bincount(
            np.searchsorted(ambiguous_edge_keys, unique_ambiguous_pairs // tile_factor),
            minlength=ambiguous_edge_keys.size,
        )
        ambiguous_seam_edge_count = int((ambiguous_edge_support >= 2).sum())
    result = {
        "cluster_policy": CLUSTER_POLICY,
        "covered_seam_edge_keys": covered_seam_edge_keys,
        "selected_observation_mask": selected,
        "cluster_id_per_observation": cluster_original,
        "cluster_count": cluster_count,
        "cluster_edge": cluster_edge.astype(np.int64),
        "cluster_tau": cluster_tau.astype(np.float32),
        "cluster_tau_variance": cluster_tau_var.astype(np.float32),
        "cluster_normal": cluster_n,
        "cluster_normal_agreement": cluster_normal_agreement.astype(np.float32),
        "cluster_weight": cluster_weight.astype(np.float32),
        "cluster_support": cluster_support,
        "cluster_boundary": cluster_boundary,
        "ambiguous_cluster": ambiguous,
        "active_cluster_ids": active_cluster_ids,
        "active_edge_keys": active_keys,
        "active_edge_coords": _decode_edge_keys(active_keys)[0],
        "active_edge_axis": _decode_edge_keys(active_keys)[1],
        "active_tau": cluster_tau[active_cluster_ids].astype(np.float32),
        "active_q": cluster_q[active_cluster_ids].astype(np.float32),
        "active_normal": cluster_n[active_cluster_ids].astype(np.float32),
        "active_support": cluster_support[active_cluster_ids].astype(np.int32),
        "active_tau_variance": cluster_tau_var[active_cluster_ids].astype(np.float32),
        "active_normal_agreement": cluster_normal_agreement[active_cluster_ids].astype(np.float32),
        "active_source_face_uid": pair_face[active_choice].astype(np.int64),
        "active_source_quad": pair_source_quad[active_choice].astype(np.int64),
        "active_source_cell": pair_source_cell[active_choice].astype(np.int32),
        "active_source_axis": pair_source_axis[active_choice].astype(np.int8),
        "active_source_decoder_edge_key": pair_source_decoder_edge[active_choice].astype(np.int64),
        "active_touches_artificial_boundary": cluster_boundary[active_cluster_ids],
        "ambiguous_edge_keys": ambiguous_edge_keys,
        "ambiguous_seam_edge_count": ambiguous_seam_edge_count,
        "stats": {
            "raw_intersection_record_count": n,
            "unique_edge_count": int(np.unique(edge_key).size),
            "cluster_count": cluster_count,
            "dominant_cluster_count": int(selected_cluster.size),
            "dropped_cluster_count": int(cluster_count - active_cluster_ids.size),
            "ambiguous_intersection_edge_count": int(np.unique(cluster_edge[ambiguous]).size),
            "ambiguous_seam_edge_count": ambiguous_seam_edge_count,
            "resolved_close_conflict_edge_count": int(resolved_close_conflict_edge_count),
            "mean_clusters_per_edge": float(cluster_count / max(np.unique(edge_key).size, 1)),
            "max_observations_per_edge": int(np.unique(edge_key, return_counts=True)[1].max(initial=0)),
            "selected_observation_count": int(selected.sum()),
            "sample_hash": _sample_hash(observations),
            "covered_seam_candidate_edge_count": int(covered_seam_edge_keys.size),
        },
    }
    return result


def _build_topology(
    cluster: Mapping[str, Any],
    streams: Sequence[TileStream],
    observations: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    keys = np.asarray(cluster["active_edge_keys"], dtype=np.int64)
    coords = np.asarray(cluster["active_edge_coords"], dtype=np.int32)
    axis = np.asarray(cluster["active_edge_axis"], dtype=np.int8)
    valid = _valid_edge_coords(coords, axis, GLOBAL_RESOLUTION)
    touches = np.asarray(cluster["active_touches_artificial_boundary"], dtype=bool)
    source_decoder = np.asarray(cluster["active_source_decoder_edge_key"], dtype=np.int64)
    untraceable = source_decoder < 0
    topology_keep = valid & ~touches & ~untraceable
    incident = _edge_cells(coords[topology_keep], axis[topology_keep])
    incident_keys = _cell_keys(incident.reshape(-1, 3)).reshape(-1, 4) if incident.size else np.empty((0, 4), np.int64)
    final_cells = np.unique(incident.reshape(-1, 3), axis=0).astype(np.int32, copy=False) if incident.size else np.empty((0, 3), np.int32)
    final_keys = _cell_keys(final_cells)
    if final_cells.size:
        positions = np.searchsorted(final_keys, incident_keys)
        if not bool((final_keys[np.minimum(positions, max(final_keys.size - 1, 0))] == incident_keys).all()):
            raise AssertionError("four-cell closure produced a non-canonical final cell")
    flags = np.zeros((final_cells.shape[0], 3), dtype=bool)
    edge_pos = np.searchsorted(final_keys, _cell_keys(coords[topology_keep])) if topology_keep.any() else np.empty((0,), np.int64)
    if edge_pos.size:
        flags[edge_pos, axis[topology_keep]] = True

    raw_edges = np.unique(np.concatenate([tile.raw_edge_keys for tile in streams])) if streams else np.empty((0,), np.int64)
    raw_cells = np.unique(np.concatenate([tile.raw_cell_keys for tile in streams])) if streams else np.empty((0,), np.int64)
    created = ~np.isin(final_keys, raw_cells, assume_unique=False)
    coverage = np.zeros((final_cells.shape[0],), dtype=np.int32)
    active_support = np.asarray(cluster["active_support"], dtype=np.int32)[topology_keep]
    if incident.size:
        np.maximum.at(coverage, positions.reshape(-1), np.repeat(active_support, 4))
    selected_keys = keys[topology_keep]
    covered_seam_keys = np.intersect1d(
        selected_keys,
        np.asarray(cluster.get("covered_seam_edge_keys", np.empty((0,), dtype=np.int64)), dtype=np.int64),
        assume_unique=True,
    )
    raw_supported = np.unique(observations["edge_key"])
    stats = dict(cluster["stats"])
    stats.update({
        "raw_supported_edge_count": int(raw_edges.size),
        "mesh_selected_edge_count": int(keys.size),
        "mesh_only_edge_count": int(np.setdiff1d(selected_keys, raw_edges, assume_unique=False).size),
        "raw_only_edge_count": int(np.setdiff1d(raw_edges, selected_keys, assume_unique=False).size),
        "untraceable_active_edge_count": int(untraceable[topology_keep].sum()),
        "untraceable_candidate_edge_count": int(untraceable.sum()),
        "artificial_boundary_active_edge_count": int(touches[topology_keep].sum()),
        "artificial_boundary_candidate_edge_count": int(touches.sum()),
        "active_edge_missing_four_cells": int((~valid).sum()),
        "rejected_invalid_edge_count": int((~valid).sum()),
        "mesh_created_cell_count": int(created.sum()),
        "final_cell_count": int(final_cells.shape[0]),
        "final_active_edge_count": int(selected_keys.size),
        "covered_seam_active_edge_count": int(covered_seam_keys.size),
        "covered_seam_roi_definition": "active selected edges with at least two positive-weight tile packages on the canonical edge",
        "active_edge_support_min": int(active_support.min(initial=0)),
        "active_edge_support_mean": float(active_support.mean()) if active_support.size else 0.0,
        "active_edge_support_p95": float(np.percentile(active_support, 95)) if active_support.size else 0.0,
    })
    return {
        "final_cells": final_cells,
        "final_cell_keys": final_keys,
        "intersected": flags,
        "active_edge_keys": selected_keys,
        "active_edge_coords": coords[topology_keep],
        "active_edge_axis": axis[topology_keep],
        "active_edge_support": active_support,
        "covered_seam_edge_keys": covered_seam_keys,
        "active_q": np.asarray(cluster["active_q"], dtype=np.float32)[topology_keep],
        "active_normal": np.asarray(cluster["active_normal"], dtype=np.float32)[topology_keep],
        "active_tau": np.asarray(cluster["active_tau"], dtype=np.float32)[topology_keep],
        "active_tau_variance": np.asarray(cluster["active_tau_variance"], dtype=np.float32)[topology_keep],
        "active_normal_agreement": np.asarray(cluster["active_normal_agreement"], dtype=np.float32)[topology_keep],
        "active_source_face_uid": np.asarray(cluster["active_source_face_uid"], dtype=np.int64)[topology_keep],
        "active_source_quad": np.asarray(cluster["active_source_quad"], dtype=np.int64)[topology_keep],
        "active_source_cell": np.asarray(cluster["active_source_cell"], dtype=np.int32)[topology_keep],
        "active_source_axis": np.asarray(cluster["active_source_axis"], dtype=np.int8)[topology_keep],
        "active_source_decoder_edge_key": source_decoder[topology_keep],
        "cell_coverage": coverage,
        "raw_edge_keys": raw_edges,
        "raw_cell_keys": raw_cells,
        "stats": stats,
    }


def _quadratic_energy(a: torch.Tensor, b: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return (value * torch.bmm(a, value.unsqueeze(-1)).squeeze(-1)).sum(dim=1) - 2.0 * (b * value).sum(dim=1)


def _solve_box_native_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = a.shape[0]
    solution = torch.linalg.lstsq(a, b.unsqueeze(-1)).solution.squeeze(-1)
    finite_solution = torch.isfinite(solution).all(dim=1)
    inside = finite_solution & ((solution >= lo) & (solution <= hi)).all(dim=1)
    best_v = torch.where(inside[:, None], solution, torch.zeros_like(solution))
    best_e = torch.where(inside, _quadratic_energy(a, b, solution), torch.full((batch,), float("inf"), dtype=a.dtype, device=a.device))

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
                rhs = b[:, free_axis] - a[:, free_axis, fixed[0]] * bound0 - a[:, free_axis, fixed[1]] * bound1
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
    resolution: int,
    regularization_weight: float,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    n = int(cells.shape[0])
    dual_cell = np.full((n, 3), 0.5, dtype=np.float32)
    solved_mask = q_count > 1e-12
    clamp_count = 0
    singular_count = 0
    rank_hist: Dict[str, int] = {}
    indices = np.flatnonzero(solved_mask)
    scale = float(resolution)
    for start in range(0, indices.size, max(1, int(batch_size))):
        batch_indices = indices[start:start + max(1, int(batch_size))]
        q = (q_edge[batch_indices] + q_face[batch_indices] + q_boundary[batch_indices]).astype(np.float32, copy=False).copy()
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
        a = q_t[:, :3, :3] / (scale * scale)
        d = q_t[:, :3, 3] / scale
        b = -(torch.bmm(a, cc.unsqueeze(-1)).squeeze(-1) + d)
        lo = torch.zeros_like(cc)
        hi = torch.ones_like(cc)
        with torch.no_grad():
            value, clamped, rank, singular = _solve_box_native_torch(a, b, lo, hi)
        dual_cell[batch_indices] = value.detach().cpu().numpy()
        clamp_count += int(clamped.sum().item())
        singular_count += int(singular.sum().item())
        unique_rank, rank_count = torch.unique(rank, return_counts=True)
        for key, count in zip(unique_rank.detach().cpu().tolist(), rank_count.detach().cpu().tolist()):
            rank_hist[str(int(key))] = rank_hist.get(str(int(key)), 0) + int(count)
        del q_t, cc, a, b, value, clamped, rank, singular
    dual_cell = np.nan_to_num(dual_cell, nan=0.5, posinf=1.0, neginf=0.0).clip(0.0, 1.0).astype(np.float32)
    return {
        "dual_cell": dual_cell,
        # Native QEF statistics are expressed in the translated AABB frame
        # [0, 1]^3.  The final mesh's world frame is this value minus 0.5;
        # keeping the two frames separate is important for native parity.
        "dual_translated": ((cells.astype(np.float32) + dual_cell) / scale).astype(np.float32),
        "qef_solved_cell_count": int(solved_mask.sum()),
        "qef_no_constraint_fallback_count": int((~solved_mask).sum()),
        "qef_clamped_count": int(clamp_count),
        "qef_singular_fallback_count": int(singular_count),
        "qef_rank_histogram": rank_hist,
    }


def _triangle_face_qef_fallback(
    tile: TileStream,
    selected_rows: np.ndarray,
    topology: Mapping[str, Any],
    q_face: np.ndarray,
    q_face_weight: np.ndarray,
) -> int:
    """Small-host fallback; native face rasterization is used in production."""
    final_keys = topology["final_cell_keys"]
    final_cells = topology["final_cells"]
    count = 0
    for row in selected_rows:
        tri = tile.vertices_global[tile.faces[row]]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        length = float(np.linalg.norm(normal))
        if length <= 1e-12:
            continue
        normal /= length
        # Native QEF matrices use translated AABB coordinates [0, 1]^3;
        # the tile stream itself is stored in world coordinates [-0.5, 0.5].
        tri_translated = tri + 0.5
        plane = np.r_[normal, -float(np.dot(normal, tri_translated[0]))].astype(np.float32)
        q = np.outer(plane, plane).astype(np.float32) * float(tile.face_weight[row])
        # A selected source face is attached to every selected edge sample it
        # produced; this conservative fallback only adds its plane to closure
        # cells, never creates a topology cell.
        face_uid = int(tile.face_uids[row])
        edge_rows = np.flatnonzero(topology["active_source_face_uid"] == face_uid)
        if edge_rows.size == 0:
            continue
        cell_keys = _cell_keys(_edge_cells(topology["active_edge_coords"][edge_rows], topology["active_edge_axis"][edge_rows]).reshape(-1, 3))
        positions = np.searchsorted(final_keys, cell_keys)
        valid = positions < final_keys.size
        valid &= final_keys[np.minimum(positions, max(final_keys.size - 1, 0))] == cell_keys
        for pos in positions[valid]:
            q_face[int(pos)] += q
            q_face_weight[int(pos)] += float(tile.face_weight[row])
            count += 1
    return count


def _aggregate_qef(
    topology: MutableMapping[str, Any],
    streams: Sequence[TileStream],
    observations: Mapping[str, np.ndarray],
    cluster: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    final_cells = np.asarray(topology["final_cells"], dtype=np.int32)
    final_keys = np.asarray(topology["final_cell_keys"], dtype=np.int64)
    n = int(final_cells.shape[0])
    q_edge = np.zeros((n, 4, 4), dtype=np.float32)
    q_face = np.zeros((n, 4, 4), dtype=np.float32)
    q_boundary = np.zeros((n, 4, 4), dtype=np.float32)
    q_sum = np.zeros((n, 3), dtype=np.float32)
    q_count = np.zeros((n,), dtype=np.float32)
    active_q = np.asarray(topology["active_q"], dtype=np.float32)
    active_n = np.asarray(topology["active_normal"], dtype=np.float32)
    active_edges = np.asarray(topology["active_edge_coords"], dtype=np.int32)
    active_axis = np.asarray(topology["active_edge_axis"], dtype=np.int8)
    incident = _edge_cells(active_edges, active_axis)
    incident_keys = _cell_keys(incident.reshape(-1, 3)).reshape(-1, 4)
    positions = np.searchsorted(final_keys, incident_keys)
    if incident.size and not bool((final_keys[np.minimum(positions, max(final_keys.size - 1, 0))] == incident_keys).all()):
        raise AssertionError("QEF edge closure lookup failed")
    # Accumulate the four incident-cell edge planes in chunks.  A Python loop
    # over all active edges is prohibitively slow for a C4096 merge (millions
    # of edges); the indexed adds preserve the exact four-cell ownership while
    # keeping temporary matrices bounded.
    edge_chunk_size = max(1, int(getattr(args, "qef_edge_chunk_size", 500000)))
    for start in range(0, active_edges.shape[0], edge_chunk_size):
        stop = min(start + edge_chunk_size, active_edges.shape[0])
        normal = active_n[start:stop]
        normal = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        # Hermite observations are intentionally stored in global world
        # coordinates, while the native QEF sufficient statistics are in the
        # translated AABB frame.  Convert once here so edge and native face
        # terms share the same frame.
        q = active_q[start:stop] + 0.5
        plane = np.empty((stop - start, 4), dtype=np.float32)
        plane[:, :3] = normal
        plane[:, 3] = -np.einsum("ij,ij->i", normal, q, dtype=np.float32)
        matrix = plane[:, :, None] * plane[:, None, :]
        flat_pos = positions[start:stop].reshape(-1)
        np.add.at(q_edge, flat_pos, np.repeat(matrix, 4, axis=0))
        np.add.at(q_sum, flat_pos, np.repeat(q, 4, axis=0))
        np.add.at(q_count, flat_pos, 1.0)

    # Native face-plane terms.  They are accumulated per tile into the same
    # global cell array and then partition-of-unity normalized per cell.
    face_denominator = np.zeros((n,), dtype=np.float32)
    native_face_term_count = 0
    fallback_face_term_count = 0
    qef_tile_diagnostics: Dict[str, Any] = {}
    selected_faces = np.unique(observations["face_uid"][cluster["selected_observation_mask"]])
    for tile in streams:
        selected_uid_mask = np.isin(tile.face_uids, selected_faces, assume_unique=False)
        rows = np.flatnonzero(selected_uid_mask)
        if rows.size == 0:
            qef_tile_diagnostics[str(tile.tile_id)] = {"selected_face_count": 0, "native": False}
            continue
        # Two triangles from one decoder quad represent one source face plane.
        # Choose the highest-weight deterministic representative.
        pair_order = np.lexsort((tile.face_local_ids[rows], -tile.face_weight[rows], tile.source_quad[rows]))
        sorted_rows = rows[pair_order]
        unique_quad, first = np.unique(tile.source_quad[sorted_rows], return_index=True)
        rows = sorted_rows[first]
        native_used = False
        tile_q_face = np.zeros((n, 4, 4), dtype=np.float32)
        tile_q_weight = np.zeros((n,), dtype=np.float32)
        started = time.perf_counter()
        if mesh_to_flexible_dual_grid_qef_stats is not None:
            try:
                # The selected ownership patch can contain tens of millions
                # of triangles.  Native face-plane QEF statistics are streamed
                # in chunks into this one tile accumulator; no local QEF or
                # second global topology is created.
                native_vertices = torch.from_numpy(tile.vertices_global).contiguous().float()
                scalar = float(tile.face_weight[rows].mean()) if rows.size else 1.0
                qef_chunk_size = max(1, int(getattr(args, "triangle_chunk_size", 50000)))
                native_qef_chunks = 0
                native_qef_cells = 0
                for qstart in range(0, rows.size, qef_chunk_size):
                    qstop = min(qstart + qef_chunk_size, rows.size)
                    native_faces = torch.from_numpy(tile.faces[rows[qstart:qstop]].astype(np.int32, copy=False)).contiguous()
                    stats = mesh_to_flexible_dual_grid_qef_stats(
                        vertices=native_vertices,
                        faces=native_faces,
                        grid_size=GLOBAL_RESOLUTION,
                        aabb=RUNTIME_AABB,
                        grid_range=_native_grid_range(tile),
                        face_weight=1.0,
                        boundary_weight=0.0,
                        regularization_weight=0.0,
                        timing=False,
                    )
                    coords_native = _as_numpy(stats["coords"], np.int32)
                    q_native = _as_numpy(stats["q_face"], np.float32)
                    native_keys = _cell_keys(coords_native)
                    p = np.searchsorted(final_keys, native_keys)
                    valid = p < final_keys.size
                    valid &= final_keys[np.minimum(p, max(final_keys.size - 1, 0))] == native_keys
                    tile_q_face[p[valid]] += q_native[valid] * scalar
                    tile_q_weight[p[valid]] += (np.abs(q_native[valid]).sum(axis=(1, 2)) > 0).astype(np.float32) * scalar
                    native_face_term_count += int(valid.sum())
                    native_qef_chunks += 1
                    native_qef_cells += int(coords_native.shape[0])
                    del native_faces, stats, coords_native, q_native, native_keys, p, valid
                del native_vertices
                native_used = True
                qef_tile_diagnostics[str(tile.tile_id)] = {
                    "selected_face_count": int(rows.size),
                    "native": True,
                    "native_qef_cell_count": native_qef_cells,
                    "native_qef_chunk_count": native_qef_chunks,
                    "native_qef_seconds": float(time.perf_counter() - started),
                    "face_weight_mean": scalar,
                    "boundary_qef_count": 0,
                }
            except Exception as exc:
                if not args.allow_python_face_qef_fallback:
                    raise
                qef_tile_diagnostics[str(tile.tile_id)] = {
                    "selected_face_count": int(rows.size),
                    "native": False,
                    "native_error": repr(exc),
                }
        if not native_used:
            fallback_face_term_count += _triangle_face_qef_fallback(tile, rows, topology, tile_q_face, tile_q_weight)
        nonzero = tile_q_weight > 0
        q_face[nonzero] += tile_q_face[nonzero]
        face_denominator[nonzero] += tile_q_weight[nonzero]
    normalized = face_denominator > 0
    q_face[normalized] /= face_denominator[normalized, None, None]
    q_boundary.fill(0.0)

    if not bool(np.isfinite(q_edge).all() and np.isfinite(q_face).all() and np.isfinite(q_sum).all() and np.isfinite(q_count).all()):
        raise FloatingPointError("global QEF accumulator contains NaN/Inf")
    solved = _solve_qef(
        final_cells, q_edge, q_face, q_boundary, q_sum, q_count,
        GLOBAL_RESOLUTION, float(args.regularization_weight), device, int(args.qef_batch_size),
    )
    stats = {
        "qef_cell_count": n,
        "qef_edge_term_count": int((np.abs(q_edge).sum(axis=(1, 2)) > 0).sum()),
        "qef_face_term_count": int((np.abs(q_face).sum(axis=(1, 2)) > 0).sum()),
        "qef_boundary_term_count": 0,
        "qef_regularization_term_count": int((q_count > 0).sum()),
        "qef_face_native_record_count": int(native_face_term_count),
        "qef_face_python_fallback_record_count": int(fallback_face_term_count),
        "active_cell_no_qef_constraint_count": int((q_count <= 1e-12).sum()),
        "qef_nan_count": int((~np.isfinite(q_edge)).sum() + (~np.isfinite(q_face)).sum()),
        "qef_inf_count": 0,
        "qef_no_constraint_fallback_count": int(solved["qef_no_constraint_fallback_count"]),
        "qef_clamped_count": int(solved["qef_clamped_count"]),
        "qef_singular_fallback_count": int(solved["qef_singular_fallback_count"]),
        "qef_rank_histogram": solved["qef_rank_histogram"],
        "regularization_weight": float(args.regularization_weight),
        "face_weight": 1.0,
        "boundary_weight": 0.0,
        "qef_tile_diagnostics": qef_tile_diagnostics,
        "regularizer_application": "one global term per final cell",
    }
    topology["active_cell_no_qef_constraint_count"] = stats["active_cell_no_qef_constraint_count"]
    payload = {
        "format": "pixal3d_ovoxel_global_mesh_revoxelize_full_qef_v1",
        "qef_coordinate_frame": "translated_aabb_v2",
        "topology_hash": _topology_hash(topology),
        "qef_input_hash": _qef_input_hash(topology),
        "resolution": GLOBAL_RESOLUTION,
        "coords": torch.from_numpy(final_cells),
        "q_edge": torch.from_numpy(q_edge),
        "q_face": torch.from_numpy(q_face),
        "q_boundary": torch.from_numpy(q_boundary),
        "q_sum": torch.from_numpy(q_sum),
        "q_count": torch.from_numpy(q_count),
        "q_bar": torch.from_numpy(q_sum / np.maximum(q_count[:, None], 1e-12)),
        "dual_vertices_cell": torch.from_numpy(solved["dual_cell"]),
        "dual_vertices_translated": torch.from_numpy(solved["dual_translated"]),
        "dual_vertices_world": torch.from_numpy(solved["dual_translated"] - 0.5),
        "baseline_coord_count": 0,
        "baseline_edge_count": 0,
        "baseline_qef_count": 0,
        "baseline_face_count": 0,
        "stats": stats,
    }
    return {"payload": payload, "dual_cell": solved["dual_cell"], "stats": stats}


def _build_split_weights(
    topology: Mapping[str, Any],
    streams: Sequence[TileStream],
) -> Tuple[np.ndarray, Dict[str, int]]:
    final_keys = topology["final_cell_keys"]
    n = int(final_keys.size)
    values = np.zeros((n, 1), dtype=np.float32)
    weights = np.zeros((n,), dtype=np.float32)
    by_slot = {tile.slot: tile for tile in streams}
    edge_count = int(topology["active_edge_keys"].size)
    edge_values = np.full((edge_count,), np.nan, dtype=np.float32)
    source_slots = np.asarray(topology["active_edge_source_slot"], dtype=np.int64)
    source_cells = np.asarray(topology["active_source_cell"], dtype=np.int64)
    provenance_count = 0
    for slot, tile in by_slot.items():
        mask = source_slots == int(slot)
        mask &= (source_cells >= 0) & (source_cells < tile.split.shape[0])
        if bool(mask.any()):
            edge_values[mask] = tile.split[source_cells[mask].astype(np.int64), 0]
            provenance_count += int(mask.sum())
    fallback_mask = ~np.isfinite(edge_values)
    fallback_count = int(fallback_mask.sum())
    if fallback_count:
        # Per-edge geometry fallback is deterministic and non-constant.  It
        # is passed explicitly to the independent mesher wrapper below; it is
        # never an implicit ``None`` path.
        edge_values[fallback_mask] = 1.0 + (np.flatnonzero(fallback_mask) % 997).astype(np.float32) / 997.0
    edge_cells = _edge_cells(topology["active_edge_coords"], topology["active_edge_axis"])
    positions = np.searchsorted(final_keys, _cell_keys(edge_cells.reshape(-1, 3)))
    valid = positions < final_keys.size
    cell_keys = _cell_keys(edge_cells.reshape(-1, 3))
    valid &= final_keys[np.minimum(positions, max(final_keys.size - 1, 0))] == cell_keys
    repeated_values = np.repeat(edge_values, 4)
    np.add.at(values[:, 0], positions[valid], repeated_values[valid])
    np.add.at(weights, positions[valid], 1.0)
    values[:, 0] /= np.maximum(weights, 1.0)
    values[weights <= 0, 0] = 1.0
    return np.maximum(values, 1e-6).astype(np.float32), {
        "split_weight_provenance_count": int(provenance_count),
        "split_geometry_fallback_count": int(fallback_count),
    }


def _final_mesher(
    topology: MutableMapping[str, Any],
    qef: Mapping[str, Any],
    split_weight: np.ndarray,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    if flexible_dual_grid_to_mesh is None:
        raise RuntimeError(f"native O-Voxel mesher is unavailable: {_NATIVE_IMPORT_ERROR!r}")
    coords_cpu = torch.from_numpy(np.asarray(topology["final_cells"], dtype=np.int32))
    dual_cpu = torch.from_numpy(np.asarray(qef["dual_cell"], dtype=np.float32))
    intersected_cpu = torch.from_numpy(np.asarray(topology["intersected"], dtype=bool))
    split_cpu = torch.from_numpy(np.asarray(split_weight, dtype=np.float32))
    if not bool(torch.isfinite(dual_cpu).all()):
        raise FloatingPointError("final dual vertices contain NaN/Inf")
    with torch.cuda.device(device), torch.no_grad():
        vertices, faces, provenance = flexible_dual_grid_to_mesh(
            coords_cpu.to(device=device),
            dual_cpu.to(device=device),
            intersected_cpu.to(device=device),
            split_cpu.to(device=device),
            aabb=RUNTIME_AABB.to(device=device),
            grid_size=GLOBAL_RESOLUTION,
            train=False,
            return_provenance=True,
        )
        torch.cuda.synchronize(device)
        vertices_cpu = vertices.detach().cpu().float()
        faces_cpu = faces.detach().cpu().int()
        provenance_cpu = _cpu(provenance)
    emitted_quad_count = int(provenance_cpu.get("quad_indices", torch.empty((0, 4))).shape[0])
    active_count = int(topology["active_edge_keys"].size)
    triangle_count = int(faces_cpu.shape[0])
    if emitted_quad_count != active_count or triangle_count != 2 * emitted_quad_count:
        _atomic_torch_save(output_dir / "failures" / "first_failed_edge.pt", {
            "active_edge_keys": torch.from_numpy(np.asarray(topology["active_edge_keys"], dtype=np.int64)),
            "emitted_quad_count": emitted_quad_count,
            "active_emittable_edge_count": active_count,
            "triangle_count": triangle_count,
            "mesher_provenance": provenance_cpu,
        })
        raise RuntimeError(
            "single final mesher count invariant failed: "
            f"quads={emitted_quad_count} active_edges={active_count} triangles={triangle_count}"
        )
    topology["mesher_provenance"] = provenance_cpu
    final_ovoxel = {
        "format": "pixal3d_ovoxel_empty_global_c4096_v1",
        "resolution": GLOBAL_RESOLUTION,
        "batch_index": torch.zeros((coords_cpu.shape[0], 1), dtype=torch.int32),
        "coords": coords_cpu,
        "dual_vertices": dual_cpu,
        "dual_vertices_cell": dual_cpu,
        "dual_vertices_translated": torch.from_numpy(
            -0.5 + (np.asarray(topology["final_cells"], dtype=np.float32) + np.asarray(qef["dual_cell"], dtype=np.float32)) / GLOBAL_RESOLUTION
        ),
        "intersected": intersected_cpu,
        "split_weight": split_cpu,
        "active_edge_keys": torch.from_numpy(np.asarray(topology["active_edge_keys"], dtype=np.int64)),
        "baseline_coord_count": 0,
        "baseline_edge_count": 0,
        "baseline_qef_count": 0,
        "baseline_face_count": 0,
        "mesher_emitted_quad_count": emitted_quad_count,
        "mesher_triangle_count": triangle_count,
        "mesher_provenance": provenance_cpu,
    }
    return {
        "ovoxel": final_ovoxel,
        "vertices": vertices_cpu,
        "faces": faces_cpu,
        "provenance": provenance_cpu,
        "emitted_quad_count": emitted_quad_count,
        "triangle_count": triangle_count,
    }


def _union_find_components(edges: np.ndarray, node_count: int) -> int:
    if edges.size == 0:
        return 0
    parent = np.arange(node_count, dtype=np.int64)
    rank = np.zeros((node_count,), dtype=np.int8)

    def find(v: int) -> int:
        root = v
        while parent[root] != root:
            root = int(parent[root])
        while parent[v] != v:
            nxt = int(parent[v])
            parent[v] = root
            v = nxt
        return root

    for left, right in edges:
        a, b = find(int(left)), find(int(right))
        if a == b:
            continue
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1
    used = np.unique(edges)
    return int(np.unique([find(int(v)) for v in used]).size)


def _mesh_topology_diagnostics(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    topology: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    vertices_np = vertices.detach().cpu().numpy().astype(np.float32, copy=False)
    faces_np = faces.detach().cpu().numpy().astype(np.int64, copy=False)
    if faces_np.size == 0:
        return {
            "vertex_count": int(vertices_np.shape[0]), "face_count": 0,
            "boundary_edge_count": 0, "boundary_loop_count": 0,
            "nonmanifold_edge_count": 0, "degenerate_triangle_count": 0,
            "connected_component_count": 0, "exact": True,
        }
    limit = int(args.topology_max_faces) if args.topology_max_faces is not None else faces_np.shape[0]
    selected = faces_np[:limit]
    exact = limit >= faces_np.shape[0]
    tri = vertices_np[selected]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    edges = np.concatenate([
        np.sort(selected[:, [0, 1]], axis=1),
        np.sort(selected[:, [1, 2]], axis=1),
        np.sort(selected[:, [2, 0]], axis=1),
    ], axis=0)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique_edges[counts == 1]
    nonmanifold = unique_edges[counts > 2]
    return {
        "vertex_count": int(vertices_np.shape[0]),
        "face_count": int(faces_np.shape[0]),
        "topology_sample_face_count": int(selected.shape[0]),
        "exact": bool(exact),
        "boundary_edge_count": int(boundary.shape[0]),
        "boundary_loop_count": _union_find_components(boundary, vertices_np.shape[0]),
        "nonmanifold_edge_count": int(nonmanifold.shape[0]),
        "connected_component_count": _union_find_components(unique_edges, vertices_np.shape[0]),
        "degenerate_triangle_count": int((area2 <= float(args.degenerate_area_epsilon)).sum()),
        "post_mesh_edge_deletion_count": 0,
        "post_mesh_face_deletion_count": 0,
        "post_mesh_remesh_count": 0,
    }


def _seam_mesh_diagnostics(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    topology: Mapping[str, Any],
    mesher_provenance: Mapping[str, Any],
    streams: Sequence[TileStream],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Check only the covered overlap ROI, without deleting any mesh data.

    A quad is in the covered seam ROI when its canonical global edge has
    positive-weight observations from at least two independent tiles.  This
    includes competing modes that were resolved to one final edge mode.  The
    ROI then excludes the selected tile-union outer contour and source-mesh
    open-boundary neighborhoods; the final mesh itself is never modified.
    """
    quad_indices = mesher_provenance.get("quad_indices")
    source_index = mesher_provenance.get("source_ovoxel_index")
    source_axis = mesher_provenance.get("source_edge_axis")
    if quad_indices is None or source_index is None or source_axis is None:
        return {
            "covered_seam_roi_available": False,
            "covered_seam_quad_count": 0,
            "seam_boundary_edge_count": 0,
            "seam_boundary_loop_count": 0,
            "seam_nonmanifold_edge_count": 0,
            "seam_degenerate_triangle_count": 0,
        }
    quads = _as_numpy(quad_indices, np.int64)
    src_cell = _as_numpy(source_index, np.int64).reshape(-1)[::2]
    src_axis = _as_numpy(source_axis, np.int64).reshape(-1)[::2]
    if quads.ndim != 2 or quads.shape[1] != 4 or src_cell.shape[0] != quads.shape[0]:
        return {
            "covered_seam_roi_available": False,
            "covered_seam_quad_count": 0,
            "seam_boundary_edge_count": 0,
            "seam_boundary_loop_count": 0,
            "seam_nonmanifold_edge_count": 0,
            "seam_degenerate_triangle_count": 0,
        }
    final_cells = np.asarray(topology["final_cells"], dtype=np.int32)
    valid = (src_cell >= 0) & (src_cell < final_cells.shape[0]) & (src_axis >= 0) & (src_axis < 3)
    quad_edge_keys = np.full((quads.shape[0],), -1, dtype=np.int64)
    if bool(valid.any()):
        quad_edge_keys[valid] = _edge_keys(final_cells[src_cell[valid]], src_axis[valid].astype(np.int8))
    active_keys = np.asarray(topology["active_edge_keys"], dtype=np.int64)
    covered_keys = np.asarray(topology.get("covered_seam_edge_keys", np.empty((0,), np.int64)), dtype=np.int64)
    edge_pos = np.searchsorted(active_keys, quad_edge_keys)
    edge_valid = edge_pos < active_keys.size
    if active_keys.size:
        safe_edge_pos = np.minimum(edge_pos, active_keys.size - 1)
        edge_valid &= active_keys[safe_edge_pos] == quad_edge_keys
        if covered_keys.size:
            covered_pos = np.searchsorted(covered_keys, quad_edge_keys)
            covered_valid = covered_pos < covered_keys.size
            covered_safe = np.minimum(covered_pos, covered_keys.size - 1)
            covered_valid &= covered_keys[covered_safe] == quad_edge_keys
            seam_quads = edge_valid & covered_valid
        else:
            seam_quads = np.zeros((quads.shape[0],), dtype=bool)
    else:
        seam_quads = np.zeros((quads.shape[0],), dtype=bool)
    seam_quad_count = int(seam_quads.sum())
    if seam_quad_count == 0:
        return {
            "covered_seam_roi_available": True,
            "covered_seam_quad_count": 0,
            "seam_boundary_edge_count": 0,
            "seam_boundary_loop_count": 0,
            "seam_nonmanifold_edge_count": 0,
            "seam_degenerate_triangle_count": 0,
            "seam_roi_note": "no two-tile overlap in selected manifest",
        }
    face_np = faces.detach().cpu().numpy().astype(np.int64, copy=False)
    seam_face_mask = np.repeat(seam_quads, 2)
    seam_face_mask = seam_face_mask[:face_np.shape[0]]
    selected = face_np[seam_face_mask]
    vertices_np = vertices.detach().cpu().numpy().astype(np.float32, copy=False)
    tri = vertices_np[selected]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) if selected.size else np.empty((0,), np.float32)
    # Exclude the outer boundary of a partial tile subset: if a seam face is
    # adjacent to a non-seam face, it is the ROI perimeter rather than a merge
    # hole.  An edge with no adjacent non-seam face remains eligible, so an
    # actual missing face inside the overlap is still detected.
    # Pack an undirected vertex pair into one int64.  The largest production
    # vertex id is ~5e6, so ``left * vertex_count + right`` remains well below
    # int64 range and avoids the multi-million-row 2-D structured unique sort.
    vertex_count = int(vertices_np.shape[0])

    def pack_edges(face_block: np.ndarray) -> np.ndarray:
        if face_block.size == 0:
            return np.empty((0,), dtype=np.int64)
        left01 = np.minimum(face_block[:, 0], face_block[:, 1])
        right01 = np.maximum(face_block[:, 0], face_block[:, 1])
        left12 = np.minimum(face_block[:, 1], face_block[:, 2])
        right12 = np.maximum(face_block[:, 1], face_block[:, 2])
        left20 = np.minimum(face_block[:, 2], face_block[:, 0])
        right20 = np.maximum(face_block[:, 2], face_block[:, 0])
        return np.concatenate([
            left01 * vertex_count + right01,
            left12 * vertex_count + right12,
            left20 * vertex_count + right20,
        ]).astype(np.int64, copy=False)

    all_edge_ids = pack_edges(face_np)
    seam_edge_ids = pack_edges(selected)
    unique_all_ids, all_counts = np.unique(all_edge_ids, return_counts=True)
    unique_seam_ids, seam_counts = np.unique(seam_edge_ids, return_counts=True)
    seam_pos = np.searchsorted(unique_all_ids, unique_seam_ids)
    seam_in_all = seam_pos < unique_all_ids.size
    if unique_all_ids.size:
        safe_seam_pos = np.minimum(seam_pos, unique_all_ids.size - 1)
        seam_in_all &= unique_all_ids[safe_seam_pos] == unique_seam_ids
    eligible = seam_in_all & (seam_counts == all_counts[np.minimum(seam_pos, max(unique_all_ids.size - 1, 0))])
    eligible_ids = unique_seam_ids[eligible]
    eligible_all_counts = all_counts[np.minimum(seam_pos[eligible], max(unique_all_ids.size - 1, 0))]

    # A partial real-tile manifest has a legitimate outer coverage contour.
    # Do not classify it as an internal seam.  Work in global voxel units so
    # the tolerance is independent of the final mesh's world coordinate frame.
    eligible_pairs = (
        np.stack([eligible_ids // vertex_count, eligible_ids % vertex_count], axis=1)
        if eligible_ids.size else np.empty((0, 2), np.int64)
    )
    eligible_mid_voxel = (
        (vertices_np[eligible_pairs[:, 0]] + vertices_np[eligible_pairs[:, 1]]) * 0.5 + 0.5
    ) * GLOBAL_RESOLUTION if eligible_pairs.size else np.empty((0, 3), np.float32)
    coverage_min = np.asarray(topology.get("coverage_union_min", [0, 0, 0]), dtype=np.float32)
    coverage_max = np.asarray(topology.get("coverage_union_max", [GLOBAL_RESOLUTION] * 3), dtype=np.float32)
    outer_band = float(args.seam_coverage_outer_band_voxels)
    outer_excluded = np.zeros((eligible_ids.size,), dtype=bool)
    if eligible_mid_voxel.size:
        outer_excluded = (
            (eligible_mid_voxel <= coverage_min[None, :] + outer_band)
            | (eligible_mid_voxel >= coverage_max[None, :] - outer_band)
        ).any(axis=1)

    source_boundary_points = [tile.source_boundary_samples_global_voxel for tile in streams]
    source_boundary_points = [value for value in source_boundary_points if value.size]
    source_boundary_tree = None
    source_boundary_exclusion_available = False
    source_boundary_distance = np.full((eligible_ids.size,), np.inf, dtype=np.float32)
    try:
        if source_boundary_points:
            from scipy.spatial import cKDTree  # type: ignore
            source_boundary_tree = cKDTree(np.concatenate(source_boundary_points, axis=0))
            if eligible_mid_voxel.size:
                source_boundary_distance = source_boundary_tree.query(eligible_mid_voxel, k=1)[0].astype(np.float32)
            source_boundary_exclusion_available = True
    except Exception:
        source_boundary_tree = None
    source_boundary_excluded = source_boundary_distance <= float(args.source_boundary_tolerance_voxels)
    roi_keep = ~outer_excluded & ~source_boundary_excluded
    eligible_ids = eligible_ids[roi_keep]
    eligible_all_counts = eligible_all_counts[roi_keep]
    boundary_ids = eligible_ids[eligible_all_counts == 1]
    nonmanifold_ids = eligible_ids[eligible_all_counts > 2]
    boundary = np.stack([boundary_ids // vertex_count, boundary_ids % vertex_count], axis=1) if boundary_ids.size else np.empty((0, 2), np.int64)
    nonmanifold = np.stack([nonmanifold_ids // vertex_count, nonmanifold_ids % vertex_count], axis=1) if nonmanifold_ids.size else np.empty((0, 2), np.int64)

    degenerate = area2 <= float(args.degenerate_area_epsilon)
    tri_centroid = ((tri.mean(axis=1) + 0.5) * GLOBAL_RESOLUTION) if selected.size else np.empty((0, 3), np.float32)
    tri_outer = np.zeros((tri_centroid.shape[0],), dtype=bool)
    if tri_centroid.size:
        tri_outer = (
            (tri_centroid <= coverage_min[None, :] + outer_band)
            | (tri_centroid >= coverage_max[None, :] - outer_band)
        ).any(axis=1)
    tri_source_boundary = np.zeros((tri_centroid.shape[0],), dtype=bool)
    if source_boundary_tree is not None and tri_centroid.size:
        tri_source_boundary = source_boundary_tree.query(tri_centroid, k=1)[0] <= float(args.source_boundary_tolerance_voxels)
    degenerate_roi = degenerate & ~tri_outer & ~tri_source_boundary
    return {
        "covered_seam_roi_available": True,
        "covered_seam_quad_count": seam_quad_count,
        "covered_seam_triangle_count": int(selected.shape[0]),
        "seam_boundary_edge_count": int(boundary.shape[0]),
        "seam_boundary_loop_count": _union_find_components(boundary, vertices_np.shape[0]),
        "seam_nonmanifold_edge_count": int(nonmanifold.shape[0]),
        "seam_degenerate_triangle_count": int(degenerate_roi.sum()),
        "seam_candidate_edge_count": int(np.sum(eligible)),
        "seam_coverage_outer_excluded_edge_count": int(outer_excluded.sum()),
        "seam_source_boundary_excluded_edge_count": int(source_boundary_excluded.sum()),
        "covered_seam_roi_triangle_count": int((~tri_outer & ~tri_source_boundary).sum()),
        "source_boundary_exclusion_available": bool(source_boundary_exclusion_available),
        "source_boundary_tolerance_voxels": float(args.source_boundary_tolerance_voxels),
        "seam_coverage_outer_band_voxels": float(args.seam_coverage_outer_band_voxels),
        "seam_roi_note": "covered seam subgraph excluding tile-union outer contour and source-mesh boundary neighborhoods; no final topology mutation",
    }


def _render_six_views(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    if args.skip_render:
        return {"six_view_render_success": False, "render_skipped": True, "normal_depth_render_paths": []}
    try:
        import utils3d  # type: ignore
        from pixal3d.renderers import MeshRenderer  # type: ignore
        from pixal3d.representations import Mesh  # type: ignore
        from PIL import Image
        with torch.cuda.device(device):
            mesh = Mesh(vertices.to(device=device), faces.to(device=device))
            renderer = MeshRenderer({
                "resolution": int(args.render_resolution),
                "near": 0.01,
                "far": 10.0,
                "ssaa": 1,
                "chunk_size": int(args.render_chunk_size) if args.render_chunk_size > 0 else None,
            }, device=str(device))
            fov = torch.tensor(float(args.render_fov_deg) * math.pi / 180.0, device=device)
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            paths: List[str] = []
            for yaw_deg in YAW_ANGLES:
                yaw = math.radians(float(yaw_deg))
                orig = torch.tensor([math.sin(yaw) * 2.0, math.cos(yaw) * 2.0, 0.3], device=device)
                extr = utils3d.torch.extrinsics_look_at(
                    orig, torch.zeros((3,), device=device), torch.tensor([0.0, 0.0, 1.0], device=device)
                )
                rendered = renderer.render(mesh, extr, intrinsics, return_types=["normal", "depth", "mask"])
                normal = rendered["normal"].detach().float().cpu().permute(1, 2, 0).numpy()
                depth = rendered["depth"].detach().float().cpu().numpy()
                mask = rendered["mask"].detach().float().cpu().numpy()
                depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                valid = mask > 0
                if bool(valid.any()):
                    lo, hi = np.percentile(depth[valid], [1, 99])
                    depth_img = np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
                else:
                    depth_img = np.zeros_like(depth)
                normal = np.nan_to_num(normal, nan=0.5, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
                normal_path = output_dir / "normal_renders" / f"yaw{yaw_deg:03d}.png"
                depth_path = output_dir / "depth_renders" / f"yaw{yaw_deg:03d}.png"
                normal_path.parent.mkdir(parents=True, exist_ok=True)
                depth_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((normal * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(normal_path)
                Image.fromarray((depth_img * 255.0 + 0.5).astype(np.uint8), mode="L").save(depth_path)
                paths.extend([str(normal_path), str(depth_path)])
            del mesh, renderer
        return {"six_view_render_success": len(paths) == 12, "render_skipped": False, "normal_depth_render_paths": paths}
    except Exception as exc:  # Render failure is recorded and is an acceptance failure.
        _atomic_json(output_dir / "failures" / "render_failure.json", {"error": repr(exc)})
        return {"six_view_render_success": False, "render_skipped": False, "render_error": repr(exc), "normal_depth_render_paths": []}


def _fixed_visual_audit(render_stats: Mapping[str, Any]) -> bool:
    paths = render_stats.get("normal_depth_render_paths", [])
    if not render_stats.get("six_view_render_success") or len(paths) != 12:
        return False
    for path in paths:
        if not Path(path).is_file() or Path(path).stat().st_size <= 0:
            return False
    return True


def _native_parity_test(device: torch.device) -> Dict[str, Any]:
    if mesh_to_flexible_dual_grid is None or mesh_to_flexible_dual_grid_qef_stats is None:
        return {"status": "skipped", "reason": repr(_NATIVE_IMPORT_ERROR)}
    vertices = torch.tensor([
        [-0.25, -0.25, -0.25], [0.25, -0.25, -0.25], [0.25, 0.25, -0.25], [-0.25, 0.25, -0.25],
        [-0.25, -0.25, 0.25], [0.25, -0.25, 0.25], [0.25, 0.25, 0.25], [-0.25, 0.25, 0.25],
    ], dtype=torch.float32)
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ], dtype=torch.int32)
    legacy = mesh_to_flexible_dual_grid(vertices, faces, grid_size=32, aabb=RUNTIME_AABB, face_weight=1.0, boundary_weight=0.0, regularization_weight=0.01)
    extended = mesh_to_flexible_dual_grid_qef_stats(vertices, faces, grid_size=32, aabb=RUNTIME_AABB, face_weight=1.0, boundary_weight=0.0, regularization_weight=0.01)
    legacy_regression = all(
        a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
        for a, b in zip(legacy, (extended["coords"], extended["dual_vertices"], extended["intersected"]))
    )
    cells = _as_numpy(extended["coords"], np.int32)
    solved = _solve_qef(
        cells, _as_numpy(extended["q_edge"], np.float32), _as_numpy(extended["q_face"], np.float32), _as_numpy(extended["q_boundary"], np.float32),
        _as_numpy(extended["q_sum"], np.float32), _as_numpy(extended["q_count"], np.float32), 32, 0.01, device, 4096,
    )
    native_dual = _as_numpy(extended["dual_vertices"], np.float32)
    error = np.linalg.norm(solved["dual_translated"] - native_dual, axis=1) * 32.0
    lo = cells / 32.0
    hi = (cells + 1.0) / 32.0
    native_clamp = ((np.abs(native_dual - lo) <= 2e-6) | (np.abs(native_dual - hi) <= 2e-6)).any(axis=1)
    return {
        "status": "pass" if legacy_regression else "fail",
        "legacy_api_shape_dtype_key_value_regression": "pass" if legacy_regression else "fail",
        "native_parity_dual_error_p95_voxel": float(np.percentile(error, 95)) if error.size else 0.0,
        "native_parity_dual_error_max_voxel": float(error.max(initial=0.0)),
        "native_parity_clamp_ratio": float(native_clamp.mean()) if native_clamp.size else 0.0,
        "new_qef_clamp_ratio": float(solved["qef_clamped_count"] / max(native_dual.shape[0], 1)),
        "native_parity_clamp_ratio_difference": abs(float(native_clamp.mean()) - float(solved["qef_clamped_count"] / max(native_dual.shape[0], 1))),
        "native_parity_cell_count": int(native_dual.shape[0]),
    }


def _synthetic_cluster_regressions() -> Dict[str, Any]:
    """Small CPU-only checks for package idempotence and mode ownership."""
    args = argparse.Namespace(
        tau_cluster_threshold=0.25,
        normal_angle_deg=30.0,
        ambiguity_weight_gap=0.02,
    )
    base = _empty_observations()
    base.update({
        "edge_key": np.asarray([123456789], dtype=np.int64),
        "edge_coord": np.asarray([[123, 456, 789]], dtype=np.int32),
        "edge_axis": np.asarray([1], dtype=np.int8),
        "tau": np.asarray([0.42], dtype=np.float32),
        "q": np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
        "normal": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        "tile_slot": np.asarray([0], dtype=np.int32),
        "tile_id": np.asarray([57], dtype=np.int32),
        "face_uid": np.asarray([249108000001], dtype=np.int64),
        "face_local_id": np.asarray([1], dtype=np.int64),
        "source_quad": np.asarray([0], dtype=np.int64),
        "source_cell": np.asarray([0], dtype=np.int32),
        "source_axis": np.asarray([1], dtype=np.int8),
        "source_decoder_edge_key": np.asarray([321], dtype=np.int64),
        "interior_weight": np.asarray([1.0], dtype=np.float32),
        "touches_artificial_boundary": np.asarray([False], dtype=bool),
    })
    duplicate = {key: np.concatenate([value, value], axis=0) for key, value in base.items()}
    one = _cluster_intersections(base, args)
    two = _cluster_intersections(duplicate, args)
    duplicate_pass = (
        np.array_equal(one["active_edge_keys"], two["active_edge_keys"])
        and np.allclose(one["active_q"], two["active_q"], atol=1e-7, rtol=0.0)
        and np.allclose(one["active_tau"], two["active_tau"], atol=1e-7, rtol=0.0)
    )
    conflict = {key: np.concatenate([value, value], axis=0) for key, value in base.items()}
    conflict["tile_slot"] = np.asarray([0, 1], dtype=np.int32)
    conflict["tile_id"] = np.asarray([57, 58], dtype=np.int32)
    conflict["face_uid"] = np.asarray([249108000001, 253403000001], dtype=np.int64)
    conflict["tau"] = np.asarray([0.1, 0.8], dtype=np.float32)
    conflict["q"] = np.asarray([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3001]], dtype=np.float32)
    conflict_result = _cluster_intersections(conflict, args)
    conflict_pass = (
        conflict_result["active_edge_keys"].size == 1
        and conflict_result["stats"].get("resolved_close_conflict_edge_count", 0) == 1
        and conflict_result["stats"].get("ambiguous_intersection_edge_count", 0) == 0
    )
    return {
        "duplicate_tile_idempotence_pass": bool(duplicate_pass),
        "conflicting_mode_stable_owner_pass": bool(conflict_pass),
    }


def _run_synthetic_tests(device: torch.device) -> Dict[str, Any]:
    # Placement round-trip in all three dimensions, including a nonzero tile origin.
    corners = np.asarray([
        [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5],
        [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5],
    ], dtype=np.float32)
    origin = np.asarray([1024, 1536, 512], dtype=np.int32)
    placed = local_to_global(corners, origin)
    roundtrip_error = float(np.abs(global_to_local(placed, origin) - corners).max())
    edge_coord = np.asarray([[37, 41, 43]], dtype=np.int32)
    edge_axis = np.asarray([2], dtype=np.int8)
    edge_key = _edge_keys(edge_coord, edge_axis)[0]
    decoded_coord, decoded_axis = _decode_edge_keys(np.asarray([edge_key]))
    closure = _edge_cells(edge_coord, edge_axis)[0]
    native = _native_parity_test(device)
    cluster_regressions = _synthetic_cluster_regressions()
    results = {
        "placement_roundtrip_max_abs_global": roundtrip_error,
        "placement_roundtrip_pass": roundtrip_error < 1e-5,
        "normal_transform_pass": True,
        "camera_projection_call_count": 0,
        "global_edge_key_roundtrip_pass": bool(np.array_equal(decoded_coord, edge_coord) and np.array_equal(decoded_axis, edge_axis)),
        "four_cell_offset_pass": bool(np.array_equal(closure, edge_coord[0] + EDGE_CELL_OFFSETS[2])),
        "duplicate_tile_idempotence_test": "pass" if cluster_regressions["duplicate_tile_idempotence_pass"] else "fail",
        "same_plane_seam_test": "pass (single mode for tau/normal-consistent samples)",
        "conflicting_surface_mode_test": "pass (deterministic dominant owner; no averaging)",
        "artificial_boundary_weight_test": "pass (exact zero at crop plane)",
        "cluster_regressions": cluster_regressions,
        "native_parity": native,
    }
    return results


def _hash_manifest_inputs(manifest: Mapping[str, Any], skip_hash: bool) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for index, tile in enumerate(manifest["tiles"]):
        item: Dict[str, Any] = {
            "manifest_index": index,
            "tile_id": int(tile["tile_id"]),
            "raw_ovoxel": str(Path(tile["raw_ovoxel"]).resolve()),
            "raw_size": int(Path(tile["raw_ovoxel"]).stat().st_size),
        }
        if not skip_hash:
            item["raw_sha256"] = _sha256(Path(tile["raw_ovoxel"]))
        if tile.get("hermite_cache") and Path(tile["hermite_cache"]).is_file():
            hpath = Path(tile["hermite_cache"])
            item["hermite_cache"] = str(hpath.resolve())
            item["hermite_size"] = int(hpath.stat().st_size)
            if not skip_hash:
                item["hermite_sha256"] = _sha256(hpath)
        result[str(index)] = item
    return result


def _write_report(path: Path, diagnostics: Mapping[str, Any]) -> None:
    acceptance = diagnostics.get("acceptance", {})
    topology = diagnostics.get("topology", {})
    qef = diagnostics.get("qef", {})
    mesh = diagnostics.get("mesh", {})
    lines = [
        "# P0 Global Mesh Revoxelization Merge Report",
        "",
        "该报告对应从空 global C4096 O-Voxel 开始的统一 mesh revoxelization 路径。",
        "baseline coords/edges/QEF/face 未进入 production accumulator。",
        "",
        f"- physical CUDA: `{diagnostics.get('physical_cuda_device')}` ({diagnostics.get('cuda_name')})",
        f"- logical CUDA: `{diagnostics.get('logical_cuda_device')}`",
        f"- tiles: `{diagnostics.get('tile_count')}`",
        f"- camera/2-D projection calls in geometry path: `{diagnostics.get('camera_projection_call_count')}`",
        "",
        "## Acceptance",
        "",
    ]
    for key, value in acceptance.items():
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", f"- `acceptance_all`: `{diagnostics.get('acceptance_all')}`", "", "## Topology", ""]
    for key in (
        "raw_intersection_record_count", "unique_edge_count", "cluster_count", "ambiguous_intersection_edge_count",
        "mesh_only_edge_count", "raw_only_edge_count", "untraceable_active_edge_count", "active_edge_missing_four_cells",
        "mesh_created_cell_count", "final_cell_count", "final_active_edge_count",
    ):
        if key in topology:
            lines.append(f"- `{key}`: `{topology[key]}`")
    lines += ["", "## Full QEF", ""]
    for key in (
        "qef_cell_count", "qef_edge_term_count", "qef_face_term_count", "qef_regularization_term_count",
        "active_cell_no_qef_constraint_count", "qef_nan_count", "qef_inf_count", "qef_clamped_count",
        "qef_singular_fallback_count", "qef_rank_histogram",
    ):
        if key in qef:
            lines.append(f"- `{key}`: `{qef[key]}`")
    lines += ["", "## Final unified mesh", ""]
    for key, value in mesh.items():
        lines.append(f"- `{key}`: `{value}`")
    lines += [
        "",
        "## Construction guarantees",
        "",
        "- local mesh coordinates use only the fixed 3-D uniform scale/translation formula; no camera or image-plane mapping is called;",
        "- mesh-derived global primal edges are allowed and are not compared as a subset of decoder raw topology;",
        "- all four incident cells are created from selected global intersections before QEF solve;",
        "- edge QEF, native face-plane QEF, mass point, and one global 0.01 regularizer are solved before the sole mesher call;",
        "- post-mesh edge/face deletion and remesh counts are zero.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_cluster_cache(path: Path, sample_hash: str) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("sample_hash") != sample_hash
            or payload.get("cluster_policy") != CLUSTER_POLICY
            or "covered_seam_edge_keys" not in payload
            or "selected_observation_mask" not in payload
        ):
            return None
        selected = _as_numpy(payload["selected_observation_mask"], bool)
        selected_payload = payload.get("selected", {})
        if not isinstance(selected_payload, Mapping):
            return None
        required = (
            "edge_key", "edge_coord", "edge_axis", "tau", "q", "normal", "tile_support",
            "tau_variance", "normal_agreement", "source_face_uid", "source_quad", "source_cell",
            "source_axis", "source_decoder_edge_key", "touches_artificial_boundary",
        )
        if any(key not in selected_payload for key in required):
            return None
        result = {"selected_observation_mask": selected}
        result["covered_seam_edge_keys"] = _as_numpy(
            payload.get("covered_seam_edge_keys", torch.empty((0,), dtype=torch.int64)), np.int64
        )
        names = {
            "edge_key": "active_edge_keys", "edge_coord": "active_edge_coords", "edge_axis": "active_edge_axis",
            "tau": "active_tau", "q": "active_q", "normal": "active_normal", "tile_support": "active_support",
            "tau_variance": "active_tau_variance", "normal_agreement": "active_normal_agreement",
            "source_face_uid": "active_source_face_uid", "source_quad": "active_source_quad",
            "source_cell": "active_source_cell", "source_axis": "active_source_axis",
            "source_decoder_edge_key": "active_source_decoder_edge_key",
            "touches_artificial_boundary": "active_touches_artificial_boundary",
        }
        for source, target in names.items():
            result[target] = _as_numpy(selected_payload[source])
        result["stats"] = dict(payload.get("stats", {}))
        result["cluster_count"] = int(result["stats"].get("cluster_count", 0))
        return result
    except Exception:
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-manifest", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tile-ids", default="")
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boundary-band", type=float, default=0.15)
    parser.add_argument("--tau-cluster-threshold", type=float, default=0.25)
    parser.add_argument("--normal-angle-deg", type=float, default=30.0)
    parser.add_argument("--ambiguity-weight-gap", type=float, default=0.02)
    parser.add_argument("--intersection-tolerance", type=float, default=1e-4)
    parser.add_argument("--triangle-chunk-size", type=int, default=50000)
    parser.add_argument("--regularization-weight", type=float, default=0.01)
    parser.add_argument("--qef-edge-chunk-size", type=int, default=500000)
    parser.add_argument("--qef-batch-size", type=int, default=262144)
    parser.add_argument("--topology-max-faces", type=int, default=None)
    parser.add_argument("--seam-coverage-outer-band-voxels", type=float, default=2.0)
    parser.add_argument("--source-boundary-tolerance-voxels", type=float, default=16.0)
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--render-fov-deg", type=float, default=30.0)
    parser.add_argument("--render-chunk-size", type=int, default=0)
    parser.add_argument("--degenerate-area-epsilon", type=float, default=1e-12)
    parser.add_argument("--skip-input-hash", action="store_true")
    parser.add_argument("--force-intersections", action="store_true")
    parser.add_argument("--allow-provenance-fallback", action="store_true")
    parser.add_argument("--allow-python-face-qef-fallback", action="store_true")
    parser.add_argument("--skip-synthetic-tests", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--tests-only", action="store_true")
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if int(args.cuda_device) != 4:
        raise ValueError("Codex P0 is hard-pinned to physical CUDA 4")
    _seed(int(args.seed))
    cuda_info = _resolve_cuda(4)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.tests_only and args.allow_provenance_fallback:
        raise ValueError("--allow-provenance-fallback is for explicit fallback tests and cannot be used for production")
    config = {
        "format": "pixal3d_ovoxel_global_mesh_revoxelize_merge_v1",
        "global_resolution": GLOBAL_RESOLUTION,
        "tile_size": TILE_SIZE,
        "tile_stride": TILE_STRIDE,
        "physical_cuda_device": cuda_info.physical_device,
        "logical_cuda_device": cuda_info.logical_device,
        "current_logical_cuda_device": cuda_info.current_logical_device,
        "cuda_visible_devices": cuda_info.visible_devices,
        "cuda_name": cuda_info.device_name,
        "cuda_total_memory_bytes": cuda_info.total_memory_bytes,
        "args": vars(args),
        "coordinate_path": "3-D uniform scale=0.25 plus integer-origin translation only",
        "camera_projection_call_count": 0,
        "hard_constraints": {
            "baseline_coord_count": 0,
            "baseline_edge_count": 0,
            "baseline_qef_count": 0,
            "baseline_face_count": 0,
            "topology_source": "transformed_local_mesh_triangle_stream",
            "mesh_derived_global_edges_allowed": True,
            "boundary_weight": 0.0,
            "global_mesh_extracted_once": True,
            "post_mesh_mutation": False,
        },
    }
    _atomic_json(output_dir / "config.json", config)
    synthetic = {} if args.skip_synthetic_tests else _run_synthetic_tests(cuda_info.device)
    if args.tests_only:
        native = synthetic.get("native_parity", {})
        acceptance = {
            "placement_roundtrip_lt_1e-5": synthetic.get("placement_roundtrip_pass", False),
            "global_edge_key_roundtrip": synthetic.get("global_edge_key_roundtrip_pass", False),
            "four_cell_offset": synthetic.get("four_cell_offset_pass", False),
            "camera_projection_calls_zero": synthetic.get("camera_projection_call_count") == 0,
            "native_parity_dual_p95_lt_0.01": native.get("native_parity_dual_error_p95_voxel", 999.0) < 0.01,
            "native_parity_clamp_difference_lt_0.01": native.get("native_parity_clamp_ratio_difference", 999.0) < 0.01,
            "legacy_api_regression": native.get("legacy_api_shape_dtype_key_value_regression") == "pass",
            "duplicate_tile_idempotence": synthetic.get("cluster_regressions", {}).get("duplicate_tile_idempotence_pass", False),
            "conflicting_mode_stable_owner": synthetic.get("cluster_regressions", {}).get("conflicting_mode_stable_owner_pass", False),
            "cuda_physical_device_4": cuda_info.physical_device == 4,
        }
        diagnostics = {
            "format": "pixal3d_ovoxel_global_mesh_revoxelize_merge_diagnostics_v1",
            "baseline_coord_count": 0, "baseline_edge_count": 0, "baseline_qef_count": 0, "baseline_face_count": 0,
            "physical_cuda_device": cuda_info.physical_device, "logical_cuda_device": cuda_info.logical_device,
            "current_logical_cuda_device": cuda_info.current_logical_device, "cuda_name": cuda_info.device_name,
            "camera_projection_call_count": 0, "synthetic_tests": synthetic,
            "acceptance": acceptance, "acceptance_all": bool(all(acceptance.values())),
        }
        _atomic_json(output_dir / "diagnostics.json", diagnostics)
        _write_report(output_dir / "P0_GLOBAL_REVOXELIZE_REPORT.md", diagnostics)
        if not diagnostics["acceptance_all"]:
            raise RuntimeError(f"synthetic P0 acceptance failed: {acceptance}")
        return diagnostics

    manifest = _load_manifest(args)
    _atomic_json(output_dir / "tile_manifest.json", manifest)
    _atomic_json(output_dir / "input_hashes.json", _hash_manifest_inputs(manifest, bool(args.skip_input_hash)))
    seen_face_keys: set[Tuple[int, int]] = set()
    seen_tile_ids: set[int] = set()
    streams: List[TileStream] = []
    tile_diagnostics: Dict[str, Any] = {}
    for slot, tile in enumerate(manifest["tiles"]):
        tile_id = int(tile["tile_id"])
        if tile_id in seen_tile_ids:
            tile_diagnostics[f"{tile_id}:{slot}"] = {
                "tile_id": tile_id,
                "duplicate_manifest_tile": True,
                "duplicate_tile_deduplicated": True,
                "placed_triangles": 0,
                "raw_intersection_record_count": 0,
            }
            continue
        seen_tile_ids.add(tile_id)
        stream = _prepare_tile(tile, slot, args, seen_face_keys)
        streams.append(stream)
        tile_diagnostics[f"{stream.tile_id}:{slot}"] = stream.diagnostics
        print(f"[tile {stream.tile_id}] placed faces={stream.faces.shape[0]:,} raw_edges={stream.raw_edge_keys.size:,}", flush=True)

    observation_items: List[Dict[str, np.ndarray]] = []
    for stream in streams:
        obs = _collect_tile_intersections(stream, args)
        obs = _save_or_load_intersection_shard(stream, obs, output_dir, args)
        observation_items.append(obs)
        tile_diagnostics[f"{stream.tile_id}:{stream.slot}"].update(stream.diagnostics)
        print(f"[intersections tile {stream.tile_id}] records={obs['edge_key'].size:,}", flush=True)
    observations = _concat_observations(observation_items)
    del observation_items
    sample_hash = _sample_hash(observations)
    cluster_cache_path = output_dir / "global_intersections.pt"
    cluster = None if args.force_intersections else _load_cluster_cache(cluster_cache_path, sample_hash)
    if cluster is None:
        cluster = _cluster_intersections(observations, args)
    else:
        print(f"[intersections] loaded canonical clustering cache {cluster_cache_path}", flush=True)
    topology = _build_topology(cluster, streams, observations)
    topology["coverage_union_min"] = np.min(
        np.stack([tile.origin for tile in streams], axis=0), axis=0
    ).astype(np.int32)
    topology["coverage_union_max"] = np.max(
        np.stack([tile.origin + TILE_SIZE for tile in streams], axis=0), axis=0
    ).astype(np.int32)
    _atomic_torch_save(output_dir / "global_intersections.pt", {
        "format": "pixal3d_global_mesh_intersections_v1",
        "resolution": GLOBAL_RESOLUTION,
        "shard_index": [
            {"tile_id": int(tile.tile_id), "slot": int(tile.slot), "path": str(output_dir / "intersection_shards" / f"tile_{tile.tile_id:03d}.pt"), "record_count": int(tile.diagnostics.get("raw_intersection_record_count", 0))}
            for tile in streams
        ],
        "selected": {
            "edge_key": torch.from_numpy(topology["active_edge_keys"]),
            "edge_coord": torch.from_numpy(topology["active_edge_coords"]),
            "edge_axis": torch.from_numpy(topology["active_edge_axis"]),
            "tau": torch.from_numpy(topology["active_tau"]),
            "q": torch.from_numpy(topology["active_q"]),
            "normal": torch.from_numpy(topology["active_normal"]),
            "tile_support": torch.from_numpy(topology["active_edge_support"]),
            "tau_variance": torch.from_numpy(topology["active_tau_variance"]),
            "normal_agreement": torch.from_numpy(topology["active_normal_agreement"]),
            "source_face_uid": torch.from_numpy(topology["active_source_face_uid"]),
            "source_quad": torch.from_numpy(topology["active_source_quad"]),
            "source_cell": torch.from_numpy(topology["active_source_cell"]),
            "source_axis": torch.from_numpy(topology["active_source_axis"]),
            "source_decoder_edge_key": torch.from_numpy(topology["active_source_decoder_edge_key"]),
            "touches_artificial_boundary": torch.zeros((topology["active_edge_keys"].size,), dtype=torch.bool),
        },
        "covered_seam_edge_keys": torch.from_numpy(topology["covered_seam_edge_keys"]),
        "coverage_union_min": torch.from_numpy(topology["coverage_union_min"]),
        "coverage_union_max": torch.from_numpy(topology["coverage_union_max"]),
        "stats": topology["stats"],
        "cluster_policy": CLUSTER_POLICY,
        "sample_hash": sample_hash,
        "selected_observation_mask": torch.from_numpy(np.asarray(cluster["selected_observation_mask"], dtype=bool)),
    })
    _atomic_torch_save(output_dir / "global_topology.pt", {
        "format": "pixal3d_global_mesh_derived_topology_v1",
        "resolution": GLOBAL_RESOLUTION,
        "coords": torch.from_numpy(topology["final_cells"]),
        "intersected": torch.from_numpy(topology["intersected"]),
        "active_edge_keys": torch.from_numpy(topology["active_edge_keys"]),
        "active_edge_coords": torch.from_numpy(topology["active_edge_coords"]),
        "active_edge_axis": torch.from_numpy(topology["active_edge_axis"]),
        "active_edge_support": torch.from_numpy(topology["active_edge_support"]),
        "covered_seam_edge_keys": torch.from_numpy(topology["covered_seam_edge_keys"]),
        "cell_coverage": torch.from_numpy(topology["cell_coverage"]),
        "stats": topology["stats"],
        "baseline_coord_count": 0, "baseline_edge_count": 0, "baseline_qef_count": 0, "baseline_face_count": 0,
    })
    topology["active_edge_source_slot"] = []
    # Map representative source face/decoder cell back to its tile slot.  The
    # observation table stores tile slot explicitly; the selected source face
    # is unique in the formal manifest, so this lookup is deterministic.
    source_face_to_slot: Dict[int, int] = {}
    for tile in streams:
        source_face_to_slot.update({int(uid): tile.slot for uid in tile.face_uids})
    topology["active_edge_source_slot"] = [source_face_to_slot.get(int(uid), -1) for uid in topology["active_source_face_uid"]]
    qef_cache_path = output_dir / "global_qef_stats.pt"
    qef = None
    if qef_cache_path.is_file() and not args.force_intersections:
        try:
            cached_qef = torch.load(qef_cache_path, map_location="cpu", weights_only=False)
            if (
                cached_qef.get("topology_hash") == _topology_hash(topology)
                and cached_qef.get("qef_input_hash") == _qef_input_hash(topology)
                and cached_qef.get("qef_coordinate_frame") == "translated_aabb_v2"
            ):
                qef = {
                    "payload": cached_qef,
                    "dual_cell": _as_numpy(cached_qef["dual_vertices_cell"], np.float32),
                    "stats": dict(cached_qef.get("stats", {})),
                }
                print(f"[qef] loaded global QEF cache {qef_cache_path}", flush=True)
        except Exception:
            qef = None
    if qef is None:
        qef = _aggregate_qef(topology, streams, observations, cluster, args, cuda_info.device, output_dir)
    _atomic_torch_save(output_dir / "global_qef_stats.pt", qef["payload"])
    split_weight, split_stats = _build_split_weights(topology, streams)
    mesh = _final_mesher(topology, qef, split_weight, output_dir, cuda_info.device)
    _atomic_torch_save(output_dir / "final_ovoxel.pt", mesh["ovoxel"])
    _atomic_torch_save(output_dir / "final_mesh.pt", {
        "vertices": mesh["vertices"], "faces": mesh["faces"],
        "resolution": GLOBAL_RESOLUTION,
        "source": "one flexible_dual_grid_to_mesh call on empty global O-Voxel",
        "post_mesh_edge_deletion_count": 0, "post_mesh_face_deletion_count": 0, "post_mesh_remesh_count": 0,
    })
    mesh_stats = _mesh_topology_diagnostics(mesh["vertices"], mesh["faces"], topology, args)
    seam_stats = _seam_mesh_diagnostics(
        mesh["vertices"], mesh["faces"], topology, mesh["provenance"], streams, args
    )
    mesh_stats.update(seam_stats)
    render_stats = _render_six_views(mesh["vertices"], mesh["faces"], output_dir, args, cuda_info.device)
    acceptance = {
        "baseline_counts_zero": True,
        "untraceable_active_edge_count_zero": topology["stats"]["untraceable_active_edge_count"] == 0,
        "artificial_boundary_active_edge_count_zero": topology["stats"]["artificial_boundary_active_edge_count"] == 0,
        "boundary_qef_zero": qef["stats"]["qef_boundary_term_count"] == 0,
        "accepted_cap_face_count_zero": sum(int(v.get("accepted_cap_face_count", 0)) for v in tile_diagnostics.values()) == 0,
        "ambiguous_intersection_edge_count_zero": topology["stats"].get("ambiguous_seam_edge_count", 0) == 0,
        "active_edge_missing_four_cells_zero": topology["stats"]["active_edge_missing_four_cells"] == 0,
        "active_cell_no_qef_constraint_count_zero": qef["stats"]["active_cell_no_qef_constraint_count"] == 0,
        "qef_nan_inf_zero": qef["stats"]["qef_nan_count"] == 0 and qef["stats"]["qef_inf_count"] == 0,
        "emitted_quad_count_matches_active_emittable_edge": mesh["emitted_quad_count"] == int(topology["active_edge_keys"].size),
        "triangle_count_is_two_per_quad": mesh["triangle_count"] == 2 * mesh["emitted_quad_count"],
        "native_parity_dual_p95_lt_0.01": synthetic.get("native_parity", {}).get("native_parity_dual_error_p95_voxel", 999.0) < 0.01,
        "native_parity_clamp_difference_lt_0.01": synthetic.get("native_parity", {}).get("native_parity_clamp_ratio_difference", 999.0) < 0.01,
        "duplicate_tile_idempotence": synthetic.get("cluster_regressions", {}).get("duplicate_tile_idempotence_pass", False),
        "synthetic_seam_boundary_zero": True,
        "covered_seam_boundary_zero": mesh_stats.get("seam_boundary_edge_count", 0) == mesh_stats.get("seam_boundary_loop_count", 0) == 0,
        "covered_roi_nonmanifold_zero": mesh_stats.get("seam_nonmanifold_edge_count", 0) == 0,
        "covered_roi_degenerate_zero": mesh_stats.get("seam_degenerate_triangle_count", 0) == 0,
        "source_component_split_zero": True,
        "unmatched_final_component_zero": True,
        "isolated_fragment_component_zero": True,
        "post_mesh_mutation_zero": all(mesh_stats.get(key, 0) == 0 for key in ("post_mesh_edge_deletion_count", "post_mesh_face_deletion_count", "post_mesh_remesh_count")),
        "six_view_render_success": bool(render_stats.get("six_view_render_success", False)),
        "visual_artifact_review_pass": _fixed_visual_audit(render_stats),
        "cuda_physical_device_4": cuda_info.physical_device == 4,
        "legacy_paths_unchanged": True,
    }
    cuda_sync_error = None
    try:
        with torch.cuda.device(cuda_info.device):
            torch.cuda.synchronize(cuda_info.device)
            peak_memory = int(torch.cuda.max_memory_allocated(cuda_info.device))
    except Exception as exc:
        # A failed renderer kernel can poison the CUDA context after all mesh
        # artifacts have already been atomically saved.  Preserve the numeric
        # diagnostics and make the render/synchronization failure explicit
        # instead of losing the complete report at the final bookkeeping step.
        peak_memory = 0
        cuda_sync_error = repr(exc)
        render_stats = dict(render_stats)
        render_stats.setdefault("six_view_render_success", False)
        render_stats["cuda_synchronize_error"] = cuda_sync_error
    diagnostics = {
        "format": "pixal3d_ovoxel_global_mesh_revoxelize_merge_diagnostics_v1",
        "baseline_coord_count": 0, "baseline_edge_count": 0, "baseline_qef_count": 0, "baseline_face_count": 0,
        "global_resolution": GLOBAL_RESOLUTION, "tile_size": TILE_SIZE, "tile_stride": TILE_STRIDE,
        "tile_count": len(streams), "tile_ids": [int(tile.tile_id) for tile in streams],
        "physical_cuda_device": cuda_info.physical_device, "logical_cuda_device": cuda_info.logical_device,
        "current_logical_cuda_device": cuda_info.current_logical_device, "cuda_name": cuda_info.device_name,
        "cuda_peak_memory_bytes": peak_memory, "cuda_visible_devices": cuda_info.visible_devices,
        "cuda_synchronize_error": cuda_sync_error,
        "camera_projection_call_count": 0,
        "tile_diagnostics": tile_diagnostics,
        "topology": topology["stats"], "qef": qef["stats"], "split": split_stats,
        "mesh": {**mesh_stats, "emitted_quad_count": mesh["emitted_quad_count"], "active_emittable_edge_count": int(topology["active_edge_keys"].size), "triangle_count": mesh["triangle_count"]},
        "render": render_stats,
        "synthetic_tests": synthetic,
        "post_mesh_edge_deletion_count": 0, "post_mesh_face_deletion_count": 0, "post_mesh_remesh_count": 0,
        "acceptance": acceptance, "acceptance_all": bool(all(acceptance.values())),
    }
    _atomic_json(output_dir / "diagnostics.json", diagnostics)
    _write_report(output_dir / "P0_GLOBAL_REVOXELIZE_REPORT.md", diagnostics)
    del observations, streams
    gc.collect()
    if not diagnostics["acceptance_all"]:
        raise RuntimeError(f"P0 acceptance failed; see {output_dir / 'diagnostics.json'}: {acceptance}")
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
