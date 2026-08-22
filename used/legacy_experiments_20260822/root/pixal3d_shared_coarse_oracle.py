#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final-PureHR shared-coarse oracle.

This module implements the field-space experiment specified in ``Codex.md``.
It intentionally has no flow sampler, encoder, guidance, Euler step,
``_xstart_to_pred`` call, re-encoding, timestep weight, or PBR clamp.  The
only model operation permitted by this route is one decode of an already
materialized final PureHR endpoint when a direct final PBR field is not
available.

The default command is a strict Phase-A preflight.  Endpoint roots may be
repeated on the command line; legacy roots are matched by their exact
``tile_camera.json`` box and require the explicit ``--allow-box-reuse`` opt-in.
The preflight refuses to substitute range-null/MRA/per-step artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix, load_npz, save_npz
from scipy.sparse.linalg import lsmr
from scipy.spatial import cKDTree


FORMAT = "pixal3d_final_purehr_shared_coarse_oracle_v1"
CANONICAL_IMAGE_SIZE = 4096
TILE_SIZE = 1024
TILE_STRIDE = 512
FINE_RESOLUTION = 1024
COARSE_RESOLUTION = 256
PBR_CHANNELS = 6
GROUPS: Dict[str, slice] = {
    "RGB": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}

# The formal valid ensemble used by the available CUDA4 fixed-field cache is
# 49 layout positions with the empty Tile06 omitted.  A later preparation
# batch also omitted Tile35; that mismatch must never be silently compared.
EXPECTED_FORMAL_48 = frozenset(set(range(49)) - {6})
FORMAL_VALID_TILE_IDS = EXPECTED_FORMAL_48
PHASE_A_TILE_IDS = frozenset({18, 19, 20, 25, 26, 27, 32, 33, 34})
CONSENSUS_MIN_DONORS = 2
NUMERICAL_EPS = 1e-12

_DIRECT_FIELD_NAMES = frozenset(
    {
        "h.pt",
        "H.pt",
        "field.pt",
        "pbr_field.pt",
        "final_field.pt",
        "pure_hr_field.pt",
        "purehr_field.pt",
        "final_purehr_field.pt",
    }
)
_ENDPOINT_NAMES = frozenset(
    {
        "pure_hr_endpoint.pt",
        "purehr_endpoint.pt",
        "final_purehr_endpoint.pt",
        "endpoint.pt",
        "endpoints.pt",
    }
)
_FORBIDDEN_MARKERS = (
    "range_null",
    "mra",
    "projector",
    "step_",
    "projection",
    "trajectory",
    "perstep",
    "initial_state",
)


@dataclass(frozen=True)
class Candidate:
    tile_id: int
    path: Path
    kind: str
    rejection: Optional[str] = None
    match_mode: str = "tile_dir"
    source_root: Optional[Path] = None


@dataclass
class TileField:
    tile_id: int
    box: Tuple[int, int, int, int]
    transform: Any
    geometry: Any
    points: torch.Tensor
    G: torch.Tensor
    H: torch.Tensor
    Delta: torch.Tensor
    hidden: torch.Tensor
    observed: torch.Tensor
    fine_coords: torch.Tensor
    coarse_coords: torch.Tensor
    hidden_rows: torch.Tensor
    pure_hidden_ids: torch.Tensor
    P_full: Optional[csr_matrix]
    P_hidden: csr_matrix
    projector: Any
    projector_coverage: torch.Tensor
    projector_info: Dict[str, Any]
    coarse: torch.Tensor
    C: torch.Tensor
    D: torch.Tensor
    raw_consensus: Optional[torch.Tensor] = None
    donor_count: Optional[torch.Tensor] = None
    C_shared: Optional[torch.Tensor] = None
    C_private: Optional[torch.Tensor] = None
    Y_null: Optional[torch.Tensor] = None
    Y_shared: Optional[torch.Tensor] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(_jsonable(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in keys})
        temporary = Path(handle.name)
    os.replace(temporary, path)


def expected_layout(
    canonical_size: int = CANONICAL_IMAGE_SIZE,
    tile_size: int = TILE_SIZE,
    stride: int = TILE_STRIDE,
) -> List[Tuple[int, int, int, int]]:
    """Return the strict row-major 7x7 4096/1024/512 layout."""
    if canonical_size != 4096 or tile_size != 1024 or stride != 512:
        raise ValueError("the oracle only accepts the official 4096/1024/512 layout")
    starts = list(range(0, canonical_size - tile_size + 1, stride))
    if len(starts) != 7 or starts[-1] != canonical_size - tile_size:
        raise ValueError("official layout does not land on the canonical edge")
    return [
        (x, y, x + tile_size, y + tile_size)
        for y in starts
        for x in starts
    ]


def phase_a_ids() -> set[int]:
    return set(PHASE_A_TILE_IDS)


def _parse_ids(text: Optional[str]) -> Optional[set[int]]:
    if text is None or not str(text).strip():
        return None
    return {int(part.strip()) for part in str(text).split(",") if part.strip()}


def _tile_dir(root: Path, tile_id: int) -> Path:
    candidates = [
        root / "tiles" / f"tile_{tile_id:02d}",
        root / "tiles" / f"tile_{tile_id}",
        root / f"tile_{tile_id:02d}",
        root / f"tile_{tile_id}",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def _candidate_files(root: Optional[Path], tile_id: int, names: Iterable[str]) -> List[Path]:
    if root is None or not root.exists():
        return []
    wanted = {name.lower() for name in names}
    tile_root = _tile_dir(root, tile_id)
    paths: List[Path] = []
    if tile_root.is_dir():
        for path in tile_root.rglob("*.pt"):
            if path.name.lower() in wanted:
                paths.append(path)
    # Also support a flat user-supplied directory with tile_XX files.
    for path in root.glob(f"tile_{tile_id:02d}_*.pt"):
        if path.name.lower() in wanted:
            paths.append(path)
    return sorted(set(paths))


def _tile_camera_box(path: Path) -> Optional[Tuple[int, int, int, int]]:
    """Read the candidate's exact canonical crop box when it is recorded."""
    current = path.parent
    for _ in range(5):
        camera_path = current / "tile_camera.json"
        payload = _layout_payload(camera_path)
        if payload is not None and isinstance(payload.get("box"), (list, tuple)):
            box = tuple(int(value) for value in payload["box"])
            if len(box) == 4:
                return box  # type: ignore[return-value]
        if current.parent == current:
            break
        current = current.parent
    return None


def _candidate_files_by_box(
    root: Optional[Path],
    expected_box: Sequence[int],
    names: Iterable[str],
) -> List[Path]:
    """Find endpoint files by their recorded crop, independent of old tile IDs.

    Several earlier experiments used the 4x4, stride-1024 tile numbering.  The
    canonical box is the stable identity across those experiments and the
    official 7x7, stride-512 layout.
    """
    if root is None or not root.exists():
        return []
    wanted = {name.lower() for name in names}
    target = tuple(int(value) for value in expected_box)
    paths: List[Path] = []
    for path in root.rglob("*.pt"):
        if path.name.lower() not in wanted:
            continue
        if _tile_camera_box(path) == target:
            paths.append(path)
    return sorted(set(paths))


def _pure_hr_route(path: Path) -> Optional[Any]:
    """Return route metadata proving that an endpoint is the pure-HR branch."""
    def find(value: Any) -> Optional[Any]:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_lower = str(key).lower().replace("-", "_")
                if key_lower in {"pure_hr", "purehr"}:
                    return child
                found = find(child)
                if found is not None:
                    return found
        elif isinstance(value, (list, tuple)):
            for child in value:
                found = find(child)
                if found is not None:
                    return found
        return None

    current = path.parent
    for _ in range(6):
        for metadata_name in ("provenance.json", "summary.json", "tile_preparation_summary.json"):
            payload = _layout_payload(current / metadata_name)
            if payload is not None:
                route = find(payload)
                if route is not None:
                    return route
        if current.parent == current:
            break
        current = current.parent
    return None


def _candidate_rejection(path: Path, *, kind: str) -> Optional[str]:
    lowered = "/".join(part.lower() for part in path.parts)
    for marker in _FORBIDDEN_MARKERS:
        # The reusable PureHR endpoint is materialized by a few scripts whose
        # output directory contains "perstep".  The endpoint filename and its
        # route metadata, rather than the parent experiment name, determine
        # whether it is the prohibited guided/per-step artifact.
        if (
            marker in {"perstep", "step_"}
            and kind == "final_endpoint"
            and path.name.lower() in {"pure_hr_endpoint.pt", "purehr_endpoint.pt", "final_purehr_endpoint.pt"}
        ):
            continue
        if marker in lowered:
            return f"path contains prohibited prior-artifact marker: {marker}"
    if kind == "final_endpoint" and path.name.lower() not in {
        "pure_hr_endpoint.pt",
        "purehr_endpoint.pt",
        "final_purehr_endpoint.pt",
    }:
        route = _pure_hr_route(path)
        if route is None:
            return "generic endpoint has no summary metadata identifying the PureHR route"
    route = _pure_hr_route(path)
    if route is not None:
        route_text = json.dumps(_jsonable(route), ensure_ascii=False).lower()
        for marker in ("guided", "mra", "projector", "range_null", "per-step", "per step"):
            if marker in route_text:
                return f"PureHR route metadata contains prohibited marker: {marker}"
    return None


def _candidate_provenance(path: Path) -> Dict[str, Any]:
    """Collect the nearest endpoint provenance without making it an input.

    The oracle never infers PureHR from a filename alone.  Materialized
    endpoints carry a small manifest, while older official PureHR caches carry
    the same route under the experiment summary.  Keeping this helper
    side-effect free also makes candidate rejection easy to test with tmp dirs.
    """
    current = path.parent
    for _ in range(6):
        for name in ("provenance.json", "summary.json"):
            payload = _layout_payload(current / name)
            if payload is not None:
                return dict(payload)
        if current.parent == current:
            break
        current = current.parent
    return {}


def _candidate_with_checks(
    *,
    tile_id: int,
    path: Path,
    kind: str,
    expected_box: Sequence[int],
    match_mode: str,
    source_root: Path,
) -> Candidate:
    rejection = _candidate_rejection(path, kind=kind)
    actual_box = _tile_camera_box(path)
    if rejection is None and actual_box is not None and actual_box != tuple(int(value) for value in expected_box):
        rejection = f"candidate tile_camera box {actual_box} != official box {tuple(int(value) for value in expected_box)}"
    return Candidate(
        tile_id=tile_id,
        path=path,
        kind=kind,
        rejection=rejection,
        match_mode=match_mode,
        source_root=source_root,
    )


def discover_candidates(
    tile_id: int,
    pure_field_dir: Optional[Path],
    pure_endpoint_dirs: Optional[Sequence[Path]],
) -> Tuple[List[Candidate], List[Candidate]]:
    direct: List[Candidate] = []
    endpoint: List[Candidate] = []
    expected_box = expected_layout()[tile_id]
    if pure_field_dir is not None:
        for path in _candidate_files(pure_field_dir, tile_id, _DIRECT_FIELD_NAMES):
            direct.append(
                _candidate_with_checks(
                    tile_id=tile_id,
                    path=path,
                    kind="direct_field",
                    expected_box=expected_box,
                    match_mode="tile_dir",
                    source_root=pure_field_dir,
                )
            )
    for root in pure_endpoint_dirs or ():
        for path in _candidate_files(root, tile_id, _ENDPOINT_NAMES):
            endpoint.append(
                _candidate_with_checks(
                    tile_id=tile_id,
                    path=path,
                    kind="final_endpoint",
                    expected_box=expected_box,
                    match_mode="tile_dir",
                    source_root=root,
                )
            )
        for path in _candidate_files_by_box(root, expected_box, _ENDPOINT_NAMES):
            if path not in {item.path for item in endpoint}:
                endpoint.append(
                    _candidate_with_checks(
                        tile_id=tile_id,
                        path=path,
                        kind="final_endpoint",
                        expected_box=expected_box,
                        match_mode="tile_camera_box",
                        source_root=root,
                    )
                )
    return direct, endpoint


def _layout_payload(path: Path) -> Optional[Mapping[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, Mapping) else None


def _validate_layout_metadata(root: Path, expected_boxes: Sequence[Sequence[int]]) -> Dict[str, Any]:
    paths = [root / "tile_layout.json", root / "summary.json"]
    found: List[Dict[str, Any]] = []
    errors: List[str] = []
    for path in paths:
        payload = _layout_payload(path)
        if payload is None:
            continue
        nested = payload.get("tile_layout") if isinstance(payload.get("tile_layout"), Mapping) else payload
        if not isinstance(nested, Mapping):
            continue
        record = {"path": str(path), "keys": sorted(str(k) for k in nested.keys())}
        found.append(record)
        for key, expected in (
            ("canonical_image_size", 4096),
            ("tile_size", 1024),
            ("stride", 512),
            ("tile_count", 49),
        ):
            if key in nested and int(nested[key]) != expected:
                errors.append(f"{path}: {key}={nested[key]!r}, expected {expected}")
        if "boxes" in nested and [list(box) for box in nested["boxes"]] != [list(box) for box in expected_boxes]:
            errors.append(f"{path}: tile boxes differ from the official 4096/1024/512 layout")
    if not found:
        errors.append(f"{root}: no tile_layout.json or summary.json with layout metadata")
    return {"root": str(root), "found": found, "errors": errors, "valid": not errors and bool(found)}


def _validate_endpoint_layout_metadata(root: Path, expected_boxes: Sequence[Sequence[int]]) -> Dict[str, Any]:
    """Validate endpoint provenance while allowing a partial official subset.

    Endpoint caches are often intentionally produced for only a few tiles, so
    ``tile_count`` need not be 49.  Every recorded box must nevertheless be
    one of the official 4096/1024/512 boxes.
    """
    expected = {tuple(int(value) for value in box) for box in expected_boxes}
    paths = [root / "tile_layout.json", root / "summary.json"]
    found: List[Dict[str, Any]] = []
    errors: List[str] = []
    for path in paths:
        payload = _layout_payload(path)
        if payload is None:
            continue
        nested = payload.get("tile_layout") if isinstance(payload.get("tile_layout"), Mapping) else payload
        if not isinstance(nested, Mapping):
            continue
        record = {"path": str(path), "keys": sorted(str(k) for k in nested.keys())}
        found.append(record)
        for key, expected_value in (
            ("canonical_image_size", 4096),
            ("tile_size", 1024),
            ("stride", 512),
        ):
            if key in nested and int(nested[key]) != expected_value:
                errors.append(f"{path}: {key}={nested[key]!r}, expected {expected_value}")
        if "boxes" in nested:
            try:
                boxes = [tuple(int(value) for value in box) for box in nested["boxes"]]
            except Exception:
                errors.append(f"{path}: invalid boxes metadata")
            else:
                invalid = [box for box in boxes if box not in expected]
                if invalid:
                    errors.append(f"{path}: boxes outside the official 4096/1024/512 layout: {invalid[:4]}")
                if len(set(boxes)) != len(boxes):
                    errors.append(f"{path}: duplicate tile boxes")
    if not found:
        return {
            "root": str(root),
            "found": [],
            "errors": [],
            "warnings": [f"{root}: no endpoint layout metadata; candidates will be checked by tile camera box"],
            "valid": None,
        }
    return {"root": str(root), "found": found, "errors": errors, "warnings": [], "valid": not errors}


def _required_static_paths(source_dir: Path, tile_id: int) -> Dict[str, Path]:
    tile = _tile_dir(source_dir, tile_id)
    return {
        "global_pbr_reference": tile / "global_pbr_reference.pt",
        "hidden_mask": tile / "hidden_mask.pt",
        "observed_mask": tile / "observed_mask.pt",
        "tile_camera": tile / "tile_camera.json",
    }


def _required_context_paths(context_dir: Path, tile_id: int) -> Dict[str, Path]:
    tile = _tile_dir(context_dir, tile_id)
    return {"fixed_shape_norm": tile / "fixed_shape_norm.pt", "tile_camera": tile / "tile_camera.json"}


def _operator_cache_paths(operator_cache_dir: Path, tile_id: int) -> Dict[str, Path]:
    tile = _tile_dir(operator_cache_dir, tile_id)
    return {
        "P_hidden": tile / "mra_P_hidden.npz",
        "support": tile / "mra_support.pt",
        "metadata": tile / "mra_operator.json",
    }


def _stable_operator_candidates(operator_cache_dir: Path, tile_id: int) -> List[Path]:
    name = f"tile_{tile_id:02d}_mra_P_hidden_float64.npz"
    if not operator_cache_dir.exists():
        return []
    return sorted(path for path in operator_cache_dir.rglob(name) if path.is_file())


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _csr_digest(operator: csr_matrix) -> Dict[str, Any]:
    operator = operator.tocsr()
    return {
        "shape": [int(v) for v in operator.shape],
        "nnz": int(operator.nnz),
        "data_digest": _array_digest(operator.data),
        "indices_digest": _array_digest(operator.indices),
        "indptr_digest": _array_digest(operator.indptr),
    }


def _load_operator_cache(
    *,
    operator_cache_dir: Path,
    tile_id: int,
    fine_coords: torch.Tensor,
    hidden_mask: torch.Tensor,
    rebuilt_coarse_coords: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[csr_matrix, Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load and validate the formal Sparse-MRA operator without re-partitioning it.

    ``mra_P_hidden.npz`` already contains ``P_full[hidden][:, pure_hidden]``.
    The returned matrix is therefore the production ``P_h``; no second column
    slicing is permitted here.
    """
    paths = _operator_cache_paths(operator_cache_dir, tile_id)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"tile {tile_id}: incomplete formal operator cache: {missing}")
    support = _load_torch(paths["support"])
    if not isinstance(support, Mapping):
        raise ValueError(f"tile {tile_id}: mra_support.pt is not a mapping")
    required = ("fine_coords", "coarse_coords", "hidden_rows", "pure_hidden_ids")
    if any(not isinstance(support.get(key), torch.Tensor) for key in required):
        raise ValueError(f"tile {tile_id}: mra_support.pt lacks {required}")
    cached_fine = support["fine_coords"].detach().cpu().to(torch.int32)
    cached_coarse = support["coarse_coords"].detach().cpu().to(torch.int32)
    cached_hidden_rows = support["hidden_rows"].detach().cpu().to(torch.int64).reshape(-1)
    cached_pure_ids = support["pure_hidden_ids"].detach().cpu().to(torch.int64).reshape(-1)
    rebuilt_fine = fine_coords.detach().cpu().to(torch.int32)
    rebuilt_coarse = rebuilt_coarse_coords.detach().cpu().to(torch.int32)
    if not torch.equal(rebuilt_fine, cached_fine):
        raise ValueError(f"tile {tile_id}: rebuilt C1024 coords differ from support cache (row order is not exact)")
    expected_hidden_rows = torch.where(hidden_mask.detach().cpu().bool())[0].to(torch.int64)
    if not torch.equal(expected_hidden_rows, cached_hidden_rows):
        raise ValueError(f"tile {tile_id}: hidden row ids differ from the formal support cache")
    if not torch.equal(rebuilt_coarse, cached_coarse):
        raise ValueError(f"tile {tile_id}: rebuilt C256 coords differ from the formal support cache")
    if cached_pure_ids.numel() and int(cached_pure_ids.min()) < 0:
        raise ValueError(f"tile {tile_id}: pure-hidden basis ids contain a negative id")

    source_operator = load_npz(paths["P_hidden"]).tocsr()
    if source_operator.shape != (int(cached_hidden_rows.numel()), int(cached_pure_ids.numel())):
        raise ValueError(
            f"tile {tile_id}: P_h shape {source_operator.shape} != cache partition "
            f"{cached_hidden_rows.numel(), cached_pure_ids.numel()}"
        )
    source_float64 = source_operator.astype(np.float64).tocsr()
    stable_candidates = _stable_operator_candidates(operator_cache_dir, tile_id)
    stable_check: Dict[str, Any] = {"present": False, "candidates": [str(p) for p in stable_candidates]}
    if stable_candidates:
        stable = load_npz(stable_candidates[0]).tocsr().astype(np.float64)
        if stable.shape != source_float64.shape:
            raise ValueError(f"tile {tile_id}: stable float64 P_h shape differs from formal cache")
        if not np.array_equal(stable.indices, source_float64.indices) or not np.array_equal(stable.indptr, source_float64.indptr):
            raise ValueError(f"tile {tile_id}: stable float64 P_h sparsity structure differs from formal cache")
        data_error = float(np.max(np.abs(stable.data - source_float64.data))) if stable.data.size else 0.0
        tolerance = float(getattr(args, "operator_data_tolerance", 1e-12))
        if data_error > tolerance:
            raise ValueError(f"tile {tile_id}: stable float64 P_h data error {data_error:.3e} > {tolerance:.3e}")
        stable_check = {
            "present": True,
            "path": str(stable_candidates[0]),
            "shape_exact": True,
            "indices_exact": True,
            "indptr_exact": True,
            "data_max_abs": data_error,
            "data_tolerance": tolerance,
            "source_digest": _csr_digest(source_float64),
            "stable_digest": _csr_digest(stable),
        }

    metadata = _layout_payload(paths["metadata"]) or {}
    row_nnz = np.diff(source_float64.indptr)
    hidden_coverage = torch.from_numpy((row_nnz > 0).copy())
    provenance = {
        "tile_id": int(tile_id),
        "cache_dir": str(operator_cache_dir.resolve()),
        "paths": {key: str(value.resolve()) for key, value in paths.items()},
        "support": {
            "fine_coords": [int(v) for v in cached_fine.shape],
            "coarse_coords": [int(v) for v in cached_coarse.shape],
            "hidden_rows": int(cached_hidden_rows.numel()),
            "pure_hidden_ids": int(cached_pure_ids.numel()),
            "fine_coords_exact": True,
            "hidden_rows_exact": True,
            "coarse_coords_exact": True,
        },
        "P_hidden": {
            **_csr_digest(source_float64),
            "source_dtype": str(source_operator.dtype),
            "loaded_dtype": str(source_float64.dtype),
            "row_nnz_min": int(row_nnz.min()) if row_nnz.size else 0,
            "row_nnz_max": int(row_nnz.max()) if row_nnz.size else 0,
            "uncovered_rows": int((row_nnz == 0).sum()),
            "coverage_ratio": float((row_nnz > 0).mean()) if row_nnz.size else 0.0,
        },
        "stable_float64_copy": stable_check,
        "formal_metadata": metadata,
        "column_partition_reused_verbatim": True,
        "repartitioned_in_oracle": False,
    }
    return source_float64, provenance, cached_fine, cached_coarse, cached_pure_ids


def _load_official_projector(operator: csr_matrix, info: Mapping[str, Any], args: argparse.Namespace) -> Any:
    """Instantiate the formal StableSparseMRAProjector used by production MRA."""
    from pixal3d_pbr_sparse_mra_projector_batch import StableSparseMRAProjector

    return StableSparseMRAProjector(
        operator,
        dict(info),
        atol=float(args.lsmr_atol),
        btol=float(args.lsmr_btol),
        maxiter=int(args.lsmr_maxiter),
        conlim=float(args.lsmr_conlim),
        channel_workers=int(getattr(args, "lsmr_channel_workers", 6)),
    )


def partition_hidden_operator(
    P_full: csr_matrix,
    observed_mask: torch.Tensor,
    *,
    hidden_rows: Optional[torch.Tensor] = None,
) -> Tuple[csr_matrix, np.ndarray]:
    """Reference-only partition helper used by regression tests.

    Production never calls this helper: it consumes the already-partitioned
    formal cache.  It makes the boundary rule explicit for synthetic tests.
    """
    observed = observed_mask.detach().cpu().bool().numpy()
    if hidden_rows is None:
        hidden = np.flatnonzero(~observed)
    else:
        hidden = hidden_rows.detach().cpu().to(torch.int64).numpy()
    touching_observed = np.zeros(P_full.shape[1], dtype=bool)
    if observed.any():
        observed_part = P_full[observed]
        if observed_part.nnz:
            touching_observed[np.unique(observed_part.indices)] = True
    active = np.asarray(P_full.getnnz(axis=0)).reshape(-1) > 0
    pure_ids = np.flatnonzero(active & ~touching_observed)
    return P_full[hidden][:, pure_ids].tocsr(), pure_ids.astype(np.int64, copy=False)


def _donor_query_field(donor: Any) -> torch.Tensor:
    """Single source of truth for donor consensus field selection."""
    return donor.C


def construct_oracles(
    H: torch.Tensor,
    G: torch.Tensor,
    C: torch.Tensor,
    C_shared: torch.Tensor,
    hidden: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct null/shared field oracles while preserving observed H exactly."""
    hidden = hidden.detach().cpu().bool().reshape(-1)
    observed = ~hidden
    C_private = C - C_shared
    Y_null = H.clone()
    Y_shared = H.clone()
    Y_null[hidden] = H[hidden] - C[hidden]
    Y_shared[hidden] = H[hidden] - C_private[hidden]
    if not torch.equal(Y_null[observed], H[observed]) or not torch.equal(Y_shared[observed], H[observed]):
        raise AssertionError("oracle construction changed observed PureHR rows")
    return Y_null, Y_shared, C_private


def _validate_operator_cache_presence(operator_cache_dir: Optional[Path], tile_ids: set[int]) -> Dict[str, Any]:
    if operator_cache_dir is None:
        return {"enabled": False, "missing": [], "complete": None}
    missing: Dict[str, List[int]] = {key: [] for key in ("P_hidden", "support", "metadata")}
    for tile_id in sorted(tile_ids):
        for key, path in _operator_cache_paths(operator_cache_dir, tile_id).items():
            if not path.is_file():
                missing[key].append(int(tile_id))
    missing_nonempty = {key: ids for key, ids in missing.items() if ids}
    return {
        "enabled": True,
        "root": str(operator_cache_dir.resolve()),
        "missing": missing_nonempty,
        "complete": not missing_nonempty,
    }


def preflight(
    *,
    source_dir: Path,
    context_dir: Path,
    pure_field_dir: Optional[Path],
    pure_endpoint_dirs: Optional[Sequence[Path]],
    allow_box_reuse: bool,
    tile_ids: set[int],
    output_dir: Path,
    operator_cache_dir: Optional[Path] = None,
    phase: str = "phase-a",
) -> Dict[str, Any]:
    boxes = expected_layout()
    layout = {
        "format": FORMAT,
        "canonical_image_size": CANONICAL_IMAGE_SIZE,
        "tile_size": TILE_SIZE,
        "stride": TILE_STRIDE,
        "tile_count": len(boxes),
        "boxes": [list(box) for box in boxes],
        "phase_a_tile_ids": sorted(PHASE_A_TILE_IDS),
        "formal_valid_tile_ids": sorted(FORMAL_VALID_TILE_IDS),
    }
    source_layout = _validate_layout_metadata(source_dir, boxes)
    context_layout = _validate_layout_metadata(context_dir, boxes)
    endpoint_layouts = [
        _validate_endpoint_layout_metadata(root, boxes)
        for root in (pure_endpoint_dirs or ())
    ]
    operator_presence = _validate_operator_cache_presence(operator_cache_dir, tile_ids)
    static_missing: Dict[str, List[int]] = {key: [] for key in ("global_pbr_reference", "hidden_mask", "observed_mask", "tile_camera")}
    context_missing: Dict[str, List[int]] = {key: [] for key in ("fixed_shape_norm", "tile_camera")}
    candidate_records: Dict[str, Any] = {}
    rejected: List[Dict[str, Any]] = []
    selected_static: set[int] = set()
    selected_context: set[int] = set()
    direct_ids: set[int] = set()
    endpoint_ids: set[int] = set()

    for tile_id in sorted(tile_ids):
        static = _required_static_paths(source_dir, tile_id)
        context = _required_context_paths(context_dir, tile_id)
        for key, path in static.items():
            if not path.is_file():
                static_missing[key].append(tile_id)
        for key, path in context.items():
            if not path.is_file():
                context_missing[key].append(tile_id)
        if all(path.is_file() for path in static.values()):
            selected_static.add(tile_id)
        if all(path.is_file() for path in context.values()):
            selected_context.add(tile_id)
        direct, endpoint = discover_candidates(tile_id, pure_field_dir, pure_endpoint_dirs)
        accepted_direct = [item for item in direct if item.rejection is None]
        accepted_endpoint = [item for item in endpoint if item.rejection is None]
        for item in direct + endpoint:
            if item.rejection is not None:
                rejected.append({"tile_id": tile_id, "kind": item.kind, "path": str(item.path), "reason": item.rejection})
        if accepted_direct:
            direct_ids.add(tile_id)
        if accepted_endpoint:
            endpoint_ids.add(tile_id)
        candidate_records[str(tile_id)] = {
            "direct": [_jsonable(item.__dict__) for item in direct],
            "endpoint": [_jsonable(item.__dict__) for item in endpoint],
        }

    final_ids = direct_ids | endpoint_ids
    missing_final = sorted(tile_ids - final_ids)
    errors = list(source_layout["errors"]) + list(context_layout["errors"])
    warnings: List[str] = []
    for endpoint_layout in endpoint_layouts:
        layout_errors = list(endpoint_layout.get("errors", []))
        if layout_errors and not allow_box_reuse:
            errors.extend(
                f"endpoint source rejected without --allow-box-reuse: {error}"
                for error in layout_errors
            )
        else:
            warnings.extend(layout_errors)
        warnings.extend(endpoint_layout.get("warnings", []))
    if not source_dir.is_dir():
        errors.append(f"source directory does not exist: {source_dir}")
    if not context_dir.is_dir():
        errors.append(f"context directory does not exist: {context_dir}")
    if operator_presence.get("complete") is False:
        errors.append(f"formal operator cache is incomplete: {operator_presence['missing']}")
    errors.extend(
        f"missing static {key}: {ids}"
        for key, ids in static_missing.items()
        if ids
    )
    errors.extend(
        f"missing fixed context {key}: {ids}"
        for key, ids in context_missing.items()
        if ids
    )
    if missing_final:
        errors.append(
            "missing final PureHR field/endpoint for tiles "
            f"{missing_final}; existing range-null, MRA, step, and projection files are not valid H inputs"
        )
    reference_preparation: Dict[str, Any] = {}
    if phase == "full":
        # Full is intentionally tied to the recorded preparation manifests;
        # a hard-coded "all except 6" list is not enough to establish that
        # the actual reference ensemble contains Tile35.
        for label, root in (("source", source_dir), ("context", context_dir)):
            summary_path = root / "tile_preparation_summary.json"
            payload = _layout_payload(summary_path)
            prepared = set(int(v) for v in (payload or {}).get("prepared_tile_ids", []))
            skipped = set(int(v) for v in (payload or {}).get("skipped_tile_ids", []))
            missing = sorted(set(FORMAL_VALID_TILE_IDS) - prepared)
            reference_preparation[label] = {
                "path": str(summary_path),
                "exists": summary_path.is_file(),
                "prepared_tile_ids": sorted(prepared),
                "skipped_tile_ids": sorted(skipped),
                "missing_formal_tile_ids": missing,
                "exact_formal_set": prepared == set(FORMAL_VALID_TILE_IDS),
            }
            if missing or prepared != set(FORMAL_VALID_TILE_IDS):
                errors.append(
                    f"{label} tile_preparation_summary.json does not contain the exact formal 48-tile set; "
                    f"missing={missing}, skipped={sorted(skipped)}"
                )
    result = {
        "format": FORMAT,
        "status": "ready" if not errors else "blocked_or_invalid",
        "source_dir": str(source_dir.resolve()),
        "context_dir": str(context_dir.resolve()),
        "operator_cache_dir": str(operator_cache_dir.resolve()) if operator_cache_dir else None,
        "phase": str(phase),
        "pure_field_dir": str(pure_field_dir.resolve()) if pure_field_dir else None,
        "pure_endpoint_dirs": [str(root.resolve()) for root in pure_endpoint_dirs or ()],
        "allow_box_reuse": bool(allow_box_reuse),
        "requested_tile_ids": sorted(tile_ids),
        "static_complete_tile_ids": sorted(selected_static),
        "context_complete_tile_ids": sorted(selected_context),
        "final_field_or_endpoint_tile_ids": sorted(final_ids),
        "missing_final_tile_ids": missing_final,
        "rejected_candidates": rejected,
        "candidates": candidate_records,
        "layout": layout,
        "source_layout": source_layout,
        "context_layout": context_layout,
        "endpoint_layouts": endpoint_layouts,
        "operator_cache": operator_presence,
        "reference_preparation": reference_preparation,
        "missing_static": static_missing,
        "missing_context": context_missing,
        "warnings": warnings,
        "errors": errors,
        "strict_route": {
            "flow_sampler_called": False,
            "shape_encoder_called": False,
            "pbr_encoder_called": False,
            "endpoint_decode_allowed_once": True,
            "normal_equation_used": False,
            "pbr_clamp_used": False,
            "visibility_or_geometry_changed": False,
            "weights_added": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "tile_layout.json", layout)
    _atomic_json(output_dir / "preflight.json", result)
    return result


def _linear_keys(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    xyz = coords.to(torch.int64)
    return (xyz[:, 0] * int(resolution) + xyz[:, 1]) * int(resolution) + xyz[:, 2]


def build_prolongation(
    coarse_coords: torch.Tensor,
    fine_points: torch.Tensor,
    *,
    coarse_resolution: int = COARSE_RESOLUTION,
) -> Tuple[csr_matrix, Dict[str, Any]]:
    """Build the sparse-support-renormalized trilinear P operator.

    The formula matches ``MeshWithVoxel.query_attrs`` at cell-center query
    points.  It is deliberately explicit so the operator can be saved and
    audited; no learned or Gaussian weights are introduced.
    """
    coarse_coords = coarse_coords.detach().cpu().to(torch.int64)
    fine_points = fine_points.detach().cpu().to(torch.float32)
    if coarse_coords.ndim != 2 or coarse_coords.shape[1] != 3:
        raise ValueError(f"coarse_coords must be [N,3], got {tuple(coarse_coords.shape)}")
    if fine_points.ndim != 2 or fine_points.shape[1] != 3:
        raise ValueError(f"fine_points must be [N,3], got {tuple(fine_points.shape)}")
    fine_count = int(fine_points.shape[0])
    coarse_count = int(coarse_coords.shape[0])
    if coarse_count == 0:
        return csr_matrix((fine_count, 0), dtype=np.float64), {
            "fine_rows": fine_count,
            "coarse_columns": 0,
            "nnz": 0,
            "uncovered_rows": fine_count,
            "coverage_ratio": 0.0,
            "dtype": "float64",
            "support_rule": "sparse trilinear with valid-neighbor renormalization",
        }
    keys = _linear_keys(coarse_coords, coarse_resolution)
    order = torch.argsort(keys, stable=True)
    sorted_keys = keys.index_select(0, order)
    if sorted_keys.numel() > 1 and bool((sorted_keys[1:] == sorted_keys[:-1]).any().item()):
        raise ValueError("duplicate coarse support coordinates")
    grid = (fine_points + 0.5) * float(coarse_resolution)
    base = torch.floor(grid - 0.5).to(torch.int64)
    frac = grid - (base.to(torch.float32) + 0.5)
    row_parts: List[torch.Tensor] = []
    col_parts: List[torch.Tensor] = []
    weight_parts: List[torch.Tensor] = []
    row_sum = torch.zeros(fine_count, dtype=torch.float64)
    for bits in range(8):
        bit = torch.tensor([(bits >> 0) & 1, (bits >> 1) & 1, (bits >> 2) & 1], dtype=torch.int64)
        neighbour = base + bit
        weight = torch.where(bit.bool(), frac, 1.0 - frac).prod(dim=1).to(torch.float64)
        valid = ((neighbour >= 0) & (neighbour < int(coarse_resolution))).all(dim=1)
        neighbour_keys = _linear_keys(neighbour, coarse_resolution)
        positions = torch.searchsorted(sorted_keys, neighbour_keys)
        valid &= positions < sorted_keys.numel()
        safe = positions.clamp_max(max(0, int(sorted_keys.numel()) - 1))
        if sorted_keys.numel():
            valid &= sorted_keys.index_select(0, safe) == neighbour_keys
        rows = torch.where(valid)[0]
        if rows.numel():
            cols = order.index_select(0, safe.index_select(0, rows))
            values = weight.index_select(0, rows)
            row_parts.append(rows)
            col_parts.append(cols)
            weight_parts.append(values)
            row_sum.index_add_(0, rows, values)
    if not row_parts:
        matrix = csr_matrix((fine_count, coarse_count), dtype=np.float64)
    else:
        rows = torch.cat(row_parts)
        cols = torch.cat(col_parts)
        values = torch.cat(weight_parts) / row_sum.index_select(0, rows).clamp_min(1e-15)
        matrix = csr_matrix(
            (values.numpy(), (rows.numpy(), cols.numpy())),
            shape=(fine_count, coarse_count),
            dtype=np.float64,
        )
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
    row_nnz = np.diff(matrix.indptr)
    info = {
        "fine_rows": fine_count,
        "coarse_columns": coarse_count,
        "nnz": int(matrix.nnz),
        "uncovered_rows": int((row_nnz == 0).sum()),
        "coverage_ratio": float((row_nnz > 0).mean()) if row_nnz.size else 0.0,
        "row_nnz": {
            "mean": float(row_nnz.mean()) if row_nnz.size else 0.0,
            "min": int(row_nnz.min()) if row_nnz.size else 0,
            "max": int(row_nnz.max()) if row_nnz.size else 0,
        },
        "dtype": "float64",
        "support_rule": "sparse trilinear with valid-neighbor renormalization",
    }
    return matrix, info


def apply_operator(operator: csr_matrix, value: torch.Tensor) -> torch.Tensor:
    array = value.detach().cpu().to(torch.float64).numpy()
    return torch.from_numpy(np.asarray(operator.dot(array), dtype=np.float64))


def solve_direct_lsmr(
    operator: csr_matrix,
    field: torch.Tensor,
    *,
    label: str = "field",
    atol: float = 1e-8,
    btol: float = 1e-8,
    maxiter: int = 500,
    conlim: float = 1e12,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Solve A x ~= field with direct float64 LSMR, never A.T@A."""
    if operator.dtype != np.float64:
        operator = operator.astype(np.float64)
    operator = operator.tocsr()
    values = field.detach().cpu().to(torch.float64).numpy()
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != operator.shape[0]:
        raise ValueError(f"{label}: field shape {values.shape} does not match operator rows {operator.shape[0]}")
    active = np.asarray(operator.getnnz(axis=0)).reshape(-1) > 0
    active_ids = np.flatnonzero(active)
    reduced = operator[:, active_ids]
    solution = np.zeros((operator.shape[1], values.shape[1]), dtype=np.float64)
    channel_info: List[Dict[str, Any]] = []
    for channel in range(values.shape[1]):
        started = time.perf_counter()
        result = lsmr(
            reduced,
            values[:, channel],
            damp=0.0,
            atol=float(atol),
            btol=float(btol),
            conlim=float(conlim),
            maxiter=int(maxiter),
        )
        x, istop, iterations, normr, normar, norm_a, cond_a, normx = result
        residual = reduced.dot(x) - values[:, channel]
        rhs = values[:, channel]
        relative_residual = float(np.linalg.norm(residual) / (np.linalg.norm(rhs) + 1e-12))
        # D is intentionally the component outside the range of P, so its
        # value-space residual need not be small.  The least-squares
        # correctness condition is stationarity: P^T(Px-b) ~= 0.  Compute it
        # explicitly for the report and accept an iteration-limited solve only
        # when that direct transpose residual is sufficiently small.  This
        # remains a direct float64 LSMR route; no normal equation is formed.
        transpose_residual = reduced.T.dot(residual)
        transpose_rhs = reduced.T.dot(rhs)
        relative_transpose_residual = float(
            np.linalg.norm(transpose_residual) / (np.linalg.norm(transpose_rhs) + 1e-12)
        )
        scaled_transpose_residual = float(
            np.linalg.norm(transpose_residual)
            / (float(norm_a) * float(np.linalg.norm(residual)) + 1e-12)
        )
        transpose_tolerance = max(1e-4, 1_000.0 * float(atol))
        scaled_transpose_tolerance = max(1e-5, 1_000.0 * float(atol))
        converged = bool(
            int(istop) in (0, 1, 2)
            or relative_transpose_residual <= transpose_tolerance
            or scaled_transpose_residual <= scaled_transpose_tolerance
        )
        if not converged:
            raise RuntimeError(
                f"{label}: direct float64 LSMR failed on channel {channel}: "
                f"istop={istop}, relative_residual={relative_residual:.3e}, "
                f"relative_transpose_residual={relative_transpose_residual:.3e}"
            )
        solution[active_ids, channel] = x
        channel_info.append(
            {
                "channel": int(channel),
                "istop": int(istop),
                "iterations": int(iterations),
                "normr": float(normr),
                "normar": float(normar),
                "normA": float(norm_a),
                "condA": float(cond_a),
                "normx": float(normx),
                "relative_residual": relative_residual,
                "relative_transpose_residual": relative_transpose_residual,
                "scaled_transpose_residual": scaled_transpose_residual,
                "transpose_tolerance": transpose_tolerance,
                "scaled_transpose_tolerance": scaled_transpose_tolerance,
                "seconds": float(time.perf_counter() - started),
                "converged": converged,
            }
        )
    info = {
        "label": label,
        "solver": "scipy.sparse.linalg.lsmr",
        "dtype": "float64",
        "normal_equation_used": False,
        "operator_shape": [int(v) for v in operator.shape],
        "operator_nnz": int(operator.nnz),
        "active_columns": int(active_ids.size),
        "inactive_columns": int((~active).sum()),
        "uncovered_rows": int((np.diff(operator.indptr) == 0).sum()),
        "atol": float(atol),
        "btol": float(btol),
        "maxiter": int(maxiter),
        "channels": channel_info,
    }
    return torch.from_numpy(solution), info


def _field_energy(value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    data = value.detach().to(torch.float64)
    if mask is not None:
        data = data[mask]
    return float(data.square().sum().item())


def _field_range(value: torch.Tensor) -> Dict[str, Any]:
    data = value.detach().to(torch.float64)
    if data.numel() == 0:
        return {"min": [], "max": [], "out_of_01_ratio": []}
    out = ((data < 0.0) | (data > 1.0)).to(torch.float64).mean(dim=0)
    return {
        "min": data.amin(dim=0).cpu().tolist(),
        "max": data.amax(dim=0).cpu().tolist(),
        "out_of_01_ratio": out.cpu().tolist(),
        "finite": bool(torch.isfinite(data).all().item()),
        "clamp_applied": False,
    }


def _metric_summary(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p50": None, "p90": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _pairwise_metrics(
    candidates: Sequence[Tuple[int, torch.Tensor, torch.Tensor]],
    sample_limit: int,
) -> Dict[str, Any]:
    """Compare transported coarse proposals with the specified null rules.

    Zero-norm vectors have no defined cosine and are excluded rather than
    converted to a misleading zero.  Relative disagreement uses the mean of
    the two norms, exactly as required by the experiment definition.
    """
    ordered = sorted(candidates, key=lambda item: int(item[0]))
    cosine_values: Dict[str, List[float]] = {name: [] for name in GROUPS}
    disagreement_values: Dict[str, List[float]] = {name: [] for name in GROUPS}
    zero_norm_pairs: Dict[str, int] = {name: 0 for name in GROUPS}
    pair_valid_rows = 0
    pair_count = 0
    for left_index in range(len(ordered)):
        _, left, left_valid = ordered[left_index]
        for right_index in range(left_index + 1, len(ordered)):
            _, right, right_valid = ordered[right_index]
            valid = left_valid.bool() & right_valid.bool()
            indices = torch.where(valid)[0]
            if indices.numel() == 0:
                continue
            if indices.numel() > int(sample_limit):
                indices = indices[torch.linspace(0, indices.numel() - 1, int(sample_limit)).round().long()]
            a = left.index_select(0, indices).to(torch.float64)
            b = right.index_select(0, indices).to(torch.float64)
            pair_count += 1
            pair_valid_rows += int(indices.numel())
            for name, group in GROUPS.items():
                av = a[:, group]
                bv = b[:, group]
                norm_a = torch.linalg.vector_norm(av, dim=1)
                norm_b = torch.linalg.vector_norm(bv, dim=1)
                cosine_valid = (norm_a > 1e-12) & (norm_b > 1e-12)
                zero_norm_pairs[name] += int((~cosine_valid).sum().item())
                if bool(cosine_valid.any().item()):
                    cosine = (av[cosine_valid] * bv[cosine_valid]).sum(dim=1) / (
                        norm_a[cosine_valid] * norm_b[cosine_valid]
                    )
                    cosine_values[name].extend(cosine.cpu().tolist())
                relative = torch.linalg.vector_norm(av - bv, dim=1) / (
                    0.5 * (norm_a + norm_b) + 1e-12
                )
                disagreement_values[name].extend(relative.cpu().tolist())
    return {
        "pair_count": int(pair_count),
        "pair_valid_rows": int(pair_valid_rows),
        "sample_limit_per_pair": int(sample_limit),
        "zero_norm_pair_count": zero_norm_pairs,
        "valid_cosine_pair_count": {name: len(values) for name, values in cosine_values.items()},
        "cosine": {name: _metric_summary(values) for name, values in cosine_values.items()},
        "relative_disagreement": {name: _metric_summary(values) for name, values in disagreement_values.items()},
        "_cosine_values": cosine_values,
        "_disagreement_values": disagreement_values,
    }


def _donor_histogram(counts: torch.Tensor) -> Dict[str, int]:
    counts = counts.detach().cpu().to(torch.int64)
    return {
        "0/1": int((counts <= 1).sum().item()),
        "2": int((counts == 2).sum().item()),
        "3": int((counts == 3).sum().item()),
        "4": int((counts == 4).sum().item()),
        "5+": int((counts >= 5).sum().item()),
        "total": int(counts.numel()),
    }


def _save_field_artifacts(root: Path, record: TileField) -> None:
    tile = root / "fields" / f"tile_{record.tile_id:02d}"
    tile.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("G", record.G),
        ("H", record.H),
        ("Delta", record.Delta),
        ("coarse", record.coarse),
        ("C", record.C),
        ("D", record.D),
    ):
        _atomic_torch(tile / f"{name}.pt", value.detach().cpu())
    for name, value in (
        ("fine_coords", record.geometry.coords),
        ("fine_points", record.points),
        ("hidden_mask", record.hidden),
        ("observed_mask", record.observed),
        ("projector_coverage", record.projector_coverage),
    ):
        _atomic_torch(tile / f"{name}.pt", value.detach().cpu())
    if record.P_full is not None:
        save_npz(tile / "P_full.npz", record.P_full)
    if not (tile / "P_hidden.npz").is_file():
        save_npz(tile / "P_hidden.npz", record.P_hidden)
    # Keep the formal cache name in the field artifact directory so downstream
    # audits cannot mistake this for an independently rebuilt operator.
    if not (tile / "mra_P_hidden.npz").is_file():
        save_npz(tile / "mra_P_hidden.npz", record.P_hidden)
    _atomic_json(tile / "projector.json", record.projector_info)
    if record.raw_consensus is not None:
        _atomic_torch(tile / "raw_consensus.pt", record.raw_consensus.detach().cpu())
    if record.donor_count is not None:
        _atomic_torch(tile / "donor_count.pt", record.donor_count.detach().cpu())
    if record.C_shared is not None:
        _atomic_torch(tile / "C_shared.pt", record.C_shared.detach().cpu())
    if record.C_private is not None:
        _atomic_torch(tile / "C_private.pt", record.C_private.detach().cpu())
    if record.Y_null is not None:
        _atomic_torch(tile / "Y_null.pt", record.Y_null.detach().cpu())
    if record.Y_shared is not None:
        _atomic_torch(tile / "Y_shared.pt", record.Y_shared.detach().cpu())
    _atomic_json(tile / "diagnostics.json", record.diagnostics)


def _load_tensor_field(path: Path) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    payload = _load_torch(path)
    if isinstance(payload, torch.Tensor):
        return payload.to(torch.float32), None
    if not isinstance(payload, Mapping):
        raise ValueError(f"field payload is not a tensor/mapping: {path}")
    value: Optional[torch.Tensor] = None
    for key in ("H", "field", "pbr", "final_pbr", "tensor", "attrs", "features", "raw"):
        candidate = payload.get(key)
        if isinstance(candidate, torch.Tensor) and candidate.ndim == 2 and candidate.shape[1] == 6:
            value = candidate
            break
    if value is None:
        raise ValueError(f"{path}: no direct six-channel PBR field found")
    coords = payload.get("coords")
    if not isinstance(coords, torch.Tensor):
        coords = None
    return value.to(torch.float32), coords.to(torch.int32) if coords is not None else None


def _normalise_coords(coords: torch.Tensor) -> torch.Tensor:
    coords = coords.detach().cpu().to(torch.int32)
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"sparse coordinates must be [N,3] or [N,4], got {tuple(coords.shape)}")
    if coords.shape[1] == 4:
        if bool((coords[:, 0] != 0).any().item()):
            raise ValueError("batched sparse coordinates must have batch index zero")
        return coords[:, 1:]
    return coords


def _align_direct_field(path: Path, fine_coords: torch.Tensor) -> torch.Tensor:
    values, coords = _load_tensor_field(path)
    if values.shape != (fine_coords.shape[0], PBR_CHANNELS):
        raise ValueError(f"{path}: PBR field shape {tuple(values.shape)} != fine support {(fine_coords.shape[0], PBR_CHANNELS)}")
    if coords is not None and not torch.equal(_normalise_coords(coords), _normalise_coords(fine_coords)):
        raise ValueError(f"{path}: field coordinates do not exactly match the fixed C1024 support")
    if not torch.isfinite(values).all():
        raise ValueError(f"{path}: direct PBR field is non-finite")
    return values.contiguous()


def _sparse_from_payload(payload: Mapping[str, Any], feature_key: str, coord_key: str = "coords") -> Any:
    from pixal3d.modules.sparse import SparseTensor

    coords = payload.get(coord_key)
    features = payload.get(feature_key)
    if not isinstance(coords, torch.Tensor) or not isinstance(features, torch.Tensor):
        raise ValueError(f"endpoint payload lacks tensor {coord_key}/{feature_key}")
    return SparseTensor(features.to(torch.float32).contiguous(), coords.to(torch.int32).contiguous())


def _load_context_sparse(path: Path) -> Any:
    payload = _load_torch(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"context sparse payload is not a mapping: {path}")
    return _sparse_from_payload(payload, "features")


def _load_endpoint_field(
    *,
    candidate: Candidate,
    context_shape_path: Path,
    fine_points: torch.Tensor,
    pipeline_holder: Dict[str, Any],
    model_path: str,
    query_chunk_size: int,
    low_vram: bool,
) -> torch.Tensor:
    """Decode one explicitly supplied final endpoint exactly once."""
    if candidate.rejection is not None:
        raise ValueError(f"endpoint candidate rejected: {candidate.rejection}")
    payload = _load_torch(candidate.path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"final endpoint must be a mapping: {candidate.path}")
    if any(marker in str(payload.get("format", "")).lower() for marker in _FORBIDDEN_MARKERS):
        raise ValueError(f"endpoint format is a prohibited prior/per-step artifact: {candidate.path}")
    fixed_shape = _load_context_sparse(context_shape_path)
    if "shape_norm" in payload and "hr_tex_norm" in payload and "shape_coords" in payload:
        from pixal3d.modules.sparse import SparseTensor

        shape_norm = SparseTensor(payload["shape_norm"].to(torch.float32), payload["shape_coords"].to(torch.int32))
        tex_norm = SparseTensor(payload["hr_tex_norm"].to(torch.float32), payload["hr_tex_coords"].to(torch.int32))
    else:
        shape_norm = fixed_shape
        feature_key = "norm" if isinstance(payload.get("norm"), torch.Tensor) else "features"
        tex_norm = _sparse_from_payload(payload, feature_key)
    if not torch.equal(shape_norm.coords, tex_norm.coords):
        raise ValueError(f"{candidate.path}: shape and final texture endpoint supports differ")
    if not torch.equal(shape_norm.coords, fixed_shape.coords):
        raise ValueError(f"{candidate.path}: endpoint shape support differs from the official fixed-shape support")
    if not torch.equal(shape_norm.feats, fixed_shape.feats):
        raise ValueError(f"{candidate.path}: endpoint shape features differ from the official fixed-shape context")
    # The oracle is field-only: never let an endpoint payload replace the
    # fixed geometry-conditioned shape context.
    shape_norm = fixed_shape
    if pipeline_holder.get("pipeline") is None:
        from inference import init_pipeline

        pipeline_holder["pipeline"] = init_pipeline(model_path, device="cuda", low_vram=bool(low_vram))
    pipeline = pipeline_holder["pipeline"]
    import pixal3d_cross_tile_pbr_perstep as base

    shape_denorm = base._denormalize_slat(shape_norm, pipeline.shape_slat_normalization)
    # The endpoint is loaded from CPU only as a cache boundary.  Decode on
    # the requested CUDA device; this is materialization of H, not an extra
    # flow/encoder operation.
    shape_denorm = base._sparse_to_device(shape_denorm, torch.device("cuda"))
    tex_norm = base._sparse_to_device(tex_norm, torch.device("cuda"))
    _, field_value, _ = base._decode_endpoint(
        pipeline=pipeline,
        shape_denorm=shape_denorm,
        texture_norm=tex_norm,
        query_points=fine_points.to("cuda"),
        query_chunk_size=int(query_chunk_size),
        label=f"final PureHR endpoint {candidate.path}",
    )
    if field_value.shape[1] != PBR_CHANNELS or not torch.isfinite(field_value).all():
        raise ValueError(f"decoded final endpoint has invalid PBR field: {candidate.path}")
    return field_value.detach().cpu().to(torch.float32)


def _load_geometry_dependencies() -> Tuple[Any, Any]:
    import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
    import o_voxel

    return core, o_voxel


def _load_baseline(path: Path) -> Any:
    payload = _load_torch(path)
    mesh = payload.get("mesh", payload) if isinstance(payload, Mapping) else payload
    from pixal3d.representations import MeshWithVoxel

    if not isinstance(mesh, MeshWithVoxel):
        raise ValueError(f"baseline is not MeshWithVoxel: {path}")
    return mesh.to("cpu")


def _payload_tensor(path: Path) -> torch.Tensor:
    payload = _load_torch(path)
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, Mapping):
        for key in ("tensor", "features", "mask", "value"):
            if isinstance(payload.get(key), torch.Tensor):
                return payload[key]
    raise ValueError(f"no tensor in {path}")


def _make_transform(core: Any, source_dir: Path, tile_id: int, box: Sequence[int], global_camera: Mapping[str, Any]) -> Any:
    camera_path = _tile_dir(source_dir, tile_id) / "tile_camera.json"
    if camera_path.is_file():
        payload = json.loads(camera_path.read_text(encoding="utf-8"))
        transform = core.TileCameraTransform(**payload)
    else:
        transform = core._derive_tile_camera(tile_id=tile_id, box=box, global_camera=global_camera, extend_pixel=0)
    if tuple(int(v) for v in transform.box) != tuple(int(v) for v in box):
        raise ValueError(f"tile {tile_id}: camera box {transform.box} != official box {tuple(box)}")
    return transform


def prepare_tile_fields(
    *,
    args: argparse.Namespace,
    selected_ids: set[int],
    source_dir: Path,
    context_dir: Path,
    pure_field_dir: Optional[Path],
    pure_endpoint_dirs: Optional[Sequence[Path]],
    operator_cache_dir: Path,
    output_dir: Path,
) -> Tuple[Dict[int, TileField], Dict[str, Any]]:
    core, _ = _load_geometry_dependencies()
    global_camera = json.loads((source_dir / "global_camera.json").read_text(encoding="utf-8"))
    baseline = _load_baseline(source_dir / "global_baseline_mesh.pt")
    vertices = baseline.vertices.detach().cpu().to(torch.float32)
    faces = baseline.faces.detach().cpu().to(torch.int64)
    face_min, face_max, face_finite = core._project_face_bboxes(
        vertices,
        faces,
        mesh_scale=float(global_camera["mesh_scale"]),
        global_camera=global_camera,
        chunk_size=int(args.face_projection_chunk_size),
    )
    records: Dict[int, TileField] = {}
    pipeline_holder: Dict[str, Any] = {}
    boxes = expected_layout()
    projector_dir = output_dir / "operators"
    purehr_provenance: Dict[str, Any] = {}
    operator_provenance: Dict[str, Any] = {}
    support_alignment: Dict[str, Any] = {}
    for tile_id in sorted(selected_ids):
        box = boxes[tile_id]
        transform = _make_transform(core, source_dir, tile_id, box, global_camera)
        geometry = core._prepare_tile_geometry(
            global_vertices=vertices,
            global_faces=faces,
            global_face_min=face_min,
            global_face_max=face_max,
            global_face_finite=face_finite,
            global_camera=global_camera,
            transform=transform,
        )
        static = _required_static_paths(source_dir, tile_id)
        G = _payload_tensor(static["global_pbr_reference"]).to(torch.float32)
        hidden = _payload_tensor(static["hidden_mask"]).to(torch.bool).reshape(-1)
        observed = _payload_tensor(static["observed_mask"]).to(torch.bool).reshape(-1)
        count = int(geometry.coords.shape[0])
        if G.shape != (count, PBR_CHANNELS):
            raise ValueError(f"tile {tile_id}: G shape {tuple(G.shape)} != {(count, PBR_CHANNELS)}")
        if hidden.shape[0] != count or observed.shape[0] != count or bool((hidden & observed).any().item()) or bool((hidden | observed).logical_not().any().item()):
            raise ValueError(f"tile {tile_id}: hidden/observed masks are not an exact complement on C1024 support")
        fine_points = -0.5 + (geometry.coords.to(torch.float32) + 0.5) / float(FINE_RESOLUTION)
        direct, endpoints = discover_candidates(tile_id, pure_field_dir, pure_endpoint_dirs)
        direct = [item for item in direct if item.rejection is None]
        endpoints = [item for item in endpoints if item.rejection is None]
        if direct:
            h_candidate = direct[0]
            H = _align_direct_field(h_candidate.path, geometry.coords)
            h_source = str(h_candidate.path)
        elif endpoints:
            h_candidate = endpoints[0]
            H = _load_endpoint_field(
                candidate=h_candidate,
                context_shape_path=_required_context_paths(context_dir, tile_id)["fixed_shape_norm"],
                fine_points=fine_points,
                pipeline_holder=pipeline_holder,
                model_path=str(args.model_path),
                query_chunk_size=int(args.query_chunk_size),
                low_vram=bool(args.low_vram),
            )
            h_source = str(h_candidate.path)
        else:
            raise RuntimeError(f"tile {tile_id}: no valid final PureHR field/endpoint")
        if H.shape != G.shape or not torch.isfinite(H).all():
            raise ValueError(f"tile {tile_id}: H is invalid or does not match G")
        Delta = H - G
        # C256/P_h are loaded from the formal Sparse-MRA route.  The C256
        # voxelization below is an audit of cache provenance only; it is never
        # used to create a second projector or to slice P_h again.
        coarse_coords, _, _ = _voxelize_support(geometry.vertices, geometry.faces, COARSE_RESOLUTION)
        P_hidden, op_provenance, cached_fine, cached_coarse, pure_hidden_ids = _load_operator_cache(
            operator_cache_dir=operator_cache_dir,
            tile_id=tile_id,
            fine_coords=geometry.coords,
            hidden_mask=hidden,
            rebuilt_coarse_coords=coarse_coords,
            args=args,
        )
        projector = _load_official_projector(P_hidden, op_provenance, args)
        hidden_ids = torch.where(hidden)[0].to(torch.int64)
        delta_hidden = Delta.index_select(0, hidden_ids)
        coarse, solve_info = projector.solve(
            delta_hidden,
            label=f"A Delta tile {tile_id}",
            x0=None,
            atol=float(args.lsmr_atol),
            btol=float(args.lsmr_btol),
            maxiter=int(args.lsmr_maxiter),
        )
        C_hidden64 = projector.apply(coarse).to(torch.float64)
        C = torch.zeros_like(Delta)
        C[hidden_ids] = C_hidden64.to(torch.float32)
        D = Delta - C
        pt_d = apply_operator(P_hidden.T.tocsr(), D.index_select(0, hidden_ids).to(torch.float64))
        pt_delta = apply_operator(P_hidden.T.tocsr(), delta_hidden.to(torch.float64))
        l2_pt_d = float(torch.linalg.vector_norm(pt_d).item())
        l2_pt_delta = float(torch.linalg.vector_norm(pt_delta).item())
        stationarity = {
            "max_abs_Pt_D": float(pt_d.abs().max().item()) if pt_d.numel() else 0.0,
            "l2_Pt_D": l2_pt_d,
            "l2_Pt_Delta": l2_pt_delta,
            "relative_Pt_D": l2_pt_d / (l2_pt_delta + float(args.stationarity_epsilon)),
            "denominator": "l2(P_h^T Delta_h) + epsilon",
        }
        projector_info = {
            **op_provenance,
            "solve": solve_info,
            "stationarity": stationarity,
            "algebraic_hidden_reconstruction_max_abs": float(
                (C[hidden_ids].to(torch.float64) + D[hidden_ids].to(torch.float64) - delta_hidden.to(torch.float64)).abs().max().item()
            ) if hidden_ids.numel() else 0.0,
            "H_source": h_source,
            "H_source_match_mode": h_candidate.match_mode,
            "H_source_root": str(h_candidate.source_root) if h_candidate.source_root else None,
            "H_source_provenance": _candidate_provenance(h_candidate.path),
            "A_definition": "official StableSparseMRAProjector.solve on Delta[hidden_rows]",
            "P_definition": "formal mra_P_hidden.npz = P_full[hidden_rows][:, pure_hidden_ids]",
            "normal_equation_used": False,
        }
        projector_coverage = torch.zeros(count, dtype=torch.bool)
        projector_coverage[hidden_ids] = torch.from_numpy((np.diff(P_hidden.indptr) > 0).copy())
        support_alignment[str(tile_id)] = {
            "fine_coords_exact": True,
            "hidden_rows_exact": True,
            "coarse_coords_exact": True,
            "fine_coords_shape": [int(v) for v in cached_fine.shape],
            "coarse_coords_shape": [int(v) for v in cached_coarse.shape],
            "hidden_rows": int(hidden_ids.numel()),
            "pure_hidden_ids": int(pure_hidden_ids.numel()),
        }
        purehr_provenance[str(tile_id)] = {
            "path": h_source,
            "match_mode": h_candidate.match_mode,
            "source_root": str(h_candidate.source_root) if h_candidate.source_root else None,
            "provenance": _candidate_provenance(h_candidate.path),
        }
        operator_provenance[str(tile_id)] = op_provenance
        record = TileField(
            tile_id=tile_id,
            box=tuple(int(v) for v in box),
            transform=transform,
            geometry=geometry,
            points=fine_points,
            G=G,
            H=H,
            Delta=Delta,
            hidden=hidden,
            observed=observed,
            fine_coords=cached_fine,
            coarse_coords=cached_coarse,
            hidden_rows=hidden_ids,
            pure_hidden_ids=pure_hidden_ids,
            P_full=None,
            P_hidden=P_hidden,
            projector=projector,
            projector_coverage=projector_coverage,
            projector_info=projector_info,
            coarse=coarse.to(torch.float64),
            C=C,
            D=D,
        )
        records[tile_id] = record
        projector_dir.mkdir(parents=True, exist_ok=True)
        save_npz(projector_dir / f"tile_{tile_id:02d}_P_hidden.npz", P_hidden)
        _save_field_artifacts(output_dir, record)
        print(f"[field] tile={tile_id} rows={count:,} hidden={int(hidden.sum()):,} H={h_source}")
    _atomic_json(output_dir / "purehr_provenance.json", {"tiles": purehr_provenance, "donor_field": "C = P_h A_h (H-G)"})
    _atomic_json(output_dir / "operator_provenance.json", {"tiles": operator_provenance})
    _atomic_json(output_dir / "support_alignment.json", {"tiles": support_alignment})
    return records, {
        "global_camera": global_camera,
        "baseline": baseline,
        "pipeline_loaded": pipeline_holder.get("pipeline") is not None,
        "purehr_provenance": purehr_provenance,
        "operator_provenance": operator_provenance,
        "support_alignment": support_alignment,
    }


def _prepared_field_paths(root: Path, tile_id: int) -> Dict[str, Path]:
    tile = root / "fields" / f"tile_{tile_id:02d}"
    return {
        name: tile / filename
        for name, filename in {
            "G": "G.pt",
            "H": "H.pt",
            "Delta": "Delta.pt",
            "coarse": "coarse.pt",
            "C": "C.pt",
            "D": "D.pt",
            "fine_coords": "fine_coords.pt",
            "fine_points": "fine_points.pt",
            "hidden": "hidden_mask.pt",
            "observed": "observed_mask.pt",
            "projector_coverage": "projector_coverage.pt",
            "P_hidden": "P_hidden.npz",
            "formal_P_hidden": "mra_P_hidden.npz",
            "projector": "projector.json",
        }.items()
    }


def load_prepared_tile_fields(
    *,
    args: argparse.Namespace,
    selected_ids: set[int],
    source_dir: Path,
    context_dir: Path,
    operator_cache_dir: Path,
    output_dir: Path,
) -> Tuple[Dict[int, TileField], Dict[str, Any]]:
    """Reload completed field preparation without rerunning endpoint decode or LSMR.

    This is an explicit resume path for a consensus/render interruption.  It
    rebuilds geometry only to restore the query/render objects, then verifies
    every saved field and support tensor against the same formal cache used by
    the original preparation.
    """
    core, _ = _load_geometry_dependencies()
    global_camera = json.loads((source_dir / "global_camera.json").read_text(encoding="utf-8"))
    baseline = _load_baseline(source_dir / "global_baseline_mesh.pt")
    vertices = baseline.vertices.detach().cpu().to(torch.float32)
    faces = baseline.faces.detach().cpu().to(torch.int64)
    face_min, face_max, face_finite = core._project_face_bboxes(
        vertices,
        faces,
        mesh_scale=float(global_camera["mesh_scale"]),
        global_camera=global_camera,
        chunk_size=int(args.face_projection_chunk_size),
    )
    boxes = expected_layout()
    records: Dict[int, TileField] = {}
    purehr_provenance: Dict[str, Any] = {}
    operator_provenance: Dict[str, Any] = {}
    support_alignment: Dict[str, Any] = {}
    for tile_id in sorted(selected_ids):
        paths = _prepared_field_paths(output_dir, tile_id)
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"tile {tile_id}: cannot reuse incomplete prepared fields: {missing}")
        box = boxes[tile_id]
        transform = _make_transform(core, source_dir, tile_id, box, global_camera)
        geometry = core._prepare_tile_geometry(
            global_vertices=vertices,
            global_faces=faces,
            global_face_min=face_min,
            global_face_max=face_max,
            global_face_finite=face_finite,
            global_camera=global_camera,
            transform=transform,
        )
        static = _required_static_paths(source_dir, tile_id)
        G = _payload_tensor(paths["G"]).to(torch.float32)
        H = _payload_tensor(paths["H"]).to(torch.float32)
        Delta = _payload_tensor(paths["Delta"]).to(torch.float32)
        coarse = _payload_tensor(paths["coarse"]).to(torch.float64)
        C = _payload_tensor(paths["C"]).to(torch.float32)
        D = _payload_tensor(paths["D"]).to(torch.float32)
        fine_coords = _payload_tensor(paths["fine_coords"]).to(torch.int32)
        fine_points = _payload_tensor(paths["fine_points"]).to(torch.float32)
        hidden = _payload_tensor(paths["hidden"]).to(torch.bool).reshape(-1)
        observed = _payload_tensor(paths["observed"]).to(torch.bool).reshape(-1)
        projector_coverage = _payload_tensor(paths["projector_coverage"]).to(torch.bool).reshape(-1)
        count = int(geometry.coords.shape[0])
        expected_points = -0.5 + (geometry.coords.to(torch.float32) + 0.5) / float(FINE_RESOLUTION)
        for name, value in {
            "G": G,
            "H": H,
            "Delta": Delta,
            "C": C,
            "D": D,
        }.items():
            if value.shape != (count, PBR_CHANNELS) or not torch.isfinite(value).all():
                raise ValueError(f"tile {tile_id}: reused {name} has invalid shape/finite values")
        if coarse.ndim != 2 or coarse.shape[1] != PBR_CHANNELS or not torch.isfinite(coarse).all():
            raise ValueError(f"tile {tile_id}: reused coarse coefficients are invalid")
        if not torch.equal(fine_coords, geometry.coords.to(torch.int32)):
            raise ValueError(f"tile {tile_id}: reused fine_coords differ from rebuilt geometry")
        if not torch.equal(fine_points, expected_points):
            raise ValueError(f"tile {tile_id}: reused fine_points differ from rebuilt geometry")
        if hidden.shape[0] != count or observed.shape[0] != count or projector_coverage.shape[0] != count:
            raise ValueError(f"tile {tile_id}: reused masks/coverage do not match rebuilt geometry")
        if bool((hidden & observed).any().item()) or bool((hidden | observed).logical_not().any().item()):
            raise ValueError(f"tile {tile_id}: reused hidden/observed masks are not an exact complement")
        static_G = _payload_tensor(static["global_pbr_reference"]).to(torch.float32)
        if not torch.equal(G, static_G):
            raise ValueError(f"tile {tile_id}: reused G differs from the formal static reference")
        if not torch.equal(Delta, H - G) or not torch.equal(D, Delta - C):
            raise ValueError(f"tile {tile_id}: reused field algebra is not exact")

        coarse_coords, _, _ = _voxelize_support(geometry.vertices, geometry.faces, COARSE_RESOLUTION)
        P_hidden, op_provenance, cached_fine, cached_coarse, pure_hidden_ids = _load_operator_cache(
            operator_cache_dir=operator_cache_dir,
            tile_id=tile_id,
            fine_coords=geometry.coords,
            hidden_mask=hidden,
            rebuilt_coarse_coords=coarse_coords,
            args=args,
        )
        try:
            saved_operator = load_npz(paths["P_hidden"]).tocsr().astype(np.float64)
        except ValueError:
            # A prior interruption can leave the convenience alias half
            # written while the formal cache-named copy is complete.  Repair
            # only this narrow artifact atomically; both files must still
            # match the formal cache exactly.
            saved_operator = load_npz(paths["formal_P_hidden"]).tocsr().astype(np.float64)
            temporary = paths["P_hidden"].with_name(f".{paths['P_hidden'].name}.{time.time_ns()}.tmp.npz")
            save_npz(temporary, saved_operator)
            os.replace(temporary, paths["P_hidden"])
        if not (
            np.array_equal(saved_operator.indices, P_hidden.indices)
            and np.array_equal(saved_operator.indptr, P_hidden.indptr)
            and np.array_equal(saved_operator.data, P_hidden.data)
        ):
            raise ValueError(f"tile {tile_id}: saved prepared P_hidden differs from the formal cache")
        projector = _load_official_projector(P_hidden, op_provenance, args)
        saved_projector_info = json.loads(paths["projector"].read_text(encoding="utf-8"))
        projector_info = {**op_provenance, **saved_projector_info, "resume_reused": True}
        hidden_ids = torch.where(hidden)[0].to(torch.int64)
        expected_coverage = torch.zeros(count, dtype=torch.bool)
        expected_coverage[hidden_ids] = torch.from_numpy((np.diff(P_hidden.indptr) > 0).copy())
        if not torch.equal(projector_coverage, expected_coverage):
            raise ValueError(f"tile {tile_id}: reused projector coverage differs from P_hidden")

        h_source = Path(str(saved_projector_info.get("H_source", "")))
        direct, endpoints = discover_candidates(tile_id, None, [h_source.parent] if h_source else None)
        all_candidates = [item for item in direct + endpoints if item.rejection is None]
        h_candidate = next((item for item in all_candidates if item.path == h_source), None)
        if h_candidate is None:
            h_candidate = Candidate(
                tile_id=tile_id,
                path=h_source,
                kind="endpoint" if h_source.name.lower() in _ENDPOINT_NAMES else "field",
                match_mode=str(saved_projector_info.get("H_source_match_mode", "reused_artifact")),
                source_root=Path(str(saved_projector_info["H_source_root"])) if saved_projector_info.get("H_source_root") else None,
            )
        purehr_provenance[str(tile_id)] = {
            "path": str(h_source),
            "match_mode": h_candidate.match_mode,
            "source_root": str(h_candidate.source_root) if h_candidate.source_root else None,
            "provenance": saved_projector_info.get("H_source_provenance") or _candidate_provenance(h_source),
        }
        operator_provenance[str(tile_id)] = op_provenance
        support_alignment[str(tile_id)] = {
            "fine_coords_exact": True,
            "hidden_rows_exact": True,
            "coarse_coords_exact": True,
            "fine_coords_shape": [int(v) for v in cached_fine.shape],
            "coarse_coords_shape": [int(v) for v in cached_coarse.shape],
            "hidden_rows": int(hidden_ids.numel()),
            "pure_hidden_ids": int(pure_hidden_ids.numel()),
            "resume_reused": True,
        }
        records[tile_id] = TileField(
            tile_id=tile_id,
            box=tuple(int(v) for v in box),
            transform=transform,
            geometry=geometry,
            points=fine_points,
            G=G,
            H=H,
            Delta=Delta,
            hidden=hidden,
            observed=observed,
            fine_coords=cached_fine,
            coarse_coords=cached_coarse,
            hidden_rows=hidden_ids,
            pure_hidden_ids=pure_hidden_ids,
            P_full=None,
            P_hidden=P_hidden,
            projector=projector,
            projector_coverage=projector_coverage,
            projector_info=projector_info,
            coarse=coarse,
            C=C,
            D=D,
        )
        print(f"[field-reuse] tile={tile_id} rows={count:,}")
    _atomic_json(output_dir / "purehr_provenance.json", {"tiles": purehr_provenance, "donor_field": "C = P_h A_h (H-G)", "resume_reused": True})
    _atomic_json(output_dir / "operator_provenance.json", {"tiles": operator_provenance, "resume_reused": True})
    _atomic_json(output_dir / "support_alignment.json", {"tiles": support_alignment, "resume_reused": True})
    return records, {
        "global_camera": global_camera,
        "baseline": baseline,
        "pipeline_loaded": False,
        "purehr_provenance": purehr_provenance,
        "operator_provenance": operator_provenance,
        "support_alignment": support_alignment,
    }


def _voxelize_support(vertices: torch.Tensor, faces: torch.Tensor, resolution: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import o_voxel

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
    if coords.numel() == 0:
        raise RuntimeError(f"C{resolution} support is empty")
    return coords, dual_vertices_world.detach().cpu().to(torch.float32), intersected.detach().cpu()


def _make_query_mesh(core: Any, baseline: Any, geometry: Any, attrs: torch.Tensor) -> Any:
    return core._make_local_reference_mesh(geometry, attrs.to(torch.float32), baseline)


def _query_donor_coarse(
    *,
    core: Any,
    base: Any,
    target: TileField,
    donor: TileField,
    target_points: torch.Tensor,
    global_camera: Mapping[str, Any],
    baseline: Any,
    query_chunk_size: int,
    query_mesh: Optional[Any] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transport the donor's coarse field ``C``, never ``Delta``.

    The coverage mesh is an independent support indicator.  A zero-valued C
    proposal is therefore still valid whenever the donor's pure-hidden
    projector support covers the query point.
    """
    q_global, _ = core._local_q_to_global_q(
        target_points * (2.0 * float(target.transform.mesh_scale)),
        global_camera=global_camera,
        transform=target.transform,
    )
    donor_points, donor_uv = core._global_q_to_local_q(
        q_global,
        global_camera=global_camera,
        transform=donor.transform,
    )
    donor_points = donor_points / (2.0 * float(donor.transform.mesh_scale))
    inside = (
        torch.isfinite(donor_points).all(dim=1)
        & torch.isfinite(donor_uv).all(dim=1)
        & (donor_uv[:, 0] >= 0.0)
        & (donor_uv[:, 0] < float(TILE_SIZE))
        & (donor_uv[:, 1] >= 0.0)
        & (donor_uv[:, 1] < float(TILE_SIZE))
        & (donor_points >= -0.5).all(dim=1)
        & (donor_points <= 0.5).all(dim=1)
    )
    # Query only geometrically valid rows.  Querying the complete target set
    # and masking afterward is mathematically equivalent but needlessly moves
    # out-of-tile points through the CUDA sparse sampler.
    query_rows = torch.where(inside)[0]
    coarse = torch.zeros((donor_points.shape[0], PBR_CHANNELS), dtype=torch.float32)
    coverage = torch.zeros((donor_points.shape[0],), dtype=torch.float32)
    if query_rows.numel():
        if query_mesh is None:
            query_attrs = torch.cat(
                [
                    _donor_query_field(donor).to(torch.float32),
                    donor.projector_coverage.to(torch.float32).reshape(-1, 1),
                ],
                dim=1,
            )
            query_mesh = _make_query_mesh(core, baseline, donor.geometry, query_attrs)
        queried = base._query_mesh_chunked(
            query_mesh,
            donor_points.index_select(0, query_rows),
            int(query_chunk_size),
        ).detach().cpu().to(torch.float32)
        if queried.ndim != 2 or queried.shape[1] != PBR_CHANNELS + 1:
            raise ValueError(f"donor query mesh must return [N,7], got {tuple(queried.shape)}")
        coarse[query_rows] = queried[:, :PBR_CHANNELS]
        coverage[query_rows] = queried[:, PBR_CHANNELS]
    valid = inside & torch.isfinite(coarse).all(dim=1) & torch.isfinite(coverage) & (coverage > 1e-6)
    return coarse, valid, q_global.detach().cpu()


def _query_self_c(
    *,
    core: Any,
    base: Any,
    record: TileField,
    baseline: Any,
    global_camera: Mapping[str, Any],
    query_chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    values, valid, _ = _query_donor_coarse(
        core=core,
        base=base,
        target=record,
        donor=record,
        target_points=record.points,
        global_camera=global_camera,
        baseline=baseline,
        query_chunk_size=query_chunk_size,
    )
    return values, valid


def compute_l2_consensus(
    candidates: Sequence[Tuple[int, torch.Tensor, torch.Tensor]],
    *,
    min_donors: int = CONSENSUS_MIN_DONORS,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Return parameter-free float64 mean and donor count in canonical order."""
    ordered = sorted(candidates, key=lambda item: int(item[0]))
    if not ordered:
        return torch.empty((0, 0), dtype=torch.float64), torch.empty((0,), dtype=torch.int32), []
    ids = [int(item[0]) for item in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("consensus candidates contain duplicate donor ids")
    values = torch.stack([item[1].detach().cpu().to(torch.float64) for item in ordered], dim=1)
    valid = torch.stack([item[2].detach().cpu().bool() for item in ordered], dim=1)
    if values.ndim != 3 or valid.shape != values.shape[:2]:
        raise ValueError("consensus candidate shapes must be [rows,channels] and [rows]")
    counts = valid.to(torch.int32).sum(dim=1)
    raw = torch.zeros((values.shape[0], values.shape[2]), dtype=torch.float64)
    evidence = counts >= int(min_donors)
    if bool(evidence.any().item()):
        weighted = values * valid[:, :, None].to(torch.float64)
        raw[evidence] = weighted.sum(dim=1)[evidence] / counts[evidence, None].to(torch.float64)
    return raw, counts, ids


def _group_norms(value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    value = value.detach().to(torch.float64)
    if mask is not None:
        value = value[mask]
    return {name: float(torch.linalg.vector_norm(value[:, sl]).item()) if value.numel() else 0.0 for name, sl in GROUPS.items()}


def _ratio_by_group(numerator: torch.Tensor, denominator: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    n = _group_norms(numerator, mask)
    d = _group_norms(denominator, mask)
    return {name: n[name] / (d[name] + 1e-12) for name in GROUPS}


def _inner_cosine_by_group(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> Dict[str, Any]:
    left = left.detach().to(torch.float64)[mask]
    right = right.detach().to(torch.float64)[mask]
    result: Dict[str, Any] = {}
    for name, sl in GROUPS.items():
        a = left[:, sl]
        b = right[:, sl]
        inner = float((a * b).sum().item()) if a.numel() else 0.0
        na = float(torch.linalg.vector_norm(a).item()) if a.numel() else 0.0
        nb = float(torch.linalg.vector_norm(b).item()) if b.numel() else 0.0
        result[name] = {"inner_product": inner, "cosine": inner / (na * nb + 1e-12)}
    return result


def _domain_stats(value: torch.Tensor, masks: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    return {
        domain: {name: _field_range(value[mask][:, group]) for name, group in GROUPS.items()}
        for domain, mask in masks.items()
    }


def _distance_summary(value: torch.Tensor) -> Dict[str, Any]:
    value = value.detach().to(torch.float64).reshape(-1)
    if not value.numel():
        return {"count": 0, "mean": None, "median": None, "p10": None, "p50": None, "p90": None}
    return _metric_summary(value.cpu().tolist())


def _boundary_stats(record: TileField, donor_count_full: torch.Tensor, output_dir: Path) -> Dict[str, Any]:
    hidden_ids = record.hidden_rows
    hidden_coverage = record.projector_coverage[hidden_ids]
    uncovered = ~hidden_coverage
    observed_coords = record.fine_coords[record.observed].to(torch.float32)
    hidden_coords = record.fine_coords[hidden_ids].to(torch.float32)
    distances = torch.full((hidden_ids.numel(),), float("nan"), dtype=torch.float32)
    if observed_coords.numel() and hidden_coords.numel():
        tree = cKDTree(observed_coords.numpy())
        chunks: List[np.ndarray] = []
        for start in range(0, int(hidden_coords.shape[0]), 250_000):
            chunks.append(tree.query(hidden_coords[start : start + 250_000].numpy(), k=1)[0].astype(np.float32))
        distances = torch.from_numpy(np.concatenate(chunks))
    y_shared_delta = (record.Y_shared[hidden_ids] - record.H[hidden_ids]) if record.Y_shared is not None else torch.zeros_like(record.H[hidden_ids])
    y_null_delta = (record.Y_null[hidden_ids] - record.H[hidden_ids]) if record.Y_null is not None else torch.zeros_like(record.H[hidden_ids])
    shared_norm = torch.linalg.vector_norm(record.C_shared[hidden_ids].to(torch.float64), dim=1) if record.C_shared is not None else torch.zeros(hidden_ids.numel(), dtype=torch.float64)
    high_shared = shared_norm >= (shared_norm.median() if shared_norm.numel() else 0.0)
    stats = {
        "tile_id": int(record.tile_id),
        "row_count": int(hidden_ids.numel()),
        "covered_hidden_rows": int(hidden_coverage.sum().item()),
        "uncovered_hidden_rows": int(uncovered.sum().item()),
        "coverage_ratio": float(hidden_coverage.to(torch.float64).mean().item()) if hidden_coverage.numel() else 0.0,
        "distance_voxel_units": {
            "all_hidden": _distance_summary(distances),
            "covered": _distance_summary(distances[hidden_coverage]),
            "uncovered": _distance_summary(distances[uncovered]),
            "high_shared": _distance_summary(distances[high_shared]),
            "low_shared": _distance_summary(distances[~high_shared]),
        },
        "spatial_distribution": {
            "uncovered_min": hidden_coords[uncovered].amin(dim=0).tolist() if bool(uncovered.any().item()) else None,
            "uncovered_max": hidden_coords[uncovered].amax(dim=0).tolist() if bool(uncovered.any().item()) else None,
            "uncovered_sample_coords": hidden_coords[uncovered][: min(256, int(uncovered.sum().item()))].to(torch.int32).tolist(),
        },
        "uncovered": {
            "Y_shared_minus_H": _group_norms(y_shared_delta, uncovered),
            "Y_null_minus_H": _group_norms(y_null_delta, uncovered),
            "donor_count": _distance_summary(donor_count_full[hidden_ids][uncovered].to(torch.float32)),
        },
    }
    return stats


def _physical_domain_stats(record: TileField) -> Dict[str, Any]:
    masks = {"full": torch.ones(record.G.shape[0], dtype=torch.bool), "observed": record.observed, "hidden": record.hidden}
    fields = {"G": record.G, "H": record.H, "Y_null": record.Y_null, "Y_shared": record.Y_shared}
    return {name: _domain_stats(value, masks) for name, value in fields.items() if value is not None}


def build_consensus(
    *,
    records: Dict[int, TileField],
    selected_ids: set[int],
    global_camera: Mapping[str, Any],
    baseline: Any,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    core, _ = _load_geometry_dependencies()
    import pixal3d_cross_tile_pbr_perstep as base

    target_diagnostics: Dict[str, Any] = {}
    pairwise_pool: Dict[str, List[float]] = {name: [] for name in GROUPS}
    disagreement_pool: Dict[str, List[float]] = {name: [] for name in GROUPS}
    macro_cosine: Dict[str, List[float]] = {name: [] for name in GROUPS}
    macro_disagreement: Dict[str, List[float]] = {name: [] for name in GROUPS}
    aggregate_energy: Dict[str, Dict[str, float]] = {
        key: {name: 0.0 for name in GROUPS}
        for key in ("Delta", "C", "D", "C_shared", "C_private", "Y_null_minus_G", "Y_shared_minus_G")
    }
    boundary_records: Dict[str, Any] = {}
    leakage_records: Dict[str, Any] = {}
    donor_field = "C = P_h A_h (H-G)"
    # The donor field and its independent support indicator are queried in one
    # canonical mesh per donor.  Keep these meshes on CUDA across target
    # queries so the same donor is not repeatedly transferred CPU -> CUDA.
    donor_query_meshes: Dict[int, Any] = {}
    for donor_id in sorted(selected_ids):
        donor = records[donor_id]
        query_attrs = torch.cat(
            [
                _donor_query_field(donor).to(torch.float32),
                donor.projector_coverage.to(torch.float32).reshape(-1, 1),
            ],
            dim=1,
        )
        donor_query_meshes[donor_id] = _make_query_mesh(core, baseline, donor.geometry, query_attrs).to("cuda")
    for target_id in sorted(selected_ids):
        target = records[target_id]
        hidden_ids = target.hidden_rows
        target_points_hidden = target.points.index_select(0, hidden_ids)
        candidates: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
        for donor_id in sorted(selected_ids):
            donor_values, donor_valid, _ = _query_donor_coarse(
                core=core,
                base=base,
                target=target,
                donor=records[donor_id],
                target_points=target_points_hidden,
                global_camera=global_camera,
                baseline=baseline,
                query_chunk_size=int(args.query_chunk_size),
                query_mesh=donor_query_meshes[donor_id],
            )
            candidates.append((int(donor_id), donor_values, donor_valid))
        raw_hidden, donor_count_hidden, canonical_ids = compute_l2_consensus(
            candidates,
            min_donors=CONSENSUS_MIN_DONORS,
        )
        raw_full = torch.zeros_like(target.Delta, dtype=torch.float64)
        donor_count_full = torch.zeros(target.Delta.shape[0], dtype=donor_count_hidden.dtype)
        raw_full[hidden_ids] = raw_hidden
        donor_count_full[hidden_ids] = donor_count_hidden
        c_shared, consensus_solve = target.projector.solve(
            raw_hidden,
            label=f"A S tile {target_id}",
            x0=None,
            atol=float(args.lsmr_atol),
            btol=float(args.lsmr_btol),
            maxiter=int(args.lsmr_maxiter),
        )
        C_shared_hidden64 = target.projector.apply(c_shared).to(torch.float64)
        C_shared = torch.zeros_like(target.Delta)
        C_shared[hidden_ids] = C_shared_hidden64.to(torch.float32)
        # Observed rows begin as H and are never overwritten by G.
        Y_null, Y_shared, C_private = construct_oracles(
            target.H,
            target.G,
            target.C,
            C_shared,
            target.hidden,
        )
        # Equivalent construction is retained as an explicit algebra check.
        y_shared_explicit = target.G[hidden_ids] + C_shared[hidden_ids] + target.D[hidden_ids]
        y_null_explicit = target.G[hidden_ids] + target.D[hidden_ids]
        if not torch.equal(Y_null[target.observed], target.H[target.observed]):
            raise AssertionError(f"tile {target_id}: null oracle changed observed rows")
        if not torch.equal(Y_shared[target.observed], target.H[target.observed]):
            raise AssertionError(f"tile {target_id}: shared oracle changed observed rows")
        if not torch.equal(C_shared[target.observed], torch.zeros_like(C_shared[target.observed])):
            raise AssertionError(f"tile {target_id}: C_shared has observed support")

        target.raw_consensus = raw_full
        target.donor_count = donor_count_full
        target.C_shared = C_shared
        target.C_private = C_private
        target.Y_null = Y_null
        target.Y_shared = Y_shared
        _atomic_torch(output_dir / "fields" / f"tile_{target_id:02d}" / "coarse_coefficients.pt", c_shared.detach().cpu())

        pairwise_raw = _pairwise_metrics(candidates, int(args.diagnostic_sample_limit))
        pairwise = {key: value for key, value in pairwise_raw.items() if not str(key).startswith("_")}
        for name in GROUPS:
            pairwise_pool[name].extend(pairwise_raw["_cosine_values"][name])
            disagreement_pool[name].extend(pairwise_raw["_disagreement_values"][name])
            if pairwise_raw["cosine"][name]["mean"] is not None:
                macro_cosine[name].append(float(pairwise_raw["cosine"][name]["mean"]))
            if pairwise_raw["relative_disagreement"][name]["mean"] is not None:
                macro_disagreement[name].append(float(pairwise_raw["relative_disagreement"][name]["mean"]))

        self_candidate = next(item for item in candidates if item[0] == target_id)
        self_values, self_valid = self_candidate[1], self_candidate[2]
        covered_self = self_valid & target.projector_coverage[hidden_ids]
        self_error = self_values - target.C[hidden_ids]
        self_relative = float(
            torch.linalg.vector_norm(self_error[covered_self].to(torch.float64)).item()
            / (torch.linalg.vector_norm(target.C[hidden_ids][covered_self].to(torch.float64)).item() + 1e-12)
        ) if bool(covered_self.any().item()) else 0.0
        self_transport = {
            "donor_field": donor_field,
            "valid_rows": int(covered_self.sum().item()),
            "coords_support_valid": bool(covered_self.any().item()),
            "mean_abs": float(self_error[covered_self].abs().mean().item()) if bool(covered_self.any().item()) else 0.0,
            "max_abs": float(self_error[covered_self].abs().max().item()) if bool(covered_self.any().item()) else 0.0,
            "relative_l2": self_relative,
        }
        if self_relative > float(args.self_transport_tolerance):
            raise RuntimeError(
                f"STOP Phase-A: target=self coarse transport failed on tile {target_id}: "
                f"relative_l2={self_relative:.3e} > {float(args.self_transport_tolerance):.3e}"
            )

        pt_d = apply_operator(target.P_hidden.T.tocsr(), target.D[hidden_ids].to(torch.float64))
        pt_delta = apply_operator(target.P_hidden.T.tocsr(), target.Delta[hidden_ids].to(torch.float64))
        pt_shared_residual = apply_operator(
            target.P_hidden.T.tocsr(),
            raw_hidden - C_shared_hidden64,
        )
        pt_raw_consensus = apply_operator(target.P_hidden.T.tocsr(), raw_hidden)
        identity = {
            "I1_Delta_equals_H_minus_G_max_abs": float((target.Delta - (target.H - target.G)).abs().max().item()),
            "I2_C_plus_D_equals_Delta_max_abs": float((target.C + target.D - target.Delta).abs().max().item()),
            "I3_stationarity": target.projector_info["stationarity"],
            "I4_shared_stationarity_l2": float(torch.linalg.vector_norm(pt_shared_residual).item()),
            "I4_shared_stationarity_relative": float(
                torch.linalg.vector_norm(pt_shared_residual).item()
                / (torch.linalg.vector_norm(pt_raw_consensus).item() + float(args.stationarity_epsilon))
            ),
            "I5_Y_null_equals_H_minus_C_max_abs": float((Y_null[hidden_ids] - (target.H[hidden_ids] - target.C[hidden_ids])).abs().max().item()) if hidden_ids.numel() else 0.0,
            "I6_Y_shared_equals_H_minus_private_max_abs": float((Y_shared[hidden_ids] - (target.H[hidden_ids] - C_private[hidden_ids])).abs().max().item()) if hidden_ids.numel() else 0.0,
            "I7_observed_null_exact": bool(torch.equal(Y_null[target.observed], target.H[target.observed])),
            "I8_observed_shared_exact": bool(torch.equal(Y_shared[target.observed], target.H[target.observed])),
            "I9_observed_C_exact_zero": bool(torch.equal(target.C[target.observed], torch.zeros_like(target.C[target.observed]))),
            "observed_identity_exact": True,
            "observed_max_abs": 0.0,
            "explicit_Y_null_max_abs": float((Y_null[hidden_ids] - y_null_explicit).abs().max().item()) if hidden_ids.numel() else 0.0,
            "explicit_Y_shared_max_abs": float((Y_shared[hidden_ids] - y_shared_explicit).abs().max().item()) if hidden_ids.numel() else 0.0,
            "P_transpose_D_max_abs": float(pt_d.abs().max().item()) if pt_d.numel() else 0.0,
            "P_transpose_Delta_l2": float(torch.linalg.vector_norm(pt_delta).item()),
        }

        ratios = {
            "R_C": _ratio_by_group(target.C, target.Delta, target.hidden),
            "R_D": _ratio_by_group(target.D, target.Delta, target.hidden),
            "R_shared": _ratio_by_group(C_shared, target.C, target.hidden),
            "R_private": _ratio_by_group(C_private, target.C, target.hidden),
            "R_preserve_shared": _ratio_by_group(Y_shared - target.G, target.Delta, target.hidden),
            "R_preserve_null": _ratio_by_group(Y_null - target.G, target.Delta, target.hidden),
        }
        for key, value in (("Delta", target.Delta), ("C", target.C), ("D", target.D), ("C_shared", C_shared), ("C_private", C_private), ("Y_null_minus_G", Y_null - target.G), ("Y_shared_minus_G", Y_shared - target.G)):
            norms = _group_norms(value, target.hidden)
            for name in GROUPS:
                aggregate_energy[key][name] += norms[name] ** 2

        no_evidence = donor_count_hidden < CONSENSUS_MIN_DONORS
        evidence = ~no_evidence
        shared_hidden = C_shared[hidden_ids]
        leakage = {
            "donor_field": donor_field,
            "no_evidence_rows": int(no_evidence.sum().item()),
            "evidence_rows": int(evidence.sum().item()),
            "R_noEvidence": float(
                torch.linalg.vector_norm(shared_hidden[no_evidence].to(torch.float64)).item()
                / (torch.linalg.vector_norm(shared_hidden.to(torch.float64)).item() + 1e-12)
            ) if hidden_ids.numel() else 0.0,
            "shared_energy_on_evidence_rows": _group_norms(shared_hidden, evidence),
            "shared_energy_on_no_evidence_rows": _group_norms(shared_hidden, no_evidence),
        }
        leakage_records[str(target_id)] = leakage
        boundary = _boundary_stats(target, donor_count_full, output_dir)
        boundary_records[str(target_id)] = boundary
        target.diagnostics = {
            "tile_id": int(target_id),
            "donor_field": donor_field,
            "donor_ids_canonical": canonical_ids,
            "donor_count_histogram": _donor_histogram(donor_count_full),
            "pairwise": pairwise,
            "self_transport": self_transport,
            "identity": identity,
            "consensus_solver": consensus_solve,
            "ratios": ratios,
            "shared_private": _inner_cosine_by_group(C_shared, C_private, target.hidden),
            "shared_private_norms": {
                "shared_norm_ratio": _ratio_by_group(C_shared, target.C, target.hidden),
                "private_norm_ratio": _ratio_by_group(C_private, target.C, target.hidden),
                "shared_squared_norm_ratio": {name: ( _group_norms(C_shared, target.hidden)[name] ** 2 / (_group_norms(target.C, target.hidden)[name] ** 2 + 1e-12)) for name in GROUPS},
                "private_squared_norm_ratio": {name: ( _group_norms(C_private, target.hidden)[name] ** 2 / (_group_norms(target.C, target.hidden)[name] ** 2 + 1e-12)) for name in GROUPS},
            },
            "consensus_projection_leakage": leakage,
            "boundary_uncovered": boundary,
            "pbr_domain_stats": _physical_domain_stats(target),
            "range": {name: _field_range(value) for name, value in {"G": target.G, "H": target.H, "Delta": target.Delta, "C": target.C, "D": target.D, "S": raw_full, "C_shared": C_shared, "C_private": C_private, "Y_null": Y_null, "Y_shared": Y_shared}.items()},
            "donor_order_invariance": _order_invariance(candidates, donor_count_hidden),
        }
        target_diagnostics[str(target_id)] = target.diagnostics
        _save_field_artifacts(output_dir, target)
        print(
            f"[consensus] tile={target_id} donor>=2={int((donor_count_hidden >= CONSENSUS_MIN_DONORS).sum()):,}/"
            f"{donor_count_hidden.numel():,} self_rel={self_relative:.3e}"
        )

    pooled_pairwise = {
        "all_valid_pair_pooled": {
            "cosine": {name: _metric_summary(pairwise_pool[name]) for name in GROUPS},
            "relative_disagreement": {name: _metric_summary(disagreement_pool[name]) for name in GROUPS},
        },
        "per_tile_macro_mean": {
            "cosine": {name: _metric_summary(macro_cosine[name]) for name in GROUPS},
            "relative_disagreement": {name: _metric_summary(macro_disagreement[name]) for name in GROUPS},
        },
    }
    coarse_agreement = {
        "format": FORMAT,
        "donor_field": donor_field,
        "tiles": {key: value["pairwise"] for key, value in target_diagnostics.items()},
        "global": pooled_pairwise,
    }
    shared_private_stats = {
        "format": FORMAT,
        "donor_field": donor_field,
        "tiles": {key: {"ratios": value["ratios"], "shared_private": value["shared_private"], "norms": value["shared_private_norms"]} for key, value in target_diagnostics.items()},
        "aggregate_squared_energy": aggregate_energy,
    }
    pbr_domain_stats = {key: value["pbr_domain_stats"] for key, value in target_diagnostics.items()}
    result = {
        "format": FORMAT,
        "donor_field": donor_field,
        "tiles": target_diagnostics,
        "aggregate": {
            "energy": aggregate_energy,
            "R_C": {name: math.sqrt(aggregate_energy["C"][name]) / (math.sqrt(aggregate_energy["Delta"][name]) + 1e-12) for name in GROUPS},
            "R_D": {name: math.sqrt(aggregate_energy["D"][name]) / (math.sqrt(aggregate_energy["Delta"][name]) + 1e-12) for name in GROUPS},
            "R_shared": {name: math.sqrt(aggregate_energy["C_shared"][name]) / (math.sqrt(aggregate_energy["C"][name]) + 1e-12) for name in GROUPS},
            "R_private": {name: math.sqrt(aggregate_energy["C_private"][name]) / (math.sqrt(aggregate_energy["C"][name]) + 1e-12) for name in GROUPS},
            "R_preserve_shared": {name: math.sqrt(aggregate_energy["Y_shared_minus_G"][name]) / (math.sqrt(aggregate_energy["Delta"][name]) + 1e-12) for name in GROUPS},
            "R_preserve_null": {name: math.sqrt(aggregate_energy["Y_null_minus_G"][name]) / (math.sqrt(aggregate_energy["Delta"][name]) + 1e-12) for name in GROUPS},
        },
    }
    _atomic_json(output_dir / "coarse_agreement.json", coarse_agreement)
    _atomic_json(output_dir / "shared_private_stats.json", shared_private_stats)
    _atomic_json(output_dir / "pbr_domain_stats.json", pbr_domain_stats)
    _atomic_json(output_dir / "boundary_uncovered_stats.json", {"tiles": boundary_records})
    _atomic_json(output_dir / "consensus_projection_leakage.json", {"tiles": leakage_records})
    _atomic_json(output_dir / "consensus_diagnostics.json", result)
    _atomic_json(output_dir / "shared_private_energy.json", result["aggregate"])
    metric_rows: List[Dict[str, Any]] = []
    for tile_id in sorted(records):
        record = records[tile_id]
        for variant_name, value in {"Global": record.G, "PureHR": record.H, "Null-only": record.Y_null, "Shared-coarse": record.Y_shared}.items():
            if value is None:
                continue
            stats = _field_range(value)
            for group_name, group_slice in GROUPS.items():
                metric_rows.append({
                    "tile_id": int(tile_id), "variant": variant_name, "group": group_name,
                    "min": float(min(stats["min"][group_slice.start:group_slice.stop])),
                    "max": float(max(stats["max"][group_slice.start:group_slice.stop])),
                    "out_of_01_ratio": float(np.mean(stats["out_of_01_ratio"][group_slice.start:group_slice.stop])),
                    "hidden_rows": int(record.hidden.sum().item()),
                })
    _write_csv(output_dir / "metrics.csv", metric_rows)
    return result


def load_consensus_artifacts(records: Dict[int, TileField], output_dir: Path) -> Dict[str, Any]:
    """Restore a completed consensus stage after a visualization/render stop."""
    pbr_domain_stats: Dict[str, Any] = {}
    for tile_id, record in sorted(records.items()):
        tile = output_dir / "fields" / f"tile_{tile_id:02d}"
        required = {
            "raw_consensus": tile / "raw_consensus.pt",
            "donor_count": tile / "donor_count.pt",
            "C_shared": tile / "C_shared.pt",
            "C_private": tile / "C_private.pt",
            "Y_null": tile / "Y_null.pt",
            "Y_shared": tile / "Y_shared.pt",
            "diagnostics": tile / "diagnostics.json",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"tile {tile_id}: incomplete consensus artifacts: {missing}")
        record.raw_consensus = _payload_tensor(required["raw_consensus"]).to(torch.float64)
        record.donor_count = _payload_tensor(required["donor_count"]).to(torch.int64).reshape(-1)
        record.C_shared = _payload_tensor(required["C_shared"]).to(torch.float32)
        record.C_private = _payload_tensor(required["C_private"]).to(torch.float32)
        record.Y_null = _payload_tensor(required["Y_null"]).to(torch.float32)
        record.Y_shared = _payload_tensor(required["Y_shared"]).to(torch.float32)
        if record.raw_consensus.shape != (record.G.shape[0], PBR_CHANNELS):
            raise ValueError(f"tile {tile_id}: raw consensus shape is invalid")
        if record.donor_count.shape[0] != record.G.shape[0]:
            raise ValueError(f"tile {tile_id}: donor count shape is invalid")
        for name, value in {
            "C_shared": record.C_shared,
            "C_private": record.C_private,
            "Y_null": record.Y_null,
            "Y_shared": record.Y_shared,
        }.items():
            if value.shape != record.G.shape or not torch.isfinite(value).all():
                raise ValueError(f"tile {tile_id}: reused {name} is invalid")
        record.diagnostics = json.loads(required["diagnostics"].read_text(encoding="utf-8"))
        # Recompute this report from channel-sliced domains so a resumed run
        # also repairs reports produced by older builds that repeated the full
        # six-channel range under every PBR group.
        record.diagnostics["pbr_domain_stats"] = _physical_domain_stats(record)
        pbr_domain_stats[str(tile_id)] = record.diagnostics["pbr_domain_stats"]
        _atomic_json(required["diagnostics"], record.diagnostics)
    _atomic_json(output_dir / "pbr_domain_stats.json", pbr_domain_stats)
    summary_path = output_dir / "consensus_diagnostics.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    result["resume_reused"] = True
    return result


def _order_invariance(candidates: Sequence[Tuple[int, torch.Tensor, torch.Tensor]], donor_count: torch.Tensor) -> Dict[str, Any]:
    if not candidates:
        return {"max_abs": 0.0, "mean_abs": 0.0}
    ordered = sorted(candidates, key=lambda item: int(item[0]))
    forward = torch.zeros_like(ordered[0][1], dtype=torch.float64)
    reverse = torch.zeros_like(ordered[0][1], dtype=torch.float64)
    for _, value, valid in ordered:
        forward += value.to(torch.float64) * valid[:, None].to(torch.float64)
    for _, value, valid in reversed(ordered):
        reverse += value.to(torch.float64) * valid[:, None].to(torch.float64)
    diff = (forward - reverse).abs()
    valid = donor_count >= 2
    if not bool(valid.any().item()):
        return {"max_abs": 0.0, "mean_abs": 0.0, "valid_rows": 0}
    return {"max_abs": float(diff[valid].max().item()), "mean_abs": float(diff[valid].mean().item()), "valid_rows": int(valid.sum().item())}


def _visualize_phase_a(records: Dict[int, TileField], output_dir: Path, global_camera: Mapping[str, Any], sample_limit: int) -> None:
    """Write compact point-raster sheets for Tile26/27 when fields exist."""
    try:
        from PIL import Image, ImageDraw
        core, _ = _load_geometry_dependencies()
        import pixal3d_cross_tile_pbr_perstep as base
    except Exception as exc:
        _atomic_json(output_dir / "visualization_error.json", {"error": repr(exc)})
        return
    (output_dir / "visuals").mkdir(parents=True, exist_ok=True)
    variants = ("G", "H", "Delta", "C", "D", "S", "C_shared", "C_private", "Y_null", "Y_shared")
    for tile_id in sorted(PHASE_A_TILE_IDS & set(records)):
        record = records[tile_id]
        values_by_name = {
            "G": record.G,
            "H": record.H,
            "Delta": record.Delta,
            "C": record.C,
            "D": record.D,
            "S": record.raw_consensus if record.raw_consensus is not None else torch.zeros_like(record.G),
            "C_shared": record.C_shared if record.C_shared is not None else torch.zeros_like(record.G),
            "C_private": record.C_private if record.C_private is not None else torch.zeros_like(record.G),
            "Y_null": record.Y_null if record.Y_null is not None else torch.zeros_like(record.G),
            "Y_shared": record.Y_shared if record.Y_shared is not None else torch.zeros_like(record.G),
        }
        points = record.points
        if points.shape[0] > int(sample_limit):
            ids = torch.linspace(0, points.shape[0] - 1, int(sample_limit)).round().long()
            points = points.index_select(0, ids)
            values_by_name = {key: value.index_select(0, ids) for key, value in values_by_name.items()}
        _, uv_full, _ = base._local_to_global(points, transform=record.transform, global_camera=global_camera)
        uv = (uv_full / float(CANONICAL_IMAGE_SIZE) * 384.0).round().long().clamp(0, 383)
        for group_name, channel_slice in GROUPS.items():
            panel = Image.new("RGB", (384 * len(variants), 430), (20, 20, 20))
            draw = ImageDraw.Draw(panel)
            for index, variant in enumerate(variants):
                image = torch.zeros((384, 384, 3), dtype=torch.float32)
                values = values_by_name[variant][:, channel_slice].to(torch.float32)
                if values.shape[1] == 1:
                    values = values.repeat(1, 3)
                elif values.shape[1] > 3:
                    values = values[:, :3]
                finite = torch.isfinite(values).all(dim=1)
                image[uv[finite, 1], uv[finite, 0]] = values[finite]
                # Display normalization only.  The saved/oracle fields are
                # never clamped to the physical [0,1] range.
                image = (image - image.amin()) / (image.amax() - image.amin()).clamp_min(1e-6)
                rgb = (image.mul(255.0).clamp(0, 255).to(torch.uint8).numpy())
                tile_image = Image.fromarray(rgb, mode="RGB")
                panel.paste(tile_image, (384 * index, 0))
                draw.text((384 * index + 5, 390), variant, fill=(255, 255, 255))
            panel.save(output_dir / "visuals" / f"tile_{tile_id:02d}_{group_name}.png")
        # This is a depth debug split only.  It is deliberately not called
        # front/back; true front/back images come from the renderer camera.
        depth = points[:, 2]
        for label, mask in (("near_depth_debug", depth <= depth.median()), ("far_depth_debug", depth > depth.median())):
            image = Image.new("RGB", (384, 384), (20, 20, 20))
            front_values = values_by_name["Y_shared"][:, :3]
            valid = mask & torch.isfinite(front_values).all(dim=1)
            pixels = torch.zeros((384, 384, 3), dtype=torch.float32)
            pixels[uv[valid, 1], uv[valid, 0]] = front_values[valid]
            pixels = (pixels - pixels.amin()) / (pixels.amax() - pixels.amin()).clamp_min(1e-6)
            image = Image.fromarray((pixels.mul(255).clamp(0, 255).to(torch.uint8).numpy()), mode="RGB")
            image.save(output_dir / "visuals" / f"tile_{tile_id:02d}_Y_shared_{label}.png")


def _render_variants(
    *,
    records: Dict[int, TileField],
    selected_ids: set[int],
    baseline: Any,
    global_camera: Mapping[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Stitch the four field variants with the repository's fixed stitcher.

    Every variant uses the same local geometry, tile ownership, weld
    tolerance, camera, and (when requested) environment map.  This function
    never changes faces or geometry based on the PBR variant.
    """
    core, _ = _load_geometry_dependencies()
    variants = {
        "Global": lambda record: record.G,
        "PureHR": lambda record: record.H,
        "Null-only": lambda record: record.Y_null,
        "Shared-coarse": lambda record: record.Y_shared,
    }
    result: Dict[str, Any] = {"geometry_identity": {}}
    stitched_meshes: Dict[str, Any] = {}
    reference_tile_geometry: Optional[Dict[str, str]] = None
    reference_stitched_geometry: Optional[Tuple[str, str]] = None
    reference_image = None
    for candidate in (Path(args.source_dir) / "canonical_4096.png", Path(args.source_dir) / "input_original.png"):
        if candidate.is_file():
            reference_image = candidate
            break
    for variant_name, get_field in variants.items():
        patches = []
        geometry_digest: Dict[str, str] = {}
        for tile_id in sorted(selected_ids):
            record = records[tile_id]
            value = get_field(record)
            if value is None:
                raise RuntimeError(f"variant {variant_name} is unavailable for tile {tile_id}")
            vertex_bytes = record.geometry.vertices.detach().cpu().contiguous().numpy().tobytes()
            face_bytes = record.geometry.faces.detach().cpu().contiguous().numpy().tobytes()
            geometry_digest[str(tile_id)] = hashlib.sha256(vertex_bytes + face_bytes).hexdigest()
            # The repository's canonical face-corner sampler is CUDA-backed;
            # decoded production meshes arrive on CUDA as well.  Preserve that
            # route for oracle fields rather than invoking flex_gemm with a
            # CPU MeshWithVoxel.
            local_mesh = core._make_local_reference_mesh(record.geometry, value, baseline).to("cuda")
            patch = core._local_mesh_to_global_patch(
                tile_id=tile_id,
                box=record.box,
                local_mesh=local_mesh,
                global_camera=global_camera,
                transform=record.transform,
                query_chunk_size=int(args.query_chunk_size),
            )
            patches.append(patch)
            del local_mesh
        stitched, stitch_stats = core._stitch_tile_patches_nearest(
            patches,
            layout=dict(core.PBR_LAYOUT),
            global_camera=global_camera,
            face_chunk_size=int(args.render_face_chunk_size),
            weld_tolerance=float(args.stitch_tolerance),
        )
        stitched_meshes[variant_name] = stitched
        if reference_tile_geometry is None:
            reference_tile_geometry = geometry_digest
        elif geometry_digest != reference_tile_geometry:
            raise AssertionError(f"variant {variant_name}: local tile geometry changed")
        stitched_vertices_digest = hashlib.sha256(stitched.vertices.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        stitched_faces_digest = hashlib.sha256(stitched.faces.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        if reference_stitched_geometry is None:
            reference_stitched_geometry = (stitched_vertices_digest, stitched_faces_digest)
        elif reference_stitched_geometry != (stitched_vertices_digest, stitched_faces_digest):
            raise AssertionError(f"variant {variant_name}: stitched geometry changed")
        variant_dir = output_dir / "variants" / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        _atomic_torch(variant_dir / "global_merged_mesh.pt", {"mesh": stitched})
        result[variant_name] = {
            "mesh_pt": str((variant_dir / "global_merged_mesh.pt").resolve()),
            "stitch": stitch_stats,
            "geometry_digest": geometry_digest,
            "stitched_vertices_digest": stitched_vertices_digest,
            "stitched_faces_digest": stitched_faces_digest,
        }
        if args.render:
            from types import SimpleNamespace

            render_args = SimpleNamespace(
                render_resolution=int(args.render_resolution),
                metric_resolution=int(args.metric_resolution),
                render_ssaa=int(args.render_ssaa),
                render_peel_layers=int(args.render_peel_layers),
                render_face_chunk_size=int(args.render_face_chunk_size),
                use_envmap_bg=False,
                envmap=str(args.envmap),
                lpips_net="vgg",
                skip_lpips=True,
            )
            envmap = core.load_envmap(str(args.envmap), device="cuda")
            result[variant_name]["render"] = core._render(
                stitched,
                output_dir=variant_dir / "render",
                camera=global_camera,
                reference_image=reference_image,
                args=render_args,
                envmap=envmap,
            )
    result["geometry_identity"]["same_geometry_per_variant"] = True
    result["geometry_identity"]["tile_geometry_digests"] = reference_tile_geometry or {}
    result["geometry_identity"]["stitched_vertices_digest"] = reference_stitched_geometry[0] if reference_stitched_geometry else None
    result["geometry_identity"]["stitched_faces_digest"] = reference_stitched_geometry[1] if reference_stitched_geometry else None
    if bool(getattr(args, "render_multiview", False)) and bool(args.render):
        import pixal3d_pbr_range_null_perstep_experiment as range_null

        envmap = core.load_envmap(str(args.envmap), device="cuda")
        result["multiview"] = range_null._render_multiview_variants(
            meshes=stitched_meshes,
            baseline_mesh=baseline,
            global_camera=global_camera,
            output_root=output_dir,
            args=args,
            envmap=envmap,
        )
        del envmap
    _atomic_json(output_dir / "geometry_identity.json", result["geometry_identity"])
    _atomic_json(output_dir / "variants.json", result)
    return result


def _phase_a_questions(output_dir: Path, extra: Mapping[str, Any]) -> List[str]:
    """Render the required Q1-Q7 answers from saved Phase-A diagnostics."""
    consensus = extra.get("consensus", extra) if isinstance(extra, Mapping) else {}
    tiles = consensus.get("tiles", {}) if isinstance(consensus, Mapping) else {}
    hidden_total = 0
    evidence_total = 0
    for tile_id, diagnostics in tiles.items():
        hidden_path = output_dir / "fields" / f"tile_{int(tile_id):02d}" / "hidden_mask.pt"
        if hidden_path.is_file():
            hidden_total += int(_payload_tensor(hidden_path).to(torch.bool).sum().item())
        histogram = diagnostics.get("donor_count_histogram", {})
        evidence_total += sum(int(histogram.get(key, 0)) for key in ("2", "3", "4", "5+"))
    q1 = (
        f"{evidence_total:,}/{hidden_total:,} hidden C1024 rows have donor_count >= 2 "
        f"({evidence_total / hidden_total:.3%} weighted)."
        if hidden_total
        else "no hidden rows recorded"
    )

    agreement_path = output_dir / "coarse_agreement.json"
    agreement = json.loads(agreement_path.read_text(encoding="utf-8")) if agreement_path.is_file() else {}
    global_agreement = agreement.get("global", {}).get("all_valid_pair_pooled", {})
    cosine = global_agreement.get("cosine", {})
    disagreement = global_agreement.get("relative_disagreement", {})
    q2 = "pooled cosine mean " + ", ".join(
        f"{name}={float(cosine.get(name, {}).get('mean', float('nan'))):.4f}" for name in GROUPS
    ) + "; relative-disagreement mean " + ", ".join(
        f"{name}={float(disagreement.get(name, {}).get('mean', float('nan'))):.4f}" for name in GROUPS
    ) + "."

    aggregate = consensus.get("aggregate", {}) if isinstance(consensus, Mapping) else {}
    r_shared = aggregate.get("R_shared", {})
    r_preserve_shared = aggregate.get("R_preserve_shared", {})
    r_preserve_null = aggregate.get("R_preserve_null", {})
    q3 = "||C_shared||/||C|| = " + ", ".join(
        f"{name}={float(r_shared.get(name, float('nan'))):.4f}" for name in GROUPS
    ) + "."
    q4 = "preserved ||Y-G||/||Delta||, shared vs null: " + "; ".join(
        f"{name}={float(r_preserve_shared.get(name, float('nan'))):.4f} vs {float(r_preserve_null.get(name, float('nan'))):.4f}"
        for name in GROUPS
    ) + "."

    variants = extra.get("variants", {}) if isinstance(extra, Mapping) else {}
    if isinstance(variants, Mapping) and variants.get("multiview"):
        q5 = "Tile26/27 real front/back and fixed-view artifacts were generated; inspect variants/multiview and renderer metrics for the color/material judgment."
    elif isinstance(variants, Mapping) and variants.get("skipped"):
        q5 = "not judged yet: --no-render was used; Tile26/27 real front/back render artifacts are pending."
    else:
        q5 = "renderer artifacts are present; visual closeness is reported as an inspection question, not a thresholded claim."

    q6 = "PBR OOB weighted comparison unavailable."
    metrics_path = output_dir / "metrics.csv"
    if metrics_path.is_file():
        rows = list(csv.DictReader(metrics_path.open("r", encoding="utf-8")))
        weighted: Dict[str, Dict[str, float]] = {"Null-only": {}, "Shared-coarse": {}}
        weights: Dict[str, Dict[str, float]] = {"Null-only": {}, "Shared-coarse": {}}
        for row in rows:
            variant = row.get("variant", "")
            group = row.get("group", "")
            if variant in weighted and group in GROUPS:
                weight = float(row.get("hidden_rows", 0) or 0)
                weighted[variant][group] = weighted[variant].get(group, 0.0) + float(row.get("out_of_01_ratio", 0.0)) * weight
                weights[variant][group] = weights[variant].get(group, 0.0) + weight
        comparisons = []
        for name in GROUPS:
            null = weighted["Null-only"].get(name, 0.0) / max(weights["Null-only"].get(name, 1.0), 1.0)
            shared = weighted["Shared-coarse"].get(name, 0.0) / max(weights["Shared-coarse"].get(name, 1.0), 1.0)
            comparisons.append(f"{name}={shared:.4%} vs {null:.4%}")
        q6 = "weighted OOB ratio Shared-coarse vs Null-only: " + ", ".join(comparisons) + "."

    leakage_path = output_dir / "consensus_projection_leakage.json"
    q7 = "no-evidence leakage unavailable."
    if leakage_path.is_file():
        leakage = json.loads(leakage_path.read_text(encoding="utf-8")).get("tiles", {})
        values = [float(item.get("R_noEvidence", 0.0)) for item in leakage.values()]
        if values:
            q7 = f"R_noEvidence macro mean={float(np.mean(values)):.4%}, max={max(values):.4%}; see consensus_projection_leakage.json."
    return [
        f"Q1. {q1}",
        f"Q2. {q2}",
        f"Q3. {q3}",
        f"Q4. {q4}",
        f"Q5. {q5}",
        f"Q6. {q6}",
        f"Q7. {q7}",
    ]


def _write_report(output_dir: Path, preflight_result: Mapping[str, Any], status: str, extra: Optional[Mapping[str, Any]] = None) -> None:
    lines = [
        "# Final-PureHR Shared-Coarse Oracle",
        "",
        f"- status: `{status}`",
        f"- format: `{FORMAT}`",
        "- layout: canonical 4096, tile 1024, stride 512, 7x7 row-major",
        "- formulas: `Delta=H-G`, `c=P_h^dagger Delta_h`, `C_h=P_h c`, `D=Delta-C`",
        "- consensus: donor-local continuous query of `C=P_h A_h(H-G)`; equal float64 mean only when donor count >= 2",
        "- solver: formal `StableSparseMRAProjector`, direct float64 LSMR, x0=None; normal equations disabled",
        "- prohibited substitutions: range-null, MRA, step projection, per-step flow, Euler, re-encode, visibility/Gaussian/facing weights, and PBR clamp",
        "",
    ]
    if status != "ready":
        lines.extend(
            [
                "## Blocking input",
                "",
                "A final PureHR PBR field or a final PureHR endpoint is required for every selected tile. The preflight did not find one and therefore did not run the oracle.",
                "",
                "```json",
                json.dumps(_jsonable({"missing_final_tile_ids": preflight_result.get("missing_final_tile_ids"), "errors": preflight_result.get("errors"), "rejected_candidates": preflight_result.get("rejected_candidates")}), ensure_ascii=False, indent=2),
                "```",
            ]
        )
    if status == "completed" and extra:
        lines.extend(["", "## Phase-A questions Q1-Q7", ""])
        lines.extend(_phase_a_questions(output_dir, extra))
        lines.append("\nThese are diagnostic measurements; no single threshold is used to declare the method successful.")
    if extra:
        lines.extend(["", "## Run summary", "", "```json", json.dumps(_jsonable(extra), ensure_ascii=False, indent=2), "```"])
    report = "\n".join(lines) + "\n"
    (output_dir / "SHARED_COARSE_ORACLE_REPORT.md").write_text(report, encoding="utf-8")
    (output_dir / "PHASE_A_SHARED_COARSE_REPORT.md").write_text(report, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "smoke", "phase-a", "full"), default="preflight")
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--source-dir", type=Path, default=Path("outputs/pbr_range_null_perstep_cuda4_full"))
    parser.add_argument("--context-dir", type=Path, default=Path("outputs/cross_tile_pbr_perstep_guided_cuda4_full"))
    parser.add_argument("--operator-cache-dir", type=Path, default=Path("outputs/pbr_sparse_mra_delta_perstep_cuda4"))
    parser.add_argument("--pure-field-dir", type=Path, default=None)
    parser.add_argument(
        "--pure-endpoint-dir",
        type=Path,
        action="append",
        default=None,
        help="repeatable endpoint root; candidates are matched by official tile ID or tile_camera box",
    )
    parser.add_argument(
        "--allow-box-reuse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="allow a partial/legacy endpoint layout after exact tile_camera-box and fixed-shape checks",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pbr_shared_coarse_oracle_phase_a_v2"))
    parser.add_argument(
        "--reuse-prepared-fields",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="resume from complete fields/tile_XX artifacts in output-dir; rebuild geometry only",
    )
    parser.add_argument(
        "--reuse-consensus",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="resume from complete consensus artifacts in output-dir; skip donor queries and LSMR",
    )
    parser.add_argument("--tile-ids", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="/home/nvme04/yyyan/download/model/Pixal3D")
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query-chunk-size", type=int, default=250000)
    parser.add_argument("--face-projection-chunk-size", type=int, default=500000)
    parser.add_argument("--lsmr-atol", type=float, default=1e-7)
    parser.add_argument("--lsmr-btol", type=float, default=1e-7)
    parser.add_argument("--lsmr-maxiter", type=int, default=500)
    parser.add_argument("--lsmr-conlim", type=float, default=1e12)
    parser.add_argument("--lsmr-channel-workers", type=int, default=6)
    parser.add_argument("--operator-data-tolerance", type=float, default=1e-12)
    parser.add_argument("--stationarity-epsilon", type=float, default=1e-12)
    parser.add_argument("--self-transport-tolerance", type=float, default=1e-4)
    parser.add_argument("--diagnostic-sample-limit", type=int, default=10000)
    parser.add_argument("--visual-sample-limit", type=int, default=200000)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--render-resolution", type=int, default=4096)
    parser.add_argument("--metric-resolution", type=int, default=512)
    parser.add_argument("--render-ssaa", type=int, default=1)
    parser.add_argument("--render-peel-layers", type=int, default=8)
    parser.add_argument("--render-face-chunk-size", type=int, default=500000)
    parser.add_argument("--stitch-tolerance", type=float, default=1.0 / 1024.0)
    parser.add_argument("--envmap", type=str, default="studio")
    parser.add_argument("--render-multiview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-envmap-bg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--multiview-resolution", type=int, default=512)
    parser.add_argument("--multiview-ssaa", type=int, default=1)
    parser.add_argument("--multiview-peel-layers", type=int, default=4)
    parser.add_argument("--multiview-radius-scale", type=float, default=1.0)
    parser.add_argument("--multiview-turntable-frames", type=int, default=24)
    return parser


def _select_cuda_device(requested: int) -> Tuple[int, Optional[int]]:
    """Map the physical CUDA4 request to a logical index under CUDA_VISIBLE_DEVICES."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; requested cuda4 is unavailable")
    requested = int(requested)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    visible_ids: List[int] = []
    if visible and all(part.strip().lstrip("-").isdigit() for part in visible.split(",")):
        visible_ids = [int(part.strip()) for part in visible.split(",")]
    if visible_ids and requested in visible_ids:
        logical = visible_ids.index(requested)
        physical: Optional[int] = requested
    else:
        logical = requested
        physical = requested if not visible_ids else None
    if logical < 0 or logical >= torch.cuda.device_count():
        raise ValueError(
            f"requested CUDA device {requested} is unavailable: visible={visible!r}, count={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(logical)
    return logical, physical


def run(args: argparse.Namespace) -> int:
    logical_cuda, physical_cuda = _select_cuda_device(int(args.cuda_device))
    print(
        f"[cuda] requested_physical={args.cuda_device} logical={logical_cuda} "
        f"name={torch.cuda.get_device_name(logical_cuda)}"
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()
    context_dir = Path(args.context_dir).expanduser().resolve()
    operator_cache_dir = Path(args.operator_cache_dir).expanduser().resolve()
    pure_field_dir = Path(args.pure_field_dir).expanduser().resolve() if args.pure_field_dir else None
    endpoint_values = list(args.pure_endpoint_dir or [])
    if not endpoint_values:
        completion_default = Path("outputs/purehr_completion_phase_a_cuda4")
        if completion_default.exists():
            endpoint_values.append(completion_default)
    pure_endpoint_dirs = [Path(path).expanduser().resolve() for path in endpoint_values]
    requested = _parse_ids(args.tile_ids)
    if args.phase == "phase-a":
        selected = phase_a_ids() if requested is None else requested
        if selected != phase_a_ids():
            raise ValueError(f"Phase-A must be exactly the Tile26/27 3x3 neighborhood {sorted(PHASE_A_TILE_IDS)}")
    elif args.phase == "full":
        selected = set(FORMAL_VALID_TILE_IDS) if requested is None else requested
        if selected != set(FORMAL_VALID_TILE_IDS):
            raise ValueError(f"full phase must use the formal tile ensemble {sorted(FORMAL_VALID_TILE_IDS)}")
    elif args.phase == "smoke":
        selected = {18} if requested is None else requested
        if len(selected) != 1:
            raise ValueError("smoke phase accepts exactly one tile; use --tile-ids 18")
    else:
        selected = phase_a_ids() if requested is None else requested
    result = preflight(
        source_dir=source_dir,
        context_dir=context_dir,
        pure_field_dir=pure_field_dir,
        pure_endpoint_dirs=pure_endpoint_dirs,
        allow_box_reuse=bool(args.allow_box_reuse),
        tile_ids=selected,
        output_dir=output_dir,
        operator_cache_dir=operator_cache_dir,
        phase=str(args.phase),
    )
    _write_report(output_dir, result, "preflight_only" if args.phase == "preflight" else result["status"])
    print(json.dumps(_jsonable({"status": result["status"], "missing_final_tile_ids": result["missing_final_tile_ids"], "errors": result["errors"]}), ensure_ascii=False, indent=2))
    if args.phase == "preflight" or result["status"] != "ready":
        return 0 if args.phase == "preflight" else 2
    if bool(args.reuse_prepared_fields):
        records, run_info = load_prepared_tile_fields(
            args=args,
            selected_ids=selected,
            source_dir=source_dir,
            context_dir=context_dir,
            operator_cache_dir=operator_cache_dir,
            output_dir=output_dir,
        )
    else:
        records, run_info = prepare_tile_fields(
            args=args,
            selected_ids=selected,
            source_dir=source_dir,
            context_dir=context_dir,
            pure_field_dir=pure_field_dir,
            pure_endpoint_dirs=pure_endpoint_dirs,
            operator_cache_dir=operator_cache_dir,
            output_dir=output_dir,
        )
    if args.phase == "smoke":
        smoke_summary = {
            "status": "smoke_completed",
            "format": FORMAT,
            "selected_tile_ids": sorted(selected),
            "run_info": {
                "pipeline_loaded": run_info["pipeline_loaded"],
                "operator_provenance": run_info["operator_provenance"],
                "support_alignment": run_info["support_alignment"],
                "purehr_provenance": run_info["purehr_provenance"],
            },
            "flow_sampler_called": False,
            "encoder_called": False,
            "euler_called": False,
        }
        _atomic_json(output_dir / "summary.json", smoke_summary)
        _write_report(output_dir, result, "smoke_completed", smoke_summary)
        return 0
    if bool(args.reuse_consensus):
        consensus = load_consensus_artifacts(records, output_dir)
    else:
        consensus = build_consensus(
            records=records,
            selected_ids=selected,
            global_camera=run_info["global_camera"],
            baseline=run_info["baseline"],
            args=args,
            output_dir=output_dir,
        )
    _visualize_phase_a(records, output_dir, run_info["global_camera"], int(args.visual_sample_limit))
    if bool(args.render):
        variants = _render_variants(
            records=records,
            selected_ids=selected,
            baseline=run_info["baseline"],
            global_camera=run_info["global_camera"],
            output_dir=output_dir,
            args=args,
        )
    else:
        variants = {"skipped": True, "reason": "--no-render"}
    summary = {
        "status": "completed",
        "format": FORMAT,
        "selected_tile_ids": sorted(selected),
        "cuda": {"requested_physical": int(args.cuda_device), "logical": int(logical_cuda), "physical": physical_cuda, "name": torch.cuda.get_device_name(logical_cuda)},
        "run_info": {"pipeline_loaded": run_info["pipeline_loaded"], "purehr_provenance": run_info["purehr_provenance"], "operator_provenance": run_info["operator_provenance"], "support_alignment": run_info["support_alignment"]},
        "flow_sampler_called": False,
        "encoder_called": False,
        "euler_called": False,
        "xstart_to_pred_called": False,
        "per_step_projection_called": False,
        "consensus": consensus,
        "variants": variants,
    }
    _atomic_json(output_dir / "summary.json", summary)
    _write_report(output_dir, result, "completed", summary)
    return 0


def main() -> None:
    args = _build_parser().parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
