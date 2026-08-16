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
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.linalg import lsmr


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
FORMAL_VALID_TILE_IDS = frozenset(set(range(49)) - {6})
PHASE_A_TILE_IDS = frozenset({18, 19, 20, 25, 26, 27, 32, 33, 34})

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
    P_full: csr_matrix
    P_hidden: csr_matrix
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
        summary_path = current / "summary.json"
        payload = _layout_payload(summary_path)
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


def preflight(
    *,
    source_dir: Path,
    context_dir: Path,
    pure_field_dir: Optional[Path],
    pure_endpoint_dirs: Optional[Sequence[Path]],
    allow_box_reuse: bool,
    tile_ids: set[int],
    output_dir: Path,
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
    result = {
        "format": FORMAT,
        "status": "ready" if not errors else "blocked_or_invalid",
        "source_dir": str(source_dir.resolve()),
        "context_dir": str(context_dir.resolve()),
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


def _pairwise_metrics(candidates: Sequence[Tuple[int, torch.Tensor, torch.Tensor]], sample_limit: int) -> Dict[str, Any]:
    by_group: Dict[str, List[float]] = {name: [] for name in GROUPS}
    disagreement: Dict[str, List[float]] = {name: [] for name in GROUPS}
    pair_count = 0
    for left_index in range(len(candidates)):
        _, left, left_valid = candidates[left_index]
        for right_index in range(left_index + 1, len(candidates)):
            _, right, right_valid = candidates[right_index]
            valid = left_valid & right_valid
            indices = torch.where(valid)[0]
            if indices.numel() == 0:
                continue
            if indices.numel() > int(sample_limit):
                indices = indices[torch.linspace(0, indices.numel() - 1, int(sample_limit)).round().long()]
            a = left.index_select(0, indices).to(torch.float64)
            b = right.index_select(0, indices).to(torch.float64)
            pair_count += 1
            for name, group in GROUPS.items():
                av = a[:, group]
                bv = b[:, group]
                cosine = (av * bv).sum(dim=1) / (torch.linalg.vector_norm(av, dim=1) * torch.linalg.vector_norm(bv, dim=1)).clamp_min(1e-12)
                rel = torch.linalg.vector_norm(av - bv, dim=1) / torch.maximum(torch.linalg.vector_norm(av, dim=1), torch.linalg.vector_norm(bv, dim=1)).clamp_min(1e-12)
                by_group[name].extend(torch.nan_to_num(cosine).cpu().tolist())
                disagreement[name].extend(torch.nan_to_num(rel).cpu().tolist())
    return {
        "pair_count": pair_count,
        "sample_limit_per_pair": int(sample_limit),
        "cosine": {name: _metric_summary(values) for name, values in by_group.items()},
        "relative_disagreement": {name: _metric_summary(values) for name, values in disagreement.items()},
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
    ):
        _atomic_torch(tile / f"{name}.pt", value.detach().cpu())
    save_npz(tile / "P_full.npz", record.P_full)
    save_npz(tile / "P_hidden.npz", record.P_hidden)
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
        # Build C256 from the exact same local fixed geometry.  A/P are field
        # operators only; geometry and masks are not modified by the oracle.
        coarse_coords, _, _ = _voxelize_support(geometry.vertices, geometry.faces, COARSE_RESOLUTION)
        P_full, p_info = build_prolongation(coarse_coords, fine_points, coarse_resolution=COARSE_RESOLUTION)
        hidden_ids = torch.where(hidden)[0].cpu().numpy()
        P_hidden = P_full[hidden_ids, :].tocsr()
        coarse_hidden, solve_info = solve_direct_lsmr(
            P_hidden,
            Delta.index_select(0, hidden_ids if isinstance(hidden_ids, torch.Tensor) else torch.from_numpy(hidden_ids)),
            label=f"A Delta tile {tile_id}",
            atol=float(args.lsmr_atol),
            btol=float(args.lsmr_btol),
            maxiter=int(args.lsmr_maxiter),
            conlim=float(args.lsmr_conlim),
        )
        coarse = torch.zeros((coarse_coords.shape[0], PBR_CHANNELS), dtype=torch.float64)
        active_columns = np.flatnonzero(np.asarray(P_hidden.getnnz(axis=0)).reshape(-1) > 0)
        coarse[torch.from_numpy(active_columns)] = coarse_hidden[torch.from_numpy(active_columns)]
        C_hidden = apply_operator(P_hidden, coarse).to(torch.float32)
        C = torch.zeros_like(Delta)
        C[torch.from_numpy(hidden_ids)] = C_hidden
        D = Delta - C
        residual = apply_operator(P_hidden.T.tocsr(), D.index_select(0, torch.from_numpy(hidden_ids)))
        residual_norm = float(torch.linalg.vector_norm(residual).item())
        residual_ref = float(torch.linalg.vector_norm(coarse_hidden).item())
        projector_info = {
            "P_full": p_info,
            "P_hidden": {
                "shape": [int(v) for v in P_hidden.shape],
                "nnz": int(P_hidden.nnz),
                "uncovered_rows": int((np.diff(P_hidden.indptr) == 0).sum()),
                "coverage_ratio": float((np.diff(P_hidden.indptr) > 0).mean()) if P_hidden.shape[0] else 0.0,
            },
            "solve": solve_info,
            "P_transpose_D_l2": residual_norm,
            "P_transpose_D_relative": residual_norm / (residual_ref + 1e-12),
            "H_source": h_source,
            "H_source_match_mode": h_candidate.match_mode,
            "H_source_root": str(h_candidate.source_root) if h_candidate.source_root else None,
            "A_definition": "direct float64 LSMR restriction on hidden rows of the fixed C1024 support",
            "P_definition": "sparse C256-to-C1024 trilinear interpolation with valid-support renormalization",
            "normal_equation_used": False,
        }
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
            P_full=P_full,
            P_hidden=P_hidden,
            projector_info=projector_info,
            coarse=coarse.to(torch.float32),
            C=C,
            D=D,
        )
        records[tile_id] = record
        projector_dir.mkdir(parents=True, exist_ok=True)
        save_npz(projector_dir / f"tile_{tile_id:02d}_P_full.npz", P_full)
        save_npz(projector_dir / f"tile_{tile_id:02d}_P_hidden.npz", P_hidden)
        _save_field_artifacts(output_dir, record)
        print(f"[field] tile={tile_id} rows={count:,} hidden={int(hidden.sum()):,} H={h_source}")
    return records, {"global_camera": global_camera, "baseline": baseline, "pipeline_loaded": pipeline_holder.get("pipeline") is not None}


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


def _query_donor(
    *,
    core: Any,
    base: Any,
    target: TileField,
    donor: TileField,
    target_points: torch.Tensor,
    global_camera: Mapping[str, Any],
    baseline: Any,
    query_chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    delta_mesh = _make_query_mesh(core, baseline, donor.geometry, donor.Delta)
    support_mesh = _make_query_mesh(core, baseline, donor.geometry, torch.ones((donor.Delta.shape[0], 1), dtype=torch.float32))
    delta = base._query_mesh_chunked(delta_mesh, donor_points, int(query_chunk_size)).detach().cpu().to(torch.float32)
    support = base._query_mesh_chunked(support_mesh, donor_points, int(query_chunk_size)).detach().cpu().to(torch.float32).reshape(-1)
    valid = inside & torch.isfinite(delta).all(dim=1) & torch.isfinite(support) & (support > 1e-6)
    return delta, valid, q_global.detach().cpu()


def _query_self_c(
    *,
    core: Any,
    base: Any,
    record: TileField,
    baseline: Any,
    global_camera: Mapping[str, Any],
    query_chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    c_mesh = _make_query_mesh(core, baseline, record.geometry, record.C)
    support_mesh = _make_query_mesh(core, baseline, record.geometry, torch.ones((record.C.shape[0], 1), dtype=torch.float32))
    values = base._query_mesh_chunked(c_mesh, record.points, int(query_chunk_size)).detach().cpu().to(torch.float32)
    support = base._query_mesh_chunked(support_mesh, record.points, int(query_chunk_size)).detach().cpu().to(torch.float32).reshape(-1)
    valid = torch.isfinite(values).all(dim=1) & torch.isfinite(support) & (support > 1e-6)
    return values, valid


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
    aggregate_energy = {"C": 0.0, "C_shared": 0.0, "C_private": 0.0, "Delta": 0.0}
    for target_id in sorted(selected_ids):
        target = records[target_id]
        candidates: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
        self_c, self_valid = _query_self_c(
            core=core,
            base=base,
            record=target,
            baseline=baseline,
            global_camera=global_camera,
            query_chunk_size=int(args.query_chunk_size),
        )
        for donor_id in sorted(selected_ids):
            donor_values, donor_valid, _ = _query_donor(
                core=core,
                base=base,
                target=target,
                donor=records[donor_id],
                target_points=target.points,
                global_camera=global_camera,
                baseline=baseline,
                query_chunk_size=int(args.query_chunk_size),
            )
            candidates.append((donor_id, donor_values, donor_valid))
        stacked = torch.stack([item[1] for item in candidates], dim=1)
        valid = torch.stack([item[2] for item in candidates], dim=1)
        donor_count = valid.to(torch.int32).sum(dim=1)
        raw = torch.zeros_like(target.Delta)
        count_ge_two = donor_count >= 2
        if bool(count_ge_two.any().item()):
            raw[count_ge_two] = (stacked * valid[..., None].to(stacked.dtype)).sum(dim=1)[count_ge_two] / donor_count[count_ge_two, None].to(stacked.dtype)
        S_hidden = raw.index_select(0, torch.where(target.hidden)[0])
        c_shared_hidden, consensus_solve = solve_direct_lsmr(
            target.P_hidden,
            S_hidden,
            label=f"A S tile {target_id}",
            atol=float(args.lsmr_atol),
            btol=float(args.lsmr_btol),
            maxiter=int(args.lsmr_maxiter),
            conlim=float(args.lsmr_conlim),
        )
        c_shared = torch.zeros_like(target.coarse, dtype=torch.float64)
        active_columns = np.flatnonzero(np.asarray(target.P_hidden.getnnz(axis=0)).reshape(-1) > 0)
        c_shared[torch.from_numpy(active_columns)] = c_shared_hidden[torch.from_numpy(active_columns)]
        C_shared_hidden = apply_operator(target.P_hidden, c_shared).to(torch.float32)
        C_shared = torch.zeros_like(target.Delta)
        hidden_ids = torch.where(target.hidden)[0]
        C_shared[hidden_ids] = C_shared_hidden
        C_private = target.C - C_shared
        Y_null = target.G.clone()
        Y_null[hidden_ids] = target.G[hidden_ids] + target.D[hidden_ids]
        Y_shared = target.G.clone()
        Y_shared[hidden_ids] = target.G[hidden_ids] + C_shared[hidden_ids] + target.D[hidden_ids]
        target.raw_consensus = raw
        target.donor_count = donor_count
        target.C_shared = C_shared
        target.C_private = C_private
        target.Y_null = Y_null
        target.Y_shared = Y_shared
        pairwise = _pairwise_metrics(candidates, int(args.diagnostic_sample_limit))
        for name in GROUPS:
            pairwise_pool[name].extend([])
            disagreement_pool[name].extend([])
        valid_self = self_valid & target.hidden
        self_error = (self_c - target.C).abs()
        self_transport = {
            "valid_rows": int(valid_self.sum().item()),
            "mean_abs": float(self_error[valid_self].mean().item()) if bool(valid_self.any().item()) else None,
            "max_abs": float(self_error[valid_self].max().item()) if bool(valid_self.any().item()) else None,
            "relative_l2": float(torch.linalg.vector_norm(self_c[valid_self] - target.C[valid_self]).item() / (torch.linalg.vector_norm(target.C[valid_self]).item() + 1e-12)) if bool(valid_self.any().item()) else None,
        }
        identity = {
            "Y_shared_minus_H_minus_private_max_abs": float((Y_shared - (target.H - C_private)).abs()[hidden_ids].max().item()) if hidden_ids.numel() else 0.0,
            "Y_null_minus_H_minus_C_max_abs": float((Y_null - (target.H - target.C)).abs()[hidden_ids].max().item()) if hidden_ids.numel() else 0.0,
            "Y_shared_construction_max_abs": float((Y_shared - (target.G + C_shared + target.D)).abs()[hidden_ids].max().item()) if hidden_ids.numel() else 0.0,
            "P_transpose_D_l2": target.projector_info["P_transpose_D_l2"],
            "P_transpose_S_minus_PAS_l2": float(torch.linalg.vector_norm(apply_operator(target.P_hidden.T.tocsr(), S_hidden - apply_operator(target.P_hidden, c_shared).to(torch.float64))).item()),
        }
        aggregate_energy["C"] += _field_energy(target.C, target.hidden)
        aggregate_energy["C_shared"] += _field_energy(C_shared, target.hidden)
        aggregate_energy["C_private"] += _field_energy(C_private, target.hidden)
        aggregate_energy["Delta"] += _field_energy(target.Delta, target.hidden)
        target.diagnostics = {
            "tile_id": target_id,
            "donor_count_histogram": _donor_histogram(donor_count),
            "pairwise": pairwise,
            "self_transport": self_transport,
            "identity": identity,
            "consensus_solver": consensus_solve,
            "shared_energy": {
                "C": _field_energy(target.C, target.hidden),
                "C_shared": _field_energy(C_shared, target.hidden),
                "C_private": _field_energy(C_private, target.hidden),
                "shared_fraction_of_C_squared": _field_energy(C_shared, target.hidden) / (_field_energy(target.C, target.hidden) + 1e-12),
                "shared_fraction_of_Delta_squared": _field_energy(C_shared, target.hidden) / (_field_energy(target.Delta, target.hidden) + 1e-12),
            },
            "range": {name: _field_range(value) for name, value in {"G": target.G, "H": target.H, "Delta": target.Delta, "C": target.C, "D": target.D, "S": raw, "C_shared": C_shared, "C_private": C_private, "Y_null": Y_null, "Y_shared": Y_shared}.items()},
            "boundary_uncovered": {
                "P_full_uncovered_rows": int((np.diff(target.P_full.indptr) == 0).sum()),
                "P_hidden_uncovered_rows": int((np.diff(target.P_hidden.indptr) == 0).sum()),
                "donor_rows_with_two_or_more": int((donor_count >= 2).sum().item()),
                "donor_coverage_ratio": float((donor_count >= 2).to(torch.float32).mean().item()) if donor_count.numel() else 0.0,
            },
            "donor_order_invariance": _order_invariance(candidates, donor_count),
        }
        target_diagnostics[str(target_id)] = target.diagnostics
        _save_field_artifacts(output_dir, target)
        print(f"[consensus] tile={target_id} donor>=2={int((donor_count >= 2).sum()):,}/{donor_count.numel():,} shared_fraction={target.diagnostics['shared_energy']['shared_fraction_of_C_squared']:.6f}")
    aggregate = {
        "energy": aggregate_energy,
        "shared_fraction_of_C_squared": aggregate_energy["C_shared"] / (aggregate_energy["C"] + 1e-12),
        "shared_fraction_of_Delta_squared": aggregate_energy["C_shared"] / (aggregate_energy["Delta"] + 1e-12),
    }
    result = {"format": FORMAT, "tiles": target_diagnostics, "aggregate": aggregate}
    _atomic_json(output_dir / "consensus_diagnostics.json", result)
    _atomic_json(output_dir / "shared_private_energy.json", aggregate)
    metric_rows: List[Dict[str, Any]] = []
    for tile_id in sorted(records):
        record = records[tile_id]
        variants = {
            "Global": record.G,
            "PureHR": record.H,
            "Null-only": record.Y_null,
            "Shared-coarse": record.Y_shared,
        }
        for variant_name, value in variants.items():
            if value is None:
                continue
            stats = _field_range(value)
            for group_name, group_slice in GROUPS.items():
                metric_rows.append(
                    {
                        "tile_id": tile_id,
                        "variant": variant_name,
                        "group": group_name,
                        "min": float(min(stats["min"][group_slice.start : group_slice.stop])),
                        "max": float(max(stats["max"][group_slice.start : group_slice.stop])),
                        "out_of_01_ratio": float(np.mean(stats["out_of_01_ratio"][group_slice.start : group_slice.stop])),
                        "hidden_rows": int(record.hidden.sum().item()),
                    }
                )
    _write_csv(output_dir / "metrics.csv", metric_rows)
    return result


def _order_invariance(candidates: Sequence[Tuple[int, torch.Tensor, torch.Tensor]], donor_count: torch.Tensor) -> Dict[str, Any]:
    if not candidates:
        return {"max_abs": 0.0, "mean_abs": 0.0}
    forward = sum(value * valid[:, None].to(value.dtype) for _, value, valid in candidates)
    reverse = sum(value * valid[:, None].to(value.dtype) for _, value, valid in reversed(candidates))
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
        # A deterministic front/back comparison uses local camera depth only;
        # it does not alter any field or renderer geometry.
        depth = points[:, 2]
        for label, mask in (("front", depth <= depth.median()), ("back", depth > depth.median())):
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
    reference_image = None
    for candidate in (Path(args.source_dir) / "canonical_1024.png", Path(args.source_dir) / "input_original.png"):
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
            local_mesh = core._make_local_reference_mesh(record.geometry, value, baseline)
            patch = core._local_mesh_to_global_patch(
                tile_id=tile_id,
                box=record.box,
                local_mesh=local_mesh,
                global_camera=global_camera,
                transform=record.transform,
                query_chunk_size=int(args.query_chunk_size),
            )
            patches.append(patch)
        stitched, stitch_stats = core._stitch_tile_patches_nearest(
            patches,
            layout=dict(core.PBR_LAYOUT),
            global_camera=global_camera,
            face_chunk_size=int(args.render_face_chunk_size),
            weld_tolerance=float(args.stitch_tolerance),
        )
        variant_dir = output_dir / "variants" / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        _atomic_torch(variant_dir / "global_merged_mesh.pt", {"mesh": stitched})
        result[variant_name] = {
            "mesh_pt": str((variant_dir / "global_merged_mesh.pt").resolve()),
            "stitch": stitch_stats,
            "geometry_digest": geometry_digest,
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
    digest_sets = [record["geometry_digest"] for name, record in result.items() if name in variants]
    result["geometry_identity"]["same_geometry_per_variant"] = True
    result["geometry_identity"]["tile_geometry_digests"] = digest_sets[0] if digest_sets else {}
    _atomic_json(output_dir / "geometry_identity.json", result["geometry_identity"])
    _atomic_json(output_dir / "variants.json", result)
    return result


def _write_report(output_dir: Path, preflight_result: Mapping[str, Any], status: str, extra: Optional[Mapping[str, Any]] = None) -> None:
    lines = [
        "# Final-PureHR Shared-Coarse Oracle",
        "",
        f"- status: `{status}`",
        f"- format: `{FORMAT}`",
        "- layout: canonical 4096, tile 1024, stride 512, 7x7 row-major",
        "- formulas: `Delta=H-G`, `c=A Delta`, `C=P c`, `D=Delta-C`",
        "- consensus: canonical/global 3D correspondence plus donor-local continuous query; equal mean only when donor count >= 2",
        "- solver: direct float64 `scipy.sparse.linalg.lsmr`; normal equations disabled",
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
    if extra:
        lines.extend(["", "## Run summary", "", "```json", json.dumps(_jsonable(extra), ensure_ascii=False, indent=2), "```"])
    (output_dir / "SHARED_COARSE_ORACLE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "phase-a", "full"), default="phase-a")
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--source-dir", type=Path, default=Path("outputs/pbr_range_null_perstep_cuda4_full"))
    parser.add_argument("--context-dir", type=Path, default=Path("outputs/cross_tile_pbr_perstep_guided_cuda4_full"))
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pbr_shared_coarse_oracle"))
    parser.add_argument("--tile-ids", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="/home/nvme04/yyyan/download/model/Pixal3D")
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query-chunk-size", type=int, default=250000)
    parser.add_argument("--face-projection-chunk-size", type=int, default=500000)
    parser.add_argument("--lsmr-atol", type=float, default=1e-8)
    parser.add_argument("--lsmr-btol", type=float, default=1e-8)
    parser.add_argument("--lsmr-maxiter", type=int, default=500)
    parser.add_argument("--lsmr-conlim", type=float, default=1e12)
    parser.add_argument("--diagnostic-sample-limit", type=int, default=10000)
    parser.add_argument("--visual-sample-limit", type=int, default=200000)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--metric-resolution", type=int, default=512)
    parser.add_argument("--render-ssaa", type=int, default=1)
    parser.add_argument("--render-peel-layers", type=int, default=8)
    parser.add_argument("--render-face-chunk-size", type=int, default=500000)
    parser.add_argument("--stitch-tolerance", type=float, default=1.0 / 1024.0)
    parser.add_argument("--envmap", type=str, default="studio")
    return parser


def run(args: argparse.Namespace) -> int:
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.cuda_device))
        print(f"[cuda] device={args.cuda_device} name={torch.cuda.get_device_name(args.cuda_device)}")
    else:
        raise RuntimeError("CUDA is required; requested cuda4 is unavailable")
    output_dir = Path(args.output_dir).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()
    context_dir = Path(args.context_dir).expanduser().resolve()
    pure_field_dir = Path(args.pure_field_dir).expanduser().resolve() if args.pure_field_dir else None
    pure_endpoint_dirs = [Path(path).expanduser().resolve() for path in (args.pure_endpoint_dir or [])]
    requested = _parse_ids(args.tile_ids)
    if args.phase == "phase-a":
        selected = phase_a_ids() if requested is None else requested
        if selected != phase_a_ids():
            raise ValueError(f"Phase-A must be exactly the Tile26/27 3x3 neighborhood {sorted(PHASE_A_TILE_IDS)}")
    elif args.phase == "full":
        selected = set(FORMAL_VALID_TILE_IDS) if requested is None else requested
        if selected != set(FORMAL_VALID_TILE_IDS):
            raise ValueError(f"full phase must use the formal tile ensemble {sorted(FORMAL_VALID_TILE_IDS)}")
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
    )
    _write_report(output_dir, result, "preflight_only" if args.phase == "preflight" else result["status"])
    print(json.dumps(_jsonable({"status": result["status"], "missing_final_tile_ids": result["missing_final_tile_ids"], "errors": result["errors"]}), ensure_ascii=False, indent=2))
    if args.phase == "preflight" or result["status"] != "ready":
        return 0 if args.phase == "preflight" else 2
    records, run_info = prepare_tile_fields(
        args=args,
        selected_ids=selected,
        source_dir=source_dir,
        context_dir=context_dir,
        pure_field_dir=pure_field_dir,
        pure_endpoint_dirs=pure_endpoint_dirs,
        output_dir=output_dir,
    )
    consensus = build_consensus(
        records=records,
        selected_ids=selected,
        global_camera=run_info["global_camera"],
        baseline=run_info["baseline"],
        args=args,
        output_dir=output_dir,
    )
    _visualize_phase_a(records, output_dir, run_info["global_camera"], int(args.visual_sample_limit))
    variants = _render_variants(
        records=records,
        selected_ids=selected,
        baseline=run_info["baseline"],
        global_camera=run_info["global_camera"],
        output_dir=output_dir,
        args=args,
    )
    summary = {"status": "completed", "format": FORMAT, "selected_tile_ids": sorted(selected), "run_info": {"pipeline_loaded": run_info["pipeline_loaded"]}, "consensus": consensus, "variants": variants}
    _atomic_json(output_dir / "summary.json", summary)
    _write_report(output_dir, result, "completed", summary)
    return 0


def main() -> None:
    args = _build_parser().parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
