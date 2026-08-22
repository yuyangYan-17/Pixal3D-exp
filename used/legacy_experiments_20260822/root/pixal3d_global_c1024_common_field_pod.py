#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global C1024 common-field POD diagnostic.

This module is deliberately independent from the historical shared-coarse
oracle.  It compares final PureHR fields and a single global C1024 baseline
only on direct sparse-query support.  No C256 operator, projector, range/null
field, MRA, fusion, flow, or re-encoding is used here.
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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix


FORMAT = "pixal3d_global_c1024_common_field_pod_v1"
CANONICAL_IMAGE_SIZE = 4096
TILE_SIZE = 1024
TILE_STRIDE = 512
RESOLUTION = 1024
PBR_CHANNELS = 6
PHASE_A_TILE_IDS = frozenset({18, 19, 20, 25, 26, 27, 32, 33, 34})
QUARTETS = (
    (18, 19, 25, 26),
    (19, 20, 26, 27),
    (25, 26, 32, 33),
    (26, 27, 33, 34),
)
GROUPS: Dict[str, slice] = {
    "RGB": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}
FORBIDDEN_MARKERS = (
    "range_null",
    "mra",
    "projector",
    "step_",
    "projection",
    "trajectory",
    "perstep",
    "guided",
    "gaussian",
)


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
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
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
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
        mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in keys})
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def official_layout() -> List[Tuple[int, int, int, int]]:
    starts = list(range(0, CANONICAL_IMAGE_SIZE - TILE_SIZE + 1, TILE_STRIDE))
    if len(starts) != 7 or starts[-1] != CANONICAL_IMAGE_SIZE - TILE_SIZE:
        raise RuntimeError("official 4096/1024/512 layout is unavailable")
    return [(x, y, x + TILE_SIZE, y + TILE_SIZE) for y in starts for x in starts]


def _tile_dir(root: Path, tile_id: int) -> Path:
    for candidate in (
        root / "tiles" / f"tile_{tile_id:02d}",
        root / "tiles" / f"tile_{tile_id}",
        root / f"tile_{tile_id:02d}",
        root / f"tile_{tile_id}",
    ):
        if candidate.is_dir():
            return candidate
    return root / "tiles" / f"tile_{tile_id:02d}"


def _payload_tensor(path: Path, keys: Iterable[str] = ("tensor", "features", "mask", "value")) -> torch.Tensor:
    payload = _load_torch(path)
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, Mapping):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise ValueError(f"no tensor payload in {path}")


def _normalise_coords(coords: torch.Tensor) -> torch.Tensor:
    coords = coords.detach().cpu().to(torch.int32)
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"coordinates must be [N,3] or [N,4], got {tuple(coords.shape)}")
    if coords.shape[1] == 4:
        if bool((coords[:, 0] != 0).any().item()):
            raise ValueError("batched coordinates must have batch index zero")
        return coords[:, 1:]
    return coords


def _field_payload(path: Path) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    payload = _load_torch(path)
    if isinstance(payload, torch.Tensor):
        value = payload
        coords = None
    elif isinstance(payload, Mapping):
        value = None
        for key in ("H", "field", "pbr", "final_pbr", "tensor", "attrs", "features", "raw"):
            candidate = payload.get(key)
            if isinstance(candidate, torch.Tensor) and candidate.ndim == 2 and candidate.shape[1] == PBR_CHANNELS:
                value = candidate
                break
        if value is None:
            raise ValueError(f"{path}: no six-channel final field")
        coords = payload.get("coords") if isinstance(payload.get("coords"), torch.Tensor) else None
    else:
        raise ValueError(f"{path}: invalid final field payload")
    value = value.detach().cpu().to(torch.float32).contiguous()
    if value.ndim != 2 or value.shape[1] != PBR_CHANNELS or not torch.isfinite(value).all():
        raise ValueError(f"{path}: field is not finite [N,6]")
    return value, (_normalise_coords(coords) if coords is not None else None)


def _recorded_route(tile_dir: Path) -> Mapping[str, Any]:
    """Return a compact route record, walking to the experiment root."""
    current = tile_dir
    for _ in range(6):
        for name in ("projector.json", "provenance.json", "summary.json", "purehr_provenance.json"):
            path = current / name
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, Mapping):
                continue
            # The endpoint manifest stores one record per tile at the root.
            if isinstance(payload.get("tiles"), Mapping):
                tile_name = tile_dir.name
                digits = "".join(ch for ch in tile_name if ch.isdigit())
                if digits and isinstance(payload["tiles"].get(digits), Mapping):
                    payload = payload["tiles"][digits]
            summary: Dict[str, Any] = {}
            for key in (
                "tile_id", "canonical_box", "route", "pure_HR", "H_source",
                "H_source_match_mode", "H_source_root", "endpoint", "model_path",
                "guidance_used", "MRA_used", "cross_tile_used", "flow_route",
                "noise_timestep", "flow_steps", "sampler_params",
            ):
                if key in payload:
                    value = payload[key]
                    if key == "sampler_params" and isinstance(value, Mapping):
                        value = {str(k): value[k] for k in ("texture_steps", "texture_guidance_strength", "texture_guidance_rescale", "texture_rescale_t") if k in value}
                    summary[key] = value
            if summary:
                return summary
        if current.parent == current:
            break
        current = current.parent
    return {}


def _route_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(_jsonable(payload), ensure_ascii=False).lower()


def _field_candidates(root: Optional[Path], tile_id: int) -> List[Path]:
    if root is None or not root.exists():
        return []
    tile = _tile_dir(root, tile_id)
    names = {"h.pt", "field.pt", "pbr_field.pt", "final_field.pt", "pure_hr_field.pt", "purehr_field.pt"}
    paths = [path for path in tile.rglob("*.pt") if path.name.lower() in names] if tile.is_dir() else []
    return sorted(set(paths))


def _endpoint_candidates(root: Optional[Path], tile_id: int) -> List[Path]:
    if root is None or not root.exists():
        return []
    tile = _tile_dir(root, tile_id)
    names = {"pure_hr_endpoint.pt", "purehr_endpoint.pt", "final_purehr_endpoint.pt"}
    paths = [path for path in tile.rglob("*.pt") if path.name.lower() in names] if tile.is_dir() else []
    return sorted(set(paths))


def _candidate_rejection(path: Path, kind: str) -> Optional[str]:
    lowered = "/".join(part.lower() for part in path.parts)
    if any(marker in lowered for marker in FORBIDDEN_MARKERS):
        # Existing endpoint root names include "perstep" for the cache's
        # historical producer; the endpoint itself is accepted only when its
        # provenance proves the official pure-HR route.
        if kind == "endpoint" and path.name.lower() == "pure_hr_endpoint.pt":
            pass
        elif kind == "field" and path.name.lower() == "h.pt":
            pass
        else:
            return "path contains a prohibited artifact marker"
    route = _recorded_route(path.parent)
    text = _route_text(route)
    if kind == "endpoint":
        route_name = str(route.get("route", "")).lower()
        pure_route = route.get("pure_HR")
        pure_route_name = str(pure_route.get("route", "")).lower() if isinstance(pure_route, Mapping) else ""
        if route and route_name and "official pure hr" not in route_name and "purehr" not in route_name and "pure_hr" not in route_name and "official pure hr" not in pure_route_name:
            return "endpoint provenance does not identify official PureHR"
        if route.get("guidance_used") is True or route.get("MRA_used") is True or route.get("cross_tile_used") is True:
            return "endpoint provenance contains a prohibited route marker"
    if kind == "field":
        source = route.get("H_source") if isinstance(route, Mapping) else None
        if source is None and isinstance(route.get("provenance"), Mapping):
            source = route["provenance"].get("H_source")
        if source is not None:
            source_text = str(source).lower()
            if any(marker in source_text for marker in ("range_null", "mra", "guided", "projector", "step_", "trajectory")):
                return "cached field provenance points to a prohibited artifact"
    return None


def _candidate_record(path: Path, kind: str, tile_id: int, expected_box: Sequence[int]) -> Dict[str, Any]:
    tile_camera = path.parent / "tile_camera.json"
    box = None
    if tile_camera.is_file():
        try:
            payload = json.loads(tile_camera.read_text(encoding="utf-8"))
            box = tuple(int(v) for v in payload.get("box", ()))
        except Exception:
            box = None
    rejection = _candidate_rejection(path, kind)
    official = tuple(int(v) for v in expected_box)
    if box is not None and box != official:
        rejection = f"tile camera box {box} != official box {official}"
    return {
        "tile_id": int(tile_id),
        "path": str(path.resolve()),
        "kind": kind,
        "box": list(box) if box is not None else None,
        "rejection": rejection,
        "sha256": _sha256(path) if path.is_file() else None,
        "route_metadata": _recorded_route(path.parent),
    }


def _support_path(support_dir: Path, tile_id: int) -> Path:
    return _tile_dir(support_dir, tile_id) / "fine_coords.pt"


def preflight(
    *,
    output_dir: Path,
    source_dir: Path,
    context_dir: Path,
    support_dir: Path,
    baseline_path: Path,
    pure_field_dir: Optional[Path],
    pure_endpoint_dir: Optional[Path],
    tile_ids: Sequence[int] = tuple(sorted(PHASE_A_TILE_IDS)),
) -> Dict[str, Any]:
    boxes = official_layout()
    errors: List[str] = []
    tiles: Dict[str, Any] = {}
    accepted_fields: Dict[int, Path] = {}
    accepted_endpoints: Dict[int, Path] = {}
    for tile_id in sorted(int(v) for v in tile_ids):
        if tile_id < 0 or tile_id >= len(boxes):
            errors.append(f"tile {tile_id} is outside official layout")
            continue
        static = _tile_dir(source_dir, tile_id)
        context = _tile_dir(context_dir, tile_id)
        support = _support_path(support_dir, tile_id)
        required = {
            "global_pbr_reference": static / "global_pbr_reference.pt",
            "hidden_mask": static / "hidden_mask.pt",
            "observed_mask": static / "observed_mask.pt",
            "tile_camera": static / "tile_camera.json",
            "fixed_shape_norm": context / "fixed_shape_norm.pt",
            "context_tile_camera": context / "tile_camera.json",
            "fine_coords": support,
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            errors.append(f"tile {tile_id}: missing {missing}")
        field_records = [_candidate_record(p, "field", tile_id, boxes[tile_id]) for p in _field_candidates(pure_field_dir, tile_id)]
        endpoint_records = [_candidate_record(p, "endpoint", tile_id, boxes[tile_id]) for p in _endpoint_candidates(pure_endpoint_dir, tile_id)]
        accepted_field_records = [r for r in field_records if r["rejection"] is None]
        accepted_endpoint_records = [r for r in endpoint_records if r["rejection"] is None]
        if accepted_field_records:
            accepted_fields[tile_id] = Path(accepted_field_records[0]["path"])
        elif accepted_endpoint_records:
            accepted_endpoints[tile_id] = Path(accepted_endpoint_records[0]["path"])
        else:
            errors.append(f"tile {tile_id}: no legal final PureHR field or endpoint")
        if not (static / "tile_camera.json").is_file():
            continue
        try:
            camera = json.loads((static / "tile_camera.json").read_text(encoding="utf-8"))
            if tuple(int(v) for v in camera.get("box", ())) != boxes[tile_id]:
                errors.append(f"tile {tile_id}: static tile camera box is not official")
        except Exception as exc:
            errors.append(f"tile {tile_id}: invalid tile_camera.json: {exc}")
        tiles[str(tile_id)] = {
            "official_box": list(boxes[tile_id]),
            "required": {name: str(path.resolve()) for name, path in required.items()},
            "missing": missing,
            "field_candidates": field_records,
            "endpoint_candidates": endpoint_records,
            "selected_field": str(accepted_fields[tile_id].resolve()) if tile_id in accepted_fields else None,
            "selected_endpoint": str(accepted_endpoints[tile_id].resolve()) if tile_id in accepted_endpoints else None,
        }
    if not baseline_path.is_file():
        errors.append(f"global baseline artifact is missing: {baseline_path}")
    result = {
        "format": FORMAT,
        "status": "ready" if not errors else "blocked_or_invalid",
        "phase_a_tile_ids": sorted(int(v) for v in tile_ids),
        "layout": {
            "canonical_image_size": CANONICAL_IMAGE_SIZE,
            "tile_size": TILE_SIZE,
            "stride": TILE_STRIDE,
            "tile_count": len(boxes),
            "boxes": [list(box) for box in boxes],
        },
        "source_dir": str(source_dir.resolve()),
        "context_dir": str(context_dir.resolve()),
        "support_dir": str(support_dir.resolve()),
        "baseline_path": str(baseline_path.resolve()),
        "baseline_sha256": _sha256(baseline_path) if baseline_path.is_file() else None,
        "pure_field_dir": str(pure_field_dir.resolve()) if pure_field_dir else None,
        "pure_endpoint_dir": str(pure_endpoint_dir.resolve()) if pure_endpoint_dir else None,
        "tiles": tiles,
        "errors": errors,
        "prohibited_routes": list(FORBIDDEN_MARKERS),
    }
    _atomic_json(output_dir / "preflight.json", result)
    return result


def _linear_keys(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    value = coords.to(torch.int64)
    return (value[:, 0] * int(resolution) + value[:, 1]) * int(resolution) + value[:, 2]


def build_sparse_query_matrix(
    active_coords: torch.Tensor,
    query_points: torch.Tensor,
    resolution: int = RESOLUTION,
) -> Tuple[csr_matrix, torch.Tensor, Dict[str, Any]]:
    """Build sparse-support trilinear interpolation rows.

    The point convention is the same as ``MeshWithVoxel``: ``origin`` is
    ``[-0.5, -0.5, -0.5]`` and an active coordinate denotes the center of a
    ``1 / resolution`` cell.  Missing neighbours are omitted and the row is
    renormalized.  No inverse or projection is involved.
    """
    coords = _normalise_coords(active_coords).to(torch.int64)
    points = query_points.detach().cpu().to(torch.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"query_points must be [N,3], got {tuple(points.shape)}")
    if coords.numel() and bool(((coords < 0) | (coords >= int(resolution))).any().item()):
        raise ValueError("active coordinates lie outside the requested resolution")
    row_count = int(points.shape[0])
    col_count = int(coords.shape[0])
    if col_count:
        keys = _linear_keys(coords, resolution)
        order = torch.argsort(keys, stable=True)
        sorted_keys = keys.index_select(0, order)
        if sorted_keys.numel() > 1 and bool((sorted_keys[1:] == sorted_keys[:-1]).any().item()):
            raise ValueError("active coordinates contain duplicates")
    else:
        order = torch.empty(0, dtype=torch.int64)
        sorted_keys = torch.empty(0, dtype=torch.int64)
    grid = (points + 0.5) * float(resolution)
    finite = torch.isfinite(grid).all(dim=1)
    safe_grid = torch.where(finite[:, None], grid, torch.zeros_like(grid))
    base = torch.floor(safe_grid - 0.5).to(torch.int64)
    frac = safe_grid - (base.to(torch.float32) + 0.5)
    row_parts: List[torch.Tensor] = []
    col_parts: List[torch.Tensor] = []
    weight_parts: List[torch.Tensor] = []
    row_sum = torch.zeros(row_count, dtype=torch.float64)
    for bits in range(8):
        bit = torch.tensor([(bits >> 0) & 1, (bits >> 1) & 1, (bits >> 2) & 1], dtype=torch.int64)
        neighbour = base + bit
        weight = torch.where(bit.bool(), frac, 1.0 - frac).prod(dim=1).to(torch.float64)
        valid = finite & ((neighbour >= 0) & (neighbour < int(resolution))).all(dim=1)
        if sorted_keys.numel():
            neighbour_keys = _linear_keys(neighbour, resolution)
            positions = torch.searchsorted(sorted_keys, neighbour_keys)
            valid &= positions < sorted_keys.numel()
            safe = positions.clamp_max(sorted_keys.numel() - 1)
            valid &= sorted_keys.index_select(0, safe) == neighbour_keys
        else:
            safe = torch.zeros(row_count, dtype=torch.int64)
            valid &= False
        rows = torch.where(valid)[0]
        if rows.numel():
            cols = order.index_select(0, safe.index_select(0, rows))
            values = weight.index_select(0, rows)
            row_parts.append(rows)
            col_parts.append(cols)
            weight_parts.append(values)
            row_sum.index_add_(0, rows, values)
    if row_parts:
        rows = torch.cat(row_parts)
        cols = torch.cat(col_parts)
        values = torch.cat(weight_parts) / row_sum.index_select(0, rows).clamp_min(1e-15)
        matrix = csr_matrix((values.numpy(), (rows.numpy(), cols.numpy())), shape=(row_count, col_count), dtype=np.float64)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
    else:
        matrix = csr_matrix((row_count, col_count), dtype=np.float64)
    valid_rows = torch.from_numpy((np.diff(matrix.indptr) > 0).copy())
    row_nnz = np.diff(matrix.indptr)
    metadata = {
        "resolution": int(resolution),
        "fine_rows": row_count,
        "active_columns": col_count,
        "nnz": int(matrix.nnz),
        "valid_rows": int(valid_rows.sum().item()),
        "invalid_rows": int((~valid_rows).sum().item()),
        "coverage_ratio": float(valid_rows.float().mean().item()) if row_count else 0.0,
        "row_nnz_min": int(row_nnz.min()) if row_nnz.size else 0,
        "row_nnz_max": int(row_nnz.max()) if row_nnz.size else 0,
        "row_nnz_mean": float(row_nnz.mean()) if row_nnz.size else 0.0,
        "support_rule": "8-neighbor trilinear with missing-neighbor renormalization",
        "operator": "coefficients to query values; no inverse/projector",
    }
    return matrix, valid_rows, metadata


def apply_query_matrix(matrix: csr_matrix, values: torch.Tensor) -> torch.Tensor:
    array = values.detach().cpu().to(torch.float64).numpy()
    return torch.from_numpy(np.asarray(matrix.dot(array), dtype=np.float64)).to(torch.float32)


def _metric(left: torch.Tensor, right: torch.Tensor) -> Dict[str, float]:
    diff = (left.detach().to(torch.float64) - right.detach().to(torch.float64)).reshape(-1)
    ref = right.detach().to(torch.float64).reshape(-1)
    return {
        "max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.abs().mean().item()) if diff.numel() else 0.0,
        "relative_l2": float(torch.linalg.vector_norm(diff).item() / (torch.linalg.vector_norm(ref).item() + 1e-12)),
    }


def uncentered_pod(columns: torch.Tensor, directional: bool = True) -> Dict[str, Any]:
    """Return raw and optionally normalized uncentered POD from [D,K] columns."""
    matrix = columns.detach().cpu().to(torch.float64)
    if matrix.ndim != 2:
        raise ValueError(f"POD input must be [D,K], got {tuple(matrix.shape)}")
    gram = matrix.T @ matrix
    eigenvalues, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues.index_select(0, order).clamp_min(0.0)
    vectors = vectors.index_select(1, order)
    sigma = torch.sqrt(eigenvalues)
    total = float(eigenvalues.sum().item())
    energy = eigenvalues / total if total > 0.0 else torch.zeros_like(eigenvalues)
    result: Dict[str, Any] = {
        "gram": gram,
        "eigenvalues": eigenvalues,
        "sigma": sigma,
        "energy_ratio": energy,
        "cumulative_energy_ratio": torch.cumsum(energy, dim=0),
        "V": vectors,
        "norms": torch.linalg.vector_norm(matrix, dim=0),
        "participating_columns": list(range(int(matrix.shape[1]))),
    }
    if directional:
        norms = result["norms"]
        participating = torch.where(norms > 1e-12)[0]
        if participating.numel():
            normalized = matrix.index_select(1, participating) / norms.index_select(0, participating)[None, :]
            dgram = normalized.T @ normalized
            deig, dvec = torch.linalg.eigh(dgram)
            dorder = torch.argsort(deig, descending=True)
            deig = deig.index_select(0, dorder).clamp_min(0.0)
            dvec = dvec.index_select(1, dorder)
            dtotal = float(deig.sum().item())
            denergy = deig / dtotal if dtotal > 0.0 else torch.zeros_like(deig)
        else:
            deig = torch.zeros(0, dtype=torch.float64)
            dvec = torch.zeros((0, 0), dtype=torch.float64)
            denergy = deig
        result.update({
            "directional_gram": dgram if participating.numel() else torch.zeros((0, 0), dtype=torch.float64),
            "directional_eigenvalues": deig,
            "directional_sigma": torch.sqrt(deig),
            "directional_energy_ratio": denergy,
            "directional_cumulative_energy_ratio": torch.cumsum(denergy, dim=0),
            "directional_V": dvec,
            "directional_participating_columns": participating.tolist(),
        })
    return result


def pairwise_cosine_from_gram(gram: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
    denominator = norms[:, None] * norms[None, :]
    result = torch.full_like(gram, float("nan"), dtype=torch.float64)
    valid = denominator > 1e-12
    result[valid] = gram[valid] / denominator[valid]
    return result


def _select_cuda_device(requested: int) -> Tuple[int, Optional[int]]:
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested} is unavailable in this environment")
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
        raise RuntimeError(f"requested cuda{requested} is unavailable: visible={visible!r}, count={torch.cuda.device_count()}")
    torch.cuda.set_device(logical)
    return logical, physical


def _query_mesh_chunked(mesh: Any, points: torch.Tensor, chunk_size: int) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    for start in range(0, int(points.shape[0]), int(chunk_size)):
        rows.append(mesh.query_attrs(points[start : start + int(chunk_size)]).to(torch.float32).detach().cpu())
    return torch.cat(rows, dim=0) if rows else torch.empty((0, int(mesh.attrs.shape[1])), dtype=torch.float32)


def _load_mesh_query_object(path: Path, device: torch.device) -> Tuple[Any, Any, Dict[str, Any]]:
    payload = _load_torch(path)
    mesh = payload.get("mesh", payload) if isinstance(payload, Mapping) else payload
    from pixal3d.representations import MeshWithVoxel

    if not isinstance(mesh, MeshWithVoxel):
        raise ValueError(f"global baseline is not MeshWithVoxel: {path}")
    if mesh.attrs.ndim != 2 or mesh.attrs.shape[1] != PBR_CHANNELS:
        raise ValueError("global baseline attrs must have six channels")
    coords = mesh.coords.detach().cpu().to(torch.int32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"global baseline coords must be [N,3], got {tuple(coords.shape)}")
    if coords.numel() and bool(((coords < 0) | (coords >= RESOLUTION)).any().item()):
        raise ValueError("global baseline coordinates lie outside the C1024 voxel domain")
    origin = mesh.origin.detach().cpu().to(torch.float32)
    voxel_size = float(mesh.voxel_size)
    # Sparse MeshWithVoxel stores cropped spatial extents; voxel_size and
    # coordinate bounds, rather than those extents, establish the C1024 grid.
    voxel_shape = tuple(int(value) for value in mesh.voxel_shape)
    if len(voxel_shape) < 3 or any(value <= 0 for value in voxel_shape[-3:]):
        raise ValueError(f"global baseline voxel_shape must have positive spatial extents, got {voxel_shape}")
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError(f"global baseline voxel_size must be positive and finite, got {voxel_size}")
    if not math.isclose(voxel_size, 1.0 / RESOLUTION, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"global baseline voxel_size is not C{RESOLUTION}: {voxel_size}")
    if not torch.isfinite(mesh.attrs.detach().cpu()).all():
        raise ValueError("global baseline attrs contain non-finite values")
    points = origin[None, :] + (coords.to(torch.float32) + 0.5) * voxel_size
    query_mesh = MeshWithVoxel(
        vertices=torch.empty((1, 3), dtype=torch.float32, device=device),
        faces=torch.empty((0, 3), dtype=torch.int32, device=device),
        origin=origin.tolist(),
        voxel_size=voxel_size,
        coords=coords.to(device=device),
        attrs=mesh.attrs.detach().cpu().to(device=device, dtype=torch.float32),
        voxel_shape=torch.Size(voxel_shape),
        layout=dict(mesh.layout),
    )
    metadata = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "resolution": int(RESOLUTION),
        "attrs_channels": int(PBR_CHANNELS),
        "coords": int(coords.shape[0]),
        "coords_shape": list(coords.shape),
        "attrs_shape": list(mesh.attrs.shape),
        "origin": origin.tolist(),
        "voxel_size": voxel_size,
        "voxel_shape": list(voxel_shape),
        "layout": {key: [value.start, value.stop] for key, value in mesh.layout.items()},
        "query_definition": "origin + (coords + 0.5) * voxel_size, then MeshWithVoxel.query_attrs",
    }
    return mesh, query_mesh, {"points": points, **metadata}


def _load_local_field(path: Path, support_coords: torch.Tensor) -> torch.Tensor:
    field, coords = _field_payload(path)
    if field.shape[0] != support_coords.shape[0]:
        raise ValueError(f"{path}: field rows {field.shape[0]} != support rows {support_coords.shape[0]}")
    if coords is not None and not torch.equal(coords, support_coords):
        raise ValueError(f"{path}: cached field coordinate order differs from C1024 support")
    return field


def _load_endpoint_field(
    endpoint_path: Path,
    context_path: Path,
    query_points: torch.Tensor,
    device: torch.device,
    model_path: str,
    query_chunk_size: int,
    pipeline_holder: Dict[str, Any],
) -> torch.Tensor:
    """Decode a legal final endpoint once, without sampling or re-encoding."""
    payload = _load_torch(endpoint_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"endpoint is not a mapping: {endpoint_path}")
    fixed = _load_torch(context_path)
    if not isinstance(fixed, Mapping):
        raise ValueError(f"fixed shape context is not a mapping: {context_path}")
    from pixal3d.modules.sparse import SparseTensor
    from inference import init_pipeline
    import pixal3d_cross_tile_pbr_perstep as base

    shape = SparseTensor(fixed["features"].to(torch.float32), fixed["coords"].to(torch.int32))
    feature_key = "norm" if isinstance(payload.get("norm"), torch.Tensor) else "features"
    texture = SparseTensor(payload[feature_key].to(torch.float32), payload["coords"].to(torch.int32))
    if not torch.equal(shape.coords, texture.coords):
        raise ValueError("endpoint shape and texture supports differ")
    if not torch.equal(shape.coords, fixed["coords"].to(torch.int32)) or not torch.equal(shape.feats, fixed["features"].to(torch.float32)):
        raise ValueError("endpoint shape support differs from fixed-shape context")
    if pipeline_holder.get("pipeline") is None:
        pipeline_holder["pipeline"] = init_pipeline(model_path, device="cuda", low_vram=True)
    pipeline = pipeline_holder["pipeline"]
    shape_denorm = base._denormalize_slat(shape, pipeline.shape_slat_normalization)
    shape_denorm = base._sparse_to_device(shape_denorm, device)
    texture = base._sparse_to_device(texture, device)
    decoded = pipeline.decode_latent(
        shape_denorm,
        base._denormalize_slat(texture, pipeline.tex_slat_normalization),
        RESOLUTION,
    )
    if len(decoded) != 1:
        raise RuntimeError(f"endpoint decoder returned {len(decoded)} meshes")
    mesh = decoded[0]
    points = query_points.to(device=device)
    field = _query_mesh_chunked(mesh, points, query_chunk_size)
    if field.shape[1] != PBR_CHANNELS or not torch.isfinite(field).all():
        raise ValueError(f"decoded endpoint field is invalid: {endpoint_path}")
    return field


def _global_to_local(core: Any, global_points: torch.Tensor, global_camera: Mapping[str, Any], transform: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_global = global_points * (2.0 * float(global_camera["mesh_scale"]))
    q_local, uv_tile = core._global_q_to_local_q(q_global, global_camera=global_camera, transform=transform)
    local_points = q_local / (2.0 * float(transform.mesh_scale))
    return q_global, local_points, uv_tile


def _local_to_global(core: Any, local_points: torch.Tensor, global_camera: Mapping[str, Any], transform: Any) -> torch.Tensor:
    q_local = local_points * (2.0 * float(transform.mesh_scale))
    q_global, _ = core._local_q_to_global_q(q_local, global_camera=global_camera, transform=transform)
    return q_global


def _plot_heatmap(path: Path, matrix: np.ndarray, labels: Sequence[str], title: str, cmap: str = "viridis") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap=cmap, aspect="equal")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "N/A" if not np.isfinite(value) else f"{value:.3g}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_support(path: Path, points: torch.Tensor, values: torch.Tensor, title: str, max_points: int = 150000) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = int(points.shape[0])
    if count > max_points:
        indices = torch.linspace(0, count - 1, max_points).round().to(torch.long)
        points = points.index_select(0, indices)
        values = values.index_select(0, indices)
    points = points.cpu().numpy()
    values = values.cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(points[:, 0], points[:, 2], c=values, s=1, cmap="viridis", rasterized=True)
    ax.set_xlabel("global x")
    ax.set_ylabel("global z")
    ax.set_title(title + " (physical O-Voxel x/z projection)")
    fig.colorbar(scatter, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_pod(path: Path, sigma: torch.Tensor, energy: torch.Tensor, title: str, cumulative: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = energy.detach().cpu().numpy() if cumulative else (energy.detach().cpu().numpy())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(values) + 1), values, marker="o")
    ax.set_xlabel("mode")
    ax.set_ylabel("cumulative energy ratio" if cumulative else "energy ratio")
    ax.set_title(title)
    ax.set_xticks(np.arange(1, len(values) + 1))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_amplitude(path: Path, tile_ids: Sequence[int], norms: torch.Tensor, pc1: torch.Tensor, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(tile_ids))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.18, norms.detach().cpu().numpy(), width=0.36, label="||x_i||")
    ax.bar(x + 0.18, pc1.detach().cpu().numpy(), width=0.36, label="v1 coefficient")
    ax.set_xticks(x, [str(v) for v in tile_ids])
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _pod_for_mask(
    deltas: Mapping[int, torch.Tensor],
    tile_ids: Sequence[int],
    mask: torch.Tensor,
    group: slice,
    chunk_size: int = 250000,
) -> Dict[str, Any]:
    indices = torch.where(mask.detach().cpu().bool())[0]
    width = len(range(*group.indices(PBR_CHANNELS)))
    gram = torch.zeros((len(tile_ids), len(tile_ids)), dtype=torch.float64)
    norms_sq = torch.zeros(len(tile_ids), dtype=torch.float64)
    for start in range(0, int(indices.numel()), int(chunk_size)):
        rows = indices[start : start + int(chunk_size)]
        block = torch.stack(
            [deltas[int(tile_id)].index_select(0, rows)[:, group].reshape(-1).to(torch.float64) for tile_id in tile_ids],
            dim=1,
        )
        gram += block.T @ block
        norms_sq += (block * block).sum(dim=0)
    pod = uncentered_pod_from_gram(gram, norms_sq)
    pod["sample_voxels"] = int(indices.numel())
    pod["group_channels"] = int(width)
    return pod


def uncentered_pod_from_gram(gram: torch.Tensor, norms_sq: torch.Tensor) -> Dict[str, Any]:
    gram = gram.detach().cpu().to(torch.float64)
    norms_sq = norms_sq.detach().cpu().to(torch.float64)
    eigenvalues, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues.index_select(0, order).clamp_min(0.0)
    vectors = vectors.index_select(1, order)
    sigma = torch.sqrt(eigenvalues)
    total = float(eigenvalues.sum().item())
    energy = eigenvalues / total if total > 0.0 else torch.zeros_like(eigenvalues)
    norms = torch.sqrt(norms_sq.clamp_min(0.0))
    directional = torch.where(norms > 1e-12, norms, torch.ones_like(norms))
    dgram = gram / directional[:, None] / directional[None, :]
    active = torch.where(norms > 1e-12)[0]
    if active.numel():
        deig, dvec = torch.linalg.eigh(dgram.index_select(0, active).index_select(1, active))
        dorder = torch.argsort(deig, descending=True)
        deig = deig.index_select(0, dorder).clamp_min(0.0)
        dvec = dvec.index_select(1, dorder)
        dtotal = float(deig.sum().item())
        denergy = deig / dtotal if dtotal > 0 else torch.zeros_like(deig)
    else:
        deig = torch.zeros(0, dtype=torch.float64)
        dvec = torch.zeros((0, 0), dtype=torch.float64)
        denergy = deig
    return {
        "gram": gram,
        "sigma": sigma,
        "energy_ratio": energy,
        "cumulative_energy_ratio": torch.cumsum(energy, 0),
        "V": vectors,
        "norms": norms,
        "rho1": float(energy[0].item()) if energy.numel() else None,
        "rho12": float(energy[:2].sum().item()) if energy.numel() else None,
        "directional_gram": dgram,
        "directional_sigma": torch.sqrt(deig),
        "directional_energy_ratio": denergy,
        "directional_cumulative_energy_ratio": torch.cumsum(denergy, 0),
        "directional_V": dvec,
        "directional_participating_columns": active.tolist(),
        "directional_rho1": float(denergy[0].item()) if denergy.numel() else None,
        "directional_rho12": float(denergy[:2].sum().item()) if denergy.numel() else None,
        "cosine": pairwise_cosine_from_gram(gram, norms),
    }


def _pod_scalar(value: Optional[float]) -> str:
    return "N/A" if value is None or not math.isfinite(float(value)) else f"{float(value):.9g}"


def _save_pod_outputs(
    quartet_dir: Path,
    quartet: Sequence[int],
    domain: str,
    group_name: str,
    pod: Mapping[str, Any],
) -> None:
    group_dir = quartet_dir / domain
    group_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        key: value
        for key, value in pod.items()
        if isinstance(value, (torch.Tensor, int, float, str, list, tuple, type(None)))
    }
    _atomic_torch(group_dir / f"pod_{group_name}.pt", payload)
    sigma = pod["sigma"]
    energy = pod["energy_ratio"]
    _plot_pod(group_dir / f"pod_spectrum_{group_name}.png", sigma, energy, f"{domain} {group_name} scree")
    _plot_pod(group_dir / f"pod_cumulative_{group_name}.png", sigma, pod["cumulative_energy_ratio"], f"{domain} {group_name} cumulative", cumulative=True)
    _plot_heatmap(group_dir / f"cosine_{group_name}.png", pod["cosine"].numpy(), [str(v) for v in quartet], f"{domain} {group_name} pairwise cosine", cmap="coolwarm")
    v = pod["V"][:, 0] if pod["V"].numel() else torch.zeros(len(quartet), dtype=torch.float64)
    _plot_amplitude(group_dir / f"amplitude_pc1_{group_name}.png", quartet, pod["norms"], v, f"{domain} {group_name} amplitude / PC1")


def _load_tile_camera(source_dir: Path, tile_id: int, box: Sequence[int]) -> Mapping[str, Any]:
    path = _tile_dir(source_dir, tile_id) / "tile_camera.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tuple(int(v) for v in payload.get("box", ())) != tuple(int(v) for v in box):
        raise ValueError(f"tile {tile_id}: camera box does not match official layout")
    return payload


def _source_mask(source_dir: Path, tile_id: int, name: str) -> torch.Tensor:
    return _payload_tensor(_tile_dir(source_dir, tile_id) / f"{name}.pt").detach().cpu().to(torch.bool).reshape(-1)


def _common_mask(mask_by_tile: Mapping[int, torch.Tensor], tile_ids: Sequence[int]) -> torch.Tensor:
    result = torch.ones_like(mask_by_tile[int(tile_ids[0])], dtype=torch.bool)
    for tile_id in tile_ids:
        result &= mask_by_tile[int(tile_id)]
    return result


def _build_report(
    output_dir: Path,
    summary: Mapping[str, Any],
    coverage_rows: Sequence[Mapping[str, Any]],
    pod_rows: Sequence[Mapping[str, Any]],
    cosine_rows: Sequence[Mapping[str, Any]],
    loto_rows: Sequence[Mapping[str, Any]],
) -> None:
    gate: Mapping[str, Any] = {}
    gate_path = output_dir / "correctness_gate.json"
    if gate_path.is_file():
        try:
            payload = json.loads(gate_path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                gate = payload
        except (OSError, json.JSONDecodeError):
            gate = {}
    lines = [
        "# Global C1024 Common-Field POD Diagnostic",
        "",
        f"- status: `{summary.get('status')}`",
        f"- GPU: `{summary.get('cuda', {}).get('name', 'N/A')}` (requested physical cuda{summary.get('cuda', {}).get('requested_physical', 'N/A')})",
        "- definition: `D_i = H_i - G` on one global baseline C1024 active O-Voxel support",
        "- no mean centering; raw and per-tile directional uncentered POD are both reported",
        "- no C256, projector, LSMR, range/null, MRA, fusion, guidance, re-encode, or trajectory artifact",
        "",
        "## 1. Global support and provenance",
        "",
        f"- active global C1024 O-Voxels: `{summary.get('global_support', {}).get('count', 'N/A')}`",
        f"- global baseline: `{summary.get('global_support', {}).get('artifact_path', 'N/A')}`",
        f"- global baseline SHA256: `{summary.get('global_support', {}).get('artifact_sha256', 'N/A')}`",
        f"- global self-query correctness: `{json.dumps(summary.get('global_support', {}).get('self_query', {}), ensure_ascii=False)}`",
    ]
    provenance = summary.get("purehr_provenance", {})
    if isinstance(provenance, Mapping) and provenance:
        lines.extend(["", "PureHR final-field provenance:", "", "| tile | selected field/endpoint | kind | endpoint source recorded in provenance |", "|---:|---|---|---|"])
        for tile_id in sorted(provenance, key=lambda value: int(value)):
            record = provenance[tile_id] if isinstance(provenance[tile_id], Mapping) else {}
            route = record.get("route_metadata", {}) if isinstance(record, Mapping) else {}
            h_source = route.get("H_source", "") if isinstance(route, Mapping) else ""
            lines.append(f"| {tile_id} | `{record.get('path', 'N/A')}` | {record.get('kind', 'N/A')} | `{h_source}` |")
    if gate:
        lines.extend([
            "",
            "## 1.1 Correctness gate",
            "",
            f"- status: `{gate.get('status', 'N/A')}`",
            f"- mask epsilon: `{gate.get('mask_epsilon', 'N/A')}`",
            f"- invalid rows excluded from POD: `{gate.get('invalid_rows_are_excluded', 'N/A')}`",
            f"- missing-value sentinel: `{gate.get('missing_value_sentinel', 'NaN')}`",
            "",
            "### Global/local round-trip",
            "",
            "| tile | max abs | mean abs | relative L2 |",
            "|---:|---:|---:|---:|",
        ])
        for tile_id, metrics in sorted((gate.get("roundtrip", {}) or {}).items(), key=lambda item: int(item[0])):
            lines.append(f"| {tile_id} | {_pod_scalar(metrics.get('max_abs'))} | {_pod_scalar(metrics.get('mean_abs'))} | {_pod_scalar(metrics.get('relative_l2'))} |")
        lines.extend(["", "### Q operator cross-check (Tile26/27)", "", "| tile | group | max abs | mean abs | relative L2 | samples |", "|---:|---|---:|---:|---:|---:|"])
        for tile_id, record in sorted((gate.get("tile26_tile27_query_correctness", {}) or {}).items(), key=lambda item: int(item[0])):
            for group_name, metrics in (record.get("channel_metrics", {}) or {}).items():
                lines.append(f"| {tile_id} | {group_name} | {_pod_scalar(metrics.get('max_abs'))} | {_pod_scalar(metrics.get('mean_abs'))} | {_pod_scalar(metrics.get('relative_l2'))} | {record.get('sample_count', 'N/A')} |")
        lines.extend(["", "### Legacy tile-local reference cross-check", "", "These existing `global_pbr_reference.pt` files are used only as a cross-check; `G_query` remains the sole global field definition.", "", "| tile | max abs | mean abs | relative L2 |", "|---:|---:|---:|---:|"])
        for tile_id, metrics in sorted((gate.get("global_reference_consistency", {}) or {}).items(), key=lambda item: int(item[0])):
            lines.append(f"| {tile_id} | {_pod_scalar(metrics.get('max_abs'))} | {_pod_scalar(metrics.get('mean_abs'))} | {_pod_scalar(metrics.get('relative_l2'))} |")
    lines.extend([
        "",
        "## 2. Tile coverage",
        "",
        "| tile | valid global support | hidden | observed | mixed |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in coverage_rows:
        lines.append(f"| {row['tile_id']} | {row['valid']} | {row['hidden']} | {row['observed']} | {row['mixed']} |")
    overlap_path = output_dir / "coverage" / "pairwise_overlap.csv"
    overlap_rows: List[Mapping[str, str]] = []
    if overlap_path.is_file():
        with overlap_path.open("r", encoding="utf-8", newline="") as handle:
            overlap_rows = list(csv.DictReader(handle))
    overlap_ids = sorted({int(row["tile_i"]) for row in overlap_rows} | {int(row["tile_j"]) for row in overlap_rows}) if overlap_rows else []
    lines.extend(["", "Pairwise direct overlap is also saved in `coverage/pairwise_overlap.csv`; 9x9 heatmaps are in `coverage/`.", ""])
    if overlap_ids:
        overlap_lookup = {(int(row["tile_i"]), int(row["tile_j"])): row for row in overlap_rows}
        for key, title in (("common_valid_voxels", "direct valid overlap"), ("common_hidden_voxels", "direct hidden overlap"), ("common_observed_voxels", "direct observed overlap")):
            lines.extend([f"### 9x9 {title}", "", "| tile | " + " | ".join(str(tile_id) for tile_id in overlap_ids) + " |", "|---:|" + "---:|" * len(overlap_ids)])
            for left_id in overlap_ids:
                values = [overlap_lookup[(left_id, right_id)].get(key, "N/A") for right_id in overlap_ids]
                lines.append("| " + str(left_id) + " | " + " | ".join(values) + " |")
            lines.append("")
    lines.extend(["## 3. Quartet support", "", "| quartet | valid | hidden | observed | mixed |", "|---|---:|---:|---:|---:|"])
    for row in summary.get("quartets", []):
        lines.append(f"| {','.join(map(str, row['quartet']))} | {row['common_valid_count']} | {row['common_hidden_count']} | {row['common_observed_count']} | {row['mixed_count']} |")
    lines.extend(["", "## 4. POD metrics", "", "| quartet | domain | group | samples | raw rho1 | raw rho12 | directional rho1 | directional rho12 |", "|---|---|---|---:|---:|---:|---:|---:|"])
    for row in pod_rows:
        lines.append(
            f"| {','.join(map(str, row['quartet']))} | {row['domain']} | {row['group']} | {row['sample_voxels']} | "
            f"{_pod_scalar(row.get('rho1'))} | {_pod_scalar(row.get('rho12'))} | {_pod_scalar(row.get('directional_rho1'))} | {_pod_scalar(row.get('directional_rho12'))} |"
        )
    lines.extend(["", "## 5. Pairwise cosine", "", "The complete matrices are in `pairwise_cosine.csv` and each quartet/domain directory. Diagonal entries are one when the norm is nonzero; zero-norm pairs are reported as `N/A`.", ""])
    for row in summary.get("cosine_summary", []):
        lines.append(f"- {','.join(map(str, row['quartet']))} / {row['domain']} / {row['group']}: mean={_pod_scalar(row.get('mean'))}, min={_pod_scalar(row.get('min'))}, max={_pod_scalar(row.get('max'))}.")
    lines.extend(["", "## 6. Leave-one-tile-out stability", "", "| quartet | domain | group | omitted | full rho1 | LOTO rho1 | full rho12 | LOTO rho12 | s_LOTO |", "|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in loto_rows:
        lines.append(f"| {','.join(map(str, row['quartet']))} | {row['domain']} | {row['group']} | {row['omitted_tile']} | {_pod_scalar(row.get('full_rho1'))} | {_pod_scalar(row.get('loto_rho1'))} | {_pod_scalar(row.get('full_rho12'))} | {_pod_scalar(row.get('loto_rho12'))} | {_pod_scalar(row.get('s_loto'))} |")
    lines.extend([
        "",
        "## 7. Required interpretation",
        "",
        "1. `H_i-G` is evaluated only where every tile in a quartet has direct sparse C1024 support; missing rows never enter POD and are never filled with zero.",
        "2. Hidden and observed analyses use strict query fractions (`>= 1 - 1e-6`); mixed visibility is excluded from both.",
        "3. A shared direction with different amplitudes would appear as high raw/directional concentration and positive pairwise cosine, while stable LOTO vectors provide the robustness check. The measurements above are descriptive and no hard pass threshold is imposed.",
        "4. The decision about Wavelet x POD is therefore based on the complete quartet, hidden/observed, cosine, and LOTO evidence above; this run does not implement fusion or Wavelet.",
        "",
        f"- machine-readable summary: `{(output_dir / 'summary.json').resolve()}`",
        f"- correctness gate: `{(output_dir / 'correctness_gate.json').resolve()}`",
    ])
    (output_dir / "GLOBAL_C1024_COMMON_FIELD_POD_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()
    context_dir = Path(args.context_dir).expanduser().resolve()
    support_dir = Path(args.support_dir).expanduser().resolve()
    baseline_path = Path(args.baseline_path).expanduser().resolve()
    pure_field_dir = Path(args.pure_field_dir).expanduser().resolve() if args.pure_field_dir else None
    pure_endpoint_dir = Path(args.pure_endpoint_dir).expanduser().resolve() if args.pure_endpoint_dir else None
    preflight_result = preflight(
        output_dir=output_dir,
        source_dir=source_dir,
        context_dir=context_dir,
        support_dir=support_dir,
        baseline_path=baseline_path,
        pure_field_dir=pure_field_dir,
        pure_endpoint_dir=pure_endpoint_dir,
    )
    if args.phase == "preflight":
        print(json.dumps(_jsonable(preflight_result), ensure_ascii=False, indent=2))
        return 0 if preflight_result["status"] == "ready" else 2
    if preflight_result["status"] != "ready":
        print(json.dumps(_jsonable(preflight_result), ensure_ascii=False, indent=2))
        return 2

    logical_cuda, physical_cuda = _select_cuda_device(int(args.cuda_device))
    device = torch.device("cuda", logical_cuda)
    import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core

    global_camera = json.loads((source_dir / "global_camera.json").read_text(encoding="utf-8"))
    global_mesh_cpu, global_query_mesh, global_meta = _load_mesh_query_object(baseline_path, device)
    global_points = global_meta.pop("points")
    global_coords = global_mesh_cpu.coords.detach().cpu().to(torch.int32)
    _atomic_torch(output_dir / "global_support" / "coords.pt", global_coords)
    _atomic_torch(output_dir / "global_support" / "points.pt", global_points)
    global_query = _query_mesh_chunked(global_query_mesh, global_points.to(device=device), int(args.query_chunk_size))
    _atomic_torch(output_dir / "global_support" / "G.pt", global_query)
    self_query = _metric(global_query, global_mesh_cpu.attrs.detach().cpu().to(torch.float32))
    global_meta["self_query"] = self_query
    global_meta["count"] = int(global_coords.shape[0])
    global_meta["artifact_path"] = global_meta["path"]
    global_meta["artifact_sha256"] = global_meta.pop("sha256")
    _atomic_json(output_dir / "global_support" / "meta.json", global_meta)

    q_global = global_points * (2.0 * float(global_camera["mesh_scale"]))
    tile_data: Dict[int, Dict[str, Any]] = {}
    pipeline_holder: Dict[str, Any] = {}
    tile_coverage_rows: List[Dict[str, Any]] = []
    roundtrip: Dict[str, Any] = {}
    query_correctness: Dict[str, Any] = {}
    selected_field_paths: Dict[int, Path] = {}
    for key, record in preflight_result["tiles"].items():
        tile_id = int(key)
        if record.get("selected_field"):
            selected_field_paths[tile_id] = Path(record["selected_field"])
    for tile_id in sorted(PHASE_A_TILE_IDS):
        tile_root = _tile_dir(source_dir, tile_id)
        support_coords = _normalise_coords(_payload_tensor(_support_path(support_dir, tile_id), keys=("tensor", "coords")))
        transform = core.TileCameraTransform(**json.loads((tile_root / "tile_camera.json").read_text(encoding="utf-8")))
        local_centers = -0.5 + (support_coords.to(torch.float32) + 0.5) / float(RESOLUTION)
        q_global_centers = _local_to_global(core, local_centers, global_camera, transform)
        global_centers = q_global_centers / (2.0 * float(global_camera["mesh_scale"]))
        reference_query = _query_mesh_chunked(global_query_mesh, global_centers.to(device=device), int(args.query_chunk_size))
        reference_cached = _payload_tensor(tile_root / "global_pbr_reference.pt").detach().cpu().to(torch.float32)
        if reference_cached.shape != reference_query.shape:
            raise ValueError(f"tile {tile_id}: global_pbr_reference shape does not match support")
        reference_consistency = _metric(reference_query, reference_cached)
        q_global_local, local_points, uv_tile = _global_to_local(core, global_points, global_camera, transform)
        roundtrip_q = _local_to_global(core, local_points, global_camera, transform)
        error = (roundtrip_q - q_global_local).to(torch.float64)
        roundtrip[tile_id] = {
            "max_abs": float(error.abs().max().item()),
            "mean_abs": float(error.abs().mean().item()),
            "relative_l2": float(torch.linalg.vector_norm(error).item() / (torch.linalg.vector_norm(q_global_local.to(torch.float64)).item() + 1e-12)),
        }
        Q, valid, q_meta = build_sparse_query_matrix(support_coords, local_points, RESOLUTION)
        if tile_id in selected_field_paths:
            local_field = _load_local_field(selected_field_paths[tile_id], support_coords)
            field_source = str(selected_field_paths[tile_id].resolve())
            field_kind = "cached_final_purehr_field"
        else:
            endpoint_path = Path(record["selected_endpoint"])
            local_field = _load_endpoint_field(
                endpoint_path,
                _tile_dir(context_dir, tile_id) / "fixed_shape_norm.pt",
                -0.5 + (support_coords.to(torch.float32) + 0.5) / float(RESOLUTION),
                device,
                str(args.model_path),
                int(args.query_chunk_size),
                pipeline_holder,
            )
            field_source = str(endpoint_path.resolve())
            field_kind = "decoded_final_purehr_endpoint"
        H = apply_query_matrix(Q, local_field)
        invalid = ~valid
        H[invalid] = float("nan")
        delta = H - global_query
        delta[invalid] = float("nan")
        observed_values = apply_query_matrix(Q, _source_mask(source_dir, tile_id, "observed_mask").to(torch.float32)[:, None]).reshape(-1)
        hidden_values = apply_query_matrix(Q, _source_mask(source_dir, tile_id, "hidden_mask").to(torch.float32)[:, None]).reshape(-1)
        eps = float(args.mask_epsilon)
        observed_global = valid & (observed_values >= 1.0 - eps)
        hidden_global = valid & (hidden_values >= 1.0 - eps)
        mixed_global = valid & ~(observed_global | hidden_global)
        tile_out = output_dir / "tiles" / f"tile_{tile_id:02d}"
        _atomic_torch(tile_out / "H_on_global.pt", H)
        _atomic_torch(tile_out / "Delta_on_global.pt", delta)
        _atomic_torch(tile_out / "valid_mask.pt", valid)
        _atomic_torch(tile_out / "observed_mask_global.pt", observed_global)
        _atomic_torch(tile_out / "hidden_mask_global.pt", hidden_global)
        _atomic_torch(tile_out / "mixed_mask_global.pt", mixed_global)
        query_record: Dict[str, Any] = {
            "tile_id": tile_id,
            "field_source": field_source,
            "field_kind": field_kind,
            "support_coords": str(_support_path(support_dir, tile_id).resolve()),
            "support_count": int(support_coords.shape[0]),
            "query": q_meta,
            "roundtrip": roundtrip[tile_id],
            "valid_count": int(valid.sum().item()),
            "invalid_count": int((~valid).sum().item()),
            "observed_count": int(observed_global.sum().item()),
            "hidden_count": int(hidden_global.sum().item()),
            "mixed_count": int(mixed_global.sum().item()),
            "global_uv_finite": int(torch.isfinite(uv_tile).all(dim=1).sum().item()),
            "global_reference_consistency": reference_consistency,
        }
        if tile_id in (26, 27):
            sample_valid = torch.where(valid)[0]
            generator = torch.Generator().manual_seed(42 + tile_id)
            if sample_valid.numel() > int(args.correctness_sample_count):
                selection = torch.randperm(sample_valid.numel(), generator=generator)[: int(args.correctness_sample_count)]
                sample_valid = sample_valid.index_select(0, selection)
            sample_local = local_points.index_select(0, sample_valid)
            from pixal3d.representations import MeshWithVoxel
            tile_query_mesh = MeshWithVoxel(
                vertices=torch.empty((1, 3), dtype=torch.float32, device=device),
                faces=torch.empty((0, 3), dtype=torch.int32, device=device),
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1.0 / RESOLUTION,
                coords=support_coords.to(device=device),
                attrs=local_field.to(device=device),
                voxel_shape=torch.Size([1, PBR_CHANNELS, RESOLUTION, RESOLUTION, RESOLUTION]),
                layout={"base_color": slice(0, 3), "metallic": slice(3, 4), "roughness": slice(4, 5), "alpha": slice(5, 6)},
            )
            q_values = apply_query_matrix(Q[sample_valid.numpy()], local_field)
            reference = _query_mesh_chunked(tile_query_mesh, sample_local.to(device=device), int(args.query_chunk_size))
            channel_metrics = {}
            for name, group in GROUPS.items():
                channel_metrics[name] = _metric(q_values[:, group], reference[:, group])
            query_correctness[str(tile_id)] = {
                "sample_count": int(sample_valid.numel()),
                "channel_metrics": channel_metrics,
                "query_operator": q_meta,
            }
            del tile_query_mesh
        _atomic_json(tile_out / "query_meta.json", query_record)
        tile_data[tile_id] = {
            "delta": delta,
            "valid": valid,
            "hidden": hidden_global,
            "observed": observed_global,
            "mixed": mixed_global,
            "H": H,
            "query_meta": query_record,
        }
        tile_coverage_rows.append({"tile_id": tile_id, "valid": int(valid.sum().item()), "hidden": int(hidden_global.sum().item()), "observed": int(observed_global.sum().item()), "mixed": int(mixed_global.sum().item())})
        del Q, local_points, local_field, H, delta, observed_values, hidden_values

    pairwise_gate: Dict[str, Any] = {}
    for left_id in sorted(PHASE_A_TILE_IDS):
        for right_id in sorted(PHASE_A_TILE_IDS):
            left = tile_data[left_id]
            right = tile_data[right_id]
            common_valid = left["valid"] & right["valid"]
            pairwise_gate[f"{left_id},{right_id}"] = {
                "tile_i": left_id,
                "tile_j": right_id,
                "common_valid_voxels": int(common_valid.sum().item()),
                "common_hidden_voxels": int((common_valid & left["hidden"] & right["hidden"]).sum().item()),
                "common_observed_voxels": int((common_valid & left["observed"] & right["observed"]).sum().item()),
            }
    gate = {
        "format": FORMAT,
        "status": "passed",
        "cuda": {"requested_physical": int(args.cuda_device), "logical": logical_cuda, "physical": physical_cuda, "name": torch.cuda.get_device_name(logical_cuda)},
        "global_support_count": int(global_coords.shape[0]),
        "global_self_query": self_query,
        "global_reference_consistency": {str(tile_id): tile_data[tile_id]["query_meta"]["global_reference_consistency"] for tile_id in sorted(PHASE_A_TILE_IDS)},
        "roundtrip": {str(k): v for k, v in roundtrip.items()},
        "tile26_tile27_query_correctness": query_correctness,
        "mask_counts": {
            str(tile_id): {
                "valid": int(tile_data[tile_id]["valid"].sum().item()),
                "invalid": int((~tile_data[tile_id]["valid"]).sum().item()),
                "hidden": int(tile_data[tile_id]["hidden"].sum().item()),
                "observed": int(tile_data[tile_id]["observed"].sum().item()),
                "mixed": int(tile_data[tile_id]["mixed"].sum().item()),
            }
            for tile_id in sorted(PHASE_A_TILE_IDS)
        },
        "pairwise_overlap_counts": pairwise_gate,
        "mask_epsilon": float(args.mask_epsilon),
        "invalid_rows_are_excluded": True,
        "zero_missing_sentinel": False,
        "missing_value_sentinel": "NaN in H_on_global.pt and Delta_on_global.pt; validity is carried by valid_mask.pt",
    }
    for tile_id in (26, 27):
        record = query_correctness.get(str(tile_id))
        if record is None or record["sample_count"] < min(10000, int(args.correctness_sample_count)):
            gate["status"] = "failed"
            gate.setdefault("failures", []).append(f"Tile{tile_id} has fewer than the required correctness samples")
        if record is not None:
            for group_name, metrics in record["channel_metrics"].items():
                if metrics["max_abs"] > float(args.correctness_tolerance) and metrics["relative_l2"] > float(args.correctness_relative_tolerance):
                    gate["status"] = "failed"
                    gate.setdefault("failures", []).append(f"Tile{tile_id} {group_name} query operator mismatch")
    _atomic_json(output_dir / "correctness_gate.json", gate)
    if gate["status"] != "passed":
        summary = {"status": "blocked_correctness_gate", "correctness_gate": gate}
        _atomic_json(output_dir / "summary.json", summary)
        _build_report(output_dir, summary, tile_coverage_rows, [], [], [])
        return 3

    coverage = torch.zeros(global_coords.shape[0], dtype=torch.int16)
    for tile_id in sorted(PHASE_A_TILE_IDS):
        coverage += tile_data[tile_id]["valid"].to(torch.int16)
    _atomic_torch(output_dir / "coverage" / "coverage_count.pt", coverage)
    unique, counts = torch.unique(coverage, return_counts=True)
    histogram = {str(int(k)): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
    _atomic_json(output_dir / "coverage" / "coverage_histogram.json", histogram)
    labels = [str(v) for v in sorted(PHASE_A_TILE_IDS)]
    valid_matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    hidden_matrix = np.zeros_like(valid_matrix)
    observed_matrix = np.zeros_like(valid_matrix)
    overlap_rows: List[Dict[str, Any]] = []
    for i, left_id in enumerate(sorted(PHASE_A_TILE_IDS)):
        for j, right_id in enumerate(sorted(PHASE_A_TILE_IDS)):
            left = tile_data[left_id]
            right = tile_data[right_id]
            valid_both = left["valid"] & right["valid"]
            hidden_both = valid_both & left["hidden"] & right["hidden"]
            observed_both = valid_both & left["observed"] & right["observed"]
            valid_matrix[i, j] = int(valid_both.sum().item())
            hidden_matrix[i, j] = int(hidden_both.sum().item())
            observed_matrix[i, j] = int(observed_both.sum().item())
            overlap_rows.append({"tile_i": left_id, "tile_j": right_id, "common_valid_voxels": int(valid_both.sum().item()), "common_hidden_voxels": int(hidden_both.sum().item()), "common_observed_voxels": int(observed_both.sum().item())})
    _write_csv(output_dir / "coverage" / "pairwise_overlap.csv", overlap_rows)
    _plot_heatmap(output_dir / "coverage" / "pairwise_overlap_all.png", valid_matrix, labels, "pairwise direct valid overlap")
    _plot_heatmap(output_dir / "coverage" / "pairwise_overlap_hidden.png", hidden_matrix, labels, "pairwise hidden overlap")
    _plot_heatmap(output_dir / "coverage" / "pairwise_overlap_observed.png", observed_matrix, labels, "pairwise observed overlap")
    _plot_support(output_dir / "coverage" / "coverage_count.png", global_points, coverage.to(torch.float32), "global support coverage")
    _plot_support(output_dir / "coverage" / "hidden_mask.png", global_points, torch.stack([tile_data[t]["hidden"] for t in sorted(PHASE_A_TILE_IDS)]).sum(0).to(torch.float32), "strict hidden coverage")
    _plot_support(output_dir / "coverage" / "observed_mask.png", global_points, torch.stack([tile_data[t]["observed"] for t in sorted(PHASE_A_TILE_IDS)]).sum(0).to(torch.float32), "strict observed coverage")
    quartet_summaries: List[Dict[str, Any]] = []
    pod_rows: List[Dict[str, Any]] = []
    cosine_rows: List[Dict[str, Any]] = []
    loto_rows: List[Dict[str, Any]] = []
    cosine_summaries: List[Dict[str, Any]] = []
    for quartet in QUARTETS:
        quartet_dir = output_dir / "quartets" / "_".join(map(str, quartet))
        base_valid = _common_mask({tile_id: tile_data[tile_id]["valid"] for tile_id in quartet}, quartet)
        hidden_common = base_valid & _common_mask({tile_id: tile_data[tile_id]["hidden"] for tile_id in quartet}, quartet)
        observed_common = base_valid & _common_mask({tile_id: tile_data[tile_id]["observed"] for tile_id in quartet}, quartet)
        quartet_record = {
            "quartet": list(quartet),
            "common_valid_count": int(base_valid.sum().item()),
            "common_hidden_count": int(hidden_common.sum().item()),
            "common_observed_count": int(observed_common.sum().item()),
            "mixed_count": int((base_valid & ~(hidden_common | observed_common)).sum().item()),
        }
        quartet_summaries.append(quartet_record)
        _atomic_torch(quartet_dir / "common_valid_mask.pt", base_valid)
        _atomic_torch(quartet_dir / "common_hidden_mask.pt", hidden_common)
        _atomic_torch(quartet_dir / "common_observed_mask.pt", observed_common)
        domains = {"ALL_VALID": base_valid, "ALL_HIDDEN": hidden_common, "ALL_OBSERVED": observed_common}
        for domain, domain_mask in domains.items():
            for group_name, group in GROUPS.items():
                pod = _pod_for_mask({tile_id: tile_data[tile_id]["delta"] for tile_id in quartet}, quartet, domain_mask, group, int(args.pod_chunk_size))
                _save_pod_outputs(quartet_dir, quartet, domain, group_name, pod)
                row = {"quartet": list(quartet), "domain": domain, "group": group_name, "sample_voxels": pod["sample_voxels"], "rho1": pod["rho1"], "rho12": pod["rho12"], "directional_rho1": pod["directional_rho1"], "directional_rho12": pod["directional_rho12"]}
                pod_rows.append(row)
                cosine = pod["cosine"]
                finite_cos = cosine[torch.isfinite(cosine) & ~torch.eye(len(quartet), dtype=torch.bool)]
                cosine_summaries.append({"quartet": list(quartet), "domain": domain, "group": group_name, "mean": float(finite_cos.mean().item()) if finite_cos.numel() else None, "min": float(finite_cos.min().item()) if finite_cos.numel() else None, "max": float(finite_cos.max().item()) if finite_cos.numel() else None})
                for i, left_id in enumerate(quartet):
                    for j, right_id in enumerate(quartet):
                        cosine_rows.append({"quartet": "_".join(map(str, quartet)), "domain": domain, "group": group_name, "tile_i": left_id, "tile_j": right_id, "cosine": float(cosine[i, j].item()) if torch.isfinite(cosine[i, j]) else None})
                full_v = pod["V"][:, 0] if pod["V"].numel() else torch.zeros(len(quartet), dtype=torch.float64)
                for omit in quartet:
                    remaining = tuple(tile_id for tile_id in quartet if tile_id != omit)
                    loto_mask = _common_mask({tile_id: tile_data[tile_id]["valid"] for tile_id in remaining}, remaining)
                    if domain == "ALL_HIDDEN":
                        loto_mask &= _common_mask({tile_id: tile_data[tile_id]["hidden"] for tile_id in remaining}, remaining)
                    elif domain == "ALL_OBSERVED":
                        loto_mask &= _common_mask({tile_id: tile_data[tile_id]["observed"] for tile_id in remaining}, remaining)
                    loto = _pod_for_mask({tile_id: tile_data[tile_id]["delta"] for tile_id in remaining}, remaining, loto_mask, group, int(args.pod_chunk_size))
                    restricted = torch.tensor([full_v[list(quartet).index(tile_id)].item() for tile_id in remaining], dtype=torch.float64)
                    if torch.linalg.vector_norm(restricted).item() > 1e-12 and loto["V"].numel():
                        restricted /= torch.linalg.vector_norm(restricted)
                        stability = abs(float(torch.dot(restricted, loto["V"][:, 0]).item()))
                    else:
                        stability = None
                    loto_rows.append({"quartet": list(quartet), "domain": domain, "group": group_name, "omitted_tile": omit, "full_rho1": pod["rho1"], "loto_rho1": loto["rho1"], "full_rho12": pod["rho12"], "loto_rho12": loto["rho12"], "s_loto": stability, "loto_sample_voxels": loto["sample_voxels"]})
    _write_csv(output_dir / "pod_metrics.csv", pod_rows)
    _write_csv(output_dir / "pairwise_cosine.csv", cosine_rows)
    _write_csv(output_dir / "loto_stability.csv", loto_rows)
    summary = {
        "status": "completed",
        "format": FORMAT,
        "selected_tile_ids": sorted(PHASE_A_TILE_IDS),
        "cuda": {"requested_physical": int(args.cuda_device), "logical": logical_cuda, "physical": physical_cuda, "name": torch.cuda.get_device_name(logical_cuda)},
        "preflight": str((output_dir / "preflight.json").resolve()),
        "correctness_gate": str((output_dir / "correctness_gate.json").resolve()),
        "global_support": {"count": int(global_coords.shape[0]), "artifact_path": global_meta["artifact_path"], "artifact_sha256": global_meta["artifact_sha256"], "self_query": self_query},
        "purehr_provenance": {
            str(tile_id): next(
                (
                    {
                        "path": candidate["path"],
                        "kind": candidate["kind"],
                        "sha256": candidate.get("sha256"),
                        "route_metadata": candidate.get("route_metadata", {}),
                    }
                    for candidate in preflight_result["tiles"][str(tile_id)]["field_candidates"] + preflight_result["tiles"][str(tile_id)]["endpoint_candidates"]
                    if candidate["path"] == (preflight_result["tiles"][str(tile_id)].get("selected_field") or preflight_result["tiles"][str(tile_id)].get("selected_endpoint"))
                ),
                {"path": preflight_result["tiles"][str(tile_id)].get("selected_field") or preflight_result["tiles"][str(tile_id)].get("selected_endpoint")},
            )
            for tile_id in sorted(PHASE_A_TILE_IDS)
        },
        "coverage": {"histogram": histogram, "pairwise_csv": str((output_dir / "coverage/pairwise_overlap.csv").resolve())},
        "tile_coverage": tile_coverage_rows,
        "quartets": quartet_summaries,
        "pod_metrics_csv": str((output_dir / "pod_metrics.csv").resolve()),
        "pairwise_cosine_csv": str((output_dir / "pairwise_cosine.csv").resolve()),
        "loto_stability_csv": str((output_dir / "loto_stability.csv").resolve()),
        "cosine_summary": cosine_summaries,
        "prohibited_operations": {"wavelet": False, "c256": False, "projector": False, "lsmr": False, "fusion": False, "flow": False, "reencode": False, "euler": False},
    }
    _atomic_json(output_dir / "summary.json", summary)
    _build_report(output_dir, summary, tile_coverage_rows, pod_rows, cosine_rows, loto_rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--source-dir", type=Path, default=Path("outputs/pbr_range_null_perstep_cuda4_full"))
    parser.add_argument("--context-dir", type=Path, default=Path("outputs/cross_tile_pbr_perstep_guided_cuda4_full"))
    parser.add_argument("--support-dir", type=Path, default=Path("outputs/pbr_shared_coarse_oracle_phase_a_v2/fields"))
    parser.add_argument("--baseline-path", type=Path, default=Path("outputs/visibility_guided_pbr_flow_cuda4_perstep_fixed095_all_full/global_baseline_mesh.pt"))
    parser.add_argument("--pure-field-dir", type=Path, default=Path("outputs/pbr_shared_coarse_oracle_phase_a_v2/fields"))
    parser.add_argument("--pure-endpoint-dir", type=Path, default=Path("outputs/purehr_completion_phase_a_cuda4"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/global_c1024_common_field_pod_phaseA_cuda4"))
    parser.add_argument("--model-path", type=str, default="/home/nvme04/yyyan/download/model/Pixal3D")
    parser.add_argument("--query-chunk-size", type=int, default=250000)
    parser.add_argument("--pod-chunk-size", type=int, default=250000)
    parser.add_argument("--correctness-sample-count", type=int, default=10000)
    parser.add_argument("--correctness-tolerance", type=float, default=5e-4)
    parser.add_argument("--correctness-relative-tolerance", type=float, default=5e-5)
    parser.add_argument("--mask-epsilon", type=float, default=1e-6)
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
