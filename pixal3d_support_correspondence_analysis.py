#!/usr/bin/env python3
"""Quantify global/local sparse-support correspondence for projective tiles.

This is a CPU-only post-processing utility.  It consumes checkpoints written by
``pixal3d_projective_tile_generation_eval.py --save-decoded-support`` and does
not run a model, densify a 1024^3 grid, or export PLY/GLB files.

Two coordinate conventions are intentionally kept separate:

* decoded C1024 O-Voxel material coordinates are voxel centers:
  ``q = 2 * (coord + 0.5) / 1024 - 1``;
* C32/C64 shape-SLat coordinates in ``support_debug.pt`` are endpoint lattices:
  ``q = 2 * coord / (resolution - 1) - 1``.

Every local C1024 center is mapped back with the exact
``_centered_tile_q_to_global_q`` implementation used by the generation script.
There is no clipping, bbox/centroid normalization, or coordinate compression.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import __version__ as scipy_version
from scipy.spatial import cKDTree

from pixal3d_projective_tile_generation_eval import (
    IMAGE_CANONICAL,
    TileCameraTransform,
    _centered_tile_q_to_global_q,
    _global_q_to_centered_tile_q,
)


SCRIPT_VERSION = "pixal3d_support_correspondence_v2"
C1024 = 1024
C64 = 64
KEY_BITS_C1024 = 10
DEFAULT_ROUTE_SUBDIR = (
    "native_c32_shape512_projected_plus_native_c64_shape1024_texture"
)
DEFAULT_SUPPORT_SUBDIR = "projective_camera_support_c32"


@dataclass
class TileState:
    tile_id: int
    category: str
    box: Tuple[int, int, int, int]
    transform: TileCameraTransform
    output_dir: Path
    local_keys: np.ndarray
    local_base_color: Optional[np.ndarray]
    local_uv_full: np.ndarray
    local_uv_tile: np.ndarray
    global_region_keys: np.ndarray
    global_region_uv_tile: np.ndarray
    matched_keys: np.ndarray
    global_only_keys: np.ndarray
    local_only_keys: np.ndarray
    distance_query_q_global: np.ndarray
    summary: Dict[str, Any]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_tile_ids(value: str) -> List[int]:
    try:
        tile_ids = [int(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError as exc:
        raise ValueError(f"invalid --tiles value {value!r}") from exc
    if not tile_ids:
        raise ValueError("--tiles must contain at least one tile id")
    if len(tile_ids) != len(set(tile_ids)):
        raise ValueError("--tiles must not contain duplicates")
    if any(tile_id < 0 for tile_id in tile_ids):
        raise ValueError("--tiles must contain non-negative ids")
    return tile_ids


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a dictionary checkpoint")
    return payload


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _required_paths(
    run_dir: Path,
    tile_ids: Sequence[int],
    *,
    route_subdir: str,
    support_subdir: str,
) -> Dict[str, Path]:
    paths: Dict[str, Path] = {
        "global_camera": run_dir / "global_camera.json",
        "global_decoded_support": (
            run_dir / "global_baseline_1024" / "decoded_support.pt"
        ),
    }
    for tile_id in tile_ids:
        tile_dir = run_dir / "tiles" / f"tile_{tile_id:02d}"
        prefix = f"tile_{tile_id:02d}"
        paths[f"{prefix}_camera"] = tile_dir / "tile_camera.json"
        paths[f"{prefix}_decoded_support"] = (
            tile_dir / route_subdir / "decoded_support.pt"
        )
        paths[f"{prefix}_support_debug"] = (
            tile_dir / support_subdir / "support_debug.pt"
        )
        paths[f"{prefix}_fusion_debug"] = (
            tile_dir / support_subdir / "tile_c64_fusion_debug.pt"
        )
    return paths


def _validate_required_paths(paths: Mapping[str, Path]) -> None:
    missing = [(name, path) for name, path in paths.items() if not path.is_file()]
    if not missing:
        return
    details = "\n".join(f"  - {name}: {path}" for name, path in missing)
    raise FileNotFoundError(
        "required support-analysis checkpoint(s) are missing:\n"
        f"{details}\n"
        "Generate them with pixal3d_projective_tile_generation_eval.py "
        "--save-decoded-support using the same tile ids."
    )


def _checkpoint_tensor(
    payload: Mapping[str, Any],
    name: str,
    *,
    columns: Optional[int] = None,
) -> torch.Tensor:
    value = payload.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"checkpoint field {name!r} must be a tensor")
    if columns is not None and (value.ndim != 2 or value.shape[1] != columns):
        raise ValueError(
            f"checkpoint field {name!r} must have shape [N,{columns}], "
            f"got {tuple(value.shape)}"
        )
    return value.detach().cpu().contiguous()


def _validate_decoded_support(
    payload: Mapping[str, Any],
    *,
    path: Path,
    require_faces: bool,
) -> None:
    if payload.get("format") != "pixal3d_decoded_support_v1":
        raise ValueError(
            f"{path}: unsupported format {payload.get('format')!r}; "
            "expected 'pixal3d_decoded_support_v1'"
        )
    vertices = _checkpoint_tensor(payload, "vertices", columns=3)
    coords = _checkpoint_tensor(payload, "ovoxel_coords_c1024", columns=3)
    attrs = _checkpoint_tensor(payload, "ovoxel_attrs")
    if attrs.ndim != 2 or attrs.shape[0] != coords.shape[0]:
        raise ValueError(
            f"{path}: ov voxel attrs must be [N,C] and align with coordinates"
        )
    if require_faces:
        faces = _checkpoint_tensor(payload, "faces", columns=3)
        if faces.numel() and (
            int(faces.min().item()) < 0
            or int(faces.max().item()) >= int(vertices.shape[0])
        ):
            raise ValueError(f"{path}: faces reference invalid vertex indices")
    voxel_shape = tuple(int(value) for value in payload.get("voxel_shape", ()))
    if len(voxel_shape) < 3:
        raise ValueError(
            f"{path}: voxel_shape must include three spatial axes, got "
            f"{voxel_shape}"
        )
    spatial_shape = voxel_shape[-3:]
    if any(value < 1 or value > C1024 for value in spatial_shape):
        raise ValueError(
            f"{path}: decoded sparse spatial shape exceeds C1024: "
            f"{spatial_shape}"
        )
    voxel_size = float(payload.get("voxel_size", float("nan")))
    if not math.isclose(
        voxel_size,
        1.0 / C1024,
        rel_tol=1e-7,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{path}: expected C1024 voxel_size={1.0 / C1024}, got "
            f"{voxel_size}"
        )
    if coords.numel() and (
        int(coords.min().item()) < 0 or int(coords.max().item()) >= C1024
    ):
        raise ValueError(f"{path}: C1024 O-Voxel coordinates are out of range")
    if coords.numel():
        spatial_limit = torch.tensor(spatial_shape, dtype=coords.dtype)[None]
        if bool((coords >= spatial_limit).any().item()):
            raise ValueError(
                f"{path}: O-Voxel coordinates exceed saved sparse spatial shape "
                f"{spatial_shape}"
            )


def _layout_slice(
    payload: Mapping[str, Any],
    name: str,
) -> Optional[slice]:
    layout = payload.get("layout")
    if not isinstance(layout, Mapping):
        return None
    spec = layout.get(name)
    if not isinstance(spec, Mapping):
        return None
    start = spec.get("start")
    stop = spec.get("stop")
    step = spec.get("step")
    return slice(
        None if start is None else int(start),
        None if stop is None else int(stop),
        None if step is None else int(step),
    )


def _extract_base_color(
    payload: Mapping[str, Any],
    attrs: torch.Tensor,
) -> Optional[np.ndarray]:
    attr_slice = _layout_slice(payload, "base_color")
    if attr_slice is None:
        return None
    values = attrs[:, attr_slice]
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(
            "base_color layout must select exactly three attribute channels"
        )
    return values.to(torch.float32).numpy()


def _voxel_centers_to_q(
    coords: torch.Tensor,
    *,
    resolution: int = C1024,
) -> torch.Tensor:
    """Convert material voxel indices to center q; never endpoint q."""
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("voxel coordinates must be [N,3]")
    return (
        2.0 * (coords.to(torch.float32) + 0.5) / float(resolution) - 1.0
    )


def _q_to_center_voxel_indices_no_clip(
    q: torch.Tensor,
    *,
    resolution: int = C1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize q to the nearest voxel center, dropping invalid rows.

    For centers ``q_i = 2*(i+0.5)/R-1``, the continuous center index is
    ``(q+1)*R/2-0.5``.  PyTorch ``round`` selects the nearest center and uses
    round-half-to-even at an exact half-grid tie.  No clamp is permitted.
    """
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q must be [N,3]")
    continuous_center_index = (
        (q + 1.0) * (float(resolution) / 2.0) - 0.5
    )
    indices = torch.round(continuous_center_index).to(torch.int64)
    finite = torch.isfinite(q).all(dim=1)
    valid = finite & ((indices >= 0) & (indices < int(resolution))).all(dim=1)
    return indices, valid


def _q_to_endpoint_indices_no_clip(
    q: torch.Tensor,
    *,
    resolution: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize shape-SLat q to an endpoint lattice without clipping."""
    if resolution <= 1:
        raise ValueError("endpoint resolution must exceed one")
    indices = torch.round(
        (q.to(torch.float64) + 1.0) * (float(resolution - 1) / 2.0)
    ).to(torch.int64)
    finite = torch.isfinite(q).all(dim=1)
    inside = (q.abs() <= 1.0).all(dim=1)
    valid = (
        finite
        & inside
        & ((indices >= 0) & (indices < int(resolution))).all(dim=1)
    )
    return indices, valid


def _pack_coords(coords: np.ndarray, *, bits: int = KEY_BITS_C1024) -> np.ndarray:
    values = np.asarray(coords)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"coordinates must be [N,3], got {values.shape}")
    maximum = (1 << bits) - 1
    if values.size and (values.min() < 0 or values.max() > maximum):
        raise ValueError(f"coordinates exceed the {bits}-bit key domain")
    values64 = values.astype(np.int64, copy=False)
    return (
        (values64[:, 0] << (2 * bits))
        | (values64[:, 1] << bits)
        | values64[:, 2]
    )


def _unpack_keys(keys: np.ndarray, *, bits: int = KEY_BITS_C1024) -> np.ndarray:
    values = np.asarray(keys, dtype=np.int64)
    mask = (1 << bits) - 1
    return np.stack(
        [
            (values >> (2 * bits)) & mask,
            (values >> bits) & mask,
            values & mask,
        ],
        axis=1,
    )


def _unique_with_payload(
    keys: np.ndarray,
    *,
    attrs: Optional[np.ndarray] = None,
    uv: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """Sort/deduplicate packed keys and aggregate aligned payloads.

    Attributes are averaged across collisions.  UV uses the first source row in
    stable key order and is diagnostic only.
    """
    keys = np.asarray(keys, dtype=np.int64)
    if keys.ndim != 1:
        raise ValueError("packed keys must be one dimensional")
    if attrs is not None and attrs.shape[0] != keys.shape[0]:
        raise ValueError("attrs and keys must have equal row counts")
    if uv is not None and uv.shape[0] != keys.shape[0]:
        raise ValueError("uv and keys must have equal row counts")
    if keys.size == 0:
        empty_counts = np.empty((0,), dtype=np.int64)
        empty_attrs = (
            None
            if attrs is None
            else np.empty((0, attrs.shape[1]), dtype=np.float32)
        )
        empty_uv = (
            None if uv is None else np.empty((0, uv.shape[1]), dtype=np.float32)
        )
        return keys.copy(), empty_attrs, empty_uv, empty_counts

    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    unique_keys = sorted_keys[starts]
    ends = np.r_[starts[1:], sorted_keys.shape[0]]
    counts = (ends - starts).astype(np.int64, copy=False)

    unique_attrs: Optional[np.ndarray] = None
    if attrs is not None:
        sorted_attrs = np.asarray(attrs, dtype=np.float32)[order]
        sums = np.add.reduceat(sorted_attrs, starts, axis=0)
        unique_attrs = (sums / counts[:, None]).astype(np.float32, copy=False)

    unique_uv: Optional[np.ndarray] = None
    if uv is not None:
        unique_uv = np.asarray(uv, dtype=np.float32)[order[starts]]

    return unique_keys, unique_attrs, unique_uv, counts


def _multiplicity_summary(counts: np.ndarray) -> Dict[str, Any]:
    counts = np.asarray(counts, dtype=np.int64)
    if counts.size == 0:
        return {
            "source_rows": 0,
            "unique_tokens": 0,
            "collision_rows": 0,
            "collision_fraction": 0.0,
            "keys_with_collision": 0,
            "max_multiplicity": 0,
        }
    source_rows = int(counts.sum())
    unique_tokens = int(counts.size)
    collision_rows = source_rows - unique_tokens
    return {
        "source_rows": source_rows,
        "unique_tokens": unique_tokens,
        "collision_rows": int(collision_rows),
        "collision_fraction": float(collision_rows / max(source_rows, 1)),
        "keys_with_collision": int((counts > 1).sum()),
        "max_multiplicity": int(counts.max()),
    }


def _tile_category(box: Sequence[int]) -> str:
    x0, y0, x1, y1 = (int(value) for value in box)
    if x0 == 0 or y0 == 0 or x1 == IMAGE_CANONICAL or y1 == IMAGE_CANONICAL:
        return "edge"
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    half_stride = 256.0
    if (
        abs(center_x - IMAGE_CANONICAL / 2.0) <= half_stride
        and abs(center_y - IMAGE_CANONICAL / 2.0) <= half_stride
    ):
        return "center"
    return "mid"


def _tile_margin_band_counts(
    state_keys: np.ndarray,
    state_uv: np.ndarray,
    *,
    matched_keys: np.ndarray,
    only_keys: np.ndarray,
    width: int,
    height: int,
    only_name: str,
) -> Dict[str, Any]:
    """Count exact support classes in center/transition/edge image bands."""
    uv = np.asarray(state_uv, dtype=np.float64)
    if uv.shape != (state_keys.shape[0], 2):
        raise ValueError("tile-margin UV rows must align with support keys")
    margin = np.minimum.reduce(
        [
            uv[:, 0],
            uv[:, 1],
            float(width) - uv[:, 0],
            float(height) - uv[:, 1],
        ]
    )
    finite = np.isfinite(uv).all(axis=1)
    inside = (
        finite
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < float(width))
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < float(height))
    )
    edge_threshold = min(width, height) * 0.125
    center_threshold = min(width, height) * 0.25
    labels = {
        "edge_lt_128px": inside & (margin < edge_threshold),
        "transition_128_to_256px": (
            inside
            & (margin >= edge_threshold)
            & (margin < center_threshold)
        ),
        "center_ge_256px": inside & (margin >= center_threshold),
        "outside_image": ~inside,
    }
    matched = np.isin(state_keys, matched_keys, assume_unique=True)
    only = np.isin(state_keys, only_keys, assume_unique=True)
    rows: Dict[str, Any] = {}
    for label, mask in labels.items():
        rows[label] = {
            "all_tokens": int(mask.sum()),
            "matched_tokens": int((mask & matched).sum()),
            f"{only_name}_tokens": int((mask & only).sum()),
        }
    return {
        "thresholds": {
            "edge_lt_pixels": float(edge_threshold),
            "center_ge_pixels": float(center_threshold),
        },
        "counts": rows,
        "representative_uv_policy": (
            "one representative projected center per unique quantized C1024 key"
        ),
    }


def _load_transform(path: Path) -> TileCameraTransform:
    payload = _read_json(path)
    if "box" in payload:
        payload["box"] = tuple(int(value) for value in payload["box"])
    try:
        return TileCameraTransform(**payload)
    except TypeError as exc:
        raise ValueError(f"{path}: invalid TileCameraTransform fields: {exc}") from exc


def _assert_saved_transform_matches(
    checkpoint: Mapping[str, Any],
    transform: TileCameraTransform,
    *,
    path: Path,
) -> None:
    saved = checkpoint.get("tile_camera")
    if not isinstance(saved, Mapping):
        raise ValueError(f"{path}: missing tile_camera metadata")
    expected = asdict(transform)
    for name, expected_value in expected.items():
        actual = saved.get(name)
        if isinstance(expected_value, (tuple, list)):
            if tuple(actual) != tuple(expected_value):
                raise ValueError(f"{path}: tile_camera.{name} mismatch")
        elif isinstance(expected_value, float):
            if actual is None or not math.isclose(
                float(actual), expected_value, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError(f"{path}: tile_camera.{name} mismatch")
        elif actual != expected_value:
            raise ValueError(f"{path}: tile_camera.{name} mismatch")


def _inverse_local_support(
    coords: torch.Tensor,
    *,
    transform: TileCameraTransform,
    global_camera: Mapping[str, float],
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Map every local O-Voxel center to global q with the official inverse."""
    count = int(coords.shape[0])
    mapped_indices = np.empty((count, 3), dtype=np.int64)
    q_global_all = np.empty((count, 3), dtype=np.float32)
    uv_full_all = np.empty((count, 2), dtype=np.float32)
    valid_all = np.zeros((count,), dtype=bool)
    max_q_error = 0.0
    max_pixel_error = 0.0
    max_depth_error = 0.0

    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        q_local = _voxel_centers_to_q(coords[start:end], resolution=C1024)
        q_global, _, uv_full, stats = _centered_tile_q_to_global_q(
            q_local,
            global_camera=global_camera,
            transform=transform,
            validate_roundtrip=True,
        )
        indices, valid = _q_to_center_voxel_indices_no_clip(
            q_global,
            resolution=C1024,
        )
        mapped_indices[start:end] = indices.numpy()
        q_global_all[start:end] = q_global.numpy()
        uv_full_all[start:end] = uv_full.numpy()
        valid_all[start:end] = valid.numpy()
        max_q_error = max(max_q_error, float(stats["q_roundtrip_max_abs"]))
        max_pixel_error = max(
            max_pixel_error, float(stats["pixel_roundtrip_max"])
        )
        max_depth_error = max(
            max_depth_error, float(stats["normalized_depth_q_error_max"])
        )

    stats = {
        "input_local_c1024_rows": count,
        "global_lattice_valid_rows": int(valid_all.sum()),
        "global_lattice_invalid_rows_dropped": int((~valid_all).sum()),
        "global_lattice_invalid_fraction": float((~valid_all).mean())
        if count
        else 0.0,
        "q_roundtrip_max_abs": max_q_error,
        "pixel_roundtrip_max": max_pixel_error,
        "normalized_depth_q_error_max": max_depth_error,
        "quantization": (
            "round((q_global + 1) * 1024 / 2 - 0.5), PyTorch "
            "round-half-to-even for exact ties, no clamp"
        ),
    }
    return mapped_indices, q_global_all, uv_full_all, {**stats, "valid": valid_all}


def _forward_global_region(
    global_coords: torch.Tensor,
    global_keys_raw: np.ndarray,
    *,
    transform: TileCameraTransform,
    global_camera: Mapping[str, float],
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Select decoded global centers lying in the tile's local canonical cube."""
    region_keys_parts: List[np.ndarray] = []
    region_uv_parts: List[np.ndarray] = []
    c64_keys_parts: List[np.ndarray] = []
    input_rows = int(global_coords.shape[0])
    finite_rows = 0

    for start in range(0, input_rows, chunk_size):
        end = min(start + chunk_size, input_rows)
        q_global = _voxel_centers_to_q(
            global_coords[start:end],
            resolution=C1024,
        )
        q_local, uv_tile, _, _ = _global_q_to_centered_tile_q(
            q_global,
            global_camera=global_camera,
            transform=transform,
        )
        finite = torch.isfinite(q_local).all(dim=1) & torch.isfinite(uv_tile).all(
            dim=1
        )
        inside = finite & (q_local.abs() <= 1.0).all(dim=1)
        finite_rows += int(finite.sum().item())
        if not bool(inside.any().item()):
            continue
        inside_np = inside.numpy()
        region_keys_parts.append(global_keys_raw[start:end][inside_np])
        region_uv_parts.append(uv_tile[inside].numpy())
        c64_indices, c64_valid = _q_to_endpoint_indices_no_clip(
            q_local[inside],
            resolution=C64,
        )
        if not bool(c64_valid.all().item()):
            raise RuntimeError(
                "inside local canonical rows unexpectedly failed C64 quantization"
            )
        c64_keys_parts.append(
            _pack_coords(c64_indices.numpy(), bits=6)
        )

    region_keys_raw = (
        np.concatenate(region_keys_parts)
        if region_keys_parts
        else np.empty((0,), dtype=np.int64)
    )
    region_uv_raw = (
        np.concatenate(region_uv_parts, axis=0)
        if region_uv_parts
        else np.empty((0, 2), dtype=np.float32)
    )
    c64_keys_raw = (
        np.concatenate(c64_keys_parts)
        if c64_keys_parts
        else np.empty((0,), dtype=np.int64)
    )
    region_keys, _, region_uv, region_counts = _unique_with_payload(
        region_keys_raw,
        uv=region_uv_raw,
    )
    _, _, _, c64_counts = _unique_with_payload(c64_keys_raw)
    if region_uv is None:
        raise AssertionError("region UV aggregation unexpectedly returned None")
    stats = {
        "global_decoded_c1024_rows": input_rows,
        "finite_camera_rows": finite_rows,
        "global_centers_inside_local_canonical_cube": int(region_keys_raw.size),
        "global_region_unique_c1024_tokens": int(region_keys.size),
        "global_region_c1024_duplicate_summary": _multiplicity_summary(
            region_counts
        ),
        "global_to_local_c64_many_to_one": _multiplicity_summary(c64_counts),
        "region_definition": (
            "decoded global O-Voxel centers transformed with the official "
            "global->centered-tile camera mapping and |q_local|<=1; no bbox "
            "normalization, clipping, or image-visibility filter"
        ),
        "c64_quantization": (
            "endpoint lattice round((q_local+1)*(64-1)/2), no clamp"
        ),
    }
    return region_keys, region_uv, stats


def _load_projected_support_stats(
    support_path: Path,
    fusion_path: Path,
    *,
    transform: TileCameraTransform,
) -> Dict[str, Any]:
    support = _torch_load(support_path)
    fusion = _torch_load(fusion_path)
    _assert_saved_transform_matches(support, transform, path=support_path)
    _assert_saved_transform_matches(fusion, transform, path=fusion_path)

    selected = _checkpoint_tensor(support, "selected_global_c128", columns=4)
    q_local = _checkpoint_tensor(support, "q_local", columns=3)
    kept = support.get("kept_mask")
    if not isinstance(kept, torch.Tensor) or kept.ndim != 1:
        raise ValueError(f"{support_path}: kept_mask must be a one-dimensional tensor")
    kept = kept.detach().cpu().to(torch.bool)
    if kept.shape[0] != q_local.shape[0]:
        raise ValueError(f"{support_path}: kept_mask and q_local row mismatch")

    coords1024 = _checkpoint_tensor(
        support, "coords1024_local_unique", columns=4
    )
    coords32 = _checkpoint_tensor(support, "coords32_unique", columns=4)
    coords64 = _checkpoint_tensor(
        support, "coords64_projected_unique", columns=4
    )
    projected64 = _checkpoint_tensor(
        fusion, "projected_coords64_unique", columns=4
    )
    native64 = _checkpoint_tensor(
        fusion, "native_coords64_unique", columns=4
    )
    fused64 = _checkpoint_tensor(fusion, "fused_coords64_unique", columns=4)

    kept_rows = int(kept.sum().item())
    projected_c64_unique = int(coords64.shape[0])
    if int(selected.shape[0]) != int(q_local.shape[0]):
        raise ValueError(
            f"{support_path}: selected_global_c128 and q_local row mismatch"
        )
    coords64_keys = np.sort(
        _pack_coords(coords64[:, 1:4].numpy(), bits=6)
    )
    projected64_keys = np.sort(
        _pack_coords(projected64[:, 1:4].numpy(), bits=6)
    )
    if not np.array_equal(coords64_keys, projected64_keys):
        raise ValueError(
            f"{fusion_path}: projected C64 set differs from support_debug"
        )

    projected_native_overlap = (
        int(projected64.shape[0])
        + int(native64.shape[0])
        - int(fused64.shape[0])
    )
    return {
        "selected_global_c1024_endpoint_rows": int(selected.shape[0]),
        "projected_rows_kept_in_local_canonical_cube": kept_rows,
        "projected_local_c1024_unique_tokens": int(coords1024.shape[0]),
        "projected_c32_unique_tokens": int(coords32.shape[0]),
        "projected_c64_unique_tokens": projected_c64_unique,
        "projected_c1024_to_c64_many_to_one": {
            "source_rows": kept_rows,
            "unique_tokens": projected_c64_unique,
            "collision_rows": kept_rows - projected_c64_unique,
            "collision_fraction": float(
                (kept_rows - projected_c64_unique) / max(kept_rows, 1)
            ),
        },
        "learned_native_c64_unique_tokens": int(native64.shape[0]),
        "projected_native_c64_exact_overlap_tokens": projected_native_overlap,
        "fused_c64_unique_tokens": int(fused64.shape[0]),
        "source_coordinate_convention": (
            "global one-step learned C1024 support uses endpoint q; projected "
            "C32/C64 coordinates also use endpoint lattices"
        ),
    }


def _keys_membership_indices(
    sorted_reference: np.ndarray,
    sorted_query: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return query membership and reference indices for sorted unique keys."""
    if sorted_query.size == 0:
        return np.empty((0,), dtype=bool), np.empty((0,), dtype=np.int64)
    indices = np.searchsorted(sorted_reference, sorted_query)
    inside = indices < sorted_reference.size
    matched = np.zeros(sorted_query.shape, dtype=bool)
    matched[inside] = sorted_reference[indices[inside]] == sorted_query[inside]
    return matched, indices


def _material_difference_summary(
    first: Optional[np.ndarray],
    second: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:
    if first is None or second is None:
        return None
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 3:
        raise ValueError("base-color arrays must be aligned [N,3]")
    if first.shape[0] == 0:
        return {
            "matched_tokens": 0,
            "mean_abs": None,
            "rmse": None,
            "psnr_db_data_range_1": None,
            "l2_p50": None,
            "l2_p95": None,
            "per_channel_mean_abs": [None, None, None],
        }
    delta = first.astype(np.float64) - second.astype(np.float64)
    abs_delta = np.abs(delta)
    mse = float(np.mean(delta * delta))
    l2 = np.linalg.norm(delta, axis=1)
    return {
        "matched_tokens": int(first.shape[0]),
        "mean_abs": float(abs_delta.mean()),
        "rmse": float(math.sqrt(mse)),
        "psnr_db_data_range_1": (
            None if mse == 0.0 else float(-10.0 * math.log10(mse))
        ),
        "identical": bool(mse == 0.0),
        "l2_p50": float(np.quantile(l2, 0.50)),
        "l2_p95": float(np.quantile(l2, 0.95)),
        "per_channel_mean_abs": [
            float(value) for value in abs_delta.mean(axis=0)
        ],
    }


def _distance_summary(distances: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(distances, dtype=np.float64)
    if values.size == 0:
        return {
            "sample_size": 0,
            "mean": None,
            "rmse": None,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    quantiles = np.quantile(values, [0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "sample_size": int(values.size),
        "mean": float(values.mean()),
        "rmse": float(math.sqrt(float(np.mean(values * values)))),
        "min": float(values.min()),
        "p25": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p75": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "max": float(values.max()),
    }


def _save_distance_histogram(
    values: np.ndarray,
    path: Path,
    *,
    title: str,
    xlabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5), dpi=150)
    values = np.asarray(values, dtype=np.float64)
    if values.size:
        axis.hist(values, bins=80, color="#4472C4", alpha=0.9)
        p50, p95 = np.quantile(values, [0.50, 0.95])
        axis.axvline(p50, color="#00A651", linestyle="--", label=f"p50={p50:.6g}")
        axis.axvline(p95, color="#D62728", linestyle="--", label=f"p95={p95:.6g}")
        axis.legend()
    else:
        axis.text(0.5, 0.5, "no valid samples", ha="center", va="center")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("count")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _subsample_rows(
    values: np.ndarray,
    *,
    maximum: int,
) -> np.ndarray:
    if values.shape[0] <= maximum:
        return values
    indices = np.linspace(0, values.shape[0] - 1, maximum).round().astype(np.int64)
    return values[indices]


def _draw_overlay_points(
    image: Image.Image,
    uv: np.ndarray,
    *,
    color: Tuple[int, int, int],
    maximum: int,
) -> int:
    points = _subsample_rows(np.asarray(uv), maximum=maximum)
    if points.size == 0:
        return 0
    pixels = np.rint(points).astype(np.int64)
    valid = (
        np.isfinite(points).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < image.width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < image.height)
    )
    pixels = pixels[valid]
    draw = ImageDraw.Draw(image)
    for x, y in pixels:
        draw.point((int(x), int(y)), fill=color)
        if image.width <= 1536:
            draw.point((int(x) + 1, int(y)), fill=color)
            draw.point((int(x), int(y) + 1), fill=color)
    return int(pixels.shape[0])


def _save_correspondence_overlay(
    state: TileState,
    reference_path: Path,
    *,
    max_points: int,
) -> Dict[str, Any]:
    if reference_path.is_file():
        image = Image.open(reference_path).convert("RGB")
    else:
        image = Image.new(
            "RGB",
            (state.transform.output_width, state.transform.output_height),
            (24, 24, 24),
        )

    local_matched, local_matched_indices = _keys_membership_indices(
        state.local_keys, state.matched_keys
    )
    if not bool(local_matched.all()):
        raise AssertionError("matched keys were not found in local keys")
    local_only_matched, local_only_indices = _keys_membership_indices(
        state.local_keys, state.local_only_keys
    )
    if not bool(local_only_matched.all()):
        raise AssertionError("local-only keys were not found in local keys")
    global_only_matched, global_only_indices = _keys_membership_indices(
        state.global_region_keys, state.global_only_keys
    )
    if not bool(global_only_matched.all()):
        raise AssertionError("global-only keys were not found in region keys")

    shown_global_only = _draw_overlay_points(
        image,
        state.global_region_uv_tile[global_only_indices],
        color=(45, 130, 255),
        maximum=max_points,
    )
    shown_local_only = _draw_overlay_points(
        image,
        state.local_uv_tile[local_only_indices],
        color=(255, 60, 55),
        maximum=max_points,
    )
    shown_matched = _draw_overlay_points(
        image,
        state.local_uv_tile[local_matched_indices],
        color=(40, 230, 85),
        maximum=max_points,
    )

    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 54), fill=(0, 0, 0))
    draw.text(
        (10, 7),
        (
            "green=matched  blue=global-only  red=local-only\n"
            f"tile {state.tile_id:02d} ({state.category}); "
            f"displayed {shown_matched}/{shown_global_only}/{shown_local_only}"
        ),
        fill=(255, 255, 255),
    )
    output_path = state.output_dir / "matched_global_only_local_only_overlay.png"
    image.save(output_path)
    return {
        "path": str(output_path),
        "displayed_matched_points": shown_matched,
        "displayed_global_only_points": shown_global_only,
        "displayed_local_only_points": shown_local_only,
        "maximum_points_per_class_before_visibility_filter": int(max_points),
    }


def _save_correspondence_keys(state: TileState) -> Path:
    path = state.output_dir / "correspondence_keys.npz"
    np.savez_compressed(
        path,
        key_encoding=np.asarray(
            ["x<<20 | y<<10 | z; each coordinate is uint10"], dtype="U64"
        ),
        local_mapped_unique_keys_int64=state.local_keys,
        global_region_unique_keys_int64=state.global_region_keys,
        matched_keys_int64=state.matched_keys,
        global_only_keys_int64=state.global_only_keys,
        local_only_keys_int64=state.local_only_keys,
        matched_coords_c1024_uint16=_unpack_keys(state.matched_keys).astype(
            np.uint16
        ),
        global_only_coords_c1024_uint16=_unpack_keys(
            state.global_only_keys
        ).astype(np.uint16),
        local_only_coords_c1024_uint16=_unpack_keys(
            state.local_only_keys
        ).astype(np.uint16),
    )
    return path


def _sample_unique_q(
    q_valid: np.ndarray,
    representative_indices: np.ndarray,
    *,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    q_unique = q_valid[representative_indices]
    if q_unique.shape[0] <= sample_size:
        return q_unique.astype(np.float32, copy=True)
    rng = np.random.default_rng(seed)
    selected = rng.choice(q_unique.shape[0], size=sample_size, replace=False)
    selected.sort()
    return q_unique[selected].astype(np.float32, copy=True)


def _analyze_tile(
    *,
    tile_id: int,
    run_dir: Path,
    output_root: Path,
    route_subdir: str,
    support_subdir: str,
    global_camera: Mapping[str, float],
    global_coords: torch.Tensor,
    global_keys_raw: np.ndarray,
    global_keys: np.ndarray,
    global_base_color: Optional[np.ndarray],
    chunk_size: int,
    distance_sample_size: int,
    seed: int,
    q_tolerance: float,
    pixel_tolerance: float,
    overlay_max_points: int,
    write_overlays: bool,
) -> TileState:
    tile_dir = run_dir / "tiles" / f"tile_{tile_id:02d}"
    route_dir = tile_dir / route_subdir
    support_dir = tile_dir / support_subdir
    output_dir = output_root / f"tile_{tile_id:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = _load_transform(tile_dir / "tile_camera.json")
    category = _tile_category(transform.box)

    local_path = route_dir / "decoded_support.pt"
    local_payload = _torch_load(local_path)
    _validate_decoded_support(
        local_payload,
        path=local_path,
        require_faces=False,
    )
    local_coords = _checkpoint_tensor(
        local_payload, "ovoxel_coords_c1024", columns=3
    )
    local_attrs = _checkpoint_tensor(local_payload, "ovoxel_attrs")
    local_base_raw = _extract_base_color(local_payload, local_attrs)
    local_vertex_count = int(
        _checkpoint_tensor(local_payload, "vertices", columns=3).shape[0]
    )
    del local_attrs

    mapped_indices, q_global, uv_full, inverse_stats = _inverse_local_support(
        local_coords,
        transform=transform,
        global_camera=global_camera,
        chunk_size=chunk_size,
    )
    valid = inverse_stats.pop("valid")
    if float(inverse_stats["q_roundtrip_max_abs"]) > q_tolerance:
        raise RuntimeError(
            f"tile {tile_id:02d}: q roundtrip error "
            f"{inverse_stats['q_roundtrip_max_abs']:.6g} exceeds "
            f"--roundtrip-q-tolerance={q_tolerance:.6g}"
        )
    if float(inverse_stats["pixel_roundtrip_max"]) > pixel_tolerance:
        raise RuntimeError(
            f"tile {tile_id:02d}: pixel roundtrip error "
            f"{inverse_stats['pixel_roundtrip_max']:.6g} exceeds "
            f"--roundtrip-pixel-tolerance={pixel_tolerance:.6g}"
        )

    valid_indices = mapped_indices[valid]
    valid_keys_raw = _pack_coords(valid_indices)
    valid_base = None if local_base_raw is None else local_base_raw[valid]
    valid_uv_full = uv_full[valid]
    (
        local_keys,
        local_base,
        local_uv_full,
        local_counts,
    ) = _unique_with_payload(
        valid_keys_raw,
        attrs=valid_base,
        uv=valid_uv_full,
    )
    if local_uv_full is None:
        raise AssertionError("local UV aggregation unexpectedly returned None")
    order = np.argsort(valid_keys_raw, kind="stable")
    sorted_keys = valid_keys_raw[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    representative_valid_indices = order[starts]
    distance_query = _sample_unique_q(
        q_global[valid],
        representative_valid_indices,
        sample_size=distance_sample_size,
        seed=seed + tile_id * 1009,
    )
    x0, y0, _, _ = transform.box
    local_uv_tile = np.stack(
        [
            (local_uv_full[:, 0] - float(x0))
            * float(transform.crop_to_output_scale_x),
            (local_uv_full[:, 1] - float(y0))
            * float(transform.crop_to_output_scale_y),
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    global_region_keys, global_region_uv, global_region_stats = (
        _forward_global_region(
            global_coords,
            global_keys_raw,
            transform=transform,
            global_camera=global_camera,
            chunk_size=chunk_size,
        )
    )
    # "Strict match" means exact equality against the full decoded global
    # material support.  The forward-projected global region is used only to
    # define which missing global tokens are relevant to this tile.  A handful
    # of quantized local keys can match full-global keys whose *center* lies just
    # outside the forward region; retain those exact matches and report them.
    matched = np.intersect1d(global_keys, local_keys, assume_unique=True)
    region_matched = np.intersect1d(
        global_region_keys, local_keys, assume_unique=True
    )
    global_only = np.setdiff1d(
        global_region_keys, local_keys, assume_unique=True
    )
    local_only = np.setdiff1d(local_keys, global_keys, assume_unique=True)
    local_full_membership, _ = _keys_membership_indices(global_keys, local_keys)

    projected_stats = _load_projected_support_stats(
        support_dir / "support_debug.pt",
        support_dir / "tile_c64_fusion_debug.pt",
        transform=transform,
    )

    global_material_agreement: Optional[Dict[str, Any]] = None
    if (
        local_base is not None
        and global_base_color is not None
        and matched.size
    ):
        local_found, local_indices = _keys_membership_indices(local_keys, matched)
        global_found, global_indices = _keys_membership_indices(global_keys, matched)
        if not bool(local_found.all() and global_found.all()):
            raise AssertionError("matched material keys failed exact lookup")
        global_material_agreement = _material_difference_summary(
            local_base[local_indices],
            global_base_color[global_indices],
        )

    visible_unique = (
        np.isfinite(local_uv_tile).all(axis=1)
        & (local_uv_tile[:, 0] >= 0)
        & (local_uv_tile[:, 0] < transform.output_width)
        & (local_uv_tile[:, 1] >= 0)
        & (local_uv_tile[:, 1] < transform.output_height)
    )
    union_count = int(matched.size + global_only.size + local_only.size)
    summary: Dict[str, Any] = {
        "tile_id": tile_id,
        "tile_category": category,
        "box": list(transform.box),
        "route_subdir": route_subdir,
        "support_subdir": support_subdir,
        "decoded_tokens": {
            "global_c1024_ovoxel_tokens": int(global_keys.size),
            "local_c1024_ovoxel_rows": int(local_coords.shape[0]),
            "local_decoder_vertices": local_vertex_count,
        },
        "projected_support_tokens": projected_stats,
        "local_to_global_c1024": {
            **inverse_stats,
            "valid_unique_global_c1024_tokens": int(local_keys.size),
            "visible_unique_tokens_in_tile_image": int(visible_unique.sum()),
            "many_to_one": _multiplicity_summary(local_counts),
        },
        "global_to_local_region": global_region_stats,
        "strict_correspondence": {
            "matched_tokens": int(matched.size),
            "matched_tokens_inside_forward_global_region": int(
                region_matched.size
            ),
            "matched_tokens_outside_forward_region_after_quantization": int(
                matched.size - region_matched.size
            ),
            "global_only_tokens": int(global_only.size),
            "local_only_tokens": int(local_only.size),
            "union_tokens": union_count,
            "jaccard": float(matched.size / max(union_count, 1)),
            "agreement_over_global_region": float(
                region_matched.size / max(global_region_keys.size, 1)
            ),
            "agreement_over_local": float(
                matched.size / max(local_keys.size, 1)
            ),
            "local_tokens_absent_from_full_global_support": int(
                (~local_full_membership).sum()
            ),
            "local_tokens_present_in_full_global_but_outside_tile_region": int(
                matched.size - region_matched.size
            ),
            "definition": (
                "matched/local-only use exact equality against the full decoded "
                "global C1024 set after official camera inverse and center-lattice "
                "quantization; global-only is restricted to the forward-projected "
                "tile canonical region"
            ),
        },
        "base_color_local_vs_global_on_exact_matches": global_material_agreement,
        "tile_margin_bands": {
            "local_support": _tile_margin_band_counts(
                local_keys,
                local_uv_tile,
                matched_keys=matched,
                only_keys=local_only,
                width=int(transform.output_width),
                height=int(transform.output_height),
                only_name="local_only",
            ),
            "global_region_support": _tile_margin_band_counts(
                global_region_keys,
                global_region_uv,
                matched_keys=region_matched,
                only_keys=global_only,
                width=int(transform.output_width),
                height=int(transform.output_height),
                only_name="global_only",
            ),
        },
        "coordinate_conventions": {
            "decoded_material_c1024": (
                "q=2*(coord+0.5)/1024-1 (voxel center)"
            ),
            "projected_shape_c32_c64": (
                "q=2*coord/(resolution-1)-1 (endpoint lattice)"
            ),
            "camera_inverse": (
                "pixal3d_projective_tile_generation_eval."
                "_centered_tile_q_to_global_q"
            ),
            "prohibited_operations": [
                "clamp",
                "bbox normalization",
                "centroid normalization",
                "coordinate compression",
            ],
        },
    }

    state = TileState(
        tile_id=tile_id,
        category=category,
        box=transform.box,
        transform=transform,
        output_dir=output_dir,
        local_keys=local_keys,
        local_base_color=local_base,
        local_uv_full=local_uv_full,
        local_uv_tile=local_uv_tile,
        global_region_keys=global_region_keys,
        global_region_uv_tile=global_region_uv,
        matched_keys=matched,
        global_only_keys=global_only,
        local_only_keys=local_only,
        distance_query_q_global=distance_query,
        summary=summary,
    )
    keys_path = _save_correspondence_keys(state)
    summary["correspondence_keys_npz"] = str(keys_path)
    if write_overlays:
        summary["correspondence_overlay"] = _save_correspondence_overlay(
            state,
            tile_dir / "reference_tile.png",
            max_points=overlay_max_points,
        )
    _atomic_json(output_dir / "summary.json", summary)

    del local_payload, local_coords, local_base_raw
    del mapped_indices, q_global, uv_full, valid
    return state


def _query_distances(
    states: Sequence[TileState],
    *,
    global_coords: torch.Tensor,
    global_vertices: torch.Tensor,
    reference_limit: int,
    seed: int,
    workers: int,
    leafsize: int,
) -> Dict[str, Any]:
    """Run sampled queries against exact (or explicitly sampled) references."""
    rng = np.random.default_rng(seed)
    timings: Dict[str, Any] = {}

    coords_np = global_coords.numpy().astype(np.float64, copy=False)
    ov_reference_count = int(coords_np.shape[0])
    ov_reference_exact = reference_limit <= 0 or ov_reference_count <= reference_limit
    if not ov_reference_exact:
        selected = rng.choice(
            ov_reference_count, size=reference_limit, replace=False
        )
        coords_reference = coords_np[selected]
    else:
        coords_reference = coords_np
    started = time.perf_counter()
    ov_tree = cKDTree(
        coords_reference,
        leafsize=leafsize,
        compact_nodes=True,
        balanced_tree=True,
    )
    timings["ovoxel_tree_build_seconds"] = time.perf_counter() - started
    for state in states:
        q = state.distance_query_q_global.astype(np.float64, copy=False)
        center_indices = (q + 1.0) * (C1024 / 2.0) - 0.5
        started = time.perf_counter()
        distance_grid, _ = ov_tree.query(center_indices, k=1, workers=workers)
        query_seconds = time.perf_counter() - started
        distance_q = np.asarray(distance_grid) * (2.0 / C1024)
        distance_world = distance_q / 2.0
        state.summary["nearest_global_ovoxel_center_distance"] = {
            **_distance_summary(distance_q),
            "units": "global normalized q",
            "world_coordinate_scale": "world distance = q distance / 2",
            "world_distance_summary": _distance_summary(distance_world),
            "query_seconds": query_seconds,
            "reference_tokens": int(coords_reference.shape[0]),
            "reference_is_full_exact_set": bool(ov_reference_exact),
        }
        np.savez_compressed(
            state.output_dir / "nearest_global_ovoxel_center_distances.npz",
            distance_q=distance_q.astype(np.float32),
            distance_world=distance_world.astype(np.float32),
        )
        _save_distance_histogram(
            distance_q,
            state.output_dir / "nearest_global_ovoxel_center_distance_hist.png",
            title=(
                f"Tile {state.tile_id:02d}: nearest global O-Voxel center"
            ),
            xlabel="distance in global normalized q",
        )
    del ov_tree, coords_reference

    vertices_np = global_vertices.numpy().astype(np.float64, copy=False) * 2.0
    vertex_reference_count = int(vertices_np.shape[0])
    vertex_reference_exact = (
        reference_limit <= 0 or vertex_reference_count <= reference_limit
    )
    if not vertex_reference_exact:
        selected = rng.choice(
            vertex_reference_count, size=reference_limit, replace=False
        )
        vertex_reference = vertices_np[selected]
    else:
        vertex_reference = vertices_np
    started = time.perf_counter()
    vertex_tree = cKDTree(
        vertex_reference,
        leafsize=leafsize,
        compact_nodes=True,
        balanced_tree=True,
    )
    timings["surface_vertex_tree_build_seconds"] = time.perf_counter() - started
    for state in states:
        q = state.distance_query_q_global.astype(np.float64, copy=False)
        started = time.perf_counter()
        distance_q, _ = vertex_tree.query(q, k=1, workers=workers)
        query_seconds = time.perf_counter() - started
        distance_q = np.asarray(distance_q)
        distance_world = distance_q / 2.0
        state.summary["nearest_global_surface_vertex_proxy_distance"] = {
            **_distance_summary(distance_q),
            "units": "global normalized q",
            "world_coordinate_scale": "world distance = q distance / 2",
            "query_seconds": query_seconds,
            "reference_vertices": int(vertex_reference.shape[0]),
            "reference_is_full_exact_set": bool(vertex_reference_exact),
            "proxy_definition": (
                "Euclidean distance from a transformed local material-voxel "
                "center to the nearest decoded global mesh vertex. This is a "
                "vertex-sampled proxy/upper bound, not exact point-to-triangle "
                "surface distance."
            ),
        }
        np.savez_compressed(
            state.output_dir / "nearest_global_surface_vertex_proxy_distances.npz",
            distance_q=distance_q.astype(np.float32),
            distance_world=distance_world.astype(np.float32),
        )
        _save_distance_histogram(
            distance_q,
            (
                state.output_dir
                / "nearest_global_surface_vertex_proxy_distance_hist.png"
            ),
            title=(
                f"Tile {state.tile_id:02d}: nearest global surface vertex proxy"
            ),
            xlabel="distance in global normalized q",
        )
        _atomic_json(state.output_dir / "summary.json", state.summary)
    del vertex_tree, vertex_reference

    timings.update(
        {
            "ovoxel_reference_total": ov_reference_count,
            "ovoxel_reference_used": (
                ov_reference_count if ov_reference_exact else reference_limit
            ),
            "surface_vertex_reference_total": vertex_reference_count,
            "surface_vertex_reference_used": (
                vertex_reference_count
                if vertex_reference_exact
                else reference_limit
            ),
            "reference_limit": int(reference_limit),
            "seed": int(seed),
        }
    )
    return timings


def _overlap_box(
    first: Sequence[int],
    second: Sequence[int],
) -> Optional[Tuple[int, int, int, int]]:
    x0 = max(int(first[0]), int(second[0]))
    y0 = max(int(first[1]), int(second[1]))
    x1 = min(int(first[2]), int(second[2]))
    y1 = min(int(first[3]), int(second[3]))
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


def _keys_in_full_image_box(
    state: TileState,
    box: Optional[Sequence[int]],
) -> np.ndarray:
    if box is None:
        return np.empty((0,), dtype=np.int64)
    x0, y0, x1, y1 = (float(value) for value in box)
    uv = state.local_uv_full
    mask = (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= x0)
        & (uv[:, 0] < x1)
        & (uv[:, 1] >= y0)
        & (uv[:, 1] < y1)
    )
    return state.local_keys[mask]


def _pairwise_analysis(
    states: Sequence[TileState],
    *,
    global_keys: np.ndarray,
    output_root: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pair_dir = output_root / "pairwise"
    pair_dir.mkdir(parents=True, exist_ok=True)
    for first, second in itertools.combinations(states, 2):
        all_intersection = np.intersect1d(
            first.local_keys, second.local_keys, assume_unique=True
        )
        all_union_count = int(
            first.local_keys.size
            + second.local_keys.size
            - all_intersection.size
        )
        overlap = _overlap_box(first.box, second.box)
        first_overlap_keys = _keys_in_full_image_box(first, overlap)
        second_overlap_keys = _keys_in_full_image_box(second, overlap)
        overlap_intersection = np.intersect1d(
            first_overlap_keys,
            second_overlap_keys,
            assume_unique=True,
        )
        overlap_union_count = int(
            first_overlap_keys.size
            + second_overlap_keys.size
            - overlap_intersection.size
        )
        double_local_only = np.setdiff1d(
            overlap_intersection,
            global_keys,
            assume_unique=True,
        )

        base_color_agreement: Optional[Dict[str, Any]] = None
        if (
            first.local_base_color is not None
            and second.local_base_color is not None
            and overlap_intersection.size
        ):
            first_found, first_indices = _keys_membership_indices(
                first.local_keys, overlap_intersection
            )
            second_found, second_indices = _keys_membership_indices(
                second.local_keys, overlap_intersection
            )
            if not bool(first_found.all() and second_found.all()):
                raise AssertionError("pairwise material keys failed exact lookup")
            base_color_agreement = _material_difference_summary(
                first.local_base_color[first_indices],
                second.local_base_color[second_indices],
            )

        pair_name = f"tile_{first.tile_id:02d}_tile_{second.tile_id:02d}"
        pair_path = pair_dir / f"{pair_name}_keys.npz"
        np.savez_compressed(
            pair_path,
            overlap_box=np.asarray(
                [] if overlap is None else overlap, dtype=np.int32
            ),
            first_overlap_keys_int64=first_overlap_keys,
            second_overlap_keys_int64=second_overlap_keys,
            exact_shared_overlap_keys_int64=overlap_intersection,
            double_tile_local_only_keys_int64=double_local_only,
            double_tile_local_only_coords_c1024_uint16=_unpack_keys(
                double_local_only
            ).astype(np.uint16),
        )
        row: Dict[str, Any] = {
            "tile_a": first.tile_id,
            "tile_b": second.tile_id,
            "category_a": first.category,
            "category_b": second.category,
            "overlap_box": None if overlap is None else list(overlap),
            "all_valid_support_a": int(first.local_keys.size),
            "all_valid_support_b": int(second.local_keys.size),
            "all_exact_shared_tokens": int(all_intersection.size),
            "all_union_tokens": all_union_count,
            "all_jaccard": float(
                all_intersection.size / max(all_union_count, 1)
            ),
            "all_agreement_over_smaller": float(
                all_intersection.size
                / max(min(first.local_keys.size, second.local_keys.size), 1)
            ),
            "overlap_support_a": int(first_overlap_keys.size),
            "overlap_support_b": int(second_overlap_keys.size),
            "overlap_exact_shared_tokens": int(overlap_intersection.size),
            "overlap_union_tokens": overlap_union_count,
            "overlap_jaccard": float(
                overlap_intersection.size / max(overlap_union_count, 1)
            ),
            "overlap_agreement_over_smaller": float(
                overlap_intersection.size
                / max(
                    min(first_overlap_keys.size, second_overlap_keys.size),
                    1,
                )
            ),
            "double_tile_local_only_tokens": int(double_local_only.size),
            "base_color_agreement_on_exact_overlap": base_color_agreement,
            "keys_npz": str(pair_path),
            "overlap_definition": (
                "representative transformed support center projects inside the "
                "intersection of the two canonical 4096 tile boxes"
            ),
        }
        _atomic_json(pair_dir / f"{pair_name}.json", row)
        rows.append(row)
    return rows


def _flatten_per_tile(state: TileState) -> Dict[str, Any]:
    summary = state.summary
    correspondence = summary["strict_correspondence"]
    inverse = summary["local_to_global_c1024"]
    projected = summary["projected_support_tokens"]
    forward = summary["global_to_local_region"]
    row: Dict[str, Any] = {
        "tile_id": state.tile_id,
        "tile_category": state.category,
        "box": json.dumps(list(state.box)),
        "global_c1024_ovoxel_tokens": summary["decoded_tokens"][
            "global_c1024_ovoxel_tokens"
        ],
        "local_c1024_ovoxel_rows": summary["decoded_tokens"][
            "local_c1024_ovoxel_rows"
        ],
        "projected_global_c1024_endpoint_rows": projected[
            "selected_global_c1024_endpoint_rows"
        ],
        "projected_c32_unique_tokens": projected[
            "projected_c32_unique_tokens"
        ],
        "projected_c64_unique_tokens": projected[
            "projected_c64_unique_tokens"
        ],
        "native_c64_unique_tokens": projected[
            "learned_native_c64_unique_tokens"
        ],
        "fused_c64_unique_tokens": projected["fused_c64_unique_tokens"],
        "local_to_global_valid_rows": inverse["global_lattice_valid_rows"],
        "local_to_global_valid_unique": inverse[
            "valid_unique_global_c1024_tokens"
        ],
        "local_to_global_collision_rows": inverse["many_to_one"][
            "collision_rows"
        ],
        "global_region_unique_tokens": forward[
            "global_region_unique_c1024_tokens"
        ],
        "global_to_local_c64_collision_rows": forward[
            "global_to_local_c64_many_to_one"
        ]["collision_rows"],
        "matched_tokens": correspondence["matched_tokens"],
        "global_only_tokens": correspondence["global_only_tokens"],
        "local_only_tokens": correspondence["local_only_tokens"],
        "jaccard": correspondence["jaccard"],
        "q_roundtrip_max_abs": inverse["q_roundtrip_max_abs"],
        "pixel_roundtrip_max": inverse["pixel_roundtrip_max"],
    }
    for prefix in (
        "nearest_global_ovoxel_center_distance",
        "nearest_global_surface_vertex_proxy_distance",
    ):
        values = summary.get(prefix)
        if isinstance(values, Mapping):
            for name in ("sample_size", "mean", "p50", "p95", "p99", "max"):
                row[f"{prefix}_{name}"] = values.get(name)
            for name in ("p50", "p95", "p99"):
                value = values.get(name)
                row[f"{prefix}_{name}_global_voxels"] = (
                    None if value is None else float(value) * C1024 / 2.0
                )
    margin_bands = summary.get("tile_margin_bands", {})
    for support_name in ("local_support", "global_region_support"):
        support_bands = margin_bands.get(support_name, {})
        for band_name, values in support_bands.get("counts", {}).items():
            for count_name, value in values.items():
                row[
                    f"{support_name}_{band_name}_{count_name}"
                ] = value
    return row


def _flatten_pairwise(row: Mapping[str, Any]) -> Dict[str, Any]:
    flat = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "base_color_agreement_on_exact_overlap",
            "overlap_definition",
        }
    }
    if isinstance(flat.get("overlap_box"), list):
        flat["overlap_box"] = json.dumps(flat["overlap_box"])
    material = row.get("base_color_agreement_on_exact_overlap")
    if isinstance(material, Mapping):
        for name in (
            "matched_tokens",
            "mean_abs",
            "rmse",
            "psnr_db_data_range_1",
            "l2_p50",
            "l2_p95",
        ):
            flat[f"base_color_{name}"] = material.get(name)
    return flat


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _git_head(cwd: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _run_self_checks() -> Dict[str, Any]:
    rng = np.random.default_rng(20260727)
    coords = rng.integers(0, C1024, size=(10000, 3), dtype=np.int64)
    tensor = torch.from_numpy(coords)
    q = _voxel_centers_to_q(tensor)
    reconstructed, valid = _q_to_center_voxel_indices_no_clip(q)
    if not bool(valid.all().item()):
        raise AssertionError("exact center self-check unexpectedly produced invalid rows")
    if not np.array_equal(reconstructed.numpy(), coords):
        raise AssertionError("center q conversion is not exactly invertible")
    keys = _pack_coords(coords)
    unpacked = _unpack_keys(keys)
    if not np.array_equal(unpacked, coords):
        raise AssertionError("packed key encoding is not exactly invertible")
    endpoint_coords = rng.integers(0, C64, size=(10000, 3), dtype=np.int64)
    endpoint_q = (
        torch.from_numpy(endpoint_coords).to(torch.float64)
        * (2.0 / float(C64 - 1))
        - 1.0
    )
    endpoint_reconstructed, endpoint_valid = _q_to_endpoint_indices_no_clip(
        endpoint_q,
        resolution=C64,
    )
    if not bool(endpoint_valid.all().item()) or not np.array_equal(
        endpoint_reconstructed.numpy(), endpoint_coords
    ):
        raise AssertionError("endpoint C64 conversion is not exactly invertible")
    return {
        "c1024_center_roundtrip_rows": int(coords.shape[0]),
        "c1024_center_roundtrip_exact": True,
        "packed_key_roundtrip_exact": True,
        "c64_endpoint_roundtrip_rows": int(endpoint_coords.shape[0]),
        "c64_endpoint_roundtrip_exact": True,
    }


def run(args: argparse.Namespace) -> None:
    started_total = time.perf_counter()
    tile_ids = _parse_tile_ids(args.tiles)
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "support_correspondence_analysis"
    )
    required = _required_paths(
        run_dir,
        tile_ids,
        route_subdir=str(args.route_subdir),
        support_subdir=str(args.support_subdir),
    )
    _validate_required_paths(required)

    self_checks = _run_self_checks()
    output_root.mkdir(parents=True, exist_ok=True)
    effective_config: Dict[str, Any] = {
        **vars(args),
        "script_version": SCRIPT_VERSION,
        "run_dir": str(run_dir),
        "output_dir": str(output_root),
        "tile_ids": tile_ids,
        "command": shlex.join(sys.argv),
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy_version,
        "git_head": _git_head(Path.cwd()),
        "required_inputs": {
            name: {"path": str(path), "bytes": path.stat().st_size}
            for name, path in required.items()
        },
        "self_checks": self_checks,
        "coordinate_policy": {
            "decoded_c1024": "voxel centers",
            "shape_c32_c64": "endpoint lattice",
            "camera_inverse": (
                "imported exact _centered_tile_q_to_global_q from generator"
            ),
            "no_clamp": True,
            "no_bbox_or_centroid_normalization": True,
            "no_dense_1024_cube": True,
            "no_ply_or_glb": True,
        },
    }
    _atomic_json(output_root / "effective_config.json", effective_config)
    print("[effective-config]")
    print(json.dumps(effective_config, ensure_ascii=False, indent=2))

    global_camera = _read_json(required["global_camera"])
    for key in ("distance", "mesh_scale"):
        if key not in global_camera or not math.isfinite(float(global_camera[key])):
            raise ValueError(f"global_camera.json has invalid {key!r}")

    global_path = required["global_decoded_support"]
    global_payload = _torch_load(global_path)
    _validate_decoded_support(
        global_payload,
        path=global_path,
        require_faces=True,
    )
    global_coords = _checkpoint_tensor(
        global_payload, "ovoxel_coords_c1024", columns=3
    ).to(torch.int64)
    global_attrs = _checkpoint_tensor(global_payload, "ovoxel_attrs")
    global_base_raw = _extract_base_color(global_payload, global_attrs)
    global_vertices = _checkpoint_tensor(global_payload, "vertices", columns=3)
    global_faces = _checkpoint_tensor(global_payload, "faces", columns=3)
    global_keys_raw = _pack_coords(global_coords.numpy())
    (
        global_keys,
        global_base_color,
        _,
        global_counts,
    ) = _unique_with_payload(global_keys_raw, attrs=global_base_raw)
    if global_keys.size != global_coords.shape[0]:
        raise ValueError(
            "global decoded O-Voxel coordinates contain duplicate rows; "
            "the analysis requires a one-to-one raw key order for camera regions"
        )
    # The unique operation sorts keys. Reorder the reference coordinates to keep
    # exact key/attribute lookup aligned while retaining the original coordinates
    # for chunked camera transforms and KD-tree construction.
    global_order = np.argsort(global_keys_raw, kind="stable")
    if not np.array_equal(global_keys, global_keys_raw[global_order]):
        raise AssertionError("global key sort self-check failed")
    if global_base_color is None and global_base_raw is not None:
        raise AssertionError("global base-color aggregation failed")

    global_summary = {
        "decoded_c1024_ovoxel_rows": int(global_coords.shape[0]),
        "decoded_c1024_ovoxel_unique_tokens": int(global_keys.size),
        "decoded_vertices": int(global_vertices.shape[0]),
        "decoded_faces": int(global_faces.shape[0]),
        "decoded_ovoxel_duplicate_summary": _multiplicity_summary(global_counts),
        "base_color_available": global_base_color is not None,
    }
    route_summary_path = run_dir / "global_baseline_1024" / "summary.json"
    if route_summary_path.is_file():
        route_summary = _read_json(route_summary_path)
        for name in (
            "global_c32_tokens",
            "global_c64_tokens",
            "global_c1024_support_tokens",
        ):
            if name in route_summary:
                global_summary[name] = route_summary[name]

    del global_payload, global_attrs, global_faces

    states: List[TileState] = []
    for tile_id in tile_ids:
        print(f"[tile {tile_id:02d}] exact support analysis")
        state = _analyze_tile(
            tile_id=tile_id,
            run_dir=run_dir,
            output_root=output_root,
            route_subdir=str(args.route_subdir),
            support_subdir=str(args.support_subdir),
            global_camera=global_camera,
            global_coords=global_coords,
            global_keys_raw=global_keys_raw,
            global_keys=global_keys,
            global_base_color=global_base_color,
            chunk_size=int(args.chunk_size),
            distance_sample_size=int(args.distance_sample_size),
            seed=int(args.seed),
            q_tolerance=float(args.roundtrip_q_tolerance),
            pixel_tolerance=float(args.roundtrip_pixel_tolerance),
            overlay_max_points=int(args.overlay_max_points),
            write_overlays=bool(args.write_overlays),
        )
        states.append(state)
        correspondence = state.summary["strict_correspondence"]
        print(
            f"[tile {tile_id:02d}] category={state.category} "
            f"matched={correspondence['matched_tokens']:,} "
            f"global_only={correspondence['global_only_tokens']:,} "
            f"local_only={correspondence['local_only_tokens']:,} "
            f"jaccard={correspondence['jaccard']:.6f}"
        )

    distance_timings: Optional[Dict[str, Any]] = None
    if bool(args.distance_analysis):
        print("[distance] building sparse KD-tree references; no dense grid")
        distance_timings = _query_distances(
            states,
            global_coords=global_coords,
            global_vertices=global_vertices,
            reference_limit=int(args.distance_reference_limit),
            seed=int(args.seed),
            workers=int(args.knn_workers),
            leafsize=int(args.kdtree_leafsize),
        )

    pairwise = _pairwise_analysis(
        states,
        global_keys=global_keys,
        output_root=output_root,
    )
    per_tile_rows = [_flatten_per_tile(state) for state in states]
    pairwise_rows = [_flatten_pairwise(row) for row in pairwise]
    _write_csv(output_root / "per_tile.csv", per_tile_rows)
    _write_csv(output_root / "pairwise.csv", pairwise_rows)

    category_map = {
        str(state.tile_id): state.category for state in states
    }
    summary: Dict[str, Any] = {
        "format": SCRIPT_VERSION,
        "run_dir": str(run_dir),
        "output_dir": str(output_root),
        "tiles": [state.summary for state in states],
        "tile_category_map": category_map,
        "global": global_summary,
        "pairwise": pairwise,
        "distance_analysis": distance_timings,
        "definitions": {
            "strict_match": (
                "exact equality of packed global C1024 integer keys after "
                "mapping local decoded material-voxel centers through the "
                "official inverse camera transform"
            ),
            "global_region": (
                "decoded global material-voxel centers whose official forward "
                "camera transform lies inside |q_local|<=1"
            ),
            "global_only_local_only": (
                "local-only is mapped local support absent from the full decoded "
                "global set; global-only is decoded global support in the "
                "forward-projected local canonical region absent from local"
            ),
            "tile_categories": (
                "edge if the tile box touches the 4096 boundary; center if its "
                "center lies within 256 px of the full-image center; mid otherwise"
            ),
            "pairwise_overlap": (
                "exact mapped global keys whose representative source centers "
                "project into the intersection of the two 4096 tile boxes"
            ),
            "surface_distance_proxy": (
                "nearest decoded global mesh vertex, not point-to-triangle distance"
            ),
        },
        "artifacts": {
            "effective_config": str(output_root / "effective_config.json"),
            "per_tile_csv": str(output_root / "per_tile.csv"),
            "pairwise_csv": str(output_root / "pairwise.csv"),
        },
        "total_seconds": time.perf_counter() - started_total,
    }
    _atomic_json(output_root / "summary.json", summary)
    print(
        f"[done] summary={output_root / 'summary.json'} "
        f"seconds={summary['total_seconds']:.3f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze sparse C1024 support correspondence for projective Pixal3D "
            "tile checkpoints without model inference or dense grids."
        )
    )
    parser.add_argument(
        "--run-dir",
        help=(
            "Run directory produced by pixal3d_projective_tile_generation_eval.py "
            "--save-decoded-support."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help="Default: RUN_DIR/support_correspondence_analysis",
    )
    parser.add_argument("--tiles", default="24,26,27")
    parser.add_argument("--route-subdir", default=DEFAULT_ROUTE_SUBDIR)
    parser.add_argument("--support-subdir", default=DEFAULT_SUPPORT_SUBDIR)
    parser.add_argument("--chunk-size", type=int, default=262_144)
    parser.add_argument("--distance-sample-size", type=int, default=50_000)
    parser.add_argument(
        "--distance-reference-limit",
        type=int,
        default=0,
        help=(
            "0 uses every global O-Voxel/vertex as the KD-tree reference. A "
            "positive value enables an explicitly approximate fixed-seed subset."
        ),
    )
    parser.add_argument(
        "--distance-analysis",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=20_260_727)
    parser.add_argument("--knn-workers", type=int, default=1)
    parser.add_argument("--kdtree-leafsize", type=int, default=32)
    parser.add_argument("--roundtrip-q-tolerance", type=float, default=5e-5)
    parser.add_argument("--roundtrip-pixel-tolerance", type=float, default=1e-2)
    parser.add_argument("--overlay-max-points", type=int, default=60_000)
    parser.add_argument(
        "--write-overlays",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--self-test-only",
        action="store_true",
        help="Run coordinate/key convention checks without reading a run directory.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test_only:
        print(json.dumps(_run_self_checks(), indent=2))
        return
    if not args.run_dir:
        parser.error("--run-dir is required unless --self-test-only is used")
    if int(args.chunk_size) < 1:
        parser.error("--chunk-size must be positive")
    if int(args.distance_sample_size) < 1:
        parser.error("--distance-sample-size must be positive")
    if int(args.distance_reference_limit) < 0:
        parser.error("--distance-reference-limit must be non-negative")
    if int(args.knn_workers) == 0:
        parser.error("--knn-workers must not be zero")
    if int(args.kdtree_leafsize) < 1:
        parser.error("--kdtree-leafsize must be positive")
    if float(args.roundtrip_q_tolerance) <= 0:
        parser.error("--roundtrip-q-tolerance must be positive")
    if float(args.roundtrip_pixel_tolerance) <= 0:
        parser.error("--roundtrip-pixel-tolerance must be positive")
    if int(args.overlay_max_points) < 1:
        parser.error("--overlay-max-points must be positive")
    try:
        run(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
