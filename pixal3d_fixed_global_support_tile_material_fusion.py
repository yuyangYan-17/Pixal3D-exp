#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fuse v7 tile materials onto an immutable global Pixal3D O-Voxel support.

This is a post-generation experiment.  It consumes checkpoints produced by
``pixal3d_projective_tile_generation_eval.py --save-decoded-support`` and does
not run any Pixal3D diffusion or decoder model.

The geometry invariant is deliberately strict:

* global vertices, faces, and sparse C1024 O-Voxel coordinates never change;
* no dense 1024^3 tensor is constructed;
* no local point can create a new global point;
* only the global ``base_color`` slice is blended;
* metallic, roughness, alpha, and all unmatched attributes remain bitwise
  identical to the global control.

Every local material coordinate is interpreted at its C1024 voxel center,

    q_local = 2 * (coord + 0.5) / 1024 - 1,

then transformed point-by-point with the exact v7
``_centered_tile_q_to_global_q`` camera inverse.  Continuous global voxel
coordinates are rounded without clipping.  A candidate survives only when the
rounded integer key already exists in the immutable global sparse support.

Within a tile, collisions on the same global key retain the continuously
mapped point nearest that global voxel center.  Across tiles, ``winner_center``
chooses the tile whose evidence is closest to its image center, while
``weighted_mean`` averages all tile candidates using the selected tent/edge
confidence.  The final update is

    new_base_color = global + blend_alpha * (candidate - global).

No color alignment, histogram matching, smoothing, inpainting, UV baking,
GLB/PLY export, or other postprocessing is performed.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pixal3d.representations import MeshWithVoxel  # noqa: E402
from pixal3d_projective_tile_generation_eval import (  # noqa: E402
    TileCameraTransform,
    _centered_tile_q_to_global_q,
)
from render_pixal3d_raw_ovoxel import (  # noqa: E402
    LPIPSEvaluator,
    composite_on_black,
    image_to_tensor,
    load_envmap,
    psnr_metric,
    render_and_evaluate_mesh,
    ssim_metric,
)


GRID_C1024 = 1024
CANONICAL_SIZE = 4096
DEFAULT_ROUTE = (
    "native_c32_shape512_projected_plus_native_c64_shape1024_texture"
)
DEFAULT_TILE_IDS = (24, 26, 27)
CHECKPOINT_FORMAT = "pixal3d_decoded_support_v1"


@dataclass
class GlobalSupport:
    path: Path
    vertices: torch.Tensor
    faces: torch.Tensor
    coords: torch.Tensor
    attrs: torch.Tensor
    origin: List[float]
    voxel_size: float
    voxel_shape: torch.Size
    layout: Dict[str, slice]
    serialized_layout: Dict[str, Dict[str, Optional[int]]]


@dataclass
class TileSupport:
    path: Path
    coords: torch.Tensor
    attrs: torch.Tensor
    layout: Dict[str, slice]


@dataclass
class TileCandidates:
    tile_id: int
    global_rows: torch.Tensor
    global_keys: torch.Tensor
    local_rows: torch.Tensor
    base_colors: torch.Tensor
    confidence: torch.Tensor
    tent_confidence: torch.Tensor
    edge_confidence: torch.Tensor
    center_distance_voxels: torch.Tensor
    uv_tile: torch.Tensor
    stats: Dict[str, Any]


@dataclass
class FusionResult:
    global_rows: torch.Tensor
    global_keys: torch.Tensor
    candidate_base_colors: torch.Tensor
    contributor_counts: torch.Tensor
    winner_tile_ids: torch.Tensor
    winner_local_rows: torch.Tensor
    winner_confidence: torch.Tensor
    winner_center_distance_voxels: torch.Tensor
    disagreement_rgb_rms: torch.Tensor
    sorted_candidate_order: torch.Tensor
    sorted_group_ids: torch.Tensor
    stats: Dict[str, Any]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSON does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _load_torch_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint must contain a mapping: {path}")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path}: expected format {CHECKPOINT_FORMAT!r}, "
            f"got {payload.get('format')!r}"
        )
    return payload


def _slice_from_serialized(name: str, value: Any) -> slice:
    if isinstance(value, slice):
        result = value
    elif isinstance(value, Mapping):
        if "start" not in value or "stop" not in value:
            raise ValueError(f"layout[{name!r}] lacks start/stop: {value!r}")
        result = slice(
            None if value["start"] is None else int(value["start"]),
            None if value["stop"] is None else int(value["stop"]),
            None if value.get("step") is None else int(value["step"]),
        )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) not in (2, 3):
            raise ValueError(f"invalid sequence layout[{name!r}]: {value!r}")
        result = slice(*[None if item is None else int(item) for item in value])
    else:
        raise ValueError(f"invalid layout[{name!r}]: {value!r}")
    if result.step not in (None, 1):
        raise ValueError(f"strided attribute layouts are unsupported: {name}")
    return result


def _normalize_layout(
    value: Any,
    *,
    channels: int,
) -> Tuple[Dict[str, slice], Dict[str, Dict[str, Optional[int]]]]:
    if not isinstance(value, Mapping):
        raise TypeError("decoded-support checkpoint layout must be a mapping")
    required = ("base_color", "metallic", "roughness", "alpha")
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"checkpoint layout is missing: {', '.join(missing)}")
    layout = {
        str(name): _slice_from_serialized(str(name), item)
        for name, item in value.items()
    }
    for name, item in layout.items():
        start = 0 if item.start is None else int(item.start)
        stop = channels if item.stop is None else int(item.stop)
        if start < 0 or stop <= start or stop > channels:
            raise ValueError(
                f"layout[{name!r}]={item!r} is invalid for {channels} channels"
            )
    base = layout["base_color"]
    if base.start != 0 or base.stop != 3 or base.step not in (None, 1):
        raise ValueError(
            "this experiment modifies only base_color channels 0:3; "
            f"checkpoint declares {base!r}"
        )
    serialized = {
        name: {
            "start": item.start,
            "stop": item.stop,
            "step": item.step,
        }
        for name, item in layout.items()
    }
    return layout, serialized


def _validate_coords(coords: torch.Tensor, *, label: str) -> torch.Tensor:
    coords = torch.as_tensor(coords).detach().cpu().contiguous()
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{label} coords must have shape [N,3]")
    if coords.dtype.is_floating_point:
        if not torch.equal(coords, coords.round()):
            raise ValueError(f"{label} coords contain non-integer values")
    coords = coords.to(torch.int32)
    if coords.numel() and (
        int(coords.min().item()) < 0
        or int(coords.max().item()) >= GRID_C1024
    ):
        raise ValueError(f"{label} coords leave the C1024 lattice")
    return coords


def _validate_attrs(
    attrs: torch.Tensor,
    *,
    rows: int,
    label: str,
) -> torch.Tensor:
    attrs = torch.as_tensor(attrs).detach().cpu().contiguous()
    if attrs.ndim != 2 or attrs.shape[0] != rows:
        raise ValueError(f"{label} attrs must be [N,C] aligned with coords")
    if attrs.shape[1] < 6:
        raise ValueError(f"{label} attrs need at least six PBR channels")
    return attrs


def _load_global_support(path: Path) -> GlobalSupport:
    payload = _load_torch_mapping(path)
    required = (
        "vertices",
        "faces",
        "ovoxel_coords_c1024",
        "ovoxel_attrs",
        "origin",
        "voxel_size",
        "voxel_shape",
        "layout",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"global checkpoint is missing: {', '.join(missing)}")
    vertices = torch.as_tensor(payload["vertices"]).detach().cpu().contiguous()
    faces = torch.as_tensor(payload["faces"]).detach().cpu().contiguous()
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("global vertices must have shape [V,3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("global faces must have shape [F,3]")
    if faces.numel() and (
        int(faces.min().item()) < 0
        or int(faces.max().item()) >= vertices.shape[0]
    ):
        raise ValueError("global faces reference invalid vertex rows")
    coords = _validate_coords(
        torch.as_tensor(payload["ovoxel_coords_c1024"]),
        label="global",
    )
    attrs = _validate_attrs(
        torch.as_tensor(payload["ovoxel_attrs"]),
        rows=int(coords.shape[0]),
        label="global",
    )
    layout, serialized_layout = _normalize_layout(
        payload["layout"],
        channels=int(attrs.shape[1]),
    )
    origin = torch.as_tensor(payload["origin"]).reshape(-1).tolist()
    if len(origin) != 3:
        raise ValueError("global origin must contain three values")
    voxel_size = float(payload["voxel_size"])
    if not math.isclose(
        voxel_size,
        1.0 / GRID_C1024,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"global voxel_size must be 1/1024, got {voxel_size}"
        )
    voxel_shape = torch.Size(int(item) for item in payload["voxel_shape"])
    if (
        len(voxel_shape) != 5
        or int(voxel_shape[0]) != 1
        or int(voxel_shape[1]) != int(attrs.shape[1])
        or any(int(item) < 1 or int(item) > GRID_C1024 for item in voxel_shape[-3:])
    ):
        raise ValueError(f"unexpected global voxel_shape: {tuple(voxel_shape)}")
    if coords.numel():
        spatial_shape = torch.tensor(voxel_shape[-3:], dtype=torch.int32)
        if bool((coords >= spatial_shape[None]).any().item()):
            raise ValueError(
                "global voxel_shape does not cover all sparse C1024 coords"
            )
    return GlobalSupport(
        path=path.resolve(),
        vertices=vertices,
        faces=faces,
        coords=coords,
        attrs=attrs,
        origin=[float(item) for item in origin],
        voxel_size=voxel_size,
        voxel_shape=voxel_shape,
        layout=layout,
        serialized_layout=serialized_layout,
    )


def _load_tile_support(
    path: Path,
    *,
    global_layout: Mapping[str, slice],
    global_channels: int,
) -> TileSupport:
    payload = _load_torch_mapping(path)
    required = (
        "ovoxel_coords_c1024",
        "ovoxel_attrs",
        "voxel_size",
        "voxel_shape",
        "layout",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"tile checkpoint is missing: {', '.join(missing)}")
    coords = _validate_coords(
        torch.as_tensor(payload["ovoxel_coords_c1024"]),
        label=f"tile checkpoint {path}",
    )
    attrs = _validate_attrs(
        torch.as_tensor(payload["ovoxel_attrs"]),
        rows=int(coords.shape[0]),
        label=f"tile checkpoint {path}",
    )
    if attrs.shape[1] != global_channels:
        raise ValueError(
            f"{path}: tile/global attribute channels differ: "
            f"{attrs.shape[1]} vs {global_channels}"
        )
    layout, _ = _normalize_layout(
        payload["layout"],
        channels=int(attrs.shape[1]),
    )
    if layout != dict(global_layout):
        raise ValueError(f"{path}: tile/global PBR layouts differ")
    if not math.isclose(
        float(payload["voxel_size"]),
        1.0 / GRID_C1024,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{path}: tile voxel_size is not 1/1024")
    voxel_shape = tuple(int(item) for item in payload["voxel_shape"])
    if (
        len(voxel_shape) != 5
        or voxel_shape[0] != 1
        or voxel_shape[1] != int(attrs.shape[1])
        or any(item < 1 or item > GRID_C1024 for item in voxel_shape[-3:])
    ):
        raise ValueError(f"{path}: unexpected voxel_shape {voxel_shape}")
    if coords.numel():
        spatial_shape = torch.tensor(voxel_shape[-3:], dtype=torch.int32)
        if bool((coords >= spatial_shape[None]).any().item()):
            raise ValueError(
                f"{path}: voxel_shape does not cover all sparse C1024 coords"
            )
    return TileSupport(
        path=path.resolve(),
        coords=coords,
        attrs=attrs,
        layout=layout,
    )


def _parse_tile_ids(value: str) -> List[int]:
    result: List[int] = []
    seen: set[int] = set()
    for text in value.split(","):
        text = text.strip()
        if not text:
            continue
        tile_id = int(text)
        if tile_id < 0 or tile_id >= 49:
            raise ValueError(f"tile id must be in [0,48], got {tile_id}")
        if tile_id not in seen:
            result.append(tile_id)
            seen.add(tile_id)
    if not result:
        raise ValueError("--tile-ids selected no tiles")
    return result


def _load_tile_transform(tile_dir: Path, tile_id: int) -> TileCameraTransform:
    camera_path = tile_dir / "tile_camera.json"
    camera = _read_json(camera_path)
    fields = {item.name for item in dataclasses.fields(TileCameraTransform)}
    missing = sorted(fields - set(camera))
    if missing:
        raise ValueError(
            f"{camera_path}: missing camera transform fields: {', '.join(missing)}"
        )
    transform = TileCameraTransform(**{name: camera[name] for name in fields})
    if int(transform.tile_id) != int(tile_id):
        raise ValueError(
            f"{camera_path}: tile id {transform.tile_id} != requested {tile_id}"
        )
    expected_box = (
        (tile_id % 7) * 512,
        (tile_id // 7) * 512,
        (tile_id % 7) * 512 + 1024,
        (tile_id // 7) * 512 + 1024,
    )
    if tuple(int(item) for item in transform.box) != expected_box:
        raise ValueError(
            f"{camera_path}: box {transform.box} != canonical {expected_box}"
        )
    return transform


def _validate_camera_consistency(
    *,
    global_camera: Mapping[str, Any],
    transform: TileCameraTransform,
    route_summary: Mapping[str, Any],
    tile_id: int,
) -> None:
    for field, global_field in (
        ("global_distance", "distance"),
        ("global_mesh_scale", "mesh_scale"),
    ):
        if not math.isclose(
            float(getattr(transform, field)),
            float(global_camera[global_field]),
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError(
                f"tile {tile_id}: {field} disagrees with global camera"
            )
    saved = route_summary.get("tile_camera_transform")
    if isinstance(saved, Mapping):
        for name in (
            "camera_angle_x",
            "distance",
            "fx",
            "fy",
            "cx",
            "cy",
            "tile_center_full_x",
            "tile_center_full_y",
        ):
            if not math.isclose(
                float(saved[name]),
                float(getattr(transform, name)),
                rel_tol=0.0,
                abs_tol=1e-7,
            ):
                raise ValueError(
                    f"tile {tile_id}: route summary and tile_camera.json "
                    f"disagree on {name}"
                )


def _pack_c1024_keys(coords: torch.Tensor) -> torch.Tensor:
    values = coords.to(torch.int64)
    return (
        (values[:, 0] * GRID_C1024 + values[:, 1]) * GRID_C1024
        + values[:, 2]
    )


def _global_key_index(
    coords: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    keys = _pack_c1024_keys(coords)
    sorted_keys, order = torch.sort(keys)
    if sorted_keys.numel() > 1 and bool(
        (sorted_keys[1:] == sorted_keys[:-1]).any().item()
    ):
        duplicates = int(
            (sorted_keys[1:] == sorted_keys[:-1]).sum().item()
        )
        raise ValueError(
            f"global sparse O-Voxel support has {duplicates} duplicate keys"
        )
    return sorted_keys.contiguous(), order.to(torch.long).contiguous()


def _confidence_from_uv(
    uv_tile: torch.Tensor,
    *,
    width: int,
    height: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    half_w = float(width) / 2.0
    half_h = float(height) / 2.0
    wx = (
        1.0
        - ((uv_tile[:, 0] - half_w) / max(half_w, 1.0)).abs()
    ).clamp(0.0, 1.0)
    wy = (
        1.0
        - ((uv_tile[:, 1] - half_h) / max(half_h, 1.0)).abs()
    ).clamp(0.0, 1.0)
    return wx * wy, torch.minimum(wx, wy)


def _load_foreground_mask(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(
            f"foreground-mask gate requested but mask is missing: {path}"
        )
    with Image.open(path) as image:
        values = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(values))


def _foreground_gate(
    uv_tile: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float,
    output_width: int,
    output_height: int,
) -> torch.Tensor:
    height, width = int(mask.shape[0]), int(mask.shape[1])
    x = torch.floor(
        uv_tile[:, 0] * (float(width) / float(output_width))
    ).to(torch.long)
    y = torch.floor(
        uv_tile[:, 1] * (float(height) / float(output_height))
    ).to(torch.long)
    # uv_tile was already checked to be inside the image.  This clamp only
    # protects nearest-neighbour image indexing from floating point roundoff;
    # it never changes a 3D material coordinate.
    x = x.clamp(0, width - 1)
    y = y.clamp(0, height - 1)
    return mask[y, x] >= float(threshold)


def _front_surface_gate(
    uv_tile: torch.Tensor,
    depth: torch.Tensor,
    *,
    output_width: int,
    output_height: int,
    resolution: int,
    tolerance: float,
) -> torch.Tensor:
    x = torch.floor(
        uv_tile[:, 0] * (float(resolution) / float(output_width))
    ).to(torch.long)
    y = torch.floor(
        uv_tile[:, 1] * (float(resolution) / float(output_height))
    ).to(torch.long)
    x = x.clamp(0, resolution - 1)
    y = y.clamp(0, resolution - 1)
    pixel = y * resolution + x
    zbuffer = torch.full(
        (resolution * resolution,),
        float("inf"),
        dtype=torch.float32,
    )
    zbuffer.scatter_reduce_(
        0,
        pixel,
        depth.to(torch.float32),
        reduce="amin",
        include_self=True,
    )
    return depth <= zbuffer[pixel] + float(tolerance)


def _select_nearest_per_global_key(
    global_rows: torch.Tensor,
    distance: torch.Tensor,
) -> torch.Tensor:
    if global_rows.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    order = torch.arange(global_rows.shape[0], dtype=torch.long)
    order = order[
        torch.argsort(distance[order], stable=True)
    ]
    order = order[
        torch.argsort(global_rows[order], stable=True)
    ]
    rows_sorted = global_rows[order]
    first = torch.ones(rows_sorted.shape[0], dtype=torch.bool)
    first[1:] = rows_sorted[1:] != rows_sorted[:-1]
    return order[first]


def _quantiles(values: torch.Tensor) -> Dict[str, Optional[float]]:
    values = values.detach().cpu().to(torch.float32)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "mean": float(values.mean().item()),
        "p50": float(torch.quantile(values, 0.50).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
    }


@torch.inference_mode()
def _map_one_tile(
    *,
    tile_id: int,
    tile_support: TileSupport,
    transform: TileCameraTransform,
    global_camera: Mapping[str, float],
    global_sorted_keys: torch.Tensor,
    global_sorted_rows: torch.Tensor,
    base_color_slice: slice,
    mapping_device: torch.device,
    chunk_size: int,
    confidence_mode: str,
    min_confidence: float,
    foreground_mask: Optional[torch.Tensor],
    foreground_threshold: float,
    front_surface_gate: bool,
    zbuffer_resolution: int,
    front_depth_tolerance: float,
) -> TileCandidates:
    coords = tile_support.coords
    row_chunks: List[torch.Tensor] = []
    ids_chunks: List[torch.Tensor] = []
    continuous_chunks: List[torch.Tensor] = []
    uv_chunks: List[torch.Tensor] = []
    depth_chunks: List[torch.Tensor] = []
    inverse_roundtrip_q_max = 0.0
    inverse_roundtrip_pixel_max = 0.0
    finite_inverse_rows = 0

    for start in range(0, int(coords.shape[0]), int(chunk_size)):
        stop = min(start + int(chunk_size), int(coords.shape[0]))
        local_ids = coords[start:stop].to(
            mapping_device,
            dtype=torch.float32,
            non_blocking=True,
        )
        q_local = (
            2.0 * (local_ids + 0.5) / float(GRID_C1024) - 1.0
        )
        q_global, _, uv_full, inverse_stats = (
            _centered_tile_q_to_global_q(
                q_local,
                global_camera=global_camera,
                transform=transform,
                validate_roundtrip=True,
            )
        )
        continuous_global = (
            (q_global + 1.0) * (float(GRID_C1024) / 2.0) - 0.5
        )
        rounded_global = torch.round(continuous_global)
        x0, y0, _, _ = transform.box
        uv_tile = torch.stack(
            [
                (uv_full[:, 0] - float(x0))
                * float(transform.crop_to_output_scale_x),
                (uv_full[:, 1] - float(y0))
                * float(transform.crop_to_output_scale_y),
            ],
            dim=1,
        )
        local_depth = float(transform.distance) - (
            q_local[:, 2] / (2.0 * float(transform.mesh_scale))
        )
        finite = (
            torch.isfinite(q_global).all(dim=1)
            & torch.isfinite(uv_tile).all(dim=1)
            & torch.isfinite(continuous_global).all(dim=1)
            & torch.isfinite(local_depth)
        )
        finite_inverse_rows += int(finite.sum().item())
        inverse_roundtrip_q_max = max(
            inverse_roundtrip_q_max,
            float(inverse_stats.get("q_roundtrip_max_abs", 0.0)),
        )
        inverse_roundtrip_pixel_max = max(
            inverse_roundtrip_pixel_max,
            float(inverse_stats.get("pixel_roundtrip_max", 0.0)),
        )
        if bool(finite.any().item()):
            row_chunks.append(
                torch.arange(start, stop, dtype=torch.long)[
                    finite.detach().cpu()
                ]
            )
            ids_chunks.append(
                rounded_global[finite].to(torch.int64).cpu()
            )
            continuous_chunks.append(
                continuous_global[finite].to(torch.float32).cpu()
            )
            uv_chunks.append(uv_tile[finite].to(torch.float32).cpu())
            depth_chunks.append(
                local_depth[finite].to(torch.float32).cpu()
            )

    if not row_chunks:
        raise RuntimeError(f"tile {tile_id}: inverse mapping produced no rows")
    local_rows = torch.cat(row_chunks)
    rounded_ids = torch.cat(ids_chunks)
    continuous_global = torch.cat(continuous_chunks)
    uv_tile = torch.cat(uv_chunks)
    depth = torch.cat(depth_chunks)
    stats: Dict[str, Any] = {
        "tile_id": int(tile_id),
        "checkpoint": str(tile_support.path),
        "raw_local_ovoxel_rows": int(coords.shape[0]),
        "finite_inverse_rows": int(finite_inverse_rows),
        "inverse_q_roundtrip_max_abs": float(inverse_roundtrip_q_max),
        "inverse_pixel_roundtrip_max": float(
            inverse_roundtrip_pixel_max
        ),
        "coordinate_formula": "q_local=2*(coord+0.5)/1024-1",
        "global_quantization": (
            "continuous index=(q_global+1)*512-0.5; round; no clamp"
        ),
    }

    inside_global = (
        (rounded_ids >= 0) & (rounded_ids < GRID_C1024)
    ).all(dim=1)
    stats["inside_global_lattice_rows"] = int(inside_global.sum().item())
    local_rows = local_rows[inside_global]
    rounded_ids = rounded_ids[inside_global]
    continuous_global = continuous_global[inside_global]
    uv_tile = uv_tile[inside_global]
    depth = depth[inside_global]

    inside_image = (
        (uv_tile[:, 0] >= 0.0)
        & (uv_tile[:, 0] < float(transform.output_width))
        & (uv_tile[:, 1] >= 0.0)
        & (uv_tile[:, 1] < float(transform.output_height))
    )
    stats["inside_tile_image_rows"] = int(inside_image.sum().item())
    local_rows = local_rows[inside_image]
    rounded_ids = rounded_ids[inside_image]
    continuous_global = continuous_global[inside_image]
    uv_tile = uv_tile[inside_image]
    depth = depth[inside_image]

    tent, edge = _confidence_from_uv(
        uv_tile,
        width=int(transform.output_width),
        height=int(transform.output_height),
    )
    confidence = tent if confidence_mode == "tent" else edge
    positive_confidence = confidence > float(min_confidence)
    stats["positive_confidence_rows"] = int(
        positive_confidence.sum().item()
    )
    local_rows = local_rows[positive_confidence]
    rounded_ids = rounded_ids[positive_confidence]
    continuous_global = continuous_global[positive_confidence]
    uv_tile = uv_tile[positive_confidence]
    depth = depth[positive_confidence]
    tent = tent[positive_confidence]
    edge = edge[positive_confidence]
    confidence = confidence[positive_confidence]

    if foreground_mask is not None:
        keep = _foreground_gate(
            uv_tile,
            foreground_mask,
            threshold=float(foreground_threshold),
            output_width=int(transform.output_width),
            output_height=int(transform.output_height),
        )
        stats["foreground_gate_enabled"] = True
        stats["foreground_gate_input_rows"] = int(keep.shape[0])
        stats["foreground_gate_pass_rows"] = int(keep.sum().item())
        local_rows = local_rows[keep]
        rounded_ids = rounded_ids[keep]
        continuous_global = continuous_global[keep]
        uv_tile = uv_tile[keep]
        depth = depth[keep]
        tent = tent[keep]
        edge = edge[keep]
        confidence = confidence[keep]
    else:
        stats["foreground_gate_enabled"] = False
        stats["foreground_gate_input_rows"] = int(local_rows.shape[0])
        stats["foreground_gate_pass_rows"] = int(local_rows.shape[0])

    if front_surface_gate:
        keep = _front_surface_gate(
            uv_tile,
            depth,
            output_width=int(transform.output_width),
            output_height=int(transform.output_height),
            resolution=int(zbuffer_resolution),
            tolerance=float(front_depth_tolerance),
        )
        stats["front_surface_gate_enabled"] = True
        stats["front_surface_gate_input_rows"] = int(keep.shape[0])
        stats["front_surface_gate_pass_rows"] = int(keep.sum().item())
        stats["front_surface_zbuffer_resolution"] = int(
            zbuffer_resolution
        )
        stats["front_surface_depth_tolerance"] = float(
            front_depth_tolerance
        )
        local_rows = local_rows[keep]
        rounded_ids = rounded_ids[keep]
        continuous_global = continuous_global[keep]
        uv_tile = uv_tile[keep]
        depth = depth[keep]
        tent = tent[keep]
        edge = edge[keep]
        confidence = confidence[keep]
    else:
        stats["front_surface_gate_enabled"] = False
        stats["front_surface_gate_input_rows"] = int(local_rows.shape[0])
        stats["front_surface_gate_pass_rows"] = int(local_rows.shape[0])

    candidate_keys = _pack_c1024_keys(rounded_ids.to(torch.int32))
    positions = torch.searchsorted(global_sorted_keys, candidate_keys)
    valid_position = positions < global_sorted_keys.shape[0]
    matched = torch.zeros_like(valid_position)
    if bool(valid_position.any().item()):
        valid_rows = torch.where(valid_position)[0]
        matched[valid_rows] = (
            global_sorted_keys[positions[valid_rows]]
            == candidate_keys[valid_rows]
        )
    stats["existing_global_support_match_rows"] = int(
        matched.sum().item()
    )
    stats["point_birth_rows_rejected"] = int((~matched).sum().item())
    local_rows = local_rows[matched]
    rounded_ids = rounded_ids[matched]
    continuous_global = continuous_global[matched]
    uv_tile = uv_tile[matched]
    tent = tent[matched]
    edge = edge[matched]
    confidence = confidence[matched]
    candidate_keys = candidate_keys[matched]
    positions = positions[matched]
    global_rows = global_sorted_rows[positions].to(torch.long)

    base_colors = tile_support.attrs[
        local_rows,
        base_color_slice,
    ].to(torch.float32)
    finite_color = torch.isfinite(base_colors).all(dim=1)
    stats["finite_base_color_rows"] = int(finite_color.sum().item())
    local_rows = local_rows[finite_color]
    rounded_ids = rounded_ids[finite_color]
    continuous_global = continuous_global[finite_color]
    uv_tile = uv_tile[finite_color]
    tent = tent[finite_color]
    edge = edge[finite_color]
    confidence = confidence[finite_color]
    candidate_keys = candidate_keys[finite_color]
    global_rows = global_rows[finite_color]
    base_colors = base_colors[finite_color]

    center_distance = torch.linalg.vector_norm(
        continuous_global - rounded_ids.to(torch.float32),
        dim=1,
    )
    keep_rows = _select_nearest_per_global_key(
        global_rows,
        center_distance,
    )
    stats["within_tile_candidate_rows"] = int(global_rows.shape[0])
    stats["within_tile_unique_global_keys"] = int(keep_rows.shape[0])
    stats["within_tile_collision_rows_removed"] = int(
        global_rows.shape[0] - keep_rows.shape[0]
    )
    stats["center_distance_voxels"] = _quantiles(
        center_distance[keep_rows]
    )
    stats["selected_confidence"] = _quantiles(confidence[keep_rows])
    if keep_rows.numel() == 0:
        raise RuntimeError(
            f"tile {tile_id}: no local material matches global support"
        )

    order = torch.argsort(global_rows[keep_rows], stable=True)
    keep_rows = keep_rows[order]
    return TileCandidates(
        tile_id=int(tile_id),
        global_rows=global_rows[keep_rows].contiguous(),
        global_keys=candidate_keys[keep_rows].contiguous(),
        local_rows=local_rows[keep_rows].contiguous(),
        base_colors=base_colors[keep_rows].contiguous(),
        confidence=confidence[keep_rows].contiguous(),
        tent_confidence=tent[keep_rows].contiguous(),
        edge_confidence=edge[keep_rows].contiguous(),
        center_distance_voxels=center_distance[keep_rows].contiguous(),
        uv_tile=uv_tile[keep_rows].contiguous(),
        stats=stats,
    )


def _concatenate_candidates(
    tiles: Sequence[TileCandidates],
) -> Dict[str, torch.Tensor]:
    return {
        "global_rows": torch.cat([item.global_rows for item in tiles]),
        "global_keys": torch.cat([item.global_keys for item in tiles]),
        "local_rows": torch.cat([item.local_rows for item in tiles]),
        "tile_ids": torch.cat(
            [
                torch.full(
                    (item.global_rows.shape[0],),
                    int(item.tile_id),
                    dtype=torch.int16,
                )
                for item in tiles
            ]
        ),
        "base_colors": torch.cat([item.base_colors for item in tiles]),
        "confidence": torch.cat([item.confidence for item in tiles]),
        "tent_confidence": torch.cat(
            [item.tent_confidence for item in tiles]
        ),
        "edge_confidence": torch.cat(
            [item.edge_confidence for item in tiles]
        ),
        "center_distance_voxels": torch.cat(
            [item.center_distance_voxels for item in tiles]
        ),
    }


def _first_per_sorted_group(values: torch.Tensor) -> torch.Tensor:
    first = torch.ones(values.shape[0], dtype=torch.bool)
    if values.numel() > 1:
        first[1:] = values[1:] != values[:-1]
    return torch.where(first)[0]


def _fuse_candidates(
    candidates: Mapping[str, torch.Tensor],
    *,
    fusion_mode: str,
) -> FusionResult:
    rows = candidates["global_rows"].to(torch.long)
    if rows.numel() == 0:
        raise RuntimeError("no material candidates survived mapping")
    order = torch.argsort(rows, stable=True)
    rows_sorted = rows[order]
    keys_sorted = candidates["global_keys"][order]
    colors_sorted = candidates["base_colors"][order].to(torch.float32)
    confidence_sorted = candidates["confidence"][order].to(torch.float32)
    tile_sorted = candidates["tile_ids"][order].to(torch.long)
    local_sorted = candidates["local_rows"][order].to(torch.long)
    distance_sorted = candidates["center_distance_voxels"][order].to(
        torch.float32
    )

    unique_rows, group_ids, counts = torch.unique_consecutive(
        rows_sorted,
        return_inverse=True,
        return_counts=True,
    )
    first = _first_per_sorted_group(rows_sorted)
    unique_keys = keys_sorted[first]
    groups = int(unique_rows.shape[0])

    color_sum = torch.zeros((groups, 3), dtype=torch.float32)
    color_sq_sum = torch.zeros((groups, 3), dtype=torch.float32)
    color_sum.index_add_(0, group_ids, colors_sorted)
    color_sq_sum.index_add_(0, group_ids, colors_sorted.square())
    mean_color = color_sum / counts[:, None].to(torch.float32)
    variance = (
        color_sq_sum / counts[:, None].to(torch.float32)
        - mean_color.square()
    ).clamp_min(0.0)
    disagreement = torch.sqrt(variance.sum(dim=1))

    # Deterministic winner: highest tile-center confidence, then lowest
    # continuous-to-global-center distance, then lowest tile/local row.
    winner_order = torch.arange(rows_sorted.shape[0], dtype=torch.long)
    winner_order = winner_order[
        torch.argsort(local_sorted[winner_order], stable=True)
    ]
    winner_order = winner_order[
        torch.argsort(tile_sorted[winner_order], stable=True)
    ]
    winner_order = winner_order[
        torch.argsort(distance_sorted[winner_order], stable=True)
    ]
    winner_order = winner_order[
        torch.argsort(
            confidence_sorted[winner_order],
            descending=True,
            stable=True,
        )
    ]
    winner_order = winner_order[
        torch.argsort(rows_sorted[winner_order], stable=True)
    ]
    winner_first = _first_per_sorted_group(rows_sorted[winner_order])
    winner_indices = winner_order[winner_first]
    if not torch.equal(rows_sorted[winner_indices], unique_rows):
        raise RuntimeError("winner selection lost global-row alignment")

    if fusion_mode == "winner_center":
        fused_color = colors_sorted[winner_indices]
    elif fusion_mode == "weighted_mean":
        weights = confidence_sorted.clamp_min(torch.finfo(torch.float32).eps)
        weight_sum = torch.zeros(groups, dtype=torch.float32)
        weighted_color_sum = torch.zeros((groups, 3), dtype=torch.float32)
        weight_sum.index_add_(0, group_ids, weights)
        weighted_color_sum.index_add_(
            0,
            group_ids,
            colors_sorted * weights[:, None],
        )
        fused_color = weighted_color_sum / weight_sum[:, None]
    else:
        raise ValueError(f"unsupported fusion mode: {fusion_mode}")

    conflict = counts > 1
    stats = {
        "fusion_mode": str(fusion_mode),
        "candidate_rows_after_within_tile_dedup": int(rows.shape[0]),
        "matched_global_keys": int(groups),
        "single_tile_global_keys": int((counts == 1).sum().item()),
        "cross_tile_conflict_global_keys": int(conflict.sum().item()),
        "cross_tile_conflict_candidate_rows": int(
            counts[conflict].sum().item()
        ),
        "max_contributors_per_global_key": int(counts.max().item()),
        "disagreement_rgb_rms_all": _quantiles(disagreement),
        "disagreement_rgb_rms_conflicts": _quantiles(
            disagreement[conflict]
        ),
        "winner_tile_histogram": {
            str(int(tile_id)): int(count)
            for tile_id, count in zip(
                *torch.unique(
                    tile_sorted[winner_indices],
                    return_counts=True,
                )
            )
        },
    }
    return FusionResult(
        global_rows=unique_rows.contiguous(),
        global_keys=unique_keys.contiguous(),
        candidate_base_colors=fused_color.contiguous(),
        contributor_counts=counts.contiguous(),
        winner_tile_ids=tile_sorted[winner_indices].to(torch.int16).contiguous(),
        winner_local_rows=local_sorted[winner_indices].contiguous(),
        winner_confidence=confidence_sorted[winner_indices].contiguous(),
        winner_center_distance_voxels=distance_sorted[
            winner_indices
        ].contiguous(),
        disagreement_rgb_rms=disagreement.contiguous(),
        sorted_candidate_order=order.contiguous(),
        sorted_group_ids=group_ids.contiguous(),
        stats=stats,
    )


def _pair_overlap_diagnostics(
    tiles: Sequence[TileCandidates],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for left_index, left in enumerate(tiles):
        for right in tiles[left_index + 1 :]:
            # Each tile is sorted by immutable global row.  Check overlap in
            # that ordering; packed keys need not follow source checkpoint row
            # order and therefore are not a valid searchsorted domain here.
            positions = torch.searchsorted(
                right.global_rows,
                left.global_rows,
            )
            valid = positions < right.global_rows.shape[0]
            shared = torch.zeros_like(valid)
            if bool(valid.any().item()):
                valid_rows = torch.where(valid)[0]
                shared[valid_rows] = (
                    right.global_rows[positions[valid_rows]]
                    == left.global_rows[valid_rows]
                )
            left_rows = torch.where(shared)[0]
            right_rows = positions[shared]
            if left_rows.numel():
                color_l2 = torch.linalg.vector_norm(
                    left.base_colors[left_rows]
                    - right.base_colors[right_rows],
                    dim=1,
                )
                mapping_distance_delta = (
                    left.center_distance_voxels[left_rows]
                    - right.center_distance_voxels[right_rows]
                ).abs()
            else:
                color_l2 = torch.empty(0)
                mapping_distance_delta = torch.empty(0)
            records.append(
                {
                    "tile_a": int(left.tile_id),
                    "tile_b": int(right.tile_id),
                    "shared_global_keys": int(left_rows.numel()),
                    "base_color_l2": _quantiles(color_l2),
                    "center_distance_delta_voxels": _quantiles(
                        mapping_distance_delta
                    ),
                    "tile_a_shared_confidence": _quantiles(
                        left.confidence[left_rows]
                    ),
                    "tile_b_shared_confidence": _quantiles(
                        right.confidence[right_rows]
                    ),
                }
            )
    return records


def _tensor_sha256(tensor: torch.Tensor, chunk_bytes: int = 16 << 20) -> str:
    value = tensor.detach().cpu().contiguous()
    byte_view = value.view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, int(byte_view.numel()), int(chunk_bytes)):
        digest.update(byte_view[start : start + chunk_bytes].numpy().tobytes())
    return digest.hexdigest()


def _apply_material_blend(
    global_attrs: torch.Tensor,
    fusion: FusionResult,
    *,
    base_color_slice: slice,
    blend_alpha: float,
) -> torch.Tensor:
    result = global_attrs.clone()
    if float(blend_alpha) == 0.0:
        return result
    global_base = global_attrs[
        fusion.global_rows,
        base_color_slice,
    ].to(torch.float32)
    blended = global_base + float(blend_alpha) * (
        fusion.candidate_base_colors - global_base
    )
    if not bool(torch.isfinite(blended).all().item()):
        raise RuntimeError("blended base colors contain non-finite values")
    result[fusion.global_rows, base_color_slice] = blended.to(result.dtype)
    return result


def _check_attribute_invariants(
    *,
    global_attrs: torch.Tensor,
    modified_attrs: torch.Tensor,
    fusion: FusionResult,
    base_color_slice: slice,
) -> Dict[str, Any]:
    if global_attrs.shape != modified_attrs.shape:
        raise RuntimeError("modified global attribute shape changed")
    matched = torch.zeros(global_attrs.shape[0], dtype=torch.bool)
    matched[fusion.global_rows] = True
    unmatched_bitwise = torch.equal(
        modified_attrs[~matched],
        global_attrs[~matched],
    )
    base_indices = set(
        range(
            int(base_color_slice.start),
            int(base_color_slice.stop),
        )
    )
    other_indices = [
        index
        for index in range(global_attrs.shape[1])
        if index not in base_indices
    ]
    non_base_bitwise = torch.equal(
        modified_attrs[:, other_indices],
        global_attrs[:, other_indices],
    )
    alpha_zero_attrs = _apply_material_blend(
        global_attrs,
        fusion,
        base_color_slice=base_color_slice,
        blend_alpha=0.0,
    )
    alpha_zero_recovers_control = torch.equal(
        alpha_zero_attrs,
        global_attrs,
    )
    checks = {
        "unmatched_attrs_bitwise_unchanged": bool(unmatched_bitwise),
        "non_base_color_channels_bitwise_unchanged": bool(
            non_base_bitwise
        ),
        "blend_alpha_zero_recovers_control_attrs": bool(
            alpha_zero_recovers_control
        ),
        "matched_rows": int(matched.sum().item()),
        "unmatched_rows": int((~matched).sum().item()),
    }
    failed = [name for name, passed in checks.items() if isinstance(passed, bool) and not passed]
    if failed:
        raise RuntimeError(
            "attribute invariant failure: " + ", ".join(failed)
        )
    return checks


def _build_mesh(
    support: GlobalSupport,
    attrs: torch.Tensor,
    *,
    device: torch.device,
) -> MeshWithVoxel:
    mesh = MeshWithVoxel(
        vertices=support.vertices,
        faces=support.faces,
        origin=support.origin,
        voxel_size=support.voxel_size,
        coords=support.coords,
        attrs=attrs,
        voxel_shape=support.voxel_shape,
        layout=support.layout,
    )
    return mesh.to(device)


def _metric_pair(
    reference: Image.Image,
    prediction: Image.Image,
    *,
    metric_resolution: int,
    lpips_evaluator: Optional[LPIPSEvaluator],
) -> Dict[str, Optional[float]]:
    target = (int(metric_resolution), int(metric_resolution))
    reference_tensor = image_to_tensor(reference, target)
    prediction_tensor = image_to_tensor(prediction, target)
    return {
        "psnr_db": float(psnr_metric(reference_tensor, prediction_tensor)),
        "ssim": float(ssim_metric(reference_tensor, prediction_tensor)),
        "lpips": (
            None
            if lpips_evaluator is None
            else float(
                lpips_evaluator.evaluate(
                    reference_tensor,
                    prediction_tensor,
                )
            )
        ),
    }


def _save_diff(
    reference: Image.Image,
    prediction: Image.Image,
    path: Path,
) -> None:
    ref = np.asarray(reference.convert("RGB"), dtype=np.float32)
    pred = np.asarray(prediction.convert("RGB"), dtype=np.float32)
    if ref.shape != pred.shape:
        raise ValueError("diff images must have equal shapes")
    difference = np.abs(ref - pred).mean(axis=2) / 255.0
    red = np.clip(difference * 3.0, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(difference * 4.0 - 2.0), 0.0, 1.0)
    blue = np.clip(1.0 - difference * 3.0, 0.0, 1.0)
    image = np.stack([red, green, blue], axis=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((image * 255.0).astype(np.uint8), mode="RGB").save(path)


def _crop_global_render(
    *,
    render_path: Path,
    box: Sequence[int],
    output_path: Path,
) -> Dict[str, Any]:
    with Image.open(render_path) as source:
        render = composite_on_black(source)
    x0, y0, x1, y1 = (int(item) for item in box)
    scale_x = float(render.width) / float(CANONICAL_SIZE)
    scale_y = float(render.height) / float(CANONICAL_SIZE)
    crop_box = (
        int(round(x0 * scale_x)),
        int(round(y0 * scale_y)),
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
    )
    if (
        crop_box[0] < 0
        or crop_box[1] < 0
        or crop_box[2] > render.width
        or crop_box[3] > render.height
        or crop_box[2] <= crop_box[0]
        or crop_box[3] <= crop_box[1]
    ):
        raise RuntimeError(
            f"invalid crop {crop_box} for render size {render.size}"
        )
    crop = render.crop(crop_box)
    if crop.size != (1024, 1024):
        crop = crop.resize((1024, 1024), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)
    return {
        "source_render": str(render_path),
        "source_render_size": list(render.size),
        "canonical_box": [x0, y0, x1, y1],
        "source_crop_box": list(crop_box),
        "saved_crop": str(output_path),
        "native_1024_crop": (
            crop_box[2] - crop_box[0] == 1024
            and crop_box[3] - crop_box[1] == 1024
        ),
    }


def _evaluate_tile_crops(
    *,
    run_dir: Path,
    output_dir: Path,
    tile_ids: Sequence[int],
    transforms: Mapping[int, TileCameraTransform],
    control_render: Path,
    fused_render: Path,
    metric_resolution: int,
    lpips_net: str,
    metric_device: str,
    skip_lpips: bool,
) -> Tuple[List[Dict[str, Any]], Optional[LPIPSEvaluator]]:
    evaluator: Optional[LPIPSEvaluator] = None
    if not skip_lpips:
        device = torch.device(
            metric_device
            if metric_device.startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )
        evaluator = LPIPSEvaluator(lpips_net, device)
    records: List[Dict[str, Any]] = []
    for tile_id in tile_ids:
        tile_output = output_dir / "tiles" / f"tile_{tile_id:02d}"
        reference_path = (
            run_dir / "tiles" / f"tile_{tile_id:02d}" / "reference_tile.png"
        )
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"tile reference does not exist: {reference_path}"
            )
        with Image.open(reference_path) as source:
            reference = composite_on_black(source)
        if reference.size != (1024, 1024):
            reference = reference.resize((1024, 1024), Image.Resampling.LANCZOS)
        saved_reference = tile_output / "reference.png"
        tile_output.mkdir(parents=True, exist_ok=True)
        reference.save(saved_reference)
        control_crop_path = tile_output / "global_control_crop.png"
        fused_crop_path = tile_output / "global_fused_crop.png"
        control_crop = _crop_global_render(
            render_path=control_render,
            box=transforms[tile_id].box,
            output_path=control_crop_path,
        )
        fused_crop = _crop_global_render(
            render_path=fused_render,
            box=transforms[tile_id].box,
            output_path=fused_crop_path,
        )
        with Image.open(control_crop_path) as image:
            control_image = image.convert("RGB")
        with Image.open(fused_crop_path) as image:
            fused_image = image.convert("RGB")
        control_metrics = _metric_pair(
            reference,
            control_image,
            metric_resolution=int(metric_resolution),
            lpips_evaluator=evaluator,
        )
        fused_metrics = _metric_pair(
            reference,
            fused_image,
            metric_resolution=int(metric_resolution),
            lpips_evaluator=evaluator,
        )
        diff_path = tile_output / "fused_abs_diff.png"
        _save_diff(reference, fused_image, diff_path)
        delta = {
            "psnr_gain_db": (
                float(fused_metrics["psnr_db"])
                - float(control_metrics["psnr_db"])
            ),
            "ssim_gain": (
                float(fused_metrics["ssim"])
                - float(control_metrics["ssim"])
            ),
            "lpips_reduction": (
                None
                if fused_metrics["lpips"] is None
                or control_metrics["lpips"] is None
                else float(control_metrics["lpips"])
                - float(fused_metrics["lpips"])
            ),
        }
        record = {
            "tile_id": int(tile_id),
            "box": list(transforms[tile_id].box),
            "reference_png": str(saved_reference),
            "control_crop": control_crop,
            "fused_crop": fused_crop,
            "fused_diff_png": str(diff_path),
            "control_metrics": control_metrics,
            "fused_metrics": fused_metrics,
            "fused_minus_control": delta,
        }
        _atomic_json(tile_output / "metrics.json", record)
        records.append(record)
    return records, evaluator


def _save_disagreement_maps(
    *,
    tiles: Sequence[TileCandidates],
    fusion: FusionResult,
    output_dir: Path,
    run_dir: Path,
) -> Dict[int, str]:
    sorted_unique_rows = fusion.global_rows
    paths: Dict[int, str] = {}
    maximum = max(
        float(fusion.disagreement_rgb_rms.max().item()),
        1e-8,
    )
    for tile in tiles:
        positions = torch.searchsorted(sorted_unique_rows, tile.global_rows)
        found = positions < sorted_unique_rows.shape[0]
        matched = torch.zeros_like(found)
        if bool(found.any().item()):
            valid_rows = torch.where(found)[0]
            matched[valid_rows] = (
                sorted_unique_rows[positions[valid_rows]]
                == tile.global_rows[valid_rows]
            )
        found = matched
        values = torch.zeros(tile.global_rows.shape[0], dtype=torch.float32)
        values[found] = fusion.disagreement_rgb_rms[positions[found]]
        x = torch.floor(tile.uv_tile[:, 0]).to(torch.long).clamp(0, 1023)
        y = torch.floor(tile.uv_tile[:, 1]).to(torch.long).clamp(0, 1023)
        flat = torch.zeros(1024 * 1024, dtype=torch.float32)
        flat.scatter_reduce_(
            0,
            y * 1024 + x,
            values,
            reduce="amax",
            include_self=True,
        )
        heat = (flat.reshape(1024, 1024) / maximum).clamp(0.0, 1.0)
        red = heat
        green = (1.0 - (heat - 0.5).abs() * 2.0).clamp(0.0, 1.0)
        blue = 1.0 - heat
        color = torch.stack([red, green, blue], dim=2).numpy()
        with Image.open(
            run_dir / "tiles" / f"tile_{tile.tile_id:02d}" / "reference_tile.png"
        ) as source:
            reference = composite_on_black(source).resize((1024, 1024))
        overlay = Image.blend(
            reference,
            Image.fromarray((color * 255.0).astype(np.uint8), mode="RGB"),
            0.55,
        )
        path = (
            output_dir
            / "tiles"
            / f"tile_{tile.tile_id:02d}"
            / "candidate_disagreement_overlay.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(path)
        paths[int(tile.tile_id)] = str(path)
    return paths


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        return
    keys: List[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in record.items()
                }
            )


def _make_contact_sheet(
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> str:
    if not records:
        raise ValueError("cannot create an empty contact sheet")
    thumb = 384
    label_height = 34
    columns = (
        ("reference_png", "reference"),
        ("control_crop", "global control"),
        ("fused_crop", "fused material"),
        ("fused_diff_png", "fused abs diff"),
    )
    canvas = Image.new(
        "RGB",
        (len(columns) * thumb, len(records) * (thumb + label_height)),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, record in enumerate(records):
        y = row_index * (thumb + label_height)
        for column_index, (key, label) in enumerate(columns):
            value = record[key]
            if isinstance(value, Mapping):
                value = value["saved_crop"]
            with Image.open(str(value)) as source:
                image = composite_on_black(source).resize(
                    (thumb, thumb),
                    Image.Resampling.LANCZOS,
                )
            x = column_index * thumb
            canvas.paste(image, (x, y + label_height))
            draw.text(
                (x + 8, y + 9),
                f"tile {int(record['tile_id']):02d} | {label}",
                fill=(255, 255, 255),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return str(output_path)


def _metric_delta(
    fused: Mapping[str, Any],
    control: Mapping[str, Any],
) -> Dict[str, Optional[float]]:
    return {
        "psnr_gain_db": (
            None
            if fused.get("psnr_db") is None or control.get("psnr_db") is None
            else float(fused["psnr_db"]) - float(control["psnr_db"])
        ),
        "ssim_gain": (
            None
            if fused.get("ssim") is None or control.get("ssim") is None
            else float(fused["ssim"]) - float(control["ssim"])
        ),
        "lpips_reduction": (
            None
            if fused.get("lpips") is None or control.get("lpips") is None
            else float(control["lpips"]) - float(fused["lpips"])
        ),
    }


def _effective_render_config(
    args: argparse.Namespace,
    run_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    global_baseline = run_summary.get("global_baseline_1024", {})
    official = run_summary.get("official_renderer", {})
    skip_lpips = args.skip_lpips
    if skip_lpips is None:
        skip_lpips = global_baseline.get("lpips") is None
    return {
        "envmap": (
            str(args.envmap)
            if args.envmap is not None
            else str(
                global_baseline.get(
                    "envmap",
                    official.get("envmap", "studio"),
                )
            )
        ),
        "render_resolution": int(
            args.render_resolution
            if args.render_resolution is not None
            else global_baseline.get("render_resolution", CANONICAL_SIZE)
        ),
        "metric_resolution": int(
            args.metric_resolution
            if args.metric_resolution is not None
            else global_baseline.get("metric_resolution", 1024)
        ),
        "ssaa": int(
            args.ssaa
            if args.ssaa is not None
            else global_baseline.get("ssaa", official.get("ssaa", 2))
        ),
        "peel_layers": int(
            args.peel_layers
            if args.peel_layers is not None
            else global_baseline.get(
                "peel_layers",
                official.get("peel_layers", 8),
            )
        ),
        "use_envmap_bg": bool(
            args.use_envmap_bg
            if args.use_envmap_bg is not None
            else global_baseline.get(
                "use_envmap_bg",
                official.get("use_envmap_bg", False),
            )
        ),
        "face_chunk_size": int(args.face_chunk_size),
        "lpips_net": str(args.lpips_net),
        "metric_device": str(args.metric_device),
        "skip_lpips": bool(skip_lpips),
    }


def _reproduction_command(
    *,
    script: Path,
    args: argparse.Namespace,
    tile_ids: Sequence[int],
    render: Mapping[str, Any],
) -> str:
    command = [
        sys.executable,
        str(script),
        "--run-dir",
        str(Path(args.run_dir).expanduser().resolve()),
        "--output-dir",
        str(Path(args.output_dir).expanduser().resolve()),
        "--tile-ids",
        ",".join(str(item) for item in tile_ids),
        "--route-name",
        str(args.route_name),
        "--fusion-mode",
        str(args.fusion_mode),
        "--confidence-mode",
        str(args.confidence_mode),
        "--blend-alpha",
        str(float(args.blend_alpha)),
        "--min-confidence",
        str(float(args.min_confidence)),
        "--mapping-device",
        str(args.mapping_device),
        "--mapping-chunk-size",
        str(int(args.mapping_chunk_size)),
        "--foreground-threshold",
        str(float(args.foreground_threshold)),
        "--zbuffer-resolution",
        str(int(args.zbuffer_resolution)),
        "--front-depth-tolerance",
        str(float(args.front_depth_tolerance)),
        "--envmap",
        str(render["envmap"]),
        "--render-resolution",
        str(int(render["render_resolution"])),
        "--metric-resolution",
        str(int(render["metric_resolution"])),
        "--ssaa",
        str(int(render["ssaa"])),
        "--peel-layers",
        str(int(render["peel_layers"])),
        "--face-chunk-size",
        str(int(render["face_chunk_size"])),
        "--lpips-net",
        str(render["lpips_net"]),
        "--metric-device",
        str(render["metric_device"]),
    ]
    if args.cuda_device is not None:
        command.extend(["--cuda-device", str(int(args.cuda_device))])
    command.extend(
        [
            (
                "--foreground-mask-gate"
                if args.foreground_mask_gate
                else "--no-foreground-mask-gate"
            ),
            (
                "--front-surface-gate"
                if args.front_surface_gate
                else "--no-front-surface-gate"
            ),
            (
                "--use-envmap-bg"
                if render["use_envmap_bg"]
                else "--no-use-envmap-bg"
            ),
            "--skip-lpips" if render["skip_lpips"] else "--no-skip-lpips",
        ]
    )
    if args.dry_run:
        command.append("--dry-run")
    return shlex.join(command)


def _validate_cli(args: argparse.Namespace) -> None:
    if not 0.0 <= float(args.blend_alpha) <= 1.0:
        raise ValueError("--blend-alpha must be in [0,1]")
    if float(args.min_confidence) < 0.0:
        raise ValueError("--min-confidence must be non-negative")
    if int(args.mapping_chunk_size) < 1:
        raise ValueError("--mapping-chunk-size must be positive")
    if not 0.0 <= float(args.foreground_threshold) <= 1.0:
        raise ValueError("--foreground-threshold must be in [0,1]")
    if int(args.zbuffer_resolution) < 1:
        raise ValueError("--zbuffer-resolution must be positive")
    if float(args.front_depth_tolerance) < 0.0:
        raise ValueError("--front-depth-tolerance must be non-negative")
    for name in ("render_resolution", "metric_resolution", "ssaa", "peel_layers"):
        value = getattr(args, name)
        if value is not None and int(value) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args.face_chunk_size) < 0:
        raise ValueError("--face-chunk-size must be non-negative")
    if args.cuda_device is not None and int(args.cuda_device) < 0:
        raise ValueError("--cuda-device must be non-negative")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _validate_cli(args)
    tile_ids = _parse_tile_ids(str(args.tile_ids))
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"v7 run directory does not exist: {run_dir}")

    run_summary = _read_json(run_dir / "summary.json")
    if run_summary.get("format") != (
        "pixal3d_global_baseline_vs_tile_cascade_v7_official_renderer"
    ):
        raise ValueError(
            f"{run_dir}: expected a v7 official-renderer run, got "
            f"{run_summary.get('format')!r}"
        )
    global_camera = _read_json(run_dir / "global_camera.json")
    for name in ("camera_angle_x", "distance", "mesh_scale"):
        if name not in global_camera:
            raise ValueError(f"global_camera.json is missing {name}")
    global_checkpoint_path = (
        run_dir / "global_baseline_1024" / "decoded_support.pt"
    )

    # Resolve every required input before creating output files.  This makes a
    # missing-checkpoint dry-run fail clearly and without a partial experiment.
    tile_inputs: Dict[int, Dict[str, Path]] = {}
    for tile_id in tile_ids:
        tile_dir = run_dir / "tiles" / f"tile_{tile_id:02d}"
        route_dir = tile_dir / str(args.route_name)
        required_paths = {
            "tile_dir": tile_dir,
            "checkpoint": route_dir / "decoded_support.pt",
            "route_summary": route_dir / "summary.json",
            "camera": tile_dir / "tile_camera.json",
            "reference": tile_dir / "reference_tile.png",
        }
        for name, path in required_paths.items():
            if name == "tile_dir":
                if not path.is_dir():
                    raise FileNotFoundError(
                        f"tile directory does not exist: {path}"
                    )
            elif not path.is_file():
                raise FileNotFoundError(
                    f"tile {tile_id} required {name} is missing: {path}"
                )
        if args.foreground_mask_gate:
            mask = tile_dir / "foreground_mask.png"
            if not mask.is_file():
                raise FileNotFoundError(
                    f"tile {tile_id} foreground mask is missing: {mask}"
                )
            required_paths["foreground_mask"] = mask
        tile_inputs[tile_id] = required_paths
    if not global_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"global decoded support is missing: {global_checkpoint_path}"
        )
    canonical_1024 = run_dir / "canonical_1024.png"
    if not canonical_1024.is_file():
        raise FileNotFoundError(
            f"global metric reference is missing: {canonical_1024}"
        )

    if args.cuda_device is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("--cuda-device was set but CUDA is unavailable")
        torch.cuda.set_device(int(args.cuda_device))
    mapping_device = torch.device(str(args.mapping_device))
    if mapping_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--mapping-device=cuda requested but CUDA is unavailable; use cpu"
        )

    render_config = _effective_render_config(args, run_summary)
    if render_config["render_resolution"] < 1:
        raise ValueError("effective render resolution must be positive")
    reproduction = _reproduction_command(
        script=Path(__file__).resolve(),
        args=args,
        tile_ids=tile_ids,
        render=render_config,
    )
    effective_config = {
        "format": "pixal3d_fixed_global_support_tile_material_fusion_config_v1",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "global_checkpoint": str(global_checkpoint_path),
        "tile_ids": [int(item) for item in tile_ids],
        "route_name": str(args.route_name),
        "fusion_mode": str(args.fusion_mode),
        "confidence_mode": str(args.confidence_mode),
        "blend_alpha": float(args.blend_alpha),
        "mapping_device": str(mapping_device),
        "cuda_device": (
            None if args.cuda_device is None else int(args.cuda_device)
        ),
        "mapping_chunk_size": int(args.mapping_chunk_size),
        "min_confidence": float(args.min_confidence),
        "foreground_mask_gate": bool(args.foreground_mask_gate),
        "foreground_threshold": float(args.foreground_threshold),
        "front_surface_gate": bool(args.front_surface_gate),
        "zbuffer_resolution": int(args.zbuffer_resolution),
        "front_depth_tolerance": float(args.front_depth_tolerance),
        "global_geometry_fixed": True,
        "point_birth_allowed": False,
        "coordinate_clamp_used": False,
        "bbox_or_centroid_normalization_used": False,
        "modified_channels": "base_color 0:3 only",
        "color_alignment_used": False,
        "material_postprocessing_used": False,
        "render": render_config,
        "dry_run": bool(args.dry_run),
        "reproduction_command": reproduction,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "effective_config.json", effective_config)
    print("[effective-config]")
    print(json.dumps(effective_config, indent=2, ensure_ascii=False))
    print("[reproduce]")
    print(reproduction)

    global_support = _load_global_support(global_checkpoint_path)
    immutable_hash_before = {
        "vertices_sha256": _tensor_sha256(global_support.vertices),
        "faces_sha256": _tensor_sha256(global_support.faces),
        "ovoxel_coords_sha256": _tensor_sha256(global_support.coords),
    }
    sorted_global_keys, sorted_global_rows = _global_key_index(
        global_support.coords
    )
    base_color_slice = global_support.layout["base_color"]

    transforms: Dict[int, TileCameraTransform] = {}
    tile_candidates: List[TileCandidates] = []
    for tile_id in tile_ids:
        paths = tile_inputs[tile_id]
        route_summary = _read_json(paths["route_summary"])
        transform = _load_tile_transform(paths["tile_dir"], tile_id)
        _validate_camera_consistency(
            global_camera=global_camera,
            transform=transform,
            route_summary=route_summary,
            tile_id=tile_id,
        )
        transforms[tile_id] = transform
        tile_support = _load_tile_support(
            paths["checkpoint"],
            global_layout=global_support.layout,
            global_channels=int(global_support.attrs.shape[1]),
        )
        foreground_mask = (
            _load_foreground_mask(paths["foreground_mask"])
            if args.foreground_mask_gate
            else None
        )
        mapped = _map_one_tile(
            tile_id=tile_id,
            tile_support=tile_support,
            transform=transform,
            global_camera=global_camera,
            global_sorted_keys=sorted_global_keys,
            global_sorted_rows=sorted_global_rows,
            base_color_slice=base_color_slice,
            mapping_device=mapping_device,
            chunk_size=int(args.mapping_chunk_size),
            confidence_mode=str(args.confidence_mode),
            min_confidence=float(args.min_confidence),
            foreground_mask=foreground_mask,
            foreground_threshold=float(args.foreground_threshold),
            front_surface_gate=bool(args.front_surface_gate),
            zbuffer_resolution=int(args.zbuffer_resolution),
            front_depth_tolerance=float(args.front_depth_tolerance),
        )
        tile_candidates.append(mapped)
        tile_output = output_dir / "tiles" / f"tile_{tile_id:02d}"
        _atomic_json(tile_output / "mapping_stats.json", mapped.stats)
        print(
            f"[tile-map] tile={tile_id:02d} "
            f"raw={mapped.stats['raw_local_ovoxel_rows']:,} "
            f"matched={mapped.stats['existing_global_support_match_rows']:,} "
            f"unique={mapped.global_rows.shape[0]:,}"
        )
        del tile_support

    candidates = _concatenate_candidates(tile_candidates)
    fusion = _fuse_candidates(
        candidates,
        fusion_mode=str(args.fusion_mode),
    )
    pair_diagnostics = _pair_overlap_diagnostics(tile_candidates)
    pair_csv = output_dir / "overlap_seam_diagnostics.csv"
    _write_csv(pair_csv, pair_diagnostics)
    _atomic_json(
        output_dir / "overlap_seam_diagnostics.json",
        {
            "definition": (
                "same immutable global C1024 key proposed by two different "
                "tiles after within-tile nearest-center selection"
            ),
            "pairs": pair_diagnostics,
        },
    )
    disagreement_maps = _save_disagreement_maps(
        tiles=tile_candidates,
        fusion=fusion,
        output_dir=output_dir,
        run_dir=run_dir,
    )

    sorted_order = fusion.sorted_candidate_order
    provenance_payload = {
        "format": "pixal3d_fixed_global_support_material_provenance_v1",
        "fusion_mode": str(args.fusion_mode),
        "confidence_mode": str(args.confidence_mode),
        "candidate_global_rows": candidates["global_rows"][
            sorted_order
        ].contiguous(),
        "candidate_global_keys": candidates["global_keys"][
            sorted_order
        ].contiguous(),
        "candidate_tile_ids": candidates["tile_ids"][
            sorted_order
        ].contiguous(),
        "candidate_local_rows": candidates["local_rows"][
            sorted_order
        ].contiguous(),
        "candidate_confidence": candidates["confidence"][
            sorted_order
        ].contiguous(),
        "candidate_tent_confidence": candidates["tent_confidence"][
            sorted_order
        ].contiguous(),
        "candidate_edge_confidence": candidates["edge_confidence"][
            sorted_order
        ].contiguous(),
        "candidate_center_distance_voxels": candidates[
            "center_distance_voxels"
        ][sorted_order].contiguous(),
        "candidate_group_ids": fusion.sorted_group_ids,
        "fused_global_rows": fusion.global_rows,
        "fused_global_keys": fusion.global_keys,
        "contributor_counts": fusion.contributor_counts,
        "winner_tile_ids": fusion.winner_tile_ids,
        "winner_local_rows": fusion.winner_local_rows,
        "winner_confidence": fusion.winner_confidence,
        "winner_center_distance_voxels": (
            fusion.winner_center_distance_voxels
        ),
        "disagreement_rgb_rms": fusion.disagreement_rgb_rms,
    }
    provenance_path = output_dir / "fusion_provenance.pt"
    _atomic_torch_save(provenance_path, provenance_payload)

    aggregate_mapping = {
        **fusion.stats,
        "global_support_rows": int(global_support.coords.shape[0]),
        "matched_global_fraction": float(
            fusion.global_rows.shape[0]
            / max(global_support.coords.shape[0], 1)
        ),
        "tiles": [item.stats for item in tile_candidates],
        "pair_diagnostics_json": str(
            output_dir / "overlap_seam_diagnostics.json"
        ),
        "pair_diagnostics_csv": str(pair_csv),
        "disagreement_maps": {
            str(key): value for key, value in disagreement_maps.items()
        },
        "provenance_checkpoint": str(provenance_path),
    }
    _atomic_json(output_dir / "mapping_summary.json", aggregate_mapping)

    immutable_hash_after_mapping = {
        "vertices_sha256": _tensor_sha256(global_support.vertices),
        "faces_sha256": _tensor_sha256(global_support.faces),
        "ovoxel_coords_sha256": _tensor_sha256(global_support.coords),
    }
    geometry_unchanged_after_mapping = (
        immutable_hash_after_mapping == immutable_hash_before
    )
    if not geometry_unchanged_after_mapping:
        raise RuntimeError("global geometry/support changed during mapping")

    if args.dry_run:
        summary = {
            "format": (
                "pixal3d_fixed_global_support_tile_material_fusion_summary_v1"
            ),
            "status": "dry_run_mapping_complete",
            "effective_config": effective_config,
            "mapping": aggregate_mapping,
            "immutable_geometry": {
                "before": immutable_hash_before,
                "after_mapping": immutable_hash_after_mapping,
                "unchanged": True,
            },
            "modified_attrs_checkpoint": None,
            "renders": None,
            "tile_crop_metrics": None,
        }
        _atomic_json(output_dir / "summary.json", summary)
        print(
            f"[dry-run-done] mapping only; summary={output_dir / 'summary.json'}"
        )
        return summary

    modified_attrs = _apply_material_blend(
        global_support.attrs,
        fusion,
        base_color_slice=base_color_slice,
        blend_alpha=float(args.blend_alpha),
    )
    attribute_invariants = _check_attribute_invariants(
        global_attrs=global_support.attrs,
        modified_attrs=modified_attrs,
        fusion=fusion,
        base_color_slice=base_color_slice,
    )
    global_base = global_support.attrs[
        fusion.global_rows,
        base_color_slice,
    ].to(torch.float32)
    final_base = modified_attrs[
        fusion.global_rows,
        base_color_slice,
    ].to(torch.float32)
    blend_change = torch.linalg.vector_norm(final_base - global_base, dim=1)
    attribute_invariants["base_color_change_l2"] = _quantiles(blend_change)

    modified_attrs_path = output_dir / "global_modified_attrs.pt"
    _atomic_torch_save(
        modified_attrs_path,
        {
            "format": "pixal3d_fixed_global_support_modified_attrs_v1",
            "source_global_checkpoint": str(global_support.path),
            "ovoxel_attrs": modified_attrs,
            "layout": global_support.serialized_layout,
            "modified_global_rows": fusion.global_rows,
            "modified_global_keys": fusion.global_keys,
            "candidate_base_colors_before_blend": (
                fusion.candidate_base_colors
            ),
            "base_colors_after_blend": final_base,
            "blend_alpha": float(args.blend_alpha),
            "fusion_mode": str(args.fusion_mode),
            "confidence_mode": str(args.confidence_mode),
            "geometry_reused_from_source_checkpoint": True,
            "invariants": attribute_invariants,
        },
    )

    if not torch.cuda.is_available():
        raise RuntimeError("native PBR rendering requires CUDA")
    render_device = torch.device(
        f"cuda:{torch.cuda.current_device()}"
    )
    envmap = load_envmap(
        str(render_config["envmap"]),
        device=render_device,
    )
    control_mesh = _build_mesh(
        global_support,
        global_support.attrs,
        device=render_device,
    )
    control_render = render_and_evaluate_mesh(
        control_mesh,
        camera_angle_x=float(global_camera["camera_angle_x"]),
        distance=float(global_camera["distance"]),
        output_dir=output_dir / "global_control",
        reference_image=canonical_1024,
        resolution=int(render_config["render_resolution"]),
        metric_resolution=int(render_config["metric_resolution"]),
        envmap=envmap,
        envmap_name=str(render_config["envmap"]),
        ssaa=int(render_config["ssaa"]),
        peel_layers=int(render_config["peel_layers"]),
        face_chunk_size=int(render_config["face_chunk_size"]),
        use_envmap_bg=bool(render_config["use_envmap_bg"]),
        lpips_net=str(render_config["lpips_net"]),
        metric_device=str(render_config["metric_device"]),
        skip_lpips=bool(render_config["skip_lpips"]),
    )
    fused_mesh = MeshWithVoxel(
        vertices=control_mesh.vertices,
        faces=control_mesh.faces,
        origin=global_support.origin,
        voxel_size=global_support.voxel_size,
        coords=control_mesh.coords,
        attrs=modified_attrs.to(render_device),
        voxel_shape=global_support.voxel_shape,
        layout=global_support.layout,
    )
    render_mesh_invariants = {
        "vertices_equal_before_fused_render": bool(
            torch.equal(fused_mesh.vertices, control_mesh.vertices)
        ),
        "faces_equal_before_fused_render": bool(
            torch.equal(fused_mesh.faces, control_mesh.faces)
        ),
        "ovoxel_coords_equal_before_fused_render": bool(
            torch.equal(fused_mesh.coords, control_mesh.coords)
        ),
    }
    if not all(render_mesh_invariants.values()):
        raise RuntimeError(
            "fused render mesh changed immutable global geometry/support"
        )
    fused_render = render_and_evaluate_mesh(
        fused_mesh,
        camera_angle_x=float(global_camera["camera_angle_x"]),
        distance=float(global_camera["distance"]),
        output_dir=output_dir / "global_fused",
        reference_image=canonical_1024,
        resolution=int(render_config["render_resolution"]),
        metric_resolution=int(render_config["metric_resolution"]),
        envmap=envmap,
        envmap_name=str(render_config["envmap"]),
        ssaa=int(render_config["ssaa"]),
        peel_layers=int(render_config["peel_layers"]),
        face_chunk_size=int(render_config["face_chunk_size"]),
        use_envmap_bg=bool(render_config["use_envmap_bg"]),
        lpips_net=str(render_config["lpips_net"]),
        metric_device=str(render_config["metric_device"]),
        skip_lpips=bool(render_config["skip_lpips"]),
    )
    render_mesh_invariants.update(
        {
            "vertices_equal_after_fused_render": bool(
                torch.equal(fused_mesh.vertices, control_mesh.vertices)
            ),
            "faces_equal_after_fused_render": bool(
                torch.equal(fused_mesh.faces, control_mesh.faces)
            ),
            "ovoxel_coords_equal_after_fused_render": bool(
                torch.equal(fused_mesh.coords, control_mesh.coords)
            ),
        }
    )
    if not all(render_mesh_invariants.values()):
        raise RuntimeError(
            "native renderer changed immutable global geometry/support"
        )
    tile_metric_records, crop_evaluator = _evaluate_tile_crops(
        run_dir=run_dir,
        output_dir=output_dir,
        tile_ids=tile_ids,
        transforms=transforms,
        control_render=Path(control_render["render_png"]),
        fused_render=Path(fused_render["render_png"]),
        metric_resolution=int(render_config["metric_resolution"]),
        lpips_net=str(render_config["lpips_net"]),
        metric_device=str(render_config["metric_device"]),
        skip_lpips=bool(render_config["skip_lpips"]),
    )
    if crop_evaluator is not None:
        crop_evaluator.model.cpu()
        del crop_evaluator
    contact_sheet = _make_contact_sheet(
        tile_metric_records,
        output_dir / "tile_24_26_27_contact_sheet.png",
    )
    _write_csv(
        output_dir / "tile_crop_metrics.csv",
        [
            {
                "tile_id": item["tile_id"],
                "box": item["box"],
                "control_psnr_db": item["control_metrics"]["psnr_db"],
                "control_ssim": item["control_metrics"]["ssim"],
                "control_lpips": item["control_metrics"]["lpips"],
                "fused_psnr_db": item["fused_metrics"]["psnr_db"],
                "fused_ssim": item["fused_metrics"]["ssim"],
                "fused_lpips": item["fused_metrics"]["lpips"],
                **item["fused_minus_control"],
            }
            for item in tile_metric_records
        ],
    )

    immutable_hash_after_render = {
        "vertices_sha256": _tensor_sha256(global_support.vertices),
        "faces_sha256": _tensor_sha256(global_support.faces),
        "ovoxel_coords_sha256": _tensor_sha256(global_support.coords),
    }
    geometry_unchanged_after_render = (
        immutable_hash_after_render == immutable_hash_before
    )
    if not geometry_unchanged_after_render:
        raise RuntimeError("global geometry/support changed during rendering")

    summary = {
        "format": "pixal3d_fixed_global_support_tile_material_fusion_summary_v1",
        "status": "success",
        "effective_config": effective_config,
        "mapping": aggregate_mapping,
        "attribute_invariants": attribute_invariants,
        "render_mesh_invariants": render_mesh_invariants,
        "immutable_geometry": {
            "before": immutable_hash_before,
            "after_mapping": immutable_hash_after_mapping,
            "after_render": immutable_hash_after_render,
            "unchanged": True,
        },
        "modified_attrs_checkpoint": str(modified_attrs_path),
        "global_control": control_render,
        "global_fused": fused_render,
        "global_fused_minus_control": _metric_delta(
            fused_render,
            control_render,
        ),
        "tile_crop_metrics": tile_metric_records,
        "tile_crop_metrics_csv": str(
            output_dir / "tile_crop_metrics.csv"
        ),
        "contact_sheet": contact_sheet,
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(
        f"[done] matched_global={fusion.global_rows.shape[0]:,} "
        f"summary={output_dir / 'summary.json'}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=(
            "v7 projective-tile run made with --save-decoded-support"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tile-ids",
        default=",".join(str(item) for item in DEFAULT_TILE_IDS),
        help="comma-separated tiles; default: 24,26,27",
    )
    parser.add_argument("--route-name", default=DEFAULT_ROUTE)
    parser.add_argument(
        "--fusion-mode",
        choices=("winner_center", "weighted_mean"),
        default="winner_center",
    )
    parser.add_argument(
        "--confidence-mode",
        choices=("tent", "edge"),
        default="tent",
        help=(
            "winner_center maximizes this image-center confidence; "
            "weighted_mean uses it as the averaging weight"
        ),
    )
    parser.add_argument("--blend-alpha", type=float, default=0.25)
    parser.add_argument("--min-confidence", type=float, default=1e-6)
    parser.add_argument("--mapping-device", default="cuda")
    parser.add_argument("--mapping-chunk-size", type=int, default=250000)
    parser.add_argument(
        "--foreground-mask-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument(
        "--front-surface-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--zbuffer-resolution", type=int, default=256)
    parser.add_argument(
        "--front-depth-tolerance",
        type=float,
        default=4.0 / GRID_C1024,
        help="camera-depth tolerance for the approximate tile z-buffer gate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="perform mapping/fusion statistics only; do not save attrs or render",
    )
    parser.add_argument("--cuda-device", type=int, default=None)

    # None means inherit the exact values recorded by the source v7 run.
    parser.add_argument("--envmap", default=None)
    parser.add_argument("--render-resolution", type=int, default=None)
    parser.add_argument("--metric-resolution", type=int, default=None)
    parser.add_argument("--ssaa", type=int, default=None)
    parser.add_argument("--peel-layers", type=int, default=None)
    parser.add_argument("--face-chunk-size", type=int, default=0)
    parser.add_argument(
        "--use-envmap-bg",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--lpips-net",
        choices=("alex", "vgg", "squeeze"),
        default="vgg",
    )
    parser.add_argument("--metric-device", default="cuda")
    parser.add_argument(
        "--skip-lpips",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"[fatal] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
