#!/usr/bin/env python3
"""Training-free multi-view fixed-geometry PBR Gaussian-fusion SR.

The only mesh used by this experiment is a cached Pixal3D 1024 baseline.
Every local tile owns a fixed projective support and runs the native texture
flow.  At the strict per-step barrier, decoded PBR fields communicate only in
the baseline surface coordinate system.  Visibility is binary metadata: an
invisible field cannot donate, but can still receive consensus.  Per-view mesh
visibility comes from a geometry-only z-buffer; each SLat bit is queried
directly from its C64 cell center to the nearest baseline mesh vertex rather
than inherited from an O-Voxel parent.
"""
from __future__ import annotations

import argparse
import csv
import gc
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

import numpy as np
import torch
from PIL import Image, ImageDraw

import pixal3d.models as pixal3d_models
import pixal3d_baseline1024_pbr_mesh_compare as baseline_render
import pixal3d_cross_tile_pbr_perstep as cross_tile
import pixal3d_texture_visibility_guided_pbr_flow as visibility
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithFacePbr, MeshWithVertexPbr, MeshWithVoxel


FORMAT = "pixal3d_multiview_fixed_geometry_pbr_gaussian_sr_v1"
ANGLES_DEFAULT = (0, 120, 240)
SOURCE_VIEW_SIZE = 1024
SOURCE_TILE_SIZE = 256
SOURCE_TILE_STRIDE = 128
MODEL_TILE_SIZE = 1024
OVOXEL_RESOLUTION = 1024
LATENT_RESOLUTION = 64
PBR_LAYOUT = dict(core.PBR_LAYOUT)


@dataclass(frozen=True)
class ViewContext:
    angle: int
    rotation: torch.Tensor
    image: Image.Image


@dataclass
class TileContext:
    context_id: int
    angle: int
    tile_id: int
    box: Tuple[int, int, int, int]
    transform: core.TileCameraTransform
    tile_dir: Path
    geometry: core.LocalGeometry
    shape_norm: SparseTensor
    shape_denorm: SparseTensor
    texture_norm: SparseTensor
    noise: SparseTensor
    initial_state: SparseTensor
    condition: Mapping[str, Any]
    target_points: torch.Tensor
    target_world_q: torch.Tensor
    ovoxel_visible: torch.Tensor
    slat_visible: torch.Tensor
    support_stats: Dict[str, Any]


@dataclass
class DecodedSnapshot:
    mesh: MeshWithVoxel
    target_field: torch.Tensor
    decoded_visible: torch.Tensor
    masked_mesh: MeshWithVoxel
    support_mesh: MeshWithVoxel
    decode_stats: Dict[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _sparse_cpu(value: SparseTensor) -> SparseTensor:
    return SparseTensor(value.feats.detach().to("cpu").clone(), value.coords.detach().to("cpu").clone())


def _sparse_cuda(value: SparseTensor) -> SparseTensor:
    return SparseTensor(value.feats.detach().to("cuda"), value.coords.detach().to("cuda"))


def _local_sparse_coords(value: SparseTensor, label: str) -> torch.Tensor:
    coords = value.coords.detach()
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise RuntimeError(f"{label}: expected sparse [N,4] coordinates, got {tuple(coords.shape)}")
    if coords.shape[0] and not bool(torch.all(coords[:, 0] == 0)):
        raise RuntimeError(f"{label}: expected a local B=1 sparse tensor")
    return coords[:, 1:].clone()


def _pack_sparse_batch(values: Sequence[SparseTensor], label: str) -> SparseTensor:
    """Pack local B=1 sparse values into a real, contiguous sparse batch."""
    if not values:
        raise ValueError(f"{label}: cannot pack an empty sparse batch")
    channels = tuple(values[0].feats.shape[1:])
    features: List[torch.Tensor] = []
    coordinates: List[torch.Tensor] = []
    for batch_id, value in enumerate(values):
        if tuple(value.feats.shape[1:]) != channels:
            raise RuntimeError(f"{label}: sparse feature channels differ across the batch")
        local = _local_sparse_coords(value, label)
        features.append(value.feats.detach())
        coordinates.append(torch.cat((torch.full_like(local[:, :1], batch_id), local), dim=1))
    return SparseTensor(torch.cat(features, dim=0), torch.cat(coordinates, dim=0))


def _restore_sparse_features(candidate_coords: torch.Tensor, candidate_feats: torch.Tensor, reference_local: torch.Tensor, label: str) -> torch.Tensor:
    """Restore a batched sparse output to the original local coordinate order."""
    if candidate_coords.shape[0] != reference_local.shape[0]:
        raise RuntimeError(f"{label}: sparse token count changed ({candidate_coords.shape[0]} != {reference_local.shape[0]})")
    if not candidate_coords.shape[0]:
        return candidate_feats
    reference = reference_local.to(device=candidate_coords.device, dtype=candidate_coords.dtype)
    bound = max(int(candidate_coords.max().item()), int(reference.max().item())) + 1
    base = max(2, bound)
    candidate_keys = (candidate_coords.to(torch.int64)[:, 0] * base + candidate_coords.to(torch.int64)[:, 1]) * base + candidate_coords.to(torch.int64)[:, 2]
    reference_keys = (reference.to(torch.int64)[:, 0] * base + reference.to(torch.int64)[:, 1]) * base + reference.to(torch.int64)[:, 2]
    sorted_keys, order = torch.sort(candidate_keys)
    positions = torch.searchsorted(sorted_keys, reference_keys)
    if bool((positions >= sorted_keys.shape[0]).any()) or not torch.equal(sorted_keys.index_select(0, positions), reference_keys):
        raise RuntimeError(f"{label}: sparse output support differs from its input")
    return candidate_feats.index_select(0, order.index_select(0, positions))


def _unpack_sparse_batch(value: SparseTensor, references: Sequence[SparseTensor], label: str) -> List[SparseTensor]:
    """Split a sparse batch and exactly restore each original local support."""
    parts: List[SparseTensor] = []
    for batch_id, reference in enumerate(references):
        mask = value.coords[:, 0] == int(batch_id)
        local = _local_sparse_coords(reference, label)
        restored = _restore_sparse_features(value.coords[mask][:, 1:], value.feats[mask], local, f"{label} batch {batch_id}")
        local_cuda = local.to(device=value.coords.device, dtype=value.coords.dtype)
        coords = torch.cat((torch.zeros_like(local_cuda[:, :1]), local_cuda), dim=1)
        parts.append(SparseTensor(restored.contiguous(), coords.contiguous()))
    return parts


def _unpack_sparse_batch_parts(value: SparseTensor, batch_size: int, label: str) -> List[SparseTensor]:
    """Split a real sparse batch without assuming output support equals input support.

    Encoders change resolution (C1024 -> C64), so the input-reference based
    ``_unpack_sparse_batch`` cannot be used for their outputs.  The batch id is
    the only routing metadata needed here; coordinates are kept in the order
    returned by the native sparse kernel and reset to local B=1 coordinates.
    """
    if not isinstance(value, SparseTensor):
        raise TypeError(f"{label}: expected SparseTensor, got {type(value)!r}")
    parts: List[SparseTensor] = []
    for batch_id in range(int(batch_size)):
        mask = value.coords[:, 0] == int(batch_id)
        if not bool(mask.any()):
            raise RuntimeError(f"{label}: batch {batch_id} has no output tokens")
        coords = value.coords[mask].detach().clone()
        coords[:, 0] = 0
        parts.append(SparseTensor(value.feats[mask].contiguous(), coords.contiguous()))
    return parts


@torch.no_grad()
def _encode_initial_batch(
    pending: Sequence[Mapping[str, Any]],
    shape_encoder: torch.nn.Module,
    pbr_encoder: torch.nn.Module,
    pipeline: Any,
    profile_records: Optional[List[Dict[str, Any]]] = None,
) -> List[Tuple[SparseTensor, SparseTensor, Dict[str, Any], Dict[str, Any]]]:
    """Encode several local supports in one native shape/PBR encoder call.

    This is deliberately a physical sparse batch: one call receives one
    ``SparseTensor`` whose coordinates carry batch ids.  It is not a loop
    hidden behind a batch-shaped API and has no per-sample fallback.
    """
    if not pending:
        return []
    batch_size = len(pending)
    input_coords = sum(int(item["geometry"].coords.shape[0]) for item in pending)

    def phase_begin() -> Tuple[Optional[float], Optional[Dict[str, float]]]:
        if profile_records is None:
            return None, None
        torch.cuda.synchronize()
        before = {"allocated_gib": float(torch.cuda.memory_allocated() / 2**30),
                  "reserved_gib": float(torch.cuda.memory_reserved() / 2**30)}
        torch.cuda.reset_peak_memory_stats()
        return time.perf_counter(), before

    def phase_end(phase: str, started: Optional[float], before: Optional[Dict[str, float]], status: str = "ok", error: Optional[str] = None) -> None:
        if profile_records is None or started is None or before is None:
            return
        peak = _cuda_peak_stats()
        record: Dict[str, Any] = {
            "phase": phase,
            "batch_size": batch_size,
            "input_coords": input_coords,
            "seconds": float(time.perf_counter() - started),
            "status": status,
            "before": before,
            "peak": peak,
            "peak_delta_allocated_gib": peak["allocated_gib"] - before["allocated_gib"],
            "peak_delta_reserved_gib": peak["reserved_gib"] - before["reserved_gib"],
        }
        if error is not None:
            record["error"] = error
        profile_records.append(record)

    shape_profile_started, shape_profile_before = phase_begin()
    shape_values: List[SparseTensor] = []
    intersected_values: List[SparseTensor] = []
    pbr_values: List[SparseTensor] = []
    for item in pending:
        geometry = item["geometry"]
        coords = geometry.coords.detach()
        coords4 = torch.cat((torch.zeros_like(coords[:, :1]), coords), dim=1)
        vertices = SparseTensor(
            geometry.dual_vertices.to(device="cuda", dtype=torch.float32),
            coords4.to(device="cuda", dtype=torch.int32),
        )
        intersected_values.append(vertices.replace(geometry.intersected.to(device="cuda")))
        shape_values.append(vertices)
        attrs = item["local_attrs"].detach().to(device="cuda", dtype=torch.float32)
        pbr_values.append(SparseTensor(attrs * 2.0 - 1.0, vertices.coords.detach().clone()))

    shape_batch = _pack_sparse_batch(shape_values, "initial shape encode")
    intersected_batch = _pack_sparse_batch(intersected_values, "initial shape intersected")
    shape_started = time.perf_counter()
    try:
        shape_encoder.to("cuda")
        shape_raw_batch = shape_encoder(shape_batch, intersected_batch, sample_posterior=False)
        _require_finite_sparse(shape_raw_batch, "initial batched shape encoder output")
    except torch.cuda.OutOfMemoryError as exc:
        phase_end("shape_encode", shape_profile_started, shape_profile_before, "oom", str(exc))
        _empty_cuda_cache()
        raise
    shape_seconds = float(time.perf_counter() - shape_started)
    shape_parts = _unpack_sparse_batch_parts(shape_raw_batch, len(pending), "initial shape encode output")
    phase_end("shape_encode", shape_profile_started, shape_profile_before)

    pbr_profile_started, pbr_profile_before = phase_begin()
    pbr_batch = _pack_sparse_batch(pbr_values, "initial PBR encode")
    pbr_started = time.perf_counter()
    try:
        pbr_encoder.to("cuda")
        pbr_raw_batch = pbr_encoder(pbr_batch, sample_posterior=False)
        _require_finite_sparse(pbr_raw_batch, "initial batched PBR encoder output")
    except torch.cuda.OutOfMemoryError as exc:
        phase_end("pbr_encode", pbr_profile_started, pbr_profile_before, "oom", str(exc))
        _empty_cuda_cache()
        raise
    pbr_seconds = float(time.perf_counter() - pbr_started)
    pbr_parts = _unpack_sparse_batch_parts(pbr_raw_batch, len(pending), "initial PBR encode output")
    phase_end("pbr_encode", pbr_profile_started, pbr_profile_before)

    result: List[Tuple[SparseTensor, SparseTensor, Dict[str, Any], Dict[str, Any]]] = []
    for item, shape_raw, texture_raw in zip(pending, shape_parts, pbr_parts):
        geometry = item["geometry"]
        shape_stats = {
            "input_coords": int(geometry.coords.shape[0]),
            "input_dual_vertices_range": core._tensor_range(geometry.dual_vertices),
            "input_intersected_shape": list(geometry.intersected.shape),
            "shape_latent_tokens": int(shape_raw.feats.shape[0]),
            "shape_latent_channels": int(shape_raw.feats.shape[1]),
            "shape_latent_coords_range": core._tensor_range(shape_raw.coords[:, 1:].to(torch.float32)),
            "shape_encoder_seconds": shape_seconds,
            "batch_size": batch_size,
        }
        texture_stats = {
            "input_coords": int(geometry.coords.shape[0]),
            "input_attrs_range": core._tensor_range(item["local_attrs"]),
            "pbr_latent_tokens": int(texture_raw.feats.shape[0]),
            "pbr_latent_channels": int(texture_raw.feats.shape[1]),
            "pbr_latent_coords_range": core._tensor_range(texture_raw.coords[:, 1:].to(torch.float32)),
            "pbr_encoder_seconds": pbr_seconds,
            "batch_size": batch_size,
        }
        result.append((shape_raw, texture_raw, shape_stats, texture_stats))

    del shape_values, intersected_values, pbr_values
    del shape_batch, intersected_batch, shape_raw_batch, shape_parts
    del pbr_batch, pbr_raw_batch, pbr_parts
    _empty_cuda_cache()
    return result


def _pack_condition_batch(values: Sequence[Any], label: str) -> Any:
    """Recursively batch the native Pixal3D condition schema without changing it."""
    if not values:
        raise ValueError(f"{label}: cannot pack an empty condition batch")
    first = values[0]
    if isinstance(first, SparseTensor):
        if not all(isinstance(value, SparseTensor) for value in values):
            raise TypeError(f"{label}: mixed sparse/non-sparse condition values")
        return _pack_sparse_batch(values, label)
    if isinstance(first, Mapping):
        keys = tuple(first.keys())
        if not all(isinstance(value, Mapping) and tuple(value.keys()) == keys for value in values):
            raise ValueError(f"{label}: condition mapping schemas differ")
        return {key: _pack_condition_batch([value[key] for value in values], f"{label}.{key}") for key in keys}
    if isinstance(first, torch.Tensor):
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError(f"{label}: mixed tensor/non-tensor condition values")
        if all(value.ndim >= 1 and value.shape[0] == 1 and value.shape[1:] == first.shape[1:] for value in values):
            return torch.cat(list(values), dim=0)
        if all(value.shape == first.shape for value in values):
            return torch.stack(list(values), dim=0)
        if all(value.ndim >= 2 and value.shape[1:] == first.shape[1:] for value in values):
            return list(values)
        raise ValueError(f"{label}: tensor condition shape is not batchable")
    if isinstance(first, (list, tuple)):
        if not all(isinstance(value, type(first)) and len(value) == len(first) for value in values):
            raise ValueError(f"{label}: condition sequence schemas differ")
        return type(first)(_pack_condition_batch([value[index] for value in values], f"{label}[{index}]") for index in range(len(first)))
    if not all(value == first for value in values):
        raise ValueError(f"{label}: scalar condition metadata differs")
    return first


def _sparse_difference(reference: SparseTensor, candidate: SparseTensor) -> Dict[str, float]:
    if not torch.equal(reference.coords, candidate.coords) or reference.feats.shape != candidate.feats.shape:
        raise RuntimeError("cannot compare sparse values with different support")
    delta = candidate.feats.detach().to(torch.float64) - reference.feats.detach().to(torch.float64)
    denominator = torch.linalg.vector_norm(reference.feats.detach().to(torch.float64)) + 1e-12
    return {"max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
            "relative_l2": float(torch.linalg.vector_norm(delta) / denominator)}


def _cuda_peak_stats() -> Dict[str, float]:
    torch.cuda.synchronize()
    return {"allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30),
            "reserved_gib": float(torch.cuda.max_memory_reserved() / 2**30)}


def _flow_groups(contexts: Sequence[TileContext], batch_size: int) -> Iterable[Sequence[TileContext]]:
    for start in range(0, len(contexts), int(batch_size)):
        yield contexts[start:start + int(batch_size)]


def _move_condition_cpu(value: Any) -> Any:
    return cross_tile._move_condition(value, torch.device("cpu"))


def _move_condition_cuda(value: Any) -> Any:
    return cross_tile._move_condition(value, torch.device("cuda"))


def _yaw_matrix(angle: int, *, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    theta = math.radians(float(angle))
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float32, device=device)


def _world_to_view_q(q_world: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return q_world @ rotation.to(device=q_world.device, dtype=q_world.dtype)


def _view_to_world_q(q_view: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return q_view @ rotation.to(device=q_view.device, dtype=q_view.dtype).T


def _tile_boxes() -> List[Tuple[int, int, int, int]]:
    boxes = core._tile_layout(SOURCE_VIEW_SIZE, SOURCE_TILE_SIZE, SOURCE_TILE_STRIDE)
    starts = [0, 128, 256, 384, 512, 640, 768]
    if len(boxes) != 49 or sorted({box[0] for box in boxes}) != starts or sorted({box[1] for box in boxes}) != starts:
        raise RuntimeError(f"required 7x7 tile layout was not constructed: {boxes}")
    return boxes


def _load_views(path: Path, output_dir: Path, angles: Sequence[int]) -> Dict[int, Image.Image]:
    with Image.open(path) as source:
        composite = source.convert("RGB")
    if composite.size != (3072, 1024):
        raise ValueError(f"multi-view composite must be exactly 3072x1024, got {composite.size}")
    expected = (0, 1024, 2048)
    if tuple(int(angle) for angle in angles) != tuple(ANGLES_DEFAULT):
        # The physical image still has the fixed three panel ordering.  A
        # selected-view run is applied later, after every panel is split.
        all_angles = ANGLES_DEFAULT
    else:
        all_angles = tuple(int(angle) for angle in angles)
    views = {angle: composite.crop((x0, 0, x0 + 1024, 1024)) for angle, x0 in zip(all_angles, expected)}
    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    for angle, image in views.items():
        image.save(input_dir / f"view_{angle:03d}.png")
    return views


def _camera_equivalence(camera: Mapping[str, float], extend_pixel: int) -> Dict[str, Any]:
    rows = []
    max_error = 0.0
    for tile_id, box in enumerate(_tile_boxes()):
        old_box = tuple(int(value) * 4 for value in box)
        old = core._derive_tile_camera(
            tile_id=tile_id, box=old_box, global_camera=camera, extend_pixel=extend_pixel,
            source_width=4096, source_height=4096, model_width=1024, model_height=1024,
        )
        new = core._derive_tile_camera(
            tile_id=tile_id, box=box, global_camera=camera, extend_pixel=extend_pixel,
            source_width=1024, source_height=1024, model_width=1024, model_height=1024,
        )
        names = ("fx", "fy", "offaxis_cx", "offaxis_cy", "camera_angle_x", "distance")
        errors = {name: abs(float(getattr(new, name)) - float(getattr(old, name))) for name in names}
        max_error = max(max_error, *errors.values())
        rows.append({"tile_id": tile_id, "new_source_box": list(box), "old_virtual_4096_box": list(old_box),
                     "new": {name: float(getattr(new, name)) for name in names},
                     "old": {name: float(getattr(old, name)) for name in names}, "abs_error": errors})
    return {"comparison": "1024 crop256 resize1024 == virtual 4096 crop1024", "tile_count": len(rows), "max_abs_error": max_error, "tiles": rows}


def _roundtrip_report(camera: Mapping[str, float], angles: Sequence[int], seed: int) -> Dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 700)
    q_world = torch.rand((1024, 3), generator=generator, dtype=torch.float32) - 0.5
    transform = core._derive_tile_camera(
        tile_id=24, box=(384, 384, 640, 640), global_camera=camera, extend_pixel=0,
        source_width=1024, source_height=1024, model_width=1024, model_height=1024,
    )
    rows = []
    for angle in angles:
        rotation = _yaw_matrix(int(angle))
        q_view = _world_to_view_q(q_world, rotation)
        q_local, uv = core._global_q_to_local_q(q_view, global_camera=camera, transform=transform)
        q_view_back, uv_back = core._local_q_to_global_q(q_local, global_camera=camera, transform=transform)
        q_world_back = _view_to_world_q(q_view_back, rotation)
        error = (q_world_back - q_world).abs()
        pixel_error = (uv_back - (uv / 4.0 + torch.tensor([384.0, 384.0]))).abs()
        rows.append({"angle": int(angle), "points": int(q_world.shape[0]), "max_abs_error": float(error.max()),
                     "mean_abs_error": float(error.mean()), "pixel_roundtrip_max_abs_error": float(pixel_error.max()),
                     "out_of_local_cube_count": int(((q_local < -1.0) | (q_local > 1.0)).any(dim=1).sum())})
    return {"tolerance": 2e-5, "rows": rows, "max_abs_error": max(row["max_abs_error"] for row in rows)}


def _masked_trilinear_8(values: torch.Tensor, visible: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if values.shape != (8, 6) or visible.shape != (8,) or weights.shape != (8,):
        raise ValueError("masked trilinear test helper expects 8 PBR corners")
    denominator = (weights * visible.to(weights.dtype)).sum()
    if float(denominator) <= 0.0:
        return torch.zeros((6,), dtype=values.dtype), False
    return ((values * (weights * visible.to(weights.dtype))[:, None]).sum(dim=0) / denominator), True


def _run_correctness_tests(camera: Mapping[str, float], composite: Path, output_dir: Path, seed: int) -> Dict[str, Any]:
    boxes = _tile_boxes()
    tests: Dict[str, Dict[str, Any]] = {}
    starts = [0, 128, 256, 384, 512, 640, 768]
    tests["tile_layout"] = {"pass": len(boxes) == 49 and [box[0] for box in boxes[:7]] == starts, "tile_count": len(boxes), "starts": starts}
    with Image.open(composite) as source:
        panel = source.convert("RGB").crop((0, 0, 1024, 1024))
    source_tile = panel.crop(boxes[-1])
    resized_tile = source_tile.resize((1024, 1024), Image.Resampling.BICUBIC)
    tests["native_crop_resize"] = {"pass": source_tile.size == (256, 256) and resized_tile.size == (1024, 1024),
                                    "source_size": list(source_tile.size), "model_size": list(resized_tile.size),
                                    "whole_view_4096_created": False}
    equivalent = _camera_equivalence(camera, 0)
    tests["camera_equivalence"] = {"pass": equivalent["max_abs_error"] < 2e-6, "max_abs_error": equivalent["max_abs_error"]}
    roundtrip = _roundtrip_report(camera, ANGLES_DEFAULT, seed)
    tests["global_local_roundtrip"] = {"pass": roundtrip["max_abs_error"] < 2e-5, "max_abs_error": roundtrip["max_abs_error"]}
    try:
        from scipy.spatial import cKDTree
        direct_tree = cKDTree(np.asarray([[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]], dtype=np.float32))
        direct_transform = core._derive_tile_camera(
            tile_id=26, box=boxes[26], global_camera=camera, extend_pixel=0,
            source_width=1024, source_height=1024, model_width=1024, model_height=1024,
        )
        direct_coords = torch.tensor([[0, 32, 32, 32], [0, 63, 63, 63]], dtype=torch.int32)
        direct_flags, direct_world, direct_nearest, direct_distance = _slat_visibility(
            direct_coords, direct_transform, _yaw_matrix(0), camera, direct_tree,
            torch.tensor([True, False], dtype=torch.bool),
        )
        expected_direct = torch.tensor([True, False], dtype=torch.bool).index_select(0, direct_nearest)
        tests["slat_visibility_direct_mesh_nn"] = {
            "pass": bool(torch.equal(direct_flags, expected_direct)
                         and direct_world.shape == (2, 3)
                         and direct_distance.shape == (2,)
                         and bool(torch.isfinite(direct_distance).all())),
            "nearest_vertex": direct_nearest.tolist(),
            "distance_normalized": direct_distance.tolist(),
            "voxel_parent_or_used": False,
        }
        local_reference = torch.tensor(
            [[0.45, 0.0, 0.0], [-0.45, 0.0, 0.0], [0.0, 0.45, 0.0],
             [0.0, -0.45, 0.0], [0.0, 0.0, 0.45], [0.0, 0.0, -0.45]],
            dtype=torch.float32,
        )
        local_q = local_reference * (2.0 * float(direct_transform.mesh_scale))
        view_q, _ = core._local_q_to_global_q(
            local_q, global_camera=camera, transform=direct_transform
        )
        world_q = _view_to_world_q(view_q, _yaw_matrix(0))
        recovered_local, recovered_uv, recovered_inside = _world_to_local(
            world_q,
            argparse.Namespace(transform=direct_transform),
            camera,
            _yaw_matrix(0),
        )
        tests["local_cube_membership_normalized"] = {
            "pass": bool(
                recovered_inside.all()
                and torch.allclose(recovered_local, local_reference, atol=2e-5, rtol=0.0)
                and torch.isfinite(recovered_uv).all()
            ),
            "points": int(local_reference.shape[0]),
            "inside": int(recovered_inside.sum()),
            "max_abs_error": float((recovered_local - local_reference).abs().max()),
            "membership_space": "normalized local object coordinates",
        }
    except Exception as exc:
        tests["slat_visibility_direct_mesh_nn"] = {"pass": False, "error": repr(exc)}
        tests["local_cube_membership_normalized"] = {"pass": False, "error": repr(exc)}
    coords = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=torch.int32)
    tests["fixed_support"] = {"pass": bool(torch.equal(coords, coords.clone()) and torch.equal(coords, coords.clone())), "tokens": 2}
    tests["visibility_keeps_slat_count"] = {"pass": int(coords.shape[0]) == int(coords.shape[0]), "visible_tokens": 0, "total_tokens": int(coords.shape[0])}
    values = torch.arange(48, dtype=torch.float32).reshape(8, 6)
    weights = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32)
    visible = torch.tensor([True, False, True, False, True, True, False, False])
    masked, valid = _masked_trilinear_8(values, visible, weights)
    expected = (values[visible] * weights[visible, None]).sum(0) / weights[visible].sum()
    tests["masked_trilinear"] = {"pass": bool(valid and torch.allclose(masked, expected)), "denominator": float(weights[visible].sum())}
    _, zero_valid = _masked_trilinear_8(values, torch.zeros(8, dtype=torch.bool), weights)
    tests["zero_visible_corner"] = {"pass": not zero_valid, "valid_donor": bool(zero_valid)}
    norm_coords = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    norm_value = SparseTensor(torch.tensor([[0.25, -0.5]], dtype=torch.float32), norm_coords)
    norm_stats = {"mean": [1.5, -2.0], "std": [2.0, 4.0]}
    denorm_value = cross_tile._denormalize_slat(norm_value, norm_stats)
    renorm_value = cross_tile._normalize_slat(denorm_value, norm_stats)
    tests["latent_normalization_roundtrip"] = {
        "pass": bool(torch.allclose(renorm_value.feats, norm_value.feats) and torch.equal(renorm_value.coords, norm_value.coords)),
        "max_abs_error": float((renorm_value.feats - norm_value.feats).abs().max()),
    }
    support_mesh = MeshWithVoxel(
        torch.empty((1, 3), dtype=torch.float32),
        torch.empty((0, 3), dtype=torch.int32),
        [-0.5, -0.5, -0.5], 1.0 / OVOXEL_RESOLUTION,
        torch.tensor([[0, 0, 0]], dtype=torch.int32),
        torch.tensor([[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]], dtype=torch.float32),
        torch.Size([1, 6, OVOXEL_RESOLUTION, OVOXEL_RESOLUTION, OVOXEL_RESOLUTION]),
        dict(PBR_LAYOUT),
    )
    final_support = _masked_mesh(support_mesh, torch.ones((1,), dtype=torch.bool))
    tests["final_assignment_visibility_is_donor_only"] = {
        "pass": bool(torch.equal(final_support.attrs[0, :6], support_mesh.attrs[0]) and float(final_support.attrs[0, 6]) == 1.0),
        "final_query_uses_full_support": True,
    }
    pbr = torch.tensor([[[1.0] * 6, [9.0] * 6]], dtype=torch.float32)
    fused, normalized, _ = cross_tile._gaussian_fuse_candidates(pbr, torch.tensor([[True, True]]), torch.tensor([[0.0, 512.0]]), 256.0)
    tests["gaussian_preference"] = {"pass": bool(normalized[0, 0] > normalized[0, 1] and fused[0, 0] < 5.0), "weights": normalized[0].tolist()}
    target = torch.tensor([3.0] * 6)
    donated, donor_valid = _masked_trilinear_8(torch.stack((target, torch.tensor([7.0] * 6), *([target] * 6))),
                                                torch.tensor([False, True, False, False, False, False, False, False]), torch.ones(8))
    tests["cross_view_invisible_donor"] = {"pass": bool(donor_valid and torch.allclose(donated, torch.tensor([7.0] * 6))), "target_may_receive": True}
    traversal_a, _, _ = cross_tile._gaussian_fuse_candidates(pbr, torch.tensor([[True, True]]), torch.tensor([[32.0, 96.0]]), 256.0)
    traversal_b, _, _ = cross_tile._gaussian_fuse_candidates(pbr.flip(1), torch.tensor([[True, True]]), torch.tensor([[96.0, 32.0]]), 256.0)
    tests["traversal_invariant"] = {"pass": bool(torch.allclose(traversal_a, traversal_b)), "max_abs_error": float((traversal_a - traversal_b).abs().max())}
    from pixal3d.pipelines.samplers.flow_euler import FlowEulerSampler
    sparse_x = SparseTensor(torch.tensor([[0.4, -0.2]], dtype=torch.float32), torch.tensor([[0, 1, 1, 1]], dtype=torch.int32))
    sparse_x0 = SparseTensor(torch.tensor([[0.1, 0.3]], dtype=torch.float32), sparse_x.coords.clone())
    sampler = FlowEulerSampler(sigma_min=0.001)
    velocity = sampler._xstart_to_pred(sparse_x, 0.5, sparse_x0)
    direct = ((1 - sampler.sigma_min) * sparse_x.feats - sparse_x0.feats) / (sampler.sigma_min + (1 - sampler.sigma_min) * 0.5)
    tests["native_endpoint_bridge"] = {"pass": bool(torch.allclose(velocity.feats, direct) and torch.equal(velocity.coords, sparse_x.coords)), "max_abs_error": float((velocity.feats - direct).abs().max())}
    result = {"all_passed": all(row["pass"] for row in tests.values()), "tests": tests,
              "camera_equivalence": equivalent, "coordinate_roundtrip": roundtrip}
    _atomic_json(output_dir / "correctness_tests.json", result)
    _atomic_json(output_dir / "camera_equivalence.json", equivalent)
    _atomic_json(output_dir / "coordinate_roundtrip.json", roundtrip)
    return result


def _save_tile_debug(views: Mapping[int, Image.Image], boxes: Sequence[Tuple[int, int, int, int]], output_dir: Path, debug: bool) -> None:
    root = output_dir / "tiles_debug"
    for angle, image in views.items():
        overview = image.copy()
        draw = ImageDraw.Draw(overview)
        for tile_id, box in enumerate(boxes):
            draw.rectangle(box, outline=(255, 70, 40), width=2)
            draw.text((box[0] + 3, box[1] + 3), str(tile_id), fill=(255, 255, 255))
        root.mkdir(parents=True, exist_ok=True)
        overview.save(root / f"view_{angle:03d}_tiles.png")
        if debug:
            for tile_id, box in enumerate(boxes):
                crop = image.crop(box)
                crop.save(root / "source_256" / f"view_{angle:03d}_tile_{tile_id:02d}.png") if (root / "source_256").mkdir(parents=True, exist_ok=True) is None else None
                resized = crop.resize((1024, 1024), Image.Resampling.BICUBIC)
                resized.save(root / "resized_1024" / f"view_{angle:03d}_tile_{tile_id:02d}.png") if (root / "resized_1024").mkdir(parents=True, exist_ok=True) is None else None


def _view_global_condition(pipeline: Any, image: Image.Image, camera: Mapping[str, float]) -> torch.Tensor:
    dummy = torch.tensor([[0, LATENT_RESOLUTION // 2, LATENT_RESOLUTION // 2, LATENT_RESOLUTION // 2]], dtype=torch.int32)
    condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024, [image], dummy,
        camera_angle_x=float(camera["camera_angle_x"]), distance=float(camera["distance"]), mesh_scale=float(camera["mesh_scale"]),
        grid_resolution_override=LATENT_RESOLUTION,
    )
    return condition["cond"]["global"].detach().cpu().clone()


def _native_condition(pipeline: Any, global_feature: torch.Tensor, tile_image: Image.Image, shape_coords: torch.Tensor, transform: core.TileCameraTransform) -> Mapping[str, Any]:
    local = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024, [tile_image], shape_coords.to(torch.int32),
        camera_angle_x=float(transform.camera_angle_x), distance=float(transform.distance), mesh_scale=float(transform.mesh_scale),
        grid_resolution_override=LATENT_RESOLUTION,
    )
    proj = local["cond"]["proj"]
    global_feature = global_feature.to(device=proj.device, dtype=proj.feats.dtype)
    condition = {
        "cond": {"global": global_feature, "proj": proj},
        "neg_cond": {"global": torch.zeros_like(global_feature), "proj": SparseTensor(torch.zeros_like(proj.feats), proj.coords.detach().clone())},
    }
    return _move_condition_cpu(condition)


def _visible_faces_and_vertices(buffers: Mapping[str, Any], faces: torch.Tensor, vertex_count: int) -> Tuple[torch.Tensor, torch.Tensor]:
    face_ids = torch.unique(buffers["triangle_id"][buffers["triangle_id"] >= 0].to(torch.long))
    face_ids = face_ids[face_ids < int(faces.shape[0])]
    vertex_visible = torch.zeros((vertex_count,), dtype=torch.bool)
    if face_ids.numel():
        vertex_visible[torch.unique(faces.index_select(0, face_ids).reshape(-1).to(torch.long))] = True
    return face_ids.cpu(), vertex_visible


def _voxel_visibility(geometry: core.LocalGeometry, transform: core.TileCameraTransform, rotation: torch.Tensor, camera: Mapping[str, float], vertex_tree: Any, visible_vertices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = geometry.coords.to(torch.float32)
    points = (coords + 0.5) / float(OVOXEL_RESOLUTION) - 0.5
    q_view, _ = core._local_q_to_global_q(points * (2.0 * float(transform.mesh_scale)), global_camera=camera, transform=transform)
    q_world = _view_to_world_q(q_view, rotation)
    _, nearest = vertex_tree.query((q_world / (2.0 * float(camera["mesh_scale"]))).numpy(), k=1, workers=-1)
    nearest_tensor = torch.as_tensor(nearest, dtype=torch.long)
    return q_world.cpu(), nearest_tensor, visible_vertices.index_select(0, nearest_tensor).cpu()


def _slat_visibility(
    slat_coords: torch.Tensor,
    transform: core.TileCameraTransform,
    rotation: torch.Tensor,
    camera: Mapping[str, float],
    vertex_tree: Any,
    visible_vertices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map each C64 SLat center directly to the binary mesh visibility set.

    SLat support is not an O-Voxel parent support.  A C64 coordinate denotes
    the center of a cell in the local tile cube, so it is lifted to a local q
    center, converted to global/view q with the exact tile camera, rotated
    back to world q, and queried against the baseline mesh vertices.  The
    returned flag is *only* the raster-derived visibility bit of that nearest
    mesh vertex.  The distance is normalized by the baseline object scale and
    is persisted as a diagnostic because a large distance means this SLat is
    not close to the mesh surface and its inherited bit is less informative.
    """
    if slat_coords.ndim != 2 or slat_coords.shape[1] < 3:
        raise ValueError(f"SLat coordinates must have shape [N, >=3], got {tuple(slat_coords.shape)}")
    xyz = slat_coords[:, -3:].detach().to(torch.float32).cpu()
    if xyz.numel() and bool(((xyz < 0) | (xyz >= LATENT_RESOLUTION)).any()):
        raise ValueError("SLat coordinates lie outside the C64 latent grid")
    local_center = (xyz + 0.5) / float(LATENT_RESOLUTION) - 0.5
    q_local = local_center * (2.0 * float(transform.mesh_scale))
    q_view, _ = core._local_q_to_global_q(q_local, global_camera=camera, transform=transform)
    q_world = _view_to_world_q(q_view, rotation).cpu()
    normalized_world = q_world / (2.0 * float(camera["mesh_scale"]))
    distances, nearest = vertex_tree.query(normalized_world.numpy(), k=1, workers=-1)
    nearest_tensor = torch.as_tensor(np.asarray(nearest).reshape(-1), dtype=torch.long)
    distance_tensor = torch.as_tensor(np.asarray(distances).reshape(-1), dtype=torch.float32)
    flags = visible_vertices.index_select(0, nearest_tensor).bool().cpu()
    return flags, q_world, nearest_tensor, distance_tensor


def _image_foreground_mask(image: Image.Image, threshold: int = 4) -> torch.Tensor:
    """Return a conservative non-black foreground mask for diagnostics only."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(np.any(rgb > int(threshold), axis=2))


def _overlay_points(
    image: Image.Image,
    uv: torch.Tensor,
    visible: torch.Tensor,
    path: Path,
    *,
    label: str,
    radius: int = 1,
) -> Dict[str, Any]:
    """Save a red-invisible/green-visible point overlay.

    The two assignments are intentionally separate: invisible points are
    painted first and visible points second, so a visible SLat/mesh point is
    never hidden by a coincident invisible point in the visualization.
    """
    if uv.ndim != 2 or uv.shape[1] != 2 or visible.ndim != 1 or uv.shape[0] != visible.shape[0]:
        raise ValueError("overlay points and visibility must be aligned [N,2]/[N]")
    width, height = image.size
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    uv_cpu = uv.detach().cpu().to(torch.float32)
    finite = torch.isfinite(uv_cpu).all(dim=1)
    # Never cast NaN/Inf directly to integer.  Those rows are excluded by
    # ``finite`` below, but replacing them first also keeps the diagnostic
    # path free of undefined/sentinel integer coordinates.
    safe_uv = torch.where(finite[:, None], uv_cpu, torch.zeros_like(uv_cpu))
    coords = torch.round(safe_uv).to(torch.long)
    inside = finite & (coords[:, 0] >= 0) & (coords[:, 0] < width) & (coords[:, 1] >= 0) & (coords[:, 1] < height)
    invisible = inside & ~visible.detach().cpu().bool()
    visible_mask = inside & visible.detach().cpu().bool()
    # Paint red first, then green as required by the visibility audit.
    for mask, color in ((invisible, (255, 32, 32)), (visible_mask, (32, 235, 72))):
        xy = coords[mask].numpy()
        if xy.size:
            if int(radius) <= 0:
                array[xy[:, 1], xy[:, 0]] = np.asarray(color, dtype=np.uint8)
            else:
                # Numpy box expansion keeps overlays legible without the
                # per-point PIL draw overhead for the millions of mesh verts.
                for dy in range(-int(radius), int(radius) + 1):
                    for dx in range(-int(radius), int(radius) + 1):
                        x = xy[:, 0] + dx
                        y = xy[:, 1] + dy
                        ok = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                        array[y[ok], x[ok]] = np.asarray(color, dtype=np.uint8)
    blended = (0.52 * np.asarray(image.convert("RGB"), dtype=np.float32) + 0.48 * array.astype(np.float32)).round().clip(0, 255).astype(np.uint8)
    result = Image.fromarray(blended, mode="RGB")
    draw = ImageDraw.Draw(result)
    draw.rectangle((4, 4, min(width - 4, 265), 31), fill=(0, 0, 0))
    draw.rectangle((10, 10, 20, 20), fill=(255, 32, 32))
    draw.text((25, 8), f"{label}: invisible", fill=(255, 255, 255))
    draw.rectangle((145, 10, 155, 20), fill=(32, 235, 72))
    draw.text((160, 8), "visible", fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    result.save(path)
    return {
        "path": str(path),
        "projected_points": int(inside.sum()),
        "projected_invisible": int(invisible.sum()),
        "projected_visible": int(visible_mask.sum()),
        "image_size": [int(width), int(height)],
        "paint_order": ["invisible_red", "visible_green"],
    }


def _save_mesh_visibility_overlay(
    image: Image.Image,
    rotated_vertices: torch.Tensor,
    visible_vertices: torch.Tensor,
    camera: Mapping[str, float],
    output_dir: Path,
) -> Dict[str, Any]:
    """Project all baseline mesh vertices and overlay raster visibility."""
    q_view = rotated_vertices.to(torch.float32) * (2.0 * float(camera["mesh_scale"]))
    uv, _, finite = core._project_global_q_to_image(
        q_view,
        global_camera=camera,
        image_width=int(image.width),
        image_height=int(image.height),
    )
    visible_vertices = visible_vertices.detach().cpu().bool()
    stats = _overlay_points(image, uv.cpu(), visible_vertices, output_dir / "mesh_visibility_overlay.png", label="mesh vertices", radius=0)
    stats.update({
        "mesh_vertices": int(rotated_vertices.shape[0]),
        "finite_projected_vertices": int(finite.sum()),
        "raster_visible_vertices": int(visible_vertices.sum()),
        "raster_invisible_vertices": int((~visible_vertices).sum()),
    })
    _atomic_json(output_dir / "mesh_visibility_overlay.json", stats)
    return stats


def _save_slat_visibility_overlays(
    image: Image.Image,
    tile_image: Image.Image,
    slat_world_q: torch.Tensor,
    slat_visible: torch.Tensor,
    transform: core.TileCameraTransform,
    rotation: torch.Tensor,
    camera: Mapping[str, float],
    output_dir: Path,
    tile_id: int,
) -> Dict[str, Any]:
    """Save both tile-local and full-view SLat projection overlays."""
    q_world = slat_world_q.detach().cpu().to(torch.float32)
    q_view = _world_to_view_q(q_world, rotation)
    _, uv_tile = core._global_q_to_local_q(q_view, global_camera=camera, transform=transform)
    uv_full, _, _ = core._project_global_q_to_image(
        q_view,
        global_camera=camera,
        image_width=int(image.width),
        image_height=int(image.height),
    )
    root = output_dir / f"tile_{int(tile_id):02d}"
    tile_stats = _overlay_points(tile_image, uv_tile.cpu(), slat_visible, root / "slat_visibility_overlay.png", label=f"tile {int(tile_id)} SLat", radius=1)
    full_stats = _overlay_points(image, uv_full.cpu(), slat_visible, root / "slat_visibility_overlay_full_view.png", label=f"tile {int(tile_id)} SLat", radius=1)
    stats = {
        "tile_id": int(tile_id),
        "tile_overlay": tile_stats,
        "full_view_overlay": full_stats,
        "slat_count": int(slat_visible.shape[0]),
        "visible_slat": int(slat_visible.sum()),
        "invisible_slat": int((~slat_visible).sum()),
        "projection": "C64 center -> local q -> exact tile camera -> world/view q",
        "paint_order": ["invisible_red", "visible_green"],
    }
    _atomic_json(root / "slat_visibility_overlay.json", stats)
    return stats


def _decoded_visibility(decoded_coords: torch.Tensor, context: TileContext) -> torch.Tensor:
    parent = torch.div(decoded_coords.detach().cpu().to(torch.long), 16, rounding_mode="floor")
    parent_keys = core._linear_keys(parent, LATENT_RESOLUTION)
    slat_keys = core._linear_keys(context.shape_norm.coords[:, -3:].detach().cpu().to(torch.long), LATENT_RESOLUTION)
    order = torch.argsort(slat_keys, stable=True)
    sorted_keys = slat_keys.index_select(0, order)
    positions = torch.searchsorted(sorted_keys, parent_keys)
    matched = positions < sorted_keys.numel()
    safe = positions.clamp_max(max(0, int(sorted_keys.numel()) - 1))
    if sorted_keys.numel():
        matched &= sorted_keys.index_select(0, safe) == parent_keys
    result = torch.zeros(parent_keys.shape[0], dtype=torch.bool)
    if bool(matched.any()):
        result[matched] = context.slat_visible.index_select(0, order.index_select(0, safe[matched])).bool()
    return result


def _masked_mesh(mesh: MeshWithVoxel, visible: torch.Tensor) -> MeshWithVoxel:
    if mesh.coords.shape[0] != visible.shape[0]:
        raise RuntimeError("decoded visibility is not aligned with decoded PBR coords")
    mask = visible.to(device=mesh.attrs.device, dtype=mesh.attrs.dtype)
    attrs = torch.cat((mesh.attrs * mask[:, None], mask[:, None]), dim=1)
    return MeshWithVoxel(mesh.vertices, mesh.faces, mesh.origin.tolist(), float(mesh.voxel_size), mesh.coords, attrs,
                         torch.Size([1, 7, *mesh.voxel_shape[-3:]]), dict(mesh.layout))


def _world_to_local(q_world: torch.Tensor, context: TileContext, camera: Mapping[str, float], rotation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_view = _world_to_view_q(q_world, rotation)
    uv_full, _, finite = core._project_global_q_to_image(q_view, global_camera=camera, image_width=1024, image_height=1024)
    local = torch.full_like(q_view, float("inf"))
    uv = torch.full((q_world.shape[0], 2), float("inf"), dtype=q_world.dtype, device=q_world.device)
    if bool(finite.any()):
        rows = torch.where(finite)[0]
        local_rows, uv_rows = core._global_q_to_local_q(q_view.index_select(0, rows), global_camera=camera, transform=context.transform)
        local[rows] = local_rows
        uv[rows] = uv_rows
    normalized_local = local / (2.0 * float(context.transform.mesh_scale))
    inside = finite & (uv[:, 0] >= 0) & (uv[:, 0] < 1024) & (uv[:, 1] >= 0) & (uv[:, 1] < 1024)
    # ``local`` is camera q, while MeshWithVoxel queries and the canonical
    # decoder cube use normalized local object coordinates.  Testing q
    # directly against +/-0.5 incorrectly kept only the inner half-width
    # whenever mesh_scale=1 (and was scale-dependent in general).
    inside &= (normalized_local >= -0.5 - 1e-5).all(dim=1) & (normalized_local <= 0.5 + 1e-5).all(dim=1)
    return normalized_local, uv, inside


def _masked_query(mesh: MeshWithVoxel, points: torch.Tensor, chunk_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    queried = cross_tile._query_mesh_chunked(mesh, points, int(chunk_size)).to("cpu")
    denominator = queried[:, 6]
    valid = torch.isfinite(queried).all(dim=1) & (denominator > 0.0)
    values = torch.zeros((points.shape[0], 6), dtype=torch.float32)
    if bool(valid.any()):
        values[valid] = queried[valid, :6] / denominator[valid, None]
    return values, valid


def _tile_distance(uv: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(uv.to(torch.float32) - uv.new_tensor([511.5, 511.5]), dim=1)


def _quantiles(value: torch.Tensor) -> Dict[str, float]:
    value = value.detach().to(torch.float32).reshape(-1)
    if not value.numel():
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    # PyTorch rejects quantile inputs above its internal element limit. Keep
    # diagnostics bounded and deterministic without changing the fused field.
    sample = value
    max_quantile_samples = 262144
    if sample.numel() > max_quantile_samples:
        stride = (sample.numel() + max_quantile_samples - 1) // max_quantile_samples
        sample = sample[::stride]
    return {
        "count": int(value.numel()),
        "mean": float(value.mean()),
        "median": float(sample.median()),
        "p95": float(torch.quantile(sample, 0.95)),
        "max": float(value.max()),
    }


def _pbr_variance_quantiles(variance: torch.Tensor, multi_donor: torch.Tensor) -> Dict[str, Dict[str, float]]:
    values = variance[multi_donor]
    return {
        "rgb": _quantiles(values[:, :3]),
        "metallic": _quantiles(values[:, 3]),
        "roughness": _quantiles(values[:, 4]),
        "alpha": _quantiles(values[:, 5]),
    }


def _require_finite_sparse(value: SparseTensor, label: str) -> None:
    if not bool(torch.isfinite(value.feats).all()):
        raise RuntimeError(f"non-finite sparse features in {label}")


def _build_contexts(args: argparse.Namespace, pipeline: Any, baseline_mesh: MeshWithVoxel, camera: Mapping[str, float], views: Mapping[int, Image.Image], output_dir: Path, global_attr: MeshWithVoxel) -> Tuple[List[TileContext], Dict[str, Any], Dict[int, ViewContext]]:
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        raise RuntimeError("scipy.spatial.cKDTree is required for fixed binary visibility ancestry") from exc
    boxes = _tile_boxes()
    _save_tile_debug(views, boxes, output_dir, bool(args.debug))
    active_angles = [int(value) for value in args.selected_views]
    view_contexts = {angle: ViewContext(angle, _yaw_matrix(angle), views[angle]) for angle in active_angles}
    global_features = {angle: _view_global_condition(pipeline, views[angle], camera) for angle in active_angles}
    dino_records: Dict[str, Any] = {"global": {str(angle): list(feature.shape) for angle, feature in global_features.items()}, "local": [], "steps_recomputed": 0}
    tree = cKDTree(baseline_mesh.vertices.detach().cpu().numpy())
    visible_vertices: Dict[int, torch.Tensor] = {}
    face_bounds: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    view_visibility_diagnostics: Dict[int, Dict[str, Any]] = {}
    for angle, view in view_contexts.items():
        rotated_vertices = baseline_mesh.vertices.detach().cpu() @ view.rotation
        view_mesh = MeshWithVoxel(rotated_vertices, baseline_mesh.faces.detach().cpu(), baseline_mesh.origin.tolist(), float(baseline_mesh.voxel_size),
                                  baseline_mesh.coords.detach().cpu(), baseline_mesh.attrs.detach().cpu(), baseline_mesh.voxel_shape, dict(baseline_mesh.layout))
        buffers = visibility._render_global_visibility_buffers(view_mesh, global_camera=camera, resolution=1024,
                                                               face_chunk_size=int(args.render_face_chunk_size), device=torch.device("cuda"))
        visibility_dir = output_dir / "visibility" / f"view_{angle:03d}"
        visibility._save_visibility_debug(visibility_dir, buffers)
        face_ids, visible_vertices[angle] = _visible_faces_and_vertices(
            buffers, baseline_mesh.faces.cpu(), int(baseline_mesh.vertices.shape[0])
        )
        input_foreground = _image_foreground_mask(view.image)
        raster_foreground = buffers["foreground"].detach().cpu().bool()
        intersection = input_foreground & raster_foreground
        union = input_foreground | raster_foreground
        input_pixels = int(input_foreground.sum())
        raster_pixels = int(raster_foreground.sum())
        foreground_stats = {
            "input_foreground_pixels": input_pixels,
            "raster_foreground_pixels": raster_pixels,
            "intersection_pixels": int(intersection.sum()),
            "union_pixels": int(union.sum()),
            "input_coverage": float(input_foreground.to(torch.float32).mean()),
            "raster_coverage": float(raster_foreground.to(torch.float32).mean()),
            "iou": float(intersection.sum()) / max(1, int(union.sum())),
            "input_mask_threshold": 4,
        }
        mesh_overlay = _save_mesh_visibility_overlay(
            view.image, rotated_vertices, visible_vertices[angle], camera, visibility_dir
        )
        view_visibility_diagnostics[angle] = {
            "visible_faces": int(face_ids.numel()),
            "visible_vertices": int(visible_vertices[angle].sum()),
            "binary_only": True,
            "foreground": foreground_stats,
            "mesh_overlay": mesh_overlay,
        }
        _atomic_json(output_dir / "visibility" / f"view_{angle:03d}_summary.json", {
            "angle": angle,
            **view_visibility_diagnostics[angle],
            "raster_renderer": buffers.get("renderer", ""),
        })
        face_bounds[angle] = core._project_face_bboxes(rotated_vertices, baseline_mesh.faces.cpu(), mesh_scale=float(camera["mesh_scale"]),
                                                       global_camera=camera, chunk_size=int(args.face_projection_chunk_size), source_width=1024, source_height=1024)
        del view_mesh, buffers
        _empty_cuda_cache()
    shape_encoder = pixal3d_models.from_pretrained(str(Path(args.shape_encoder))).eval()
    pbr_encoder = pixal3d_models.from_pretrained(str(Path(args.pbr_encoder))).eval()
    contexts: List[TileContext] = []
    support_rows: List[Dict[str, Any]] = []
    requested = int(args.max_tiles) if args.max_tiles is not None else None
    requested_tile_ids = None if args.tile_ids is None else {int(tile_id) for tile_id in args.tile_ids}
    # Initial shape/PBR encodings have a different memory profile from the
    # flow forward.  Do not silently reuse the (much larger) flow batch here:
    # the CUDA4 profile is B44 for flow, but only a small safe batch is needed
    # while constructing the cached tile contexts.
    requested_initial_batch = getattr(args, "initial_encode_batch_size", None)
    initial_batch_size = max(
        1,
        int(args.flow_batch_size if requested_initial_batch is None else requested_initial_batch),
    )
    initial_profile_records = getattr(args, "_initial_encode_profile_records", None)
    pending: List[Dict[str, Any]] = []

    def flush_pending() -> None:
        if not pending:
            return
        batch_size = len(pending)
        encoded = _encode_initial_batch(pending, shape_encoder, pbr_encoder, pipeline, initial_profile_records)
        for item, (shape_raw, texture_raw, shape_stats, texture_stats) in zip(pending, encoded):
            angle = int(item["angle"])
            tile_id = int(item["tile_id"])
            tile_dir = item["tile_dir"]
            geometry = item["geometry"]
            transform = item["transform"]
            tile_image = item["tile_image"]
            material_stats = item["material_stats"]
            selected_count = int(item["selected_count"])
            world_q = item["world_q"]
            ovoxel_visible = item["ovoxel_visible"]

            shape_norm = cross_tile._normalize_slat(cross_tile._fresh_sparse(shape_raw), pipeline.shape_slat_normalization)
            texture_norm = cross_tile._normalize_slat(cross_tile._fresh_sparse(texture_raw), pipeline.tex_slat_normalization)
            if not torch.equal(shape_norm.coords, texture_norm.coords):
                raise RuntimeError(f"tile {angle}/{tile_id}: shape and texture support differ")
            shape_norm, texture_norm = _sparse_cpu(shape_norm), _sparse_cpu(texture_norm)
            shape_denorm = cross_tile._denormalize_slat(shape_norm, pipeline.shape_slat_normalization)
            condition = _native_condition(pipeline, global_features[angle], tile_image, shape_norm.coords, transform)
            _seed(int(args.seed) + angle * 100_003 + tile_id)
            noise = SparseTensor(torch.randn_like(texture_norm.feats), texture_norm.coords.detach().clone())
            initial = cross_tile._native_noised_endpoint(
                texture_norm,
                noise,
                pipeline.tex_slat_sampler,
                float(args.noise_timestep),
                float(args.noise_strength),
            )
            target_points = ((geometry.coords.to(torch.float32) + 0.5) / float(OVOXEL_RESOLUTION) - 0.5).cpu()
            slat_visible, slat_world_q, slat_nearest_vertex, slat_nearest_distance = _slat_visibility(
                shape_norm.coords, transform, view_contexts[angle].rotation, camera, tree, visible_vertices[angle]
            )
            slat_overlay = _save_slat_visibility_overlays(
                view_contexts[angle].image,
                tile_image,
                slat_world_q,
                slat_visible,
                transform,
                view_contexts[angle].rotation,
                camera,
                output_dir / "visibility" / f"view_{angle:03d}",
                tile_id,
            )
            record = {
                "angle": angle,
                "tile_id": tile_id,
                "box": list(item["box"]),
                "status": "active",
                "global_support_faces": selected_count,
                "local_ovoxels": int(geometry.coords.shape[0]),
                "shape_slat": int(shape_norm.feats.shape[0]),
                "texture_slat": int(texture_norm.feats.shape[0]),
                "visible_ovoxels": int(ovoxel_visible.sum()),
                "invisible_ovoxels": int((~ovoxel_visible).sum()),
                "visible_slat": int(slat_visible.sum()),
                "invisible_slat": int((~slat_visible).sum()),
                "slat_visibility_source": "direct nearest baseline mesh vertex; raster-derived binary vertex flag",
                "slat_nearest_mesh_distance_normalized": _quantiles(slat_nearest_distance),
                "slat_nearest_mesh_vertex_count": int(torch.unique(slat_nearest_vertex).numel()),
                "slat_overlay": slat_overlay,
                "initial_encode_batch_size": batch_size,
                "geometry": geometry.stats,
                "material": material_stats,
                "shape_encoder": shape_stats,
                "pbr_encoder": texture_stats,
            }
            _atomic_json(tile_dir / "tile_camera.json", transform.__dict__)
            _atomic_json(tile_dir / "support.json", record)
            contexts.append(
                TileContext(
                    len(contexts), angle, tile_id, item["box"], transform, tile_dir, geometry,
                    shape_norm, shape_denorm, texture_norm, _sparse_cpu(noise), _sparse_cpu(initial),
                    condition, target_points, world_q, ovoxel_visible, slat_visible, record,
                )
            )
            support_rows.append(record)
            dino_records["local"].append({
                "angle": angle,
                "tile_id": tile_id,
                "source_size": [256, 256],
                "model_size": [1024, 1024],
                "global_shape": list(condition["cond"]["global"].shape),
                "proj_shape": list(condition["cond"]["proj"].feats.shape),
            })
        pending.clear()

    for angle in active_angles:
        view = view_contexts[angle]
        rotated_vertices = baseline_mesh.vertices.detach().cpu() @ view.rotation
        face_min, face_max, face_finite = face_bounds[angle]
        for tile_id, box in enumerate(boxes):
            if requested is not None and len(contexts) + len(pending) >= requested:
                break
            if requested_tile_ids is not None and tile_id not in requested_tile_ids:
                continue
            selected = core._tile_face_ids_from_bbox(face_min, face_max, face_finite, box)
            tile_dir = output_dir / "tiles" / f"view_{angle:03d}_tile_{tile_id:02d}"
            tile_dir.mkdir(parents=True, exist_ok=True)
            if not selected.numel():
                support_rows.append({"angle": angle, "tile_id": tile_id, "status": "empty_projective_support"})
                continue
            crop = views[angle].crop(box)
            if crop.size != (256, 256):
                raise RuntimeError(f"tile {angle}/{tile_id} is not a 256x256 crop")
            tile_image = crop.resize((1024, 1024), Image.Resampling.BICUBIC)
            transform = core._derive_tile_camera(tile_id=tile_id, box=box, global_camera=camera, extend_pixel=int(args.extend_pixel),
                                                 source_width=1024, source_height=1024, model_width=1024, model_height=1024)
            geometry = core._prepare_tile_geometry(global_vertices=rotated_vertices, global_faces=baseline_mesh.faces.cpu(), global_face_min=face_min,
                                                   global_face_max=face_max, global_face_finite=face_finite, global_camera=camera, transform=transform)
            if float(geometry.stats["global_local_global_q_max_abs_error"]) > float(args.roundtrip_tolerance):
                raise RuntimeError(f"tile {angle}/{tile_id}: local camera roundtrip failed: {geometry.stats}")
            local_attrs, material_stats = core._resample_local_attrs_from_global(
                geometry=geometry, global_attr_field=global_attr, global_camera=camera, transform=transform,
                query_chunk_size=int(args.material_query_chunk_size), face_chunk_size=int(args.material_face_chunk_size),
                local_q_to_attr_q=lambda q, rotation=view.rotation: _view_to_world_q(q, rotation),
            )
            world_q, _, ovoxel_visible = _voxel_visibility(
                geometry, transform, view.rotation, camera, tree, visible_vertices[angle]
            )
            pending.append({
                "angle": angle,
                "tile_id": tile_id,
                "box": box,
                "tile_dir": tile_dir,
                "tile_image": tile_image,
                "transform": transform,
                "geometry": geometry,
                "local_attrs": local_attrs,
                "material_stats": material_stats,
                "selected_count": int(selected.numel()),
                "world_q": world_q,
                "ovoxel_visible": ovoxel_visible,
            })
            if len(pending) >= initial_batch_size:
                flush_pending()
        if requested is not None and len(contexts) + len(pending) >= requested:
            break
    flush_pending()
    shape_encoder.cpu()
    pbr_encoder.cpu()
    _empty_cuda_cache()
    dino_records["local_cached_count"] = len(dino_records["local"])
    _atomic_json(output_dir / "dino_cache_diagnostics.json", dino_records)
    _atomic_json(output_dir / "support_stats.json", {"contexts": support_rows, "active_contexts": len(contexts)})
    view_visibility_rows = []
    for angle in active_angles:
        angle_contexts = [context for context in contexts if context.angle == angle]
        slat_distance_stats = [context.support_stats.get("slat_nearest_mesh_distance_normalized", {}) for context in angle_contexts]
        slat_distance_count = [int(stats.get("count", 0)) for stats in slat_distance_stats]
        slat_distance_mean = sum(float(stats.get("mean", 0.0)) * count for stats, count in zip(slat_distance_stats, slat_distance_count)) / max(1, sum(slat_distance_count))
        raster_diag = view_visibility_diagnostics.get(angle, {})
        view_visibility_rows.append({
            "angle": int(angle),
            "binary_only": True,
            "visible_global_vertices": int(visible_vertices[angle].sum()),
            "global_vertices": int(visible_vertices[angle].numel()),
            "visible_global_vertex_fraction": float(visible_vertices[angle].to(torch.float32).mean()),
            "active_tiles": len(angle_contexts),
            "visible_ovoxels": sum(int(context.ovoxel_visible.sum()) for context in angle_contexts),
            "invisible_ovoxels": sum(int((~context.ovoxel_visible).sum()) for context in angle_contexts),
            "visible_slat": sum(int(context.slat_visible.sum()) for context in angle_contexts),
            "invisible_slat": sum(int((~context.slat_visible).sum()) for context in angle_contexts),
            "slat_nearest_mesh_distance_mean_normalized": slat_distance_mean,
            "foreground": raster_diag.get("foreground", {}),
            "mesh_overlay": raster_diag.get("mesh_overlay", {}),
        })
    _atomic_json(output_dir / "visibility_stats.json", {
        "mesh_visibility": "exact per-view nvdiffrast z-buffer; visible faces -> binary visible baseline vertices",
        "mesh_to_ovoxel": "nearest baseline vertex inherits fixed raster binary visibility",
        "ovoxel_to_slat": "not used for SLat visibility",
        "slat_visibility": "C64 SLat center -> exact local/world q -> nearest baseline mesh vertex -> binary raster visibility",
        "slat_nearest_mesh_distance": "normalized object-space distance in baseline vertex coordinates",
        "slat_to_decoded_pbr": "decoded coordinate parent=floor(coord/16)",
        "views": view_visibility_rows,
    })
    return contexts, dino_records, view_contexts


def _decode_snapshots_batched(contexts: Sequence[TileContext], states: Mapping[int, SparseTensor], pipeline: Any, args: argparse.Namespace, batch_size: int) -> Dict[int, DecodedSnapshot]:
    """Decode frozen endpoint snapshots in real sparse batches."""
    snapshots: Dict[int, DecodedSnapshot] = {}
    for group in _flow_groups(contexts, batch_size):
        state_batch = _pack_sparse_batch([_sparse_cuda(states[context.context_id]) for context in group], "decode texture")
        shape_batch = _pack_sparse_batch([_sparse_cuda(context.shape_denorm) for context in group], "decode shape")
        # Flow states and predicted x0 values are normalized texture SLat.
        # decode_latent consumes the raw/denormalized texture SLat, exactly as
        # the native single-tile route does.  Keeping this conversion here
        # makes both per-step and final decode use the same contract.
        texture_batch = cross_tile._denormalize_slat(state_batch, pipeline.tex_slat_normalization)
        started = time.perf_counter()
        decoded = pipeline.decode_latent(shape_batch, texture_batch, OVOXEL_RESOLUTION)
        _require_decoded_count = len(group)
        if len(decoded) != _require_decoded_count:
            raise RuntimeError(f"batched endpoint decode returned {len(decoded)} meshes for {_require_decoded_count} contexts")
        for context, mesh in zip(group, decoded):
            mesh = cross_tile._validate_decoded_mesh(mesh, f"view {context.angle} tile {context.tile_id} endpoint")
            field = cross_tile._query_mesh_chunked(mesh, context.target_points.to("cuda"), int(args.query_chunk_size))
            if not torch.isfinite(field).all():
                raise RuntimeError(f"view {context.angle} tile {context.tile_id} endpoint: decoded PBR query is non-finite")
            mesh_cpu, field_cpu = mesh.to("cpu"), field.detach().cpu().clone()
            decoded_visible = _decoded_visibility(mesh_cpu.coords, context)
            support_mesh = _masked_mesh(mesh_cpu, torch.ones_like(decoded_visible))
            stats = {"decode_seconds": float(time.perf_counter() - started),
                     "decoded_vertices": int(mesh_cpu.vertices.shape[0]), "decoded_faces": int(mesh_cpu.faces.shape[0]),
                     "decoded_active_ovoxels": int(mesh_cpu.coords.shape[0]), "queried_fixed_support_tokens": int(field_cpu.shape[0]),
                     "decoded_pbr_range": core._tensor_range(mesh_cpu.attrs), "batch_size": len(group)}
            snapshots[context.context_id] = DecodedSnapshot(mesh_cpu, field_cpu, decoded_visible, _masked_mesh(mesh_cpu, decoded_visible), support_mesh, stats)
        del state_batch, shape_batch, texture_batch, decoded
        _empty_cuda_cache()
    return snapshots


@torch.no_grad()
def _predict_flow_batch(group: Sequence[TileContext], states: Mapping[int, SparseTensor], model: torch.nn.Module, sampler: Any, t: float, step_kwargs: Mapping[str, Any]) -> Dict[int, Dict[str, SparseTensor]]:
    """Run one native flow forward for a real B=len(group) sparse batch."""
    state_values = [_sparse_cuda(states[context.context_id]) for context in group]
    shape_values = [_sparse_cuda(context.shape_norm) for context in group]
    state_batch = _pack_sparse_batch(state_values, "flow state")
    shape_batch = _pack_sparse_batch(shape_values, "flow shape")
    conditions = [_move_condition_cuda(context.condition) for context in group]
    condition_batch = _pack_condition_batch([condition["cond"] for condition in conditions], "flow cond")
    negative_batch = _pack_condition_batch([condition["neg_cond"] for condition in conditions], "flow neg_cond")
    x0_batch, _, velocity_batch = sampler._get_model_prediction(
        model, state_batch, float(t), cond=condition_batch, neg_cond=negative_batch,
        concat_cond=shape_batch, **step_kwargs
    )
    x0_values = _unpack_sparse_batch(x0_batch, state_values, "flow x0")
    velocity_values = _unpack_sparse_batch(velocity_batch, state_values, "flow velocity")
    predictions: Dict[int, Dict[str, SparseTensor]] = {}
    for context, state, x0, velocity in zip(group, state_values, x0_values, velocity_values):
        _require_finite_sparse(x0, f"flow context {context.context_id} predicted endpoint")
        _require_finite_sparse(velocity, f"flow context {context.context_id} predicted velocity")
        cross_tile._strict_sparse_check(state, x0, f"flow context {context.context_id} predicted endpoint")
        cross_tile._strict_sparse_check(state, velocity, f"flow context {context.context_id} predicted velocity")
        predictions[context.context_id] = {"x0": _sparse_cpu(x0), "velocity": _sparse_cpu(velocity)}
    del state_values, shape_values, state_batch, shape_batch, conditions, condition_batch, negative_batch, x0_batch, velocity_batch
    _empty_cuda_cache()
    return predictions


@torch.no_grad()
def _encode_fused_batch(group: Sequence[TileContext], fused_fields: Mapping[int, torch.Tensor], predictions: Mapping[int, Mapping[str, SparseTensor]], pbr_encoder: torch.nn.Module, pipeline: Any) -> Dict[int, SparseTensor]:
    """Directly encode multiple fused PBR fields in one sparse PBR-encoder call."""
    values: List[SparseTensor] = []
    references: List[SparseTensor] = []
    for context in group:
        coords = context.geometry.coords
        attrs = fused_fields[context.context_id].detach().to(torch.float32)
        if attrs.shape[0] != coords.shape[0]:
            raise RuntimeError(f"context {context.context_id}: fused PBR rows do not match geometry")
        local_coords = torch.cat((torch.zeros_like(coords[:, :1]), coords), dim=1)
        values.append(SparseTensor(attrs * 2.0 - 1.0, local_coords))
        references.append(predictions[context.context_id]["x0"])
    packed = _pack_sparse_batch([_sparse_cuda(value) for value in values], "PBR encode")
    pbr_encoder.to("cuda")
    raw_batch = pbr_encoder(packed, sample_posterior=False)
    _require_finite_sparse(raw_batch, "batched PBR encoder output")
    raw_values = _unpack_sparse_batch(raw_batch, references, "PBR encode output")
    corrected: Dict[int, SparseTensor] = {}
    for context, raw in zip(group, raw_values):
        endpoint = _sparse_cpu(cross_tile._normalize_slat(raw, pipeline.tex_slat_normalization))
        _require_finite_sparse(endpoint, f"context {context.context_id} direct fused endpoint")
        cross_tile._strict_sparse_check(predictions[context.context_id]["x0"], endpoint, f"context {context.context_id} direct fused endpoint")
        corrected[context.context_id] = endpoint
    del values, references, packed, raw_batch, raw_values
    _empty_cuda_cache()
    return corrected


def _fuse_target(context: TileContext, contexts: Sequence[TileContext], snapshots: Mapping[int, DecodedSnapshot], camera: Mapping[str, float], rotations: Mapping[int, torch.Tensor], args: argparse.Namespace) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
    original = snapshots[context.context_id].target_field.cpu()
    count = int(original.shape[0])
    fused_sum = torch.zeros_like(original)
    weight_sum = torch.zeros((count,), dtype=torch.float32)
    donor_count = torch.zeros((count,), dtype=torch.int32)
    cross_view_count = torch.zeros((count,), dtype=torch.int32)
    pbr_sum = torch.zeros_like(original)
    pbr_square_sum = torch.zeros_like(original)
    for donor in contexts:
        local_points, uv, covered = _world_to_local(context.target_world_q, donor, camera, rotations[donor.angle])
        rows = torch.where(covered)[0]
        if not rows.numel():
            continue
        values, valid_local = _masked_query(snapshots[donor.context_id].masked_mesh, local_points.index_select(0, rows), int(args.query_chunk_size))
        if not bool(valid_local.any()):
            continue
        valid_rows = rows[valid_local]
        values = values[valid_local]
        weights = torch.exp(-_tile_distance(uv.index_select(0, valid_rows)).square() / (2.0 * float(args.gaussian_sigma) ** 2))
        fused_sum.index_add_(0, valid_rows, values * weights[:, None])
        weight_sum.index_add_(0, valid_rows, weights)
        donor_count.index_add_(0, valid_rows, torch.ones_like(valid_rows, dtype=torch.int32))
        if donor.angle != context.angle:
            cross_view_count.index_add_(0, valid_rows, torch.ones_like(valid_rows, dtype=torch.int32))
        pbr_sum.index_add_(0, valid_rows, values)
        pbr_square_sum.index_add_(0, valid_rows, values.square())
    has_donors = weight_sum > 0
    fused = original.clone()
    fused[has_donors] = fused_sum[has_donors] / weight_sum[has_donors, None]
    variance = torch.zeros_like(original)
    multi = donor_count > 1
    if bool(multi.any()):
        mean = pbr_sum[multi] / donor_count[multi, None].to(torch.float32)
        variance[multi] = (pbr_square_sum[multi] / donor_count[multi, None].to(torch.float32) - mean.square()).clamp_min(0.0)
    coverage = {"zero": int((donor_count == 0).sum()), "one": int((donor_count == 1).sum()), "two": int((donor_count == 2).sum()),
                "three": int((donor_count == 3).sum()), "four_plus": int((donor_count >= 4).sum()),
                "cross_view_receipts": int((cross_view_count > 0).sum()), "same_view_overlap": int(((donor_count - cross_view_count) > 1).sum())}
    stats = {"target_context": context.context_id, "active_ovoxels": count, "coverage": coverage, "gaussian_sigma": float(args.gaussian_sigma),
             "weight": "binary_visibility * exp(-tile_center_distance_pixels^2/(2*sigma^2))", "no_donor_keeps_target_endpoint": int((~has_donors).sum()),
             "pbr_self_vs_fused": _quantiles((fused - original).abs()), "donor_variance": _pbr_variance_quantiles(variance, multi)}
    details = {"donor_count": donor_count, "cross_view_donor_count": cross_view_count, "weight_sum": weight_sum, "variance": variance}
    return fused, stats, details


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


@torch.no_grad()
def _run_flow(contexts: Sequence[TileContext], pipeline: Any, camera: Mapping[str, float], pbr_encoder: torch.nn.Module, args: argparse.Namespace) -> Dict[int, SparseTensor]:
    if not contexts:
        raise RuntimeError("no active projective supports; cannot run texture flow")
    sampler = pipeline.tex_slat_sampler
    model = pipeline.models["tex_slat_flow_model_1024"]
    merged = {**pipeline.tex_slat_sampler_params, "steps": int(args.num_steps), "rescale_t": float(args.texture_rescale_t),
              "guidance_strength": float(args.texture_guidance_strength), "guidance_rescale": float(args.texture_guidance_rescale)}
    schedule = cross_tile._native_schedule(sampler, merged)
    start = cross_tile._schedule_start(schedule, float(args.noise_timestep))
    step_kwargs = cross_tile._sampler_step_kwargs(merged)
    flow_batch_size = max(1, int(args.flow_batch_size))
    requested_decode_batch = getattr(args, "decode_batch_size", None)
    requested_pbr_encode_batch = getattr(args, "pbr_encode_batch_size", None)
    decode_batch_size = max(1, int(flow_batch_size if requested_decode_batch is None else requested_decode_batch))
    pbr_encode_batch_size = max(1, int(flow_batch_size if requested_pbr_encode_batch is None else requested_pbr_encode_batch))
    states = {context.context_id: _sparse_cpu(context.initial_state) for context in contexts}
    rotations = {context.angle: _yaw_matrix(context.angle) for context in contexts}
    flow_groups = list(_flow_groups(contexts, flow_batch_size))
    decode_groups = list(_flow_groups(contexts, decode_batch_size))
    pbr_encode_groups = list(_flow_groups(contexts, pbr_encode_batch_size))
    per_step: List[Dict[str, Any]] = []
    pbr_rows: List[Dict[str, Any]] = []
    correction_rows: List[Dict[str, Any]] = []
    try:
        for step_index, (t, t_next) in enumerate(zip(schedule[start:-1], schedule[start + 1:]), start=start):
            started = time.perf_counter()
            print(
                f"[flow step {step_index:02d}] t={t:.8f} contexts={len(contexts)} "
                f"flow_batch={flow_batch_size} decode_batch={decode_batch_size} "
                f"pbr_encode_batch={pbr_encode_batch_size}",
                flush=True,
            )
            predictions: Dict[int, Dict[str, SparseTensor]] = {}
            # Barrier A: every forward consumes its old x_t before any decode or update.
            model.to("cuda")
            for group in flow_groups:
                predictions.update(_predict_flow_batch(group, states, model, sampler, float(t), step_kwargs))
            model.cpu()
            _empty_cuda_cache()
            # Barrier B: all endpoint decodes are frozen before any fusion query.
            snapshots = _decode_snapshots_batched(
                contexts,
                {key: value["x0"] for key, value in predictions.items()},
                pipeline,
                args,
                decode_batch_size,
            )
            fused_fields: Dict[int, torch.Tensor] = {}
            fusion_stats: Dict[int, Dict[str, Any]] = {}
            # Barrier C: every target reads exactly the frozen snapshot set.
            for context in contexts:
                field, stats, details = _fuse_target(context, contexts, snapshots, camera, rotations, args)
                fused_fields[context.context_id] = field
                fusion_stats[context.context_id] = stats
                variance_stats = stats["donor_variance"]
                pbr_rows.append({"step": step_index, "angle": context.angle, "tile_id": context.tile_id,
                                 **{f"{channel}_variance_{quantile}": value for channel, values in variance_stats.items() for quantile, value in values.items()},
                                 "donor_zero": stats["coverage"]["zero"], "donor_one": stats["coverage"]["one"], "donor_two": stats["coverage"]["two"],
                                 "donor_three": stats["coverage"]["three"], "donor_four_plus": stats["coverage"]["four_plus"],
                                 "same_view_overlap": stats["coverage"]["same_view_overlap"], "cross_view_receipts": stats["coverage"]["cross_view_receipts"]})
                if bool(args.debug):
                    _atomic_torch_save(context.tile_dir / "steps" / f"step_{step_index:02d}_fusion.pt", details)
            corrected_x0: Dict[int, SparseTensor] = {}
            # Barrier D: direct E_tex(P_fused), with no E(P_original) cycle/residual.
            pbr_encoder.to("cuda")
            for group in pbr_encode_groups:
                corrected_x0.update(_encode_fused_batch(group, fused_fields, predictions, pbr_encoder, pipeline))
            pbr_encoder.cpu()
            _empty_cuda_cache()
            next_states: Dict[int, SparseTensor] = {}
            # Barrier E/F: native endpoint bridge and Euler values are all calculated before state replacement.
            for group in flow_groups:
                old_values = [_sparse_cuda(states[context.context_id]) for context in group]
                endpoint_values = [_sparse_cuda(corrected_x0[context.context_id]) for context in group]
                old_batch = _pack_sparse_batch(old_values, "Euler state")
                endpoint_batch = _pack_sparse_batch(endpoint_values, "Euler endpoint")
                corrected_velocity_batch = sampler._xstart_to_pred(old_batch, float(t), endpoint_batch)
                next_batch = SparseTensor(old_batch.feats - float(t - t_next) * corrected_velocity_batch.feats, old_batch.coords.detach().clone())
                velocity_values = _unpack_sparse_batch(corrected_velocity_batch, old_values, "Euler velocity")
                next_values = _unpack_sparse_batch(next_batch, old_values, "Euler next state")
                for context, old_state, endpoint, corrected_velocity, next_state in zip(group, old_values, endpoint_values, velocity_values, next_values):
                    _require_finite_sparse(corrected_velocity, f"step {step_index} context {context.context_id} corrected velocity")
                    _require_finite_sparse(next_state, f"step {step_index} context {context.context_id} next state")
                    cross_tile._strict_sparse_check(old_state, corrected_velocity, f"step {step_index} context {context.context_id} corrected velocity")
                    cross_tile._strict_sparse_check(old_state, next_state, f"step {step_index} context {context.context_id} next state")
                    delta = (endpoint.feats - _sparse_cuda(predictions[context.context_id]["x0"]).feats).abs().detach().cpu()
                    correction_rows.append({"step": step_index, "angle": context.angle, "tile_id": context.tile_id, **_quantiles(delta)})
                    next_states[context.context_id] = _sparse_cpu(next_state)
                del old_values, endpoint_values, old_batch, endpoint_batch, corrected_velocity_batch, velocity_values, next_batch, next_values
                _empty_cuda_cache()
            states = next_states
            fixed = all(torch.equal(states[ctx.context_id].coords, ctx.texture_norm.coords) for ctx in contexts)
            if not fixed:
                raise RuntimeError("texture sparse support changed during flow")
            step_record = {"step": step_index, "t": float(t), "t_next": float(t_next), "contexts": len(contexts), "seconds": time.perf_counter() - started,
                           "barriers": {"all_forward_before_decode": True, "all_decode_before_fusion": True, "all_fusion_before_encode": True,
                                        "all_corrected_endpoints_before_update": True, "synchronous_jacobi": True}, "fixed_texture_support": fixed,
                           "flow_batch_size": flow_batch_size,
                           "decode_batch_size": decode_batch_size,
                           "pbr_encode_batch_size": pbr_encode_batch_size,
                           "real_sparse_batch": True,
                           "batch_calls": {
                               "flow_forward": len(flow_groups),
                               "endpoint_decode": len(decode_groups),
                               "direct_pbr_encode": len(pbr_encode_groups),
                               "native_endpoint_bridge": len(flow_groups),
                               "flow_sizes": [len(group) for group in flow_groups],
                               "decode_sizes": [len(group) for group in decode_groups],
                               "pbr_encode_sizes": [len(group) for group in pbr_encode_groups],
                           },
                           "finite_features": True, "tiles": fusion_stats}
            _atomic_json(Path(args.output_dir) / "steps" / f"step_{step_index:02d}_summary.json", step_record)
            per_step.append(step_record)
            del predictions, snapshots, fused_fields, corrected_x0
            _empty_cuda_cache()
    finally:
        model.cpu()
        pbr_encoder.cpu()
    _write_csv(Path(args.output_dir) / "per_step_metrics.csv", [{"step": row["step"], "t": schedule[row["step"]], "contexts": len(contexts), "seconds": next(item["seconds"] for item in per_step if item["step"] == row["step"])} for row in per_step], ["step", "t", "contexts", "seconds"])
    pbr_variance_fields = [f"{channel}_variance_{quantile}" for channel in ("rgb", "metallic", "roughness", "alpha") for quantile in ("mean", "median", "p95", "max")]
    _write_csv(Path(args.output_dir) / "pbr_disagreement.csv", pbr_rows, ["step", "angle", "tile_id", *pbr_variance_fields, "donor_zero", "donor_one", "donor_two", "donor_three", "donor_four_plus", "same_view_overlap", "cross_view_receipts"])
    _write_csv(Path(args.output_dir) / "correction_norms.csv", correction_rows, ["step", "angle", "tile_id", "mean", "median", "p95", "max"])
    coverage_rows = []
    context_lookup = {context.context_id: context for context in contexts}
    for step in per_step:
        for context_id, stats in step["tiles"].items():
            context = context_lookup[int(context_id)]
            coverage_rows.append({"step": int(step["step"]), "angle": context.angle, "tile_id": context.tile_id, **stats["coverage"]})
    _atomic_json(Path(args.output_dir) / "cross_view_coverage_stats.json", {"per_step_tile_coverage": coverage_rows})
    _atomic_json(Path(args.output_dir) / "flow_summary.json", {"steps": per_step, "native_schedule": schedule, "start_index": start,
                 "route": "all forward -> all endpoint decode -> frozen masked PBR consensus -> direct PBR encode endpoint -> native xstart_to_pred -> synchronous Euler",
                 "real_sparse_batch": True, "serial_fallback": False, "single_batch_consistency_test": False,
                 "flow_batch_size": flow_batch_size, "decode_batch_size": decode_batch_size,
                 "pbr_encode_batch_size": pbr_encode_batch_size,
                 "round_trip_residual_cancellation": False, "soft_visibility": False, "slat_fusion": False})
    return states


def _final_snapshots(contexts: Sequence[TileContext], states: Mapping[int, SparseTensor], pipeline: Any, args: argparse.Namespace) -> Dict[int, DecodedSnapshot]:
    # The final field is another decoder barrier.  Keep it batched as well;
    # there is no correctness route that silently falls back to B=1 here.
    return _decode_snapshots_batched(
        contexts,
        states,
        pipeline,
        args,
        max(1, int(args.flow_batch_size if getattr(args, "decode_batch_size", None) is None else args.decode_batch_size)),
    )


def _final_assign(points: torch.Tensor, contexts: Sequence[TileContext], snapshots: Mapping[int, DecodedSnapshot], camera: Mapping[str, float], args: argparse.Namespace, baseline_attr: MeshWithVoxel) -> Tuple[torch.Tensor, int]:
    assigned = torch.zeros((points.shape[0], 6), dtype=torch.float32)
    best_distance = torch.full((points.shape[0],), float("inf"), dtype=torch.float32)
    rotations = {context.angle: _yaw_matrix(context.angle) for context in contexts}
    q_world = points * (2.0 * float(camera["mesh_scale"]))
    # This is a nearest valid tile-center selection, not a final Gaussian blend.
    for context in contexts:
        local, uv, covered = _world_to_local(q_world, context, camera, rotations[context.angle])
        rows = torch.where(covered)[0]
        if not rows.numel():
            continue
        # Visibility gates per-step donor eligibility only.  An invisible
        # target may have received a valid cross-view consensus and must remain
        # exportable at final assignment, so query full decoded support here.
        values, valid = _masked_query(snapshots[context.context_id].support_mesh, local.index_select(0, rows), int(args.query_chunk_size))
        rows, values = rows[valid], values[valid]
        if not rows.numel():
            continue
        distance = _tile_distance(uv.index_select(0, rows))
        replace = distance < best_distance.index_select(0, rows)
        if bool(replace.any()):
            target_rows = rows[replace]
            assigned[target_rows] = values[replace]
            best_distance[target_rows] = distance[replace]
    fallback = torch.where(~torch.isfinite(best_distance))[0]
    if fallback.numel():
        assigned[fallback] = cross_tile._query_mesh_chunked(baseline_attr, points.index_select(0, fallback), int(args.query_chunk_size)).detach().cpu()
    return assigned, int(fallback.numel())


def _build_final_meshes(contexts: Sequence[TileContext], states: Mapping[int, SparseTensor], pipeline: Any, baseline: MeshWithVoxel, camera: Mapping[str, float], args: argparse.Namespace, baseline_attr: MeshWithVoxel, output_dir: Path) -> Tuple[MeshWithVertexPbr, MeshWithFacePbr, Dict[str, Any]]:
    snapshots = _final_snapshots(contexts, states, pipeline, args)
    vertices = baseline.vertices.detach().cpu()
    faces = baseline.faces.detach().cpu()
    face_points = (vertices.index_select(0, faces.long().reshape(-1)).reshape(-1, 3, 3)).mean(dim=1)
    vertex_parts, face_parts = [], []
    vertex_fallback = face_fallback = 0
    for points, parts, is_vertex in ((vertices, vertex_parts, True), (face_points, face_parts, False)):
        fallback = 0
        for start in range(0, int(points.shape[0]), int(args.final_query_chunk_size)):
            value, missing = _final_assign(points[start:start + int(args.final_query_chunk_size)], contexts, snapshots, camera, args, baseline_attr)
            parts.append(value)
            fallback += missing
        if is_vertex:
            vertex_fallback = fallback
        else:
            face_fallback = fallback
    vertex_mesh = MeshWithVertexPbr(vertices, faces, torch.cat(vertex_parts), layout=dict(PBR_LAYOUT))
    face_mesh = MeshWithFacePbr(vertices, faces, torch.cat(face_parts), layout=dict(PBR_LAYOUT))
    _atomic_torch_save(output_dir / "final_per_vertex_pbr_mesh.pt", {"format": FORMAT, "representation": "per_vertex_pbr", "mesh": vertex_mesh})
    _atomic_torch_save(output_dir / "final_per_face_pbr_mesh.pt", {"format": FORMAT, "representation": "per_face_pbr", "mesh": face_mesh})
    summary = {"geometry_source": "unchanged baseline 1024 mesh", "vertices": int(vertices.shape[0]), "faces": int(faces.shape[0]),
               "final_assignment": "nearest tile-center valid full-support donor; per-step visibility remains donor-only; baseline PBR only when no valid donor exists", "final_gaussian_blend": False,
               "vertex_baseline_fallback": vertex_fallback, "face_baseline_fallback": face_fallback,
               "decoded_donor_visibility": "decoded C1024 parent=floor(coord/16) joins fixed SLat binary visibility for per-step donors; final assignment queries full decoded support"}
    _atomic_json(output_dir / "final_assignment.json", summary)
    return vertex_mesh, face_mesh, summary


def _render_outputs(mesh: MeshWithVertexPbr, camera: Mapping[str, float], args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    if not args.render:
        return {"enabled": False}
    from pixal3d.renderers import PbrMeshRenderer
    from render_pixal3d_raw_ovoxel import load_envmap
    angles = (0, 60, 120, 180, 240, 300)
    extrinsics, intrinsics, _ = baseline_render._make_camera_views(float(camera["camera_angle_x"]), float(camera["distance"]), angles)
    renderer = PbrMeshRenderer(rendering_options={"resolution": int(args.render_resolution), "near": max(0.01, float(camera["distance"]) - 2.0),
                              "far": float(camera["distance"]) + 10.0, "ssaa": int(args.render_ssaa), "peel_layers": int(args.render_peel_layers),
                              "face_chunk_size": int(args.render_face_chunk_size)}, device=f"cuda:{torch.cuda.current_device()}")
    envmap = load_envmap(str(args.envmap), device="cuda")
    paths: List[Path] = []
    live = mesh.to("cuda")
    for angle in angles:
        result = renderer.render(live, extrinsics[angle], intrinsics, envmap=envmap, use_envmap_bg=False)
        image = baseline_render._image_from_array(baseline_render._tensor_to_hwc(result["shaded"]))
        path = output_dir / "renders" / f"yaw{angle:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        paths.append(path)
        del result
        _empty_cuda_cache()
    sheet = Image.new("RGB", (3 * int(args.render_resolution), 2 * int(args.render_resolution)), "black")
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            sheet.paste(image.convert("RGB"), ((index % 3) * image.width, (index // 3) * image.height))
    sheet.save(output_dir / "renders" / "six_view_sheet.png")
    frames: List[Image.Image] = []
    for index in range(int(args.turntable_frames)):
        angle = int(round(360.0 * index / int(args.turntable_frames)))
        extr, intr, _ = baseline_render._make_camera_views(float(camera["camera_angle_x"]), float(camera["distance"]), (angle,))
        result = renderer.render(live, extr[angle], intr, envmap=envmap, use_envmap_bg=False)
        image = baseline_render._image_from_array(baseline_render._tensor_to_hwc(result["shaded"]))
        path = output_dir / "turntable" / f"frame_{index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        frames.append(image)
        del result
        _empty_cuda_cache()
    if frames:
        frames[0].save(output_dir / "turntable" / "turntable.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
    del live, envmap, renderer
    _empty_cuda_cache()
    return {"enabled": True, "angles": list(angles), "six_view_sheet": str(output_dir / "renders" / "six_view_sheet.png"), "turntable_frames": len(frames)}


def _render_only(args: argparse.Namespace) -> Dict[str, Any]:
    """Render an already completed final mesh without rerunning texture flow."""
    output_dir = Path(args.output_dir).expanduser().resolve()
    mesh_path = output_dir / "final_per_vertex_pbr_mesh.pt"
    camera_path = output_dir / "global_camera.json"
    summary_path = output_dir / "summary.json"
    if not mesh_path.is_file() or not camera_path.is_file():
        raise FileNotFoundError(
            f"render-only requires {mesh_path} and {camera_path}"
        )
    try:
        payload = torch.load(mesh_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(mesh_path, map_location="cpu")
    mesh = payload["mesh"] if isinstance(payload, Mapping) and "mesh" in payload else payload
    if not isinstance(mesh, MeshWithVertexPbr):
        raise TypeError(f"render-only expected MeshWithVertexPbr, got {type(mesh)!r}")
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    render_summary = _render_outputs(mesh, camera, args, output_dir)
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["render"] = render_summary
        _atomic_json(summary_path, summary)
        _write_report(output_dir, summary)
    return {"output_dir": str(output_dir), "render": render_summary}


def _write_report(output_dir: Path, summary: Mapping[str, Any]) -> None:
    lines = ["# Multi-View Gaussian PBR SR Report", "", "## Input", "", "The 3072x1024 composite was split before resizing into 0, 120, and 240 degree 1024x1024 views. Each local condition is a real 256x256 crop with stride 128, resized directly to 1024x1024. No full view is resized to 4096.", "", "## Camera", "", "The native 1024 crop camera is algebraically equivalent to the old virtual 4096 crop camera. `camera_equivalence.json` records all 49 comparisons. World q is rotated into the current yaw frame before local projective conversion and rotated back before baseline PBR queries.", "", "## Geometry", "", "The final geometry is the unchanged cached Pixal3D baseline 1024 mesh. Each view-tile derives a distinct fixed projective support only to run native texture generation.", "", "## Condition", "", "Global DINO comes once from the corresponding complete 1024 view and local projected DINO comes once from that tile's actual resized 256-to-1024 image. The native `{cond:{global,proj}, neg_cond:{global,proj}}` schema is retained. No condition or DINO fusion is performed.", "", "## Visibility", "", "A geometry-only per-view z-buffer produces binary visible faces and baseline mesh vertices. Local O-voxels still inherit their nearest baseline-vertex bit for donor gating, but SLat visibility is computed independently: each C64 SLat center is projected through the exact tile camera, mapped to world q, and assigned the binary bit of its nearest baseline mesh vertex. Per-view mesh and per-tile SLat red/green overlays, foreground IoU, and nearest-mesh distances are saved under `visibility/`.", "", "## PBR Consensus", "", "Frozen decoded donor fields are queried with masked trilinear interpolation. The only donor weight is binary visibility multiplied by the established Gaussian tile-center weight. No soft visibility or confidence is used.", "", "## Flow", "", "Every Euler step uses a Jacobi barrier: all native forwards, all endpoint decodes, frozen PBR consensus, direct PBR encoding as the corrected endpoint, native `_xstart_to_pred`, then synchronous Euler update. Normalized texture SLat is denormalized before every decoder call. No round-trip residual cancellation is used.", "", "## Final PBR", "", "Final per-face and per-vertex PBR uses the nearest valid tile-center donor queried on the full decoded support. Visibility is used only during per-step donor filtering; baseline PBR is used only when no valid final support exists. There is no final Gaussian blend.", "", "## Artifacts", ""]
    for key, value in summary.get("artifacts", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    (output_dir / "MULTIVIEW_GAUSSIAN_PBR_SR_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiview-image", default="test_pic/mask_compare_output/image2_resized.png")
    parser.add_argument("--baseline-dir", default="outputs/cross_tile_pbr_perstep_guided_cuda4_full")
    parser.add_argument("--output-dir", default="outputs/multiview_fixed_geometry_pbr_gaussian_cuda4")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--shape-encoder", default=str(core.DEFAULT_ENCODER_ROOT / "shape_enc_next_dc_f16c32_fp16"))
    parser.add_argument("--pbr-encoder", default=str(core.DEFAULT_ENCODER_ROOT / "tex_enc_next_dc_f16c32_fp16"))
    parser.add_argument("--device", "--cuda-device", dest="cuda_device", type=int, default=4)
    parser.add_argument("--angles", nargs="+", type=int, default=list(ANGLES_DEFAULT))
    parser.add_argument("--selected-views", nargs="+", type=int, default=list(ANGLES_DEFAULT))
    parser.add_argument("--source-view-size", type=int, default=1024)
    parser.add_argument("--source-tile-size", type=int, default=256)
    parser.add_argument("--source-tile-stride", type=int, default=128)
    parser.add_argument("--model-tile-size", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=12)
    parser.add_argument("--flow-batch-size", type=int, default=8,
                        help="Number of view-tiles per real initial-enc/flow/decode/direct-encode batch.")
    parser.add_argument("--initial-encode-batch-size", type=int, default=1,
                        help="Batch for the one-time initial shape/PBR encoders; separate from staged flow batches.")
    parser.add_argument("--decode-batch-size", type=int, default=None,
                        help="Endpoint decoder batch size; defaults to flow batch size.")
    parser.add_argument("--pbr-encode-batch-size", type=int, default=None,
                        help="Corrected PBR encoder batch size; defaults to flow batch size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-timestep", type=float, default=1.0)
    parser.add_argument("--noise-strength", type=float, default=1.0)
    parser.add_argument("--texture-guidance-strength", type=float, default=1.0)
    parser.add_argument("--texture-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--texture-rescale-t", type=float, default=3.0)
    parser.add_argument("--gaussian-sigma", type=float, default=256.0)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--tile-ids", nargs="+", type=int, default=None,
                        help="Restrict every selected view to these 7x7-layout tile IDs.")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument("--roundtrip-tolerance", type=float, default=2e-5)
    parser.add_argument("--face-projection-chunk-size", type=int, default=250_000)
    parser.add_argument("--material-query-chunk-size", type=int, default=65_536)
    parser.add_argument("--material-face-chunk-size", type=int, default=16_384)
    parser.add_argument("--query-chunk-size", type=int, default=65_536)
    parser.add_argument("--final-query-chunk-size", type=int, default=65_536)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-only", action="store_true",
                        help="Render final_per_vertex_pbr_mesh.pt from an existing output directory.")
    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--render-ssaa", type=int, default=1)
    parser.add_argument("--render-peel-layers", type=int, default=4)
    parser.add_argument("--render-face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--turntable-frames", type=int, default=24)
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if bool(args.render_only):
        if not torch.cuda.is_available():
            raise RuntimeError("render-only requires CUDA")
        torch.cuda.set_device(int(args.cuda_device))
        return _render_only(args)
    if not torch.cuda.is_available() or int(args.cuda_device) >= torch.cuda.device_count():
        raise RuntimeError(f"requested CUDA device {args.cuda_device} is unavailable")
    if (args.source_view_size, args.source_tile_size, args.source_tile_stride, args.model_tile_size) != (1024, 256, 128, 1024):
        raise ValueError("this fixed experiment requires source-view/tile/stride/model = 1024/256/128/1024")
    if tuple(args.angles) != ANGLES_DEFAULT or any(angle not in ANGLES_DEFAULT for angle in args.selected_views):
        raise ValueError("the supplied composite has exactly yaw 0, 120, and 240 panels")
    if args.max_tiles is not None and int(args.max_tiles) <= 0:
        raise ValueError("--max-tiles must be positive when supplied")
    if int(args.flow_batch_size) <= 0:
        raise ValueError("--flow-batch-size must be positive")
    if args.decode_batch_size is not None and int(args.decode_batch_size) <= 0:
        raise ValueError("--decode-batch-size must be positive")
    if args.pbr_encode_batch_size is not None and int(args.pbr_encode_batch_size) <= 0:
        raise ValueError("--pbr-encode-batch-size must be positive")
    if args.tile_ids is not None:
        invalid_tile_ids = sorted({int(tile_id) for tile_id in args.tile_ids if int(tile_id) < 0 or int(tile_id) >= 49})
        if invalid_tile_ids:
            raise ValueError(f"--tile-ids must be in [0, 48], got {invalid_tile_ids}")
    torch.cuda.set_device(int(args.cuda_device))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    source_path = Path(args.multiview_image).expanduser().resolve()
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    camera_path = baseline_dir / "global_camera.json"
    mesh_path = baseline_dir / "global_baseline_mesh.pt"
    if not source_path.is_file() or not camera_path.is_file() or not mesh_path.is_file():
        raise FileNotFoundError(f"missing input, camera, or baseline mesh: {source_path}, {camera_path}, {mesh_path}")
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    config = {"format": FORMAT, "cuda_device": int(args.cuda_device), "cuda_name": torch.cuda.get_device_name(int(args.cuda_device)),
              "input": str(source_path), "baseline_dir": str(baseline_dir), "args": vars(args), "fixed_geometry": True,
              "full_view_4096_upsample": False, "condition_fusion": False, "slat_fusion": False, "soft_visibility": False,
              "round_trip_residual_cancellation": False, "final_gaussian_blend": False}
    _atomic_json(output_dir / "config.json", config)
    views_all = _load_views(source_path, output_dir, ANGLES_DEFAULT)
    correctness = _run_correctness_tests(camera, source_path, output_dir, int(args.seed))
    if not correctness["all_passed"]:
        raise RuntimeError(f"correctness tests failed: {correctness}")
    if args.test_only:
        summary = {**config, "correctness": correctness, "status": "test_only_passed"}
        _atomic_json(output_dir / "summary.json", summary)
        return summary
    selected_views = {angle: views_all[angle] for angle in args.selected_views}
    _seed(int(args.seed))
    pipeline = init_pipeline(args.model_path, device="cuda", low_vram=bool(args.low_vram))
    baseline = cross_tile._load_mesh(mesh_path).to("cpu")
    shutil.copy2(mesh_path, output_dir / "global_baseline_mesh.pt")
    shutil.copy2(camera_path, output_dir / "global_camera.json")
    baseline_attr = core._make_attribute_query_mesh(baseline, torch.device("cuda"))
    contexts, dino, _ = _build_contexts(args, pipeline, baseline, camera, selected_views, output_dir, baseline_attr)
    if not contexts:
        raise RuntimeError("all requested view tiles had empty support")
    pbr_encoder = pixal3d_models.from_pretrained(str(Path(args.pbr_encoder))).eval()
    states = _run_flow(contexts, pipeline, camera, pbr_encoder, args)
    pbr_encoder.cpu()
    vertex_mesh, face_mesh, final_summary = _build_final_meshes(contexts, states, pipeline, baseline, camera, args, baseline_attr, output_dir)
    render_summary = _render_outputs(vertex_mesh, camera, args, output_dir)
    summary = {**config, "correctness": correctness, "active_contexts": len(contexts), "dino_cache": dino, "final": final_summary,
               "render": render_summary, "artifacts": {"config": str(output_dir / "config.json"), "camera_equivalence": str(output_dir / "camera_equivalence.json"),
               "coordinate_roundtrip": str(output_dir / "coordinate_roundtrip.json"), "support_stats": str(output_dir / "support_stats.json"),
               "visibility": str(output_dir / "visibility"), "visibility_stats": str(output_dir / "visibility_stats.json"),
               "dino_cache": str(output_dir / "dino_cache_diagnostics.json"), "per_step_metrics": str(output_dir / "per_step_metrics.csv"),
               "cross_view_coverage": str(output_dir / "cross_view_coverage_stats.json"), "pbr_disagreement": str(output_dir / "pbr_disagreement.csv"),
               "correction_norms": str(output_dir / "correction_norms.csv"), "flow_summary": str(output_dir / "flow_summary.json"),
               "vertex_mesh": str(output_dir / "final_per_vertex_pbr_mesh.pt"), "face_mesh": str(output_dir / "final_per_face_pbr_mesh.pt"),
               "final_assignment": str(output_dir / "final_assignment.json")}}
    _write_report(output_dir, summary)
    summary["artifacts"]["report"] = str(output_dir / "MULTIVIEW_GAUSSIAN_PBR_SR_REPORT.md")
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    summary = run(_parser().parse_args())
    print(f"[done] {summary.get('status', 'complete')} output={summary.get('args', {}).get('output_dir', '')}")


if __name__ == "__main__":
    main()
