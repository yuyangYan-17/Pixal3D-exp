#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pixal3D 2048 tile cascade with projective tile recanonicalization.

The complete image first produces a full-object C128 sparse support.  For each
4096 -> 1024 crop, every C128 point whose complete-image projection lies inside
the crop is retained, including occluded/back-side points inferred by the global
model.  Those points are deterministically re-parameterized into a new tile
canonical coordinate system q_tile in [-1, 1]^3:

    global C128 q_global
      -> complete-image camera projection
      -> crop-local pixel coordinates
      -> inverse projection through the centered Pixal3D tile camera
      -> q_tile, preserving q_tile.z = q_global.z

No depth slab, z-buffer, visibility test, fitted local cube, or point-depth
selection is used.  Small numerical/projective overflow outside [-1, 1] is
clamped rather than discarded and is reported in the trace.

Each tile uses its projectively recanonicalized C32 support as the sparse
structure result, runs the official 512 shape flow on the high-resolution crop,
upsamples to tile C64, and maps the resulting C64 coordinates back through the
exact inverse projective transform to continuous global C128 coordinates.
The formal shape and texture trajectories remain a single C128 master state.
During selected latter Euler steps, each tile C64 point reads the current
feature of exactly one mapped global C128 master point.  The tile model predicts
a velocity for that point from the crop view.  Velocities from multiple local
points and overlapping tiles are tent-weighted and averaged on the same global
master row.  Uncovered master rows retain the global velocity.

The script targets the current Pixal3D-exp branch, whose projection condition
supports sparse grid_indices and grid_resolution_override and whose samplers
expose timestep_schedule() and sample_once().
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
)
os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "1")

from inference import (  # noqa: E402
    MODEL_PATH,
    distance_from_fov,
    get_camera_params_wild_moge,
    init_pipeline,
    load_moge_model,
)
from pixal3d.modules.sparse import SparseTensor  # noqa: E402

try:  # Required only when final decoding/export is enabled.
    import o_voxel  # type: ignore
except Exception:  # pragma: no cover
    o_voxel = None


GRID_LR = 32
GRID_TILE_HR = 64
GRID_MASTER = 128
RESOLUTION_LR = 512
RESOLUTION_TILE_HR = 1024
RESOLUTION_MASTER = 2048
CANONICAL_IMAGE_SIZE = 4096
DEFAULT_TILE_SIZE = 1024
DEFAULT_TILE_STRIDE = 512

PIXAL3D_EXPORT_ROTATION = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass
class RecordedFlow:
    result: Any
    name: str

    @property
    def trajectory(self) -> Any:
        value = getattr(self.result, "trajectory", None)
        if value is None:
            raise RuntimeError(f"{self.name} did not record a trajectory")
        return value

    @property
    def samples(self) -> SparseTensor:
        value = getattr(self.result, "samples", None)
        if not isinstance(value, SparseTensor):
            raise TypeError(f"{self.name}.samples is not a SparseTensor")
        return value


@dataclass
class TileCameraTransform:
    box: Tuple[int, int, int, int]
    output_size: int
    camera_angle_x: float
    distance: float
    mesh_scale: float
    full_focal_pixels: float
    tile_focal_pixels: float
    crop_scale_x: float
    crop_scale_y: float


@dataclass
class TileExpert:
    tile_id: int
    box: Tuple[int, int, int, int]
    transform: TileCameraTransform
    input_base_rows: torch.Tensor              # [M]
    input_local_coords32: torch.Tensor         # [N32,4]
    local_coords64: torch.Tensor               # [K,4], tile-space integer IDs
    local_tile_camera_points64: torch.Tensor   # [K,3]
    local_global_camera_points64: torch.Tensor # [K,3]
    mapped_global_coords128: torch.Tensor      # [K,4], global-space integer IDs
    shape_condition_cpu: Mapping[str, Any]
    texture_condition_cpu: Mapping[str, Any]
    input_transform_stats: Mapping[str, Any]
    output_transform_stats: Mapping[str, Any]
    lr_trace_path: Optional[str]
    local_to_master_row: Optional[torch.Tensor] = None  # [K]
    active_local_rows: Optional[torch.Tensor] = None    # [Ka]
    active_tent_weights: Optional[torch.Tensor] = None  # [Ka]


@dataclass
class OnlineFlowResult:
    samples: SparseTensor
    times: List[float]
    time_intervals: List[float]
    states: List[torch.Tensor]
    velocities: List[torch.Tensor]
    step_records: List[Dict[str, Any]]
    covered_rows_union: int


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _randn(
    rows: int,
    channels: int,
    *,
    device: torch.device,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if rows <= 0 or channels <= 0:
        raise ValueError(f"invalid random tensor shape ({rows}, {channels})")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn(
        rows,
        channels,
        generator=generator,
        device=device,
        dtype=dtype,
    )


def _normalization(
    values: Mapping[str, Sequence[float]],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    std = torch.as_tensor(values["std"], device=device, dtype=dtype)[None]
    mean = torch.as_tensor(values["mean"], device=device, dtype=dtype)[None]
    return std, mean


def _denormalize_sparse(
    value: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
) -> SparseTensor:
    std, mean = _normalization(normalization, value.device, value.dtype)
    return value.replace(value.feats * std + mean)


def _normalize_sparse(
    value: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
) -> SparseTensor:
    std, mean = _normalization(normalization, value.device, value.dtype)
    return value.replace((value.feats - mean) / std)


def _features(value: Any) -> torch.Tensor:
    return value.feats if hasattr(value, "feats") else value


def _tree_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, SparseTensor):
        return SparseTensor(
            feats=value.feats.detach().to(device="cpu", copy=True),
            coords=value.coords.detach().to(device="cpu", copy=True),
        )
    if isinstance(value, Mapping):
        return {key: _tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_tree_to_cpu(item) for item in value]
    return value


def _tree_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=False)
    if isinstance(value, SparseTensor):
        return SparseTensor(
            feats=value.feats.to(device=device, non_blocking=False),
            coords=value.coords.to(device=device, non_blocking=False),
        )
    if isinstance(value, Mapping):
        return {key: _tree_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_tree_to_device(item, device) for item in value]
    return value


def _sample_once_kwargs(sampler_params: Mapping[str, Any]) -> Dict[str, Any]:
    excluded = {
        "steps",
        "rescale_t",
        "verbose",
        "tqdm_desc",
        "record_trajectory",
        "trajectory_device",
        "return_model_history",
    }
    return {key: value for key, value in sampler_params.items() if key not in excluded}


def _validate_trajectory(flow: RecordedFlow, expected_steps: int) -> None:
    trajectory = flow.trajectory
    if len(trajectory.states) != expected_steps + 1:
        raise RuntimeError(f"{flow.name}: invalid state count")
    if len(trajectory.velocities) != expected_steps:
        raise RuntimeError(f"{flow.name}: invalid velocity count")
    if len(trajectory.times) != expected_steps + 1:
        raise RuntimeError(f"{flow.name}: invalid time count")
    if len(trajectory.time_intervals) != expected_steps:
        raise RuntimeError(f"{flow.name}: invalid interval count")


def _run_recorded_flow(
    *,
    pipeline: Any,
    sampler: Any,
    flow_model: torch.nn.Module,
    noise: SparseTensor,
    condition: Mapping[str, Any],
    sampler_params: Mapping[str, Any],
    description: str,
    concat_cond: Optional[SparseTensor] = None,
) -> RecordedFlow:
    if concat_cond is not None and not torch.equal(noise.coords, concat_cond.coords):
        raise RuntimeError(f"{description}: noise/concat coordinates differ")
    if pipeline.low_vram:
        flow_model.to(pipeline.device)
    kwargs: Dict[str, Any] = {
        **condition,
        **dict(sampler_params),
        "verbose": True,
        "tqdm_desc": description,
        "record_trajectory": True,
        "trajectory_device": "cpu",
        "return_model_history": False,
    }
    if concat_cond is not None:
        kwargs["concat_cond"] = concat_cond
    started = time.perf_counter()
    result = sampler.sample(flow_model, noise, **kwargs)
    _sync_cuda()
    print(
        f"[recorded-flow] {description}: tokens={noise.feats.shape[0]:,} "
        f"channels={noise.feats.shape[1]} seconds={time.perf_counter()-started:.3f}"
    )
    if pipeline.low_vram:
        flow_model.cpu()
        _empty_cuda_cache()
    wrapped = RecordedFlow(result=result, name=description)
    _validate_trajectory(wrapped, int(sampler_params.get("steps", 12)))
    if not torch.equal(wrapped.samples.coords, noise.coords):
        raise RuntimeError(f"{description}: sampler changed sparse coordinates")
    return wrapped


def _endpoint_indices_to_q(indices: torch.Tensor, resolution: int) -> torch.Tensor:
    if resolution <= 1:
        raise ValueError("resolution must exceed one")
    return indices.to(torch.float32) * (2.0 / float(resolution - 1)) - 1.0


def _q_to_endpoint_indices(q: torch.Tensor, resolution: int) -> torch.Tensor:
    return torch.round((q + 1.0) * (float(resolution - 1) / 2.0)).to(torch.int32)


def _quantize_decoder_candidates(
    candidates: torch.Tensor,
    *,
    target_grid: int,
    source_resolution: int = RESOLUTION_LR,
) -> torch.Tensor:
    if candidates.ndim != 2 or candidates.shape[1] != 4:
        raise ValueError(f"decoder candidates must be [N,4], got {tuple(candidates.shape)}")
    # Match the official Pixal3D cascade exactly: decoder coordinates live in
    # the 512 endpoint convention and are mapped to [0, target_grid-1] with
    # round(), not to [0, target_grid] with floor().
    quantized_xyz = torch.round(
        (candidates[:, 1:].to(torch.float32) + 0.5)
        / float(source_resolution)
        * float(target_grid - 1)
    ).to(torch.int32)
    quantized = torch.cat(
        [candidates[:, :1].to(torch.int32), quantized_xyz],
        dim=1,
    )
    quantized = torch.unique(quantized, dim=0)
    valid = ((quantized[:, 1:] >= 0) & (quantized[:, 1:] < target_grid)).all(dim=1)
    quantized = quantized[valid]
    if quantized.numel() == 0:
        raise RuntimeError(f"decoder upsample produced no C{target_grid} coordinates")
    return quantized


def _learned_upsample(
    pipeline: Any,
    lr_shape_denormalized: SparseTensor,
    *,
    target_grid: int,
) -> torch.Tensor:
    decoder = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        decoder.to(pipeline.device)
        decoder.low_vram = True
    try:
        candidates = decoder.upsample(lr_shape_denormalized, upsample_times=4)
    finally:
        if pipeline.low_vram:
            decoder.cpu()
            decoder.low_vram = False
            _empty_cuda_cache()
    return _quantize_decoder_candidates(candidates, target_grid=target_grid)


def _coord_key_rows(coords: torch.Tensor) -> Dict[Tuple[int, int, int, int], int]:
    cpu = coords.detach().to(device="cpu", dtype=torch.int64)
    mapping: Dict[Tuple[int, int, int, int], int] = {}
    for row, values in enumerate(cpu.tolist()):
        key = tuple(int(value) for value in values)
        if key in mapping:
            raise RuntimeError(f"duplicate sparse coordinate {key}")
        mapping[key] = row
    return mapping


def _global_coords_to_camera(
    coords: torch.Tensor,
    *,
    grid_resolution: int,
    camera: Mapping[str, float],
) -> torch.Tensor:
    q = _endpoint_indices_to_q(coords[:, 1:4], grid_resolution).to(coords.device)
    center = torch.tensor(
        [0.0, 0.0, -float(camera["distance"])],
        device=coords.device,
        dtype=q.dtype,
    )
    return center[None] + q / (2.0 * float(camera["mesh_scale"]))


def _camera_to_global_q(
    camera_points: torch.Tensor,
    *,
    camera: Mapping[str, float],
) -> torch.Tensor:
    center = torch.tensor(
        [0.0, 0.0, -float(camera["distance"])],
        device=camera_points.device,
        dtype=camera_points.dtype,
    )
    return 2.0 * float(camera["mesh_scale"]) * (camera_points - center[None])


def _project_camera_points(
    camera_points: torch.Tensor,
    camera_angle_x: torch.Tensor | float,
    resolution: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points = camera_points
    if points.ndim == 2:
        points = points.unsqueeze(0)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("camera_points must be [K,3] or [B,K,3]")
    batch = points.shape[0]
    fov = torch.as_tensor(camera_angle_x, device=points.device, dtype=points.dtype)
    if fov.ndim == 0:
        fov = fov[None]
    if fov.numel() == 1 and batch > 1:
        fov = fov.expand(batch)
    if fov.shape != (batch,):
        raise ValueError(f"camera_angle_x must broadcast to [{batch}]")
    focal = float(resolution) / (2.0 * torch.tan(fov / 2.0))
    depth = -points[..., 2]
    denom = depth.clamp_min(1e-8)
    x = focal[:, None] * points[..., 0] / denom + float(resolution) / 2.0
    y = -focal[:, None] * points[..., 1] / denom + float(resolution) / 2.0
    pixels = torch.stack([x, y], dim=-1)
    valid = (
        (depth > 0)
        & (x >= 0)
        & (x < resolution)
        & (y >= 0)
        & (y < resolution)
        & torch.isfinite(pixels).all(dim=-1)
    )
    return pixels, depth, valid


def _get_tile_projection_condition(
    *,
    pipeline: Any,
    image_cond_model: torch.nn.Module,
    image: Image.Image,
    coords: torch.Tensor,
    transform: TileCameraTransform,
    grid_resolution: int,
) -> Mapping[str, Any]:
    """Use the ordinary Pixal3D centered-camera projection in tile space."""
    return pipeline.get_proj_cond_shape(
        image_cond_model,
        [image],
        coords,
        camera_angle_x=float(transform.camera_angle_x),
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
        grid_resolution_override=int(grid_resolution),
    )


def _tile_layout(
    canonical_size: int,
    tile_size: int,
    tile_stride: int,
) -> List[Tuple[int, int, int, int]]:
    starts = list(range(0, canonical_size - tile_size + 1, tile_stride))
    if not starts or starts[-1] != canonical_size - tile_size:
        raise ValueError("tile layout does not land on the final image edge")
    return [
        (x0, y0, x0 + tile_size, y0 + tile_size)
        for y0 in starts
        for x0 in starts
    ]


def _rows_and_tent_weights(
    uv: torch.Tensor,
    valid: torch.Tensor,
    box: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    x0, y0, x1, y1 = (float(value) for value in box)
    in_box = (
        valid
        & (uv[:, 0] >= x0)
        & (uv[:, 0] < x1)
        & (uv[:, 1] >= y0)
        & (uv[:, 1] < y1)
    )
    rows = torch.where(in_box)[0]
    if rows.numel() == 0:
        return rows, torch.empty(0, device=uv.device, dtype=torch.float32)
    local_x = (uv[rows, 0] - x0) / (x1 - x0)
    local_y = (uv[rows, 1] - y0) / (y1 - y0)
    weights = (
        (1.0 - (2.0 * local_x - 1.0).abs())
        * (1.0 - (2.0 * local_y - 1.0).abs())
    ).clamp_min(1e-3)
    return rows, weights.to(torch.float32)


def _focal_pixels(camera_angle_x: float, resolution: int) -> float:
    return float(resolution) / (2.0 * math.tan(float(camera_angle_x) / 2.0))


def _build_tile_camera_transform(
    *,
    box: Sequence[int],
    camera: Mapping[str, float],
    canonical_size: int,
    output_size: int,
    extend_pixel: int = 0,
) -> TileCameraTransform:
    """Construct a centered Pixal3D camera representing one raw image crop.

    The crop keeps the complete-image pixel rays.  Its focal length in output
    pixels is the complete-image focal length multiplied by the crop resize
    factor.  The tile FOV and Pixal3D distance are then derived exactly as in
    official inference.  No point depth statistics enter this construction.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    crop_w = x1 - x0
    crop_h = y1 - y0
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"invalid tile box {tuple(box)}")
    scale_x = float(output_size) / float(crop_w)
    scale_y = float(output_size) / float(crop_h)
    full_focal = _focal_pixels(float(camera["camera_angle_x"]), canonical_size)
    tile_fx = full_focal * scale_x
    tile_fy = full_focal * scale_y
    if not math.isclose(tile_fx, tile_fy, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            "current Pixal3D projection accepts one symmetric FOV; "
            f"tile focal mismatch fx={tile_fx}, fy={tile_fy}"
        )
    tile_fov = 2.0 * math.atan(float(output_size) / (2.0 * tile_fx))
    tile_mesh_scale = float(camera["mesh_scale"])
    tile_distance = distance_from_fov(
        tile_fov,
        torch.tensor([-1.0, 0.0, 0.0]),
        torch.tensor([0 - int(extend_pixel), output_size - 1 + int(extend_pixel)]),
        tile_mesh_scale,
        output_size,
    )["distance_from_x"]
    return TileCameraTransform(
        box=(x0, y0, x1, y1),
        output_size=int(output_size),
        camera_angle_x=float(tile_fov),
        distance=float(tile_distance),
        mesh_scale=float(tile_mesh_scale),
        full_focal_pixels=float(full_focal),
        tile_focal_pixels=float(tile_fx),
        crop_scale_x=float(scale_x),
        crop_scale_y=float(scale_y),
    )


def _tile_coords_to_camera(
    coords: torch.Tensor,
    *,
    grid_resolution: int,
    transform: TileCameraTransform,
) -> torch.Tensor:
    q = _endpoint_indices_to_q(coords[:, 1:4], grid_resolution).to(coords.device)
    center = torch.tensor(
        [0.0, 0.0, -float(transform.distance)],
        device=coords.device,
        dtype=q.dtype,
    )
    return center[None] + q / (2.0 * float(transform.mesh_scale))


def _global_q_to_tile_q(
    q_global: torch.Tensor,
    *,
    camera: Mapping[str, float],
    transform: TileCameraTransform,
    clamp: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Projectively recanonicalize global q into the tile's new [-1,1]^3.

    The complete normalized depth coordinate is preserved exactly before the
    optional numerical clamp: q_tile.z = q_global.z.  Every supplied row is
    returned; no visibility/depth/range row filtering occurs.
    """
    if q_global.ndim != 2 or q_global.shape[1] != 3:
        raise ValueError("q_global must be [N,3]")
    dtype = q_global.dtype
    device = q_global.device
    global_center = torch.tensor(
        [0.0, 0.0, -float(camera["distance"])],
        device=device,
        dtype=dtype,
    )
    global_points = global_center[None] + q_global / (2.0 * float(camera["mesh_scale"]))
    uv, _, valid = _project_camera_points(
        global_points,
        float(camera["camera_angle_x"]),
        CANONICAL_IMAGE_SIZE,
    )
    uv = uv[0]
    valid = valid[0]
    if not bool(valid.all().item()):
        raise RuntimeError("selected global tile rows include invalid complete-camera projections")

    x0, y0, _, _ = transform.box
    u_tile = (uv[:, 0] - float(x0)) * float(transform.crop_scale_x)
    v_tile = (uv[:, 1] - float(y0)) * float(transform.crop_scale_y)
    qz = q_global[:, 2]
    depth_tile = float(transform.distance) - qz / (2.0 * float(transform.mesh_scale))
    if bool((depth_tile <= 0).any().item()):
        raise RuntimeError("tile canonical depth became non-positive")
    x_tile = (
        (u_tile - float(transform.output_size) / 2.0)
        * depth_tile
        / float(transform.tile_focal_pixels)
    )
    y_tile = -(
        (v_tile - float(transform.output_size) / 2.0)
        * depth_tile
        / float(transform.tile_focal_pixels)
    )
    q_raw = torch.stack(
        [
            2.0 * float(transform.mesh_scale) * x_tile,
            2.0 * float(transform.mesh_scale) * y_tile,
            qz,
        ],
        dim=1,
    )
    overflow = (q_raw.abs() - 1.0).clamp_min(0.0)
    row_overflow = (overflow > 0).any(dim=1)
    stats = {
        "rows": int(q_raw.shape[0]),
        "clamped_rows": int(row_overflow.sum().item()),
        "clamped_fraction": float(row_overflow.float().mean().item()) if q_raw.shape[0] else 0.0,
        "max_overflow": float(overflow.max().item()) if overflow.numel() else 0.0,
        "q_raw_min": [float(v) for v in q_raw.amin(dim=0).detach().cpu().tolist()] if q_raw.shape[0] else [],
        "q_raw_max": [float(v) for v in q_raw.amax(dim=0).detach().cpu().tolist()] if q_raw.shape[0] else [],
    }
    return (q_raw.clamp(-1.0, 1.0) if clamp else q_raw), stats


def _tile_q_to_global_q(
    q_tile: torch.Tensor,
    *,
    transform: TileCameraTransform,
    camera: Mapping[str, float],
    clamp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Exact inverse projective recanonicalization, preserving q_global.z."""
    if q_tile.ndim != 2 or q_tile.shape[1] != 3:
        raise ValueError("q_tile must be [N,3]")
    st = float(transform.mesh_scale)
    qz = q_tile[:, 2]
    tile_points = torch.stack(
        [
            q_tile[:, 0] / (2.0 * st),
            q_tile[:, 1] / (2.0 * st),
            qz / (2.0 * st) - float(transform.distance),
        ],
        dim=1,
    )
    uv_tile, _, valid = _project_camera_points(
        tile_points,
        float(transform.camera_angle_x),
        int(transform.output_size),
    )
    uv_tile = uv_tile[0]
    # Do not reject rows solely because the model's canonical back half projects
    # slightly outside the nominal image edge. Positive depth/finite values are
    # the only hard requirements here.
    finite = torch.isfinite(uv_tile).all(dim=1) & torch.isfinite(tile_points).all(dim=1)
    positive_depth = (-tile_points[:, 2]) > 0
    if not bool((finite & positive_depth).all().item()):
        raise RuntimeError("tile coordinates produced invalid camera rays")

    x0, y0, _, _ = transform.box
    u_full = uv_tile[:, 0] / float(transform.crop_scale_x) + float(x0)
    v_full = uv_tile[:, 1] / float(transform.crop_scale_y) + float(y0)
    sg = float(camera["mesh_scale"])
    depth_global = float(camera["distance"]) - qz / (2.0 * sg)
    x_global = (
        (u_full - CANONICAL_IMAGE_SIZE / 2.0)
        * depth_global
        / float(transform.full_focal_pixels)
    )
    y_global = -(
        (v_full - CANONICAL_IMAGE_SIZE / 2.0)
        * depth_global
        / float(transform.full_focal_pixels)
    )
    global_points = torch.stack([x_global, y_global, -depth_global], dim=1)
    q_raw = torch.stack(
        [
            2.0 * sg * x_global,
            2.0 * sg * y_global,
            qz,
        ],
        dim=1,
    )
    overflow = (q_raw.abs() - 1.0).clamp_min(0.0)
    row_overflow = (overflow > 0).any(dim=1)
    stats = {
        "rows": int(q_raw.shape[0]),
        "clamped_rows": int(row_overflow.sum().item()),
        "clamped_fraction": float(row_overflow.float().mean().item()) if q_raw.shape[0] else 0.0,
        "max_overflow": float(overflow.max().item()) if overflow.numel() else 0.0,
        "q_raw_min": [float(v) for v in q_raw.amin(dim=0).detach().cpu().tolist()] if q_raw.shape[0] else [],
        "q_raw_max": [float(v) for v in q_raw.amax(dim=0).detach().cpu().tolist()] if q_raw.shape[0] else [],
    }
    q_out = q_raw.clamp(-1.0, 1.0) if clamp else q_raw
    return q_out, tile_points, global_points, stats


def _selected_global_coords_to_local_coords(
    base_coords128: torch.Tensor,
    input_rows: torch.Tensor,
    *,
    camera: Mapping[str, float],
    transform: TileCameraTransform,
    grid_resolution: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    selected = base_coords128.index_select(0, input_rows)
    q_global = _endpoint_indices_to_q(selected[:, 1:4], GRID_MASTER).to(selected.device)
    q_tile, stats = _global_q_to_tile_q(
        q_global,
        camera=camera,
        transform=transform,
        clamp=True,
    )
    xyz = _q_to_endpoint_indices(q_tile, grid_resolution).clamp(0, grid_resolution - 1)
    batch = torch.zeros((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
    coords = torch.cat([batch, xyz], dim=1).to(torch.int32)
    coords = torch.unique(coords, dim=0)
    if coords.numel() == 0:
        raise RuntimeError("tile projective coordinate quantization is empty")
    stats = {
        **stats,
        "input_rows": int(selected.shape[0]),
        "unique_coords": int(coords.shape[0]),
        "quantization_merge_rows": int(selected.shape[0] - coords.shape[0]),
    }
    return coords, stats


def _local_coords_to_global128(
    local_coords: torch.Tensor,
    *,
    transform: TileCameraTransform,
    camera: Mapping[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Map each tile C64 row to one global C128 coordinate ID.

    The integer IDs are never compared across spaces.  The mapping is:

        tile C64 ID -> q_tile -> inverse projective transform -> q_global
        -> nearest endpoint-aligned global C128 ID.

    Row order is preserved, so mapped_global_coords128[k] identifies the global
    master point whose current feature and velocity correspond to local row k.
    Multiple local rows and multiple tiles may map to the same global row; their
    velocities are averaged later.
    """
    q_tile = _endpoint_indices_to_q(
        local_coords[:, 1:4], GRID_TILE_HR
    ).to(local_coords.device)
    q_global, tile_points, global_points, stats = _tile_q_to_global_q(
        q_tile,
        transform=transform,
        camera=camera,
        clamp=True,
    )
    mapped_xyz = _q_to_endpoint_indices(q_global, GRID_MASTER).clamp(
        0, GRID_MASTER - 1
    )
    mapped = torch.cat(
        [
            torch.zeros(
                (mapped_xyz.shape[0], 1),
                device=mapped_xyz.device,
                dtype=torch.int32,
            ),
            mapped_xyz.to(torch.int32),
        ],
        dim=1,
    )
    unique_count = int(torch.unique(mapped, dim=0).shape[0])
    stats = {
        **stats,
        "mapped_rows": int(mapped.shape[0]),
        "mapped_unique_global_coords": unique_count,
        "many_to_one_rows": int(mapped.shape[0] - unique_count),
    }
    return mapped, tile_points, global_points, stats


def _estimate_camera(
    *,
    image_1024: Image.Image,
    output_dir: Path,
    manual_fov: float,
    mesh_scale: float,
    extend_pixel: int,
    image_resolution: int,
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if manual_fov > 0:
        distance = distance_from_fov(
            float(manual_fov),
            torch.tensor([-1.0, 0.0, 0.0]),
            torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
            float(mesh_scale),
            int(image_resolution),
        )["distance_from_x"]
        return {
            "camera_angle_x": float(manual_fov),
            "distance": float(distance),
            "mesh_scale": float(mesh_scale),
        }
    temporary = output_dir / f"_joint_tile_moge_{int(time.time()*1000)}.png"
    image_1024.save(temporary)
    print("[MoGe-2] loading global-image camera estimator")
    model = load_moge_model(device="cuda")
    try:
        params = get_camera_params_wild_moge(
            str(temporary),
            model,
            device="cuda",
            mesh_scale=float(mesh_scale),
            extend_pixel=int(extend_pixel),
            image_resolution=int(image_resolution),
        )
    finally:
        model.cpu()
        del model
        temporary.unlink(missing_ok=True)
        _empty_cuda_cache()
    return {
        "camera_angle_x": float(params["camera_angle_x"]),
        "distance": float(params["distance"]),
        "mesh_scale": float(params["mesh_scale"]),
    }


def _build_sampler_params(
    args: argparse.Namespace,
    pipeline: Any,
) -> Dict[str, Dict[str, Any]]:
    return {
        "ss": {
            **pipeline.sparse_structure_sampler_params,
            "steps": int(args.steps),
            "guidance_strength": float(args.ss_guidance_strength),
            "guidance_rescale": float(args.ss_guidance_rescale),
            "rescale_t": float(args.ss_rescale_t),
        },
        "shape": {
            **pipeline.shape_slat_sampler_params,
            "steps": int(args.steps),
            "guidance_strength": float(args.shape_guidance_strength),
            "guidance_rescale": float(args.shape_guidance_rescale),
            "rescale_t": float(args.shape_rescale_t),
        },
        "texture": {
            **pipeline.tex_slat_sampler_params,
            "steps": int(args.steps),
            "guidance_strength": float(args.texture_guidance_strength),
            "guidance_rescale": float(args.texture_guidance_rescale),
            "rescale_t": float(args.texture_rescale_t),
        },
    }


def _global_initial_support(
    *,
    pipeline: Any,
    image_512: Image.Image,
    camera: Mapping[str, float],
    sampler_params: Mapping[str, Mapping[str, Any]],
    seed: int,
    max_num_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, RecordedFlow]:
    cond_ss = pipeline.get_proj_cond_ss(
        [image_512],
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
    )
    _seed_everything(seed)
    coords32 = pipeline.sample_sparse_structure(
        cond_ss,
        resolution=GRID_LR,
        sampler_params=dict(sampler_params["ss"]),
    )
    del cond_ss
    if coords32.numel() == 0:
        raise RuntimeError("global sparse structure is empty")
    print(f"[global-support] C32={coords32.shape[0]:,}")

    cond_lr = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_512,
        [image_512],
        coords32,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
        grid_resolution_override=GRID_LR,
    )
    model = pipeline.models["shape_slat_flow_model_512"]
    noise = SparseTensor(
        feats=_randn(
            coords32.shape[0],
            int(model.in_channels),
            device=pipeline.device,
            seed=seed + 101,
        ),
        coords=coords32,
    )
    flow = _run_recorded_flow(
        pipeline=pipeline,
        sampler=pipeline.shape_slat_sampler,
        flow_model=model,
        noise=noise,
        condition=cond_lr,
        sampler_params=sampler_params["shape"],
        description="Global shape SLat 512",
    )
    lr_denorm = _denormalize_sparse(flow.samples, pipeline.shape_slat_normalization)
    coords128 = _learned_upsample(pipeline, lr_denorm, target_grid=GRID_MASTER)
    if coords128.shape[0] > max_num_tokens:
        raise RuntimeError(
            f"C128_base has {coords128.shape[0]:,} tokens, exceeding "
            f"--max-num-tokens={max_num_tokens:,}"
        )
    print(f"[global-support] C128_base={coords128.shape[0]:,}")
    return coords32, coords128, flow


def _prepare_one_tile(
    *,
    pipeline: Any,
    tile_id: int,
    box: Tuple[int, int, int, int],
    image_4096: Image.Image,
    base_coords128: torch.Tensor,
    input_rows: torch.Tensor,
    camera: Mapping[str, float],
    sampler_params: Mapping[str, Mapping[str, Any]],
    base_seed: int,
    trace_dir: Path,
    save_lr_trace: bool,
    extend_pixel: int,
) -> TileExpert:
    transform = _build_tile_camera_transform(
        box=box,
        camera=camera,
        canonical_size=CANONICAL_IMAGE_SIZE,
        output_size=RESOLUTION_TILE_HR,
        extend_pixel=extend_pixel,
    )
    coords32, input_stats = _selected_global_coords_to_local_coords(
        base_coords128,
        input_rows,
        camera=camera,
        transform=transform,
        grid_resolution=GRID_LR,
    )
    tile_image_1024 = image_4096.crop(box).convert("RGB")
    if tile_image_1024.size != (RESOLUTION_TILE_HR, RESOLUTION_TILE_HR):
        tile_image_1024 = tile_image_1024.resize(
            (RESOLUTION_TILE_HR, RESOLUTION_TILE_HR), Image.Resampling.LANCZOS
        )
    tile_image_512 = tile_image_1024.resize(
        (RESOLUTION_LR, RESOLUTION_LR), Image.Resampling.LANCZOS
    )
    print(
        f"[tile-camera] tile={tile_id:02d} box={box} "
        f"fov={transform.camera_angle_x:.8f} distance={transform.distance:.8f} "
        f"mesh_scale={transform.mesh_scale:.8f} focal={transform.tile_focal_pixels:.3f}"
    )
    print(
        f"[tile-prepare] tile={tile_id:02d} base_rows={input_rows.numel():,} "
        f"C32_local={coords32.shape[0]:,} "
        f"input_clamped={input_stats['clamped_rows']:,}/{input_stats['rows']:,} "
        f"max_overflow={input_stats['max_overflow']:.6f}"
    )

    cond_lr = _get_tile_projection_condition(
        pipeline=pipeline,
        image_cond_model=pipeline.image_cond_model_shape_512,
        image=tile_image_512,
        coords=coords32,
        transform=transform,
        grid_resolution=GRID_LR,
    )
    model_lr = pipeline.models["shape_slat_flow_model_512"]
    lr_seed = base_seed + tile_id * 100 + 1
    noise_lr = SparseTensor(
        feats=_randn(
            coords32.shape[0],
            int(model_lr.in_channels),
            device=pipeline.device,
            seed=lr_seed,
        ),
        coords=coords32,
    )
    lr_flow = _run_recorded_flow(
        pipeline=pipeline,
        sampler=pipeline.shape_slat_sampler,
        flow_model=model_lr,
        noise=noise_lr,
        condition=cond_lr,
        sampler_params=sampler_params["shape"],
        description=f"Tile {tile_id:02d} shape SLat 512",
    )
    lr_denorm = _denormalize_sparse(lr_flow.samples, pipeline.shape_slat_normalization)
    local_coords64 = _learned_upsample(
        pipeline,
        lr_denorm,
        target_grid=GRID_TILE_HR,
    )
    (
        mapped,
        local_tile_camera_points,
        local_global_camera_points,
        output_stats,
    ) = _local_coords_to_global128(
        local_coords64,
        transform=transform,
        camera=camera,
    )
    print(
        f"[tile-prepare] tile={tile_id:02d} C64_local={local_coords64.shape[0]:,} "
        f"mapped_C128_unique={torch.unique(mapped, dim=0).shape[0]:,} "
        f"output_clamped={output_stats['clamped_rows']:,}/{output_stats['rows']:,} "
        f"max_overflow={output_stats['max_overflow']:.6f}"
    )
    shape_condition = _get_tile_projection_condition(
        pipeline=pipeline,
        image_cond_model=pipeline.image_cond_model_shape_1024,
        image=tile_image_1024,
        coords=local_coords64,
        transform=transform,
        grid_resolution=GRID_TILE_HR,
    )
    texture_condition = _get_tile_projection_condition(
        pipeline=pipeline,
        image_cond_model=pipeline.image_cond_model_tex_1024,
        image=tile_image_1024,
        coords=local_coords64,
        transform=transform,
        grid_resolution=GRID_TILE_HR,
    )
    shape_condition_cpu = _tree_to_cpu(shape_condition)
    texture_condition_cpu = _tree_to_cpu(texture_condition)

    trace_path_value: Optional[str] = None
    if save_lr_trace:
        path = trace_dir / "tiles" / f"tile_{tile_id:04d}_support.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "format": "pixal3d_2048_tile_point_velocity_support_v1",
                "tile_id": tile_id,
                "box_4096": list(box),
                "transform": asdict(transform),
                "input_transform_stats": dict(input_stats),
                "output_transform_stats": dict(output_stats),
                "input_base_rows": input_rows.cpu(),
                "input_base_coords128": base_coords128.index_select(0, input_rows).cpu(),
                "coords32_local": coords32.cpu(),
                "coords64_local": local_coords64.cpu(),
                "coords128_mapped": mapped.cpu(),
                "tile_camera_points64": local_tile_camera_points.cpu(),
                "global_camera_points64": local_global_camera_points.cpu(),
                "shape_512": {
                    "times": torch.as_tensor(lr_flow.trajectory.times),
                    "time_intervals": torch.as_tensor(lr_flow.trajectory.time_intervals),
                    "states": [state.detach().cpu() for state in lr_flow.trajectory.states],
                    "velocities": [v.detach().cpu() for v in lr_flow.trajectory.velocities],
                    "final_samples": lr_flow.samples.feats.detach().cpu(),
                },
                "seed_shape_512": lr_seed,
            },
            temporary,
        )
        temporary.replace(path)
        trace_path_value = str(path.resolve())

    del cond_lr, noise_lr, lr_denorm, lr_flow, shape_condition, texture_condition
    _empty_cuda_cache()
    return TileExpert(
        tile_id=int(tile_id),
        box=tuple(int(v) for v in box),
        transform=transform,
        input_base_rows=input_rows.detach().cpu(),
        input_local_coords32=coords32.detach().cpu(),
        local_coords64=local_coords64.detach().cpu(),
        local_tile_camera_points64=local_tile_camera_points.detach().cpu(),
        local_global_camera_points64=local_global_camera_points.detach().cpu(),
        mapped_global_coords128=mapped.detach().cpu(),
        shape_condition_cpu=shape_condition_cpu,
        texture_condition_cpu=texture_condition_cpu,
        input_transform_stats=dict(input_stats),
        output_transform_stats=dict(output_stats),
        lr_trace_path=trace_path_value,
    )


def _build_master_support(
    base_coords128: torch.Tensor,
    tile_experts: Sequence[TileExpert],
    max_num_tokens: int,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Union C128_base with the single mapped global ID of every tile row."""
    base_cpu = base_coords128.detach().to(device="cpu", dtype=torch.int64)
    base_keys = [tuple(int(v) for v in row) for row in base_cpu.tolist()]
    if len(set(base_keys)) != len(base_keys):
        raise RuntimeError("C128_base contains duplicate coordinates")

    known = set(base_keys)
    extras: set[Tuple[int, int, int, int]] = set()
    total_local_rows = 0
    total_unique_tile_coords = 0
    for expert in tile_experts:
        mapped = expert.mapped_global_coords128.detach().to(
            device="cpu", dtype=torch.int64
        )
        total_local_rows += int(mapped.shape[0])
        unique_mapped = torch.unique(mapped, dim=0)
        total_unique_tile_coords += int(unique_mapped.shape[0])
        for values in unique_mapped.tolist():
            key = tuple(int(v) for v in values)
            if key not in known:
                extras.add(key)

    ordered = base_keys + sorted(extras)
    if len(ordered) > max_num_tokens:
        raise RuntimeError(
            f"C128_master has {len(ordered):,} tokens, exceeding "
            f"--max-num-tokens={max_num_tokens:,}"
        )
    master = torch.tensor(
        ordered, dtype=base_coords128.dtype, device=base_coords128.device
    )
    return master, {
        "base_tokens": len(base_keys),
        "tile_local_rows_total": total_local_rows,
        "tile_unique_mapped_coords_sum": total_unique_tile_coords,
        "added_unique_tokens": len(extras),
        "master_tokens": len(ordered),
    }


def _bind_tile_experts_to_master(
    *,
    tile_experts: Sequence[TileExpert],
    master_coords128: torch.Tensor,
    camera: Mapping[str, float],
) -> List[TileExpert]:
    """Build a one-to-one row correspondence for each local row.

    One-to-one here means one local row reads/writes one global master row.  The
    relation is many-to-one globally: several local rows or several tiles may
    map to the same master row, and their velocities are averaged there.
    """
    del camera
    master_map = _coord_key_rows(master_coords128)
    usable: List[TileExpert] = []
    for expert in tile_experts:
        mapped_cpu = expert.mapped_global_coords128.detach().to(
            device="cpu", dtype=torch.int64
        )
        if mapped_cpu.shape[0] != expert.local_coords64.shape[0]:
            raise RuntimeError(
                f"tile {expert.tile_id}: mapped/local row counts differ "
                f"({mapped_cpu.shape[0]} vs {expert.local_coords64.shape[0]})"
            )
        master_rows = torch.tensor(
            [master_map[tuple(int(v) for v in row)] for row in mapped_cpu.tolist()],
            dtype=torch.long,
        )
        count = int(master_rows.numel())
        if count == 0:
            print(f"[tile-bind] tile={expert.tile_id:02d} has no local rows")
            continue

        # All generated C64 rows retain their global point correspondence.
        # The tent weight only blends overlapping image tiles; it does not
        # change point identity or perform latent interpolation.
        uv, _, _ = _project_camera_points(
            expert.local_tile_camera_points64,
            float(expert.transform.camera_angle_x),
            int(expert.transform.output_size),
        )
        uv = uv[0]
        local_x = (uv[:, 0] / float(expert.transform.output_size)).clamp(0.0, 1.0)
        local_y = (uv[:, 1] / float(expert.transform.output_size)).clamp(0.0, 1.0)
        tent_weights = (
            (1.0 - (2.0 * local_x - 1.0).abs())
            * (1.0 - (2.0 * local_y - 1.0).abs())
        ).clamp_min(1e-3).to(torch.float32)
        local_rows = torch.arange(count, dtype=torch.long)

        expert.local_to_master_row = master_rows
        expert.active_local_rows = local_rows
        expert.active_tent_weights = tent_weights.detach().cpu()
        unique_master = int(torch.unique(master_rows).numel())
        duplicate_rows = count - unique_master
        print(
            f"[tile-bind] tile={expert.tile_id:02d} local={count:,} "
            f"unique_master={unique_master:,} many_to_one={duplicate_rows:,}"
        )
        usable.append(expert)
    return usable


@torch.no_grad()
def _run_online_master_flow(
    *,
    pipeline: Any,
    sampler: Any,
    flow_model: torch.nn.Module,
    master_state: SparseTensor,
    global_condition: Mapping[str, Any],
    tile_experts: Sequence[TileExpert],
    sampler_params: Mapping[str, Any],
    replace_last_n: int,
    replace_alpha: float,
    stage: str,
    global_concat_cond: Optional[SparseTensor] = None,
    save_step_states: bool = True,
) -> OnlineFlowResult:
    """Run one synchronized C128 trajectory with point-identity tile fusion.

    Shapes per tile with K local rows and C latent channels:

      local_to_master_row: [K]
      master_state.feats:  [N,C]
      tile_state.feats:    [K,C] = index_select(master_state, rows)
      tile_velocity:       [K,C]

    Multiple local rows/tiles writing the same master row are accumulated with
    index_add and normalized by their scalar tent weights.  No latent or
    velocity is spread to neighboring C128 lattice IDs.
    """
    if stage not in {"shape", "texture"}:
        raise ValueError(stage)
    steps = int(sampler_params.get("steps", 12))
    if not 0 <= replace_last_n <= steps:
        raise ValueError(f"{stage}: replace_last_n must be in [0,{steps}]")
    if not 0.0 <= replace_alpha <= 1.0:
        raise ValueError("replace_alpha must be in [0,1]")

    times = [
        float(v)
        for v in sampler.timestep_schedule(
            steps,
            float(sampler_params.get("rescale_t", 1.0)),
        )
    ]
    intervals = [times[i] - times[i + 1] for i in range(steps)]
    start_step = steps - replace_last_n
    step_kwargs = _sample_once_kwargs(sampler_params)
    if pipeline.low_vram:
        flow_model.to(pipeline.device)
    device = master_state.device
    global_condition_device = _tree_to_device(global_condition, device)
    if global_concat_cond is not None and not torch.equal(
        master_state.coords, global_concat_cond.coords
    ):
        raise RuntimeError(f"{stage}: master/concat coordinates differ")

    states_cpu: List[torch.Tensor] = []
    velocities_cpu: List[torch.Tensor] = []
    if save_step_states:
        states_cpu.append(master_state.feats.detach().cpu().clone())
    records: List[Dict[str, Any]] = []
    union_covered = torch.zeros(
        master_state.feats.shape[0], dtype=torch.bool, device=device
    )

    progress = tqdm(range(steps), desc=f"C128 master joint {stage}", dynamic_ncols=True)
    for step in progress:
        t = times[step]
        t_next = times[step + 1]
        dt = intervals[step]

        global_call = {**global_condition_device, **step_kwargs}
        if global_concat_cond is not None:
            global_call["concat_cond"] = global_concat_cond
        global_out = sampler.sample_once(
            flow_model,
            master_state,
            t,
            t_next,
            **global_call,
        )
        global_velocity = _features(global_out.pred_v).to(torch.float32)  # [N,C]
        merged = global_velocity.clone()                                  # [N,C]
        velocity_sum = torch.zeros_like(global_velocity)                  # [N,C]
        weight_sum = torch.zeros(
            (global_velocity.shape[0], 1), device=device, dtype=torch.float32
        )                                                                 # [N,1]
        tile_calls = 0
        local_rows_evaluated = 0

        if step >= start_step:
            for expert in tile_experts:
                if (
                    expert.local_to_master_row is None
                    or expert.active_local_rows is None
                    or expert.active_tent_weights is None
                ):
                    raise RuntimeError(f"tile {expert.tile_id}: incomplete master binding")

                local_to_master = expert.local_to_master_row.to(
                    device=device, dtype=torch.long
                )                                                         # [K]
                local_coords = expert.local_coords64.to(device=device)    # [K,4]
                tile_features = master_state.feats.index_select(
                    0, local_to_master
                )                                                         # [K,C]
                tile_state = SparseTensor(feats=tile_features, coords=local_coords)

                condition_cpu = (
                    expert.shape_condition_cpu
                    if stage == "shape"
                    else expert.texture_condition_cpu
                )
                tile_call = {**_tree_to_device(condition_cpu, device), **step_kwargs}
                tile_shape_features: Optional[torch.Tensor] = None
                if global_concat_cond is not None:
                    tile_shape_features = global_concat_cond.feats.index_select(
                        0, local_to_master
                    )                                                     # [K,Cshape]
                    tile_call["concat_cond"] = SparseTensor(
                        feats=tile_shape_features,
                        coords=local_coords,
                    )

                tile_out = sampler.sample_once(
                    flow_model,
                    tile_state,
                    t,
                    t_next,
                    **tile_call,
                )
                tile_velocity = _features(tile_out.pred_v).to(torch.float32)  # [K,C]

                active_local = expert.active_local_rows.to(
                    device=device, dtype=torch.long
                )                                                         # [Ka]
                active_master = local_to_master.index_select(
                    0, active_local
                )                                                         # [Ka]
                tent = expert.active_tent_weights.to(
                    device=device, dtype=torch.float32
                )                                                         # [Ka]
                selected_velocity = tile_velocity.index_select(
                    0, active_local
                )                                                         # [Ka,C]

                velocity_sum.index_add_(
                    0,
                    active_master,
                    selected_velocity * tent[:, None],
                )
                weight_sum.index_add_(
                    0,
                    active_master,
                    tent[:, None],
                )
                tile_calls += 1
                local_rows_evaluated += int(active_local.numel())

                del (
                    tile_features,
                    tile_state,
                    tile_call,
                    tile_out,
                    tile_velocity,
                    active_master,
                    selected_velocity,
                )
                if tile_shape_features is not None:
                    del tile_shape_features

            covered = weight_sum[:, 0] > 0
            union_covered |= covered
            if torch.any(covered):
                local_mean = velocity_sum[covered] / weight_sum[covered]  # [Nc,C]
                merged[covered] = (
                    (1.0 - replace_alpha) * global_velocity[covered]
                    + replace_alpha * local_mean
                )
        else:
            covered = torch.zeros(
                global_velocity.shape[0], dtype=torch.bool, device=device
            )

        next_state = master_state.replace(
            master_state.feats - float(dt) * merged.to(master_state.dtype)
        )
        if not torch.isfinite(next_state.feats).all():
            raise RuntimeError(f"{stage}: non-finite master state at step {step}")

        covered_count = int(covered.sum().item())
        if covered_count:
            local_mean_all = velocity_sum[covered] / weight_sum[covered]
            global_covered = global_velocity[covered]
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    local_mean_all.flatten()[None],
                    global_covered.flatten()[None],
                ).item()
            )
            norm_ratio = float(
                local_mean_all.norm().item()
                / max(global_covered.norm().item(), 1e-12)
            )
        else:
            cosine = 1.0
            norm_ratio = 1.0

        record = {
            "step": step,
            "t": t,
            "t_next": t_next,
            "dt": dt,
            "replacement_active": step >= start_step,
            "covered_rows": covered_count,
            "covered_ratio": covered_count / float(master_state.feats.shape[0]),
            "tile_experts_called": tile_calls,
            "local_rows_evaluated": local_rows_evaluated,
            "fallback_global_rows": int(master_state.feats.shape[0] - covered_count),
            "local_vs_global_cosine_covered": cosine,
            "local_to_global_norm_ratio_covered": norm_ratio,
        }
        records.append(record)
        progress.set_postfix(
            covered=f"{covered_count}/{master_state.feats.shape[0]}",
            local=local_rows_evaluated,
            tiles=tile_calls,
            replace=int(step >= start_step),
            cos=f"{cosine:.4f}",
        )

        master_state = next_state
        if save_step_states:
            velocities_cpu.append(merged.detach().cpu().clone())
            states_cpu.append(master_state.feats.detach().cpu().clone())
        del global_out, global_velocity, merged, velocity_sum, weight_sum

    if pipeline.low_vram:
        flow_model.cpu()
        _empty_cuda_cache()
    return OnlineFlowResult(
        samples=master_state,
        times=times,
        time_intervals=intervals,
        states=states_cpu,
        velocities=velocities_cpu,
        step_records=records,
        covered_rows_union=int(union_covered.sum().item()),
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_global_trace(
    *,
    path: Path,
    camera: Mapping[str, float],
    base_coords128: torch.Tensor,
    master_coords128: torch.Tensor,
    global_lr_flow: RecordedFlow,
    tile_experts: Sequence[TileExpert],
    shape_flow: OnlineFlowResult,
    texture_flow: OnlineFlowResult,
    shape_final_norm: SparseTensor,
    texture_final_norm: SparseTensor,
    sampler_params: Mapping[str, Mapping[str, Any]],
    master_stats: Mapping[str, int],
) -> None:
    tile_payload = []
    for expert in tile_experts:
        tile_payload.append(
            {
                "tile_id": expert.tile_id,
                "box": list(expert.box),
                "transform": asdict(expert.transform),
                "input_transform_stats": dict(expert.input_transform_stats),
                "output_transform_stats": dict(expert.output_transform_stats),
                "input_base_rows": expert.input_base_rows,
                "coords32_local": expert.input_local_coords32,
                "coords64_local": expert.local_coords64.cpu(),
                "tile_camera_points64": expert.local_tile_camera_points64.cpu(),
                "global_camera_points64": expert.local_global_camera_points64.cpu(),
                "coords128_mapped": expert.mapped_global_coords128.cpu(),
                "local_to_master_row": (
                    None
                    if expert.local_to_master_row is None
                    else expert.local_to_master_row.cpu()
                ),
                "active_local_rows": expert.active_local_rows,
                "active_tent_weights": expert.active_tent_weights,
                "lr_trace_path": expert.lr_trace_path,
            }
        )
    payload = {
        "format": "pixal3d_joint_tile_cascade_2048_master128_point_velocity_v1",
        "coordinate_system": {
            "global_master_grid": GRID_MASTER,
            "global_flow_model_grid": GRID_MASTER,
            "tile_lr_grid": GRID_LR,
            "tile_hr_grid": GRID_TILE_HR,
            "tile_projection": "global q -> crop pixels -> centered tile camera q, preserving normalized z",
            "master_update": "one Euler update per step",
            "resampling": "tile ID -> q_tile -> q_global -> one C128 ID; direct feature gather; tent-weighted multi-tile velocity mean",
        },
        "camera": dict(camera),
        "sampler_params": {key: dict(value) for key, value in sampler_params.items()},
        "master_stats": dict(master_stats),
        "base_coords128": base_coords128.cpu(),
        "master_coords128": master_coords128.cpu(),
        "global_shape_512": {
            "times": torch.as_tensor(global_lr_flow.trajectory.times),
            "time_intervals": torch.as_tensor(global_lr_flow.trajectory.time_intervals),
            "states": [state.detach().cpu() for state in global_lr_flow.trajectory.states],
            "velocities": [v.detach().cpu() for v in global_lr_flow.trajectory.velocities],
            "final_samples": global_lr_flow.samples.feats.detach().cpu(),
        },
        "tiles": tile_payload,
        "shape_2048": {
            "times": torch.as_tensor(shape_flow.times),
            "time_intervals": torch.as_tensor(shape_flow.time_intervals),
            "states": shape_flow.states,
            "velocities": shape_flow.velocities,
            "step_records": shape_flow.step_records,
            "covered_rows_union": shape_flow.covered_rows_union,
            "final_normalized": shape_final_norm.feats.detach().cpu(),
        },
        "texture_2048": {
            "times": torch.as_tensor(texture_flow.times),
            "time_intervals": torch.as_tensor(texture_flow.time_intervals),
            "states": texture_flow.states,
            "velocities": texture_flow.velocities,
            "step_records": texture_flow.step_records,
            "covered_rows_union": texture_flow.covered_rows_union,
            "final_normalized": texture_final_norm.feats.detach().cpu(),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _export_glb(
    *,
    pipeline: Any,
    shape_slat: SparseTensor,
    texture_slat: SparseTensor,
    output_path: Path,
    texture_size: int,
    decimation_target: int,
    postprocess_cache_dir: Path,
    camera: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    if o_voxel is None:
        raise RuntimeError("o_voxel is unavailable; run with --no-decode or fix the environment")
    meshes = pipeline.decode_latent(shape_slat, texture_slat, RESOLUTION_MASTER)
    mesh = meshes[0]
    vertices = int(mesh.vertices.shape[0])
    faces = int(mesh.faces.shape[0])
    effective_target = faces if decimation_target <= 0 else min(faces, int(decimation_target))
    print(f"[decode] vertices={vertices:,} faces={faces:,}")

    export_kwargs = {
        "vertices": mesh.vertices,
        "faces": mesh.faces,
        "attr_volume": mesh.attrs,
        "coords": mesh.coords,
        "attr_layout": pipeline.pbr_attr_layout,
        "grid_size": RESOLUTION_MASTER,
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "decimation_target": effective_target,
        "texture_size": int(texture_size),
        "remesh": False,
        "use_tqdm": True,
        "verbose": False,
    }

    # Save the exact decoder/postprocess inputs before UV baking.  The repository
    # evaluator render_pixal3d_cache_no_uv.py consumes this cache and uses the
    # same aligned Pixal3D camera for PSNR/SSIM/LPIPS.
    try:
        from pixal3d_directory_texture_eval import save_to_glb_cache
    except Exception as exc:
        raise RuntimeError(
            "cannot import save_to_glb_cache from pixal3d_directory_texture_eval.py"
        ) from exc

    cache_manifest = save_to_glb_cache(
        postprocess_cache_dir,
        export_kwargs,
        extra_metadata={
            "camera_params": dict(camera),
            "pipeline_resolution": RESOLUTION_MASTER,
            "actual_grid_resolution": GRID_MASTER,
            "seed": int(seed),
            "decoder_vertices": vertices,
            "decoder_faces": faces,
            "flow_experiment": "joint_tile_cascade_master128_point_velocity",
        },
        overwrite=True,
    )
    # render_pixal3d_cache_no_uv.py currently reads grid_size from the top-level
    # manifest even though save_to_glb_cache stores it in python_meta.pt. Mirror
    # the scalar here so alignment verification can run without changing the
    # evaluator script.
    manifest_path = postprocess_cache_dir / "manifest.json"
    published_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    published_manifest["grid_size"] = int(RESOLUTION_MASTER)
    published_manifest["aabb"] = export_kwargs["aabb"]
    _atomic_json(manifest_path, published_manifest)
    cache_manifest = published_manifest
    print(f"[postprocess-cache] {postprocess_cache_dir}")

    glb = o_voxel.postprocess.to_glb(**export_kwargs)
    glb.apply_transform(PIXAL3D_EXPORT_ROTATION)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(output_path), extension_webp=False)
    print(f"[done] GLB saved to {output_path}")
    return {
        "decoder_vertices": vertices,
        "decoder_faces": faces,
        "effective_decimation_target": effective_target,
        "postprocess_cache": str(postprocess_cache_dir),
        "postprocess_cache_manifest": cache_manifest,
    }


def _run_aligned_render_eval(
    *,
    trace_dir: Path,
    reference_image: Path,
    light: str,
    render_resolution: int,
    metric_resolution: int,
    blender_samples: int,
    lpips_net: str,
    blender: str,
) -> Dict[str, Any]:
    cache_dir = trace_dir / "postprocess_cache"
    output_dir = trace_dir / "aligned_eval"
    script_path = Path(__file__).resolve().parent / "render_pixal3d_cache_no_uv.py"
    if not script_path.is_file():
        raise FileNotFoundError(
            f"aligned evaluator is missing: {script_path}"
        )
    command = [
        sys.executable,
        str(script_path),
        "--cache-dir", str(cache_dir),
        "--output-dir", str(output_dir),
        "--reference-image", str(reference_image),
        "--lights", str(light),
        "--engine", "cycles",
        "--material-mode", "pbr",
        "--render-resolution", str(int(render_resolution)),
        "--metric-resolution", str(int(metric_resolution)),
        "--samples", str(int(blender_samples)),
        "--lpips-net", str(lpips_net),
        "--metric-device", "cuda",
        "--blender", str(blender),
        "--overwrite-renders",
    ]
    print("[render-eval-command] " + shlex.join(command))
    process = subprocess.run(command, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"aligned render evaluator failed with exit code {process.returncode}")

    metrics_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = metrics_payload.get("rows", metrics_payload.get("metrics", []))
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("status") == "success":
                print(
                    "[metrics] "
                    f"light={row.get('light')} "
                    f"PSNR={row.get('psnr_db')} "
                    f"SSIM={row.get('ssim')} "
                    f"LPIPS={row.get('lpips')}"
                )
    print(f"[render-eval] output={output_dir}")
    return {
        "aligned_eval_dir": str(output_dir),
        "aligned_eval_metrics_json": str(metrics_path),
        "aligned_eval_metrics_csv": str(csv_path),
        "aligned_eval_payload": metrics_payload,
    }


def _parse_tile_ids(value: Optional[str]) -> Optional[set[int]]:
    if value is None or not value.strip():
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}



def _active_unique_master_count(expert: TileExpert) -> int:
    if expert.local_to_master_row is None or expert.active_local_rows is None:
        return 0
    active = expert.active_local_rows.to(dtype=torch.long)
    rows = expert.local_to_master_row.index_select(0, active)
    return int(torch.unique(rows).numel())


def run_experiment(args: argparse.Namespace) -> None:
    if args.steps != 12:
        raise ValueError("this experiment currently requires exactly 12 Euler steps")
    if args.tile_size != 1024 or args.tile_stride != 512:
        raise ValueError("this implementation requires tile-size=1024, tile-stride=512")

    output_path = Path(args.output).expanduser().resolve()
    trace_dir = Path(args.trace_dir).expanduser().resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = init_pipeline(
        args.model_path,
        device="cuda",
        low_vram=bool(args.low_vram),
    )
    canonical = pipeline.preprocess_canonical_images(Image.open(args.image))
    image_4096: Image.Image = canonical["image_4096"]
    image_1024: Image.Image = canonical["image_1024"]
    image_512: Image.Image = canonical["image_512"]
    image_4096.save(trace_dir / "canonical_4096.png")
    image_1024.save(trace_dir / "canonical_1024.png")
    image_512.save(trace_dir / "canonical_512.png")
    metric_reference_path = trace_dir / "metric_reference_rgb.png"
    reference_rgba = image_1024.convert("RGBA")
    reference_black = Image.new("RGBA", reference_rgba.size, (0, 0, 0, 255))
    reference_black.alpha_composite(reference_rgba)
    reference_black.convert("RGB").save(metric_reference_path)

    camera = _estimate_camera(
        image_1024=image_1024,
        output_dir=trace_dir,
        manual_fov=float(args.fov),
        mesh_scale=float(args.mesh_scale),
        extend_pixel=int(args.extend_pixel),
        image_resolution=int(args.camera_image_resolution),
    )
    print(
        f"[camera] fov={camera['camera_angle_x']:.8f} "
        f"distance={camera['distance']:.8f} mesh_scale={camera['mesh_scale']:.8f}"
    )
    sampler_params = _build_sampler_params(args, pipeline)
    _, base_coords128, global_lr_flow = _global_initial_support(
        pipeline=pipeline,
        image_512=image_512,
        camera=camera,
        sampler_params=sampler_params,
        seed=int(args.seed),
        max_num_tokens=int(args.max_num_tokens),
    )

    base_camera_points = _global_coords_to_camera(
        base_coords128,
        grid_resolution=GRID_MASTER,
        camera=camera,
    )
    base_uv, _, base_valid = _project_camera_points(
        base_camera_points,
        float(camera["camera_angle_x"]),
        CANONICAL_IMAGE_SIZE,
    )
    base_uv = base_uv[0]
    base_valid = base_valid[0]

    boxes = _tile_layout(
        CANONICAL_IMAGE_SIZE,
        int(args.tile_size),
        int(args.tile_stride),
    )
    selected_ids = _parse_tile_ids(args.tile_ids)
    tile_experts: List[TileExpert] = []
    processed = 0
    for tile_id, box in enumerate(boxes):
        if selected_ids is not None and tile_id not in selected_ids:
            continue
        rows, _ = _rows_and_tent_weights(base_uv, base_valid, box)
        if rows.numel() < int(args.min_tile_tokens):
            print(
                f"[tile-skip] tile={tile_id:02d} tokens={rows.numel():,} "
                f"< min={args.min_tile_tokens}"
            )
            continue
        if args.max_tiles is not None and processed >= int(args.max_tiles):
            break
        try:
            expert = _prepare_one_tile(
                pipeline=pipeline,
                tile_id=tile_id,
                box=box,
                image_4096=image_4096,
                base_coords128=base_coords128,
                input_rows=rows,
                camera=camera,
                sampler_params=sampler_params,
                base_seed=int(args.seed) + 1000,
                trace_dir=trace_dir,
                save_lr_trace=bool(args.save_tile_lr_traces),
                extend_pixel=int(args.extend_pixel),
            )
        except Exception as exc:
            if args.strict_tiles:
                raise
            print(f"[tile-error] tile={tile_id:02d}: {type(exc).__name__}: {exc}")
            _empty_cuda_cache()
            continue
        tile_experts.append(expert)
        processed += 1

    if not tile_experts:
        raise RuntimeError("no usable tile experts were prepared")
    master_coords128, master_stats = _build_master_support(
        base_coords128,
        tile_experts,
        int(args.max_num_tokens),
    )
    print(
        f"[master-support] base={master_stats['base_tokens']:,} "
        f"added={master_stats['added_unique_tokens']:,} "
        f"master={master_stats['master_tokens']:,}"
    )
    tile_experts = _bind_tile_experts_to_master(
        tile_experts=tile_experts,
        master_coords128=master_coords128,
        camera=camera,
    )
    if not tile_experts:
        raise RuntimeError("all prepared tiles became inactive after master binding")

    cond_global_shape = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [image_1024],
        master_coords128,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
        grid_resolution_override=GRID_MASTER,
    )
    shape_model = pipeline.models["shape_slat_flow_model_1024"]
    shape_noise = SparseTensor(
        feats=_randn(
            master_coords128.shape[0],
            int(shape_model.in_channels),
            device=pipeline.device,
            seed=int(args.seed) + 202,
        ),
        coords=master_coords128,
    )
    shape_online = _run_online_master_flow(
        pipeline=pipeline,
        sampler=pipeline.shape_slat_sampler,
        flow_model=shape_model,
        master_state=shape_noise,
        global_condition=cond_global_shape,
        tile_experts=tile_experts,
        sampler_params=sampler_params["shape"],
        replace_last_n=int(args.shape_replace_last_n),
        replace_alpha=float(args.replace_alpha),
        stage="shape",
        save_step_states=bool(args.save_step_states),
    )
    shape_norm = shape_online.samples
    shape_denorm = _denormalize_sparse(shape_norm, pipeline.shape_slat_normalization)
    del cond_global_shape, shape_noise
    _empty_cuda_cache()

    cond_global_texture = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [image_1024],
        master_coords128,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
        grid_resolution_override=GRID_MASTER,
    )
    texture_model = pipeline.models["tex_slat_flow_model_1024"]
    texture_channels = int(texture_model.in_channels) - int(shape_norm.feats.shape[1])
    if texture_channels <= 0:
        raise RuntimeError(f"invalid texture noise channel count {texture_channels}")
    texture_noise = SparseTensor(
        feats=_randn(
            master_coords128.shape[0],
            texture_channels,
            device=pipeline.device,
            seed=int(args.seed) + 303,
        ),
        coords=master_coords128,
    )
    texture_online = _run_online_master_flow(
        pipeline=pipeline,
        sampler=pipeline.tex_slat_sampler,
        flow_model=texture_model,
        master_state=texture_noise,
        global_condition=cond_global_texture,
        tile_experts=tile_experts,
        sampler_params=sampler_params["texture"],
        replace_last_n=int(args.texture_replace_last_n),
        replace_alpha=float(args.replace_alpha),
        stage="texture",
        global_concat_cond=shape_norm,
        save_step_states=bool(args.save_step_states),
    )
    texture_norm = texture_online.samples
    texture_denorm = _denormalize_sparse(texture_norm, pipeline.tex_slat_normalization)

    trace_path = trace_dir / "joint_master128_trace.pt"
    _save_global_trace(
        path=trace_path,
        camera=camera,
        base_coords128=base_coords128,
        master_coords128=master_coords128,
        global_lr_flow=global_lr_flow,
        tile_experts=tile_experts,
        shape_flow=shape_online,
        texture_flow=texture_online,
        shape_final_norm=shape_norm,
        texture_final_norm=texture_norm,
        sampler_params=sampler_params,
        master_stats=master_stats,
    )
    export_meta: Dict[str, Any] = {}
    if not args.no_decode:
        export_meta = _export_glb(
            pipeline=pipeline,
            shape_slat=shape_denorm,
            texture_slat=texture_denorm,
            output_path=output_path,
            texture_size=int(args.texture_size),
            decimation_target=int(args.decimation_target),
            postprocess_cache_dir=trace_dir / "postprocess_cache",
            camera=camera,
            seed=int(args.seed),
        )
        if args.render_eval:
            export_meta.update(
                _run_aligned_render_eval(
                    trace_dir=trace_dir,
                    reference_image=metric_reference_path,
                    light=str(args.light),
                    render_resolution=int(args.render_resolution),
                    metric_resolution=int(args.metric_resolution),
                    blender_samples=int(args.blender_samples),
                    lpips_net=str(args.lpips_net),
                    blender=str(args.blender),
                )
            )
    else:
        print("[done] --no-decode: fused 2048 latents and trajectories were saved")

    summary = {
        "format": "pixal3d_joint_tile_cascade_2048_master128_point_velocity_summary_v1",
        "image": str(Path(args.image).expanduser().resolve()),
        "output": str(output_path),
        "trace": str(trace_path),
        "camera": camera,
        "master_stats": master_stats,
        "processed_tiles": processed,
        "usable_tile_experts": len(tile_experts),
        "shape_replace_last_n": int(args.shape_replace_last_n),
        "texture_replace_last_n": int(args.texture_replace_last_n),
        "replace_alpha": float(args.replace_alpha),
        "shape_covered_rows": shape_online.covered_rows_union,
        "texture_covered_rows": texture_online.covered_rows_union,
        "tile_transforms": [
            {
                "tile_id": expert.tile_id,
                "box": list(expert.box),
                "local_tokens": int(expert.local_coords64.shape[0]),
                "active_rows": int(expert.active_local_rows.numel())
                if expert.active_local_rows is not None
                else 0,
                "unique_master_rows": _active_unique_master_count(expert),
                "transform": asdict(expert.transform),
                "input_transform_stats": dict(expert.input_transform_stats),
                "output_transform_stats": dict(expert.output_transform_stats),
            }
            for expert in tile_experts
        ],
        **export_meta,
    }
    _atomic_json(trace_dir / "summary.json", summary)
    print(f"[summary] {trace_dir / 'summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pixal3D 2048 master128 + tile-local bidirectional velocity fusion"
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--max-num-tokens", type=int, default=49152)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--tile-stride", type=int, default=DEFAULT_TILE_STRIDE)
    parser.add_argument("--min-tile-tokens", type=int, default=100)
    parser.add_argument("--tile-ids", default=None, help="comma-separated tile ids")
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--strict-tiles", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tile-scale-multiplier", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tile-fit-quantile", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tile-fit-margin", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shape-replace-last-n", type=int, default=6)
    parser.add_argument("--texture-replace-last-n", type=int, default=6)
    parser.add_argument("--replace-alpha", type=float, default=1.0)
    parser.add_argument("--save-step-states", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-tile-lr-traces", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--fov", type=float, default=-1.0)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument("--camera-image-resolution", type=int, default=1024)

    parser.add_argument("--ss-guidance-strength", type=float, default=7.5)
    parser.add_argument("--ss-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--ss-rescale-t", type=float, default=1.0)
    parser.add_argument("--shape-guidance-strength", type=float, default=7.5)
    parser.add_argument("--shape-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--shape-rescale-t", type=float, default=1.0)
    parser.add_argument("--texture-guidance-strength", type=float, default=1.0)
    parser.add_argument("--texture-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--texture-rescale-t", type=float, default=1.0)

    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-decode", action="store_true")
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--decimation-target", type=int, default=0)

    parser.add_argument("--render-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--light", default="studio")
    parser.add_argument("--render-resolution", type=int, default=2048)
    parser.add_argument("--metric-resolution", type=int, default=1024)
    parser.add_argument("--blender-samples", type=int, default=64)
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--blender", default="blender")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if any(
        value is not None
        for value in (args.tile_scale_multiplier, args.tile_fit_quantile, args.tile_fit_margin)
    ):
        print(
            "[deprecated] --tile-scale-multiplier/--tile-fit-quantile/"
            "--tile-fit-margin are ignored; projective tile recanonicalization "
            "uses no fitted 3-D cube or depth slab"
        )
    run_experiment(args)


if __name__ == "__main__":
    main()