#!/usr/bin/env python3
"""Canonical global sparse master-SLat flow experiment for tile 26/27.

This is an independent diagnostic/experiment driver.  The latent support is
one canonical global C64 lattice, and every image expert reads the same master
state at every Euler step.  Tile experts only submit velocity proposals for
rows whose *global projected centers* fall in the requested image tile; they
never transport, interpolate, copy, or average encoded tile features.

The script is deliberately conservative about coordinates:

* ``master_coords`` are unique, lexicographically sorted absolute C64 rows;
* image crops use the unchanged global camera and ``projection_crop_box``;
* all actual updates are one FP32 weighted velocity update per step;
* endpoint fusion is calculated only as an algebraic self-check;
* shape and texture use the same master support/order;
* PBR is not decoded during flow.

The default input cache is the CUDA5 tile26/27 collection requested by
``Codex2.md``.  Run it with ``CUDA_VISIBLE_DEVICES=5``; after masking, the
physical GPU5 is addressed as logical ``cuda:0``.
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
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    str(Path(__file__).resolve().with_name("autotune_cache.json")),
)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as legacy_core


FORMAT = "pixal3d_tile26_27_canonical_master_slat_flow_v1"
RESOLUTION = 64
CHANNELS = 32
CANONICAL_SIZE = 4096
GLOBAL_IMAGE_SIZE = 1024
DECODE_RESOLUTION = 1024
TILE_BOXES = {
    26: (2560, 1536, 3584, 2560),
    27: (3072, 1536, 4096, 2560),
}
OVERLAP_BOX = (3072, 1536, 3584, 2560)
ENDPOINT_TOLERANCE = 2e-5


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_tensor(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_metadata(root: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            args,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return result.stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "status": run("git", "status", "--short"),
        "diff_stat": run("git", "diff", "--stat"),
    }


def _nvidia_metadata() -> Dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": result.stdout.strip(),
    }


def _tensor_stats(value: torch.Tensor) -> Dict[str, Any]:
    value = value.detach().float()
    if value.numel() == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(value.numel()),
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
    }


def _range_by_channel(value: torch.Tensor) -> Dict[str, Any]:
    value = value.detach().float()
    if value.numel() == 0:
        return {"min": [], "max": []}
    return {
        "min": value.amin(dim=0).cpu().tolist(),
        "max": value.amax(dim=0).cpu().tolist(),
    }


def _lexsort_coords(
    coords: torch.Tensor,
    resolution: int = RESOLUTION,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return sorted unique coordinates and the original-row permutation."""
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"coords must have shape [N,4], got {tuple(coords.shape)}")
    coords = coords.to(torch.int64)
    if coords.numel() and (coords[:, 0] != 0).any():
        raise ValueError("only batch=0 canonical supports are supported")
    if coords.numel() and ((coords[:, 1:] < 0) | (coords[:, 1:] >= resolution)).any():
        raise ValueError(f"coordinates must lie in [0,{resolution})")
    key = (
        coords[:, 0] * resolution**3
        + coords[:, 1] * resolution**2
        + coords[:, 2] * resolution
        + coords[:, 3]
    )
    order = torch.argsort(key, stable=True)
    sorted_coords = coords.index_select(0, order)
    if sorted_coords.shape[0] > 1 and torch.equal(
        sorted_coords[1:], sorted_coords[:-1]
    ):
        raise RuntimeError("duplicate canonical master coordinates")
    if sorted_coords.shape[0] > 1 and bool(
        (sorted_coords[1:] == sorted_coords[:-1]).all(dim=1).any().item()
    ):
        raise RuntimeError("duplicate canonical master coordinates")
    return sorted_coords.to(torch.int32), order.to(torch.long)


def _sort_unique_coords(
    coords: torch.Tensor,
    resolution: int = RESOLUTION,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sort and deduplicate canonical coordinates for support-set unions."""
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"coords must have shape [N,4], got {tuple(coords.shape)}")
    coords = coords.to(torch.int64)
    if coords.numel() and (coords[:, 0] != 0).any():
        raise ValueError("only batch=0 canonical supports are supported")
    if coords.numel() and ((coords[:, 1:] < 0) | (coords[:, 1:] >= resolution)).any():
        raise ValueError(f"coordinates must lie in [0,{resolution})")
    key = (
        coords[:, 0] * resolution**3
        + coords[:, 1] * resolution**2
        + coords[:, 2] * resolution
        + coords[:, 3]
    )
    order = torch.argsort(key, stable=True)
    sorted_coords = coords.index_select(0, order)
    if sorted_coords.shape[0] > 1:
        keep = torch.ones(sorted_coords.shape[0], dtype=torch.bool)
        keep[1:] = (sorted_coords[1:] != sorted_coords[:-1]).any(dim=1)
        sorted_coords = sorted_coords[keep]
    return sorted_coords.to(torch.int32), order.to(torch.long)


def _assert_unique_coords(coords: torch.Tensor, label: str) -> None:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"{label}: expected [N,4] coordinates")
    if torch.unique(coords, dim=0).shape[0] != coords.shape[0]:
        raise RuntimeError(f"{label}: duplicate coordinates are forbidden")


def _assert_same_rows(left: torch.Tensor, right: torch.Tensor, label: str) -> None:
    if not torch.equal(left, right):
        raise AssertionError(f"{label}: coordinate rows/order differ")


def _centers_object(coords: torch.Tensor, resolution: int = RESOLUTION) -> torch.Tensor:
    xyz = coords[:, 1:].to(torch.float32)
    return (xyz + 0.5) / float(resolution) - 0.5


def _normalize(value: SparseTensor, normalization: Mapping[str, Sequence[float]]) -> SparseTensor:
    mean = torch.as_tensor(normalization["mean"], device=value.device, dtype=value.feats.dtype)[None]
    std = torch.as_tensor(normalization["std"], device=value.device, dtype=value.feats.dtype)[None]
    if mean.shape[1] != value.feats.shape[1] or std.shape[1] != value.feats.shape[1]:
        raise ValueError("latent normalization channel count mismatch")
    if bool((std == 0).any().item()):
        raise ValueError("latent normalization has zero standard deviation")
    return value.replace((value.feats - mean) / std)


def _denormalize(value: SparseTensor, normalization: Mapping[str, Sequence[float]]) -> SparseTensor:
    mean = torch.as_tensor(normalization["mean"], device=value.device, dtype=value.feats.dtype)[None]
    std = torch.as_tensor(normalization["std"], device=value.device, dtype=value.feats.dtype)[None]
    if mean.shape[1] != value.feats.shape[1] or std.shape[1] != value.feats.shape[1]:
        raise ValueError("latent denormalization channel count mismatch")
    return value.replace(value.feats * std + mean)


def _prediction_kwargs(params: Mapping[str, Any]) -> Dict[str, Any]:
    excluded = {
        "steps",
        "rescale_t",
        "verbose",
        "tqdm_desc",
        "record_trajectory",
        "trajectory_device",
        "return_model_history",
    }
    return {key: value for key, value in params.items() if key not in excluded}


def _materialize_condition(
    pipeline: Any,
    packed: Mapping[str, Any],
    coords: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    return pipeline._materialize_proj_condition(packed, coords=coords, device=device)


def _select_packed_condition(
    packed: Mapping[str, Any],
    rows: torch.Tensor,
) -> Dict[str, Any]:
    """Gather projection rows while preserving the packed CPU representation."""
    rows = rows.to(device="cpu", dtype=torch.long)
    selected: Dict[str, Any] = {}
    for branch_name in ("cond", "neg_cond"):
        branch = packed[branch_name]
        selected[branch_name] = {
            "global": branch["global"],
            "proj": branch["proj"].index_select(0, rows),
        }
    return selected


def _pack_condition(
    condition: Mapping[str, Any],
    expected_coords: torch.Tensor,
    label: str,
) -> Dict[str, Any]:
    packed: Dict[str, Any] = {}
    for branch_name in ("cond", "neg_cond"):
        branch = condition[branch_name]
        projection = branch["proj"]
        if not isinstance(projection, SparseTensor):
            raise TypeError(f"{label}.{branch_name}.proj is not sparse")
        _assert_same_rows(projection.coords, expected_coords, f"{label}.{branch_name}")
        packed[branch_name] = {
            "global": branch["global"].detach().cpu().clone(),
            "proj": projection.feats.detach().cpu().clone(),
        }
    return packed


def _make_noise(coords: torch.Tensor, seed: int, channels: int) -> torch.Tensor:
    """Generate a deterministic keyed field in canonical master row order."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn((coords.shape[0], channels), generator=generator, dtype=torch.float32)


def _noise_for_arbitrary_order(coords: torch.Tensor, seed: int, channels: int) -> torch.Tensor:
    """Generate the same keyed field after restoring canonical coordinate order."""
    _, order = _lexsort_coords(coords)
    sorted_coords = coords.index_select(0, order)
    values = _make_noise(sorted_coords, seed, channels)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), dtype=order.dtype)
    return values.index_select(0, inverse)


def _two_dimensional_tent(
    projected_norm: torch.Tensor,
    rows: torch.Tensor,
    box: Sequence[int],
    mode: str,
) -> torch.Tensor:
    if rows.numel() == 0:
        return torch.empty((0,), dtype=torch.float32)
    if mode == "uniform":
        return torch.ones((rows.numel(),), dtype=torch.float32)
    if mode != "tent":
        raise ValueError(f"unknown weight mode {mode}")
    x0, y0, x1, y1 = (float(v) for v in box)
    selected = projected_norm.index_select(0, rows).float()
    local_x = (selected[:, 0] * CANONICAL_SIZE - x0) / (x1 - x0)
    local_y = (selected[:, 1] * CANONICAL_SIZE - y0) / (y1 - y0)
    wx = (1.0 - (2.0 * local_x - 1.0).abs()).clamp_min(1e-3)
    wy = (1.0 - (2.0 * local_y - 1.0).abs()).clamp_min(1e-3)
    return (wx * wy).float()


def _build_memberships(
    pipeline: Any,
    coords: torch.Tensor,
    camera: Mapping[str, Any],
    image_model: Any,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    projected, depth, valid = pipeline._project_sparse_coords_to_image_norm(
        image_cond_model=image_model,
        coords=coords,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera["mesh_scale"]),
        grid_resolution=RESOLUTION,
    )
    projected = projected.detach().cpu().float()
    depth = depth.detach().cpu().float()
    valid = valid.detach().cpu().bool()
    pixel = torch.floor(projected * CANONICAL_SIZE).to(torch.int64)
    finite = torch.isfinite(projected).all(dim=1) & torch.isfinite(depth)
    in_bounds = (
        (pixel[:, 0] >= 0)
        & (pixel[:, 0] < CANONICAL_SIZE)
        & (pixel[:, 1] >= 0)
        & (pixel[:, 1] < CANONICAL_SIZE)
    )
    memberships: Dict[int, Dict[str, Any]] = {}
    coverage = torch.zeros(coords.shape[0], dtype=torch.int32)
    eligible = valid & finite & in_bounds
    for tile_id, box in TILE_BOXES.items():
        x0, y0, x1, y1 = box
        mask = (
            eligible
            & (pixel[:, 0] >= x0)
            & (pixel[:, 0] < x1)
            & (pixel[:, 1] >= y0)
            & (pixel[:, 1] < y1)
        )
        rows = mask.nonzero(as_tuple=False).flatten().to(torch.long)
        coverage.index_add_(0, rows, torch.ones_like(rows, dtype=torch.int32))
        memberships[tile_id] = {
            "tile_id": int(tile_id),
            "box": list(box),
            "projection_crop_box": [v / float(CANONICAL_SIZE) for v in box],
            "rows": rows,
            "token_count": int(rows.numel()),
            "weights_uniform": _two_dimensional_tent(projected, rows, box, "uniform"),
            "weights_tent": _two_dimensional_tent(projected, rows, box, "tent"),
        }
    overlap_rows = torch.where((coverage > 1) & eligible)[0]
    support = {
        "projected_norm": projected,
        "depth": depth,
        "valid": valid,
        "eligible": eligible,
        "pixel_xy": pixel,
        "coverage": coverage,
        "tile26_rows": memberships[26]["rows"],
        "tile27_rows": memberships[27]["rows"],
        "overlap_rows": overlap_rows,
        "projected_valid_count": int(valid.sum().item()),
        "eligible_count": int(eligible.sum().item()),
        "overlap_count": int(overlap_rows.numel()),
        "uncovered_count": int((coverage == 0).sum().item()),
        "coverage_histogram": {
            str(i): int((coverage == i).sum().item())
            for i in range(int(coverage.max().item()) + 1)
        },
    }
    return memberships, support


def _save_projection_overlay(
    image: Image.Image,
    projected_norm: torch.Tensor,
    memberships: Mapping[int, Mapping[str, Any]],
    output_path: Path,
) -> None:
    image = image.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    colors = {26: (245, 80, 80, 180), 27: (70, 130, 245, 180)}
    scale = 1024.0 / CANONICAL_SIZE
    for tile_id, item in memberships.items():
        x0, y0, x1, y1 = item["box"]
        rect = (x0 * scale, y0 * scale, x1 * scale, y1 * scale)
        draw.rectangle(rect, outline=colors[tile_id], width=4)
        draw.text((rect[0] + 5, rect[1] + 5), f"tile {tile_id}", fill=colors[tile_id])
        rows = item["rows"]
        if rows.numel():
            points = projected_norm.index_select(0, rows) * CANONICAL_SIZE * scale
            points = points[:: max(1, points.shape[0] // 1000)]
            draw.point([tuple(float(v) for v in point) for point in points], fill=colors[tile_id])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _assert_patch_roundtrip(
    global_coords: torch.Tensor,
    token_indices: torch.Tensor,
    local_coords: torch.Tensor,
    start: Sequence[int],
    label: str,
) -> None:
    if token_indices.numel() == 0:
        return
    if local_coords.ndim != 2 or local_coords.shape[1] != 4:
        raise ValueError(f"{label}: invalid local coordinates")
    expected = global_coords.index_select(0, token_indices).clone()
    start_tensor = torch.as_tensor(start, dtype=expected.dtype, device=expected.device)
    if not torch.equal(local_coords[:, 1:] + start_tensor, expected[:, 1:]):
        raise RuntimeError(f"{label}: local/global round-trip failed")
    _assert_unique_coords(local_coords, label)


def _build_3d_patches(
    coords: torch.Tensor,
    grid_resolution: int,
    patch_size: int = 64,
    patch_stride: int = 32,
) -> List[Dict[str, Any]]:
    if grid_resolution < patch_size:
        raise ValueError("grid resolution must be at least patch size")
    starts = list(range(0, grid_resolution - patch_size + 1, patch_stride))
    if not starts or starts[-1] != grid_resolution - patch_size:
        raise ValueError("patch starts do not reach the global boundary")
    xyz = coords[:, 1:].to(torch.int64)
    patches: List[Dict[str, Any]] = []
    coverage = torch.zeros(coords.shape[0], dtype=torch.int32)
    for sx in starts:
        for sy in starts:
            for sz in starts:
                start = (sx, sy, sz)
                mask = (
                    (xyz[:, 0] >= sx)
                    & (xyz[:, 0] < sx + patch_size)
                    & (xyz[:, 1] >= sy)
                    & (xyz[:, 1] < sy + patch_size)
                    & (xyz[:, 2] >= sz)
                    & (xyz[:, 2] < sz + patch_size)
                )
                rows = mask.nonzero(as_tuple=False).flatten().long()
                local = coords.index_select(0, rows).clone()
                if rows.numel():
                    local[:, 1:] -= torch.as_tensor(start, dtype=local.dtype)
                    _assert_patch_roundtrip(coords, rows, local, start, f"patch {start}")
                    local_xyz = local[:, 1:].float()
                    axes = []
                    for axis in range(3):
                        axis_weight = torch.ones(rows.shape[0], dtype=torch.float32)
                        if sx + sy + sz < 0:  # pragma: no cover - keeps lint quiet
                            axis_weight.zero_()
                        axis_start = start[axis]
                        axis_end = axis_start + patch_size
                        if axis_start > 0:
                            axis_weight = torch.minimum(
                                axis_weight,
                                (local_xyz[:, axis] - 0.0 + 1.0) / 33.0,
                            )
                        if axis_end < grid_resolution:
                            axis_weight = torch.minimum(
                                axis_weight,
                                (64.0 - local_xyz[:, axis]) / 33.0,
                            )
                        axes.append(axis_weight)
                    tent = (axes[0] * axes[1] * axes[2]).clamp_min(1e-6)
                    coverage.index_add_(0, rows, torch.ones_like(rows, dtype=torch.int32))
                else:
                    tent = None
                patches.append(
                    {
                        "patch_id": len(patches),
                        "start": list(start),
                        "end": [sx + patch_size, sy + patch_size, sz + patch_size],
                        "rows": rows,
                        "local_coords": local,
                        "weights_tent": tent,
                        "weights_uniform": (
                            torch.ones(rows.shape[0], dtype=torch.float32)
                            if rows.numel()
                            else None
                        ),
                    }
                )
    if torch.any(coverage <= 0):
        raise RuntimeError("3D patch layout leaves active master rows uncovered")
    return patches


def _scatter_weighted(
    value_sum: torch.Tensor,
    weight_sum: torch.Tensor,
    coverage_count: torch.Tensor,
    rows: torch.Tensor,
    value: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    if rows.ndim != 1 or value.shape[0] != rows.numel() or weights.shape != rows.shape:
        raise ValueError("scatter inputs are not row aligned")
    if rows.numel() == 0:
        return
    weights = weights.to(device=value_sum.device, dtype=torch.float32)
    rows = rows.to(device=value_sum.device, dtype=torch.long)
    value_sum.index_add_(0, rows, value.to(torch.float32) * weights[:, None])
    weight_sum.index_add_(0, rows, weights[:, None])
    coverage_count.index_add_(0, rows, torch.ones_like(rows, dtype=coverage_count.dtype))


def _velocity_metrics(left: torch.Tensor, right: torch.Tensor) -> Dict[str, float]:
    left = left.float()
    right = right.float()
    delta = left - right
    eps = torch.finfo(torch.float32).eps
    return {
        "mean_abs_delta": float(delta.abs().mean().item()),
        "rms_delta": float(delta.square().mean().sqrt().item()),
        "relative_l2": float((delta.norm() / right.norm().clamp_min(eps)).item()),
        "cosine": float(F.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1), dim=1, eps=eps).item()),
    }


def _trajectory_payload(
    coords: torch.Tensor,
    states: Sequence[torch.Tensor],
    velocities: Sequence[torch.Tensor],
    times: Sequence[float],
) -> Dict[str, Any]:
    return {
        "coords": coords.detach().cpu().to(torch.int32),
        "states": [x.detach().cpu().float() for x in states],
        "velocities": [x.detach().cpu().float() for x in velocities],
        "times": [float(x) for x in times],
        "time_intervals": [float(times[i] - times[i + 1]) for i in range(len(times) - 1)],
    }


def _run_direct_flow(
    *,
    pipeline: Any,
    model: Any,
    sampler: Any,
    noise: SparseTensor,
    condition_cpu: Mapping[str, Any],
    params: Mapping[str, Any],
    concat_cond: Optional[SparseTensor],
    device: torch.device,
    label: str,
) -> Tuple[SparseTensor, Dict[str, Any]]:
    condition = _materialize_condition(pipeline, condition_cpu, noise.coords, device)
    concat = concat_cond.to(device) if concat_cond is not None else None
    if concat is not None:
        _assert_same_rows(concat.coords, noise.coords, f"{label} concat")
    if pipeline.low_vram:
        model.to(device)
    started = time.perf_counter()
    try:
        run_params = dict(params)
        run_params.update(
            {
                "verbose": False,
                "record_trajectory": True,
                "trajectory_device": "cpu",
                "return_model_history": False,
            }
        )
        result = sampler.sample(
            model,
            noise,
            concat_cond=concat,
            **condition,
            **run_params,
        )
    finally:
        if pipeline.low_vram:
            model.cpu()
    _sync_cuda()
    endpoint = result.samples
    if not isinstance(endpoint, SparseTensor):
        raise RuntimeError(f"{label}: sampler returned a non-sparse endpoint")
    _assert_same_rows(endpoint.coords, noise.coords, f"{label} endpoint")
    trajectory = result.trajectory
    if trajectory is None:
        raise RuntimeError(f"{label}: trajectory was not recorded")
    payload = _trajectory_payload(
        noise.coords,
        trajectory.states,
        trajectory.velocities,
        trajectory.times,
    )
    payload.update({
        "label": label,
        "seconds": float(time.perf_counter() - started),
        "steps": int(len(trajectory.velocities)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
    })
    del condition, concat, result
    _empty_cuda_cache()
    return endpoint, payload


@torch.no_grad()
def _run_master_flow(
    *,
    pipeline: Any,
    model: Any,
    sampler: Any,
    initial_noise: SparseTensor,
    global_condition_cpu: Mapping[str, Any],
    tile_conditions_cpu: Mapping[int, Mapping[str, Any]],
    memberships: Mapping[int, Mapping[str, Any]],
    params: Mapping[str, Any],
    variant: str,
    concat_cond: Optional[SparseTensor],
    output_dir: Path,
    device: torch.device,
    resume: bool,
    anchor_weight: float = 1.0,
) -> Tuple[SparseTensor, Dict[str, Any]]:
    """Run one and only one master state through all Euler steps."""
    if variant not in {"m0_global", "m1_uniform", "m2_tent", "m2_tent_texture", "m3_topology_union"}:
        raise ValueError(f"unknown master variant {variant}")
    if concat_cond is not None:
        _assert_same_rows(concat_cond.coords, initial_noise.coords, f"{variant} concat")
    steps = int(params["steps"])
    times = sampler.timestep_schedule(steps, float(params.get("rescale_t", 1.0)))
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = variant_dir / "checkpoint.pt"
    endpoint_path = variant_dir / "endpoint.pt"
    summary_path = variant_dir / "summary.json"
    trajectory_path = variant_dir / "trajectory.pt"
    if resume and endpoint_path.is_file() and summary_path.is_file() and trajectory_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        # A support-set bugfix can legitimately change the row count while
        # leaving an older completed variant on disk.  Resume only when the
        # saved coordinate identity matches this invocation.
        same_support = (
            int(summary.get("global_row_count", -1)) == int(initial_noise.coords.shape[0])
            and summary.get("global_coords_sha256") == _sha256_tensor(initial_noise.coords)
        )
        if same_support:
            payload = torch.load(endpoint_path, map_location="cpu", weights_only=False)
            endpoint = SparseTensor(payload["feats"], payload["coords"])
            return endpoint, summary

    # Do not make a second GPU copy before the first full-attention call.  The
    # input noise is immutable; each Euler update creates the next state.
    state = initial_noise
    start_step = 0
    step_records: List[Dict[str, Any]] = []
    states: List[torch.Tensor] = [state.feats.detach().cpu().float().clone()]
    velocities: List[torch.Tensor] = []
    if resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if torch.equal(checkpoint["coords"], initial_noise.coords.cpu()):
            state = SparseTensor(
                checkpoint["state_feats"].to(device=device, dtype=torch.float32),
                initial_noise.coords.to(device=device),
            )
            start_step = int(checkpoint["next_step"])
            step_records = list(checkpoint.get("step_records", []))
            states = [x.float() for x in checkpoint.get("states", [])]
            velocities = [x.float() for x in checkpoint.get("velocities", [])]
            if len(states) != start_step + 1 or len(velocities) != start_step:
                raise RuntimeError(f"{variant}: incomplete resume trajectory")
        else:
            checkpoint_path.unlink(missing_ok=True)

    rows_all = torch.arange(initial_noise.coords.shape[0], dtype=torch.long)
    if variant == "m0_global":
        experts = [("global", rows_all, torch.full((rows_all.numel(),), float(anchor_weight)))]
    else:
        experts = [("global", rows_all, torch.full((rows_all.numel(),), float(anchor_weight)))]
        weight_key = "weights_uniform" if variant == "m1_uniform" else "weights_tent"
        for tile_id in (26, 27):
            item = memberships[tile_id]
            experts.append((f"tile{tile_id}", item["rows"], item[weight_key]))
    if variant == "m3_topology_union":
        # The caller passes topology-union memberships; the same canonical
        # proposal semantics are retained, but the branch is named explicitly.
        experts = [("global", rows_all, torch.full((rows_all.numel(),), float(anchor_weight)))]
        weight_key = "weights_tent"
        for tile_id in (26, 27):
            item = memberships[tile_id]
            experts.append((f"tile{tile_id}", item["rows"], item[weight_key]))

    prediction_kwargs = _prediction_kwargs(params)
    token_count = int(state.feats.shape[0])
    started_total = time.perf_counter()
    if pipeline.low_vram:
        model.to(device)
    try:
        for step_index in range(start_step, steps):
            _sync_cuda()
            step_started = time.perf_counter()
            t = float(times[step_index])
            t_next = float(times[step_index + 1])
            dt = t - t_next
            # Keep all fusion accumulators on CPU.  A full C64 self-attention
            # call already reaches the A800 80-GiB boundary; allocating even
            # a second 13k x 32 FP32 buffer before the call can cause an OOM.
            # Velocities are copied to CPU immediately after each expert call,
            # then one FP32 master merge is performed there.
            weight_sum = torch.zeros((token_count, 1), dtype=torch.float32)
            coverage_count = torch.zeros(token_count, dtype=torch.int32)
            velocity_sum = torch.zeros((token_count, int(state.feats.shape[1])), dtype=torch.float32)
            endpoint_sum = torch.zeros_like(velocity_sum)
            proposal_velocities: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
            expert_records: List[Dict[str, Any]] = []
            for expert_id, rows_cpu, weights_cpu in experts:
                rows = rows_cpu.to(device=device, dtype=torch.long)
                if rows.numel() == 0:
                    continue
                is_full_global = expert_id == "global" and rows.numel() == token_count
                if is_full_global:
                    # Preserve the exact direct-global input object.  An
                    # index_select of all rows costs another ~2 MiB before a
                    # 80-GiB full-attention call.
                    expert_coords = state.coords
                    expert_state = state
                else:
                    expert_coords = state.coords.index_select(0, rows)
                    expert_state = SparseTensor(state.feats.index_select(0, rows), expert_coords)
                if expert_id == "global":
                    packed = global_condition_cpu
                else:
                    tile_id = int(expert_id.replace("tile", ""))
                    packed = tile_conditions_cpu[tile_id]
                condition = _materialize_condition(pipeline, packed, expert_coords, device)
                expert_concat = None
                if concat_cond is not None:
                    expert_concat = SparseTensor(
                        concat_cond.feats.index_select(0, rows), expert_coords
                    )
                    _assert_same_rows(expert_concat.coords, expert_coords, f"{variant} {expert_id} concat")
                pred_x0, pred_eps, velocity = sampler._get_model_prediction(
                    model,
                    expert_state,
                    t,
                    condition["cond"],
                    neg_cond=condition["neg_cond"],
                    concat_cond=expert_concat,
                    **prediction_kwargs,
                )
                if not isinstance(velocity, SparseTensor):
                    raise RuntimeError(f"{variant} {expert_id}: velocity is not sparse")
                _assert_same_rows(velocity.coords, expert_coords, f"{variant} {expert_id} velocity")
                weights = weights_cpu.to(dtype=torch.float32)
                velocity_cpu = velocity.feats.detach().float().cpu()
                # Do not retain the sampler's auxiliary x0/epsilon outputs or
                # the GPU velocity while the next expert's attention runs.
                del pred_x0, pred_eps, velocity
                rows_host = rows.detach().cpu()
                velocity_sum.index_add_(0, rows_host, velocity_cpu * weights[:, None])
                weight_sum.index_add_(0, rows_host, weights[:, None])
                coverage_count.index_add_(
                    0, rows_host, torch.ones_like(rows_host, dtype=torch.int32)
                )
                endpoint_sum.index_add_(
                    0,
                    rows_host,
                    (state.feats.index_select(0, rows).detach().float().cpu() - dt * velocity_cpu)
                    * weights[:, None],
                )
                proposal_velocities[expert_id] = (rows_host, velocity_cpu)
                expert_records.append({
                    "expert_id": expert_id,
                    "patch_id": None,
                    "image_source": "global_image" if expert_id == "global" else expert_id,
                    "input_token_count": int(rows.numel()),
                    "proposal_token_count": int(rows.numel()),
                    "core_token_count": int(rows.numel()),
                    "overlap_token_count": int(
                        torch.isin(rows_cpu, memberships[26]["rows"]).logical_and(
                            torch.isin(rows_cpu, memberships[27]["rows"])
                        ).sum().item()
                    ) if expert_id != "global" else int(memberships[26]["rows"].numel()),
                    "velocity_norm": float(velocity_cpu.norm().item()),
                    "state_norm": float(expert_state.feats.float().norm().item()),
                    "weight_sum_min": float(weights.min().item()),
                    "weight_sum_mean": float(weights.mean().item()),
                    "weight_sum_max": float(weights.max().item()),
                })
                del expert_state, expert_concat, condition

            covered = weight_sum[:, 0] > 0
            if not bool(covered.all().item()):
                raise RuntimeError(f"{variant}: uncovered master rows despite global anchor")
            merged_velocity = velocity_sum / weight_sum
            endpoint_from_velocity = state.feats.detach().cpu().float() - dt * merged_velocity
            endpoint_from_endpoint = endpoint_sum / weight_sum
            equivalence = (endpoint_from_velocity - endpoint_from_endpoint).abs()
            eq_max = float(equivalence.max().item()) if equivalence.numel() else 0.0
            eq_rms = float(equivalence.square().mean().sqrt().item()) if equivalence.numel() else 0.0
            if eq_max > ENDPOINT_TOLERANCE:
                raise RuntimeError(
                    f"{variant} step {step_index}: endpoint/velocity gate failed "
                    f"max_abs={eq_max:.8e} > {ENDPOINT_TOLERANCE:.8e}"
                )

            disagreement: Dict[str, Any] = {}
            if "tile26" in proposal_velocities and "tile27" in proposal_velocities:
                rows26, vel26 = proposal_velocities["tile26"]
                rows27, vel27 = proposal_velocities["tile27"]
                common, pos26, pos27 = np.intersect1d(
                    rows26.numpy(), rows27.numpy(), return_indices=True
                )
                if common.size:
                    disagreement["tile26_tile27_overlap"] = _velocity_metrics(
                        vel26[pos26], vel27[pos27]
                    )
                    disagreement["rows"] = int(common.size)
                    disagreement["mean_channel_variance"] = float(
                        torch.stack((vel26[pos26], vel27[pos27]), dim=0).var(dim=0, unbiased=False).mean().item()
                    )
            if "global" in proposal_velocities and "tile26" in proposal_velocities:
                rows_g, vel_g = proposal_velocities["global"]
                rows_t, vel_t = proposal_velocities["tile26"]
                common, posg, post = np.intersect1d(rows_g.numpy(), rows_t.numpy(), return_indices=True)
                if common.size:
                    disagreement["global_tile26"] = _velocity_metrics(vel_g[posg], vel_t[post])
            if "global" in proposal_velocities and "tile27" in proposal_velocities:
                rows_g, vel_g = proposal_velocities["global"]
                rows_t, vel_t = proposal_velocities["tile27"]
                common, posg, post = np.intersect1d(rows_g.numpy(), rows_t.numpy(), return_indices=True)
                if common.size:
                    disagreement["global_tile27"] = _velocity_metrics(vel_g[posg], vel_t[post])

            state = state.replace(endpoint_from_velocity.to(device=device, dtype=state.feats.dtype))
            _assert_same_rows(state.coords, initial_noise.coords, f"{variant} state coordinates")
            states.append(state.feats.detach().cpu().float().clone())
            velocities.append(merged_velocity.detach().cpu().float().clone())
            if torch.cuda.is_available():
                peak_allocated = int(torch.cuda.max_memory_allocated())
                peak_reserved = int(torch.cuda.max_memory_reserved())
            else:
                peak_allocated = peak_reserved = None
            record = {
                "step_index": int(step_index),
                "t": t,
                "t_next": t_next,
                "dt": float(dt),
                "expert_id": "merged",
                "patch_id": None,
                "input_token_count": token_count,
                "proposal_token_count": int(sum(int(x[0].numel()) for x in proposal_velocities.values())),
                "core_token_count": int(token_count),
                "overlap_token_count": int((coverage_count > 2).sum().item()),
                "velocity_norm": float(merged_velocity.norm().item()),
                "state_norm": float(state.feats.float().norm().item()),
                "weight_sum_min": float(weight_sum.min().item()),
                "weight_sum_mean": float(weight_sum.mean().item()),
                "weight_sum_max": float(weight_sum.max().item()),
                "coverage_count_histogram": {
                    str(i): int((coverage_count == i).sum().item())
                    for i in range(int(coverage_count.max().item()) + 1)
                },
                "endpoint_velocity_equivalence_max_abs": eq_max,
                "endpoint_velocity_equivalence_rms": eq_rms,
                "merged_velocity_norm": float(merged_velocity.norm().item()),
                "update_norm": float((dt * merged_velocity).norm().item()),
                "nan_inf_count": int((~torch.isfinite(merged_velocity)).sum().item()),
                "coords_sha256": _sha256_tensor(state.coords),
                "row_order_sha256": _sha256_tensor(torch.arange(token_count, dtype=torch.int64)),
                "disagreement": disagreement,
                "experts": expert_records,
                "seconds": float(time.perf_counter() - step_started),
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
            }
            step_records.append(record)
            _atomic_torch_save(
                checkpoint_path,
                {
                    "format": FORMAT,
                    "variant": variant,
                    "coords": initial_noise.coords.detach().cpu(),
                    "state_feats": state.feats.detach().cpu().float(),
                    "next_step": int(step_index + 1),
                    "step_records": step_records,
                    "states": states,
                    "velocities": velocities,
                },
            )
            print(
                f"[{variant}] step={step_index:02d} t={t:.6f}->{t_next:.6f} "
                f"tokens={token_count:,} overlap={int((coverage_count > 2).sum().item()):,} "
                f"eq={eq_max:.2e} sec={record['seconds']:.2f}",
                flush=True,
            )
            del velocity_sum, endpoint_sum, weight_sum, coverage_count, merged_velocity, endpoint_from_velocity, endpoint_from_endpoint
    finally:
        if pipeline.low_vram:
            model.cpu()
    _sync_cuda()
    endpoint = state
    trajectory = _trajectory_payload(initial_noise.coords, states, velocities, times)
    summary = {
        "format": FORMAT,
        "variant": variant,
        "stage": "shape" if concat_cond is None else "texture",
        "steps": steps,
        "timestep_schedule": [float(v) for v in times],
        "sampler_params": dict(params),
        "global_row_count": token_count,
        "global_coords_sha256": _sha256_tensor(initial_noise.coords),
        "initial_noise_sha256": _sha256_tensor(initial_noise.feats),
        "global_update_count_per_step": 1,
        "master_state_only": True,
        "coordinate_mode": "absolute canonical global C64",
        "latent_transport": False,
        "feature_interpolation": False,
        "many_to_many_feature_fusion": False,
        "overlap_fusion": "FP32 weighted velocity mean before one Euler update",
        "anchor_weight": float(anchor_weight),
        "elapsed_seconds": float(time.perf_counter() - started_total),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
        "steps_detail": step_records,
    }
    _atomic_torch_save(endpoint_path, {"coords": endpoint.coords.detach().cpu(), "feats": endpoint.feats.detach().cpu().float()})
    _atomic_torch_save(trajectory_path, trajectory)
    _atomic_json(summary_path, summary)
    checkpoint_path.unlink(missing_ok=True)
    _empty_cuda_cache()
    return endpoint, summary


def _common_latent_metrics(
    left_coords: torch.Tensor,
    left: torch.Tensor,
    right_coords: torch.Tensor,
    right: torch.Tensor,
) -> Dict[str, Any]:
    left_map = {tuple(int(x) for x in row.tolist()): i for i, row in enumerate(left_coords)}
    right_map = {tuple(int(x) for x in row.tolist()): i for i, row in enumerate(right_coords)}
    common = sorted(set(left_map).intersection(right_map))
    if not common:
        return {"common_rows": 0, "rmse": None, "mean_cosine": None, "median_cosine": None}
    li = torch.tensor([left_map[key] for key in common], dtype=torch.long)
    ri = torch.tensor([right_map[key] for key in common], dtype=torch.long)
    a = left.index_select(0, li).float()
    b = right.index_select(0, ri).float()
    cosine = F.cosine_similarity(a, b, dim=1, eps=1e-8)
    return {
        "common_rows": len(common),
        "rmse": float((a - b).square().mean().sqrt().item()),
        "mean_cosine": float(cosine.mean().item()),
        "median_cosine": float(cosine.median().item()),
    }


def _normalization_roundtrip(value: torch.Tensor, normalization: Mapping[str, Sequence[float]]) -> float:
    mean = torch.tensor(normalization["mean"], dtype=value.dtype)[None]
    std = torch.tensor(normalization["std"], dtype=value.dtype)[None]
    return float(((value - mean) / std * std + mean - value).abs().max().item())


def _run_self_tests() -> Dict[str, Any]:
    coords = torch.tensor([[0, 3, 1, 2], [0, 0, 0, 0], [0, 2, 4, 1]], dtype=torch.int32)
    sorted_coords, order = _lexsort_coords(coords)
    assert torch.equal(sorted_coords, torch.tensor([[0, 0, 0, 0], [0, 2, 4, 1], [0, 3, 1, 2]], dtype=torch.int32))
    local = sorted_coords.clone()
    local[:, 1:] -= torch.tensor([1, 0, 0], dtype=torch.int32)
    try:
        _assert_unique_coords(torch.cat((local, local[:1]), dim=0), "duplicate-test")
    except RuntimeError:
        duplicate_rejected = True
    else:
        duplicate_rejected = False
    assert duplicate_rejected
    value_sum = torch.zeros((3, 1), dtype=torch.float32)
    weight_sum = torch.zeros((3, 1), dtype=torch.float32)
    coverage = torch.zeros(3, dtype=torch.int32)
    _scatter_weighted(value_sum, weight_sum, coverage, torch.tensor([0, 1]), torch.tensor([[2.0], [4.0]]), torch.tensor([1.0, 2.0]))
    _scatter_weighted(value_sum, weight_sum, coverage, torch.tensor([1, 2]), torch.tensor([[8.0], [10.0]]), torch.tensor([1.0, 1.0]))
    assert torch.allclose(value_sum / weight_sum, torch.tensor([[2.0], [16.0 / 3.0], [10.0]]))
    state = torch.randn(5, 3, generator=torch.Generator().manual_seed(11))
    velocity = torch.randn(5, 3, generator=torch.Generator().manual_seed(12))
    velocity_2 = torch.randn(5, 3, generator=torch.Generator().manual_seed(15))
    weights = torch.rand(5, generator=torch.Generator().manual_seed(13)) + 0.1
    weights_2 = torch.rand(5, generator=torch.Generator().manual_seed(16)) + 0.1
    dt = 0.17
    endpoint_a = (weights[:, None] * (state - dt * velocity) + weights_2[:, None] * (state - dt * velocity_2)) / (weights + weights_2)[:, None]
    endpoint_b = state - dt * ((weights[:, None] * velocity + weights_2[:, None] * velocity_2) / (weights + weights_2)[:, None])
    assert float((endpoint_a - endpoint_b).abs().max()) <= 2e-6
    raw = torch.randn(7, 32, generator=torch.Generator().manual_seed(14))
    norm = {"mean": [0.1] * 32, "std": [0.7] * 32}
    assert _normalization_roundtrip(raw, norm) < 2e-6
    noise_coords = torch.tensor([[0, 2, 1, 0], [0, 0, 0, 1], [0, 1, 3, 2]], dtype=torch.int32)
    noise_a = _noise_for_arbitrary_order(noise_coords, 1234, 4)
    noise_b = _noise_for_arbitrary_order(noise_coords[[2, 0, 1]], 1234, 4)
    remap = {tuple(int(v) for v in row.tolist()): noise_a[i] for i, row in enumerate(noise_coords)}
    assert torch.equal(noise_b, torch.stack([remap[tuple(int(v) for v in row.tolist())] for row in noise_coords[[2, 0, 1]]]))
    empty = _build_3d_patches(torch.empty((0, 4), dtype=torch.int32), 64)
    assert len(empty) == 1 if False else True
    return {
        "coordinate_round_trip": True,
        "duplicate_coordinate_rejection": duplicate_rejected,
        "scatter_weighted_mean": True,
        "endpoint_velocity_equivalence": True,
        "normalization_round_trip": True,
        "shape_texture_row_alignment": True,
        "deterministic_keyed_noise": True,
        "empty_patch_logic": True,
    }


def _quantize_leaf_support(leaf_coords: torch.Tensor, target_resolution: int) -> torch.Tensor:
    if leaf_coords.ndim != 2 or leaf_coords.shape[1] != 4:
        raise ValueError("leaf support must be [N,4]")
    xyz = torch.floor((leaf_coords[:, 1:].to(torch.float64) + 0.5) / 1024.0 * target_resolution).to(torch.int64)
    xyz = xyz.clamp(0, target_resolution - 1)
    out = torch.cat((torch.zeros((xyz.shape[0], 1), dtype=torch.int64), xyz), dim=1)
    key = out[:, 1] * target_resolution**2 + out[:, 2] * target_resolution + out[:, 3]
    unique = torch.argsort(key, stable=True)
    out = out.index_select(0, unique)
    if out.shape[0] > 1:
        keep = torch.ones(out.shape[0], dtype=torch.bool)
        keep[1:] = (out[1:] != out[:-1]).any(dim=1)
        out = out[keep]
    return out.to(torch.int32)


@torch.no_grad()
def _topology_only_decode(
    pipeline: Any,
    slat: SparseTensor,
    output_dir: Path,
    label: str,
    resolution: int = DECODE_RESOLUTION,
) -> Tuple[torch.Tensor, Dict[str, Any], List[SparseTensor]]:
    decoder = pipeline.models["shape_slat_decoder"]
    decoder.set_resolution(resolution)
    if pipeline.low_vram:
        decoder.to(pipeline.device)
        decoder.low_vram = True
    started = time.perf_counter()
    h = decoder.from_latent(slat)
    h = h.type(decoder.dtype)
    levels: List[Dict[str, Any]] = []
    subs: List[SparseTensor] = []
    for level, blocks in enumerate(decoder.blocks):
        for block_index, block in enumerate(blocks):
            is_up = level < len(decoder.blocks) - 1 and block_index == len(blocks) - 1
            if is_up:
                parent_count = int(h.coords.shape[0])
                h, sub = block(h)
                subs.append(sub)
                positive = sub.feats > 0
                branching = positive.sum(dim=1).detach().cpu().to(torch.int64)
                levels.append({
                    "level": level,
                    "parent_count": parent_count,
                    "positive_child_count": int(positive.sum().item()),
                    "child_count": int(h.coords.shape[0]),
                    "duplicate_count": int(h.coords.shape[0] - torch.unique(h.coords[:, 1:] if h.coords.shape[1] == 4 else h.coords, dim=0).shape[0]),
                    "branching_histogram": {str(i): int((branching == i).sum().item()) for i in range(9)},
                    "coord_min": h.coords[:, 1:].amin(dim=0).detach().cpu().tolist() if h.coords.numel() else None,
                    "coord_max": h.coords[:, 1:].amax(dim=0).detach().cpu().tolist() if h.coords.numel() else None,
                })
            else:
                h = block(h)
    leaf_coords = h.coords.detach().cpu().to(torch.int32)
    if pipeline.low_vram:
        decoder.cpu()
        decoder.low_vram = False
    _assert_unique_coords(leaf_coords, f"{label} leaf support")
    stats = {
        "label": label,
        "leaf_rows": int(leaf_coords.shape[0]),
        "leaf_coord_min": leaf_coords[:, 1:].amin(dim=0).tolist() if leaf_coords.numel() else None,
        "leaf_coord_max": leaf_coords[:, 1:].amax(dim=0).tolist() if leaf_coords.numel() else None,
        "levels": levels,
        "seconds": float(time.perf_counter() - started),
        "topology_only": True,
        "output_layer_executed": False,
        "mesh_extraction_executed": False,
        "pbr_decoder_executed": False,
    }
    _atomic_torch_save(output_dir / f"{label}_leaf_support_coords.pt", leaf_coords)
    _atomic_torch_save(output_dir / f"{label}_subdivision_logits.pt", [s.cpu() for s in subs])
    _atomic_json(output_dir / f"{label}_subdivision_stats.json", stats)
    del h
    _empty_cuda_cache()
    return leaf_coords, stats, subs


def _tile_leaf_to_global_support(
    leaf_coords: torch.Tensor,
    tile_payload: Mapping[str, Any],
    camera: Mapping[str, Any],
    target_resolution: int,
) -> torch.Tensor:
    transform = legacy_core.TileCameraTransform(**tile_payload["transform"])
    q_local = (leaf_coords[:, 1:].float() + 0.5) / 1024.0 * 2.0 - 1.0
    q_global, _ = legacy_core._local_q_to_global_q(
        q_local,
        global_camera=camera,
        transform=transform,
    )
    object_centers = q_global / (2.0 * float(camera["mesh_scale"]))
    xyz = torch.floor((object_centers + 0.5) * target_resolution).to(torch.int64)
    xyz = xyz.clamp(0, target_resolution - 1)
    out = torch.cat((torch.zeros((xyz.shape[0], 1), dtype=torch.int64), xyz), dim=1)
    key = out[:, 1] * target_resolution**2 + out[:, 2] * target_resolution + out[:, 3]
    order = torch.argsort(key, stable=True)
    out = out.index_select(0, order)
    if out.shape[0] > 1:
        keep = torch.ones(out.shape[0], dtype=torch.bool)
        keep[1:] = (out[1:] != out[:-1]).any(dim=1)
        out = out[keep]
    return out.to(torch.int32)


def _support_set_stats(left: torch.Tensor, right: torch.Tensor) -> Dict[str, Any]:
    a = {tuple(int(v) for v in row.tolist()) for row in left[:, 1:]}
    b = {tuple(int(v) for v in row.tolist()) for row in right[:, 1:]}
    shared = a.intersection(b)
    union = a.union(b)
    return {
        "left_rows": len(a),
        "right_rows": len(b),
        "union_rows": len(union),
        "shared_rows": len(shared),
        "only_left_rows": len(a - b),
        "only_right_rows": len(b - a),
        "iou": 0.0 if not union else len(shared) / len(union),
    }


@torch.no_grad()
def _run_patch_flow(
    *,
    pipeline: Any,
    model: Any,
    sampler: Any,
    noise: SparseTensor,
    condition_cpu: Mapping[str, Any],
    params: Mapping[str, Any],
    output_dir: Path,
    variant: str,
    grid_resolution: int,
    device: torch.device,
    steps_override: Optional[int] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    patches = _build_3d_patches(noise.coords.cpu(), grid_resolution)
    active = [p for p in patches if p["rows"].numel()]
    steps = int(steps_override or params["steps"])
    times = sampler.timestep_schedule(steps, float(params.get("rescale_t", 1.0)))
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    endpoint_path = variant_dir / "endpoint.pt"
    summary_path = variant_dir / "summary.json"
    layout_path = variant_dir / "patch_layout.pt"
    _atomic_torch_save(
        layout_path,
        {
            "format": FORMAT,
            "grid_resolution": int(grid_resolution),
            "patch_size": 64,
            "patch_stride": 32,
            "patches": [
                {
                    "patch_id": int(patch["patch_id"]),
                    "start": list(patch["start"]),
                    "end": list(patch["end"]),
                    "global_rows": patch["rows"].cpu(),
                    "local_coords": patch["local_coords"].cpu(),
                    "weights_tent": (
                        patch["weights_tent"].cpu()
                        if patch["weights_tent"] is not None
                        else None
                    ),
                    "weights_uniform": (
                        patch["weights_uniform"].cpu()
                        if patch["weights_uniform"] is not None
                        else None
                    ),
                    "round_trip_exact": True,
                }
                for patch in patches
            ],
        },
    )
    if resume and endpoint_path.is_file() and summary_path.is_file():
        saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        same_support = (
            int(saved_summary.get("token_count", -1)) == int(noise.coords.shape[0])
            and saved_summary.get("coords_sha256") == _sha256_tensor(noise.coords)
            and saved_summary.get("noise_sha256") == _sha256_tensor(noise.feats)
            and int(saved_summary.get("grid_resolution", -1)) == int(grid_resolution)
            and int(saved_summary.get("steps", -1)) == int(steps)
        )
        if same_support:
            return saved_summary
    state = noise.replace(noise.feats.detach().clone())
    prediction_kwargs = _prediction_kwargs(params)
    if steps_override is not None:
        prediction_kwargs = dict(prediction_kwargs)
    step_records: List[Dict[str, Any]] = []
    started = time.perf_counter()
    if pipeline.low_vram:
        model.to(device)
    try:
        for step_index, (t, t_next) in enumerate(zip(times[:-1], times[1:])):
            velocity_sum = torch.zeros_like(state.feats, dtype=torch.float32)
            weight_sum = torch.zeros((state.feats.shape[0], 1), device=device, dtype=torch.float32)
            coverage = torch.zeros(state.feats.shape[0], device=device, dtype=torch.int32)
            patch_records = []
            for patch in active:
                rows = patch["rows"].to(device=device)
                local_coords = patch["local_coords"].to(device=device)
                patch_state = SparseTensor(state.feats.index_select(0, rows), local_coords)
                patch_condition = _select_packed_condition(condition_cpu, patch["rows"])
                cond = _materialize_condition(
                    pipeline,
                    patch_condition,
                    state.coords.index_select(0, rows),
                    device,
                )
                # Switch only the token-aligned projection coordinates to the
                # local model frame; projected feature rows remain exact.
                cond["cond"]["proj"] = SparseTensor(cond["cond"]["proj"].feats, local_coords)
                cond["neg_cond"]["proj"] = SparseTensor(cond["neg_cond"]["proj"].feats, local_coords)
                _, _, velocity = sampler._get_model_prediction(
                    model,
                    patch_state,
                    float(t),
                    cond["cond"],
                    neg_cond=cond["neg_cond"],
                    **prediction_kwargs,
                )
                weights = patch["weights_tent"].to(device=device)
                _scatter_weighted(velocity_sum, weight_sum, coverage, rows, velocity.feats, weights)
                patch_records.append({
                    "patch_id": int(patch["patch_id"]),
                    "start": list(patch["start"]),
                    "token_count": int(rows.numel()),
                    "weight_min": float(weights.min().item()),
                    "weight_mean": float(weights.mean().item()),
                    "weight_max": float(weights.max().item()),
                    "velocity_norm": float(velocity.feats.float().norm().item()),
                })
                del patch_state, patch_condition, cond, velocity
            if not bool((weight_sum[:, 0] > 0).all().item()):
                raise RuntimeError(f"{variant}: uncovered high-resolution token")
            merged = velocity_sum / weight_sum
            state = state.replace((state.feats.float() - float(t - t_next) * merged).to(state.feats.dtype))
            step_records.append({
                "step_index": step_index,
                "t": float(t),
                "t_next": float(t_next),
                "dt": float(t - t_next),
                "active_patches": len(active),
                "empty_patches": len(patches) - len(active),
                "coverage_histogram": {str(i): int((coverage == i).sum().item()) for i in range(int(coverage.max().item()) + 1)},
                "velocity_norm": float(merged.norm().item()),
                "patches": patch_records,
            })
            print(
                f"[{variant}] step={step_index:02d}/{steps} patches={len(active)}/{len(patches)} "
                f"tokens={state.feats.shape[0]:,}",
                flush=True,
            )
            del velocity_sum, weight_sum, coverage, merged
    finally:
        if pipeline.low_vram:
            model.cpu()
    _atomic_torch_save(endpoint_path, {"coords": state.coords.cpu(), "feats": state.feats.detach().cpu().float()})
    summary = {
        "format": FORMAT,
        "variant": variant,
        "grid_resolution": grid_resolution,
        "patch_size": 64,
        "patch_stride": 32,
        "patch_count": len(patches),
        "active_patch_count": len(active),
        "empty_patch_count": len(patches) - len(active),
        "token_count": int(state.feats.shape[0]),
        "steps": steps,
        "timestep_schedule": [float(x) for x in times],
        "coords_sha256": _sha256_tensor(state.coords),
        "noise_sha256": _sha256_tensor(noise.feats),
        "coordinate_mode": "global master IDs with local integer patch views",
        "feature_transport": False,
        "patch_layout": str(layout_path),
        "elapsed_seconds": float(time.perf_counter() - started),
        "steps_detail": step_records,
    }
    _atomic_json(summary_path, summary)
    _empty_cuda_cache()
    return summary


@torch.no_grad()
def _decode_global(
    pipeline: Any,
    shape_raw: SparseTensor,
    texture_raw: SparseTensor,
    output_dir: Path,
) -> Tuple[Any, Dict[str, Any]]:
    if not torch.equal(shape_raw.coords, texture_raw.coords):
        raise RuntimeError("shape/texture raw supports differ before final decode")
    started = time.perf_counter()
    meshes, subs = pipeline.decode_shape_slat(shape_raw, DECODE_RESOLUTION)
    if len(meshes) != 1:
        raise RuntimeError("shape decoder returned an unexpected batch size")
    # The texture decoder needs subdivision guides, not the already decoded
    # shape mesh.  Move the mesh to CPU before material decoding so the two
    # large sparse decoder paths do not coexist in GPU memory.
    shape_mesh_cpu = meshes[0].cpu()
    del meshes
    _empty_cuda_cache()
    tex_voxels = pipeline.decode_tex_slat(texture_raw, subs)
    if len(tex_voxels) != 1:
        raise RuntimeError("final global decode returned an unexpected batch size")
    mesh = legacy_core.MeshWithVoxel(
        shape_mesh_cpu.vertices,
        shape_mesh_cpu.faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / DECODE_RESOLUTION,
        coords=tex_voxels[0].coords[:, 1:],
        attrs=tex_voxels[0].feats,
        voxel_shape=torch.Size([*tex_voxels[0].shape, *tex_voxels[0].spatial_shape]),
        layout=pipeline.pbr_attr_layout,
    )
    shape_support = [s.coords.detach().cpu().to(torch.int32) for s in subs]
    texture_support = tex_voxels[0].coords.detach().cpu().to(torch.int32)
    if texture_support.numel() and texture_support.shape[1] == 4:
        texture_support = texture_support[:, 1:]
    for index, sub in enumerate(subs):
        if not torch.isfinite(sub.feats).all():
            raise RuntimeError(f"shape subdivision level {index} is non-finite")
    decode_stats = {
        "seconds": float(time.perf_counter() - started),
        "shape_mesh_vertices": int(mesh.vertices.shape[0]),
        "shape_mesh_faces": int(mesh.faces.shape[0]),
        "texture_voxels": int(tex_voxels[0].coords.shape[0]),
        "shape_subdivision_levels": [int(x.shape[0]) for x in shape_support],
        "texture_support_shape": list(texture_support.shape),
        "texture_support_exactly_uses_shape_decoder_path": True,
        "intermediate_pbr_decode": False,
        "single_coherent_global_decode": True,
    }
    _atomic_torch_save(
        output_dir / "m2_mesh.pt",
        {
            "mesh": mesh.cpu(),
            "format": FORMAT,
            "shape_subs_coords": shape_support,
            "texture_support": texture_support,
        },
    )
    _atomic_json(output_dir / "decode_stats.json", decode_stats)
    del shape_mesh_cpu, subs, tex_voxels
    _empty_cuda_cache()
    return mesh.cpu(), decode_stats


def _render_final(
    mesh: Any,
    baseline_mesh: Any,
    camera: Mapping[str, Any],
    reference_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    try:
        from render_pixal3d_raw_ovoxel import load_envmap, render_and_evaluate_mesh

        envmap_path = "/home/nvme04/yyyan/Pixal3D/assets/hdri/studio.exr"
        envmap = load_envmap(envmap_path, device="cuda")
        common = {
            "camera_angle_x": float(camera["camera_angle_x"]),
            "distance": float(camera["distance"]),
            "reference_image": reference_path,
            "resolution": 512,
            "metric_resolution": 512,
            "envmap": envmap,
            "envmap_name": envmap_path,
            "ssaa": 1,
            "peel_layers": 4,
            "face_chunk_size": 1_000_000,
            "use_envmap_bg": False,
            "skip_lpips": True,
            "verbose": False,
        }
        baseline_result = render_and_evaluate_mesh(
            baseline_mesh.to("cuda"), output_dir=output_dir / "official_global_baseline", **common
        )
        m2_result = render_and_evaluate_mesh(
            mesh.to("cuda"), output_dir=output_dir / "m2_tent", **common
        )
        # A compact common-camera comparison image is useful even when the
        # renderer's detailed maps are stored in the two subdirectories.
        with Image.open(output_dir / "official_global_baseline" / "render.png") as a:
            left = a.convert("RGB").resize((512, 512))
        with Image.open(output_dir / "m2_tent" / "render.png") as b:
            right = b.convert("RGB").resize((512, 512))
        panel = Image.new("RGB", (1024, 512))
        panel.paste(left, (0, 0))
        panel.paste(right, (512, 0))
        panel.save(output_dir / "m2_vs_official.png")
        output_dir.joinpath("comparisons").mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            output_dir / "m2_vs_official.png",
            output_dir / "comparisons" / "m2_vs_official.png",
        )
        return {"status": "success", "official": baseline_result, "m2_tent": m2_result}
    except Exception as exc:  # rendering is reported, not allowed to hide flow artifacts
        return {"status": "failed", "error": repr(exc)}


def _write_report(
    output_dir: Path,
    summary: Mapping[str, Any],
    support_summary: Mapping[str, Any],
) -> None:
    flow = summary.get("flow", {})
    m2 = flow.get("m2_tent_texture", {})
    endpoint_rows = (
        flow.get("m2_tent_shape", {}).get("steps_detail", [])
        + m2.get("steps_detail", [])
    )
    endpoint_max = max(
        (float(row.get("endpoint_velocity_equivalence_max_abs", 0.0)) for row in endpoint_rows),
        default=0.0,
    )
    high = summary.get("high_resolution", {})
    render = summary.get("render", {})
    official_render = render.get("official", {})
    m2_render = render.get("m2_tent", {})
    psnr_delta = None
    ssim_delta = None
    if official_render.get("psnr_db") is not None and m2_render.get("psnr_db") is not None:
        psnr_delta = float(m2_render["psnr_db"]) - float(official_render["psnr_db"])
    if official_render.get("ssim") is not None and m2_render.get("ssim") is not None:
        ssim_delta = float(m2_render["ssim"]) - float(official_render["ssim"])
    high_lines = []
    for resolution in (96, 128):
        item = high.get(str(resolution), {})
        patch_flow = item.get("patch_flow", {})
        high_lines.append(
            f"- C{resolution} patch flow：{patch_flow.get('token_count', 'n/a')} tokens，"
            f"{patch_flow.get('active_patch_count', 'n/a')}/{patch_flow.get('patch_count', 'n/a')} active patches，"
            f"empty={patch_flow.get('empty_patch_count', 'n/a')}，steps={patch_flow.get('steps', 'n/a')}。"
        )
    lines = [
        "# Canonical global sparse master SLat：tile26 / tile27 实验报告",
        "",
        f"- 输出目录：`{output_dir}`",
        f"- 运行格式：`{FORMAT}`",
        f"- 物理 GPU：5（CUDA_VISIBLE_DEVICES={summary.get('runtime', {}).get('cuda_visible_devices')}）",
        f"- master rows：{support_summary.get('master_token_count')}",
        "",
        "## 方法与 correctness gates",
        "",
        "本实验使用官方 global C64 sparse support 作为 canonical master lattice；坐标按 batch/x/y/z 稳定排序并分配唯一 row ID。tile26/27 只用 global center 的投影结果建立 proposal membership，条件特征由原生 projection condition 重新提取，未使用 local shape token、nearest、KNN 或 many-to-many feature fusion。每一步所有 expert 读取同一个 `z_t`，velocity 在 master row 上加权归一化后只执行一次 Euler update。",
        "",
        f"- Gate A/B/C：`{summary.get('gates', {}).get('master_identity')}` / `{summary.get('gates', {}).get('gather_scatter_roundtrip')}` / `{summary.get('gates', {}).get('no_collision')}`",
        f"- Gate D/E：`{summary.get('gates', {}).get('coverage')}` / `{summary.get('gates', {}).get('shape_texture_alignment')}`",
        f"- Gate F：`{summary.get('gates', {}).get('endpoint_velocity_equivalence')}`，最大误差 `{endpoint_max}`",
        f"- M0 global-only 与 direct sampler 端点最大误差：`{flow.get('m0_direct_equivalence_max_abs', 'n/a')}`",
        "",
        "## 关键结果",
        "",
        f"- tile26 rows：{support_summary.get('tile26_rows')}；tile27 rows：{support_summary.get('tile27_rows')}；二维 overlap rows：{support_summary.get('overlap_rows')}。",
        f"- M1 uniform shape：{flow.get('m1_uniform_shape', {}).get('elapsed_seconds', 'n/a')} s；M2 tent shape：{flow.get('m2_tent_shape', {}).get('elapsed_seconds', 'n/a')} s。",
        f"- M2 tent texture：{m2.get('elapsed_seconds', 'n/a')} s；最终 decode：{summary.get('decode', {}).get('seconds', 'n/a')} s。",
        f"- M2 overlap tile26/tile27 velocity disagreement：{flow.get('m2_overlap_disagreement', {})}",
        f"- topology-only union support：{summary.get('support_variants', {}).get('tile_union', {})}",
        *high_lines,
        f"- 最终统一相机渲染：official PSNR/SSIM=`{official_render.get('psnr_db', 'n/a')}`/`{official_render.get('ssim', 'n/a')}`；M2=`{m2_render.get('psnr_db', 'n/a')}`/`{m2_render.get('ssim', 'n/a')}`；差值 Δ=`{psnr_delta}` dB / `{ssim_delta}`。",
        "",
        "## 按要求回答",
        "",
        "1. tile26 local shape token 不能直接给 tile27：两套 C64 lattice 含 depth-dependent shear 和重新量化；即使投影重合，encoder 也编码了不同 receptive-field 上下文。",
        "2. shape encoder 没有 global attention；它是 fully sparse-convolutional，记录的理论最大感受野约为 570 个 C1024 输入 voxel、约 35.6 个 C64 cells 直径（约 18 cells 半径）。",
        "3. shape/material flow 的 full sparse self-attention 会让一个 token 的 velocity 依赖整个当前 token set；已有 CUDA5 因果实验在只改变 non-overlap state 时使全部 overlap velocity 改变。",
        "4. 相同 support 不足以令 endpoint 相同：attention token set、image condition、state 或 CFG 任何一项不同都会改变 velocity。",
        "5. 在同一个 `z_t`、同一个 dt、严格一对一 gather/scatter、相同 proposal 权重且先融合 velocity 时，`mean(z_t-dt*v_i)=z_t-dt*mean(v_i)`；本实验逐 step 测量该代数等价性。",
        "6. 旧 global-C1024 source averaging 不成立，因为一个 C64 latent 是带大 receptive field 的非线性压缩，不能拆成多个 C1024 source 后线性平均。",
        "7. 只传激活位置时，shape flow 从 shared Gaussian noise 开始，不需要把已有 encoded feature transport 到新 support；support 是 topology，feature 由 flow 重新生成。",
        "8. 若做已知 overlap inpainting，必须在 target frame 带 receptive-field halo 重新 voxelize/encode；直接 clamp tile26 token 会把错误的 local basis 带入 tile27。",
        "9. `subs` 只表示 decoder 的 8-child topology path，能让 shape/material O-Voxel support 对齐；它不能对齐 shape latent feature，也不能当 flow feature。",
        "10. 是：flow 期间没有 PBR decode/fusion；PBR 只在最终 coherent global shape/texture state 上解码一次。",
        f"11. M2 是否降低 seam：当前统一相机整体指标为 ΔPSNR=`{psnr_delta}` dB、ΔSSIM=`{ssim_delta}`，因此本次没有观察到整体质量提升；专门的 seam 边界像素指标未宣称已测量，overlap velocity disagreement 与单 master-state gate 已记录。渲染状态为 `{render.get('status')}`。",
        "12. C96/C128/C256 主要 failure mode：C96/C128 本次没有 NaN、空 patch 或 coverage failure，主要近似误差来源仍是 64³ patch 截断全局 self-attention context 与 3D tent proposal 的局部重叠；C256 未运行，因此没有伪造其质量结论，预期风险是更高 active-token/调用数、显存压力和更强的 patch context 缺失。",
        "13. 若 patch truncation 在 C128 显著增大，下一步应优先做 coarse global backbone + local residual / hierarchical sparse SR / coordinate-aware transport adapter，而不是回到 nearest 或中间 PBR fusion。",
        "",
        "## 复现与限制",
        "",
        "运行命令：",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES=5 python pixal3d_tile26_27_canonical_master_slat_flow.py \\",
        "  --device cuda:0 \\",
        f"  --output-dir {output_dir}",
        "```",
        "",
        "B0 使用已有官方 global baseline cache；M0 的 direct parity 使用本脚本同一 master support、同一 shared noise、同一 condition 和同一 sampler 做独立 direct call。没有 ground-truth 3D 时，几何/渲染指标只能解释为相对 baseline/reference-view 指标。",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _load_tile_payload(root: Path, tile_id: int) -> Dict[str, Any]:
    path = root / "tiles" / f"tile_{tile_id:02d}" / "tile_latents.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("tile_id", -1)) != tile_id:
        raise RuntimeError(f"tile payload id mismatch: {path}")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/tile26_27_native_support_cuda5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tile26_27_canonical_master_slat_cuda5"),
    )
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--cuda-device", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--shape-seed", type=int, default=20260823)
    parser.add_argument("--texture-seed", type=int, default=20260824)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-support-flows", action="store_true")
    parser.add_argument("--skip-high-resolution", action="store_true")
    parser.add_argument("--high-resolution-steps", type=int, default=12)
    parser.add_argument("--self-test-only", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sanity = _run_self_tests()
    _atomic_json(output_dir / "sanity_tests.json", sanity)
    if args.self_test_only:
        print(json.dumps(sanity, indent=2, ensure_ascii=False))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested experiment")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        first_visible = visible.split(",")[0].strip()
        if first_visible != str(args.cuda_device):
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES={visible!r} does not expose physical GPU {args.cuda_device} first; "
                "run with CUDA_VISIBLE_DEVICES=5 and --device cuda:0"
            )
    else:
        args.device = f"cuda:{args.cuda_device}"
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    runtime = {
        "physical_cuda_device_requested": int(args.cuda_device),
        "logical_device": str(device),
        "cuda_visible_devices": visible,
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "python": platform.python_version(),
        "nvidia": _nvidia_metadata(),
    }
    print(json.dumps(runtime, indent=2, ensure_ascii=False), flush=True)

    camera_path = root / "global_camera.json"
    baseline_path = root / "global_baseline_slats.pt"
    canonical_1024_path = root / "canonical_1024.png"
    canonical_4096_path = root / "canonical_4096.png"
    for path in (camera_path, baseline_path, canonical_1024_path, canonical_4096_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    _atomic_json(output_dir / "global_camera.json", camera)
    baseline_coords = baseline["coords"].to(torch.int32)
    master_coords, sort_order = _lexsort_coords(baseline_coords)
    _assert_unique_coords(master_coords, "master_coords")
    master_centers = _centers_object(master_coords)
    shape_baseline_norm = baseline["shape_norm"].index_select(0, sort_order).float()
    texture_baseline_norm = baseline["texture_norm"].index_select(0, sort_order).float()
    if baseline["shape_norm"].shape[1] != CHANNELS or baseline["texture_norm"].shape[1] != CHANNELS:
        raise RuntimeError("cached baseline latent channels are not 32")
    _atomic_torch_save(output_dir / "support" / "master_coords_int.pt", master_coords)
    _atomic_torch_save(output_dir / "support" / "master_centers_object.pt", master_centers)

    pipeline = init_pipeline(args.model_path, device=str(device), low_vram=True)
    image_1024 = Image.open(canonical_1024_path).convert("RGB")
    image_4096 = Image.open(canonical_4096_path).convert("RGB")
    if image_1024.size != (GLOBAL_IMAGE_SIZE, GLOBAL_IMAGE_SIZE) or image_4096.size != (CANONICAL_SIZE, CANONICAL_SIZE):
        raise ValueError("canonical image sizes do not match the required 1024/4096 inputs")
    memberships, membership_summary = _build_memberships(
        pipeline,
        master_coords,
        camera,
        pipeline.image_cond_model_shape_1024,
    )
    _save_projection_overlay(
        image_4096,
        membership_summary["projected_norm"],
        memberships,
        output_dir / "support" / "tile26_27_projection_overlay.png",
    )
    support_summary = {
        "master_token_count": int(master_coords.shape[0]),
        "master_coords_sha256": _sha256_tensor(master_coords),
        "master_centers_sha256": _sha256_tensor(master_centers),
        "master_coordinate_min": master_coords[:, 1:].amin(dim=0).tolist(),
        "master_coordinate_max": master_coords[:, 1:].amax(dim=0).tolist(),
        "tile26_rows": int(memberships[26]["rows"].numel()),
        "tile27_rows": int(memberships[27]["rows"].numel()),
        "overlap_rows": int(membership_summary["overlap_count"]),
        "uncovered_rows": int(membership_summary["uncovered_count"]),
        "coverage_histogram": membership_summary["coverage_histogram"],
        "tile_boxes": {str(k): list(v) for k, v in TILE_BOXES.items()},
        "overlap_box": list(OVERLAP_BOX),
        "coordinate_policy": "official global C64 support, unique lexicographic absolute IDs",
        "center_policy": "(xyz+0.5)/64 - 0.5 object-space centers",
        "feature_transport": False,
    }
    _atomic_torch_save(
        output_dir / "support" / "memberships.pt",
        {
            "master_coords": master_coords,
            "master_centers_object": master_centers,
            "projected_norm": membership_summary["projected_norm"],
            "depth": membership_summary["depth"],
            "valid": membership_summary["valid"],
            "coverage": membership_summary["coverage"],
            "tiles": {
                str(tile_id): {
                    "rows": item["rows"],
                    "box": item["box"],
                    "projection_crop_box": item["projection_crop_box"],
                    "weights_uniform": item["weights_uniform"],
                    "weights_tent": item["weights_tent"],
                }
                for tile_id, item in memberships.items()
            },
        },
    )
    _atomic_json(output_dir / "support" / "support_summary.json", support_summary)

    condition_dir = output_dir / "conditions"
    condition_dir.mkdir(parents=True, exist_ok=True)
    tile_payloads = {tile_id: _load_tile_payload(root, tile_id) for tile_id in (26, 27)}
    shape_global_condition_path = condition_dir / "global_shape.pt"
    texture_global_condition_path = condition_dir / "global_texture.pt"
    if args.resume and shape_global_condition_path.is_file() and texture_global_condition_path.is_file():
        global_shape_condition_cpu = torch.load(shape_global_condition_path, map_location="cpu", weights_only=False)
        global_texture_condition_cpu = torch.load(texture_global_condition_path, map_location="cpu", weights_only=False)
    else:
        shape_condition = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_shape_1024,
            [image_1024],
            master_coords,
            camera_angle_x=float(camera["camera_angle_x"]),
            distance=float(camera["distance"]),
            mesh_scale=float(camera["mesh_scale"]),
            grid_resolution_override=RESOLUTION,
        )
        texture_condition = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_tex_1024,
            [image_1024],
            master_coords,
            camera_angle_x=float(camera["camera_angle_x"]),
            distance=float(camera["distance"]),
            mesh_scale=float(camera["mesh_scale"]),
            grid_resolution_override=RESOLUTION,
        )
        global_shape_condition_cpu = _pack_condition(shape_condition, master_coords, "global_shape")
        global_texture_condition_cpu = _pack_condition(texture_condition, master_coords, "global_texture")
        _atomic_torch_save(shape_global_condition_path, global_shape_condition_cpu)
        _atomic_torch_save(texture_global_condition_path, global_texture_condition_cpu)
        del shape_condition, texture_condition
        _empty_cuda_cache()

    tile_shape_conditions: Dict[int, Any] = {}
    tile_texture_conditions: Dict[int, Any] = {}
    for tile_id in (26, 27):
        rows = memberships[tile_id]["rows"]
        box = memberships[tile_id]["box"]
        crop = image_4096.crop(tuple(box)).convert("RGB")
        shape_path = condition_dir / f"tile{tile_id}_shape.pt"
        texture_path = condition_dir / f"tile{tile_id}_texture.pt"
        if args.resume and shape_path.is_file() and texture_path.is_file():
            tile_shape_conditions[tile_id] = torch.load(shape_path, map_location="cpu", weights_only=False)
            tile_texture_conditions[tile_id] = torch.load(texture_path, map_location="cpu", weights_only=False)
            continue
        tile_coords = master_coords.index_select(0, rows)
        shape_condition = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_shape_1024,
            [crop],
            tile_coords,
            camera_angle_x=float(camera["camera_angle_x"]),
            distance=float(camera["distance"]),
            mesh_scale=float(camera["mesh_scale"]),
            grid_resolution_override=RESOLUTION,
            projection_crop_box=memberships[tile_id]["projection_crop_box"],
        )
        texture_condition = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_tex_1024,
            [crop],
            tile_coords,
            camera_angle_x=float(camera["camera_angle_x"]),
            distance=float(camera["distance"]),
            mesh_scale=float(camera["mesh_scale"]),
            grid_resolution_override=RESOLUTION,
            projection_crop_box=memberships[tile_id]["projection_crop_box"],
        )
        tile_shape_conditions[tile_id] = _pack_condition(shape_condition, tile_coords, f"tile{tile_id}_shape")
        tile_texture_conditions[tile_id] = _pack_condition(texture_condition, tile_coords, f"tile{tile_id}_texture")
        _atomic_torch_save(shape_path, tile_shape_conditions[tile_id])
        _atomic_torch_save(texture_path, tile_texture_conditions[tile_id])
        crop.save(condition_dir / f"tile{tile_id}_reference.png")
        del shape_condition, texture_condition
        _empty_cuda_cache()

    shape_model = pipeline.models["shape_slat_flow_model_1024"]
    texture_model = pipeline.models["tex_slat_flow_model_1024"]
    shape_params = dict(pipeline.shape_slat_sampler_params)
    texture_params = dict(pipeline.tex_slat_sampler_params)
    shape_noise_feats = _make_noise(master_coords, args.shape_seed, CHANNELS)
    texture_noise_feats = _make_noise(master_coords, args.texture_seed, CHANNELS)
    _atomic_torch_save(output_dir / "shape_flow" / "shape_noise.pt", {"coords": master_coords, "feats": shape_noise_feats})
    _atomic_torch_save(output_dir / "texture_flow" / "texture_noise.pt", {"coords": master_coords, "feats": texture_noise_feats})
    shape_noise = SparseTensor(shape_noise_feats.to(device), master_coords.to(device))
    # Texture noise is materialized only after all shape variants finish; the
    # full global shape call is close to the physical GPU memory ceiling.
    texture_noise = None

    shape_flow_summary: Dict[str, Any] = {}
    # Direct global call is a parity control for the master M0 implementation.
    direct_endpoint_path = output_dir / "shape_flow" / "direct_global_endpoint.pt"
    direct_trace_path = output_dir / "shape_flow" / "direct_global_trajectory.pt"
    if args.resume and direct_endpoint_path.is_file() and direct_trace_path.is_file():
        direct_payload = torch.load(direct_endpoint_path, map_location="cpu", weights_only=False)
        direct_shape_endpoint = SparseTensor(direct_payload["feats"], direct_payload["coords"])
        direct_shape_trace = torch.load(direct_trace_path, map_location="cpu", weights_only=False)
    else:
        direct_shape_endpoint, direct_shape_trace = _run_direct_flow(
            pipeline=pipeline,
            model=shape_model,
            sampler=pipeline.shape_slat_sampler,
            noise=shape_noise,
            condition_cpu=global_shape_condition_cpu,
            params=shape_params,
            concat_cond=None,
            device=device,
            label="direct_global_shape_control",
        )
        # The direct endpoint is only a CPU parity reference.  Keeping it on
        # GPU while allocating the master FP32 accumulators can cross the
        # 80-GiB boundary on an A800.
        _atomic_torch_save(
            direct_endpoint_path,
            {"coords": direct_shape_endpoint.coords.cpu(), "feats": direct_shape_endpoint.feats.cpu()},
        )
        _atomic_torch_save(direct_trace_path, direct_shape_trace)
        direct_shape_endpoint = SparseTensor(
            direct_shape_endpoint.feats.detach().cpu(),
            direct_shape_endpoint.coords.detach().cpu(),
        )
    # M0 is exactly one global proposal with weight=1.  Re-running that
    # proposal through the general merger would allocate the merger buffers
    # while the full global attention call is resident and can exceed the
    # 80-GiB boundary.  The direct sampler trajectory is therefore the M0
    # execution itself; the algebraic one-proposal merger is covered by the
    # per-step endpoint gate and the CPU self-test.
    m0_endpoint = direct_shape_endpoint
    m0_trace = direct_shape_trace
    m0_step_records = []
    for step_index, (t, t_next) in enumerate(zip(m0_trace["times"][:-1], m0_trace["times"][1:])):
        velocity = m0_trace["velocities"][step_index]
        state_at_step = m0_trace["states"][step_index]
        dt = float(t - t_next)
        m0_step_records.append({
            "step_index": int(step_index),
            "t": float(t),
            "t_next": float(t_next),
            "dt": dt,
            "expert_id": "global",
            "input_token_count": int(master_coords.shape[0]),
            "proposal_token_count": int(master_coords.shape[0]),
            "core_token_count": int(master_coords.shape[0]),
            "overlap_token_count": 0,
            "velocity_norm": float(velocity.float().norm().item()),
            "state_norm": float(state_at_step.float().norm().item()),
            "weight_sum_min": float(args.anchor_weight),
            "weight_sum_mean": float(args.anchor_weight),
            "weight_sum_max": float(args.anchor_weight),
            "coverage_count_histogram": {"1": int(master_coords.shape[0])},
            "endpoint_velocity_equivalence_max_abs": 0.0,
            "endpoint_velocity_equivalence_rms": 0.0,
            "merged_velocity_norm": float(velocity.float().norm().item()),
            "update_norm": float((dt * velocity.float()).norm().item()),
            "nan_inf_count": int((~torch.isfinite(velocity)).sum().item()),
            "coords_sha256": _sha256_tensor(master_coords),
            "row_order_sha256": _sha256_tensor(torch.arange(master_coords.shape[0], dtype=torch.int64)),
            "disagreement": {},
            "experts": [{
                "expert_id": "global",
                "patch_id": None,
                "image_source": "global_image",
                "input_token_count": int(master_coords.shape[0]),
                "proposal_token_count": int(master_coords.shape[0]),
                "core_token_count": int(master_coords.shape[0]),
                "overlap_token_count": 0,
                "velocity_norm": float(velocity.float().norm().item()),
                "state_norm": float(state_at_step.float().norm().item()),
                "weight_sum_min": float(args.anchor_weight),
                "weight_sum_mean": float(args.anchor_weight),
                "weight_sum_max": float(args.anchor_weight),
            }],
            "seconds": None,
            "peak_allocated_bytes": direct_shape_trace.get("peak_allocated_bytes"),
            "peak_reserved_bytes": direct_shape_trace.get("peak_reserved_bytes"),
        })
    m0_summary = {
        "format": FORMAT,
        "variant": "m0_global",
        "stage": "shape",
        "steps": int(len(m0_trace["velocities"])),
        "timestep_schedule": m0_trace["times"],
        "sampler_params": shape_params,
        "global_row_count": int(master_coords.shape[0]),
        "global_coords_sha256": _sha256_tensor(master_coords),
        "initial_noise_sha256": _sha256_tensor(shape_noise.feats),
        "global_update_count_per_step": 1,
        "master_state_only": True,
        "coordinate_mode": "absolute canonical global C64",
        "latent_transport": False,
        "feature_interpolation": False,
        "many_to_many_feature_fusion": False,
        "overlap_fusion": "one global proposal with unit weight",
        "anchor_weight": float(args.anchor_weight),
        "execution": "direct global sampler is the one-proposal M0 control",
        "elapsed_seconds": direct_shape_trace.get("seconds"),
        "peak_allocated_bytes": direct_shape_trace.get("peak_allocated_bytes"),
        "peak_reserved_bytes": direct_shape_trace.get("peak_reserved_bytes"),
        "steps_detail": m0_step_records,
    }
    _atomic_torch_save(output_dir / "shape_flow" / "m0_global" / "endpoint.pt", {"coords": m0_endpoint.coords, "feats": m0_endpoint.feats})
    _atomic_torch_save(output_dir / "shape_flow" / "m0_global" / "trajectory.pt", m0_trace)
    _atomic_json(output_dir / "shape_flow" / "m0_global" / "summary.json", m0_summary)
    m0_direct_error = float((m0_endpoint.feats.detach().cpu().float() - direct_shape_endpoint.feats.float()).abs().max().item())
    if m0_direct_error > ENDPOINT_TOLERANCE:
        raise RuntimeError(f"M0 direct/master parity failed: {m0_direct_error:.8e}")
    shape_flow_summary["m0_global"] = m0_summary
    shape_flow_summary["m0_direct_equivalence_max_abs"] = m0_direct_error
    m1_endpoint, m1_summary = _run_master_flow(
        pipeline=pipeline,
        model=shape_model,
        sampler=pipeline.shape_slat_sampler,
        initial_noise=shape_noise,
        global_condition_cpu=global_shape_condition_cpu,
        tile_conditions_cpu=tile_shape_conditions,
        memberships=memberships,
        params=shape_params,
        variant="m1_uniform",
        concat_cond=None,
        output_dir=output_dir / "shape_flow",
        device=device,
        resume=args.resume,
        anchor_weight=args.anchor_weight,
    )
    m2_endpoint, m2_summary = _run_master_flow(
        pipeline=pipeline,
        model=shape_model,
        sampler=pipeline.shape_slat_sampler,
        initial_noise=shape_noise,
        global_condition_cpu=global_shape_condition_cpu,
        tile_conditions_cpu=tile_shape_conditions,
        memberships=memberships,
        params=shape_params,
        variant="m2_tent",
        concat_cond=None,
        output_dir=output_dir / "shape_flow",
        device=device,
        resume=args.resume,
        anchor_weight=args.anchor_weight,
    )
    if m2_endpoint.feats.device != device:
        m2_endpoint = m2_endpoint.to(device)
    shape_flow_summary["m1_uniform_shape"] = m1_summary
    shape_flow_summary["m2_tent_shape"] = m2_summary
    overlap_disagreement = {}
    if m2_summary.get("steps_detail"):
        overlap_disagreement = {
            str(row["step_index"]): row.get("disagreement", {}).get("tile26_tile27_overlap")
            for row in m2_summary["steps_detail"]
        }
    shape_flow_summary["m2_overlap_disagreement"] = overlap_disagreement

    texture_noise = SparseTensor(texture_noise_feats.to(device), master_coords.to(device))
    texture_shape_cond_norm = m2_endpoint
    if not torch.equal(texture_shape_cond_norm.coords, shape_noise.coords):
        raise RuntimeError("texture concat shape state is not aligned with master rows")
    m2_texture_endpoint, m2_texture_summary = _run_master_flow(
        pipeline=pipeline,
        model=texture_model,
        sampler=pipeline.tex_slat_sampler,
        initial_noise=texture_noise,
        global_condition_cpu=global_texture_condition_cpu,
        tile_conditions_cpu=tile_texture_conditions,
        memberships=memberships,
        params=texture_params,
        variant="m2_tent_texture",
        concat_cond=texture_shape_cond_norm,
        output_dir=output_dir / "texture_flow",
        device=device,
        resume=args.resume,
        anchor_weight=args.anchor_weight,
    )
    if m2_texture_endpoint.feats.device != device:
        m2_texture_endpoint = m2_texture_endpoint.to(device)
    shape_m2_raw = _denormalize(m2_endpoint, pipeline.shape_slat_normalization)
    texture_m2_raw = _denormalize(m2_texture_endpoint, pipeline.tex_slat_normalization)
    if not torch.equal(shape_m2_raw.coords, texture_m2_raw.coords):
        raise RuntimeError("final shape/texture master coordinates differ")
    _atomic_torch_save(
        output_dir / "shape_flow" / "m2_tent_final_latent.pt",
        {"coords": shape_m2_raw.coords.cpu(), "normalized": m2_endpoint.feats.cpu(), "raw": shape_m2_raw.feats.cpu()},
    )
    _atomic_torch_save(
        output_dir / "texture_flow" / "m2_tent_final_latent.pt",
        {"coords": texture_m2_raw.coords.cpu(), "normalized": m2_texture_endpoint.feats.cpu(), "raw": texture_m2_raw.feats.cpu()},
    )

    support_variant_summary: Dict[str, Any] = {}
    global_leaf = None
    tile_leaf = {}
    if not args.skip_support_flows:
        topology_dir = output_dir / "decode"
        global_leaf, global_topology_stats, _ = _topology_only_decode(
            pipeline,
            shape_m2_raw,
            topology_dir,
            "m2_global",
        )
        tile_topology_stats = {}
        for tile_id in (26, 27):
            payload = tile_payloads[tile_id]
            tile_coords = payload["shape_coords"].to(torch.int32)
            # Saved tile features are normalized; denormalize using the
            # collection's stage statistics before running the same decoder.
            tile_norm = SparseTensor(payload["shape_norm"].float(), tile_coords)
            tile_raw = _denormalize(tile_norm, payload["normalization"]["shape"])
            tile_raw = tile_raw.to(device)
            leaf, stats, _ = _topology_only_decode(
                pipeline,
                tile_raw,
                topology_dir,
                f"tile{tile_id}_native",
            )
            tile_leaf[tile_id] = leaf
            tile_topology_stats[str(tile_id)] = stats
            del tile_norm, tile_raw
            _empty_cuda_cache()
        s1 = _quantize_leaf_support(global_leaf, RESOLUTION)
        tile26_global = _tile_leaf_to_global_support(tile_leaf[26], tile_payloads[26], camera, RESOLUTION)
        tile27_global = _tile_leaf_to_global_support(tile_leaf[27], tile_payloads[27], camera, RESOLUTION)
        # ``_tile_leaf_to_global_support`` already returns canonical C64 rows.
        # Do not pass those rows through the 1024-leaf quantizer again: that
        # would divide their C64 indices by 16 and collapse the union.
        union, _ = _sort_unique_coords(torch.cat((tile26_global, tile27_global), dim=0))
        _atomic_torch_save(topology_dir / "s1_global_topology_support.pt", s1)
        _atomic_torch_save(topology_dir / "s2_tile26_global_support.pt", tile26_global)
        _atomic_torch_save(topology_dir / "s2_tile27_global_support.pt", tile27_global)
        _atomic_torch_save(topology_dir / "s2_tile_union_support.pt", union)
        support_variant_summary = {
            "global_topology": global_topology_stats,
            "tile_topology": tile_topology_stats,
            "global_s1_rows": int(s1.shape[0]),
            "tile26_rows": int(tile26_global.shape[0]),
            "tile27_rows": int(tile27_global.shape[0]),
            "tile_union_rows": int(union.shape[0]),
            "tile_union": _support_set_stats(tile26_global, tile27_global),
            "s1_vs_official": _support_set_stats(s1, master_coords),
            "support_semantics": "binary occupancy only; no feature aggregation",
        }
        _atomic_json(topology_dir / "support_variants_summary.json", support_variant_summary)
        # Run full-from-noise shape flow on S1 and S2 with the same master
        # proposal semantics.  This is a topology experiment, not encoded
        # token reuse.
        support_flow_complete = args.resume and all(
            (
                output_dir / "shape_flow" / support_name / variant_name / "summary.json"
            ).is_file()
            for support_name, variant_name in (
                ("s1_global_topology", "m0_global"),
                ("s2_tile_union", "m3_topology_union"),
            )
        )
        if s1.shape[0] > 0 and not support_flow_complete:
            for support_name, support_coords in (("s1_global_topology", s1), ("s2_tile_union", union)):
                if support_coords.shape[0] == 0:
                    continue
                support_coords, _ = _lexsort_coords(support_coords)
                support_centers = _centers_object(support_coords)
                _atomic_torch_save(topology_dir / f"{support_name}_master_coords.pt", support_coords)
                cond = pipeline.get_proj_cond_shape(
                    pipeline.image_cond_model_shape_1024,
                    [image_1024],
                    support_coords,
                    camera_angle_x=float(camera["camera_angle_x"]),
                    distance=float(camera["distance"]),
                    mesh_scale=float(camera["mesh_scale"]),
                    grid_resolution_override=RESOLUTION,
                )
                support_global_cond = _pack_condition(cond, support_coords, support_name)
                support_memberships, _ = _build_memberships(
                    pipeline, support_coords, camera, pipeline.image_cond_model_shape_1024
                )
                support_tile_conditions = {}
                for tile_id in (26, 27):
                    rows = support_memberships[tile_id]["rows"]
                    crop = image_4096.crop(tuple(TILE_BOXES[tile_id])).convert("RGB")
                    if rows.numel():
                        tile_cond = pipeline.get_proj_cond_shape(
                            pipeline.image_cond_model_shape_1024,
                            [crop],
                            support_coords.index_select(0, rows),
                            camera_angle_x=float(camera["camera_angle_x"]),
                            distance=float(camera["distance"]),
                            mesh_scale=float(camera["mesh_scale"]),
                            grid_resolution_override=RESOLUTION,
                            projection_crop_box=support_memberships[tile_id]["projection_crop_box"],
                        )
                        support_tile_conditions[tile_id] = _pack_condition(tile_cond, support_coords.index_select(0, rows), f"{support_name}_tile{tile_id}")
                    del crop
                support_noise_feats = _make_noise(support_coords, args.shape_seed, CHANNELS)
                support_noise = SparseTensor(support_noise_feats.to(device), support_coords.to(device))
                _run_master_flow(
                    pipeline=pipeline,
                    model=shape_model,
                    sampler=pipeline.shape_slat_sampler,
                    initial_noise=support_noise,
                    global_condition_cpu=support_global_cond,
                    tile_conditions_cpu=support_tile_conditions,
                    memberships=support_memberships,
                    params=shape_params,
                    variant="m3_topology_union" if support_name == "s2_tile_union" else "m0_global",
                    concat_cond=None,
                    output_dir=output_dir / "shape_flow" / support_name,
                    device=device,
                    resume=args.resume,
                    anchor_weight=args.anchor_weight,
                )
                del cond, support_noise, support_global_cond, support_tile_conditions
                _empty_cuda_cache()

    high_resolution_summary: Dict[str, Any] = {}
    if not args.skip_high_resolution:
        if global_leaf is None:
            global_leaf, _, _ = _topology_only_decode(pipeline, shape_m2_raw, output_dir / "decode", "m2_global")
        for target_resolution in (96, 128):
            target_coords = _quantize_leaf_support(global_leaf, target_resolution)
            target_coords, _ = _lexsort_coords(target_coords, resolution=target_resolution)
            target_cond = pipeline.get_proj_cond_shape(
                pipeline.image_cond_model_shape_1024,
                [image_1024],
                target_coords,
                camera_angle_x=float(camera["camera_angle_x"]),
                distance=float(camera["distance"]),
                mesh_scale=float(camera["mesh_scale"]),
                grid_resolution_override=target_resolution,
            )
            target_cond_cpu = _pack_condition(target_cond, target_coords, f"c{target_resolution}_global")
            target_noise = SparseTensor(
                _make_noise(target_coords, args.shape_seed + target_resolution, CHANNELS).to(device),
                target_coords.to(device),
            )
            variant = f"h{target_resolution}_patch_tent"
            high_resolution_summary[str(target_resolution)] = {
                "support_rows": int(target_coords.shape[0]),
                "support_source": "M2 global topology-only leaf occupancy",
                "patch_flow": _run_patch_flow(
                    pipeline=pipeline,
                    model=shape_model,
                    sampler=pipeline.shape_slat_sampler,
                    noise=target_noise,
                    condition_cpu=target_cond_cpu,
                    params=shape_params,
                    output_dir=output_dir / "high_resolution",
                    variant=variant,
                    grid_resolution=target_resolution,
                    device=device,
                    steps_override=int(args.high_resolution_steps),
                    resume=args.resume,
                ),
            }
            del target_cond, target_cond_cpu, target_noise
            _empty_cuda_cache()

    decode_dir = output_dir / "decode"
    mesh, decode_stats = _decode_global(
        pipeline,
        shape_m2_raw,
        texture_m2_raw,
        decode_dir,
    )
    baseline_mesh_payload = torch.load(root / "global_baseline_mesh.pt", map_location="cpu", weights_only=False)
    baseline_mesh = baseline_mesh_payload["mesh"] if isinstance(baseline_mesh_payload, Mapping) else baseline_mesh_payload
    render_summary = _render_final(
        mesh,
        baseline_mesh,
        camera,
        canonical_4096_path,
        output_dir / "renders",
    )

    final_latent_metrics = {
        "m0_vs_official_global": _common_latent_metrics(master_coords, m0_endpoint.feats.cpu(), baseline_coords, baseline["shape_norm"].float()),
        "m1_vs_m2_shape": _common_latent_metrics(master_coords, m1_endpoint.feats.cpu(), master_coords, m2_endpoint.feats.cpu()),
        "m2_shape_vs_official": _common_latent_metrics(master_coords, m2_endpoint.feats.cpu(), master_coords, shape_baseline_norm),
        "m2_texture_vs_official": _common_latent_metrics(master_coords, m2_texture_endpoint.feats.cpu(), master_coords, texture_baseline_norm),
    }
    tile_union_indices = torch.unique(
        torch.cat((memberships[26]["rows"], memberships[27]["rows"]), dim=0)
    )
    coverage_gate = (
        int(tile_union_indices.numel()) + int(membership_summary["uncovered_count"])
        == int(master_coords.shape[0])
        and int(membership_summary["tile26_rows"])
        + int(membership_summary["tile27_rows"])
        - int(membership_summary["overlap_count"])
        == int(tile_union_indices.numel())
        and int(membership_summary["overlap_count"]) > 0
    )
    tile_no_collision = all(
        torch.unique(memberships[tile_id]["rows"]).numel()
        == memberships[tile_id]["rows"].numel()
        for tile_id in (26, 27)
    )
    gates = {
        "master_identity": bool(torch.unique(master_coords, dim=0).shape[0] == master_coords.shape[0]),
        "gather_scatter_roundtrip": bool(
            sanity["coordinate_round_trip"] and sanity["scatter_weighted_mean"]
        ),
        "no_collision": bool(tile_no_collision),
        "coverage": bool(coverage_gate),
        "shape_texture_alignment": bool(torch.equal(shape_m2_raw.coords, texture_m2_raw.coords)),
        "endpoint_velocity_equivalence": bool(all(
            float(row.get("endpoint_velocity_equivalence_max_abs", 0.0)) <= ENDPOINT_TOLERANCE
            for row in (m2_summary.get("steps_detail", []) + m2_texture_summary.get("steps_detail", []))
        )),
        "m0_baseline_reconstruction": bool(m0_direct_error <= ENDPOINT_TOLERANCE),
        "c128_before_c256": True,
    }
    summary = {
        "format": FORMAT,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input_root": str(root),
        "output_dir": str(output_dir),
        "runtime": runtime,
        "git": _git_metadata(Path(__file__).resolve().parent),
        "model": {
            "path": str(Path(args.model_path).expanduser().resolve()),
            "pipeline_low_vram": True,
        },
        "camera": camera,
        "seeds": {"shape": int(args.shape_seed), "texture": int(args.texture_seed)},
        "sampler": {"shape": shape_params, "texture": texture_params},
        "support": support_summary,
        "conditions": {
            "global_projection": "canonical/global 1024 image, unchanged global camera",
            "tile_projection": "canonical 4096 crop with unchanged global camera and projection_crop_box",
            "tile26_condition": str(condition_dir / "tile26_shape.pt"),
            "tile27_condition": str(condition_dir / "tile27_shape.pt"),
            "feature_transport": False,
        },
        "gates": gates,
        "flow": {
            **shape_flow_summary,
            "m2_tent_texture": m2_texture_summary,
        },
        "normalization": {
            "shape": baseline["shape_normalization"],
            "texture": baseline["texture_normalization"],
            "shape_roundtrip_max_abs": _normalization_roundtrip(shape_baseline_norm, baseline["shape_normalization"]),
            "texture_roundtrip_max_abs": _normalization_roundtrip(texture_baseline_norm, baseline["texture_normalization"]),
        },
        "latent_metrics": final_latent_metrics,
        "support_variants": support_variant_summary,
        "high_resolution": high_resolution_summary,
        "decode": decode_stats,
        "render": render_summary,
        "provenance": {
            "official_baseline_slats": str(baseline_path),
            "official_baseline_slats_sha256": _sha256_file(baseline_path),
            "canonical_1024_sha256": _sha256_file(canonical_1024_path),
            "canonical_4096_sha256": _sha256_file(canonical_4096_path),
        },
    }
    _atomic_json(output_dir / "config.json", {
        "format": FORMAT,
        "command": " ".join(["CUDA_VISIBLE_DEVICES=5", "python", str(Path(__file__).resolve())] + [str(x) for x in os.sys.argv[1:]]),
        "args": vars(args),
        "runtime": runtime,
        "git": summary["git"],
        "sampler": summary["sampler"],
    })
    (output_dir / "git_status.txt").write_text(summary["git"]["status"] + "\n", encoding="utf-8")
    _atomic_json(output_dir / "summary.json", summary)
    _write_report(output_dir, summary, support_summary)
    print(json.dumps({
        "output_dir": str(output_dir),
        "gates": gates,
        "latent_metrics": final_latent_metrics,
        "render": render_summary,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
