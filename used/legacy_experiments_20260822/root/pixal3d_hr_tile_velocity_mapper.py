#!/usr/bin/env python3
"""Train the Codex.md HR-tile velocity mapper on the cached small dataset.

The frozen Pixal3D denoiser is evaluated three times on one shared flow state:

    G = F(x_t, global canonical-1024 condition)
    H = F(gather(x_t), canonical-4096 HR tile condition)
    L = F(gather(x_t), canonical-1024 matching LR crop condition)

Only the point-wise channel mapper Phi(G, H-L) is optimized.  Sparse support,
row order, noisy state, timestep, local token context, and decoder space are
never changed.  Train/test splitting is by object, not by view or tile.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

from dataclasses import asdict, dataclass
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor


CANONICAL_SIZE = 4096
GLOBAL_SIZE = 1024
TILE_SIZE = 1024
TILE_STRIDE = 512
GRID_RESOLUTION = 64
SIGMA_MIN = 1e-5
TIME_BIN_CENTERS = (0.1, 0.3, 0.5, 0.7, 0.9)
# This is deliberately conservative: a frozen sparse-transformer forward does
# not retain gradients, but each sparse token still creates several temporary
# hidden/QKV/MLP buffers.  The runtime OOM fallback below remains the final
# authority because allocator fragmentation and attention backends vary.
AUTO_BYTES_PER_MODEL_TOKEN = 256 * 1024


@dataclass(frozen=True)
class TileItem:
    split: str
    object_id: str
    view_name: str
    tile_id: int
    object_dir: str
    view_dir: str
    tile_rows: tuple[int, ...]
    owner_positions: tuple[int, ...]

    @property
    def key(self) -> str:
        return f"{self.object_id}/{self.view_name}/tile_{self.tile_id:03d}"


@dataclass(frozen=True)
class BatchPlan:
    """GPU- and token-aware upper bounds for one optimizer mini-batch."""

    max_examples: int
    model_token_budget: int
    requested_batch_size: int
    requested_model_token_budget: int
    free_memory_bytes: int
    total_memory_bytes: int
    memory_fraction: float
    bytes_per_model_token: int
    item_model_tokens_min: int
    item_model_tokens_median: int
    item_model_tokens_max: int
    item_owner_tokens_median: int


def compare_loss(before: float, after: float) -> dict[str, float | str]:
    """Describe an MSE change; positive gain always means an improvement."""
    before = float(before)
    after = float(after)
    delta = after - before
    scale = max(abs(before), 1e-12)
    gain_percent = 100.0 * (before - after) / scale
    tolerance = max(1e-12, scale * 1e-9)
    if delta < -tolerance:
        status = "improved"
    elif delta > tolerance:
        status = "degraded"
    else:
        status = "unchanged"
    return {
        "status": status,
        "delta": delta,
        "gain_percent": gain_percent,
        "magnitude_percent": abs(gain_percent),
    }


def _format_loss_change(comparison: Mapping[str, float | str]) -> str:
    status = str(comparison["status"])
    magnitude = float(comparison["magnitude_percent"])
    delta = float(comparison["delta"])
    if status == "unchanged":
        return f"unchanged(0.000%, delta={delta:+.6f})"
    return f"{status}({magnitude:.3f}%, delta={delta:+.6f})"


def design_batch_plan(
    model_token_counts: Sequence[int],
    owner_token_counts: Sequence[int],
    *,
    free_memory_bytes: int,
    total_memory_bytes: int,
    requested_batch_size: int,
    requested_model_token_budget: int,
    max_auto_batch_size: int,
    memory_fraction: float,
    bytes_per_model_token: int = AUTO_BYTES_PER_MODEL_TOKEN,
) -> BatchPlan:
    """Choose batch limits from available GPU memory and sparse-token counts."""
    if not model_token_counts or len(model_token_counts) != len(owner_token_counts):
        raise ValueError("model/owner token counts must be non-empty and aligned")
    if min(model_token_counts) <= 0 or min(owner_token_counts) <= 0:
        raise ValueError("all token counts must be positive")
    if requested_batch_size < 0 or requested_model_token_budget < 0:
        raise ValueError("requested batch limits cannot be negative")
    if max_auto_batch_size <= 0:
        raise ValueError("max_auto_batch_size must be positive")
    if not 0.0 < memory_fraction < 1.0:
        raise ValueError("memory_fraction must be in (0,1)")
    if bytes_per_model_token <= 0:
        raise ValueError("bytes_per_model_token must be positive")

    model_counts = np.asarray(model_token_counts, dtype=np.int64)
    owner_counts = np.asarray(owner_token_counts, dtype=np.int64)
    minimum = int(model_counts.min())
    median = max(1, int(np.median(model_counts)))
    maximum = int(model_counts.max())
    owner_median = max(1, int(np.median(owner_counts)))
    automatic_budget = int(
        max(0, free_memory_bytes) * memory_fraction / bytes_per_model_token
    )
    # A singleton must always be schedulable.  It may still fail in the real
    # model, in which case the runtime raises an actionable singleton OOM.
    automatic_budget = max(maximum, automatic_budget)
    token_budget = (
        int(requested_model_token_budget)
        if requested_model_token_budget > 0
        else automatic_budget
    )
    automatic_examples = max(1, token_budget // median)
    automatic_examples = min(max_auto_batch_size, automatic_examples)
    max_examples = (
        int(requested_batch_size)
        if requested_batch_size > 0
        else automatic_examples
    )
    return BatchPlan(
        max_examples=max_examples,
        model_token_budget=token_budget,
        requested_batch_size=int(requested_batch_size),
        requested_model_token_budget=int(requested_model_token_budget),
        free_memory_bytes=int(free_memory_bytes),
        total_memory_bytes=int(total_memory_bytes),
        memory_fraction=float(memory_fraction),
        bytes_per_model_token=int(bytes_per_model_token),
        item_model_tokens_min=minimum,
        item_model_tokens_median=median,
        item_model_tokens_max=maximum,
        item_owner_tokens_median=owner_median,
    )


class LowRankVelocityMapper(nn.Module):
    """Bias-free per-token map with Phi(G, 0)=0 and exact zero init."""

    def __init__(
        self,
        channels: int = 32,
        hidden_channels: int = 96,
        rank: int = 8,
    ) -> None:
        super().__init__()
        if min(channels, hidden_channels, rank) <= 0:
            raise ValueError("channel and rank values must be positive")
        self.channels = int(channels)
        self.rank = int(rank)
        self.g_norm = nn.RMSNorm(channels)
        self.d_norm = nn.RMSNorm(channels)
        self.modulator = nn.Sequential(
            nn.Linear(2 * channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, rank),
        )
        self.base_map = nn.Linear(channels, channels, bias=False)
        self.down_map = nn.Linear(channels, rank, bias=False)
        self.up_map = nn.Linear(rank, channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.base_map.weight)
        nn.init.xavier_uniform_(self.down_map.weight)
        nn.init.xavier_uniform_(self.up_map.weight)
        for layer in self.modulator[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        final = self.modulator[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, g: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        if g.ndim != 2 or g.shape != d.shape:
            raise ValueError(f"G and D must be aligned [N,C], got {g.shape}, {d.shape}")
        if g.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {g.shape[1]}")
        condition = torch.cat((self.g_norm(g), self.d_norm(d)), dim=-1)
        modulation = torch.tanh(self.modulator(condition))
        return self.base_map(d) + self.up_map(modulation * self.down_map(d))


def tile_boxes() -> list[tuple[int, int, int, int]]:
    starts = list(range(0, CANONICAL_SIZE - TILE_SIZE + 1, TILE_STRIDE))
    boxes = [
        (x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE)
        for y0 in starts
        for x0 in starts
    ]
    if len(boxes) != 49:
        raise RuntimeError(f"expected 49 tiles, got {len(boxes)}")
    return boxes


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _coords_digest(coords: torch.Tensor | np.ndarray) -> str:
    if isinstance(coords, torch.Tensor):
        array = coords.detach().cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(coords)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def discover_complete_objects(dataset_root: Path) -> list[str]:
    manifest = json.loads((dataset_root / "dataset.json").read_text(encoding="utf-8"))
    result = []
    for object_id in manifest["objects"]:
        object_dir = dataset_root / object_id
        required = (
            object_dir / "slat" / "shape_gt_c64.npz",
            object_dir / "slat" / "texture_gt_c64.npz",
            object_dir / "views" / "views.json",
        )
        if all(path.is_file() for path in required):
            result.append(str(object_id))
    if len(result) < 2:
        raise RuntimeError("at least two complete objects are required")
    return result


def split_objects(
    object_ids: Sequence[str], seed: int, test_fraction: float
) -> tuple[list[str], list[str]]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0,1)")
    shuffled = list(object_ids)
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, int(math.ceil(len(shuffled) * test_fraction)))
    test_count = min(test_count, len(shuffled) - 1)
    return sorted(shuffled[test_count:]), sorted(shuffled[:test_count])


def owner_assignments(
    projected_uv: np.ndarray,
    valid_mask: np.ndarray,
    tile_indices: Sequence[np.ndarray],
    min_tokens: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Assign each projected global row to at most one executable tile."""
    boxes = tile_boxes()
    eligible_tiles = [
        tile_id for tile_id, rows in enumerate(tile_indices)
        if len(rows) >= min_tokens
    ]
    owner = np.full(len(projected_uv), -1, dtype=np.int16)
    best = np.full(len(projected_uv), np.inf, dtype=np.float64)
    for tile_id in eligible_tiles:
        rows = np.asarray(tile_indices[tile_id], dtype=np.int64)
        x0, y0, x1, y1 = boxes[tile_id]
        center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
        distance = np.square(projected_uv[rows] - center[None]).sum(axis=1)
        improve = distance < best[rows]
        # Strict comparison and ascending tile traversal provide deterministic
        # tile-index tie-breaking.
        changed = rows[improve]
        owner[changed] = tile_id
        best[changed] = distance[improve]
    owner[~np.asarray(valid_mask, dtype=np.bool_)] = -1
    owner_counts = np.asarray([(owner == tile_id).sum() for tile_id in range(49)])
    return owner, owner_counts, eligible_tiles


def _select_tiles(owner_counts: np.ndarray, eligible: Sequence[int], count: int) -> list[int]:
    ranked = sorted(
        (tile_id for tile_id in eligible if owner_counts[tile_id] > 0),
        key=lambda tile_id: (-int(owner_counts[tile_id]), tile_id),
    )
    return ranked[:count]


def build_items(
    dataset_root: Path,
    object_ids: Sequence[str],
    split: str,
    views_per_object: int,
    tiles_per_view: int,
    min_tokens: int,
) -> tuple[list[TileItem], dict[str, Any]]:
    items: list[TileItem] = []
    view_audits: list[dict[str, Any]] = []
    for object_id in object_ids:
        object_dir = dataset_root / object_id
        view_dirs = sorted((object_dir / "views").glob("view_*"))[:views_per_object]
        for view_dir in view_dirs:
            with np.load(view_dir / "tile_indices.npz") as data:
                indices = [data[f"tile_{tile_id:03d}"].astype(np.int64) for tile_id in range(49)]
                owner, owner_counts, eligible = owner_assignments(
                    data["projected_uv_4096"], data["valid_mask"], indices, min_tokens
                )
                chosen = _select_tiles(owner_counts, eligible, tiles_per_view)
                for tile_id in chosen:
                    rows = indices[tile_id]
                    owner_positions = np.flatnonzero(owner[rows] == tile_id)
                    if len(rows) < min_tokens or len(owner_positions) == 0:
                        raise RuntimeError("selected tile violates token/owner constraints")
                    items.append(TileItem(
                        split=split,
                        object_id=object_id,
                        view_name=view_dir.name,
                        tile_id=tile_id,
                        object_dir=str(object_dir),
                        view_dir=str(view_dir),
                        tile_rows=tuple(int(value) for value in rows),
                        owner_positions=tuple(int(value) for value in owner_positions),
                    ))
                claimed = owner >= 0
                view_audits.append({
                    "object_id": object_id,
                    "view": view_dir.name,
                    "eligible_tiles": len(eligible),
                    "chosen_tiles": chosen,
                    "claimed_global_rows": int(claimed.sum()),
                    "owner_count_sum": int(owner_counts.sum()),
                    "owner_unique": bool(int(owner_counts.sum()) == int(claimed.sum())),
                })
    if not items:
        raise RuntimeError(f"no {split} tile items passed the constraints")
    return items, {"split": split, "items": len(items), "views": view_audits}


def item_model_token_counts(items: Sequence[TileItem]) -> dict[str, int]:
    """Count G + H + L sparse tokens used by each three-way example."""
    global_tokens: dict[str, int] = {}
    result: dict[str, int] = {}
    for item in items:
        if item.object_id not in global_tokens:
            shape_path = Path(item.object_dir) / "slat" / "shape_gt_c64.npz"
            with np.load(shape_path) as shape:
                global_tokens[item.object_id] = int(len(shape["coords"]))
        result[item.key] = global_tokens[item.object_id] + 2 * len(item.tile_rows)
    return result


def _normalization_tensors(
    normalization: Mapping[str, Sequence[float]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.as_tensor(normalization["mean"], device=device, dtype=torch.float32)[None]
    std = torch.as_tensor(normalization["std"], device=device, dtype=torch.float32)[None]
    if torch.any(std == 0):
        raise ValueError("latent normalization has zero standard deviation")
    return mean, std


class LatentStore:
    def __init__(self, pipeline: Any, device: torch.device) -> None:
        self.pipeline = pipeline
        self.device = device
        self._cpu: dict[str, dict[str, torch.Tensor]] = {}

    def _load(self, item: TileItem) -> dict[str, torch.Tensor]:
        if item.object_id in self._cpu:
            return self._cpu[item.object_id]
        slat_dir = Path(item.object_dir) / "slat"
        with np.load(slat_dir / "shape_gt_c64.npz") as shape, np.load(
            slat_dir / "texture_gt_c64.npz"
        ) as texture:
            shape_coords = shape["coords"].astype(np.int32)
            texture_coords = texture["coords"].astype(np.int32)
            if not np.array_equal(shape_coords, texture_coords):
                raise RuntimeError(f"{item.object_id}: shape/texture support differs")
            batch = np.zeros((len(shape_coords), 1), dtype=np.int32)
            record = {
                "coords": torch.from_numpy(np.concatenate((batch, shape_coords), axis=1)),
                "shape_raw": torch.from_numpy(shape["feats"].astype(np.float32)),
                "texture_raw": torch.from_numpy(texture["feats"].astype(np.float32)),
            }
        with np.load(Path(item.view_dir) / "tile_indices.npz") as tile_data:
            if not np.array_equal(tile_data["global_coords_c64"], shape_coords):
                raise RuntimeError(f"{item.object_id}: cached tile rows use another support")
        self._cpu[item.object_id] = record
        return record

    def get(self, item: TileItem, branch: str) -> dict[str, torch.Tensor]:
        record = self._load(item)
        normalization = (
            self.pipeline.shape_slat_normalization
            if branch == "shape"
            else self.pipeline.tex_slat_normalization
        )
        mean, std = _normalization_tensors(normalization, self.device)
        raw = record[f"{branch}_raw"].to(self.device)
        output = {
            "coords": record["coords"].to(self.device),
            "x0": (raw - mean) / std,
        }
        if branch == "texture":
            shape_mean, shape_std = _normalization_tensors(
                self.pipeline.shape_slat_normalization, self.device
            )
            output["shape_cond"] = (
                record["shape_raw"].to(self.device) - shape_mean
            ) / shape_std
        return output


def camera_to_world_blender(camera: Mapping[str, Any]) -> torch.Tensor:
    """Convert cached OpenCV W2C to ProjGrid C2W, including SLat axis rotation."""
    extrinsic = np.asarray(camera["extrinsics_world_to_camera_opencv"], dtype=np.float64)
    if extrinsic.shape != (4, 4):
        raise ValueError("cached camera extrinsic must be 4x4")
    cv_to_blender_camera = np.diag([1.0, -1.0, -1.0, 1.0])
    slat_to_world = np.eye(4, dtype=np.float64)
    slat_to_world[:3, :3] = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    world_to_camera = cv_to_blender_camera @ extrinsic @ np.linalg.inv(slat_to_world)
    return torch.from_numpy(np.linalg.inv(world_to_camera).astype(np.float32))


def source_crop_box(
    camera: Mapping[str, Any], tile_box: Sequence[int] | None = None
) -> tuple[float, float, float, float]:
    preprocess = camera["canonical_preprocess"]
    source_width, source_height = (float(value) for value in preprocess["source_size"])
    left, top, right, bottom = (float(value) for value in preprocess["square_extent_source"])
    side_x, side_y = right - left, bottom - top
    if tile_box is None:
        raw_box = (left, top, right, bottom)
    else:
        x0, y0, x1, y1 = (float(value) for value in tile_box)
        raw_box = (
            left + x0 * side_x / CANONICAL_SIZE,
            top + y0 * side_y / CANONICAL_SIZE,
            left + x1 * side_x / CANONICAL_SIZE,
            top + y1 * side_y / CANONICAL_SIZE,
        )
    return (
        raw_box[0] / source_width,
        raw_box[1] / source_height,
        raw_box[2] / source_width,
        raw_box[3] / source_height,
    )


def condition_cache_path(cache_root: Path, branch: str, item: TileItem, kind: str) -> Path:
    base = cache_root / branch / item.object_id / item.view_name
    if kind == "global":
        return base / "global.pt"
    return base / f"tile_{item.tile_id:03d}_{kind}.pt"


def _pack_positive_condition(
    condition: Mapping[str, Mapping[str, Any]], coords: torch.Tensor
) -> dict[str, Any]:
    positive = condition["cond"]
    projection = positive["proj"]
    if not isinstance(projection, SparseTensor):
        raise TypeError("projected condition is not sparse")
    if not torch.equal(projection.coords, coords):
        raise RuntimeError("condition extraction changed sparse row order")
    return {
        "global": positive["global"].detach().cpu(),
        "proj": projection.feats.detach().cpu(),
        "coord_digest": _coords_digest(coords),
        "tokens": int(coords.shape[0]),
    }


@torch.inference_mode()
def prepare_conditions(
    pipeline: Any,
    branch: str,
    items: Sequence[TileItem],
    cache_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    image_model = (
        pipeline.image_cond_model_shape_1024
        if branch == "shape"
        else pipeline.image_cond_model_tex_1024
    )
    missing = []
    seen_global: set[tuple[str, str]] = set()
    for item in items:
        view_key = (item.object_id, item.view_name)
        if view_key not in seen_global:
            seen_global.add(view_key)
            if not condition_cache_path(cache_root, branch, item, "global").is_file():
                missing.append((item, "global"))
        for kind in ("hr", "lr"):
            if not condition_cache_path(cache_root, branch, item, kind).is_file():
                missing.append((item, kind))
    if not missing:
        return {"branch": branch, "computed": 0, "reused": len(seen_global) + 2 * len(items)}

    original_low_vram = bool(pipeline.low_vram)
    image_model.to(device)
    pipeline.low_vram = False
    computed = 0
    started = time.perf_counter()
    try:
        for index, (item, kind) in enumerate(missing, start=1):
            view_dir = Path(item.view_dir)
            camera = json.loads((view_dir / "camera.json").read_text(encoding="utf-8"))
            coords_np = np.load(view_dir / "tile_indices.npz")["global_coords_c64"].astype(np.int32)
            coords = torch.from_numpy(np.concatenate(
                (np.zeros((len(coords_np), 1), dtype=np.int32), coords_np), axis=1
            )).to(device)
            transform = camera_to_world_blender(camera).to(device)
            box = tile_boxes()[item.tile_id]
            if kind == "global":
                image = Image.open(view_dir / "image_1024.png").convert("RGB")
                query_coords = coords
                crop_box = source_crop_box(camera)
            else:
                rows = torch.as_tensor(item.tile_rows, device=device, dtype=torch.long)
                query_coords = coords.index_select(0, rows)
                crop_box = source_crop_box(camera, box)
                if kind == "hr":
                    with Image.open(view_dir / "image_4096.png") as source:
                        image = source.convert("RGB").crop(box)
                else:
                    lr_box = tuple(int(value // 4) for value in box)
                    with Image.open(view_dir / "image_1024.png") as source:
                        image = source.convert("RGB").crop(lr_box)
            condition = pipeline.get_proj_cond_shape(
                image_cond_model=image_model,
                image=[image],
                coords=query_coords,
                camera_angle_x=math.radians(float(camera["fov_degrees"])),
                distance=float(camera["radius"]),
                mesh_scale=1.0,
                grid_resolution_override=GRID_RESOLUTION,
                projection_crop_box=crop_box,
                transform_matrix=transform,
            )
            packed = _pack_positive_condition(condition, query_coords)
            packed.update({"kind": kind, "item": item.key, "branch": branch})
            _atomic_torch_save(condition_cache_path(cache_root, branch, item, kind), packed)
            computed += 1
            print(
                f"[condition:{branch}] {index}/{len(missing)} {item.key} {kind} "
                f"tokens={query_coords.shape[0]:,}",
                flush=True,
            )
            del condition, packed, query_coords, coords
            torch.cuda.empty_cache()
    finally:
        pipeline.low_vram = original_low_vram
        image_model.cpu()
        torch.cuda.empty_cache()
    return {
        "branch": branch,
        "computed": computed,
        "reused": len(seen_global) + 2 * len(items) - computed,
        "seconds": time.perf_counter() - started,
    }


class ConditionStore:
    def __init__(self, cache_root: Path, branch: str, device: torch.device) -> None:
        self.cache_root = cache_root
        self.branch = branch
        self.device = device
        self._cpu: dict[str, dict[str, Any]] = {}

    def get(self, item: TileItem, kind: str, coords: torch.Tensor) -> dict[str, Any]:
        path = condition_cache_path(self.cache_root, self.branch, item, kind)
        key = str(path)
        if key not in self._cpu:
            self._cpu[key] = torch.load(path, map_location="cpu", weights_only=True)
        packed = self._cpu[key]
        if packed["tokens"] != int(coords.shape[0]):
            raise RuntimeError(f"{path}: cached token count mismatch")
        if packed["coord_digest"] != _coords_digest(coords):
            raise RuntimeError(f"{path}: cached coordinate order mismatch")
        projection = packed["proj"].to(self.device)
        return {
            "global": packed["global"].to(self.device),
            "proj": SparseTensor(feats=projection, coords=coords),
        }


def _batch_sparse(
    feats: Sequence[torch.Tensor], coords: Sequence[torch.Tensor]
) -> SparseTensor:
    if not feats or len(feats) != len(coords):
        raise ValueError("batched sparse features/coordinates must be non-empty and aligned")
    return SparseTensor.from_tensor_list(list(feats), list(coords))


def _batch_conditions(conditions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine cached one-sample conditions into a real sparse mini-batch."""
    if not conditions:
        raise ValueError("cannot batch an empty condition list")
    global_values = [condition["global"] for condition in conditions]
    if all(
        isinstance(value, torch.Tensor)
        and value.ndim >= 1
        and value.shape[0] == 1
        and value.shape[1:] == global_values[0].shape[1:]
        for value in global_values
    ):
        global_batch: torch.Tensor | list[torch.Tensor] = torch.cat(
            global_values, dim=0
        )
    else:
        # SLatFlowModel converts a list of [L,C] tensors into VarLenTensor.
        global_batch = [
            value.squeeze(0)
            if isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[0] == 1
            else value
            for value in global_values
        ]
    projections = [condition["proj"] for condition in conditions]
    if not all(isinstance(value, SparseTensor) for value in projections):
        raise TypeError("all projected conditions must be SparseTensor values")
    return {
        "global": global_batch,
        "proj": _batch_sparse(
            [value.feats for value in projections],
            [value.coords for value in projections],
        ),
    }


@torch.no_grad()
def compute_velocity_batch(
    model: nn.Module,
    branch: str,
    items: Sequence[TileItem],
    times: Sequence[float],
    noise_seeds: Sequence[int],
    latent_store: LatentStore,
    condition_store: ConditionStore,
    device: torch.device,
) -> list[dict[str, torch.Tensor | float | int]]:
    """Run the three frozen flows as one sparse mini-batch."""
    if not items or len(items) != len(times) or len(items) != len(noise_seeds):
        raise ValueError("items, times, and noise seeds must be non-empty and aligned")

    samples: list[dict[str, Any]] = []
    batch_latents: dict[str, dict[str, torch.Tensor]] = {}
    for item, t, noise_seed in zip(items, times, noise_seeds):
        if item.object_id not in batch_latents:
            batch_latents[item.object_id] = latent_store.get(item, branch)
        latent = batch_latents[item.object_id]
        coords = latent["coords"]
        x0 = latent["x0"]
        generator = torch.Generator(device=device)
        generator.manual_seed(int(noise_seed))
        epsilon = torch.randn(
            x0.shape, generator=generator, device=device, dtype=torch.float32
        )
        x_t = (1.0 - float(t)) * x0 + (
            SIGMA_MIN + (1.0 - SIGMA_MIN) * float(t)
        ) * epsilon
        rows = torch.as_tensor(item.tile_rows, device=device, dtype=torch.long)
        owner_positions = torch.as_tensor(
            item.owner_positions, device=device, dtype=torch.long
        )
        tile_coords = coords.index_select(0, rows)
        tile_state_feats = x_t.index_select(0, rows)
        samples.append({
            "item": item,
            "t": float(t),
            "coords": coords,
            "x_t": x_t,
            "target": (1.0 - SIGMA_MIN) * epsilon - x0,
            "rows": rows,
            "owner_positions": owner_positions,
            "tile_coords": tile_coords,
            "tile_state_feats": tile_state_feats,
            "shape_cond": latent.get("shape_cond"),
            "state_gather_max_error": float(
                (tile_state_feats - x_t.index_select(0, rows)).abs().max().item()
            ),
        })

    global_state = _batch_sparse(
        [sample["x_t"] for sample in samples],
        [sample["coords"] for sample in samples],
    )
    global_conditions = _batch_conditions([
        condition_store.get(sample["item"], "global", sample["coords"])
        for sample in samples
    ])
    timestep = torch.as_tensor(
        [1000.0 * sample["t"] for sample in samples],
        device=device,
        dtype=torch.float32,
    )
    global_kwargs: dict[str, Any] = {}
    if branch == "texture":
        global_kwargs["concat_cond"] = _batch_sparse(
            [sample["shape_cond"] for sample in samples],
            [sample["coords"] for sample in samples],
        )
    g = model(global_state, timestep, global_conditions, **global_kwargs)
    if not torch.equal(g.coords, global_state.coords):
        raise RuntimeError("batched global flow changed support/order")
    g_parts, _ = g.to_tensor_list()
    del global_conditions, global_kwargs, global_state, g

    tile_state = _batch_sparse(
        [sample["tile_state_feats"] for sample in samples],
        [sample["tile_coords"] for sample in samples],
    )
    local_kwargs: dict[str, Any] = {}
    if branch == "texture":
        local_kwargs["concat_cond"] = _batch_sparse(
            [
                sample["shape_cond"].index_select(0, sample["rows"])
                for sample in samples
            ],
            [sample["tile_coords"] for sample in samples],
        )
    hr_conditions = _batch_conditions([
        condition_store.get(sample["item"], "hr", sample["tile_coords"])
        for sample in samples
    ])
    h = model(tile_state, timestep, hr_conditions, **local_kwargs)
    if not torch.equal(h.coords, tile_state.coords):
        raise RuntimeError("batched HR flow changed support/order")
    h_parts, _ = h.to_tensor_list()
    del hr_conditions, h

    lr_conditions = _batch_conditions([
        condition_store.get(sample["item"], "lr", sample["tile_coords"])
        for sample in samples
    ])
    l = model(tile_state, timestep, lr_conditions, **local_kwargs)
    if not torch.equal(l.coords, tile_state.coords):
        raise RuntimeError("batched LR flow changed support/order")
    l_parts, _ = l.to_tensor_list()

    result: list[dict[str, torch.Tensor | float | int]] = []
    for sample, g_part, h_part, l_part in zip(samples, g_parts, h_parts, l_parts):
        global_rows = sample["rows"].index_select(0, sample["owner_positions"])
        local_rows = sample["owner_positions"]
        result.append({
            "g": g_part.index_select(0, global_rows).float(),
            "h": h_part.index_select(0, local_rows).float(),
            "l": l_part.index_select(0, local_rows).float(),
            "target": sample["target"].index_select(0, global_rows).float(),
            "state_gather_max_error": sample["state_gather_max_error"],
            "owner_tokens": int(local_rows.numel()),
            "t": sample["t"],
        })
    return result


def compute_velocity_example(
    model: nn.Module,
    branch: str,
    item: TileItem,
    t: float,
    noise_seed: int,
    latent_store: LatentStore,
    condition_store: ConditionStore,
    device: torch.device,
) -> dict[str, torch.Tensor | float | int]:
    """Compatibility wrapper for callers that need a singleton example."""
    return compute_velocity_batch(
        model=model,
        branch=branch,
        items=[item],
        times=[t],
        noise_seeds=[noise_seed],
        latent_store=latent_store,
        condition_store=condition_store,
        device=device,
    )[0]


def _sample_logit_normal(generator: torch.Generator) -> float:
    return float(torch.sigmoid(torch.randn((), generator=generator)).item())


def _take_training_batch(
    order: list[TileItem],
    model_tokens: Mapping[str, int],
    max_examples: int,
    model_token_budget: int,
) -> list[TileItem]:
    """Pop a batch without crossing the configured model-token budget."""
    selected: list[TileItem] = []
    selected_tokens = 0
    while order and len(selected) < max_examples:
        candidate = order[-1]
        candidate_tokens = int(model_tokens[candidate.key])
        if selected and selected_tokens + candidate_tokens > model_token_budget:
            break
        selected.append(order.pop())
        selected_tokens += candidate_tokens
    if not selected:
        # An explicit token budget may be smaller than one large item.  Let the
        # singleton run (or produce the actionable singleton OOM below).
        selected.append(order.pop())
    return selected


def train_mapper(
    pipeline: Any,
    branch: str,
    train_items: Sequence[TileItem],
    cache_root: Path,
    output_dir: Path,
    device: torch.device,
    steps: int,
    learning_rate: float,
    seed: int,
    batch_size: int,
    batch_token_budget: int,
    max_auto_batch_size: int,
    batch_memory_fraction: float,
) -> tuple[LowRankVelocityMapper, list[dict[str, Any]], dict[str, Any]]:
    model_name = (
        "shape_slat_flow_model_1024" if branch == "shape" else "tex_slat_flow_model_1024"
    )
    model = pipeline.models[model_name].to(device).eval()
    model.requires_grad_(False)
    mapper = LowRankVelocityMapper(channels=32).to(device)
    zero_probe = mapper(torch.randn(7, 32, device=device), torch.randn(7, 32, device=device))
    if torch.count_nonzero(zero_probe).item() != 0:
        raise RuntimeError("mapper is not exactly identity-initialized")
    optimizer = torch.optim.Adam(mapper.parameters(), lr=learning_rate)
    latent_store = LatentStore(pipeline, device)
    condition_store = ConditionStore(cache_root, branch, device)
    model_tokens = item_model_token_counts(train_items)
    owner_tokens = {item.key: len(item.owner_positions) for item in train_items}
    torch.cuda.empty_cache()
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    batch_plan = design_batch_plan(
        [model_tokens[item.key] for item in train_items],
        [owner_tokens[item.key] for item in train_items],
        free_memory_bytes=free_memory,
        total_memory_bytes=total_memory,
        requested_batch_size=batch_size,
        requested_model_token_budget=batch_token_budget,
        max_auto_batch_size=max_auto_batch_size,
        memory_fraction=batch_memory_fraction,
    )
    print(
        f"[batch:{branch}] max_examples={batch_plan.max_examples} "
        f"model_token_budget={batch_plan.model_token_budget:,} "
        f"item_tokens(min/median/max)={batch_plan.item_model_tokens_min:,}/"
        f"{batch_plan.item_model_tokens_median:,}/{batch_plan.item_model_tokens_max:,} "
        f"free_vram={batch_plan.free_memory_bytes / 2**30:.2f}GiB "
        f"mode={'manual' if batch_size > 0 or batch_token_budget > 0 else 'auto'}",
        flush=True,
    )
    order_rng = random.Random(seed + (0 if branch == "shape" else 100_000))
    time_generator = torch.Generator(device="cpu")
    time_generator.manual_seed(seed + (11 if branch == "shape" else 100_011))
    order: list[TileItem] = []
    logs: list[dict[str, Any]] = []
    samples_seen = 0
    oom_reductions = 0
    effective_max_examples = batch_plan.max_examples
    effective_token_budget = batch_plan.model_token_budget
    mapper.train()
    for step in range(steps):
        if not order:
            order = list(train_items)
            order_rng.shuffle(order)
        batch_items = _take_training_batch(
            order,
            model_tokens,
            effective_max_examples,
            effective_token_budget,
        )
        times = [_sample_logit_normal(time_generator) for _ in batch_items]
        branch_seed_offset = 0 if branch == "shape" else 500_000
        noise_seeds = [
            seed * 1_000_003 + samples_seen + offset + branch_seed_offset
            for offset in range(len(batch_items))
        ]
        while True:
            try:
                batch_features = compute_velocity_batch(
                    model=model,
                    branch=branch,
                    items=batch_items,
                    times=times,
                    noise_seeds=noise_seeds,
                    latent_store=latent_store,
                    condition_store=condition_store,
                    device=device,
                )
                break
            except torch.OutOfMemoryError as error:
                gc.collect()
                torch.cuda.empty_cache()
                if len(batch_items) == 1:
                    item = batch_items[0]
                    raise RuntimeError(
                        f"CUDA OOM on singleton {item.key} with "
                        f"{model_tokens[item.key]:,} model tokens; lower "
                        "--min-tokens/tiles-per-view or use a larger GPU"
                    ) from error
                reduced_size = max(1, len(batch_items) // 2)
                deferred = batch_items[reduced_size:]
                # `order` is popped from the end, so reverse to preserve order.
                order.extend(reversed(deferred))
                batch_items = batch_items[:reduced_size]
                times = times[:reduced_size]
                noise_seeds = noise_seeds[:reduced_size]
                effective_max_examples = min(effective_max_examples, reduced_size)
                effective_token_budget = min(
                    effective_token_budget,
                    sum(model_tokens[item.key] for item in batch_items),
                )
                oom_reductions += 1
                print(
                    f"[batch:{branch}] CUDA OOM; retrying batch_size="
                    f"{len(batch_items)} model_tokens="
                    f"{sum(model_tokens[item.key] for item in batch_items):,}",
                    flush=True,
                )

        samples_seen += len(batch_items)
        g = torch.cat([features["g"] for features in batch_features], dim=0)
        h = torch.cat([features["h"] for features in batch_features], dim=0)
        l = torch.cat([features["l"] for features in batch_features], dim=0)
        target = torch.cat(
            [features["target"] for features in batch_features], dim=0
        )
        d = h - l
        baseline = F.mse_loss(g, target)
        direct = F.mse_loss(g + d, target)
        optimizer.zero_grad(set_to_none=True)
        prediction = g + mapper(g, d)
        loss = F.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()
        comparison = compare_loss(float(baseline.item()), float(loss.detach().item()))
        batch_owner_tokens = sum(
            int(features["owner_tokens"]) for features in batch_features
        )
        batch_model_tokens = sum(model_tokens[item.key] for item in batch_items)
        record = {
            "step": step + 1,
            "item": batch_items[0].key,
            "item_keys": "|".join(item.key for item in batch_items),
            "batch_size": len(batch_items),
            "t": float(np.mean(times)),
            "t_min": min(times),
            "t_max": max(times),
            "owner_tokens": batch_owner_tokens,
            "model_tokens": batch_model_tokens,
            "loss": float(loss.detach().item()),
            "baseline_mse": float(baseline.item()),
            "direct_mse": float(direct.item()),
            "loss_delta_vs_before": comparison["delta"],
            "loss_gain_percent_vs_before": comparison["gain_percent"],
            "loss_change_vs_before": comparison["status"],
            "state_gather_max_error": max(
                float(features["state_gather_max_error"])
                for features in batch_features
            ),
        }
        logs.append(record)
        print(
            f"[train:{branch}] {step + 1:04d}/{steps} batch={len(batch_items)} "
            f"owner_tokens={batch_owner_tokens:,} model_tokens={batch_model_tokens:,} "
            f"before(G)={record['baseline_mse']:.6f} "
            f"after(Phi)={record['loss']:.6f} "
            f"change={_format_loss_change(comparison)}",
            flush=True,
        )
    batch_report = {
        "initial_plan": asdict(batch_plan),
        "effective_max_examples": effective_max_examples,
        "effective_model_token_budget": effective_token_budget,
        "oom_reductions": oom_reductions,
        "optimizer_steps": steps,
        "examples_seen": samples_seen,
    }
    checkpoint = {
        "format": "pixal3d_hr_tile_velocity_mapper_v2",
        "branch": branch,
        "mapper": mapper.state_dict(),
        "channels": 32,
        "steps": steps,
        "learning_rate": learning_rate,
        "seed": seed,
        "batching": batch_report,
    }
    _atomic_torch_save(output_dir / branch / "mapper.pt", checkpoint)
    (output_dir / branch).mkdir(parents=True, exist_ok=True)
    with (output_dir / branch / "train_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(logs[0]))
        writer.writeheader()
        writer.writerows(logs)
    model.cpu()
    torch.cuda.empty_cache()
    return mapper, logs, batch_report


def _metric_sums() -> dict[str, float]:
    return {name: 0.0 for name in ("G", "H", "L", "direct", "Phi")}


@torch.inference_mode()
def evaluate_mapper(
    pipeline: Any,
    branch: str,
    mapper: LowRankVelocityMapper,
    test_items: Sequence[TileItem],
    cache_root: Path,
    device: torch.device,
    items_per_bin: int,
    seed: int,
) -> dict[str, Any]:
    model_name = (
        "shape_slat_flow_model_1024" if branch == "shape" else "tex_slat_flow_model_1024"
    )
    model = pipeline.models[model_name].to(device).eval()
    model.requires_grad_(False)
    mapper.eval()
    latent_store = LatentStore(pipeline, device)
    condition_store = ConditionStore(cache_root, branch, device)
    overall = _metric_sums()
    total_elements = 0
    examples: list[dict[str, Any]] = []
    by_bin: dict[str, dict[str, Any]] = {}
    for bin_index, t in enumerate(TIME_BIN_CENTERS):
        bin_sums = _metric_sums()
        bin_elements = 0
        selected = [
            test_items[(bin_index * items_per_bin + offset) % len(test_items)]
            for offset in range(items_per_bin)
        ]
        for offset, item in enumerate(selected):
            features = compute_velocity_example(
                model=model,
                branch=branch,
                item=item,
                t=t,
                noise_seed=seed * 2_000_003 + bin_index * 101 + offset
                + (0 if branch == "shape" else 700_000),
                latent_store=latent_store,
                condition_store=condition_store,
                device=device,
            )
            g = features["g"]
            h = features["h"]
            l = features["l"]
            target = features["target"]
            assert all(isinstance(value, torch.Tensor) for value in (g, h, l, target))
            d = h - l
            predictions = {
                "G": g,
                "H": h,
                "L": l,
                "direct": g + d,
                "Phi": g + mapper(g, d),
            }
            elements = int(target.numel())
            metrics = {
                name: float((prediction - target).square().sum().item() / elements)
                for name, prediction in predictions.items()
            }
            comparison = compare_loss(metrics["G"], metrics["Phi"])
            for name, value in metrics.items():
                weighted = value * elements
                overall[name] += weighted
                bin_sums[name] += weighted
            total_elements += elements
            bin_elements += elements
            examples.append({
                "item": item.key,
                "t": t,
                "owner_tokens": features["owner_tokens"],
                **{f"mse_{name}": value for name, value in metrics.items()},
                "phi_delta_vs_before": comparison["delta"],
                "phi_gain_percent_vs_before": comparison["gain_percent"],
                "phi_change_vs_before": comparison["status"],
            })
            print(
                f"[test:{branch}] bin={bin_index} t={t:.1f} {item.key} "
                f"before(G)={metrics['G']:.6f} after(Phi)={metrics['Phi']:.6f} "
                f"change={_format_loss_change(comparison)}",
                flush=True,
            )
        bin_metrics = {name: value / bin_elements for name, value in bin_sums.items()}
        bin_comparison = compare_loss(bin_metrics["G"], bin_metrics["Phi"])
        by_bin[str(bin_index)] = {
            "range": [bin_index / 5.0, (bin_index + 1) / 5.0],
            "center": t,
            "elements": bin_elements,
            "metrics": bin_metrics,
            "phi_improvement_percent": bin_comparison["gain_percent"],
            "phi_change_vs_before": bin_comparison,
        }
    metrics = {name: value / total_elements for name, value in overall.items()}
    phi_comparison = compare_loss(metrics["G"], metrics["Phi"])
    direct_comparison = compare_loss(metrics["G"], metrics["direct"])
    result = {
        "branch": branch,
        "test_object_ids": sorted({item.object_id for item in test_items}),
        "examples": len(examples),
        "elements": total_elements,
        "metrics": metrics,
        "phi_improvement_percent": phi_comparison["gain_percent"],
        "phi_change_vs_before": phi_comparison,
        "direct_improvement_percent": direct_comparison["gain_percent"],
        "direct_change_vs_before": direct_comparison,
        "success": bool(metrics["Phi"] < metrics["G"]),
        "time_bins": by_bin,
        "per_example": examples,
    }
    model.cpu()
    torch.cuda.empty_cache()
    return result


def alignment_audit(items: Sequence[TileItem]) -> dict[str, Any]:
    image_errors = []
    coordinate_errors = []
    for item in items:
        view_dir = Path(item.view_dir)
        box = tile_boxes()[item.tile_id]
        with Image.open(view_dir / "image_4096.png") as hr_source, Image.open(
            view_dir / "image_1024.png"
        ) as lr_source:
            hr = hr_source.convert("RGB").crop(box).resize(
                (TILE_SIZE // 4, TILE_SIZE // 4), Image.Resampling.LANCZOS
            )
            lr_box = tuple(int(value // 4) for value in box)
            lr = lr_source.convert("RGB").crop(lr_box)
            difference = np.abs(
                np.asarray(hr, dtype=np.int16) - np.asarray(lr, dtype=np.int16)
            )
            image_errors.append((float(difference.mean()), int(difference.max())))
        camera = json.loads((view_dir / "camera.json").read_text(encoding="utf-8"))
        with np.load(view_dir / "tile_indices.npz") as data:
            uv = data["projected_uv_4096"][np.asarray(item.tile_rows, dtype=np.int64)]
        preprocess = camera["canonical_preprocess"]
        left, top, right, bottom = (float(v) for v in preprocess["square_extent_source"])
        side = right - left
        raw_uv = np.empty_like(uv, dtype=np.float64)
        raw_uv[:, 0] = left + uv[:, 0] * side / CANONICAL_SIZE
        raw_uv[:, 1] = top + uv[:, 1] * side / CANONICAL_SIZE
        x0, y0, _, _ = box
        raw_start = np.array([
            left + x0 * side / CANONICAL_SIZE,
            top + y0 * side / CANONICAL_SIZE,
        ])
        raw_extent = side * TILE_SIZE / CANONICAL_SIZE
        via_nested_crop = (raw_uv - raw_start[None]) * TILE_SIZE / raw_extent
        direct = uv - np.array([x0, y0], dtype=np.float64)[None]
        coordinate_errors.append(float(np.abs(via_nested_crop - direct).max()))
    return {
        "items_checked": len(items),
        "hr_downsample_vs_lr_crop_mean_abs_pixel": float(np.mean([v[0] for v in image_errors])),
        "hr_downsample_vs_lr_crop_max_abs_pixel": int(max(v[1] for v in image_errors)),
        "global_to_nested_tile_vs_direct_crop_max_pixel_error": max(coordinate_errors),
        "note": "Lanczos boundary support can make crop-then-resize differ slightly from resize-then-crop; coordinate mapping is audited separately.",
    }


def save_metric_plot(results: Mapping[str, Any], output_path: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    names = ["G", "H", "L", "direct", "Phi"]
    branches = list(results)
    width = 0.36
    x = np.arange(len(names))
    figure, (axis, delta_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    for index, branch in enumerate(branches):
        values = [results[branch]["metrics"][name] for name in names]
        offsets = x + (index - (len(branches) - 1) / 2) * width
        axis.bar(offsets, values, width, label=branch)
        baseline = float(results[branch]["metrics"]["G"])
        relative = [100.0 * (value / baseline - 1.0) for value in values]
        delta_axis.bar(offsets, relative, width, label=branch)
    axis.set_xticks(x, names)
    axis.set_ylabel("Test flow MSE (lower is better)")
    axis.set_yscale("log")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    delta_axis.axhline(0.0, color="black", linewidth=0.8)
    delta_axis.set_xticks(x, names)
    delta_axis.set_ylabel("MSE change relative to G (%)")
    delta_axis.set_title("Negative means improvement")
    delta_axis.grid(axis="y", alpha=0.25)
    delta_axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("assets/small_glb_dataset_full"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hr_tile_velocity_mapper_small"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--branches", type=str, default="texture")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--views-per-object", type=int, default=999)
    parser.add_argument("--tiles-per-view", type=int, default=10)
    parser.add_argument("--min-tokens", type=int, default=1000)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--batch-size", type=int, default=0,
        help="maximum examples per optimizer step; 0 selects it from GPU/token stats",
    )
    parser.add_argument(
        "--batch-token-budget", type=int, default=0,
        help="maximum G+H+L sparse tokens per batch; 0 derives it from free VRAM",
    )
    parser.add_argument(
        "--max-auto-batch-size", type=int, default=8,
        help="safety cap for automatically selected example batch size",
    )
    parser.add_argument(
        "--batch-memory-fraction", type=float, default=0.65,
        help="fraction of currently free VRAM available to the automatic token budget",
    )
    parser.add_argument("--eval-items-per-bin", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.views_per_object,
        args.tiles_per_view,
        args.min_tokens,
        args.train_steps,
        args.eval_items_per_bin,
    ) <= 0:
        raise SystemExit("view/tile/token/step/eval counts must be positive")
    branches = [value.strip() for value in args.branches.split(",") if value.strip()]
    if not branches or any(branch not in ("shape", "texture") for branch in branches):
        raise SystemExit("--branches must contain shape and/or texture")
    if args.batch_size < 0 or args.batch_token_budget < 0:
        raise SystemExit("--batch-size and --batch-token-budget cannot be negative")
    if args.max_auto_batch_size <= 0:
        raise SystemExit("--max-auto-batch-size must be positive")
    if not 0.0 < args.batch_memory_fraction < 1.0:
        raise SystemExit("--batch-memory-fraction must be in (0,1)")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.cuda_device)
    if torch.cuda.current_device() != args.cuda_device:
        raise RuntimeError("failed to select the requested CUDA device")
    device = torch.device("cuda")
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    cache_root = (
        args.cache_dir.resolve()
        if args.cache_dir is not None
        else output_dir / "condition_cache"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    object_ids = discover_complete_objects(dataset_root)
    train_ids, test_ids = split_objects(object_ids, args.seed, args.test_fraction)
    train_items, train_audit = build_items(
        dataset_root, train_ids, "train", args.views_per_object,
        args.tiles_per_view, args.min_tokens,
    )
    test_items, test_audit = build_items(
        dataset_root, test_ids, "test", args.views_per_object,
        args.tiles_per_view, args.min_tokens,
    )
    split_report = {
        "seed": args.seed,
        "policy": "object-disjoint deterministic shuffle",
        "train_objects": train_ids,
        "test_objects": test_ids,
        "train_items": [asdict(item) for item in train_items],
        "test_items": [asdict(item) for item in test_items],
    }
    _json_dump(output_dir / "split.json", split_report)
    data_audit = {
        "train": train_audit,
        "test": test_audit,
        "alignment": alignment_audit(train_items[:2] + test_items[:2]),
    }
    _json_dump(output_dir / "data_audit.json", data_audit)
    print(
        f"[cuda] physical={torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())}\n"
        f"[split] train={len(train_ids)} objects/{len(train_items)} tiles "
        f"test={len(test_ids)} objects/{len(test_items)} tiles",
        flush=True,
    )

    pipeline = init_pipeline(args.model_path, device="cuda", low_vram=False)
    condition_reports: dict[str, Any] = {}
    batch_reports: dict[str, Any] = {}
    results: dict[str, Any] = {}
    max_gather_error = 0.0
    for branch in branches:
        condition_reports[branch] = prepare_conditions(
            pipeline, branch, train_items + test_items, cache_root, device
        )
        mapper, train_logs, batch_report = train_mapper(
            pipeline=pipeline,
            branch=branch,
            train_items=train_items,
            cache_root=cache_root,
            output_dir=output_dir,
            device=device,
            steps=args.train_steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            batch_size=args.batch_size,
            batch_token_budget=args.batch_token_budget,
            max_auto_batch_size=args.max_auto_batch_size,
            batch_memory_fraction=args.batch_memory_fraction,
        )
        batch_reports[branch] = batch_report
        max_gather_error = max(
            max_gather_error,
            max(float(record["state_gather_max_error"]) for record in train_logs),
        )
        result = evaluate_mapper(
            pipeline=pipeline,
            branch=branch,
            mapper=mapper,
            test_items=test_items,
            cache_root=cache_root,
            device=device,
            items_per_bin=args.eval_items_per_bin,
            seed=args.seed,
        )
        results[branch] = result
        _json_dump(output_dir / branch / "test_metrics.json", result)
        del mapper
        torch.cuda.empty_cache()
    plot = save_metric_plot(results, output_dir / "test_flow_mse.png")
    summary = {
        "format": "pixal3d_hr_tile_velocity_mapper_small_v2",
        "cuda_device": args.cuda_device,
        "cuda_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "model_path": args.model_path,
        "arguments": vars(args) | {
            "dataset_root": str(args.dataset_root),
            "output_dir": str(args.output_dir),
            "cache_dir": None if args.cache_dir is None else str(args.cache_dir),
        },
        "split": {"train": train_ids, "test": test_ids},
        "invariants": {
            "support_changed": False,
            "local_requantization": False,
            "global_position_coords_used_for_all_flows": True,
            "shared_global_noise_then_gather": True,
            "max_state_gather_error": max_gather_error,
            "owner_write_multiplicity_max": 1,
            "mapper_zero_initialized": True,
            "loss": "owner-only standard flow MSE, token-weighted across each mini-batch",
            "loss_before_reference": "frozen unmodified global prediction G",
        },
        "condition_cache": condition_reports,
        "batching": batch_reports,
        "results": results,
        "plot": plot,
        "seconds": time.perf_counter() - started,
    }
    _json_dump(output_dir / "summary.json", summary)
    print("\n[result]", flush=True)
    for branch, result in results.items():
        comparison = result["phi_change_vs_before"]
        print(
            f"  {branch}: before(G)={result['metrics']['G']:.8f} "
            f"after(Phi)={result['metrics']['Phi']:.8f} "
            f"change={_format_loss_change(comparison)} "
            f"success={result['success']}",
            flush=True,
        )
    print(f"[saved] {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
