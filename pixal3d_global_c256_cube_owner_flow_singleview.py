#!/usr/bin/env python3
"""Global C256, local C64 cube, hard-owner or Gaussian Shape/Texture flow.

The global sparse row table is the only state.  Cubes are integer translations
of that table. Each Euler step either selects the fixed nearest-centre owner
velocity or blends all containing-cube velocities with normalized 3-D Gaussian
weights based on global cell-centre to cube-centre distance.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH", str(Path(__file__).with_name("autotune_cache.json")))

import numpy as np
import o_voxel
import torch
from PIL import Image

import pixal3d_global4096_singleview_shared_slat_shape_tex_sr as expc
import pixal3d_global4096_tile_endpoint_rollout_sync as legacy
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as camera_core
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor

FORMAT = "pixal3d_global_c256_cube_owner_flow_singleview_v1"
GRID = 256
CUBE = 64
STRIDE = 32
STARTS = tuple(range(0, GRID - CUBE + 1, STRIDE))
CANONICAL_IMAGE_SIZE = 4096
PROJECTION_IMAGE_SIZE = 1024
DINO_PATCH_SIZE = 16
LOCAL_CONDITION_FORMAT = "pixal3d_global_c256_cube_local_condition_v2"
VOXELIZER = {
    "function": "o_voxel.convert.mesh_to_flexible_dual_grid",
    "grid_size": 4096,
    "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    "face_weight": 1.0,
    "boundary_weight": 0.2,
    "regularization_weight": 1e-2,
}


def _jsonable(x: Any) -> Any:
    if isinstance(x, Mapping): return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [_jsonable(v) for v in x]
    if isinstance(x, Path): return str(x)
    if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp); os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(chunk): h.update(data)
    return h.hexdigest()


def tensor_sha256(x: torch.Tensor) -> str:
    y = x.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(y.dtype).encode()); h.update(np.asarray(y.shape, dtype=np.int64).tobytes())
    h.update(y.numpy().tobytes(order="C"))
    return h.hexdigest()


def linear_keys(xyz: torch.Tensor, resolution: int = GRID) -> torch.Tensor:
    q = xyz.to(torch.int64)
    return (q[:, 0] * resolution + q[:, 1]) * resolution + q[:, 2]


def cube_layout(grid: int = GRID, cube: int = CUBE, stride: int = STRIDE) -> list[dict[str, Any]]:
    if (grid, cube, stride) != (GRID, CUBE, STRIDE):
        starts = tuple(range(0, grid - cube + 1, stride))
        if not starts or starts[-1] != grid - cube: raise ValueError("layout does not land on final edge")
    else: starts = STARTS
    return [{"cube_id": i, "start": (sx, sy, sz)}
            for i, (sx, sy, sz) in enumerate((a, b, c) for a in starts for b in starts for c in starts)]


def physical_boundary_q(xyz_boundary: torch.Tensor, resolution: int = GRID) -> torch.Tensor:
    """Map lattice cell-boundary coordinates to the physical ``[-1, 1]`` box."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    return 2.0 * xyz_boundary.to(torch.float64) / float(resolution) - 1.0


def align_projected_crop_box(
    projected_xy: torch.Tensor,
    image_width: int = CANONICAL_IMAGE_SIZE,
    image_height: int = CANONICAL_IMAGE_SIZE,
    multiple: int = DINO_PATCH_SIZE,
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    """Return an in-bounds pixel-edge bbox expanded outwards to ``multiple``.

    ``projected_xy`` is expressed in canonical pixel-edge coordinates.  The
    returned ``(x0, y0, x1, y1)`` follows PIL's right/bottom-exclusive crop
    convention.  Clipping precedes outward alignment, so a partially off-frame
    cube uses every available source pixel and both crop dimensions remain
    divisible by the DINO patch size.
    """
    if projected_xy.ndim != 2 or projected_xy.shape[1] != 2 or not projected_xy.numel():
        raise ValueError("projected_xy must be nonempty [N,2]")
    if image_width <= 0 or image_height <= 0 or multiple <= 0:
        raise ValueError("image dimensions and crop multiple must be positive")
    if image_width % multiple or image_height % multiple:
        raise ValueError("canonical image dimensions must be divisible by crop multiple")
    points = projected_xy.detach().cpu().to(torch.float64)
    if not torch.isfinite(points).all():
        raise ValueError("projected_xy contains NaN/Inf")
    raw_lo = points.amin(0)
    raw_hi = points.amax(0)
    clipped_lo = torch.maximum(raw_lo, torch.zeros(2, dtype=torch.float64))
    clipped_hi = torch.minimum(
        raw_hi,
        torch.tensor([float(image_width), float(image_height)], dtype=torch.float64),
    )
    if bool((clipped_hi <= clipped_lo).any()):
        raise RuntimeError(
            "projected cube does not intersect the canonical condition image: "
            f"raw_lo={raw_lo.tolist()} raw_hi={raw_hi.tolist()}"
        )

    x0 = max(0, int(math.floor(float(clipped_lo[0]))) // multiple * multiple)
    y0 = max(0, int(math.floor(float(clipped_lo[1]))) // multiple * multiple)
    x1_unaligned = int(math.ceil(float(clipped_hi[0])))
    y1_unaligned = int(math.ceil(float(clipped_hi[1])))
    x1 = min(image_width, ((x1_unaligned + multiple - 1) // multiple) * multiple)
    y1 = min(image_height, ((y1_unaligned + multiple - 1) // multiple) * multiple)
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("aligned projected crop is empty")
    if (x1 - x0) % multiple or (y1 - y0) % multiple:
        raise RuntimeError("aligned projected crop is not patch divisible")
    box = (x0, y0, x1, y1)
    return box, {
        "raw_bbox_pixel_edges_4096": [
            float(raw_lo[0]), float(raw_lo[1]), float(raw_hi[0]), float(raw_hi[1])
        ],
        "clipped_bbox_pixel_edges_4096": [
            float(clipped_lo[0]), float(clipped_lo[1]),
            float(clipped_hi[0]), float(clipped_hi[1]),
        ],
        "crop_box_4096": list(box),
        "crop_size": [x1 - x0, y1 - y0],
        "partially_outside_image": bool(
            (raw_lo[0] < 0) or (raw_lo[1] < 0)
            or (raw_hi[0] > image_width) or (raw_hi[1] > image_height)
        ),
    }


def cube_projection_crop(
    start256: Sequence[int],
    camera: Mapping[str, Any],
    image_width: int = CANONICAL_IMAGE_SIZE,
    image_height: int = CANONICAL_IMAGE_SIZE,
    multiple: int = DINO_PATCH_SIZE,
) -> dict[str, Any]:
    """Project a physical C4096 1024³ cube and build its local image crop.

    Projection uses the exact centered global camera path used by
    ``ProjGrid``.  The model normalizes a 1024 projection with ``+0.5`` pixel
    centres before mapping it into a crop; scaling those normalized pixel-edge
    coordinates to 4096 avoids a two-pixel convention drift.
    """
    if len(start256) != 3:
        raise ValueError("start256 must contain three coordinates")
    start = torch.as_tensor(start256, dtype=torch.float64)
    if bool(((start < 0) | (start + CUBE > GRID)).any()):
        raise ValueError(f"cube start outside C{GRID}: {tuple(start256)}")
    corner_bits = torch.tensor(
        [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)],
        dtype=torch.float64,
    )
    corners_q = physical_boundary_q(start[None] + corner_bits * CUBE)
    uv_projection, depth, finite = camera_core._project_global_q_to_image(
        corners_q,
        global_camera=camera,
        image_width=PROJECTION_IMAGE_SIZE,
        image_height=PROJECTION_IMAGE_SIZE,
    )
    if not bool(finite.all()):
        raise RuntimeError(
            f"cube {tuple(int(v) for v in start256)} crosses/behind the camera plane"
        )
    scale = torch.tensor(
        [
            float(image_width) / PROJECTION_IMAGE_SIZE,
            float(image_height) / PROJECTION_IMAGE_SIZE,
        ],
        dtype=torch.float64,
    )
    projected_pixel_edges = (uv_projection.to(torch.float64) + 0.5) * scale[None]
    box, diagnostics = align_projected_crop_box(
        projected_pixel_edges,
        image_width=image_width,
        image_height=image_height,
        multiple=multiple,
    )
    x0, y0, x1, y1 = box
    return {
        **diagnostics,
        "projection_crop_box": (
            x0 / float(image_width), y0 / float(image_height),
            x1 / float(image_width), y1 / float(image_height),
        ),
        "cube_corners_q": corners_q.tolist(),
        "cube_corner_depth": depth.detach().cpu().tolist(),
        "projection_convention": (
            "global camera C1024 pixel centres -> normalized pixel edges -> canonical C4096"
        ),
    }


def attach_cube_projection_crops(
    records: Sequence[Mapping[str, Any]],
    camera: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach deterministic C4096 crop metadata to all 343 cube records."""
    crop_rows: list[dict[str, Any]] = []
    for rec in records:
        crop = cube_projection_crop(rec["start"], camera)
        rec["condition_crop"] = crop
        crop_rows.append({
            "cube_id": int(rec["cube_id"]),
            "start": list(rec["start"]),
            "membership_tokens": int(rec["global_row_ids"].numel()),
            "owned_tokens": int(rec.get("owned_row_ids", torch.empty(0)).numel()),
            **crop,
        })
    sizes = torch.tensor([row["crop_size"] for row in crop_rows], dtype=torch.float64)
    return {
        "format": LOCAL_CONDITION_FORMAT,
        "source_grid": [CANONICAL_IMAGE_SIZE, CANONICAL_IMAGE_SIZE],
        "projection_grid": [PROJECTION_IMAGE_SIZE, PROJECTION_IMAGE_SIZE],
        "crop_alignment": DINO_PATCH_SIZE,
        "cube_count": len(crop_rows),
        "crop_width": _distribution(sizes[:, 0]),
        "crop_height": _distribution(sizes[:, 1]),
        "partially_outside_cubes": sum(row["partially_outside_image"] for row in crop_rows),
        "cubes": crop_rows,
    }


def center_translate_scale_to_local_c64(global_xyz: torch.Tensor, start256: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the physical C4096-cube centre transform, then quantize C64.

    Global C256 cell centres are first expressed in canonical [-1,1].  A
    C4096 cube has width 1024, i.e. one quarter of the global extent, so after
    subtracting its centre we multiply coordinates by four.  Converting the
    result to C64 cell indices is algebraically identical to ``xyz-start``;
    both are returned and checked to prevent accidentally feeding global
    coordinates to the local flow model.
    """
    xyz = global_xyz.to(torch.float64)
    start = torch.as_tensor(start256, dtype=torch.float64, device=xyz.device)
    global_q = 2.0 * (xyz + 0.5) / 256.0 - 1.0
    cube_center_q = 2.0 * (start + 32.0) / 256.0 - 1.0
    local_q = (global_q - cube_center_q) * 4.0
    local_continuous = (local_q + 1.0) * 64.0 / 2.0 - 0.5
    local_xyz = torch.round(local_continuous).to(torch.int32)
    exact = global_xyz.to(torch.int32) - torch.as_tensor(start256, dtype=torch.int32, device=global_xyz.device)
    if not torch.equal(local_xyz, exact): raise RuntimeError("centre-translate/scale is not equal to exact C64 relabel")
    if local_continuous.numel() and float((local_continuous - local_xyz).abs().max()) > 1e-12:
        raise RuntimeError("centre-translate/scale produced non-integral local coordinates")
    return local_xyz, local_q


def build_cube_records(global_coords: torch.Tensor, grid: int = GRID, cube: int = CUBE,
                       stride: int = STRIDE) -> tuple[list[dict[str, Any]], torch.Tensor]:
    if global_coords.ndim != 2 or global_coords.shape[1] != 4: raise ValueError("global_coords must be [N,4]")
    xyz = global_coords[:, 1:].cpu().to(torch.int32)
    records: list[dict[str, Any]] = []
    coverage = torch.zeros(xyz.shape[0], dtype=torch.int16)
    for item in cube_layout(grid, cube, stride):
        start = torch.tensor(item["start"], dtype=torch.int32)
        # Select by the requested physical C4096 1024/512 cube.  Multiplying
        # C256 integer coordinates and starts by 16 is exact.
        xyz4096 = xyz.to(torch.int64) * 16
        start4096 = start.to(torch.int64) * 16
        inside = ((xyz4096 >= start4096) & (xyz4096 < start4096 + 1024)).all(1)
        ids = torch.where(inside)[0].to(torch.int64)
        local, local_q = center_translate_scale_to_local_c64(xyz.index_select(0, ids), item["start"])
        if ids.numel():
            if not torch.equal(local + start, xyz.index_select(0, ids)): raise RuntimeError("integer roundtrip failed")
            if torch.unique(linear_keys(local, cube)).numel() != ids.numel(): raise RuntimeError("local coords not unique")
        coverage.index_add_(0, ids, torch.ones_like(ids, dtype=torch.int16))
        records.append({**item, "start4096": tuple(int(v) * 16 for v in item["start"]),
                        "global_row_ids": ids, "local_xyz": local,
                        "coordinate_transform": "subtract C4096 cube centre, scale x4, relabel C64"})
    if len(records) != 343: raise RuntimeError(f"expected 343 cubes, got {len(records)}")
    if bool(((coverage < 1) | (coverage > 8)).any()): raise RuntimeError("coverage outside [1,8]")
    return records, coverage


def build_owner_map(global_coords: torch.Tensor, records: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, dict[str, Any]]:
    xyz = global_coords[:, 1:].cpu().to(torch.float64)
    n = xyz.shape[0]
    owner = torch.full((n,), -1, dtype=torch.int16)
    best = torch.full((n,), float("inf"), dtype=torch.float64)
    tie_rows = torch.zeros(n, dtype=torch.bool)
    for rec in records:
        ids = rec["global_row_ids"].to(torch.int64)
        if not ids.numel(): continue
        center = torch.tensor(rec["start"], dtype=torch.float64) + 32.0
        d2 = ((xyz.index_select(0, ids) + 0.5 - center) ** 2).sum(1)
        old = best.index_select(0, ids)
        equal = d2 == old
        tie_rows[ids[equal]] = True
        better = d2 < old  # iteration is cube_id order, so equality keeps lowest ID
        chosen = ids[better]
        best[chosen] = d2[better]
        owner[chosen] = int(rec["cube_id"])
    if bool((owner < 0).any()): raise RuntimeError("row without owner")
    owned_hist: dict[str, int] = {}
    margins = []
    for rec in records:
        ids = rec["global_row_ids"].to(torch.int64)
        owned = ids[owner.index_select(0, ids) == int(rec["cube_id"])]
        rec["owned_row_ids"] = owned
        owned_hist[str(rec["cube_id"])] = int(owned.numel())
        if owned.numel():
            local = global_coords[owned, 1:].cpu().float() - torch.tensor(rec["start"]).float()
            margins.append(torch.minimum(local + .5, 64.0 - (local + .5)).amin(1))
    margin = torch.cat(margins) if margins else torch.empty(0)
    return owner, {"owner_histogram": owned_hist, "tie_row_count": int(tie_rows.sum()),
                   "owner_face_margin_histogram": histogram(margin.tolist())}


def histogram(values: Iterable[Any]) -> dict[str, int]:
    c = Counter(values); return {str(k): int(c[k]) for k in sorted(c)}


def validate_owner_scatter(owner: torch.Tensor, proposals: Sequence[tuple[int, torch.Tensor, torch.Tensor]],
                           channels: int) -> torch.Tensor:
    out = torch.empty((owner.numel(), channels), dtype=torch.float32)
    writes = torch.zeros(owner.numel(), dtype=torch.int16)
    for cube_id, row_ids, velocity in proposals:
        mask = owner.index_select(0, row_ids) == int(cube_id)
        ids = row_ids[mask]
        vals = velocity[mask].float()
        out.index_copy_(0, ids, vals)
        writes.index_add_(0, ids, torch.ones_like(ids, dtype=torch.int16))
    if not torch.all(writes == 1):
        raise RuntimeError(f"owner write count must equal one; histogram={histogram(writes.tolist())}")
    if not torch.isfinite(out).all(): raise FloatingPointError("non-finite owner velocity")
    return out


def build_gaussian_weight_table(global_coords: torch.Tensor, records: Sequence[Mapping[str, Any]],
                                sigma: float) -> dict[int, torch.Tensor]:
    """Return per-incidence normalized weights for 3-D cube velocity fusion."""
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("gaussian sigma must be finite and positive")
    xyz = global_coords[:, 1:].cpu().to(torch.float64) + 0.5
    weight_sum = torch.zeros(global_coords.shape[0], dtype=torch.float64)
    raw: dict[int, torch.Tensor] = {}
    for rec in records:
        ids = rec["global_row_ids"].cpu().long()
        if not ids.numel():
            continue
        center = torch.as_tensor(rec["start"], dtype=torch.float64) + 32.0
        d2 = ((xyz.index_select(0, ids) - center) ** 2).sum(1)
        weight = torch.exp(-d2 / (2.0 * float(sigma) ** 2))
        if not torch.isfinite(weight).all() or bool((weight <= 0).any()):
            raise FloatingPointError("invalid Gaussian incidence weight")
        cube_id = int(rec["cube_id"])
        raw[cube_id] = weight
        weight_sum.index_add_(0, ids, weight)
    if not torch.isfinite(weight_sum).all() or bool((weight_sum <= 0).any()):
        raise RuntimeError("every global row must have positive Gaussian weight sum")
    normalized: dict[int, torch.Tensor] = {}
    normalized_sum = torch.zeros_like(weight_sum)
    for rec in records:
        cube_id = int(rec["cube_id"])
        if cube_id not in raw:
            continue
        ids = rec["global_row_ids"].cpu().long()
        weight = (raw[cube_id] / weight_sum.index_select(0, ids)).float().contiguous()
        normalized[cube_id] = weight
        normalized_sum.index_add_(0, ids, weight.double())
    if not torch.allclose(normalized_sum, torch.ones_like(normalized_sum), atol=2e-7, rtol=0):
        raise RuntimeError("normalized Gaussian weights do not sum to one per row")
    return normalized


def validate_gaussian_fusion(global_rows: int,
                             proposals: Sequence[tuple[int, torch.Tensor, torch.Tensor]],
                             weights: Mapping[int, torch.Tensor], channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse exactly one proposal from every nonempty cube using fixed weights."""
    expected = set(weights)
    actual = [int(cube_id) for cube_id, _, _ in proposals]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError(f"Gaussian proposals must contain each nonempty cube exactly once; "
                           f"expected={len(expected)} actual={len(actual)} unique={len(set(actual))}")
    out = torch.zeros((global_rows, channels), dtype=torch.float32)
    weight_sum = torch.zeros(global_rows, dtype=torch.float32)
    for cube_id, row_ids, velocity in proposals:
        ids = row_ids.cpu().long()
        value = velocity.cpu().float()
        weight = weights[int(cube_id)]
        if value.shape != (ids.numel(), channels) or weight.shape != (ids.numel(),):
            raise RuntimeError(f"cube {cube_id} Gaussian proposal/weight shape mismatch")
        out.index_add_(0, ids, value * weight[:, None])
        weight_sum.index_add_(0, ids, weight)
    if not torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=2e-6, rtol=0):
        raise RuntimeError("Gaussian fused weight sum must equal one for every row")
    if not torch.isfinite(out).all():
        raise FloatingPointError("non-finite Gaussian fused velocity")
    return out, weight_sum


def jacobi_update(state: torch.Tensor, owner_velocity: torch.Tensor, t: float, t_next: float) -> torch.Tensor:
    if state.shape != owner_velocity.shape: raise ValueError("state/velocity shape mismatch")
    return state - float(t - t_next) * owner_velocity


def pack_groups(records: Sequence[Mapping[str, Any]], flow_batch_size: int,
                max_batch_tokens: int, require_owned: bool = True) -> list[list[Mapping[str, Any]]]:
    if flow_batch_size <= 0 or max_batch_tokens <= 0: raise ValueError("batch limits must be positive")
    groups: list[list[Mapping[str, Any]]] = []; current = []; tokens = 0
    for rec in records:
        n = int(rec["global_row_ids"].numel())
        if n == 0 or (require_owned and int(rec.get("owned_row_ids", torch.empty(0)).numel()) == 0): continue
        if n > max_batch_tokens: raise RuntimeError(f"cube {rec['cube_id']} exceeds max-batch-tokens")
        if current and (len(current) >= flow_batch_size or tokens + n > max_batch_tokens):
            groups.append(current); current = []; tokens = 0
        current.append(rec); tokens += n
    if current: groups.append(current)
    return groups


def _empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def _distribution(values: Sequence[int] | torch.Tensor) -> dict[str, Any]:
    x = torch.as_tensor(values, dtype=torch.float64)
    if not x.numel(): return {"count": 0}
    return {"count": int(x.numel()), "min": float(x.min()), "p50": float(torch.quantile(x, .5)),
            "p95": float(torch.quantile(x, .95)), "p99": float(torch.quantile(x, .99)), "max": float(x.max())}


def _load_mesh_geometry(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, str]]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    mesh = artifact.get("mesh", artifact) if isinstance(artifact, dict) else artifact
    vertices = mesh.vertices.float().cpu().contiguous(); faces = mesh.faces.int().cpu().contiguous()
    hashes = {"mesh_file_sha256": sha256_file(path), "vertices_sha256": tensor_sha256(vertices),
              "faces_sha256": tensor_sha256(faces)}
    return vertices, faces, hashes


def prepare_support(args: argparse.Namespace, out: Path) -> tuple[torch.Tensor, list[dict[str, Any]], torch.Tensor, dict[str, Any]]:
    vertices, faces, hashes = _load_mesh_geometry(Path(args.baseline_mesh))
    fp = {**hashes, "voxelizer": VOXELIZER, "schema": FORMAT}
    fp_hash = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()
    cache = out / "support" / "c4096_occupancy.pt"
    coords4096 = None; reused = False
    if cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if payload.get("fingerprint_sha256") != fp_hash: raise RuntimeError("C4096 cache fingerprint mismatch; refusing reuse")
        coords4096 = payload["coords"].int().contiguous(); reused = True
    if coords4096 is None:
        started = time.perf_counter()
        raw = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices=vertices, faces=faces, grid_size=4096, aabb=VOXELIZER["aabb"],
            face_weight=1.0, boundary_weight=.2, regularization_weight=1e-2)[0].int().cpu()
        keys = linear_keys(raw, 4096); order = torch.argsort(keys, stable=True); raw = raw[order]; keys = keys[order]
        keep = torch.ones(keys.numel(), dtype=torch.bool); keep[1:] = keys[1:] != keys[:-1]
        coords4096 = raw[keep].contiguous()
        atomic_save(cache, {"format": FORMAT, "fingerprint_sha256": fp_hash, "fingerprint": fp, "coords": coords4096})
        voxel_seconds = time.perf_counter() - started
    else: voxel_seconds = 0.0
    xyz256 = torch.div(coords4096.long(), 16, rounding_mode="floor")
    xyz256 = torch.unique(xyz256, dim=0, sorted=True).int()
    xyz256 = xyz256[torch.argsort(linear_keys(xyz256), stable=True)].contiguous()
    coords = torch.cat((torch.zeros((xyz256.shape[0], 1), dtype=torch.int32), xyz256), 1)
    records, coverage = build_cube_records(coords)
    owner, owner_stats = build_owner_map(coords, records)
    support_hash = tensor_sha256(coords)
    owner_hash = tensor_sha256(owner)
    atomic_save(out / "support" / "global_c256_support.pt", {"format": FORMAT, "coords": coords,
                "global_row_id": torch.arange(coords.shape[0]), "support_sha256": support_hash, "fingerprint": fp})
    atomic_save(out / "cubes" / "owner_map.pt", {"owner_cube_id": owner, "owner_sha256": owner_hash,
                "support_sha256": support_hash})
    token_rows = []
    for rec in records:
        ids, local = rec["global_row_ids"], rec["local_xyz"]
        payload = {"cube_id": rec["cube_id"], "start": rec["start"], "global_row_ids": ids,
                   "local_coords": torch.cat((torch.zeros((ids.numel(), 1), dtype=torch.int32), local), 1),
                   "owned_row_ids": rec["owned_row_ids"]}
        atomic_save(out / "cubes" / f"cube_{int(rec['cube_id']):03d}.pt", payload)
        token_rows.append({"cube_id": rec["cube_id"], "sx": rec["start"][0], "sy": rec["start"][1],
                           "sz": rec["start"][2], "membership_tokens": ids.numel(),
                           "owned_tokens": rec["owned_row_ids"].numel()})
    (out / "cubes").mkdir(parents=True, exist_ok=True)
    with (out / "cubes" / "token_histogram.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(token_rows[0])); w.writeheader(); w.writerows(token_rows)
    counts4096 = torch.unique(xyz256, dim=0, return_counts=True)[1]  # retained for schema; detailed counts below recomputed
    child_keys = linear_keys(torch.div(coords4096.long(), 16, rounding_mode="floor"), GRID)
    child_counts = torch.unique(child_keys, return_counts=True)[1]
    c4096_stats = {"unique_c4096_token_count": int(coords4096.shape[0]), "coord_min": coords4096.min(0).values,
                   "coord_max": coords4096.max(0).values, "voxelize_seconds": voxel_seconds, "cache_reused": reused}
    c256_stats = {"unique_c256_token_count": int(coords.shape[0]), "coord_min": xyz256.min(0).values,
                  "coord_max": xyz256.max(0).values, "multiplicity_histogram": histogram(child_counts.tolist()),
                  "support_sha256": support_hash}
    atomic_json(out / "support" / "c4096_stats.json", c4096_stats)
    atomic_json(out / "support" / "global_c256_stats.json", c256_stats)
    stats = {**owner_stats, "owner_sha256": owner_hash, "coverage_histogram": histogram(coverage.tolist()),
             "total_incidence_rows": int(sum(r["global_row_ids"].numel() for r in records)),
             "membership_distribution": _distribution([r["global_row_ids"].numel() for r in records if r["global_row_ids"].numel()]),
             "empty_cubes": sum(not r["global_row_ids"].numel() for r in records),
             "nonempty_cubes": sum(bool(r["global_row_ids"].numel()) for r in records),
             "flow_active_cubes": sum(bool(r["owned_row_ids"].numel()) for r in records)}
    atomic_json(out / "cubes" / "coverage_owner_stats.json", stats)
    atomic_json(out / "cubes" / "layout.json", {"starts": STARTS, "cube_count": 343, "records": token_rows})
    atomic_json(out / "inputs" / "fingerprints.json", fp)
    return coords, records, owner, {"support": c256_stats, "cubes": stats, "fingerprint_sha256": fp_hash}


def flow_condition_records(
    records: Sequence[Mapping[str, Any]], velocity_fusion: str,
) -> list[Mapping[str, Any]]:
    """Return exactly the cubes whose velocity will be evaluated."""
    if velocity_fusion not in {"owner", "gaussian"}:
        raise ValueError(f"unknown velocity fusion: {velocity_fusion}")
    require_owned = velocity_fusion == "owner"
    return [
        rec for rec in records
        if int(rec["global_row_ids"].numel()) > 0
        and (not require_owned or int(rec.get("owned_row_ids", torch.empty(0)).numel()) > 0)
    ]


def _condition_fingerprint(
    args: argparse.Namespace,
    coords: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    stage: str,
    camera: Mapping[str, Any],
) -> dict[str, Any]:
    active = flow_condition_records(records, str(args.velocity_fusion))
    return {
        "format": LOCAL_CONDITION_FORMAT,
        "stage": stage,
        "condition_image_4096_sha256": sha256_file(Path(args.condition_image_4096)),
        "condition_image_4096": str(Path(args.condition_image_4096).resolve()),
        "support_sha256": tensor_sha256(coords),
        "camera": {k: float(camera[k]) for k in ("camera_angle_x", "distance", "mesh_scale")},
        "global_projection_grid": GRID,
        "projection_image_size": PROJECTION_IMAGE_SIZE,
        "canonical_image_size": CANONICAL_IMAGE_SIZE,
        "crop_source": "projected physical cube boundary",
        "crop_alignment": DINO_PATCH_SIZE,
        "preserve_crop_resolution": True,
        "naf_target": "configured nominal scale applied independently to crop height/width",
        "velocity_fusion": str(args.velocity_fusion),
        "active_cube_ids": [int(rec["cube_id"]) for rec in active],
        "crop_boxes_4096": {
            str(int(rec["cube_id"])): list(rec["condition_crop"]["crop_box_4096"])
            for rec in active
        },
        "model": f"image_cond_model_{'shape' if stage == 'shape' else 'tex'}_1024",
    }


def _validate_cube_condition(
    payload: Mapping[str, Any],
    rec: Mapping[str, Any],
    cube_fingerprint_sha256: str,
) -> dict[str, torch.Tensor]:
    cube_id = int(rec["cube_id"])
    row_ids = rec["global_row_ids"].cpu().long()
    if payload.get("fingerprint_sha256") != cube_fingerprint_sha256:
        raise RuntimeError(f"cube {cube_id} local condition cache mismatch")
    cached_rows = payload.get("global_row_ids")
    if not isinstance(cached_rows, torch.Tensor) or not torch.equal(cached_rows.long(), row_ids):
        raise RuntimeError(f"cube {cube_id} local condition rows mismatch")
    if tuple(payload.get("crop_box_4096", ())) != tuple(rec["condition_crop"]["crop_box_4096"]):
        raise RuntimeError(f"cube {cube_id} local condition crop mismatch")
    glob, proj = payload.get("global"), payload.get("proj")
    if not isinstance(glob, torch.Tensor) or glob.ndim != 3 or glob.shape[0] != 1:
        raise RuntimeError(f"cube {cube_id} local global token has invalid shape")
    if not isinstance(proj, torch.Tensor) or proj.ndim != 2 or proj.shape[0] != row_ids.numel():
        raise RuntimeError(f"cube {cube_id} local projected token has invalid shape")
    if not torch.isfinite(glob).all() or not torch.isfinite(proj).all():
        raise FloatingPointError(f"cube {cube_id} local condition contains NaN/Inf")
    return {
        "global": glob.detach().cpu().contiguous(),
        "proj": proj.detach().cpu().contiguous(),
        "global_row_ids": row_ids,
    }


@torch.no_grad()
def build_condition(
    pipeline: Any,
    args: argparse.Namespace,
    coords: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    camera: Mapping[str, Any],
    stage: str,
    out: Path,
) -> dict[str, Any]:
    if stage not in {"shape", "texture"}:
        raise ValueError("stage must be shape or texture")
    active = flow_condition_records(records, str(args.velocity_fusion))
    if not active:
        raise RuntimeError("no flow-active cube can build a local condition")
    for rec in active:
        if "condition_crop" not in rec:
            raise RuntimeError("cube projection crops must be attached before condition extraction")
    fp = _condition_fingerprint(args, coords, records, stage, camera)
    fp_hash = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()
    root = out / "conditions" / f"{stage}_local_cubes"
    root.mkdir(parents=True, exist_ok=True)
    cubes: dict[int, dict[str, torch.Tensor]] = {}
    pending: list[tuple[Mapping[str, Any], Path, str]] = []
    extraction_rows: list[dict[str, Any]] = []
    for rec in active:
        cube_id = int(rec["cube_id"])
        descriptor = {
            "condition_fingerprint_sha256": fp_hash,
            "cube_id": cube_id,
            "start": list(rec["start"]),
            "global_row_ids_sha256": tensor_sha256(rec["global_row_ids"].long()),
            "crop_box_4096": list(rec["condition_crop"]["crop_box_4096"]),
        }
        cube_hash = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
        path = root / f"cube_{cube_id:03d}.pt"
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            cubes[cube_id] = _validate_cube_condition(payload, rec, cube_hash)
            extraction_rows.append({
                "cube_id": cube_id,
                "tokens": int(rec["global_row_ids"].numel()),
                "crop_box_4096": list(rec["condition_crop"]["crop_box_4096"]),
                "crop_size": list(rec["condition_crop"]["crop_size"]),
                "cache_reused": True,
                "seconds": 0.0,
            })
        else:
            pending.append((rec, path, cube_hash))

    image = Image.open(args.condition_image_4096).convert("RGB")
    if image.size != (CANONICAL_IMAGE_SIZE, CANONICAL_IMAGE_SIZE):
        raise RuntimeError(
            f"local condition image must be 4096x4096, got {image.size}"
        )
    model = pipeline.image_cond_model_shape_1024 if stage == "shape" else pipeline.image_cond_model_tex_1024
    low_vram = bool(getattr(pipeline, "low_vram", False))
    if pending and low_vram:
        model.to(pipeline.device)
        pipeline.low_vram = False
    try:
        for index, (rec, path, cube_hash) in enumerate(pending, start=1):
            started = time.perf_counter()
            cube_id = int(rec["cube_id"])
            box = tuple(int(v) for v in rec["condition_crop"]["crop_box_4096"])
            crop = image.crop(box).convert("RGB")
            if crop.size != tuple(rec["condition_crop"]["crop_size"]):
                raise RuntimeError(f"cube {cube_id} PIL crop size mismatch")
            if crop.width % DINO_PATCH_SIZE or crop.height % DINO_PATCH_SIZE:
                raise RuntimeError(f"cube {cube_id} crop is not DINO patch aligned")
            subset_coords = coords.index_select(0, rec["global_row_ids"].long()).to(pipeline.device)
            condition = pipeline.get_proj_cond_shape(
                model,
                [crop],
                subset_coords,
                camera_angle_x=float(camera["camera_angle_x"]),
                distance=float(camera["distance"]),
                mesh_scale=float(camera["mesh_scale"]),
                grid_resolution_override=GRID,
                projection_crop_box=rec["condition_crop"]["projection_crop_box"],
                transform_matrix=None,
                preserve_image_resolution=True,
            )
            glob = condition["cond"]["global"].detach().cpu().contiguous()
            proj = condition["cond"]["proj"].feats.detach().cpu().contiguous()
            payload = {
                "format": LOCAL_CONDITION_FORMAT,
                "stage": stage,
                "fingerprint": fp,
                "fingerprint_sha256": cube_hash,
                "condition_fingerprint_sha256": fp_hash,
                "cube_id": cube_id,
                "start": tuple(rec["start"]),
                "global_row_ids": rec["global_row_ids"].cpu().long(),
                "crop_box_4096": box,
                "projection_crop_box": tuple(rec["condition_crop"]["projection_crop_box"]),
                "crop_size": crop.size,
                "preserve_image_resolution": True,
                "global": glob,
                "proj": proj,
            }
            atomic_save(path, payload)
            cubes[cube_id] = _validate_cube_condition(payload, rec, cube_hash)
            elapsed = time.perf_counter() - started
            extraction_rows.append({
                "cube_id": cube_id,
                "tokens": int(rec["global_row_ids"].numel()),
                "crop_box_4096": list(box),
                "crop_size": list(crop.size),
                "cache_reused": False,
                "seconds": elapsed,
            })
            print(
                f"[condition-{stage}] cube={cube_id:03d} "
                f"crop={crop.width}x{crop.height} tokens={proj.shape[0]:,} "
                f"{index}/{len(pending)} seconds={elapsed:.1f}",
                flush=True,
            )
            del condition, crop, subset_coords, glob, proj, payload
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if pending and low_vram:
            pipeline.low_vram = True
            model.cpu()
        image.close()
        _empty_cuda()

    extraction_rows.sort(key=lambda row: row["cube_id"])
    if set(cubes) != {int(rec["cube_id"]) for rec in active}:
        raise RuntimeError(f"{stage} local condition cube coverage mismatch")
    manifest = {
        "format": LOCAL_CONDITION_FORMAT,
        "stage": stage,
        "fingerprint": fp,
        "fingerprint_sha256": fp_hash,
        "active_cube_count": len(active),
        "cache_reused_count": sum(bool(row["cache_reused"]) for row in extraction_rows),
        "dino_per_cube": True,
        "naf_per_cube": bool(getattr(model, "use_naf_upsample", False)),
        "global_token_source": "same local C4096 crop as projected tokens",
        "records": extraction_rows,
    }
    atomic_json(out / "conditions" / f"{stage}_local_manifest.json", manifest)
    return {"cubes": cubes, "fingerprint_sha256": fp_hash, "manifest": manifest}


def _local_coords(rec: Mapping[str, Any]) -> torch.Tensor:
    xyz = rec["local_xyz"].to(torch.int32)
    return torch.cat((torch.zeros((xyz.shape[0], 1), dtype=torch.int32), xyz), 1)


def _pack_state(group: Sequence[Mapping[str, Any]], state: torch.Tensor, device: torch.device) -> SparseTensor:
    values = [SparseTensor(state.index_select(0, r["global_row_ids"]), _local_coords(r)) for r in group]
    return legacy._pack_sparse_batch(values, "cube flow state").to(device)


def _pack_condition(group: Sequence[Mapping[str, Any]], condition: Mapping[str, Any],
                    packed_coords: torch.Tensor, device: torch.device) -> dict[str, Any]:
    by_cube = condition.get("cubes")
    if not isinstance(by_cube, Mapping):
        raise TypeError("local condition must contain a per-cube mapping")
    proj_parts: list[torch.Tensor] = []
    global_parts: list[torch.Tensor] = []
    for rec in group:
        cube_id = int(rec["cube_id"])
        local = by_cube.get(cube_id)
        if not isinstance(local, Mapping):
            raise RuntimeError(f"missing local condition for cube {cube_id}")
        cached_rows = local.get("global_row_ids")
        expected_rows = rec["global_row_ids"].cpu().long()
        if not isinstance(cached_rows, torch.Tensor) or not torch.equal(cached_rows, expected_rows):
            raise RuntimeError(f"cube {cube_id} condition row order changed")
        proj = local.get("proj")
        glob = local.get("global")
        if not isinstance(proj, torch.Tensor) or proj.shape[0] != expected_rows.numel():
            raise RuntimeError(f"cube {cube_id} projected local token shape mismatch")
        if not isinstance(glob, torch.Tensor) or glob.ndim != 3 or glob.shape[0] != 1:
            raise RuntimeError(f"cube {cube_id} global local token shape mismatch")
        proj_parts.append(proj)
        global_parts.append(glob)
    proj = torch.cat(proj_parts, 0).to(device)
    glob = torch.cat(global_parts, 0).to(device)
    if proj.shape[0] != packed_coords.shape[0]:
        raise RuntimeError("packed local condition is not sparse-row aligned")
    return {"cond": {"global": glob, "proj": SparseTensor(proj, packed_coords)},
            "neg_cond": {"global": torch.zeros_like(glob), "proj": SparseTensor(torch.zeros_like(proj), packed_coords)}}


def _pack_concat(group: Sequence[Mapping[str, Any]], concat: torch.Tensor, device: torch.device) -> SparseTensor:
    return _pack_state(group, concat, device)


def _prediction_kwargs(params: Mapping[str, Any]) -> dict[str, Any]:
    return legacy._prediction_kwargs(params)


@torch.no_grad()
def _one_prediction(group: Sequence[Mapping[str, Any]], state: torch.Tensor, condition: Mapping[str, Any],
                    sampler: Any, model: Any, params: Mapping[str, Any], t: float, t_next: float,
                    device: torch.device, concat: Optional[torch.Tensor] = None) -> tuple[list[torch.Tensor], dict[str, Any]]:
    packed = _pack_state(group, state, device)
    cond = _pack_condition(group, condition, packed.coords, device)
    concat_st = _pack_concat(group, concat, device) if concat is not None else None
    started = time.perf_counter()
    result = sampler.sample_once(model, packed, float(t), float(t_next), cond=cond["cond"],
                                 neg_cond=cond["neg_cond"], concat_cond=concat_st,
                                 **_prediction_kwargs(params))
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    parts = legacy._split_sparse_batch(result.pred_v, len(group), "cube pred_v")
    values = []
    for rec, part in zip(group, parts):
        if not torch.equal(part.coords.cpu(), _local_coords(rec)): raise RuntimeError("model changed cube row order")
        values.append(part.feats.detach().float().cpu().contiguous())
    return values, {"seconds": seconds, "tokens": int(packed.feats.shape[0]), "batch_size": len(group)}


@torch.no_grad()
def probe_max_batch(stage: str, rec: Mapping[str, Any], state: torch.Tensor,
                    condition: Mapping[str, Any], sampler: Any, model: Any,
                    params: Mapping[str, Any], device: torch.device, out: Path,
                    concat: Optional[torch.Tensor], max_candidate: int = 343) -> int:
    """Find the largest B<=max_candidate for repeated largest-cube real CFG forward."""
    final_path = out / "smoke" / f"{stage}_largest_cube.json"
    if final_path.is_file():
        cached = json.loads(final_path.read_text())
        if (cached.get("status") == "passed" and cached.get("cube_id") == int(rec["cube_id"])
                and cached.get("membership_tokens") == int(rec["global_row_ids"].numel())
                and int(cached.get("max_successful_batch_size", 0)) <= int(max_candidate)):
            return int(cached["max_successful_batch_size"])
    attempts: list[dict[str, Any]] = []
    model.to(device).eval()

    def attempt(batch: int) -> bool:
        _empty_cuda(); torch.cuda.reset_peak_memory_stats(device)
        free_before, total = torch.cuda.mem_get_info(device)
        row = {"batch_size": batch, "tokens_per_item": int(rec["global_row_ids"].numel()),
               "total_tokens": batch * int(rec["global_row_ids"].numel()), "free_before_bytes": int(free_before),
               "total_memory_bytes": int(total)}
        try:
            _, timing = _one_prediction([rec] * batch, state, condition, sampler, model, params, 1.0, .99,
                                        device, concat)
            row.update(timing); row["success"] = True
            row["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            row["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
            ok = True
        except torch.cuda.OutOfMemoryError as exc:
            row.update({"success": False, "oom": True, "error": str(exc)[:1000],
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))})
            ok = False
        attempts.append(row)
        atomic_json(out / "smoke" / f"{stage}_largest_cube.in_progress.json",
                    {"stage": stage, "cube_id": int(rec["cube_id"]), "attempts": attempts})
        _empty_cuda(); return ok

    success = 0; candidate = 1; failure = max_candidate + 1
    while candidate <= max_candidate:
        if attempt(candidate):
            success = candidate
            if candidate == max_candidate: break
            candidate = min(max_candidate, candidate * 2)
            if candidate == success: break
        else:
            failure = candidate; break
    if not success:
        payload = {"format": FORMAT, "stage": stage, "status": "blocked_single_cube_oom", "attempts": attempts}
        atomic_json(out / "smoke" / f"{stage}_largest_cube.json", payload)
        raise RuntimeError(f"{stage} largest cube OOM at B=1")
    if failure <= max_candidate and failure - success > 1:
        lo, hi = success + 1, failure - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if attempt(mid): success = mid; lo = mid + 1
            else: failure = mid; hi = mid - 1
    payload = {"format": FORMAT, "stage": stage, "status": "passed", "cube_id": int(rec["cube_id"]),
               "membership_tokens": int(rec["global_row_ids"].numel()), "max_successful_batch_size": success,
               "first_oom_batch_size": failure if failure <= max_candidate else None,
               "search_cap": max_candidate, "shape": "[B, variable_sparse_tokens, channels]", "attempts": attempts}
    atomic_json(final_path, payload)
    progress = out / "smoke" / f"{stage}_largest_cube.in_progress.json"
    if progress.exists(): progress.unlink()
    model.cpu(); _empty_cuda()
    return success


def _edge_pairs(coords: torch.Tensor, owner: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = coords[:, 1:].long(); keys = linear_keys(xyz); all_pairs = []
    for axis, delta in enumerate((GRID * GRID, GRID, 1)):
        valid = xyz[:, axis] < GRID - 1
        src = torch.where(valid)[0]; target_keys = keys[src] + delta
        pos = torch.searchsorted(keys, target_keys)
        hit = (pos < keys.numel()) & (keys[pos.clamp_max(keys.numel()-1)] == target_keys)
        if hit.any(): all_pairs.append(torch.stack((src[hit], pos[hit]), 1))
    edges = torch.cat(all_pairs, 0) if all_pairs else torch.empty((0, 2), dtype=torch.int64)
    cross = owner[edges[:, 0]] != owner[edges[:, 1]]
    return edges[~cross], edges[cross]


def _jump_stats(value: torch.Tensor, edges: torch.Tensor) -> dict[str, Any]:
    if not edges.numel(): return {"edges": 0}
    a, b = value[edges[:, 0]].float(), value[edges[:, 1]].float(); d = a - b
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
    return {"edges": int(edges.shape[0]), "l1_mean": float(d.abs().mean()),
            "l2_mean": float(d.norm(dim=1).mean()), "cosine_mean": float(cos.mean())}


def _proposal_disagreement(owner: torch.Tensor, owner_velocity: torch.Tensor,
                           proposals: Sequence[tuple[int, torch.Tensor, torch.Tensor]]) -> dict[str, Any]:
    l1 = []; l2 = []; cos = []
    for cube_id, ids, value in proposals:
        halo = owner.index_select(0, ids) != int(cube_id)
        if not halo.any(): continue
        a = value[halo].float(); b = owner_velocity.index_select(0, ids[halo]).float(); d = a - b
        l1.append(d.abs().mean(1)); l2.append(d.norm(dim=1)); cos.append(torch.nn.functional.cosine_similarity(a, b, dim=1))
    if not l1: return {"rows": 0}
    l1t, l2t, cost = torch.cat(l1), torch.cat(l2), torch.cat(cos)
    return {"rows": int(l1t.numel()), "l1": _distribution(l1t), "l2": _distribution(l2t),
            "cosine": _distribution(cost)}


def _capture_rng() -> dict[str, Any]:
    return {"torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state()}


@torch.no_grad()
def run_flow(stage: str, initial: torch.Tensor, coords: torch.Tensor, records: Sequence[Mapping[str, Any]],
             owner: torch.Tensor, condition: Mapping[str, Any], sampler: Any, model: Any,
             params: Mapping[str, Any], out: Path, device: torch.device, flow_batch_size: int,
             max_batch_tokens: int, concat: Optional[torch.Tensor], resume: bool,
             velocity_fusion: str = "owner", gaussian_sigma: float = 32.0) -> tuple[torch.Tensor, dict[str, Any]]:
    root = out / stage; checkpoint = root / "checkpoint.pt"
    steps = int(params["steps"]); schedule = sampler.timestep_schedule(steps, float(params.get("rescale_t", 1.0)))
    if velocity_fusion not in {"owner", "gaussian"}:
        raise ValueError(f"unknown velocity fusion: {velocity_fusion}")
    gaussian_weights = build_gaussian_weight_table(coords, records, gaussian_sigma) if velocity_fusion == "gaussian" else {}
    weight_hash = hashlib.sha256()
    for cube_id in sorted(gaussian_weights):
        weight_hash.update(str(cube_id).encode()); weight_hash.update(tensor_sha256(gaussian_weights[cube_id]).encode())
    fusion_fingerprint = {"mode": velocity_fusion, "gaussian_sigma_c256_cells": float(gaussian_sigma),
                          "gaussian_weight_sha256": weight_hash.hexdigest() if gaussian_weights else None}
    if gaussian_weights:
        weight_path = out / "cubes" / "gaussian_velocity_weights.pt"
        if not weight_path.is_file():
            atomic_save(weight_path, {"format": FORMAT, "support_sha256": tensor_sha256(coords),
                        "sigma_c256_cells": float(gaussian_sigma), "weights_by_cube_id": gaussian_weights,
                        "weight_sha256": fusion_fingerprint["gaussian_weight_sha256"]})
            all_weights = torch.cat(list(gaussian_weights.values()))
            atomic_json(out / "cubes" / "gaussian_velocity_weight_stats.json",
                        {**fusion_fingerprint, "formula": "exp(-distance_sq/(2*sigma^2)), normalized per global row",
                         "incidence_weights": _distribution(all_weights),
                         "nonempty_cubes": len(gaussian_weights)})
    hashes = {"support": tensor_sha256(coords), "owner": tensor_sha256(owner), "condition": condition["fingerprint_sha256"],
              "noise": tensor_sha256(initial), "concat": tensor_sha256(concat) if concat is not None else None,
              "velocity_fusion": fusion_fingerprint}
    state = initial.float().cpu().contiguous(); next_step = 0; step_rows: list[dict[str, Any]] = []
    if resume and checkpoint.is_file():
        cp = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if cp.get("hashes") != hashes or cp.get("schedule") != schedule or cp.get("sampler_params") != dict(params):
            raise RuntimeError(f"{stage} resume fingerprint mismatch")
        state = cp["state"].float(); next_step = int(cp["next_step"]); step_rows = cp["records"]
    groups = pack_groups(records, flow_batch_size, max_batch_tokens, require_owned=velocity_fusion == "owner")
    same_edges, cross_edges = _edge_pairs(coords, owner)
    model.to(device).eval()
    for step in range(next_step, steps):
        t, t_next = schedule[step], schedule[step + 1]
        frozen = state.clone()  # Jacobi barrier
        proposals: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        timings = []; torch.cuda.reset_peak_memory_stats(device); started = time.perf_counter()
        for group in groups:
            velocities, timing = _one_prediction(group, frozen, condition, sampler, model, params, t, t_next, device, concat)
            timings.append(timing)
            for rec, velocity in zip(group, velocities):
                proposals.append((int(rec["cube_id"]), rec["global_row_ids"], velocity))
        if velocity_fusion == "gaussian":
            fused_velocity, fused_weight_sum = validate_gaussian_fusion(
                int(coords.shape[0]), proposals, gaussian_weights, int(initial.shape[1]))
        else:
            fused_velocity = validate_owner_scatter(owner, proposals, int(initial.shape[1]))
            fused_weight_sum = torch.ones(owner.numel(), dtype=torch.float32)
        state = jacobi_update(frozen, fused_velocity, t, t_next)
        if not torch.isfinite(state).all(): raise FloatingPointError(f"{stage} step {step}: non-finite state")
        record = {"step": step, "t": t, "t_next": t_next, "seconds": time.perf_counter()-started,
                  "physical_batches": len(groups), "flow_batch_size": flow_batch_size,
                  "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                  "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                  "velocity_fusion": fusion_fingerprint,
                  "normalized_weight_sum": _distribution(fused_weight_sum),
                  "velocity_norm": _distribution(fused_velocity.norm(dim=1)),
                  "proposal_disagreement": _proposal_disagreement(owner, fused_velocity, proposals),
                  "boundary": {"velocity_same_owner": _jump_stats(fused_velocity, same_edges),
                               "velocity_cross_owner": _jump_stats(fused_velocity, cross_edges),
                               "latent_same_owner": _jump_stats(state, same_edges),
                               "latent_cross_owner": _jump_stats(state, cross_edges)}, "batch_timings": timings}
        step_rows.append(record)
        atomic_save(root / "fused_velocity_last.pt", {"step": step, "velocity": fused_velocity,
                    "velocity_fusion": fusion_fingerprint})
        atomic_save(checkpoint, {"format": FORMAT, "stage": stage, "coords": coords, "state": state,
                    "next_step": step + 1, "schedule": schedule, "sampler_params": dict(params),
                    "hashes": hashes, "records": step_rows, "rng_state": _capture_rng()})
        atomic_json(root / "flow_summary.json", {"stage": stage, "steps_completed": step+1, "steps": steps,
                    "schedule": schedule, "records": step_rows, "flow_batch_size": flow_batch_size,
                    "max_batch_tokens": max_batch_tokens, "jacobi_barrier": True,
                    "velocity_fusion": fusion_fingerprint})
        for name, key in (("proposal_disagreement_by_step.json", "proposal_disagreement"),
                          ("owner_boundary_by_step.json", "boundary")):
            path = out / "diagnostics" / name
            prior = json.loads(path.read_text()) if path.is_file() else {}
            prior[stage] = [r[key] for r in step_rows]
            atomic_json(path, prior)
        print(f"[{stage}] step {step+1}/{steps} B={flow_batch_size} batches={len(groups)} seconds={record['seconds']:.1f}", flush=True)
    model.cpu(); _empty_cuda()
    summary = {"stage": stage, "steps": steps, "steps_completed": len(step_rows), "schedule": schedule,
               "records": step_rows, "flow_batch_size": flow_batch_size, "max_batch_tokens": max_batch_tokens,
               "hashes": hashes, "jacobi_barrier": True, "velocity_fusion": fusion_fingerprint,
               "velocity_or_endpoint_averaging": velocity_fusion == "gaussian"}
    atomic_json(root / "flow_summary.json", summary)
    return state, summary


def _norm_tensors(normalization: Mapping[str, Any], channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.as_tensor(normalization["mean"], dtype=torch.float32)[None]
    std = torch.as_tensor(normalization["std"], dtype=torch.float32)[None]
    if mean.shape != (1, channels) or std.shape != (1, channels): raise RuntimeError("normalization channel mismatch")
    return mean, std


def denormalize(features: torch.Tensor, normalization: Mapping[str, Any]) -> torch.Tensor:
    mean, std = _norm_tensors(normalization, features.shape[1]); return features.float() * std + mean


def normalize(features: torch.Tensor, normalization: Mapping[str, Any]) -> torch.Tensor:
    mean, std = _norm_tensors(normalization, features.shape[1]); return (features.float() - mean) / std


def _official_c64_tokens(baseline_mesh: Path) -> Optional[int]:
    path = baseline_mesh.parent / "baseline_c64_latents.pt"
    if not path.is_file(): return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    shape = payload.get("shape_slat")
    if isinstance(shape, Mapping) and isinstance(shape.get("coords"), torch.Tensor): return int(shape["coords"].shape[0])
    if hasattr(shape, "coords"): return int(shape.coords.shape[0])
    return None


@torch.no_grad()
def decode_render(pipeline: Any, shape_norm: torch.Tensor, texture_norm: torch.Tensor,
                  coords: torch.Tensor, camera: Mapping[str, Any], args: argparse.Namespace,
                  out: Path, device: torch.device) -> dict[str, Any]:
    shape_denorm = denormalize(shape_norm, pipeline.shape_slat_normalization)
    texture_denorm = denormalize(texture_norm, pipeline.tex_slat_normalization)
    if not torch.allclose(normalize(shape_denorm, pipeline.shape_slat_normalization), shape_norm, atol=2e-5, rtol=2e-5):
        raise RuntimeError("shape normalization roundtrip failed")
    if not torch.allclose(normalize(texture_denorm, pipeline.tex_slat_normalization), texture_norm, atol=2e-5, rtol=2e-5):
        raise RuntimeError("texture normalization roundtrip failed")
    atomic_save(out / "shape" / "final_state_normalized.pt", {"coords": coords, "features": shape_norm})
    atomic_save(out / "shape" / "final_state_denormalized.pt", {"coords": coords, "features": shape_denorm})
    atomic_save(out / "texture" / "final_state_normalized.pt", {"coords": coords, "features": texture_norm})
    atomic_save(out / "texture" / "final_state_denormalized.pt", {"coords": coords, "features": texture_denorm})
    summary_path = out / "decode" / "summary.json"
    vertex_path = out / "final" / "final_per_vertex_pbr_mesh.pt"
    if summary_path.is_file() and vertex_path.is_file() and json.loads(summary_path.read_text()).get("status") == "complete":
        summary = json.loads(summary_path.read_text())
        payload = torch.load(vertex_path, map_location="cpu", weights_only=False)
        vertex_mesh = payload.get("mesh", payload) if isinstance(payload, Mapping) else payload
        summary["decode_cache_reused"] = True
    else:
        shape = SparseTensor(shape_denorm.to(device), coords.to(device)); texture = SparseTensor(texture_denorm.to(device), coords.to(device))
        torch.cuda.reset_peak_memory_stats(device); started = time.perf_counter()
        try:
            decoded = pipeline.decode_latent(shape, texture, 4096)
        except torch.cuda.OutOfMemoryError as exc:
            summary = {"status": "oom", "error": str(exc), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                       "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), "tokens": int(coords.shape[0])}
            atomic_json(summary_path, summary); _empty_cuda(); return summary
        if len(decoded) != 1: raise RuntimeError(f"decode returned B={len(decoded)}")
        native = decoded[0]
        atomic_save(out / "final" / "final_material_mesh.pt", {"format": FORMAT, "mesh": native.cpu()})
        vertex_mesh, face_mesh = expc._native_mesh_to_pbr(native, device)
        atomic_save(vertex_path, {"format": FORMAT, "mesh": vertex_mesh})
        atomic_save(out / "final" / "final_per_face_pbr_mesh.pt", {"format": FORMAT, "mesh": face_mesh})
        summary = {"status": "complete", "resolution": 4096, "vertices": int(vertex_mesh.vertices.shape[0]),
                   "faces": int(vertex_mesh.faces.shape[0]), "seconds": time.perf_counter()-started,
                   "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                   "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                   "mode": "single global pipeline.decode_latent(C256,C256,4096)"}
        atomic_json(summary_path, summary)
    if args.render:
        gt_image = Image.open(args.canonical_gt).convert("RGB")
        mask_image = Image.open(args.foreground_mask).convert("L")
        gt = np.asarray(gt_image, dtype=np.float32) / 255.0
        mask = np.asarray(mask_image, dtype=np.float32) / 255.0
        render = legacy._render_one(vertex_mesh, camera, out / "final", device, "final", 4096,
                                    reference=gt, foreground=mask)
        renders: dict[str, Mapping[str, Any]] = {"final": render}
        expc_root = Path(args.baseline_mesh).parent.parent
        expc_rgb = expc_root / "final" / "final_render_rgb_4096.png"
        expc_alpha = expc_root / "final" / "final_render_alpha_4096.png"
        if expc_rgb.is_file(): renders["baseline_to_4096_exp_c"] = {"rgb_path": expc_rgb, "alpha_path": expc_alpha}
        baseline_dir = out / "comparisons" / "official_baseline1024"
        baseline_rgb = baseline_dir / "official_baseline1024_render_rgb_4096.png"
        baseline_alpha = baseline_dir / "official_baseline1024_render_alpha_4096.png"
        if not baseline_rgb.is_file():
            payload = torch.load(args.baseline_mesh, map_location="cpu", weights_only=False)
            native_baseline = payload.get("mesh", payload) if isinstance(payload, Mapping) else payload
            baseline_vertex, _ = expc._native_mesh_to_pbr(native_baseline, device)
            legacy._render_one(baseline_vertex, camera, baseline_dir, device, "official_baseline1024", 4096,
                               reference=gt, foreground=mask)
            del payload, native_baseline, baseline_vertex
            _empty_cuda()
        if baseline_rgb.is_file(): renders["official_baseline1024"] = {"rgb_path": baseline_rgb, "alpha_path": baseline_alpha}
        legacy._compute_global_metrics(gt_image, mask_image, out, renders)
        metrics = json.loads((out / "metrics_4096.json").read_text())
        rows = {row["variant"]: row for row in metrics.get("rows", [])}
        final_row = rows.get("final", {})
        metrics["deltas_final_minus_comparison"] = {
            name: {key: (float(final_row[key]) - float(row[key]))
                   for key in ("psnr_db", "foreground_psnr_db", "ssim", "foreground_ssim", "alpha_iou", "lpips_alex_native_512_patch")
                   if final_row.get(key) is not None and row.get(key) is not None}
            for name, row in rows.items() if name != "final"
        }
        atomic_json(out / "metrics_4096.json", metrics)
        summary["render"] = render
        atomic_json(out / "decode" / "summary.json", summary)
    return summary


def write_report(out: Path, status: str, meta: Mapping[str, Any], shape_summary: Optional[Mapping[str, Any]] = None,
                 texture_summary: Optional[Mapping[str, Any]] = None, decode_summary: Optional[Mapping[str, Any]] = None) -> None:
    cubes = meta.get("cubes", {}); support = meta.get("support", {})
    smoke_shape_path = out / "smoke" / "shape_largest_cube.json"
    smoke_tex_path = out / "smoke" / "texture_largest_cube.json"
    smoke_shape = json.loads(smoke_shape_path.read_text()) if smoke_shape_path.is_file() else {}
    smoke_tex = json.loads(smoke_tex_path.read_text()) if smoke_tex_path.is_file() else {}
    metrics_path = out / "metrics_4096.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    metric_rows = {row["variant"]: row for row in metrics.get("rows", [])}
    final_metrics = metric_rows.get("final", {})
    metric_deltas = metrics.get("deltas_final_minus_comparison", {})
    config_path = out / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    fusion = config.get("fusion", {"mode": "owner"})
    gaussian = fusion.get("mode") == "gaussian"
    flow_description = (f"每一步冻结唯一 global `X_t`，收齐全部非空 cube velocity 后，按三维中心距离 Gaussian "
                        f"(σ={fusion.get('gaussian_sigma_c256_cells')} C256 cells) 逐 row 归一化融合，再同步 Euler 更新。"
                        if gaussian else
                        "每一步冻结唯一 global `X_t`，收齐 hard-owner velocity 后同步 Euler 更新；无 velocity/endpoint averaging。")
    title = "Global C256 Cube Gaussian Velocity Fusion Flow" if gaussian else "Global C256 Cube Owner Flow"
    active_description = cubes.get("nonempty_cubes") if gaussian else cubes.get("flow_active_cubes")
    text = f"""# {title} 报告

最终状态：`{status}`

- fresh C4096 tokens：{meta.get('c4096_tokens', 'unknown')}；fresh C256 tokens：{support.get('unique_c256_token_count', 'unknown')}。
- cube：343；empty/nonempty/owner-active：{cubes.get('empty_cubes')}/{cubes.get('nonempty_cubes')}/{cubes.get('flow_active_cubes')}；本次 flow-active：{active_description}。
- coverage histogram：`{json.dumps(cubes.get('coverage_histogram', {}), ensure_ascii=False)}`；每 row 固定唯一 owner，owner SHA-256 `{cubes.get('owner_sha256')}`。
- membership token 分布：`{json.dumps(cubes.get('membership_distribution', {}), ensure_ascii=False)}`；official baseline C64 tokens：{meta.get('official_c64_tokens')}。
- Shape 最大 cube 的真实 CFG `[B,...]` 最大成功 batch：{smoke_shape.get('max_successful_batch_size')}；首次 OOM：{smoke_shape.get('first_oom_batch_size')}。
- Texture 最大 cube smoke batch：{smoke_tex.get('max_successful_batch_size')}；首次 OOM：{smoke_tex.get('first_oom_batch_size')}。
- Shape/Texture steps：{(shape_summary or {}).get('steps_completed', 0)}/{(shape_summary or {}).get('steps', 12)}，{(texture_summary or {}).get('steps_completed', 0)}/{(texture_summary or {}).get('steps', 12)}。
- {flow_description}
- 每个 flow-active 3D cube 先用 global camera 投影其物理边界到 canonical 4096；bbox 向外对齐到 16 像素后保持长宽比原样送入 DINOv3，NAF 按模型原有倍率生成对应矩形超分特征。
- 每个 cube 的 projected token 与 `global`（CLS + register）token 都来自该 cube 自己的 4096 crop；投影仍使用 global C256 坐标，只在进入 flow 时 relabel 为 local C64。
- local/off-axis camera、visibility routing、PBR fusion：均未使用。
- global decode：{(decode_summary or {}).get('status', 'not_started')}；vertices/faces：{(decode_summary or {}).get('vertices')}/{(decode_summary or {}).get('faces')}。
- yaw=0 final GT：PSNR {final_metrics.get('psnr_db')}，foreground PSNR {final_metrics.get('foreground_psnr_db')}，SSIM {final_metrics.get('ssim')}，foreground SSIM {final_metrics.get('foreground_ssim')}，LPIPS {final_metrics.get('lpips_alex_native_512_patch')}，alpha IoU {final_metrics.get('alpha_iou')}。
- 相对 baseline→4096 Exp-C / official baseline1024 的差值：`{json.dumps(metric_deltas, ensure_ascii=False)}`。

风险边界：局部 C64 support 的 full-attention 上下文可能 OOD；{('Gaussian 融合会混合重叠 cube 的 OOD proposal' if gaussian else 'hard owner 会形成 Voronoi 边界')}；单视图不能提供背面证据；fixed support 不能在 baseline occupancy 之外自由创建拓扑。逐步 proposal disagreement 与 same/cross-owner jump 见 `diagnostics/`。
"""
    path = out / "REPORT.md"; path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".REPORT.", suffix=".tmp", dir=out); os.close(fd)
    Path(tmp).write_text(text, encoding="utf-8"); os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    base = Path("outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024")
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--physical-cuda", type=int, default=4)
    p.add_argument("--input-image", default="assets/images/0_img.png")
    p.add_argument(
        "--condition-image-4096", "--condition-image",
        dest="condition_image_4096",
        default=str(base / "inputs/canonical_foreground_rgb_4096.png"),
        help="canonical 4096 image cropped independently for every 3-D cube",
    )
    p.add_argument("--canonical-gt", default=str(base / "inputs/canonical_foreground_rgb_4096.png"))
    p.add_argument("--foreground-mask", default=str(base / "inputs/canonical_foreground_mask_4096.png"))
    p.add_argument("--baseline-mesh", default=str(base / "baseline/baseline_1024_mesh.pt"))
    p.add_argument("--camera-json", default=str(base / "global_camera.json"))
    p.add_argument("--output-dir", default="outputs/global_c256_cube_owner_flow_singleview_cuda4")
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--grid-resolution", type=int, default=256)
    p.add_argument("--cube-size", type=int, default=64)
    p.add_argument("--cube-stride", type=int, default=32)
    p.add_argument("--shape-steps", type=int, default=12)
    p.add_argument("--texture-steps", type=int, default=12)
    p.add_argument("--shape-seed", type=int, default=43)
    p.add_argument("--texture-seed", type=int, default=44)
    p.add_argument("--flow-batch-size", type=int, default=343, help="upper bound for empirical B search")
    p.add_argument("--max-batch-tokens", type=int, default=10_000_000)
    p.add_argument("--velocity-fusion", choices=("owner", "gaussian"), default="owner")
    p.add_argument("--gaussian-sigma", type=float, default=32.0,
                   help="3-D Gaussian sigma in global C256 cells")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if (args.grid_resolution, args.cube_size, args.cube_stride) != (256, 64, 32):
        raise ValueError("this experiment is fixed to C256/C64/stride32")
    if not math.isfinite(args.gaussian_sigma) or args.gaussian_sigma <= 0:
        raise ValueError("--gaussian-sigma must be finite and positive")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and [x.strip() for x in visible.split(",")] != [str(args.physical_cuda)]:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}, expected physical CUDA {args.physical_cuda} only")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device = torch.device(args.device); torch.cuda.set_device(device)
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    for name in ("input_image", "condition_image_4096", "canonical_gt", "foreground_mask", "baseline_mesh", "camera_json"):
        if not Path(getattr(args, name)).is_file(): raise FileNotFoundError(getattr(args, name))
    config = {"format": FORMAT, "status": "running", "args": vars(args),
              "runtime": {"CUDA_VISIBLE_DEVICES": visible, "logical_device": str(device),
                          "gpu": torch.cuda.get_device_name(device), "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory},
              "fusion": {"mode": args.velocity_fusion, "gaussian_sigma_c256_cells": args.gaussian_sigma,
                         "distance": "global cell centre to C64 cube centre", "normalized_per_global_row": True},
              "local_condition": {"image_crop_4096": True, "crop_alignment_pixels": DINO_PATCH_SIZE,
                                  "preserve_crop_resolution": True, "local_global_token": True},
              "forbidden_routes": {"local_camera": False, "visibility": False,
                                   "endpoint_average": False, "pbr_fusion": False}}
    atomic_json(out / "config.json", config)
    status = "blocked_geometry"; meta: dict[str, Any] = {}; shape_summary = texture_summary = decode_summary = None
    try:
        coords, records, owner, meta = prepare_support(args, out)
        c4096_stats = json.loads((out / "support/c4096_stats.json").read_text())
        meta["c4096_tokens"] = c4096_stats["unique_c4096_token_count"]
        meta["official_c64_tokens"] = _official_c64_tokens(Path(args.baseline_mesh))
        camera = json.loads(Path(args.camera_json).read_text())
        projection_diagnostics = attach_cube_projection_crops(records, camera)
        projection_diagnostics["source"] = str(Path(args.condition_image_4096).resolve())
        projection_diagnostics["global_projection_grid"] = GRID
        projection_diagnostics["local_camera"] = False
        projection_diagnostics["native_out_of_bounds_behavior"] = "crop clipped to canonical image; border sampler preserved"
        atomic_json(out / "conditions/projection_diagnostics.json", projection_diagnostics)
        pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=args.low_vram)
        status = "blocked_condition"
        shape_cond = build_condition(pipeline, args, coords, records, camera, "shape", out)
        shape_model = pipeline.models["shape_slat_flow_model_1024"]
        shape_channels = int(shape_model.in_channels)
        shape_noise = torch.randn((coords.shape[0], shape_channels), generator=torch.Generator().manual_seed(args.shape_seed))
        atomic_save(out / "shape/initial_noise.pt", {"coords": coords, "features": shape_noise,
                    "noise_sha256": tensor_sha256(shape_noise), "seed": args.shape_seed})
        active_records = flow_condition_records(records, args.velocity_fusion)
        largest = max(active_records, key=lambda r: r["global_row_ids"].numel())
        shape_params = dict(pipeline.shape_slat_sampler_params); shape_params["steps"] = args.shape_steps
        max_shape_b = probe_max_batch("shape", largest, shape_noise, shape_cond, pipeline.shape_slat_sampler,
                                      shape_model, shape_params, device, out, None, args.flow_batch_size)
        # The measured largest-cube B is the flow B requested by the user.
        status = "blocked_shape_flow"
        shape_norm, shape_summary = run_flow("shape", shape_noise, coords, records, owner, shape_cond,
                    pipeline.shape_slat_sampler, shape_model, shape_params, out, device, max_shape_b,
                    args.max_batch_tokens, None, args.resume, args.velocity_fusion, args.gaussian_sigma)
        del shape_cond
        _empty_cuda()
        texture_cond = build_condition(pipeline, args, coords, records, camera, "texture", out)
        tex_model = pipeline.models["tex_slat_flow_model_1024"]
        tex_channels = int(tex_model.in_channels) - int(shape_norm.shape[1])
        if tex_channels <= 0: raise RuntimeError("invalid texture noise channel count")
        tex_noise = torch.randn((coords.shape[0], tex_channels), generator=torch.Generator().manual_seed(args.texture_seed))
        atomic_save(out / "texture/initial_noise.pt", {"coords": coords, "features": tex_noise,
                    "noise_sha256": tensor_sha256(tex_noise), "seed": args.texture_seed})
        tex_params = dict(pipeline.tex_slat_sampler_params); tex_params["steps"] = args.texture_steps
        max_tex_b = probe_max_batch("texture", largest, tex_noise, texture_cond, pipeline.tex_slat_sampler,
                                    tex_model, tex_params, device, out, shape_norm, max_shape_b)
        status = "blocked_texture_flow"
        tex_norm, texture_summary = run_flow("texture", tex_noise, coords, records, owner, texture_cond,
                    pipeline.tex_slat_sampler, tex_model, tex_params, out, device, min(max_shape_b, max_tex_b),
                    args.max_batch_tokens, shape_norm, args.resume, args.velocity_fusion, args.gaussian_sigma)
        del texture_cond
        _empty_cuda()
        status = "flow_complete_decode_blocked"
        decode_summary = decode_render(pipeline, shape_norm, tex_norm, coords, camera, args, out, device)
        if decode_summary.get("status") == "complete": status = "complete"
    except Exception as exc:
        atomic_json(out / "failure.json", {"status": status, "type": type(exc).__name__, "error": str(exc)})
        write_report(out, status, meta, shape_summary, texture_summary, decode_summary)
        config["status"] = status; config["error"] = str(exc); atomic_json(out / "config.json", config)
        raise
    write_report(out, status, meta, shape_summary, texture_summary, decode_summary)
    config["status"] = status; config["empirical_shape_flow_batch_size"] = max_shape_b
    config["empirical_texture_flow_batch_size"] = min(max_shape_b, max_tex_b)
    atomic_json(out / "config.json", config)
    print(f"[done] status={status} output={out}", flush=True)


if __name__ == "__main__": main()
