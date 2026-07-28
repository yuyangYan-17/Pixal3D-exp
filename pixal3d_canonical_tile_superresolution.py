#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training-free 2048 Pixal3D synchronization in exact canonical space.

This is the executable implementation of ``CODEX_TASKS.md``.  It keeps the
ordinary global trajectory as the sole low-frequency master and uses each
1024 crop only for a local clean-endpoint residual:

    shared spatial noise
      -> independent global/local endpoint predictions
      -> robust merge(local endpoint - broadcast global endpoint)
      -> coverage-aware zero-parent-mean high-pass
      -> synchronized local Euler update

Sparse topology is handled separately.  Projected global support is mandatory;
tile-native support enters only after foreground/connectivity filtering and
multi-tile or global-surface consensus.  Shape and texture are generated from
the beginning on one sparse logical C128 support and decoded once at 2048.

Evaluation intentionally follows the non-modified routes in
``pixal3d_projective_tile_generation_eval_projected_c64_only_copy.py``:
native ``MeshWithVoxel`` outputs are rendered by
``render_utils.render_frames/PbrMeshRenderer`` and compared with the canonical
full image and exact canonical tile crops.  The modified global-geometry/tile-
material route from that reference script is never called.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
)

import argparse
import csv
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from inference import MODEL_PATH, init_pipeline  # noqa: E402
from pixal3d.modules.sparse import SparseTensor  # noqa: E402
from pixal3d.pipelines.canonical_tile_sync import (  # noqa: E402
    C128MasterAtomSpace,
    C128NativeWindow,
    CommonAtomSpace,
    build_c128_master_atom_space,
    build_common_atom_space,
    endpoint_indices_to_q,
    q_to_endpoint_indices,
    raised_cosine_tile_weights,
    run_c128_2048_coupled_endpoint_flow,
    run_coupled_endpoint_flow,
    shared_c128_master_local_noise,
    shared_spatial_noise,
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

import pixal3d_projective_tile_generation_eval as projective  # noqa: E402
import pixal3d_projective_tile_generation_eval_projected_c64_only_copy as evaluation  # noqa: E402


GRID_SS_NOISE = 16
GRID_SS = 32
GRID_SHAPE = 64
GRID_UNIFIED = 128
GRID_DECODED = 1024
IMAGE_512 = 512
IMAGE_1024 = 1024
IMAGE_CANONICAL = 4096
IMAGE_TARGET = 2048


@dataclass
class TileContext:
    tile_id: int
    box: Tuple[int, int, int, int]
    transform: evaluation.TileCameraTransform
    image_1024: Image.Image
    image_512: Image.Image
    foreground: Image.Image
    camera: Dict[str, float]
    native32: Optional[torch.Tensor] = None
    anchor32: Optional[torch.Tensor] = None
    support32: Optional[torch.Tensor] = None
    anchor64: Optional[torch.Tensor] = None
    candidate64: Optional[torch.Tensor] = None
    support64: Optional[torch.Tensor] = None
    candidate_confidence64: Optional[torch.Tensor] = None
    accepted_c128_codes: Optional[torch.Tensor] = None


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _log_cuda_memory(label: str) -> Dict[str, int]:
    if not torch.cuda.is_available():
        row = {"allocated": 0, "reserved": 0, "max_allocated": 0}
    else:
        _sync_cuda()
        row = {
            "allocated": int(torch.cuda.memory_allocated()),
            "reserved": int(torch.cuda.memory_reserved()),
            "max_allocated": int(torch.cuda.max_memory_allocated()),
        }
    print(
        f"[cuda-memory] {label}: "
        f"allocated={row['allocated'] / 2**30:.3f}GiB "
        f"reserved={row['reserved'] / 2**30:.3f}GiB "
        f"max_allocated={row['max_allocated'] / 2**30:.3f}GiB"
    )
    return row


def _release_cuda(label: str) -> Dict[str, int]:
    _empty_cuda_cache()
    _sync_cuda()
    return _log_cuda_memory(label)


def _move_to_cpu(*values: Any) -> None:
    for value in values:
        if value is not None and hasattr(value, "cpu"):
            value.cpu()


def _iter_cuda_tensors(value: Any, path: str = "root"):
    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            yield path, value
        return
    if isinstance(value, SparseTensor):
        yield from _iter_cuda_tensors(value.feats, f"{path}.feats")
        yield from _iter_cuda_tensors(value.coords, f"{path}.coords")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_cuda_tensors(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_cuda_tensors(item, f"{path}[{index}]")


def _assert_no_cuda_tensors(label: str, value: Any) -> None:
    retained = list(_iter_cuda_tensors(value, label))
    if retained:
        preview = ", ".join(path for path, _ in retained[:8])
        raise RuntimeError(
            f"{label} unexpectedly retains CUDA tensors: {preview}"
        )


def _assert_compact_coupled_result(label: str, result: Any) -> None:
    if hasattr(result, "trajectory"):
        raise RuntimeError(f"{label} must not retain a trajectory")
    for index, record in enumerate(result.step_records):
        if any(isinstance(value, torch.Tensor) for value in record.values()):
            raise RuntimeError(
                f"{label} step record {index} retains a tensor"
            )


def _seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    evaluation._atomic_json(path, payload)


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
        return value.to(device=device)
    if isinstance(value, SparseTensor):
        return SparseTensor(
            feats=value.feats.to(device=device),
            coords=value.coords.to(device=device),
        )
    if isinstance(value, Mapping):
        return {key: _tree_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_tree_to_device(item, device) for item in value]
    return value


def _denormalize(
    value: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
) -> SparseTensor:
    mean = torch.as_tensor(
        normalization["mean"], device=value.device, dtype=value.dtype
    )[None]
    std = torch.as_tensor(
        normalization["std"], device=value.device, dtype=value.dtype
    )[None]
    return value.replace(value.feats * std + mean)


def _sampler_step_kwargs(params: Mapping[str, Any]) -> Dict[str, Any]:
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


def _parse_tile_ids(value: Optional[str], count: int) -> List[int]:
    if value is None or not value.strip():
        return list(range(int(count)))
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    invalid = [item for item in result if item < 0 or item >= int(count)]
    if invalid:
        raise ValueError(f"invalid tile IDs {invalid}; expected 0..{count - 1}")
    return result


def _exact_transforms(
    global_camera: Mapping[str, float],
    tile: TileContext,
):
    def local_to_global(q: torch.Tensor) -> torch.Tensor:
        return projective._centered_tile_q_to_global_q(
            q,
            global_camera=global_camera,
            transform=tile.transform,
            validate_roundtrip=False,
        )[0]

    def global_to_local(q: torch.Tensor) -> torch.Tensor:
        return evaluation._global_q_to_centered_tile_q(
            q,
            global_camera=global_camera,
            transform=tile.transform,
        )[0]

    return local_to_global, global_to_local


def _filter_local_centers_in_global_cube(
    coords: torch.Tensor,
    *,
    resolution: int,
    forward,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if coords.shape[0] == 0:
        return coords, {
            "input_rows": 0,
            "kept_rows": 0,
            "outside_rows_dropped_no_clamp": 0,
            "q_global_min": None,
            "q_global_max": None,
        }
    q_local = endpoint_indices_to_q(coords[:, 1:4], int(resolution))
    q_global = forward(q_local)
    finite = torch.isfinite(q_global).all(dim=1)
    inside = finite & (q_global.abs() <= 1.0).all(dim=1)
    kept = coords[inside]
    return kept, {
        "input_rows": int(coords.shape[0]),
        "kept_rows": int(kept.shape[0]),
        "outside_rows_dropped_no_clamp": int((~inside).sum().item()),
        "q_global_min": [
            float(value) for value in q_global.amin(dim=0).tolist()
        ],
        "q_global_max": [
            float(value) for value in q_global.amax(dim=0).tolist()
        ],
    }


def _project_global_support_to_local(
    global_coords: torch.Tensor,
    *,
    global_resolution: int,
    local_resolution: int,
    inverse,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    q_global = endpoint_indices_to_q(
        global_coords[:, 1:4], int(global_resolution)
    )
    q_local = inverse(q_global)
    finite = torch.isfinite(q_local).all(dim=1)
    inside = finite & (q_local.abs() <= 1.0).all(dim=1)
    local_ids = q_to_endpoint_indices(
        q_local[inside], int(local_resolution)
    )
    valid = (
        (local_ids >= 0) & (local_ids < int(local_resolution))
    ).all(dim=1)
    local_ids = local_ids[valid]
    coords = torch.cat(
        [
            torch.zeros((local_ids.shape[0], 1), dtype=torch.int64),
            local_ids,
        ],
        dim=1,
    ).to(torch.int32)
    unique = torch.unique(coords, dim=0)
    return unique, {
        "global_rows": int(global_coords.shape[0]),
        "inside_rows": int(inside.sum().item()),
        "quantized_rows": int(coords.shape[0]),
        "unique_local_rows": int(unique.shape[0]),
        "outside_rows_dropped_no_clamp": int((~inside).sum().item()),
        "quantization_merges": int(coords.shape[0] - unique.shape[0]),
    }


def _coordinate_codes(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    xyz = coords[:, 1:4].to(torch.int64)
    return (
        (xyz[:, 0] * int(resolution) + xyz[:, 1]) * int(resolution)
        + xyz[:, 2]
    )


def _union_coords(*supports: torch.Tensor) -> torch.Tensor:
    valid = [support for support in supports if support is not None and support.numel()]
    if not valid:
        raise RuntimeError("cannot union empty supports")
    return torch.unique(torch.cat(valid, dim=0).to(torch.int32), dim=0)


def _filter_native_c32(
    native: torch.Tensor,
    anchor: torch.Tensor,
    tile: TileContext,
    global_camera: Mapping[str, float],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Foreground + radius-one connectivity gate for local topology candidates."""
    if native.numel() == 0:
        return native, {"input": 0, "kept": 0}
    q = endpoint_indices_to_q(native[:, 1:4], GRID_SS)
    # Local centered camera projection is equivalent to mapping through the
    # exact crop transform; the returned full UV is converted to tile pixels.
    _, _, uv_full, _ = projective._centered_tile_q_to_global_q(
        q,
        global_camera=global_camera,
        transform=tile.transform,
        validate_roundtrip=False,
    )
    x0, y0, _, _ = tile.box
    uv_tile = torch.stack(
        [
            (uv_full[:, 0] - float(x0))
            * float(tile.transform.crop_to_output_scale_x),
            (uv_full[:, 1] - float(y0))
            * float(tile.transform.crop_to_output_scale_y),
        ],
        dim=1,
    )
    pixels = torch.round(uv_tile).to(torch.int64)
    in_image = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < IMAGE_1024)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < IMAGE_1024)
    )
    mask_array = np.asarray(tile.foreground.convert("L"), dtype=np.uint8)
    foreground = torch.zeros(native.shape[0], dtype=torch.bool)
    rows = torch.where(in_image)[0]
    if rows.numel():
        foreground[rows] = torch.from_numpy(
            mask_array[
                pixels[rows, 1].numpy(),
                pixels[rows, 0].numpy(),
            ]
            >= 127
        )

    anchor_xyz = {
        tuple(int(v) for v in row)
        for row in anchor[:, 1:4].to(torch.int64).tolist()
    }
    connected = []
    for xyz in native[:, 1:4].to(torch.int64).tolist():
        connected.append(
            any(
                (xyz[0] + dx, xyz[1] + dy, xyz[2] + dz) in anchor_xyz
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
            )
        )
    connected_tensor = torch.tensor(connected, dtype=torch.bool)
    keep = foreground & connected_tensor
    kept = native[keep]
    return kept, {
        "input": int(native.shape[0]),
        "foreground": int(foreground.sum().item()),
        "anchor_radius1_connected": int(connected_tensor.sum().item()),
        "kept": int(kept.shape[0]),
    }

def _tile_token_weights(
    tile: TileContext,
    coords: torch.Tensor,
    resolution: int,
    global_camera: Mapping[str, float],
) -> torch.Tensor:
    q = endpoint_indices_to_q(coords[:, 1:4], int(resolution))
    _, _, uv_full, _ = projective._centered_tile_q_to_global_q(
        q,
        global_camera=global_camera,
        transform=tile.transform,
        validate_roundtrip=False,
    )
    x0, y0, _, _ = tile.box
    uv_tile = torch.stack(
        [
            (uv_full[:, 0] - float(x0))
            * float(tile.transform.crop_to_output_scale_x),
            (uv_full[:, 1] - float(y0))
            * float(tile.transform.crop_to_output_scale_y),
        ],
        dim=1,
    )
    return raised_cosine_tile_weights(
        uv_tile, IMAGE_1024, IMAGE_1024, minimum=1e-4
    )


def _full_dense_coords(resolution: int) -> torch.Tensor:
    axis = torch.arange(int(resolution), dtype=torch.int32)
    xyz = torch.cartesian_prod(axis, axis, axis)
    return torch.cat(
        [torch.zeros((xyz.shape[0], 1), dtype=torch.int32), xyz], dim=1
    )


def _features_to_dense_noise(
    features: torch.Tensor,
    resolution: int,
) -> torch.Tensor:
    channels = int(features.shape[1])
    expected = int(resolution) ** 3
    if features.shape[0] != expected:
        raise ValueError(f"dense features have {features.shape[0]} rows, expected {expected}")
    return (
        features.reshape(resolution, resolution, resolution, channels)
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .contiguous()
    )


def _random_features(
    rows: int,
    channels: int,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randn(
        int(rows), int(channels), generator=generator, device=device
    )


def _save_atom_space(path: Path, space: CommonAtomSpace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "pixal3d_common_atom_space_v1",
            "stage": space.stage,
            "global_resolution": space.global_resolution,
            "local_resolution": space.local_resolution,
            "target_resolution": space.target_resolution,
            "atom_coords": space.atom_coords,
            "atom_ids": space.atom_ids,
            "global_coords": space.global_coords,
            "global_atom_indices": space.global_atom_indices,
            "global_parent_rows": space.global_parent_rows,
            "atom_reference_parent": space.atom_reference_parent,
            "local": [
                {
                    "tile_id": mapping.tile_id,
                    "coords": mapping.coords,
                    "atom_indices": mapping.atom_indices,
                    "token_indices": mapping.token_indices,
                    "overlap_fractions": mapping.overlap_fractions,
                    "token_atom_counts": mapping.token_atom_counts,
                    "token_atom_mass": mapping.token_atom_mass,
                }
                for mapping in space.local_mappings
            ],
            "diagnostics": dict(space.diagnostics),
        },
        path,
    )


def _save_c128_master_space(
    path: Path,
    space: C128MasterAtomSpace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "pixal3d_c128_master_atom_space_v1",
            "stage": space.stage,
            "target_resolution": space.target_resolution,
            "local_resolution": space.local_resolution,
            "atom_coords": space.atom_coords,
            "atom_ids": space.atom_ids,
            "coarse_parent": space.coarse_parent,
            "coarse_parent_count": space.coarse_parent_count,
            "local": [
                {
                    "tile_id": mapping.tile_id,
                    "coords": mapping.coords,
                    "atom_indices": mapping.atom_indices,
                    "token_indices": mapping.token_indices,
                    "overlap_fractions": mapping.overlap_fractions,
                    "token_atom_counts": mapping.token_atom_counts,
                    "token_atom_mass": mapping.token_atom_mass,
                    "diagnostics": dict(mapping.diagnostics),
                }
                for mapping in space.local_mappings
            ],
            "diagnostics": dict(space.diagnostics),
        },
        path,
    )


def _prepare_tiles(
    *,
    image_4096: Image.Image,
    foreground_4096: Image.Image,
    global_camera: Mapping[str, float],
    tile_ids: Sequence[int],
    extend_pixel: int,
) -> List[TileContext]:
    boxes = evaluation._tile_layout(IMAGE_CANONICAL, IMAGE_1024, 512)
    tiles: List[TileContext] = []
    for tile_id in tile_ids:
        box = boxes[int(tile_id)]
        transform = evaluation._derive_tile_camera(
            tile_id=int(tile_id),
            box=box,
            global_camera=global_camera,
            extend_pixel=int(extend_pixel),
            offaxis_shift_y_sign=1,
        )
        tile_image = image_4096.crop(box).resize(
            (IMAGE_1024, IMAGE_1024), Image.Resampling.LANCZOS
        ).convert("RGB")
        tile_foreground = foreground_4096.crop(box).resize(
            (IMAGE_1024, IMAGE_1024), Image.Resampling.NEAREST
        ).convert("L")
        tiles.append(
            TileContext(
                tile_id=int(tile_id),
                box=tuple(int(value) for value in box),
                transform=transform,
                image_1024=tile_image,
                image_512=tile_image.resize(
                    (IMAGE_512, IMAGE_512), Image.Resampling.LANCZOS
                ),
                foreground=tile_foreground,
                camera={
                    "camera_angle_x": float(transform.camera_angle_x),
                    "distance": float(transform.distance),
                    "mesh_scale": float(transform.mesh_scale),
                },
            )
        )
    return tiles


def _build_stage_atom_space(
    *,
    stage: str,
    global_coords: torch.Tensor,
    resolution: int,
    tiles: Sequence[TileContext],
    local_supports: Sequence[torch.Tensor],
    global_camera: Mapping[str, float],
    target_resolution: int,
    chunk_size: int,
) -> CommonAtomSpace:
    forwards = []
    inverses = []
    for tile in tiles:
        forward, inverse = _exact_transforms(global_camera, tile)
        forwards.append(forward)
        inverses.append(inverse)
    return build_common_atom_space(
        stage=stage,
        global_coords=global_coords.detach().cpu(),
        global_resolution=int(resolution),
        local_coords=[
            support.detach().cpu() for support in local_supports
        ],
        local_resolution=int(resolution),
        local_to_global=forwards,
        global_to_local=inverses,
        tile_ids=[tile.tile_id for tile in tiles],
        target_resolution=int(target_resolution),
        chunk_size=int(chunk_size),
    )


def _build_c128_master_space(
    *,
    stage: str,
    master_coords128: torch.Tensor,
    global_coords64: torch.Tensor,
    tiles: Sequence[TileContext],
    local_supports64: Sequence[torch.Tensor],
    global_camera: Mapping[str, float],
    chunk_size: int,
) -> C128MasterAtomSpace:
    forwards = []
    inverses = []
    for tile in tiles:
        forward, inverse = _exact_transforms(global_camera, tile)
        forwards.append(forward)
        inverses.append(inverse)
    master_codes = _coordinate_codes(master_coords128, GRID_UNIFIED)
    coarse_parent = _nearest_global_c64_parent_rows(
        master_codes, global_coords64.detach().cpu()
    )
    return build_c128_master_atom_space(
        stage=stage,
        master_coords=master_coords128.detach().cpu(),
        coarse_parent=coarse_parent,
        coarse_parent_count=int(global_coords64.shape[0]),
        local_coords=[
            support.detach().cpu() for support in local_supports64
        ],
        local_to_global=forwards,
        global_to_local=inverses,
        tile_ids=[tile.tile_id for tile in tiles],
        target_resolution=GRID_UNIFIED,
        local_resolution=GRID_SHAPE,
        chunk_size=int(chunk_size),
    )


def _run_shared_sparse_structures(
    *,
    pipeline: Any,
    image_512: Image.Image,
    global_camera: Mapping[str, float],
    tiles: Sequence[TileContext],
    params: Mapping[str, Any],
    seed: int,
    atom_chunk_size: int,
    output_dir: Path,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Run dense SS flows with shared canonical noise on their overlap."""
    model = pipeline.models["sparse_structure_flow_model"]
    dense_resolution = int(model.resolution)
    if dense_resolution != GRID_SS_NOISE:
        raise RuntimeError(
            f"expected sparse flow C{GRID_SS_NOISE} noise, got C{dense_resolution}"
        )
    channels = int(model.in_channels)
    full = _full_dense_coords(dense_resolution)
    valid_local: List[torch.Tensor] = []
    local_filter_stats: List[Dict[str, Any]] = []
    for tile in tiles:
        forward, _ = _exact_transforms(global_camera, tile)
        valid, stats = _filter_local_centers_in_global_cube(
            full,
            resolution=dense_resolution,
            forward=forward,
        )
        valid_local.append(valid)
        local_filter_stats.append(stats)
    atom_space = _build_stage_atom_space(
        stage="ss16",
        global_coords=full,
        resolution=dense_resolution,
        tiles=tiles,
        local_supports=valid_local,
        global_camera=global_camera,
        target_resolution=4 * dense_resolution,
        chunk_size=int(atom_chunk_size),
    )
    _save_atom_space(output_dir / "atoms" / "ss16.pt", atom_space)
    global_noise, local_valid_noise, atom_noise = shared_spatial_noise(
        atom_space,
        channels,
        seed=int(seed),
        namespace="noise/ss16",
        device=pipeline.device,
    )
    del atom_noise

    global_condition = pipeline.get_proj_cond_ss(
        [image_512],
        camera_angle_x=float(global_camera["camera_angle_x"]),
        distance=float(global_camera["distance"]),
        mesh_scale=float(global_camera["mesh_scale"]),
    )
    _seed_everything(seed)
    global_coords32 = pipeline.sample_sparse_structure(
        global_condition,
        resolution=GRID_SS,
        sampler_params=dict(params),
        noise=_features_to_dense_noise(global_noise, dense_resolution),
    )
    del global_condition
    tile_stats: List[Dict[str, Any]] = []
    for index, tile in enumerate(tiles):
        condition = pipeline.get_proj_cond_ss(
            [tile.image_512],
            camera_angle_x=float(tile.camera["camera_angle_x"]),
            distance=float(tile.camera["distance"]),
            mesh_scale=float(tile.camera["mesh_scale"]),
        )
        full_noise = _random_features(
            full.shape[0],
            channels,
            seed=int(seed) + 100000 + int(tile.tile_id),
            device=pipeline.device,
        )
        valid_codes = _coordinate_codes(
            valid_local[index], dense_resolution
        ).to(device=pipeline.device, dtype=torch.long)
        full_noise.index_copy_(
            0,
            valid_codes,
            local_valid_noise[index],
        )
        _seed_everything(seed + 100000 + tile.tile_id)
        native = pipeline.sample_sparse_structure(
            condition,
            resolution=GRID_SS,
            sampler_params=dict(params),
            noise=_features_to_dense_noise(full_noise, dense_resolution),
        ).detach().cpu()
        forward, inverse = _exact_transforms(global_camera, tile)
        native, domain_stats = _filter_local_centers_in_global_cube(
            native, resolution=GRID_SS, forward=forward
        )
        anchor, anchor_stats = _project_global_support_to_local(
            global_coords32.detach().cpu(),
            global_resolution=GRID_SS,
            local_resolution=GRID_SS,
            inverse=inverse,
        )
        native, gate_stats = _filter_native_c32(
            native, anchor, tile, global_camera
        )
        tile.native32 = native
        tile.anchor32 = anchor
        tile.support32 = _union_coords(anchor, native)
        tile_stats.append(
            {
                "tile_id": int(tile.tile_id),
                "dense_noise_domain": local_filter_stats[index],
                "native_domain": domain_stats,
                "anchor": anchor_stats,
                "candidate_gate": gate_stats,
                "support32_rows": int(tile.support32.shape[0]),
            }
        )
        del condition, full_noise
        _empty_cuda_cache()
    diagnostics = {
        "atom_space": dict(atom_space.diagnostics),
        "global_c32_rows": int(global_coords32.shape[0]),
        "tiles": tile_stats,
        "noise_namespace": "noise/ss16",
        "outside_local_noise": (
            "independent stateless run noise; no shared physical global atom"
        ),
    }
    _atomic_json(output_dir / "stages" / "ss16_ss32.json", diagnostics)
    result_coords = global_coords32.detach().cpu()
    _move_to_cpu(
        pipeline.models["sparse_structure_flow_model"],
        pipeline.models["sparse_structure_decoder"],
        pipeline.image_cond_model_ss,
    )
    del (
        atom_space,
        global_noise,
        local_valid_noise,
        valid_local,
        full,
        model,
        global_coords32,
    )
    for tile in tiles:
        # Only sparse C32 coordinates survive; no dense state/noise is kept.
        if hasattr(tile, "dense_noise"):
            tile.dense_noise = None
    _release_cuda("ss-complete")
    return result_coords, diagnostics


def _prepare_shape_conditions(
    *,
    pipeline: Any,
    image_model: Any,
    global_image: Image.Image,
    global_coords: torch.Tensor,
    global_camera: Mapping[str, float],
    tiles: Sequence[TileContext],
    local_supports: Sequence[torch.Tensor],
    resolution: int,
) -> Tuple[Mapping[str, Any], List[Mapping[str, Any]]]:
    global_condition = pipeline.get_proj_cond_shape(
        image_model,
        [global_image],
        global_coords.to(pipeline.device),
        camera_angle_x=float(global_camera["camera_angle_x"]),
        distance=float(global_camera["distance"]),
        mesh_scale=float(global_camera["mesh_scale"]),
        grid_resolution_override=int(resolution),
    )
    local_conditions: List[Mapping[str, Any]] = []
    for tile, support in zip(tiles, local_supports):
        condition = pipeline.get_proj_cond_shape(
            image_model,
            [tile.image_512 if global_image.size == (512, 512) else tile.image_1024],
            support.to(pipeline.device),
            camera_angle_x=float(tile.camera["camera_angle_x"]),
            distance=float(tile.camera["distance"]),
            mesh_scale=float(tile.camera["mesh_scale"]),
            grid_resolution_override=int(resolution),
        )
        local_conditions.append(_tree_to_cpu(condition))
        del condition
        _empty_cuda_cache()
    return _tree_to_cpu(global_condition), local_conditions


def _run_continuous_stage(
    *,
    pipeline: Any,
    stage: str,
    sampler: Any,
    model: torch.nn.Module,
    atom_space: CommonAtomSpace,
    global_condition: Mapping[str, Any],
    local_conditions: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    seed: int,
    tiles: Sequence[TileContext],
    global_camera: Mapping[str, float],
    huber_delta: float,
    robust_iterations: int,
    global_concat: Optional[SparseTensor] = None,
    local_concat: Optional[Sequence[SparseTensor]] = None,
    support_confidence: Optional[Sequence[torch.Tensor]] = None,
) -> Any:
    device = pipeline.device
    channels = int(model.in_channels)
    if global_concat is not None:
        channels -= int(global_concat.feats.shape[1])
    if channels < 1:
        raise RuntimeError(f"{stage}: invalid generated channel count {channels}")
    global_noise, local_noise, _ = shared_spatial_noise(
        atom_space,
        channels,
        seed=int(seed),
        namespace=f"noise/{stage}",
        device=device,
    )
    global_state = SparseTensor(
        feats=global_noise,
        coords=atom_space.global_coords.to(device),
    )
    local_states = [
        SparseTensor(
            feats=noise,
            coords=mapping.coords.to(device),
        )
        for noise, mapping in zip(local_noise, atom_space.local_mappings)
    ]
    local_weights = [
        _tile_token_weights(
            tile,
            mapping.coords,
            atom_space.local_resolution,
            global_camera,
        ).to(device)
        for tile, mapping in zip(tiles, atom_space.local_mappings)
    ]
    if support_confidence is not None:
        support_confidence = [
            value.to(device) for value in support_confidence
        ]
    if pipeline.low_vram:
        model.to(device)
    progress = tqdm(total=int(params["steps"]), desc=stage, dynamic_ncols=True)

    def update(completed: int, total: int, record: Mapping[str, float]) -> None:
        progress.n = completed
        progress.set_postfix(
            atoms=int(record["atom_count"]),
            covered=int(record["covered_atoms"]),
            low=f"{record['max_abs_Rg_unified_minus_global']:.2e}",
        )
        progress.refresh()

    result = run_coupled_endpoint_flow(
        sampler=sampler,
        model=model,
        atom_space=atom_space,
        global_state=global_state,
        local_states=local_states,
        global_condition=_tree_to_device(global_condition, device),
        local_conditions=[
            _tree_to_device(value, device) for value in local_conditions
        ],
        steps=int(params["steps"]),
        rescale_t=float(params.get("rescale_t", 1.0)),
        sampler_step_kwargs=_sampler_step_kwargs(params),
        global_concat_cond=global_concat,
        local_concat_cond=local_concat,
        local_token_weights=local_weights,
        support_confidence=support_confidence,
        huber_delta=float(huber_delta),
        robust_iterations=int(robust_iterations),
        invariant_tolerance=2e-5,
        progress_callback=update,
    )
    progress.close()
    if pipeline.low_vram:
        model.cpu()
        _empty_cuda_cache()
    return result


def _prepare_c128_master_local_conditions(
    *,
    pipeline: Any,
    image_model: Any,
    global_image: Image.Image,
    master_coords128: torch.Tensor,
    global_camera: Mapping[str, float],
    tiles: Sequence[TileContext],
    atom_space: C128MasterAtomSpace,
) -> Tuple[Mapping[str, Any], List[Mapping[str, Any]]]:
    global_condition = pipeline.get_proj_cond_shape(
        image_model,
        [global_image],
        master_coords128.to(pipeline.device),
        camera_angle_x=float(global_camera["camera_angle_x"]),
        distance=float(global_camera["distance"]),
        mesh_scale=float(global_camera["mesh_scale"]),
        grid_resolution_override=GRID_UNIFIED,
    )
    local_conditions: List[Mapping[str, Any]] = []
    for tile, mapping in zip(tiles, atom_space.local_mappings):
        condition = pipeline.get_proj_cond_shape(
            image_model,
            [tile.image_1024],
            mapping.coords.to(pipeline.device),
            camera_angle_x=float(tile.camera["camera_angle_x"]),
            distance=float(tile.camera["distance"]),
            mesh_scale=float(tile.camera["mesh_scale"]),
            grid_resolution_override=GRID_SHAPE,
        )
        local_conditions.append(_tree_to_cpu(condition))
        del condition
        _release_cuda(f"condition-tile-{tile.tile_id:02d}")
    result = _tree_to_cpu(global_condition), local_conditions
    del global_condition
    _release_cuda("condition-global-c128")
    return result


def _native_c128_windows(
    pipeline: Any,
    coords128: torch.Tensor,
) -> Tuple[C128NativeWindow, ...]:
    if GRID_UNIFIED != 128:
        raise RuntimeError("native 2048 windows require C128")
    patches = pipeline._build_2048_overlap_patches(
        coords128.to(pipeline.device),
        grid_resolution=GRID_UNIFIED,
        patch_size=GRID_SHAPE,
        patch_stride=GRID_SS,
    )
    windows: List[C128NativeWindow] = []
    for patch in patches:
        if patch["token_indices"].numel() == 0:
            continue
        if patch["local_coords"] is None or patch["weights"] is None:
            raise RuntimeError("active native C64 window is incomplete")
        windows.append(
            C128NativeWindow(
                window_index=int(patch["patch_index"]),
                token_indices=patch["token_indices"].detach().cpu(),
                local_coords=patch["local_coords"].detach().cpu(),
                weights=patch["weights"].detach().cpu(),
                start=tuple(int(value) for value in patch["start"]),
                end=tuple(int(value) for value in patch["end"]),
            )
        )
    if not windows:
        raise RuntimeError("fixed C128 support activates no native C64 window")
    return tuple(windows)


def _mapping_support_confidence(
    tile: TileContext,
    coords: torch.Tensor,
) -> torch.Tensor:
    if tile.anchor64 is None:
        raise RuntimeError(f"tile {tile.tile_id} is missing C64 anchors")
    anchor_codes = torch.sort(
        _coordinate_codes(tile.anchor64, GRID_SHAPE)
    ).values
    codes = _coordinate_codes(coords, GRID_SHAPE)
    positions = torch.searchsorted(anchor_codes, codes)
    safe = positions.clamp_max(max(0, anchor_codes.shape[0] - 1))
    is_anchor = (positions < anchor_codes.shape[0]) & (
        anchor_codes.index_select(0, safe) == codes
    )
    return torch.where(
        is_anchor,
        torch.ones_like(codes, dtype=torch.float32),
        torch.full_like(codes, 0.85, dtype=torch.float32),
    )


def _run_c128_stage(
    *,
    pipeline: Any,
    stage: str,
    sampler: Any,
    model: torch.nn.Module,
    atom_space: C128MasterAtomSpace,
    windows: Sequence[C128NativeWindow],
    global_condition: Mapping[str, Any],
    local_conditions: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    seed: int,
    tiles: Sequence[TileContext],
    global_camera: Mapping[str, float],
    huber_delta: float,
    robust_iterations: int,
    global_concat: Optional[SparseTensor] = None,
    local_concat: Optional[Sequence[SparseTensor]] = None,
) -> Any:
    device = pipeline.device
    channels = int(model.in_channels)
    if global_concat is not None:
        channels -= int(global_concat.feats.shape[1])
    if channels < 1:
        raise RuntimeError(f"{stage}: invalid generated channel count {channels}")
    global_noise, local_noise, atom_noise = shared_c128_master_local_noise(
        atom_space,
        channels,
        seed=int(seed),
        namespace=f"noise/{stage}",
        device=device,
    )
    global_state = SparseTensor(
        feats=global_noise,
        coords=atom_space.atom_coords.to(device),
    )
    local_states = [
        SparseTensor(
            feats=noise,
            coords=mapping.coords.to(device),
        )
        for noise, mapping in zip(local_noise, atom_space.local_mappings)
    ]
    local_weights = [
        _tile_token_weights(
            tile,
            mapping.coords,
            GRID_SHAPE,
            global_camera,
        ).to(device)
        for tile, mapping in zip(tiles, atom_space.local_mappings)
    ]
    support_confidence = [
        _mapping_support_confidence(tile, mapping.coords).to(device)
        for tile, mapping in zip(tiles, atom_space.local_mappings)
    ]
    global_condition_device = _tree_to_device(global_condition, device)
    local_conditions_device = [
        _tree_to_device(value, device) for value in local_conditions
    ]
    model.to(device)
    progress = tqdm(total=int(params["steps"]), desc=stage, dynamic_ncols=True)

    def update(completed: int, total: int, record: Mapping[str, float]) -> None:
        progress.n = completed
        progress.set_postfix(
            c128=int(record["atom_count"]),
            covered=int(record["covered_atoms"]),
            low=f"{record['max_abs_Pcoarse_unified_minus_global']:.2e}",
        )
        progress.refresh()

    result = run_c128_2048_coupled_endpoint_flow(
        sampler=sampler,
        model=model,
        atom_space=atom_space,
        global_windows=windows,
        global_state=global_state,
        local_states=local_states,
        global_condition=global_condition_device,
        local_conditions=local_conditions_device,
        steps=int(params["steps"]),
        rescale_t=float(params.get("rescale_t", 1.0)),
        sampler_step_kwargs=_sampler_step_kwargs(params),
        global_concat_cond=global_concat,
        local_concat_cond=local_concat,
        local_token_weights=local_weights,
        support_confidence=support_confidence,
        huber_delta=float(huber_delta),
        robust_iterations=int(robust_iterations),
        invariant_tolerance=2e-5,
        progress_callback=update,
    )
    progress.close()
    _assert_compact_coupled_result(stage, result)
    del (
        global_state,
        local_states,
        global_noise,
        local_noise,
        atom_noise,
        local_weights,
        support_confidence,
        global_condition_device,
        local_conditions_device,
    )
    _release_cuda(f"{stage}-temporary-state-released")
    return result


def _upsample_shape512(
    pipeline: Any,
    value: SparseTensor,
) -> torch.Tensor:
    denorm = _denormalize(value, pipeline.shape_slat_normalization)
    return evaluation._learned_upsample_shape512_to_c64(pipeline, denorm)


def _candidate_global_codes(
    tile: TileContext,
    coords: torch.Tensor,
    global_camera: Mapping[str, float],
    source_resolution: int,
    target_resolution: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    forward, _ = _exact_transforms(global_camera, tile)
    q_local = endpoint_indices_to_q(
        coords[:, 1:4], int(source_resolution)
    )
    q_global = forward(q_local)
    inside = torch.isfinite(q_global).all(dim=1) & (
        q_global.abs() <= 1.0
    ).all(dim=1)
    xyz = torch.floor(
        (q_global[inside] + 1.0) * (float(target_resolution) / 2.0)
    ).to(torch.int64)
    valid = ((xyz >= 0) & (xyz < int(target_resolution))).all(dim=1)
    rows = torch.where(inside)[0][valid]
    xyz = xyz[valid]
    codes = (
        (xyz[:, 0] * int(target_resolution) + xyz[:, 1])
        * int(target_resolution)
        + xyz[:, 2]
    )
    return rows, codes


def _codes_to_coords(codes: torch.Tensor, resolution: int) -> torch.Tensor:
    codes = codes.to(torch.int64)
    z = torch.remainder(codes, int(resolution))
    quotient = torch.div(codes, int(resolution), rounding_mode="floor")
    y = torch.remainder(quotient, int(resolution))
    x = torch.div(quotient, int(resolution), rounding_mode="floor")
    return torch.stack(
        [torch.zeros_like(x), x, y, z], dim=1
    ).to(torch.int32)


def _topology_consensus_c128(
    *,
    shape512_atom_space: CommonAtomSpace,
    global_anchor_codes: torch.Tensor,
    tiles: Sequence[TileContext],
    global_camera: Mapping[str, float],
    surface_band_cells: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Fuse exact local-C32 topology evidence on canonical C128 keys."""
    target = GRID_UNIFIED
    if target != 128:
        raise RuntimeError("2048 topology consensus requires C128")
    global_anchor_codes = torch.unique(
        global_anchor_codes.detach().cpu().to(torch.int64), sorted=True
    )
    global_xyz = _codes_to_coords(
        global_anchor_codes, target
    )[:, 1:4].to(torch.float64)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(global_xyz.numpy())
    except Exception as exc:
        raise RuntimeError("scipy is required for sparse topology consensus") from exc

    candidate_codes: List[torch.Tensor] = []
    tile_ids: List[torch.Tensor] = []
    candidate32_counts: List[int] = []
    for tile, mapping in zip(
        tiles, shape512_atom_space.local_mappings
    ):
        if tile.anchor32 is None:
            raise RuntimeError(f"tile {tile.tile_id} is missing C32 anchors")
        anchor_codes32 = torch.sort(
            _coordinate_codes(tile.anchor32, GRID_SS)
        ).values
        support_codes32 = _coordinate_codes(mapping.coords, GRID_SS)
        positions = torch.searchsorted(anchor_codes32, support_codes32)
        safe = positions.clamp_max(max(0, anchor_codes32.shape[0] - 1))
        is_anchor = (positions < anchor_codes32.shape[0]) & (
            anchor_codes32.index_select(0, safe) == support_codes32
        )
        candidate_token = ~is_anchor
        candidate32_counts.append(int(candidate_token.sum().item()))
        candidate_edges = candidate_token.index_select(
            0, mapping.token_indices
        )
        codes = torch.unique(
            shape512_atom_space.atom_ids.index_select(
                0, mapping.atom_indices[candidate_edges]
            ),
            sorted=True,
        )
        candidate_codes.append(codes)
        tile_ids.append(
            torch.full((codes.shape[0],), tile.tile_id, dtype=torch.int64)
        )
    if candidate_codes:
        all_codes = torch.cat(candidate_codes)
        all_tile_ids = torch.cat(tile_ids)
        unique_pairs = torch.unique(
            torch.stack([all_codes, all_tile_ids], dim=1), dim=0
        )
        unique_codes, counts = torch.unique(
            unique_pairs[:, 0], return_counts=True
        )
        count_lookup = {
            int(code): int(count)
            for code, count in zip(unique_codes.tolist(), counts.tolist())
        }
    else:
        count_lookup = {}

    records: List[Dict[str, Any]] = []
    accepted_all: List[torch.Tensor] = []
    for tile, codes, candidate32_count in zip(
        tiles, candidate_codes, candidate32_counts
    ):
        if tile.candidate64 is None or tile.anchor64 is None:
            raise RuntimeError(f"tile {tile.tile_id} is missing C64 support")
        xyz = _codes_to_coords(codes, target)[:, 1:4]
        if xyz.numel():
            distance, _ = tree.query(xyz.numpy(), k=1, p=np.inf)
            narrow = torch.from_numpy(distance <= float(surface_band_cells))
        else:
            narrow = torch.zeros(0, dtype=torch.bool)
        consensus = torch.tensor(
            [count_lookup.get(int(code), 0) >= 2 for code in codes.tolist()],
            dtype=torch.bool,
        )
        accepted_codes = codes[consensus | narrow]
        tile.accepted_c128_codes = accepted_codes
        accepted_all.append(accepted_codes)

        candidate64_rows, candidate64_codes = _candidate_global_codes(
            tile,
            tile.candidate64,
            global_camera,
            source_resolution=GRID_SHAPE,
            target_resolution=target,
        )
        accepted64_mask = torch.isin(
            candidate64_codes, accepted_codes
        )
        accepted64 = tile.candidate64.index_select(
            0, candidate64_rows[accepted64_mask]
        )
        tile.support64 = _union_coords(tile.anchor64, accepted64)
        anchor_codes = set(_coordinate_codes(tile.anchor64, GRID_SHAPE).tolist())
        confidence = torch.tensor(
            [
                1.0 if int(code) in anchor_codes else 0.85
                for code in _coordinate_codes(tile.support64, GRID_SHAPE).tolist()
            ],
            dtype=torch.float32,
        )
        tile.candidate_confidence64 = confidence
        records.append(
            {
                "tile_id": int(tile.tile_id),
                "anchor_c128": int(global_anchor_codes.shape[0]),
                "candidate_c32_tokens": int(candidate32_count),
                "candidate_c128_keys": int(codes.shape[0]),
                "accepted_c128_keys": int(accepted_codes.shape[0]),
                "anchor64": int(tile.anchor64.shape[0]),
                "candidate64": int(tile.candidate64.shape[0]),
                "candidate64_mapped_inside": int(candidate64_rows.shape[0]),
                "accepted_candidate64": int(accepted64.shape[0]),
                "accepted_multi_tile": int(consensus.sum().item()),
                "accepted_surface_band": int((narrow & ~consensus).sum().item()),
                "final_support64": int(tile.support64.shape[0]),
            }
        )
    master_codes = torch.unique(
        torch.cat([global_anchor_codes, *accepted_all]), sorted=True
    )
    return _codes_to_coords(master_codes, target), {
        "target_resolution": target,
        "global_anchor_c128_codes": int(global_anchor_codes.shape[0]),
        "final_master_c128_codes": int(master_codes.shape[0]),
        "surface_band_cells": int(surface_band_cells),
        "acceptance": (
            "mandatory global C1024-to-C128 anchor OR exact local-C32 "
            "footprint key with distinct-tile count>=2 OR global-anchor "
            "Chebyshev narrow band"
        ),
        "tiles": records,
    }


def _remove_existing_coords(
    candidates: torch.Tensor,
    existing: torch.Tensor,
    resolution: int,
) -> torch.Tensor:
    existing_codes = torch.sort(_coordinate_codes(existing, resolution)).values
    candidate_codes = _coordinate_codes(candidates, resolution)
    positions = torch.searchsorted(existing_codes, candidate_codes)
    safe = positions.clamp_max(max(0, existing_codes.shape[0] - 1))
    found = (positions < existing_codes.shape[0]) & (
        existing_codes.index_select(0, safe) == candidate_codes
    )
    return candidates[~found]


def _decoded_c1024_to_unified_codes(coords: torch.Tensor) -> torch.Tensor:
    """Use decoded O-Voxel cell-center semantics, not endpoint semantics."""
    xyz = torch.floor(
        (coords[:, 1:4].to(torch.float64) + 0.5)
        / float(GRID_DECODED)
        * float(GRID_UNIFIED)
    ).to(torch.int64)
    valid = ((xyz >= 0) & (xyz < GRID_UNIFIED)).all(dim=1)
    xyz = xyz[valid]
    codes = (
        (xyz[:, 0] * GRID_UNIFIED + xyz[:, 1]) * GRID_UNIFIED
        + xyz[:, 2]
    )
    return torch.unique(codes, sorted=True)


def _nearest_global_c64_parent_rows(
    c128_codes: torch.Tensor,
    global_coords64: torch.Tensor,
) -> torch.Tensor:
    """Assign every native C128 master cell to a coarse global C64 parent."""
    if c128_codes.numel() == 0:
        return torch.zeros(0, dtype=torch.int64)
    c128_xyz = _codes_to_coords(c128_codes, GRID_UNIFIED)[:, 1:4]
    c128_q = (
        (c128_xyz.to(torch.float64) + 0.5)
        * (2.0 / float(GRID_UNIFIED))
        - 1.0
    )
    global_q = endpoint_indices_to_q(
        global_coords64[:, 1:4], GRID_SHAPE
    )
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(global_q.numpy())
        _, rows = tree.query(c128_q.numpy(), k=1, workers=-1)
        return torch.from_numpy(np.asarray(rows, dtype=np.int64))
    except Exception:
        output = []
        for begin in range(0, c128_q.shape[0], 4096):
            output.append(
                torch.cdist(
                    c128_q[begin : begin + 4096], global_q
                ).argmin(dim=1)
            )
        return torch.cat(output)


def _save_native_mesh_checkpoint(mesh: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    layout = {
        key: [int(value.start), int(value.stop)]
        for key, value in mesh.layout.items()
    }
    torch.save(
        {
            "format": "pixal3d_mesh_with_voxel_native_v1",
            "vertices": mesh.vertices.detach().cpu(),
            "faces": mesh.faces.detach().cpu(),
            "origin": torch.as_tensor(mesh.origin).detach().cpu(),
            "voxel_size": torch.as_tensor(mesh.voxel_size).detach().cpu(),
            "coords": mesh.coords.detach().cpu(),
            "attrs": mesh.attrs.detach().cpu(),
            "voxel_shape": list(mesh.voxel_shape),
            "layout": layout,
        },
        path,
    )


def _crop_global_render(
    render_path: Path,
    box: Sequence[int],
    output_path: Path,
) -> Path:
    with Image.open(render_path) as source:
        image = composite_on_black(source)
    x0, y0, x1, y1 = (int(value) for value in box)
    scale_x = image.width / float(IMAGE_CANONICAL)
    scale_y = image.height / float(IMAGE_CANONICAL)
    crop = image.crop(
        (
            int(round(x0 * scale_x)),
            int(round(y0 * scale_y)),
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
        )
    ).resize((IMAGE_1024, IMAGE_1024), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)
    return output_path


def _silhouette_iou(reference_mask: Image.Image, rendered_alpha: Image.Image) -> float:
    reference = np.asarray(
        reference_mask.resize((IMAGE_1024, IMAGE_1024), Image.Resampling.NEAREST)
    ) >= 127
    alpha = np.asarray(
        rendered_alpha.convert("L").resize(
            (IMAGE_1024, IMAGE_1024), Image.Resampling.NEAREST
        )
    ) >= 127
    union = np.logical_or(reference, alpha).sum()
    return 1.0 if union == 0 else float(np.logical_and(reference, alpha).sum() / union)


def _evaluate_global_tile_crops(
    *,
    tiles: Sequence[TileContext],
    baseline_render: Path,
    baseline_alpha: Path,
    unified_render: Path,
    unified_alpha: Path,
    output_dir: Path,
    metric_resolution: int,
    skip_lpips: bool,
    lpips_net: str,
    metric_device: str,
) -> Tuple[List[Dict[str, Any]], Path]:
    evaluator = None
    if not skip_lpips:
        device = torch.device(
            metric_device
            if metric_device.startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )
        evaluator = LPIPSEvaluator(lpips_net, device)
    rows: List[Dict[str, Any]] = []
    panels: List[Image.Image] = []
    try:
        for tile in tiles:
            tile_dir = output_dir / "tiles" / f"tile_{tile.tile_id:02d}" / "evaluation"
            tile_dir.mkdir(parents=True, exist_ok=True)
            reference_path = tile_dir / "reference.png"
            tile.image_1024.save(reference_path)
            baseline_path = _crop_global_render(
                baseline_render, tile.box, tile_dir / "global_baseline.png"
            )
            unified_path = _crop_global_render(
                unified_render, tile.box, tile_dir / "unified_sr.png"
            )
            baseline_alpha_path = _crop_global_render(
                baseline_alpha, tile.box, tile_dir / "global_baseline_alpha.png"
            )
            unified_alpha_path = _crop_global_render(
                unified_alpha, tile.box, tile_dir / "unified_sr_alpha.png"
            )
            target_size = (int(metric_resolution), int(metric_resolution))
            reference_tensor = image_to_tensor(
                composite_on_black(tile.image_1024), target_size
            )
            baseline_tensor = image_to_tensor(
                Image.open(baseline_path).convert("RGB"), target_size
            )
            unified_tensor = image_to_tensor(
                Image.open(unified_path).convert("RGB"), target_size
            )
            baseline_lpips = (
                None
                if evaluator is None
                else evaluator.evaluate(reference_tensor, baseline_tensor)
            )
            unified_lpips = (
                None
                if evaluator is None
                else evaluator.evaluate(reference_tensor, unified_tensor)
            )
            baseline_metrics = {
                "psnr_db": psnr_metric(reference_tensor, baseline_tensor),
                "ssim": ssim_metric(reference_tensor, baseline_tensor),
                "lpips": baseline_lpips,
                "silhouette_iou": _silhouette_iou(
                    tile.foreground, Image.open(baseline_alpha_path)
                ),
            }
            unified_metrics = {
                "psnr_db": psnr_metric(reference_tensor, unified_tensor),
                "ssim": ssim_metric(reference_tensor, unified_tensor),
                "lpips": unified_lpips,
                "silhouette_iou": _silhouette_iou(
                    tile.foreground, Image.open(unified_alpha_path)
                ),
            }
            row = {
                "tile_id": int(tile.tile_id),
                "box": list(tile.box),
                **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
                **{f"unified_{key}": value for key, value in unified_metrics.items()},
                "psnr_gain_db": (
                    unified_metrics["psnr_db"] - baseline_metrics["psnr_db"]
                ),
                "ssim_gain": unified_metrics["ssim"] - baseline_metrics["ssim"],
                "lpips_reduction": (
                    None
                    if baseline_lpips is None or unified_lpips is None
                    else baseline_lpips - unified_lpips
                ),
                "silhouette_iou_gain": (
                    unified_metrics["silhouette_iou"]
                    - baseline_metrics["silhouette_iou"]
                ),
                "reference_png": str(reference_path),
                "baseline_render_png": str(baseline_path),
                "unified_render_png": str(unified_path),
            }
            rows.append(row)
            panel = Image.new("RGB", (IMAGE_1024 * 3, IMAGE_1024 + 42), (18, 18, 18))
            panel.paste(tile.image_1024.convert("RGB"), (0, 42))
            panel.paste(Image.open(baseline_path).convert("RGB"), (IMAGE_1024, 42))
            panel.paste(Image.open(unified_path).convert("RGB"), (2 * IMAGE_1024, 42))
            draw = ImageDraw.Draw(panel)
            draw.text((8, 12), f"tile {tile.tile_id} reference", fill="white")
            draw.text(
                (IMAGE_1024 + 8, 12),
                f"global {baseline_metrics['psnr_db']:.3f} dB",
                fill="white",
            )
            draw.text(
                (2 * IMAGE_1024 + 8, 12),
                f"unified {unified_metrics['psnr_db']:.3f} dB",
                fill="white",
            )
            panels.append(panel.resize((1536, 533), Image.Resampling.LANCZOS))
            _atomic_json(tile_dir / "metrics.json", row)
    finally:
        if evaluator is not None:
            evaluator.model.cpu()
            del evaluator
            _empty_cuda_cache()
    csv_path = output_dir / "tile_crop_metrics.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    contact_path = output_dir / "tile_crop_contact_sheet.png"
    if panels:
        contact = Image.new(
            "RGB",
            (1536, 533 * len(panels)),
            (18, 18, 18),
        )
        for index, panel in enumerate(panels):
            contact.paste(panel, (0, index * 533))
        contact.save(contact_path)
    return rows, contact_path


def _mean_metric(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    return None if not values else float(np.mean(values))


def run(args: argparse.Namespace) -> None:
    """Execute native-window C128 generation followed by direct 2048 decode."""
    if int(args.decode_resolution) != IMAGE_TARGET:
        raise ValueError("--decode-resolution must be exactly 2048")
    expected_input_resolution = int(args.decode_resolution) // 16
    if expected_input_resolution != GRID_UNIFIED or GRID_UNIFIED != 128:
        raise RuntimeError(
            "direct decode invariant failed: expected_input_resolution "
            "must equal GRID_UNIFIED == 128"
        )
    if args.cuda_device is not None:
        torch.cuda.set_device(int(args.cuda_device))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    pipeline = init_pipeline(
        args.model_path,
        device="cuda",
        low_vram=bool(args.low_vram),
    )
    if not bool(args.low_vram):
        for model in pipeline.models.values():
            _move_to_cpu(model)
        _move_to_cpu(
            pipeline.image_cond_model_ss,
            pipeline.image_cond_model_shape_512,
            pipeline.image_cond_model_shape_1024,
            pipeline.image_cond_model_tex_1024,
        )
        pipeline.low_vram = True
        print("[memory] forced stagewise CPU offload for the C128 pipeline")
        _release_cuda("pipeline-models-parked")

    canonical = pipeline.preprocess_canonical_images(Image.open(args.image))
    image_4096: Image.Image = canonical["image_4096"]
    image_1024: Image.Image = canonical["image_1024"]
    image_512: Image.Image = canonical["image_512"]
    foreground_4096: Image.Image = canonical[
        "foreground_mask_4096"
    ].convert("L")
    image_4096.save(output_dir / "canonical_4096.png")
    image_1024.save(output_dir / "canonical_1024.png")
    image_512.save(output_dir / "canonical_512.png")
    foreground_4096.save(output_dir / "canonical_foreground_4096.png")
    _atomic_json(output_dir / "canonical_metadata.json", canonical["metadata"])

    global_camera = evaluation._estimate_camera(
        image_1024=image_1024,
        output_dir=output_dir,
        manual_fov=float(args.fov),
        mesh_scale=float(args.mesh_scale),
        extend_pixel=int(args.extend_pixel),
        image_resolution=int(args.camera_image_resolution),
        moge_model_path=args.moge_model_path,
    )
    _atomic_json(output_dir / "global_camera.json", global_camera)
    boxes = evaluation._tile_layout(
        IMAGE_CANONICAL, IMAGE_1024, IMAGE_1024 // 2
    )
    if len(boxes) != 49:
        raise RuntimeError(f"canonical tile layout must contain 49 tiles, got {len(boxes)}")
    tile_ids = _parse_tile_ids(args.tile_ids, len(boxes))
    if args.max_tiles is not None:
        tile_ids = tile_ids[: int(args.max_tiles)]
    tiles = _prepare_tiles(
        image_4096=image_4096,
        foreground_4096=foreground_4096,
        global_camera=global_camera,
        tile_ids=tile_ids,
        extend_pixel=int(args.extend_pixel),
    )
    for tile in tiles:
        tile_dir = output_dir / "tiles" / f"tile_{tile.tile_id:02d}"
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile.image_1024.save(tile_dir / "reference.png")
        tile.foreground.save(tile_dir / "foreground.png")
        _atomic_json(tile_dir / "camera.json", asdict(tile.transform))

    params = evaluation._sampler_params(args, pipeline)
    config = {
        **vars(args),
        "format": "pixal3d_canonical_endpoint_2048_c128_config_v1",
        "image": str(Path(args.image).expanduser().resolve()),
        "output_dir": str(output_dir),
        "output_description": (
            "canonical 4096 input; native C128/H2048 generation; "
            "direct 2048 decode and render"
        ),
        "canonical_input_resolution": IMAGE_CANONICAL,
        "target_resolution": IMAGE_TARGET,
        "tile_size": IMAGE_1024,
        "tile_stride": IMAGE_1024 // 2,
        "available_tile_count": len(boxes),
        "selected_tile_ids": tile_ids,
        "tile_count": len(tiles),
        "global_camera": global_camera,
        "method": {
            "global_trajectory": (
                "native C64 windows merged as one C128 clean-endpoint master; "
                "never modified by local tiles"
            ),
            "noise": "stateless C128 atom white noise with exact local-C64 overlap",
            "flow_sync": (
                "local-C64 minus native-global-C128 clean endpoint; Huber merge; "
                "coverage-aware zero-C64-projector-mean high-pass"
            ),
            "topology": (
                "local-C32 exact footprints and global C1024 anchors reach "
                "consensus directly on canonical C128 keys"
            ),
            "decode": "one native sparse C128 latent -> one direct 2048 decode",
            "renderer": (
                "native MeshWithVoxel render_utils.render_frames/"
                "PbrMeshRenderer"
            ),
            "stagewise_cpu_offload": True,
            "modified_material_magic_route": False,
            "raw_velocity_average": False,
            "post_generation_coordinate_average": False,
        },
    }
    _atomic_json(output_dir / "effective_config.json", config)
    print(json.dumps(config, ensure_ascii=False, indent=2, default=str))
    memory_stats: Dict[str, Dict[str, int]] = {
        "start": _log_cuda_memory("start")
    }

    # Passes 1-3: canonical 4096 input, shared SS noise, global/local C32.
    global_coords32, ss_diagnostics = _run_shared_sparse_structures(
        pipeline=pipeline,
        image_512=image_512,
        global_camera=global_camera,
        tiles=tiles,
        params=params["ss"],
        seed=int(args.seed),
        atom_chunk_size=int(args.atom_chunk_size),
        output_dir=output_dir,
    )
    local_coords32 = [tile.support32 for tile in tiles]
    if any(value is None for value in local_coords32):
        raise RuntimeError("a tile is missing C32 support")
    local_coords32 = [value for value in local_coords32 if value is not None]
    print(
        f"[tokens] SS global_C32={global_coords32.shape[0]:,} "
        f"local_C32_total={sum(value.shape[0] for value in local_coords32):,}"
    )

    # Pass 4: Shape512 remains a C32 model, with a C128 common atom target.
    atom512 = _build_stage_atom_space(
        stage="shape512",
        global_coords=global_coords32,
        resolution=GRID_SS,
        tiles=tiles,
        local_supports=local_coords32,
        global_camera=global_camera,
        target_resolution=GRID_UNIFIED,
        chunk_size=int(args.atom_chunk_size),
    )
    if atom512.target_resolution != GRID_UNIFIED:
        raise RuntimeError("Shape512 atom target must be C128")
    _save_atom_space(output_dir / "atoms" / "shape512_c128.pt", atom512)
    global_cond512, local_cond512 = _prepare_shape_conditions(
        pipeline=pipeline,
        image_model=pipeline.image_cond_model_shape_512,
        global_image=image_512,
        global_coords=global_coords32,
        global_camera=global_camera,
        tiles=tiles,
        local_supports=local_coords32,
        resolution=GRID_SS,
    )
    shape512 = _run_continuous_stage(
        pipeline=pipeline,
        stage="shape512_c128_atoms",
        sampler=pipeline.shape_slat_sampler,
        model=pipeline.models["shape_slat_flow_model_512"],
        atom_space=atom512,
        global_condition=global_cond512,
        local_conditions=local_cond512,
        params=params["shape"],
        seed=int(args.seed) + 1001,
        tiles=tiles,
        global_camera=global_camera,
        huber_delta=float(args.huber_delta),
        robust_iterations=int(args.robust_iterations),
    )
    _assert_compact_coupled_result("shape512", shape512)
    shape512_diagnostics = {
        "format": "pixal3d_shape512_c128_atoms_v1",
        "atom_space": dict(atom512.diagnostics),
        "elapsed_seconds": shape512.elapsed_seconds,
        "final_fusion": dict(shape512.final_fusion.diagnostics),
        "step_records": list(shape512.step_records),
    }
    _atomic_json(output_dir / "stages" / "shape512_c128.json", shape512_diagnostics)
    global_coords64 = _upsample_shape512(
        pipeline, shape512.global_samples
    ).detach().cpu()
    local_upsampled64 = [
        _upsample_shape512(pipeline, value).detach().cpu()
        for value in shape512.local_samples
    ]
    for tile, candidates in zip(tiles, local_upsampled64):
        _, inverse = _exact_transforms(global_camera, tile)
        anchor64, anchor_stats = _project_global_support_to_local(
            global_coords64,
            global_resolution=GRID_SHAPE,
            local_resolution=GRID_SHAPE,
            inverse=inverse,
        )
        tile.anchor64 = anchor64
        tile.candidate64 = _remove_existing_coords(
            candidates, anchor64, GRID_SHAPE
        )
        _atomic_json(
            output_dir
            / "tiles"
            / f"tile_{tile.tile_id:02d}"
            / "c64_preconsensus.json",
            {
                "anchor": anchor_stats,
                "upsampled_rows": int(candidates.shape[0]),
                "native_candidate_rows": int(tile.candidate64.shape[0]),
            },
        )
    print(
        f"[tokens] Shape512 atoms_C128={atom512.atom_count:,} "
        f"global_support_C64={global_coords64.shape[0]:,}"
    )

    # Ordinary global C64 Shape1024/Texture1024 baseline is generated first.
    shape_model = pipeline.models["shape_slat_flow_model_1024"]
    global_shape_condition64 = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [image_1024],
        global_coords64.to(device),
        camera_angle_x=float(global_camera["camera_angle_x"]),
        distance=float(global_camera["distance"]),
        mesh_scale=float(global_camera["mesh_scale"]),
        grid_resolution_override=GRID_SHAPE,
    )
    _seed_everything(int(args.seed) + 2000)
    baseline_shape_denorm = pipeline.sample_shape_slat(
        global_shape_condition64,
        shape_model,
        global_coords64.to(device),
        dict(params["shape"]),
    )
    del global_shape_condition64
    _release_cuda("ordinary-global-shape-c64-complete")

    texture_model = pipeline.models["tex_slat_flow_model_1024"]
    global_texture_condition64 = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [image_1024],
        baseline_shape_denorm.coords,
        camera_angle_x=float(global_camera["camera_angle_x"]),
        distance=float(global_camera["distance"]),
        mesh_scale=float(global_camera["mesh_scale"]),
        grid_resolution_override=GRID_SHAPE,
    )
    _seed_everything(int(args.seed) + 3000)
    baseline_texture_denorm = pipeline.sample_tex_slat(
        global_texture_condition64,
        texture_model,
        baseline_shape_denorm,
        dict(params["texture"]),
    )
    del global_texture_condition64
    _release_cuda("ordinary-global-texture-c64-complete")
    if not torch.equal(
        baseline_shape_denorm.coords, baseline_texture_denorm.coords
    ):
        raise RuntimeError("ordinary C64 baseline shape/texture supports differ")

    # Learned global surface anchors are quantized directly C1024 -> C128.
    global_c1024, global_upsample_stats = (
        evaluation._learned_upsample_shape1024_to_c1024(
            pipeline, baseline_shape_denorm
        )
    )
    global_upsample_stats = dict(global_upsample_stats)
    if "unique_c128_tokens" in global_upsample_stats:
        global_upsample_stats["unique_c1024_tokens"] = (
            global_upsample_stats.pop("unique_c128_tokens")
        )
    global_c1024 = global_c1024.detach().cpu()
    global_anchor_codes128 = _decoded_c1024_to_unified_codes(global_c1024)
    if global_anchor_codes128.numel() > int(args.max_final_tokens):
        raise RuntimeError(
            f"global C128 anchors={global_anchor_codes128.numel():,} exceed "
            f"--max-final-tokens={int(args.max_final_tokens):,}"
        )

    # Local C32 evidence and global anchors meet only on canonical C128 keys.
    master_coords128, topology = _topology_consensus_c128(
        shape512_atom_space=atom512,
        global_anchor_codes=global_anchor_codes128,
        tiles=tiles,
        global_camera=global_camera,
        surface_band_cells=int(args.surface_band_cells),
    )
    if master_coords128.shape[0] > int(args.max_final_tokens):
        raise RuntimeError(
            f"final C128 support={master_coords128.shape[0]:,} exceeds "
            f"--max-final-tokens={int(args.max_final_tokens):,}"
        )
    _atomic_json(output_dir / "stages" / "topology_c128.json", topology)
    local_coords64 = [tile.support64 for tile in tiles]
    if any(value is None for value in local_coords64):
        raise RuntimeError("a tile is missing accepted C64 flow support")
    local_coords64 = [value for value in local_coords64 if value is not None]

    # Shape512 is now fully consumed.
    del (
        shape512,
        global_cond512,
        local_cond512,
        global_coords32,
        local_coords32,
        local_upsampled64,
    )
    _move_to_cpu(
        pipeline.models["shape_slat_flow_model_512"],
        pipeline.image_cond_model_shape_512,
    )
    for tile in tiles:
        tile.native32 = None
        tile.anchor32 = None
        tile.support32 = None
    memory_stats["shape512_released"] = _release_cuda("shape512-released")

    # Fixed master support is shared byte-for-byte by shape and texture.
    atom128 = _build_c128_master_space(
        stage="shape1024_texture1024_c128_master",
        master_coords128=master_coords128,
        global_coords64=global_coords64,
        tiles=tiles,
        local_supports64=local_coords64,
        global_camera=global_camera,
        chunk_size=int(args.atom_chunk_size),
    )
    del atom512
    _save_c128_master_space(output_dir / "atoms" / "c128_master.pt", atom128)
    master_coords128 = atom128.atom_coords
    xyz128 = master_coords128[:, 1:4]
    if bool(((xyz128 < 0) | (xyz128 >= GRID_UNIFIED)).any().item()):
        raise RuntimeError("final C128 coordinates must lie in [0,127]")
    windows = _native_c128_windows(pipeline, master_coords128)
    print(
        f"[tokens] topology_C128={master_coords128.shape[0]:,} "
        f"local_C64_total={sum(m.coords.shape[0] for m in atom128.local_mappings):,} "
        f"native_windows={len(windows)}"
    )

    # Shape1024: native C64 windows form the C128 global master while tile
    # C64 endpoints contribute only exact-footprint high-frequency residual.
    global_shape_condition128, local_shape_conditions64 = (
        _prepare_c128_master_local_conditions(
            pipeline=pipeline,
            image_model=pipeline.image_cond_model_shape_1024,
            global_image=image_1024,
            master_coords128=master_coords128,
            global_camera=global_camera,
            tiles=tiles,
            atom_space=atom128,
        )
    )
    shape1024 = _run_c128_stage(
        pipeline=pipeline,
        stage="shape1024_c128_master",
        sampler=pipeline.shape_slat_sampler,
        model=shape_model,
        atom_space=atom128,
        windows=windows,
        global_condition=global_shape_condition128,
        local_conditions=local_shape_conditions64,
        params=params["shape"],
        seed=int(args.seed) + 2001,
        tiles=tiles,
        global_camera=global_camera,
        huber_delta=float(args.huber_delta),
        robust_iterations=int(args.robust_iterations),
    )
    shape1024_diagnostics = {
        "format": "pixal3d_shape1024_native_c128_master_v1",
        "global_master_tokens_c128": int(master_coords128.shape[0]),
        "local_tokens_c64": [
            int(mapping.coords.shape[0]) for mapping in atom128.local_mappings
        ],
        "native_window_count": len(windows),
        "elapsed_seconds": shape1024.elapsed_seconds,
        "final_fusion": dict(shape1024.final_fusion.diagnostics),
        "step_records": list(shape1024.step_records),
    }
    _atomic_json(
        output_dir / "stages" / "shape1024_c128.json",
        shape1024_diagnostics,
    )
    shape_global_final = shape1024.global_samples
    shape_local_final = shape1024.local_samples
    shape_final_fusion = shape1024.final_fusion
    del shape1024
    del global_shape_condition128, local_shape_conditions64
    _move_to_cpu(shape_model, pipeline.image_cond_model_shape_1024)
    memory_stats["shape1024_released"] = _release_cuda(
        "shape1024-model-and-conditions-released"
    )

    # Texture1024 uses exactly the same C128 support and native windows.
    global_texture_condition128, local_texture_conditions64 = (
        _prepare_c128_master_local_conditions(
            pipeline=pipeline,
            image_model=pipeline.image_cond_model_tex_1024,
            global_image=image_1024,
            master_coords128=master_coords128,
            global_camera=global_camera,
            tiles=tiles,
            atom_space=atom128,
        )
    )
    texture1024 = _run_c128_stage(
        pipeline=pipeline,
        stage="texture1024_c128_master",
        sampler=pipeline.tex_slat_sampler,
        model=texture_model,
        atom_space=atom128,
        windows=windows,
        global_condition=global_texture_condition128,
        local_conditions=local_texture_conditions64,
        params=params["texture"],
        seed=int(args.seed) + 3001,
        tiles=tiles,
        global_camera=global_camera,
        huber_delta=float(args.huber_delta),
        robust_iterations=int(args.robust_iterations),
        global_concat=shape_global_final,
        local_concat=shape_local_final,
    )
    texture1024_diagnostics = {
        "format": "pixal3d_texture1024_native_c128_master_v1",
        "global_master_tokens_c128": int(master_coords128.shape[0]),
        "native_window_count": len(windows),
        "noise_namespace": "noise/texture1024_c128_master",
        "elapsed_seconds": texture1024.elapsed_seconds,
        "final_fusion": dict(texture1024.final_fusion.diagnostics),
        "step_records": list(texture1024.step_records),
    }
    _atomic_json(
        output_dir / "stages" / "texture1024_c128.json",
        texture1024_diagnostics,
    )
    texture_global_final = texture1024.global_samples
    texture_local_final = texture1024.local_samples
    texture_final_fusion = texture1024.final_fusion
    del texture1024
    del global_texture_condition128, local_texture_conditions64
    _move_to_cpu(texture_model, pipeline.image_cond_model_tex_1024)
    memory_stats["texture1024_released"] = _release_cuda(
        "texture1024-model-and-conditions-released"
    )

    coords128 = atom128.atom_coords.to(device)
    unified_shape_norm = SparseTensor(
        feats=shape_final_fusion.unified_atoms,
        coords=coords128,
    )
    unified_texture_norm = SparseTensor(
        feats=texture_final_fusion.unified_atoms,
        coords=coords128,
    )
    if not torch.equal(unified_shape_norm.coords, unified_texture_norm.coords):
        raise RuntimeError("final C128 shape/texture supports differ")
    if bool(
        (
            (unified_shape_norm.coords[:, 1:4] < 0)
            | (unified_shape_norm.coords[:, 1:4] >= 128)
        ).any().item()
    ):
        raise RuntimeError("final shape/texture coordinates are outside [0,127]")
    unified_shape_denorm = _denormalize(
        unified_shape_norm, pipeline.shape_slat_normalization
    )
    unified_texture_denorm = _denormalize(
        unified_texture_norm, pipeline.tex_slat_normalization
    )
    if not torch.equal(
        unified_shape_denorm.coords, unified_texture_denorm.coords
    ):
        raise RuntimeError("denormalized C128 shape/texture supports differ")

    master_codes = atom128.atom_ids
    anchor_present = torch.isin(global_anchor_codes128, master_codes)
    if not bool(anchor_present.all().item()):
        raise RuntimeError("a global C1024-derived C128 anchor was lost")
    detail_codes = int(master_codes.shape[0] - global_anchor_codes128.shape[0])
    final_support_summary = {
        "format": "pixal3d_final_c128_support_v1",
        "global_c1024_rows": int(global_c1024.shape[0]),
        "global_anchor_c128_codes": int(global_anchor_codes128.shape[0]),
        "topology_detail_c128_codes": detail_codes,
        "final_c128_tokens": int(master_coords128.shape[0]),
        "represented_global_c64_parents": int(
            torch.unique(atom128.coarse_parent).shape[0]
        ),
        "missing_global_anchor_c128_codes": 0,
        "missing_global_anchor_policy": (
            "all global anchors are mandatory native C128 master cells; "
            "topology births use nearest active global C64 only as the "
            "coarse-projector grouping, never as a copied latent"
        ),
        "global_shape1024_to_c1024": global_upsample_stats,
        "logical_grid": [128, 128, 128],
        "effective_bandwidth": [128, 128, 64],
        "coordinate_semantics": (
            "global decoded C1024 uses cell centers; final flow/decode support "
            "uses native C128 cells; local flow uses endpoint C64 footprints"
        ),
        "z_policy": (
            "local C64 evidence remains low-pass along z and spans multiple "
            "logical C128 cells; no independent z samples are fabricated"
        ),
        "shape_final_fusion": dict(shape_final_fusion.diagnostics),
        "texture_final_fusion": dict(texture_final_fusion.diagnostics),
    }
    _atomic_json(
        output_dir / "stages" / "final_support_c128.json",
        final_support_summary,
    )
    print(
        f"[tokens] final_shape_C128={unified_shape_norm.coords.shape[0]:,} "
        f"final_texture_C128={unified_texture_norm.coords.shape[0]:,}"
    )

    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "pixal3d_canonical_endpoint_2048_c128_latents_v1",
            "coords128": coords128.detach().cpu(),
            "shape_normalized": unified_shape_norm.feats.detach().cpu(),
            "shape_denormalized": unified_shape_denorm.feats.detach().cpu(),
            "texture_normalized": unified_texture_norm.feats.detach().cpu(),
            "texture_denormalized": unified_texture_denorm.feats.detach().cpu(),
            "baseline_global_coords64": baseline_shape_denorm.coords.detach().cpu(),
            "baseline_global_shape_denormalized": (
                baseline_shape_denorm.feats.detach().cpu()
            ),
            "baseline_global_texture_denormalized": (
                baseline_texture_denorm.feats.detach().cpu()
            ),
            "support_summary": final_support_summary,
        },
        traces_dir / "unified_c128_latents.pt",
    )

    summary: Dict[str, Any] = {
        "format": "pixal3d_canonical_endpoint_2048_c128_v1",
        "image": str(Path(args.image).expanduser().resolve()),
        "canonical_input_resolution": IMAGE_CANONICAL,
        "target_decode_resolution": IMAGE_TARGET,
        "tiles": tile_ids,
        "global_camera": global_camera,
        "ss": ss_diagnostics,
        "topology_c128": topology,
        "shape512_final": shape512_diagnostics["final_fusion"],
        "shape1024_c128_final": shape1024_diagnostics["final_fusion"],
        "texture1024_c128_final": texture1024_diagnostics["final_fusion"],
        "final_support": final_support_summary,
        "comparison_protocol": (
            "native MeshWithVoxel/PbrMeshRenderer full 2048 render against "
            "canonical 4096 input resized by the evaluator; tile crops use "
            "the original canonical boxes by scale; modified material route excluded"
        ),
        "direct_decode": "C128 -> 2048",
        "post_generation_coordinate_average": False,
    }

    # Preserve only the four denormalized decode inputs on CUDA.
    del (
        unified_shape_norm,
        unified_texture_norm,
        shape_global_final,
        shape_local_final,
        shape_final_fusion,
        texture_global_final,
        texture_local_final,
        texture_final_fusion,
        atom128,
        windows,
        coords128,
        master_coords128,
        global_anchor_codes128,
        global_c1024,
        global_coords64,
        local_coords64,
        shape_model,
        texture_model,
    )
    for tile in tiles:
        tile.candidate64 = None
        tile.support64 = None
        tile.candidate_confidence64 = None
        tile.accepted_c128_codes = None
    _assert_no_cuda_tensors(
        "decode-guard-metadata",
        {
            "params": params,
            "ss": ss_diagnostics,
            "topology": topology,
            "shape512": shape512_diagnostics,
            "shape1024": shape1024_diagnostics,
            "texture1024": texture1024_diagnostics,
            "support": final_support_summary,
        },
    )
    memory_stats["decode_ready"] = _release_cuda("decode-ready-four-latents")
    summary["cuda_memory"] = memory_stats

    if not bool(args.decode):
        del (
            baseline_shape_denorm,
            baseline_texture_denorm,
            unified_shape_denorm,
            unified_texture_denorm,
        )
        _release_cuda("no-decode-latents-released")
        _atomic_json(output_dir / "summary.json", summary)
        print(f"[summary] {output_dir / 'summary.json'}")
        return

    if int(args.decode_resolution) // 16 != GRID_UNIFIED:
        raise RuntimeError("decode must consume native C128 coordinates directly")
    if not torch.equal(
        unified_shape_denorm.coords, unified_texture_denorm.coords
    ):
        raise RuntimeError("direct decode shape/texture supports differ")
    print(
        f"[decode] ordinary global C64 -> {IMAGE_1024} baseline; "
        f"tokens={baseline_shape_denorm.coords.shape[0]:,}"
    )
    baseline_meshes = pipeline.decode_latent(
        baseline_shape_denorm,
        baseline_texture_denorm,
        IMAGE_1024,
    )
    baseline_mesh = baseline_meshes[0]
    _save_native_mesh_checkpoint(
        baseline_mesh, output_dir / "global_baseline" / "mesh_with_voxel.pt"
    )
    baseline_envmap = load_envmap(args.envmap, device=device)
    baseline_eval = render_and_evaluate_mesh(
        baseline_mesh,
        camera_angle_x=float(global_camera["camera_angle_x"]),
        distance=float(global_camera["distance"]),
        output_dir=output_dir / "global_baseline" / "aligned_eval",
        reference_image=output_dir / "canonical_4096.png",
        resolution=int(args.render_resolution),
        metric_resolution=int(args.metric_resolution),
        envmap=baseline_envmap,
        envmap_name=str(args.envmap),
        ssaa=int(args.render_ssaa),
        peel_layers=int(args.render_peel_layers),
        face_chunk_size=int(args.render_face_chunk_size),
        use_envmap_bg=bool(args.use_envmap_bg),
        lpips_net=str(args.lpips_net),
        metric_device=str(args.metric_device),
        skip_lpips=bool(args.skip_lpips),
    )
    del (
        baseline_meshes,
        baseline_mesh,
        baseline_shape_denorm,
        baseline_texture_denorm,
        baseline_envmap,
    )
    _move_to_cpu(
        pipeline.models["shape_slat_decoder"],
        pipeline.models["tex_slat_decoder"],
    )
    memory_stats["baseline_render_released"] = _release_cuda(
        "baseline-mesh-latents-envmap-released"
    )

    print(
        f"[decode] direct native C128 -> {IMAGE_TARGET}; "
        f"tokens={unified_shape_denorm.coords.shape[0]:,}"
    )
    unified_meshes = pipeline.decode_latent(
        unified_shape_denorm,
        unified_texture_denorm,
        IMAGE_TARGET,
    )
    unified_mesh = unified_meshes[0]
    _save_native_mesh_checkpoint(
        unified_mesh, output_dir / "unified_sr" / "mesh_with_voxel.pt"
    )
    unified_envmap = load_envmap(args.envmap, device=device)
    unified_eval = render_and_evaluate_mesh(
        unified_mesh,
        camera_angle_x=float(global_camera["camera_angle_x"]),
        distance=float(global_camera["distance"]),
        output_dir=output_dir / "unified_sr" / "aligned_eval",
        reference_image=output_dir / "canonical_4096.png",
        resolution=int(args.render_resolution),
        metric_resolution=int(args.metric_resolution),
        envmap=unified_envmap,
        envmap_name=str(args.envmap),
        ssaa=int(args.render_ssaa),
        peel_layers=int(args.render_peel_layers),
        face_chunk_size=int(args.render_face_chunk_size),
        use_envmap_bg=bool(args.use_envmap_bg),
        lpips_net=str(args.lpips_net),
        metric_device=str(args.metric_device),
        skip_lpips=bool(args.skip_lpips),
    )
    del (
        unified_meshes,
        unified_mesh,
        unified_shape_denorm,
        unified_texture_denorm,
        unified_envmap,
    )
    _move_to_cpu(
        pipeline.models["shape_slat_decoder"],
        pipeline.models["tex_slat_decoder"],
    )
    memory_stats["unified_render_released"] = _release_cuda(
        "unified-mesh-latents-envmap-released"
    )

    tile_rows, contact_sheet = _evaluate_global_tile_crops(
        tiles=tiles,
        baseline_render=Path(baseline_eval["render_png"]),
        baseline_alpha=Path(baseline_eval["render_outputs"]["alpha"]),
        unified_render=Path(unified_eval["render_png"]),
        unified_alpha=Path(unified_eval["render_outputs"]["alpha"]),
        output_dir=output_dir,
        metric_resolution=int(args.metric_resolution),
        skip_lpips=bool(args.skip_lpips),
        lpips_net=str(args.lpips_net),
        metric_device=str(args.metric_device),
    )
    summary.update(
        {
            "global_baseline_metrics": {
                key: baseline_eval.get(key)
                for key in ("psnr_db", "ssim", "lpips")
            },
            "unified_sr_metrics": {
                key: unified_eval.get(key)
                for key in ("psnr_db", "ssim", "lpips")
            },
            "global_metric_delta": {
                "psnr_gain_db": (
                    float(unified_eval["psnr_db"])
                    - float(baseline_eval["psnr_db"])
                ),
                "ssim_gain": (
                    float(unified_eval["ssim"])
                    - float(baseline_eval["ssim"])
                ),
                "lpips_reduction": (
                    None
                    if baseline_eval["lpips"] is None
                    else float(baseline_eval["lpips"])
                    - float(unified_eval["lpips"])
                ),
            },
            "mean_tile_metrics": {
                key: _mean_metric(tile_rows, key)
                for key in (
                    "baseline_psnr_db",
                    "unified_psnr_db",
                    "psnr_gain_db",
                    "baseline_ssim",
                    "unified_ssim",
                    "ssim_gain",
                    "baseline_lpips",
                    "unified_lpips",
                    "lpips_reduction",
                    "baseline_silhouette_iou",
                    "unified_silhouette_iou",
                )
            },
            "tile_crop_metrics_csv": str(output_dir / "tile_crop_metrics.csv"),
            "tile_contact_sheet": str(contact_sheet),
            "baseline_render": baseline_eval,
            "unified_render": unified_eval,
            "cuda_memory": memory_stats,
        }
    )
    _atomic_json(output_dir / "summary.json", summary)
    print(f"[summary] {output_dir / 'summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--moge-model-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--tile-ids", default=None)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--fov", type=float, default=-1.0)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument("--camera-image-resolution", type=int, default=1024)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--ss-guidance-strength", type=float, default=7.5)
    parser.add_argument("--ss-guidance-rescale", type=float, default=0.7)
    parser.add_argument("--ss-rescale-t", type=float, default=5.0)
    parser.add_argument("--shape-steps", type=int, default=12)
    parser.add_argument("--shape-guidance-strength", type=float, default=7.5)
    parser.add_argument("--shape-guidance-rescale", type=float, default=0.5)
    parser.add_argument("--shape-rescale-t", type=float, default=3.0)
    parser.add_argument("--texture-steps", type=int, default=12)
    parser.add_argument("--texture-guidance-strength", type=float, default=1.0)
    parser.add_argument("--texture-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--texture-rescale-t", type=float, default=3.0)

    parser.add_argument("--atom-chunk-size", type=int, default=2048)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--robust-iterations", type=int, default=3)
    parser.add_argument("--surface-band-cells", type=int, default=2)
    parser.add_argument("--max-final-tokens", type=int, default=262144)
    parser.add_argument("--decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--decode-resolution", type=int, choices=(2048,), default=2048
    )

    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--render-resolution", type=int, default=2048)
    parser.add_argument("--metric-resolution", type=int, default=1024)
    parser.add_argument("--render-ssaa", type=int, default=2)
    parser.add_argument("--render-peel-layers", type=int, default=8)
    parser.add_argument("--render-face-chunk-size", type=int, default=1000000)
    parser.add_argument(
        "--use-envmap-bg", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--metric-device", default="cuda")
    parser.add_argument("--skip-lpips", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_tiles is not None and int(args.max_tiles) < 1:
        raise ValueError("--max-tiles must be positive")
    if int(args.atom_chunk_size) < 1:
        raise ValueError("--atom-chunk-size must be positive")
    if int(args.robust_iterations) < 0:
        raise ValueError("--robust-iterations must be non-negative")
    if float(args.huber_delta) <= 0:
        raise ValueError("--huber-delta must be positive")
    if int(args.max_final_tokens) < 1:
        raise ValueError("--max-final-tokens must be positive")
    if int(args.decode_resolution) != 2048:
        raise ValueError("--decode-resolution must be exactly 2048")
    if int(args.decode_resolution) // 16 != GRID_UNIFIED:
        raise RuntimeError("decode input resolution must be C128")
    run(args)


if __name__ == "__main__":
    main()
