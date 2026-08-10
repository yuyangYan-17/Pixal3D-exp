#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pixal3D global-C256 / projected-tile-C64 3D Haar velocity fusion.

This experiment preserves Pixal3D's ordinary global prior:

    full image -> SS C32 -> shape512 -> learned C64 -> shape1024
    -> decoder subdivision C1024 -> quantized global C256 support

Only 1024 image crops are passed to the local branches.  Each global C256
token is projected through the exact global-camera/crop/local-camera transform
and quantized to a tile-local C64 token.  Shape and texture then use one global
C256 state.  At every Euler step:

1. predict the ordinary full-image global C256 velocity;
2. transport the current global x_t to every tile-local C64 support;
3. predict local C64 velocities from the corresponding image crops;
4. transport all local velocities back to global C256, averaging overlaps;
5. scatter the global and local-union velocities to [B,C,256,256,256];
6. apply one-level orthonormal 3D Haar DWT, producing
   [B,C,8,128,128,128];
7. retain global LLL and replace the other seven bands with local bands;
8. invert the DWT, gather only active global C256 tokens, and update x_t.

Texture repeats the same synchronized flow while using the completed global
shape latent (and its transported local views) as concat conditioning.  The
final global C256 shape/texture latents are decoded once at resolution 4096,
rendered with Pixal3D's official renderer, and evaluated against the canonical
4096 image.

The exact camera and evaluation helpers are shared with
``pixal3d_projective_tile_generation_eval_projected_c64_only.py``.  That module
is not modified.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
)
os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "1")

import torch
from PIL import Image

import pixal3d_projective_tile_generation_eval_projected_c64_only as base
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from render_pixal3d_raw_ovoxel import load_envmap, render_and_evaluate_mesh


GRID_SS = 32
GRID_COARSE = 64
GRID_GLOBAL = 256
GRID_LOCAL = 64
GRID_SUBDIVIDED = 1024
IMAGE_CANONICAL = 4096
IMAGE_FLOW = 1024
IMAGE_LR = 512
DECODE_GLOBAL = 4096
DEFAULT_TILE_SIZE = 1024
DEFAULT_TILE_STRIDE = 512
HAAR_BANDS = (
    "LLL",
    "LLH",
    "LHL",
    "LHH",
    "HLL",
    "HLH",
    "HHL",
    "HHH",
)


@dataclass
class TileTransportC256:
    """Fixed direct correspondence between global C256 and one local C64."""

    tile_id: int
    box: Tuple[int, int, int, int]
    transform: base.TileCameraTransform
    local_coords: torch.Tensor
    edge_global: torch.Tensor
    edge_local: torch.Tensor
    edge_forward_weight: torch.Tensor
    global_token_rows: torch.Tensor
    stats: Dict[str, Any]
    condition_cpu: Optional[Dict[str, Dict[str, torch.Tensor]]] = None


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _quantize_global_c1024_to_c256(
    coords1024: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Quantize decoder-subdivided C1024 points onto fixed global C256."""
    if coords1024.ndim != 2 or coords1024.shape[1] != 4:
        raise ValueError(
            f"global C1024 coordinates must be [N,4], got {coords1024.shape}"
        )
    if bool((coords1024[:, 0] != 0).any().item()):
        raise ValueError("only batch zero is supported")
    xyz1024 = coords1024[:, 1:4].to(torch.float32)
    if bool(
        ((xyz1024 < 0) | (xyz1024 >= GRID_SUBDIVIDED)).any().item()
    ):
        raise ValueError("global C1024 support contains out-of-range points")
    xyz256 = torch.floor(
        (xyz1024 + 0.5)
        / float(GRID_SUBDIVIDED)
        * float(GRID_GLOBAL)
    ).to(torch.int32)
    per_source = torch.cat(
        [coords1024[:, :1].to(torch.int32), xyz256],
        dim=1,
    )
    if bool(
        ((per_source[:, 1:] < 0) | (per_source[:, 1:] >= GRID_GLOBAL))
        .any()
        .item()
    ):
        raise RuntimeError("C1024 -> C256 quantization produced invalid indices")
    coords256, source_to_global = torch.unique(
        per_source,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    counts = torch.bincount(
        source_to_global,
        minlength=coords256.shape[0],
    )
    stats = {
        "quantization": "floor((c1024 + 0.5) / 1024 * 256), then unique",
        "source_c1024_points": int(coords1024.shape[0]),
        "global_c256_tokens": int(coords256.shape[0]),
        "merged_source_rows": int(coords1024.shape[0] - coords256.shape[0]),
        "sources_per_token_min": int(counts.min().item()),
        "sources_per_token_mean": float(counts.float().mean().item()),
        "sources_per_token_max": int(counts.max().item()),
    }
    return coords256, source_to_global.to(torch.long), stats


def _haar_analysis_axis(
    value: torch.Tensor,
    axis: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    moved = value.movedim(axis, -1)
    if moved.shape[-1] % 2:
        raise ValueError(
            f"Haar DWT axis length must be even, got {moved.shape[-1]}"
        )
    scale = math.sqrt(2.0)
    low = (moved[..., 0::2] + moved[..., 1::2]) / scale
    high = (moved[..., 0::2] - moved[..., 1::2]) / scale
    return low.movedim(-1, axis), high.movedim(-1, axis)


def _haar_synthesis_axis(
    low: torch.Tensor,
    high: torch.Tensor,
    axis: int,
) -> torch.Tensor:
    if low.shape != high.shape:
        raise ValueError(f"Haar low/high shapes differ: {low.shape}, {high.shape}")
    low_moved = low.movedim(axis, -1)
    high_moved = high.movedim(axis, -1)
    scale = math.sqrt(2.0)
    even = (low_moved + high_moved) / scale
    odd = (low_moved - high_moved) / scale
    output = torch.empty(
        (*low_moved.shape[:-1], low_moved.shape[-1] * 2),
        device=low.device,
        dtype=low.dtype,
    )
    output[..., 0::2] = even
    output[..., 1::2] = odd
    return output.movedim(-1, axis)


def _haar_dwt3d_stacked(value: torch.Tensor) -> torch.Tensor:
    """Return one-level 3D Haar bands as [B,C,8,X/2,Y/2,Z/2]."""
    if value.ndim != 5:
        raise ValueError(f"expected dense [B,C,X,Y,Z], got {value.shape}")
    x_low, x_high = _haar_analysis_axis(value, 2)
    ll, lh = _haar_analysis_axis(x_low, 3)
    hl, hh = _haar_analysis_axis(x_high, 3)
    lll, llh = _haar_analysis_axis(ll, 4)
    lhl, lhh = _haar_analysis_axis(lh, 4)
    hll, hlh = _haar_analysis_axis(hl, 4)
    hhl, hhh = _haar_analysis_axis(hh, 4)
    return torch.stack(
        (lll, llh, lhl, lhh, hll, hlh, hhl, hhh),
        dim=2,
    )


def _haar_idwt3d_stacked(bands: torch.Tensor) -> torch.Tensor:
    """Invert stacked bands [B,C,8,X,Y,Z] to [B,C,2X,2Y,2Z]."""
    if bands.ndim != 6 or bands.shape[2] != 8:
        raise ValueError(
            f"expected Haar bands [B,C,8,X,Y,Z], got {bands.shape}"
        )
    ll = _haar_synthesis_axis(bands[:, :, 0], bands[:, :, 1], 4)
    lh = _haar_synthesis_axis(bands[:, :, 2], bands[:, :, 3], 4)
    hl = _haar_synthesis_axis(bands[:, :, 4], bands[:, :, 5], 4)
    hh = _haar_synthesis_axis(bands[:, :, 6], bands[:, :, 7], 4)
    x_low = _haar_synthesis_axis(ll, lh, 3)
    x_high = _haar_synthesis_axis(hl, hh, 3)
    return _haar_synthesis_axis(x_low, x_high, 2)


def _wavelet_self_test(device: torch.device) -> Dict[str, float]:
    generator = torch.Generator(device=device)
    generator.manual_seed(20260728)
    global_dense = torch.randn(
        (1, 3, 8, 8, 8),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    local_dense = torch.randn(
        global_dense.shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    global_bands = _haar_dwt3d_stacked(global_dense)
    local_bands = _haar_dwt3d_stacked(local_dense)
    roundtrip = _haar_idwt3d_stacked(global_bands)
    fused_bands = local_bands.clone()
    fused_bands[:, :, 0] = global_bands[:, :, 0]
    fused_dense = _haar_idwt3d_stacked(fused_bands)
    check_bands = _haar_dwt3d_stacked(fused_dense)
    roundtrip_error = float((roundtrip - global_dense).abs().max().item())
    low_error = float(
        (check_bands[:, :, 0] - global_bands[:, :, 0]).abs().max().item()
    )
    high_error = float(
        (check_bands[:, :, 1:] - local_bands[:, :, 1:]).abs().max().item()
    )
    tolerance = 2e-6
    if max(roundtrip_error, low_error, high_error) >= tolerance:
        raise RuntimeError(
            "3D Haar self-test failed: "
            f"roundtrip={roundtrip_error:.8e}, low={low_error:.8e}, "
            f"high={high_error:.8e}, required<{tolerance:.8e}"
        )
    return {
        "roundtrip_max_abs_error": roundtrip_error,
        "global_lll_invariant_max_abs_error": low_error,
        "local_high_band_invariant_max_abs_error": high_error,
        "tolerance_exclusive": tolerance,
    }


def _scatter_active_velocity_c256(
    features: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    """Scatter active sparse features to dense [B,C,256,256,256]."""
    if features.ndim != 2 or coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("features/coords must be [T,C] and [T,4]")
    if features.shape[0] != coords.shape[0]:
        raise ValueError("feature and coordinate counts differ")
    if bool((coords[:, 0] != 0).any().item()):
        raise ValueError("dense wavelet fusion currently supports batch zero")
    xyz = coords[:, 1:4].to(torch.long)
    if bool(((xyz < 0) | (xyz >= GRID_GLOBAL)).any().item()):
        raise ValueError("global coordinates are outside C256")
    dense = torch.zeros(
        (1, features.shape[1], GRID_GLOBAL, GRID_GLOBAL, GRID_GLOBAL),
        device=features.device,
        dtype=torch.float32,
    )
    linear = (xyz[:, 0] * GRID_GLOBAL + xyz[:, 1]) * GRID_GLOBAL + xyz[:, 2]
    dense.view(1, features.shape[1], -1)[0, :, linear] = (
        features.to(torch.float32).transpose(0, 1)
    )
    return dense


def _gather_active_velocity_c256(
    dense: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    """Gather active C256 coordinates from dense [B,C,256,256,256]."""
    expected = (GRID_GLOBAL, GRID_GLOBAL, GRID_GLOBAL)
    if dense.ndim != 5 or dense.shape[0] != 1:
        raise ValueError(f"expected dense [1,C,256,256,256], got {dense.shape}")
    if tuple(dense.shape[2:]) != expected:
        raise ValueError(f"dense spatial shape must be {expected}")
    xyz = coords[:, 1:4].to(torch.long)
    linear = (xyz[:, 0] * GRID_GLOBAL + xyz[:, 1]) * GRID_GLOBAL + xyz[:, 2]
    return (
        dense.reshape(1, dense.shape[1], -1)[0, :, linear]
        .transpose(0, 1)
        .contiguous()
    )


def _fuse_velocity_wavelet_c256(
    global_velocity: torch.Tensor,
    local_union_velocity: torch.Tensor,
    coords256: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Use global LLL and local seven high bands, then gather active C256."""
    global_dense = _scatter_active_velocity_c256(global_velocity, coords256)
    local_dense = _scatter_active_velocity_c256(local_union_velocity, coords256)
    global_bands = _haar_dwt3d_stacked(global_dense)
    local_bands = _haar_dwt3d_stacked(local_dense)
    if tuple(global_bands.shape[2:]) != (8, 128, 128, 128):
        raise RuntimeError(
            f"unexpected C256 Haar shape {tuple(global_bands.shape)}"
        )
    fused_bands = local_bands
    fused_bands[:, :, 0] = global_bands[:, :, 0]
    global_lll_rms = float(
        global_bands[:, :, 0].square().mean().sqrt().item()
    )
    local_high_rms = float(
        fused_bands[:, :, 1:].square().mean().sqrt().item()
    )
    del global_dense, local_dense, global_bands
    fused_dense = _haar_idwt3d_stacked(fused_bands)
    fused_sparse = _gather_active_velocity_c256(fused_dense, coords256)
    stats = {
        "dense_shape": [1, int(global_velocity.shape[1]), 256, 256, 256],
        "wavelet_shape": [
            1,
            int(global_velocity.shape[1]),
            8,
            128,
            128,
            128,
        ],
        "low_frequency_band": "LLL",
        "low_frequency_source": "global",
        "high_frequency_source": "local_union",
        "global_lll_rms": global_lll_rms,
        "local_high_band_rms": local_high_rms,
        "fused_active_velocity_norm": float(fused_sparse.norm().item()),
    }
    del fused_bands, fused_dense
    return fused_sparse, stats


def _build_tile_transport_c256(
    *,
    global_coords256: torch.Tensor,
    selected_global_rows: torch.Tensor,
    global_camera: Mapping[str, float],
    transform: base.TileCameraTransform,
    output_dir: Path,
    boundary_epsilon: float,
) -> TileTransportC256:
    """Create and persist exact global-C256/local-C64 correspondence."""
    selected_global_rows = selected_global_rows.to(
        device=global_coords256.device,
        dtype=torch.long,
    )
    if selected_global_rows.numel() == 0:
        raise ValueError("selected global rows are empty")
    selected_coords = global_coords256.index_select(0, selected_global_rows)
    q_global = base._endpoint_indices_to_q(
        selected_coords[:, 1:4],
        GRID_GLOBAL,
    ).to(global_coords256.device)
    q_local, uv_tile, _, transform_stats = (
        base._global_q_to_centered_tile_q(
            q_global,
            global_camera=global_camera,
            transform=transform,
        )
    )
    _, kept, quant_stats = base._quantize_local_q_without_geometry_clip(
        q_local,
        resolution=GRID_LOCAL,
        lattice_name="projected global C256 -> local C64",
        epsilon=float(boundary_epsilon),
    )
    global_rows = selected_global_rows[kept]
    q_global_kept = q_global[kept]
    q_local_kept = q_local[kept]
    uv_kept = uv_tile[kept]
    local_xyz_per_global = base._q_to_endpoint_indices(
        q_local_kept,
        GRID_LOCAL,
    )
    local_per_global = torch.cat(
        [
            torch.zeros(
                (local_xyz_per_global.shape[0], 1),
                device=local_xyz_per_global.device,
                dtype=torch.int32,
            ),
            local_xyz_per_global,
        ],
        dim=1,
    )
    local_coords, global_to_local = torch.unique(
        local_per_global,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    edge_global = global_rows
    edge_local = global_to_local.to(torch.long)
    local_degree = torch.bincount(
        edge_local,
        minlength=local_coords.shape[0],
    ).to(torch.float32)
    edge_forward_weight = 1.0 / local_degree.index_select(0, edge_local)
    check = torch.zeros(
        local_coords.shape[0],
        device=local_coords.device,
        dtype=torch.float32,
    )
    check.index_add_(0, edge_local, edge_forward_weight)
    normalization_error = float((check - 1.0).abs().max().item())
    if normalization_error >= 1e-6:
        raise RuntimeError(
            "global x_t -> local C64 averaging failed: "
            f"max sum error={normalization_error:.8e}"
        )

    stats = {
        "tile_id": int(transform.tile_id),
        "box": list(transform.box),
        "mapping": "exact camera global C256 centers -> unique local C64",
        "selected_global_c256_tokens": int(selected_global_rows.shape[0]),
        "kept_global_c256_tokens": int(global_rows.shape[0]),
        "local_c64_tokens": int(local_coords.shape[0]),
        "correspondence_edges": int(edge_global.shape[0]),
        "global_to_local": (
            "arithmetic mean of every global C256 x_t mapped to the local C64"
        ),
        "local_to_global": (
            "direct correspondence, followed by arithmetic mean over tiles"
        ),
        "forward_weight_sum_max_error": normalization_error,
        "global_edges_per_local_min": int(local_degree.min().item()),
        "global_edges_per_local_mean": float(local_degree.mean().item()),
        "global_edges_per_local_max": int(local_degree.max().item()),
        "local_quantization": quant_stats,
        "transform": transform_stats,
        "tile_camera": asdict(transform),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    base._atomic_json(output_dir / "transport_stats.json", stats)
    torch.save(
        {
            "global_coords_c256": global_coords256.index_select(
                0, global_rows
            ).detach().cpu(),
            "global_token_rows": global_rows.detach().cpu(),
            "local_coords_c64": local_coords.detach().cpu(),
            "edge_global_c256": edge_global.detach().cpu(),
            "edge_local_c64": edge_local.detach().cpu(),
            "edge_global_to_local_average_weight": (
                edge_forward_weight.detach().cpu()
            ),
            "q_global": q_global_kept.detach().cpu(),
            "q_local": q_local_kept.detach().cpu(),
            "uv_tile": uv_kept.detach().cpu(),
            "tile_camera": asdict(transform),
        },
        output_dir / "global_c256_local_c64_correspondence.pt",
    )
    return TileTransportC256(
        tile_id=int(transform.tile_id),
        box=tuple(int(v) for v in transform.box),
        transform=transform,
        local_coords=local_coords.detach().cpu(),
        edge_global=edge_global.detach().cpu(),
        edge_local=edge_local.detach().cpu(),
        edge_forward_weight=edge_forward_weight.detach().cpu(),
        global_token_rows=global_rows.detach().cpu(),
        stats=stats,
    )


def _prepare_tile_transports(
    *,
    args: argparse.Namespace,
    image_4096: Image.Image,
    global_coords256: torch.Tensor,
    uv_full_4096: torch.Tensor,
    finite_global: torch.Tensor,
    global_camera: Mapping[str, float],
    output_dir: Path,
) -> Tuple[List[TileTransportC256], List[Dict[str, Any]]]:
    boxes = base._tile_layout(
        IMAGE_CANONICAL,
        int(args.tile_size),
        int(args.tile_stride),
    )
    selected_ids = base._parse_tile_ids(args.tile_ids)
    if selected_ids is not None:
        invalid = sorted(value for value in selected_ids if value not in range(len(boxes)))
        if invalid:
            raise ValueError(
                f"invalid tile ids {invalid}; valid range is 0-{len(boxes) - 1}"
            )
    transports: List[TileTransportC256] = []
    records: List[Dict[str, Any]] = []
    attempted = 0
    for tile_id, box in enumerate(boxes):
        if selected_ids is not None and tile_id not in selected_ids:
            continue
        if args.max_tiles is not None and attempted >= int(args.max_tiles):
            break
        attempted += 1
        tile_dir = output_dir / "tiles" / f"tile_{tile_id:02d}"
        base._prepare_tile_reference(image_4096, box, tile_dir)
        rows = base._rows_inside_tile(uv_full_4096, finite_global, box)
        transform = base._derive_tile_camera(
            tile_id=tile_id,
            box=box,
            global_camera=global_camera,
            extend_pixel=int(args.extend_pixel),
            offaxis_shift_y_sign=int(args.offaxis_shift_y_sign),
        )
        base._atomic_json(tile_dir / "tile_camera.json", asdict(transform))
        if rows.numel() == 0:
            record = {
                "status": "skipped",
                "tile_id": tile_id,
                "box": list(box),
                "reason": "no projected global C256 token",
                "selected_global_c256_tokens": 0,
                "local_c64_tokens": 0,
            }
            records.append(record)
            base._atomic_json(tile_dir / "summary.json", record)
            continue
        try:
            transport = _build_tile_transport_c256(
                global_coords256=global_coords256,
                selected_global_rows=rows,
                global_camera=global_camera,
                transform=transform,
                output_dir=tile_dir / "transport",
                boundary_epsilon=float(args.boundary_epsilon),
            )
            outside_fraction = float(
                transport.stats["local_quantization"]["hard_outside_fraction"]
            )
            if outside_fraction > float(args.max_outside_fraction):
                raise RuntimeError(
                    f"outside fraction {outside_fraction:.6f} exceeds "
                    f"{float(args.max_outside_fraction):.6f}"
                )
            if transport.local_coords.shape[0] < int(args.min_tile_tokens):
                record = {
                    "status": "skipped",
                    "reason": "local C64 support below --min-tile-tokens",
                    **transport.stats,
                }
            else:
                transports.append(transport)
                record = {
                    "status": "active",
                    **transport.stats,
                    "reference_png": str(tile_dir / "reference_tile.png"),
                    "correspondence_path": str(
                        tile_dir
                        / "transport"
                        / "global_c256_local_c64_correspondence.pt"
                    ),
                }
                print(
                    f"[transport] tile={tile_id:02d} "
                    f"global_C256={transport.global_token_rows.shape[0]:,} "
                    f"local_C64={transport.local_coords.shape[0]:,}"
                )
            records.append(record)
            base._atomic_json(tile_dir / "summary.json", record)
        except Exception as exc:
            record = {
                "status": "failed",
                "tile_id": tile_id,
                "box": list(box),
                "selected_global_c256_tokens": int(rows.shape[0]),
                "reason": f"{type(exc).__name__}: {exc}",
            }
            records.append(record)
            base._atomic_json(tile_dir / "summary.json", record)
            print(f"[transport-error] tile={tile_id:02d}: {record['reason']}")
    if not transports:
        raise RuntimeError("no tile produced a usable global-C256/local-C64 map")
    return transports, records


def _prepare_stage_conditions(
    *,
    pipeline: Any,
    stage_name: str,
    image_1024: Image.Image,
    image_4096: Image.Image,
    global_coords256: torch.Tensor,
    global_camera: Mapping[str, float],
    transports: Sequence[TileTransportC256],
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Any]]:
    if stage_name == "shape":
        image_cond_model = pipeline.image_cond_model_shape_1024
    elif stage_name == "texture":
        image_cond_model = pipeline.image_cond_model_tex_1024
    else:
        raise ValueError("stage_name must be shape or texture")
    started = time.perf_counter()
    global_condition = pipeline.get_proj_cond_shape(
        image_cond_model,
        [image_1024.convert("RGB")],
        global_coords256,
        camera_angle_x=float(global_camera["camera_angle_x"]),
        distance=float(global_camera["distance"]),
        mesh_scale=float(global_camera["mesh_scale"]),
        grid_resolution_override=GRID_GLOBAL,
    )
    global_packed = base._pack_proj_condition_cpu(
        global_condition,
        expected_coords=global_coords256,
        name=f"global_{stage_name}_C256",
    )
    del global_condition
    _empty_cuda_cache()

    tile_records: List[Dict[str, Any]] = []
    for transport in transports:
        tile_started = time.perf_counter()
        local_coords = transport.local_coords.to(
            device=pipeline.device,
            dtype=torch.int32,
        )
        tile_image = base._composite_on_black(
            image_4096.crop(transport.box)
        )
        if tile_image.size != (IMAGE_FLOW, IMAGE_FLOW):
            tile_image = tile_image.resize(
                (IMAGE_FLOW, IMAGE_FLOW),
                Image.Resampling.LANCZOS,
            )
        camera = transport.transform
        local_condition = pipeline.get_proj_cond_shape(
            image_cond_model,
            [tile_image],
            local_coords,
            camera_angle_x=float(camera.camera_angle_x),
            distance=float(camera.distance),
            mesh_scale=float(camera.mesh_scale),
            grid_resolution_override=GRID_LOCAL,
        )
        transport.condition_cpu = base._pack_proj_condition_cpu(
            local_condition,
            expected_coords=local_coords,
            name=f"tile_{transport.tile_id:02d}_{stage_name}_C64",
        )
        elapsed = time.perf_counter() - tile_started
        tile_records.append(
            {
                "tile_id": int(transport.tile_id),
                "local_c64_tokens": int(local_coords.shape[0]),
                "seconds": float(elapsed),
            }
        )
        del local_coords, tile_image, local_condition
        _empty_cuda_cache()
        print(
            f"[condition-{stage_name}] tile={transport.tile_id:02d} "
            f"local_C64={transport.local_coords.shape[0]:,} "
            f"seconds={elapsed:.3f}"
        )
    return global_packed, {
        "stage": stage_name,
        "global_image": "canonical image resized to 1024",
        "local_images": "only canonical 1024 crops",
        "global_grid": GRID_GLOBAL,
        "local_grid": GRID_LOCAL,
        "active_tiles": len(transports),
        "total_seconds": float(time.perf_counter() - started),
        "tiles": tile_records,
    }


def _transport_global_features_to_local(
    global_features: torch.Tensor,
    transport: TileTransportC256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = global_features.device
    edge_global = transport.edge_global.to(device=device, dtype=torch.long)
    edge_local = transport.edge_local.to(device=device, dtype=torch.long)
    forward_weight = transport.edge_forward_weight.to(
        device=device,
        dtype=torch.float32,
    )
    local = torch.zeros(
        (transport.local_coords.shape[0], global_features.shape[1]),
        device=device,
        dtype=torch.float32,
    )
    local.index_add_(
        0,
        edge_local,
        global_features.index_select(0, edge_global).to(torch.float32)
        * forward_weight[:, None],
    )
    return local, edge_global, edge_local


@torch.no_grad()
def _run_wavelet_synchronized_flow(
    *,
    pipeline: Any,
    stage_name: str,
    model: torch.nn.Module,
    sampler: Any,
    global_coords256: torch.Tensor,
    global_condition_cpu: Mapping[str, Mapping[str, torch.Tensor]],
    transports: Sequence[TileTransportC256],
    params: Mapping[str, Any],
    seed: int,
    concat_global: Optional[SparseTensor] = None,
) -> Tuple[SparseTensor, float, Dict[str, Any]]:
    """Run global C256 with local-C64 high-frequency replacement every step."""
    if concat_global is not None and not torch.equal(
        concat_global.coords, global_coords256
    ):
        raise RuntimeError(f"{stage_name}: concat condition coords differ")
    latent_channels = int(model.in_channels)
    if concat_global is not None:
        latent_channels -= int(concat_global.feats.shape[1])
    if latent_channels <= 0:
        raise RuntimeError(f"{stage_name}: invalid latent channel count")
    state = SparseTensor(
        feats=base._randn(
            global_coords256.shape[0],
            latent_channels,
            device=pipeline.device,
            seed=int(seed),
        ),
        coords=global_coords256,
    )
    t_seq = sampler.timestep_schedule(
        int(params["steps"]),
        float(params.get("rescale_t", 1.0)),
    )
    prediction_kwargs = {
        key: value
        for key, value in params.items()
        if key
        not in {
            "steps",
            "rescale_t",
            "verbose",
            "tqdm_desc",
            "record_trajectory",
            "trajectory_device",
            "return_model_history",
        }
    }
    global_condition = base._materialize_proj_condition(
        global_condition_cpu,
        coords=global_coords256,
        device=pipeline.device,
    )
    local_concat_cpu: Dict[int, torch.Tensor] = {}
    if concat_global is not None:
        for transport in transports:
            local_concat, _, _ = _transport_global_features_to_local(
                concat_global.feats,
                transport,
            )
            local_concat_cpu[transport.tile_id] = local_concat.detach().cpu()
            del local_concat

    coverage_count = torch.zeros(
        global_coords256.shape[0],
        device=pipeline.device,
        dtype=torch.float32,
    )
    tile_coverage_count = torch.zeros(
        global_coords256.shape[0],
        device=pipeline.device,
        dtype=torch.int32,
    )
    for transport in transports:
        global_rows = torch.unique(
            transport.edge_global.to(
                device=pipeline.device,
                dtype=torch.long,
            )
        )
        coverage_count.index_add_(
            0,
            global_rows,
            torch.ones_like(global_rows, dtype=torch.float32),
        )
        tile_coverage_count.index_add_(
            0,
            global_rows,
            torch.ones_like(global_rows, dtype=torch.int32),
        )
    covered = coverage_count > 0
    wavelet_check = _wavelet_self_test(pipeline.device)
    print(
        f"[haar-self-test] roundtrip="
        f"{wavelet_check['roundtrip_max_abs_error']:.8e} "
        f"low={wavelet_check['global_lll_invariant_max_abs_error']:.8e} "
        f"high={wavelet_check['local_high_band_invariant_max_abs_error']:.8e}"
    )

    if pipeline.low_vram:
        model.to(pipeline.device)
    started = time.perf_counter()
    step_records: List[Dict[str, Any]] = []
    try:
        for step_index, (t_value, t_next) in enumerate(
            zip(t_seq[:-1], t_seq[1:])
        ):
            step_started = time.perf_counter()
            dt = float(t_value - t_next)
            _, _, global_velocity = sampler._get_model_prediction(
                model,
                state,
                float(t_value),
                global_condition["cond"],
                neg_cond=global_condition["neg_cond"],
                concat_cond=concat_global,
                **prediction_kwargs,
            )
            if not torch.equal(global_velocity.coords, global_coords256):
                raise RuntimeError(f"{stage_name}: global velocity changed coords")

            local_velocity_sum = torch.zeros(
                state.feats.shape,
                device=pipeline.device,
                dtype=torch.float32,
            )
            local_velocity_count = torch.zeros(
                global_coords256.shape[0],
                device=pipeline.device,
                dtype=torch.float32,
            )
            local_velocity_norm_sum = 0.0
            tile_calls = 0
            for transport in transports:
                if transport.condition_cpu is None:
                    raise RuntimeError(
                        f"tile {transport.tile_id} has no {stage_name} condition"
                    )
                local_coords = transport.local_coords.to(
                    device=pipeline.device,
                    dtype=torch.int32,
                )
                local_state_feats, edge_global, edge_local = (
                    _transport_global_features_to_local(
                        state.feats,
                        transport,
                    )
                )
                local_state = SparseTensor(
                    feats=local_state_feats.to(state.feats.dtype),
                    coords=local_coords,
                )
                local_condition = base._materialize_proj_condition(
                    transport.condition_cpu,
                    coords=local_coords,
                    device=pipeline.device,
                )
                local_concat = None
                if concat_global is not None:
                    local_concat = SparseTensor(
                        feats=local_concat_cpu[transport.tile_id].to(
                            device=pipeline.device,
                            dtype=concat_global.feats.dtype,
                        ),
                        coords=local_coords,
                    )
                _, _, local_velocity = sampler._get_model_prediction(
                    model,
                    local_state,
                    float(t_value),
                    local_condition["cond"],
                    neg_cond=local_condition["neg_cond"],
                    concat_cond=local_concat,
                    **prediction_kwargs,
                )
                if not torch.equal(local_velocity.coords, local_coords):
                    raise RuntimeError(
                        f"{stage_name}: tile {transport.tile_id} changed coords"
                    )
                local_velocity_sum.index_add_(
                    0,
                    edge_global,
                    local_velocity.feats.to(torch.float32).index_select(
                        0, edge_local
                    ),
                )
                local_velocity_count.index_add_(
                    0,
                    edge_global,
                    torch.ones_like(edge_global, dtype=torch.float32),
                )
                local_velocity_norm_sum += float(
                    local_velocity.feats.float().norm().item()
                )
                tile_calls += 1
                del (
                    local_coords,
                    local_state_feats,
                    edge_global,
                    edge_local,
                    local_state,
                    local_condition,
                    local_concat,
                    local_velocity,
                )

            # A complete local-union field is required before the dense DWT.
            # Covered active tokens use the arithmetic tile mean.  Uncovered
            # active tokens fall back to the ordinary global velocity so the
            # local dense field does not create artificial zero boundaries.
            local_union_velocity = global_velocity.feats.to(torch.float32).clone()
            local_union_velocity[covered] = (
                local_velocity_sum[covered]
                / local_velocity_count[covered, None]
            )
            fused_velocity, wavelet_stats = _fuse_velocity_wavelet_c256(
                global_velocity.feats,
                local_union_velocity,
                global_coords256,
            )
            state = state.replace(
                state.feats - dt * fused_velocity.to(state.feats.dtype)
            )
            if not torch.equal(state.coords, global_coords256):
                raise RuntimeError(f"{stage_name}: Euler update changed coords")
            _sync_cuda()
            record = {
                "step_index": int(step_index),
                "t": float(t_value),
                "t_next": float(t_next),
                "dt": dt,
                "global_velocity_norm": float(
                    global_velocity.feats.float().norm().item()
                ),
                "local_union_velocity_norm": float(
                    local_union_velocity.norm().item()
                ),
                "fused_velocity_norm": float(fused_velocity.norm().item()),
                "local_model_calls": int(tile_calls),
                "mean_local_velocity_norm": float(
                    local_velocity_norm_sum / max(tile_calls, 1)
                ),
                "wavelet": wavelet_stats,
                "seconds": float(time.perf_counter() - step_started),
            }
            step_records.append(record)
            print(
                f"[C256-{stage_name}-wavelet] step={step_index:02d} "
                f"t={float(t_value):.8f}->{float(t_next):.8f} "
                f"tiles={tile_calls} "
                f"global_v={record['global_velocity_norm']:.4f} "
                f"local_v={record['local_union_velocity_norm']:.4f} "
                f"fused_v={record['fused_velocity_norm']:.4f} "
                f"seconds={record['seconds']:.3f}"
            )
            del (
                global_velocity,
                local_velocity_sum,
                local_velocity_count,
                local_union_velocity,
                fused_velocity,
            )
            _empty_cuda_cache()
    finally:
        if pipeline.low_vram:
            model.cpu()
        for transport in transports:
            transport.condition_cpu = None
        _empty_cuda_cache()
    elapsed = time.perf_counter() - started
    diagnostics = {
        "stage": stage_name,
        "state": "one global C256 sparse x_t",
        "global_grid": GRID_GLOBAL,
        "local_grid": GRID_LOCAL,
        "global_tokens": int(global_coords256.shape[0]),
        "active_tiles": int(len(transports)),
        "covered_global_tokens": int(covered.sum().item()),
        "covered_global_fraction": float(covered.float().mean().item()),
        "overlap_global_tokens": int((tile_coverage_count > 1).sum().item()),
        "overlap_rule": "arithmetic mean over local tile velocities",
        "uncovered_rule": "ordinary global velocity fallback",
        "local_noise_initializations": 0,
        "local_state_rule": "transport current global x_t every step",
        "wavelet": {
            "family": "orthonormal Haar",
            "dense_velocity_shape": [
                1,
                latent_channels,
                256,
                256,
                256,
            ],
            "band_shape": [
                1,
                latent_channels,
                8,
                128,
                128,
                128,
            ],
            "band_order": list(HAAR_BANDS),
            "LLL_source": "normal full-image global velocity",
            "other_seven_band_source": "local C64 union velocity",
            "compute_dtype": "float32",
            "self_test": wavelet_check,
        },
        "global_updates_per_step": 1,
        "steps": step_records,
        "elapsed_seconds": float(elapsed),
    }
    del global_condition, local_concat_cpu
    _empty_cuda_cache()
    return state, elapsed, diagnostics


def _load_final_latents(
    path: Path,
    *,
    device: torch.device,
) -> Tuple[SparseTensor, SparseTensor, Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    required = ("coords", "shape_denorm_feats", "texture_denorm_feats")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path} is missing: {', '.join(missing)}")
    coords = torch.as_tensor(payload["coords"], device=device, dtype=torch.int32)
    shape = torch.as_tensor(payload["shape_denorm_feats"], device=device)
    texture = torch.as_tensor(payload["texture_denorm_feats"], device=device)
    if shape.shape[0] != coords.shape[0] or texture.shape[0] != coords.shape[0]:
        raise ValueError("saved final latents are not coordinate-aligned")
    return (
        SparseTensor(feats=shape, coords=coords),
        SparseTensor(feats=texture, coords=coords),
        dict(payload),
    )


def _decode_render_evaluate(
    *,
    pipeline: Any,
    shape_denorm: SparseTensor,
    texture_denorm: SparseTensor,
    output_dir: Path,
    global_camera: Mapping[str, float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    final_dir = output_dir / "final_global_4096"
    final_dir.mkdir(parents=True, exist_ok=True)
    decoded, mesh = base._decode_normal_mesh_with_ovoxel(
        pipeline=pipeline,
        shape_latent=shape_denorm,
        texture_latent=texture_denorm,
        label="Global C256 wavelet-fused 4096",
        resolution=DECODE_GLOBAL,
    )
    decoder_summary = {
        "resolution": DECODE_GLOBAL,
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "active_voxels": int(mesh.coords.shape[0]),
        "sample_type": type(mesh).__name__,
    }
    checkpoint = final_dir / "mesh_with_ovoxel.pt"
    mesh_cpu = mesh.to("cpu")
    torch.save(mesh_cpu, checkpoint)
    print(f"[checkpoint] decoded model saved before rendering: {checkpoint}")
    pipeline.to(torch.device("cpu"))
    del decoded, mesh
    _empty_cuda_cache()
    render_mesh = mesh_cpu.to("cuda")
    del mesh_cpu
    _empty_cuda_cache()
    envmap = load_envmap(str(args.envmap), device="cuda")
    try:
        metric_row = render_and_evaluate_mesh(
            render_mesh,
            camera_angle_x=float(global_camera["camera_angle_x"]),
            distance=float(global_camera["distance"]),
            output_dir=final_dir / "aligned_eval",
            reference_image=output_dir / "canonical_4096.png",
            resolution=int(args.render_resolution),
            metric_resolution=int(args.metric_resolution),
            envmap=envmap,
            envmap_name=str(args.envmap),
            ssaa=int(args.render_ssaa),
            peel_layers=int(args.render_peel_layers),
            face_chunk_size=int(args.render_face_chunk_size),
            use_envmap_bg=bool(args.use_envmap_bg),
            lpips_net=str(args.lpips_net),
            metric_device=str(args.metric_device),
            skip_lpips=bool(args.skip_lpips),
        )
    except Exception as exc:
        failure = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "mesh_checkpoint": str(checkpoint),
        }
        base._atomic_json(final_dir / "render_failure.json", failure)
        raise
    comparison = base._save_extra_comparisons(
        Path(metric_row["original_png"]),
        Path(metric_row["render_png"]),
        final_dir / "comparisons",
    )
    del render_mesh, envmap
    _empty_cuda_cache()
    return {
        "decoder": decoder_summary,
        "mesh_checkpoint": str(checkpoint),
        "render_and_metrics": {**metric_row, **comparison},
    }


def run(args: argparse.Namespace) -> None:
    if int(args.tile_size) != DEFAULT_TILE_SIZE:
        raise ValueError(f"this route requires --tile-size={DEFAULT_TILE_SIZE}")
    if int(args.tile_stride) != DEFAULT_TILE_STRIDE:
        raise ValueError(f"this route requires --tile-stride={DEFAULT_TILE_STRIDE}")
    if args.cuda_device is not None:
        torch.cuda.set_device(int(args.cuda_device))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base._atomic_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "physical_cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            ),
            "decode_resolution": DECODE_GLOBAL,
            "global_grid_resolution": GRID_GLOBAL,
            "local_grid_resolution": GRID_LOCAL,
        },
    )
    base._seed_everything(int(args.seed))
    pipeline = init_pipeline(
        args.model_path,
        device="cuda",
        low_vram=bool(args.low_vram),
    )
    canonical = pipeline.preprocess_canonical_images(Image.open(args.image))
    image_4096: Image.Image = canonical["image_4096"]
    image_1024: Image.Image = canonical["image_1024"]
    image_512: Image.Image = canonical["image_512"]
    foreground_mask = canonical["foreground_mask_4096"].convert("L")
    image_4096.save(output_dir / "canonical_4096.png")
    image_1024.save(output_dir / "canonical_1024.png")
    image_512.save(output_dir / "canonical_512.png")
    foreground_mask.save(output_dir / "canonical_foreground_mask_4096.png")
    base._atomic_json(
        output_dir / "canonical_metadata.json",
        canonical["metadata"],
    )
    global_camera = base._estimate_camera(
        image_1024=image_1024,
        output_dir=output_dir,
        manual_fov=float(args.fov),
        mesh_scale=float(args.mesh_scale),
        extend_pixel=int(args.extend_pixel),
        image_resolution=int(args.camera_image_resolution),
        moge_model_path=args.moge_model_path,
    )
    base._atomic_json(output_dir / "global_camera.json", global_camera)
    params = base._sampler_params(args, pipeline)
    latent_path = output_dir / "global_c256_wavelet_latents.pt"

    if bool(args.resume_final_latents):
        shape_denorm, texture_denorm, payload = _load_final_latents(
            latent_path,
            device=pipeline.device,
        )
        final = _decode_render_evaluate(
            pipeline=pipeline,
            shape_denorm=shape_denorm,
            texture_denorm=texture_denorm,
            output_dir=output_dir,
            global_camera=global_camera,
            args=args,
        )
        summary = {
            "format": "pixal3d_global_c256_local_c64_haar_velocity_fusion_v1",
            "resumed_from_final_latents": True,
            "image": str(Path(args.image).expanduser().resolve()),
            "latent_payload": {
                "resolution": payload.get("resolution"),
                "grid_resolution": payload.get("grid_resolution"),
            },
            **final,
            "visual_metrics": {
                key: final["render_and_metrics"].get(key)
                for key in ("psnr_db", "ssim", "lpips")
            },
        }
        base._atomic_json(output_dir / "summary.json", summary)
        return

    print("[global-prior] SS C32 -> shape512 -> learned C64 -> shape1024")
    coords32, coords64, coarse_shape = (
        base._run_global_official_geometry_to_shape1024(
            pipeline=pipeline,
            image_512=image_512,
            image_1024=image_1024,
            camera=global_camera,
            params=params,
            seed=int(args.seed),
            max_tokens=int(args.max_num_tokens),
        )
    )
    coords1024, subdivision_stats = (
        base._learned_subdivide_shape1024_to_c1024(
            pipeline,
            coarse_shape.shape_denorm,
        )
    )
    coords256, source_to_global, quantization_stats = (
        _quantize_global_c1024_to_c256(coords1024)
    )
    if coords256.shape[0] > int(args.max_num_tokens):
        raise RuntimeError(
            f"global C256 has {coords256.shape[0]:,} tokens, exceeding "
            f"--max-num-tokens={int(args.max_num_tokens):,}"
        )
    print(
        f"[global-support] dense_C1024={coords1024.shape[0]:,} "
        f"global_C256={coords256.shape[0]:,}"
    )
    support_dir = output_dir / "global_support"
    support_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "coords_c32": coords32.detach().cpu(),
            "coords_c64": coords64.detach().cpu(),
            "coords_c1024": coords1024.detach().cpu(),
            "coords_global_c256": coords256.detach().cpu(),
            "c1024_source_to_global_c256": source_to_global.detach().cpu(),
        },
        support_dir / "global_support_and_mapping.pt",
    )
    support_summary = {
        "route": (
            "ordinary Pixal3D SS C32 -> shape512 -> learned C64 -> "
            "shape1024 -> decoder subdivision C1024 -> fixed C256"
        ),
        "global_c32_tokens": int(coords32.shape[0]),
        "global_c64_tokens": int(coords64.shape[0]),
        "dense_global_c1024_points": int(coords1024.shape[0]),
        "global_c256_tokens": int(coords256.shape[0]),
        "subdivision": subdivision_stats,
        "c1024_to_c256": quantization_stats,
        "shape512_seconds": float(coarse_shape.shape512_seconds),
        "coarse_shape1024_seconds": float(coarse_shape.shape1024_seconds),
    }
    base._atomic_json(support_dir / "summary.json", support_summary)
    del coarse_shape, coords1024, source_to_global
    _empty_cuda_cache()

    q_global = base._endpoint_indices_to_q(
        coords256[:, 1:4],
        GRID_GLOBAL,
    ).to(coords256.device)
    _, uv_global_1024, uv_full_4096, _, finite_global = (
        base._project_global_q_to_1024_and_4096(
            q_global,
            global_camera=global_camera,
        )
    )
    projection_summary = {
        "source": "global C256 flow tokens",
        "finite_tokens": int(finite_global.sum().item()),
        "total_tokens": int(coords256.shape[0]),
        "uv_global_1024_min": uv_global_1024.amin(dim=0).cpu().tolist(),
        "uv_global_1024_max": uv_global_1024.amax(dim=0).cpu().tolist(),
        "uv_full_4096_min": uv_full_4096.amin(dim=0).cpu().tolist(),
        "uv_full_4096_max": uv_full_4096.amax(dim=0).cpu().tolist(),
    }
    base._atomic_json(
        support_dir / "projection_summary.json",
        projection_summary,
    )
    transports, tile_records = _prepare_tile_transports(
        args=args,
        image_4096=image_4096,
        global_coords256=coords256,
        uv_full_4096=uv_full_4096,
        finite_global=finite_global,
        global_camera=global_camera,
        output_dir=output_dir,
    )
    base._write_csv(output_dir / "tile_transport_summary.csv", tile_records)
    del (
        q_global,
        uv_global_1024,
        uv_full_4096,
        finite_global,
        coords32,
        coords64,
    )
    _empty_cuda_cache()

    shape_condition_cpu, shape_condition_stats = _prepare_stage_conditions(
        pipeline=pipeline,
        stage_name="shape",
        image_1024=image_1024,
        image_4096=image_4096,
        global_coords256=coords256,
        global_camera=global_camera,
        transports=transports,
    )
    shape_norm, shape_seconds, shape_flow = _run_wavelet_synchronized_flow(
        pipeline=pipeline,
        stage_name="shape",
        model=pipeline.models["shape_slat_flow_model_1024"],
        sampler=pipeline.shape_slat_sampler,
        global_coords256=coords256,
        global_condition_cpu=shape_condition_cpu,
        transports=transports,
        params=params["shape"],
        seed=int(args.seed) + 401,
    )
    shape_denorm = base._denormalize_sparse(
        shape_norm,
        pipeline.shape_slat_normalization,
    )
    base._atomic_json(output_dir / "shape_flow.json", shape_flow)
    del shape_condition_cpu
    _empty_cuda_cache()

    texture_condition_cpu, texture_condition_stats = _prepare_stage_conditions(
        pipeline=pipeline,
        stage_name="texture",
        image_1024=image_1024,
        image_4096=image_4096,
        global_coords256=coords256,
        global_camera=global_camera,
        transports=transports,
    )
    texture_norm, texture_seconds, texture_flow = (
        _run_wavelet_synchronized_flow(
            pipeline=pipeline,
            stage_name="texture",
            model=pipeline.models["tex_slat_flow_model_1024"],
            sampler=pipeline.tex_slat_sampler,
            global_coords256=coords256,
            global_condition_cpu=texture_condition_cpu,
            transports=transports,
            params=params["texture"],
            seed=int(args.seed) + 501,
            concat_global=shape_norm,
        )
    )
    texture_denorm = base._denormalize_sparse(
        texture_norm,
        pipeline.tex_slat_normalization,
    )
    base._atomic_json(output_dir / "texture_flow.json", texture_flow)
    del texture_condition_cpu
    _empty_cuda_cache()

    torch.save(
        {
            "resolution": DECODE_GLOBAL,
            "grid_resolution": GRID_GLOBAL,
            "coords": coords256.detach().cpu(),
            "shape_norm_feats": shape_norm.feats.detach().cpu(),
            "shape_denorm_feats": shape_denorm.feats.detach().cpu(),
            "texture_norm_feats": texture_norm.feats.detach().cpu(),
            "texture_denorm_feats": texture_denorm.feats.detach().cpu(),
            "global_camera": dict(global_camera),
        },
        latent_path,
    )
    final = _decode_render_evaluate(
        pipeline=pipeline,
        shape_denorm=shape_denorm,
        texture_denorm=texture_denorm,
        output_dir=output_dir,
        global_camera=global_camera,
        args=args,
    )
    metric_row = final["render_and_metrics"]
    summary = {
        "format": "pixal3d_global_c256_local_c64_haar_velocity_fusion_v1",
        "image": str(Path(args.image).expanduser().resolve()),
        "global_camera": global_camera,
        "support": support_summary,
        "projection": projection_summary,
        "transport": {
            "tile_size": int(args.tile_size),
            "tile_stride": int(args.tile_stride),
            "active_tiles": int(len(transports)),
            "local_inputs": "1024 crops only",
            "mapping": "global C256 <-> exact-camera local C64",
            "overlap": "arithmetic mean",
            "tiles": tile_records,
        },
        "shape": {
            "condition_extraction": shape_condition_stats,
            "flow": shape_flow,
            "seconds": float(shape_seconds),
        },
        "texture": {
            "condition_extraction": texture_condition_stats,
            "flow": texture_flow,
            "seconds": float(texture_seconds),
        },
        "fusion": (
            "each step: global C256 x_t -> local C64; local velocities -> "
            "global C256 arithmetic-mean union; dense C256 Haar; global LLL "
            "+ local seven high bands; inverse Haar; gather active C256"
        ),
        "latents": str(latent_path),
        **final,
        "visual_metrics": {
            "psnr_db": metric_row.get("psnr_db"),
            "ssim": metric_row.get("ssim"),
            "lpips": metric_row.get("lpips"),
        },
    }
    base._atomic_json(
        output_dir / "final_global_4096" / "summary.json",
        summary,
    )
    base._atomic_json(output_dir / "summary.json", summary)
    print(
        f"[done] mesh={summary['mesh_checkpoint']} "
        f"render={metric_row.get('render_png')} "
        f"PSNR={metric_row.get('psnr_db')} "
        f"SSIM={metric_row.get('ssim')} LPIPS={metric_row.get('lpips')}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image")
    parser.add_argument("--output-dir")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--moge-model-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--resume-final-latents",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--tile-stride", type=int, default=DEFAULT_TILE_STRIDE)
    parser.add_argument("--tile-ids", default=None)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument(
        "--min-tile-tokens",
        type=int,
        default=1,
        help="minimum local C64 tokens; default retains every nonempty crop",
    )
    parser.add_argument("--max-num-tokens", type=int, default=100_000_000)
    parser.add_argument("--boundary-epsilon", type=float, default=1e-5)
    parser.add_argument(
        "--max-outside-fraction",
        type=float,
        default=1.0,
        help=(
            "optional whole-tile rejection threshold after strict per-point "
            "outside rows have already been dropped; 1.0 retains every tile "
            "with at least one valid local C64 token"
        ),
    )
    parser.add_argument(
        "--offaxis-shift-y-sign",
        type=int,
        choices=(-1, 1),
        default=1,
    )
    parser.add_argument("--fov", type=float, default=-1.0)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument("--camera-image-resolution", type=int, default=1024)
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
    parser.add_argument(
        "--low-vram",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--render-resolution", type=int, default=2048)
    parser.add_argument("--metric-resolution", type=int, default=1024)
    parser.add_argument("--render-ssaa", type=int, default=1)
    parser.add_argument("--render-peel-layers", type=int, default=4)
    parser.add_argument("--render-face-chunk-size", type=int, default=4_000_000)
    parser.add_argument(
        "--use-envmap-bg",
        action=argparse.BooleanOptionalAction,
        default=False,
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
        default=False,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if bool(args.self_test):
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if args.cuda_device is not None and device.type == "cuda":
            torch.cuda.set_device(int(args.cuda_device))
            device = torch.device("cuda")
        result = _wavelet_self_test(device)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if not args.image or not args.output_dir:
        raise ValueError("--image and --output-dir are required")
    if int(args.min_tile_tokens) < 1:
        raise ValueError("--min-tile-tokens must be positive")
    if int(args.max_num_tokens) < int(args.min_tile_tokens):
        raise ValueError("--max-num-tokens must be >= --min-tile-tokens")
    if args.max_tiles is not None and int(args.max_tiles) < 1:
        raise ValueError("--max-tiles must be positive")
    if args.cuda_device is not None and int(args.cuda_device) < 0:
        raise ValueError("--cuda-device must be non-negative")
    for name in ("ss_steps", "shape_steps", "texture_steps"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "render_resolution",
        "metric_resolution",
        "render_ssaa",
        "render_peel_layers",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args.render_face_chunk_size) < 0:
        raise ValueError("--render-face-chunk-size must be non-negative")
    if float(args.boundary_epsilon) < 0:
        raise ValueError("--boundary-epsilon must be non-negative")
    if not 0.0 <= float(args.max_outside_fraction) <= 1.0:
        raise ValueError("--max-outside-fraction must lie in [0,1]")
    run(args)


if __name__ == "__main__":
    main()
