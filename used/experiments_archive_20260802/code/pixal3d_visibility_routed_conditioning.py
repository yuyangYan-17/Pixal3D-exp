#!/usr/bin/env python3
"""Visibility-routed per-token image conditioning for Pixal3D local tiles.

This experiment reuses the established global baseline, local C64 support,
decoder, local-to-global return, ownership, welding, renderer, and metrics.
Only the image contribution inside each 1024 shape/texture flow Transformer
block is changed.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
)

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
import pixal3d_tile_encoded_query_noise_flow_overlap_render as baseline
import pixal3d_tile_online_canonical_posterior_shape_texture as posterior
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from pixal3d.modules.sparse.spatial.spatial2channel import (
    SparseSpatial2Channel,
)


FORMAT_VERSION = "pixal3d_visibility_routed_conditioning_v1"
CONDITIONING_MODES = (
    "local",
    "proj_mask",
    "hg_block",
    "hzero_block",
    "hg_velocity",
    "hg_soft",
)


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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
        json.dump(
            posterior._json_value(dict(payload)),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _rms(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(value.detach().float().square().mean().sqrt().item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left64 = left.detach().double().reshape(-1)
    right64 = right.detach().double().reshape(-1)
    denominator = float(
        (
            torch.linalg.vector_norm(left64)
            * torch.linalg.vector_norm(right64)
        ).item()
    )
    if denominator <= 1e-24:
        return 0.0
    return float(torch.dot(left64, right64).item() / denominator)


def _linear_keys(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    return core._linear_keys(coords.to(torch.int64), int(resolution))


def _first_hit_visibility(
    q_local: torch.Tensor,
    transform: core.TileCameraTransform,
) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor]:
    """One exact depth winner per raster pixel; no depth tolerance is used."""
    points = core._camera_q_to_points(
        q_local.float(),
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
    )
    uv, depth, finite = core._project_points(
        points,
        fx=float(transform.fx),
        fy=float(transform.fy),
        cx=float(transform.cx),
        cy=float(transform.cy),
    )
    width = int(transform.output_width)
    height = int(transform.output_height)
    pixels = torch.floor(uv).to(torch.int64)
    valid = (
        finite
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    valid_rows = torch.where(valid)[0]
    visible = torch.zeros(q_local.shape[0], dtype=torch.bool)
    if valid_rows.numel():
        keys = (
            pixels[valid_rows, 1] * int(width)
            + pixels[valid_rows, 0]
        )
        valid_depth = depth.index_select(0, valid_rows)
        # Stable depth sort followed by stable pixel sort is lexicographic
        # (pixel, depth, source row); the first row of each pixel is first-hit.
        by_depth = torch.argsort(valid_depth, stable=True)
        by_pixel = torch.argsort(keys[by_depth], stable=True)
        order = by_depth[by_pixel]
        ordered_keys = keys[order]
        keep = torch.ones(order.shape[0], dtype=torch.bool)
        keep[1:] = ordered_keys[1:] != ordered_keys[:-1]
        winners = valid_rows.index_select(0, order[keep])
        visible[winners] = True
    return (
        visible,
        {
            "surface_primitives": int(q_local.shape[0]),
            "valid_projected_primitives": int(valid_rows.numel()),
            "first_hit_primitives": int(visible.sum().item()),
            "occupied_raster_pixels": int(visible.sum().item()),
            "raster_resolution": [width, height],
            "pixel_quantization": "floor continuous camera projection",
            "z_buffer_rule": (
                "one minimum camera-depth source row per occupied raster "
                "pixel; stable source order breaks exact depth ties"
            ),
            "depth_tolerance": None,
        },
        uv,
    )


def _trace_encoder_source_to_c64(
    coords_xyz: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Compose the four released SparseSpatial2Channel indice maps."""
    coords4 = torch.cat(
        (torch.zeros_like(coords_xyz[:, :1]), coords_xyz), dim=1
    ).to(torch.int32)
    current = SparseTensor(
        torch.ones((coords4.shape[0], 1), dtype=torch.float32), coords4
    )
    source_to_current = torch.arange(coords4.shape[0], dtype=torch.long)
    stage_rows = [int(coords4.shape[0])]
    for _ in range(4):
        downsample = SparseSpatial2Channel(2)
        output = downsample(current)
        cache = current.get_spatial_cache("spatial2channel_2")
        if cache is None:
            raise RuntimeError("released SparseSpatial2Channel indice map missing")
        new_coords, current_to_next, _ = cache
        if not torch.equal(output.coords, new_coords):
            raise RuntimeError("SparseSpatial2Channel cache/output mismatch")
        source_to_current = current_to_next.index_select(
            0, source_to_current
        )
        current = output
        stage_rows.append(int(current.coords.shape[0]))
    simplified = coords4.clone()
    simplified[:, 1:] //= 16
    if not torch.equal(
        current.coords.index_select(0, source_to_current), simplified
    ):
        raise AssertionError(
            "four released encoder indice maps differ from floor(coord/16)"
        )
    return (
        current.coords.detach().cpu().to(torch.int32),
        source_to_current.detach().cpu().to(torch.long),
        {
            "encoder_downsampling_class": (
                "pixal3d.modules.sparse.spatial.spatial2channel."
                "SparseSpatial2Channel"
            ),
            "stages": 4,
            "factor_per_stage": 2,
            "stage_active_rows": stage_rows,
            "composed_factor": 16,
            "floor_coord_div_16_bitwise_validated": True,
        },
    )


def _aggregate_visibility_to_tokens(
    *,
    token_coords: torch.Tensor,
    final_coords: torch.Tensor,
    source_to_final: torch.Tensor,
    source_visible: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    final_total = torch.zeros(final_coords.shape[0], dtype=torch.int64)
    final_visible = torch.zeros_like(final_total)
    final_total.index_add_(
        0, source_to_final, torch.ones_like(source_to_final)
    )
    final_visible.index_add_(
        0, source_to_final, source_visible.to(torch.int64)
    )
    final_keys = _linear_keys(final_coords[:, 1:], 64)
    token_keys = _linear_keys(token_coords[:, 1:], 64)
    order = torch.argsort(final_keys)
    sorted_keys = final_keys[order]
    positions = torch.searchsorted(sorted_keys, token_keys)
    in_bounds = positions < sorted_keys.shape[0]
    safe = positions.clamp_max(max(0, sorted_keys.shape[0] - 1))
    matched = in_bounds & (
        sorted_keys.index_select(0, safe) == token_keys
    )
    totals = torch.zeros(token_coords.shape[0], dtype=torch.int64)
    visible = torch.zeros_like(totals)
    if bool(matched.any()):
        token_rows = torch.where(matched)[0]
        final_rows = order.index_select(
            0, positions.index_select(0, token_rows)
        )
        totals[token_rows] = final_total.index_select(0, final_rows)
        visible[token_rows] = final_visible.index_select(0, final_rows)
    ratio = visible.float() / totals.clamp_min(1).float()
    hard = visible > 0
    return {
        "hard_visibility": hard,
        "visibility_ratio": ratio,
        "visible_surface_counts": visible,
        "total_surface_counts": totals,
    }


def _save_visibility_visualizations(
    *,
    tile_image: Image.Image,
    source_uv: torch.Tensor,
    source_visible: torch.Tensor,
    token_coords: torch.Tensor,
    hard_visibility: torch.Tensor,
    transform: core.TileCameraTransform,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_panel = tile_image.convert("RGB").copy()
    draw = ImageDraw.Draw(source_panel)
    step = max(1, int(source_uv.shape[0]) // 25000)
    for row in range(0, int(source_uv.shape[0]), step):
        x, y = source_uv[row].tolist()
        color = (0, 255, 80) if bool(source_visible[row]) else (255, 48, 48)
        draw.point((float(x), float(y)), fill=color)
    source_panel.save(output_dir / "visible_ovoxels.png")

    xyz = token_coords[:, 1:].float()
    object_points = (xyz + 0.5) / 64.0 - 0.5
    q_local = object_points * (2.0 * float(transform.mesh_scale))
    points = core._camera_q_to_points(
        q_local,
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
    )
    token_uv, _, finite = core._project_points(
        points,
        fx=float(transform.fx),
        fy=float(transform.fy),
        cx=float(transform.cx),
        cy=float(transform.cy),
    )
    token_panel = tile_image.convert("RGB").copy()
    draw = ImageDraw.Draw(token_panel)
    for row in torch.where(finite)[0].tolist():
        x, y = token_uv[row].tolist()
        color = (0, 255, 80) if bool(hard_visibility[row]) else (64, 128, 255)
        draw.ellipse(
            (float(x) - 1, float(y) - 1, float(x) + 1, float(y) + 1),
            fill=color,
        )
    token_panel.save(output_dir / "c64_token_visibility.png")


def _build_baseline_visibility(
    *,
    tile: posterior.TileRuntime,
    baseline_mesh: Any,
    global_q: torch.Tensor,
    global_uv: torch.Tensor,
    global_finite: torch.Tensor,
    global_camera: Mapping[str, float],
    tile_image: Image.Image,
    output_dir: Path,
) -> Dict[str, Any]:
    mapping = core._map_global_ovoxels_to_local(
        global_mesh=baseline_mesh,
        global_q=global_q,
        global_uv_4096=global_uv,
        finite_projection=global_finite,
        global_camera=global_camera,
        transform=tile.transform,
    )
    if mapping.local_coords.shape[0] == 0:
        raise RuntimeError("baseline visibility has no local surface O-Voxels")
    source_visible, zbuffer_stats, source_uv = _first_hit_visibility(
        mapping.local_q, tile.transform
    )
    final_coords, source_to_final, downsample_stats = (
        _trace_encoder_source_to_c64(mapping.local_coords)
    )
    tensors = _aggregate_visibility_to_tokens(
        token_coords=tile.coords_cpu,
        final_coords=final_coords,
        source_to_final=source_to_final,
        source_visible=source_visible,
    )
    no_evidence = tensors["total_surface_counts"] == 0
    summary = {
        "tile_id": int(tile.tile_id),
        "source": "global baseline first-hit visibility",
        "tokens": int(tile.tokens),
        "visible_tokens": int(tensors["hard_visibility"].sum().item()),
        "back_tokens": int((~tensors["hard_visibility"]).sum().item()),
        "visible_fraction": float(
            tensors["hard_visibility"].float().mean().item()
        ),
        "no_surface_evidence_tokens": int(no_evidence.sum().item()),
        "no_surface_evidence_fraction": float(no_evidence.float().mean().item()),
        "no_surface_evidence_policy": "hard_visibility=0",
        "z_buffer": zbuffer_stats,
        "encoder_mapping": downsample_stats,
        "mapping": mapping.stats,
    }
    payload = {
        "coords": tile.coords_cpu,
        **tensors,
        "source": "global baseline first-hit visibility",
    }
    _atomic_torch_save(output_dir / "visibility_mask.pt", payload)
    _atomic_json(output_dir / "visibility_summary.json", summary)
    _save_visibility_visualizations(
        tile_image=tile_image,
        source_uv=source_uv,
        source_visible=source_visible,
        token_coords=tile.coords_cpu,
        hard_visibility=tensors["hard_visibility"],
        transform=tile.transform,
        output_dir=output_dir,
    )
    return {**payload, "summary": summary}


def _decoded_shape_visibility(
    *,
    pipeline: Any,
    tile: posterior.TileRuntime,
    baseline_mask: Mapping[str, Any],
    output_dir: Path,
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """Decode final routed shape and remap its first-hit C1024 support."""
    captured: Dict[str, SparseTensor] = {}
    decoder = pipeline.models["shape_slat_decoder"]

    def capture_output(_module: Any, _inputs: Any, output: SparseTensor) -> None:
        captured["output"] = output.detach().cpu()

    handle = decoder.output_layer.register_forward_hook(capture_output)
    try:
        shape_norm = posterior._sparse_from_cpu(
            tile.shape_final_norm_cpu,
            tile.coords_cpu,
            device=torch.device("cuda"),
        )
        shape_raw = baseline._denormalize(
            shape_norm, pipeline.shape_slat_normalization
        )
        with torch.no_grad():
            pipeline.decode_shape_slat(shape_raw, core.OVOXEL_RESOLUTION)
        output = captured.get("output")
        if output is None:
            raise RuntimeError("shape decoder output hook did not fire")
        surface = (output.feats[:, 3:6] > 0).any(dim=1)
        surface_coords = output.coords[surface].to(torch.int32)
        if surface_coords.shape[0] == 0:
            raise RuntimeError("routed shape decode has no surface O-Voxel")
        parents = surface_coords.clone()
        parents[:, 1:] //= 16
        token_key_set = set(
            _linear_keys(tile.coords_cpu[:, 1:], 64).tolist()
        )
        if not set(_linear_keys(parents[:, 1:], 64).tolist()).issubset(
            token_key_set
        ):
            raise RuntimeError(
                "decoded C1024 descendants do not map to input C64 support"
            )
        q_local = (
            ((surface_coords[:, 1:].float() + 0.5) / 1024.0 - 0.5)
            * (2.0 * float(tile.transform.mesh_scale))
        )
        surface_visible, zbuffer_stats, _ = _first_hit_visibility(
            q_local, tile.transform
        )
        final_coords, source_to_final = torch.unique(
            parents, dim=0, sorted=True, return_inverse=True
        )
        tensors = _aggregate_visibility_to_tokens(
            token_coords=tile.coords_cpu,
            final_coords=final_coords,
            source_to_final=source_to_final,
            source_visible=surface_visible,
        )
        no_evidence = tensors["total_surface_counts"] == 0
        summary = {
            "tile_id": int(tile.tile_id),
            "source": "routed final shape decode first-hit visibility",
            "tokens": int(tile.tokens),
            "surface_ovoxels": int(surface_coords.shape[0]),
            "visible_tokens": int(tensors["hard_visibility"].sum().item()),
            "back_tokens": int((~tensors["hard_visibility"]).sum().item()),
            "visible_fraction": float(
                tensors["hard_visibility"].float().mean().item()
            ),
            "no_surface_evidence_tokens": int(no_evidence.sum().item()),
            "no_surface_evidence_fraction": float(
                no_evidence.float().mean().item()
            ),
            "no_surface_evidence_policy": "hard_visibility=0",
            "decoder_mapping": (
                "released four-stage C64->C1024 subdivision ancestry; "
                "floor(descendant_coord/16) bitwise validated against input "
                "C64 support"
            ),
            "z_buffer": zbuffer_stats,
        }
        payload = {
            "coords": tile.coords_cpu,
            **tensors,
            "source": summary["source"],
        }
        _atomic_torch_save(output_dir / "texture_visibility_mask.pt", payload)
        _atomic_json(
            output_dir / "texture_visibility_summary.json", summary
        )
        return {**payload, "summary": summary}, False, None
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        fallback = {
            key: value
            for key, value in baseline_mask.items()
            if key != "summary"
        }
        summary = {
            **dict(baseline_mask["summary"]),
            "source": "baseline shape mask explicit fallback",
            "fallback_reason": reason,
        }
        _atomic_torch_save(
            output_dir / "texture_visibility_mask.pt", fallback
        )
        _atomic_json(
            output_dir / "texture_visibility_summary.json", summary
        )
        return {**fallback, "summary": summary}, True, reason
    finally:
        handle.remove()
        _empty_cuda_cache()


def _global_context(
    *,
    pipeline: Any,
    image_model: Any,
    image_1024: Image.Image,
    tile: posterior.TileRuntime,
) -> torch.Tensor:
    condition = posterior._make_condition(
        pipeline=pipeline,
        image_model=image_model,
        image=image_1024,
        coords_cpu=tile.coords_cpu,
        transform=tile.transform,
    )
    return condition["cond"]["global"].detach().cpu()


def _load_hr_conditions(
    *,
    tiles: Sequence[posterior.TileRuntime],
    source_cache_dir: Path,
    latent_name: str,
) -> Dict[int, Mapping[str, Any]]:
    output: Dict[int, Mapping[str, Any]] = {}
    for tile in tiles:
        path = (
            source_cache_dir
            / "per_tile"
            / f"tile_{tile.tile_id:02d}"
            / f"{latent_name}_conditions.pt"
        )
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("format") != "matched_lr_hr_condition_v1":
            raise RuntimeError(f"unexpected condition cache {path}")
        if not torch.equal(saved["coords"], tile.coords_cpu):
            raise RuntimeError(f"condition support mismatch in {path}")
        output[tile.tile_id] = saved["hr"]
    return output


def _make_routed_prediction_condition(
    *,
    hr: Mapping[str, Any],
    global_back: torch.Tensor,
    mask: Mapping[str, Any],
    mode: str,
    record_diagnostics: bool,
    intervention: bool,
) -> Dict[str, Any]:
    front = hr["cond"]
    negative = hr["neg_cond"]
    proj_front = front["proj"]
    proj_zero = proj_front.replace(torch.zeros_like(proj_front.feats))
    if mode == "proj_mask":
        back_global = front["global"]
        token_visibility = mask["hard_visibility"]
        routing_kind = "hard"
    elif mode == "hg_block":
        back_global = global_back
        token_visibility = mask["hard_visibility"]
        routing_kind = "hard"
    elif mode == "hzero_block":
        back_global = negative["global"]
        token_visibility = mask["hard_visibility"]
        routing_kind = "hard"
    elif mode == "hg_soft":
        back_global = global_back
        token_visibility = mask["visibility_ratio"]
        routing_kind = "soft"
    else:
        raise ValueError(mode)
    routed = {
        "mode": "visibility_routed",
        "global_front": front["global"],
        "proj_front": proj_front,
        "global_back": back_global,
        "proj_back": proj_zero,
        "token_visibility": token_visibility,
        "mask_coords": mask["coords"],
        "routing_kind": routing_kind,
        "record_diagnostics": bool(record_diagnostics),
        "record_self_attention_intervention": bool(intervention),
    }
    return {"cond": routed, "neg_cond": negative}


def _make_global_zero_projected_condition(
    *,
    hr: Mapping[str, Any],
    global_back: torch.Tensor,
) -> Dict[str, Any]:
    proj = hr["cond"]["proj"]
    return {
        "cond": {
            "global": global_back,
            "proj": proj.replace(torch.zeros_like(proj.feats)),
        },
        "neg_cond": hr["neg_cond"],
    }


def _collect_block_diagnostics(model: Any) -> List[Dict[str, float]]:
    output: List[Dict[str, float]] = []
    for block_index, block in enumerate(model.blocks):
        tensors = block.last_routing_tensors
        if not tensors:
            continue
        output.append(
            {
                "block": int(block_index),
                **{
                    name: float(value.detach().float().cpu().item())
                    for name, value in tensors.items()
                },
            }
        )
    return output


def _reuse_flow_outputs(
    *,
    output_dir: Path,
    tiles: Sequence[posterior.TileRuntime],
    latent_name: str,
    reuse_root: Path,
    route_enabled: bool,
    mode: str,
) -> Dict[str, Any]:
    """Load an explicitly named, coordinate-identical completed flow."""
    reuse_root = reuse_root.expanduser().resolve()
    reused_tiles: List[Dict[str, Any]] = []
    for tile in tiles:
        candidates = (
            reuse_root
            / f"tile_{tile.tile_id:02d}"
            / f"{latent_name}_trace.pt",
            reuse_root
            / "per_tile"
            / f"tile_{tile.tile_id:02d}"
            / f"{latent_name}_trace.pt",
        )
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise FileNotFoundError(
                f"{latent_name}: no reusable trace for tile "
                f"{tile.tile_id} under {reuse_root}"
            )
        payload = torch.load(source, map_location="cpu", weights_only=False)
        coords = payload.get("coords")
        final = payload.get("final_norm")
        trace = payload.get("trace", [])
        if not isinstance(coords, torch.Tensor) or not torch.equal(
            coords.to(torch.int32), tile.coords_cpu.to(torch.int32)
        ):
            raise RuntimeError(
                f"{latent_name}: reusable coords differ for tile "
                f"{tile.tile_id}"
            )
        if (
            not isinstance(final, torch.Tensor)
            or final.ndim != 2
            or final.shape != (tile.tokens, 32)
        ):
            raise RuntimeError(
                f"{latent_name}: invalid reusable final shape for tile "
                f"{tile.tile_id}: {getattr(final, 'shape', None)}"
            )
        final = final.detach().cpu().float().contiguous()
        if not torch.isfinite(final).all():
            raise RuntimeError(
                f"{latent_name}: reusable final contains non-finite values "
                f"for tile {tile.tile_id}"
            )
        if latent_name == "shape":
            tile.shape_final_norm_cpu = final
            tile.shape_trace = list(trace)
        elif latent_name == "texture":
            tile.texture_final_norm_cpu = final
            tile.texture_trace = list(trace)
        else:
            raise ValueError(latent_name)
        tile_dir = output_dir / f"tile_{tile.tile_id:02d}"
        _atomic_torch_save(
            tile_dir / f"{latent_name}_trace.pt",
            {
                "format": FORMAT_VERSION,
                "coords": tile.coords_cpu,
                "final_norm": final,
                "trace": trace,
                "reused_from": str(source),
            },
        )
        _atomic_json(
            tile_dir / f"{latent_name}_trace.json",
            {
                "steps": trace,
                "reused_from": str(source),
                "reuse_validation": (
                    "exact C64 coords, [tokens,32] shape, finite float32"
                ),
            },
        )
        reused_tiles.append(
            {
                "tile_id": int(tile.tile_id),
                "tokens": int(tile.tokens),
                "source": str(source),
            }
        )

    source_summary_path = reuse_root / "summary.json"
    source_flow: Dict[str, Any] = {}
    if source_summary_path.is_file():
        source_summary = json.loads(source_summary_path.read_text("utf-8"))
        candidate = source_summary.get(f"{latent_name}_flow")
        if isinstance(candidate, dict):
            source_flow = dict(candidate)
    return {
        **source_flow,
        "latent": latent_name,
        "conditioning_mode": mode,
        "route_enabled": bool(route_enabled and mode != "local"),
        "steps": int(source_flow.get("steps", 12)),
        "tiles": len(tiles),
        "tokens": int(sum(tile.tokens for tile in tiles)),
        "elapsed_seconds": 0.0,
        "reused": True,
        "reused_from": str(reuse_root),
        "reuse_validation": (
            "every tile exact C64 coords; final [tokens,32]; finite float32"
        ),
        "reused_tiles": reused_tiles,
    }


def _call_parameters(
    sampler: Any,
    params: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[Tuple[float, float]]]:
    call_params = dict(params)
    steps = int(call_params.pop("steps"))
    rescale_t = float(call_params.pop("rescale_t", 1.0))
    inference_parameters = inspect.signature(
        sampler._inference_model
    ).parameters
    if "guidance_interval" in inference_parameters:
        call_params.setdefault("guidance_interval", (0.0, 1.0))
    sequence = sampler.timestep_schedule(steps, rescale_t)
    return call_params, [
        (float(sequence[index]), float(sequence[index + 1]))
        for index in range(steps)
    ]


def _predict(
    *,
    sampler: Any,
    model: Any,
    state: SparseTensor,
    timestep: float,
    condition_cpu: Mapping[str, Any],
    call_params: Mapping[str, Any],
    concat_cond: Optional[SparseTensor],
) -> SparseTensor:
    parameters = dict(call_params)
    if concat_cond is not None:
        if not torch.equal(state.coords, concat_cond.coords):
            raise RuntimeError("texture concat condition coordinates differ")
        parameters["concat_cond"] = concat_cond
    condition = posterior._move_nested(condition_cpu, state.device)
    _, _, velocity = sampler._get_model_prediction(
        model,
        state,
        float(timestep),
        **dict(condition),
        **parameters,
    )
    if not isinstance(velocity, SparseTensor):
        raise TypeError("flow prediction is not SparseTensor")
    if not torch.equal(velocity.coords, state.coords):
        raise RuntimeError("flow prediction coordinates differ")
    if not torch.isfinite(velocity.feats).all():
        raise RuntimeError("flow prediction contains non-finite values")
    return velocity


@torch.no_grad()
def _run_flow(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    device: torch.device,
    output_dir: Path,
    tiles: Sequence[posterior.TileRuntime],
    latent_name: str,
    hr_conditions: Mapping[int, Mapping[str, Any]],
    global_context: torch.Tensor,
    masks: Mapping[int, Mapping[str, Any]],
    route_enabled: bool,
    params: Mapping[str, Any],
) -> Dict[str, Any]:
    if latent_name == "shape":
        sampler = pipeline.shape_slat_sampler
        model = pipeline.models["shape_slat_flow_model_1024"]
        channels = int(model.in_channels)
        seed_offset = 201
    elif latent_name == "texture":
        sampler = pipeline.tex_slat_sampler
        model = pipeline.models["tex_slat_flow_model_1024"]
        channels = int(model.in_channels) - 32
        seed_offset = 301
    else:
        raise ValueError(latent_name)
    if channels != 32:
        raise RuntimeError(f"{latent_name}: expected 32 noise channels")
    if bool(args.low_vram):
        model.to(device)
    call_params, time_pairs = _call_parameters(sampler, params)
    states = {
        tile.tile_id: posterior._random_sparse_state(
            tile,
            channels,
            device=device,
            seed=int(args.seed) + tile.tile_id * 1000 + seed_offset,
        )
        for tile in tiles
    }
    concat_conditions: Dict[int, Optional[SparseTensor]] = {}
    for tile in tiles:
        if latent_name == "texture":
            if tile.shape_final_norm_cpu is None:
                raise RuntimeError("texture requires final normalized shape")
            concat_conditions[tile.tile_id] = posterior._sparse_from_cpu(
                tile.shape_final_norm_cpu, tile.coords_cpu, device=device
            )
        else:
            concat_conditions[tile.tile_id] = None

    mode = str(args.conditioning_mode)
    block_mode = mode in {
        "proj_mask",
        "hg_block",
        "hzero_block",
        "hg_soft",
    }
    use_route = bool(route_enabled and mode != "local")
    step_summaries: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for step_index, (timestep, t_previous) in enumerate(time_pairs):
        tile_rows: List[Dict[str, Any]] = []
        for tile in tqdm(
            tiles,
            desc=f"{latent_name} {mode} step {step_index:02d}",
            leave=False,
        ):
            state = states[tile.tile_id]
            hr = hr_conditions[tile.tile_id]
            mask = masks[tile.tile_id]
            concat = concat_conditions[tile.tile_id]
            block_rows: List[Dict[str, float]] = []
            if not use_route:
                velocity_local = _predict(
                    sampler=sampler,
                    model=model,
                    state=state,
                    timestep=timestep,
                    condition_cpu=hr,
                    call_params=call_params,
                    concat_cond=concat,
                )
                velocity = velocity_local
            elif block_mode:
                routed_condition = _make_routed_prediction_condition(
                    hr=hr,
                    global_back=global_context,
                    mask=mask,
                    mode=mode,
                    record_diagnostics=True,
                    intervention=bool(
                        args.record_self_attention_intervention
                        and tile.tile_id == int(args.intervention_tile_id)
                        and step_index == int(args.intervention_step)
                    ),
                )
                velocity = _predict(
                    sampler=sampler,
                    model=model,
                    state=state,
                    timestep=timestep,
                    condition_cpu=routed_condition,
                    call_params=call_params,
                    concat_cond=concat,
                )
                block_rows = _collect_block_diagnostics(model)
                velocity_local = _predict(
                    sampler=sampler,
                    model=model,
                    state=state,
                    timestep=timestep,
                    condition_cpu=hr,
                    call_params=call_params,
                    concat_cond=concat,
                )
            elif mode == "hg_velocity":
                velocity_local = _predict(
                    sampler=sampler,
                    model=model,
                    state=state,
                    timestep=timestep,
                    condition_cpu=hr,
                    call_params=call_params,
                    concat_cond=concat,
                )
                global_condition = _make_global_zero_projected_condition(
                    hr=hr, global_back=global_context
                )
                velocity_global = _predict(
                    sampler=sampler,
                    model=model,
                    state=state,
                    timestep=timestep,
                    condition_cpu=global_condition,
                    call_params=call_params,
                    concat_cond=concat,
                )
                blend = mask["hard_visibility"].to(
                    device=device, dtype=velocity_local.dtype
                )[:, None]
                velocity = velocity_local.replace(
                    blend * velocity_local.feats
                    + (1.0 - blend) * velocity_global.feats
                )
            else:
                raise ValueError(mode)

            hard = mask["hard_visibility"].to(device=device)
            difference = velocity.feats.float() - velocity_local.feats.float()
            tile_row = {
                "tile_id": int(tile.tile_id),
                "tokens": int(tile.tokens),
                "visible_tokens": int(hard.sum().item()),
                "back_tokens": int((~hard).sum().item()),
                "routed_velocity_rms": _rms(velocity.feats),
                "local_hr_velocity_rms": _rms(velocity_local.feats),
                "routed_local_velocity_cosine": _cosine(
                    velocity.feats, velocity_local.feats
                ),
                "front_velocity_difference_rms": _rms(difference[hard]),
                "back_velocity_difference_rms": _rms(difference[~hard]),
                "block_diagnostics": block_rows,
            }
            states[tile.tile_id] = (
                state - float(timestep - t_previous) * velocity
            )
            if latent_name == "shape":
                tile.shape_trace.append(tile_row)
            else:
                tile.texture_trace.append(tile_row)
            tile_rows.append(tile_row)

        total_tokens = sum(row["tokens"] for row in tile_rows)

        def weighted(name: str) -> float:
            return float(
                sum(row[name] * row["tokens"] for row in tile_rows)
                / max(1, total_tokens)
            )

        summary = {
            "latent": latent_name,
            "conditioning_mode": mode,
            "route_enabled": use_route,
            "step": int(step_index),
            "timestep": float(timestep),
            "t_previous": float(t_previous),
            "tiles": len(tile_rows),
            "tokens": int(total_tokens),
            "visible_tokens": int(
                sum(row["visible_tokens"] for row in tile_rows)
            ),
            "back_tokens": int(sum(row["back_tokens"] for row in tile_rows)),
            "routed_velocity_rms_token_weighted": weighted(
                "routed_velocity_rms"
            ),
            "local_hr_velocity_rms_token_weighted": weighted(
                "local_hr_velocity_rms"
            ),
            "routed_local_velocity_cosine_token_weighted": weighted(
                "routed_local_velocity_cosine"
            ),
            "front_velocity_difference_rms_token_weighted": weighted(
                "front_velocity_difference_rms"
            ),
            "back_velocity_difference_rms_token_weighted": weighted(
                "back_velocity_difference_rms"
            ),
            "tiles_detail": tile_rows,
        }
        step_summaries.append(summary)
        _atomic_json(
            output_dir
            / "statistics"
            / f"{latent_name}_steps"
            / f"step_{step_index:02d}.json",
            summary,
        )

    _sync_cuda()
    for tile in tiles:
        final_cpu = states[tile.tile_id].feats.detach().cpu().float()
        if latent_name == "shape":
            tile.shape_final_norm_cpu = final_cpu
            trace = tile.shape_trace
        else:
            tile.texture_final_norm_cpu = final_cpu
            trace = tile.texture_trace
        tile_dir = output_dir / f"tile_{tile.tile_id:02d}"
        _atomic_torch_save(
            tile_dir / f"{latent_name}_trace.pt",
            {
                "format": FORMAT_VERSION,
                "coords": tile.coords_cpu,
                "final_norm": final_cpu,
                "trace": trace,
            },
        )
        _atomic_json(tile_dir / f"{latent_name}_trace.json", {"steps": trace})
    elapsed = float(time.perf_counter() - started)
    if bool(args.low_vram):
        model.cpu()
    del states, concat_conditions
    _empty_cuda_cache()
    return {
        "latent": latent_name,
        "conditioning_mode": mode,
        "route_enabled": use_route,
        "steps": len(time_pairs),
        "tiles": len(tiles),
        "tokens": sum(tile.tokens for tile in tiles),
        "elapsed_seconds": elapsed,
        "step_statistics": step_summaries,
    }


def _masked_psnr(
    reference: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> Optional[float]:
    if not bool(mask.any()):
        return None
    error = np.square(reference[mask] - prediction[mask]).mean()
    if error <= 1e-20:
        return float("inf")
    return float(10.0 * math.log10(1.0 / float(error)))


def _seam_energy(array: np.ndarray, boundaries: Sequence[int]) -> float:
    values: List[np.ndarray] = []
    for boundary in boundaries:
        if 1 <= boundary < array.shape[1] - 1:
            left = array[:, boundary] - array[:, boundary - 1]
            right = array[:, boundary + 1] - array[:, boundary]
            values.append(np.abs(left - right).reshape(-1))
        if 1 <= boundary < array.shape[0] - 1:
            upper = array[boundary] - array[boundary - 1]
            lower = array[boundary + 1] - array[boundary]
            values.append(np.abs(upper - lower).reshape(-1))
    if not values:
        return 0.0
    return float(np.concatenate(values).mean())


def _regional_lpips(
    reference: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    net: str,
) -> float:
    """Average the spatial LPIPS map over an explicitly fixed pixel region."""
    import lpips

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net=net, spatial=True).eval().to(device)
    reference_tensor = (
        torch.from_numpy(reference)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .mul(2.0)
        .sub(1.0)
    )
    prediction_tensor = (
        torch.from_numpy(prediction)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .mul(2.0)
        .sub(1.0)
    )
    with torch.no_grad():
        spatial = model(reference_tensor, prediction_tensor, normalize=False)
    if spatial.shape[-2:] != mask.shape:
        spatial = torch.nn.functional.interpolate(
            spatial,
            size=mask.shape,
            mode="bilinear",
            align_corners=False,
        )
    mask_tensor = torch.from_numpy(mask).to(
        device=spatial.device,
        dtype=torch.bool,
    )
    if not bool(mask_tensor.any()):
        raise RuntimeError("regional LPIPS mask is empty")
    result = float(spatial[0, 0][mask_tensor].mean().item())
    del model, reference_tensor, prediction_tensor, spatial, mask_tensor
    _empty_cuda_cache()
    return result


def _boundary_metrics(
    output_dir: Path,
    *,
    lpips_net: str,
    skip_lpips: bool,
) -> Dict[str, Any]:
    eval_dir = output_dir / "aligned_eval"
    reference = (
        np.asarray(Image.open(eval_dir / "original.png").convert("RGB"))
        .astype(np.float32)
        / 255.0
    )
    render = (
        np.asarray(Image.open(eval_dir / "render.png").convert("RGB"))
        .astype(np.float32)
        / 255.0
    )
    height, width = reference.shape[:2]
    canonical_boundaries = list(range(768, 3329, 512))
    boundaries = [
        int(round(value * width / 4096.0))
        for value in canonical_boundaries
    ]
    band = np.zeros((height, width), dtype=bool)
    # The fixed measurement band is exactly one render pixel wide. Its
    # location is determined before measuring any generated output.
    band_width = 1
    for value in boundaries:
        band[:, max(0, value) : min(width, value + band_width)] = True
        band[max(0, value) : min(height, value + band_width), :] = True
    interior = ~band
    channels: Dict[str, float] = {}
    for name in ("render", "base_color", "roughness", "metallic"):
        image = np.asarray(Image.open(eval_dir / f"{name}.png")).astype(
            np.float32
        )
        if image.max(initial=0.0) > 1.0:
            image /= 255.0
        channels[name] = _seam_energy(image, boundaries)
    if skip_lpips:
        boundary_lpips = None
        interior_lpips = None
        lpips_note = "regional LPIPS explicitly skipped by --skip-lpips"
    else:
        boundary_lpips = _regional_lpips(
            reference,
            render,
            band,
            net=lpips_net,
        )
        interior_lpips = _regional_lpips(
            reference,
            render,
            interior,
            net=lpips_net,
        )
        lpips_note = (
            f"LPIPS-{lpips_net} spatial output averaged over the fixed "
            "one-pixel boundary band and its complement"
        )
    return {
        "tile_layout": "4096 canonical; tile=1024, stride=512, 7x7",
        "render_resolution": [width, height],
        "canonical_boundary_positions": canonical_boundaries,
        "boundary_positions": boundaries,
        "boundary_band_width_pixels": band_width,
        "boundary_band_derivation": (
            "fixed nearest-tile-center ownership midlines at canonical "
            "coordinates 768+512k, selected before output measurement"
        ),
        "boundary_fraction": float(band.mean()),
        "boundary_psnr_db": _masked_psnr(reference, render, band),
        "interior_psnr_db": _masked_psnr(reference, render, interior),
        "boundary_lpips": boundary_lpips,
        "interior_lpips": interior_lpips,
        "lpips_note": lpips_note,
        "gradient_jump_energy": channels,
    }


def _update_suite_summary(suite_root: Path) -> None:
    configurations: Dict[str, Any] = {}
    for child in sorted(suite_root.iterdir()):
        path = child / "summary.json"
        if child.is_dir() and path.is_file():
            try:
                configurations[child.name] = json.loads(
                    path.read_text("utf-8")
                )
            except Exception:
                continue
    rows: Dict[str, Any] = {}
    for name, summary in configurations.items():
        evaluation = summary.get("evaluation")
        rows[name] = {
            "status": "complete" if evaluation else "flow_only",
            "conditioning": summary.get("configuration"),
            "successful_tiles": summary.get("successful_tiles"),
            "render_metrics": (
                evaluation.get("render_metrics") if evaluation else None
            ),
            "boundary_metrics": (
                evaluation.get("boundary_metrics") if evaluation else None
            ),
            "geometry_diagnostics": (
                evaluation.get("geometry_diagnostics") if evaluation else None
            ),
            "material_diagnostics": (
                evaluation.get("material_diagnostics") if evaluation else None
            ),
        }
    _atomic_json(
        suite_root / "summary.json",
        {
            "format": FORMAT_VERSION,
            "configurations": rows,
            "success_claim": False,
            "success_claim_reason": (
                "the generated summary never auto-claims success; the "
                "cross-configuration and multiview decision is recorded in "
                "EXPERIMENT_REPORT.md"
            ),
        },
    )


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(int(args.cuda_device))
    device = torch.device("cuda")
    output_dir = Path(args.output_dir).expanduser().resolve()
    suite_root = Path(args.suite_root).expanduser().resolve()
    source_cache_dir = Path(args.source_cache_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[cuda] requested={args.cuda_device} current={torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())}"
    )
    pipeline = init_pipeline(
        args.model_path, device="cuda", low_vram=bool(args.low_vram)
    )
    run_started = time.perf_counter()
    (
        image_4096,
        image_1024,
        global_camera,
        baseline_mesh,
        global_q,
        global_uv,
        global_finite,
        global_face_uv,
        global_face_finite,
        global_source,
    ) = posterior._prepare_global_source(
        args=args,
        pipeline=pipeline,
        output_dir=output_dir,
        source_cache_dir=source_cache_dir,
    )
    tiles, source_records = posterior._prepare_tiles(
        args=args,
        pipeline=pipeline,
        device=device,
        source_cache_dir=source_cache_dir,
        image_4096=image_4096,
        image_1024=image_1024,
        global_camera=global_camera,
        baseline_mesh=baseline_mesh,
        global_ovoxel_q=global_q,
        global_ovoxel_uv=global_uv,
        global_ovoxel_finite=global_finite,
        global_face_uv=global_face_uv,
        global_face_finite=global_face_finite,
    )
    if any(row["status"] == "failed" for row in source_records):
        raise RuntimeError("one or more source tiles failed")

    baseline_masks: Dict[int, Dict[str, Any]] = {}
    for tile in tqdm(tiles, desc="baseline first-hit visibility"):
        tile_dir = output_dir / f"tile_{tile.tile_id:02d}"
        baseline_masks[tile.tile_id] = _build_baseline_visibility(
            tile=tile,
            baseline_mesh=baseline_mesh,
            global_q=global_q,
            global_uv=global_uv,
            global_finite=global_finite,
            global_camera=global_camera,
            tile_image=image_4096.crop(tile.box),
            output_dir=tile_dir,
        )

    _, shape_params, texture_params = posterior._sampler_params(args)
    shape_hr = _load_hr_conditions(
        tiles=tiles,
        source_cache_dir=source_cache_dir,
        latent_name="shape",
    )
    shape_global = _global_context(
        pipeline=pipeline,
        image_model=pipeline.image_cond_model_shape_1024,
        image_1024=image_1024,
        tile=tiles[0],
    )
    if args.reuse_shape_from:
        shape_flow = _reuse_flow_outputs(
            output_dir=output_dir,
            tiles=tiles,
            latent_name="shape",
            reuse_root=Path(args.reuse_shape_from),
            route_enabled=bool(args.route_shape),
            mode=str(args.conditioning_mode),
        )
    else:
        shape_flow = _run_flow(
            args=args,
            pipeline=pipeline,
            device=device,
            output_dir=output_dir,
            tiles=tiles,
            latent_name="shape",
            hr_conditions=shape_hr,
            global_context=shape_global,
            masks=baseline_masks,
            route_enabled=bool(args.route_shape),
            params=shape_params,
        )

    texture_masks: Dict[int, Dict[str, Any]] = {}
    texture_fallbacks: List[Dict[str, Any]] = []
    for tile in tqdm(tiles, desc="texture visibility"):
        tile_dir = output_dir / f"tile_{tile.tile_id:02d}"
        if args.texture_mask_source == "baseline_shape":
            texture_masks[tile.tile_id] = baseline_masks[tile.tile_id]
            _atomic_json(
                tile_dir / "texture_visibility_summary.json",
                {
                    **baseline_masks[tile.tile_id]["summary"],
                    "source": "baseline shape mask explicitly requested",
                },
            )
        else:
            mask, fallback, reason = _decoded_shape_visibility(
                pipeline=pipeline,
                tile=tile,
                baseline_mask=baseline_masks[tile.tile_id],
                output_dir=tile_dir,
            )
            texture_masks[tile.tile_id] = mask
            if fallback:
                texture_fallbacks.append(
                    {"tile_id": tile.tile_id, "reason": reason}
                )

    texture_hr = _load_hr_conditions(
        tiles=tiles,
        source_cache_dir=source_cache_dir,
        latent_name="texture",
    )
    texture_global = _global_context(
        pipeline=pipeline,
        image_model=pipeline.image_cond_model_tex_1024,
        image_1024=image_1024,
        tile=tiles[0],
    )
    if args.reuse_texture_from:
        texture_flow = _reuse_flow_outputs(
            output_dir=output_dir,
            tiles=tiles,
            latent_name="texture",
            reuse_root=Path(args.reuse_texture_from),
            route_enabled=bool(args.route_texture),
            mode=str(args.conditioning_mode),
        )
    else:
        texture_flow = _run_flow(
            args=args,
            pipeline=pipeline,
            device=device,
            output_dir=output_dir,
            tiles=tiles,
            latent_name="texture",
            hr_conditions=texture_hr,
            global_context=texture_global,
            masks=texture_masks,
            route_enabled=bool(args.route_texture),
            params=texture_params,
        )

    if bool(args.decode):
        evaluation = posterior._decode_and_evaluate(
            args=args,
            pipeline=pipeline,
            device=device,
            output_dir=output_dir,
            tiles=tiles,
            global_camera=global_camera,
            baseline_mesh=baseline_mesh,
        )
        geometry = evaluation.get("geometry_diagnostics", {})
        vertex_count = int(geometry.get("vertices", 0) or 0)
        largest_count = int(
            geometry.get("largest_component_vertices", 0) or 0
        )
        geometry["largest_component_vertex_ratio"] = (
            float(largest_count / vertex_count) if vertex_count else None
        )
        evaluation["boundary_metrics"] = _boundary_metrics(
            output_dir,
            lpips_net=args.lpips_net,
            skip_lpips=bool(args.skip_lpips),
        )
    else:
        evaluation = None
    mask_summaries = [mask["summary"] for mask in baseline_masks.values()]
    texture_mask_summaries = [
        mask["summary"] for mask in texture_masks.values()
    ]

    def aggregate_visibility(
        rows: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "visible_tokens": int(
                sum(int(row["visible_tokens"]) for row in rows)
            ),
            "back_tokens": int(
                sum(int(row["back_tokens"]) for row in rows)
            ),
            "no_surface_evidence_tokens": int(
                sum(int(row["no_surface_evidence_tokens"]) for row in rows)
            ),
            "per_tile": list(rows),
        }

    summary = {
        "format": FORMAT_VERSION,
        "image": str(Path(args.image).expanduser().resolve()),
        "output_dir": str(output_dir),
        "cuda_device": int(args.cuda_device),
        "seed": int(args.seed),
        "global_seed": int(args.global_seed),
        "configuration": {
            "conditioning_mode": args.conditioning_mode,
            "route_shape": bool(args.route_shape),
            "route_texture": bool(args.route_texture),
            "texture_mask_source": args.texture_mask_source,
            "test_time_intervention": True,
            "weight_updates": 0,
            "latent_fusion": False,
            "velocity_fusion": args.conditioning_mode == "hg_velocity",
            "reuse_shape_from": args.reuse_shape_from,
            "reuse_texture_from": args.reuse_texture_from,
        },
        "global_source": global_source,
        "successful_tiles": len(tiles),
        "skipped_tiles": sum(
            row["status"] == "skipped" for row in source_records
        ),
        "source_tiles": source_records,
        "visibility": aggregate_visibility(mask_summaries),
        "texture_visibility": aggregate_visibility(texture_mask_summaries),
        "texture_mask_fallbacks": texture_fallbacks,
        "shape_flow": shape_flow,
        "texture_flow": texture_flow,
        "evaluation": evaluation,
        "generation_seconds": float(time.perf_counter() - run_started),
        "success_claim": False,
        "success_claim_reason": "requires cross-configuration comparison",
    }
    _atomic_json(output_dir / "summary.json", summary)
    _update_suite_summary(suite_root)
    print(f"[done] summary={output_dir / 'summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=str(Path(__file__).parent / "assets" / "choose" / "0_img.png"),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/visibility_routed_conditioning/joint_hg",
    )
    parser.add_argument(
        "--suite-root", default="outputs/visibility_routed_conditioning"
    )
    parser.add_argument(
        "--source-cache-dir",
        default="outputs/joint_online_canonical_posterior/source_cache",
    )
    parser.add_argument("--reuse-shape-from", default=None)
    parser.add_argument("--reuse-texture-from", default=None)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--moge-model-path", default=None)
    parser.add_argument("--global-mesh-cache", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument(
        "--conditioning-mode",
        choices=CONDITIONING_MODES,
        default="hg_block",
    )
    parser.add_argument(
        "--route-shape",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--route-texture",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--texture-mask-source",
        choices=("routed_shape", "baseline_shape"),
        default="routed_shape",
    )
    parser.add_argument(
        "--decode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--low-vram",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--record-self-attention-intervention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--intervention-tile-id", type=int, default=24)
    parser.add_argument("--intervention-step", type=int, default=0)
    parser.add_argument("--tile-ids", default=None)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--min-tile-ovoxels", type=int, default=1001)
    parser.add_argument("--max-num-tokens", type=int, default=1_000_000)
    parser.add_argument("--face-projection-chunk-size", type=int, default=250_000)
    parser.add_argument("--vertex-weld-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--save-mesh-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--shape-encoder",
        default=str(core.DEFAULT_ENCODER_ROOT / "shape_enc_next_dc_f16c32_fp16"),
    )
    parser.add_argument(
        "--pbr-encoder",
        default=str(core.DEFAULT_ENCODER_ROOT / "tex_enc_next_dc_f16c32_fp16"),
    )
    parser.add_argument("--fov", type=float, default=-1.0)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--extend-pixel", type=int, default=0)
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
    parser.add_argument("--envmap", default="studio")
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--metric-resolution", type=int, default=1024)
    parser.add_argument("--render-ssaa", type=int, default=2)
    parser.add_argument("--render-peel-layers", type=int, default=8)
    parser.add_argument("--render-face-chunk-size", type=int, default=4_000_000)
    parser.add_argument(
        "--render-multiview",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--multiview-resolution", type=int, default=512)
    parser.add_argument("--multiview-ssaa", type=int, default=1)
    parser.add_argument("--multiview-peel-layers", type=int, default=2)
    parser.add_argument("--multiview-radius-scale", type=float, default=1.0)
    parser.add_argument(
        "--multiview-yaws-degrees", default="0,-45,45,-90,90,180"
    )
    parser.add_argument(
        "--multiview-pitches-degrees", default="0,0,0,0,0,0"
    )
    parser.add_argument(
        "--use-envmap-bg",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--skip-lpips",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg"
    )
    parser.add_argument("--surface-samples", type=int, default=10_000)
    parser.add_argument("--overlap-samples", type=int, default=2_048)
    parser.add_argument("--nearest-chunk-size", type=int, default=1_024)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not Path(args.image).expanduser().is_file():
        raise FileNotFoundError(args.image)
    if int(args.cuda_device) != 4:
        raise ValueError("Codex.md requires physical CUDA device 4")
    if args.conditioning_mode == "local" and (
        bool(args.route_shape) or bool(args.route_texture)
    ):
        print("[config] local mode ignores route-shape/route-texture flags")
    requested = core._parse_tile_ids(args.tile_ids)
    if requested is not None:
        invalid = sorted(value for value in requested if value not in range(49))
        if invalid:
            raise ValueError(f"invalid tile IDs: {invalid}")
    for name in (
        "shape_steps",
        "texture_steps",
        "min_tile_ovoxels",
        "max_num_tokens",
        "render_resolution",
        "metric_resolution",
        "surface_samples",
        "overlap_samples",
        "nearest_chunk_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
