#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired fixed-shape texture-condition ablation for Pixal3D.

This experiment keeps the global-baseline mesh, local dual-grid, shape
encoder, texture sampler, decoder, local-to-global return, stitcher and
renderer fixed.  The only ablated component is the texture image condition:

* local_tile_condition: the existing local LR tile image condition;
* global_1024_routed_condition: one complete canonical 1024 image extraction
  shared by every tile, with its DINO pixel feature maps queried at points
  routed through the existing local-q -> global-q -> global-camera projection.

There is deliberately no shape flow, PBR encoder, texture reference endpoint,
G_tex, prefix/suffix flow, velocity/x0 intervention, CCA, mask, fusion or new
network in this file.  Texture flow starts at one shared t=1 random texture
SLat state and uses the native full FlowEulerSampler for both groups.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

import pixal3d.models as pixal3d_models
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as base
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVertexPbr, MeshWithVoxel


LOCAL_TILE_CONDITION = "local_tile_condition"
GLOBAL_ROUTED_CONDITION = "global_1024_routed_condition"
GROUPS = (LOCAL_TILE_CONDITION, GLOBAL_ROUTED_CONDITION)


@dataclass
class GlobalTextureFeatureCache:
    global_tokens: torch.Tensor
    pixel_feature_maps: Tuple[torch.Tensor, ...]
    map_names: Tuple[str, ...]
    image_resolution: int
    image_model_image_size: int
    feature_extraction_calls: int


def _safe_tensor_stats(value: torch.Tensor, *, include_range: bool = True) -> Dict[str, Any]:
    value = value.detach()
    result: Dict[str, Any] = {
        "shape": [int(v) for v in value.shape],
        "dtype": str(value.dtype),
        "device": str(value.device),
        "numel": int(value.numel()),
    }
    if value.numel():
        value_f = value.to(torch.float32)
        result.update(
            {
                "mean": float(value_f.mean().item()),
                "std": float(value_f.std(unbiased=False).item()),
                "min": float(value_f.amin().item()),
                "max": float(value_f.amax().item()),
                "l2": float(torch.linalg.vector_norm(value_f.reshape(-1)).item()),
            }
        )
        if include_range and value.ndim >= 2 and value.shape[-1] <= 64:
            result["channel_mean"] = [
                float(v) for v in value_f.reshape(-1, value.shape[-1]).mean(0).cpu().tolist()
            ]
            result["channel_std"] = [
                float(v) for v in value_f.reshape(-1, value.shape[-1]).std(0, unbiased=False).cpu().tolist()
            ]
    return result


def _pil_to_image_model_tensor(
    image: Image.Image,
    *,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    resized = image.convert("RGB").resize(
        (int(image_size), int(image_size)), Image.Resampling.LANCZOS
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .to(device=device)
    )


@torch.no_grad()
def _extract_global_texture_feature_cache(
    pipeline: Any,
    canonical_image: Image.Image,
) -> GlobalTextureFeatureCache:
    """Extract complete-image DINO maps once, matching the native extractor."""
    model = pipeline.image_cond_model_tex_1024
    device = torch.device(pipeline.device)
    if pipeline.low_vram:
        model.to(device)
    model.eval()
    image_size = int(model.image_size)
    image_tensor = _pil_to_image_model_tensor(
        canonical_image, image_size=image_size, device=device
    )
    image_for_naf = image_tensor.clone() if bool(model.use_naf_upsample) else None
    normalized = None
    z = None
    patch_tokens = None
    patch_spatial = None
    low_map = None
    try:
        normalized = model.transform(image_tensor)
        # This is the one complete-image image_cond_model_tex_1024 extraction.
        # The following split is copied from DinoV3ProjFeatureExtractor.forward
        # so that arbitrary routed pixels can be queried without a tile crop.
        z = model.extract_features(normalized)
        batch = int(z.shape[0])
        num_register = int(getattr(model.model.config, "num_register_tokens", 4))
        cls_token = z[:, 0:1]
        register_tokens = z[:, 1 : 1 + num_register]
        patch_tokens = z[:, 1 + num_register :]
        patch_spatial = patch_tokens.reshape(
            batch, int(model.patch_number), int(model.patch_number), -1
        )
        low_map = patch_spatial.permute(0, 3, 1, 2).contiguous().detach()
        maps: List[torch.Tensor] = [low_map]
        map_names: List[str] = ["dino_patch_lr"]
        if bool(model.use_naf_upsample):
            model._load_naf()
            high_map = model.naf_model(
                image_for_naf,
                low_map,
                model.naf_target_size,
            ).detach()
            maps.append(high_map)
            map_names.append("dino_patch_naf_hr")
        global_tokens = torch.cat([cls_token, register_tokens], dim=1).detach()
        for tensor in (global_tokens, *maps):
            if not torch.isfinite(tensor).all():
                raise RuntimeError("complete canonical1024 texture features contain non-finite values")
        base._sync_cuda()
        cache = GlobalTextureFeatureCache(
            global_tokens=global_tokens,
            pixel_feature_maps=tuple(maps),
            map_names=tuple(map_names),
            image_resolution=int(image_size),
            image_model_image_size=int(image_size),
            feature_extraction_calls=1,
        )
    finally:
        del image_tensor
        for value in (normalized, z, patch_tokens, patch_spatial, low_map):
            if value is not None:
                del value
        if image_for_naf is not None:
            del image_for_naf
        if pipeline.low_vram:
            model.cpu()
            base._empty_cuda_cache()
    print(
        "[global-texture-features] "
        f"one complete image extraction maps={list(cache.map_names)} "
        f"global={tuple(cache.global_tokens.shape)}"
    )
    return cache


@torch.no_grad()
def _sample_feature_maps(
    feature_maps: Sequence[torch.Tensor],
    uv_pixels: torch.Tensor,
    *,
    image_resolution: int,
    chunk_size: int,
) -> torch.Tensor:
    """Sample maps using the exact ProjGrid pixel-center normalization."""
    if int(chunk_size) <= 0:
        raise ValueError("feature query chunk size must be positive")
    if uv_pixels.ndim != 2 or uv_pixels.shape[1] != 2:
        raise ValueError(f"uv_pixels must have shape [N, 2], got {tuple(uv_pixels.shape)}")
    chunks: List[torch.Tensor] = []
    for feature_map in feature_maps:
        if feature_map.ndim != 4 or feature_map.shape[0] != 1:
            raise ValueError(f"feature map must have shape [1, C, H, W], got {tuple(feature_map.shape)}")
        rows: List[torch.Tensor] = []
        for start in range(0, int(uv_pixels.shape[0]), int(chunk_size)):
            end = min(start + int(chunk_size), int(uv_pixels.shape[0]))
            grid = (uv_pixels[start:end] + 0.5) / float(image_resolution) * 2.0 - 1.0
            grid = grid.reshape(1, end - start, 1, 2).to(
                device=feature_map.device, dtype=feature_map.dtype
            )
            sampled = F.grid_sample(
                feature_map,
                grid,
                mode="bilinear",
                align_corners=False,
                padding_mode="border",
            )
            rows.append(sampled.squeeze(0).squeeze(-1).transpose(0, 1))
        chunks.append(torch.cat(rows, dim=0) if rows else feature_map.new_empty((0, feature_map.shape[1])))
    return torch.cat(chunks, dim=1)


def _latent_coords_to_local_q(coords: torch.Tensor) -> torch.Tensor:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"latent coords must have shape [N, 4], got {tuple(coords.shape)}")
    one_dim = torch.linspace(
        -1.0,
        1.0,
        int(base.LATENT_RESOLUTION),
        device=coords.device,
        dtype=torch.float32,
    )
    xyz = coords[:, 1:4].to(torch.long)
    if xyz.numel() and bool(((xyz < 0) | (xyz >= int(base.LATENT_RESOLUTION))).any().item()):
        raise RuntimeError("fixed local SLat support is outside the C64 grid")
    # This is the same coordinate-to-q ordering used by ProjGrid when it
    # receives grid_indices: q_local = [x, y, z] in camera-q convention.
    return one_dim[xyz]


def _route_sanity_rows(
    *,
    indices: torch.Tensor,
    q_local: torch.Tensor,
    q_global: torch.Tensor,
    uv_4096: torch.Tensor,
    uv_1024: torch.Tensor,
    sampled_features: torch.Tensor,
    count: int,
) -> List[Dict[str, Any]]:
    if indices.numel() == 0:
        return []
    take = indices[: min(int(count), int(indices.shape[0]))]
    rows: List[Dict[str, Any]] = []
    for index in take.tolist():
        rows.append(
            {
                "support_row": int(index),
                "q_local": [float(v) for v in q_local[index].cpu().tolist()],
                "q_global": [float(v) for v in q_global[index].cpu().tolist()],
                "uv_4096": [float(v) for v in uv_4096[index].cpu().tolist()],
                "uv_canonical_1024": [float(v) for v in uv_1024[index].cpu().tolist()],
                "projected_feature_prefix": [
                    float(v) for v in sampled_features[index, : min(8, sampled_features.shape[1])].to(torch.float32).cpu().tolist()
                ],
            }
        )
    return rows


@torch.no_grad()
def _make_global_routed_condition(
    *,
    cache: GlobalTextureFeatureCache,
    shape_coords: torch.Tensor,
    global_camera: Mapping[str, float],
    transform: base.TileCameraTransform,
    query_chunk_size: int,
    sanity_token_count: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Route local C64 support through local->global->global camera projection."""
    q_local = _latent_coords_to_local_q(shape_coords)
    q_global, uv_from_local = base._local_q_to_global_q(
        q_local,
        global_camera=global_camera,
        transform=transform,
    )
    uv_reprojected, global_depth, finite = base._project_global_q_to_4096(
        q_global,
        global_camera=global_camera,
    )
    if not bool(finite.all().item()):
        raise RuntimeError("global routed condition produced invalid global projections")
    route_pixel_error = (uv_reprojected - uv_from_local).abs()
    if float(route_pixel_error.max().item()) > 2e-3:
        raise RuntimeError(
            "local->global->global-camera route is not self-consistent: "
            f"max pixel error={float(route_pixel_error.max().item()):.6g}"
        )
    uv_1024 = uv_reprojected * (
        float(base.GLOBAL_IMAGE_SIZE) / float(base.CANONICAL_IMAGE_SIZE)
    )
    sampled_features = _sample_feature_maps(
        cache.pixel_feature_maps,
        uv_1024,
        image_resolution=int(cache.image_resolution),
        chunk_size=int(query_chunk_size),
    )
    coords = shape_coords.to(device=sampled_features.device, dtype=torch.int32)
    projected = SparseTensor(sampled_features, coords)
    neg_projected = SparseTensor(torch.zeros_like(sampled_features), coords)
    condition = {
        "cond": {
            "global": cache.global_tokens,
            "proj": projected,
        },
        "neg_cond": {
            "global": torch.zeros_like(cache.global_tokens),
            "proj": neg_projected,
        },
    }
    in_bounds = (
        (uv_1024[:, 0] >= 0.0)
        & (uv_1024[:, 0] < float(base.GLOBAL_IMAGE_SIZE))
        & (uv_1024[:, 1] >= 0.0)
        & (uv_1024[:, 1] < float(base.GLOBAL_IMAGE_SIZE))
    )
    route_stats = {
        "source_image": "complete canonical_1024",
        "source_image_path_role": "one shared global image; no tile crop",
        "image_model": "image_cond_model_tex_1024",
        "global_image_condition_calls": int(cache.feature_extraction_calls),
        "global_token_cache_shared_across_tiles": True,
        "global_tokens": _safe_tensor_stats(cache.global_tokens),
        "pixel_feature_maps": {
            name: _safe_tensor_stats(feature_map, include_range=False)
            for name, feature_map in zip(cache.map_names, cache.pixel_feature_maps)
        },
        "query": {
            "support_order_preserved": True,
            "support_coords_shape": [int(v) for v in shape_coords.shape],
            "local_coordinate_to_q": "C64 integer coords -> linspace(-1,1,64) in x,y,z order",
            "route": "local SLat q -> existing local_q_to_global_q -> global camera reprojection -> canonical1024 UV",
            "projection_map_query": "bilinear grid_sample with ProjGrid pixel-center normalization and border padding",
            "direct_local_coord_into_global_map": False,
            "tile_crop_for_global_condition": False,
            "texture_condition_4096": False,
            "canonical_1024_uv_range": base._tensor_range(uv_1024),
            "global_q_range": base._tensor_range(q_global),
            "global_depth_range": base._tensor_range(global_depth[:, None]),
            "in_bounds_tokens": int(in_bounds.sum().item()),
            "out_of_bounds_tokens": int((~in_bounds).sum().item()),
            "route_pixel_error_max": float(route_pixel_error.max().item()),
            "route_pixel_error_mean": float(route_pixel_error.mean().item()),
            "sampled_feature_shape": [int(v) for v in sampled_features.shape],
            "sampled_feature_range": base._tensor_range(sampled_features),
        },
        "sanity_tokens": _route_sanity_rows(
            indices=torch.arange(shape_coords.shape[0], device=shape_coords.device),
            q_local=q_local,
            q_global=q_global,
            uv_4096=uv_reprojected,
            uv_1024=uv_1024,
            sampled_features=sampled_features,
            count=int(sanity_token_count),
        ),
    }
    return condition, route_stats


@torch.no_grad()
def _make_local_tile_condition(
    *,
    pipeline: Any,
    lr_tile_image: Image.Image,
    shape_coords: torch.Tensor,
    transform: base.TileCameraTransform,
) -> Dict[str, Any]:
    condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [lr_tile_image.convert("RGB")],
        shape_coords,
        camera_angle_x=float(transform.camera_angle_x),
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
        grid_resolution_override=int(base.LATENT_RESOLUTION),
    )
    projected = condition["cond"]["proj"]
    if not torch.equal(projected.coords.to(torch.int32), shape_coords.to(torch.int32)):
        raise RuntimeError("local tile condition changed the fixed shape support order")
    return condition


def _clone_sparse(value: SparseTensor) -> SparseTensor:
    # Keep the sparse spatial caches attached to the fixed shape support.
    # Pixal3D's native texture sampler constructs its noise with
    # shape_slat.replace(feats=...), and the decoder relies on those caches
    # when the shape subdivision guides the texture decoder.
    return value.replace(value.feats.detach().clone())


@torch.no_grad()
def _run_one_texture_flow(
    *,
    pipeline: Any,
    fixed_shape_norm: SparseTensor,
    texture_noise: SparseTensor,
    condition: Mapping[str, Any],
    texture_params: Mapping[str, Any],
    description: str,
) -> Tuple[SparseTensor, Dict[str, Any]]:
    model = pipeline.models["tex_slat_flow_model_1024"]
    sampler = pipeline.tex_slat_sampler
    merged_params = {**pipeline.tex_slat_sampler_params, **dict(texture_params)}
    device = torch.device(pipeline.device)
    if pipeline.low_vram:
        model.to(device)
    started = time.perf_counter()
    try:
        result = sampler.sample(
            model,
            _clone_sparse(texture_noise),
            cond=condition["cond"],
            neg_cond=condition["neg_cond"],
            concat_cond=fixed_shape_norm,
            **merged_params,
            verbose=True,
            tqdm_desc=description,
            record_trajectory=False,
            return_model_history=False,
        )
    finally:
        if pipeline.low_vram:
            model.cpu()
    base._sync_cuda()
    output = getattr(result, "samples", result)
    if not isinstance(output, SparseTensor):
        raise RuntimeError(f"{description} returned {type(output)!r}, expected SparseTensor")
    if not torch.equal(output.coords.to(torch.int32), fixed_shape_norm.coords.to(torch.int32)):
        raise RuntimeError(f"{description} changed the fixed local support")
    return output, {
        "flow_seconds": float(time.perf_counter() - started),
        "flow_steps": int(merged_params["steps"]),
        "sampler": {
            "steps": int(merged_params["steps"]),
            "rescale_t": float(merged_params["rescale_t"]),
            "guidance_strength": float(merged_params.get("guidance_strength", 0.0)),
            "guidance_rescale": float(merged_params.get("guidance_rescale", 0.0)),
        },
        "execution": "native FlowEulerSampler.sample from its direct t=1 noise state",
        "record_trajectory": False,
        "texture_reference_or_endpoint": False,
        "support_preserved": True,
    }


@torch.no_grad()
def _run_paired_texture_flows(
    *,
    pipeline: Any,
    fixed_shape_norm: SparseTensor,
    conditions: Mapping[str, Mapping[str, Any]],
    texture_params: Mapping[str, Any],
    seed: int,
    tile_id: int,
) -> Tuple[Dict[str, SparseTensor], Dict[str, Any]]:
    model = pipeline.models["tex_slat_flow_model_1024"]
    device = torch.device(pipeline.device)
    texture_channels = int(model.in_channels) - int(fixed_shape_norm.feats.shape[1])
    if texture_channels <= 0:
        raise RuntimeError(
            f"texture flow channels are not positive: model={model.in_channels} "
            f"fixed_shape={fixed_shape_norm.feats.shape[1]}"
        )
    coords = fixed_shape_norm.coords.to(device=device, dtype=torch.int32)
    base._seed_everything(int(seed))
    texture_noise = fixed_shape_norm.replace(
        torch.randn(
            int(coords.shape[0]),
            texture_channels,
            device=device,
            dtype=torch.float32,
        )
    )
    merged_params = {**pipeline.tex_slat_sampler_params, **dict(texture_params)}
    schedule = [
        float(v)
        for v in pipeline.tex_slat_sampler.timestep_schedule(
            int(merged_params["steps"]), float(merged_params["rescale_t"])
        )
    ]
    outputs: Dict[str, SparseTensor] = {}
    branch_stats: Dict[str, Any] = {}
    for group in GROUPS:
        output, stats = _run_one_texture_flow(
            pipeline=pipeline,
            fixed_shape_norm=fixed_shape_norm,
            texture_noise=texture_noise,
            condition=conditions[group],
            texture_params=texture_params,
            description=f"Tile {tile_id:02d} {group} texture flow",
        )
        outputs[group] = output
        branch_stats[group] = stats
    noise_l2 = float(torch.linalg.vector_norm(texture_noise.feats.to(torch.float32)).item())
    flow_stats = {
        "tile_id": int(tile_id),
        "seed": int(seed),
        "fixed_shape": True,
        "shape_flow_used": False,
        "texture_tokens": int(coords.shape[0]),
        "texture_channels": int(texture_channels),
        "shared_texture_noise": True,
        "noise_created_once": True,
        "noise_range": base._tensor_range(texture_noise.feats),
        "noise_l2": noise_l2,
        "native_timestep_schedule": schedule,
        "sampler_params_identical": True,
        "condition_only_difference": True,
        "branches": branch_stats,
    }
    del texture_noise
    base._empty_cuda_cache()
    return outputs, flow_stats


@torch.no_grad()
def _decode_to_patch(
    *,
    pipeline: Any,
    fixed_shape_denorm: SparseTensor,
    texture_denorm: SparseTensor,
    tile_id: int,
    box: Sequence[int],
    global_camera: Mapping[str, float],
    transform: base.TileCameraTransform,
    query_chunk_size: int,
) -> Tuple[base.ReturnedTilePatch, MeshWithVoxel]:
    decoded = pipeline.decode_latent(
        fixed_shape_denorm,
        texture_denorm,
        int(base.OVOXEL_RESOLUTION),
    )
    base._sync_cuda()
    if len(decoded) != 1:
        raise RuntimeError(f"tile {tile_id:02d} decode returned {len(decoded)} meshes")
    mesh = base._validate_mesh(decoded[0], f"tile {tile_id:02d} texture decode")
    patch = base._local_mesh_to_global_patch(
        tile_id=int(tile_id),
        box=box,
        local_mesh=mesh,
        global_camera=global_camera,
        transform=transform,
        query_chunk_size=int(query_chunk_size),
    )
    return patch, mesh


def _save_input_comparison(
    *,
    canonical_path: Path,
    baseline_path: Path,
    group_paths: Mapping[str, Path],
    baseline_metrics: Optional[Mapping[str, Any]],
    group_metrics: Mapping[str, Optional[Mapping[str, Any]]],
    output_path: Path,
) -> None:
    entries = [
        (canonical_path, "canonical input", None),
        (baseline_path, "ordinary global baseline", baseline_metrics),
        (
            group_paths[LOCAL_TILE_CONDITION],
            LOCAL_TILE_CONDITION,
            group_metrics.get(LOCAL_TILE_CONDITION),
        ),
        (
            group_paths[GLOBAL_ROUTED_CONDITION],
            GLOBAL_ROUTED_CONDITION,
            group_metrics.get(GLOBAL_ROUTED_CONDITION),
        ),
    ]
    panel = 420
    header = 70
    canvas = Image.new("RGB", (panel * len(entries), panel + header), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (path, title, metrics) in enumerate(entries):
        if path.is_file():
            with Image.open(path) as image:
                image = ImageOps.contain(image.convert("RGB"), (panel - 8, panel - 8))
            canvas.paste(
                image,
                (
                    index * panel + (panel - image.width) // 2,
                    header + (panel - image.height) // 2,
                ),
            )
        draw.text((index * panel + 8, 8), title, fill=(255, 255, 255))
        if metrics is not None:
            draw.text(
                (index * panel + 8, 34),
                f"PSNR {metrics.get('psnr_db')} SSIM {metrics.get('ssim')} LPIPS {metrics.get('lpips')}",
                fill=(220, 220, 220),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _save_multiview_group_sheet(
    *,
    baseline_paths: Sequence[Path],
    local_paths: Sequence[Path],
    global_paths: Sequence[Path],
    output_path: Path,
) -> None:
    count = min(len(baseline_paths), len(local_paths), len(global_paths))
    panel = 320
    header = 42
    canvas = Image.new("RGB", (panel * 3, (panel + header) * count), "black")
    draw = ImageDraw.Draw(canvas)
    columns = (
        ("baseline", baseline_paths),
        (LOCAL_TILE_CONDITION, local_paths),
        (GLOBAL_ROUTED_CONDITION, global_paths),
    )
    for row in range(count):
        for column, (title, paths) in enumerate(columns):
            path = paths[row]
            if path.is_file():
                with Image.open(path) as image:
                    image = ImageOps.contain(image.convert("RGB"), (panel - 4, panel - 4))
                canvas.paste(
                    image,
                    (
                        column * panel + (panel - image.width) // 2,
                        row * (panel + header) + header,
                    ),
                )
            draw.text(
                (column * panel + 5, row * (panel + header) + 10),
                title if row == 0 else title,
                fill=(255, 255, 255),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _draw_route_sanity(
    *,
    canonical_image: Image.Image,
    boxes: Sequence[Sequence[int]],
    route_records: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> Dict[str, Any]:
    scale = float(base.GLOBAL_IMAGE_SIZE) / float(base.CANONICAL_IMAGE_SIZE)
    canvas = canonical_image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    palette = (
        (239, 83, 80),
        (255, 167, 38),
        (253, 216, 53),
        (102, 187, 106),
        (38, 198, 218),
        (66, 165, 245),
        (126, 87, 194),
        (236, 64, 122),
    )
    for tile_id, box in enumerate(boxes):
        x0, y0, x1, y1 = [int(round(float(v) * scale)) for v in box]
        color = palette[tile_id % len(palette)]
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=2)
        draw.text((x0 + 4, y0 + 4), f"{tile_id:02d}", fill=color)
    point_count = 0
    for record in route_records:
        tile_id = int(record["tile_id"])
        color = palette[tile_id % len(palette)]
        for row in record.get("route", {}).get("sanity_tokens", []):
            x, y = row["uv_canonical_1024"]
            px = int(round(float(x)))
            py = int(round(float(y)))
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
            point_count += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {
        "path": str(output_path),
        "resolution": [int(canvas.width), int(canvas.height)],
        "source_image": "canonical_1024",
        "tile_boxes_scaled_from_4096": True,
        "sampled_token_points": int(point_count),
        "point_semantics": "sampled fixed-shape support rows after local->global->canonical1024 routing",
    }


def _mean_multiview_metrics(multiview: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not multiview or not multiview.get("enabled"):
        return {"views": 0, "psnr_db": None, "ssim": None}
    rows = multiview.get("pair_metrics", [])
    if not rows:
        return {"views": 0, "psnr_db": None, "ssim": None}
    return {
        "views": int(len(rows)),
        "psnr_db": float(np.mean([row["baseline_vs_stitched_psnr_db"] for row in rows])),
        "ssim": float(np.mean([row["baseline_vs_stitched_ssim"] for row in rows])),
    }


def _validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if int(args.cuda_device) < 0 or int(args.cuda_device) >= torch.cuda.device_count():
        raise ValueError(
            f"--cuda-device={args.cuda_device} is unavailable; "
            f"visible CUDA device count is {torch.cuda.device_count()}"
        )
    if args.max_tiles is not None and int(args.max_tiles) < 1:
        raise ValueError("--max-tiles must be positive")
    if int(args.feature_query_chunk_size) < 1:
        raise ValueError("--feature-query-chunk-size must be positive")
    if int(args.sanity_token_count) < 1:
        raise ValueError("--sanity-token-count must be positive")
    if args.render_multiview and int(args.multiview_turntable_frames) != 24:
        raise ValueError(
            "30 fixed multiview evaluation requires --multiview-turntable-frames=24"
        )


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    torch.cuda.set_device(int(args.cuda_device))
    device = torch.device("cuda")
    print(
        f"[cuda] requested/current index={int(args.cuda_device)} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())}"
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.image).expanduser().resolve()
    with Image.open(source_path) as source:
        source_image = source.convert("RGB")
    source_image.save(output_dir / "input_original.png")

    pipeline = base.init_pipeline(
        args.model_path,
        device="cuda",
        low_vram=bool(args.low_vram),
    )
    canonical = pipeline.preprocess_canonical_images(source_image)
    image_4096: Image.Image = canonical["image_4096"]
    image_1024: Image.Image = canonical["image_1024"]
    image_512: Image.Image = canonical["image_512"]
    image_4096.save(output_dir / "canonical_4096.png")
    image_1024.save(output_dir / "canonical_1024.png")
    image_512.save(output_dir / "canonical_512.png")
    canonical["foreground_mask_4096"].save(output_dir / "canonical_foreground_mask_4096.png")
    base._atomic_json(output_dir / "canonical_metadata.json", canonical["metadata"])

    global_camera = base._estimate_camera(
        image_1024=image_1024,
        output_dir=output_dir,
        manual_fov=float(args.fov),
        mesh_scale=float(args.mesh_scale),
        extend_pixel=int(args.extend_pixel),
        moge_model_path=args.moge_model_path,
    )
    base._atomic_json(output_dir / "global_camera.json", global_camera)
    ss_params, shape_params, texture_params = base._sampler_overrides(args)

    print("[global-baseline] running ordinary Pixal3D 1024_cascade")
    base._seed_everything(int(args.seed))
    baseline_started = time.perf_counter()
    baseline_output, baseline_latents = pipeline.run(
        image_1024,
        camera_params=global_camera,
        seed=int(args.seed),
        sparse_structure_sampler_params=ss_params,
        shape_slat_sampler_params=shape_params,
        tex_slat_sampler_params=texture_params,
        preprocess_image=False,
        return_latent=True,
        pipeline_type="1024_cascade",
        max_num_tokens=int(args.max_num_tokens),
    )
    baseline_seconds = time.perf_counter() - baseline_started
    if len(baseline_output) != 1:
        raise RuntimeError(f"global baseline returned {len(baseline_output)} meshes")
    baseline_live = base._validate_mesh(baseline_output[0], "global ordinary 1024 baseline")
    baseline_shape_slat, baseline_texture_slat, decoded_resolution = baseline_latents
    if int(decoded_resolution) != int(base.OVOXEL_RESOLUTION):
        raise RuntimeError(f"baseline decoder resolution is {decoded_resolution}, expected 1024")
    baseline_dir = output_dir / "global_baseline_1024"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    envmap = (
        base.load_envmap(str(args.envmap), device="cuda")
        if (args.render or args.render_multiview)
        else None
    )
    baseline_render: Optional[Dict[str, Any]] = None
    if args.render:
        baseline_render = base._render(
            baseline_live,
            output_dir=baseline_dir / "aligned_eval",
            camera=global_camera,
            reference_image=output_dir / "canonical_1024.png",
            args=args,
            envmap=envmap,
        )
    baseline_mesh = baseline_live.to("cpu")
    baseline_summary = {
        "route": "ordinary pipeline.run with pipeline_type=1024_cascade",
        "generation_seconds": float(baseline_seconds),
        "decoder_resolution": int(decoded_resolution),
        "vertices": int(baseline_mesh.vertices.shape[0]),
        "faces": int(baseline_mesh.faces.shape[0]),
        "active_ovoxels": int(baseline_mesh.coords.shape[0]),
        "shape_slat_tokens": int(baseline_shape_slat.feats.shape[0]),
        "texture_slat_tokens": int(baseline_texture_slat.feats.shape[0]),
        "render": baseline_render,
    }
    base._atomic_json(baseline_dir / "summary.json", baseline_summary)
    del baseline_output, baseline_live, baseline_latents
    del baseline_shape_slat, baseline_texture_slat
    base._empty_cuda_cache()

    print("[global-texture-features] extracting complete canonical1024 once")
    feature_cache = _extract_global_texture_feature_cache(pipeline, image_1024)
    base._atomic_json(
        output_dir / "global_texture_feature_cache.json",
        {
            "source_image": str(output_dir / "canonical_1024.png"),
            "source_is_complete_canonical_1024": True,
            "feature_extraction_calls": int(feature_cache.feature_extraction_calls),
            "map_names": list(feature_cache.map_names),
            "global_tokens": _safe_tensor_stats(feature_cache.global_tokens),
            "pixel_feature_maps": {
                name: _safe_tensor_stats(feature_map, include_range=False)
                for name, feature_map in zip(feature_cache.map_names, feature_cache.pixel_feature_maps)
            },
        },
    )

    face_min, face_max, face_finite = base._project_face_bboxes(
        baseline_mesh.vertices,
        baseline_mesh.faces,
        mesh_scale=float(global_camera["mesh_scale"]),
        global_camera=global_camera,
        chunk_size=int(args.face_projection_chunk_size),
    )
    shape_encoder = pixal3d_models.from_pretrained(
        str(Path(args.shape_encoder).expanduser())
    ).eval()

    boxes = base._tile_layout()
    requested_ids = base._parse_tile_ids(args.tile_ids)
    if requested_ids is not None:
        invalid = sorted(tile_id for tile_id in requested_ids if tile_id not in range(len(boxes)))
        if invalid:
            raise ValueError(f"invalid tile ids {invalid}; valid ids are 0..{len(boxes) - 1}")

    patches_by_group: Dict[str, List[base.ReturnedTilePatch]] = {
        group: [] for group in GROUPS
    }
    tile_records: List[Dict[str, Any]] = []
    route_records: List[Dict[str, Any]] = []
    attempted = 0
    for tile_id, box in enumerate(boxes):
        if requested_ids is not None and tile_id not in requested_ids:
            continue
        if args.max_tiles is not None and attempted >= int(args.max_tiles):
            break
        attempted += 1
        tile_dir = output_dir / "tiles" / f"tile_{tile_id:02d}"
        tile_dir.mkdir(parents=True, exist_ok=True)
        lr_tile_image = base._make_lr_tile_image(image_1024, box)
        lr_tile_image.save(tile_dir / "tile_lr_condition_reference.png")
        transform = base._derive_tile_camera(
            tile_id=int(tile_id),
            box=box,
            global_camera=global_camera,
            extend_pixel=int(args.extend_pixel),
        )
        base._atomic_json(tile_dir / "tile_camera.json", asdict(transform))
        selected_face_ids = base._tile_face_ids_from_bbox(
            face_min, face_max, face_finite, box
        )
        selected_face_count = int(selected_face_ids.shape[0])
        print(f"[tile {tile_id:02d}] bbox_faces={selected_face_count:,} box={box}")
        if selected_face_count == 0:
            record = {
                "status": "skipped",
                "tile_id": int(tile_id),
                "box": list(box),
                "reason": "no triangle projection bbox intersects tile",
            }
            tile_records.append(record)
            base._write_tile_summary(tile_dir, record)
            continue
        started = time.perf_counter()
        try:
            geometry = base._prepare_tile_geometry(
                global_vertices=baseline_mesh.vertices,
                global_faces=baseline_mesh.faces,
                global_face_min=face_min,
                global_face_max=face_max,
                global_face_finite=face_finite,
                global_camera=global_camera,
                transform=transform,
            )
            if geometry.stats["global_local_global_q_max_abs_error"] > float(args.roundtrip_tolerance):
                raise RuntimeError(
                    "global/local camera round-trip exceeded tolerance: "
                    f"{geometry.stats['global_local_global_q_max_abs_error']}"
                )
            shape_reference, shape_stats = base._encode_local_shape(
                encoder=shape_encoder,
                local_coords=geometry.coords,
                local_dual_vertices=geometry.dual_vertices,
                local_intersected=geometry.intersected,
                device=device,
                low_vram=bool(args.low_vram),
            )
            fixed_shape_norm = base._normalize_slat(
                shape_reference, pipeline.shape_slat_normalization
            )
            fixed_shape_denorm = base._denormalize_slat(
                fixed_shape_norm, pipeline.shape_slat_normalization
            )
            local_condition = _make_local_tile_condition(
                pipeline=pipeline,
                lr_tile_image=lr_tile_image,
                shape_coords=fixed_shape_norm.coords,
                transform=transform,
            )
            global_condition, route_stats = _make_global_routed_condition(
                cache=feature_cache,
                shape_coords=fixed_shape_norm.coords,
                global_camera=global_camera,
                transform=transform,
                query_chunk_size=int(args.feature_query_chunk_size),
                sanity_token_count=int(args.sanity_token_count),
            )
            conditions = {
                LOCAL_TILE_CONDITION: local_condition,
                GLOBAL_ROUTED_CONDITION: global_condition,
            }
            base._atomic_json(
                tile_dir / "global1024_route.json",
                {
                    "tile_id": int(tile_id),
                    "box": list(box),
                    "route": route_stats,
                },
            )
            route_records.append(
                {"tile_id": int(tile_id), "box": list(box), "route": route_stats}
            )

            flow_outputs, flow_stats = _run_paired_texture_flows(
                pipeline=pipeline,
                fixed_shape_norm=fixed_shape_norm,
                conditions=conditions,
                texture_params=texture_params,
                seed=int(args.seed),
                tile_id=int(tile_id),
            )
            tile_patches: Dict[str, base.ReturnedTilePatch] = {}
            texture_decode_stats: Dict[str, Any] = {}
            for group in GROUPS:
                texture_norm = flow_outputs[group]
                texture_denorm = base._denormalize_slat(
                    texture_norm, pipeline.tex_slat_normalization
                )
                decode_started = time.perf_counter()
                patch, decoded_mesh = _decode_to_patch(
                    pipeline=pipeline,
                    fixed_shape_denorm=fixed_shape_denorm,
                    texture_denorm=texture_denorm,
                    tile_id=int(tile_id),
                    box=box,
                    global_camera=global_camera,
                    transform=transform,
                    query_chunk_size=int(args.material_query_chunk_size),
                )
                texture_decode_stats[group] = {
                    "decode_seconds": float(time.perf_counter() - decode_started),
                    "decoded_vertices": int(decoded_mesh.vertices.shape[0]),
                    "decoded_faces": int(decoded_mesh.faces.shape[0]),
                    "returned_patch": patch.stats,
                    "texture_slat": base._tensor_range(texture_denorm.feats),
                }
                tile_patches[group] = patch
                patches_by_group[group].append(patch)
                del texture_denorm, decoded_mesh, texture_norm

            tile_record = {
                "status": "success",
                "tile_id": int(tile_id),
                "box": list(box),
                "tile_seconds": float(time.perf_counter() - started),
                "projected_bbox_faces": int(selected_face_count),
                "geometry": geometry.stats,
                "shape_encoder": shape_stats,
                "fixed_shape": {
                    "support": "one shape encoder output retained unchanged for both branches",
                    "tokens": int(fixed_shape_norm.feats.shape[0]),
                    "channels": int(fixed_shape_norm.feats.shape[1]),
                    "normalized_range": base._tensor_range(fixed_shape_norm.feats),
                },
                "conditions": {
                    LOCAL_TILE_CONDITION: {
                        "source": "current LR tile image condition",
                        "image": str(tile_dir / "tile_lr_condition_reference.png"),
                        "support_preserved": True,
                    },
                    GLOBAL_ROUTED_CONDITION: route_stats,
                },
                "texture_flow": flow_stats,
                "texture_decode": texture_decode_stats,
                "same_fixed_shape": True,
                "same_noise": True,
                "same_sampler": True,
                "same_decoder": True,
                "same_local_to_global": True,
                "tile_only_condition_difference": True,
            }
            tile_records.append(tile_record)
            base._write_tile_summary(tile_dir, tile_record)
            print(
                f"[tile {tile_id:02d}] success "
                f"tokens={fixed_shape_norm.feats.shape[0]:,} "
                f"local_faces={tile_patches[LOCAL_TILE_CONDITION].faces.shape[0]:,} "
                f"global_faces={tile_patches[GLOBAL_ROUTED_CONDITION].faces.shape[0]:,} "
                f"seconds={tile_record['tile_seconds']:.2f}"
            )
            del (
                geometry,
                shape_reference,
                fixed_shape_norm,
                fixed_shape_denorm,
                local_condition,
                global_condition,
                conditions,
                flow_outputs,
                flow_stats,
                tile_patches,
            )
            base._empty_cuda_cache()
        except Exception as exc:
            traceback.print_exc()
            record = {
                "status": "failed",
                "tile_id": int(tile_id),
                "box": list(box),
                "projected_bbox_faces": int(selected_face_count),
                "tile_seconds": float(time.perf_counter() - started),
                "reason": f"{type(exc).__name__}: {exc}",
            }
            tile_records.append(record)
            base._write_tile_summary(tile_dir, record)
            print(f"[tile {tile_id:02d}] FAILED: {record['reason']}")
            base._empty_cuda_cache()

    del shape_encoder, feature_cache
    base._empty_cuda_cache()
    successful_rows = [row for row in tile_records if row["status"] == "success"]
    failed_rows = [row for row in tile_records if row["status"] == "failed"]
    skipped_rows = [row for row in tile_records if row["status"] == "skipped"]
    route_sanity = _draw_route_sanity(
        canonical_image=image_1024,
        boxes=boxes,
        route_records=route_records,
        output_path=output_dir / "global1024_route_sanity.png",
    )
    base._atomic_json(output_dir / "global1024_route_sanity.json", route_sanity)

    group_results: Dict[str, Any] = {}
    group_meshes: Dict[str, MeshWithVertexPbr] = {}
    group_render_paths: Dict[str, Path] = {}
    group_input_metrics: Dict[str, Optional[Mapping[str, Any]]] = {}
    for group in GROUPS:
        patches = patches_by_group[group]
        if not patches:
            group_results[group] = {
                "status": "no_successful_tiles",
                "successful_tiles": 0,
            }
            group_input_metrics[group] = None
            continue
        group_dir = output_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        # Keep the stitch policy byte-for-byte aligned with the existing
        # fixed-shape local texture experiment.  The current 4x4 layout is
        # disjoint, so that experiment uses direct concatenation; overlapping
        # layouts retain the existing nearest-owner/weld path.
        if len(boxes) == 16 and base.TILE_STRIDE == base.TILE_SIZE:
            stitched_mesh, stitch_stats = base._stitch_tile_patches(
                patches,
                layout=baseline_mesh.layout,
            )
            stitch_stats["layout_policy"] = (
                "4x4 disjoint tiles; direct concat without overlap owner/weld"
            )
        else:
            stitched_mesh, stitch_stats = base._stitch_tile_patches_nearest(
                patches,
                layout=baseline_mesh.layout,
                global_camera=global_camera,
                face_chunk_size=int(args.face_projection_chunk_size),
                weld_tolerance=float(args.stitch_tolerance),
            )
        group_meshes[group] = stitched_mesh
        stitched_patch = base.ReturnedTilePatch(
            tile_id=-1,
            box=(0, 0, int(base.CANONICAL_IMAGE_SIZE), int(base.CANONICAL_IMAGE_SIZE)),
            vertices=stitched_mesh.vertices,
            faces=stitched_mesh.faces,
            vertex_attrs=stitched_mesh.vertex_attrs,
            stats=stitch_stats,
        )
        glb_stats = (
            base._export_tiled_glb(
                [stitched_patch],
                group_dir / f"{group}.glb",
            )
            if args.export_glb
            else {"enabled": False}
        )
        overlap_stats = base._save_tile_overlap_visualization(
            image_4096=image_4096,
            boxes=boxes,
            successful_ids=[patch.tile_id for patch in patches],
            output_path=group_dir / "tile_overlap_coverage.png",
        )
        render_stats: Dict[str, Any] = {
            "enabled": False,
            "overlap_visualization": overlap_stats,
        }
        if args.render:
            aligned = base._render(
                stitched_mesh,
                output_dir=group_dir / "aligned_eval",
                camera=global_camera,
                reference_image=output_dir / "canonical_1024.png",
                args=args,
                envmap=envmap,
            )
            against_baseline = (
                base._render(
                    stitched_mesh,
                    output_dir=group_dir / "against_global_baseline",
                    camera=global_camera,
                    reference_image=Path(str(baseline_render["render_png"])),
                    args=args,
                    envmap=envmap,
                )
                if baseline_render is not None
                else None
            )
            render_stats.update(
                {
                    "aligned": aligned,
                    "against_global_baseline": against_baseline,
                    "input_metrics": base._metric_subset(aligned),
                    "baseline_metrics": base._metric_subset(against_baseline),
                }
            )
            group_render_paths[group] = Path(str(aligned["render_png"]))
            group_input_metrics[group] = base._metric_subset(aligned)
        else:
            group_input_metrics[group] = None
        multiview = (
            base._render_multiview_comparison(
                baseline_mesh,
                stitched_mesh,
                output_dir=group_dir / "multiview",
                camera=global_camera,
                args=args,
                envmap=envmap,
            )
            if args.render_multiview
            else {"enabled": False}
        )
        group_results[group] = {
            "status": "success",
            "successful_tiles": int(len(patches)),
            "stitch": stitch_stats,
            "glb": glb_stats,
            "render": render_stats,
            "multiview": multiview,
            "mean_multiview_vs_global_baseline": _mean_multiview_metrics(multiview),
            "condition": (
                "local LR tile image projected condition"
                if group == LOCAL_TILE_CONDITION
                else "one complete canonical1024 global token plus local->global routed pixel features"
            ),
        }
        base._atomic_json(group_dir / "summary.json", group_results[group])

    if args.render and baseline_render is not None and group_render_paths:
        _save_input_comparison(
            canonical_path=output_dir / "canonical_1024.png",
            baseline_path=Path(str(baseline_render["render_png"])),
            group_paths=group_render_paths,
            baseline_metrics=base._metric_subset(baseline_render),
            group_metrics=group_input_metrics,
            output_path=output_dir / "comparison_input_baseline_local_global.png",
        )
    if args.render_multiview and all(
        group in group_results and group_results[group].get("multiview", {}).get("enabled")
        for group in GROUPS
    ):
        _save_multiview_group_sheet(
            baseline_paths=[
                Path(v) for v in group_results[LOCAL_TILE_CONDITION]["multiview"]["baseline_frame_pngs"]
            ],
            local_paths=[
                Path(v) for v in group_results[LOCAL_TILE_CONDITION]["multiview"]["stitched_local_frame_pngs"]
            ],
            global_paths=[
                Path(v) for v in group_results[GLOBAL_ROUTED_CONDITION]["multiview"]["stitched_local_frame_pngs"]
            ],
            output_path=output_dir / "comparison_30view_baseline_local_global.png",
        )

    summary = {
        "format": "pixal3d_paired_fixed_shape_texture_condition_ablation_v1",
        "image": str(source_path),
        "cuda_device": int(args.cuda_device),
        "global_camera": global_camera,
        "protocol": {
            "fixed_shape": True,
            "shape_source": "global baseline mesh -> local C1024 dual-grid -> shape encoder",
            "shape_flow_used": False,
            "pbr_encoder_used": False,
            "texture_flow": "native full FlowEulerSampler from direct t=1 random texture SLat noise",
            "same_noise_seed_per_tile": int(args.seed),
            "same_noise_across_groups": True,
            "same_sampler_across_groups": True,
            "same_decoder_across_groups": True,
            "same_local_to_global_transform": True,
            "same_stitcher": (
                "existing fixed-shape local texture stitch policy: direct concat "
                "for disjoint 4x4, nearest-owner/weld otherwise"
            ),
            "same_renderer": "base._render and base._render_multiview_comparison",
            "global_source": "complete canonical_1024",
            "global_image_condition_calls": 1,
            "global_token_shared_across_tiles": True,
            "global_projected_map_query": "bilinear pixel-aligned DINO patch maps with local->global camera route",
            "global_direct_local_coord_query": False,
            "global_tile_crop": False,
            "global_texture_condition_4096": False,
            "forbidden_paths": {
                "G_tex": False,
                "prefix_suffix_flow": False,
                "velocity_intervention": False,
                "x0_intervention": False,
                "CCA": False,
                "mask": False,
                "fusion": False,
                "new_network": False,
            },
            "multiview_policy": "six fixed views plus 24 turntable frames, total 30",
        },
        "global_baseline_1024": baseline_summary,
        "tile_policy": {
            "canonical_image_size": int(base.CANONICAL_IMAGE_SIZE),
            "global_condition_image_size": int(base.GLOBAL_IMAGE_SIZE),
            "tile_size": int(base.TILE_SIZE),
            "tile_stride": int(base.TILE_STRIDE),
            "tile_count": int(len(boxes)),
            "local_condition": "each LR 256 crop from canonical1024 resized to 1024 for native tex_1024 condition",
            "global_condition": "complete canonical1024 image extraction once; each tile queries the same cached maps",
            "support_order": "exact fixed shape encoder output order",
        },
        "successful_tiles": int(len(successful_rows)),
        "failed_tiles": int(len(failed_rows)),
        "skipped_tiles": int(len(skipped_rows)),
        "route_sanity": route_sanity,
        "groups": group_results,
        "tiles": tile_records,
    }
    base._atomic_json(output_dir / "summary.json", summary)
    print(
        f"[done] success={len(successful_rows)} failed={len(failed_rows)} "
        f"skipped={len(skipped_rows)} summary={output_dir / 'summary.json'}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--feature-query-chunk-size",
        type=int,
        default=16_384,
        help="maximum routed support rows per grid_sample call",
    )
    parser.add_argument(
        "--sanity-token-count",
        type=int,
        default=32,
        help="sampled routed support rows stored in route JSON/visualization",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
