#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render global Pixal3D versus a global-anchored tile-detail FDG mesh.

The ordinary global route is unchanged.  The tile route uses the decoded
global O-Voxels and the released VAE encoders only to discover active local
C64 query coordinates:

    global O-Voxel / corresponding global mesh part
    -> exact global-to-local camera mapping
    -> shape/PBR encoders
    -> common active C64 coordinates (white x positions)
    -> normalize the encoded global-derived shape/PBR latents as clean anchors
    -> initialize fresh shape noise on those coordinates
    -> at every Euler step evaluate matched HR-tile and LR-tile conditions
    -> simulate global velocity with x_t -> encoded clean anchor
    -> add the HR-minus-LR condition velocity residual on the full support
    -> repeat the same anchored residual flow for texture
    -> local FDG decode
    -> exact local mesh-vertex to global absolute-coordinate mapping
    -> assign each projected triangle to the nearest successful tile center
    -> compact and concatenate the owned triangle meshes
    -> weld near-identical global vertices
    -> keep PBR sampled onto local dual vertices and interpolate it on faces.

This is deliberately a visual experiment.  It reports only full-frame image
metrics for:

1. the ordinary global Pixal3D render against the original image; and
2. the directly merged tile FDG mesh render against the original image.

Per-tile Chamfer, O-Voxel IoU, and per-tile render metrics are not computed.
Each successful tile reports shape/texture latent feature differences between
the projected-global encoder result and the anchored detail-flow result, plus
per-step anchor and HR-minus-LR residual diagnostics.
The merged tile mesh also receives a configurable multi-view render sheet.
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
from dataclasses import dataclass
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

import numpy as np
import torch
import utils3d
from PIL import Image, ImageDraw
from tqdm import tqdm

import pixal3d.models as pixal3d_models
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVertexPbr, MeshWithVoxel
from pixal3d.utils import render_utils
from render_pixal3d_raw_ovoxel import load_envmap


@dataclass
class TileFlowLatents:
    shape_norm: SparseTensor
    shape_denorm: SparseTensor
    texture_norm: SparseTensor
    texture_denorm: SparseTensor
    stats: Dict[str, Any]


@dataclass
class ReturnedTileOVoxels:
    tile_id: int
    coords: torch.Tensor
    attrs: torch.Tensor
    center_distance_normalized: torch.Tensor
    center_weight: torch.Tensor
    stats: Dict[str, Any]


@dataclass
class ReturnedTileMesh:
    tile_id: int
    tile_center_4096: Tuple[float, float]
    vertices: torch.Tensor
    faces: torch.Tensor
    vertex_attrs: torch.Tensor
    vertex_center_weights: torch.Tensor
    vertex_uv_4096: torch.Tensor
    layout: Dict[str, Any]
    stats: Dict[str, Any]


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _randn(
    rows: int,
    channels: int,
    *,
    device: torch.device,
    seed: int,
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
        dtype=torch.float32,
    )


def _denormalize(
    value: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
) -> SparseTensor:
    std = torch.as_tensor(
        normalization["std"], device=value.device, dtype=value.dtype
    )[None]
    mean = torch.as_tensor(
        normalization["mean"], device=value.device, dtype=value.dtype
    )[None]
    return value.replace(value.feats * std + mean)


def _normalize(
    value: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
) -> SparseTensor:
    std = torch.as_tensor(
        normalization["std"], device=value.device, dtype=value.dtype
    )[None]
    mean = torch.as_tensor(
        normalization["mean"], device=value.device, dtype=value.dtype
    )[None]
    if std.shape[1] != value.feats.shape[1]:
        raise ValueError(
            "normalization channel count does not match latent features: "
            f"stats={std.shape[1]} latent={value.feats.shape[1]}"
        )
    if bool((std == 0).any().item()):
        raise ValueError("latent normalization contains a zero standard deviation")
    return value.replace((value.feats - mean) / std)


def _extract_samples(result: Any, label: str) -> SparseTensor:
    samples = getattr(result, "samples", result)
    if not isinstance(samples, SparseTensor):
        raise TypeError(f"{label}: sampler did not return SparseTensor")
    return samples


def _feature_statistics(feats: torch.Tensor) -> Dict[str, Any]:
    values = feats.detach().to(dtype=torch.float32)
    if values.ndim != 2 or values.numel() == 0:
        raise ValueError("latent features must be a non-empty [N,C] tensor")
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "abs_mean": float(values.abs().mean().item()),
        "rms": float(values.square().mean().sqrt().item()),
        "mean_token_l2": float(
            torch.linalg.vector_norm(values, dim=1).mean().item()
        ),
        "channel_mean": [
            float(value)
            for value in values.mean(dim=0).to(device="cpu").tolist()
        ],
        "channel_std": [
            float(value)
            for value in values.std(dim=0, unbiased=False)
            .to(device="cpu")
            .tolist()
        ],
    }


def _latent_feature_difference(
    *,
    encoded: SparseTensor,
    flowed_denorm: SparseTensor,
    label: str,
) -> Dict[str, Any]:
    if not torch.equal(encoded.coords, flowed_denorm.coords):
        raise RuntimeError(f"{label}: encoded/flowed coordinates differ")
    if encoded.feats.shape != flowed_denorm.feats.shape:
        raise RuntimeError(
            f"{label}: feature shapes differ: encoded={tuple(encoded.feats.shape)} "
            f"flowed={tuple(flowed_denorm.feats.shape)}"
        )
    encoded_feats = encoded.feats.detach().to(dtype=torch.float32)
    flowed_feats = flowed_denorm.feats.detach().to(dtype=torch.float32)
    difference = flowed_feats - encoded_feats
    encoded_norm = torch.linalg.vector_norm(encoded_feats)
    relative_l2 = torch.linalg.vector_norm(difference) / encoded_norm.clamp_min(
        1e-12
    )
    encoded_token_norm = torch.linalg.vector_norm(encoded_feats, dim=1)
    flowed_token_norm = torch.linalg.vector_norm(flowed_feats, dim=1)
    cosine = (encoded_feats * flowed_feats).sum(dim=1) / (
        encoded_token_norm * flowed_token_norm
    ).clamp_min(1e-12)
    both_zero = (encoded_token_norm <= 1e-12) & (
        flowed_token_norm <= 1e-12
    )
    cosine = torch.where(both_zero, torch.ones_like(cosine), cosine)
    return {
        "label": label,
        "tokens": int(encoded_feats.shape[0]),
        "channels": int(encoded_feats.shape[1]),
        "coordinate_alignment": "exact common C64 coordinate order",
        "comparison_space": (
            "decoder-ready denormalized latent features; "
            "difference is anchored_detail_flow minus projected_global_encoder"
        ),
        "projected_global_encoder": _feature_statistics(encoded_feats),
        "anchored_detail_flow": _feature_statistics(flowed_feats),
        "flow_minus_encoder": {
            **_feature_statistics(difference),
            "mae": float(difference.abs().mean().item()),
            "rmse": float(difference.square().mean().sqrt().item()),
            "max_abs": float(difference.abs().max().item()),
            "relative_l2": float(relative_l2.item()),
            "token_cosine_similarity_mean": float(cosine.mean().item()),
            "token_cosine_similarity_std": float(
                cosine.std(unbiased=False).item()
            ),
        },
    }


def _aggregate_latent_feature_differences(
    tile_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    successful = [
        row
        for row in tile_records
        if row.get("status") == "success"
        and isinstance(row.get("latent_feature_difference"), Mapping)
    ]
    for latent_name in ("shape", "texture"):
        rows = [
            row["latent_feature_difference"][latent_name]
            for row in successful
            if latent_name in row["latent_feature_difference"]
        ]
        if not rows:
            continue
        feature_weights = [
            int(row["tokens"]) * int(row["channels"]) for row in rows
        ]
        token_weights = [int(row["tokens"]) for row in rows]
        feature_total = sum(feature_weights)
        token_total = sum(token_weights)
        diffs = [row["flow_minus_encoder"] for row in rows]
        output[latent_name] = {
            "tiles": len(rows),
            "tokens_sum_across_tiles": token_total,
            "feature_values_sum_across_tiles": feature_total,
            "mae_feature_weighted": float(
                sum(
                    float(row["mae"]) * weight
                    for row, weight in zip(diffs, feature_weights)
                )
                / feature_total
            ),
            "rmse_feature_weighted": float(
                math.sqrt(
                    sum(
                        float(row["rmse"]) ** 2 * weight
                        for row, weight in zip(diffs, feature_weights)
                    )
                    / feature_total
                )
            ),
            "max_abs_across_tiles": float(
                max(float(row["max_abs"]) for row in diffs)
            ),
            "token_cosine_similarity_weighted": float(
                sum(
                    float(row["token_cosine_similarity_mean"]) * weight
                    for row, weight in zip(diffs, token_weights)
                )
                / token_total
            ),
        }
    return output


def _make_lr_reference_tile(
    image_1024: Image.Image,
    box_4096: Sequence[int],
) -> Image.Image:
    if image_1024.size != (1024, 1024):
        raise ValueError(
            "the global-condition image must be canonical 1024x1024, got "
            f"{image_1024.size}"
        )
    if len(box_4096) != 4:
        raise ValueError("tile box must contain four coordinates")
    scaled = tuple(int(value) // 4 for value in box_4096)
    if any(int(value) % 4 != 0 for value in box_4096):
        raise ValueError("canonical 4096 tile coordinates must be divisible by four")
    crop = image_1024.convert("RGB").crop(scaled)
    if crop.size != (256, 256):
        raise ValueError(
            "the downsampled global-image tile must be 256x256, got "
            f"{crop.size}"
        )
    return crop.resize((1024, 1024), Image.Resampling.LANCZOS)


def _prediction_features(
    prediction: Any,
    *,
    coords: torch.Tensor,
    label: str,
) -> torch.Tensor:
    if not isinstance(prediction, SparseTensor):
        raise TypeError(f"{label}: expected SparseTensor prediction")
    if not torch.equal(prediction.coords, coords):
        raise RuntimeError(f"{label}: model prediction changed sparse coordinates")
    return prediction.feats.to(dtype=torch.float32)


@torch.no_grad()
def _sample_global_anchor_detail_flow(
    *,
    pipeline: Any,
    sampler: Any,
    model: torch.nn.Module,
    noise: SparseTensor,
    clean_anchor: SparseTensor,
    hr_condition: Mapping[str, Any],
    lr_condition: Mapping[str, Any],
    params: Mapping[str, Any],
    label: str,
    concat_cond: Optional[SparseTensor] = None,
) -> Tuple[SparseTensor, float, Dict[str, Any]]:
    """Euler sample v*=v_anchor+(v_HR-v_LR) on the full fixed C64 support."""
    if not torch.equal(noise.coords, clean_anchor.coords):
        raise RuntimeError(f"{label}: noise/clean-anchor coordinates differ")
    if noise.feats.shape != clean_anchor.feats.shape:
        raise RuntimeError(f"{label}: noise/clean-anchor feature shapes differ")
    if concat_cond is not None and not torch.equal(
        noise.coords, concat_cond.coords
    ):
        raise RuntimeError(f"{label}: noise/concat coordinates differ")
    if pipeline.low_vram:
        model.to(pipeline.device)

    call_params = dict(params)
    steps = int(call_params.pop("steps"))
    rescale_t = float(call_params.pop("rescale_t", 1.0))
    call_params.pop("verbose", None)
    call_params.pop("tqdm_desc", None)
    inference_parameters = inspect.signature(
        sampler._inference_model
    ).parameters
    if "guidance_interval" in inference_parameters:
        call_params.setdefault("guidance_interval", (0.0, 1.0))
    if concat_cond is not None:
        call_params["concat_cond"] = concat_cond
    t_seq = sampler.timestep_schedule(steps, rescale_t)
    t_pairs = [(t_seq[index], t_seq[index + 1]) for index in range(steps)]

    sample = noise
    step_records: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for step_index, (t, t_prev) in enumerate(
        tqdm(t_pairs, desc=label, disable=False)
    ):
        _, _, velocity_hr = sampler._get_model_prediction(
            model,
            sample,
            float(t),
            **dict(hr_condition),
            **call_params,
        )
        _, _, velocity_lr = sampler._get_model_prediction(
            model,
            sample,
            float(t),
            **dict(lr_condition),
            **call_params,
        )
        velocity_hr_feats = _prediction_features(
            velocity_hr,
            coords=noise.coords,
            label=f"{label} HR step {step_index}",
        )
        velocity_lr_feats = _prediction_features(
            velocity_lr,
            coords=noise.coords,
            label=f"{label} LR step {step_index}",
        )
        anchor_velocity = sampler._xstart_to_pred(
            sample,
            float(t),
            clean_anchor,
        )
        anchor_velocity_feats = _prediction_features(
            anchor_velocity,
            coords=noise.coords,
            label=f"{label} anchor step {step_index}",
        )
        condition_delta = velocity_hr_feats - velocity_lr_feats
        fused_velocity = sample.replace(
            (
                anchor_velocity_feats + condition_delta
            ).to(dtype=sample.feats.dtype)
        )
        anchor_clean_prediction = sampler._pred_to_xstart(
            sample,
            float(t),
            anchor_velocity,
        )
        anchor_clean_feats = _prediction_features(
            anchor_clean_prediction,
            coords=noise.coords,
            label=f"{label} anchor clean prediction step {step_index}",
        )
        anchor_error_max = float(
            (
                anchor_clean_feats - clean_anchor.feats.to(torch.float32)
            ).abs().max().item()
        )
        step_records.append(
            {
                "step": int(step_index),
                "t": float(t),
                "t_prev": float(t_prev),
                "sigma_t": float(
                    sampler.sigma_min
                    + (1.0 - sampler.sigma_min) * float(t)
                ),
                "anchor_velocity_rms": float(
                    anchor_velocity_feats.square().mean().sqrt().item()
                ),
                "hr_velocity_rms": float(
                    velocity_hr_feats.square().mean().sqrt().item()
                ),
                "lr_velocity_rms": float(
                    velocity_lr_feats.square().mean().sqrt().item()
                ),
                "hr_minus_lr_velocity_rms": float(
                    condition_delta.square().mean().sqrt().item()
                ),
                "anchor_clean_endpoint_max_abs_error": anchor_error_max,
            }
        )
        sample = sample - float(t - t_prev) * fused_velocity

    _sync_cuda()
    elapsed = time.perf_counter() - started
    if not torch.equal(sample.coords, noise.coords):
        raise RuntimeError(f"{label}: sampler changed query coordinates")
    if pipeline.low_vram:
        model.cpu()
        _empty_cuda_cache()
    print(
        f"[flow] {label}: tokens={noise.feats.shape[0]:,} "
        f"noise_channels={noise.feats.shape[1]} seconds={elapsed:.3f}"
    )
    return sample, elapsed, {
        "formula": "v*=v_anchor+(v_HR-v_LR)",
        "anchor_formula": (
            "v_anchor=((1-sigma_min)*x_t-z_global_local_encoded)/sigma_t"
        ),
        "condition_pair": (
            "same x_t, C64 support/order, local camera, flow checkpoint, CFG; "
            "only the complete HR-tile versus LR-tile DINO conditions differ"
        ),
        "token_scope": (
            "full common C64 support; no front/back mask, z-buffer, spatial "
            "gate, or token locking; Pixal3D model supplies front/back behavior"
        ),
        "steps": step_records,
        "elapsed_seconds": float(elapsed),
    }


def _run_tile_query_flows(
    *,
    pipeline: Any,
    hr_tile_image: Image.Image,
    lr_tile_image: Image.Image,
    query_coords: torch.Tensor,
    encoded_shape: SparseTensor,
    encoded_pbr: SparseTensor,
    transform: core.TileCameraTransform,
    shape_params: Mapping[str, Any],
    texture_params: Mapping[str, Any],
    seed: int,
    tile_id: int,
) -> TileFlowLatents:
    coords = query_coords.to(
        device=pipeline.device, dtype=torch.int32
    ).contiguous()
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("query coordinates must have shape [N,4]")
    if bool((coords[:, 0] != 0).any().item()):
        raise ValueError("query coordinates must use batch zero")
    if bool(((coords[:, 1:] < 0) | (coords[:, 1:] >= 64)).any().item()):
        raise ValueError("query coordinates lie outside the local C64 grid")
    if int(torch.unique(coords, dim=0).shape[0]) != int(coords.shape[0]):
        raise ValueError("query coordinates must be unique")
    for label, encoded in (
        ("shape", encoded_shape),
        ("texture/PBR", encoded_pbr),
    ):
        if not torch.equal(encoded.coords, coords):
            raise RuntimeError(
                f"{label} encoded anchor is not aligned to query coordinates"
            )

    hr_tile_rgb = hr_tile_image.convert("RGB")
    lr_tile_rgb = lr_tile_image.convert("RGB")
    tile_camera = {
        "camera_angle_x": float(transform.camera_angle_x),
        "distance": float(transform.distance),
        "mesh_scale": float(transform.mesh_scale),
    }
    shape_hr_condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [hr_tile_rgb],
        coords,
        camera_angle_x=tile_camera["camera_angle_x"],
        distance=tile_camera["distance"],
        mesh_scale=tile_camera["mesh_scale"],
        grid_resolution_override=64,
    )
    shape_lr_condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [lr_tile_rgb],
        coords,
        camera_angle_x=tile_camera["camera_angle_x"],
        distance=tile_camera["distance"],
        mesh_scale=tile_camera["mesh_scale"],
        grid_resolution_override=64,
    )
    shape_model = pipeline.models["shape_slat_flow_model_1024"]
    shape_anchor_norm = _normalize(
        encoded_shape, pipeline.shape_slat_normalization
    )
    shape_noise_seed = int(seed) + 201
    shape_noise_feats = _randn(
        int(coords.shape[0]),
        int(shape_model.in_channels),
        device=pipeline.device,
        seed=shape_noise_seed,
    )
    shape_noise = SparseTensor(shape_noise_feats, coords)
    shape_norm, shape_seconds, shape_anchor_flow = (
        _sample_global_anchor_detail_flow(
            pipeline=pipeline,
            sampler=pipeline.shape_slat_sampler,
            model=shape_model,
            noise=shape_noise,
            clean_anchor=shape_anchor_norm,
            hr_condition=shape_hr_condition,
            lr_condition=shape_lr_condition,
            params=shape_params,
            label=f"Tile {tile_id:02d} global-anchor detail shape 1024",
        )
    )
    shape_denorm = _denormalize(
        shape_norm, pipeline.shape_slat_normalization
    )
    del (
        shape_hr_condition,
        shape_lr_condition,
        shape_anchor_norm,
        shape_noise,
    )
    _empty_cuda_cache()

    texture_hr_condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [hr_tile_rgb],
        coords,
        camera_angle_x=tile_camera["camera_angle_x"],
        distance=tile_camera["distance"],
        mesh_scale=tile_camera["mesh_scale"],
        grid_resolution_override=64,
    )
    texture_lr_condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [lr_tile_rgb],
        coords,
        camera_angle_x=tile_camera["camera_angle_x"],
        distance=tile_camera["distance"],
        mesh_scale=tile_camera["mesh_scale"],
        grid_resolution_override=64,
    )
    texture_model = pipeline.models["tex_slat_flow_model_1024"]
    texture_channels = int(texture_model.in_channels) - int(
        shape_norm.feats.shape[1]
    )
    if texture_channels <= 0:
        raise RuntimeError(
            f"invalid texture noise channel count {texture_channels}"
        )
    texture_anchor_norm = _normalize(
        encoded_pbr, pipeline.tex_slat_normalization
    )
    if int(texture_anchor_norm.feats.shape[1]) != texture_channels:
        raise RuntimeError(
            "encoded texture anchor channels do not match texture flow noise: "
            f"anchor={texture_anchor_norm.feats.shape[1]} "
            f"noise={texture_channels}"
        )
    texture_noise_seed = int(seed) + 301
    texture_noise_feats = _randn(
        int(coords.shape[0]),
        texture_channels,
        device=pipeline.device,
        seed=texture_noise_seed,
    )
    texture_noise = SparseTensor(texture_noise_feats, coords)
    texture_norm, texture_seconds, texture_anchor_flow = (
        _sample_global_anchor_detail_flow(
            pipeline=pipeline,
            sampler=pipeline.tex_slat_sampler,
            model=texture_model,
            noise=texture_noise,
            clean_anchor=texture_anchor_norm,
            hr_condition=texture_hr_condition,
            lr_condition=texture_lr_condition,
            params=texture_params,
            label=f"Tile {tile_id:02d} global-anchor detail texture 1024",
            concat_cond=shape_norm,
        )
    )
    texture_denorm = _denormalize(
        texture_norm, pipeline.tex_slat_normalization
    )
    stats = {
        "query_c64_tokens": int(coords.shape[0]),
        "shape_noise_seed": shape_noise_seed,
        "shape_noise_channels": int(shape_noise_feats.shape[1]),
        "shape_noise_mean": float(shape_noise_feats.mean().item()),
        "shape_noise_std": float(shape_noise_feats.std().item()),
        "shape_flow_seconds": float(shape_seconds),
        "texture_noise_seed": texture_noise_seed,
        "texture_noise_channels": int(texture_noise_feats.shape[1]),
        "texture_noise_mean": float(texture_noise_feats.mean().item()),
        "texture_noise_std": float(texture_noise_feats.std().item()),
        "texture_flow_seconds": float(texture_seconds),
        "shape_anchor_detail_flow": shape_anchor_flow,
        "texture_anchor_detail_flow": texture_anchor_flow,
        "encoder_features_reused": True,
        "encoder_coordinates_reused_as_queries": True,
        "encoded_features_role": (
            "normalized clean endpoint for simulated global-anchor velocity "
            "at every Euler step"
        ),
        "detail_condition_role": (
            "HR-tile minus LR-tile velocity on every common C64 token; "
            "front/back distinction is left to the trained Pixal3D model"
        ),
    }
    del (
        texture_hr_condition,
        texture_lr_condition,
        texture_anchor_norm,
        texture_noise,
    )
    _empty_cuda_cache()
    return TileFlowLatents(
        shape_norm=shape_norm,
        shape_denorm=shape_denorm,
        texture_norm=texture_norm,
        texture_denorm=texture_denorm,
        stats=stats,
    )


def _return_local_mesh_to_global(
    *,
    tile_id: int,
    local_mesh: MeshWithVoxel,
    global_camera: Mapping[str, float],
    transform: core.TileCameraTransform,
) -> ReturnedTileMesh:
    """Return one decoder-native local FDG mesh to global object space."""
    local_vertices = local_mesh.vertices.to(dtype=torch.float32)
    if local_vertices.ndim != 2 or local_vertices.shape[1] != 3:
        raise ValueError("local decoder vertices must have shape [N,3]")
    local_q = local_vertices * (2.0 * float(transform.mesh_scale))
    global_q, uv_4096 = core._local_q_to_global_q(
        local_q,
        global_camera=global_camera,
        transform=transform,
    )
    global_vertices = global_q / (
        2.0 * float(global_camera["mesh_scale"])
    )
    finite = (
        torch.isfinite(global_vertices).all(dim=1)
        & torch.isfinite(uv_4096).all(dim=1)
    )
    if not bool(finite.all().item()):
        raise RuntimeError(
            "local FDG mesh produced non-finite global vertices/projections"
        )
    faces = local_mesh.faces.to(dtype=torch.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("local decoder faces must have shape [M,3]")
    if faces.shape[0] == 0:
        raise RuntimeError("local FDG decoder produced no triangle faces")
    with torch.no_grad():
        local_vertex_attrs = local_mesh.query_vertex_attrs().to(
            dtype=torch.float32
        )
    if (
        local_vertex_attrs.ndim != 2
        or local_vertex_attrs.shape[0] != local_vertices.shape[0]
    ):
        raise RuntimeError(
            "local O-Voxel PBR sampling is not aligned with FDG vertices"
        )
    if not bool(torch.isfinite(local_vertex_attrs).all().item()):
        raise RuntimeError("local vertex PBR contains non-finite values")
    own_center = uv_4096.new_tensor(
        [
            float(transform.tile_center_full_x),
            float(transform.tile_center_full_y),
        ]
    )
    tile_width = float(transform.box[2] - transform.box[0])
    tile_height = float(transform.box[3] - transform.box[1])
    half_diagonal = max(
        math.hypot(tile_width * 0.5, tile_height * 0.5),
        1e-12,
    )
    vertex_center_distance = torch.linalg.vector_norm(
        uv_4096 - own_center[None],
        dim=1,
    )
    vertex_center_weights = 1.0 / (
        1.0 + (vertex_center_distance / half_diagonal).square()
    )
    stats = {
        "tile_id": int(tile_id),
        "local_decoder_vertices": int(local_vertices.shape[0]),
        "local_decoder_faces": int(faces.shape[0]),
        "local_vertex_pbr_rows": int(local_vertex_attrs.shape[0]),
        "local_vertex_pbr_channels": int(local_vertex_attrs.shape[1]),
        "local_material_rule": (
            "query local decoded sparse O-Voxel PBR at local FDG dual "
            "vertices before any local-to-global geometry transform"
        ),
        "local_vertex_pbr_mean": [
            float(value)
            for value in local_vertex_attrs.mean(dim=0).tolist()
        ],
        "local_vertex_pbr_min": [
            float(value)
            for value in local_vertex_attrs.amin(dim=0).tolist()
        ],
        "local_vertex_pbr_max": [
            float(value)
            for value in local_vertex_attrs.amax(dim=0).tolist()
        ],
        "local_to_global_formula": (
            "local_object * (2*tile_mesh_scale) -> inverse tile camera "
            "mapping -> global_q / (2*global_mesh_scale)"
        ),
        "global_object_min": [
            float(value) for value in global_vertices.amin(dim=0).tolist()
        ],
        "global_object_max": [
            float(value) for value in global_vertices.amax(dim=0).tolist()
        ],
        "projected_uv_4096_min": [
            float(value) for value in uv_4096.amin(dim=0).tolist()
        ],
        "projected_uv_4096_max": [
            float(value) for value in uv_4096.amax(dim=0).tolist()
        ],
    }
    print(
        f"[tile {int(tile_id):02d} mesh->global] "
        f"vertices={local_vertices.shape[0]:,} faces={faces.shape[0]:,}"
    )
    return ReturnedTileMesh(
        tile_id=int(tile_id),
        tile_center_4096=(
            float(transform.tile_center_full_x),
            float(transform.tile_center_full_y),
        ),
        vertices=global_vertices.detach().to(
            device="cpu", dtype=torch.float32
        ),
        faces=faces.detach().to(device="cpu", dtype=torch.int64),
        vertex_attrs=local_vertex_attrs.detach().to(
            device="cpu", dtype=torch.float32
        ),
        vertex_center_weights=vertex_center_weights.detach().to(
            device="cpu", dtype=torch.float32
        ),
        vertex_uv_4096=uv_4096.detach().to(
            device="cpu", dtype=torch.float32
        ),
        layout=dict(local_mesh.layout),
        stats=stats,
    )


def _filter_tile_faces_by_nearest_successful_center(
    tile: ReturnedTileMesh,
    *,
    owner_tile_ids: torch.Tensor,
    owner_centers: torch.Tensor,
    chunk_size: int,
    canonical_size: int = 4096,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Dict[str, Any],
]:
    """Keep triangles whose projected centroids are closest to this tile."""
    if int(chunk_size) <= 0:
        raise ValueError("face ownership chunk size must be positive")
    if owner_tile_ids.ndim != 1 or owner_centers.shape != (
        owner_tile_ids.shape[0],
        2,
    ):
        raise ValueError("owner tile ids and centers are not aligned")
    matching_owner = torch.where(
        owner_tile_ids.to(torch.int64) == int(tile.tile_id)
    )[0]
    if matching_owner.numel() != 1:
        raise RuntimeError(
            f"tile {tile.tile_id} does not have exactly one owner center"
        )
    expected_owner = int(matching_owner.item())
    faces = tile.faces.to(device="cpu", dtype=torch.int64)
    uv = tile.vertex_uv_4096.to(device="cpu", dtype=torch.float32)
    centers = owner_centers.to(device="cpu", dtype=torch.float32)
    kept_chunks: List[torch.Tensor] = []
    outside_canonical = 0
    nonfinite = 0
    lost_to_nearer_tile = 0
    for start in range(0, int(faces.shape[0]), int(chunk_size)):
        face_chunk = faces[start : start + int(chunk_size)]
        tri_uv = uv.index_select(0, face_chunk.reshape(-1)).reshape(-1, 3, 2)
        finite = torch.isfinite(tri_uv).all(dim=(1, 2))
        centroid = tri_uv.mean(dim=1)
        inside = (
            (centroid[:, 0] >= 0.0)
            & (centroid[:, 0] < float(canonical_size))
            & (centroid[:, 1] >= 0.0)
            & (centroid[:, 1] < float(canonical_size))
        )
        distances_sq = (
            centroid[:, None, :] - centers[None, :, :]
        ).square().sum(dim=2)
        nearest = torch.argmin(distances_sq, dim=1)
        keep = finite & inside & (nearest == expected_owner)
        nonfinite += int((~finite).sum().item())
        outside_canonical += int((finite & ~inside).sum().item())
        lost_to_nearer_tile += int(
            (finite & inside & (nearest != expected_owner)).sum().item()
        )
        if bool(keep.any().item()):
            kept_chunks.append(face_chunk[keep])
    if not kept_chunks:
        return (
            torch.empty((0, 3), dtype=torch.int64),
            torch.empty((0, 3), dtype=torch.float32),
            torch.empty(
                (0, int(tile.vertex_attrs.shape[1])),
                dtype=torch.float32,
            ),
            torch.empty((0,), dtype=torch.float32),
            {
                "tile_id": int(tile.tile_id),
                "input_vertices": int(tile.vertices.shape[0]),
                "input_faces": int(faces.shape[0]),
                "owned_faces": 0,
                "compact_owned_vertices": 0,
                "faces_nonfinite_projection": int(nonfinite),
                "faces_outside_canonical_image": int(outside_canonical),
                "faces_owned_by_nearer_successful_tile": int(
                    lost_to_nearer_tile
                ),
                "ownership_rule": (
                    "minimum squared 4096-image distance from projected "
                    "triangle centroid to successful tile center"
                ),
            },
        )
    owned_source_faces = torch.cat(kept_chunks, dim=0)
    used_vertices, inverse = torch.unique(
        owned_source_faces.reshape(-1),
        sorted=True,
        return_inverse=True,
    )
    compact_faces = inverse.reshape(-1, 3).to(torch.int64)
    compact_vertices = tile.vertices.index_select(0, used_vertices)
    compact_attrs = tile.vertex_attrs.index_select(0, used_vertices)
    compact_weights = tile.vertex_center_weights.index_select(
        0, used_vertices
    )
    stats = {
        "tile_id": int(tile.tile_id),
        "input_vertices": int(tile.vertices.shape[0]),
        "input_faces": int(faces.shape[0]),
        "owned_faces": int(compact_faces.shape[0]),
        "compact_owned_vertices": int(compact_vertices.shape[0]),
        "unused_vertices_dropped": int(
            tile.vertices.shape[0] - compact_vertices.shape[0]
        ),
        "faces_nonfinite_projection": int(nonfinite),
        "faces_outside_canonical_image": int(outside_canonical),
        "faces_owned_by_nearer_successful_tile": int(lost_to_nearer_tile),
        "ownership_rule": (
            "minimum squared 4096-image distance from projected triangle "
            "centroid to successful tile center; lower tile id breaks ties"
        ),
    }
    return (
        compact_faces,
        compact_vertices,
        compact_attrs,
        compact_weights,
        stats,
    )


def _weld_vertices_and_remap_faces(
    *,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    vertex_attrs: torch.Tensor,
    vertex_weights: torch.Tensor,
    tolerance: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Quantized global near-point welding with GPU key reduction."""
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [N,3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [M,3]")
    if (
        vertex_attrs.ndim != 2
        or vertex_attrs.shape[0] != vertices.shape[0]
    ):
        raise ValueError("vertex_attrs must be [N,C] aligned with vertices")
    if vertex_weights.shape != (vertices.shape[0],):
        raise ValueError("vertex_weights must be [N] aligned with vertices")
    if float(tolerance) <= 0.0 or vertices.shape[0] == 0:
        return (
            vertices.to(dtype=torch.float32).contiguous(),
            faces.to(dtype=torch.int32).contiguous(),
            vertex_attrs.to(dtype=torch.float32).contiguous(),
            {
                "enabled": False,
                "tolerance": float(tolerance),
                "input_vertices": int(vertices.shape[0]),
                "output_vertices": int(vertices.shape[0]),
                "input_faces": int(faces.shape[0]),
                "output_faces": int(faces.shape[0]),
                "vertices_welded": 0,
                "degenerate_faces_removed": 0,
            },
        )

    live_vertices = vertices.to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    quantized = torch.round(
        live_vertices / float(tolerance)
    ).to(torch.int64)
    q_min = quantized.amin(dim=0)
    shifted = quantized - q_min[None]
    spans = shifted.amax(dim=0) + 1
    span_values = [int(value) for value in spans.detach().cpu().tolist()]
    key_volume = (
        span_values[0] * span_values[1] * span_values[2]
    )
    if key_volume >= 2**63:
        raise RuntimeError(
            "vertex-weld quantization exceeds signed int64 key capacity; "
            "increase --vertex-weld-tolerance"
        )
    keys = (
        (shifted[:, 0] * spans[1] + shifted[:, 1]) * spans[2]
        + shifted[:, 2]
    )
    _, inverse = torch.unique(
        keys, sorted=True, return_inverse=True
    )
    cluster_count = int(inverse.max().item()) + 1
    welded = torch.zeros(
        (cluster_count, 3),
        device=device,
        dtype=torch.float32,
    )
    welded.index_add_(0, inverse, live_vertices)
    counts = torch.zeros(
        (cluster_count,), device=device, dtype=torch.float32
    )
    counts.index_add_(
        0,
        inverse,
        torch.ones_like(inverse, dtype=torch.float32),
    )
    welded = welded / counts[:, None]
    live_attrs = vertex_attrs.to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    live_weights = vertex_weights.to(
        device=device, dtype=torch.float32, non_blocking=True
    ).clamp_min(1e-12)
    welded_attrs = torch.zeros(
        (cluster_count, int(live_attrs.shape[1])),
        device=device,
        dtype=torch.float32,
    )
    welded_attr_weights = torch.zeros(
        (cluster_count,), device=device, dtype=torch.float32
    )
    welded_attrs.index_add_(
        0,
        inverse,
        live_attrs * live_weights[:, None],
    )
    welded_attr_weights.index_add_(0, inverse, live_weights)
    welded_attrs = welded_attrs / welded_attr_weights[:, None].clamp_min(
        1e-12
    )
    live_faces = faces.to(
        device=device, dtype=torch.int64, non_blocking=True
    )
    remapped = inverse.index_select(0, live_faces.reshape(-1)).reshape(-1, 3)
    nondegenerate = (
        (remapped[:, 0] != remapped[:, 1])
        & (remapped[:, 1] != remapped[:, 2])
        & (remapped[:, 0] != remapped[:, 2])
    )
    remapped = remapped[nondegenerate]
    if welded.shape[0] >= 2**31:
        raise RuntimeError("merged mesh exceeds int32 vertex-index capacity")
    stats = {
        "enabled": True,
        "tolerance": float(tolerance),
        "quantized_key_spans": span_values,
        "input_vertices": int(vertices.shape[0]),
        "output_vertices": int(welded.shape[0]),
        "vertices_welded": int(vertices.shape[0] - welded.shape[0]),
        "input_faces": int(faces.shape[0]),
        "output_faces": int(remapped.shape[0]),
        "degenerate_faces_removed": int(
            faces.shape[0] - remapped.shape[0]
        ),
        "welded_material_rule": (
            "source vertex PBR weighted by inverse normalized distance to "
            "its own tile center"
        ),
    }
    return (
        welded.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        remapped.detach().to(device="cpu", dtype=torch.int32).contiguous(),
        welded_attrs.detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous(),
        stats,
    )


def _merge_tile_meshes_by_nearest_center(
    *,
    tiles: Sequence[ReturnedTileMesh],
    face_projection_chunk_size: int,
    vertex_weld_tolerance: float,
    device: torch.device,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Dict[str, Any],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    if not tiles:
        raise RuntimeError("no successful tile meshes to merge")
    ordered = sorted(tiles, key=lambda item: int(item.tile_id))
    reference_layout = dict(ordered[0].layout)
    if any(dict(tile.layout) != reference_layout for tile in ordered[1:]):
        raise RuntimeError("successful tile meshes use inconsistent PBR layouts")
    owner_tile_ids = torch.tensor(
        [int(tile.tile_id) for tile in ordered], dtype=torch.int64
    )
    owner_centers = torch.tensor(
        [tile.tile_center_4096 for tile in ordered],
        dtype=torch.float32,
    )
    vertex_chunks: List[torch.Tensor] = []
    face_chunks: List[torch.Tensor] = []
    attr_chunks: List[torch.Tensor] = []
    weight_chunks: List[torch.Tensor] = []
    ownership_rows: List[Dict[str, Any]] = []
    vertex_offset = 0
    for tile in ordered:
        (
            compact_faces,
            compact_vertices,
            compact_attrs,
            compact_weights,
            stats,
        ) = (
            _filter_tile_faces_by_nearest_successful_center(
                tile,
                owner_tile_ids=owner_tile_ids,
                owner_centers=owner_centers,
                chunk_size=int(face_projection_chunk_size),
            )
        )
        ownership_rows.append(stats)
        if compact_faces.shape[0] == 0:
            print(
                f"[tile {tile.tile_id:02d} ownership] no owned faces"
            )
            continue
        face_chunks.append(compact_faces + int(vertex_offset))
        vertex_chunks.append(compact_vertices)
        attr_chunks.append(compact_attrs)
        weight_chunks.append(compact_weights)
        vertex_offset += int(compact_vertices.shape[0])
        print(
            f"[tile {tile.tile_id:02d} ownership] "
            f"faces={compact_faces.shape[0]:,} "
            f"vertices={compact_vertices.shape[0]:,}"
        )
    if not vertex_chunks or not face_chunks:
        raise RuntimeError("nearest-center ownership produced no mesh")
    concatenated_vertices = torch.cat(vertex_chunks, dim=0).to(torch.float32)
    concatenated_faces = torch.cat(face_chunks, dim=0).to(torch.int64)
    concatenated_attrs = torch.cat(attr_chunks, dim=0).to(torch.float32)
    concatenated_weights = torch.cat(weight_chunks, dim=0).to(torch.float32)
    (
        welded_vertices,
        welded_faces,
        welded_attrs,
        weld_stats,
    ) = (
        _weld_vertices_and_remap_faces(
            vertices=concatenated_vertices,
            faces=concatenated_faces,
            vertex_attrs=concatenated_attrs,
            vertex_weights=concatenated_weights,
            tolerance=float(vertex_weld_tolerance),
            device=device,
        )
    )
    stats = {
        "successful_tile_meshes": len(ordered),
        "tile_ids": [int(tile.tile_id) for tile in ordered],
        "successful_tile_centers_4096": {
            str(tile.tile_id): [
                float(tile.tile_center_4096[0]),
                float(tile.tile_center_4096[1]),
            ]
            for tile in ordered
        },
        "concatenated_owned_vertices_before_weld": int(
            concatenated_vertices.shape[0]
        ),
        "concatenated_owned_faces_before_weld": int(
            concatenated_faces.shape[0]
        ),
        "merged_vertices": int(welded_vertices.shape[0]),
        "merged_faces": int(welded_faces.shape[0]),
        "merged_vertex_pbr_rows": int(welded_attrs.shape[0]),
        "merged_vertex_pbr_channels": int(welded_attrs.shape[1]),
        "vertex_welding": weld_stats,
        "geometry_route": (
            "local decoder FDG mesh -> exact inverse tile camera mapping -> "
            "nearest-successful-tile-center triangle ownership -> compact -> "
            "quantized global near-vertex welding"
        ),
    }
    print(
        "[global direct mesh] "
        f"vertices={welded_vertices.shape[0]:,} "
        f"faces={welded_faces.shape[0]:,} "
        f"welded={weld_stats['vertices_welded']:,}"
    )
    return (
        welded_vertices,
        welded_faces,
        welded_attrs,
        reference_layout,
        ownership_rows,
        stats,
    )


def _global_carrier_metadata(
    global_template: MeshWithVoxel,
    *,
    resolution: int,
) -> Tuple[torch.Tensor, float, torch.Size, Dict[str, Any]]:
    if int(resolution) <= core.OVOXEL_RESOLUTION:
        raise ValueError("global carrier resolution must exceed 1024")
    origin = torch.as_tensor(
        global_template.origin, dtype=torch.float32
    ).reshape(-1)
    if origin.numel() != 3:
        raise ValueError("global O-Voxel origin must contain three values")
    base_size = torch.as_tensor(
        global_template.voxel_size, dtype=torch.float64
    ).reshape(-1)
    if base_size.numel() == 1:
        base_size = base_size.repeat(3)
    if base_size.numel() != 3 or not torch.allclose(
        base_size, base_size[:1].expand_as(base_size), atol=1e-12, rtol=0.0
    ):
        raise ValueError("global O-Voxel voxel_size must be isotropic")
    extent = float(base_size[0].item()) * float(core.OVOXEL_RESOLUTION)
    carrier_size = extent / float(resolution)
    attr_channels = int(global_template.attrs.shape[1])
    voxel_shape = torch.Size(
        [1, attr_channels, int(resolution), int(resolution), int(resolution)]
    )
    stats = {
        "source_resolution": int(core.OVOXEL_RESOLUTION),
        "carrier_resolution": int(resolution),
        "origin": [float(value) for value in origin.tolist()],
        "source_voxel_size": float(base_size[0].item()),
        "carrier_voxel_size": float(carrier_size),
        "spatial_extent": float(extent),
        "voxel_shape": list(voxel_shape),
        "quantization": (
            "floor((global_absolute_object - origin) / carrier_voxel_size)"
        ),
    }
    return origin, float(carrier_size), voxel_shape, stats


def _floor_quantize_global_points(
    points: torch.Tensor,
    *,
    origin: torch.Tensor,
    voxel_size: float,
    resolution: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("global absolute O-Voxel points must have shape [N,3]")
    origin = origin.to(device=points.device, dtype=points.dtype)
    continuous = (points - origin[None]) / float(voxel_size)
    floored = torch.floor(continuous).to(torch.int64)
    valid = torch.isfinite(points).all(dim=1) & (
        (floored >= 0) & (floored < int(resolution))
    ).all(dim=1)
    centers = origin[None] + (
        floored.to(points.dtype) + 0.5
    ) * float(voxel_size)
    error_in_cells = torch.linalg.vector_norm(
        (points - centers) / float(voxel_size), dim=1
    )
    return floored.to(torch.int32), valid, error_in_cells


def _same_tile_floor_dedup_rows(
    coords: torch.Tensor,
    error_in_cells: torch.Tensor,
    *,
    resolution: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    coords_cpu = coords.to(device="cpu", dtype=torch.int64).contiguous()
    errors = error_in_cells.to(
        device="cpu", dtype=torch.float64
    ).contiguous().numpy()
    keys = core._linear_keys(coords_cpu, int(resolution)).numpy()
    original = np.arange(coords_cpu.shape[0], dtype=np.int64)
    order = np.lexsort((original, errors, keys))
    sorted_keys = keys[order]
    first = np.ones(order.shape[0], dtype=bool)
    first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    counts = np.unique(keys, return_counts=True)[1]
    selected = torch.from_numpy(order[first]).to(torch.long)
    return selected, {
        "floor_candidates": int(coords.shape[0]),
        "unique_floor_cells_kept": int(selected.shape[0]),
        "same_tile_floor_collision_candidates_discarded": int(
            coords.shape[0] - selected.shape[0]
        ),
        "same_tile_floor_collision_cells": int((counts > 1).sum()),
        "same_tile_floor_max_candidates_in_one_cell": int(counts.max()),
        "same_tile_floor_selection": (
            "keep candidate closest to the selected carrier cell center; "
            "original decoder row breaks exact ties"
        ),
    }


def _return_local_ovoxels_to_global_carrier(
    *,
    tile_id: int,
    local_mesh: MeshWithVoxel,
    global_camera: Mapping[str, float],
    transform: core.TileCameraTransform,
    carrier_origin: torch.Tensor,
    carrier_voxel_size: float,
    carrier_resolution: int,
) -> ReturnedTileOVoxels:
    local_absolute = core._ovoxel_indices_to_object(
        local_mesh.coords,
        origin=local_mesh.origin,
        voxel_size=local_mesh.voxel_size,
    )
    local_q = local_absolute * (2.0 * float(transform.mesh_scale))
    local_points = core._camera_q_to_points(
        local_q,
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
    )
    local_uv, _, local_finite = core._project_points(
        local_points,
        fx=float(transform.fx),
        fy=float(transform.fy),
        cx=float(transform.cx),
        cy=float(transform.cy),
    )
    center = local_uv.new_tensor(
        [float(transform.cx), float(transform.cy)]
    )[None]
    center_distance_pixels = torch.linalg.vector_norm(
        local_uv - center, dim=1
    )
    tile_width = float(transform.box[2] - transform.box[0])
    tile_height = float(transform.box[3] - transform.box[1])
    half_diagonal = math.hypot(tile_width * 0.5, tile_height * 0.5)
    center_distance_normalized = center_distance_pixels / max(
        half_diagonal, 1e-12
    )
    center_weight = 1.0 / (
        1.0 + center_distance_normalized.square()
    )

    global_q, _ = core._local_q_to_global_q(
        local_q,
        global_camera=global_camera,
        transform=transform,
    )
    global_absolute = global_q / (
        2.0 * float(global_camera["mesh_scale"])
    )
    global_coords, valid_grid, error_in_cells = (
        _floor_quantize_global_points(
            global_absolute,
            origin=carrier_origin,
            voxel_size=float(carrier_voxel_size),
            resolution=int(carrier_resolution),
        )
    )
    valid = (
        valid_grid
        & local_finite
        & torch.isfinite(center_distance_normalized)
        & torch.isfinite(center_weight)
        & torch.isfinite(error_in_cells)
    )
    valid_rows = torch.where(valid)[0]
    if valid_rows.numel() == 0:
        raise RuntimeError("tile decode has no O-Voxel in global carrier")
    valid_coords = global_coords.index_select(0, valid_rows)
    valid_error = error_in_cells.index_select(0, valid_rows)
    keep_relative, collision_stats = _same_tile_floor_dedup_rows(
        valid_coords,
        valid_error,
        resolution=int(carrier_resolution),
    )
    keep_relative = keep_relative.to(valid_rows.device)
    kept_rows = valid_rows.index_select(0, keep_relative)
    coords_unique = global_coords.index_select(0, kept_rows)
    attrs_unique = local_mesh.attrs.index_select(0, kept_rows)
    distances_unique = center_distance_normalized.index_select(0, kept_rows)
    weights_unique = center_weight.index_select(0, kept_rows)
    kept_error = error_in_cells.index_select(0, kept_rows)
    stats = {
        "tile_id": int(tile_id),
        "local_decoder_vertices_ignored": int(local_mesh.vertices.shape[0]),
        "local_decoder_faces_ignored": int(local_mesh.faces.shape[0]),
        "local_decoder_ovoxels": int(local_mesh.coords.shape[0]),
        "local_absolute_coordinate_formula": (
            "local_origin + (local_relative_index + 0.5) * local_voxel_size"
        ),
        "global_absolute_coordinate_formula": (
            "local absolute -> local camera q -> global camera q -> "
            "global absolute object coordinate"
        ),
        "global_carrier_resolution": int(carrier_resolution),
        "global_floor_candidates_in_bounds": int(valid_rows.shape[0]),
        "global_floor_candidates_out_of_bounds_or_nonfinite": int(
            local_mesh.coords.shape[0] - valid_rows.shape[0]
        ),
        **collision_stats,
        "returned_unique_global_carrier_ovoxels": int(
            coords_unique.shape[0]
        ),
        "floor_cell_center_error_cells_mean": float(kept_error.mean().item()),
        "floor_cell_center_error_cells_max": float(kept_error.max().item()),
        "tile_center_distance_normalized_min": float(
            distances_unique.min().item()
        ),
        "tile_center_distance_normalized_mean": float(
            distances_unique.mean().item()
        ),
        "tile_center_distance_normalized_max": float(
            distances_unique.max().item()
        ),
        "tile_center_weight_min": float(weights_unique.min().item()),
        "tile_center_weight_mean": float(weights_unique.mean().item()),
        "tile_center_weight_max": float(weights_unique.max().item()),
        "tile_center_weight_formula": (
            "w = 1 / (1 + (pixel_distance / tile_half_diagonal)^2)"
        ),
        "global_absolute_min": [
            float(value)
            for value in global_absolute.index_select(
                0, kept_rows
            ).amin(dim=0).tolist()
        ],
        "global_absolute_max": [
            float(value)
            for value in global_absolute.index_select(
                0, kept_rows
            ).amax(dim=0).tolist()
        ],
    }
    print(
        f"[tile {int(tile_id):02d} carrier C{int(carrier_resolution)}] "
        f"local={local_mesh.coords.shape[0]:,} "
        f"in_bounds={valid_rows.shape[0]:,} "
        f"same_tile_floor_drop="
        f"{collision_stats['same_tile_floor_collision_candidates_discarded']:,} "
        f"kept={coords_unique.shape[0]:,}"
    )
    return ReturnedTileOVoxels(
        tile_id=int(tile_id),
        coords=coords_unique.detach().to(device="cpu", dtype=torch.int32),
        attrs=attrs_unique.detach().to(device="cpu", dtype=torch.float32),
        center_distance_normalized=distances_unique.detach().to(
            device="cpu", dtype=torch.float32
        ),
        center_weight=weights_unique.detach().to(
            device="cpu", dtype=torch.float32
        ),
        stats=stats,
    )


def _fuse_tile_ovoxels_by_center_weight(
    *,
    tiles: Sequence[ReturnedTileOVoxels],
    resolution: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Dict[str, Any],
]:
    if not tiles:
        raise RuntimeError("no successful returned tile O-Voxels to fuse")
    coords = torch.cat([tile.coords for tile in tiles], dim=0)
    attrs = torch.cat([tile.attrs for tile in tiles], dim=0)
    distances = torch.cat(
        [tile.center_distance_normalized for tile in tiles], dim=0
    )
    weights = torch.cat([tile.center_weight for tile in tiles], dim=0)
    tile_ids = torch.cat(
        [
            torch.full(
                (tile.coords.shape[0],),
                int(tile.tile_id),
                dtype=torch.int32,
            )
            for tile in tiles
        ],
        dim=0,
    )
    coords64 = coords.to(torch.int64)
    keys = core._linear_keys(coords64, int(resolution)).numpy()
    weight_np = weights.to(torch.float64).numpy()
    tile_np = tile_ids.to(torch.int64).numpy()
    original = np.arange(coords.shape[0], dtype=np.int64)
    order = np.lexsort((original, tile_np, -weight_np, keys))
    sorted_keys = keys[order]
    first = np.ones(order.shape[0], dtype=bool)
    first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    counts = np.unique(keys, return_counts=True)[1]
    selected = torch.from_numpy(order[first]).to(torch.long)
    selected_tile_ids = tile_ids.index_select(0, selected)
    candidate_by_tile: Dict[str, int] = {}
    winner_by_tile: Dict[str, int] = {}
    for tile in tiles:
        label = str(int(tile.tile_id))
        candidate_by_tile[label] = int(tile.coords.shape[0])
        winner_by_tile[label] = int(
            (selected_tile_ids == int(tile.tile_id)).sum().item()
        )
    stats = {
        "successful_tiles": len(tiles),
        "tile_ids": [int(tile.tile_id) for tile in tiles],
        "global_carrier_resolution": int(resolution),
        "candidates_after_same_tile_floor_dedup": int(coords.shape[0]),
        "unique_global_carrier_ovoxels": int(selected.shape[0]),
        "cross_tile_conflict_candidates_discarded": int(
            coords.shape[0] - selected.shape[0]
        ),
        "cross_tile_contested_global_cells": int((counts > 1).sum()),
        "cross_tile_max_candidates_in_one_cell": int(counts.max()),
        "cross_tile_selection": (
            "highest source-tile center weight wins; smaller tile id then "
            "original candidate row break exact ties"
        ),
        "tile_center_weight_formula": (
            "w = 1 / (1 + (pixel_distance / tile_half_diagonal)^2)"
        ),
        "candidate_ovoxels_by_tile": candidate_by_tile,
        "winning_ovoxels_by_tile": winner_by_tile,
        "winning_fraction_by_tile": {
            tile_id: (
                float(winner_by_tile[tile_id])
                / float(candidate_by_tile[tile_id])
            )
            for tile_id in candidate_by_tile
        },
    }
    print(
        f"[global carrier C{int(resolution)} fuse] "
        f"candidates={coords.shape[0]:,} "
        f"contested_cells={stats['cross_tile_contested_global_cells']:,} "
        f"cross_tile_drop="
        f"{stats['cross_tile_conflict_candidates_discarded']:,} "
        f"kept={selected.shape[0]:,}"
    )
    return (
        coords.index_select(0, selected).contiguous(),
        attrs.index_select(0, selected).contiguous(),
        distances.index_select(0, selected).contiguous(),
        weights.index_select(0, selected).contiguous(),
        selected_tile_ids.contiguous(),
        stats,
    )


def _direct_mesh_with_local_vertex_pbr(
    *,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    vertex_attrs: torch.Tensor,
    layout: Mapping[str, Any],
) -> MeshWithVertexPbr:
    """Build the globally merged mesh from locally baked vertex PBR."""
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("direct merged vertices must have shape [N,3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("direct merged faces must have shape [M,3]")
    if (
        vertex_attrs.ndim != 2
        or vertex_attrs.shape[0] != vertices.shape[0]
    ):
        raise ValueError("vertex PBR must be [N,C] aligned with vertices")
    return MeshWithVertexPbr(
        vertices=vertices.to(dtype=torch.float32).contiguous(),
        faces=faces.to(dtype=torch.int32).contiguous(),
        vertex_attrs=vertex_attrs.to(dtype=torch.float32).contiguous(),
        layout=dict(layout),
    )


def _global_ovoxels_to_surface_mesh(
    *,
    coords: torch.Tensor,
    attrs: torch.Tensor,
    origin: torch.Tensor,
    voxel_size: float,
    voxel_shape: torch.Size,
    layout: Mapping[str, Any],
    resolution: int,
    chunk_size: int,
) -> Tuple[MeshWithVoxel, Dict[str, Any]]:
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        raise ValueError("fused global O-Voxel coords must be non-empty [N,3]")
    if attrs.ndim != 2 or attrs.shape[0] != coords.shape[0]:
        raise ValueError("fused global O-Voxel attrs are not aligned")
    if int(chunk_size) <= 0:
        raise ValueError("voxel mesh chunk size must be positive")
    coords_np = coords.to(torch.int64).contiguous().numpy()
    keys = core._linear_keys(coords.to(torch.int64), int(resolution)).numpy()
    sorted_keys = np.sort(keys)
    if bool((sorted_keys[1:] == sorted_keys[:-1]).any()):
        raise RuntimeError("fused global C4096 O-Voxel coords are not unique")
    directions = np.asarray(
        [
            [-1, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
        ],
        dtype=np.int64,
    )
    face_corners = torch.tensor(
        [
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
            [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
            [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
            [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        ],
        dtype=torch.float32,
    )
    triangle_pattern = torch.tensor(
        [[0, 1, 2], [0, 2, 3]], dtype=torch.int64
    )
    origin_cpu = origin.to(device="cpu", dtype=torch.float32)
    vertex_chunks: List[torch.Tensor] = []
    face_chunks: List[torch.Tensor] = []
    total_vertices = 0
    exposed_quads = 0
    direction_quads = [0] * 6
    for start in range(0, int(coords_np.shape[0]), int(chunk_size)):
        block = coords_np[start : start + int(chunk_size)]
        for direction_index, direction in enumerate(directions):
            neighbor = block + direction[None]
            in_bounds = (
                (neighbor >= 0) & (neighbor < int(resolution))
            ).all(axis=1)
            neighbor_keys = (
                (
                    neighbor[:, 0] * int(resolution)
                    + neighbor[:, 1]
                )
                * int(resolution)
                + neighbor[:, 2]
            )
            positions = np.searchsorted(sorted_keys, neighbor_keys)
            occupied = np.zeros(block.shape[0], dtype=bool)
            test = in_bounds & (positions < sorted_keys.shape[0])
            occupied[test] = (
                sorted_keys[positions[test]] == neighbor_keys[test]
            )
            exposed_block = block[~occupied]
            if exposed_block.shape[0] == 0:
                continue
            base_coords = torch.from_numpy(exposed_block).to(torch.float32)
            vertices = origin_cpu[None, None] + (
                base_coords[:, None] + face_corners[direction_index][None]
            ) * float(voxel_size)
            vertices = vertices.reshape(-1, 3).contiguous()
            quad_count = int(exposed_block.shape[0])
            faces = (
                torch.arange(
                    quad_count, dtype=torch.int64
                )[:, None, None]
                * 4
                + triangle_pattern[None]
                + int(total_vertices)
            ).reshape(-1, 3)
            vertex_chunks.append(vertices)
            face_chunks.append(faces)
            total_vertices += int(vertices.shape[0])
            exposed_quads += quad_count
            direction_quads[direction_index] += quad_count
    if not face_chunks:
        raise RuntimeError("fused global O-Voxels produced no exposed faces")
    if total_vertices >= 2**31:
        raise RuntimeError(
            "C4096 voxel surface exceeds int32 mesh vertex-index capacity"
        )
    vertices = torch.cat(vertex_chunks, dim=0)
    faces = torch.cat(face_chunks, dim=0).to(torch.int32)
    mesh = MeshWithVoxel(
        vertices=vertices,
        faces=faces,
        origin=[float(value) for value in origin_cpu.tolist()],
        voxel_size=float(voxel_size),
        coords=coords.to(dtype=torch.int32).contiguous(),
        attrs=attrs.to(dtype=torch.float32).contiguous(),
        voxel_shape=voxel_shape,
        layout=dict(layout),
    )
    candidate_quads = int(coords.shape[0]) * 6
    stats = {
        "mesh_source": "union surface of fused global C4096 O-Voxel cells",
        "active_global_c4096_ovoxels": int(coords.shape[0]),
        "candidate_voxel_quads": candidate_quads,
        "internal_neighbor_quads_removed": int(
            candidate_quads - exposed_quads
        ),
        "exposed_surface_quads": int(exposed_quads),
        "exposed_quads_by_direction_xyz_negative_positive": direction_quads,
        "surface_vertices_with_per_quad_duplication": int(vertices.shape[0]),
        "vertex_sharing": (
            "disabled across exposed quads so opposite faces cannot cancel "
            "the renderer's generated vertex normals"
        ),
        "surface_triangles": int(faces.shape[0]),
        "surface_extraction_chunk_size": int(chunk_size),
        "surface_rule": (
            "emit a quad only when the adjacent C4096 cell is unoccupied"
        ),
    }
    print(
        "[global C4096 mesh] "
        f"ovoxels={coords.shape[0]:,} "
        f"exposed_quads={exposed_quads:,} "
        f"triangles={faces.shape[0]:,}"
    )
    return mesh, stats


def _metric_subset(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "psnr_db": row.get("psnr_db"),
        "ssim": row.get("ssim"),
        "lpips": row.get("lpips"),
        "render_png": row.get("render_png"),
        "comparison_png": row.get("comparison_png"),
        "vertices": row.get("decoder_vertices"),
        "faces": row.get("decoder_faces"),
        "active_ovoxels": row.get("active_voxels"),
        "render_seconds": row.get("render_seconds"),
    }


def _label_panel(
    image: Image.Image,
    label: str,
    *,
    metrics: Optional[Mapping[str, Any]] = None,
) -> Image.Image:
    panel = image.convert("RGB").resize((640, 640), Image.Resampling.LANCZOS)
    output = Image.new("RGB", (640, 694), (0, 0, 0))
    output.paste(panel, (0, 54))
    draw = ImageDraw.Draw(output)
    draw.text((10, 8), label, fill=(255, 255, 255))
    if metrics is not None:
        psnr = metrics.get("psnr_db")
        ssim = metrics.get("ssim")
        lpips = metrics.get("lpips")
        text = (
            f"PSNR={psnr:.3f}  SSIM={ssim:.4f}"
            if psnr is not None and ssim is not None
            else "metrics unavailable"
        )
        if lpips is not None:
            text += f"  LPIPS={lpips:.4f}"
        draw.text((10, 29), text, fill=(220, 220, 220))
    return output


def _save_three_way_comparison(
    *,
    original_path: Path,
    global_render_path: Path,
    tile_render_path: Path,
    global_metrics: Mapping[str, Any],
    tile_metrics: Mapping[str, Any],
    output_path: Path,
) -> None:
    with Image.open(original_path) as image:
        original = _label_panel(image, "Original canonical 1024")
    with Image.open(global_render_path) as image:
        global_panel = _label_panel(
            image, "Ordinary global Pixal3D 1024", metrics=global_metrics
        )
    with Image.open(tile_render_path) as image:
        tile_panel = _label_panel(
            image,
            "Tile-flow direct local-FDG mesh merge",
            metrics=tile_metrics,
        )
    sheet = Image.new("RGB", (original.width * 3, original.height), (0, 0, 0))
    sheet.paste(original, (0, 0))
    sheet.paste(global_panel, (original.width, 0))
    sheet.paste(tile_panel, (original.width * 2, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _parse_angle_csv(value: str, *, label: str) -> List[float]:
    try:
        angles = [
            float(item.strip())
            for item in str(value).split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise ValueError(f"{label} must be a comma-separated float list") from exc
    if not angles or not all(math.isfinite(angle) for angle in angles):
        raise ValueError(f"{label} must contain finite angles")
    return angles


def _save_multiview_sheet(
    *,
    frame_paths: Sequence[Path],
    labels: Sequence[str],
    output_path: Path,
) -> None:
    if len(frame_paths) != len(labels) or not frame_paths:
        raise ValueError("multi-view frame paths and labels are not aligned")
    panel_size = 512
    header = 38
    columns = min(3, len(frame_paths))
    rows = (len(frame_paths) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * panel_size, rows * (panel_size + header)),
        (0, 0, 0),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (path, label) in enumerate(zip(frame_paths, labels)):
        column = index % columns
        row = index // columns
        x = column * panel_size
        y = row * (panel_size + header)
        with Image.open(path) as image:
            panel = image.convert("RGB").resize(
                (panel_size, panel_size), Image.Resampling.LANCZOS
            )
        sheet.paste(panel, (x, y + header))
        draw.text((x + 10, y + 10), label, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _render_merged_mesh_multiview(
    mesh: MeshWithVertexPbr,
    *,
    output_dir: Path,
    camera: Mapping[str, float],
    args: argparse.Namespace,
    envmap: Any,
) -> Dict[str, Any]:
    yaws = _parse_angle_csv(
        args.multiview_yaws_degrees,
        label="--multiview-yaws-degrees",
    )
    pitches = _parse_angle_csv(
        args.multiview_pitches_degrees,
        label="--multiview-pitches-degrees",
    )
    if len(pitches) == 1 and len(yaws) > 1:
        pitches = pitches * len(yaws)
    if len(yaws) != len(pitches):
        raise ValueError(
            "--multiview-yaws-degrees and --multiview-pitches-degrees "
            "must have equal lengths, or provide one pitch"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    live_mesh = mesh.to(torch.device("cuda"))
    radius = float(camera["distance"]) * float(args.multiview_radius_scale)
    fov = torch.tensor(
        float(camera["camera_angle_x"]),
        device="cuda",
        dtype=torch.float32,
    )
    intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
    target = torch.zeros(3, device="cuda", dtype=torch.float32)
    up = torch.tensor([0.0, 1.0, 0.0], device="cuda")
    extrinsics: List[torch.Tensor] = []
    intrinsics_list: List[torch.Tensor] = []
    labels: List[str] = []
    for yaw_degrees, pitch_degrees in zip(yaws, pitches):
        yaw = math.radians(float(yaw_degrees))
        pitch = math.radians(float(pitch_degrees))
        direction = torch.tensor(
            [
                math.sin(yaw) * math.cos(pitch),
                math.sin(pitch),
                math.cos(yaw) * math.cos(pitch),
            ],
            device="cuda",
            dtype=torch.float32,
        )
        camera_position = target + direction * radius
        extrinsics.append(
            utils3d.torch.extrinsics_look_at(
                camera_position,
                target,
                up,
            )
        )
        intrinsics_list.append(intrinsics)
        labels.append(
            f"yaw={float(yaw_degrees):g}°, pitch={float(pitch_degrees):g}°"
        )
    near = max(0.01, radius - 2.0)
    far = radius + 10.0
    started = time.perf_counter()
    renders = render_utils.render_frames(
        live_mesh,
        extrinsics=extrinsics,
        intrinsics=intrinsics_list,
        options={
            "resolution": int(args.multiview_resolution),
            "near": float(near),
            "far": float(far),
            "ssaa": int(args.multiview_ssaa),
            "peel_layers": int(args.multiview_peel_layers),
            "face_chunk_size": int(args.render_face_chunk_size),
        },
        verbose=True,
        envmap=envmap,
        use_envmap_bg=bool(args.use_envmap_bg),
    )
    shaded_frames = renders.get("shaded")
    if not shaded_frames or len(shaded_frames) != len(labels):
        raise RuntimeError("multi-view renderer did not return all shaded frames")
    frame_paths: List[Path] = []
    for index, (frame, label) in enumerate(zip(shaded_frames, labels)):
        path = output_dir / f"view_{index:02d}.png"
        Image.fromarray(np.asarray(frame)).convert("RGB").save(path)
        frame_paths.append(path)
        print(f"[multiview] {label} -> {path.name}")
    sheet_path = output_dir / "multiview_sheet.png"
    _save_multiview_sheet(
        frame_paths=frame_paths,
        labels=labels,
        output_path=sheet_path,
    )
    del live_mesh, renders
    _empty_cuda_cache()
    return {
        "enabled": True,
        "resolution": int(args.multiview_resolution),
        "ssaa": int(args.multiview_ssaa),
        "peel_layers": int(args.multiview_peel_layers),
        "radius": float(radius),
        "radius_scale": float(args.multiview_radius_scale),
        "yaw_degrees": yaws,
        "pitch_degrees": pitches,
        "frame_pngs": [str(path) for path in frame_paths],
        "sheet_png": str(sheet_path),
        "render_seconds": float(time.perf_counter() - started),
    }


def _sampler_params(args: argparse.Namespace) -> Tuple[Dict[str, Any], ...]:
    return (
        {
            "steps": int(args.ss_steps),
            "guidance_strength": float(args.ss_guidance_strength),
            "guidance_rescale": float(args.ss_guidance_rescale),
            "rescale_t": float(args.ss_rescale_t),
        },
        {
            "steps": int(args.shape_steps),
            "guidance_strength": float(args.shape_guidance_strength),
            "guidance_rescale": float(args.shape_guidance_rescale),
            "rescale_t": float(args.shape_rescale_t),
        },
        {
            "steps": int(args.texture_steps),
            "guidance_strength": float(args.texture_guidance_strength),
            "guidance_rescale": float(args.texture_guidance_rescale),
            "rescale_t": float(args.texture_rescale_t),
        },
    )


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(int(args.cuda_device))
    device = torch.device("cuda")
    print(
        f"[cuda] index={int(args.cuda_device)} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())}"
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = init_pipeline(
        args.model_path,
        device="cuda",
        low_vram=bool(args.low_vram),
    )

    canonical = pipeline.preprocess_canonical_images(Image.open(args.image))
    image_4096: Image.Image = canonical["image_4096"]
    image_1024: Image.Image = canonical["image_1024"]
    image_512: Image.Image = canonical["image_512"]
    image_4096.save(output_dir / "canonical_4096.png")
    image_1024.save(output_dir / "canonical_1024.png")
    image_512.save(output_dir / "canonical_512.png")
    _atomic_json(output_dir / "canonical_metadata.json", canonical["metadata"])
    global_camera = core._estimate_camera(
        image_1024=image_1024,
        output_dir=output_dir,
        manual_fov=float(args.fov),
        mesh_scale=float(args.mesh_scale),
        extend_pixel=int(args.extend_pixel),
        moge_model_path=args.moge_model_path,
    )
    _atomic_json(output_dir / "global_camera.json", global_camera)

    ss_params, shape_params, texture_params = _sampler_params(args)
    print("[global] ordinary Pixal3D 1024_cascade")
    _seed_everything(int(args.seed))
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
        raise RuntimeError("ordinary global route did not return exactly one mesh")
    baseline_live = core._validate_mesh(
        baseline_output[0], "ordinary global Pixal3D 1024"
    )
    _, _, baseline_resolution = baseline_latents
    if int(baseline_resolution) != core.OVOXEL_RESOLUTION:
        raise RuntimeError("ordinary global decoder did not use resolution 1024")

    envmap = load_envmap(str(args.envmap), device="cuda")
    baseline_dir = output_dir / "global_baseline_1024"
    baseline_metric = core._render(
        baseline_live,
        output_dir=baseline_dir / "aligned_eval",
        camera=global_camera,
        reference_image=output_dir / "canonical_1024.png",
        args=args,
        envmap=envmap,
    )
    baseline_mesh = baseline_live.to("cpu")
    baseline_summary = {
        "generation_seconds": float(baseline_seconds),
        "decoder_resolution": int(baseline_resolution),
        "metrics": _metric_subset(baseline_metric),
    }
    _atomic_json(baseline_dir / "summary.json", baseline_summary)
    if args.save_mesh_checkpoints:
        torch.save(baseline_mesh, baseline_dir / "mesh_with_ovoxel.pt")
    del baseline_output, baseline_latents, baseline_live
    _empty_cuda_cache()

    global_ovoxel_object = core._ovoxel_indices_to_object(
        baseline_mesh.coords,
        origin=baseline_mesh.origin,
        voxel_size=baseline_mesh.voxel_size,
    )
    global_ovoxel_q = global_ovoxel_object * (
        2.0 * float(global_camera["mesh_scale"])
    )
    global_ovoxel_uv, _, global_ovoxel_finite = (
        core._project_global_q_to_4096(
            global_ovoxel_q, global_camera=global_camera
        )
    )
    global_face_uv, global_face_finite = core._project_face_centers(
        baseline_mesh.vertices,
        baseline_mesh.faces,
        mesh_scale=float(global_camera["mesh_scale"]),
        global_camera=global_camera,
        chunk_size=int(args.face_projection_chunk_size),
    )

    shape_encoder = pixal3d_models.from_pretrained(
        str(Path(args.shape_encoder).expanduser())
    ).eval()
    pbr_encoder = pixal3d_models.from_pretrained(
        str(Path(args.pbr_encoder).expanduser())
    ).eval()
    if not args.low_vram:
        shape_encoder.to(device)
        pbr_encoder.to(device)

    boxes = core._tile_layout()
    requested_ids = core._parse_tile_ids(args.tile_ids)
    if requested_ids is not None:
        invalid = sorted(tile_id for tile_id in requested_ids if tile_id not in range(49))
        if invalid:
            raise ValueError(f"invalid tile ids {invalid}; valid ids are 0..48")

    returned_tile_meshes: List[ReturnedTileMesh] = []
    tile_records: List[Dict[str, Any]] = []
    attempted_eligible = 0
    for tile_id, box in enumerate(boxes):
        if requested_ids is not None and tile_id not in requested_ids:
            continue
        projected_count = int(
            core._inside_tile(
                global_ovoxel_uv, global_ovoxel_finite, box
            ).sum().item()
        )
        tile_dir = output_dir / "tiles" / f"tile_{tile_id:02d}"
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_image = image_4096.crop(box).convert("RGB")
        lr_tile_image = _make_lr_reference_tile(image_1024, box)
        tile_image.save(tile_dir / "tile_reference.png")
        lr_tile_image.save(tile_dir / "tile_reference_lr_from_global_1024.png")
        if projected_count < int(args.min_tile_ovoxels):
            record = {
                "status": "skipped",
                "tile_id": tile_id,
                "box": list(box),
                "projected_global_ovoxels": projected_count,
                "reason": "projected global O-Voxels below threshold",
            }
            tile_records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            continue
        if args.max_tiles is not None and attempted_eligible >= int(args.max_tiles):
            break
        attempted_eligible += 1
        tile_started = time.perf_counter()
        transform = core._derive_tile_camera(
            tile_id=tile_id,
            box=box,
            global_camera=global_camera,
            extend_pixel=int(args.extend_pixel),
        )
        try:
            mapping = core._map_global_ovoxels_to_local(
                global_mesh=baseline_mesh,
                global_q=global_ovoxel_q,
                global_uv_4096=global_ovoxel_uv,
                finite_projection=global_ovoxel_finite,
                global_camera=global_camera,
                transform=transform,
            )
            (
                local_geometry_vertices,
                local_geometry_faces,
                _,
                _,
                geometry_stats,
            ) = core._prepare_tile_geometry(
                global_vertices=baseline_mesh.vertices,
                global_faces=baseline_mesh.faces,
                global_face_uv=global_face_uv,
                global_face_finite=global_face_finite,
                global_camera=global_camera,
                transform=transform,
            )
            encoded_shape, shape_encoder_stats = core._encode_local_shape(
                encoder=shape_encoder,
                vertices=local_geometry_vertices,
                faces=local_geometry_faces,
                device=device,
                low_vram=bool(args.low_vram),
            )
            encoded_pbr, pbr_encoder_stats = core._encode_local_pbr(
                encoder=pbr_encoder,
                coords=mapping.local_coords,
                attrs=mapping.local_attrs,
                device=device,
                low_vram=bool(args.low_vram),
            )
            encoded_shape, encoded_pbr, alignment_stats = (
                core._align_latent_supports(encoded_shape, encoded_pbr)
            )
            query_coords = encoded_shape.coords.detach().clone()
            if query_coords.shape[0] > int(args.max_num_tokens):
                raise RuntimeError(
                    f"common local latent has {query_coords.shape[0]:,} tokens, "
                    f"exceeding --max-num-tokens={int(args.max_num_tokens):,}"
                )
            selected_uv = global_ovoxel_uv.index_select(
                0, mapping.source_global_rows
            )
            selected_tile_uv = torch.stack(
                (
                    selected_uv[:, 0] - float(box[0]),
                    selected_uv[:, 1] - float(box[1]),
                ),
                dim=1,
            )
            selected_global_q = global_ovoxel_q.index_select(
                0, mapping.source_global_rows
            )
            selected_depth = float(global_camera["distance"]) - (
                selected_global_q[:, 2]
                / (2.0 * float(global_camera["mesh_scale"]))
            )
            visualization_stats = core._save_selected_ovoxel_visualizations(
                tile_image=tile_image,
                ovoxel_uv=selected_tile_uv,
                ovoxel_depth=selected_depth,
                output_dir=tile_dir,
                tile_id=tile_id,
                latent_coords=query_coords,
                transform=transform,
            )
            tile_seed = int(args.seed) + tile_id * 1000
            flow_latents = _run_tile_query_flows(
                pipeline=pipeline,
                hr_tile_image=tile_image,
                lr_tile_image=lr_tile_image,
                query_coords=query_coords,
                encoded_shape=encoded_shape,
                encoded_pbr=encoded_pbr,
                transform=transform,
                shape_params=shape_params,
                texture_params=texture_params,
                seed=tile_seed,
                tile_id=tile_id,
            )
            latent_difference = {
                "shape": _latent_feature_difference(
                    encoded=encoded_shape,
                    flowed_denorm=flow_latents.shape_denorm,
                    label="shape latent",
                ),
                "texture": _latent_feature_difference(
                    encoded=encoded_pbr,
                    flowed_denorm=flow_latents.texture_denorm,
                    label="texture/PBR latent",
                ),
            }
            decode_started = time.perf_counter()
            with torch.no_grad():
                decoded = pipeline.decode_latent(
                    flow_latents.shape_denorm,
                    flow_latents.texture_denorm,
                    core.OVOXEL_RESOLUTION,
                )
            _sync_cuda()
            decode_seconds = time.perf_counter() - decode_started
            if len(decoded) != 1:
                raise RuntimeError("tile flow decoder did not return exactly one mesh")
            local_mesh = core._validate_mesh(
                decoded[0], f"tile {tile_id:02d} global-anchor detail flow"
            )
            returned_mesh = _return_local_mesh_to_global(
                tile_id=tile_id,
                local_mesh=local_mesh,
                global_camera=global_camera,
                transform=transform,
            )
            returned_tile_meshes.append(returned_mesh)
            if args.save_mesh_checkpoints:
                torch.save(
                    {
                        "query_coords": query_coords.to("cpu"),
                        "encoded_shape_latent": encoded_shape.to("cpu"),
                        "encoded_texture_latent": encoded_pbr.to("cpu"),
                        "shape_latent": flow_latents.shape_denorm.to("cpu"),
                        "texture_latent": flow_latents.texture_denorm.to("cpu"),
                    },
                    tile_dir / "global_anchor_detail_flow_latents.pt",
                )
            record = {
                "status": "success",
                "tile_id": tile_id,
                "box": list(box),
                "projected_global_ovoxels": projected_count,
                "mapping": mapping.stats,
                "geometry_encoder_input": geometry_stats,
                "shape_encoder": shape_encoder_stats,
                "pbr_encoder": pbr_encoder_stats,
                "query_alignment": alignment_stats,
                "query_visualizations": visualization_stats,
                "global_anchor_detail_flows": flow_latents.stats,
                "latent_feature_difference": latent_difference,
                "decode_seconds": float(decode_seconds),
                "returned_global_mesh": returned_mesh.stats,
                "tile_seconds": float(time.perf_counter() - tile_started),
            }
            tile_records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            print(
                f"[tile {tile_id:02d}] success "
                f"queries={query_coords.shape[0]:,} "
                f"mesh_faces={returned_mesh.faces.shape[0]:,} "
                f"shape_MAE={latent_difference['shape']['flow_minus_encoder']['mae']:.5f} "
                f"tex_MAE={latent_difference['texture']['flow_minus_encoder']['mae']:.5f}"
            )
            del (
                mapping,
                encoded_shape,
                encoded_pbr,
                flow_latents,
                decoded,
                local_mesh,
                query_coords,
            )
            _empty_cuda_cache()
        except Exception as exc:
            record = {
                "status": "failed",
                "tile_id": tile_id,
                "box": list(box),
                "projected_global_ovoxels": projected_count,
                "tile_seconds": float(time.perf_counter() - tile_started),
                "reason": f"{type(exc).__name__}: {exc}",
            }
            tile_records.append(record)
            _atomic_json(tile_dir / "summary.json", record)
            print(f"[tile {tile_id:02d}] FAILED: {record['reason']}")
            _empty_cuda_cache()

    del shape_encoder, pbr_encoder
    _empty_cuda_cache()
    (
        merged_vertices,
        merged_faces,
        merged_vertex_attrs,
        merged_layout,
        triangle_ownership_rows,
        geometry_merge_stats,
    ) = _merge_tile_meshes_by_nearest_center(
        tiles=returned_tile_meshes,
        face_projection_chunk_size=int(args.face_projection_chunk_size),
        vertex_weld_tolerance=float(args.vertex_weld_tolerance),
        device=device,
    )
    _empty_cuda_cache()
    merged_mesh = _direct_mesh_with_local_vertex_pbr(
        vertices=merged_vertices,
        faces=merged_faces,
        vertex_attrs=merged_vertex_attrs,
        layout=merged_layout,
    )
    merged_dir = output_dir / "tile_owned_direct_global_mesh"
    merged_dir.mkdir(parents=True, exist_ok=True)
    if args.save_mesh_checkpoints:
        torch.save(
            {
                "mesh_with_local_vertex_pbr": merged_mesh,
                "geometry_merge_stats": geometry_merge_stats,
                "triangle_ownership": triangle_ownership_rows,
            },
            merged_dir / "direct_merged_mesh_with_local_vertex_pbr.pt",
        )
    merged_metric = core._render(
        merged_mesh,
        output_dir=merged_dir / "aligned_eval",
        camera=global_camera,
        reference_image=output_dir / "canonical_1024.png",
        args=args,
        envmap=envmap,
    )
    global_metrics = _metric_subset(baseline_metric)
    tile_metrics = _metric_subset(merged_metric)
    comparison_path = output_dir / (
        "comparison_original_global_tile_direct_mesh.png"
    )
    _save_three_way_comparison(
        original_path=output_dir / "canonical_1024.png",
        global_render_path=Path(str(baseline_metric["render_png"])),
        tile_render_path=Path(str(merged_metric["render_png"])),
        global_metrics=global_metrics,
        tile_metrics=tile_metrics,
        output_path=comparison_path,
    )
    if bool(args.render_multiview):
        multiview_summary = _render_merged_mesh_multiview(
            merged_mesh,
            output_dir=merged_dir / "multiview",
            camera=global_camera,
            args=args,
            envmap=envmap,
        )
    else:
        multiview_summary = {"enabled": False}
    delta = {
        "tile_minus_global_psnr_db": (
            None
            if global_metrics["psnr_db"] is None
            or tile_metrics["psnr_db"] is None
            else float(tile_metrics["psnr_db"])
            - float(global_metrics["psnr_db"])
        ),
        "tile_minus_global_ssim": (
            None
            if global_metrics["ssim"] is None or tile_metrics["ssim"] is None
            else float(tile_metrics["ssim"]) - float(global_metrics["ssim"])
        ),
        "tile_minus_global_lpips": (
            None
            if global_metrics["lpips"] is None
            or tile_metrics["lpips"] is None
            else float(tile_metrics["lpips"])
            - float(global_metrics["lpips"])
        ),
    }
    merged_summary = {
        "geometry_merge": geometry_merge_stats,
        "triangle_ownership_by_tile": triangle_ownership_rows,
        "material": {
            "source": (
                "local decoded sparse O-Voxel PBR sampled onto local FDG "
                "dual vertices before geometry is returned to global"
            ),
            "interpolation": "barycentric vertex-PBR interpolation",
            "channels": int(merged_vertex_attrs.shape[1]),
        },
        "metrics": tile_metrics,
        "comparison_png": str(comparison_path),
        "multiview": multiview_summary,
    }
    _atomic_json(merged_dir / "summary.json", merged_summary)
    summary = {
        "format": "pixal3d_global_anchor_hr_lr_detail_direct_mesh_merge_v4",
        "image": str(Path(args.image).expanduser().resolve()),
        "cuda_device": int(args.cuda_device),
        "global_camera": global_camera,
        "route": {
            "query_source": (
                "common active C64 coordinates from local shape/PBR encoders"
            ),
            "shape": (
                "fresh noise -> per-step encoded-global clean anchor velocity "
                "+ (HR condition velocity - LR condition velocity) on the "
                "full common C64 support"
            ),
            "texture": (
                "fresh noise + anchored-detail normalized shape concat -> "
                "the same per-step encoded-global anchor and full-support "
                "HR-minus-LR condition residual"
            ),
            "detail_projector": (
                "HR-minus-LR isolates information absent at global image "
                "scale; no explicit spatial/front-back projector is applied, "
                "so Pixal3D's trained condition response controls front/back"
            ),
            "material": (
                "sample local decoded sparse O-Voxel PBR at local FDG dual "
                "vertices; carry vertex PBR through global transform, "
                "triangle ownership, and weighted seam welding"
            ),
            "final_mesh": (
                "decoder-native local FDG meshes returned to global object "
                "coordinates; triangle centroids assigned to the nearest "
                "successful tile center; owned meshes compacted, concatenated, "
                "and near-identical global vertices welded"
            ),
        },
        "comparison_scope": (
            "full-frame original/global/direct-tile-mesh render metrics plus "
            "per-tile same-coordinate latent feature diagnostics; no per-tile "
            "render metrics; extra merged-mesh multi-view visualization"
        ),
        "global_baseline": baseline_summary,
        "tile_owned_direct_global_mesh": merged_summary,
        "metric_delta": delta,
        "latent_feature_difference_aggregate": (
            _aggregate_latent_feature_differences(tile_records)
        ),
        "successful_tiles": len(returned_tile_meshes),
        "failed_tiles": sum(row["status"] == "failed" for row in tile_records),
        "skipped_tiles": sum(row["status"] == "skipped" for row in tile_records),
        "tiles": tile_records,
        "comparison_png": str(comparison_path),
        "multiview_sheet_png": multiview_summary.get("sheet_png"),
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(f"[done] summary={output_dir / 'summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--moge-model-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument(
        "--low-vram",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--shape-encoder",
        default=str(
            core.DEFAULT_ENCODER_ROOT / "shape_enc_next_dc_f16c32_fp16"
        ),
    )
    parser.add_argument(
        "--pbr-encoder",
        default=str(core.DEFAULT_ENCODER_ROOT / "tex_enc_next_dc_f16c32_fp16"),
    )
    parser.add_argument("--tile-ids", default=None)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--min-tile-ovoxels", type=int, default=1001)
    parser.add_argument("--max-num-tokens", type=int, default=1_000_000)
    parser.add_argument("--face-projection-chunk-size", type=int, default=250_000)
    parser.add_argument(
        "--vertex-weld-tolerance",
        type=float,
        default=1e-6,
        help=(
            "Global object-space quantization tolerance for welding "
            "near-identical vertices after triangle ownership; <=0 disables."
        ),
    )
    parser.add_argument(
        "--save-mesh-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
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
        "--multiview-yaws-degrees",
        default="0,-45,45,-90,90,180",
    )
    parser.add_argument(
        "--multiview-pitches-degrees",
        default="0,0,0,10,10,0",
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
        "--lpips-net",
        choices=("alex", "vgg", "squeeze"),
        default="vgg",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if int(args.cuda_device) < 0:
        raise ValueError("--cuda-device must be non-negative")
    if int(args.min_tile_ovoxels) < 1001:
        raise ValueError("--min-tile-ovoxels must be >=1001")
    if args.max_tiles is not None and int(args.max_tiles) < 1:
        raise ValueError("--max-tiles must be positive")
    if (
        int(args.max_num_tokens) < 1
        or int(args.face_projection_chunk_size) < 1
    ):
        raise ValueError("token/chunk limits must be positive")
    if not math.isfinite(float(args.vertex_weld_tolerance)):
        raise ValueError("--vertex-weld-tolerance must be finite")
    if (
        int(args.render_resolution) < 1
        or int(args.metric_resolution) < 1
        or int(args.render_ssaa) < 1
        or int(args.render_peel_layers) < 1
        or int(args.render_face_chunk_size) < 0
    ):
        raise ValueError("invalid render configuration")
    if bool(args.render_multiview):
        if (
            int(args.multiview_resolution) < 1
            or int(args.multiview_ssaa) < 1
            or int(args.multiview_peel_layers) < 1
            or not math.isfinite(float(args.multiview_radius_scale))
            or float(args.multiview_radius_scale) <= 0.0
        ):
            raise ValueError("invalid multi-view render configuration")
        yaws = _parse_angle_csv(
            args.multiview_yaws_degrees,
            label="--multiview-yaws-degrees",
        )
        pitches = _parse_angle_csv(
            args.multiview_pitches_degrees,
            label="--multiview-pitches-degrees",
        )
        if len(pitches) not in (1, len(yaws)):
            raise ValueError(
                "multi-view pitch count must be one or match yaw count"
            )
    for encoder_path in (args.shape_encoder, args.pbr_encoder):
        base = Path(encoder_path).expanduser()
        if not Path(f"{base}.json").is_file() or not Path(
            f"{base}.safetensors"
        ).is_file():
            raise FileNotFoundError(
                f"encoder checkpoint pair not found for base path {base}"
            )
    run(args)


if __name__ == "__main__":
    main()
