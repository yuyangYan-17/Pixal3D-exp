#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict global-encoder versus independent local-flow latent diagnostics.

This script is analysis-only.  It does not alter sampler trajectories, fuse
velocities, lock tokens, construct masks, merge meshes, or report render
metrics.  The primary analysis space is normalized flow space; decoder/raw
space is retained as a secondary, explicitly labelled comparison.
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import stats as scipy_stats
from tqdm import tqdm

import pixal3d.models as pixal3d_models
import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core
import pixal3d_tile_encoded_query_noise_flow_overlap_render as anchor
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor


SPACE_NORM = "normalized_flow_space"
SPACE_RAW = "decoder_raw_space"
RIDGE_LAMBDAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
MODE_COUNTS = (1, 2, 4, 8, 16, 24, 32)
LATENT_NAMES = ("shape", "texture")
PAIR_NAMES = (("global", "lr"), ("global", "hr"), ("lr", "hr"))
FORMAT_VERSION = "pixal3d_global_local_latent_relationship_v1"


@dataclass
class FlowTrajectory:
    endpoint_lr_norm: SparseTensor
    endpoint_hr_norm: SparseTensor
    per_step: Dict[int, Dict[str, Any]]
    elapsed_seconds: float
    metadata: Dict[str, Any]


@dataclass
class TileDataset:
    tile_id: int
    seed: int
    row: int
    column: int
    position_group: str
    tokens: int
    global_norm: np.ndarray
    lr_norm: np.ndarray
    hr_norm: np.ndarray
    path: Path


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


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().numpy())
    if isinstance(value, Path):
        return str(value)
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
        json.dump(
            _json_value(dict(payload)),
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
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _parse_int_csv(value: str) -> List[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("integer list must not be empty")
    if len(set(values)) != len(values):
        raise ValueError("integer list must not contain duplicates")
    return values


def _position_group(tile_id: int) -> str:
    row, column = divmod(int(tile_id), 7)
    return "central" if 2 <= row <= 4 and 2 <= column <= 4 else "edge"


def _normalization_tensors(
    normalization: Mapping[str, Sequence[float]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    channels: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.as_tensor(
        normalization["mean"], device=device, dtype=dtype
    ).reshape(1, -1)
    std = torch.as_tensor(
        normalization["std"], device=device, dtype=dtype
    ).reshape(1, -1)
    if mean.shape[1] != channels or std.shape[1] != channels:
        raise ValueError(
            f"normalization channels differ: expected={channels}, "
            f"mean={mean.shape[1]}, std={std.shape[1]}"
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("normalization contains non-finite values")
    if bool((std == 0).any().item()):
        raise ValueError("normalization contains zero std")
    return mean, std


def _normalize(
    value_raw: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
) -> SparseTensor:
    features = value_raw.feats.float()
    mean, std = _normalization_tensors(
        normalization,
        device=value_raw.device,
        dtype=torch.float32,
        channels=int(value_raw.feats.shape[1]),
    )
    return value_raw.replace((features - mean) / std)


def _denormalize(
    value_norm: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
) -> SparseTensor:
    features = value_norm.feats.float()
    mean, std = _normalization_tensors(
        normalization,
        device=value_norm.device,
        dtype=torch.float32,
        channels=int(value_norm.feats.shape[1]),
    )
    return value_norm.replace(features * std + mean)


def _space_assertions(
    *,
    encoded_raw: SparseTensor,
    flow_norm: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
    label: str,
) -> Dict[str, Any]:
    assert torch.equal(encoded_raw.coords, flow_norm.coords), (
        f"{label}: encoder and flow sparse coordinates differ"
    )
    encoded_norm = _normalize(encoded_raw, normalization)
    flow_raw = _denormalize(flow_norm, normalization)
    reconstructed_raw = _denormalize(encoded_norm, normalization)
    reconstructed_norm = _normalize(flow_raw, normalization)
    raw_error = float(
        (reconstructed_raw.feats.float() - encoded_raw.feats.float())
        .abs()
        .max()
        .item()
    )
    norm_error = float(
        (reconstructed_norm.feats.float() - flow_norm.feats.float())
        .abs()
        .max()
        .item()
    )
    max_error = max(raw_error, norm_error)
    assert raw_error < 1e-5, (
        f"{label}: encoder raw normalization roundtrip error {raw_error}"
    )
    assert norm_error < 1e-5, (
        f"{label}: flow normalized roundtrip error {norm_error}"
    )
    return {
        "encoder_space": "decoder_raw",
        "flow_space": "normalized",
        "analysis_primary_space": SPACE_NORM,
        "decoder_space_analysis": True,
        "coordinate_alignment": "exact C64 coordinates and token order",
        "normalization_roundtrip_max_error": max_error,
        "encoder_raw_roundtrip_max_error": raw_error,
        "flow_norm_roundtrip_max_error": norm_error,
    }


def _feature_stats(features: torch.Tensor) -> Dict[str, Any]:
    values = features.detach().float()
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("features must be non-empty [N,C]")
    return {
        "tokens": int(values.shape[0]),
        "channels": int(values.shape[1]),
        "per_channel_mean": values.mean(dim=0).cpu().tolist(),
        "per_channel_std": values.std(dim=0, unbiased=False).cpu().tolist(),
        "global_mean": float(values.mean().item()),
        "global_std": float(values.std(unbiased=False).item()),
    }


def _safe_cosine_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    numerator = (left * right).sum(dim=1)
    denominator = (
        torch.linalg.vector_norm(left, dim=1)
        * torch.linalg.vector_norm(right, dim=1)
    )
    result = numerator / denominator.clamp_min(1e-12)
    both_zero = (
        torch.linalg.vector_norm(left, dim=1) <= 1e-12
    ) & (torch.linalg.vector_norm(right, dim=1) <= 1e-12)
    return torch.where(both_zero, torch.ones_like(result), result)


def _comparison_stats(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    space: str,
    include_covariance: bool = True,
) -> Dict[str, Any]:
    if space not in (SPACE_NORM, SPACE_RAW):
        raise ValueError(f"unsupported comparison space {space}")
    left = left.detach().float()
    right = right.detach().float()
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] == 0:
        raise ValueError(
            f"comparison requires equal non-empty [N,C], got "
            f"{tuple(left.shape)} and {tuple(right.shape)}"
        )
    difference = right - left
    cosine = _safe_cosine_rows(left, right)
    left_centered = left - left.mean(dim=0, keepdim=True)
    right_centered = right - right.mean(dim=0, keepdim=True)
    centered_cosine = float(
        (
            (left_centered * right_centered).sum()
            / (
                torch.linalg.vector_norm(left_centered)
                * torch.linalg.vector_norm(right_centered)
            ).clamp_min(1e-12)
        ).item()
    )
    flat_left = left.reshape(-1)
    flat_right = right.reshape(-1)
    flat_left_centered = flat_left - flat_left.mean()
    flat_right_centered = flat_right - flat_right.mean()
    pearson = float(
        (
            (flat_left_centered * flat_right_centered).sum()
            / (
                torch.linalg.vector_norm(flat_left_centered)
                * torch.linalg.vector_norm(flat_right_centered)
            ).clamp_min(1e-12)
        ).item()
    )
    output = {
        "comparison_space": space,
        "tokens": int(left.shape[0]),
        "channels": int(left.shape[1]),
        "left": _feature_stats(left),
        "right": _feature_stats(right),
        "mae": float(difference.abs().mean().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "relative_l2": float(
            (
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(left).clamp_min(1e-12)
            ).item()
        ),
        "token_cosine_mean": float(cosine.mean().item()),
        "token_cosine_std": float(cosine.std(unbiased=False).item()),
        "centered_cosine": centered_cosine,
        "pearson_correlation": pearson,
    }
    if include_covariance:
        denominator = max(1, left.shape[0] - 1)
        cross_covariance = (
            left_centered.transpose(0, 1) @ right_centered
        ) / denominator
        spectrum = torch.linalg.svdvals(cross_covariance)
        output["cross_covariance"] = cross_covariance.cpu().tolist()
        output["cross_covariance_singular_values"] = spectrum.cpu().tolist()
    return output


def _randn_sparse(
    coords: torch.Tensor,
    channels: int,
    *,
    device: torch.device,
    seed: int,
) -> SparseTensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    features = torch.randn(
        int(coords.shape[0]),
        int(channels),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    return SparseTensor(features, coords.to(device=device, dtype=torch.int32))


def _prediction_features(
    prediction: Any,
    *,
    coords: torch.Tensor,
    label: str,
) -> torch.Tensor:
    if not isinstance(prediction, SparseTensor):
        raise TypeError(f"{label}: expected SparseTensor prediction")
    if not torch.equal(prediction.coords, coords):
        raise RuntimeError(f"{label}: prediction changed sparse coordinates")
    return prediction.feats.float()


def _deterministic_subset_indices(tokens: int, maximum: int) -> torch.Tensor:
    if tokens <= 0 or maximum <= 0:
        raise ValueError("token and subset counts must be positive")
    count = min(int(tokens), int(maximum))
    if count == tokens:
        return torch.arange(tokens, dtype=torch.int64)
    # Rounded linspace is deterministic, sorted, and spans the complete order.
    indices = torch.linspace(0, tokens - 1, count, dtype=torch.float64)
    indices = torch.round(indices).to(torch.int64)
    if torch.unique(indices).shape[0] != count:
        raise RuntimeError("deterministic token subset unexpectedly duplicated rows")
    return indices


@torch.no_grad()
def _sample_independent_pair(
    *,
    sampler: Any,
    model: torch.nn.Module,
    noise: SparseTensor,
    lr_condition: Mapping[str, Any],
    hr_condition: Mapping[str, Any],
    params: Mapping[str, Any],
    subset_indices: torch.Tensor,
    label: str,
    concat_cond_lr: Optional[SparseTensor] = None,
    concat_cond_hr: Optional[SparseTensor] = None,
    matched_concat_cond: Optional[SparseTensor] = None,
) -> FlowTrajectory:
    """Run independent LR/HR Euler flows plus HR response at each LR state."""
    coords = noise.coords
    for name, concat in (
        ("LR", concat_cond_lr),
        ("HR", concat_cond_hr),
        ("matched", matched_concat_cond),
    ):
        if concat is not None and not torch.equal(coords, concat.coords):
            raise RuntimeError(f"{label}: {name} concat support differs")
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
    time_sequence = sampler.timestep_schedule(steps, rescale_t)
    time_pairs = [
        (time_sequence[index], time_sequence[index + 1])
        for index in range(steps)
    ]
    sample_lr = noise.replace(noise.feats.clone())
    sample_hr = noise.replace(noise.feats.clone())
    subset_device = subset_indices.to(device=noise.device)
    per_step: Dict[int, Dict[str, Any]] = {}
    started = time.perf_counter()
    for step_index, (t_value, t_prev) in enumerate(
        tqdm(time_pairs, desc=label, leave=False)
    ):
        lr_call = dict(call_params)
        if concat_cond_lr is not None:
            lr_call["concat_cond"] = concat_cond_lr
        _, _, velocity_lr = sampler._get_model_prediction(
            model,
            sample_lr,
            float(t_value),
            **dict(lr_condition),
            **lr_call,
        )
        matched_hr_call = dict(call_params)
        if matched_concat_cond is not None:
            matched_hr_call["concat_cond"] = matched_concat_cond
        _, _, velocity_hr_matched = sampler._get_model_prediction(
            model,
            sample_lr,
            float(t_value),
            **dict(hr_condition),
            **matched_hr_call,
        )
        hr_call = dict(call_params)
        if concat_cond_hr is not None:
            hr_call["concat_cond"] = concat_cond_hr
        _, _, velocity_hr_independent = sampler._get_model_prediction(
            model,
            sample_hr,
            float(t_value),
            **dict(hr_condition),
            **hr_call,
        )
        velocity_lr_features = _prediction_features(
            velocity_lr,
            coords=coords,
            label=f"{label} LR step {step_index}",
        )
        velocity_hr_matched_features = _prediction_features(
            velocity_hr_matched,
            coords=coords,
            label=f"{label} matched HR step {step_index}",
        )
        velocity_hr_independent_features = _prediction_features(
            velocity_hr_independent,
            coords=coords,
            label=f"{label} independent HR step {step_index}",
        )
        z0_lr = sampler._pred_to_xstart(
            sample_lr, float(t_value), velocity_lr
        )
        z0_hr_matched = sampler._pred_to_xstart(
            sample_lr, float(t_value), velocity_hr_matched
        )
        z0_hr_independent = sampler._pred_to_xstart(
            sample_hr, float(t_value), velocity_hr_independent
        )
        z0_lr_features = _prediction_features(
            z0_lr,
            coords=coords,
            label=f"{label} LR z0 step {step_index}",
        )
        z0_hr_matched_features = _prediction_features(
            z0_hr_matched,
            coords=coords,
            label=f"{label} matched HR z0 step {step_index}",
        )
        z0_hr_independent_features = _prediction_features(
            z0_hr_independent,
            coords=coords,
            label=f"{label} independent HR z0 step {step_index}",
        )
        sigma_t = float(
            sampler.sigma_min
            + (1.0 - sampler.sigma_min) * float(t_value)
        )
        delta_formula = -sigma_t * (
            velocity_hr_matched_features - velocity_lr_features
        )
        delta_direct = z0_hr_matched_features - z0_lr_features
        delta_error = float(
            (delta_formula - delta_direct).abs().max().item()
        )
        if delta_error >= 1e-5:
            raise AssertionError(
                f"{label}: matched HR delta formula error {delta_error}"
            )
        index = subset_device
        per_step[int(step_index)] = {
            "t": float(t_value),
            "t_prev": float(t_prev),
            "sigma_t": sigma_t,
            "x_norm": sample_lr.feats.index_select(0, index)
            .detach()
            .float()
            .cpu(),
            "x_hr_independent_norm": sample_hr.feats.index_select(0, index)
            .detach()
            .float()
            .cpu(),
            "v_lr_norm": velocity_lr_features.index_select(0, index)
            .detach()
            .cpu(),
            "v_hr_norm": velocity_hr_matched_features.index_select(0, index)
            .detach()
            .cpu(),
            "v_hr_independent_norm": (
                velocity_hr_independent_features.index_select(0, index)
                .detach()
                .cpu()
            ),
            "z0_lr_norm": z0_lr_features.index_select(0, index)
            .detach()
            .cpu(),
            "z0_hr_norm": z0_hr_independent_features.index_select(0, index)
            .detach()
            .cpu(),
            "z0_hr_matched_norm": (
                z0_hr_matched_features.index_select(0, index)
                .detach()
                .cpu()
            ),
            "hr_minus_lr_z0_norm": delta_direct.index_select(0, index)
            .detach()
            .cpu(),
            "matched_delta_formula_max_error": delta_error,
        }
        delta_t = float(t_value - t_prev)
        sample_lr = sample_lr - delta_t * velocity_lr
        sample_hr = sample_hr - delta_t * velocity_hr_independent
    _sync_cuda()
    if not torch.equal(sample_lr.coords, coords) or not torch.equal(
        sample_hr.coords, coords
    ):
        raise RuntimeError(f"{label}: Euler trajectory changed coordinates")
    return FlowTrajectory(
        endpoint_lr_norm=sample_lr,
        endpoint_hr_norm=sample_hr,
        per_step=per_step,
        elapsed_seconds=float(time.perf_counter() - started),
        metadata={
            "trajectory": "independent LR and HR Euler trajectories",
            "matched_state": (
                "HR and LR velocities evaluated at the LR trajectory x_t; "
                "no subtraction of independently evolved states"
            ),
            "steps": steps,
            "rescale_t": rescale_t,
            "sigma_min": float(sampler.sigma_min),
            "matched_delta_formula": "-sigma(t)*(v_hr-v_lr)",
            "sampler_trajectory_modified": False,
        },
    )


def _condition(
    *,
    pipeline: Any,
    model: Any,
    image: Image.Image,
    coords: torch.Tensor,
    transform: core.TileCameraTransform,
) -> Mapping[str, Any]:
    return pipeline.get_proj_cond_shape(
        model,
        [image.convert("RGB")],
        coords,
        camera_angle_x=float(transform.camera_angle_x),
        distance=float(transform.distance),
        mesh_scale=float(transform.mesh_scale),
        grid_resolution_override=64,
    )


def _endpoint_stats(
    *,
    global_raw: SparseTensor,
    global_norm: SparseTensor,
    lr_norm: SparseTensor,
    hr_norm: SparseTensor,
    normalization: Mapping[str, Sequence[float]],
    space_assertions: Mapping[str, Any],
) -> Dict[str, Any]:
    for label, value in (
        ("global norm", global_norm),
        ("LR norm", lr_norm),
        ("HR norm", hr_norm),
    ):
        if not torch.equal(global_raw.coords, value.coords):
            raise RuntimeError(f"{label}: endpoint coordinates differ")
    lr_raw = _denormalize(lr_norm, normalization)
    hr_raw = _denormalize(hr_norm, normalization)
    values_norm = {
        "global": global_norm.feats,
        "lr": lr_norm.feats,
        "hr": hr_norm.feats,
    }
    values_raw = {
        "global": global_raw.feats,
        "lr": lr_raw.feats,
        "hr": hr_raw.feats,
    }
    output: Dict[str, Any] = {
        **dict(space_assertions),
        "normalized_flow_space": {},
        "decoder_raw_space": {},
    }
    for left, right in PAIR_NAMES:
        key = f"{left}_vs_{right}"
        output["normalized_flow_space"][key] = _comparison_stats(
            values_norm[left],
            values_norm[right],
            space=SPACE_NORM,
        )
        raw_stats = _comparison_stats(
            values_raw[left],
            values_raw[right],
            space=SPACE_RAW,
            include_covariance=False,
        )
        # Decoder/raw analysis is secondary; retain the requested MAE/RMSE
        # while avoiding misleading cross-latent scale comparisons.
        output["decoder_raw_space"][key] = {
            "comparison_space": SPACE_RAW,
            "tokens": raw_stats["tokens"],
            "channels": raw_stats["channels"],
            "mae": raw_stats["mae"],
            "rmse": raw_stats["rmse"],
            "relative_l2": raw_stats["relative_l2"],
            "token_cosine_mean": raw_stats["token_cosine_mean"],
            "token_cosine_std": raw_stats["token_cosine_std"],
            "centered_cosine": raw_stats["centered_cosine"],
            "pearson_correlation": raw_stats["pearson_correlation"],
        }
    return output


def _time_stats(
    global_norm: SparseTensor,
    trajectory: FlowTrajectory,
    subset_indices: torch.Tensor,
) -> Dict[str, Any]:
    global_subset = (
        global_norm.feats.detach().float().cpu().index_select(0, subset_indices)
    )
    output: Dict[str, Any] = {}
    for step_index, row in trajectory.per_step.items():
        gr = _comparison_stats(
            global_subset,
            row["z0_lr_norm"],
            space=SPACE_NORM,
            include_covariance=False,
        )
        gh = _comparison_stats(
            global_subset,
            row["z0_hr_norm"],
            space=SPACE_NORM,
            include_covariance=False,
        )
        output[str(step_index)] = {
            "t": float(row["t"]),
            "global_vs_lr": {
                "rmse": gr["rmse"],
                "relative_l2": gr["relative_l2"],
                "centered_cosine": gr["centered_cosine"],
                "token_cosine_mean": gr["token_cosine_mean"],
            },
            "global_vs_hr": {
                "rmse": gh["rmse"],
                "relative_l2": gh["relative_l2"],
                "centered_cosine": gh["centered_cosine"],
                "token_cosine_mean": gh["token_cosine_mean"],
            },
            "matched_hr_residual_rms": float(
                row["hr_minus_lr_z0_norm"].square().mean().sqrt().item()
            ),
        }
    return output


def _checkpoint_path(tile_dir: Path, seed: int, first_seed: int) -> Path:
    if int(seed) == int(first_seed):
        return tile_dir / "matched_latents.pt"
    return tile_dir / f"matched_latents_seed_{int(seed)}.pt"


def _save_seed_checkpoint(
    *,
    path: Path,
    tile_id: int,
    box: Sequence[int],
    seed: int,
    global_seed: int,
    coords: torch.Tensor,
    subset_indices: torch.Tensor,
    global_shape_raw: SparseTensor,
    global_shape_norm: SparseTensor,
    shape_trajectory: FlowTrajectory,
    global_texture_raw: SparseTensor,
    global_texture_norm: SparseTensor,
    texture_trajectory: FlowTrajectory,
    shape_normalization: Mapping[str, Sequence[float]],
    texture_normalization: Mapping[str, Sequence[float]],
    shape_stats: Mapping[str, Any],
    texture_stats: Mapping[str, Any],
) -> None:
    payload: Dict[str, Any] = {
        "format": FORMAT_VERSION,
        "tile_id": int(tile_id),
        "box": [int(value) for value in box],
        "seed": int(seed),
        "global_pipeline_seed": int(global_seed),
        "coords": coords.detach().cpu(),
        "per_step_token_indices": subset_indices.detach().cpu(),
        "per_step_token_sampling": (
            "deterministic rounded linspace over exact endpoint token order"
        ),
        "spaces": {
            "encoder_space": "decoder_raw",
            "flow_space": "normalized",
            "analysis_primary_space": SPACE_NORM,
            "decoder_space_analysis": True,
        },
        "shape": {
            "coords": coords.detach().cpu(),
            "global_encoded_raw": global_shape_raw.feats.detach().float().cpu(),
            "global_encoded_norm": global_shape_norm.feats.detach().float().cpu(),
            "local_lr_endpoint_norm": (
                shape_trajectory.endpoint_lr_norm.feats.detach().float().cpu()
            ),
            "local_lr_endpoint_raw": (
                _denormalize(
                    shape_trajectory.endpoint_lr_norm, shape_normalization
                )
                .feats.detach()
                .float()
                .cpu()
            ),
            "local_hr_endpoint_norm": (
                shape_trajectory.endpoint_hr_norm.feats.detach().float().cpu()
            ),
            "local_hr_endpoint_raw": (
                _denormalize(
                    shape_trajectory.endpoint_hr_norm, shape_normalization
                )
                .feats.detach()
                .float()
                .cpu()
            ),
            "per_step": shape_trajectory.per_step,
            "trajectory_metadata": shape_trajectory.metadata,
            "stats": dict(shape_stats),
        },
        "texture": {
            "coords": coords.detach().cpu(),
            "global_encoded_raw": global_texture_raw.feats.detach().float().cpu(),
            "global_encoded_norm": global_texture_norm.feats.detach().float().cpu(),
            "local_lr_endpoint_norm": (
                texture_trajectory.endpoint_lr_norm.feats.detach().float().cpu()
            ),
            "local_lr_endpoint_raw": (
                _denormalize(
                    texture_trajectory.endpoint_lr_norm, texture_normalization
                )
                .feats.detach()
                .float()
                .cpu()
            ),
            "local_hr_endpoint_norm": (
                texture_trajectory.endpoint_hr_norm.feats.detach().float().cpu()
            ),
            "local_hr_endpoint_raw": (
                _denormalize(
                    texture_trajectory.endpoint_hr_norm, texture_normalization
                )
                .feats.detach()
                .float()
                .cpu()
            ),
            "per_step": texture_trajectory.per_step,
            "trajectory_metadata": {
                **texture_trajectory.metadata,
                "concat_cond": (
                    "independent LR/HR texture trajectories use their respective "
                    "normalized shape endpoints; matched-state HR/LR image "
                    "response uses the LR normalized shape endpoint for both"
                ),
            },
            "stats": dict(texture_stats),
        },
    }
    _atomic_torch_save(path, payload)


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


def _prepare_global_source(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    output_dir: Path,
    device: torch.device,
) -> Tuple[
    Image.Image,
    Image.Image,
    Mapping[str, float],
    Any,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    canonical = pipeline.preprocess_canonical_images(Image.open(args.image))
    image_4096: Image.Image = canonical["image_4096"]
    image_1024: Image.Image = canonical["image_1024"]
    image_4096.save(output_dir / "canonical_4096.png")
    image_1024.save(output_dir / "canonical_1024.png")
    canonical["image_512"].save(output_dir / "canonical_512.png")
    _atomic_json(output_dir / "canonical_metadata.json", canonical["metadata"])
    global_camera_path = output_dir / "global_camera.json"
    if bool(args.resume) and global_camera_path.is_file():
        global_camera = json.loads(global_camera_path.read_text("utf-8"))
    else:
        global_camera = core._estimate_camera(
            image_1024=image_1024,
            output_dir=output_dir,
            manual_fov=float(args.fov),
            mesh_scale=float(args.mesh_scale),
            extend_pixel=int(args.extend_pixel),
            moge_model_path=args.moge_model_path,
        )
        _atomic_json(global_camera_path, global_camera)

    baseline_path = output_dir / "global_baseline_mesh.pt"
    if bool(args.resume) and baseline_path.is_file():
        baseline_mesh = torch.load(
            baseline_path, map_location="cpu", weights_only=False
        )
        print(f"[global] reused {baseline_path}")
    else:
        ss_params, shape_params, texture_params = _sampler_params(args)
        _seed_everything(int(args.global_seed))
        print(
            "[global] ordinary Pixal3D 1024_cascade, "
            f"fixed seed={int(args.global_seed)}"
        )
        output, latents = pipeline.run(
            image_1024,
            camera_params=global_camera,
            seed=int(args.global_seed),
            sparse_structure_sampler_params=ss_params,
            shape_slat_sampler_params=shape_params,
            tex_slat_sampler_params=texture_params,
            preprocess_image=False,
            return_latent=True,
            pipeline_type="1024_cascade",
            max_num_tokens=int(args.max_num_tokens),
        )
        if len(output) != 1:
            raise RuntimeError("global route did not return exactly one mesh")
        baseline_live = core._validate_mesh(
            output[0], "ordinary global Pixal3D 1024"
        )
        if int(latents[2]) != core.OVOXEL_RESOLUTION:
            raise RuntimeError("global decoder resolution is not 1024")
        baseline_mesh = baseline_live.to("cpu")
        _atomic_torch_save(
            baseline_path,
            {
                "format": "global_baseline_mesh_wrapper_v1",
                "mesh": baseline_mesh,
                "global_seed": int(args.global_seed),
            },
        )
        # Keep the actual mesh as the convenient runtime value.  The wrapper
        # makes the fixed global seed explicit on disk.
        del output, latents, baseline_live
        _empty_cuda_cache()
    if isinstance(baseline_mesh, Mapping) and "mesh" in baseline_mesh:
        baseline_mesh = baseline_mesh["mesh"]
    baseline_mesh = core._validate_mesh(baseline_mesh, "cached global mesh")
    global_ovoxel_object = core._ovoxel_indices_to_object(
        baseline_mesh.coords,
        origin=baseline_mesh.origin,
        voxel_size=baseline_mesh.voxel_size,
    )
    global_ovoxel_q = global_ovoxel_object * (
        2.0 * float(global_camera["mesh_scale"])
    )
    global_ovoxel_uv, _, global_ovoxel_finite = core._project_global_q_to_4096(
        global_ovoxel_q, global_camera=global_camera
    )
    global_face_uv, global_face_finite = core._project_face_centers(
        baseline_mesh.vertices,
        baseline_mesh.faces,
        mesh_scale=float(global_camera["mesh_scale"]),
        global_camera=global_camera,
        chunk_size=int(args.face_projection_chunk_size),
    )
    return (
        image_4096,
        image_1024,
        global_camera,
        baseline_mesh,
        global_ovoxel_q,
        global_ovoxel_uv,
        global_ovoxel_finite,
        global_face_uv,
        global_face_finite,
    )


def _collect_tile(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    output_dir: Path,
    device: torch.device,
    image_4096: Image.Image,
    image_1024: Image.Image,
    global_camera: Mapping[str, float],
    baseline_mesh: Any,
    global_ovoxel_q: torch.Tensor,
    global_ovoxel_uv: torch.Tensor,
    global_ovoxel_finite: torch.Tensor,
    global_face_uv: torch.Tensor,
    global_face_finite: torch.Tensor,
    shape_encoder: torch.nn.Module,
    pbr_encoder: torch.nn.Module,
    tile_id: int,
    box: Sequence[int],
    seeds: Sequence[int],
    shape_params: Mapping[str, Any],
    texture_params: Mapping[str, Any],
) -> Dict[str, Any]:
    tile_dir = output_dir / "per_tile" / f"tile_{tile_id:02d}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    projected_count = int(
        core._inside_tile(
            global_ovoxel_uv, global_ovoxel_finite, box
        ).sum().item()
    )
    base_record: Dict[str, Any] = {
        "tile_id": int(tile_id),
        "box": [int(value) for value in box],
        "row": int(tile_id // 7),
        "column": int(tile_id % 7),
        "position_group": _position_group(tile_id),
        "projected_global_ovoxels": projected_count,
    }
    if projected_count < int(args.min_tile_ovoxels):
        return {
            **base_record,
            "status": "skipped",
            "reason": "projected global O-Voxels below collection threshold",
        }
    first_seed = int(seeds[0])
    checkpoint_paths = {
        int(seed): _checkpoint_path(tile_dir, int(seed), first_seed)
        for seed in seeds
    }
    if bool(args.resume) and all(path.is_file() for path in checkpoint_paths.values()):
        print(f"[tile {tile_id:02d}] all seed checkpoints already exist")
        first = torch.load(
            checkpoint_paths[first_seed],
            map_location="cpu",
            weights_only=False,
        )
        return {
            **base_record,
            "status": "success",
            "common_tokens": int(first["coords"].shape[0]),
            "seeds": [int(seed) for seed in seeds],
            "checkpoints": {
                str(seed): str(path)
                for seed, path in checkpoint_paths.items()
            },
            "resumed": True,
        }

    started = time.perf_counter()
    transform = core._derive_tile_camera(
        tile_id=tile_id,
        box=box,
        global_camera=global_camera,
        extend_pixel=int(args.extend_pixel),
    )
    mapping = core._map_global_ovoxels_to_local(
        global_mesh=baseline_mesh,
        global_q=global_ovoxel_q,
        global_uv_4096=global_ovoxel_uv,
        finite_projection=global_ovoxel_finite,
        global_camera=global_camera,
        transform=transform,
    )
    local_vertices, local_faces, _, _, geometry_stats = (
        core._prepare_tile_geometry(
            global_vertices=baseline_mesh.vertices,
            global_faces=baseline_mesh.faces,
            global_face_uv=global_face_uv,
            global_face_finite=global_face_finite,
            global_camera=global_camera,
            transform=transform,
        )
    )
    global_shape_raw, shape_encoder_stats = core._encode_local_shape(
        encoder=shape_encoder,
        vertices=local_vertices,
        faces=local_faces,
        device=device,
        low_vram=bool(args.low_vram),
    )
    global_texture_raw, texture_encoder_stats = core._encode_local_pbr(
        encoder=pbr_encoder,
        coords=mapping.local_coords,
        attrs=mapping.local_attrs,
        device=device,
        low_vram=bool(args.low_vram),
    )
    global_shape_raw, global_texture_raw, alignment_stats = (
        core._align_latent_supports(global_shape_raw, global_texture_raw)
    )
    coords = global_shape_raw.coords.detach().clone()
    tokens = int(coords.shape[0])
    if tokens > int(args.max_num_tokens):
        raise RuntimeError(
            f"tile has {tokens:,} common tokens, exceeding "
            f"--max-num-tokens={int(args.max_num_tokens):,}"
        )
    if tokens < int(args.min_flow_tokens):
        return {
            **base_record,
            "status": "skipped",
            "common_tokens": tokens,
            "reason": "common C64 tokens below flow threshold",
            "alignment": alignment_stats,
        }
    if not torch.equal(global_shape_raw.coords, global_texture_raw.coords):
        raise AssertionError("aligned shape/texture coordinates differ")
    global_shape_norm = _normalize(
        global_shape_raw, pipeline.shape_slat_normalization
    )
    global_texture_norm = _normalize(
        global_texture_raw, pipeline.tex_slat_normalization
    )
    subset_indices = _deterministic_subset_indices(
        tokens, int(args.per_step_tokens)
    )

    tile_hr_image = image_4096.crop(tuple(box)).convert("RGB")
    tile_lr_image = anchor._make_lr_reference_tile(image_1024, box)
    tile_hr_image.save(tile_dir / "tile_reference_hr.png")
    tile_lr_image.save(tile_dir / "tile_reference_lr_from_global_1024.png")

    shape_hr_condition = _condition(
        pipeline=pipeline,
        model=pipeline.image_cond_model_shape_1024,
        image=tile_hr_image,
        coords=coords,
        transform=transform,
    )
    shape_lr_condition = _condition(
        pipeline=pipeline,
        model=pipeline.image_cond_model_shape_1024,
        image=tile_lr_image,
        coords=coords,
        transform=transform,
    )
    shape_model = pipeline.models["shape_slat_flow_model_1024"]
    if bool(args.low_vram):
        shape_model.to(device)
    shape_results: Dict[int, FlowTrajectory] = {}
    shape_seed_stats: Dict[str, Any] = {}
    for seed in seeds:
        seed = int(seed)
        if bool(args.resume) and checkpoint_paths[seed].is_file():
            continue
        noise_seed = seed + tile_id * 1000 + 201
        noise = _randn_sparse(
            coords,
            int(shape_model.in_channels),
            device=device,
            seed=noise_seed,
        )
        trajectory = _sample_independent_pair(
            sampler=pipeline.shape_slat_sampler,
            model=shape_model,
            noise=noise,
            lr_condition=shape_lr_condition,
            hr_condition=shape_hr_condition,
            params=shape_params,
            subset_indices=subset_indices,
            label=f"tile {tile_id:02d} seed {seed} shape",
        )
        shape_results[seed] = trajectory
        assertions = _space_assertions(
            encoded_raw=global_shape_raw,
            flow_norm=trajectory.endpoint_lr_norm,
            normalization=pipeline.shape_slat_normalization,
            label=f"tile {tile_id:02d} seed {seed} shape",
        )
        shape_seed_stats[str(seed)] = {
            **_endpoint_stats(
                global_raw=global_shape_raw,
                global_norm=global_shape_norm,
                lr_norm=trajectory.endpoint_lr_norm,
                hr_norm=trajectory.endpoint_hr_norm,
                normalization=pipeline.shape_slat_normalization,
                space_assertions=assertions,
            ),
            "noise_seed": noise_seed,
            "flow_seconds": trajectory.elapsed_seconds,
            "per_step": _time_stats(
                global_shape_norm, trajectory, subset_indices
            ),
        }
        del noise
        _empty_cuda_cache()
    if bool(args.low_vram):
        shape_model.cpu()
    del shape_hr_condition, shape_lr_condition
    _empty_cuda_cache()

    texture_hr_condition = _condition(
        pipeline=pipeline,
        model=pipeline.image_cond_model_tex_1024,
        image=tile_hr_image,
        coords=coords,
        transform=transform,
    )
    texture_lr_condition = _condition(
        pipeline=pipeline,
        model=pipeline.image_cond_model_tex_1024,
        image=tile_lr_image,
        coords=coords,
        transform=transform,
    )
    texture_model = pipeline.models["tex_slat_flow_model_1024"]
    texture_channels = int(texture_model.in_channels) - int(
        global_shape_norm.feats.shape[1]
    )
    if texture_channels != int(global_texture_norm.feats.shape[1]):
        raise RuntimeError(
            "texture flow channels do not match normalized texture encoder: "
            f"noise={texture_channels}, encoded={global_texture_norm.feats.shape[1]}"
        )
    if bool(args.low_vram):
        texture_model.to(device)
    texture_seed_stats: Dict[str, Any] = {}
    for seed in seeds:
        seed = int(seed)
        if bool(args.resume) and checkpoint_paths[seed].is_file():
            continue
        shape_trajectory = shape_results[seed]
        # This explicit assertion prevents decoder/raw shape latents from
        # accidentally entering the texture concat condition.
        shape_lr_concat_norm = shape_trajectory.endpoint_lr_norm
        shape_hr_concat_norm = shape_trajectory.endpoint_hr_norm
        assert torch.equal(shape_lr_concat_norm.coords, coords)
        assert torch.equal(shape_hr_concat_norm.coords, coords)
        noise_seed = seed + tile_id * 1000 + 301
        noise = _randn_sparse(
            coords,
            texture_channels,
            device=device,
            seed=noise_seed,
        )
        trajectory = _sample_independent_pair(
            sampler=pipeline.tex_slat_sampler,
            model=texture_model,
            noise=noise,
            lr_condition=texture_lr_condition,
            hr_condition=texture_hr_condition,
            params=texture_params,
            subset_indices=subset_indices,
            label=f"tile {tile_id:02d} seed {seed} texture",
            concat_cond_lr=shape_lr_concat_norm,
            concat_cond_hr=shape_hr_concat_norm,
            matched_concat_cond=shape_lr_concat_norm,
        )
        assertions = _space_assertions(
            encoded_raw=global_texture_raw,
            flow_norm=trajectory.endpoint_lr_norm,
            normalization=pipeline.tex_slat_normalization,
            label=f"tile {tile_id:02d} seed {seed} texture",
        )
        texture_stats = {
            **_endpoint_stats(
                global_raw=global_texture_raw,
                global_norm=global_texture_norm,
                lr_norm=trajectory.endpoint_lr_norm,
                hr_norm=trajectory.endpoint_hr_norm,
                normalization=pipeline.tex_slat_normalization,
                space_assertions=assertions,
            ),
            "noise_seed": noise_seed,
            "flow_seconds": trajectory.elapsed_seconds,
            "concat_cond_space": SPACE_NORM,
            "concat_cond_is_denormalized": False,
            "per_step": _time_stats(
                global_texture_norm, trajectory, subset_indices
            ),
        }
        texture_seed_stats[str(seed)] = texture_stats
        _save_seed_checkpoint(
            path=checkpoint_paths[seed],
            tile_id=tile_id,
            box=box,
            seed=seed,
            global_seed=int(args.global_seed),
            coords=coords,
            subset_indices=subset_indices,
            global_shape_raw=global_shape_raw,
            global_shape_norm=global_shape_norm,
            shape_trajectory=shape_trajectory,
            global_texture_raw=global_texture_raw,
            global_texture_norm=global_texture_norm,
            texture_trajectory=trajectory,
            shape_normalization=pipeline.shape_slat_normalization,
            texture_normalization=pipeline.tex_slat_normalization,
            shape_stats=shape_seed_stats[str(seed)],
            texture_stats=texture_stats,
        )
        print(
            f"[tile {tile_id:02d} seed {seed}] saved "
            f"tokens={tokens:,} -> {checkpoint_paths[seed].name}"
        )
        del noise, trajectory
        _empty_cuda_cache()
    if bool(args.low_vram):
        texture_model.cpu()
    del texture_hr_condition, texture_lr_condition
    _empty_cuda_cache()

    # When resuming a partially collected tile, populate summaries directly
    # from all now-complete checkpoints.
    for seed, path in checkpoint_paths.items():
        if str(seed) in shape_seed_stats and str(seed) in texture_seed_stats:
            continue
        saved = torch.load(path, map_location="cpu", weights_only=False)
        shape_seed_stats[str(seed)] = saved["shape"]["stats"]
        texture_seed_stats[str(seed)] = saved["texture"]["stats"]
    common_metadata = {
        **base_record,
        "status": "success",
        "common_tokens": tokens,
        "primary_analysis_included": tokens >= int(args.min_analysis_tokens),
        "seeds": [int(seed) for seed in seeds],
        "global_pipeline_seed": int(args.global_seed),
        "alignment": alignment_stats,
        "geometry_encoder_input": geometry_stats,
        "shape_encoder": shape_encoder_stats,
        "texture_encoder": texture_encoder_stats,
        "per_step_token_count": int(subset_indices.shape[0]),
        "per_step_token_indices": subset_indices.tolist(),
        "checkpoints": {
            str(seed): str(path) for seed, path in checkpoint_paths.items()
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _atomic_json(
        tile_dir / "shape_stats.json",
        {**common_metadata, "latent": "shape", "by_seed": shape_seed_stats},
    )
    _atomic_json(
        tile_dir / "texture_stats.json",
        {**common_metadata, "latent": "texture", "by_seed": texture_seed_stats},
    )
    del (
        mapping,
        global_shape_raw,
        global_texture_raw,
        global_shape_norm,
        global_texture_norm,
        shape_results,
    )
    _empty_cuda_cache()
    return common_metadata


def collect(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for collection")
    torch.cuda.set_device(int(args.cuda_device))
    device = torch.device("cuda")
    print(
        f"[cuda] physical index={int(args.cuda_device)} "
        f"current={torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())}"
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = init_pipeline(
        args.model_path,
        device="cuda",
        low_vram=bool(args.low_vram),
    )
    (
        image_4096,
        image_1024,
        global_camera,
        baseline_mesh,
        global_ovoxel_q,
        global_ovoxel_uv,
        global_ovoxel_finite,
        global_face_uv,
        global_face_finite,
    ) = _prepare_global_source(
        args=args,
        pipeline=pipeline,
        output_dir=output_dir,
        device=device,
    )
    _, shape_params, texture_params = _sampler_params(args)
    shape_encoder = pixal3d_models.from_pretrained(
        str(Path(args.shape_encoder).expanduser())
    ).eval()
    pbr_encoder = pixal3d_models.from_pretrained(
        str(Path(args.pbr_encoder).expanduser())
    ).eval()
    if not bool(args.low_vram):
        shape_encoder.to(device)
        pbr_encoder.to(device)

    boxes = core._tile_layout()
    requested = core._parse_tile_ids(args.tile_ids)
    if requested is not None:
        invalid = sorted(item for item in requested if item not in range(49))
        if invalid:
            raise ValueError(f"invalid tile ids {invalid}; valid range is 0..48")
    seeds = _parse_int_csv(args.seeds)
    records: List[Dict[str, Any]] = []
    attempted = 0
    for tile_id, box in enumerate(boxes):
        if requested is not None and tile_id not in requested:
            continue
        if args.max_tiles is not None and attempted >= int(args.max_tiles):
            break
        attempted += 1
        try:
            record = _collect_tile(
                args=args,
                pipeline=pipeline,
                output_dir=output_dir,
                device=device,
                image_4096=image_4096,
                image_1024=image_1024,
                global_camera=global_camera,
                baseline_mesh=baseline_mesh,
                global_ovoxel_q=global_ovoxel_q,
                global_ovoxel_uv=global_ovoxel_uv,
                global_ovoxel_finite=global_ovoxel_finite,
                global_face_uv=global_face_uv,
                global_face_finite=global_face_finite,
                shape_encoder=shape_encoder,
                pbr_encoder=pbr_encoder,
                tile_id=tile_id,
                box=box,
                seeds=seeds,
                shape_params=shape_params,
                texture_params=texture_params,
            )
        except Exception as error:
            record = {
                "tile_id": int(tile_id),
                "box": [int(value) for value in box],
                "row": int(tile_id // 7),
                "column": int(tile_id % 7),
                "position_group": _position_group(tile_id),
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
            }
            print(f"[tile {tile_id:02d}] FAILED: {record['reason']}")
        records.append(record)
        _atomic_json(
            output_dir / "collection_manifest.json",
            {
                "format": FORMAT_VERSION,
                "image": str(Path(args.image).expanduser().resolve()),
                "cuda_device": int(args.cuda_device),
                "global_pipeline_seed": int(args.global_seed),
                "local_flow_seeds": seeds,
                "tiles": records,
            },
        )
    del shape_encoder, pbr_encoder, pipeline
    _empty_cuda_cache()
    manifest = {
        "format": FORMAT_VERSION,
        "image": str(Path(args.image).expanduser().resolve()),
        "cuda_device": int(args.cuda_device),
        "global_pipeline_seed": int(args.global_seed),
        "local_flow_seeds": seeds,
        "global_camera": global_camera,
        "encoder_space": "decoder_raw",
        "flow_space": "normalized",
        "analysis_primary_space": SPACE_NORM,
        "decoder_space_analysis": True,
        "min_analysis_tokens": int(args.min_analysis_tokens),
        "successful_tiles": sum(row["status"] == "success" for row in records),
        "skipped_tiles": sum(row["status"] == "skipped" for row in records),
        "failed_tiles": sum(row["status"] == "failed" for row in records),
        "tiles": records,
    }
    _atomic_json(output_dir / "collection_manifest.json", manifest)
    return manifest


def _load_datasets(
    output_dir: Path,
    latent_name: str,
    *,
    min_tokens: int,
    seed_filter: Optional[int] = None,
) -> Tuple[List[TileDataset], List[Dict[str, Any]]]:
    if latent_name not in LATENT_NAMES:
        raise ValueError(f"unknown latent {latent_name}")
    manifest_path = output_dir / "collection_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing collection manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    datasets: List[TileDataset] = []
    excluded: List[Dict[str, Any]] = []
    for tile in manifest["tiles"]:
        if tile.get("status") != "success":
            continue
        for seed_text, path_text in tile["checkpoints"].items():
            seed = int(seed_text)
            if seed_filter is not None and seed != int(seed_filter):
                continue
            path = Path(path_text)
            if not path.is_absolute():
                path = output_dir / path
            saved = torch.load(path, map_location="cpu", weights_only=False)
            latent = saved[latent_name]
            tokens = int(saved["coords"].shape[0])
            if tokens < int(min_tokens):
                excluded.append(
                    {
                        "tile_id": int(tile["tile_id"]),
                        "seed": seed,
                        "tokens": tokens,
                        "reason": (
                            f"tokens below primary threshold {int(min_tokens)}"
                        ),
                        "path": str(path),
                    }
                )
                continue
            global_norm = (
                latent["global_encoded_norm"].float().numpy().astype(
                    np.float64, copy=False
                )
            )
            lr_norm = (
                latent["local_lr_endpoint_norm"].float().numpy().astype(
                    np.float64, copy=False
                )
            )
            hr_norm = (
                latent["local_hr_endpoint_norm"].float().numpy().astype(
                    np.float64, copy=False
                )
            )
            if (
                global_norm.shape != lr_norm.shape
                or global_norm.shape != hr_norm.shape
                or global_norm.ndim != 2
            ):
                raise RuntimeError(
                    f"{path}: endpoint latent shapes are not identical [N,C]"
                )
            datasets.append(
                TileDataset(
                    tile_id=int(tile["tile_id"]),
                    seed=seed,
                    row=int(tile["row"]),
                    column=int(tile["column"]),
                    position_group=str(tile["position_group"]),
                    tokens=tokens,
                    global_norm=global_norm,
                    lr_norm=lr_norm,
                    hr_norm=hr_norm,
                    path=path,
                )
            )
    return datasets, excluded


def _sufficient(
    datasets: Sequence[TileDataset],
    *,
    target: str = "lr",
) -> Dict[str, Any]:
    if not datasets:
        raise ValueError("cannot compute sufficient statistics for no tiles")
    channels = int(datasets[0].global_norm.shape[1])
    total = {
        "n": 0,
        "sum_x": np.zeros(channels, dtype=np.float64),
        "sum_y": np.zeros(channels, dtype=np.float64),
        "xtx": np.zeros((channels, channels), dtype=np.float64),
        "xty": np.zeros((channels, channels), dtype=np.float64),
        "yty": np.zeros((channels, channels), dtype=np.float64),
    }
    for dataset in datasets:
        cache_name = f"_sufficient_{target}"
        item = getattr(dataset, cache_name, None)
        if item is None:
            x_value = dataset.global_norm
            y_value = dataset.lr_norm if target == "lr" else dataset.hr_norm
            if x_value.shape[1] != channels or y_value.shape != x_value.shape:
                raise ValueError("dataset channel/shape mismatch")
            item = {
                "n": int(x_value.shape[0]),
                "sum_x": x_value.sum(axis=0),
                "sum_y": y_value.sum(axis=0),
                "xtx": x_value.T @ x_value,
                "xty": x_value.T @ y_value,
                "yty": y_value.T @ y_value,
            }
            setattr(dataset, cache_name, item)
        for key in ("sum_x", "sum_y", "xtx", "xty", "yty"):
            total[key] += item[key]
        total["n"] += int(item["n"])
    return total


def _centered_moments(
    sufficient: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = int(sufficient["n"])
    if count <= 0:
        raise ValueError("sufficient statistics have no samples")
    mean_x = sufficient["sum_x"] / count
    mean_y = sufficient["sum_y"] / count
    covariance_x = (
        sufficient["xtx"] - count * np.outer(mean_x, mean_x)
    ) / max(1, count - 1)
    covariance_y = (
        sufficient["yty"] - count * np.outer(mean_y, mean_y)
    ) / max(1, count - 1)
    covariance_xy = (
        sufficient["xty"] - count * np.outer(mean_x, mean_y)
    ) / max(1, count - 1)
    return mean_x, mean_y, covariance_x, covariance_y, covariance_xy


def _fit_full_ridge(
    train: Sequence[TileDataset],
    ridge_lambda: float,
) -> Dict[str, Any]:
    sufficient = _sufficient(train)
    return _fit_full_ridge_sufficient(sufficient, ridge_lambda)


def _fit_full_ridge_sufficient(
    sufficient: Mapping[str, Any],
    ridge_lambda: float,
) -> Dict[str, Any]:
    mean_x, mean_y, covariance_x, _, covariance_xy = _centered_moments(
        sufficient
    )
    scale = max(1, int(sufficient["n"]) - 1)
    regularizer = float(ridge_lambda) / scale
    matrix = covariance_x + regularizer * np.eye(covariance_x.shape[0])
    try:
        mapping = np.linalg.solve(matrix, covariance_xy)
    except np.linalg.LinAlgError:
        mapping = np.linalg.pinv(matrix, rcond=1e-12) @ covariance_xy
    intercept = mean_y - mean_x @ mapping
    return {
        "name": "full_ridge",
        "A": mapping,
        "b": intercept,
        "lambda": float(ridge_lambda),
        "train_mean_y": mean_y,
        "train_tokens": int(sufficient["n"]),
    }


def _fit_models(
    train: Sequence[TileDataset],
    ridge_lambda: float,
) -> Dict[str, Dict[str, Any]]:
    sufficient = _sufficient(train)
    mean_x, mean_y, covariance_x, covariance_y, covariance_xy = (
        _centered_moments(sufficient)
    )
    channels = int(mean_x.shape[0])
    identity = np.eye(channels, dtype=np.float64)
    variance_x = np.diag(covariance_x)
    variance_y = np.diag(covariance_y)
    covariance_diagonal = np.diag(covariance_xy)
    affine_scale = np.divide(
        covariance_diagonal,
        variance_x,
        out=np.zeros_like(variance_x),
        where=variance_x > 1e-12,
    )
    whitening_scale = np.sqrt(
        np.divide(
            variance_y,
            variance_x,
            out=np.zeros_like(variance_x),
            where=variance_x > 1e-12,
        )
    )
    return {
        "identity": {
            "name": "identity",
            "A": identity,
            "b": np.zeros(channels),
            "train_mean_y": mean_y,
        },
        "mean_only": {
            "name": "mean_only",
            "A": np.zeros((channels, channels)),
            "b": mean_y,
            "train_mean_y": mean_y,
        },
        "per_channel_affine": {
            "name": "per_channel_affine",
            "A": np.diag(affine_scale),
            "b": mean_y - mean_x * affine_scale,
            "train_mean_y": mean_y,
            "per_channel_scale": affine_scale,
        },
        "diagonal_whitening_coloring": {
            "name": "diagonal_whitening_coloring",
            "A": np.diag(whitening_scale),
            "b": mean_y - mean_x * whitening_scale,
            "train_mean_y": mean_y,
            "per_channel_scale": whitening_scale,
        },
        "full_ridge": _fit_full_ridge(train, ridge_lambda),
    }


def _predict(model: Mapping[str, Any], x_value: np.ndarray) -> np.ndarray:
    return x_value @ model["A"] + model["b"]


def _evaluation_metrics(
    datasets: Sequence[TileDataset],
    model: Mapping[str, Any],
    *,
    target: str = "lr",
) -> Dict[str, Any]:
    if not datasets:
        raise ValueError("cannot evaluate no datasets")
    total = _evaluation_from_sufficient(_sufficient(datasets, target=target), model)
    per_tile: List[Dict[str, Any]] = []
    for dataset in datasets:
        tile_metrics = _evaluation_from_sufficient(
            _sufficient([dataset], target=target), model
        )
        per_tile.append(
            {
                "tile_id": int(dataset.tile_id),
                "seed": int(dataset.seed),
                "tokens": int(dataset.tokens),
                "r2": tile_metrics["r2"],
                "rmse": tile_metrics["rmse"],
                "relative_l2": tile_metrics["relative_l2"],
                "centered_cosine": tile_metrics["centered_cosine"],
            }
        )
    return {**total, "per_tile": per_tile}


def _evaluation_from_sufficient(
    sufficient: Mapping[str, Any],
    model: Mapping[str, Any],
) -> Dict[str, Any]:
    count = int(sufficient["n"])
    mapping = np.asarray(model["A"], dtype=np.float64)
    intercept = np.asarray(model["b"], dtype=np.float64)
    train_mean_y = np.asarray(model["train_mean_y"], dtype=np.float64)
    sum_x = sufficient["sum_x"]
    sum_y = sufficient["sum_y"]
    xtx = sufficient["xtx"]
    xty = sufficient["xty"]
    yty = sufficient["yty"]
    mapped_sum_x = mapping.T @ sum_x
    predicted_sum = sum_x @ mapping + count * intercept
    predicted_cross = (
        mapping.T @ xty + np.outer(intercept, sum_y)
    )
    predicted_square = (
        mapping.T @ xtx @ mapping
        + np.outer(mapped_sum_x, intercept)
        + np.outer(intercept, mapped_sum_x)
        + count * np.outer(intercept, intercept)
    )
    error_square = (
        predicted_square - predicted_cross - predicted_cross.T + yty
    )
    per_channel_error = np.maximum(np.diag(error_square), 0.0)
    squared_error = float(per_channel_error.sum())
    target_norm_squared = float(np.trace(yty))
    baseline_square = (
        yty
        - np.outer(sum_y, train_mean_y)
        - np.outer(train_mean_y, sum_y)
        + count * np.outer(train_mean_y, train_mean_y)
    )
    per_channel_denominator = np.maximum(np.diag(baseline_square), 0.0)
    baseline_denominator = float(per_channel_denominator.sum())
    prediction_mean = predicted_sum / count
    target_mean = sum_y / count
    centered_prediction_square = float(
        np.trace(
            predicted_square - count * np.outer(prediction_mean, prediction_mean)
        )
    )
    centered_target_square = float(
        np.trace(yty - count * np.outer(target_mean, target_mean))
    )
    centered_cross = float(
        np.trace(
            predicted_cross - count * np.outer(prediction_mean, target_mean)
        )
    )
    channels = int(mapping.shape[0])
    return {
        "tokens": count,
        "channels": channels,
        "rmse": math.sqrt(max(0.0, squared_error) / (count * channels)),
        "relative_l2": math.sqrt(
            max(0.0, squared_error) / max(1e-12, target_norm_squared)
        ),
        "r2": (
            1.0 - squared_error / baseline_denominator
            if baseline_denominator > 1e-12
            else None
        ),
        "centered_cosine": centered_cross
        / max(
            1e-12,
            math.sqrt(
                max(0.0, centered_prediction_square)
                * max(0.0, centered_target_square)
            ),
        ),
        "per_channel_r2": [
            (
                1.0 - float(error) / float(denominator)
                if denominator > 1e-12
                else None
            )
            for error, denominator in zip(
                per_channel_error, per_channel_denominator
            )
        ],
    }


def _select_lambda(train: Sequence[TileDataset]) -> Dict[str, Any]:
    tile_ids = sorted({dataset.tile_id for dataset in train})
    scores: Dict[str, Optional[float]] = {}
    if len(tile_ids) < 2:
        selected = 1e-3
        return {
            "selected": selected,
            "inner_split": "unavailable: fewer than two training tiles",
            "validation_rmse_by_lambda": {
                str(value): None for value in RIDGE_LAMBDAS
            },
        }
    for ridge_lambda in RIDGE_LAMBDAS:
        squared_error = 0.0
        feature_values = 0
        for held_tile in tile_ids:
            inner_train = [
                item for item in train if item.tile_id != held_tile
            ]
            inner_test = [
                item for item in train if item.tile_id == held_tile
            ]
            if not inner_train or not inner_test:
                continue
            model = _fit_full_ridge_sufficient(
                _sufficient(inner_train), ridge_lambda
            )
            validation = _evaluation_from_sufficient(
                _sufficient(inner_test), model
            )
            squared_error += (
                float(validation["rmse"]) ** 2
                * int(validation["tokens"])
                * int(validation["channels"])
            )
            feature_values += int(validation["tokens"]) * int(
                validation["channels"]
            )
        scores[str(ridge_lambda)] = (
            math.sqrt(squared_error / feature_values)
            if feature_values > 0
            else None
        )
    finite = [
        (float(key), value)
        for key, value in scores.items()
        if value is not None and math.isfinite(float(value))
    ]
    selected = min(finite, key=lambda item: (item[1], item[0]))[0]
    return {
        "selected": selected,
        "inner_split": "leave-one-complete-training-tile-out",
        "validation_rmse_by_lambda": scores,
    }


def _matrix_diagnostics(mapping: np.ndarray) -> Dict[str, Any]:
    channels = int(mapping.shape[0])
    identity = np.eye(channels)
    return {
        "singular_values": np.linalg.svd(mapping, compute_uv=False),
        "relative_distance_to_identity": float(
            np.linalg.norm(mapping - identity) / np.linalg.norm(identity)
        ),
        "matrix": mapping,
    }


def _psd_matrix_power(
    covariance: np.ndarray,
    power: float,
    regularization: float,
) -> np.ndarray:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    floor = max(float(regularization), 1e-12)
    adjusted = np.maximum(eigenvalues, floor) ** float(power)
    return (eigenvectors * adjusted[None]) @ eigenvectors.T


def _fit_procrustes(
    train: Sequence[TileDataset],
    regularization: float,
) -> Dict[str, Any]:
    sufficient = _sufficient(train)
    mean_x, mean_y, covariance_x, covariance_y, covariance_xy = (
        _centered_moments(sufficient)
    )
    x_inverse_sqrt = _psd_matrix_power(
        covariance_x, -0.5, regularization
    )
    y_inverse_sqrt = _psd_matrix_power(
        covariance_y, -0.5, regularization
    )
    y_sqrt = _psd_matrix_power(covariance_y, 0.5, regularization)
    whitened_cross = x_inverse_sqrt @ covariance_xy @ y_inverse_sqrt
    left, singular_values, right_transpose = np.linalg.svd(
        whitened_cross, full_matrices=False
    )
    orthogonal = left @ right_transpose
    mapping = x_inverse_sqrt @ orthogonal @ y_sqrt
    intercept = mean_y - mean_x @ mapping
    return {
        "name": "orthogonal_procrustes",
        "A": mapping,
        "b": intercept,
        "Q": orthogonal,
        "whitened_cross_singular_values": singular_values,
        "train_mean_y": mean_y,
        "regularization": float(regularization),
    }


def _fit_cca(
    train: Sequence[TileDataset],
    regularization: float,
) -> Dict[str, Any]:
    sufficient = _sufficient(train)
    mean_x, mean_y, covariance_x, covariance_y, covariance_xy = (
        _centered_moments(sufficient)
    )
    x_inverse_sqrt = _psd_matrix_power(
        covariance_x, -0.5, regularization
    )
    y_inverse_sqrt = _psd_matrix_power(
        covariance_y, -0.5, regularization
    )
    whitened_cross = x_inverse_sqrt @ covariance_xy @ y_inverse_sqrt
    left, correlations, right_transpose = np.linalg.svd(
        whitened_cross, full_matrices=False
    )
    right = right_transpose.T
    correlations = np.clip(correlations, 0.0, 1.0)
    weights_x = x_inverse_sqrt @ left
    weights_y = y_inverse_sqrt @ right
    squared = np.square(correlations)
    total = max(1e-12, float(squared.sum()))
    cumulative = {
        str(count): float(squared[: min(count, len(squared))].sum() / total)
        for count in MODE_COUNTS
    }
    return {
        "mean_x": mean_x,
        "mean_y": mean_y,
        "weights_x": weights_x,
        "weights_y": weights_y,
        "U": left,
        "V": right,
        "canonical_correlations": correlations,
        "cumulative_explained_rho_squared": cumulative,
        "regularization": float(regularization),
        "train_tokens": int(sufficient["n"]),
    }


def _canonical_correlations_on_test(
    model: Mapping[str, Any],
    test: Sequence[TileDataset],
) -> np.ndarray:
    sufficient = _sufficient(test)
    _, _, covariance_x, covariance_y, covariance_xy = _centered_moments(
        sufficient
    )
    weights_x = model["weights_x"]
    weights_y = model["weights_y"]
    covariance = np.diag(weights_x.T @ covariance_xy @ weights_y)
    variance_x = np.diag(weights_x.T @ covariance_x @ weights_x)
    variance_y = np.diag(weights_y.T @ covariance_y @ weights_y)
    return np.divide(
        covariance,
        np.sqrt(np.maximum(variance_x * variance_y, 1e-24)),
        out=np.zeros_like(covariance),
        where=(variance_x > 1e-12) & (variance_y > 1e-12),
    )


def _shared_energy(
    model: Mapping[str, Any],
    test: Sequence[TileDataset],
) -> Dict[str, float]:
    # The residual lives in the original local latent coordinates.  CCA's
    # right singular vectors ``V`` live in whitened coordinates, so projecting
    # the residual directly with V mixes two incompatible bases.  Project with
    # W_R = C_RR^{-1/2} V (stored as weights_y) after centering instead.
    weights_y = np.asarray(model["weights_y"], dtype=np.float64)
    delta_count = 0
    delta_sum = np.zeros(weights_y.shape[0], dtype=np.float64)
    delta_second_moment = np.zeros(
        (weights_y.shape[0], weights_y.shape[0]), dtype=np.float64
    )
    for dataset in test:
        delta = dataset.hr_norm - dataset.lr_norm
        delta_count += int(delta.shape[0])
        delta_sum += delta.sum(axis=0)
        delta_second_moment += delta.T @ delta
    if delta_count <= 0:
        return {str(count): 0.0 for count in MODE_COUNTS}
    delta_mean = delta_sum / float(delta_count)
    centered_moment = (
        delta_second_moment
        - float(delta_count) * np.outer(delta_mean, delta_mean)
    )
    canonical_moment = weights_y.T @ centered_moment @ weights_y
    total_energy = max(0.0, float(np.trace(canonical_moment)))
    result: Dict[str, float] = {}
    for count in MODE_COUNTS:
        modes = min(count, canonical_moment.shape[0])
        shared_energy = max(
            0.0, float(np.trace(canonical_moment[:modes, :modes]))
        )
        result[str(count)] = (
            shared_energy / total_energy if total_energy > 1e-12 else 0.0
        )
    return result


def _split_definitions(
    datasets: Sequence[TileDataset],
) -> List[Dict[str, Any]]:
    tile_ids = sorted({dataset.tile_id for dataset in datasets})
    definitions: List[Dict[str, Any]] = []
    for held_tile in tile_ids:
        definitions.append(
            {
                "family": "leave_one_tile_out",
                "name": f"tile_{held_tile:02d}",
                "train_tile_ids": [
                    tile_id for tile_id in tile_ids if tile_id != held_tile
                ],
                "test_tile_ids": [held_tile],
            }
        )
    for column in range(7):
        test_ids = [
            tile_id for tile_id in tile_ids if tile_id % 7 == column
        ]
        train_ids = [
            tile_id for tile_id in tile_ids if tile_id % 7 != column
        ]
        definitions.append(
            {
                "family": "seven_fold_spatial_columns",
                "name": f"column_{column}",
                "train_tile_ids": train_ids,
                "test_tile_ids": test_ids,
            }
        )
    central_ids = [
        tile_id for tile_id in tile_ids if _position_group(tile_id) == "central"
    ]
    edge_ids = [
        tile_id for tile_id in tile_ids if _position_group(tile_id) == "edge"
    ]
    definitions.extend(
        [
            {
                "family": "central_edge",
                "name": "central_train_edge_test",
                "train_tile_ids": central_ids,
                "test_tile_ids": edge_ids,
            },
            {
                "family": "central_edge",
                "name": "edge_train_central_test",
                "train_tile_ids": edge_ids,
                "test_tile_ids": central_ids,
            },
        ]
    )
    return definitions


def _confidence_interval(values: Sequence[float]) -> Dict[str, Any]:
    finite = np.asarray(
        [value for value in values if value is not None and math.isfinite(value)],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"count": 0, "mean": None, "ci95": [None, None]}
    mean = float(finite.mean())
    if finite.size < 2:
        return {"count": int(finite.size), "mean": mean, "ci95": [None, None]}
    standard_error = float(scipy_stats.sem(finite))
    low, high = scipy_stats.t.interval(
        0.95, int(finite.size - 1), loc=mean, scale=standard_error
    )
    return {
        "count": int(finite.size),
        "mean": mean,
        "std": float(finite.std(ddof=1)),
        "ci95": [float(low), float(high)],
    }


def _run_seed_cross_validation(
    datasets: Sequence[TileDataset],
    *,
    regularization: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if not datasets:
        raise ValueError("no datasets for cross validation")
    seed_values = {item.seed for item in datasets}
    if len(seed_values) != 1:
        raise ValueError("seed cross validation requires exactly one seed")
    seed = int(next(iter(seed_values)))
    ridge_folds: List[Dict[str, Any]] = []
    procrustes_folds: List[Dict[str, Any]] = []
    cca_folds: List[Dict[str, Any]] = []
    for split in _split_definitions(datasets):
        train_ids = set(split["train_tile_ids"])
        test_ids = set(split["test_tile_ids"])
        train = [item for item in datasets if item.tile_id in train_ids]
        test = [item for item in datasets if item.tile_id in test_ids]
        split_record = {
            **split,
            "seed": seed,
            "train_tokens": sum(item.tokens for item in train),
            "test_tokens": sum(item.tokens for item in test),
            "tile_level_split": True,
            "token_random_split": False,
        }
        if not train or not test:
            skipped = {
                **split_record,
                "status": "skipped",
                "reason": "empty train or test set after token filtering",
            }
            ridge_folds.append(skipped)
            procrustes_folds.append(skipped)
            cca_folds.append(skipped)
            continue
        lambda_selection = _select_lambda(train)
        models = _fit_models(train, float(lambda_selection["selected"]))
        model_results: Dict[str, Any] = {}
        for name, model in models.items():
            diagnostics = _matrix_diagnostics(model["A"])
            model_results[name] = {
                "train": _evaluation_metrics(train, model),
                "held_out": _evaluation_metrics(test, model),
                "mapping": diagnostics,
            }
        ridge_folds.append(
            {
                **split_record,
                "status": "success",
                "ridge_lambda_selection": lambda_selection,
                "models": model_results,
            }
        )
        procrustes_model = _fit_procrustes(train, regularization)
        procrustes_folds.append(
            {
                **split_record,
                "status": "success",
                "train": _evaluation_metrics(train, procrustes_model),
                "held_out": _evaluation_metrics(test, procrustes_model),
                "mapping": _matrix_diagnostics(procrustes_model["A"]),
                "orthogonality_max_error": float(
                    np.abs(
                        procrustes_model["Q"].T @ procrustes_model["Q"]
                        - np.eye(procrustes_model["Q"].shape[0])
                    ).max()
                ),
                "whitened_cross_singular_values": procrustes_model[
                    "whitened_cross_singular_values"
                ],
            }
        )
        cca_model = _fit_cca(train, regularization)
        heldout_correlations = _canonical_correlations_on_test(cca_model, test)
        cca_folds.append(
            {
                **split_record,
                "status": "success",
                "train_canonical_correlations": cca_model[
                    "canonical_correlations"
                ],
                "held_out_canonical_correlations": heldout_correlations,
                "cumulative_explained_rho_squared": cca_model[
                    "cumulative_explained_rho_squared"
                ],
                "held_out_hr_residual_shared_energy_ratio": _shared_energy(
                    cca_model, test
                ),
            }
        )
    ridge_summary: Dict[str, Any] = {}
    for family in (
        "leave_one_tile_out",
        "seven_fold_spatial_columns",
        "central_edge",
    ):
        family_folds = [
            row
            for row in ridge_folds
            if row["family"] == family and row.get("status") == "success"
        ]
        family_summary: Dict[str, Any] = {}
        for model_name in (
            "identity",
            "mean_only",
            "per_channel_affine",
            "diagonal_whitening_coloring",
            "full_ridge",
        ):
            family_summary[model_name] = {
                "held_out_r2": _confidence_interval(
                    [
                        row["models"][model_name]["held_out"]["r2"]
                        for row in family_folds
                    ]
                ),
                "held_out_rmse": _confidence_interval(
                    [
                        row["models"][model_name]["held_out"]["rmse"]
                        for row in family_folds
                    ]
                ),
                "held_out_relative_l2": _confidence_interval(
                    [
                        row["models"][model_name]["held_out"]["relative_l2"]
                        for row in family_folds
                    ]
                ),
                "held_out_centered_cosine": _confidence_interval(
                    [
                        row["models"][model_name]["held_out"][
                            "centered_cosine"
                        ]
                        for row in family_folds
                    ]
                ),
            }
        ridge_summary[family] = family_summary
    procrustes_summary = {
        family: {
            "held_out_r2": _confidence_interval(
                [
                    row["held_out"]["r2"]
                    for row in procrustes_folds
                    if row["family"] == family
                    and row.get("status") == "success"
                ]
            ),
            "held_out_rmse": _confidence_interval(
                [
                    row["held_out"]["rmse"]
                    for row in procrustes_folds
                    if row["family"] == family
                    and row.get("status") == "success"
                ]
            ),
            "held_out_centered_cosine": _confidence_interval(
                [
                    row["held_out"]["centered_cosine"]
                    for row in procrustes_folds
                    if row["family"] == family
                    and row.get("status") == "success"
                ]
            ),
        }
        for family in (
            "leave_one_tile_out",
            "seven_fold_spatial_columns",
            "central_edge",
        )
    }
    cca_summary: Dict[str, Any] = {}
    for family in (
        "leave_one_tile_out",
        "seven_fold_spatial_columns",
        "central_edge",
    ):
        rows = [
            row
            for row in cca_folds
            if row["family"] == family and row.get("status") == "success"
        ]
        if rows:
            train_spectra = np.asarray(
                [row["train_canonical_correlations"] for row in rows]
            )
            test_spectra = np.asarray(
                [row["held_out_canonical_correlations"] for row in rows]
            )
            energy = {
                str(count): _confidence_interval(
                    [
                        row["held_out_hr_residual_shared_energy_ratio"][
                            str(count)
                        ]
                        for row in rows
                    ]
                )
                for count in MODE_COUNTS
            }
            cca_summary[family] = {
                "train_correlation_mean": train_spectra.mean(axis=0),
                "train_correlation_std": train_spectra.std(axis=0),
                "held_out_correlation_mean": test_spectra.mean(axis=0),
                "held_out_correlation_std": test_spectra.std(axis=0),
                "hr_residual_shared_energy": energy,
            }
        else:
            cca_summary[family] = {"status": "no_valid_folds"}
    return (
        {
            "seed": seed,
            "summary": ridge_summary,
            "folds": ridge_folds,
        },
        {
            "seed": seed,
            "summary": procrustes_summary,
            "folds": procrustes_folds,
        },
        {
            "seed": seed,
            "summary": cca_summary,
            "folds": cca_folds,
        },
    )


def _aggregate_basic_stats(
    output_dir: Path,
    latent_name: str,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(
        (output_dir / "per_tile").glob(f"tile_*/{latent_name}_stats.json")
    ):
        payload = json.loads(path.read_text("utf-8"))
        if payload.get("status") != "success":
            continue
        for seed, stats in payload["by_seed"].items():
            normalized = stats["normalized_flow_space"]
            raw = stats["decoder_raw_space"]
            rows.append(
                {
                    "tile_id": int(payload["tile_id"]),
                    "seed": int(seed),
                    "tokens": int(payload["common_tokens"]),
                    "position_group": payload["position_group"],
                    "normalized": normalized,
                    "raw": raw,
                }
            )
    output: Dict[str, Any] = {
        "latent": latent_name,
        "encoder_space": "decoder_raw",
        "flow_space": "normalized",
        "analysis_primary_space": SPACE_NORM,
        "decoder_space_analysis": True,
        "rows": rows,
        "aggregate": {},
    }
    for space_key in ("normalized", "raw"):
        output["aggregate"][space_key] = {}
        for left, right in PAIR_NAMES:
            pair = f"{left}_vs_{right}"
            pair_rows = [row[space_key][pair] for row in rows]
            weights = [
                row["tokens"] * int(pair_row["channels"])
                for row, pair_row in zip(rows, pair_rows)
            ]
            total = sum(weights)
            if total == 0:
                continue
            aggregate_row: Dict[str, Any] = {}
            for metric in (
                "mae",
                "rmse",
                "relative_l2",
                "token_cosine_mean",
                "centered_cosine",
                "pearson_correlation",
            ):
                if metric not in pair_rows[0]:
                    continue
                aggregate_row[f"{metric}_feature_weighted"] = float(
                    sum(
                        float(item[metric]) * weight
                        for item, weight in zip(pair_rows, weights)
                    )
                    / total
                )
            output["aggregate"][space_key][pair] = aggregate_row
    return output


def _principal_angles_degrees(
    basis_left: np.ndarray,
    basis_right: np.ndarray,
    count: int,
) -> np.ndarray:
    left_q, _ = np.linalg.qr(basis_left[:, :count])
    right_q, _ = np.linalg.qr(basis_right[:, :count])
    singular_values = np.linalg.svd(
        left_q.T @ right_q, compute_uv=False
    )
    return np.degrees(np.arccos(np.clip(singular_values, -1.0, 1.0)))


def _cross_seed_stability(
    datasets: Sequence[TileDataset],
    regularization: float,
) -> Dict[str, Any]:
    seeds = sorted({item.seed for item in datasets})
    ridge_models: Dict[int, Dict[str, Any]] = {}
    cca_models: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        seed_data = [item for item in datasets if item.seed == seed]
        selection = _select_lambda(seed_data)
        ridge_models[seed] = _fit_full_ridge(
            seed_data, float(selection["selected"])
        )
        ridge_models[seed]["lambda_selection"] = selection
        cca_models[seed] = _fit_cca(seed_data, regularization)
    pairwise: List[Dict[str, Any]] = []
    for left_index, left_seed in enumerate(seeds):
        for right_seed in seeds[left_index + 1 :]:
            left_mapping = ridge_models[left_seed]["A"]
            right_mapping = ridge_models[right_seed]["A"]
            row: Dict[str, Any] = {
                "seed_left": left_seed,
                "seed_right": right_seed,
                "ridge_relative_frobenius": float(
                    np.linalg.norm(left_mapping - right_mapping)
                    / max(1e-12, np.linalg.norm(left_mapping))
                ),
                "cca_spectrum_relative_l2": float(
                    np.linalg.norm(
                        cca_models[left_seed]["canonical_correlations"]
                        - cca_models[right_seed]["canonical_correlations"]
                    )
                    / max(
                        1e-12,
                        np.linalg.norm(
                            cca_models[left_seed]["canonical_correlations"]
                        ),
                    )
                ),
                "cca_principal_angles_degrees": {},
            }
            for count in MODE_COUNTS:
                row["cca_principal_angles_degrees"][str(count)] = (
                    _principal_angles_degrees(
                        cca_models[left_seed]["weights_y"],
                        cca_models[right_seed]["weights_y"],
                        count,
                    )
                )
            pairwise.append(row)
    return {
        "seeds": seeds,
        "ridge_by_seed": {
            str(seed): {
                "lambda_selection": ridge_models[seed]["lambda_selection"],
                "mapping": _matrix_diagnostics(ridge_models[seed]["A"]),
            }
            for seed in seeds
        },
        "cca_by_seed": {
            str(seed): {
                "canonical_correlations": cca_models[seed][
                    "canonical_correlations"
                ],
                "cumulative_explained_rho_squared": cca_models[seed][
                    "cumulative_explained_rho_squared"
                ],
            }
            for seed in seeds
        },
        "pairwise": pairwise,
    }


def _load_time_dataset(
    endpoint: TileDataset,
    latent_name: str,
    step_index: int,
) -> Tuple[TileDataset, Dict[str, Any]]:
    saved = torch.load(
        endpoint.path, map_location="cpu", weights_only=False
    )
    latent = saved[latent_name]
    indices = saved["per_step_token_indices"].to(torch.int64)
    per_step = latent["per_step"]
    row = per_step.get(step_index, per_step.get(str(step_index)))
    if row is None:
        raise KeyError(
            f"{endpoint.path}: no {latent_name} step {step_index}"
        )
    global_subset = (
        latent["global_encoded_norm"]
        .float()
        .index_select(0, indices)
        .numpy()
        .astype(np.float64, copy=False)
    )
    lr_value = row["z0_lr_norm"].float().numpy().astype(
        np.float64, copy=False
    )
    hr_independent = row["z0_hr_norm"].float().numpy().astype(
        np.float64, copy=False
    )
    matched_delta = row["hr_minus_lr_z0_norm"].float().numpy().astype(
        np.float64, copy=False
    )
    time_dataset = TileDataset(
        tile_id=endpoint.tile_id,
        seed=endpoint.seed,
        row=endpoint.row,
        column=endpoint.column,
        position_group=endpoint.position_group,
        tokens=int(global_subset.shape[0]),
        global_norm=global_subset,
        lr_norm=lr_value,
        # For shared/private energy, H-R must be the same-state condition
        # response.  Independent H is retained separately for G-H similarity.
        hr_norm=lr_value + matched_delta,
        path=endpoint.path,
    )
    gr = _comparison_stats(
        torch.from_numpy(global_subset),
        torch.from_numpy(lr_value),
        space=SPACE_NORM,
        include_covariance=False,
    )
    gh = _comparison_stats(
        torch.from_numpy(global_subset),
        torch.from_numpy(hr_independent),
        space=SPACE_NORM,
        include_covariance=False,
    )
    return time_dataset, {
        "tile_id": endpoint.tile_id,
        "seed": endpoint.seed,
        "tokens": int(global_subset.shape[0]),
        "t": float(row["t"]),
        "global_vs_lr": {
            "rmse": gr["rmse"],
            "relative_l2": gr["relative_l2"],
            "centered_cosine": gr["centered_cosine"],
        },
        "global_vs_hr": {
            "rmse": gh["rmse"],
            "relative_l2": gh["relative_l2"],
            "centered_cosine": gh["centered_cosine"],
        },
        "matched_hr_residual_rms": float(np.sqrt(np.square(matched_delta).mean())),
    }


def _time_stability(
    datasets: Sequence[TileDataset],
    latent_name: str,
    regularization: float,
) -> Dict[str, Any]:
    if not datasets:
        return {"status": "no_data"}
    first = torch.load(
        datasets[0].path, map_location="cpu", weights_only=False
    )
    step_keys = sorted(int(key) for key in first[latent_name]["per_step"].keys())
    seeds = sorted({item.seed for item in datasets})
    by_seed: Dict[str, Any] = {}
    for seed in seeds:
        endpoints = [item for item in datasets if item.seed == seed]
        seed_steps: Dict[str, Any] = {}
        for step_index in step_keys:
            time_datasets: List[TileDataset] = []
            similarities: List[Dict[str, Any]] = []
            for endpoint in endpoints:
                dataset, similarity = _load_time_dataset(
                    endpoint, latent_name, step_index
                )
                time_datasets.append(dataset)
                similarities.append(similarity)
            ridge, _, cca = _run_seed_cross_validation(
                time_datasets, regularization=regularization
            )
            ridge_spatial = ridge["summary"]["seven_fold_spatial_columns"]
            cca_spatial = cca["summary"]["seven_fold_spatial_columns"]
            weights = np.asarray(
                [row["tokens"] for row in similarities], dtype=np.float64
            )
            weight_sum = max(1.0, float(weights.sum()))

            def weighted(path: Tuple[str, str]) -> float:
                return float(
                    sum(
                        float(row[path[0]][path[1]]) * weight
                        for row, weight in zip(similarities, weights)
                    )
                    / weight_sum
                )

            seed_steps[str(step_index)] = {
                "step": step_index,
                "t": float(similarities[0]["t"]),
                "tokens": int(weights.sum()),
                "global_vs_lr": {
                    "rmse": weighted(("global_vs_lr", "rmse")),
                    "relative_l2": weighted(
                        ("global_vs_lr", "relative_l2")
                    ),
                    "centered_cosine": weighted(
                        ("global_vs_lr", "centered_cosine")
                    ),
                },
                "global_vs_hr": {
                    "rmse": weighted(("global_vs_hr", "rmse")),
                    "relative_l2": weighted(
                        ("global_vs_hr", "relative_l2")
                    ),
                    "centered_cosine": weighted(
                        ("global_vs_hr", "centered_cosine")
                    ),
                },
                "seven_fold_spatial_heldout_r2": {
                    model: ridge_spatial[model]["held_out_r2"]
                    for model in (
                        "identity",
                        "per_channel_affine",
                        "diagonal_whitening_coloring",
                        "full_ridge",
                    )
                },
                "cca_train_correlation_mean": cca_spatial.get(
                    "train_correlation_mean"
                ),
                "cca_heldout_correlation_mean": cca_spatial.get(
                    "held_out_correlation_mean"
                ),
                "matched_hr_residual_shared_energy": cca_spatial.get(
                    "hr_residual_shared_energy"
                ),
                "per_tile_similarity": similarities,
            }
        by_seed[str(seed)] = seed_steps
    aggregate_by_step: Dict[str, Any] = {}
    for step_index in step_keys:
        rows = [by_seed[str(seed)][str(step_index)] for seed in seeds]
        aggregate_by_step[str(step_index)] = {
            "step": step_index,
            "t_mean": float(np.mean([row["t"] for row in rows])),
            "global_vs_lr_rmse": _confidence_interval(
                [row["global_vs_lr"]["rmse"] for row in rows]
            ),
            "global_vs_lr_centered_cosine": _confidence_interval(
                [row["global_vs_lr"]["centered_cosine"] for row in rows]
            ),
            "global_vs_hr_rmse": _confidence_interval(
                [row["global_vs_hr"]["rmse"] for row in rows]
            ),
            "global_vs_hr_centered_cosine": _confidence_interval(
                [row["global_vs_hr"]["centered_cosine"] for row in rows]
            ),
            "full_ridge_heldout_r2": _confidence_interval(
                [
                    row["seven_fold_spatial_heldout_r2"]["full_ridge"][
                        "mean"
                    ]
                    for row in rows
                ]
            ),
            "cca_train_correlation_mean": np.mean(
                [
                    np.asarray(row["cca_train_correlation_mean"])
                    for row in rows
                    if row["cca_train_correlation_mean"] is not None
                ],
                axis=0,
            ),
            "cca_heldout_correlation_mean": np.mean(
                [
                    np.asarray(row["cca_heldout_correlation_mean"])
                    for row in rows
                    if row["cca_heldout_correlation_mean"] is not None
                ],
                axis=0,
            ),
            "matched_hr_residual_shared_energy": {
                str(count): _confidence_interval(
                    [
                        row["matched_hr_residual_shared_energy"][str(count)][
                            "mean"
                        ]
                        for row in rows
                        if row["matched_hr_residual_shared_energy"]
                        and row["matched_hr_residual_shared_energy"][
                            str(count)
                        ]["mean"]
                        is not None
                    ]
                )
                for count in MODE_COUNTS
            },
        }
    return {
        "latent": latent_name,
        "per_step_endpoint_token_sampling": (
            "deterministic subset; tile membership is never split"
        ),
        "ridge_and_cca_split": "seven spatial column folds",
        "by_seed": by_seed,
        "aggregate_by_step": aggregate_by_step,
    }


def _run_latent_analysis(
    *,
    output_dir: Path,
    latent_name: str,
    min_tokens: int,
    regularization: float,
) -> Dict[str, Any]:
    datasets, excluded = _load_datasets(
        output_dir, latent_name, min_tokens=min_tokens
    )
    seeds = sorted({item.seed for item in datasets})
    ridge_by_seed: Dict[str, Any] = {}
    procrustes_by_seed: Dict[str, Any] = {}
    cca_by_seed: Dict[str, Any] = {}
    for seed in seeds:
        seed_data = [item for item in datasets if item.seed == seed]
        ridge, procrustes, cca = _run_seed_cross_validation(
            seed_data, regularization=regularization
        )
        ridge_by_seed[str(seed)] = ridge
        procrustes_by_seed[str(seed)] = procrustes
        cca_by_seed[str(seed)] = cca
    cross_validation_dir = output_dir / "cross_validation"
    cross_validation_dir.mkdir(parents=True, exist_ok=True)
    ridge_output = {
        "latent": latent_name,
        "comparison_space": SPACE_NORM,
        "tile_level_split": True,
        "random_token_split": False,
        "ridge_lambdas": list(RIDGE_LAMBDAS),
        "excluded_low_token_tiles": excluded,
        "by_seed": ridge_by_seed,
    }
    procrustes_output = {
        "latent": latent_name,
        "comparison_space": SPACE_NORM,
        "tile_level_split": True,
        "regularization": float(regularization),
        "by_seed": procrustes_by_seed,
    }
    cca_output = {
        "latent": latent_name,
        "comparison_space": SPACE_NORM,
        "tile_level_split": True,
        "regularization": float(regularization),
        "all_32_modes_reported": True,
        "mode_counts": list(MODE_COUNTS),
        "by_seed": cca_by_seed,
    }
    _atomic_json(
        cross_validation_dir / f"ridge_{latent_name}.json", ridge_output
    )
    _atomic_json(
        cross_validation_dir / f"procrustes_{latent_name}.json",
        procrustes_output,
    )
    _atomic_json(
        cross_validation_dir / f"cca_{latent_name}.json", cca_output
    )
    basic = _aggregate_basic_stats(output_dir, latent_name)
    stability = _cross_seed_stability(datasets, regularization)
    time_stability = _time_stability(
        datasets, latent_name, regularization
    )
    summary = {
        "format": FORMAT_VERSION,
        "latent": latent_name,
        "encoder_space": "decoder_raw",
        "flow_space": "normalized",
        "analysis_primary_space": SPACE_NORM,
        "decoder_space_analysis": True,
        "primary_min_tokens_per_tile": int(min_tokens),
        "primary_dataset_tiles": sorted({item.tile_id for item in datasets}),
        "primary_dataset_seeds": seeds,
        "excluded_low_token_tiles": excluded,
        "basic_statistics": basic,
        "cross_seed_stability": stability,
        "time_stability": time_stability,
    }
    _atomic_json(output_dir / f"{latent_name}_summary.json", summary)
    return summary


def _mean_seed_model_metric(
    summary: Mapping[str, Any],
    model: str,
    metric: str = "held_out_r2",
) -> Optional[float]:
    values: List[float] = []
    for seed_result in summary["by_seed"].values():
        row = seed_result["summary"]["seven_fold_spatial_columns"][model][
            metric
        ]["mean"]
        if row is not None:
            values.append(float(row))
    return float(np.mean(values)) if values else None


def _load_cross_validation(
    output_dir: Path, prefix: str, latent_name: str
) -> Dict[str, Any]:
    return json.loads(
        (
            output_dir
            / "cross_validation"
            / f"{prefix}_{latent_name}.json"
        ).read_text("utf-8")
    )


def _best_model_summary(
    output_dir: Path, latent_name: str
) -> Dict[str, Any]:
    ridge = _load_cross_validation(output_dir, "ridge", latent_name)
    procrustes = _load_cross_validation(output_dir, "procrustes", latent_name)
    values: Dict[str, Optional[float]] = {}
    for model in (
        "identity",
        "mean_only",
        "per_channel_affine",
        "diagonal_whitening_coloring",
        "full_ridge",
    ):
        values[model] = _mean_seed_model_metric(ridge, model)
    procrustes_values = []
    for seed_result in procrustes["by_seed"].values():
        value = seed_result["summary"]["seven_fold_spatial_columns"][
            "held_out_r2"
        ]["mean"]
        if value is not None:
            procrustes_values.append(float(value))
    values["orthogonal_procrustes"] = (
        float(np.mean(procrustes_values)) if procrustes_values else None
    )
    finite = [
        (name, value)
        for name, value in values.items()
        if value is not None and math.isfinite(value)
    ]
    best = max(finite, key=lambda item: item[1]) if finite else (None, None)
    return {
        "seven_fold_spatial_heldout_r2_by_model": values,
        "best_model": best[0],
        "best_heldout_r2": best[1],
    }


def _plot_basic_by_tile(
    shape_summary: Mapping[str, Any],
    texture_summary: Mapping[str, Any],
    plots_dir: Path,
) -> None:
    for metric, filename, ylabel in (
        ("rmse", "normalized_rmse_by_tile.png", "Normalized RMSE"),
        (
            "centered_cosine",
            "normalized_cosine_by_tile.png",
            "Centered cosine",
        ),
    ):
        fig, axis = plt.subplots(figsize=(13, 5))
        for summary, label, color in (
            (shape_summary, "shape", "tab:blue"),
            (texture_summary, "texture", "tab:orange"),
        ):
            grouped: Dict[int, List[float]] = {}
            for row in summary["basic_statistics"]["rows"]:
                tile_id = int(row["tile_id"])
                value = row["normalized"]["global_vs_lr"][metric]
                grouped.setdefault(tile_id, []).append(float(value))
            tile_ids = sorted(grouped)
            means = [float(np.mean(grouped[item])) for item in tile_ids]
            axis.plot(
                tile_ids,
                means,
                marker="o",
                markersize=3,
                linewidth=1,
                label=label,
                color=color,
            )
        axis.set_xlabel("Tile ID (complete tile; mean across seeds)")
        axis.set_ylabel(ylabel)
        axis.set_title(f"Global encoded vs independent local LR: {ylabel}")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=180)
        plt.close(fig)


def _plot_heldout_r2(output_dir: Path, plots_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(13, 5))
    for latent_name, color in (("shape", "tab:blue"), ("texture", "tab:orange")):
        ridge = _load_cross_validation(output_dir, "ridge", latent_name)
        grouped: Dict[int, List[float]] = {}
        for seed_result in ridge["by_seed"].values():
            for fold in seed_result["folds"]:
                if (
                    fold.get("status") == "success"
                    and fold["family"] == "leave_one_tile_out"
                ):
                    tile_id = int(fold["test_tile_ids"][0])
                    value = fold["models"]["full_ridge"]["held_out"]["r2"]
                    if value is not None:
                        grouped.setdefault(tile_id, []).append(float(value))
        tile_ids = sorted(grouped)
        axis.plot(
            tile_ids,
            [float(np.mean(grouped[item])) for item in tile_ids],
            marker="o",
            markersize=3,
            linewidth=1,
            label=latent_name,
            color=color,
        )
    axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
    axis.set_xlabel("Held-out tile ID")
    axis.set_ylabel("Full-ridge held-out R²")
    axis.set_title("Leave-one-complete-tile-out prediction")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "heldout_r2_by_tile.png", dpi=180)
    plt.close(fig)


def _mean_cca_spectrum(output_dir: Path, latent_name: str) -> np.ndarray:
    cca = _load_cross_validation(output_dir, "cca", latent_name)
    spectra = []
    for seed_result in cca["by_seed"].values():
        value = seed_result["summary"]["seven_fold_spatial_columns"].get(
            "held_out_correlation_mean"
        )
        if value is not None:
            spectra.append(np.asarray(value, dtype=np.float64))
    return np.mean(spectra, axis=0)


def _plot_cca_endpoint(
    output_dir: Path, latent_name: str, plots_dir: Path
) -> None:
    heldout = _mean_cca_spectrum(output_dir, latent_name)
    cca = _load_cross_validation(output_dir, "cca", latent_name)
    train = np.mean(
        [
            np.asarray(
                seed_result["summary"]["seven_fold_spatial_columns"][
                    "train_correlation_mean"
                ]
            )
            for seed_result in cca["by_seed"].values()
        ],
        axis=0,
    )
    fig, axis = plt.subplots(figsize=(8, 5))
    modes = np.arange(1, len(heldout) + 1)
    axis.plot(modes, train, label="train", color="tab:gray")
    axis.plot(modes, heldout, label="held-out spatial folds", color="tab:blue")
    axis.set_xlabel("Canonical mode")
    axis.set_ylabel("Canonical correlation")
    axis.set_ylim(-1.0, 1.05)
    axis.set_title(f"CCA spectrum: {latent_name}")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"cca_spectrum_{latent_name}.png", dpi=180)
    plt.close(fig)


def _plot_time(
    shape_summary: Mapping[str, Any],
    texture_summary: Mapping[str, Any],
    plots_dir: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    for summary, latent_name, linestyle in (
        (shape_summary, "shape", "-"),
        (texture_summary, "texture", "--"),
    ):
        steps = summary["time_stability"]["aggregate_by_step"]
        ordered = [steps[key] for key in sorted(steps, key=int)]
        time_values = [row["t_mean"] for row in ordered]
        for mode, color in ((1, "tab:blue"), (4, "tab:orange"), (8, "tab:green")):
            correlations = [
                row["cca_heldout_correlation_mean"][mode - 1]
                for row in ordered
            ]
            axis.plot(
                time_values,
                correlations,
                linestyle=linestyle,
                color=color,
                label=f"{latent_name} mode {mode}",
            )
    axis.invert_xaxis()
    axis.set_xlabel("Flow t")
    axis.set_ylabel("Held-out canonical correlation")
    axis.set_title("CCA spectrum over Euler steps")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "cca_spectrum_by_step.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    for summary, latent_name, color in (
        (shape_summary, "shape", "tab:blue"),
        (texture_summary, "texture", "tab:orange"),
    ):
        steps = summary["time_stability"]["aggregate_by_step"]
        ordered = [steps[key] for key in sorted(steps, key=int)]
        axis.plot(
            [row["t_mean"] for row in ordered],
            [
                row["matched_hr_residual_shared_energy"]["8"]["mean"]
                for row in ordered
            ],
            marker="o",
            markersize=3,
            label=f"{latent_name}, k=8",
            color=color,
        )
    axis.invert_xaxis()
    axis.set_xlabel("Flow t")
    axis.set_ylabel("Matched HR residual shared-energy ratio")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("HR condition response in CCA shared modes")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(
        plots_dir / "hr_residual_shared_energy_by_step.png", dpi=180
    )
    plt.close(fig)


def _plot_mapping_stability(
    shape_summary: Mapping[str, Any],
    texture_summary: Mapping[str, Any],
    plots_dir: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    labels: List[str] = []
    shape_values: List[float] = []
    texture_values: List[float] = []
    shape_pairs = shape_summary["cross_seed_stability"]["pairwise"]
    texture_pairs = texture_summary["cross_seed_stability"]["pairwise"]
    texture_lookup = {
        (row["seed_left"], row["seed_right"]): row for row in texture_pairs
    }
    for row in shape_pairs:
        key = (row["seed_left"], row["seed_right"])
        labels.append(f"{key[0]}–{key[1]}")
        shape_values.append(float(row["ridge_relative_frobenius"]))
        texture_values.append(
            float(texture_lookup[key]["ridge_relative_frobenius"])
        )
    positions = np.arange(len(labels))
    width = 0.38
    axis.bar(positions - width / 2, shape_values, width, label="shape")
    axis.bar(positions + width / 2, texture_values, width, label="texture")
    axis.set_xticks(positions, labels)
    axis.set_xlabel("Seed pair")
    axis.set_ylabel("||A(s1)-A(s2)||F / ||A(s1)||F")
    axis.set_title("Full-ridge mapping stability across local-flow seeds")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(
        plots_dir / "mapping_stability_across_seeds.png", dpi=180
    )
    plt.close(fig)


def _pooled_spatial_metric(
    output_dir: Path,
    latent_name: str,
    model_name: str,
    *,
    train: bool = False,
) -> Dict[str, Any]:
    ridge = _load_cross_validation(output_dir, "ridge", latent_name)
    values = []
    for seed_result in ridge["by_seed"].values():
        for fold in seed_result["folds"]:
            if (
                fold.get("status") == "success"
                and fold["family"] == "seven_fold_spatial_columns"
            ):
                section = "train" if train else "held_out"
                value = fold["models"][model_name][section]["r2"]
                if value is not None:
                    values.append(float(value))
    return _confidence_interval(values)


def _pooled_cca_top_modes(
    output_dir: Path,
    latent_name: str,
    count: int = 8,
) -> Dict[str, Any]:
    cca = _load_cross_validation(output_dir, "cca", latent_name)
    train_values: List[float] = []
    heldout_values: List[float] = []
    for seed_result in cca["by_seed"].values():
        for fold in seed_result["folds"]:
            if (
                fold.get("status") == "success"
                and fold["family"] == "seven_fold_spatial_columns"
            ):
                train_values.append(
                    float(
                        np.mean(
                            fold["train_canonical_correlations"][:count]
                        )
                    )
                )
                heldout_values.append(
                    float(
                        np.mean(
                            fold["held_out_canonical_correlations"][:count]
                        )
                    )
                )
    return {
        "mode_count": count,
        "train": _confidence_interval(train_values),
        "held_out": _confidence_interval(heldout_values),
    }


def _endpoint_shared_energy(
    output_dir: Path,
    latent_name: str,
) -> Dict[str, Any]:
    cca = _load_cross_validation(output_dir, "cca", latent_name)
    output = {}
    for count in MODE_COUNTS:
        values = []
        for seed_result in cca["by_seed"].values():
            row = seed_result["summary"]["seven_fold_spatial_columns"][
                "hr_residual_shared_energy"
            ][str(count)]["mean"]
            if row is not None:
                values.append(float(row))
        output[str(count)] = _confidence_interval(values)
    return output


def _central_edge_summary(
    output_dir: Path,
    latent_name: str,
) -> Dict[str, Any]:
    ridge = _load_cross_validation(output_dir, "ridge", latent_name)
    by_direction: Dict[str, List[float]] = {
        "central_train_edge_test": [],
        "edge_train_central_test": [],
    }
    for seed_result in ridge["by_seed"].values():
        for fold in seed_result["folds"]:
            if (
                fold.get("status") == "success"
                and fold["family"] == "central_edge"
            ):
                value = fold["models"]["full_ridge"]["held_out"]["r2"]
                if value is not None:
                    by_direction[fold["name"]].append(float(value))
    return {
        name: _confidence_interval(values)
        for name, values in by_direction.items()
    }


def _diagnostic_conclusion(
    output_dir: Path,
    shape_summary: Mapping[str, Any],
    texture_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    basic: Dict[str, Any] = {}
    best: Dict[str, Any] = {}
    ridge_significance: Dict[str, Any] = {}
    cca_top8: Dict[str, Any] = {}
    shared_energy: Dict[str, Any] = {}
    central_edge: Dict[str, Any] = {}
    for latent_name, summary in (
        ("shape", shape_summary),
        ("texture", texture_summary),
    ):
        normalized = summary["basic_statistics"]["aggregate"]["normalized"][
            "global_vs_lr"
        ]
        basic[latent_name] = {
            "relative_l2": normalized["relative_l2_feature_weighted"],
            "token_cosine": normalized[
                "token_cosine_mean_feature_weighted"
            ],
            "centered_cosine": normalized[
                "centered_cosine_feature_weighted"
            ],
            "rmse": normalized["rmse_feature_weighted"],
        }
        best[latent_name] = _best_model_summary(output_dir, latent_name)
        ridge_significance[latent_name] = {
            "train": _pooled_spatial_metric(
                output_dir, latent_name, "full_ridge", train=True
            ),
            "held_out": _pooled_spatial_metric(
                output_dir, latent_name, "full_ridge"
            ),
        }
        cca_top8[latent_name] = _pooled_cca_top_modes(
            output_dir, latent_name, 8
        )
        shared_energy[latent_name] = _endpoint_shared_energy(
            output_dir, latent_name
        )
        central_edge[latent_name] = _central_edge_summary(
            output_dir, latent_name
        )
    stronger = max(
        LATENT_NAMES,
        key=lambda name: (
            basic[name]["centered_cosine"],
            -basic[name]["relative_l2"],
        ),
    )
    significant = {}
    for latent_name in LATENT_NAMES:
        interval = ridge_significance[latent_name]["held_out"]["ci95"]
        significant[latent_name] = (
            interval[0] is not None and float(interval[0]) > 0.0
        )
    endpoint_positive = any(significant.values())
    time_positive: Dict[str, bool] = {}
    for latent_name, summary in (
        ("shape", shape_summary),
        ("texture", texture_summary),
    ):
        time_positive[latent_name] = any(
            row["full_ridge_heldout_r2"]["mean"] is not None
            and float(row["full_ridge_heldout_r2"]["mean"]) > 0.0
            for row in summary["time_stability"]["aggregate_by_step"].values()
        )
    best_nonpositive = all(
        best[name]["best_heldout_r2"] is None
        or float(best[name]["best_heldout_r2"]) <= 0.0
        for name in LATENT_NAMES
    )
    train_positive = any(
        ridge_significance[name]["train"]["mean"] is not None
        and float(ridge_significance[name]["train"]["mean"]) > 0.0
        for name in LATENT_NAMES
    )
    if significant["shape"] != significant["texture"]:
        case = "D"
        reason = (
            "Only one latent type has a full-ridge spatial held-out R² "
            "95% confidence interval strictly above zero."
        )
    elif significant["shape"] and significant["texture"]:
        case = "A"
        reason = (
            "Both latent types have spatial held-out full-ridge R² "
            "confidence intervals strictly above zero; CCA held-out spectra "
            "and cross-seed principal angles quantify the remaining stability."
        )
    elif not endpoint_positive and any(time_positive.values()):
        case = "E"
        reason = (
            "Endpoint evidence is not significantly positive, while at least "
            "one intermediate timestep has positive mean spatial held-out R²."
        )
    elif best_nonpositive:
        case = "C"
        reason = (
            "No evaluated endpoint mapping beats the mean baseline in average "
            "spatial held-out R²."
        )
    else:
        case = "B"
        reason = (
            "Training fit is positive but held-out evidence is not strictly "
            "above zero, indicating tile-specific fit."
            if train_positive
            else "Held-out evidence is inconclusive and not stable."
        )
    support = case == "A"
    if case == "D":
        support_text = (
            "Support is limited to the latent type with significant held-out "
            "evidence; coupling both shape and texture is not supported."
        )
    elif support:
        support_text = (
            "There is sufficient linear/CCA evidence to investigate a "
            "global-shared plus local-private decomposition, without claiming "
            "any rendering improvement."
        )
    else:
        support_text = (
            "There is not sufficient endpoint evidence for a fixed "
            "global-shared plus local-private latent decomposition."
        )
    return {
        "questions": {
            "1_normalized_global_vs_lr": basic,
            "2_stronger_relationship": stronger,
            "3_best_heldout_mapping": best,
            "4_full_ridge_heldout_r2_significance": ridge_significance,
            "5_low_dimensional_cca_shared_subspace": cca_top8,
            "6_cross_tile_cross_seed_generalization": {
                "shape": shape_summary["cross_seed_stability"],
                "texture": texture_summary["cross_seed_stability"],
            },
            "7_tile_position_dependence": central_edge,
            "8_timestep_dependence": {
                "shape": shape_summary["time_stability"][
                    "aggregate_by_step"
                ],
                "texture": texture_summary["time_stability"][
                    "aggregate_by_step"
                ],
            },
            "9_hr_minus_lr_shared_energy": shared_energy,
            "10_shared_private_method_support": support_text,
        },
        "strict_case": case,
        "strict_case_reason": reason,
        "supports_next_shared_private_latent_experiment": support,
        "no_metric_improvement_claimed": True,
    }


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    shape_summary = _run_latent_analysis(
        output_dir=output_dir,
        latent_name="shape",
        min_tokens=int(args.min_analysis_tokens),
        regularization=float(args.cca_regularization),
    )
    texture_summary = _run_latent_analysis(
        output_dir=output_dir,
        latent_name="texture",
        min_tokens=int(args.min_analysis_tokens),
        regularization=float(args.cca_regularization),
    )
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_basic_by_tile(shape_summary, texture_summary, plots_dir)
    _plot_heldout_r2(output_dir, plots_dir)
    _plot_cca_endpoint(output_dir, "shape", plots_dir)
    _plot_cca_endpoint(output_dir, "texture", plots_dir)
    _plot_time(shape_summary, texture_summary, plots_dir)
    _plot_mapping_stability(shape_summary, texture_summary, plots_dir)
    conclusion = _diagnostic_conclusion(
        output_dir, shape_summary, texture_summary
    )
    manifest = json.loads(
        (output_dir / "collection_manifest.json").read_text("utf-8")
    )
    summary = {
        "format": FORMAT_VERSION,
        "image": manifest["image"],
        "cuda_device": manifest["cuda_device"],
        "global_pipeline_seed": manifest["global_pipeline_seed"],
        "local_flow_seeds": manifest["local_flow_seeds"],
        "encoder_space": "decoder_raw",
        "flow_space": "normalized",
        "analysis_primary_space": SPACE_NORM,
        "decoder_space_analysis": True,
        "tile_level_split": True,
        "random_token_split": False,
        "successful_tiles": manifest["successful_tiles"],
        "failed_tiles": manifest["failed_tiles"],
        "skipped_tiles": manifest["skipped_tiles"],
        "shape_summary": str(output_dir / "shape_summary.json"),
        "texture_summary": str(output_dir / "texture_summary.json"),
        "conclusion": conclusion,
        "outputs": {
            "per_tile": str(output_dir / "per_tile"),
            "cross_validation": str(output_dir / "cross_validation"),
            "plots": str(plots_dir),
        },
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/global_local_latent_relationship",
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--moge-model-path", default=None)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--seeds", default="42,43,44,45")
    parser.add_argument(
        "--mode",
        choices=("all", "collect", "analyze"),
        default="all",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    parser.add_argument("--min-flow-tokens", type=int, default=1)
    parser.add_argument("--min-analysis-tokens", type=int, default=5000)
    parser.add_argument("--per-step-tokens", type=int, default=4096)
    parser.add_argument("--max-num-tokens", type=int, default=1_000_000)
    parser.add_argument("--face-projection-chunk-size", type=int, default=250_000)
    parser.add_argument("--cca-regularization", type=float, default=1e-4)

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
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.cuda_device) < 0:
        raise ValueError("--cuda-device must be non-negative")
    _parse_int_csv(args.seeds)
    for name in (
        "ss_steps",
        "shape_steps",
        "texture_steps",
        "min_tile_ovoxels",
        "min_flow_tokens",
        "min_analysis_tokens",
        "per_step_tokens",
        "max_num_tokens",
        "face_projection_chunk_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_tiles is not None and int(args.max_tiles) <= 0:
        raise ValueError("--max-tiles must be positive")
    if (
        not math.isfinite(float(args.cca_regularization))
        or float(args.cca_regularization) <= 0.0
    ):
        raise ValueError("--cca-regularization must be finite and positive")
    if args.mode in ("all", "collect"):
        if not Path(args.image).expanduser().is_file():
            raise FileNotFoundError(args.image)
        for encoder_path in (args.shape_encoder, args.pbr_encoder):
            base = Path(encoder_path).expanduser()
            if not Path(f"{base}.json").is_file() or not Path(
                f"{base}.safetensors"
            ).is_file():
                raise FileNotFoundError(
                    f"encoder checkpoint pair not found for {base}"
                )


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    if args.mode in ("all", "collect"):
        collect(args)
    if args.mode in ("all", "analyze"):
        summary = analyze(args)
        conclusion = summary["conclusion"]
        print(
            f"[done] strict_case={conclusion['strict_case']} "
            f"supports_shared_private="
            f"{conclusion['supports_next_shared_private_latent_experiment']} "
            f"summary={Path(args.output_dir).expanduser().resolve() / 'summary.json'}"
        )


if __name__ == "__main__":
    main()
