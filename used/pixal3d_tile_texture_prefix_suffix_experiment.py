#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pixal3D fixed-shape texture G-prefix -> local-suffix experiment.

The implementation deliberately reuses the existing local tile route for
camera mapping, local dual-grid construction, global PBR queries, encoders,
decoder, stitching, rendering, and metrics.  Only the texture state is
varied.  For every tile the fixed shape SLat is encoded directly from the
global-baseline mesh and is passed unchanged as ``concat_cond`` to every
texture group.

Each prefix group uses the exact native FlowEuler endpoint bridge at one
native timestep and then calls the native sampler's per-step method over the
remaining suffix of the same timestep schedule.  Calling ``sample_once``
preserves the normal model, CFG, guidance interval, and Euler update logic;
the only changed input is the initial texture state.  The resulting nine-point
sweep is deliberately texture-only: the local shape SLat is fixed and is
passed unchanged as ``concat_cond``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as base
from pixal3d.modules.sparse import SparseTensor
from pixal3d.representations import MeshWithVertexPbr, MeshWithVoxel


GROUPS = (
    "tex_G_only",
    "tex_G_to_local_t0.6",
    "tex_G_to_local_t0.681818",
    "tex_G_to_local_t0.75",
    "tex_G_to_local_t0.807692",
    "tex_G_to_local_t0.857143",
    "tex_G_to_local_t0.9",
    "tex_G_to_local_t0.9375",
    "tex_G_to_local_t0.970588",
    "tex_local_full",
)
SWITCH_TIMES = {
    "tex_G_to_local_t0.6": 0.6,
    "tex_G_to_local_t0.681818": 0.681818,
    "tex_G_to_local_t0.75": 0.75,
    "tex_G_to_local_t0.807692": 0.807692,
    "tex_G_to_local_t0.857143": 0.857143,
    "tex_G_to_local_t0.9": 0.9,
    "tex_G_to_local_t0.9375": 0.9375,
    "tex_G_to_local_t0.970588": 0.970588,
}
SWEEP_GROUPS = tuple(SWITCH_TIMES)


def _clone_sparse(value: SparseTensor) -> SparseTensor:
    return SparseTensor(value.feats.detach().clone(), value.coords)


def _group_sort_key(name: str) -> int:
    return GROUPS.index(name)


def _feature_stats(value: Any) -> Dict[str, Any]:
    """Return compact, JSON-safe statistics for one latent/prediction."""
    features = value.feats if isinstance(value, SparseTensor) else value
    features = features.detach().to(torch.float32)
    if features.numel() == 0:
        return {"tokens": 0, "channels": 0}
    flat = features.reshape(-1)
    channel_mean = features.mean(dim=0)
    channel_std = features.std(dim=0, unbiased=False)
    return {
        "tokens": int(features.shape[0]),
        "channels": int(features.shape[1]) if features.ndim > 1 else 1,
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "l2": float(torch.linalg.vector_norm(flat).item()),
        "rms": float(torch.sqrt(torch.mean(flat.square())).item()),
        "channel_mean": [float(v) for v in channel_mean.cpu().tolist()],
        "channel_std": [float(v) for v in channel_std.cpu().tolist()],
    }


def _feature_distance(left: Any, right: Any) -> Dict[str, float]:
    """Compare two aligned sparse/dense feature tensors without gradients."""
    left_value = left.feats if isinstance(left, SparseTensor) else left
    right_value = right.feats if isinstance(right, SparseTensor) else right
    left_value = left_value.detach().to(torch.float32)
    right_value = right_value.detach().to(device=left_value.device, dtype=torch.float32)
    if left_value.shape != right_value.shape:
        raise RuntimeError(
            f"feature distance shape mismatch: {tuple(left_value.shape)} vs "
            f"{tuple(right_value.shape)}"
        )
    delta = (left_value - right_value).reshape(-1)
    left_flat = left_value.reshape(-1)
    right_flat = right_value.reshape(-1)
    left_norm = torch.linalg.vector_norm(left_flat).clamp_min(torch.finfo(torch.float32).eps)
    right_norm = torch.linalg.vector_norm(right_flat).clamp_min(torch.finfo(torch.float32).eps)
    cosine = torch.dot(left_flat, right_flat) / (left_norm * right_norm)
    return {
        "l2": float(torch.linalg.vector_norm(delta).item()),
        "rms": float(torch.sqrt(torch.mean(delta.square())).item()),
        "relative_l2_to_left": float((torch.linalg.vector_norm(delta) / left_norm).item()),
        "relative_l2_to_right": float((torch.linalg.vector_norm(delta) / right_norm).item()),
        "cosine": float(cosine.clamp(-1.0, 1.0).item()),
    }


def _trajectory_step_record(
    *,
    step: int,
    timestep: float,
    next_timestep: float,
    state: SparseTensor,
    result: Any,
    g_tex_norm: SparseTensor,
    full_state: Optional[torch.Tensor],
) -> Dict[str, Any]:
    """Record the required per-step state, velocity, x0 and distances."""
    state_after = result.pred_x_prev
    if not isinstance(state_after, SparseTensor):
        raise RuntimeError("native sampler returned a non-sparse next state")
    state_features = state.feats
    after_features = state_after.feats
    full_state_value = None
    if full_state is not None:
        full_state_value = SparseTensor(full_state, state.coords)
    return {
        "step": int(step),
        "timestep": float(timestep),
        "next_timestep": float(next_timestep),
        "time_interval": float(timestep - next_timestep),
        "state": _feature_stats(state),
        "predicted_velocity": _feature_stats(result.pred_v),
        "pred_x_0": _feature_stats(result.pred_x_0),
        "state_after": _feature_stats(state_after),
        "distance_state_to_G": _feature_distance(state, g_tex_norm),
        "distance_state_after_to_G": _feature_distance(state_after, g_tex_norm),
        "distance_state_to_full_local": (
            _feature_distance(state, full_state_value)
            if full_state_value is not None
            else None
        ),
        "distance_state_after_to_full_local": (
            _feature_distance(state_after, full_state_value)
            if full_state_value is not None
            else None
        ),
    }


def _native_suffix_flow(
    *,
    pipeline: Any,
    model: torch.nn.Module,
    sampler: Any,
    initial_state: SparseTensor,
    condition: Mapping[str, Any],
    shape_cond: SparseTensor,
    sampler_params: Mapping[str, Any],
    switch_t: float,
    description: str,
    g_tex_norm: SparseTensor,
    full_trajectory_states: Optional[Sequence[torch.Tensor]] = None,
) -> Tuple[SparseTensor, Dict[str, Any], List[torch.Tensor]]:
    """Run the unchanged native Euler/CFG step logic from ``switch_t`` to 0."""
    steps = int(sampler_params["steps"])
    rescale_t = float(sampler_params["rescale_t"])
    schedule = [float(v) for v in sampler.timestep_schedule(steps, rescale_t)]
    matches = [i for i, value in enumerate(schedule) if abs(value - switch_t) <= 1e-6]
    if len(matches) != 1:
        raise RuntimeError(
            f"switch timestep {switch_t} is not an exact native schedule point: "
            f"schedule={schedule}"
        )
    start_index = matches[0]
    suffix_schedule = schedule[start_index:]
    if len(suffix_schedule) < 2:
        raise RuntimeError(f"switch timestep {switch_t} has no suffix update")

    # sample_once consumes the same per-step arguments that sample() forwards
    # to the model.  It also goes through the sampler's normal CFG and
    # guidance-interval MRO; no custom prediction or CFG calculation occurs.
    step_kwargs = base._sampler_step_kwargs(sampler_params)
    step_kwargs["concat_cond"] = shape_cond
    if full_trajectory_states is not None and len(full_trajectory_states) != len(schedule):
        raise RuntimeError(
            "full-local trajectory has an unexpected state count: "
            f"{len(full_trajectory_states)} vs schedule {len(schedule)}"
        )
    state = initial_state
    trajectory_states: List[torch.Tensor] = [state.feats.detach().cpu().clone()]
    trajectory_records: List[Dict[str, Any]] = []
    if pipeline.low_vram:
        model.to(torch.device(pipeline.device))
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for step, (t, t_prev) in enumerate(
                zip(suffix_schedule[:-1], suffix_schedule[1:])
            ):
                result = sampler.sample_once(
                    model,
                    state,
                    float(t),
                    float(t_prev),
                    cond=condition["cond"],
                    neg_cond=condition["neg_cond"],
                    **step_kwargs,
                )
                trajectory_records.append(
                    _trajectory_step_record(
                        step=start_index + step,
                        timestep=float(t),
                        next_timestep=float(t_prev),
                        state=state,
                        result=result,
                        g_tex_norm=g_tex_norm,
                        full_state=(
                            full_trajectory_states[start_index + step]
                            if full_trajectory_states is not None
                            else None
                        ),
                    )
                )
                state = result.pred_x_prev
                if not isinstance(state, SparseTensor):
                    raise RuntimeError(
                        f"{description} step {step} returned {type(state)!r}, "
                        "expected SparseTensor"
                    )
                trajectory_states.append(state.feats.detach().cpu().clone())
    finally:
        if pipeline.low_vram:
            model.cpu()
    base._sync_cuda()
    seconds = time.perf_counter() - started
    if not torch.equal(state.coords, initial_state.coords):
        raise RuntimeError(f"{description} changed local latent support")
    return state, {
        "flow_seconds": float(seconds),
        "flow_steps": int(len(suffix_schedule) - 1),
        "switch_timestep": float(switch_t),
        "suffix_timestep_schedule": suffix_schedule,
        "sampler_execution": "native sampler.sample_once over exact native schedule suffix",
        "trajectory": trajectory_records,
        "trajectory_state_count": int(len(trajectory_states)),
    }, trajectory_states


def _native_full_texture_flow(
    *,
    pipeline: Any,
    model: torch.nn.Module,
    sampler: Any,
    initial_state: SparseTensor,
    condition: Mapping[str, Any],
    shape_cond: SparseTensor,
    sampler_params: Mapping[str, Any],
    description: str,
    g_tex_norm: SparseTensor,
) -> Tuple[SparseTensor, Dict[str, Any], List[torch.Tensor]]:
    """Run the normal local texture sampler from the shared t=1 noise state."""
    kwargs = dict(sampler_params)
    kwargs["concat_cond"] = shape_cond
    if pipeline.low_vram:
        model.to(torch.device(pipeline.device))
    started = time.perf_counter()
    try:
        with torch.no_grad():
            result = sampler.sample(
                model,
                initial_state,
                cond=condition["cond"],
                neg_cond=condition["neg_cond"],
                **kwargs,
                verbose=True,
                tqdm_desc=description,
                record_trajectory=True,
                trajectory_device="cpu",
                return_model_history=False,
            )
    finally:
        if pipeline.low_vram:
            model.cpu()
    base._sync_cuda()
    final_state = getattr(result, "samples", result)
    if not isinstance(final_state, SparseTensor):
        raise RuntimeError(f"{description} returned {type(final_state)!r}, expected SparseTensor")
    if not torch.equal(final_state.coords, initial_state.coords):
        raise RuntimeError(f"{description} changed local latent support")
    trajectory = getattr(result, "trajectory", None)
    if trajectory is None:
        raise RuntimeError(f"{description} did not return a trajectory")
    trajectory_states = [
        value.detach().cpu().clone() for value in trajectory.states
    ]
    trajectory_records: List[Dict[str, Any]] = []
    schedule = [
        float(v)
        for v in sampler.timestep_schedule(
            int(sampler_params["steps"]), float(sampler_params["rescale_t"])
        )
    ]
    if len(trajectory_states) != len(schedule) or len(trajectory.velocities) != len(schedule) - 1:
        raise RuntimeError(f"{description} trajectory length is inconsistent")
    # ``sample()`` does not expose pred_x_0 history when model history is
    # disabled.  Reconstruct the compact x0 statistics from the saved state
    # and velocity using the exact native FlowEuler relationship.
    for step, (t, t_prev) in enumerate(zip(schedule[:-1], schedule[1:])):
        state = SparseTensor(trajectory_states[step], initial_state.coords)
        next_state = SparseTensor(trajectory_states[step + 1], initial_state.coords)
        velocity = SparseTensor(trajectory.velocities[step], initial_state.coords)
        pred_x_0 = sampler._pred_to_xstart(
            state.feats,
            torch.tensor(float(t), dtype=state.feats.dtype),
            velocity.feats,
        )
        result_like = type("Result", (), {
            "pred_x_prev": next_state,
            "pred_v": velocity,
            "pred_x_0": SparseTensor(pred_x_0, initial_state.coords),
        })()
        trajectory_records.append(
            _trajectory_step_record(
                step=step,
                timestep=float(t),
                next_timestep=float(t_prev),
                state=state,
                result=result_like,
                g_tex_norm=g_tex_norm,
                full_state=trajectory_states[step],
            )
        )
    return final_state, {
        "flow_seconds": float(time.perf_counter() - started),
        "flow_steps": int(sampler_params["steps"]),
        "switch_timestep": 1.0,
        "suffix_timestep_schedule": [
            float(v)
            for v in sampler.timestep_schedule(
                int(sampler_params["steps"]), float(sampler_params["rescale_t"])
            )
        ],
        "sampler_execution": "native sampler.sample from t=1",
        "trajectory": trajectory_records,
        "trajectory_state_count": int(len(trajectory_states)),
    }, trajectory_states


def _run_texture_groups(
    *,
    pipeline: Any,
    fixed_shape_norm: SparseTensor,
    fixed_shape_denorm: SparseTensor,
    g_tex_norm: SparseTensor,
    g_tex_denorm: SparseTensor,
    condition: Mapping[str, Any],
    texture_params: Mapping[str, Any],
    seed: int,
    tile_id: int,
) -> Tuple[Dict[str, SparseTensor], Dict[str, Any]]:
    """Create one shared noise tensor and produce the complete sweep."""
    device = torch.device(pipeline.device)
    texture_model = pipeline.models["tex_slat_flow_model_1024"]
    sampler = pipeline.tex_slat_sampler
    merged_params = {**pipeline.tex_slat_sampler_params, **dict(texture_params)}
    steps = int(merged_params["steps"])
    rescale_t = float(merged_params["rescale_t"])
    schedule = [float(v) for v in sampler.timestep_schedule(steps, rescale_t)]
    coords = fixed_shape_norm.coords.to(torch.int32)
    if not torch.equal(coords, g_tex_norm.coords.to(torch.int32)):
        raise RuntimeError("fixed shape and G texture supports differ")
    texture_channels = int(texture_model.in_channels) - int(fixed_shape_norm.feats.shape[1])
    if texture_channels <= 0:
        raise RuntimeError(f"invalid texture channel count {texture_channels}")
    if int(g_tex_norm.feats.shape[1]) != texture_channels:
        raise RuntimeError(
            "G texture channels do not match texture flow input: "
            f"flow={texture_channels} G={g_tex_norm.feats.shape[1]}"
        )

    # Conditions are computed before this seed in the caller.  This makes the
    # tensor below the only texture noise source and lets every group reuse it
    # byte-for-byte.
    base._seed_everything(int(seed))
    texture_noise = SparseTensor(
        torch.randn(
            coords.shape[0], texture_channels, device=device, dtype=torch.float32
        ),
        coords,
    )
    initial_noise_range = base._tensor_range(texture_noise.feats)
    outputs: Dict[str, SparseTensor] = {}
    group_stats: Dict[str, Any] = {
        "tile_id": int(tile_id),
        "seed": int(seed),
        "texture_tokens": int(coords.shape[0]),
        "texture_channels": int(texture_channels),
        "texture_noise": {
            "range": initial_noise_range,
            "l2": float(torch.linalg.vector_norm(texture_noise.feats).item()),
            "shared_across_groups": True,
            "same_noise_source": "one torch.randn SparseTensor created once per tile",
        },
        "native_texture_timestep_schedule": schedule,
        "groups": {},
    }

    # G-only is decoded exactly as the fixed endpoint, with no texture flow.
    outputs["tex_G_only"] = g_tex_denorm
    group_stats["groups"]["tex_G_only"] = {
        "initial_state": "G_tex_denorm endpoint",
        "flow_seconds": 0.0,
        "flow_steps": 0,
        "switch_timestep": None,
        "noise_used_for_comparison": True,
        "native_noised_endpoint": "not applied; direct endpoint decode",
    }

    # Run t=1 first so every earlier switch can be compared against the exact
    # full-local state at the same native timestep.  This uses the unchanged
    # sampler.sample implementation and the same t=1 state/noise as the
    # other groups.
    local_initial = base._native_noised_endpoint(
        g_tex_norm, texture_noise, sampler, schedule[0]
    )
    local_state, local_stats, full_trajectory_states = _native_full_texture_flow(
        pipeline=pipeline,
        model=texture_model,
        sampler=sampler,
        initial_state=local_initial,
        condition=condition,
        shape_cond=fixed_shape_norm,
        sampler_params=merged_params,
        description=f"Tile {tile_id:02d} tex_local_full texture flow",
        g_tex_norm=g_tex_norm,
    )
    outputs["tex_local_full"] = base._denormalize_slat(
        local_state, pipeline.tex_slat_normalization
    )
    group_stats["groups"]["tex_local_full"] = {
        "initial_state": "native_noised_endpoint(G_tex_norm, shared_texture_noise, t=1); equals epsilon",
        "native_noised_endpoint": "x_t=(1-t)G_norm+sigma(t)epsilon",
        "noise_timestep": float(schedule[0]),
        **local_stats,
    }
    del local_initial, local_state
    base._empty_cuda_cache()

    for group_name, switch_t in SWITCH_TIMES.items():
        initial = base._native_noised_endpoint(
            g_tex_norm, texture_noise, sampler, switch_t
        )
        state, stats, _trajectory_states = _native_suffix_flow(
            pipeline=pipeline,
            model=texture_model,
            sampler=sampler,
            initial_state=initial,
            condition=condition,
            shape_cond=fixed_shape_norm,
            sampler_params=merged_params,
            switch_t=switch_t,
            description=f"Tile {tile_id:02d} {group_name} texture suffix",
            g_tex_norm=g_tex_norm,
            full_trajectory_states=full_trajectory_states,
        )
        outputs[group_name] = base._denormalize_slat(
            state, pipeline.tex_slat_normalization
        )
        group_stats["groups"][group_name] = {
            "initial_state": "native_noised_endpoint(G_tex_norm, shared_texture_noise, switch_t)",
            "native_noised_endpoint": "x_t=(1-t)G_norm+sigma(t)epsilon",
            "noise_timestep": float(switch_t),
            **stats,
        }
        del initial, state, _trajectory_states
        base._empty_cuda_cache()
    del full_trajectory_states, texture_noise
    base._empty_cuda_cache()
    return outputs, group_stats


def _save_comparison_sheet(
    *,
    entries: Sequence[Tuple[Path, str, Optional[Mapping[str, Any]]]],
    output_path: Path,
) -> None:
    panel = 384
    header = 72
    columns = 3
    rows = (len(entries) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * panel, rows * (panel + header)), "black")
    draw = ImageDraw.Draw(sheet)
    for index, (path, title, metrics) in enumerate(entries):
        x = (index % columns) * panel
        y = (index // columns) * (panel + header)
        if path.is_file():
            with Image.open(path) as image:
                image = ImageOps.contain(image.convert("RGB"), (panel, panel))
            sheet.paste(image, (x + (panel - image.width) // 2, y + header))
        else:
            draw.rectangle((x, y + header, x + panel - 1, y + header + panel - 1), fill=(60, 0, 0))
        draw.text((x + 6, y + 8), title, fill=(255, 255, 255))
        if metrics is not None:
            draw.text(
                (x + 6, y + 29),
                f"PSNR {metrics.get('psnr_db')}  SSIM {metrics.get('ssim')}",
                fill=(220, 220, 220),
            )
            draw.text(
                (x + 6, y + 50),
                f"LPIPS {metrics.get('lpips')}",
                fill=(180, 180, 180),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _mean_multiview_metrics(multiview: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    rows = list(multiview.get("pair_metrics", []))
    if not rows:
        return {"psnr_db": None, "ssim": None}
    return {
        "psnr_db": float(np.mean([row["baseline_vs_stitched_psnr_db"] for row in rows])),
        "ssim": float(np.mean([row["baseline_vs_stitched_ssim"] for row in rows])),
    }


def _pbr_statistics(mesh: Any, source: str) -> Dict[str, Any]:
    """Summarize final stitched PBR channels, including robust percentiles."""
    if isinstance(mesh, MeshWithVoxel):
        attrs = mesh.attrs
        layout = base.PBR_LAYOUT
    elif isinstance(mesh, MeshWithVertexPbr):
        attrs = mesh.vertex_attrs
        layout = mesh.layout or base.PBR_LAYOUT
    else:
        raise TypeError(f"unsupported mesh type for PBR statistics: {type(mesh)!r}")
    attrs = attrs.detach().to(torch.float32).cpu()
    channel_groups = {
        "base_color": layout["base_color"],
        "metallic": layout["metallic"],
        "roughness": layout["roughness"],
        "alpha": layout["alpha"],
    }
    stats: Dict[str, Any] = {
        "source": source,
        "attribute_tokens": int(attrs.shape[0]),
        "attribute_channels": int(attrs.shape[1]),
        "channels": {},
    }
    for name, channel_slice in channel_groups.items():
        values = attrs[:, channel_slice].reshape(-1).numpy().astype(np.float32, copy=False)
        if values.size == 0:
            stats["channels"][name] = None
            continue
        stats["channels"][name] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }
    rgb_slice = channel_groups["base_color"]
    rgb = attrs[:, rgb_slice].numpy().astype(np.float32, copy=False)
    stats["rgb"] = {
        channel: {
            "min": float(np.min(rgb[:, index])),
            "max": float(np.max(rgb[:, index])),
            "mean": float(np.mean(rgb[:, index])),
            "std": float(np.std(rgb[:, index])),
            "p05": float(np.percentile(rgb[:, index], 5)),
            "p50": float(np.percentile(rgb[:, index], 50)),
            "p95": float(np.percentile(rgb[:, index], 95)),
        }
        for index, channel in enumerate(("r", "g", "b"))
    }
    return stats


def _pbr_stat_change(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
    """Compute interpretable mean/dispersion changes between two PBR endpoints."""
    before_channels = before.get("channels", {})
    after_channels = after.get("channels", {})
    changes: Dict[str, Any] = {}
    for name in ("base_color", "metallic", "roughness", "alpha"):
        left = before_channels.get(name)
        right = after_channels.get(name)
        if not left or not right:
            continue
        changes[name] = {
            "mean_delta": float(right["mean"] - left["mean"]),
            "abs_mean_delta": float(abs(right["mean"] - left["mean"])),
            "std_delta": float(right["std"] - left["std"]),
            "p05_delta": float(right["p05"] - left["p05"]),
            "p50_delta": float(right["p50"] - left["p50"]),
            "p95_delta": float(right["p95"] - left["p95"]),
        }
    ranked = sorted(
        changes.items(), key=lambda item: item[1]["abs_mean_delta"], reverse=True
    )
    return {
        "channels": changes,
        "largest_abs_mean_shift": ranked[0][0] if ranked else None,
    }


def _input_metrics(result: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    return dict(result.get("input_metrics") or {"psnr_db": None, "ssim": None, "lpips": None})


def _analyze_sweep(group_results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Find the empirical transition interval and assemble report-ready rows."""
    rows: List[Dict[str, Any]] = []
    analysis_groups = (*SWEEP_GROUPS, "tex_local_full")
    for group_name in analysis_groups:
        result = group_results.get(group_name, {})
        metrics = _input_metrics(result)
        multiview = result.get("multiview_mean_against_baseline") or {}
        switch_t = 1.0 if group_name == "tex_local_full" else float(SWITCH_TIMES[group_name])
        rows.append(
            {
                "group": group_name,
                "switch_timestep": switch_t,
                "input_metrics": metrics,
                "multiview_mean_against_baseline": {
                    "psnr_db": multiview.get("psnr_db"),
                    "ssim": multiview.get("ssim"),
                },
                "pbr_statistics": result.get("pbr_statistics"),
                "status": result.get("status"),
            }
        )
    rows = [row for row in rows if row["input_metrics"].get("psnr_db") is not None]
    rows.sort(key=lambda row: row["switch_timestep"])
    intervals: List[Dict[str, Any]] = []
    for before, after in zip(rows[:-1], rows[1:]):
        before_psnr = float(before["input_metrics"]["psnr_db"])
        after_psnr = float(after["input_metrics"]["psnr_db"])
        before_mv = before["multiview_mean_against_baseline"].get("psnr_db")
        after_mv = after["multiview_mean_against_baseline"].get("psnr_db")
        input_delta = after_psnr - before_psnr
        mv_delta = (
            float(after_mv) - float(before_mv)
            if before_mv is not None and after_mv is not None
            else None
        )
        pbr_change = None
        if before.get("pbr_statistics") and after.get("pbr_statistics"):
            pbr_change = _pbr_stat_change(
                before["pbr_statistics"], after["pbr_statistics"]
            )
        intervals.append(
            {
                "before_group": before["group"],
                "after_group": after["group"],
                "t_before": float(before["switch_timestep"]),
                "t_after": float(after["switch_timestep"]),
                "input_psnr_delta_db": float(input_delta),
                "multiview_psnr_delta_db": mv_delta,
                "directional_sync": bool(input_delta > 0.0 and mv_delta is not None and mv_delta < 0.0),
                "pbr_change": pbr_change,
            }
        )
    positive = [item for item in intervals if item["input_psnr_delta_db"] > 0.0]
    candidates = positive or intervals
    transition = max(
        candidates,
        key=lambda item: (
            item["input_psnr_delta_db"] if positive else abs(item["input_psnr_delta_db"])
        ),
        default=None,
    )
    abs_deltas = [abs(item["input_psnr_delta_db"]) for item in intervals]
    median_abs_delta = float(np.median(abs_deltas)) if abs_deltas else 0.0
    max_positive_delta = (
        max(item["input_psnr_delta_db"] for item in positive) if positive else 0.0
    )
    sharp = bool(
        transition is not None
        and max_positive_delta >= 1.0
        and (median_abs_delta == 0.0 or max_positive_delta >= 2.0 * median_abs_delta)
    )
    synchronized = bool(
        transition is not None
        and transition.get("directional_sync")
    )
    pbr_change = transition.get("pbr_change") if transition else None
    largest_pbr = pbr_change.get("largest_abs_mean_shift") if pbr_change else None
    role = "insufficient PBR evidence"
    if largest_pbr == "metallic" or largest_pbr == "roughness":
        role = "material / PBR regime"
    elif largest_pbr == "base_color":
        role = "large-scale base-color appearance"
    elif largest_pbr == "alpha":
        role = "alpha / occupancy appearance"
    elif transition is not None:
        role = "texture identity or local detail (no dominant aggregate PBR shift)"
    return {
        "rows": rows,
        "intervals": intervals,
        "t_star": {
            "t_before": transition["t_before"] if transition else None,
            "t_after": transition["t_after"] if transition else None,
            "before_group": transition["before_group"] if transition else None,
            "after_group": transition["after_group"] if transition else None,
            "input_psnr_delta_db": transition["input_psnr_delta_db"] if transition else None,
            "transition_type": "sharp transition / basin switch" if sharp else "gradual or multi-step transition",
            "selection": "largest positive adjacent PSNR change while moving from lower t to higher t",
        },
        "input_quality_vs_global_consistency": {
            "synchronized_at_t_star": synchronized,
            "interpretation": (
                "input PSNR rises while mean multiview PSNR vs baseline falls in the same native interval"
                if synchronized
                else "the largest input-quality interval is not accompanied by a same-interval multiview PSNR decrease"
            ),
        },
        "largest_transition_pbr_shift": largest_pbr,
        "early_flow_role": role,
    }


def _aggregate_texture_trajectories(tile_records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Average compact trajectory scalars across successful tiles."""
    group_records: Dict[str, List[Mapping[str, Any]]] = {name: [] for name in GROUPS if name != "tex_G_only"}
    for tile in tile_records:
        if tile.get("status") != "success":
            continue
        for group_name, group in tile.get("flow", {}).get("groups", {}).items():
            if group_name in group_records:
                group_records[group_name].append(group)
    output: Dict[str, Any] = {
        "format": "pixal3d_texture_flow_trajectory_summary_v1",
        "scope": "successful tiles; scalar averages preserve each tile's native trajectory record",
        "groups": {},
    }
    scalar_paths = (
        ("state", "mean"),
        ("state", "std"),
        ("state", "l2"),
        ("state", "rms"),
        ("predicted_velocity", "mean"),
        ("predicted_velocity", "std"),
        ("predicted_velocity", "l2"),
        ("predicted_velocity", "rms"),
        ("pred_x_0", "mean"),
        ("pred_x_0", "std"),
        ("pred_x_0", "l2"),
        ("pred_x_0", "rms"),
        ("distance_state_to_G", "l2"),
        ("distance_state_to_G", "rms"),
        ("distance_state_to_G", "cosine"),
        ("distance_state_after_to_G", "l2"),
        ("distance_state_after_to_G", "rms"),
        ("distance_state_after_to_G", "cosine"),
        ("distance_state_to_full_local", "l2"),
        ("distance_state_to_full_local", "rms"),
        ("distance_state_to_full_local", "cosine"),
        ("distance_state_after_to_full_local", "l2"),
        ("distance_state_after_to_full_local", "rms"),
        ("distance_state_after_to_full_local", "cosine"),
    )
    for group_name, groups in group_records.items():
        per_step: Dict[int, List[Mapping[str, Any]]] = {}
        for group in groups:
            for record in group.get("trajectory", []):
                per_step.setdefault(int(record["step"]), []).append(record)
        records: List[Dict[str, Any]] = []
        for step in sorted(per_step):
            values = per_step[step]
            aggregate: Dict[str, Any] = {
                "step": int(step),
                "timestep": float(np.mean([value["timestep"] for value in values])),
                "next_timestep": float(np.mean([value["next_timestep"] for value in values])),
                "tile_count": int(len(values)),
            }
            for path, key in scalar_paths:
                numeric = [
                    value.get(path, {}).get(key)
                    for value in values
                    if value.get(path) is not None and value.get(path, {}).get(key) is not None
                ]
                if numeric:
                    aggregate[f"{path}.{key}"] = float(np.mean(numeric))
            records.append(aggregate)
        output["groups"][group_name] = {
            "tile_count": int(len(groups)),
            "records": records,
        }
    return output


def _plot_sweep_artifacts(
    output_dir: Path,
    analysis: Mapping[str, Any],
) -> Dict[str, str]:
    """Write the four required figures using the measured endpoint rows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(analysis.get("rows", []))
    x = [float(row["switch_timestep"]) for row in rows]
    labels = [f"{value:.6f}".rstrip("0").rstrip(".") for value in x]
    transition = analysis.get("t_star", {})
    paths: Dict[str, str] = {}

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for axis, metric, title, ylabel in zip(
        axes,
        ("psnr_db", "ssim", "lpips"),
        ("Input-view PSNR", "Input-view SSIM", "Input-view LPIPS"),
        ("dB", "SSIM", "LPIPS"),
    ):
        values = [row["input_metrics"].get(metric) for row in rows]
        axis.plot(x, values, marker="o", linewidth=2)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[-1].set_xticks(x, labels, rotation=35, ha="right")
    axes[-1].set_xlabel("switch timestep t_s (ascending)")
    if transition.get("t_before") is not None:
        for axis in axes:
            axis.axvspan(transition["t_before"], transition["t_after"], color="orange", alpha=0.18)
    fig.suptitle("Figure 1 — input-view quality vs switch timestep")
    fig.tight_layout()
    path = output_dir / "figure1_input_metrics_vs_timestep.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["figure1_input_metrics"] = str(path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for axis, metric, title, ylabel in zip(
        axes,
        ("psnr_db", "ssim"),
        ("Mean multiview PSNR vs global baseline", "Mean multiview SSIM vs global baseline"),
        ("dB", "SSIM"),
    ):
        values = [row["multiview_mean_against_baseline"].get(metric) for row in rows]
        axis.plot(x, values, marker="o", linewidth=2, color="tab:purple")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[-1].set_xticks(x, labels, rotation=35, ha="right")
    axes[-1].set_xlabel("switch timestep t_s (ascending)")
    if transition.get("t_before") is not None:
        for axis in axes:
            axis.axvspan(transition["t_before"], transition["t_after"], color="orange", alpha=0.18)
    fig.suptitle("Figure 2 — multiview similarity vs global baseline")
    fig.tight_layout()
    path = output_dir / "figure2_multiview_vs_baseline.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["figure2_multiview"] = str(path)

    fig, axis = plt.subplots(figsize=(10, 5.5))
    input_values = [row["input_metrics"].get("psnr_db") for row in rows]
    mv_values = [row["multiview_mean_against_baseline"].get("psnr_db") for row in rows]
    axis.plot(x, input_values, marker="o", linewidth=2, label="input PSNR", color="tab:blue")
    axis.set_ylabel("input PSNR (dB)", color="tab:blue")
    axis.tick_params(axis="y", labelcolor="tab:blue")
    axis.grid(alpha=0.25)
    right = axis.twinx()
    right.plot(x, mv_values, marker="s", linewidth=2, label="mean multiview PSNR vs baseline", color="tab:red")
    right.set_ylabel("mean multiview PSNR vs baseline (dB)", color="tab:red")
    right.tick_params(axis="y", labelcolor="tab:red")
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.set_xlabel("switch timestep t_s (ascending)")
    if transition.get("t_before") is not None:
        axis.axvspan(transition["t_before"], transition["t_after"], color="orange", alpha=0.18)
    axis.set_title("Figure 3 — input quality and global consistency")
    fig.tight_layout()
    path = output_dir / "figure3_input_vs_global_consistency.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["figure3_joint"] = str(path)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for axis, channel, title, color in zip(
        axes,
        ("base_color", "metallic", "roughness"),
        ("RGB/base-color", "metallic", "roughness"),
        ("tab:green", "tab:orange", "tab:brown"),
    ):
        means = [
            row.get("pbr_statistics", {}).get("channels", {}).get(channel, {}).get("mean")
            for row in rows
        ]
        p05 = [
            row.get("pbr_statistics", {}).get("channels", {}).get(channel, {}).get("p05")
            for row in rows
        ]
        p95 = [
            row.get("pbr_statistics", {}).get("channels", {}).get(channel, {}).get("p95")
            for row in rows
        ]
        if channel == "base_color":
            rgb_colors = {"r": "#d62728", "g": "#2ca02c", "b": "#1f77b4"}
            for rgb_channel, rgb_color in rgb_colors.items():
                rgb_values = [
                    row.get("pbr_statistics", {}).get("rgb", {}).get(rgb_channel, {}).get("mean")
                    for row in rows
                ]
                axis.plot(x, rgb_values, marker="o", linewidth=1.7, label=f"{rgb_channel} mean", color=rgb_color)
            axis.plot(x, means, linestyle="--", linewidth=2, label="RGB pooled mean", color="black")
            axis.fill_between(x, p05, p95, color="gray", alpha=0.18, label="pooled p05–p95")
            axis.legend(loc="best", ncol=2, fontsize=8)
        else:
            axis.plot(x, means, marker="o", linewidth=2, color=color, label="mean")
            axis.fill_between(x, p05, p95, color=color, alpha=0.16, label="p05–p95")
            axis.legend(loc="best", fontsize=8)
        axis.set_ylabel("value")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[-1].set_xticks(x, labels, rotation=35, ha="right")
    axes[-1].set_xlabel("switch timestep t_s (ascending)")
    if transition.get("t_before") is not None:
        for axis in axes:
            axis.axvspan(transition["t_before"], transition["t_after"], color="orange", alpha=0.18)
    fig.suptitle("Figure 4 — final PBR statistics vs switch timestep")
    fig.tight_layout()
    path = output_dir / "figure4_pbr_statistics_vs_timestep.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["figure4_pbr"] = str(path)
    return paths


def _save_pbr_key_comparison_sheet(
    group_results: Mapping[str, Mapping[str, Any]],
    analysis: Mapping[str, Any],
    output_path: Path,
) -> Optional[str]:
    transition = analysis.get("t_star", {})
    before = transition.get("before_group")
    after = transition.get("after_group")
    full = "tex_local_full"
    key_groups = [name for name in (before, after, full) if name in group_results]
    if len(key_groups) < 3:
        return None
    modes = (("shaded", "render"), ("base_color", "base_color"), ("metallic", "metallic"), ("roughness", "roughness"))
    panel = 320
    header = 72
    sheet = Image.new("RGB", (len(key_groups) * panel, len(modes) * (panel + header)), "black")
    draw = ImageDraw.Draw(sheet)
    for row_index, (title, output_key) in enumerate(modes):
        for col_index, group_name in enumerate(key_groups):
            render = group_results[group_name].get("aligned_render", {})
            paths = render.get("render_outputs", {})
            image_path = Path(str(paths.get(output_key, "/missing")))
            x = col_index * panel
            y = row_index * (panel + header)
            if image_path.is_file():
                with Image.open(image_path) as source:
                    image = ImageOps.contain(source.convert("RGB"), (panel, panel))
                sheet.paste(image, (x + (panel - image.width) // 2, y + header))
            draw.text((x + 6, y + 8), f"{title} | {group_name}", fill=(255, 255, 255))
            if title == "shaded":
                metrics = group_results[group_name].get("input_metrics", {})
                draw.text(
                    (x + 6, y + 30),
                    f"PSNR {metrics.get('psnr_db')} SSIM {metrics.get('ssim')}",
                    fill=(220, 220, 220),
                )
                draw.text((x + 6, y + 51), f"LPIPS {metrics.get('lpips')}", fill=(180, 180, 180))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def _save_key_multiview_comparison_sheet(
    group_results: Mapping[str, Mapping[str, Any]],
    analysis: Mapping[str, Any],
    output_path: Path,
    gif_path: Path,
) -> Optional[str]:
    transition = analysis.get("t_star", {})
    before = transition.get("before_group")
    after = transition.get("after_group")
    key_groups = [name for name in (before, after, "tex_local_full") if name in group_results]
    if len(key_groups) < 3:
        return None
    reference = group_results[key_groups[0]].get("multiview", {})
    baseline_paths = [Path(value) for value in reference.get("baseline_frame_pngs", [])]
    local_paths = {
        name: [Path(value) for value in group_results[name].get("multiview", {}).get("stitched_local_frame_pngs", [])]
        for name in key_groups
    }
    frame_count = len(baseline_paths)
    if frame_count == 0 or any(len(paths) != frame_count for paths in local_paths.values()):
        return None
    panel = 256
    header = 42
    columns = 1 + len(key_groups)
    sheet = Image.new("RGB", (columns * panel, frame_count * (panel + header)), "black")
    draw = ImageDraw.Draw(sheet)
    headers = ["baseline", *key_groups]
    for frame_index in range(frame_count):
        paths = [baseline_paths[frame_index], *[local_paths[name][frame_index] for name in key_groups]]
        for col_index, image_path in enumerate(paths):
            x = col_index * panel
            y = frame_index * (panel + header)
            if image_path.is_file():
                with Image.open(image_path) as source:
                    image = ImageOps.contain(source.convert("RGB"), (panel, panel))
                sheet.paste(image, (x + (panel - image.width) // 2, y + header))
            draw.text((x + 5, y + 10), headers[col_index], fill=(255, 255, 255))
        if frame_index < len(reference.get("pair_metrics", [])):
            label = reference["pair_metrics"][frame_index].get("label", f"view {frame_index}")
            draw.text((5, frame_index * (panel + header) + 27), label, fill=(180, 180, 180))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)

    turntable_start = 6
    gif_frames: List[Image.Image] = []
    for frame_index in range(turntable_start, frame_count):
        paths = [baseline_paths[frame_index], *[local_paths[name][frame_index] for name in key_groups]]
        canvas = Image.new("RGB", (columns * panel, panel), "black")
        for col_index, image_path in enumerate(paths):
            if image_path.is_file():
                with Image.open(image_path) as source:
                    image = ImageOps.contain(source.convert("RGB"), (panel, panel))
                canvas.paste(image, (col_index * panel + (panel - image.width) // 2, 0))
        gif_frames.append(canvas)
    if gif_frames:
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=120, loop=0)
    return str(output_path)


def _write_experiment_report(
    output_dir: Path,
    analysis: Mapping[str, Any],
    figure_paths: Mapping[str, str],
    pbr_sheet: Optional[str],
    multiview_sheet: Optional[str],
) -> str:
    transition = analysis.get("t_star", {})
    pbr_shift = analysis.get("largest_transition_pbr_shift") or "未能判定"
    consistency = analysis.get("input_quality_vs_global_consistency", {})
    lines = [
        "# Pixal3D Texture Flow t* 定位实验报告",
        "",
        "本报告由 CUDA 4 上的固定 shape、固定 LR tile condition、固定 texture noise 和原生 Pixal3D sampler sweep 自动生成。",
        "",
        "## A. t* 位置",
        "",
        f"- 突变区间：`[{transition.get('t_before')}, {transition.get('t_after')}]`。",
        f"- 前后组：`{transition.get('before_group')}` → `{transition.get('after_group')}`。",
        f"- 相邻 PSNR 变化：`{transition.get('input_psnr_delta_db')} dB`。",
        f"- 判定：{transition.get('transition_type')}（选择规则：{transition.get('selection')}）。",
        "",
        "## B. PSNR 变化性质",
        "",
        f"由相邻 native timestep 的最大正向 PSNR 变化判定为：**{transition.get('transition_type')}**。具体数值见 `sweep_analysis.json`。",
        "",
        "## C. t* 前后变化最大的 texture/PBR 属性",
        "",
        f"按最终 stitched mesh 的 aggregate PBR mean shift，最大变化通道为：**{pbr_shift}**。统计同时保存 min/max/mean/std/p05/p50/p95；这表示分布变化证据，不把它解释为唯一因果。",
        "",
        "## D. 输入视角与多视角 baseline 一致性",
        "",
        f"{consistency.get('interpretation')}。同步判定：`{consistency.get('synchronized_at_t_star')}`。多视角指标仅表示相对原始 global baseline 的一致性，不是真实 GT 质量。",
        "",
        "## E. early texture flow 的作用",
        "",
        f"基于 PBR aggregate statistics 的保守归因：**{analysis.get('early_flow_role')}**。该结论只描述本次 endpoint sweep 的观测，不引入任何 trajectory correction、fusion 或新网络。",
        "",
        "## 图表与关键视觉输出",
        "",
    ]
    for key, value in figure_paths.items():
        lines.append(f"- {key}: `{value}`")
    if pbr_sheet:
        lines.append(f"- t* 前后/full-local PBR 对比：`{pbr_sheet}`")
    if multiview_sheet:
        lines.append(f"- baseline 与 t* 前后/full-local 多视角对比：`{multiview_sheet}`")
    lines.extend(["", "## Sweep 数值", "", "| t_s | PSNR (dB) | SSIM | LPIPS | mean multiview PSNR vs baseline | mean multiview SSIM vs baseline |", "|---:|---:|---:|---:|---:|---:|"])
    for row in analysis.get("rows", []):
        metrics = row["input_metrics"]
        multiview = row["multiview_mean_against_baseline"]
        lines.append(
            f"| {row['switch_timestep']:.6f} | {metrics.get('psnr_db')} | {metrics.get('ssim')} | {metrics.get('lpips')} | {multiview.get('psnr_db')} | {multiview.get('ssim')} |"
        )
    path = output_dir / "experiment_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if int(args.cuda_device) < 0:
        raise ValueError("--cuda-device must be non-negative")
    if not args.render:
        raise ValueError("this experiment requires --render")
    if not args.render_multiview:
        raise ValueError("this experiment requires --render-multiview")
    if args.max_tiles is not None and int(args.max_tiles) < 1:
        raise ValueError("--max-tiles must be positive")
    for encoder_path in (args.shape_encoder, args.pbr_encoder):
        base_path = Path(encoder_path).expanduser()
        if not Path(f"{base_path}.json").is_file() or not Path(f"{base_path}.safetensors").is_file():
            raise FileNotFoundError(f"encoder checkpoint pair not found for base path {base_path}")


def _decode_to_patch(
    *,
    pipeline: Any,
    shape_slat: SparseTensor,
    texture_slat: SparseTensor,
    tile_id: int,
    box: Sequence[int],
    global_camera: Mapping[str, float],
    transform: base.TileCameraTransform,
    query_chunk_size: int,
) -> Tuple[base.ReturnedTilePatch, Dict[str, Any]]:
    started = time.perf_counter()
    with torch.no_grad():
        decoded = pipeline.decode_latent(
            shape_slat, texture_slat, base.OVOXEL_RESOLUTION
        )
    base._sync_cuda()
    decode_seconds = time.perf_counter() - started
    if len(decoded) != 1:
        raise RuntimeError("texture group decoder returned more than one mesh")
    mesh = base._validate_mesh(decoded[0], f"tile {tile_id:02d} texture group decode")
    patch = base._local_mesh_to_global_patch(
        tile_id=tile_id,
        box=box,
        local_mesh=mesh,
        global_camera=global_camera,
        transform=transform,
        query_chunk_size=query_chunk_size,
    )
    stats = {
        "decode_seconds": float(decode_seconds),
        "local_vertices": int(mesh.vertices.shape[0]),
        "local_faces": int(mesh.faces.shape[0]),
        "local_active_ovoxels": int(mesh.coords.shape[0]),
        "local_pbr_range": base._tensor_range(mesh.attrs),
        "global_patch": patch.stats,
    }
    del decoded, mesh
    base._empty_cuda_cache()
    return patch, stats


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
    source_image = Image.open(args.image).convert("RGB")
    source_image.save(output_dir / "input_original.png")

    pipeline = base.init_pipeline(
        args.model_path, device="cuda", low_vram=bool(args.low_vram)
    )
    canonical = pipeline.preprocess_canonical_images(Image.open(args.image))
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
    baseline_live = base._validate_mesh(baseline_output[0], "global ordinary Pixal3D-1024 baseline")
    baseline_shape_slat, baseline_texture_slat, decoded_resolution = baseline_latents
    if int(decoded_resolution) != base.OVOXEL_RESOLUTION:
        raise RuntimeError(f"baseline decoder resolution is {decoded_resolution}")
    envmap = base.load_envmap(str(args.envmap), device="cuda")
    baseline_dir = output_dir / "baseline_1024"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_render = base._render(
        baseline_live,
        output_dir=baseline_dir / "aligned_eval",
        camera=global_camera,
        reference_image=output_dir / "canonical_1024.png",
        args=args,
        envmap=envmap,
    )
    baseline_mesh = baseline_live.to("cpu")
    baseline_pbr_statistics = _pbr_statistics(
        baseline_mesh,
        "ordinary global 1024 MeshWithVoxel continuous PBR field",
    )
    baseline_summary = {
        "route": "ordinary pipeline.run(..., pipeline_type='1024_cascade')",
        "generation_seconds": float(baseline_seconds),
        "decoder_resolution": int(decoded_resolution),
        "vertices": int(baseline_mesh.vertices.shape[0]),
        "faces": int(baseline_mesh.faces.shape[0]),
        "active_ovoxels": int(baseline_mesh.coords.shape[0]),
        "shape_slat_tokens": int(baseline_shape_slat.feats.shape[0]),
        "texture_slat_tokens": int(baseline_texture_slat.feats.shape[0]),
        "pbr_statistics": baseline_pbr_statistics,
        "render": base._metric_subset(baseline_render),
        "render_detail": baseline_render,
    }
    base._atomic_json(baseline_dir / "summary.json", baseline_summary)
    del baseline_output, baseline_live, baseline_latents
    del baseline_shape_slat, baseline_texture_slat
    base._empty_cuda_cache()

    print("[global-analysis] projecting baseline mesh and loading encoders")
    face_min, face_max, face_finite = base._project_face_bboxes(
        baseline_mesh.vertices,
        baseline_mesh.faces,
        mesh_scale=float(global_camera["mesh_scale"]),
        global_camera=global_camera,
        chunk_size=int(args.face_projection_chunk_size),
    )
    global_attr_field = base._make_attribute_query_mesh(baseline_mesh, device)
    shape_encoder = base.pixal3d_models.from_pretrained(
        str(Path(args.shape_encoder).expanduser())
    ).eval()
    pbr_encoder = base.pixal3d_models.from_pretrained(
        str(Path(args.pbr_encoder).expanduser())
    ).eval()
    if not args.low_vram:
        shape_encoder.to(device)
        pbr_encoder.to(device)

    # Follow-up quick test layout requested by the user: 4x4 disjoint 1024
    # crops over canonical 4096, so the tile stride equals the tile size.
    boxes = base._tile_layout(stride=base.TILE_SIZE)
    requested_ids = base._parse_tile_ids(args.tile_ids)
    group_patches: Dict[str, List[base.ReturnedTilePatch]] = {
        name: [] for name in GROUPS
    }
    tile_records: List[Dict[str, Any]] = []
    attempted_tiles = 0

    for tile_id, box in enumerate(boxes):
        if requested_ids is not None and tile_id not in requested_ids:
            continue
        if args.max_tiles is not None and attempted_tiles >= int(args.max_tiles):
            break
        attempted_tiles += 1
        tile_dir = output_dir / "tiles" / f"tile_{tile_id:02d}"
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_image_lr = base._make_lr_tile_image(image_1024, box)
        tile_image_lr.save(tile_dir / "tile_lr_condition_reference.png")
        transform = base._derive_tile_camera(
            tile_id=tile_id,
            box=box,
            global_camera=global_camera,
            extend_pixel=int(args.extend_pixel),
        )
        base._atomic_json(tile_dir / "tile_camera.json", transform.__dict__)
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
                "projected_bbox_faces": 0,
                "reason": "no triangle projection bbox intersects tile",
            }
            tile_records.append(record)
            base._write_tile_summary(tile_dir, record)
            continue

        started = time.perf_counter()
        geometry = None
        local_attrs = None
        shape_reference = None
        texture_reference = None
        texture_condition = None
        tile_record: Dict[str, Any] = {
            "status": "failed",
            "tile_id": int(tile_id),
            "box": list(box),
            "projected_bbox_faces": selected_face_count,
            "groups": {},
        }
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
                    f"{geometry.stats['global_local_global_q_max_abs_error']:.3e}"
                )
            local_attrs, material_stats = base._resample_local_attrs_from_global(
                geometry=geometry,
                global_attr_field=global_attr_field,
                global_camera=global_camera,
                transform=transform,
                query_chunk_size=int(args.material_query_chunk_size),
                face_chunk_size=int(args.material_face_chunk_size),
            )
            shape_reference, shape_encoder_stats = base._encode_local_shape(
                encoder=shape_encoder,
                local_coords=geometry.coords,
                local_dual_vertices=geometry.dual_vertices,
                local_intersected=geometry.intersected,
                device=device,
                low_vram=bool(args.low_vram),
            )
            texture_reference, pbr_encoder_stats = base._encode_local_pbr(
                encoder=pbr_encoder,
                coords=geometry.coords,
                attrs=local_attrs,
                device=device,
                low_vram=bool(args.low_vram),
            )
            alignment_stats = base._latent_support_diagnostics(
                shape_reference, texture_reference
            )
            if not alignment_stats["coordinates_exactly_equal"]:
                raise RuntimeError(
                    "fixed shape/G texture encoder supports differ: "
                    + json.dumps(alignment_stats, ensure_ascii=False)
                )
            if int(shape_reference.feats.shape[0]) > int(args.max_num_tokens):
                raise RuntimeError(
                    f"local latent has {shape_reference.feats.shape[0]:,} tokens, "
                    f"exceeding --max-num-tokens={int(args.max_num_tokens):,}"
                )

            # This is the only image condition used by all texture groups.
            # It is the existing LR tile condition, resized to the model's
            # native input size, while shape remains entirely fixed.
            texture_condition = pipeline.get_proj_cond_shape(
                pipeline.image_cond_model_tex_1024,
                [tile_image_lr.convert("RGB")],
                shape_reference.coords.to(torch.int32),
                camera_angle_x=float(transform.camera_angle_x),
                distance=float(transform.distance),
                mesh_scale=float(transform.mesh_scale),
                grid_resolution_override=base.LATENT_RESOLUTION,
            )
            fixed_shape_norm = base._normalize_slat(
                shape_reference, pipeline.shape_slat_normalization
            )
            fixed_shape_denorm = base._denormalize_slat(
                fixed_shape_norm, pipeline.shape_slat_normalization
            )
            g_tex_norm = base._normalize_slat(
                texture_reference, pipeline.tex_slat_normalization
            )
            g_tex_denorm = base._denormalize_slat(
                g_tex_norm, pipeline.tex_slat_normalization
            )
            group_latents, flow_stats = _run_texture_groups(
                pipeline=pipeline,
                fixed_shape_norm=fixed_shape_norm,
                fixed_shape_denorm=fixed_shape_denorm,
                g_tex_norm=g_tex_norm,
                g_tex_denorm=g_tex_denorm,
                condition=texture_condition,
                texture_params=texture_params,
                seed=int(args.seed),
                tile_id=tile_id,
            )

            tile_record.update(
                {
                    "status": "success",
                    "tile_seconds": float(time.perf_counter() - started),
                    "geometry": geometry.stats,
                    "material_resampling": material_stats,
                    "fixed_shape": {
                        "source": "global baseline mesh -> local voxelize -> shape encoder",
                        "encoder": shape_encoder_stats,
                        "tokens": int(fixed_shape_norm.feats.shape[0]),
                        "normalization": "shape_slat_normalization",
                        "used_as_texture_concat_cond": True,
                    },
                    "G_tex": {
                        "source": "global baseline O-voxel local PBR attrs -> local PBR re-encode",
                        "encoder": pbr_encoder_stats,
                        "tokens": int(g_tex_norm.feats.shape[0]),
                        "normalization": "tex_slat_normalization",
                        "used_as_texture_endpoint_anchor": True,
                    },
                    "latent_support": alignment_stats,
                    "texture_condition": {
                        "source": "canonical/global 1024 corresponding 256x256 LR crop resized to 1024",
                        "image": str(tile_dir / "tile_lr_condition_reference.png"),
                        "local_camera": {
                            "camera_angle_x": float(transform.camera_angle_x),
                            "distance": float(transform.distance),
                            "mesh_scale": float(transform.mesh_scale),
                        },
                    },
                    "flow": flow_stats,
                }
            )

            # Decode one group at a time, retaining only CPU global patches.
            # This keeps GPU memory bounded while ensuring every group sees
            # exactly the same fixed shape and local support.
            for group_name in sorted(group_latents, key=_group_sort_key):
                group_dir = tile_dir / group_name
                group_dir.mkdir(parents=True, exist_ok=True)
                group_started = time.perf_counter()
                try:
                    patch, decode_stats = _decode_to_patch(
                        pipeline=pipeline,
                        shape_slat=fixed_shape_denorm,
                        texture_slat=group_latents[group_name],
                        tile_id=tile_id,
                        box=box,
                        global_camera=global_camera,
                        transform=transform,
                        query_chunk_size=int(args.material_query_chunk_size),
                    )
                    group_patches[group_name].append(patch)
                    group_record = {
                        "status": "success",
                        "group": group_name,
                        "tokens": int(group_latents[group_name].feats.shape[0]),
            "texture_stats": flow_stats["groups"][group_name],
                        "decode": decode_stats,
                        "group_seconds": float(time.perf_counter() - group_started),
                    }
                except Exception as exc:
                    group_record = {
                        "status": "failed",
                        "group": group_name,
                        "group_seconds": float(time.perf_counter() - group_started),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                    print(f"[tile {tile_id:02d}] {group_name} FAILED: {group_record['reason']}")
                tile_record["groups"][group_name] = group_record
                base._atomic_json(group_dir / "summary.json", group_record)

            del (
                group_latents,
                fixed_shape_norm,
                fixed_shape_denorm,
                g_tex_norm,
                g_tex_denorm,
            )
            base._empty_cuda_cache()
            print(
                f"[tile {tile_id:02d}] success tokens={shape_reference.feats.shape[0]:,} "
                f"groups={sum(v['status'] == 'success' for v in tile_record['groups'].values())}/{len(GROUPS)} "
                f"seconds={tile_record['tile_seconds']:.2f}"
            )
        except Exception as exc:
            tile_record["reason"] = f"{type(exc).__name__}: {exc}"
            tile_record["tile_seconds"] = float(time.perf_counter() - started)
            print(f"[tile {tile_id:02d}] FAILED: {tile_record['reason']}")
        finally:
            tile_record["tile_seconds"] = float(time.perf_counter() - started)
            tile_records.append(tile_record)
            base._write_tile_summary(tile_dir, tile_record)
            geometry = None
            local_attrs = None
            shape_reference = None
            texture_reference = None
            texture_condition = None
            base._empty_cuda_cache()

    del shape_encoder, pbr_encoder, global_attr_field
    base._empty_cuda_cache()
    successful_rows = [row for row in tile_records if row.get("status") == "success"]
    skipped_rows = [row for row in tile_records if row.get("status") == "skipped"]
    failed_rows = [row for row in tile_records if row.get("status") == "failed"]

    # Save a simple baseline GLB as the sixth mesh artifact.  The renderer and
    # metrics above still use the official O-Voxel field; this export is only a
    # portable geometry/base-color representation.
    baseline_export_attr_field = base._make_attribute_query_mesh(baseline_mesh, device)
    baseline_vertex_attrs = base._query_mesh_attrs_chunked(
        baseline_export_attr_field,
        baseline_mesh.vertices.to(device),
        chunk_size=int(args.material_query_chunk_size),
    ).cpu()
    baseline_patch = base.ReturnedTilePatch(
        tile_id=-1,
        box=(0, 0, base.CANONICAL_IMAGE_SIZE, base.CANONICAL_IMAGE_SIZE),
        vertices=baseline_mesh.vertices.cpu().to(torch.float32),
        faces=baseline_mesh.faces.cpu().to(torch.int32),
        vertex_attrs=baseline_vertex_attrs.to(torch.float32),
        stats={"source": "ordinary global baseline MeshWithVoxel"},
    )
    baseline_glb = base._export_tiled_glb(
        [baseline_patch], baseline_dir / "baseline_1024.glb"
    ) if args.export_glb else {"enabled": False}
    base._atomic_json(baseline_dir / "export.json", baseline_glb)
    del baseline_vertex_attrs, baseline_patch, baseline_export_attr_field
    base._empty_cuda_cache()

    group_results: Dict[str, Dict[str, Any]] = {}
    for group_name in GROUPS:
        patches = group_patches[group_name]
        if not patches:
            group_results[group_name] = {
                "status": "failed",
                "successful_tiles": 0,
                "reason": "no successful tile patches",
            }
            continue
        print(f"[stitch/render] {group_name} patches={len(patches)}")
        if len(boxes) == 16 and base.TILE_STRIDE == base.TILE_SIZE:
            # With a disjoint 4x4 layout there is no overlap ownership to
            # resolve.  Keep every returned tile patch and use the existing
            # direct-concat stitcher; this avoids an unnecessary large CPU
            # spatial weld while preserving the decoded corner attributes.
            stitched, stitch_stats = base._stitch_tile_patches(
                patches,
                layout=baseline_mesh.layout,
            )
            stitch_stats["layout_policy"] = "4x4 disjoint tiles; direct concat without overlap owner/weld"
        else:
            stitched, stitch_stats = base._stitch_tile_patches_nearest(
                patches,
                layout=baseline_mesh.layout,
                global_camera=global_camera,
                face_chunk_size=int(args.face_projection_chunk_size),
                weld_tolerance=float(args.stitch_tolerance),
            )
        group_dir = output_dir / group_name
        stitched_dir = group_dir / "stitched_global_mesh"
        stitched_dir.mkdir(parents=True, exist_ok=True)
        stitched_patch = base.ReturnedTilePatch(
            tile_id=-1,
            box=(0, 0, base.CANONICAL_IMAGE_SIZE, base.CANONICAL_IMAGE_SIZE),
            vertices=stitched.vertices,
            faces=stitched.faces,
            vertex_attrs=stitched.vertex_attrs,
            stats=stitch_stats,
        )
        glb = base._export_tiled_glb(
            [stitched_patch], stitched_dir / f"{group_name}.glb"
        ) if args.export_glb else {"enabled": False}
        overlap = base._save_tile_overlap_visualization(
            image_4096=image_4096,
            boxes=boxes,
            successful_ids=[patch.tile_id for patch in patches],
            output_path=stitched_dir / "tile_overlap_coverage.png",
        )
        aligned_render = base._render(
            stitched,
            output_dir=stitched_dir / "aligned_eval",
            camera=global_camera,
            reference_image=output_dir / "canonical_1024.png",
            args=args,
            envmap=envmap,
        )
        baseline_compare_render = base._render(
            stitched,
            output_dir=stitched_dir / "against_baseline_1024",
            camera=global_camera,
            reference_image=Path(str(baseline_render["render_png"])),
            args=args,
            envmap=envmap,
        )
        multiview = base._render_multiview_comparison(
            baseline_mesh,
            stitched,
            output_dir=group_dir / "multiview_baseline_vs_group",
            camera=global_camera,
            args=args,
            envmap=envmap,
        )
        pbr_statistics = _pbr_statistics(
            stitched,
            f"final stitched MeshWithVertexPbr for {group_name}",
        )
        group_results[group_name] = {
            "status": "success",
            "successful_tiles": int(len(patches)),
            "stitch": stitch_stats,
            "overlap": overlap,
            "glb": glb,
            "input_metrics": base._metric_subset(aligned_render),
            "baseline_metrics": base._metric_subset(baseline_compare_render),
            "aligned_render": aligned_render,
            "against_baseline_render": baseline_compare_render,
            "multiview": multiview,
            "multiview_mean_against_baseline": _mean_multiview_metrics(multiview),
            "pbr_statistics": pbr_statistics,
            "stitched_mesh": str(stitched_dir),
        }
        base._atomic_json(
            stitched_dir / "summary.json", group_results[group_name]
        )
        del stitched, stitched_patch
        base._empty_cuda_cache()

    entries: List[Tuple[Path, str, Optional[Mapping[str, Any]]]] = [
        (output_dir / "canonical_1024.png", "input/canonical_1024", None),
        (
            Path(str(baseline_render["render_png"])),
            "baseline_1024",
            base._metric_subset(baseline_render),
        ),
    ]
    for group_name in GROUPS:
        result = group_results.get(group_name, {})
        render = result.get("aligned_render")
        entries.append(
            (
                Path(str(render["render_png"])) if render else Path("/missing"),
                group_name,
                result.get("input_metrics"),
            )
        )
    comparison_path = output_dir / "comparison_input_baseline_vs_all.png"
    _save_comparison_sheet(entries=entries, output_path=comparison_path)

    analysis = _analyze_sweep(group_results)
    figure_paths = _plot_sweep_artifacts(output_dir, analysis)
    pbr_key_sheet = _save_pbr_key_comparison_sheet(
        group_results,
        analysis,
        output_dir / "comparison_t_star_pbr_channels.png",
    )
    multiview_key_sheet = _save_key_multiview_comparison_sheet(
        group_results,
        analysis,
        output_dir / "comparison_t_star_multiview.png",
        output_dir / "comparison_t_star_multiview_turntable.gif",
    )
    trajectory_summary = _aggregate_texture_trajectories(tile_records)
    base._atomic_json(output_dir / "sweep_analysis.json", analysis)
    base._atomic_json(output_dir / "trajectory_summary.json", trajectory_summary)
    report_path = _write_experiment_report(
        output_dir,
        analysis,
        figure_paths,
        pbr_key_sheet,
        multiview_key_sheet,
    )

    visual_metrics: Dict[str, Any] = {
        "format": "pixal3d_texture_prefix_suffix_visual_metrics_v2",
        "reference_image": str(output_dir / "canonical_1024.png"),
        "camera": global_camera,
        "baseline_1024": base._metric_subset(baseline_render),
        "baseline_pbr_statistics": baseline_pbr_statistics,
        "groups": {
            name: {
                "input_metrics": result.get("input_metrics"),
                "baseline_metrics": result.get("baseline_metrics"),
                "pbr_statistics": result.get("pbr_statistics"),
                "multiview_mean_against_baseline": result.get(
                    "multiview_mean_against_baseline"
                ),
                "multiview_metrics_json": (
                    result.get("multiview", {}).get("metrics_json")
                    if result.get("multiview")
                    else None
                ),
            }
            for name, result in group_results.items()
        },
        "comparison_png": str(comparison_path),
        "pbr_key_comparison_png": pbr_key_sheet,
        "multiview_key_comparison_png": multiview_key_sheet,
        "figure_paths": figure_paths,
        "same_render_policy": {
            "camera": global_camera,
            "envmap": str(args.envmap),
            "render_resolution": int(args.render_resolution),
            "metric_resolution": int(args.metric_resolution),
            "render_ssaa": int(args.render_ssaa),
            "render_peel_layers": int(args.render_peel_layers),
            "multiview_resolution": int(args.multiview_resolution),
            "multiview_ssaa": int(args.multiview_ssaa),
            "multiview_peel_layers": int(args.multiview_peel_layers),
        },
    }
    base._atomic_json(output_dir / "visual_metrics.json", visual_metrics)

    summary = {
        "format": "pixal3d_texture_G_prefix_local_suffix_fixed_shape_native_sweep_v2",
        "image": str(Path(args.image).expanduser().resolve()),
        "cuda_device": int(args.cuda_device),
        "route": {
            "baseline_1024": "ordinary global pipeline 1024_cascade",
            "fixed_shape": "global baseline mesh -> local 1024 dual-grid voxelize -> shape encoder; no shape flow",
            "G_tex": "global baseline O-voxel local PBR attrs -> local PBR re-encode; normalized as texture endpoint anchor",
            "texture_condition": "one LR tile image condition shared by all texture groups on a tile",
            "prefix_suffix": "native_noised_endpoint(G_tex_norm, shared_epsilon, t_s) -> native Euler/CFG suffix to 0",
            "tex_local_full": "native full texture sampler from the same shared t=1 noise state",
            "tile_layout": "4x4 disjoint 1024 crops over canonical 4096; no overlap",
            "stitch": "existing direct-concat stitcher for disjoint tiles; no overlap ownership, welding, or fusion",
            "no_fusion": True,
            "no_cca": True,
            "no_mask": True,
            "no_new_network": True,
        },
        "groups": ["baseline_1024", *GROUPS],
        "switch_times": SWITCH_TIMES,
        "sampler": {
            "shape": shape_params,
            "texture": texture_params,
            "suffix_schedule_policy": "exact slice of native 12-step texture schedule; all requested 1->0.6 points must be native schedule points",
        },
        "global_baseline": baseline_summary,
        "sweep_analysis": analysis,
        "trajectory_summary_json": str(output_dir / "trajectory_summary.json"),
        "report_markdown": report_path,
        "figure_paths": figure_paths,
        "pbr_key_comparison_png": pbr_key_sheet,
        "multiview_key_comparison_png": multiview_key_sheet,
        "successful_tiles": int(len(successful_rows)),
        "failed_tiles": int(len(failed_rows)),
        "skipped_tiles": int(len(skipped_rows)),
        "group_successful_tiles": {
            name: int(len(group_patches[name])) for name in GROUPS
        },
        "visual_metrics_json": str(output_dir / "visual_metrics.json"),
        "comparison_png": str(comparison_path),
        "groups_results": group_results,
        "tiles": tile_records,
    }
    base._atomic_json(output_dir / "summary.json", summary)
    print(
        f"[done] tile_success={len(successful_rows)} tile_failed={len(failed_rows)} "
        f"tile_skipped={len(skipped_rows)} summary={output_dir / 'summary.json'}"
    )


def main() -> None:
    parser = base.build_parser()
    parser.description = __doc__
    args = parser.parse_args()
    if not args.skip_lpips:
        try:
            import lpips  # noqa: F401
        except Exception:
            print("[metrics] lpips package unavailable; continuing without LPIPS")
            args.skip_lpips = True
    run(args)


if __name__ == "__main__":
    main()
