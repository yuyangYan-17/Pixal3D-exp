#!/usr/bin/env python3
"""Global 4096 irregular-master flow with instantaneous ``pred_x_0`` consensus.

This is the executable ``instant_x0_consensus_v2`` route described in
``Codex2.md``.  The older endpoint-rollout script remains available as a
legacy experiment, but this entry point replaces its flow barrier:

    freeze one global master state
    -> one official ``sample_once`` per active tile batch
    -> scatter the returned instantaneous ``pred_x_0`` by stable master ID
    -> FP32 2-D Gaussian consensus
    -> ``_xstart_to_pred`` and one Euler interval

The geometry, camera, sparse encoder, local decoder, renderer and metric
helpers are imported from the already validated tile implementation.  Its
old nested flow function is deliberately not called; the wrapper below
installs the new flow before invoking the shared end-to-end orchestration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image

from pixal3d.modules.sparse import SparseTensor
import pixal3d_global4096_tile_endpoint_rollout_sync as _legacy


FORMAT = "pixal3d_global4096_tile_x0_consensus_sync_instant_x0_consensus_v2"
CANONICAL_SIZE = 4096
GLOBAL_SIZE = 1024
TILE_SIZE = 1024
TILE_STRIDE = 512
TILE_GRID = 7
TILE_COUNT = 49
LOCAL_OVOXEL = 1024
LATENT_SIZE = 64
SIGMA_PIXELS = TILE_SIZE / 4.0
FLOW_BATCH_SIZE = 44
DECODE_BATCH_SIZE = 12
ENCODE_BATCH_SIZE = 13
# The official 13-item profile is retained for shape/PBR encoding.  The
# texture image condition additionally passes through the 1024 NAF branch;
# that branch has a known CUDA illegal-access failure at B=13 on this build,
# so its memory batch is an explicit, recorded B=1 safety setting.  This does
# not change tile condition isolation or any flow barrier semantics.
TEXTURE_CONDITION_BATCH_SIZE = 1
DEFAULT_OUTPUT_DIR = Path(
    "/home/nvme04/yyyan/Pixal3D/outputs/"
    "global4096_tile_x0_consensus_sync_cuda5"
)
DEFAULT_IMAGE = Path("/home/nvme04/yyyan/Pixal3D/assets/images/0_img.png")

# These public aliases make the correctness tests independent of the legacy
# module name.  They are data structures, camera transforms and pure support
# helpers only; no old endpoint sampler is exposed through the run path.
TileView = _legacy.TileView
MasterSupport = _legacy.MasterSupport
SupportCollisionError = _legacy.SupportCollisionError
core = _legacy.core
_tile_layout = _legacy._tile_layout
_inside_box = _legacy._inside_box
_inside_any_box = _legacy._inside_any_box
_c64_coords_from_q = _legacy._c64_coords_from_q
_coord_keys = _legacy._coord_keys
_assert_unique_coords = _legacy._assert_unique_coords
_build_master_support = _legacy._build_master_support
gaussian_weights = _legacy.gaussian_weights
_master_index_coords = _legacy._master_index_coords


def _configure_legacy_orchestration() -> None:
    """Point the shared geometry/decode orchestration at the v2 contract."""

    _legacy.FORMAT = FORMAT
    _legacy.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    # Shape and texture both consume the native 1024 tile crop.  The legacy
    # implementation used this constant for the old 512 shape condition.
    _legacy.SHAPE_IMAGE_SIZE = TILE_SIZE
    _legacy.TEXTURE_CONDITION_BATCH_SIZE = TEXTURE_CONDITION_BATCH_SIZE
    _legacy.FLOW_BATCH_SIZE = FLOW_BATCH_SIZE
    _legacy.DECODE_BATCH_SIZE = DECODE_BATCH_SIZE
    _legacy.ENCODE_BATCH_SIZE = ENCODE_BATCH_SIZE
    _legacy._run_synchronized_endpoint_flow = _run_synchronized_x0_consensus_flow


def _jsonable(value: Any) -> Any:
    return _legacy._jsonable(value)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _legacy._atomic_json(path, payload)


def _atomic_save(path: Path, payload: Any) -> None:
    _legacy._atomic_save(path, payload)


def _tensor_stats(value: torch.Tensor) -> Dict[str, Any]:
    return _legacy._stats_tensor(value)


def _sha256_bytes(hasher: "hashlib._Hash", value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "little"))
    hasher.update(value)


def _hash_tensor(hasher: "hashlib._Hash", name: str, value: torch.Tensor) -> None:
    value = value.detach().cpu().contiguous()
    _sha256_bytes(hasher, name.encode("utf-8"))
    _sha256_bytes(hasher, str(value.dtype).encode("utf-8"))
    _sha256_bytes(hasher, repr(tuple(value.shape)).encode("utf-8"))
    _sha256_bytes(hasher, value.numpy().tobytes())


def _hash_value(hasher: "hashlib._Hash", name: str, value: Any) -> None:
    """Hash nested tensor/config values without reducing them to token counts."""

    _sha256_bytes(hasher, name.encode("utf-8"))
    if isinstance(value, torch.Tensor):
        _hash_tensor(hasher, "tensor", value)
    elif isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            _hash_value(hasher, str(key), value[key])
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _hash_value(hasher, str(index), item)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        _sha256_bytes(hasher, repr(value).encode("utf-8"))
    else:
        _sha256_bytes(hasher, repr(value).encode("utf-8"))


def _flow_input_hash(
    *,
    stage: str,
    master_coords: torch.Tensor,
    initial_features: torch.Tensor,
    views: Mapping[int, TileView],
    conditions: Mapping[int, Mapping[str, Any]],
    sampler_params: Mapping[str, Any],
    schedule: Sequence[float],
    concat_features: Optional[torch.Tensor],
) -> Dict[str, str]:
    """Return independent hashes for support, conditions and initial latents."""

    support = hashlib.sha256()
    _hash_value(support, "algorithm", FORMAT)
    _hash_value(support, "stage", stage)
    _hash_tensor(support, "master_coords", master_coords)
    for tile_id in sorted(views):
        view = views[tile_id]
        _hash_value(support, f"tile_{tile_id}_box", view.box)
        _hash_tensor(support, f"tile_{tile_id}_ids", view.master_ids)
        _hash_tensor(support, f"tile_{tile_id}_coords", view.local_coords)
        _hash_tensor(support, f"tile_{tile_id}_uv", view.master_uv_4096)
        _hash_tensor(support, f"tile_{tile_id}_weight", view.gaussian_weight)

    condition = hashlib.sha256()
    _hash_value(condition, "stage", stage)
    _hash_value(condition, "sampler_params", dict(sampler_params))
    _hash_value(condition, "schedule", list(schedule))
    for tile_id in sorted(conditions):
        _hash_value(condition, f"condition_{tile_id}", conditions[tile_id])

    latent = hashlib.sha256()
    _hash_tensor(latent, "initial_features", initial_features)
    if concat_features is not None:
        _hash_tensor(latent, "concat_features", concat_features)

    return {
        "support_sha256": support.hexdigest(),
        "condition_sha256": condition.hexdigest(),
        "latent_sha256": latent.hexdigest(),
    }


def _validate_view_conditions(
    views: Sequence[TileView], conditions: Mapping[int, Mapping[str, Any]]
) -> None:
    for view in views:
        if view.tile_id not in conditions:
            raise KeyError(f"missing condition for active tile {view.tile_id}")
        condition = conditions[view.tile_id]
        if not torch.equal(condition["coords"].to(torch.int32), view.local_coords):
            raise RuntimeError(
                f"tile {view.tile_id}: cached condition coordinates differ from tile view"
            )
        for branch in ("cond", "neg_cond"):
            if branch not in condition:
                raise RuntimeError(f"tile {view.tile_id}: missing {branch}")
            if "global" not in condition[branch] or "proj" not in condition[branch]:
                raise RuntimeError(f"tile {view.tile_id}: incomplete {branch}")


def _safe_cosine_mean(
    values: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor, count: int
) -> torch.Tensor:
    norms = values.float().norm(dim=1, keepdim=True).clamp_min(1e-12)
    unit = values.float() / norms
    result = torch.zeros((count, values.shape[1]), dtype=torch.float32)
    result.index_add_(0, ids, unit * weights[:, None])
    return result


def merge_pred_x0_contributions(
    num_master: int,
    contributions: Iterable[Mapping[str, Any]],
    *,
    disagreement_threshold: float = 0.25,
    disagreement_temperature: float = 0.10,
) -> Dict[str, torch.Tensor]:
    """Merge current-time predictions and produce condition controls.

    ``contributions`` contains only values returned by the current frozen
    ``sample_once`` calls.  No endpoint rollout or feature initialization is
    performed here.  All accumulators are CPU FP32 tensors so the operation is
    deterministic up to the explicitly recorded contribution order.
    """

    rows = list(contributions)
    if num_master <= 0:
        raise ValueError("num_master must be positive")
    if not rows:
        raise ValueError("at least one tile contribution is required")
    channels = int(rows[0]["pred_x0"].shape[1])
    sum_w_x0 = torch.zeros((num_master, channels), dtype=torch.float32)
    sum_w = torch.zeros((num_master, 1), dtype=torch.float32)
    sum_uniform = torch.zeros_like(sum_w_x0)
    sum_w_norm2 = torch.zeros((num_master, 1), dtype=torch.float32)
    sum_w_unit = torch.zeros_like(sum_w_x0)
    contribution_count = torch.zeros((num_master, 1), dtype=torch.int32)
    hard_weight = torch.full((num_master, 1), -float("inf"), dtype=torch.float32)
    hard_x0 = torch.zeros_like(sum_w_x0)
    hard_tile = torch.full((num_master,), -1, dtype=torch.int16)
    single_x0 = torch.zeros_like(sum_w_x0)

    normalized_rows: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
    for row in rows:
        ids = row["master_ids"].detach().cpu().to(torch.int64).contiguous()
        values = row["pred_x0"].detach().cpu().to(torch.float32).contiguous()
        weights = row["weight"].detach().cpu().to(torch.float32).reshape(-1)
        tile_id = int(row.get("tile_id", 0))
        if values.ndim != 2 or values.shape[0] != ids.numel():
            raise ValueError("pred_x0 rows and master IDs are not aligned")
        if weights.shape[0] != ids.numel():
            raise ValueError("Gaussian weights and master IDs are not aligned")
        if bool((ids < 0).any()) or bool((ids >= num_master).any()):
            raise IndexError("master ID is outside the global state table")
        if not torch.isfinite(values).all() or not torch.isfinite(weights).all():
            raise FloatingPointError("non-finite current pred_x0 contribution")
        if bool((weights <= 0).any()):
            raise ValueError("all active Gaussian contribution weights must be positive")
        normalized_rows.append((ids, values, weights, tile_id))
        contribution_count.index_add_(0, ids, torch.ones((ids.numel(), 1), dtype=torch.int32))
        sum_w.index_add_(0, ids, weights[:, None])
        sum_w_x0.index_add_(0, ids, values * weights[:, None])
        sum_uniform.index_add_(0, ids, values)
        sum_w_norm2.index_add_(0, ids, values.square().sum(dim=1, keepdim=True) * weights[:, None])
        norms = values.norm(dim=1, keepdim=True).clamp_min(1e-12)
        sum_w_unit.index_add_(0, ids, values / norms * weights[:, None])
        better = weights > hard_weight.index_select(0, ids)[:, 0]
        if bool(better.any()):
            chosen_ids = ids[better]
            hard_weight[chosen_ids, 0] = weights[better]
            hard_x0[chosen_ids] = values[better]
            hard_tile[chosen_ids] = tile_id
        single_x0.index_copy_(0, ids, values)

    if bool((sum_w <= 0).any()):
        missing = torch.where(sum_w[:, 0] <= 0)[0][:32].tolist()
        raise RuntimeError(f"uncovered master IDs in x0 consensus: {missing}")

    gaussian = sum_w_x0 / sum_w
    # Preserve a single contribution exactly, rather than introducing a
    # needless multiply/divide roundoff in the identity case.
    gaussian[contribution_count[:, 0] == 1] = single_x0[contribution_count[:, 0] == 1]
    uniform = sum_uniform / contribution_count.to(torch.float32).clamp_min(1.0)
    hard = hard_x0.clone()

    weighted_second = sum_w_norm2 / sum_w
    center_norm2 = gaussian.square().sum(dim=1, keepdim=True)
    pairwise_rms = torch.sqrt((2.0 * (weighted_second - center_norm2)).clamp_min(0.0))[:, 0]
    pairwise_cosine = (
        (sum_w_unit / sum_w).square().sum(dim=1).clamp(-1.0, 1.0)
    )
    relative_rms = pairwise_rms / gaussian.norm(dim=1).clamp_min(1e-6)

    # The gate is fixed before quality metrics are seen.  It is a diagnostic
    # control; the Gaussian result above remains the main state update.
    gated_sum = torch.zeros_like(sum_w_x0)
    gated_weight = torch.zeros_like(sum_w)
    rejected_weight = torch.zeros_like(sum_w)
    for ids, values, weights, _tile_id in normalized_rows:
        center = gaussian.index_select(0, ids)
        residual = (values - center).norm(dim=1) / center.norm(dim=1).clamp_min(1e-6)
        gate = torch.sigmoid(
            (float(disagreement_threshold) - residual)
            / max(float(disagreement_temperature), 1e-6)
        )
        effective = weights * gate
        gated_sum.index_add_(0, ids, values * effective[:, None])
        gated_weight.index_add_(0, ids, effective[:, None])
        rejected_weight.index_add_(0, ids, weights[:, None] * (gate < 0.5).to(torch.float32)[:, None])
    gated = torch.where(gated_weight > 0, gated_sum / gated_weight.clamp_min(1e-12), gaussian)

    return {
        "gaussian_pred_x0": gaussian,
        "uniform_pred_x0": uniform,
        "hard_center_owner_pred_x0": hard,
        "disagreement_gated_pred_x0": gated,
        "hard_center_owner_tile_id": hard_tile,
        "sum_weight": sum_w,
        "participant_count": contribution_count[:, 0],
        "pairwise_rms": pairwise_rms,
        "pairwise_cosine": pairwise_cosine,
        "relative_rms": relative_rms,
        "gated_weight": gated_weight,
        "gated_rejected_weight": rejected_weight,
        "gated_rejected_fraction": rejected_weight / sum_w.clamp_min(1e-12),
    }


def _disagreement_summary(merged: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    def stats(name: str) -> Dict[str, Any]:
        value = merged[name].detach().float().reshape(-1)
        return {"name": name, **_tensor_stats(value)}

    count = merged["participant_count"].detach().cpu()
    histogram = {
        str(n): int((count == n).sum())
        for n in sorted(int(v) for v in torch.unique(count).tolist())
    }
    return {
        "pairwise_pred_x0_rms": stats("pairwise_rms"),
        "pairwise_pred_x0_cosine": stats("pairwise_cosine"),
        "relative_rms": stats("relative_rms"),
        "participant_count_histogram": histogram,
        "gated_rejected_fraction": stats("gated_rejected_fraction"),
        "gating": {
            "threshold_relative_rms": 0.25,
            "temperature_relative_rms": 0.10,
            "main_state_uses": "gaussian_pred_x0",
        },
    }


def _effective_cfg_branches(params: Mapping[str, Any], t: float) -> int:
    guidance = float(params.get("guidance_strength", 1.0))
    interval = params.get("guidance_interval")
    if interval is not None and not (float(interval[0]) <= t <= float(interval[1])):
        guidance = 1.0
    return 2 if guidance not in (0.0, 1.0) else 1


@torch.no_grad()
def _run_synchronized_x0_consensus_flow(
    *,
    stage: str,
    initial_features: torch.Tensor,
    master_coords: torch.Tensor,
    views: Mapping[int, TileView],
    conditions: Mapping[int, Mapping[str, Any]],
    sampler: Any,
    model: torch.nn.Module,
    sampler_params: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    flow_batch_size: int = FLOW_BATCH_SIZE,
    concat_features: Optional[torch.Tensor] = None,
    resume: bool = False,
    save_step_tensors: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Run the all-tile instantaneous-x0 consensus flow.

    There is intentionally no inner/suffix loop in this function.  The only
    sampler call in a batch is for the current ``(frozen, t_k, t_{k+1})`` and
    only ``out.pred_x_0``/``out.pred_v`` from that call enter the barrier.
    """

    if initial_features.ndim != 2:
        raise ValueError("initial_features must have shape [N,C]")
    if master_coords.ndim != 2 or master_coords.shape != (initial_features.shape[0], 4):
        raise ValueError("master_coords must be [N,4] aligned with initial_features")
    if flow_batch_size <= 0:
        raise ValueError("flow_batch_size must be positive")
    active = [views[key] for key in sorted(views)]
    if not active:
        raise RuntimeError(f"{stage}: no active tile views")
    _validate_view_conditions(active, conditions)
    steps = int(sampler_params["steps"])
    rescale_t = float(sampler_params.get("rescale_t", 1.0))
    times = tuple(float(v) for v in sampler.timestep_schedule(steps, rescale_t))
    if len(times) != steps + 1:
        raise RuntimeError(f"{stage}: sampler returned {len(times)} times for {steps} steps")

    stage_dir = Path(output_dir) / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    hashes = _flow_input_hash(
        stage=stage,
        master_coords=master_coords,
        initial_features=initial_features,
        views=views,
        conditions=conditions,
        sampler_params=sampler_params,
        schedule=times,
        concat_features=concat_features,
    )
    checkpoint = stage_dir / "checkpoint.pt"
    state = initial_features.detach().cpu().to(torch.float32).contiguous()
    start_step = 0
    records: List[Dict[str, Any]] = []
    if resume and checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for key in ("format", "support_sha256", "condition_sha256", "latent_sha256"):
            if saved.get(key) != (FORMAT if key == "format" else hashes[key]):
                raise RuntimeError(
                    f"{stage}: resume checkpoint rejected because {key} changed; "
                    "use --no-resume for a fresh v2 run"
                )
        if int(saved.get("steps", -1)) != steps or list(saved.get("schedule", [])) != list(times):
            raise RuntimeError(f"{stage}: resume checkpoint rejected because sampler schedule changed")
        state = saved["state"].detach().cpu().to(torch.float32).contiguous()
        start_step = int(saved["next_step"])
        records = list(saved.get("records", []))
        if start_step < 0 or start_step > steps:
            raise RuntimeError(f"{stage}: invalid checkpoint next_step={start_step}")

    _atomic_save(
        stage_dir / "initial_noise.pt",
        {"format": FORMAT, "stage": stage, "coords": master_coords.cpu(), "features": initial_features.cpu(), "hashes": hashes},
    )
    model.to(device)
    model.eval()
    try:
        for step in range(start_step, steps):
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            frozen = state.clone()
            contributions: List[Dict[str, Any]] = []
            tile_records: List[Dict[str, Any]] = []
            physical_batches = 0
            physical_forwards = 0
            logical_predictions = 0
            current_t = times[step]
            next_t = times[step + 1]
            for group_start in range(0, len(active), int(flow_batch_size)):
                group = active[group_start : group_start + int(flow_batch_size)]
                physical_batches += 1
                local, ids_concat = _legacy._pack_state_batch(group, frozen, device)
                condition = _legacy._pack_flow_condition(group, conditions, local.coords, device)
                concat = None
                if concat_features is not None:
                    concat, _ = _legacy._pack_state_batch(group, concat_features.cpu(), device)

                # Exactly one official current-time prediction for this real
                # [B,...] tile batch.  No later-state sampler field is read.
                out = sampler.sample_once(
                    model,
                    local,
                    current_t,
                    next_t,
                    cond=condition["cond"],
                    neg_cond=condition["neg_cond"],
                    concat_cond=concat,
                    **_legacy._prediction_kwargs(sampler_params),
                )
                if not hasattr(out, "pred_x_0") or not hasattr(out, "pred_v"):
                    raise RuntimeError(f"{stage}: sampler output lacks pred_x_0/pred_v")
                if not isinstance(out.pred_x_0, SparseTensor) or not isinstance(out.pred_v, SparseTensor):
                    raise RuntimeError(f"{stage}: sparse model output lost SparseTensor structure")
                x0_parts = _legacy._unpack_state_parts(out.pred_x_0, group, f"{stage} pred_x_0")
                v_parts = _legacy._unpack_state_parts(out.pred_v, group, f"{stage} pred_v")
                state_parts = _legacy._split_sparse_batch(local, len(group), f"{stage} frozen state")
                branch_count = _effective_cfg_branches(sampler_params, current_t)
                physical_forwards += branch_count
                logical_predictions += len(group)
                for view, state_part, x0_part, v_part in zip(group, state_parts, x0_parts, v_parts):
                    ids = view.master_ids.clone()
                    x0_values = x0_part.feats.detach().cpu().float()
                    v_values = v_part.feats.detach().cpu().float()
                    contributions.append(
                        {"tile_id": view.tile_id, "master_ids": ids, "pred_x0": x0_values, "pred_v": v_values, "weight": view.gaussian_weight.clone()}
                    )
                    # This is a same-call algebraic check, not a suffix rollout.
                    direct_x0 = sampler._pred_to_xstart(state_part, current_t, v_part).feats.detach().cpu().float()
                    direct_error = (direct_x0 - x0_values).abs()
                    tile_records.append(
                        {
                            "tile_id": view.tile_id,
                            "token_count": int(ids.numel()),
                            "weight": _tensor_stats(view.gaussian_weight),
                            "pred_x_0": _tensor_stats(x0_values),
                            "pred_v": _tensor_stats(v_values),
                            "same_call_pred_x0_formula_max_abs": float(direct_error.max()) if direct_error.numel() else 0.0,
                        }
                    )
                del out, local, condition, concat, x0_parts, v_parts, state_parts
                _legacy._empty_cuda_cache()

            merged = merge_pred_x0_contributions(len(state), contributions)
            weighted_v = torch.zeros_like(merged["gaussian_pred_x0"])
            for contribution in contributions:
                weighted_v.index_add_(
                    0,
                    contribution["master_ids"],
                    contribution["pred_v"] * contribution["weight"][:, None],
                )
            weighted_v = weighted_v / merged["sum_weight"].clamp_min(1e-12)
            frozen_sparse = SparseTensor(frozen, master_coords.cpu())
            merged_sparse = SparseTensor(merged["gaussian_pred_x0"], master_coords.cpu())
            effective_velocity = sampler._xstart_to_pred(frozen_sparse, current_t, merged_sparse)
            if not isinstance(effective_velocity, SparseTensor):
                raise RuntimeError(f"{stage}: sampler._xstart_to_pred did not preserve sparse state")
            effective_v = effective_velocity.feats.detach().cpu().float()
            equivalence_error = effective_v - weighted_v
            max_equivalence_error = float(equivalence_error.abs().max()) if equivalence_error.numel() else 0.0
            rms_equivalence_error = float(torch.sqrt(equivalence_error.square().mean())) if equivalence_error.numel() else 0.0
            scale = max(1.0, float(effective_v.abs().max()))
            if max_equivalence_error > 2e-4 * scale:
                raise RuntimeError(
                    f"{stage} step {step}: x0 consensus/current-v consensus mismatch "
                    f"max={max_equivalence_error:.6g} scale={scale:.6g}"
                )
            state = frozen - float(current_t - next_t) * effective_v
            if not torch.isfinite(state).all():
                raise FloatingPointError(f"{stage} step {step}: non-finite global state")

            disagreement_payload = {
                key: value
                for key, value in merged.items()
                if key in {"participant_count", "pairwise_rms", "pairwise_cosine", "relative_rms", "gated_rejected_fraction", "sum_weight", "hard_center_owner_tile_id"}
            }
            step_dir = stage_dir / f"step_{step:02d}"
            if save_step_tensors:
                _atomic_save(step_dir / "global_state.pt", {"format": FORMAT, "coords": master_coords.cpu(), "features": state})
                _atomic_save(step_dir / "merged_pred_x0.pt", {"format": FORMAT, "coords": master_coords.cpu(), "features": merged["gaussian_pred_x0"]})
                _atomic_save(step_dir / "sum_weight.pt", merged["sum_weight"])
                _atomic_save(step_dir / "condition_disagreement.pt", disagreement_payload)
                _atomic_save(step_dir / "control_uniform_pred_x0.pt", {"coords": master_coords.cpu(), "features": merged["uniform_pred_x0"]})
                _atomic_save(step_dir / "control_hard_center_owner_pred_x0.pt", {"coords": master_coords.cpu(), "features": merged["hard_center_owner_pred_x0"]})
                _atomic_save(step_dir / "control_disagreement_gated_pred_x0.pt", {"coords": master_coords.cpu(), "features": merged["disagreement_gated_pred_x0"]})
            disagreement_json = _disagreement_summary(merged)
            _atomic_json(step_dir / "effective_velocity_stats.json", _tensor_stats(effective_v))
            _atomic_json(step_dir / "per_tile_pred_x0_stats.json", {"tiles": tile_records})
            _atomic_json(
                step_dir / "x0_velocity_equivalence.json",
                {
                    "formula": "_xstart_to_pred(frozen, t, weighted_pred_x0) == weighted_current_pred_v",
                    "max_abs_error": max_equivalence_error,
                    "rms_error": rms_equivalence_error,
                    "tolerance": 2e-4 * scale,
                },
            )
            _atomic_json(step_dir / "condition_disagreement.json", disagreement_json)
            seconds = float(time.perf_counter() - started)
            peak_allocated = int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() and device.type == "cuda" else None
            peak_reserved = int(torch.cuda.max_memory_reserved(device)) if torch.cuda.is_available() and device.type == "cuda" else None
            step_record = {
                "step": step,
                "t": current_t,
                "t_next": next_t,
                "dt": float(current_t - next_t),
                "prediction_semantics": "instantaneous out.pred_x_0 at frozen x_t",
                "suffix_rollout_used": False,
                "inner_prediction_count": 0,
                "active_tile_count": len(active),
                "logical_tile_predictions": logical_predictions,
                "physical_tile_batches": physical_batches,
                "physical_model_forwards": physical_forwards,
                "flow_batch_size": int(flow_batch_size),
                "frozen_state": _tensor_stats(frozen),
                "merged_pred_x0": _tensor_stats(merged["gaussian_pred_x0"]),
                "uniform_control_pred_x0": _tensor_stats(merged["uniform_pred_x0"]),
                "hard_center_owner_control_pred_x0": _tensor_stats(merged["hard_center_owner_pred_x0"]),
                "disagreement_gated_control_pred_x0": _tensor_stats(merged["disagreement_gated_pred_x0"]),
                "effective_velocity": _tensor_stats(effective_v),
                "sum_weight": _tensor_stats(merged["sum_weight"]),
                "disagreement": disagreement_json,
                "tile_pred_x0": tile_records,
                "seconds": seconds,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
            }
            records.append(step_record)
            _atomic_save(
                checkpoint,
                {
                    "format": FORMAT,
                    "stage": stage,
                    **hashes,
                    "steps": steps,
                    "schedule": list(times),
                    "next_step": step + 1,
                    "state": state,
                    "records": records,
                    "sampler_params": dict(sampler_params),
                    "rng_state": _legacy._capture_rng_state(device),
                },
            )
    finally:
        model.cpu()
        _legacy._empty_cuda_cache()

    summary = {
        "format": FORMAT,
        "stage": stage,
        "steps": steps,
        "schedule": list(times),
        "suffix_rollout_used": False,
        "predictions_per_active_tile": steps,
        "logical_tile_predictions": steps * len(active),
        "physical_tile_batches": sum(int(r["physical_tile_batches"]) for r in records),
        "actual_model_forwards": sum(int(r["physical_model_forwards"]) for r in records),
        "active_tile_count": len(active),
        "flow_batch_size": int(flow_batch_size),
        "records": records,
        "hashes": hashes,
        "sampler_params": dict(sampler_params),
        "prediction_contract": "one official sample_once per active tile batch and timestep; only current pred_x_0 enters Gaussian barrier",
        "global_update": "sampler._xstart_to_pred(frozen, t_k, merged_pred_x0); frozen - (t_k-t_next)*v_eff",
        "serial_tile_fallback": False,
    }
    _atomic_json(stage_dir / "flow_summary.json", summary)
    _atomic_save(stage_dir / "final_state.pt", {"format": FORMAT, "coords": master_coords.cpu(), "features": state})
    return state, summary


def _write_v2_artifact_aliases(output_dir: Path) -> None:
    """Add the exact top-level artifact names requested by Codex2."""

    support_dir = output_dir / "support"
    for name in (
        "support_owner_map_4096.png",
        "support_density_map_4096.png",
        "support_overlap_count_map_4096.png",
        "support_collision_report.json",
    ):
        source = support_dir / name
        target = output_dir / name
        if source.is_file() and not target.is_file():
            target.write_bytes(source.read_bytes())


def validate_native_4096_outputs(
    output_dir: Path, *, require_all: bool = False
) -> Dict[str, Any]:
    """Validate that emitted image/depth artifacts are genuinely 4096x4096.

    The check is intentionally performed on file pixels, not only on a
    renderer option stored in JSON.  It is used after a rendered run and is
    also a small standalone correctness gate for tests.
    """

    final_dir = Path(output_dir) / "final"
    image_names = (
        "final_render_rgb_4096.png",
        "final_render_alpha_4096.png",
        "final_render_normal_camera_4096.png",
        "final_render_normal_world_4096.png",
        "final_pbr_base_color_4096.png",
        "final_pbr_metallic_4096.png",
        "final_pbr_roughness_4096.png",
        "final_pbr_alpha_4096.png",
    )
    checked: Dict[str, Any] = {"resolution": [CANONICAL_SIZE, CANONICAL_SIZE], "images": {}}
    for name in image_names:
        path = final_dir / name
        if not path.is_file():
            if require_all:
                raise FileNotFoundError(f"missing required native 4096 artifact: {path}")
            continue
        with Image.open(path) as image:
            size = list(image.size)
        if size != [CANONICAL_SIZE, CANONICAL_SIZE]:
            raise RuntimeError(f"{path}: expected native 4096x4096 pixels, got {size}")
        checked["images"][name] = size
    depth_path = final_dir / "final_depth_4096.pt"
    if depth_path.is_file():
        payload = torch.load(depth_path, map_location="cpu", weights_only=False)
        depth = payload.get("depth_camera_positive", payload) if isinstance(payload, Mapping) else payload
        if not isinstance(depth, torch.Tensor) or tuple(depth.shape[-2:]) != (CANONICAL_SIZE, CANONICAL_SIZE):
            raise RuntimeError(f"{depth_path}: expected native 4096 depth tensor, got {getattr(depth, 'shape', None)}")
        checked["depth"] = list(depth.shape)
    elif require_all:
        raise FileNotFoundError(f"missing required native 4096 depth artifact: {depth_path}")
    return checked


def build_parser() -> argparse.ArgumentParser:
    # Building a parser must be side-effect free.  In particular, the legacy
    # module is also imported by its own regression tests; do not replace its
    # sampler function merely because a v2 parser was requested.
    parser = _legacy.build_parser()
    parser.description = __doc__
    # The parser's actions hold the defaults at construction time; explicitly
    # update the visible v2 defaults as well as the module constants.
    for action in parser._actions:
        if action.dest == "output_dir":
            action.default = DEFAULT_OUTPUT_DIR
        elif action.dest == "encode_batch_size":
            action.default = ENCODE_BATCH_SIZE
        elif action.dest == "flow_batch_size":
            action.default = FLOW_BATCH_SIZE
        elif action.dest == "decode_batch_size":
            action.default = DECODE_BATCH_SIZE
    parser.add_argument(
        "--disagreement-threshold",
        type=float,
        default=0.25,
        help="fixed relative-RMS threshold recorded for the diagnostic gated control",
    )
    parser.add_argument(
        "--disagreement-temperature",
        type=float,
        default=0.10,
        help="fixed sigmoid temperature recorded for the diagnostic gated control",
    )
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Run Path A + the v2 Path B orchestration on the requested CUDA device."""

    _configure_legacy_orchestration()
    if not (0.0 < float(args.disagreement_temperature)):
        raise ValueError("--disagreement-temperature must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    old_summary = output_dir / "summary.json"
    if old_summary.is_file():
        payload = json.loads(old_summary.read_text(encoding="utf-8"))
        if payload.get("format") not in (None, FORMAT):
            raise RuntimeError(
                f"refusing to reuse non-v2 output {old_summary}; choose a fresh output directory"
            )
    for cache_name in ("final/final_per_vertex_pbr_mesh.pt", "final/final_per_face_pbr_mesh.pt"):
        cache_path = output_dir / cache_name
        if not cache_path.is_file():
            continue
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if isinstance(cache, Mapping) and cache.get("format") != FORMAT:
            raise RuntimeError(
                f"refusing to reuse incompatible decode cache {cache_path}; choose a fresh output directory"
            )
    # The current merge helper uses the fixed values in the v2 contract.  Keep
    # CLI values in the manifest so a future control implementation can use
    # them without making the main Gaussian state metric-dependent.
    summary = _legacy.run(args)
    _write_v2_artifact_aliases(output_dir)
    if bool(getattr(args, "render", False)):
        summary["native_4096_outputs"] = validate_native_4096_outputs(
            output_dir, require_all=True
        )
    summary["format"] = FORMAT
    summary["algorithm"] = {
        "name": FORMAT,
        "instantaneous_pred_x0": True,
        "suffix_rollout_used": False,
        "disagreement_threshold": float(args.disagreement_threshold),
        "disagreement_temperature": float(args.disagreement_temperature),
        "shape_tile_condition_resolution": [TILE_SIZE, TILE_SIZE],
        "texture_image_condition_batch": TEXTURE_CONDITION_BATCH_SIZE,
        "pbr_encode_batch_profile": ENCODE_BATCH_SIZE,
        "cuda_policy": "physical CUDA 5 exposed as logical cuda:0",
    }
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_json(
        output_dir / "config.json",
        {
            "format": FORMAT,
            "args": vars(args),
            "algorithm": summary["algorithm"],
            "batch_profile": {
                "flow": FLOW_BATCH_SIZE,
                "decode": DECODE_BATCH_SIZE,
                "pbr_encode": ENCODE_BATCH_SIZE,
                "texture_image_condition": TEXTURE_CONDITION_BATCH_SIZE,
                "model_batch_semantics": "one isolated tile condition per sparse batch row",
                "texture_batch_retry": "B=1 required after CUDA NAF B=13 illegal-memory-access failure",
            },
        },
    )
    return summary


def main() -> int:
    args = build_parser().parse_args()
    summary = run(args)
    print(f"[done] {summary.get('status', 'complete')} output={Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
