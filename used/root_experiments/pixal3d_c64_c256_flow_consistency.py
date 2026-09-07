#!/usr/bin/env python3
"""Diagnose coarse/fine consistency of matched C64 and C256 shape flows.

This is deliberately an inference-only experiment.  It consumes an existing
global C256 sparse support, derives C64 exclusively by integer division, and
runs the unmodified Pixal3D shape model/sampler on both supports.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor


FORMAT = "pixal3d_c64_c256_tiled_flow_consistency_v2"
DEFAULT_CASE = Path(
    "outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/"
    "exp_c_baseline4096_from1024"
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def tensor_digest(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_coords(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    coords = payload.get("coords") if isinstance(payload, Mapping) else payload
    if not isinstance(coords, torch.Tensor) or coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"invalid C256 support in {path}: expected [N,3/4] tensor")
    if coords.shape[1] == 3:
        coords = torch.cat((torch.zeros_like(coords[:, :1]), coords), dim=1)
    coords = coords.to(torch.int64)
    if torch.any(coords[:, 0] != 0):
        raise ValueError("only a single batch (batch id zero) is supported")
    if torch.any(coords[:, 1:] < 0) or torch.any(coords[:, 1:] >= 256):
        raise ValueError("C256 coordinates must lie in [0, 255]")
    if torch.unique(coords, dim=0).shape[0] != coords.shape[0]:
        raise ValueError("C256 support contains duplicate coordinates")
    return coords.to(torch.int32)


def build_parent_mapping(coords256: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return C64 coords, fine->coarse indices and CSR child offsets."""
    coarse_xyz = torch.div(coords256[:, 1:].to(torch.int64), 4, rounding_mode="floor")
    unique_xyz, inverse = torch.unique(coarse_xyz, dim=0, sorted=True, return_inverse=True)
    coords64 = torch.cat((torch.zeros_like(unique_xyz[:, :1]), unique_xyz), dim=1).to(torch.int32)
    counts = torch.bincount(inverse, minlength=coords64.shape[0])
    if inverse.numel() != coords256.shape[0] or torch.any(counts <= 0):
        raise RuntimeError("parent mapping is not total")
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    return coords64, inverse.to(torch.int64), offsets


def build_c256_tiles(
    coords256: torch.Tensor,
    tile_size: int = 64,
    stride: int = 64,
) -> Tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Partition C4096 into non-overlapping 1024-voxel cubes.

    One C256 latent cell spans 16 output voxels, hence a 1024-voxel cube is a
    64-cell local support.  The physical centre translation and x4 scale are
    algebraically identical to ``local_xyz = global_xyz - tile_start``.
    """
    if tile_size != 64 or stride != 64:
        raise ValueError("v2 requires C256 tile_size=stride=64 (1024 output voxels)")
    xyz = coords256[:, 1:].to(torch.int64).cpu()
    starts = list(range(0, 256, stride))
    records: list[Dict[str, Any]] = []
    writes = torch.zeros(coords256.shape[0], dtype=torch.int16)
    tile_id = 0
    for sx in starts:
        for sy in starts:
            for sz in starts:
                start = torch.tensor([sx, sy, sz], dtype=torch.int64)
                inside = ((xyz >= start) & (xyz < start + tile_size)).all(1)
                ids = torch.where(inside)[0].to(torch.int64)
                local = (xyz.index_select(0, ids) - start).to(torch.int32)
                global_q = 2.0 * (xyz.index_select(0, ids).double() + 0.5) / 256.0 - 1.0
                centre_q = 2.0 * (start.double() + 32.0) / 256.0 - 1.0
                local_q = (global_q - centre_q) * 4.0
                transformed = torch.round((local_q + 1.0) * 32.0 - 0.5).to(torch.int32)
                if not torch.equal(transformed, local):
                    raise RuntimeError(f"tile {tile_id} translate/scale mismatch")
                if ids.numel() and (int(local.min()) < 0 or int(local.max()) >= 64):
                    raise RuntimeError(f"tile {tile_id} local index outside C64")
                writes.index_add_(0, ids, torch.ones_like(ids, dtype=torch.int16))
                local_coords = torch.cat((torch.zeros((ids.numel(), 1), dtype=torch.int32), local), 1)
                records.append({
                    "tile_id": tile_id,
                    "start_c256": (sx, sy, sz),
                    "start_voxel4096": (sx * 16, sy * 16, sz * 16),
                    "global_row_ids": ids,
                    "local_coords": local_coords,
                    "tokens": int(ids.numel()),
                    "transform": "subtract tile centre in canonical q, scale x4, quantize to local C64; exact integer form global_xyz-start",
                })
                tile_id += 1
    if len(records) != 64 or not torch.all(writes == 1):
        raise RuntimeError("64-tile half-open partition must cover every C256 row exactly once")
    token_counts = torch.tensor([record["tokens"] for record in records], dtype=torch.float32)
    stats = {
        "voxel_grid": 4096,
        "tile_voxels": 1024,
        "stride_voxels": 1024,
        "latent_grid": 256,
        "local_latent_grid": 64,
        "tile_count": 64,
        "nonempty_tiles": int((token_counts > 0).sum()),
        "tokens": {
            "min": int(token_counts.min()),
            "mean": float(token_counts.mean()),
            "median": float(token_counts.median()),
            "max": int(token_counts.max()),
        },
        "coverage": "exactly_once_half_open",
    }
    return records, stats


def _pack_tile_batch(
    records: Sequence[Mapping[str, Any]],
    state: torch.Tensor,
    proj_features: torch.Tensor,
    global_feature: torch.Tensor,
    device: torch.device,
) -> Tuple[SparseTensor, Dict[str, Any], list[int]]:
    feature_parts = []
    projection_parts = []
    coordinate_parts = []
    lengths = []
    for batch_id, record in enumerate(records):
        ids = record["global_row_ids"]
        local = record["local_coords"].clone()
        local[:, 0] = batch_id
        feature_parts.append(state.index_select(0, ids))
        projection_parts.append(proj_features.index_select(0, ids))
        coordinate_parts.append(local)
        lengths.append(int(ids.numel()))
    coords = torch.cat(coordinate_parts, 0).to(device)
    packed = SparseTensor(torch.cat(feature_parts, 0).to(device), coords)
    projection = torch.cat(projection_parts, 0).to(device)
    glob = global_feature.repeat(len(records), 1, 1).to(device)
    condition = {
        "cond": {"global": glob, "proj": SparseTensor(projection, coords)},
        "neg_cond": {
            "global": torch.zeros_like(glob),
            "proj": SparseTensor(torch.zeros_like(projection), coords),
        },
    }
    return packed, condition, lengths


@torch.no_grad()
def tiled_c256_step(
    model: Any,
    sampler: Any,
    state: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    proj_features: torch.Tensor,
    global_feature: torch.Tensor,
    timestep: float,
    next_timestep: float,
    model_kwargs: Mapping[str, Any],
    device: torch.device,
    tile_batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[Dict[str, Any]]]:
    """Evaluate independent local-C64 flows and scatter them to global C256."""
    next_state = torch.empty_like(state, dtype=torch.float32)
    velocity = torch.empty_like(state, dtype=torch.float32)
    endpoint = torch.empty_like(state, dtype=torch.float32)
    writes = torch.zeros(state.shape[0], dtype=torch.int16)
    timings: list[Dict[str, Any]] = []
    active = [record for record in records if record["tokens"] > 0]
    for start in range(0, len(active), tile_batch_size):
        group = active[start:start + tile_batch_size]
        packed, condition, lengths = _pack_tile_batch(
            group, state, proj_features, global_feature, device
        )
        batch_started = time.perf_counter()
        result = sampler.sample_once(
            model, packed, timestep, next_timestep,
            **condition, **dict(model_kwargs),
        )
        torch.cuda.synchronize(device)
        if not torch.equal(result.pred_x_prev.coords, packed.coords):
            raise RuntimeError("tiled flow changed packed sparse row order")
        cursor = 0
        for record, length in zip(group, lengths):
            ids = record["global_row_ids"]
            stop = cursor + length
            next_state.index_copy_(0, ids, result.pred_x_prev.feats[cursor:stop].float().cpu())
            velocity.index_copy_(0, ids, result.pred_v.feats[cursor:stop].float().cpu())
            endpoint.index_copy_(0, ids, result.pred_x_0.feats[cursor:stop].float().cpu())
            writes.index_add_(0, ids, torch.ones_like(ids, dtype=torch.int16))
            cursor = stop
        timings.append({
            "first_tile_id": int(group[0]["tile_id"]),
            "last_tile_id": int(group[-1]["tile_id"]),
            "batch_size": len(group),
            "tokens": int(sum(lengths)),
            "seconds": float(time.perf_counter() - batch_started),
        })
        del packed, condition, result
        torch.cuda.empty_cache()
    if not torch.all(writes == 1):
        raise RuntimeError("tiled C256 flow must write every global row exactly once")
    return next_state, velocity, endpoint, timings


def segment_mean(values: torch.Tensor, parent: torch.Tensor, count: int) -> torch.Tensor:
    result = torch.zeros((count, values.shape[1]), device=values.device, dtype=values.dtype)
    result.index_add_(0, parent, values)
    denom = torch.bincount(parent, minlength=count).to(values.device, values.dtype).unsqueeze(1)
    return result / denom.clamp_min(1)


def per_parent_variance(values: torch.Tensor, parent: torch.Tensor, count: int) -> torch.Tensor:
    mean = segment_mean(values, parent, count)
    squared_distance = (values - mean.index_select(0, parent)).float().square().sum(1)
    result = torch.zeros(count, device=values.device, dtype=torch.float32)
    result.index_add_(0, parent, squared_distance)
    denom = torch.bincount(parent, minlength=count).to(values.device, torch.float32)
    return result / denom.clamp_min(1)


def relative_l2(lhs: torch.Tensor, rhs: torch.Tensor, eps: float = 1e-8) -> float:
    return float((lhs.float() - rhs.float()).norm() / (rhs.float().norm() + eps))


def cosine_metrics(lhs: torch.Tensor, rhs: torch.Tensor) -> Tuple[float, float, torch.Tensor]:
    per_row = F.cosine_similarity(lhs.float(), rhs.float(), dim=1, eps=1e-8)
    flattened = F.cosine_similarity(lhs.float().reshape(1, -1), rhs.float().reshape(1, -1), dim=1)
    return float(per_row.mean()), float(flattened), per_row


def metric_row(
    step: int,
    timestep: float,
    next_timestep: float,
    x64: torch.Tensor,
    v64: torch.Tensor,
    x064: torch.Tensor,
    x256: torch.Tensor,
    v256: torch.Tensor,
    x0256: torch.Tensor,
    parent: torch.Tensor,
    num_coarse: int,
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    dx = segment_mean(x256, parent, num_coarse)
    dv = segment_mean(v256, parent, num_coarse)
    dx0 = segment_mean(x0256, parent, num_coarse)
    x0_cos, x0_flat_cos, x0_per_parent = cosine_metrics(dx0, x064)
    v_cos, v_flat_cos, v_per_parent = cosine_metrics(dv, v64)
    variance = per_parent_variance(x0256, parent, num_coarse)
    low = dx0.index_select(0, parent)
    high = x0256.float() - low.float()
    coarse_energy = float(low.float().norm())
    fine_energy = float(high.norm())
    q = torch.quantile(variance, torch.tensor([0.5, 0.9, 0.95], device=variance.device))
    row: Dict[str, float] = {
        "step": int(step),
        "timestep": float(timestep),
        "next_timestep": float(next_timestep),
        "endpoint_cosine_mean": x0_cos,
        "endpoint_cosine_flat": x0_flat_cos,
        "endpoint_relative_l2": relative_l2(dx0, x064),
        "state_relative_l2": relative_l2(dx, x64),
        "velocity_cosine_mean": v_cos,
        "velocity_cosine_flat": v_flat_cos,
        "velocity_relative_l2": relative_l2(dv, v64),
        "children_variance_mean": float(variance.mean()),
        "children_variance_median": float(q[0]),
        "children_variance_p90": float(q[1]),
        "children_variance_p95": float(q[2]),
        "coarse_energy": coarse_energy,
        "fine_energy": fine_energy,
        "fine_coarse_energy_ratio": fine_energy / (coarse_energy + 1e-8),
    }
    detail = {
        "endpoint_cosine": x0_per_parent.detach().cpu(),
        "velocity_cosine": v_per_parent.detach().cpu(),
        "children_variance": variance.detach().cpu(),
    }
    return row, detail


def write_metrics(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_json(output / "metrics.json", {"format": FORMAT, "rows": list(rows)})
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(output: Path, rows: Sequence[Mapping[str, float]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [int(row["step"]) for row in rows]
    actual_t = [float(row["timestep"]) for row in rows]
    plots = (
        ("endpoint_cosine.png", "Endpoint cosine", ("endpoint_cosine_mean", "mean parent cosine")),
        ("endpoint_l2.png", "Endpoint relative L2", ("endpoint_relative_l2", "relative L2")),
        ("velocity_cosine.png", "Velocity cosine", ("velocity_cosine_mean", "mean parent cosine")),
        ("children_variance.png", "Children endpoint variance", ("children_variance_mean", "mean"), ("children_variance_median", "median"), ("children_variance_p90", "p90"), ("children_variance_p95", "p95")),
        ("fine_coarse_energy.png", "Fine / coarse endpoint energy", ("fine_coarse_energy_ratio", "ratio")),
    )
    for spec in plots:
        filename, title, *series = spec
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for key, label in series:
            ax.plot(x, [float(row[key]) for row in rows], marker="o", label=label)
        ax.set(xlabel="Flow step (actual t shown above)", ylabel=title, xticks=x)
        ax.set_xticklabels([f"{s}\n{t:.3f}" for s, t in zip(x, actual_t)])
        ax.grid(alpha=0.25)
        if len(series) > 1:
            ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)


def write_colored_ply(path: Path, coords64: torch.Tensor, values: torch.Tensor, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps

    xyz = ((coords64[:, 1:].float() + 0.5) / 64.0 - 0.5).numpy()
    v = values.float().numpy()
    lo, hi = np.quantile(v, [0.02, 0.98]) if len(v) > 1 else (float(v[0]), float(v[0]))
    normalized = np.clip((v - lo) / max(float(hi - lo), 1e-12), 0, 1)
    rgb = (colormaps["turbo"](normalized)[:, :3] * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"comment metric {label} p02 {lo} p98 {hi}\n")
        handle.write(f"element vertex {len(xyz)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nproperty float value\nend_header\n")
        for point, color, value in zip(xyz, rgb, v):
            handle.write(f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} {color[0]} {color[1]} {color[2]} {value:.7g}\n")


def write_heatmap_views(path: Path, coords64: torch.Tensor, values: torch.Tensor, label: str) -> None:
    """Write three orthographic views while keeping the PLY as the 3D source."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = ((coords64[:, 1:].float() + 0.5) / 64.0 - 0.5).numpy()
    scalar = values.float().numpy()
    lo, hi = np.quantile(scalar, [0.02, 0.98])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    projections = ((0, 1, "x", "y"), (0, 2, "x", "z"), (2, 1, "z", "y"))
    scatter = None
    for axis, (u, v, ul, vl) in zip(axes, projections):
        order = np.argsort(scalar)
        scatter = axis.scatter(xyz[order, u], xyz[order, v], c=scalar[order], s=2.5, cmap="turbo", vmin=lo, vmax=hi, linewidths=0)
        axis.set(xlabel=ul, ylabel=vl, aspect="equal", xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    fig.colorbar(scatter, ax=axes, shrink=0.82, label=label)
    fig.suptitle(f"C64 parent heatmap: {label} (color clipped p02/p98)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _denormalize(pipeline: Any, features: torch.Tensor) -> torch.Tensor:
    std = torch.as_tensor(pipeline.shape_slat_normalization["std"], device=features.device, dtype=features.dtype)
    mean = torch.as_tensor(pipeline.shape_slat_normalization["mean"], device=features.device, dtype=features.dtype)
    return features * std + mean


def decode_and_render(
    pipeline: Any,
    output: Path,
    step: int,
    coords64: torch.Tensor,
    coords256: torch.Tensor,
    camera: Mapping[str, float],
    device: torch.device,
    render_resolution: int,
    scales: Sequence[int] = (64, 256),
) -> Dict[str, Any]:
    from pixal3d.renderers import MeshRenderer
    from pixal3d.utils.render_utils import proj_camera_to_render_params

    step_dir = output / f"step_{step:02d}"
    payload = torch.load(step_dir / "latents.pt", map_location="cpu", weights_only=False)
    decode_path = step_dir / "decode.json"
    result: Dict[str, Any] = json.loads(decode_path.read_text(encoding="utf-8")) if decode_path.is_file() else {}
    for scale, coords, resolution in ((64, coords64, 1024), (256, coords256, 4096)):
        if scale not in scales:
            continue
        try:
            mesh_path = step_dir / f"x0_pred_{scale}_mesh.pt"
            if mesh_path.is_file():
                mesh = torch.load(mesh_path, map_location="cpu", weights_only=False)["mesh"]
            else:
                normalized = payload[f"x0_pred_{scale}"].to(device)
                slat = SparseTensor(feats=_denormalize(pipeline, normalized), coords=coords.to(device))
                meshes, _ = pipeline.decode_shape_slat(slat, resolution)
                if len(meshes) != 1:
                    raise RuntimeError(f"decoder returned {len(meshes)} meshes")
                mesh = meshes[0]
                atomic_torch_save(mesh_path, {"mesh": mesh.cpu(), "resolution": resolution})
            live = mesh.to(device)
            renderer = MeshRenderer(
                {"resolution": render_resolution, "near": max(0.01, float(camera["distance"]) - 2.0), "far": float(camera["distance"]) + 2.0, "ssaa": 1, "chunk_size": 500_000, "antialias": False},
                device=str(device),
            )
            extr, intr = proj_camera_to_render_params(float(camera["camera_angle_x"]), float(camera["distance"]))
            rendered = renderer.render(live, extr, intr, return_types=["normal", "mask"])
            torch.cuda.synchronize()
            normal = rendered.normal.detach().float().cpu().permute(1, 2, 0).numpy()
            mask = rendered.mask.detach().float().cpu().numpy()[..., None]
            image = np.clip(normal * mask * 255.0, 0, 255).astype(np.uint8)
            render_path = step_dir / f"x0_pred_{scale}_normal.png"
            Image.fromarray(image).save(render_path)
            result[str(scale)] = {"status": "complete", "vertices": int(mesh.vertices.shape[0]), "faces": int(mesh.faces.shape[0]), "render": str(render_path)}
            del mesh, live, renderer, rendered
        except Exception as exc:
            result[str(scale)] = {"status": "failed", "error": repr(exc)}
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(decode_path, result)
    return result


def _stage(step: int, steps: int) -> str:
    fraction = (step - 1) / max(steps - 1, 1)
    return "early" if fraction < 1 / 3 else ("middle" if fraction < 2 / 3 else "late")


def _first_growth(rows: Sequence[Mapping[str, float]], key: str) -> int | None:
    values = np.asarray([float(row[key]) for row in rows])
    baseline = values[0]
    threshold = baseline + max(0.1 * float(values.max() - values.min()), 0.25 * max(abs(baseline), 1e-12))
    found = np.flatnonzero(values >= threshold)
    return int(found[0] + 1) if len(found) else None


def make_summary(output: Path, rows: Sequence[Mapping[str, float]], decode: Mapping[str, Any]) -> None:
    endpoint = np.asarray([row["endpoint_cosine_mean"] for row in rows], dtype=float)
    velocity = np.asarray([row["velocity_cosine_mean"] for row in rows], dtype=float)
    variance_step = _first_growth(rows, "children_variance_mean")
    endpoint_drop_amount = float(np.max(endpoint[0] - endpoint))
    velocity_drop_amount = float(np.max(velocity[0] - velocity))
    endpoint_drop = int(np.argmax(endpoint[0] - endpoint) + 1) if endpoint_drop_amount > 1e-6 else None
    velocity_drop = int(np.argmax(velocity[0] - velocity) + 1) if velocity_drop_amount > 1e-6 else None
    shared_fine = [
        int(row["step"]) for row in rows
        if row["endpoint_cosine_mean"] >= 0.8
        and row["children_variance_mean"] >= np.median([r["children_variance_mean"] for r in rows])
    ]
    heatmap_note = "The PLY heatmaps are quantitative spatial diagnostics; semantic concentration requires visual inspection against the decoded renders."
    lines = [
        "# C64 / C256 shape-flow consistency summary",
        "",
        f"1. Endpoint mean parent cosine ranges from {endpoint.min():.4f} to {endpoint.max():.4f}; final={endpoint[-1]:.4f}. Endpoint relative L2 final={rows[-1]['endpoint_relative_l2']:.4f}.",
        f"2. Endpoint decline from step 1: {('none; the trajectory never drops below step 1' if endpoint_drop is None else f'largest at step {endpoint_drop} ({_stage(endpoint_drop, len(rows))})') }.",
        f"3. Velocity decline from step 1: {('none; the trajectory never drops below step 1' if velocity_drop is None else f'largest at step {velocity_drop} ({_stage(velocity_drop, len(rows))})') }.",
        f"4. Children variance first crosses the declared growth threshold at {('step ' + str(variance_step)) if variance_step else 'no detected step'}; final mean={rows[-1]['children_variance_mean']:.6g}.",
        f"5. {heatmap_note}",
        f"6. Steps satisfying absolute coarse cosine >= 0.8 plus above-median children variance: {shared_fine or 'none'}. This is the declared operational signature of shared coarse component + fine residual.",
        "",
        "The onset rules are descriptive diagnostics, not statistical significance tests. See `metrics.csv`, selected-step native decoder renders, and step 4/8/12 PLY heatmaps for the underlying evidence.",
        "",
        f"Decode status: `{json.dumps(decode, ensure_ascii=False)}`.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def run(args: argparse.Namespace) -> Path:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4":
        raise RuntimeError("this experiment must run with CUDA_VISIBLE_DEVICES=4")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    coords256_cpu = load_coords(args.c256_support.expanduser().resolve())
    coords64_cpu, parent_cpu, offsets = build_parent_mapping(coords256_cpu)
    tile_records, tile_stats = build_c256_tiles(coords256_cpu)
    counts = offsets[1:] - offsets[:-1]
    mapping_stats = {
        "num_c256_points": int(coords256_cpu.shape[0]),
        "num_c64_points": int(coords64_cpu.shape[0]),
        "children_per_parent": {"mean": float(counts.float().mean()), "median": float(counts.float().median()), "min": int(counts.min()), "max": int(counts.max())},
        "invariants": {"every_fine_has_exactly_one_parent": bool(parent_cpu.numel() == coords256_cpu.shape[0]), "all_parents_nonempty": bool(torch.all(counts > 0)), "formula": "floor(q256 / 4)"},
    }
    atomic_json(output / "mapping_stats.json", mapping_stats)
    atomic_torch_save(output / "parent_child_mapping.pt", {"coords_c256": coords256_cpu, "coords_c64": coords64_cpu, "fine_to_coarse_parent": parent_cpu, "coarse_to_fine_offsets": offsets, "coarse_to_fine_order": torch.argsort(parent_cpu, stable=True)})
    atomic_json(
        output / "tile_layout.json",
        {
            **tile_stats,
            "tiles": [
                {
                    "tile_id": int(record["tile_id"]),
                    "start_c256": list(record["start_c256"]),
                    "start_voxel4096": list(record["start_voxel4096"]),
                    "tokens": int(record["tokens"]),
                    "transform": record["transform"],
                }
                for record in tile_records
            ],
        },
    )

    camera = json.loads(args.camera.read_text(encoding="utf-8"))
    image = Image.open(args.image).convert("RGB")
    if image.size != (1024, 1024):
        raise ValueError(f"global condition image must be 1024x1024, got {image.size}")
    torch.manual_seed(args.seed64)
    np.random.seed(args.seed64)
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=True)
    if args.decode_only:
        if args.decode_step not in {1, 4, 8, 12} or args.decode_scale not in {64, 256}:
            raise ValueError("decode-only requires --decode-step in {1,4,8,12} and --decode-scale in {64,256}")
        result = decode_and_render(
            pipeline, output, args.decode_step, coords64_cpu, coords256_cpu,
            camera, device, args.render_resolution, scales=(args.decode_scale,),
        )
        print(json.dumps(result, indent=2), flush=True)
        return output
    model = pipeline.models["shape_slat_flow_model_1024"]
    sampler = pipeline.shape_slat_sampler
    params = dict(pipeline.shape_slat_sampler_params)
    params["steps"] = 12
    coords64 = coords64_cpu.to(device)
    coords256 = coords256_cpu.to(device)

    print(f"[mapping] C256={len(coords256_cpu):,} C64={len(coords64_cpu):,} children mean={counts.float().mean():.2f}", flush=True)
    cond64 = pipeline.get_proj_cond_shape(pipeline.image_cond_model_shape_1024, [image], coords64, camera_angle_x=float(camera["camera_angle_x"]), distance=float(camera["distance"]), mesh_scale=float(camera.get("mesh_scale", 1.0)), grid_resolution_override=64)
    cond256 = pipeline.get_proj_cond_shape(pipeline.image_cond_model_shape_1024, [image], coords256, camera_angle_x=float(camera["camera_angle_x"]), distance=float(camera["distance"]), mesh_scale=float(camera.get("mesh_scale", 1.0)), grid_resolution_override=256)
    global_max_diff = float((cond64["cond"]["global"] - cond256["cond"]["global"]).abs().max())
    # Make the global token literally shared; sparse full-image projection rows remain support-aligned.
    cond256["cond"]["global"] = cond64["cond"]["global"]
    cond256["neg_cond"]["global"] = cond64["neg_cond"]["global"]

    generator64 = torch.Generator(device="cpu").manual_seed(args.seed64)
    generator256 = torch.Generator(device="cpu").manual_seed(args.seed256)
    eps64_cpu = torch.randn((coords64.shape[0], int(model.in_channels)), generator=generator64)
    eps256_cpu = torch.randn((coords256.shape[0], int(model.in_channels)), generator=generator256)
    atomic_torch_save(
        output / "independent_noise.pt",
        {
            "epsilon_64": eps64_cpu,
            "epsilon_256": eps256_cpu,
            "seed64": int(args.seed64),
            "seed256": int(args.seed256),
            "relationship": "independent IID Gaussian initializations",
        },
    )
    state64 = SparseTensor(feats=eps64_cpu.to(device), coords=coords64)
    state256 = eps256_cpu.float().contiguous()

    proj256_cpu = cond256["cond"]["proj"].feats.detach().float().cpu().contiguous()
    global_cpu = cond64["cond"]["global"].detach().cpu().contiguous()
    del cond256, coords256
    torch.cuda.empty_cache()

    times = sampler.timestep_schedule(12, float(params["rescale_t"]))
    model_kwargs = {key: value for key, value in params.items() if key not in {"steps", "rescale_t"}}
    if pipeline.low_vram:
        model.to(device)
    rows = []
    selected = {1, 4, 8, 12}
    heatmap_steps = {4, 8, 12}
    started = time.perf_counter()
    for index, (timestep, next_timestep) in enumerate(zip(times[:-1], times[1:]), start=1):
        step_started = time.perf_counter()
        x64 = state64.feats.detach().float().cpu().clone()
        x256 = state256.clone()
        out64 = sampler.sample_once(model, state64, timestep, next_timestep, **cond64, **model_kwargs)
        next256, velocity256, endpoint256, tile_timings = tiled_c256_step(
            model=model,
            sampler=sampler,
            state=state256,
            records=tile_records,
            proj_features=proj256_cpu,
            global_feature=global_cpu,
            timestep=timestep,
            next_timestep=next_timestep,
            model_kwargs=model_kwargs,
            device=device,
            tile_batch_size=args.tile_batch_size,
        )
        velocity64 = out64.pred_v.feats.detach().float().cpu()
        endpoint64 = out64.pred_x_0.feats.detach().float().cpu()
        row, detail = metric_row(
            index, timestep, next_timestep,
            x64, velocity64, endpoint64,
            x256, velocity256, endpoint256,
            parent_cpu, coords64_cpu.shape[0],
        )
        row["seconds"] = float(time.perf_counter() - step_started)
        rows.append(row)
        step_dir = output / f"step_{index:02d}"
        atomic_torch_save(step_dir / "latents.pt", {
            "timestep": timestep, "next_timestep": next_timestep,
            "x_t_64": x64, "v_t_64": velocity64, "x0_pred_64": endpoint64,
            "x_t_256": x256, "v_t_256": velocity256, "x0_pred_256": endpoint256,
            "c256_tile_timings": tile_timings,
        })
        atomic_torch_save(step_dir / "per_parent_metrics.pt", detail)
        if index in heatmap_steps:
            write_colored_ply(step_dir / "children_variance.ply", coords64_cpu, detail["children_variance"], "children_variance")
            write_colored_ply(step_dir / "endpoint_cosine.ply", coords64_cpu, detail["endpoint_cosine"], "endpoint_cosine")
            write_heatmap_views(step_dir / "children_variance_views.png", coords64_cpu, detail["children_variance"], "children variance")
            write_heatmap_views(step_dir / "endpoint_cosine_views.png", coords64_cpu, detail["endpoint_cosine"], "endpoint cosine")
        state64, state256 = out64.pred_x_prev, next256
        write_metrics(output, rows)
        print(f"[step {index:02d}/12] t={timestep:.6f} endpoint_cos={row['endpoint_cosine_mean']:.5f} velocity_cos={row['velocity_cosine_mean']:.5f} child_var={row['children_variance_mean']:.6g} seconds={row['seconds']:.1f}", flush=True)
        del x64, x256, out64, detail, velocity64, endpoint64, velocity256, endpoint256, next256
        torch.cuda.empty_cache()
    if pipeline.low_vram:
        model.cpu()
    make_plots(output, rows)
    config = {
        "format": FORMAT, "seed64": int(args.seed64), "seed256": int(args.seed256), "steps": 12,
        "physical_cuda": 4, "logical_device": str(device), "gpu": torch.cuda.get_device_name(0),
        "model_path": str(args.model_path), "c256_support": str(args.c256_support), "global_image": str(args.image), "camera": camera,
        "sampler": params, "flow_parameterization": "native FlowEuler _v_to_xstart_eps: x0=(1-sigma_min)x_t-(sigma_min+(1-sigma_min)t)v",
        "conditioning": {"image": "same canonical global 1024 image", "global_token_shared_exactly": True, "pre_share_global_max_abs_diff": global_max_diff, "projection": "same input-image native full-image global/proj extraction; C256 proj rows are gathered by global physical row then paired with tile-local model coords"},
        "c256_flow": {"mode": "64 independent non-overlap tiles", **tile_stats, "tile_batch_size": int(args.tile_batch_size), "external_velocity_fusion": "none; exact half-open ownership", "coordinate_transform": "C4096 1024-voxel tile centre translation + x4 scale -> local C64 index"},
        "noise": {"mode": "independent IID per scale", "seed64": int(args.seed64), "seed256": int(args.seed256)},
        "runtime_seconds_flow": float(time.perf_counter() - started),
    }
    atomic_json(output / "config.json", config)

    decode_results: Dict[str, Any] = {}
    if not args.skip_decode:
        for step in sorted(selected):
            print(f"[decode] step={step}", flush=True)
            decode_results[str(step)] = decode_and_render(pipeline, output, step, coords64_cpu, coords256_cpu, camera, device, args.render_resolution)
    make_summary(output, rows, decode_results)
    print(f"[done] {output}", flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c256-support", type=Path, default=DEFAULT_CASE / "shape/upsample_c256/coords.pt")
    parser.add_argument("--image", type=Path, default=DEFAULT_CASE / "inputs/global_input_1024.png")
    parser.add_argument("--camera", type=Path, default=DEFAULT_CASE / "global_camera.json")
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--output", type=Path, default=Path("outputs/c64_c256_flow_consistency/tiled1024_stride1024_independent_noise_seed20260824"))
    parser.add_argument("--seed64", type=int, default=20260824)
    parser.add_argument("--seed256", type=int, default=20260825)
    parser.add_argument("--tile-batch-size", type=int, default=16)
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--decode-step", type=int, default=0)
    parser.add_argument("--decode-scale", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.tile_batch_size <= 0 or args.render_resolution <= 0:
        raise ValueError("tile-batch-size and render-resolution must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
