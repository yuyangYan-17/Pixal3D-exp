#!/usr/bin/env python3
"""Run only selected global-C256/local-C64 cubes and decode each cube locally.

The global C256 support is reused as geometry-only SLAT activation points.  No
velocity is evaluated outside ``--cube-ids`` and no global C256 decode is run.
Each selected cube gets its own projected 4096 crop, independent Shape/Texture
flow, and an independent local C64 -> 1024 native decode.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import pixal3d_global_c256_cube_owner_flow_singleview as cube_flow
import pixal3d_global4096_singleview_shared_slat_shape_tex_sr as expc
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor


FORMAT = "pixal3d_global_c256_head_cubes_local_decode_v1"


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def atomic_json(path: Path, value: Any) -> None:
    cube_flow.atomic_json(path, value)


def parse_cube_ids(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result or len(set(result)) != len(result):
        raise ValueError("--cube-ids must be a nonempty unique comma-separated list")
    if any(item < 0 or item >= 343 for item in result):
        raise ValueError("cube IDs must be in [0,342]")
    return result


def local_condition(condition: Mapping[str, Any], rec: Mapping[str, Any]) -> dict[str, Any]:
    cube_id = int(rec["cube_id"])
    payload = dict(condition["cubes"][cube_id])
    payload["global_row_ids"] = torch.arange(rec["global_row_ids"].numel(), dtype=torch.int64)
    return {"cubes": {cube_id: payload}, "fingerprint_sha256": condition["fingerprint_sha256"]}


def local_record(rec: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(rec)
    result["global_row_ids"] = torch.arange(rec["global_row_ids"].numel(), dtype=torch.int64)
    result["owned_row_ids"] = result["global_row_ids"]
    return result


@torch.no_grad()
def run_local_flow(
    *, stage: str, rec: Mapping[str, Any], condition: Mapping[str, Any], sampler: Any,
    model: Any, params: Mapping[str, Any], device: torch.device, seed: int,
    concat: torch.Tensor | None, output: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    channels = int(model.in_channels) if concat is None else int(model.in_channels) - int(concat.shape[1])
    if channels <= 0:
        raise RuntimeError(f"invalid {stage} channel count")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    state = torch.randn((rec["global_row_ids"].numel(), channels), generator=generator)
    schedule = sampler.timestep_schedule(int(params["steps"]), float(params.get("rescale_t", 1.0)))
    model.to(device).eval()
    records = []
    started = time.perf_counter()
    for step, (t, t_next) in enumerate(zip(schedule[:-1], schedule[1:])):
        torch.cuda.reset_peak_memory_stats(device)
        velocity, timing = cube_flow._one_prediction(
            [rec], state, condition, sampler, model, params, float(t), float(t_next), device, concat
        )
        state = cube_flow.jacobi_update(state, velocity[0], float(t), float(t_next))
        row = {
            "step": step, "t": float(t), "t_next": float(t_next), **timing,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        records.append(row)
        print(f"[{stage}] cube={int(rec['cube_id']):03d} step={step+1}/{len(schedule)-1} seconds={timing['seconds']:.2f}", flush=True)
    model.cpu(); empty_cuda()
    summary = {
        "stage": stage, "cube_id": int(rec["cube_id"]), "tokens": int(state.shape[0]),
        "channels": int(state.shape[1]), "steps": len(schedule)-1,
        "seed": int(seed), "seconds": time.perf_counter()-started, "records": records,
    }
    cube_flow.atomic_save(output / f"{stage}_normalized.pt", {"features": state, "local_coords": cube_flow._local_coords(rec)})
    atomic_json(output / f"{stage}_flow_summary.json", summary)
    return state, summary


@torch.no_grad()
def decode_cube(pipeline: Any, rec: Mapping[str, Any], shape: torch.Tensor, texture: torch.Tensor,
                output: Path, device: torch.device) -> dict[str, Any]:
    shape_raw = cube_flow.denormalize(shape, pipeline.shape_slat_normalization)
    texture_raw = cube_flow.denormalize(texture, pipeline.tex_slat_normalization)
    coords = cube_flow._local_coords(rec)
    cube_flow.atomic_save(output / "latents_denormalized.pt", {
        "local_coords": coords, "shape": shape_raw, "texture": texture_raw,
    })
    started = time.perf_counter()
    decoded = pipeline.decode_latent(
        SparseTensor(shape_raw.to(device), coords.to(device)),
        SparseTensor(texture_raw.to(device), coords.to(device)), 1024,
    )
    if len(decoded) != 1:
        raise RuntimeError(f"cube {rec['cube_id']} decode returned B={len(decoded)}")
    native = decoded[0]
    cube_flow.atomic_save(output / "local_material_mesh.pt", {"format": FORMAT, "mesh": native.cpu()})
    vertex, face = expc._native_mesh_to_pbr(native, device)
    vertex_path = output / "local_per_vertex_pbr_mesh.pt"
    cube_flow.atomic_save(vertex_path, {"format": FORMAT, "mesh": vertex})
    cube_flow.atomic_save(output / "local_per_face_pbr_mesh.pt", {"format": FORMAT, "mesh": face})
    result = {
        "status": "complete", "cube_id": int(rec["cube_id"]), "start_c256": list(rec["start"]),
        "resolution": 1024, "tokens": int(coords.shape[0]), "vertices": int(vertex.vertices.shape[0]),
        "faces": int(vertex.faces.shape[0]), "seconds": time.perf_counter()-started,
        "vertex_mesh": str(vertex_path.resolve()),
    }
    atomic_json(output / "decode_summary.json", result)
    del decoded, native, vertex, face
    empty_cuda()
    return result


def render_activation_points(coords: torch.Tensor, selected: list[Mapping[str, Any]], output: Path) -> str:
    xyz = coords[:, 1:].float().numpy()
    selected_masks = []
    for rec in selected:
        mask = np.zeros(xyz.shape[0], dtype=bool)
        mask[rec["global_row_ids"].numpy()] = True
        selected_masks.append(mask)
    any_selected = np.logical_or.reduce(selected_masks)
    # Deterministic thinning keeps the overview readable without changing saved support.
    base_ids = np.where(~any_selected)[0][::max(1, int((~any_selected).sum()) // 70000)]
    views = ((20, -55), (20, 35), (20, 125), (20, 215))
    fig = plt.figure(figsize=(20, 5), facecolor="black")
    colors = ("#ff5a36", "#32d7ff", "#ffd447", "#a879ff")
    for index, (elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 4, index, projection="3d", facecolor="black")
        ax.scatter(xyz[base_ids, 0], xyz[base_ids, 2], xyz[base_ids, 1], s=.08, c="#b8c0cc", alpha=.22)
        for color, rec, mask in zip(colors, selected, selected_masks):
            pts = xyz[mask]
            ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=.45, c=color, alpha=.9,
                       label=f"cube {int(rec['cube_id'])}")
        ax.view_init(elev=elev, azim=azim)
        # SLAT axis 1 is native world-up.  Plot it as matplotlib's vertical
        # axis without reversing its limits (the old overview flipped y).
        ax.set_xlim(0, 255); ax.set_ylim(0, 255); ax.set_zlim(0, 255)
        ax.set_axis_off(); ax.set_title(f"azimuth {azim}°", color="white")
        if index == 1: ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Global C256 SLAT support (gray) + inferred head cubes", color="white", fontsize=14)
    fig.tight_layout()
    path = output / "slat_activation_points_multiview.png"
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path.resolve())


def parse_args() -> argparse.Namespace:
    baseline = Path("outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024")
    source = Path("outputs/global_c256_cube_owner_flow_singleview_cuda4")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-ids", default="263,270")
    parser.add_argument("--support", default=source / "support/global_c256_support.pt")
    parser.add_argument("--condition-image-4096", default=baseline / "inputs/canonical_foreground_rgb_4096.png")
    parser.add_argument("--camera-json", default=baseline / "global_camera.json")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--output-dir", default="outputs/global_c256_head_cubes_263_270_local_decode_cuda5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-cuda", type=int, default=5)
    parser.add_argument("--shape-steps", type=int, default=12)
    parser.add_argument("--texture-steps", type=int, default=12)
    parser.add_argument("--shape-seed", type=int, default=43)
    parser.add_argument("--texture-seed", type=int, default=44)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); cube_ids = parse_cube_ids(args.cube_ids)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.physical_cuda):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}, expected {args.physical_cuda}")
    device = torch.device(args.device); torch.cuda.set_device(device)
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    support_payload = torch.load(args.support, map_location="cpu", weights_only=False)
    coords = support_payload["coords"].int().contiguous()
    records, _ = cube_flow.build_cube_records(coords)
    owner, _ = cube_flow.build_owner_map(coords, records)
    camera = json.loads(Path(args.camera_json).read_text())
    selected = [records[cube_id] for cube_id in cube_ids]
    cube_flow.attach_cube_projection_crops(selected, camera)
    if any(not rec["global_row_ids"].numel() for rec in selected):
        raise RuntimeError("selected cube is empty")
    config = {
        "format": FORMAT, "status": "running", "args": vars(args), "selected_cubes": [
            {"cube_id": int(r["cube_id"]), "start_c256": list(r["start"]),
             "membership_tokens": int(r["global_row_ids"].numel()),
             "owned_tokens": int(r["owned_row_ids"].numel()),
             "crop_box_4096": list(r["condition_crop"]["crop_box_4096"])} for r in selected],
        "global_support_tokens": int(coords.shape[0]),
        "outside_selected_flow": "SLAT support activation points only; no velocity and no decode",
        "runtime": {"CUDA_VISIBLE_DEVICES": visible, "gpu": torch.cuda.get_device_name(device)},
    }
    atomic_json(output / "config.json", config)
    activation_path = render_activation_points(coords, selected, output)
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=args.low_vram)
    condition_args = argparse.Namespace(condition_image_4096=args.condition_image_4096, velocity_fusion="owner")
    shape_condition = cube_flow.build_condition(pipeline, condition_args, coords, selected, camera, "shape", output)
    texture_condition = cube_flow.build_condition(pipeline, condition_args, coords, selected, camera, "texture", output)
    results = []
    for rec0 in selected:
        rec = local_record(rec0); cube_id = int(rec["cube_id"]); cube_out = output / f"cube_{cube_id:03d}"
        cube_out.mkdir(parents=True, exist_ok=True)
        shape_params = dict(pipeline.shape_slat_sampler_params); shape_params["steps"] = args.shape_steps
        shape, shape_summary = run_local_flow(
            stage="shape", rec=rec, condition=local_condition(shape_condition, rec0),
            sampler=pipeline.shape_slat_sampler, model=pipeline.models["shape_slat_flow_model_1024"],
            params=shape_params, device=device, seed=args.shape_seed + cube_id, concat=None, output=cube_out)
        texture_params = dict(pipeline.tex_slat_sampler_params); texture_params["steps"] = args.texture_steps
        texture, texture_summary = run_local_flow(
            stage="texture", rec=rec, condition=local_condition(texture_condition, rec0),
            sampler=pipeline.tex_slat_sampler, model=pipeline.models["tex_slat_flow_model_1024"],
            params=texture_params, device=device, seed=args.texture_seed + cube_id,
            concat=shape, output=cube_out)
        decoded = decode_cube(pipeline, rec, shape, texture, cube_out, device)
        results.append({"cube_id": cube_id, "shape": shape_summary, "texture": texture_summary, "decode": decoded})
        del shape, texture
        empty_cuda()
    config.update({"status": "flow_and_decode_complete", "activation_points_multiview": activation_path, "results": results})
    atomic_json(output / "summary.json", config)
    atomic_json(output / "config.json", config)
    print(f"[done] local flow/decode complete; output={output}", flush=True)


if __name__ == "__main__":
    main()
