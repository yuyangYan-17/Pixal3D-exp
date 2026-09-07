#!/usr/bin/env python3
"""Run C64-context/stride32 hard-owner Shape/Texture flows on Global C256."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import torch
from PIL import Image

import pixal3d_global_c256_cube_owner_flow_singleview as cube_flow
import pixal3d_global_c256_restructured_blocks_singleview as core
from inference import MODEL_PATH, init_pipeline


FORMAT = "pixal3d_global_c256_c64_stride32_owner_flow_singleview_v1"


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def owner_records(coords: torch.Tensor) -> list[dict[str, Any]]:
    xyz = coords[:, 1:4].cpu().int()
    coverage = torch.zeros(len(coords), dtype=torch.int16)
    records: list[dict[str, Any]] = []
    cube_id = 0
    for ox in range(8):
        for oy in range(8):
            for oz in range(8):
                owner_start = torch.tensor((ox, oy, oz), dtype=torch.int32) * 32
                owner_mask = ((xyz >= owner_start) & (xyz < owner_start + 32)).all(1)
                if not owner_mask.any():
                    continue
                context_start = owner_start - 16
                context_mask = ((xyz >= context_start) & (xyz < context_start + 64)).all(1)
                context_rows = torch.where(context_mask)[0].long()
                owned_rows = torch.where(owner_mask)[0].long()
                owner_positions = torch.searchsorted(context_rows, owned_rows)
                if (owner_positions >= len(context_rows)).any() or not torch.equal(
                        context_rows.index_select(0, owner_positions), owned_rows):
                    raise RuntimeError("owner rows are not a subset of context rows")
                coverage.index_add_(0, owned_rows, torch.ones(len(owned_rows), dtype=torch.int16))
                records.append({
                    "cube_id": cube_id, "owner_index": (ox, oy, oz),
                    "owner_start": tuple(int(v) for v in owner_start.tolist()),
                    "context_start": tuple(int(v) for v in context_start.tolist()),
                    "global_row_ids": context_rows, "owned_row_ids": owned_rows,
                    "owner_context_positions": owner_positions,
                    "local_xyz": xyz.index_select(0, context_rows) - context_start,
                })
                cube_id += 1
    if not torch.all(coverage == 1):
        raise RuntimeError(
            f"Global C256 C32 ownership failed: min={int(coverage.min())} max={int(coverage.max())}")
    return records


@torch.no_grad()
def build_global_point_condition(
    pipeline: Any, image: Image.Image, camera: Mapping[str, float], coords: torch.Tensor,
    stage: str, output: Path, device: torch.device,
) -> dict[str, torch.Tensor]:
    path = output / "conditions" / f"{stage}_global_points.pt"
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if torch.equal(payload["coords"].int(), coords.int()):
            return {"global": payload["global"], "proj": payload["proj"]}
    model = pipeline.image_cond_model_shape_1024 if stage == "shape" else pipeline.image_cond_model_tex_1024
    cached = core.extract_full_image_features(model, image, device)
    distance = float(camera["distance"])
    fov = float(camera["camera_angle_x"])
    transform = core.local_to_global_camera_transform(
        model, block_start=(0, 0, 0), global_resolution=256,
        global_extent=256, distance=distance, device=device)
    error = core.validate_projection_transform(
        model, transform, local_resolution=256, global_resolution=256,
        block_start=(0, 0, 0), distance=distance, fov=fov, device=device)
    glob, proj = core.project_cached_features(
        model, cached, transform=transform, fov=fov, distance=distance,
        coords=coords.to(device), grid_resolution=256)
    payload = {
        "format": FORMAT, "stage": stage, "coords": coords.cpu().int(),
        "global": glob.cpu(), "proj": proj[0].cpu(),
        "projection_max_error_pixels": error,
    }
    cube_flow.atomic_save(path, payload)
    if pipeline.low_vram:
        model.cpu()
    del cached, glob, proj
    empty_cuda()
    return {"global": payload["global"], "proj": payload["proj"]}


@torch.no_grad()
def owner_flow(
    *, stage: str, pipeline: Any, records: list[dict[str, Any]],
    point_condition: Mapping[str, torch.Tensor], output: Path, device: torch.device,
    seed: int, steps: int, concat: torch.Tensor | None,
    max_batch_size: int, max_batch_tokens: int,
) -> torch.Tensor:
    root = output / stage
    final_path = root / "final_normalized.pt"
    if final_path.is_file():
        return torch.load(final_path, map_location="cpu", weights_only=False)["features"].float()
    model = pipeline.models["shape_slat_flow_model_1024" if stage == "shape" else "tex_slat_flow_model_1024"]
    sampler = pipeline.shape_slat_sampler if stage == "shape" else pipeline.tex_slat_sampler
    params = dict(pipeline.shape_slat_sampler_params if stage == "shape" else pipeline.tex_slat_sampler_params)
    params["steps"] = steps
    channels = int(model.in_channels) if concat is None else int(model.in_channels) - int(concat.shape[1])
    token_count = sum(len(r["owned_row_ids"]) for r in records)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = torch.randn((token_count, channels), generator=generator)
    completed = 0
    for step in range(steps, 0, -1):
        path = root / f"step_{step:02d}.pt"
        if path.is_file():
            state = torch.load(path, map_location="cpu", weights_only=False)["features"].float()
            completed = step
            break
    schedule = sampler.timestep_schedule(steps, float(params.get("rescale_t", 1.0)))
    groups: list[list[dict[str, Any]]] = []
    group: list[dict[str, Any]] = []
    group_tokens = 0
    for rec in records:
        n = len(rec["global_row_ids"])
        if group and (len(group) >= max_batch_size or group_tokens + n > max_batch_tokens):
            groups.append(group)
            group, group_tokens = [], 0
        group.append(rec)
        group_tokens += n
    if group:
        groups.append(group)
    print(f"[{stage}-groups] groups={len(groups)} max_batch_size={max_batch_size} "
          f"max_batch_tokens={max_batch_tokens:,}", flush=True)
    model.to(device).eval()
    history = []
    for step in range(completed, steps):
        t, t_next = float(schedule[step]), float(schedule[step + 1])
        velocity = torch.empty_like(state)
        writes = torch.zeros(token_count, dtype=torch.int16)
        elapsed = 0.0
        for order, group in enumerate(groups, 1):
            local_condition = {"cubes": {}}
            for rec in group:
                rows = rec["global_row_ids"]
                local_condition["cubes"][int(rec["cube_id"])] = {
                    "global_row_ids": rows, "global": point_condition["global"],
                    "proj": point_condition["proj"].index_select(0, rows)}
            values, timing = cube_flow._one_prediction(
                group, state, local_condition, sampler, model, params,
                t, t_next, device, concat)
            for rec, value in zip(group, values):
                owned_v = value.index_select(0, rec["owner_context_positions"])
                velocity.index_copy_(0, rec["owned_row_ids"], owned_v)
                writes.index_add_(
                    0, rec["owned_row_ids"],
                    torch.ones(len(rec["owned_row_ids"]), dtype=torch.int16))
            elapsed += timing["seconds"]
            print(f"[{stage}-owner] step={step+1}/{steps} group={order}/{len(groups)} "
                  f"batch={len(group)} context={sum(len(r['global_row_ids']) for r in group):,} "
                  f"owner={sum(len(r['owned_row_ids']) for r in group):,}", flush=True)
            del local_condition, values
        if not torch.all(writes == 1):
            raise RuntimeError(f"{stage} owner write coverage is not one")
        state = cube_flow.jacobi_update(state, velocity, t, t_next)
        if not torch.isfinite(state).all():
            raise FloatingPointError(f"non-finite {stage} state")
        cube_flow.atomic_save(root / f"step_{step+1:02d}.pt", {"format": FORMAT, "features": state})
        history.append({"step": step + 1, "t": t, "t_next": t_next, "seconds": elapsed})
    model.cpu()
    empty_cuda()
    cube_flow.atomic_save(final_path, {"format": FORMAT, "features": state})
    core.atomic_json(root / "summary.json", {
        "format": FORMAT, "tokens": token_count, "channels": channels,
        "windows": len(records), "steps": steps, "seed": seed, "new_history": history})
    return state


def parse_args() -> argparse.Namespace:
    base = Path("outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, default=Path("assets/images/0_img.png"))
    p.add_argument("--camera", type=Path, default=base / "global_camera.json")
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--global-support", type=Path, default=Path(
        "outputs/global_c256_c32_context_owner_dec1_all_blocks_cuda5/global_support/global_c256_support.pt"))
    p.add_argument("--output", type=Path, default=Path(
        "outputs/global_c256_c64_stride32_owner_flow_cuda5"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--physical-cuda", type=int, default=5)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--shape-seed", type=int, default=48001)
    p.add_argument("--texture-seed", type=int, default=49001)
    # Empirical CUDA5 estimate: 450k context rows peaked at 35.7 GiB with an
    # 8.6 GiB resident baseline.  Scaling the activation component to a 70 GiB
    # safety ceiling gives ~1.02M rows; use 900k as the verified-next target.
    p.add_argument("--max-batch-size", type=int, default=64)
    p.add_argument("--max-batch-tokens", type=int, default=900000)
    p.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.physical_cuda):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}, expected {args.physical_cuda}")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    camera = json.loads(args.camera.read_text())
    camera["mesh_scale"] = 1.0
    coords = torch.load(args.global_support, map_location="cpu", weights_only=False)["coords"].int()
    records = owner_records(coords)
    core.atomic_json(output / "config.json", {
        "format": FORMAT, "args": vars(args), "camera": camera,
        "support": {"tokens": len(coords), "source": str(args.global_support.resolve())},
        "flow_partition": "Global C256: centered C64 context, stride32, center C32 hard owner",
        "owner_windows": len(records), "decode": "one Global C256 -> 4096 geometry/material decode",
    })
    core.atomic_json(output / "owner_windows.json", {
        "format": FORMAT, "windows": len(records),
        "context_memberships": sum(len(r["global_row_ids"]) for r in records),
        "records": [{
            "cube_id": int(r["cube_id"]), "owner_index": r["owner_index"],
            "owner_start": r["owner_start"], "context_start": r["context_start"],
            "context_tokens": len(r["global_row_ids"]), "owner_tokens": len(r["owned_row_ids"]),
        } for r in records],
    })
    print(f"[support] tokens={len(coords):,} owner_windows={len(records)} "
          f"memberships={sum(len(r['global_row_ids']) for r in records):,}", flush=True)

    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=args.low_vram)
    canonical = pipeline.preprocess_canonical_images(Image.open(args.image))
    image1024 = canonical["image_1024"]
    shape_cond = build_global_point_condition(
        pipeline, image1024, camera, coords, "shape", output, device)
    shape = owner_flow(
        stage="shape", pipeline=pipeline, records=records, point_condition=shape_cond,
        output=output, device=device, seed=args.shape_seed, steps=args.steps, concat=None,
        max_batch_size=args.max_batch_size, max_batch_tokens=args.max_batch_tokens)
    del shape_cond
    empty_cuda()
    tex_cond = build_global_point_condition(
        pipeline, image1024, camera, coords, "texture", output, device)
    texture = owner_flow(
        stage="texture", pipeline=pipeline, records=records, point_condition=tex_cond,
        output=output, device=device, seed=args.texture_seed, steps=args.steps, concat=shape,
        max_batch_size=args.max_batch_size, max_batch_tokens=args.max_batch_tokens)
    del tex_cond
    empty_cuda()
    decoded = core.decode_global(pipeline, coords, shape, texture, output, device)
    core.atomic_json(output / "summary.json", {
        "format": FORMAT, "status": "complete", "global_c256_tokens": len(coords),
        "owner_windows": len(records), "duplicate_points": 0,
        "decode": decoded, "seconds": time.perf_counter() - started})
    print(f"[done] {output}", flush=True)


if __name__ == "__main__":
    main()
