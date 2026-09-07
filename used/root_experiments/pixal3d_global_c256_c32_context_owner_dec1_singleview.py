#!/usr/bin/env python3
"""C32-context/C16-owner construction of one Global C256 SLat.

The native baseline pre-Shape1024 C64 support is partitioned into nonempty
C16 owner blocks.  Each owner predicts Shape512 velocities from a centered
C32 context (stride 16), while a single Global C64 state is updated only on
the center owner's rows after every context has completed the timestep.
The final C32 context latent is decoded once to a local C64 support; each
owner's full local output maps to its disjoint Global C256 C64 block.  Shape
and Texture then run synchronously on the unique Global C256 support and are
decoded together once at resolution 4096.
"""
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
from pixal3d.modules.sparse import SparseTensor


FORMAT = "pixal3d_global_c256_c32_context_owner_crop_dec1_singleview_v2"


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_context_records(baseline64: torch.Tensor) -> list[dict[str, Any]]:
    xyz = baseline64[:, 1:4].cpu().int()
    records: list[dict[str, Any]] = []
    coverage = torch.zeros(len(xyz), dtype=torch.int16)
    for base in core.active_blocks(baseline64):
        bid = int(base["cube_id"])
        idx = tuple(int(v) for v in base["block_index"])
        owner_start = torch.tensor(idx, dtype=torch.int32) * 16
        context_start = owner_start - 8
        context_mask = ((xyz >= context_start) & (xyz < context_start + 32)).all(1)
        owner_mask = ((xyz >= owner_start) & (xyz < owner_start + 16)).all(1)
        context_rows = torch.where(context_mask)[0].long()
        owner_rows = torch.where(owner_mask)[0].long()
        owner_positions = torch.searchsorted(context_rows, owner_rows)
        if (owner_positions >= len(context_rows)).any() or not torch.equal(
                context_rows.index_select(0, owner_positions), owner_rows):
            raise RuntimeError(f"owner rows are not a subset of context rows for block {bid}")
        coverage.index_add_(0, owner_rows, torch.ones(len(owner_rows), dtype=coverage.dtype))
        records.append({
            "cube_id": bid,
            "block_index": idx,
            "context_start": tuple(int(v) for v in context_start.tolist()),
            "owner_start": tuple(int(v) for v in owner_start.tolist()),
            "global_row_ids": context_rows,
            "owned_row_ids": owner_rows,
            "owner_context_positions": owner_positions,
            "local_xyz": xyz.index_select(0, context_rows) - context_start,
            "baseline_tokens": int(len(owner_rows)),
            "context_tokens": int(len(context_rows)),
        })
    if not torch.all(coverage == 1):
        raise RuntimeError(
            f"C16 owner partition is not exact: min={int(coverage.min())} max={int(coverage.max())}")
    return records


@torch.no_grad()
def validate_active_projection(
    model: Any, transform: torch.Tensor, rec: Mapping[str, Any], *,
    fov: float, distance: float, device: torch.device,
) -> float:
    """Validate only real context activations; padded boundary corners are outside C64."""
    local_indices = rec["local_xyz"].to(device=device, dtype=torch.long)
    global_indices = local_indices + torch.tensor(
        rec["context_start"], dtype=torch.long, device=device)
    if torch.any(global_indices < 0) or torch.any(global_indices >= 64):
        raise RuntimeError("real context activation lies outside Global C64")
    fov_t = torch.tensor([fov], device=device)
    dist_t = torch.tensor([distance], device=device)
    scale_t = torch.ones(1, device=device)
    local_px = model.proj_grid.project_grid_indices(
        fov_t, dist_t, scale_t, transform.unsqueeze(0), local_indices,
        grid_resolution=32)[0]
    global_px = model.proj_grid.project_grid_indices(
        fov_t, dist_t, scale_t, None, global_indices,
        grid_resolution=64)[0]
    error = float((local_px - global_px).abs().max().item())
    if error > 2e-3:
        raise RuntimeError(f"active local/global projection mismatch: max pixel error {error}")
    return error


@torch.no_grad()
def build_context_conditions(
    pipeline: Any, image512: Image.Image, camera: Mapping[str, float],
    records: list[dict[str, Any]], output: Path, device: torch.device,
) -> dict[str, Any]:
    root = output / "conditions" / "shape512_c32_context"
    model = pipeline.image_cond_model_shape_512
    pending = [r for r in records if not (root / f"block_{int(r['cube_id']):02d}.pt").is_file()]
    if pending:
        cached = core.extract_full_image_features(model, image512, device)
        fov = float(camera["camera_angle_x"])
        distance = float(camera["distance"])
        for order, rec in enumerate(pending, 1):
            bid = int(rec["cube_id"])
            coords = cube_flow._local_coords(rec).to(device)
            start = tuple(int(v) for v in rec["context_start"])
            transform = core.local_to_global_camera_transform(
                model, block_start=start, global_resolution=64,
                global_extent=32, distance=distance, device=device)
            error = validate_active_projection(
                model, transform, rec, distance=distance, fov=fov, device=device)
            glob, proj = core.project_cached_features(
                model, cached, transform=transform, fov=fov, distance=distance,
                coords=coords, grid_resolution=32)
            cube_flow.atomic_save(root / f"block_{bid:02d}.pt", {
                "format": FORMAT, "cube_id": bid, "block_index": rec["block_index"],
                "context_start": start, "global_row_ids": rec["global_row_ids"],
                "global": glob.cpu(), "proj": proj[0].cpu(),
                "projection_max_error_pixels": error,
                "image_condition": "full-image global token and exact global-position projected features",
            })
            print(f"[condition-shape512] block={bid:02d} {order}/{len(pending)} "
                  f"context={len(coords):,} projection_error={error:.3g}", flush=True)
            del coords, glob, proj
            empty_cuda()
        del cached
        if pipeline.low_vram:
            model.cpu()
        empty_cuda()
    cubes: dict[int, Any] = {}
    for rec in records:
        bid = int(rec["cube_id"])
        payload = torch.load(root / f"block_{bid:02d}.pt", map_location="cpu", weights_only=False)
        if not torch.equal(payload["global_row_ids"].long(), rec["global_row_ids"].long()):
            raise RuntimeError(f"cached context condition row mismatch for block {bid}")
        cubes[bid] = payload
    return {"cubes": cubes, "fingerprint_sha256": FORMAT}


@torch.no_grad()
def context_owner_shape512_flow(
    pipeline: Any, baseline64: torch.Tensor, records: list[dict[str, Any]],
    condition: Mapping[str, Any], output: Path, device: torch.device,
    seed: int, steps: int,
) -> torch.Tensor:
    root = output / "shape512_context_owner"
    final_path = root / "final_normalized.pt"
    if final_path.is_file():
        return torch.load(final_path, map_location="cpu", weights_only=False)["features"].float()
    model = pipeline.models["shape_slat_flow_model_512"]
    sampler = pipeline.shape_slat_sampler
    params = dict(pipeline.shape_slat_sampler_params)
    params["steps"] = steps
    schedule = sampler.timestep_schedule(steps, float(params.get("rescale_t", 1.0)))
    completed = 0
    state = None
    for step in range(steps, 0, -1):
        checkpoint = root / f"step_{step:02d}.pt"
        if checkpoint.is_file():
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)["features"].float()
            completed = step
            break
    if state is None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        state = torch.randn((len(baseline64), int(model.in_channels)), generator=generator)
    if state.shape[0] != len(baseline64):
        raise RuntimeError("cached Shape512 global state token mismatch")
    model.to(device).eval()
    history = []
    for step in range(completed, steps):
        t, t_next = float(schedule[step]), float(schedule[step + 1])
        velocity = torch.empty_like(state)
        write_count = torch.zeros(len(state), dtype=torch.int16)
        elapsed = 0.0
        for order, rec in enumerate(records, 1):
            values, timing = cube_flow._one_prediction(
                [rec], state, condition, sampler, model, params, t, t_next, device, None)
            owned_velocity = values[0].index_select(0, rec["owner_context_positions"])
            velocity.index_copy_(0, rec["owned_row_ids"], owned_velocity)
            write_count.index_add_(
                0, rec["owned_row_ids"], torch.ones(len(rec["owned_row_ids"]), dtype=torch.int16))
            elapsed += timing["seconds"]
            print(f"[shape512-owner] step={step+1}/{steps} block={int(rec['cube_id']):02d} "
                  f"{order}/{len(records)} context={len(rec['global_row_ids']):,} "
                  f"owner={len(rec['owned_row_ids']):,}", flush=True)
        if not torch.all(write_count == 1):
            raise RuntimeError(
                f"Shape512 owner write coverage failed: min={int(write_count.min())} max={int(write_count.max())}")
        state = cube_flow.jacobi_update(state, velocity, t, t_next)
        if not torch.isfinite(state).all():
            raise FloatingPointError(f"non-finite Shape512 state after step {step+1}")
        cube_flow.atomic_save(root / f"step_{step+1:02d}.pt", {"format": FORMAT, "features": state})
        history.append({"step": step + 1, "t": t, "t_next": t_next, "seconds": elapsed})
    model.cpu()
    empty_cuda()
    cube_flow.atomic_save(final_path, {"format": FORMAT, "features": state})
    core.atomic_json(root / "summary.json", {
        "format": FORMAT, "tokens": len(state), "channels": state.shape[1],
        "steps": steps, "seed": seed, "active_owners": len(records),
        "owner_coverage_min": 1, "owner_coverage_max": 1, "new_history": history,
    })
    return state


@torch.no_grad()
def decode_contexts_to_global_support(
    pipeline: Any, endpoint_normalized: torch.Tensor, records: list[dict[str, Any]],
    output: Path, device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    root = output / "local_c32_context_dec1"
    decoder = pipeline.models["shape_slat_decoder"]
    raw = cube_flow.denormalize(endpoint_normalized, pipeline.shape_slat_normalization)
    pending = [r for r in records if not (root / f"block_{int(r['cube_id']):02d}" / "c64_support.pt").is_file()]
    if pending and pipeline.low_vram:
        decoder.to(device)
        decoder.low_vram = True
    for order, rec in enumerate(pending, 1):
        bid = int(rec["cube_id"])
        coords32 = cube_flow._local_coords(rec)
        latent = SparseTensor(raw.index_select(0, rec["global_row_ids"]).to(device), coords32.to(device))
        candidates = decoder.upsample(latent, upsample_times=1)
        context64 = candidates.int().unique(dim=0).cpu().contiguous()
        center = ((context64[:, 1:4] >= 16) & (context64[:, 1:4] < 48)).all(1)
        center32 = context64[center]
        coords64 = center32.clone()
        coords64[:, 1:4] = (coords64[:, 1:4] - 16) * 2
        coords64 = coords64.unique(dim=0).int().contiguous()
        if len(coords64) == 0 or torch.any(coords64[:, 1:] < 0) or torch.any(coords64[:, 1:] >= 64):
            raise RuntimeError(f"invalid decoder-x1 C64 support for block {bid}")
        block_dir = root / f"block_{bid:02d}"
        block_dir.mkdir(parents=True, exist_ok=True)
        cube_flow.atomic_save(block_dir / "c64_support.pt", {
            "format": FORMAT, "block_id": bid, "block_index": rec["block_index"],
            "context_start": rec["context_start"], "coords": coords64,
            "context_tokens": len(rec["global_row_ids"]), "owner_tokens": len(rec["owned_row_ids"]),
            "decoder_upsample_times": 1, "candidate_resolution": 64,
            "full_context_candidates": len(context64), "center_candidates": len(center32),
            "owner_crop": "decoder C64 [16,48) -> owner-local C64 via (coord-16)*2",
        })
        print(f"[decoder-x1] block={bid:02d} {order}/{len(pending)} "
              f"C32-context={len(coords32):,} full={len(context64):,} "
              f"center={len(coords64):,}", flush=True)
        del latent, candidates, context64, center32, coords64
        empty_cuda()
    if pending and pipeline.low_vram:
        decoder.cpu()
        decoder.low_vram = False
    empty_cuda()

    parts: list[torch.Tensor] = []
    raw_count = 0
    for rec in records:
        bid = int(rec["cube_id"])
        payload = torch.load(root / f"block_{bid:02d}" / "c64_support.pt", map_location="cpu", weights_only=False)
        local = payload["coords"].int()
        offset = torch.tensor(rec["block_index"], dtype=torch.int32) * 64
        global_xyz = local[:, 1:4] + offset
        parts.append(torch.cat((torch.zeros((len(local), 1), dtype=torch.int32), global_xyz), 1))
        raw_count += len(local)
        rec["generated_c64_tokens"] = int(len(local))
    concatenated = torch.cat(parts, 0).int()
    coords = concatenated.unique(dim=0).int().contiguous()
    duplicates = raw_count - len(coords)
    if duplicates != 0:
        raise RuntimeError(f"disjoint Global C256 assembly has {duplicates} duplicate points")
    xyz = coords[:, 1:4]
    if torch.any(xyz < 0) or torch.any(xyz >= 256):
        raise RuntimeError("assembled support lies outside Global C256")
    coverage = torch.zeros(len(coords), dtype=torch.int16)
    flow_records: list[dict[str, Any]] = []
    for rec in records:
        idx = tuple(int(v) for v in rec["block_index"])
        start = torch.tensor(idx, dtype=torch.int32) * 64
        mask = ((xyz >= start) & (xyz < start + 64)).all(1)
        rows = torch.where(mask)[0].long()
        local_xyz = xyz.index_select(0, rows) - start
        coverage.index_add_(0, rows, torch.ones(len(rows), dtype=torch.int16))
        flow_records.append({
            "cube_id": int(rec["cube_id"]), "block_index": idx,
            "start": tuple(int(v) for v in start.tolist()),
            "global_row_ids": rows, "owned_row_ids": rows, "local_xyz": local_xyz,
        })
    if not torch.all(coverage == 1):
        raise RuntimeError("Global C256 flow ownership is not exactly one")
    cube_flow.atomic_save(output / "global_support" / "global_c256_support.pt", {
        "format": FORMAT, "coords": coords})
    core.atomic_json(output / "global_support" / "summary.json", {
        "format": FORMAT, "tokens": len(coords), "active_blocks": len(records),
        "concatenated_tokens": raw_count, "duplicate_points": duplicates,
        "coverage_min": int(coverage.min()), "coverage_max": int(coverage.max()),
        "records": [{
            "cube_id": int(r["cube_id"]), "block_index": r["block_index"],
            "owner_tokens": int(r["baseline_tokens"]), "context_tokens": int(r["context_tokens"]),
            "generated_c64_tokens": int(r["generated_c64_tokens"]),
        } for r in records],
    })
    return coords, flow_records


def parse_args() -> argparse.Namespace:
    base = Path("outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, default=Path("assets/images/0_img.png"))
    p.add_argument("--camera", type=Path, default=base / "global_camera.json")
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--baseline-c64", type=Path, default=Path(
        "outputs/global_c256_restructured_blocks_cuda5/baseline_pre_hr/baseline_c64_support.pt"))
    p.add_argument("--output", type=Path, default=Path(
        "outputs/global_c256_c32_context_owner_crop_dec1_all_blocks_cuda5"))
    p.add_argument("--shape512-endpoint", type=Path, default=Path(
        "outputs/global_c256_c32_context_owner_dec1_all_blocks_cuda5/shape512_context_owner/final_normalized.pt"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--physical-cuda", type=int, default=5)
    p.add_argument("--seed", type=int, default=45001)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--shape-seed", type=int, default=46001)
    p.add_argument("--texture-seed", type=int, default=47001)
    p.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stop-after-support", action="store_true")
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
    core.atomic_json(output / "config.json", {
        "format": FORMAT, "args": vars(args), "camera": camera,
        "first_flow": "one Global C64 noise/state; centered C32 contexts stride16; center C16 hard-owner writes",
        "support_decode": "C32 decoder x1; retain [16,48) center owner; map with (coord-16)*2 to disjoint C64 block",
        "hr_flow": "one Global C256 noise/state; 45 local C64 predictions; timestep barrier",
        "image_condition": "full image global token + correct global-position projected feature",
        "decode": "single Global C256 -> 4096 geometry/material decode",
    })

    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=args.low_vram)
    canonical = pipeline.preprocess_canonical_images(Image.open(args.image))
    image512, image1024 = canonical["image_512"], canonical["image_1024"]
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    image512.save(inputs / "global_512.png")
    image1024.save(inputs / "global_1024.png")
    baseline64 = torch.load(args.baseline_c64, map_location="cpu", weights_only=False)["coords"].int()
    records = build_context_records(baseline64)
    core.atomic_json(output / "context_blocks.json", {
        "format": FORMAT, "baseline_c64_tokens": len(baseline64), "active_blocks": len(records),
        "records": [{
            "cube_id": int(r["cube_id"]), "block_index": r["block_index"],
            "owner_start": r["owner_start"], "context_start": r["context_start"],
            "owner_tokens": int(r["baseline_tokens"]), "context_tokens": int(r["context_tokens"]),
        } for r in records],
    })
    print(f"[active] baseline C64={len(baseline64):,} owners={len(records)}", flush=True)

    if args.shape512_endpoint.is_file():
        endpoint = torch.load(
            args.shape512_endpoint, map_location="cpu", weights_only=False)["features"].float()
        if endpoint.shape != (len(baseline64), 32):
            raise RuntimeError(f"external Shape512 endpoint shape mismatch: {tuple(endpoint.shape)}")
        print(f"[reuse-shape512] {args.shape512_endpoint} tokens={len(endpoint):,}", flush=True)
    else:
        condition = build_context_conditions(pipeline, image512, camera, records, output, device)
        endpoint = context_owner_shape512_flow(
            pipeline, baseline64, records, condition, output, device, args.seed, args.steps)
        del condition
    empty_cuda()
    coords, flow_records = decode_contexts_to_global_support(
        pipeline, endpoint, records, output, device)
    print(f"[global-support] tokens={len(coords):,} duplicate_points=0 coverage=1", flush=True)
    if args.stop_after_support:
        print(f"[done-support] {output}", flush=True)
        return

    shape_condition = core.build_global_conditions(
        pipeline, image1024, camera, coords, flow_records, "shape", output, device)
    shape = core.synchronous_flow(
        stage="shape", pipeline=pipeline, records=flow_records, condition=shape_condition,
        output=output, device=device, seed=args.shape_seed, steps=args.steps, concat=None)
    del shape_condition
    empty_cuda()
    texture_condition = core.build_global_conditions(
        pipeline, image1024, camera, coords, flow_records, "texture", output, device)
    texture = core.synchronous_flow(
        stage="texture", pipeline=pipeline, records=flow_records, condition=texture_condition,
        output=output, device=device, seed=args.texture_seed, steps=args.steps, concat=shape)
    del texture_condition
    empty_cuda()
    decoded = core.decode_global(pipeline, coords, shape, texture, output, device)
    core.atomic_json(output / "summary.json", {
        "format": FORMAT, "status": "complete", "baseline_c64_tokens": len(baseline64),
        "active_blocks": len(records), "global_c256_tokens": len(coords),
        "duplicate_points": 0, "decode": decoded, "seconds": time.perf_counter() - started,
    })
    print(f"[done] {output}", flush=True)


if __name__ == "__main__":
    main()
