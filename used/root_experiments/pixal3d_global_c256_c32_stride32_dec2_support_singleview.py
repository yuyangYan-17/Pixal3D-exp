#!/usr/bin/env python3
"""Build a refined Global-C256 support from disjoint Global-C64 C32 blocks.

The native baseline pre-Shape1024 C64 support is split into non-overlapping
C32 blocks (stride 32).  A single Global-C64 Shape512 noise/state is evolved
synchronously: each local block predicts all of its rows and the global state
is updated only after every block has completed the timestep.  Each final C32
latent is then decoded with ``shape_slat_decoder.upsample(..., 2)`` into a full
local C128 support and placed into its disjoint C128 region of Global C256.
No Shape1024, Texture, or mesh decode is run here.
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


FORMAT = "pixal3d_global_c256_c32_stride32_dec2_support_singleview_v1"


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_records(coords64: torch.Tensor) -> list[dict[str, Any]]:
    xyz = coords64[:, 1:4].cpu().int()
    coverage = torch.zeros(len(coords64), dtype=torch.int16)
    records: list[dict[str, Any]] = []
    cube_id = 0
    for bx in range(2):
        for by in range(2):
            for bz in range(2):
                index = (bx, by, bz)
                start = torch.tensor(index, dtype=torch.int32) * 32
                mask = ((xyz >= start) & (xyz < start + 32)).all(1)
                if not mask.any():
                    continue
                rows = torch.where(mask)[0].long()
                coverage.index_add_(0, rows, torch.ones(len(rows), dtype=torch.int16))
                records.append({
                    "cube_id": cube_id,
                    "block_index": index,
                    "start_c64": tuple(int(v) for v in start.tolist()),
                    "global_row_ids": rows,
                    "owned_row_ids": rows,
                    "owner_context_positions": torch.arange(len(rows), dtype=torch.long),
                    "local_xyz": xyz.index_select(0, rows) - start,
                })
                cube_id += 1
    if not torch.all(coverage == 1):
        raise RuntimeError(
            f"C32/stride32 partition is not exact: min={int(coverage.min())} "
            f"max={int(coverage.max())}")
    return records


@torch.no_grad()
def build_conditions(
    pipeline: Any, image512: Image.Image, camera: Mapping[str, float],
    records: list[dict[str, Any]], output: Path, device: torch.device,
) -> dict[str, Any]:
    root = output / "conditions" / "shape512_c32_stride32"
    model = pipeline.image_cond_model_shape_512
    pending = [r for r in records if not (root / f"block_{int(r['cube_id']):02d}.pt").is_file()]
    if pending:
        cached = core.extract_full_image_features(model, image512, device)
        fov = float(camera["camera_angle_x"])
        distance = float(camera["distance"])
        for order, rec in enumerate(pending, 1):
            bid = int(rec["cube_id"])
            local_coords = cube_flow._local_coords(rec).to(device)
            start = tuple(int(v) for v in rec["start_c64"])
            transform = core.local_to_global_camera_transform(
                model, block_start=start, global_resolution=64,
                global_extent=32, distance=distance, device=device)
            error = core.validate_projection_transform(
                model, transform, local_resolution=32, global_resolution=64,
                block_start=start, distance=distance, fov=fov, device=device)
            glob, proj = core.project_cached_features(
                model, cached, transform=transform, fov=fov, distance=distance,
                coords=local_coords, grid_resolution=32)
            cube_flow.atomic_save(root / f"block_{bid:02d}.pt", {
                "format": FORMAT, "cube_id": bid,
                "block_index": rec["block_index"], "start_c64": start,
                "global_row_ids": rec["global_row_ids"],
                "global": glob.cpu(), "proj": proj[0].cpu(),
                "projection_max_error_pixels": error,
                "image_condition": "full-image global token + exact Global-C64-position projection",
            })
            print(f"[condition] {order}/{len(pending)} block={bid:02d} "
                  f"tokens={len(local_coords):,} projection_error={error:.3g}", flush=True)
            del local_coords, glob, proj
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
            raise RuntimeError(f"cached condition row mismatch for block {bid}")
        cubes[bid] = payload
    return {"cubes": cubes, "fingerprint_sha256": FORMAT}


@torch.no_grad()
def synchronous_shape512(
    pipeline: Any, coords64: torch.Tensor, records: list[dict[str, Any]],
    condition: Mapping[str, Any], output: Path, device: torch.device,
    seed: int, steps: int,
) -> torch.Tensor:
    root = output / "shape512"
    final_path = root / "final_normalized.pt"
    if final_path.is_file():
        return torch.load(final_path, map_location="cpu", weights_only=False)["features"].float()
    model = pipeline.models["shape_slat_flow_model_512"]
    sampler = pipeline.shape_slat_sampler
    params = dict(pipeline.shape_slat_sampler_params)
    params["steps"] = steps
    schedule = sampler.timestep_schedule(steps, float(params.get("rescale_t", 1.0)))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = torch.randn((len(coords64), int(model.in_channels)), generator=generator)
    completed = 0
    for step in range(steps, 0, -1):
        checkpoint = root / f"step_{step:02d}.pt"
        if checkpoint.is_file():
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)["features"].float()
            completed = step
            break
    model.to(device).eval()
    history = []
    for step in range(completed, steps):
        t, t_next = float(schedule[step]), float(schedule[step + 1])
        velocity = torch.empty_like(state)
        writes = torch.zeros(len(state), dtype=torch.int16)
        values, timing = cube_flow._one_prediction(
            records, state, condition, sampler, model, params, t, t_next, device, None)
        for rec, value in zip(records, values):
            rows = rec["global_row_ids"]
            velocity.index_copy_(0, rows, value)
            writes.index_add_(0, rows, torch.ones(len(rows), dtype=torch.int16))
        if not torch.all(writes == 1):
            raise RuntimeError("Shape512 write coverage is not exactly one")
        state = cube_flow.jacobi_update(state, velocity, t, t_next)
        if not torch.isfinite(state).all():
            raise FloatingPointError(f"non-finite Shape512 state after step {step + 1}")
        cube_flow.atomic_save(root / f"step_{step + 1:02d}.pt", {
            "format": FORMAT, "features": state})
        history.append({"step": step + 1, "t": t, "t_next": t_next, **timing})
        print(f"[shape512] step={step + 1}/{steps} blocks={len(records)} "
              f"tokens={len(coords64):,} seconds={timing['seconds']:.2f}", flush=True)
    model.cpu()
    empty_cuda()
    cube_flow.atomic_save(final_path, {"format": FORMAT, "features": state})
    core.atomic_json(root / "summary.json", {
        "format": FORMAT, "tokens": len(state), "channels": state.shape[1],
        "blocks": len(records), "steps": steps, "seed": seed, "history": history})
    return state


@torch.no_grad()
def decode_c128_blocks(
    pipeline: Any, endpoint: torch.Tensor, records: list[dict[str, Any]],
    output: Path, device: torch.device,
) -> torch.Tensor:
    root = output / "local_c128"
    decoder = pipeline.models["shape_slat_decoder"]
    raw = cube_flow.denormalize(endpoint, pipeline.shape_slat_normalization)
    pending = [r for r in records if not (root / f"block_{int(r['cube_id']):02d}" / "c128_support.pt").is_file()]
    if pending and pipeline.low_vram:
        decoder.to(device)
        decoder.low_vram = True
    for order, rec in enumerate(pending, 1):
        bid = int(rec["cube_id"])
        rows = rec["global_row_ids"]
        local_coords = cube_flow._local_coords(rec)
        latent = SparseTensor(raw.index_select(0, rows).to(device), local_coords.to(device))
        candidates = decoder.upsample(latent, upsample_times=2)
        coords128 = candidates.int().unique(dim=0).cpu().contiguous()
        xyz = coords128[:, 1:4]
        if len(coords128) == 0 or torch.any(xyz < 0) or torch.any(xyz >= 128):
            raise RuntimeError(
                f"invalid block {bid} C128 coordinates: "
                f"min={xyz.amin(0).tolist()} max={xyz.amax(0).tolist()}")
        block_dir = root / f"block_{bid:02d}"
        block_dir.mkdir(parents=True, exist_ok=True)
        cube_flow.atomic_save(block_dir / "c128_support.pt", {
            "format": FORMAT, "block_id": bid, "block_index": rec["block_index"],
            "coords": coords128, "input_tokens": len(rows),
            "decoder_upsample_times": 2, "candidate_resolution": 128,
            "crop": None,
        })
        print(f"[decoder-x2] {order}/{len(pending)} block={bid:02d} "
              f"C32={len(rows):,} C128={len(coords128):,}", flush=True)
        del latent, candidates, coords128
        empty_cuda()
    if pending and pipeline.low_vram:
        decoder.cpu()
        decoder.low_vram = False
    empty_cuda()

    parts = []
    raw_count = 0
    block_rows = []
    for rec in records:
        bid = int(rec["cube_id"])
        payload = torch.load(root / f"block_{bid:02d}" / "c128_support.pt", map_location="cpu", weights_only=False)
        local = payload["coords"].int()
        offset = torch.tensor(rec["block_index"], dtype=torch.int32) * 128
        global_xyz = local[:, 1:4] + offset
        global_coords = torch.cat((torch.zeros((len(local), 1), dtype=torch.int32), global_xyz), 1)
        parts.append(global_coords)
        raw_count += len(local)
        block_rows.append({
            "block_id": bid, "block_index": rec["block_index"],
            "input_c32_tokens": len(rec["global_row_ids"]),
            "output_c128_tokens": len(local),
            "start_c256": tuple(int(v) for v in offset.tolist()),
        })
    concatenated = torch.cat(parts, 0).int()
    coords256 = concatenated.unique(dim=0).int().contiguous()
    duplicates = raw_count - len(coords256)
    xyz = coords256[:, 1:4]
    if torch.any(xyz < 0) or torch.any(xyz >= 256):
        raise RuntimeError("assembled support lies outside Global C256")
    if duplicates != 0:
        raise RuntimeError(f"disjoint C128 assembly unexpectedly has {duplicates} duplicate points")
    cube_flow.atomic_save(output / "global_support" / "global_c256_support.pt", {
        "format": FORMAT, "coords": coords256,
        "source": "disjoint C32/stride32 Shape512 blocks decoded to full C128 with x2",
    })
    core.atomic_json(output / "global_support" / "summary.json", {
        "format": FORMAT, "tokens": len(coords256), "blocks": len(records),
        "concatenated_tokens": raw_count, "duplicate_points": duplicates,
        "coordinate_min": xyz.amin(0).tolist(), "coordinate_max": xyz.amax(0).tolist(),
        "records": block_rows,
    })
    return coords256


def parse_args() -> argparse.Namespace:
    base = Path("outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, default=Path("assets/images/0_img.png"))
    p.add_argument("--camera", type=Path, default=base / "global_camera.json")
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--baseline-c64", type=Path, default=Path(
        "outputs/global_c256_restructured_blocks_cuda5/baseline_pre_hr/baseline_c64_support.pt"))
    p.add_argument("--output", type=Path, default=Path(
        "outputs/global_c256_c32_stride32_dec2_support_cuda5"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--physical-cuda", type=int, default=5)
    p.add_argument("--seed", type=int, default=51001)
    p.add_argument("--steps", type=int, default=12)
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
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=args.low_vram)
    canonical = pipeline.preprocess_canonical_images(Image.open(args.image))
    image512 = canonical["image_512"]
    (output / "inputs").mkdir(parents=True, exist_ok=True)
    image512.save(output / "inputs" / "global_512.png")
    coords64 = torch.load(args.baseline_c64, map_location="cpu", weights_only=False)["coords"].int()
    records = build_records(coords64)
    core.atomic_json(output / "config.json", {
        "format": FORMAT, "args": vars(args), "camera": camera,
        "partition": "Global C64: disjoint C32 context, stride32; no overlap and no owner crop",
        "flow": "one Global C64 Shape512 noise/state; all blocks finish before each update",
        "support_decode": "full local C32 latent -> decoder x2 -> full local C128 -> disjoint Global C256 C128 block",
        "later_flow": None,
    })
    print(f"[input] baseline_C64={len(coords64):,} nonempty_C32_blocks={len(records)}", flush=True)
    condition = build_conditions(pipeline, image512, camera, records, output, device)
    endpoint = synchronous_shape512(
        pipeline, coords64, records, condition, output, device, args.seed, args.steps)
    del condition
    empty_cuda()
    coords256 = decode_c128_blocks(pipeline, endpoint, records, output, device)
    core.atomic_json(output / "summary.json", {
        "format": FORMAT, "status": "complete", "baseline_c64_tokens": len(coords64),
        "blocks": len(records), "global_c256_tokens": len(coords256),
        "duplicate_points": 0, "seconds": time.perf_counter() - started,
    })
    print(f"[done] Global_C256={len(coords256):,} duplicate_points=0 output={output}", flush=True)


if __name__ == "__main__":
    main()
