#!/usr/bin/env python3
"""Direct Shape1024/Texture1024 flow and one 4096 decode on Global C256."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("PIXAL3D_LOW_MEMORY_DECODER", "1")

import torch
from PIL import Image

import pixal3d_global_c256_c64_stride32_owner_flow_singleview as owner_stage
import pixal3d_global_c256_restructured_blocks_singleview as core
from inference import MODEL_PATH, init_pipeline


FORMAT = "pixal3d_global_c256_direct_shape_tex_decode_singleview_v1"


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def direct_record(coords: torch.Tensor) -> dict:
    rows = torch.arange(len(coords), dtype=torch.long)
    xyz = coords[:, 1:4].cpu().int()
    if torch.any(xyz < 0) or torch.any(xyz >= 256):
        raise RuntimeError("Global C256 coordinates lie outside [0,255]")
    return {
        "cube_id": 0,
        "owner_index": (0, 0, 0),
        "owner_start": (0, 0, 0),
        "context_start": (0, 0, 0),
        "global_row_ids": rows,
        "owned_row_ids": rows,
        "owner_context_positions": rows,
        "local_xyz": xyz,
    }


def parse_args() -> argparse.Namespace:
    base = Path("outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, default=Path("assets/images/0_img.png"))
    p.add_argument("--camera", type=Path, default=base / "global_camera.json")
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--support", type=Path, default=Path(
        "outputs/global_c256_c32_stride32_dec2_support_cuda5/global_support/global_c256_support.pt"))
    p.add_argument("--output", type=Path, default=Path(
        "outputs/global_c256_dec2_support_direct_shape_tex_decode_cuda5"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--physical-cuda", type=int, default=5)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--shape-seed", type=int, default=52001)
    p.add_argument("--texture-seed", type=int, default=53001)
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
    coords = torch.load(args.support, map_location="cpu", weights_only=False)["coords"].int()
    rec = direct_record(coords)
    core.atomic_json(output / "config.json", {
        "format": FORMAT, "args": vars(args), "camera": camera,
        "support_tokens": len(coords),
        "flow_partition": None,
        "shape_flow": "one Global C256 noise/state; one full Shape1024 sparse forward per timestep",
        "texture_flow": "one Global C256 noise/state; one full Texture1024 sparse forward per timestep; final Shape concat",
        "decode": "one Global C256 -> 4096 geometry/material decode",
    })
    print(f"[input] Global_C256={len(coords):,} direct_full_tensor=True", flush=True)
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=args.low_vram)
    canonical = pipeline.preprocess_canonical_images(Image.open(args.image))
    image1024 = canonical["image_1024"]
    (output / "inputs").mkdir(parents=True, exist_ok=True)
    image1024.save(output / "inputs" / "global_1024.png")

    shape_cond = owner_stage.build_global_point_condition(
        pipeline, image1024, camera, coords, "shape", output, device)
    shape = owner_stage.owner_flow(
        stage="shape", pipeline=pipeline, records=[rec], point_condition=shape_cond,
        output=output, device=device, seed=args.shape_seed, steps=args.steps,
        concat=None, max_batch_size=1, max_batch_tokens=len(coords) + 1)
    del shape_cond
    empty_cuda()

    texture_cond = owner_stage.build_global_point_condition(
        pipeline, image1024, camera, coords, "texture", output, device)
    texture = owner_stage.owner_flow(
        stage="texture", pipeline=pipeline, records=[rec], point_condition=texture_cond,
        output=output, device=device, seed=args.texture_seed, steps=args.steps,
        concat=shape, max_batch_size=1, max_batch_tokens=len(coords) + 1)
    del texture_cond
    empty_cuda()

    decoded = core.decode_global(pipeline, coords, shape, texture, output, device)
    core.atomic_json(output / "summary.json", {
        "format": FORMAT, "status": "complete", "global_c256_tokens": len(coords),
        "shape_steps": args.steps, "texture_steps": args.steps,
        "decode": decoded, "seconds": time.perf_counter() - started,
    })
    print(f"[done] output={output} decode={decoded}", flush=True)


if __name__ == "__main__":
    main()
