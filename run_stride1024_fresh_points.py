#!/usr/bin/env python3
"""Generate fresh C64 supports for all 4x4 disjoint 1024 crops of a 4096 image.

This intentionally keeps the crop pixels untouched and uses one fixed camera
derived from the full-image camera.  It diagnoses what independent local
generation does, without MoGe or a second foreground recenter step.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import numpy as np
import torch
from PIL import Image

import pixal3d_cascade512_1024_tiled2048_crop_condition as tiled
from pixal3d.modules.sparse import SparseTensor


ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = ROOT / "可视化网页2/reference_4096.png"
DEFAULT_CAMERA = ROOT / "可视化网页2/camera.json"
DEFAULT_OUT = ROOT / "outputs/stride1024_fresh_points_cuda4"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    p.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model-path", type=Path, default=Path(tiled.DEFAULT_MODEL_PATH))
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--seed", type=int, default=4201)
    return p.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_sparse(path: Path, value: SparseTensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"coords": value.coords.detach().cpu(), "feats": value.feats.detach().cpu()}, path)


def load_sparse(path: Path, device: torch.device) -> SparseTensor:
    value = torch.load(path, map_location="cpu", weights_only=False)
    return SparseTensor(feats=value["feats"].to(device), coords=value["coords"].to(device))


def support_stats(coords: torch.Tensor, resolution: int) -> dict:
    xyz = coords[:, 1:].detach().cpu().to(torch.int64)
    lo = xyz.amin(0).tolist() if len(xyz) else [None] * 3
    hi = xyz.amax(0).tolist() if len(xyz) else [None] * 3
    boundary = ((xyz == 0) | (xyz == resolution - 1)).any(1)
    centered75 = ((xyz >= resolution * 0.125) & (xyz < resolution * 0.875)).all(1)
    return {
        "count": int(len(xyz)), "range_min": lo, "range_max": hi,
        "boundary_ratio": float(boundary.float().mean()) if len(xyz) else 0.0,
        "central_75_ratio": float(centered75.float().mean()) if len(xyz) else 0.0,
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    from inference import init_pipeline
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4":
        raise RuntimeError("run with CUDA_VISIBLE_DEVICES=4; physical GPU 4 becomes logical cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = Image.open(args.image).convert("RGB")
    if source.size != (4096, 4096):
        raise ValueError(f"expected 4096x4096 source, got {source.size}")
    global_camera = json.loads(args.camera.read_text(encoding="utf-8"))
    # A 1024 crop keeps one quarter of each image dimension.  Recenter that
    # crop as an independent camera while retaining the source focal length.
    local_fov = 2 * math.atan(math.tan(float(global_camera["camera_angle_x"]) / 2) / 4)
    # Same convention as the native baseline: the canonical cube width spans
    # essentially the full local image at the cube center.
    local_distance = 1 / (2 * math.tan(local_fov / 2))
    camera = {"camera_angle_x": local_fov, "distance": local_distance, "mesh_scale": 1.0}
    config = {
        "format": "pixal3d_stride1024_fresh_points_v1",
        "status": "running", "image": str(args.image.resolve()),
        "layout": "4096 image, context=1024, stride=1024, 4x4 disjoint tiles",
        "path": "raw tile (no recrop/recenter/MoGe) -> fresh SS C32 -> Shape512 -> learned C64 support",
        "camera": camera, "global_camera": global_camera,
        "same_noise_seed_for_every_tile": args.seed, "steps": args.steps,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": str(device), "gpu": torch.cuda.get_device_name(device),
    }
    atomic_json(out / "config.json", config)
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=True)
    pipeline.sparse_structure_sampler_params["steps"] = args.steps
    pipeline.shape_slat_sampler_params["steps"] = args.steps
    records = []
    started = time.perf_counter()
    for tile_id in range(16):
        row, col = divmod(tile_id, 4)
        box = (col * 1024, row * 1024, (col + 1) * 1024, (row + 1) * 1024)
        root = out / f"tile_{tile_id:02d}"
        raw_path = root / "input_1024.png"
        root.mkdir(parents=True, exist_ok=True)
        image1024 = source.crop(box)
        image512 = image1024.resize((512, 512), Image.Resampling.LANCZOS)
        image1024.save(raw_path)
        image512.save(root / "input_512.png")
        c32_path, shape_path, c64_path = root / "coords_c32.pt", root / "shape_c32.pt", root / "coords_c64.pt"
        t0 = time.perf_counter()
        if c64_path.is_file():
            coords64 = torch.load(c64_path, map_location="cpu", weights_only=False)["coords"]
            coords32 = torch.load(c32_path, map_location="cpu", weights_only=False)["coords"]
            state = "cached"
        else:
            torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
            np.random.seed(args.seed); random.seed(args.seed)
            if c32_path.is_file():
                coords32 = torch.load(c32_path, map_location="cpu", weights_only=False)["coords"].to(device)
            else:
                cond = pipeline.get_proj_cond_ss([image512], camera_angle_x=local_fov,
                    distance=local_distance, mesh_scale=1.0)
                coords32 = pipeline.sample_sparse_structure(cond, 32)
                torch.save({"coords": coords32.cpu()}, c32_path)
                del cond
                tiled.empty_cuda()
            if shape_path.is_file():
                shape32 = load_sparse(shape_path, device)
            else:
                torch.manual_seed(args.seed + 1); torch.cuda.manual_seed_all(args.seed + 1)
                cond = pipeline.get_proj_cond_shape(pipeline.image_cond_model_shape_512,
                    [image512], coords32, camera_angle_x=local_fov,
                    distance=local_distance, mesh_scale=1.0)
                shape32 = pipeline.sample_shape_slat(
                    cond, pipeline.models["shape_slat_flow_model_512"], coords32)
                save_sparse(shape_path, shape32)
                del cond
                tiled.empty_cuda()
            coords64 = tiled.upsample_and_quantize(pipeline, shape32, 512, 64, True).cpu()
            torch.save({"coords": coords64}, c64_path)
            del shape32
            tiled.empty_cuda()
            state = "generated"
        record = {
            "tile_id": tile_id, "row": row, "col": col, "box_4096": box,
            "input": str(raw_path), "state": state,
            "c32": support_stats(coords32, 32), "c64": support_stats(coords64, 64),
            "seconds_this_run": time.perf_counter() - t0,
        }
        atomic_json(root / "summary.json", record)
        records.append(record)
        print(f"[tile {tile_id:02d}/15] C32={len(coords32):,} C64={len(coords64):,} {state} {record['seconds_this_run']:.1f}s", flush=True)
        del coords32, coords64
        tiled.empty_cuda()
    config["status"] = "complete"
    config["seconds"] = time.perf_counter() - started
    config["tiles"] = records
    config["total_c32"] = sum(r["c32"]["count"] for r in records)
    config["total_c64"] = sum(r["c64"]["count"] for r in records)
    atomic_json(out / "summary.json", config)
    print(json.dumps({"status": "complete", "seconds": config["seconds"],
                      "total_c32": config["total_c32"], "total_c64": config["total_c64"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
