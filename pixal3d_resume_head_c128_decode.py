#!/usr/bin/env python3
"""Recover and decode a completed independent-head C128 Flow state."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import torch

from pixal3d.modules.sparse import SparseTensor
from pixal3d_c128_block_local_cascade_shape_renoise_2048 import empty_cuda
from pixal3d_independent_head_c128_baseline import init_pipeline


ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--camera", type=Path, required=True)
    p.add_argument("--model-path", type=Path, default=Path("/home/nvme04/yyyan/download/model/Pixal3D"))
    p.add_argument("--cuda-device", type=int, default=4)
    p.add_argument("--angles", default="0,60,180,240")
    args = p.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.cuda_device):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}, expected physical CUDA {args.cuda_device}")
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    out = args.output_dir.resolve()
    payload = torch.load(out / "latents/shape_c128_denormalized.pt", map_location="cpu", weights_only=False)
    features = payload["features"]
    if isinstance(features, SparseTensor):
        features = features.feats.detach().cpu()
    features = features.float()
    coords = payload["coords"].int()
    print(f"[resume] C128 tokens={len(coords):,} feature_shape={tuple(features.shape)}", flush=True)
    pipeline = init_pipeline(args.model_path.resolve(), device)
    slat = SparseTensor(features.to(device), coords.to(device))
    print("[decode] C128 -> geometry at 2048", flush=True)
    meshes, _ = pipeline.decode_shape_slat(slat, 2048)
    if len(meshes) != 1:
        raise RuntimeError(f"expected one mesh, got {len(meshes)}")
    mesh = meshes[0]
    torch.save(
        {"format": "pixal3d_independent_head_c128_baseline_v1", "mesh": mesh.cpu()},
        out / "final/geometry_mesh.pt",
    )
    del meshes, mesh, slat, pipeline
    empty_cuda()
    from pixal3d_baseline1024_c128_8xc64_geometry import render_geometry

    camera_payload = json.loads(args.camera.resolve().read_text())
    camera = camera_payload.get("camera", camera_payload)
    mesh_payload = torch.load(out / "final/geometry_mesh.pt", map_location="cpu", weights_only=False)
    live = mesh_payload["mesh"].to(device)
    angles = tuple(int(x.strip()) % 360 for x in args.angles.split(",") if x.strip())
    render = render_geometry(live, camera, args.image.resolve(), args.mask.resolve(), out, 1024, angles, 200_000)
    summary = {
        "format": "pixal3d_independent_head_c128_baseline_v1",
        "status": "complete",
        "vertices": int(len(live.vertices)),
        "faces": int(len(live.faces)),
        "render": render,
        "resumed_from": str((out / "latents/shape_c128_denormalized.pt").resolve()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
