#!/usr/bin/env python3
"""Render the native C64 baseline x0 endpoint before C4096 re-encoding."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import torch

import pixal3d_global4096_singleview_shared_slat_shape_tex_sr as native_sr
import pixal3d_render_global4096_multiview as multiview
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor


def _atomic_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load(path: Path, device: torch.device) -> SparseTensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return SparseTensor(
        payload["endpoint_normalized_feats"].to(device),
        payload["coords"].to(device),
    )


def _denormalize(value: SparseTensor, normalization) -> SparseTensor:
    std = torch.as_tensor(normalization["std"], device=value.device)[None]
    mean = torch.as_tensor(normalization["mean"], device=value.device)[None]
    return value.replace(value.feats * std + mean)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--cuda-device", type=int, default=5)
    parser.add_argument("--angles", default="0,120,240")
    args = parser.parse_args()
    torch.cuda.set_device(args.cuda_device)
    device = torch.device("cuda", args.cuda_device)
    root = args.endpoint_dir.expanduser().resolve()
    pipeline = init_pipeline(str(args.model_path), device="cuda", low_vram=True)
    shape = _denormalize(
        _load(root / "shape_endpoint_latents" / f"step_{args.step:02d}.pt", device),
        pipeline.shape_slat_normalization,
    )
    texture = _denormalize(
        _load(root / "texture_endpoint_latents" / f"step_{args.step:02d}.pt", device),
        pipeline.tex_slat_normalization,
    )
    mesh = pipeline.decode_latent(shape, texture, 1024)[0]
    output = root / f"native_step_{args.step:02d}_visualization"
    vertex, face = native_sr._native_mesh_to_pbr(mesh, device)
    vertex_path = output / "per_vertex_pbr_mesh.pt"
    _atomic_save(vertex_path, {"mesh": vertex, "resolution": 1024})
    _atomic_save(output / "per_face_pbr_mesh.pt", {"mesh": face, "resolution": 1024})
    multiview.render(SimpleNamespace(
        mesh=vertex_path,
        camera=root / "global_camera.json",
        output_dir=output / "multiview",
        angles=args.angles,
        resolution=1024,
        face_chunk_size=4_000_000,
        device=str(device),
        force=False,
    ))
    print(f"[done] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
