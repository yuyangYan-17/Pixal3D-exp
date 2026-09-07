#!/usr/bin/env python3
"""Save all uniform-time C64 texture x0 endpoints as native 1024 O-Voxels."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import torch
from PIL import Image

from inference import MODEL_PATH, init_pipeline
from pixal3d.representations import MeshWithVoxel


FORMAT = "pixal3d_baseline1024_uniform_texture_endpoints_v1"
ROOT = Path("outputs/baseline1024_raw_ovoxel_cuda4_0_img")
DEFAULT_OUTPUT = Path("outputs/baseline1024_uniform_texture_endpoints_cuda4")


def atomic_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def empty_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "canonical_1024.png")
    parser.add_argument("--camera", type=Path, default=ROOT / "global_camera.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-cuda", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--texture-rescale-t", type=float, default=1.0)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and [x.strip() for x in visible.split(",")] != [str(args.physical_cuda)]:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}; expected physical CUDA {args.physical_cuda}")
    if args.steps != 12 or args.texture_rescale_t != 1.0:
        raise ValueError("this experiment requires 12 uniform texture steps (rescale_t=1)")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    complete = out / "summary.json"
    if args.resume and complete.is_file():
        summary = json.loads(complete.read_text(encoding="utf-8"))
        if summary.get("status") == "complete":
            print(f"[resume] complete: {out}")
            return 0

    image = Image.open(args.image).convert("RGB")
    if image.size != (1024, 1024):
        raise RuntimeError(f"expected canonical 1024 image, got {image.size}")
    camera = json.loads(args.camera.read_text(encoding="utf-8"))
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=args.low_vram)
    endpoint_rows: list[dict[str, Any]] = []

    def capture(**payload: Any) -> None:
        step = int(payload["step_index"]) + 1
        endpoint = payload["endpoint"]
        path = out / "texture_endpoint_latents" / f"step_{step:02d}.pt"
        feats = endpoint.feats.detach().float().cpu().contiguous()
        coords = endpoint.coords.detach().to(torch.int32).cpu().contiguous()
        record = {
            "step": step,
            "t": float(payload["t"]),
            "t_next": float(payload["t_next"]),
            "path": str(path.resolve()),
            "endpoint_sha256": tensor_hash(feats),
        }
        atomic_save(path, {
            "format": FORMAT,
            **record,
            "coords": coords,
            "endpoint_normalized_feats": feats,
            "formula": "x0=(1-sigma_min)*x_t-(sigma_min+(1-sigma_min)*t)*v",
        })
        endpoint_rows.append(record)
        print(f"[baseline endpoint] step={step:02d} t={record['t']:.6f}", flush=True)

    sparse_params = {
        "steps": 12,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.7,
        "rescale_t": 5.0,
    }
    shape_params = {
        "steps": 12,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.5,
        "rescale_t": 3.0,
    }
    texture_params = {
        "steps": args.steps,
        "guidance_strength": 1.0,
        "guidance_rescale": 0.0,
        "guidance_interval": (0.0, 1.0),
        "rescale_t": args.texture_rescale_t,
        "endpoint_callback": capture,
        "return_model_history": False,
    }
    print("[baseline] run 1024 cascade and capture uniform texture endpoints", flush=True)
    generated, (shape_slat, final_texture, resolution) = pipeline.run(
        image,
        camera_params=dict(camera),
        seed=args.seed,
        sparse_structure_sampler_params=sparse_params,
        shape_slat_sampler_params=shape_params,
        tex_slat_sampler_params=texture_params,
        preprocess_image=False,
        return_latent=True,
        pipeline_type="1024_cascade",
        max_num_tokens=1_000_000,
    )
    if int(resolution) != 1024 or len(endpoint_rows) != args.steps:
        raise RuntimeError(f"expected resolution=1024/endpoints=12, got {resolution}/{len(endpoint_rows)}")
    del generated, final_texture
    empty_cuda()

    atomic_save(out / "final_shape_slat.pt", {
        "format": FORMAT,
        "coords": shape_slat.coords.detach().cpu(),
        "raw_feats": shape_slat.feats.detach().float().cpu(),
    })
    print("[baseline decode] decode final shape once", flush=True)
    shape_meshes, subs = pipeline.decode_shape_slat(shape_slat, 1024)
    geometry = shape_meshes[0]
    shared_geometry = out / "ovoxel1024" / "shared_geometry.pt"
    atomic_save(shared_geometry, {
        "format": FORMAT,
        "vertices": geometry.vertices.detach().float().cpu(),
        "faces": geometry.faces.detach().to(torch.int32).cpu(),
        "resolution": 1024,
    })

    tex_mean = torch.as_tensor(pipeline.tex_slat_normalization["mean"], device=device)[None]
    tex_std = torch.as_tensor(pipeline.tex_slat_normalization["std"], device=device)[None]
    ov_rows: list[dict[str, Any]] = []
    for record in endpoint_rows:
        saved = torch.load(record["path"], map_location="cpu", weights_only=False)
        normalized = shape_slat.replace(saved["endpoint_normalized_feats"].to(device))
        raw = normalized * tex_std + tex_mean
        decoded_tex = pipeline.decode_tex_slat(raw, subs)[0]
        field_path = out / "ovoxel1024" / f"texture_endpoint_step_{record['step']:02d}.pt"
        payload = {
            "format": FORMAT,
            "step": record["step"],
            "t": record["t"],
            "t_next": record["t_next"],
            "shared_geometry": str(shared_geometry.resolve()),
            "origin": torch.tensor([-0.5, -0.5, -0.5], dtype=torch.float32),
            "voxel_size": 1.0 / 1024,
            "coords": decoded_tex.coords[:, 1:].detach().to(torch.int32).cpu().contiguous(),
            "attrs": decoded_tex.feats.detach().float().cpu().contiguous(),
            "voxel_shape": [*decoded_tex.shape, *decoded_tex.spatial_shape],
            "layout": dict(pipeline.pbr_attr_layout),
            "representation": "MeshWithVoxel material field with shared geometry",
        }
        atomic_save(field_path, payload)
        ov_rows.append({
            "step": record["step"], "t": record["t"], "t_next": record["t_next"],
            "path": str(field_path.resolve()), "active_ovoxels": int(payload["coords"].shape[0]),
            "attrs_sha256": tensor_hash(payload["attrs"]),
        })
        print(f"[ovoxel1024] step={record['step']:02d} rows={payload['coords'].shape[0]:,}", flush=True)
        del saved, normalized, raw, decoded_tex, payload
        empty_cuda()

    schedule = pipeline.tex_slat_sampler.timestep_schedule(args.steps, args.texture_rescale_t)
    summary = {
        "format": FORMAT,
        "status": "complete",
        "seed": args.seed,
        "texture_steps": args.steps,
        "texture_rescale_t": args.texture_rescale_t,
        "texture_schedule": schedule,
        "image_guidance": "conditional, guidance_strength=1",
        "shape_for_all_texture_endpoints": "one final baseline C64 shape SLat",
        "shared_geometry": str(shared_geometry.resolve()),
        "texture_endpoints": endpoint_rows,
        "ovoxel1024_endpoints": ov_rows,
    }
    atomic_json(complete, summary)
    print(f"[done] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
