#!/usr/bin/env python3
"""Run a native geometry-only Pixal3D 1024 cascade on the head crop."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    str(Path(__file__).with_name("autotune_cache.json")),
)

import numpy as np
import torch
from PIL import Image

import pixal3d_cascade512_1024_tiled2048_crop_condition as tiled
from pixal3d.modules.sparse import SparseTensor
from pixal3d_baseline1024_c128_8xc64_geometry import render_geometry, save_sparse


FORMAT = "pixal3d_head_crop_baseline1024_geometry_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--image",
        type=Path,
        default=Path("outputs/c128_head_crop_4096/turtle_head_1024.png"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/c128_head_crop_baseline1024_geometry_cuda4"),
    )
    p.add_argument("--model-path", type=Path, default=Path(tiled.DEFAULT_MODEL_PATH))
    p.add_argument(
        "--moge-model",
        type=Path,
        default=Path("/home/nvme04/yyyan/download/model/moge-2-vitl/model.pt"),
    )
    p.add_argument("--cuda-device", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--render-resolution", type=int, default=1024)
    p.add_argument("--angles", default="0,60,120,180,240,300")
    p.add_argument("--render-chunk-size", type=int, default=200_000)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    from inference import get_camera_params_wild_moge, init_pipeline, load_moge_model

    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.cuda_device):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}, expected {args.cuda_device}")
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=True)
    canonical = pipeline.preprocess_canonical_images(Image.open(args.image).convert("RGB"))
    inputs = out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    canonical["image_512"].save(inputs / "canonical_512.png")
    canonical["image_1024"].save(inputs / "canonical_1024.png")
    canonical["foreground_mask_4096"].resize(
        (1024, 1024), Image.Resampling.LANCZOS
    ).save(inputs / "foreground_mask_1024.png")

    camera_input = inputs / "canonical_1024.png"
    print("[camera] native MoGe estimate for the independent head crop", flush=True)
    moge = load_moge_model(device=str(device), model_name=str(args.moge_model))
    try:
        camera = get_camera_params_wild_moge(
            camera_input,
            moge,
            device=str(device),
            mesh_scale=1.0,
            extend_pixel=0,
            image_resolution=512,
        )
    finally:
        moge.cpu()
        del moge
        gc.collect()
        torch.cuda.empty_cache()
    tiled.atomic_json(out / "camera.json", camera)
    tiled.atomic_json(
        out / "config.json",
        {
            "format": FORMAT,
            "input": args.image,
            "pipeline": "native 1024_cascade geometry path",
            "path": "sparse structure C32 -> Shape512 C32 -> decoder upsample/quant C64 -> Shape1024 C64 -> shape-only decode1024",
            "texture": "not sampled and not decoded",
            "seed": args.seed,
            "steps": args.steps,
        },
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pipeline.sparse_structure_sampler_params["steps"] = int(args.steps)
    pipeline.shape_slat_sampler_params["steps"] = int(args.steps)
    print("[baseline 1/3] sparse structure C32", flush=True)
    cond_ss = pipeline.get_proj_cond_ss(
        [canonical["image_512"]],
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=1.0,
    )
    coords_c32 = pipeline.sample_sparse_structure(cond_ss, 32)
    tiled.atomic_save(out / "support" / "coords_c32.pt", {"coords": coords_c32.cpu()})
    del cond_ss
    tiled.empty_cuda()

    print("[baseline 2/3] Shape512 flow", flush=True)
    cond_c32 = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_512,
        [canonical["image_512"]],
        coords_c32,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=1.0,
    )
    shape_c32 = pipeline.sample_shape_slat(
        cond_c32, pipeline.models["shape_slat_flow_model_512"], coords_c32
    )
    save_sparse(out / "latents" / "shape_c32_denormalized.pt", shape_c32, False)
    del cond_c32, coords_c32
    tiled.empty_cuda()

    print("[baseline 3/3] learned support C64 + Shape1024 flow", flush=True)
    coords_c64 = tiled.upsample_and_quantize(pipeline, shape_c32, 512, 64, True)
    tiled.atomic_save(out / "support" / "coords_c64.pt", {"coords": coords_c64.cpu()})
    cond_c64 = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [canonical["image_1024"]],
        coords_c64,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=1.0,
        grid_resolution_override=64,
    )
    shape_c64 = pipeline.sample_shape_slat(
        cond_c64, pipeline.models["shape_slat_flow_model_1024"], coords_c64
    )
    save_sparse(out / "latents" / "shape_c64_denormalized.pt", shape_c64, False)
    del cond_c64, shape_c32
    tiled.empty_cuda()

    print("[decode] shape-only native decode at 1024", flush=True)
    meshes, _ = pipeline.decode_shape_slat(shape_c64, 1024)
    if len(meshes) != 1:
        raise RuntimeError(f"shape decoder returned B={len(meshes)}")
    mesh = meshes[0]
    mesh_path = out / "final" / "geometry_mesh.pt"
    tiled.atomic_save(mesh_path, {"format": FORMAT, "mesh": mesh.cpu()})
    glb_path = out / "final" / "geometry_mesh.glb"
    import trimesh

    tri = trimesh.Trimesh(
        vertices=mesh.vertices.detach().cpu().numpy(),
        faces=mesh.faces.detach().cpu().numpy(),
        process=False,
    )
    tri.export(glb_path)
    render = render_geometry(
        mesh,
        camera,
        inputs / "canonical_1024.png",
        inputs / "foreground_mask_1024.png",
        out,
        int(args.render_resolution),
        tuple(int(x) % 360 for x in args.angles.split(",")),
        int(args.render_chunk_size),
    )
    summary = {
        "format": FORMAT,
        "status": "complete",
        "texture_executed": False,
        "c64_tokens": int(coords_c64.shape[0]),
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "seconds": time.perf_counter() - started,
        "camera": camera,
        "mesh_pt": str(mesh_path.resolve()),
        "mesh_glb": str(glb_path.resolve()),
        **render,
    }
    tiled.atomic_json(out / "summary.json", summary)
    print(json.dumps(tiled._jsonable(summary), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
