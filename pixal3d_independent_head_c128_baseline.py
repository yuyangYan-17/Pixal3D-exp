#!/usr/bin/env python3
"""Run a native C128/2048 shape baseline on the saved independent head crop."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image

import pixal3d.models as models
from pixal3d.modules.sparse import SparseTensor
from pixal3d.pipelines import Pixal3DImageTo3DPipeline
from pixal3d.pipelines import samplers
from pixal3d_c128_block_local_cascade_shape_renoise_2048 import empty_cuda
from inference import IMAGE_COND_CONFIGS, MODEL_PATH, build_image_cond_model


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = Path(MODEL_PATH)
DEFAULT_LOCAL_STATE = ROOT / "outputs/c128_head_crop_baseline1024_geometry_cuda4/latents/shape_c64_denormalized.pt"
DEFAULT_LOCAL_INPUT = ROOT / "outputs/c128_head_crop_baseline1024_geometry_cuda4/inputs/canonical_1024.png"
DEFAULT_LOCAL_MASK = ROOT / "outputs/c128_head_crop_baseline1024_geometry_cuda4/inputs/foreground_mask_1024.png"
DEFAULT_LOCAL_CAMERA = ROOT / "outputs/c128_head_crop_baseline1024_geometry_cuda4/camera.json"
DEFAULT_OUT = ROOT / "outputs/c128_head_crop_native_c2048_baseline_cuda4"


def atomic_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def init_pipeline(model_path: Path, device: torch.device) -> Pixal3DImageTo3DPipeline:
    config = json.loads((model_path / "pipeline.json").read_text(encoding="utf-8"))["args"]
    loaded = {}
    for name in ("shape_slat_flow_model_1024", "shape_slat_decoder"):
        print(f"[model] loading {name}", flush=True)
        loaded[name] = models.from_pretrained(str(model_path / config["models"][name])).eval()
    sampler_config = config["shape_slat_sampler"]
    pipeline = Pixal3DImageTo3DPipeline(
        models=loaded,
        shape_slat_sampler=getattr(samplers, sampler_config["name"])(**sampler_config["args"]),
        shape_slat_sampler_params=dict(sampler_config["params"]),
        shape_slat_normalization=config["shape_slat_normalization"],
        low_vram=True,
    )
    pipeline._device = device
    return pipeline


def load_state(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    coords = payload["coords"].int()
    raw = payload["features"].float()
    if coords.shape[1] != 4 or raw.shape[0] != coords.shape[0]:
        raise ValueError("invalid independent C64 state")
    if bool(((coords[:, 1:] < 0) | (coords[:, 1:] >= 64)).any()):
        raise ValueError("independent state is not C64")
    return coords, raw


@torch.no_grad()
def promote_support(
    pipeline: Pixal3DImageTo3DPipeline,
    coords: torch.Tensor,
    raw: torch.Tensor,
    device: torch.device,
    method: str,
) -> torch.Tensor:
    """Use the trained decoder upsampler to derive a C128 support."""
    if method == "direct":
        # One C64 cell maps to the C128 cell containing the same physical
        # center. This is intentionally sparse; it avoids the topology
        # explosion caused by expanding a crop-only support through every
        # decoder subdivision before the C128 Flow stage.
        xyz128 = ((coords[:, 1:].float() + 0.5) * 2.0).floor().int()
        promoted = torch.cat((coords[:, :1], xyz128), dim=1).unique(dim=0).int().cpu()
        print(f"[support] direct C64 cell-center promotion -> C128 ({len(promoted):,} tokens)", flush=True)
        return promoted
    if method != "upsample":
        raise ValueError(f"unsupported support method: {method}")
    slat = SparseTensor(raw.to(device), coords.to(device))
    decoder = pipeline.models["shape_slat_decoder"]
    decoder.to(device)
    decoder.low_vram = True
    print("[support] independent C64 -> decoder upsample -> C128", flush=True)
    hr = decoder.upsample(slat, upsample_times=4)
    decoder.cpu()
    decoder.low_vram = False
    # For a C64 input the decoder's high resolution coordinates are C1024.
    xyz = (((hr[:, 1:].float() + 0.5) / 1024.0) * 128.0).floor().int()
    promoted = torch.cat((hr[:, :1].int(), xyz), dim=1).unique(dim=0).int().cpu()
    if not promoted.numel() or bool(((promoted[:, 1:] < 0) | (promoted[:, 1:] >= 128)).any()):
        raise RuntimeError("promoted C128 support is invalid")
    del slat, hr
    empty_cuda()
    return promoted


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", type=Path, default=DEFAULT_LOCAL_STATE)
    p.add_argument("--image", type=Path, default=DEFAULT_LOCAL_INPUT)
    p.add_argument("--mask", type=Path, default=DEFAULT_LOCAL_MASK)
    p.add_argument("--camera", type=Path, default=DEFAULT_LOCAL_CAMERA)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cuda-device", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--support-method", choices=("upsample", "direct"), default="upsample")
    p.add_argument("--angles", default="0,60,180,240")
    args = p.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.cuda_device):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}, expected physical CUDA {args.cuda_device}")
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    camera_payload = json.loads(args.camera.resolve().read_text(encoding="utf-8"))
    camera = camera_payload.get("camera", camera_payload)
    coords64, raw64 = load_state(args.state.resolve())
    pipeline = init_pipeline(args.model_path.resolve(), device)
    coords128 = promote_support(pipeline, coords64, raw64, device, args.support_method)
    atomic_save(out / "support" / "coords_c128.pt", {"coords": coords128})
    atomic_json(
        out / "manifest.json",
        {
            "format": "pixal3d_independent_head_c128_baseline_v1",
            "state_c64": str(args.state.resolve()),
            "input": str(args.image.resolve()),
            "camera": camera,
            "c64_tokens": int(len(coords64)),
            "c128_tokens": int(len(coords128)),
            "path": "saved independent C64 endpoint -> decoder upsample/quant C128 -> native Shape1024 flow -> global C2048 decode",
            "support_method": args.support_method,
            "seed": int(args.seed),
            "steps": int(args.steps),
            "cuda_device": int(args.cuda_device),
        },
    )
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    pipeline.shape_slat_sampler_params["steps"] = int(args.steps)
    print("[condition] local canonical image -> C128 projected condition", flush=True)
    extractor = build_image_cond_model(IMAGE_COND_CONFIGS["shape_1024"])
    with Image.open(args.image.resolve()) as image:
        image = image.convert("RGB")
        cond = pipeline.get_proj_cond_shape(
            extractor,
            [image],
            coords128.to(device),
            camera_angle_x=float(camera["camera_angle_x"]),
            distance=float(camera["distance"]),
            mesh_scale=float(camera.get("mesh_scale", 1.0)),
            grid_resolution_override=128,
        )
    extractor.cpu()
    del extractor
    empty_cuda()
    print("[flow] native independent head C128 Shape1024", flush=True)
    flow = pipeline.models["shape_slat_flow_model_1024"]
    state = pipeline.sample_shape_slat(cond, flow, coords128.to(device))
    state_features = state.feats.detach().cpu()
    atomic_save(
        out / "latents" / "shape_c128_denormalized.pt",
        {"coords": coords128, "features": state_features, "format": "pixal3d_independent_head_c128_baseline_v1"},
    )
    del cond, state, state_features, flow
    empty_cuda()
    print("[decode] C128 -> geometry at 2048", flush=True)
    payload = torch.load(out / "latents/shape_c128_denormalized.pt", map_location="cpu", weights_only=False)
    slat = SparseTensor(payload["features"].to(device), payload["coords"].to(device))
    meshes, _ = pipeline.decode_shape_slat(slat, 2048)
    if len(meshes) != 1:
        raise RuntimeError(f"expected one mesh, got {len(meshes)}")
    mesh = meshes[0]
    atomic_save(out / "final" / "geometry_mesh.pt", {"format": "pixal3d_independent_head_c128_baseline_v1", "mesh": mesh.cpu()})
    del meshes, mesh, slat, payload, pipeline
    gc.collect()
    empty_cuda()
    # Render after decoder teardown through the lightweight repository helper.
    from pixal3d_baseline1024_c128_8xc64_geometry import render_geometry

    mesh_payload = torch.load(out / "final/geometry_mesh.pt", map_location="cpu", weights_only=False)
    live = mesh_payload["mesh"].to(device)
    angles = tuple(int(x.strip()) % 360 for x in args.angles.split(",") if x.strip())
    render = render_geometry(live, camera, args.image.resolve(), args.mask.resolve(), out, 1024, angles, 200_000)
    atomic_json(
        out / "summary.json",
        {
            "format": "pixal3d_independent_head_c128_baseline_v1",
            "status": "complete",
            "vertices": int(len(live.vertices)),
            "faces": int(len(live.faces)),
            "render": render,
        },
    )
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
