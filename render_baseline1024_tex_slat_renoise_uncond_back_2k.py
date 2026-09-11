#!/usr/bin/env python3
"""Re-render the saved texture re-noise sweep at true 2048px per panel."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import numpy as np
import torch
from PIL import Image

from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from pixal3d.renderers import PbrMeshRenderer
from pixal3d_baseline1024_tex_slat_renoise_uncond_back_sweep import (
    array_to_image,
    atomic_json,
    back_camera,
    empty_cuda,
    error_heatmap,
    make_grid,
    make_mesh,
    make_metric_plot,
    metrics,
    tensor_to_array,
)
from render_pixal3d_raw_ovoxel import load_envmap


DEFAULT_ROOT = Path("outputs/baseline1024_tex_slat_renoise_uncond_back_cuda4")


def load_slat(path: Path, device: torch.device) -> SparseTensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return SparseTensor(
        feats=payload["feats"].to(device=device, dtype=torch.float32),
        coords=payload["coords"].to(device=device, dtype=torch.int32),
    )


def save_shaded_and_mask(
    result: dict[str, torch.Tensor], output: Path
) -> dict[str, np.ndarray]:
    output.mkdir(parents=True, exist_ok=True)
    shaded = tensor_to_array(result["shaded"])
    mask = tensor_to_array(result["mask"])
    array_to_image(shaded).save(output / "render.png")
    array_to_image(shaded).save(output / "shaded.png")
    array_to_image(mask).save(output / "mask.png")
    return {"shaded": shaded, "mask": mask}


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-cuda", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--error-scale", type=float, default=0.25)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_cuda):
        raise RuntimeError(f"set CUDA_VISIBLE_DEVICES={args.physical_cuda}")
    if args.resolution != 2048:
        raise ValueError("this high-resolution visualization is fixed to 2048px")
    root = args.root.resolve()
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or len(summary.get("results", [])) != 12:
        raise RuntimeError("the source sweep is not complete")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    camera = summary["camera"]
    print("[2k] loading pipeline and saved SLat files", flush=True)
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=True)
    shape_slat = load_slat(root / "baseline_shape_slat.pt", device)
    print("[2k] decoding fixed geometry once", flush=True)
    shape_meshes, subs = pipeline.decode_shape_slat(shape_slat, 1024)
    geometry = shape_meshes[0]

    extrinsics, intrinsics = back_camera(
        float(camera["camera_angle_x"]), float(camera["distance"])
    )
    renderer = PbrMeshRenderer(
        rendering_options={
            "resolution": 2048,
            "near": max(0.01, float(camera["distance"]) - 2.0),
            "far": float(camera["distance"]) + 10.0,
            "ssaa": 1,
            "peel_layers": 8,
            "face_chunk_size": 0,
        },
        device=str(device),
    )
    envmap = load_envmap("studio", device=device)

    def render_slat(path: Path, output: Path) -> dict[str, np.ndarray]:
        slat = load_slat(path, device)
        voxel = pipeline.decode_tex_slat(slat, subs)[0]
        mesh = make_mesh(geometry, voxel, 1024, pipeline.pbr_attr_layout)
        torch.cuda.manual_seed_all(100_222)
        result = renderer.render(
            mesh,
            extrinsics.to(device),
            intrinsics.to(device),
            envmap=envmap,
            use_envmap_bg=False,
        )
        saved = save_shaded_and_mask(result, output)
        del result, mesh, voxel, slat
        empty_cuda()
        return saved

    baseline = render_slat(
        root / "baseline_texture_slat.pt", root / "baseline_back_2k"
    )
    reference = baseline["shaded"]
    foreground = baseline["mask"]
    if foreground.ndim == 3:
        foreground = foreground[..., 0]
    foreground = foreground > 0.5
    array_to_image(foreground.astype(np.float32)).save(
        root / "baseline_back_2k" / "foreground_mask.png"
    )

    display_records: list[dict[str, Any]] = []
    for source_record in summary["results"]:
        step = int(source_record["start_step"])
        print(f"[2k {step:02d}/12] decode and render", flush=True)
        directory = root / f"start_step_{step:02d}" / "back_2k"
        rendered = render_slat(
            root / f"start_step_{step:02d}" / "final_texture_slat.pt",
            directory,
        )
        scores = metrics(reference, rendered["shaded"], foreground)
        error, heatmap = error_heatmap(
            reference, rendered["shaded"], foreground, args.error_scale
        )
        np.save(directory / "absolute_error_float32.npy", error)
        error_path = directory / "error_map.png"
        array_to_image(heatmap).save(error_path)
        display_records.append(
            {
                **source_record,
                **scores,
                "back_render": str((directory / "render.png").resolve()),
                "error_map": str(error_path.resolve()),
            }
        )

    error_grid = root / "back_error_maps_3x4.png"
    metric_plot = root / "foreground_metrics.png"
    make_grid(
        display_records,
        error_grid,
        "error_map",
        "2K back error maps",
        thumb=2048,
        label_h=448,
    )
    make_metric_plot(
        display_records,
        metric_plot,
        width=2560,
        height=1440,
    )
    summary["high_resolution_visualization"] = {
        "render_resolution": [2048, 2048],
        "error_map_panel_resolution": [2048, 2048],
        "error_grid_resolution": list(Image.open(error_grid).size),
        "metric_plot_resolution": [2560, 1440],
        "error_grid": str(error_grid.resolve()),
        "metric_plot": str(metric_plot.resolve()),
        "results": display_records,
    }
    atomic_json(summary_path, summary)
    print(f"[2k done] {error_grid}", flush=True)
    print(f"[2k done] {metric_plot}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
