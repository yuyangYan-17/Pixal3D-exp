#!/usr/bin/env python3
"""Render the cached baseline PBR mesh at 2048 for like-for-like metrics."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from PIL import Image

from pixal3d.representations import MeshWithVertexPbr
from pixal3d.renderers import PbrMeshRenderer
from pixal3d.utils import render_utils
from pixal3d_baseline1024_pbr_mesh_compare import _make_camera_views, _image_from_array, _tensor_to_hwc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/baseline1024_pbr_mesh_compare"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline1024_pbr_mesh_compare/render_2048"))
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--ssaa", type=int, default=2)
    parser.add_argument("--peel-layers", type=int, default=8)
    parser.add_argument("--face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--envmap", default="studio")
    args = parser.parse_args()

    torch.cuda.set_device(int(args.cuda_device))
    baseline_dir = args.baseline_dir.resolve()
    output_dir = args.output_dir.resolve()
    payload = torch.load(baseline_dir / "per_vertex_pbr_mesh.pt", map_location="cpu", weights_only=False)
    mesh = MeshWithVertexPbr(
        payload["vertices"], payload["faces"], payload["vertex_attrs"], layout=payload["layout"]
    ).to(f"cuda:{args.cuda_device}")
    import json
    camera = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
    angles = (0, 60, 120, 180, 240, 300)
    extrinsics, intrinsics, _ = _make_camera_views(
        float(camera["camera_angle_x"]), float(camera["distance"]), angles
    )
    from render_pixal3d_raw_ovoxel import load_envmap
    envmap = load_envmap(str(args.envmap), device=f"cuda:{args.cuda_device}")
    renderer = PbrMeshRenderer(
        rendering_options={
            "resolution": int(args.resolution),
            "near": max(0.01, float(camera["distance"]) - 2.0),
            "far": float(camera["distance"]) + 10.0,
            "ssaa": int(args.ssaa),
            "peel_layers": int(args.peel_layers),
            "face_chunk_size": int(args.face_chunk_size),
        },
        device=f"cuda:{args.cuda_device}",
    )
    paths = []
    for angle in angles:
        print(f"[baseline render] yaw={angle}", flush=True)
        result = renderer.render(mesh, extrinsics[angle], intrinsics, envmap=envmap, use_envmap_bg=False)
        path = output_dir / "renders" / f"yaw{angle:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        _image_from_array(_tensor_to_hwc(result["shaded"])).save(path)
        paths.append(path)
        del result
        gc.collect()
        torch.cuda.empty_cache()
    sheet = Image.new("RGB", (3 * int(args.resolution), 2 * int(args.resolution)), "black")
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            sheet.paste(image.convert("RGB"), ((index % 3) * image.width, (index // 3) * image.height))
    sheet.save(output_dir / "renders" / "six_view_sheet.png")
    print(f"[done] {output_dir / 'renders'}", flush=True)


if __name__ == "__main__":
    main()
