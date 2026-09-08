#!/usr/bin/env python3
"""Render the back view for every C256 endpoint-prefix sweep result."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import torch
from PIL import Image, ImageDraw

import pixal3d_global_c256_cube_owner_flow_singleview as base
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from pixal3d.renderers import PbrMeshRenderer
from pixal3d.utils import render_utils
from render_encdown_six_views import font, rotation, save_tensor_image
from render_pixal3d_raw_ovoxel import load_envmap


ROOT = Path("outputs/c256_uniform_endpoint_prefix_uncond_sweep_strict_cuda4")
BASELINE = Path("outputs/baseline1024_raw_ovoxel_cuda4_0_img")
STEPS = 12


def empty_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--camera", type=Path, default=BASELINE / "global_camera.json")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--physical-cuda", type=int, default=4)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--face-chunk-size", type=int, default=4_000_000)
    p.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return p


@torch.no_grad()
def main() -> int:
    args = parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and [x.strip() for x in visible.split(",")] != [str(args.physical_cuda)]:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r}; expected physical CUDA {args.physical_cuda}"
        )

    root = args.root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "back_views_1024"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    results = sorted(summary["results"], key=lambda row: int(row["guided_prefix_steps"]))
    if [int(row["guided_prefix_steps"]) for row in results] != list(range(1, STEPS + 1)):
        raise RuntimeError("expected exactly guided prefixes 1..12")

    shape_saved = torch.load(
        root / "shape/final_state_normalized.pt", map_location="cpu", weights_only=False
    )
    coords = shape_saved["coords"].to(torch.int32).contiguous()
    shape_norm = shape_saved["features"].float().contiguous()
    camera = json.loads(args.camera.read_text(encoding="utf-8"))

    pipeline = init_pipeline(
        str(args.model_path), device=str(device), low_vram=bool(args.low_vram)
    )
    shape_raw = base.denormalize(shape_norm, pipeline.shape_slat_normalization)
    tex_mean, tex_std = base._norm_tensors(pipeline.tex_slat_normalization, 32)

    front, intrinsics = render_utils.proj_camera_to_render_params(
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
    )
    rotate = rotation("y", 180, dtype=front.dtype).to(device)
    inverse = rotate.clone()
    inverse[:3, :3] = rotate[:3, :3].T
    extrinsics = front.to(device) @ inverse
    intrinsics = intrinsics.to(device)
    renderer = PbrMeshRenderer(
        rendering_options={
            "resolution": int(args.resolution),
            "near": max(0.01, float(camera["distance"]) - 2.0),
            "far": float(camera["distance"]) + 10.0,
            "ssaa": 1,
            "peel_layers": 8,
            "face_chunk_size": int(args.face_chunk_size),
        },
        device=str(device),
    )
    envmap = load_envmap("studio", device=device)

    records: list[dict[str, Any]] = []
    for row in results:
        prefix = int(row["guided_prefix_steps"])
        render_dir = output_dir / f"guided_prefix_{prefix:02d}"
        render_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = render_dir / "back_rgb.png"
        alpha_path = render_dir / "back_alpha.png"
        if not (args.resume and rgb_path.is_file() and alpha_path.is_file()):
            saved = torch.load(row["path"], map_location="cpu", weights_only=False)
            if not torch.equal(saved["coords"].int(), coords):
                raise RuntimeError(f"prefix {prefix}: texture/shape support mismatch")
            tex_raw = saved["features"].float() * tex_std + tex_mean
            shape = SparseTensor(shape_raw.to(device), coords.to(device))
            texture = SparseTensor(tex_raw.to(device), coords.to(device))
            print(f"[decode4096] guided prefix={prefix:02d}", flush=True)
            mesh = pipeline.decode_latent(shape, texture, 4096)[0]
            print(f"[back view] render guided prefix={prefix:02d}", flush=True)
            rendered = renderer.render(
                mesh,
                extrinsics,
                intrinsics,
                envmap=envmap,
                use_envmap_bg=False,
            )
            save_tensor_image(rendered["shaded"], rgb_path)
            save_tensor_image(rendered["mask"], alpha_path, channels=1)
            del saved, tex_raw, shape, texture, mesh, rendered
            empty_cuda()
        else:
            print(f"[resume] guided prefix={prefix:02d}", flush=True)
        records.append(
            {
                "guided_prefix_steps": prefix,
                "unconditional_suffix_steps": STEPS - prefix,
                "switch_t": 1.0 - prefix / STEPS,
                "rgb": str(rgb_path.resolve()),
                "alpha": str(alpha_path.resolve()),
            }
        )

    thumb, label_h, margin = 512, 64, 12
    columns, rows = 3, 4
    sheet = Image.new(
        "RGB",
        (columns * thumb + 4 * margin, rows * (thumb + label_h) + 5 * margin),
        (18, 21, 27),
    )
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(records):
        row, col = divmod(index, columns)
        x = margin + col * (thumb + margin)
        y = margin + row * (thumb + label_h + margin)
        prefix = int(record["guided_prefix_steps"])
        draw.text(
            (x + 6, y + 4),
            f"GUIDED PREFIX {prefix}/12",
            fill="white",
            font=font(21),
        )
        draw.text(
            (x + 6, y + 33),
            f"then UNCOND from t={float(record['switch_t']):.4f}",
            fill=(255, 208, 112),
            font=font(17),
        )
        with Image.open(record["rgb"]) as rendered:
            tile = rendered.convert("RGB").resize(
                (thumb, thumb), Image.Resampling.LANCZOS
            )
        sheet.paste(tile, (x, y + label_h))

    grid_path = output_dir / "guided_prefix_unconditional_suffix_back_grid_3x4.png"
    sheet.save(grid_path)
    manifest = {
        "status": "complete",
        "source_summary": str((root / "summary.json").resolve()),
        "shape": str((root / "shape/final_state_normalized.pt").resolve()),
        "camera": str(args.camera.resolve()),
        "view": {"name": "back", "axis": "y", "angle_deg": 180},
        "camera_convention": "front extrinsics @ inverse(object y=180 degree rotation)",
        "resolution": [int(args.resolution), int(args.resolution)],
        "metrics": None,
        "records": records,
        "grid": str(grid_path.resolve()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[done] {grid_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
