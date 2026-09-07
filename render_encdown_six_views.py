#!/usr/bin/env python3
"""Render front/back/left/right/top/bottom views of the encdown result."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from pixal3d.renderers import PbrMeshRenderer
from pixal3d.utils import render_utils
from render_pixal3d_raw_ovoxel import load_envmap, load_mesh_checkpoint


ROOT = Path("outputs/global_c256_encdown_context1024_flow_singleview_cuda4")
BASELINE = Path("outputs/baseline1024_raw_ovoxel_cuda4_0_img")
VIEWS = (
    ("front", "y", 0),
    ("right", "y", 90),
    ("back", "y", 180),
    ("left", "y", 270),
    # With the repository's front camera (camera-space Y is flipped), -90°
    # around X exposes world +Y (top), while +90° exposes world -Y.
    ("top", "x", -90),
    ("bottom", "x", 90),
)


def rotation(axis: str, angle_deg: float, *, dtype: torch.dtype) -> torch.Tensor:
    angle = math.radians(float(angle_deg))
    c, s = math.cos(angle), math.sin(angle)
    if axis == "y":
        values = ((c, 0, s, 0), (0, 1, 0, 0), (-s, 0, c, 0), (0, 0, 0, 1))
    elif axis == "x":
        values = ((1, 0, 0, 0), (0, c, -s, 0), (0, s, c, 0), (0, 0, 0, 1))
    else:
        raise ValueError(axis)
    return torch.tensor(values, dtype=dtype)


def save_tensor_image(value: torch.Tensor, path: Path, channels: int = 3) -> None:
    x = value.detach().float().cpu()
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    if x.ndim == 2:
        x = x[..., None]
    array = np.nan_to_num(x.numpy(), nan=0.0, posinf=1.0, neginf=0.0)
    array = np.uint8(np.clip(array, 0, 1) * 255.0 + 0.5)
    image = Image.fromarray(array[..., 0], "L") if channels == 1 else Image.fromarray(array[..., :3], "RGB")
    image.save(path)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", type=Path, default=ROOT / "final/final_material_mesh.pt")
    p.add_argument("--camera", type=Path, default=BASELINE / "global_camera.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "six_views_1024")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--face-chunk-size", type=int, default=4_000_000)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    camera = json.loads(args.camera.read_text(encoding="utf-8"))
    front, intrinsics = render_utils.proj_camera_to_render_params(
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
    )
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
    mesh = load_mesh_checkpoint(args.mesh, device=device)
    records = []
    with torch.inference_mode():
        for name, axis, angle in VIEWS:
            rotate = rotation(axis, angle, dtype=front.dtype).to(device)
            inverse = rotate.clone()
            inverse[:3, :3] = rotate[:3, :3].T
            extrinsics = front.to(device) @ inverse
            print(f"[six-view] rendering {name} ({axis}={angle:+d} deg)", flush=True)
            result = renderer.render(
                mesh,
                extrinsics,
                intrinsics.to(device),
                envmap=envmap,
                use_envmap_bg=False,
            )
            rgb_path = args.output_dir / f"{name}_rgb.png"
            alpha_path = args.output_dir / f"{name}_alpha.png"
            save_tensor_image(result["shaded"], rgb_path)
            save_tensor_image(result["mask"], alpha_path, channels=1)
            records.append(
                {
                    "name": name,
                    "axis": axis,
                    "angle_deg": angle,
                    "rgb": str(rgb_path.resolve()),
                    "alpha": str(alpha_path.resolve()),
                    "extrinsics": extrinsics.detach().cpu().tolist(),
                }
            )
            del result
            torch.cuda.empty_cache()

    cell = int(args.resolution)
    label_h, margin = 54, 16
    sheet = Image.new("RGB", (3 * cell + 4 * margin, 2 * (cell + label_h) + 3 * margin), (18, 21, 27))
    draw = ImageDraw.Draw(sheet)
    label_font = font(30)
    for index, record in enumerate(records):
        row, col = divmod(index, 3)
        x = margin + col * (cell + margin)
        y = margin + row * (cell + label_h + margin)
        draw.text((x + 8, y + 8), record["name"].upper(), fill=(245, 247, 250), font=label_font)
        with Image.open(record["rgb"]) as image:
            sheet.paste(image.convert("RGB"), (x, y + label_h))
    sheet_path = args.output_dir / "six_views_contact_sheet.png"
    sheet.save(sheet_path)
    manifest = {
        "status": "complete",
        "mesh": str(args.mesh.resolve()),
        "camera": str(args.camera.resolve()),
        "resolution": [int(args.resolution), int(args.resolution)],
        "camera_convention": "front extrinsics @ inverse(object axis rotation)",
        "views": records,
        "contact_sheet": str(sheet_path.resolve()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
