#!/usr/bin/env python3
"""Project the saved global C128 support and crop a 1024px turtle-head ROI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from pixal3d_tile_c1024_local_slat_and_local_decode_return_global import (
    _project_global_q_to_image,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--coords",
        type=Path,
        default=Path("outputs/baseline1024_c128_8xc64_geometry_cuda4/support/coords_c128.pt"),
    )
    p.add_argument(
        "--image-4096",
        type=Path,
        default=Path("outputs/cascade512_1024_tiled2048_square_crop_cuda4/inputs/image_4096.png"),
    )
    p.add_argument(
        "--camera",
        type=Path,
        default=Path("outputs/baseline1024_c128_8xc64_geometry_cuda4/camera.json"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/c128_head_crop_4096"),
    )
    p.add_argument("--crop", type=int, nargs=4, default=(3008, 1632, 4032, 2656))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image_4096).convert("RGB")
    if image.size != (4096, 4096):
        raise RuntimeError(f"expected a 4096 image, got {image.size}")
    camera = json.loads(args.camera.read_text(encoding="utf-8"))
    coords = torch.load(args.coords, map_location="cpu", weights_only=False)["coords"].int()
    q = 2.0 * (coords[:, 1:].double() + 0.5) / 128.0 - 1.0
    uv, depth, finite = _project_global_q_to_image(
        q,
        global_camera=camera,
        image_width=1024,
        image_height=1024,
    )
    pixels = (uv.double() + 0.5) * 4.0
    visible = finite & (depth > 0)
    visible &= (pixels[:, 0] >= 0) & (pixels[:, 0] < 4096)
    visible &= (pixels[:, 1] >= 0) & (pixels[:, 1] < 4096)
    pixels = pixels[visible]

    x0, y0, x1, y1 = (int(v) for v in args.crop)
    if x1 - x0 != 1024 or y1 - y0 != 1024:
        raise ValueError("--crop must be exactly 1024x1024")
    if min(x0, y0) < 0 or max(x1, y1) > 4096:
        raise ValueError("--crop must remain inside the 4096 image")
    inside = (
        (pixels[:, 0] >= x0) & (pixels[:, 0] < x1)
        & (pixels[:, 1] >= y0) & (pixels[:, 1] < y1)
    )

    crop = image.crop((x0, y0, x1, y1))
    crop.save(args.output_dir / "turtle_head_1024.png")

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    # Subsample only for legibility; all projected points are still counted.
    shown = pixels[:: max(1, len(pixels) // 45_000)].round().int().tolist()
    for x, y in shown:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 255, 255, 105))
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 40, 40, 255), width=16)
    draw.rectangle((x0 + 16, y0 + 16, x0 + 550, y0 + 82), fill=(0, 0, 0, 190))
    draw.text((x0 + 30, y0 + 30), "HEAD ROI 1024 x 1024", fill=(255, 255, 255, 255))
    overlay.resize((1024, 1024), Image.Resampling.LANCZOS).save(
        args.output_dir / "c128_projection_head_box_preview.png"
    )

    crop_overlay = crop.copy()
    crop_draw = ImageDraw.Draw(crop_overlay, "RGBA")
    local = pixels[inside] - torch.tensor([x0, y0], dtype=pixels.dtype)
    for x, y in local[:: max(1, len(local) // 20_000)].round().int().tolist():
        crop_draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 255, 255, 120))
    crop_overlay.save(args.output_dir / "turtle_head_1024_with_c128_projection.png")
    manifest = {
        "source_image": str(args.image_4096.resolve()),
        "coords": str(args.coords.resolve()),
        "projection": "C128 voxel centers q=2*(xyz+0.5)/128-1 through the saved global camera",
        "crop_xyxy_4096": [x0, y0, x1, y1],
        "crop_size": [1024, 1024],
        "projected_points_in_image": int(pixels.shape[0]),
        "projected_points_in_crop": int(inside.sum()),
        "crop": str((args.output_dir / "turtle_head_1024.png").resolve()),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
