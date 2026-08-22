#!/usr/bin/env python3
"""Annotate generated and baseline six-view sheets with three input-view metrics."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping

from PIL import Image, ImageDraw, ImageFont


ANGLES = (0, 60, 120, 180, 240, 300)
METRIC_ANGLES = (0, 120, 240)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    if not bold:
        names = names[::-1]
    for name in names:
        path = Path(name)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _metrics(metrics_json: Path) -> Dict[int, Dict[str, Dict[str, float]]]:
    payload: Mapping[str, Any] = json.loads(metrics_json.read_text(encoding="utf-8"))
    result: Dict[int, Dict[str, Dict[str, float]]] = {}
    for row in payload["rows"]:
        angle = int(row["angle"])
        result[angle] = {
            "baseline": row["baseline"]["input_foreground_only"],
            "generated": row["generated"]["input_foreground_only"],
        }
    return result


def _annotate(source: Path, output: Path, method: str, values: Dict[int, Dict[str, Dict[str, float]]], panel: int) -> None:
    with Image.open(source) as image:
        canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    title_font = _font(max(32, panel // 32), bold=True)
    body_font = _font(max(28, panel // 42), bold=True)
    for index, angle in enumerate(ANGLES):
        if angle not in METRIC_ANGLES:
            continue
        x = (index % 3) * panel + max(24, panel // 80)
        y = (index // 3) * panel + max(24, panel // 80)
        metric = values[angle][method]
        lines = [
            f"{angle:03d}°  {method.title()}",
            f"FG PSNR  {float(metric['psnr_db']):.2f} dB",
            f"FG SSIM  {float(metric['ssim']):.3f}",
        ]
        line_gap = max(8, panel // 160)
        title_box = draw.textbbox((0, 0), lines[0], font=title_font)
        body_boxes = [draw.textbbox((0, 0), line, font=body_font) for line in lines[1:]]
        width = max(title_box[2] - title_box[0], *(box[2] - box[0] for box in body_boxes))
        height = (title_box[3] - title_box[1]) + sum(box[3] - box[1] for box in body_boxes) + 2 * line_gap
        pad = max(16, panel // 100)
        draw.rounded_rectangle((x, y, x + width + 2 * pad, y + height + 2 * pad), radius=max(8, panel // 100), fill=(0, 0, 0, 190), outline=(255, 255, 255, 180), width=max(2, panel // 512))
        cursor_y = y + pad
        draw.text((x + pad, cursor_y), lines[0], font=title_font, fill=(255, 235, 80, 255), stroke_width=1, stroke_fill=(0, 0, 0, 255))
        cursor_y += title_box[3] - title_box[1] + line_gap
        for line, box in zip(lines[1:], body_boxes):
            draw.text((x + pad, cursor_y), line, font=body_font, fill=(255, 255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0, 255))
            cursor_y += box[3] - box[1] + line_gap
    annotated = Image.alpha_composite(canvas, overlay).convert("RGB")
    annotated.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--baseline-render-root", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    args = parser.parse_args()
    generated_sheet = args.generated_root.resolve() / "renders" / "six_view_sheet.png"
    baseline_sheet = args.baseline_render_root.resolve() / "renders" / "six_view_sheet.png"
    values = _metrics(args.metrics_json.resolve())
    for sheet, method in ((generated_sheet, "generated"), (baseline_sheet, "baseline")):
        raw = sheet.with_name("six_view_sheet_raw.png")
        if not raw.is_file():
            shutil.copy2(sheet, raw)
        with Image.open(sheet) as image:
            panel = image.width // 3
            if image.width != 3 * panel or image.height != 2 * panel:
                raise ValueError(f"expected 3x2 square sheet, got {image.size}: {sheet}")
        _annotate(raw, sheet, method, values, panel)
        print(sheet)


if __name__ == "__main__":
    main()
