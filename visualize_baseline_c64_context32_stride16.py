#!/usr/bin/env python3
"""Visualize the native baseline C64 support as overlapping C32/stride-16 blocks.

The input coordinates are the support produced by the native baseline route

    SS C32 -> Shape512 C32 -> decoder C512 candidates -> quantized C64 support.

This script performs no flow inference.  It only partitions that fixed C64
support, projects each block's active points with the baseline camera, saves
the exact projected-bbox crop from canonical_4096, and renders a numbered
exploded point-cloud view.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
GRID = 64
CONTEXT = 32
STRIDE = 16
CANONICAL_SIZE = 4096
PATCH_ALIGNMENT = 16
FORMAT = "baseline_c64_context32_stride16_visualization_v1"

DEFAULT_SUPPORT = ROOT / (
    "outputs/baseline1024_c128_8xc64_geometry_cuda4/"
    "baseline/shape_c64_denormalized.pt"
)
DEFAULT_IMAGE = ROOT / (
    "outputs/c64_to_c128_two_stage_block_flow_geometry_cuda4/"
    "inputs/canonical_4096.png"
)
DEFAULT_CAMERA = ROOT / "outputs/baseline1024_c128_8xc64_geometry_cuda4/camera.json"
DEFAULT_OUTPUT = ROOT / "outputs/baseline_c64_context32_stride16_visualization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--explode-gap", type=float, default=24.0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_coords(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "coords" not in payload:
        raise ValueError(f"support file has no coords: {path}")
    coords = torch.as_tensor(payload["coords"]).to(torch.int32)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"coords must be [N,4], got {tuple(coords.shape)}")
    if torch.any(coords[:, 0] != 0):
        raise ValueError("only a single batch is supported")
    xyz = coords[:, 1:]
    if torch.any(xyz < 0) or torch.any(xyz >= GRID):
        raise ValueError("support coordinates are outside C64")
    if len(coords.unique(dim=0)) != len(coords):
        raise ValueError("support contains duplicate coordinates")
    return coords


def make_blocks(coords: torch.Tensor) -> list[dict[str, Any]]:
    xyz = coords[:, 1:]
    starts = range(0, GRID - CONTEXT + 1, STRIDE)
    records: list[dict[str, Any]] = []
    cube_id = 0
    coverage = torch.zeros(len(coords), dtype=torch.int16)
    for sx in starts:
        for sy in starts:
            for sz in starts:
                start = torch.tensor([sx, sy, sz], dtype=torch.int32)
                rows = torch.where(
                    ((xyz >= start) & (xyz < start + CONTEXT)).all(dim=1)
                )[0]
                coverage.index_add_(
                    0, rows, torch.ones(len(rows), dtype=coverage.dtype)
                )
                records.append(
                    {
                        "cube_id": cube_id,
                        "start": [sx, sy, sz],
                        "rows": rows,
                        "tokens": int(len(rows)),
                    }
                )
                cube_id += 1
    if len(records) != 27:
        raise RuntimeError(f"expected 27 blocks, got {len(records)}")
    if torch.any(coverage < 1) or torch.any(coverage > 8):
        raise RuntimeError("overlapping block coverage must be in [1,8]")
    return records


def project_c64(coords_xyz: torch.Tensor, camera: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Match ProjGrid's C64 coordinate, Blender-axis and pixel-center path."""
    u = 2.0 * coords_xyz.to(torch.float64) / float(GRID - 1) - 1.0
    # ProjGrid maps latent (x,y,z) to camera q=(x,-z,y).
    q = torch.stack((u[:, 0], -u[:, 2], u[:, 1]), dim=1)
    mesh_scale = float(camera.get("mesh_scale", 1.0))
    points = q / (2.0 * mesh_scale)
    points[:, 2] -= float(camera["distance"])
    depth = -points[:, 2]
    focal = (
        16.0 / math.tan(float(camera["camera_angle_x"]) / 2.0)
        * CANONICAL_SIZE / 32.0
    )
    uv = torch.stack(
        (
            focal * points[:, 0] / depth + CANONICAL_SIZE / 2.0,
            -focal * points[:, 1] / depth + CANONICAL_SIZE / 2.0,
        ),
        dim=1,
    )
    finite = torch.isfinite(uv).all(1) & torch.isfinite(depth) & (depth > 0)
    return uv, finite


def aligned_bbox(points: torch.Tensor) -> tuple[list[int], list[float]]:
    if not len(points):
        raise RuntimeError("block has no finite projected points")
    raw_lo = points.amin(0)
    raw_hi = points.amax(0)
    lo = raw_lo.clamp(0.0, float(CANONICAL_SIZE))
    hi = raw_hi.clamp(0.0, float(CANONICAL_SIZE))
    if torch.any(hi <= lo):
        raise RuntimeError("projected bbox does not intersect the canonical image")
    x0 = max(0, int(math.floor(float(lo[0]))) // PATCH_ALIGNMENT * PATCH_ALIGNMENT)
    y0 = max(0, int(math.floor(float(lo[1]))) // PATCH_ALIGNMENT * PATCH_ALIGNMENT)
    x1 = min(
        CANONICAL_SIZE,
        int(math.ceil(float(hi[0]) / PATCH_ALIGNMENT)) * PATCH_ALIGNMENT,
    )
    y1 = min(
        CANONICAL_SIZE,
        int(math.ceil(float(hi[1]) / PATCH_ALIGNMENT)) * PATCH_ALIGNMENT,
    )
    x1 = max(x1, min(CANONICAL_SIZE, x0 + PATCH_ALIGNMENT))
    y1 = max(y1, min(CANONICAL_SIZE, y0 + PATCH_ALIGNMENT))
    return [x0, y0, x1, y1], [
        float(raw_lo[0]), float(raw_lo[1]), float(raw_hi[0]), float(raw_hi[1])
    ]


def color_table() -> list[tuple[int, int, int]]:
    cmap = plt.colormaps["hsv"].resampled(28)
    return [tuple(int(round(v * 255)) for v in cmap(i)[:3]) for i in range(27)]


def draw_box(ax: Any, lo: np.ndarray, hi: np.ndarray, color: tuple[float, ...]) -> None:
    corners = np.asarray(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )
    for i, a in enumerate(corners):
        for j in range(i + 1, len(corners)):
            b = corners[j]
            if np.count_nonzero(a != b) == 1:
                ax.plot(*zip(a, b), color=color, linewidth=0.55, alpha=0.62)


def render_exploded(
    coords: torch.Tensor,
    records: list[dict[str, Any]],
    colors: list[tuple[int, int, int]],
    output: Path,
    gap: float,
    dpi: int,
) -> None:
    xyz = coords[:, 1:].numpy()
    fig = plt.figure(figsize=(13.5, 12.0), facecolor="#09101c")
    ax = fig.add_subplot(111, projection="3d", facecolor="#09101c")
    for rec, rgb in zip(records, colors):
        start = np.asarray(rec["start"], dtype=np.float64)
        lattice = start / STRIDE
        # Lay each local C32 cube onto a disjoint 3x3x3 display lattice.
        # This is display-only; membership and projection retain global C64
        # coordinates.  Adjacent display cubes have exactly ``gap`` cells
        # between their wireframe boundaries.
        display_start = lattice * (CONTEXT + gap)
        offset = display_start - start
        local = xyz[rec["rows"].numpy()] - start
        shown = local + start + offset
        color = np.asarray(rgb, dtype=np.float64) / 255.0
        if len(shown):
            ax.scatter(
                shown[:, 0], shown[:, 1], shown[:, 2],
                s=2.0, c=[color], alpha=0.88, depthshade=False, linewidths=0,
            )
        lo = start + offset
        hi = lo + CONTEXT
        draw_box(ax, lo, hi, (*color, 1.0))
        center = (lo + hi) / 2.0
        label_position = center.copy()
        label_position[2] = hi[2] + 2.0
        ax.text(
            label_position[0], label_position[1], label_position[2], f"{rec['cube_id']}",
            color="white", fontsize=8.2, ha="center", va="bottom",
            bbox={"boxstyle": "round,pad=0.17", "fc": "#111827", "ec": color, "alpha": 0.92},
        )
    ax.set_title(
        "Baseline C64 support · C32 context / stride 16 · 27 numbered blocks",
        color="white", pad=18, fontsize=14,
    )
    ax.set_xlabel("latent X", color="#cbd5e1")
    ax.set_ylabel("latent Y", color="#cbd5e1")
    ax.set_zlabel("latent Z", color="#cbd5e1")
    ax.tick_params(colors="#94a3b8", labelsize=7)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.035, 0.063, 0.11, 1.0))
        axis.pane.set_edgecolor((0.25, 0.32, 0.43, 0.5))
    ax.view_init(elev=24, azim=38)
    ax.set_box_aspect((1, 1, 1))
    fig.text(
        0.5, 0.025,
        "Point duplication is intentional: overlap tokens appear in every block that contains them.",
        color="#94a3b8", ha="center", fontsize=9,
    )
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def save_crop_sheet(
    records: list[dict[str, Any]],
    colors: list[tuple[int, int, int]],
    crop_paths: list[Path],
    output: Path,
) -> None:
    columns, rows = 5, 6
    cell_w, cell_h = 420, 390
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (9, 16, 28))
    draw = ImageDraw.Draw(sheet)
    title_font, meta_font = font(19), font(13)
    for index, (rec, color, path) in enumerate(zip(records, colors, crop_paths)):
        col, row = index % columns, index // columns
        x, y = col * cell_w, row * cell_h
        crop = Image.open(path).convert("RGB")
        crop.thumbnail((cell_w - 20, 300), Image.Resampling.LANCZOS)
        px = x + (cell_w - crop.width) // 2
        py = y + 66 + (300 - crop.height) // 2
        sheet.paste(crop, (px, py))
        draw.rectangle((x + 5, y + 5, x + cell_w - 5, y + cell_h - 5), outline=color, width=3)
        draw.text((x + 13, y + 10), f"#{rec['cube_id']:02d}  start={tuple(rec['start'])}", fill=color, font=title_font)
        draw.text((x + 13, y + 37), f"bbox={tuple(rec['crop_box_4096'])}  n={rec['tokens']}", fill="white", font=meta_font)
    draw.text((12, rows * cell_h - 25), "Block ID matches exploded_blocks_numbered.png", fill="#94a3b8", font=meta_font)
    sheet.save(output)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    crops_dir = output / "crops_bbox_4096"
    crops_dir.mkdir(parents=True, exist_ok=True)
    coords = load_coords(args.support)
    image = Image.open(args.image).convert("RGB")
    if image.size != (CANONICAL_SIZE, CANONICAL_SIZE):
        raise ValueError(f"expected canonical 4096 image, got {image.size}")
    camera = json.loads(args.camera.read_text(encoding="utf-8"))
    if "camera" in camera:
        camera = camera["camera"]
    records = make_blocks(coords)
    colors = color_table()
    uv, finite = project_c64(coords[:, 1:], camera)

    overlay = image.copy()
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_font = font(30)
    crop_paths: list[Path] = []
    manifest_records: list[dict[str, Any]] = []
    for rec, color in zip(records, colors):
        rows = rec.pop("rows")
        block_finite = finite.index_select(0, rows)
        points = uv.index_select(0, rows)[block_finite]
        bbox, raw_bbox = aligned_bbox(points)
        rec["finite_projected_tokens"] = int(block_finite.sum())
        rec["raw_projected_bbox_4096"] = raw_bbox
        rec["crop_box_4096"] = bbox
        x0, y0, x1, y1 = bbox
        crop_path = crops_dir / f"block_{rec['cube_id']:02d}_bbox_{x0}_{y0}_{x1}_{y1}.png"
        image.crop((x0, y0, x1, y1)).save(crop_path)
        crop_paths.append(crop_path)
        overlay_draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=8)
        overlay_draw.text(
            (x0 + 8, y0 + 5), str(rec["cube_id"]),
            fill="white", stroke_width=4, stroke_fill=(0, 0, 0), font=overlay_font,
        )
        manifest_records.append({**rec, "crop_path": str(crop_path)})

    # Rebuild block rows because the JSON records above deliberately omit tensors.
    render_records = make_blocks(coords)
    render_exploded(
        coords, render_records, colors, output / "exploded_blocks_numbered.png",
        float(args.explode_gap), int(args.dpi),
    )
    overlay.save(output / "canonical_4096_bboxes_numbered.png")
    save_crop_sheet(records, colors, crop_paths, output / "crops_contact_sheet.png")
    summary = {
        "format": FORMAT,
        "status": "complete",
        "source_support": str(args.support.resolve()),
        "source_support_semantics": (
            "coordinates only: native SS C32 -> Shape512 C32 -> decoder C512 "
            "candidates -> C64 quantization; no post-partition flow is run"
        ),
        "source_image": str(args.image.resolve()),
        "camera": camera,
        "global_grid": GRID,
        "context": CONTEXT,
        "stride": STRIDE,
        "axis_starts": [0, 16, 32],
        "block_count": len(records),
        "support_tokens": int(len(coords)),
        "projection": (
            "exact ProjGrid C64 linspace and Blender-axis rotation, baseline camera, canonical 4096 pixels"
        ),
        "crop_rule": "finite active-point bbox, clipped to image, expanded outward to 16-pixel boundaries",
        "blocks": manifest_records,
    }
    (output / "blocks.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "blocks"}, indent=2))
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
