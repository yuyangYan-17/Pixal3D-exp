#!/usr/bin/env python3
"""Compare independent-subimage C32 support with globally cut C32 blocks.

This is a CPU-only diagnostic.  It deliberately compares the first Shape512
support before any Flow features are generated:

  independent image -> sparse_structure_sampler -> C32
  full baseline C64 -> decoder.upsample -> global C128 -> context32 cut

The output records boundary occupancy, bbox margins, cardinality, image-crop
overlap, and orthographic occupancy maps for the head blocks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw


def load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def stats(coords: torch.Tensor) -> dict[str, Any]:
    xyz = coords[:, 1:].int() if coords.shape[1] == 4 else coords.int()
    if not len(xyz):
        return {"tokens": 0}
    lo = xyz.amin(0)
    hi = xyz.amax(0)
    distance = torch.minimum(xyz, 31 - xyz).amin(1)
    boundary = distance == 0
    bbox_volume = int(torch.prod(hi - lo + 1))
    return {
        "tokens": int(len(xyz)),
        "min": lo.tolist(),
        "max": hi.tolist(),
        "span": (hi - lo + 1).tolist(),
        "bbox_fill": float(len(xyz) / max(1, bbox_volume)),
        "boundary_tokens": int(boundary.sum()),
        "boundary_fraction": float(boundary.float().mean()),
        "boundary_by_axis": [
            int(((xyz[:, axis] == 0) | (xyz[:, axis] == 31)).sum())
            for axis in range(3)
        ],
        "interior_tokens_margin1": int((distance >= 1).sum()),
        "interior_tokens_margin2": int((distance >= 2).sum()),
        "interior_tokens_margin3": int((distance >= 3).sum()),
        "distance_histogram_0_to_7": [int((distance == i).sum()) for i in range(8)],
    }


def box_intersection(a: list[int], b: list[int]) -> int:
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def occupancy_panel(coords: torch.Tensor, title: str, size: int = 320) -> Image.Image:
    xyz = coords[:, 1:].int() if coords.shape[1] == 4 else coords.int()
    panel = Image.new("RGB", (size * 3, size + 34), "white")
    draw = ImageDraw.Draw(panel)
    labels = ((0, 1, "x-y"), (0, 2, "x-z"), (1, 2, "y-z"))
    for panel_id, (axis_a, axis_b, label) in enumerate(labels):
        ox = panel_id * size
        draw.text((ox + 5, 5), label, fill="black")
        draw.rectangle((ox + 20, 24, ox + size - 12, size + 20), outline="#777777")
        if len(xyz):
            a = xyz[:, axis_a].float()
            b = xyz[:, axis_b].float()
            px = ox + 20 + (a / 31.0 * (size - 32)).round().int()
            py = size + 20 - (b / 31.0 * (size - 32)).round().int()
            for x, y in zip(px.tolist(), py.tolist()):
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#1261a0")
    draw.text((5, size + 23), title, fill="black")
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block-root",
        type=Path,
        default=Path("outputs/c128_context32_local_cascade_4096_crop_geometry_cuda4"),
    )
    parser.add_argument(
        "--independent-root",
        type=Path,
        default=Path(
            "outputs/independent_subimage_baseline_compare_cuda4/"
            "back_headtop_inner4096_baseline1024"
        ),
    )
    parser.add_argument(
        "--crop-meta",
        type=Path,
        default=Path(
            "outputs/independent_subimage_baseline_compare_cuda4/inputs/"
            "back_headtop_inner4096_bbox1024.json"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    out = args.output or args.block_root / "support/c32_distribution_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    global_coords = load(
        args.block_root / "support/baseline_derived_c128_support.pt"
    )["coords"].int()
    layout = json.loads(
        (args.block_root / "support/context32_block_layout.json").read_text()
    )
    independent = load(args.independent_root / "support/coords_c32.pt")["coords"].int()
    independent_meta = json.loads(args.crop_meta.read_text())
    independent_box = independent_meta["selected_inner_crop_4096"]

    rows: list[dict[str, Any]] = []
    for block in layout["blocks"]:
        start = torch.tensor(block["start"], dtype=torch.int32)
        xyz = global_coords[:, 1:]
        mask = ((xyz >= start) & (xyz < start + 32)).all(1)
        local = torch.cat(
            (torch.zeros((int(mask.sum()), 1), dtype=torch.int32), xyz[mask] - start),
            dim=1,
        )
        row = {"cube_id": int(block["cube_id"]), "start": block["start"]}
        row.update(stats(local))
        condition_path = args.block_root / "conditions/shape512" / (
            f"cube_{int(block['cube_id']):02d}.pt"
        )
        if condition_path.is_file():
            condition = load(condition_path)
            crop = condition.get("crop", {})
            box = crop.get("crop_box_4096")
            if box:
                intersection = box_intersection(box, independent_box)
                row["condition_crop_box_4096"] = box
                row["independent_crop_intersection_pixels"] = intersection
                row["independent_crop_iou"] = intersection / float(
                    1024 * 1024 + 1024 * 1024 - intersection
                )
        rows.append(row)

    selected_ids = [57, 58, 53, 54, 41, 42, 43]
    by_id = {row["cube_id"]: row for row in rows}
    panels = [occupancy_panel(independent, "independent C32 (2413 tokens)")]
    for cube_id in selected_ids:
        block = by_id.get(cube_id)
        if block is None:
            continue
        start = torch.tensor(block["start"], dtype=torch.int32)
        xyz = global_coords[:, 1:]
        mask = ((xyz >= start) & (xyz < start + 32)).all(1)
        local = torch.cat(
            (torch.zeros((int(mask.sum()), 1), dtype=torch.int32), xyz[mask] - start),
            dim=1,
        )
        panels.append(occupancy_panel(local, f"cube {cube_id} ({len(local)} tokens)"))

    cols = 2
    rows_count = (len(panels) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * panels[0].width, rows_count * panels[0].height), "#888888")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % cols) * panel.width, (index // cols) * panel.height))
    sheet.save(out / "c32_occupancy_head_comparison.png")

    payload = {
        "format": "c32_support_distribution_diagnostic_v1",
        "independent_crop_box_4096": independent_box,
        "independent": stats(independent),
        "blocks_sorted_by_boundary_fraction": sorted(
            rows, key=lambda item: item.get("boundary_fraction", 0.0), reverse=True
        ),
        "blocks_sorted_by_tokens": sorted(
            rows, key=lambda item: item.get("tokens", 0), reverse=True
        ),
        "head_block_ids_visualized": selected_ids,
        "occupancy_sheet": str((out / "c32_occupancy_head_comparison.png").resolve()),
    }
    (out / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
