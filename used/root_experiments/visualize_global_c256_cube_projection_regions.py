#!/usr/bin/env python3
"""Overlay projected global-C256 cube membership regions on the 1024 input."""
from __future__ import annotations

import argparse
import colorsys
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as camera_core


def cube_color(cube_id: int) -> tuple[int, int, int]:
    # Golden-angle hue plus cycling saturation/value makes adjacent IDs easy
    # to distinguish while remaining deterministic across reruns.
    hue = (float(cube_id) * 0.6180339887498949) % 1.0
    saturation = (0.62, 0.78, 0.92)[cube_id % 3]
    value = (0.90, 1.00)[(cube_id // 3) % 2]
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return tuple(int(round(channel * 255.0)) for channel in rgb)


def official_endpoint_q(xyz: torch.Tensor, resolution: int = 256) -> torch.Tensor:
    return 2.0 * xyz.to(torch.float64) / float(resolution - 1) - 1.0


def physical_boundary_q(xyz_boundary: torch.Tensor, resolution: int = 256) -> torch.Tensor:
    """Map cell-boundary coordinates [0,256] to the physical [-1,1] volume."""
    return 2.0 * xyz_boundary.to(torch.float64) / float(resolution) - 1.0


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    points = sorted(set(points))
    if len(points) <= 2:
        return points

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def align_bbox_outward(
    points: torch.Tensor,
    width: int,
    height: int,
    multiple: int = 16,
) -> tuple[tuple[int, int, int, int], list[float], list[float]]:
    """Clip a projected volume bbox, then expand it outwards to a patch multiple."""
    if points.ndim != 2 or points.shape[1] != 2 or not points.numel():
        raise ValueError("points must be nonempty [N,2]")
    if width <= 0 or height <= 0 or multiple <= 0:
        raise ValueError("image dimensions and multiple must be positive")
    if width % multiple or height % multiple:
        raise ValueError("image dimensions must be divisible by bbox alignment")
    points = points.detach().cpu().to(torch.float64)
    if not torch.isfinite(points).all():
        raise ValueError("projected points contain NaN/Inf")
    raw_lo, raw_hi = points.amin(0), points.amax(0)
    clipped_lo = torch.maximum(raw_lo, torch.zeros(2, dtype=torch.float64))
    clipped_hi = torch.minimum(
        raw_hi,
        torch.tensor([float(width), float(height)], dtype=torch.float64),
    )
    if bool((clipped_hi <= clipped_lo).any()):
        raise RuntimeError("projected volume does not intersect the image")
    x0 = int(math.floor(float(clipped_lo[0]))) // multiple * multiple
    y0 = int(math.floor(float(clipped_lo[1]))) // multiple * multiple
    x1 = min(width, math.ceil(float(clipped_hi[0]) / multiple) * multiple)
    y1 = min(height, math.ceil(float(clipped_hi[1]) / multiple) * multiple)
    raw = [float(raw_lo[0]), float(raw_lo[1]), float(raw_hi[0]), float(raw_hi[1])]
    clipped = [
        float(clipped_lo[0]), float(clipped_lo[1]),
        float(clipped_hi[0]), float(clipped_hi[1]),
    ]
    return (x0, y0, x1, y1), raw, clipped


def render_z_merged_bboxes(
    image: Image.Image,
    camera: Mapping[str, float],
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Merge the seven z cubes and draw 7x7 aligned projected prism bboxes."""
    width, height = image.size
    starts = tuple(range(0, 256 - 64 + 1, 32))
    corner_bits = torch.tensor(
        [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)],
        dtype=torch.float64,
    )
    scale = torch.tensor([width / 1024.0, height / 1024.0], dtype=torch.float64)
    records: list[dict[str, Any]] = []
    for ix, sx in enumerate(starts):
        for iy, sy in enumerate(starts):
            prism_id = ix * len(starts) + iy
            start = torch.tensor([sx, sy, 0], dtype=torch.float64)
            extent = torch.tensor([64, 64, 256], dtype=torch.float64)
            corners_c256 = start[None] + corner_bits * extent[None]
            uv_1024, depth, finite = camera_core._project_global_q_to_image(
                physical_boundary_q(corners_c256),
                global_camera=camera,
                image_width=1024,
                image_height=1024,
            )
            if not bool(finite.all()):
                raise RuntimeError(f"z-merged prism ({sx}, {sy}) crosses the camera plane")
            # Match the flow route exactly: 1024 pixel centres become normalized
            # pixel edges before scaling into the canonical source image.
            points = (uv_1024.detach().cpu().to(torch.float64) + 0.5) * scale[None]
            box, raw_box, clipped_box = align_bbox_outward(
                points, width, height, int(args.bbox_alignment)
            )
            records.append({
                "prism_id": prism_id,
                "grid_xy": [ix, iy],
                "start_c256": [sx, sy, 0],
                "extent_c256": [64, 64, 256],
                "merged_cube_starts_z": list(starts),
                "color_rgb": list(cube_color(prism_id)),
                "raw_bbox_pixel_edges": [round(value, 6) for value in raw_box],
                "clipped_bbox_pixel_edges": [round(value, 6) for value in clipped_box],
                "aligned_bbox_xyxy": list(box),
                "crop_size": [box[2] - box[0], box[3] - box[1]],
                "mean_depth": float(depth.mean()),
            })

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for item in sorted(records, key=lambda row: row["mean_depth"], reverse=True):
        x0, y0, x1, y1 = item["aligned_bbox_xyxy"]
        color = tuple(item["color_rgb"])
        draw.rectangle(
            (x0, y0, x1 - 1, y1 - 1),
            fill=(*color, int(args.prism_fill_alpha)),
            outline=(*color, int(args.prism_line_alpha)),
            width=int(args.line_width),
        )
    try:
        label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(args.label_font_size))
    except OSError:
        label_font = ImageFont.load_default()
    label_padding = max(4, int(args.label_padding))
    occupied_label_boxes: list[tuple[int, int, int, int]] = []
    for item in sorted(records, key=lambda row: row["prism_id"]):
        x0, y0, x1, y1 = item["aligned_bbox_xyxy"]
        label = str(item["prism_id"])
        text_box = draw.textbbox((0, 0), label, font=label_font, stroke_width=1)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        cell_width = text_width + label_padding
        cell_height = text_height + label_padding
        label_x = x0 + label_padding
        label_y = y0 + label_padding
        found_slot = False
        for row in range(max(1, (y1 - y0) // cell_height)):
            for column in range(max(1, (x1 - x0) // cell_width)):
                candidate_x = x0 + label_padding + column * cell_width
                candidate_y = y0 + label_padding + row * cell_height
                candidate = (
                    candidate_x - label_padding // 2,
                    candidate_y - label_padding // 2,
                    candidate_x + text_width + label_padding // 2,
                    candidate_y + text_height + label_padding // 2,
                )
                if candidate[2] >= x1 or candidate[3] >= y1:
                    continue
                intersects = any(
                    candidate[0] < used[2] and candidate[2] > used[0]
                    and candidate[1] < used[3] and candidate[3] > used[1]
                    for used in occupied_label_boxes
                )
                if not intersects:
                    label_x, label_y = candidate_x, candidate_y
                    occupied_label_boxes.append(candidate)
                    found_slot = True
                    break
            if found_slot:
                break
        background = (
            label_x - label_padding // 2,
            label_y - label_padding // 2,
            min(x1 - 1, label_x + text_width + label_padding // 2),
            min(y1 - 1, label_y + text_height + label_padding // 2),
        )
        draw.rounded_rectangle(
            background,
            radius=max(2, label_padding // 2),
            fill=(0, 0, 0, int(args.label_background_alpha)),
        )
        draw.text(
            (label_x, label_y - text_box[1]),
            label,
            font=label_font,
            fill=(255, 255, 255, 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 255),
        )
        item["label_xy"] = [label_x, label_y]
    result = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    individual_dir = (
        Path(args.individual_dir).resolve()
        if args.individual_dir is not None
        else output_path.parent / f"{output_path.stem}_individual"
    )
    individual_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(records, key=lambda row: row["prism_id"]):
        x0, y0, x1, y1 = item["aligned_bbox_xyxy"]
        color = tuple(item["color_rgb"])
        single_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        single_draw = ImageDraw.Draw(single_overlay, "RGBA")
        single_draw.rectangle(
            (x0, y0, x1 - 1, y1 - 1),
            fill=(*color, int(args.prism_fill_alpha)),
            outline=(*color, int(args.prism_line_alpha)),
            width=int(args.line_width),
        )
        label = str(item["prism_id"])
        text_box = single_draw.textbbox((0, 0), label, font=label_font, stroke_width=1)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_x, label_y = x0 + label_padding, y0 + label_padding
        single_draw.rounded_rectangle(
            (
                label_x - label_padding // 2,
                label_y - label_padding // 2,
                label_x + text_width + label_padding // 2,
                label_y + text_height + label_padding // 2,
            ),
            radius=max(2, label_padding // 2),
            fill=(0, 0, 0, int(args.label_background_alpha)),
        )
        single_draw.text(
            (label_x, label_y - text_box[1]),
            label,
            font=label_font,
            fill=(255, 255, 255, 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 255),
        )
        sx, sy, _ = item["start_c256"]
        individual_path = individual_dir / f"prism_{item['prism_id']:02d}_xy_{sx:03d}_{sy:03d}.png"
        Image.alpha_composite(image.convert("RGBA"), single_overlay).convert("RGB").save(
            individual_path,
            compress_level=int(args.png_compress_level),
        )
        item["individual_output"] = str(individual_path)
    manifest = {
        "format": "global_c256_z_merged_prism_bbox_projection_v1",
        "mode": "z-merged-bboxes",
        "input_image": str(Path(args.image).resolve()),
        "camera": str(Path(args.camera).resolve()),
        "layout": "7x7 C64xC64xC256 prisms; xy size=64 stride=32; z merged over [0,256]",
        "projection_convention": "global camera C1024 pixel centres -> normalized pixel edges -> source image",
        "image_size": [width, height],
        "prism_count": len(records),
        "bbox_alignment": int(args.bbox_alignment),
        "fill_alpha": int(args.prism_fill_alpha),
        "line_alpha": int(args.prism_line_alpha),
        "line_width": int(args.line_width),
        "label_range": [0, len(records) - 1],
        "label_font_size": int(args.label_font_size),
        "output": str(output_path),
        "individual_output_dir": str(individual_dir),
        "individual_output_count": len(records),
        "prisms": sorted(records, key=lambda row: row["prism_id"]),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def render_full_generation_cubes(
    image: Image.Image,
    cubes_dir: Path,
    camera: Mapping[str, float],
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Project all 343 physical C256 cube volumes, including empty cubes."""
    width, height = image.size
    corner_bits = torch.tensor(
        [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)],
        dtype=torch.float64,
    )
    edge_pairs = [
        (a, b) for a in range(8) for b in range(a + 1, 8)
        if sum(corner_bits[a, axis] != corner_bits[b, axis] for axis in range(3)) == 1
    ]
    projected: list[dict[str, Any]] = []
    for cube_id in range(343):
        payload = torch.load(cubes_dir / f"cube_{cube_id:03d}.pt", map_location="cpu", weights_only=False)
        start = torch.tensor(payload["start"], dtype=torch.float64)
        corners_c256 = start[None] + corner_bits * 64.0
        uv, depth, finite = camera_core._project_global_q_to_image(
            physical_boundary_q(corners_c256),
            global_camera=camera,
            image_width=width,
            image_height=height,
        )
        if not bool(finite.all()):
            continue
        uv_cpu = uv.detach().cpu().to(torch.float64)
        points = [(float(p[0]), float(p[1])) for p in uv_cpu]
        projected.append({
            "cube_id": cube_id,
            "start_c256": [int(v) for v in payload["start"]],
            "start_c4096": [int(v) * 16 for v in payload["start"]],
            "membership_rows": int(payload["global_row_ids"].numel()),
            "empty": not bool(payload["global_row_ids"].numel()),
            "color_rgb": list(cube_color(cube_id)),
            "corners_xy_1024": [[round(x, 3), round(y, 3)] for x, y in points],
            "mean_depth": float(depth.mean()),
            "hull": convex_hull(points),
            "points": points,
        })

    # Far volumes first.  Fills are completed before the wireframe pass so no
    # translucent face can erase an already drawn cube edge.
    projected.sort(key=lambda item: item["mean_depth"], reverse=True)
    composite = image.convert("RGBA")
    for item in projected:
        hull = item["hull"]
        if len(hull) >= 3:
            layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
            ImageDraw.Draw(layer, "RGBA").polygon(
                hull,
                fill=(*item["color_rgb"], int(args.cube_fill_alpha)),
            )
            composite = Image.alpha_composite(composite, layer)
    line_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    line_draw = ImageDraw.Draw(line_layer, "RGBA")
    for item in projected:
        color = (*item["color_rgb"], int(args.cube_line_alpha))
        for a, b in edge_pairs:
            line_draw.line(
                (item["points"][a], item["points"][b]),
                fill=color,
                width=int(args.line_width),
            )
    result = Image.alpha_composite(composite, line_layer).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    manifest = {
        "format": "global_c256_full_generation_cube_projection_v1",
        "mode": "full-cubes",
        "input_image": str(Path(args.image).resolve()),
        "cubes_dir": str(cubes_dir),
        "camera": str(Path(args.camera).resolve()),
        "cube_layout": "C256 size=64 stride=32 starts=[0,32,64,96,128,160,192]",
        "physical_equivalent": "C4096 size=1024 stride=512",
        "projection_coordinate": "physical cell boundaries q=2*boundary/256-1",
        "image_size": [width, height],
        "cube_count": len(projected),
        "empty_cubes_included": sum(bool(item["empty"]) for item in projected),
        "nonempty_cubes_included": sum(not bool(item["empty"]) for item in projected),
        "fill_alpha": int(args.cube_fill_alpha),
        "line_alpha": int(args.cube_line_alpha),
        "line_width": int(args.line_width),
        "output": str(output_path),
        "cubes": [
            {key: value for key, value in item.items() if key not in {"hull", "points"}}
            for item in sorted(projected, key=lambda item: item["cube_id"])
        ],
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def render_global_aabb(
    image: Image.Image,
    camera: Mapping[str, float],
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Project the single world-space generation AABB [-0.5,0.5]^3."""
    width, height = image.size
    corners_world = torch.tensor(
        [(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
        dtype=torch.float64,
    )
    # camera_core consumes canonical q and internally computes
    # world = q / (2*mesh_scale), so this maps the requested world AABB
    # exactly even if mesh_scale is not one.
    q = corners_world * (2.0 * float(camera.get("mesh_scale", 1.0)))
    uv, depth, finite = camera_core._project_global_q_to_image(
        q,
        global_camera=camera,
        image_width=width,
        image_height=height,
    )
    if not bool(finite.all()):
        raise RuntimeError("global AABB has a corner behind the camera")
    points_image = [(float(p[0]), float(p[1])) for p in uv.detach().cpu()]
    if args.aabb_expanded_canvas:
        margin = int(args.aabb_canvas_margin)
        min_x = math.floor(min(0.0, min(x for x, _ in points_image)))
        min_y = math.floor(min(0.0, min(y for _, y in points_image)))
        max_x = math.ceil(max(float(width - 1), max(x for x, _ in points_image)))
        max_y = math.ceil(max(float(height - 1), max(y for _, y in points_image)))
        offset_x = margin - min_x
        offset_y = margin - min_y
        canvas_size = (
            max_x - min_x + 1 + 2 * margin,
            max_y - min_y + 1 + 2 * margin,
        )
        base = Image.new("RGBA", canvas_size, (0, 0, 0, 255))
        base.alpha_composite(image.convert("RGBA"), dest=(offset_x, offset_y))
        points = [(x + offset_x, y + offset_y) for x, y in points_image]
    else:
        margin = 0
        offset_x = offset_y = 0
        canvas_size = image.size
        base = image.convert("RGBA")
        points = points_image
    edge_pairs = [
        (a, b) for a in range(8) for b in range(a + 1, 8)
        if sum(corners_world[a, axis] != corners_world[b, axis] for axis in range(3)) == 1
    ]
    hull = convex_hull(points)
    result = base
    fill_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    ImageDraw.Draw(fill_layer, "RGBA").polygon(
        hull,
        fill=(*args.aabb_color, int(args.aabb_fill_alpha)),
    )
    result = Image.alpha_composite(result, fill_layer)
    line_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(line_layer, "RGBA")
    for a, b in edge_pairs:
        draw.line(
            (points[a], points[b]),
            fill=(*args.aabb_color, int(args.aabb_line_alpha)),
            width=int(args.aabb_line_width),
        )
    # Corner markers make the projected 8-corner topology unambiguous.
    radius = max(2, int(args.aabb_line_width) + 1)
    for x, y in points:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(*args.aabb_color, int(args.aabb_line_alpha)),
        )
    result = Image.alpha_composite(result, line_layer).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    manifest = {
        "format": "global_generation_aabb_projection_v1",
        "mode": "global-aabb",
        "input_image": str(Path(args.image).resolve()),
        "camera": str(Path(args.camera).resolve()),
        "aabb_world": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "corners_world": corners_world.tolist(),
        "corners_xy_original_image": [[round(x, 3), round(y, 3)] for x, y in points_image],
        "corners_xy_output_canvas": [[round(x, 3), round(y, 3)] for x, y in points],
        "corner_depth": [round(float(value), 6) for value in depth.detach().cpu()],
        "edges": edge_pairs,
        "color_rgb": list(args.aabb_color),
        "fill_alpha": int(args.aabb_fill_alpha),
        "line_alpha": int(args.aabb_line_alpha),
        "line_width": int(args.aabb_line_width),
        "expanded_black_canvas": bool(args.aabb_expanded_canvas),
        "original_image_size": [width, height],
        "original_image_offset_xy": [offset_x, offset_y],
        "output_canvas_size": list(canvas_size),
        "canvas_margin": margin,
        "output": str(output_path),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def square_box(uv: torch.Tensor, width: int, height: int, padding: float) -> tuple[float, float, float, float] | None:
    if not uv.numel():
        return None
    lo, hi = uv.amin(0), uv.amax(0)
    center = (lo + hi) * 0.5
    side = max(float((hi - lo).max()), 1.0) + 2.0 * float(padding)
    x0, y0 = float(center[0] - side / 2.0), float(center[1] - side / 2.0)
    x1, y1 = float(center[0] + side / 2.0), float(center[1] + side / 2.0)
    x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(width - 1), x1), min(float(height - 1), y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def render(args: argparse.Namespace) -> dict[str, Any]:
    image_path = Path(args.image).resolve()
    support_path = Path(args.support).resolve()
    cubes_dir = Path(args.cubes_dir).resolve()
    camera_path = Path(args.camera).resolve()
    output_path = Path(args.output).resolve()
    required = [image_path, camera_path]
    if args.mode in {"point-regions", "full-cubes"}:
        required.append(cubes_dir)
    if args.mode == "point-regions":
        required.append(support_path)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    camera_payload = json.loads(camera_path.read_text(encoding="utf-8"))
    camera: Mapping[str, float] = camera_payload.get("camera", camera_payload)
    if args.mode == "z-merged-bboxes":
        return render_z_merged_bboxes(image, camera, output_path, args)
    if args.mode == "global-aabb":
        return render_global_aabb(image, camera, output_path, args)
    if args.mode == "full-cubes":
        return render_full_generation_cubes(image, cubes_dir, camera, output_path, args)
    support = torch.load(support_path, map_location="cpu", weights_only=False)
    coords = support["coords"].to(torch.int32)
    q = official_endpoint_q(coords[:, 1:4])
    uv, depth, finite = camera_core._project_global_q_to_image(
        q,
        global_camera=camera,
        image_width=width,
        image_height=height,
    )
    finite = finite.cpu() & torch.isfinite(uv.cpu()).all(1) & torch.isfinite(depth.cpu())
    uv = uv.cpu()

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    records: list[dict[str, Any]] = []
    skipped_empty = 0
    skipped_no_visible_projection = 0
    for cube_id in range(343):
        payload = torch.load(cubes_dir / f"cube_{cube_id:03d}.pt", map_location="cpu", weights_only=False)
        row_ids = payload["global_row_ids"].to(torch.int64)
        if not row_ids.numel():
            skipped_empty += 1
            continue
        valid_rows = row_ids[finite.index_select(0, row_ids)]
        points = uv.index_select(0, valid_rows) if valid_rows.numel() else torch.empty((0, 2))
        in_or_crossing = points[
            (points[:, 0] >= -width) & (points[:, 0] < 2 * width)
            & (points[:, 1] >= -height) & (points[:, 1] < 2 * height)
        ] if points.numel() else points
        box = square_box(in_or_crossing, width, height, args.padding)
        if box is None:
            skipped_no_visible_projection += 1
            continue
        color = cube_color(cube_id)
        draw.rectangle(box, fill=(*color, int(args.fill_alpha)))
        for offset in range(int(args.line_width)):
            inset = (box[0] + offset, box[1] + offset, box[2] - offset, box[3] - offset)
            if inset[2] > inset[0] and inset[3] > inset[1]:
                draw.rectangle(inset, outline=(*color, int(args.line_alpha)), width=1)
        records.append({
            "cube_id": cube_id,
            "start_c256": list(payload["start"]),
            "start_c4096": [int(value) * 16 for value in payload["start"]],
            "membership_rows": int(row_ids.numel()),
            "valid_projected_rows": int(valid_rows.numel()),
            "color_rgb": list(color),
            "square_box_xyxy_1024": [round(value, 3) for value in box],
        })

    result = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    manifest = {
        "format": "global_c256_cube_projection_regions_v1",
        "input_image": str(image_path),
        "support": str(support_path),
        "cubes_dir": str(cubes_dir),
        "camera": str(camera_path),
        "projection_coordinate": "official endpoint q = 2*index/(256-1)-1",
        "image_size": [width, height],
        "cube_count": 343,
        "drawn_nonempty_cubes": len(records),
        "skipped_empty_cubes": skipped_empty,
        "skipped_no_visible_projection": skipped_no_visible_projection,
        "fill_alpha": int(args.fill_alpha),
        "line_alpha": int(args.line_alpha),
        "line_width": int(args.line_width),
        "padding_pixels": float(args.padding),
        "regions": records,
        "output": str(output_path),
    }
    output_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    root = Path("outputs/global_c256_cube_owner_flow_singleview_cuda4")
    baseline = Path("outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=baseline / "inputs/global_input_1024.png")
    parser.add_argument("--support", default=root / "support/global_c256_support.pt")
    parser.add_argument("--cubes-dir", default=root / "cubes")
    parser.add_argument("--camera", default=baseline / "global_camera.json")
    parser.add_argument("--output", default=root / "visualizations/cube_projection_regions_on_global_input_1024.png")
    parser.add_argument(
        "--mode",
        choices=("point-regions", "full-cubes", "global-aabb", "z-merged-bboxes"),
        default="point-regions",
    )
    parser.add_argument("--fill-alpha", type=int, default=48)
    parser.add_argument("--line-alpha", type=int, default=230)
    parser.add_argument("--cube-fill-alpha", type=int, default=5)
    parser.add_argument("--cube-line-alpha", type=int, default=190)
    parser.add_argument("--prism-fill-alpha", type=int, default=30)
    parser.add_argument("--prism-line-alpha", type=int, default=220)
    parser.add_argument("--bbox-alignment", type=int, default=16)
    parser.add_argument("--label-font-size", type=int, default=48)
    parser.add_argument("--label-padding", type=int, default=12)
    parser.add_argument("--label-background-alpha", type=int, default=180)
    parser.add_argument("--individual-dir", default=None)
    parser.add_argument("--png-compress-level", type=int, choices=range(0, 10), default=2)
    parser.add_argument("--aabb-color", type=lambda value: tuple(int(x) for x in value.split(",")), default=(0, 255, 255))
    parser.add_argument("--aabb-fill-alpha", type=int, default=24)
    parser.add_argument("--aabb-line-alpha", type=int, default=255)
    parser.add_argument("--aabb-line-width", type=int, default=4)
    parser.add_argument("--aabb-expanded-canvas", action="store_true")
    parser.add_argument("--aabb-canvas-margin", type=int, default=32)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--padding", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    result = render(parse_args())
    count = result.get("drawn_nonempty_cubes", result.get("cube_count", result.get("prism_count", 1)))
    print(f"[complete] mode={result.get('mode', 'point-regions')} drawn={count} output={result['output']}")
