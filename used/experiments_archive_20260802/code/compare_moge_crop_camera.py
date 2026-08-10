#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare MoGe-estimated crop intrinsics with intrinsics derived from a global camera.

The test intentionally separates two questions:

1. Camera intrinsics:
   Run MoGe on the global canonical image and on each raw image crop.  Derive the
   exact crop intrinsics from the global MoGe intrinsics using the standard
   crop-and-resize transform, then compare fx/fy/cx/cy and ray directions.

2. Pixal3D camera distance:
   Pixal3D does not take distance from MoGe.  It derives distance from the FOV
   and the canonical cube boundary.  Therefore this script reports separately:
     - the original global Pixal3D distance (correct for a digital crop),
     - the distance obtained by recomputing from the derived crop FOV, and
     - the distance obtained from the crop MoGe FOV.

For an off-centre crop, the exact crop camera is an off-axis camera.  A camera
represented only by centred FOV + distance cannot reproduce it.  The script
therefore also compares against Pixal3D's centred-principal-point approximation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw


DEFAULT_MOGE_MODEL = "/home/nvme04/yyyan/download/model/moge-2-vitl/model.pt"
CANONICAL_SIZE = 4096
DEFAULT_TILE_SIZE = 1024
DEFAULT_TILE_STRIDE = 512
DEFAULT_OUTPUT_SIZE = 1024


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def fov_x_rad(self) -> float:
        return 2.0 * math.atan(float(self.width) / (2.0 * float(self.fx)))

    @property
    def fov_y_rad(self) -> float:
        return 2.0 * math.atan(float(self.height) / (2.0 * float(self.fy)))

    @property
    def fov_x_deg(self) -> float:
        return math.degrees(self.fov_x_rad)

    @property
    def fov_y_deg(self) -> float:
        return math.degrees(self.fov_y_rad)

    @property
    def optical_axis_shift_x_deg(self) -> float:
        # Positive means the optical axis lies left of the image centre.
        return math.degrees(
            math.atan2(float(self.width) * 0.5 - float(self.cx), float(self.fx))
        )

    @property
    def optical_axis_shift_y_deg(self) -> float:
        # Positive means the optical axis lies above the image centre.
        return math.degrees(
            math.atan2(float(self.height) * 0.5 - float(self.cy), float(self.fy))
        )

    def normalized_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx / self.width, 0.0, self.cx / self.width],
                [0.0, self.fy / self.height, self.cy / self.height],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def pixel_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "fov_x_rad": self.fov_x_rad,
                "fov_y_rad": self.fov_y_rad,
                "fov_x_deg": self.fov_x_deg,
                "fov_y_deg": self.fov_y_deg,
                "optical_axis_shift_x_deg": self.optical_axis_shift_x_deg,
                "optical_axis_shift_y_deg": self.optical_axis_shift_y_deg,
                "K_normalized": self.normalized_matrix().tolist(),
                "K_pixels": self.pixel_matrix().tolist(),
            }
        )
        return payload


@dataclass(frozen=True)
class TileSpec:
    tile_id: int
    box: Tuple[int, int, int, int]


class CameraTestError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare MoGe crop intrinsics with exact crop intrinsics derived "
            "from the global MoGe camera."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--image",
        type=str,
        help=(
            "Original source image. Exact local canonical preprocessing is "
            "implemented only when it contains a non-opaque alpha channel."
        ),
    )
    source.add_argument(
        "--canonical-4096",
        type=str,
        help="Already prepared Pixal3D canonical 4096x4096 RGB image.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--moge-model-path", type=str, default=DEFAULT_MOGE_MODEL)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tile-ids", type=str, default="24")
    parser.add_argument(
        "--boxes",
        type=str,
        default="",
        help=(
            "Optional semicolon-separated boxes x0,y0,x1,y1. When provided, "
            "--tile-ids is ignored and IDs are assigned from 0."
        ),
    )
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--tile-stride", type=int, default=DEFAULT_TILE_STRIDE)
    parser.add_argument("--tile-output-size", type=int, default=DEFAULT_OUTPUT_SIZE)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument(
        "--camera-image-resolution",
        type=int,
        default=512,
        help="Pixal3D distance_from_fov image_resolution; inference.py defaults to 512.",
    )
    parser.add_argument("--ray-grid", type=int, default=21)
    parser.add_argument("--focal-rel-tol", type=float, default=0.05)
    parser.add_argument("--principal-point-tol-px", type=float, default=8.0)
    parser.add_argument("--ray-p95-tol-deg", type=float, default=1.0)
    return parser.parse_args()


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(path)


def canonicalize_rgba_exact(source: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
    """Reproduce the alpha-input branch of Pixal3D preprocess_canonical_images."""
    if source.mode != "RGBA":
        raise CameraTestError(
            "--image must contain an RGBA alpha channel for exact lightweight "
            "canonical preprocessing. Otherwise pass the previously saved "
            "canonical_4096.png with --canonical-4096."
        )
    alpha = np.asarray(source.getchannel("A"))
    if np.all(alpha == 255):
        raise CameraTestError(
            "The source alpha channel is fully opaque. Pass the Pixal3D "
            "canonical_4096.png via --canonical-4096, or use an RGBA source "
            "with foreground alpha."
        )
    foreground = np.argwhere(alpha > 0.8 * 255)
    if foreground.size == 0:
        raise CameraTestError("Foreground alpha mask is empty")
    bbox = (
        int(np.min(foreground[:, 1])),
        int(np.min(foreground[:, 0])),
        int(np.max(foreground[:, 1])) + 1,
        int(np.max(foreground[:, 0])) + 1,
    )
    center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
    side = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    side = max(1, int(math.ceil(side * 1.1)))
    left = int(math.floor(center[0] - side / 2.0))
    top = int(math.floor(center[1] - side / 2.0))
    square_extent = (left, top, left + side, top + side)

    square_rgba = source.crop(square_extent).convert("RGBA")
    square_np = np.asarray(square_rgba).astype(np.float32) / 255.0
    rgb = square_np[:, :, :3] * square_np[:, :, 3:4]
    square_rgb = Image.fromarray(
        (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB"
    )
    canonical = square_rgb.resize(
        (CANONICAL_SIZE, CANONICAL_SIZE), Image.Resampling.LANCZOS
    )
    metadata = {
        "source_size": [source.width, source.height],
        "foreground_bbox_source": list(bbox),
        "square_extent_source": list(square_extent),
        "square_side_source": side,
        "method": "Pixal3D alpha branch reproduced without loading pipeline",
    }
    return canonical, metadata


def load_canonical_image(args: argparse.Namespace) -> Tuple[Image.Image, Dict[str, Any]]:
    if args.canonical_4096:
        image = Image.open(args.canonical_4096).convert("RGB")
        if image.size != (CANONICAL_SIZE, CANONICAL_SIZE):
            raise CameraTestError(
                f"--canonical-4096 must be 4096x4096, got {image.size}"
            )
        return image, {
            "method": "provided canonical image",
            "path": str(Path(args.canonical_4096).resolve()),
        }
    source = Image.open(args.image)
    return canonicalize_rgba_exact(source)


def load_moge(model_path: str, device: torch.device):
    try:
        from moge.model.v2 import MoGeModel
    except Exception as exc:
        raise CameraTestError(
            "Cannot import moge.model.v2.MoGeModel. Run this script inside the "
            "Pixal3D environment."
        ) from exc
    print(f"[MoGe] loading {model_path}")
    model = MoGeModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def infer_moge_intrinsics(model, image: Image.Image, device: torch.device) -> Tuple[Intrinsics, Dict[str, Any]]:
    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(device)
    output = model.infer(tensor)
    if "intrinsics" not in output:
        raise CameraTestError("MoGe output does not contain 'intrinsics'")
    matrix = output["intrinsics"].detach().float().squeeze().cpu().numpy()
    if matrix.shape != (3, 3):
        raise CameraTestError(f"Unexpected MoGe intrinsics shape {matrix.shape}")
    width, height = rgb.size
    intrinsics = Intrinsics(
        width=width,
        height=height,
        fx=float(matrix[0, 0]) * width,
        fy=float(matrix[1, 1]) * height,
        cx=float(matrix[0, 2]) * width,
        cy=float(matrix[1, 2]) * height,
    )
    extras: Dict[str, Any] = {
        "raw_normalized_intrinsics": matrix.tolist(),
        "output_keys": sorted(str(key) for key in output.keys()),
    }
    return intrinsics, extras


def scale_intrinsics(intrinsics: Intrinsics, width: int, height: int) -> Intrinsics:
    sx = float(width) / float(intrinsics.width)
    sy = float(height) / float(intrinsics.height)
    return Intrinsics(
        width=width,
        height=height,
        fx=intrinsics.fx * sx,
        fy=intrinsics.fy * sy,
        cx=intrinsics.cx * sx,
        cy=intrinsics.cy * sy,
    )


def derive_crop_intrinsics(
    global_intrinsics: Intrinsics,
    box: Sequence[int],
    output_width: int,
    output_height: int,
) -> Intrinsics:
    x0, y0, x1, y1 = (int(value) for value in box)
    crop_width = x1 - x0
    crop_height = y1 - y0
    if crop_width <= 0 or crop_height <= 0:
        raise CameraTestError(f"Invalid crop box {tuple(box)}")
    sx = float(output_width) / float(crop_width)
    sy = float(output_height) / float(crop_height)
    return Intrinsics(
        width=output_width,
        height=output_height,
        fx=global_intrinsics.fx * sx,
        fy=global_intrinsics.fy * sy,
        cx=(global_intrinsics.cx - float(x0)) * sx,
        cy=(global_intrinsics.cy - float(y0)) * sy,
    )


def centered_approximation(intrinsics: Intrinsics) -> Intrinsics:
    return Intrinsics(
        width=intrinsics.width,
        height=intrinsics.height,
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.width / 2.0,
        cy=intrinsics.height / 2.0,
    )


def pixal3d_distance_from_fov(
    camera_angle_x: float,
    mesh_scale: float,
    extend_pixel: int,
    image_resolution: int,
) -> float:
    """Equivalent to inference.py::distance_from_fov for grid_point [-1,0,0]."""
    if mesh_scale <= 0:
        raise CameraTestError("mesh_scale must be positive")
    focal_pixels = float(image_resolution) / (
        2.0 * math.tan(float(camera_angle_x) / 2.0)
    )
    x_world = -1.0 / (2.0 * float(mesh_scale))
    x_target = -float(extend_pixel)
    x_ndc = x_target - float(image_resolution) / 2.0
    if abs(x_ndc) < 1e-12:
        raise CameraTestError("distance formula has zero x_ndc denominator")
    return focal_pixels * x_world / x_ndc


def make_tile_layout(
    canonical_size: int, tile_size: int, stride: int
) -> List[TileSpec]:
    if tile_size <= 0 or stride <= 0 or tile_size > canonical_size:
        raise CameraTestError("Invalid tile size or stride")
    starts = list(range(0, canonical_size - tile_size + 1, stride))
    if not starts or starts[-1] != canonical_size - tile_size:
        raise CameraTestError(
            "Tile layout does not land on the final image edge; use a compatible stride"
        )
    tiles: List[TileSpec] = []
    for y0 in starts:
        for x0 in starts:
            tiles.append(TileSpec(len(tiles), (x0, y0, x0 + tile_size, y0 + tile_size)))
    return tiles


def parse_boxes(value: str) -> List[TileSpec]:
    boxes: List[TileSpec] = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [int(token.strip()) for token in item.split(",")]
        if len(parts) != 4:
            raise CameraTestError(f"Invalid box: {item}")
        boxes.append(TileSpec(len(boxes), tuple(parts)))
    if not boxes:
        raise CameraTestError("--boxes did not contain a valid box")
    return boxes


def select_tiles(args: argparse.Namespace) -> List[TileSpec]:
    if args.boxes.strip():
        return parse_boxes(args.boxes)
    layout = make_tile_layout(CANONICAL_SIZE, args.tile_size, args.tile_stride)
    requested = {
        int(token.strip())
        for token in args.tile_ids.split(",")
        if token.strip()
    }
    invalid = sorted(tile_id for tile_id in requested if tile_id < 0 or tile_id >= len(layout))
    if invalid:
        raise CameraTestError(
            f"Invalid tile IDs {invalid}; valid range is 0..{len(layout)-1}"
        )
    return [tile for tile in layout if tile.tile_id in requested]


def rays_for_grid(intrinsics: Intrinsics, grid_size: int) -> np.ndarray:
    if grid_size < 2:
        raise CameraTestError("ray-grid must be at least 2")
    xs = np.linspace(0.5, intrinsics.width - 0.5, grid_size, dtype=np.float64)
    ys = np.linspace(0.5, intrinsics.height - 0.5, grid_size, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    rays = np.stack(
        [
            (xx - intrinsics.cx) / intrinsics.fx,
            -(yy - intrinsics.cy) / intrinsics.fy,
            np.ones_like(xx),
        ],
        axis=-1,
    ).reshape(-1, 3)
    norm = np.linalg.norm(rays, axis=1, keepdims=True)
    return rays / np.clip(norm, 1e-12, None)


def ray_error_degrees(a: Intrinsics, b: Intrinsics, grid_size: int) -> Dict[str, float]:
    if (a.width, a.height) != (b.width, b.height):
        raise CameraTestError("Ray comparison requires equal image dimensions")
    rays_a = rays_for_grid(a, grid_size)
    rays_b = rays_for_grid(b, grid_size)
    dots = np.clip(np.sum(rays_a * rays_b, axis=1), -1.0, 1.0)
    degrees = np.degrees(np.arccos(dots))
    return {
        "mean_deg": float(np.mean(degrees)),
        "median_deg": float(np.median(degrees)),
        "p95_deg": float(np.percentile(degrees, 95)),
        "max_deg": float(np.max(degrees)),
    }


def safe_relative_error(value: float, reference: float) -> float:
    return abs(float(value) - float(reference)) / max(abs(float(reference)), 1e-12)


def compare_intrinsics(
    estimate: Intrinsics,
    reference: Intrinsics,
    ray_grid: int,
    focal_rel_tol: float,
    principal_point_tol_px: float,
    ray_p95_tol_deg: float,
) -> Dict[str, Any]:
    ray_error = ray_error_degrees(estimate, reference, ray_grid)
    fx_rel = safe_relative_error(estimate.fx, reference.fx)
    fy_rel = safe_relative_error(estimate.fy, reference.fy)
    cx_abs = abs(estimate.cx - reference.cx)
    cy_abs = abs(estimate.cy - reference.cy)
    result = {
        "fx_relative_error": fx_rel,
        "fy_relative_error": fy_rel,
        "cx_absolute_error_px": cx_abs,
        "cy_absolute_error_px": cy_abs,
        "fov_x_absolute_error_deg": abs(estimate.fov_x_deg - reference.fov_x_deg),
        "fov_y_absolute_error_deg": abs(estimate.fov_y_deg - reference.fov_y_deg),
        "normalized_K_frobenius_error": float(
            np.linalg.norm(estimate.normalized_matrix() - reference.normalized_matrix())
        ),
        "ray_angular_error": ray_error,
    }
    result["consistent_under_thresholds"] = bool(
        fx_rel <= focal_rel_tol
        and fy_rel <= focal_rel_tol
        and cx_abs <= principal_point_tol_px
        and cy_abs <= principal_point_tol_px
        and ray_error["p95_deg"] <= ray_p95_tol_deg
    )
    return result


def annotate_intrinsics(
    image: Image.Image,
    output: Path,
    derived: Intrinsics,
    moge: Intrinsics,
) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    def cross(x: float, y: float, color: Tuple[int, int, int], radius: int, width: int) -> None:
        draw.line((x - radius, y, x + radius, y), fill=color, width=width)
        draw.line((x, y - radius, x, y + radius), fill=color, width=width)

    cross(canvas.width / 2.0, canvas.height / 2.0, (255, 220, 0), 18, 3)
    cross(derived.cx, derived.cy, (0, 255, 255), 24, 4)
    cross(moge.cx, moge.cy, (255, 40, 40), 14, 4)
    draw.rectangle((0, 0, canvas.width, 76), fill=(0, 0, 0))
    draw.text((8, 8), "yellow=image center  cyan=derived principal point  red=MoGe", fill=(255, 255, 255))
    draw.text(
        (8, 30),
        f"derived fx={derived.fx:.2f} cx={derived.cx:.2f} cy={derived.cy:.2f}",
        fill=(0, 255, 255),
    )
    draw.text(
        (8, 50),
        f"MoGe    fx={moge.fx:.2f} cx={moge.cx:.2f} cy={moge.cy:.2f}",
        fill=(255, 90, 90),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def flatten_summary(tile_result: Dict[str, Any]) -> Dict[str, Any]:
    exact = tile_result["comparison_moge_vs_exact_derived"]
    centered = tile_result["comparison_moge_vs_centered_approximation"]
    distances = tile_result["pixal3d_distance_comparison"]
    return {
        "tile_id": tile_result["tile_id"],
        "box": ",".join(str(v) for v in tile_result["box"]),
        "derived_fov_x_deg": tile_result["derived_exact_intrinsics"]["fov_x_deg"],
        "moge_fov_x_deg": tile_result["crop_moge_intrinsics"]["fov_x_deg"],
        "fov_error_deg": exact["fov_x_absolute_error_deg"],
        "fx_relative_error": exact["fx_relative_error"],
        "fy_relative_error": exact["fy_relative_error"],
        "cx_error_px": exact["cx_absolute_error_px"],
        "cy_error_px": exact["cy_absolute_error_px"],
        "exact_ray_p95_deg": exact["ray_angular_error"]["p95_deg"],
        "centered_ray_p95_deg": tile_result[
            "comparison_centered_approximation_vs_exact_derived"
        ]["ray_angular_error"]["p95_deg"],
        "moge_vs_exact_consistent": exact["consistent_under_thresholds"],
        "moge_vs_centered_consistent": centered["consistent_under_thresholds"],
        "global_distance": distances["global_distance"],
        "derived_keep_global_distance": distances["derived_crop_keep_global_distance"],
        "derived_recomputed_distance": distances["derived_crop_recomputed_from_fov"],
        "crop_moge_distance": distances["crop_moge_recomputed_from_fov"],
    }


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_4096, preprocess_metadata = load_canonical_image(args)
    canonical_path = output_dir / "canonical_4096.png"
    canonical_4096.save(canonical_path)
    global_image = canonical_4096.resize((1024, 1024), Image.Resampling.LANCZOS)
    global_path = output_dir / "canonical_1024_for_moge.png"
    global_image.save(global_path)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CameraTestError("CUDA was requested but torch.cuda.is_available() is false")
    model = load_moge(args.moge_model_path, device)

    print("[MoGe] estimating global canonical camera")
    global_moge_1024, global_extras = infer_moge_intrinsics(model, global_image, device)
    global_moge_4096 = scale_intrinsics(global_moge_1024, CANONICAL_SIZE, CANONICAL_SIZE)
    global_distance = pixal3d_distance_from_fov(
        global_moge_1024.fov_x_rad,
        args.mesh_scale,
        args.extend_pixel,
        args.camera_image_resolution,
    )
    print(
        f"[global] fov={global_moge_1024.fov_x_deg:.6f} deg "
        f"fx={global_moge_1024.fx:.3f} cx={global_moge_1024.cx:.3f} "
        f"cy={global_moge_1024.cy:.3f} distance={global_distance:.8f}"
    )

    tiles = select_tiles(args)
    tile_results: List[Dict[str, Any]] = []
    for tile in tiles:
        x0, y0, x1, y1 = tile.box
        if x0 < 0 or y0 < 0 or x1 > CANONICAL_SIZE or y1 > CANONICAL_SIZE:
            raise CameraTestError(f"Tile {tile.tile_id} is outside canonical image: {tile.box}")
        raw_crop = canonical_4096.crop(tile.box).resize(
            (args.tile_output_size, args.tile_output_size), Image.Resampling.LANCZOS
        )
        tile_dir = output_dir / f"tile_{tile.tile_id:02d}"
        tile_dir.mkdir(parents=True, exist_ok=True)
        crop_path = tile_dir / "raw_crop_1024.png"
        raw_crop.save(crop_path)

        exact = derive_crop_intrinsics(
            global_moge_4096,
            tile.box,
            args.tile_output_size,
            args.tile_output_size,
        )
        centered = centered_approximation(exact)

        print(f"[MoGe] estimating tile {tile.tile_id} box={tile.box}")
        crop_moge, crop_extras = infer_moge_intrinsics(model, raw_crop, device)

        exact_comparison = compare_intrinsics(
            crop_moge,
            exact,
            args.ray_grid,
            args.focal_rel_tol,
            args.principal_point_tol_px,
            args.ray_p95_tol_deg,
        )
        centered_comparison = compare_intrinsics(
            crop_moge,
            centered,
            args.ray_grid,
            args.focal_rel_tol,
            args.principal_point_tol_px,
            args.ray_p95_tol_deg,
        )
        centered_vs_exact = compare_intrinsics(
            centered,
            exact,
            args.ray_grid,
            args.focal_rel_tol,
            args.principal_point_tol_px,
            args.ray_p95_tol_deg,
        )

        derived_recomputed_distance = pixal3d_distance_from_fov(
            exact.fov_x_rad,
            args.mesh_scale,
            args.extend_pixel,
            args.camera_image_resolution,
        )
        crop_moge_distance = pixal3d_distance_from_fov(
            crop_moge.fov_x_rad,
            args.mesh_scale,
            args.extend_pixel,
            args.camera_image_resolution,
        )

        annotation_path = tile_dir / "principal_point_comparison.png"
        annotate_intrinsics(raw_crop, annotation_path, exact, crop_moge)

        result: Dict[str, Any] = {
            "tile_id": tile.tile_id,
            "box": list(tile.box),
            "crop_path": str(crop_path),
            "principal_point_visualization": str(annotation_path),
            "derived_exact_intrinsics": exact.to_dict(),
            "pixal3d_centered_approximation": centered.to_dict(),
            "crop_moge_intrinsics": crop_moge.to_dict(),
            "crop_moge_extras": crop_extras,
            "comparison_moge_vs_exact_derived": exact_comparison,
            "comparison_moge_vs_centered_approximation": centered_comparison,
            "comparison_centered_approximation_vs_exact_derived": centered_vs_exact,
            "pixal3d_distance_comparison": {
                "important_note": (
                    "MoGe predicts intrinsics/FOV. Pixal3D recomputes distance "
                    "from FOV; this distance is not independently predicted by MoGe."
                ),
                "global_distance": global_distance,
                "derived_crop_keep_global_distance": global_distance,
                "derived_crop_recomputed_from_fov": derived_recomputed_distance,
                "crop_moge_recomputed_from_fov": crop_moge_distance,
                "derived_recomputed_over_global_ratio": (
                    derived_recomputed_distance / global_distance
                ),
                "crop_moge_over_global_ratio": crop_moge_distance / global_distance,
            },
        }
        atomic_json(tile_dir / "camera_comparison.json", result)
        tile_results.append(result)
        print(
            f"[tile {tile.tile_id}] derived_fov={exact.fov_x_deg:.6f} deg "
            f"moge_fov={crop_moge.fov_x_deg:.6f} deg "
            f"fx_rel={exact_comparison['fx_relative_error']:.6f} "
            f"cx_err={exact_comparison['cx_absolute_error_px']:.3f}px "
            f"cy_err={exact_comparison['cy_absolute_error_px']:.3f}px "
            f"ray_p95={exact_comparison['ray_angular_error']['p95_deg']:.6f}deg "
            f"consistent={exact_comparison['consistent_under_thresholds']}"
        )

    payload = {
        "experiment": "moge_vs_global_derived_crop_camera",
        "preprocess": preprocess_metadata,
        "canonical_4096": str(canonical_path),
        "global_image_for_moge": str(global_path),
        "global_moge_intrinsics_1024": global_moge_1024.to_dict(),
        "global_moge_intrinsics_4096": global_moge_4096.to_dict(),
        "global_moge_extras": global_extras,
        "global_pixal3d_camera": {
            "camera_angle_x_rad": global_moge_1024.fov_x_rad,
            "camera_angle_x_deg": global_moge_1024.fov_x_deg,
            "distance": global_distance,
            "mesh_scale": args.mesh_scale,
            "extend_pixel": args.extend_pixel,
            "camera_image_resolution": args.camera_image_resolution,
        },
        "thresholds": {
            "focal_relative_tolerance": args.focal_rel_tol,
            "principal_point_tolerance_px": args.principal_point_tol_px,
            "ray_p95_tolerance_deg": args.ray_p95_tol_deg,
        },
        "tiles": tile_results,
    }
    atomic_json(output_dir / "summary.json", payload)
    write_csv(
        output_dir / "summary.csv",
        [flatten_summary(result) for result in tile_results],
    )
    print(f"[done] {output_dir / 'summary.json'}")
    print(f"[done] {output_dir / 'summary.csv'}")


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except CameraTestError as exc:
        raise SystemExit(f"[error] {exc}") from exc


if __name__ == "__main__":
    main()
