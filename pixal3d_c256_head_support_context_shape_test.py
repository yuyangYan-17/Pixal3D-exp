#!/usr/bin/env python3
"""Test a single translated global-C256 head support with native local C64 Shape Flow.

This intentionally uses no halo and no texture stage.  The only changed input
relative to a native 1024 head run is the C64 sparse support: it is cut from the
full-object C256 support and translated exactly into [0, 64)^3.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import numpy as np
import torch
from PIL import Image, ImageDraw

from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from pixal3d.renderers import MeshRenderer
from pixal3d.utils.render_utils import proj_camera_to_render_params


FORMAT = "pixal3d_c256_head_support_context_shape_test_v1"


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def save_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def parse_triplet(value: str) -> tuple[int, int, int]:
    result = tuple(int(x.strip()) for x in value.split(","))
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return result


def support_stats(local_xyz: torch.Tensor) -> dict:
    result = {
        "tokens": int(local_xyz.shape[0]),
        "min": local_xyz.min(0).values.tolist(),
        "max": local_xyz.max(0).values.tolist(),
        "mean": local_xyz.float().mean(0).tolist(),
        "std": local_xyz.float().std(0).tolist(),
        "boundary_counts": {},
    }
    for axis, name in enumerate("xyz"):
        result["boundary_counts"][name] = {
            "equal_0": int((local_xyz[:, axis] == 0).sum()),
            "equal_63": int((local_xyz[:, axis] == 63).sum()),
            "within_2_low": int((local_xyz[:, axis] <= 2).sum()),
            "within_2_high": int((local_xyz[:, axis] >= 61).sum()),
        }
    return result


def camera_views(fov: float, distance: float, angles: tuple[int, ...]):
    front, intrinsics = proj_camera_to_render_params(fov, distance)
    views = {}
    for angle in angles:
        a = math.radians(float(angle)); c, s = math.cos(a), math.sin(a)
        rotation = torch.tensor(
            [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]],
            dtype=front.dtype, device=front.device,
        )
        inverse = rotation.clone(); inverse[:3, :3] = rotation[:3, :3].T
        views[angle] = front @ inverse
    return views, intrinsics


def tensor_rgb(value: torch.Tensor, mask: torch.Tensor | None = None) -> Image.Image:
    array = value.detach().float().cpu().permute(1, 2, 0).numpy()
    if mask is not None:
        array *= mask.detach().float().cpu().numpy()[..., None]
    return Image.fromarray((np.clip(array, 0, 1) * 255 + .5).astype(np.uint8), "RGB")


def labeled_sheet(columns: list[tuple[str, Image.Image]], path: Path) -> None:
    if not columns:
        return
    tile_w, tile_h = columns[0][1].size
    label_h = 46
    sheet = Image.new("RGB", (tile_w * len(columns), tile_h + label_h), "#111111")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(columns):
        x = index * tile_w
        sheet.paste(image, (x, label_h))
        draw.text((x + 12, 14), label, fill="white")
    sheet.save(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--support", type=Path, default=Path("outputs/global_c256_cube_owner_flow_singleview_cuda4/support/global_c256_support.pt"))
    p.add_argument("--cube-start", type=parse_triplet, default=(162, 82, 145))
    p.add_argument("--input", type=Path, default=Path("assets/0_img_part.png"))
    p.add_argument("--baseline", type=Path, default=Path("outputs/baseline1024_0_img_part_cuda5"))
    p.add_argument("--output", type=Path, default=Path("outputs/c256_head_support_context_shape_cuda5"))
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--physical-cuda", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--angles", default="0,120,240")
    p.add_argument("--render-resolution", type=int, default=1024)
    p.add_argument("--ssaa", type=int, default=2)
    p.add_argument("--chunk-size", type=int, default=1_000_000)
    p.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args(); started = time.perf_counter()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.physical_cuda):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}, expected {args.physical_cuda}")
    device = torch.device(args.device); torch.cuda.set_device(device)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    baseline_summary = json.loads((args.baseline / "summary.json").read_text())
    fov = float(baseline_summary["camera_angle_x"])
    distance = float(baseline_summary["distance"])
    angles = tuple(int(x) for x in args.angles.split(","))

    payload = torch.load(args.support, map_location="cpu", weights_only=False)
    global_coords = payload["coords"].int().contiguous()
    xyz = global_coords[:, 1:4]
    start = torch.tensor(args.cube_start, dtype=torch.int32)
    inside = ((xyz >= start) & (xyz < start + 64)).all(1)
    global_rows = torch.where(inside)[0]
    local_xyz = (xyz[inside] - start).int().contiguous()
    if not local_xyz.numel() or torch.any(local_xyz < 0) or torch.any(local_xyz >= 64):
        raise RuntimeError("invalid translated support")
    local_coords = torch.cat((torch.zeros((len(local_xyz), 1), dtype=torch.int32), local_xyz), 1)
    stats = support_stats(local_xyz)
    save_torch(output / "translated_c64_support.pt", {
        "format": FORMAT, "cube_start_c256": list(args.cube_start),
        "global_row_ids": global_rows, "global_coords": global_coords[inside],
        "local_coords": local_coords, "no_halo": True,
    })
    save_json(output / "support_stats.json", stats)
    print(f"[support] start={args.cube_start} tokens={len(local_xyz):,} local=[0,63], no halo", flush=True)

    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=args.low_vram)
    canonical = pipeline.preprocess_canonical_images(Image.open(args.input))
    image_1024 = canonical["image_1024"]
    image_1024.save(output / "canonical_head_1024.png")
    save_json(output / "canonical_preprocess.json", canonical["metadata"])

    print("[condition] head subimage + baseline local camera + translated C64 support", flush=True)
    cond = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024, [image_1024], local_coords.to(device),
        camera_angle_x=fov, distance=distance, mesh_scale=1.0,
        grid_resolution_override=64,
    )
    # Re-seed immediately before HR Shape noise so this experiment is exactly reproducible.
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    params = dict(pipeline.shape_slat_sampler_params); params["steps"] = args.steps
    print(f"[shape-flow] tokens={len(local_xyz):,} steps={args.steps} seed={args.seed}", flush=True)
    shape = pipeline.sample_shape_slat(
        cond, pipeline.models["shape_slat_flow_model_1024"], local_coords.to(device), params,
    )
    save_torch(output / "shape_slat.pt", {
        "format": FORMAT, "coords": shape.coords.cpu(), "features": shape.feats.cpu(),
        "normalization": "denormalized decoder input",
    })
    del cond; gc.collect(); torch.cuda.empty_cache()

    print("[decode] shape only, C64 -> 1024", flush=True)
    meshes, subs = pipeline.decode_shape_slat(shape, 1024)
    if len(meshes) != 1:
        raise RuntimeError(f"shape decoder returned {len(meshes)} meshes")
    mesh = meshes[0]
    save_torch(output / "shape_mesh.pt", {"format": FORMAT, "mesh": mesh.cpu()})
    del subs, shape, pipeline; gc.collect(); torch.cuda.empty_cache()

    views, intrinsics = camera_views(fov, distance, angles)
    renderer = MeshRenderer({
        "resolution": args.render_resolution,
        "near": max(.01, distance - 2), "far": distance + 10,
        "ssaa": args.ssaa, "chunk_size": args.chunk_size,
        "antialias": True,
    }, device=str(device))
    live = mesh.to(device)
    rendered_paths = {}
    comparison_columns = []
    for angle in angles:
        print(f"[render] camera normal yaw={angle}", flush=True)
        result = renderer.render(live, views[angle], intrinsics, return_types=["normal", "mask"])
        image = tensor_rgb(result.normal, result.mask)
        test_path = output / f"camera_normal_yaw{angle:03d}.png"; image.save(test_path)
        rendered_paths[str(angle)] = str(test_path.resolve())
        baseline_path = args.baseline / "per_vertex_pbr" / f"yaw{angle:03d}" / "normal.png"
        if baseline_path.is_file():
            baseline_image = Image.open(baseline_path).convert("RGB").resize(image.size, Image.Resampling.LANCZOS)
            comparison_columns.extend([(f"baseline native support | yaw {angle}", baseline_image),
                                       (f"global-crop support | yaw {angle}", image)])
        del result
    sheet = output / "baseline_vs_global_crop_camera_normal_contact_sheet.png"
    labeled_sheet(comparison_columns, sheet)
    result = {
        "format": FORMAT, "status": "complete", "hypothesis_variable": "C64 sparse support distribution/context",
        "fixed": ["head subimage", "baseline local camera", "C64 HR Shape Flow", "C64-to-1024 shape decoder"],
        "no_halo": True, "texture_flow": False, "cube_start_c256": list(args.cube_start),
        "cube_end_exclusive_c256": [x + 64 for x in args.cube_start], "support": stats,
        "seed_note": "seed reset immediately before HR Shape noise; reproducible but not bitwise baseline HR noise because baseline consumed RNG in earlier cascade stages",
        "camera_angle_x": fov, "distance": distance, "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]), "renders": rendered_paths,
        "comparison_sheet": str(sheet.resolve()), "seconds": time.perf_counter() - started,
        "cuda_visible_devices": visible, "gpu": torch.cuda.get_device_name(device),
    }
    save_json(output / "summary.json", result)
    print(f"[done] {sheet}", flush=True)


if __name__ == "__main__":
    main()
