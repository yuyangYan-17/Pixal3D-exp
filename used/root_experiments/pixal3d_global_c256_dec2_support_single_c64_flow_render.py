#!/usr/bin/env python3
"""Run one C64 block from the refined C256 support and render it with C256 points."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import torch
from PIL import Image

import pixal3d_global4096_singleview_shared_slat_shape_tex_sr as expc
import pixal3d_global_c256_cube_owner_flow_singleview as cube_flow
import pixal3d_global_c256_restructured_blocks_singleview as core
import pixal3d_render_global_slat_points_with_local_cubes as composite
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from pixal3d.renderers import PbrMeshRenderer
from pixal3d_baseline1024_pbr_mesh_compare import _make_camera_views
from render_pixal3d_raw_ovoxel import load_envmap


FORMAT = "pixal3d_global_c256_dec2_support_single_c64_flow_render_v1"


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    baseline = Path("outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, default=Path("assets/images/0_img.png"))
    p.add_argument("--camera", type=Path, default=baseline / "global_camera.json")
    p.add_argument("--support", type=Path, default=Path(
        "outputs/global_c256_c32_stride32_dec2_support_cuda5/global_support/global_c256_support.pt"))
    p.add_argument("--output", type=Path, default=Path(
        "outputs/global_c256_dec2_support_head_c64_3_1_2_cuda5"))
    p.add_argument("--block-index", default="3,1,2")
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--shape-seed", type=int, default=54001)
    p.add_argument("--texture-seed", type=int, default=55001)
    p.add_argument("--angles", default="0,60,120,180,240,300")
    p.add_argument("--resolution", type=int, default=4096)
    p.add_argument("--point-radius", type=int, default=3)
    p.add_argument("--face-chunk-size", type=int, default=1_000_000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--physical-cuda", type=int, default=5)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.physical_cuda):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}, expected {args.physical_cuda}")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    index = tuple(int(v) for v in args.block_index.split(","))
    if len(index) != 3 or any(v < 0 or v >= 4 for v in index):
        raise ValueError("--block-index must contain three values in [0,3]")
    block_id = index[0] * 16 + index[1] * 4 + index[2]
    start = tuple(v * 64 for v in index)
    start_t = torch.tensor(start, dtype=torch.int32)
    camera = json.loads(args.camera.read_text())
    camera["mesh_scale"] = 1.0

    coords = torch.load(args.support, map_location="cpu", weights_only=False)["coords"].int()
    xyz = coords[:, 1:4]
    inside = ((xyz >= start_t) & (xyz < start_t + 64)).all(1)
    source_rows = torch.where(inside)[0].long()
    if not len(source_rows):
        raise RuntimeError(f"selected C64 block {index} is empty")
    local_xyz = xyz.index_select(0, source_rows) - start_t
    local_coords = torch.cat((torch.zeros((len(local_xyz), 1), dtype=torch.int32), local_xyz), 1)
    global_coords = coords.index_select(0, source_rows)
    local_rows = torch.arange(len(local_coords), dtype=torch.long)
    record = {
        "cube_id": block_id, "block_index": index, "start": start,
        "global_row_ids": local_rows, "owned_row_ids": local_rows,
        "local_xyz": local_xyz,
    }
    core.atomic_json(output / "config.json", {
        "format": FORMAT, "args": vars(args), "camera": camera,
        "block_id": block_id, "block_index": index, "start_c256": start,
        "end_exclusive_c256": [v + 64 for v in start],
        "source_support_tokens": len(coords), "local_block_tokens": len(local_coords),
        "flow": "local C64 Shape1024 then Texture1024; fresh noise; full-image token + correct global-position projection",
        "placeholder": "all refined Global C256 points outside selected C64",
    })
    cube_flow.atomic_save(output / "local_c64_support.pt", {
        "format": FORMAT, "coords": local_coords, "global_coords": global_coords,
        "source_global_rows": source_rows, "block_index": index,
    })
    print(f"[input] Global_C256={len(coords):,} block={index} id={block_id} "
          f"local_C64={len(local_coords):,} placeholders={int((~inside).sum()):,}", flush=True)

    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=True)
    canonical = pipeline.preprocess_canonical_images(Image.open(args.image))
    image1024 = canonical["image_1024"]
    shape_cond = core.build_global_conditions(
        pipeline, image1024, camera, global_coords, [record], "shape", output, device)
    shape = core.synchronous_flow(
        stage="shape", pipeline=pipeline, records=[record], condition=shape_cond,
        output=output, device=device, seed=args.shape_seed, steps=args.steps, concat=None)
    del shape_cond
    empty_cuda()
    texture_cond = core.build_global_conditions(
        pipeline, image1024, camera, global_coords, [record], "texture", output, device)
    texture = core.synchronous_flow(
        stage="texture", pipeline=pipeline, records=[record], condition=texture_cond,
        output=output, device=device, seed=args.texture_seed, steps=args.steps, concat=shape)
    del texture_cond
    empty_cuda()

    shape_raw = cube_flow.denormalize(shape, pipeline.shape_slat_normalization)
    texture_raw = cube_flow.denormalize(texture, pipeline.tex_slat_normalization)
    decoded = pipeline.decode_latent(
        SparseTensor(shape_raw.to(device), local_coords.to(device)),
        SparseTensor(texture_raw.to(device), local_coords.to(device)), 1024)
    if len(decoded) != 1:
        raise RuntimeError("local decoder returned batch size other than one")
    native = decoded[0]
    cube_flow.atomic_save(output / "local_material_mesh.pt", {"format": FORMAT, "mesh": native.cpu()})
    local_vertex, local_face = expc._native_mesh_to_pbr(native, device)
    cube_flow.atomic_save(output / "local_per_vertex_pbr_mesh.pt", {"format": FORMAT, "mesh": local_vertex})
    cube_flow.atomic_save(output / "local_per_face_pbr_mesh.pt", {"format": FORMAT, "mesh": local_face})
    placed = composite.place_local_mesh(local_vertex, start, float(camera["mesh_scale"]))
    cube_flow.atomic_save(output / "local_mesh_in_global_coordinates.pt", {
        "format": FORMAT, "mesh": placed, "block_index": index})
    del decoded, native, local_face
    empty_cuda()

    keep = ~inside
    point_q = 2.0 * (xyz[keep].float() + 0.5) / 256.0 - 1.0
    points = point_q / (2.0 * float(camera["mesh_scale"]))
    angles = tuple(int(v) % 360 for v in args.angles.split(","))
    extrinsics, intrinsics, _ = _make_camera_views(
        camera["camera_angle_x"], camera["distance"], angles)
    render_dir = output / "global_c256_points_plus_local_c64_multiview_4096"
    render_dir.mkdir(parents=True, exist_ok=True)
    renderer = PbrMeshRenderer({
        "resolution": args.resolution, "near": 0.01,
        "far": camera["distance"] + 10, "ssaa": 1, "peel_layers": 8,
        "face_chunk_size": args.face_chunk_size,
    }, device=str(device))
    envmap = load_envmap("studio", device=device)
    live = placed.to(device)
    rgb_paths, normal_paths = [], []
    for angle in angles:
        print(f"[render] yaw={angle} points={len(points):,}", flush=True)
        cloud = composite.point_image(
            points, extrinsics[angle], intrinsics, args.resolution, args.point_radius)
        result = renderer.render(
            live, extrinsics[angle].to(device), intrinsics.to(device),
            envmap=envmap, use_envmap_bg=False)
        alpha = composite.tensor_image(result["mask"], "L")
        rgb_out = Image.composite(composite.tensor_image(result["shaded"]), cloud, alpha)
        normal_out = Image.composite(composite.tensor_image(result["normal"]), cloud, alpha)
        rgb_path = render_dir / f"view_{angle:03d}_pbr_plus_c256_points.png"
        normal_path = render_dir / f"view_{angle:03d}_camera_normal_plus_c256_points.png"
        rgb_out.save(rgb_path)
        normal_out.save(normal_path)
        rgb_paths.append((angle, rgb_path))
        normal_paths.append((angle, normal_path))
        del result, alpha, rgb_out, normal_out
    rgb_sheet = render_dir / "local_c64_pbr_plus_c256_points_contact_sheet.png"
    normal_sheet = render_dir / "local_c64_camera_normal_plus_c256_points_contact_sheet.png"
    composite.contact_sheet(rgb_paths, rgb_sheet, f"Global C256 points + local C64 {index} PBR")
    composite.contact_sheet(normal_paths, normal_sheet, f"Global C256 points + local C64 {index} camera normal")
    core.atomic_json(output / "summary.json", {
        "format": FORMAT, "status": "complete", "block_id": block_id,
        "block_index": index, "block_tokens": len(local_coords),
        "placeholder_points": len(points), "vertices": len(local_vertex.vertices),
        "faces": len(local_vertex.faces), "pbr_contact_sheet": str(rgb_sheet.resolve()),
        "camera_normal_contact_sheet": str(normal_sheet.resolve()),
        "seconds": time.perf_counter() - started,
    })
    print(f"[done] {rgb_sheet}", flush=True)


if __name__ == "__main__":
    main()
