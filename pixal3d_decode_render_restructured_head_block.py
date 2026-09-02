#!/usr/bin/env python3
"""Decode one restructured head C64 block; render all other Global C256 rows as points."""
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

import pixal3d_global_c256_cube_owner_flow_singleview as cube_flow
import pixal3d_global4096_singleview_shared_slat_shape_tex_sr as expc
import pixal3d_render_global_slat_points_with_local_cubes as composite
from inference import MODEL_PATH, init_pipeline
from pixal3d.modules.sparse import SparseTensor
from pixal3d.renderers import PbrMeshRenderer
from pixal3d_baseline1024_pbr_mesh_compare import _make_camera_views
from render_pixal3d_raw_ovoxel import load_envmap


FORMAT = "pixal3d_restructured_single_head_block_v1"


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    source = Path("outputs/global_c256_restructured_blocks_cuda5")
    baseline = Path("outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=source)
    p.add_argument("--support", type=Path, default=source / "global_support/global_c256_support.pt")
    p.add_argument("--shape", type=Path, default=source / "shape/final_normalized.pt")
    p.add_argument("--texture", type=Path, default=source / "texture/final_normalized.pt")
    p.add_argument("--camera", type=Path, default=baseline / "global_camera.json")
    p.add_argument("--output", type=Path, default=Path("outputs/global_c256_restructured_head_block38_cuda5"))
    p.add_argument("--block-id", type=int, default=38)
    p.add_argument("--block-index", default="2,1,2")
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--physical-cuda", type=int, default=5)
    p.add_argument("--angles", default="0,60,120,180,240,300")
    p.add_argument("--resolution", type=int, default=4096)
    p.add_argument("--point-radius", type=int, default=3)
    p.add_argument("--face-chunk-size", type=int, default=1_000_000)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args(); started = time.perf_counter()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.physical_cuda):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}, expected {args.physical_cuda}")
    device = torch.device(args.device); torch.cuda.set_device(device)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    index = tuple(int(v) for v in args.block_index.split(","))
    if len(index) != 3 or args.block_id != index[0] * 16 + index[1] * 4 + index[2]:
        raise ValueError("block id/index mismatch")
    start = tuple(v * 64 for v in index)
    camera = json.loads(args.camera.read_text()); mesh_scale = float(camera.get("mesh_scale", 1.0))
    coords = torch.load(args.support, map_location="cpu", weights_only=False)["coords"].int()
    xyz = coords[:, 1:4]; start_t = torch.tensor(start, dtype=torch.int32)
    inside = ((xyz >= start_t) & (xyz < start_t + 64)).all(1)
    rows = torch.where(inside)[0].long()
    local_xyz = xyz[inside] - start_t
    local_coords = torch.cat((torch.zeros((len(rows), 1), dtype=torch.int32), local_xyz), 1)
    shape_global = torch.load(args.shape, map_location="cpu", weights_only=False)["features"].float()
    tex_global = torch.load(args.texture, map_location="cpu", weights_only=False)["features"].float()
    shape = shape_global.index_select(0, rows); texture = tex_global.index_select(0, rows)
    cube_flow.atomic_save(output / "block38_normalized_latents.pt", {
        "format": FORMAT, "block_id": args.block_id, "block_index": index,
        "start_c256": start, "global_row_ids": rows, "local_coords": local_coords,
        "shape": shape, "texture": texture,
    })
    vertex_path = output / "block38_local_per_vertex_pbr_mesh.pt"
    if vertex_path.is_file():
        local_vertex = composite.load_mesh(vertex_path)
    else:
        pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=True)
        shape_raw = cube_flow.denormalize(shape, pipeline.shape_slat_normalization)
        tex_raw = cube_flow.denormalize(texture, pipeline.tex_slat_normalization)
        decoded = pipeline.decode_latent(
            SparseTensor(shape_raw.to(device), local_coords.to(device)),
            SparseTensor(tex_raw.to(device), local_coords.to(device)), 1024,
        )
        if len(decoded) != 1: raise RuntimeError("local decoder returned batch != 1")
        native = decoded[0]
        cube_flow.atomic_save(output / "block38_local_material_mesh.pt", {"format": FORMAT, "mesh": native.cpu()})
        local_vertex, local_face = expc._native_mesh_to_pbr(native, device)
        cube_flow.atomic_save(vertex_path, {"format": FORMAT, "mesh": local_vertex})
        cube_flow.atomic_save(output / "block38_local_per_face_pbr_mesh.pt", {"format": FORMAT, "mesh": local_face})
        del pipeline, decoded, native, local_face
        empty_cuda()

    placed = composite.place_local_mesh(local_vertex, start, mesh_scale)
    cube_flow.atomic_save(output / "block38_in_global_coordinates.pt", {"format": FORMAT, "mesh": placed})
    keep = ~inside
    point_q = 2.0 * (xyz[keep].float() + 0.5) / 256.0 - 1.0
    points = point_q / (2.0 * mesh_scale)
    angles = tuple(int(v) for v in args.angles.split(","))
    extrinsics, intrinsics, _ = _make_camera_views(camera["camera_angle_x"], camera["distance"], angles)
    render_dir = output / "global_slat_plus_head_block_multiview_4096"; render_dir.mkdir(parents=True, exist_ok=True)
    renderer = PbrMeshRenderer({"resolution": args.resolution, "near": .01,
        "far": camera["distance"] + 10, "ssaa": 1, "peel_layers": 8,
        "face_chunk_size": args.face_chunk_size}, device=str(device))
    envmap = load_envmap("studio", device=device); live = placed.to(device)
    rgb_paths, normal_paths = [], []
    for angle in angles:
        print(f"[render] block={args.block_id} yaw={angle} points={len(points):,}", flush=True)
        cloud = composite.point_image(points, extrinsics[angle], intrinsics, args.resolution, args.point_radius)
        result = renderer.render(live, extrinsics[angle].to(device), intrinsics.to(device),
                                 envmap=envmap, use_envmap_bg=False)
        alpha = composite.tensor_image(result["mask"], "L")
        rgb = composite.tensor_image(result["shaded"])
        normal = composite.tensor_image(result["normal"])
        rgb_out = Image.composite(rgb, cloud, alpha)
        normal_out = Image.composite(normal, cloud, alpha)
        rgb_path = render_dir / f"view_{angle:03d}_slat_plus_block38_pbr.png"
        normal_path = render_dir / f"view_{angle:03d}_slat_plus_block38_camera_normal.png"
        rgb_out.save(rgb_path); normal_out.save(normal_path)
        rgb_paths.append((angle, rgb_path)); normal_paths.append((angle, normal_path))
        del result
    rgb_sheet = render_dir / "slat_plus_block38_pbr_contact_sheet.png"
    normal_sheet = render_dir / "slat_plus_block38_camera_normal_contact_sheet.png"
    composite.contact_sheet(rgb_paths, rgb_sheet, "Global C256 SLAT + head block 38 PBR")
    composite.contact_sheet(normal_paths, normal_sheet, "Global C256 SLAT + head block 38 camera normal")
    cube_flow.atomic_json(output / "summary.json", {
        "format": FORMAT, "status": "complete", "block_id": args.block_id,
        "block_index": index, "start_c256": start, "end_exclusive_c256": [v + 64 for v in start],
        "block_tokens": int(len(rows)), "placeholder_slat_points": int(keep.sum()),
        "local_vertices": int(local_vertex.vertices.shape[0]), "local_faces": int(local_vertex.faces.shape[0]),
        "resolution": args.resolution, "angles": angles,
        "pbr_contact_sheet": str(rgb_sheet.resolve()),
        "camera_normal_contact_sheet": str(normal_sheet.resolve()),
        "seconds": time.perf_counter() - started,
    })
    print(f"[done] {rgb_sheet}", flush=True)
if __name__ == "__main__":
    main()
