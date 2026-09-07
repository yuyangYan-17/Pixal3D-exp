#!/usr/bin/env python3
"""Render full global SLAT points with selected local PBR cube decodes in place."""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image, ImageDraw

from pixal3d.representations import MeshWithVertexPbr
from pixal3d.renderers import PbrMeshRenderer
from pixal3d_baseline1024_pbr_mesh_compare import _make_camera_views
from render_pixal3d_raw_ovoxel import load_envmap


FORMAT = "pixal3d_global_slat_points_with_local_cubes_v1"


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n"); temporary = Path(f.name)
    os.replace(temporary, path)


def load_mesh(path: Path) -> MeshWithVertexPbr:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mesh = payload.get("mesh", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(mesh, MeshWithVertexPbr):
        raise TypeError(f"expected MeshWithVertexPbr: {path}")
    return mesh.cpu()


def cube_start(cube_id: int) -> tuple[int, int, int]:
    starts = tuple(range(0, 193, 32))
    ix, remainder = divmod(cube_id, 49)
    iy, iz = divmod(remainder, 7)
    return starts[ix], starts[iy], starts[iz]


def place_local_mesh(mesh: MeshWithVertexPbr, start: tuple[int, int, int], mesh_scale: float) -> MeshWithVertexPbr:
    # Local decoder vertices span [-.5,.5], i.e. q_local=[-1,1].  A C64 cube
    # occupies one quarter of C256: q_global = centre_q + q_local/4.
    centre_q = 2.0 * (torch.tensor(start, dtype=torch.float32) + 32.0) / 256.0 - 1.0
    q_local = mesh.vertices.float() * (2.0 * float(mesh_scale))
    vertices = (centre_q[None] + q_local / 4.0) / (2.0 * float(mesh_scale))
    return MeshWithVertexPbr(vertices, mesh.faces.int(), mesh.vertex_attrs.float(), layout=dict(mesh.layout))


def merge_meshes(meshes: list[MeshWithVertexPbr]) -> MeshWithVertexPbr:
    vertices, faces, attrs = [], [], []
    offset = 0
    for mesh in meshes:
        vertices.append(mesh.vertices); attrs.append(mesh.vertex_attrs)
        faces.append(mesh.faces.to(torch.int64) + offset)
        offset += int(mesh.vertices.shape[0])
    return MeshWithVertexPbr(torch.cat(vertices), torch.cat(faces).int(), torch.cat(attrs), layout=dict(meshes[0].layout))


def global_support_points(coords: torch.Tensor, cube_ids: tuple[int, ...], mesh_scale: float) -> torch.Tensor:
    xyz = coords[:, 1:].float()
    keep = torch.ones(xyz.shape[0], dtype=torch.bool)
    for cube_id in cube_ids:
        start = torch.tensor(cube_start(cube_id), dtype=torch.float32)
        keep &= ~(((xyz >= start) & (xyz < start + 64.0)).all(1))
    q = 2.0 * (xyz[keep] + 0.5) / 256.0 - 1.0
    return q / (2.0 * float(mesh_scale))


def project_points(points: torch.Tensor, extrinsic: torch.Tensor, intrinsic: torch.Tensor,
                   resolution: int) -> tuple[np.ndarray, np.ndarray]:
    p = torch.cat((points.float(), torch.ones((points.shape[0], 1))), 1)
    camera = p @ extrinsic.detach().cpu().float().T
    z = camera[:, 2]
    u = (intrinsic[0, 0].cpu() * camera[:, 0] / z + intrinsic[0, 2].cpu()) * resolution
    v = (intrinsic[1, 1].cpu() * camera[:, 1] / z + intrinsic[1, 2].cpu()) * resolution
    valid = torch.isfinite(u) & torch.isfinite(v) & (z > 0) & (u >= 0) & (u < resolution) & (v >= 0) & (v < resolution)
    uvz = torch.stack((u[valid], v[valid], z[valid]), 1)
    uvz = uvz[torch.argsort(uvz[:, 2], descending=True)]
    return uvz[:, :2].numpy(), uvz[:, 2].numpy()


def point_image(points: torch.Tensor, extrinsic: torch.Tensor, intrinsic: torch.Tensor,
                resolution: int, radius: int) -> Image.Image:
    uv, depth = project_points(points, extrinsic, intrinsic, resolution)
    image = Image.new("RGB", (resolution, resolution), (0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    if depth.size:
        near, far = float(depth.min()), float(depth.max())
        shade = 1.0 - (depth - near) / max(far - near, 1e-8)
        for (u, v), value in zip(uv, shade):
            c = int(round(115 + 120 * float(value)))
            draw.ellipse((u-radius, v-radius, u+radius, v+radius), fill=(c, c, c, 205))
    return image


def tensor_image(value: torch.Tensor, mode: str = "RGB") -> Image.Image:
    tensor = value.detach().float().cpu()
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3): tensor = tensor.permute(1, 2, 0)
    array = np.clip(np.nan_to_num(tensor.numpy()), 0.0, 1.0)
    if array.ndim == 2:
        return Image.fromarray((array * 255 + .5).astype(np.uint8), "L")
    if mode == "L" or array.shape[-1] == 1:
        return Image.fromarray((array[..., 0] * 255 + .5).astype(np.uint8), "L")
    return Image.fromarray((array * 255 + .5).astype(np.uint8), "RGB")


def contact_sheet(paths: list[tuple[int, Path]], output: Path, title: str) -> None:
    thumb = 768; margin = 20; label = 40
    sheet = Image.new("RGB", (len(paths)*(thumb+margin)+margin, thumb+label+2*margin), (16,16,16))
    draw = ImageDraw.Draw(sheet)
    for index, (angle, path) in enumerate(paths):
        with Image.open(path) as source: image = source.convert("RGB")
        x = margin + index*(thumb+margin); y = margin+label
        sheet.paste(image.resize((thumb,thumb), Image.Resampling.LANCZOS), (x,y))
        draw.text((x,margin), f"{title} · yaw {angle}°", fill="white")
    sheet.save(output)


def parse_args() -> argparse.Namespace:
    root = Path("outputs/global_c256_head_cubes_263_270_local_decode_cuda5")
    baseline = Path("outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=root)
    p.add_argument("--support", type=Path, default=Path("outputs/global_c256_cube_owner_flow_singleview_cuda4/support/global_c256_support.pt"))
    p.add_argument("--camera", type=Path, default=baseline / "global_camera.json")
    p.add_argument("--cube-ids", default="263,270")
    p.add_argument("--angles", default="0,60,120,180,240,300")
    p.add_argument("--resolution", type=int, default=4096)
    p.add_argument("--point-radius", type=int, default=3)
    p.add_argument("--face-chunk-size", type=int, default=1_000_000)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args(); started = time.perf_counter()
    cube_ids = tuple(int(v) for v in args.cube_ids.split(",")); angles = tuple(int(v) for v in args.angles.split(","))
    camera = json.loads(args.camera.read_text()); mesh_scale = float(camera.get("mesh_scale", 1.0))
    output = args.root.resolve() / "global_composite_multiview"; output.mkdir(parents=True, exist_ok=True)
    meshes = [place_local_mesh(load_mesh(args.root/f"cube_{cube_id:03d}"/"local_per_vertex_pbr_mesh.pt"),
                               cube_start(cube_id), mesh_scale) for cube_id in cube_ids]
    merged = merge_meshes(meshes)
    merged_path = output / "local_cubes_in_global_coordinates.pt"
    torch.save({"format": FORMAT, "mesh": merged}, merged_path)
    support = torch.load(args.support, map_location="cpu", weights_only=False)["coords"]
    points = global_support_points(support, cube_ids, mesh_scale)
    extrinsics, intrinsics, _ = _make_camera_views(camera["camera_angle_x"], camera["distance"], angles)
    device = torch.device(args.device)
    renderer = PbrMeshRenderer({"resolution": args.resolution, "near": .01,
                                "far": camera["distance"]+10, "ssaa": 1, "peel_layers": 8,
                                "face_chunk_size": args.face_chunk_size}, device=str(device))
    envmap = load_envmap("studio", device=device); live = merged.to(device)
    rgb_view_paths: list[tuple[int, Path]] = []
    normal_view_paths: list[tuple[int, Path]] = []
    for angle in angles:
        print(f"[global-composite] yaw={angle} points={points.shape[0]:,}", flush=True)
        cloud = point_image(points, extrinsics[angle], intrinsics, args.resolution, args.point_radius)
        rendered = renderer.render(live, extrinsics[angle].to(device), intrinsics.to(device),
                                   envmap=envmap, use_envmap_bg=False)
        rgb = tensor_image(rendered["shaded"]); alpha = tensor_image(rendered["mask"], "L")
        composite = Image.composite(rgb, cloud, alpha)
        path = output / f"view_{angle:03d}_global_slat_plus_local_pbr.png"; composite.save(path)
        camera_normal = tensor_image(rendered["normal"])
        normal_composite = Image.composite(camera_normal, cloud, alpha)
        normal_path = output / f"view_{angle:03d}_global_slat_plus_local_camera_normal.png"
        normal_composite.save(normal_path)
        rgb.save(output / f"view_{angle:03d}_local_pbr_only.png")
        camera_normal.save(output / f"view_{angle:03d}_local_camera_normal_only.png")
        cloud.save(output / f"view_{angle:03d}_slat_points_only.png")
        rgb_view_paths.append((angle,path)); normal_view_paths.append((angle,normal_path)); del rendered
    sheet = output / "global_slat_plus_local_pbr_contact_sheet.png"
    normal_sheet = output / "global_slat_plus_local_camera_normal_contact_sheet.png"
    contact_sheet(rgb_view_paths, sheet, "Full SLAT + local PBR cubes")
    contact_sheet(normal_view_paths, normal_sheet, "Full SLAT + local camera normals")
    atomic_json(output/"manifest.json", {"format":FORMAT,"status":"complete","cube_ids":list(cube_ids),
        "cube_starts_c256":[list(cube_start(v)) for v in cube_ids],"angles":list(angles),
        "resolution":args.resolution,"remaining_slat_points":int(points.shape[0]),
        "local_vertices":int(merged.vertices.shape[0]),"local_faces":int(merged.faces.shape[0]),
        "coordinate_rule":"q_global = cube_center_q + q_local/4; native world y is not flipped",
        "pbr_contact_sheet":str(sheet.resolve()),
        "camera_normal_contact_sheet":str(normal_sheet.resolve()),"seconds":time.perf_counter()-started})
    print(f"[done] {sheet}", flush=True)


if __name__ == "__main__": main()
