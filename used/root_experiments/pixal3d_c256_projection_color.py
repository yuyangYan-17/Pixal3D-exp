#!/usr/bin/env python3
"""Project Global C256 voxel centres to the reference image and export a viewer bundle."""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as camera_core


ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = ROOT / "outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024"
DEFAULT_SUPPORT = ROOT / "outputs/global_c256_cube_owner_flow_singleview_cuda4/support/global_c256_support.pt"
DEFAULT_OUTPUT = ROOT / "outputs/global_c256_projection_color_cuda5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_EXPERIMENT / "inputs/canonical_foreground_rgb_4096.png")
    parser.add_argument("--camera", type=Path, default=DEFAULT_EXPERIMENT / "global_camera.json")
    parser.add_argument("--viewer-template", type=Path, default=ROOT / "pixal3d_c256_projection_viewer.html")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def write_binary_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, uv: np.ndarray,
                     depth: np.ndarray, in_frame: np.ndarray) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment Global C256 voxel centres colored by the projected reference image\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property float image_u\nproperty float image_v\nproperty float depth\n"
        "property uchar in_frame\nend_header\n"
    ).encode("ascii")
    record = np.empty(len(xyz), dtype=np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
        ("u", "<f4"), ("v", "<f4"), ("depth", "<f4"), ("in_frame", "u1"),
    ]))
    record["x"], record["y"], record["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    record["r"], record["g"], record["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    record["u"], record["v"], record["depth"] = uv[:, 0], uv[:, 1], depth
    record["in_frame"] = in_frame.astype(np.uint8)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(record.tobytes())


def make_overlay(reference: Image.Image, uv: np.ndarray, in_frame: np.ndarray, output: Path) -> None:
    preview = reference.copy()
    preview.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    sx, sy = preview.width / reference.width, preview.height / reference.height
    draw = ImageDraw.Draw(preview, "RGBA")
    valid_uv = uv[in_frame]
    # The support is dense enough that a deterministic stride is clearer than drawing 240k opaque dots.
    stride = max(1, len(valid_uv) // 60_000)
    for u, v in valid_uv[::stride]:
        x, y = float(u * sx), float(v * sy)
        draw.ellipse((x - 0.7, y - 0.7, x + 0.7, y + 0.7), fill=(0, 255, 255, 105))
    preview.save(output)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    for path in (args.support, args.reference, args.camera, args.viewer_template):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic run")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    support_payload = torch.load(args.support, map_location="cpu", weights_only=False)
    coords = support_payload["coords"].to(torch.int32).contiguous()
    if coords.ndim != 2 or coords.shape[1] != 4 or torch.any(coords[:, 0] != 0):
        raise RuntimeError("expected a single-batch Global C256 [N,4] support")
    xyz256 = coords[:, 1:4]
    if torch.any(xyz256 < 0) or torch.any(xyz256 >= 256):
        raise RuntimeError("support contains coordinates outside C256")

    camera = json.loads(args.camera.read_text(encoding="utf-8"))
    reference = Image.open(args.reference).convert("RGB")
    width, height = reference.size
    q = (2.0 * (xyz256.to(device=device, dtype=torch.float64) + 0.5) / 256.0 - 1.0)
    uv, depth, finite = camera_core._project_global_q_to_image(
        q, global_camera=camera, image_width=width, image_height=height)
    in_frame = finite & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    # In actual Blender coordinates the camera is on -Y.  R^T maps the world
    # point (0,-1,0) back to raw q=(0,0,2*mesh_scale), which must hit the
    # principal point exactly.
    mesh_scale = float(camera.get("mesh_scale", 1.0))
    ray_q = torch.tensor([[0.0, 0.0, 2.0 * mesh_scale]], dtype=torch.float64, device=device)
    ray_uv, ray_depth, ray_finite = camera_core._project_global_q_to_image(
        ray_q, global_camera=camera, image_width=width, image_height=height)
    image_center = ray_uv.new_tensor([[width / 2.0, height / 2.0]])
    if not bool(ray_finite.all()) or not torch.allclose(ray_uv, image_center, atol=1e-8, rtol=0):
        raise RuntimeError(f"(0,-1,0) optical-axis check failed: uv={ray_uv.tolist()}")

    image_np = np.asarray(reference, dtype=np.uint8).copy()
    image_tensor = torch.from_numpy(image_np).to(device=device, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
    scale = uv.new_tensor([float(width), float(height)])
    grid = ((uv + 0.5) / scale * 2.0 - 1.0).to(torch.float32).reshape(1, -1, 1, 2)
    sampled = F.grid_sample(image_tensor, grid, mode="bilinear", padding_mode="border", align_corners=False)
    rgb = sampled[0, :, :, 0].T.clamp(0, 1)

    # Actual Blender/generation space used by ProjGrid.  The projection helper
    # works in the algebraically simplified camera coordinates, but the viewer
    # must show the preceding Blender alignment: (x,y,z) -> (x,-z,y).
    world_xyz = torch.stack((q[:, 0], -q[:, 2], q[:, 1]), dim=1) / (2.0 * mesh_scale)
    xyz_np = world_xyz.float().cpu().numpy()
    rgb_np = torch.round(rgb * 255.0).to(torch.uint8).cpu().numpy()
    uv_np = uv.float().cpu().numpy()
    depth_np = depth.float().cpu().numpy()
    in_frame_np = in_frame.cpu().numpy()

    write_binary_ply(output / "colored_c256_points.ply", xyz_np, rgb_np, uv_np, depth_np, in_frame_np)
    torch.save({
        "format": "pixal3d_c256_projection_color_v1",
        "coords_c256": coords,
        "q_global": q.float().cpu(),
        "world_xyz": world_xyz.float().cpu(),
        "image_uv": uv.float().cpu(),
        "depth": depth.float().cpu(),
        "in_frame": in_frame.cpu(),
        "rgb_u8": torch.from_numpy(rgb_np),
        "sampling": "bilinear grid_sample, align_corners=False, padding_mode=border",
    }, output / "projection_data.pt")

    shutil.copy2(args.reference, output / "reference_4096.png")
    shutil.copy2(args.camera, output / "camera.json")
    make_overlay(reference, uv_np, in_frame_np, output / "projection_overlay.png")

    # Compact offline payload: uint8 C256 xyz followed by uint8 RGB per point.
    packed = np.concatenate((xyz256.cpu().numpy().astype(np.uint8), rgb_np), axis=1).tobytes()
    texture_image = reference.resize((1024, 1024), Image.Resampling.LANCZOS).convert("RGBA")
    texture_rgba = np.asarray(texture_image, dtype=np.uint8)
    # WebGL typed-array rows start at texture v=0. Store the bottom source row
    # first so the texture is upright without UNPACK_FLIP_Y_WEBGL or an image element.
    texture_rgba_bottom_up = np.ascontiguousarray(texture_rgba[::-1])
    preview_buffer = io.BytesIO()
    texture_image.convert("RGB").save(preview_buffer, format="PNG", optimize=True)
    payload = {
        "format": "pixal3d_c256_projection_viewer_v1",
        "count": int(len(coords)),
        "gridResolution": 256,
        "camera": camera,
        "optical_axis_ray_check": {
            "world_point": [0.0, -1.0, 0.0],
            "image_uv": ray_uv[0].detach().cpu().tolist(),
            "image_pixel_description": "4096 image centre",
            "depth": float(ray_depth[0]),
            "passed": True,
        },
        "reference": "reference_4096.png",
        "referenceDataUrl": "data:image/png;base64," + base64.b64encode(preview_buffer.getvalue()).decode("ascii"),
        "referenceTextureWidth": texture_image.width,
        "referenceTextureHeight": texture_image.height,
        "referenceTextureFormat": "RGBA8_BOTTOM_UP",
        "referenceTextureBase64": base64.b64encode(texture_rgba_bottom_up.tobytes()).decode("ascii"),
        "pointRecord": "uint8 x,y,z,r,g,b",
        "pointBase64": base64.b64encode(packed).decode("ascii"),
    }
    payload_script = "window.PIXAL3D_C256_DATA=" + json.dumps(payload, separators=(",", ":")) + ";\n"
    (output / "pointcloud_data.js").write_text(payload_script, encoding="utf-8")
    viewer_text = args.viewer_template.read_text(encoding="utf-8")
    marker = '<script src="pointcloud_data.js"></script>'
    if viewer_text.count(marker) != 1:
        raise RuntimeError("viewer template data-script marker is missing or ambiguous")
    # A single-file viewer avoids every Windows file:// unique-origin path.
    (output / "viewer.html").write_text(
        viewer_text.replace(marker, "<script>" + payload_script + "</script>"), encoding="utf-8")

    fov = float(camera["camera_angle_x"])
    distance = float(camera["distance"])
    # Place the display image close to the front of the generation cube.  Any
    # position along the same frustum is projectively equivalent; this one is
    # large enough to inspect clearly in the overview.
    plane_y = -0.5 / float(camera.get("mesh_scale", 1.0)) - 0.15 / float(camera.get("mesh_scale", 1.0))
    plane_distance = distance + plane_y
    plane_half = plane_distance * math.tan(fov / 2.0)
    metadata = {
        "format": "pixal3d_c256_projection_color_v1",
        "source_support": str(args.support.resolve()),
        "source_reference": str(args.reference.resolve()),
        "source_camera": str(args.camera.resolve()),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "points": int(len(coords)),
        "in_frame_points": int(in_frame.sum()),
        "out_of_frame_points": int((~in_frame).sum()),
        "image_size": [width, height],
        "camera": camera,
        "optical_axis_ray_check": payload["optical_axis_ray_check"],
        "coordinate_convention": {
            "c256_center_to_q": "q = 2 * (xyz + 0.5) / 256 - 1",
            "viewer_world": "world_blender = (q.x, -q.z, q.y) / (2 * mesh_scale)",
            "viewer_camera_position": [0.0, -distance, 0.0],
            "viewer_camera_target": [0.0, 0.0, 0.0],
            "viewer_camera_up": [0.0, 0.0, 1.0],
            "optical_axis": "+Y",
            "ray_check": "world point (0,-1,0) projects to image centre (W/2,H/2)",
        },
        "sampling": "RGB bilinear grid_sample with (uv+0.5) pixel-centre normalization, align_corners=False, border padding",
        "image_plane": {
            "display_only": True,
            "distance_in_front_of_camera": plane_distance,
            "y": plane_y,
            "half_extent_xz": plane_half,
            "default_opacity": 0.42,
            "texture": "reference_4096.png",
        },
        "y0_plane": {
            "display_only": True,
            "corners": [[-0.5 / mesh_scale, 0, -0.5 / mesh_scale], [0.5 / mesh_scale, 0, 0.5 / mesh_scale]],
            "default_opacity": 0.34,
            "texture": "reference_4096.png",
        },
    }
    (output / "metadata.json").write_text(json.dumps(jsonable(metadata), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.txt").write_text(
        "双击 viewer.html 即可离线查看。\n"
        "左键旋转，中键/Shift+左键平移，滚轮缩放，双击复位。\n"
        "图像 X/Y 滑条选择目标像素；彩色 ray 会吸附到最近的真实 C256 投影点。\n"
        "相机成像平面、y=0 约束平面都使用 reference_4096.png，可分别调透明度。\n"
        "viewer.html 是单文件；WebGL 纹理由内嵌 RGBA Uint8Array 直接上传，不使用本地图片 URL。\n"
        "colored_c256_points.ply 是带 RGB、投影 uv、depth、in_frame 属性的二进制 PLY。\n"
        "projection_data.pt 保存完整数值诊断；metadata.json 保存本次相机和坐标约定。\n",
        encoding="utf-8",
    )
    print(json.dumps(jsonable(metadata), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
