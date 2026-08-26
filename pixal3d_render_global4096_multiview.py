#!/usr/bin/env python3
"""Render the completed Codex2 global 4096 PBR mesh from multiple views.

The mesh stays in the native ``MeshWithVertexPbr`` representation and is
rendered by Pixal3D's native PBR renderer.  No GLB export, UV unwrap, or
external renderer is involved.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import numpy as np
import torch
from PIL import Image

from pixal3d.representations import MeshWithVertexPbr
from pixal3d.renderers import PbrMeshRenderer


DEFAULT_ROOT = Path(
    "/home/nvme04/yyyan/Pixal3D/outputs/"
    "global4096_tile_x0_consensus_sync_cuda5"
)
DEFAULT_MESH = DEFAULT_ROOT / "final" / "final_per_vertex_pbr_mesh.pt"
DEFAULT_CAMERA = DEFAULT_ROOT / "global_camera.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "multiview_4096"
DEFAULT_ANGLES = (0, 120, 240)
DEFAULT_FACE_CHUNK_SIZE = 4_000_000


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _save_image(value: torch.Tensor, path: Path, channels: int | None = None) -> None:
    tensor = value.detach().float().cpu()
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
        tensor = tensor.permute(1, 2, 0)
    if tensor.ndim == 2:
        tensor = tensor[..., None]
    if tensor.ndim != 3:
        raise ValueError(f"unexpected render tensor shape: {tuple(tensor.shape)}")
    array = tensor.numpy()
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    array = np.clip(array, 0.0, 1.0)
    if channels == 1 or array.shape[-1] == 1:
        image = Image.fromarray(
            (array[..., 0] * 255.0 + 0.5).astype(np.uint8), mode="L"
        )
    elif array.shape[-1] == 3:
        image = Image.fromarray(
            (array * 255.0 + 0.5).astype(np.uint8), mode="RGB"
        )
    else:
        raise ValueError(f"cannot save {array.shape[-1]} channels as an image")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _parse_angles(value: str) -> tuple[int, ...]:
    angles = tuple(int(part.strip()) % 360 for part in value.split(",") if part.strip())
    if not angles:
        raise ValueError("--angles must contain at least one angle")
    if len(set(angles)) != len(angles):
        raise ValueError(f"duplicate angles are not allowed: {angles}")
    return angles


def _load_camera(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "camera" in payload and isinstance(payload["camera"], Mapping):
        payload = payload["camera"]
    required = ("camera_angle_x", "distance")
    if any(key not in payload for key in required):
        raise KeyError(f"camera file lacks {required}: {path}")
    return {
        "camera_angle_x": float(payload["camera_angle_x"]),
        "distance": float(payload["distance"]),
        "mesh_scale": float(payload.get("mesh_scale", 1.0)),
    }


def _load_vertex_mesh(path: Path) -> MeshWithVertexPbr:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mesh = payload.get("mesh", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(mesh, MeshWithVertexPbr):
        raise TypeError(f"expected MeshWithVertexPbr in {path}, got {type(mesh)!r}")
    if mesh.vertices.ndim != 2 or mesh.faces.ndim != 2 or mesh.vertex_attrs.ndim != 2:
        raise ValueError("mesh tensors have unexpected ranks")
    if mesh.faces.shape[-1] != 3 or mesh.vertex_attrs.shape[-1] != 6:
        raise ValueError("mesh does not have native triangle/6-channel PBR layout")
    return mesh.cpu()


def _runtime(device: torch.device) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        index = torch.cuda.current_device()
        result.update(
            {
                "torch_current_device": int(index),
                "gpu_name": torch.cuda.get_device_name(index),
                "free_memory_bytes": int(torch.cuda.mem_get_info(index)[0]),
                "total_memory_bytes": int(torch.cuda.mem_get_info(index)[1]),
            }
        )
    return result


def _output_names(output_dir: Path, angle: int) -> Sequence[Path]:
    prefix = output_dir / f"view_{angle:03d}"
    return (
        Path(f"{prefix}_render_rgb_4096.png"),
        Path(f"{prefix}_render_alpha_4096.png"),
        Path(f"{prefix}_render_normal_camera_4096.png"),
        Path(f"{prefix}_render_normal_world_4096.png"),
        Path(f"{prefix}_pbr_base_color_4096.png"),
        Path(f"{prefix}_pbr_metallic_4096.png"),
        Path(f"{prefix}_pbr_roughness_4096.png"),
        Path(f"{prefix}_pbr_alpha_4096.png"),
    )


def _make_contact_sheet(
    output_dir: Path,
    angles: Iterable[int],
    image_suffix: str = "render_rgb_4096.png",
    output_name: str = "multiview_rgb_contact_sheet.png",
    title: str = "RGB",
) -> Path:
    images = []
    for angle in angles:
        path = output_dir / f"view_{angle:03d}_{image_suffix}"
        with Image.open(path) as image:
            images.append((angle, image.convert("RGB")))
    thumb_size = 768
    margin = 24
    label_height = 48
    sheet = Image.new(
        "RGB",
        (len(images) * (thumb_size + margin) + margin, thumb_size + label_height + 2 * margin),
        (24, 24, 24),
    )
    from PIL import ImageDraw

    draw = ImageDraw.Draw(sheet)
    for index, (angle, image) in enumerate(images):
        image.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        x = margin + index * (thumb_size + margin)
        y = margin + label_height
        sheet.paste(image, (x, y))
        draw.text((x, margin), f"{title} · view {angle}°", fill=(255, 255, 255))
    path = output_dir / output_name
    sheet.save(path)
    return path


def render(args: argparse.Namespace) -> Dict[str, Any]:
    mesh_path = Path(args.mesh).resolve()
    camera_path = Path(args.camera).resolve()
    output_dir = Path(args.output_dir).resolve()
    angles = _parse_angles(args.angles)
    resolution = int(args.resolution)
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    if not camera_path.is_file():
        raise FileNotFoundError(camera_path)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    camera = _load_camera(camera_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "multiview_manifest.json"
    if manifest_path.is_file() and not args.force:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            saved.get("status") == "complete"
            and tuple(saved.get("angles_deg", ())) == angles
            and int(saved.get("resolution", -1)) == resolution
            and saved.get("mesh") == str(mesh_path)
            and all(path.is_file() for angle in angles for path in _output_names(output_dir, angle))
        ):
            print(f"[multiview] cache hit: {manifest_path}")
            return saved

    from render_pixal3d_raw_ovoxel import load_envmap

    started = time.time()
    mesh = _load_vertex_mesh(mesh_path)
    print(
        f"[multiview] mesh vertices={mesh.vertices.shape[0]:,} "
        f"faces={mesh.faces.shape[0]:,} device={device} angles={angles}"
    )
    # Use the exact yaw convention from the existing 1024 comparison driver.
    from pixal3d_baseline1024_pbr_mesh_compare import _make_camera_views

    view_extrinsics, intrinsics, _ = _make_camera_views(
        camera["camera_angle_x"], camera["distance"], angles
    )
    if intrinsics.shape != (3, 3):
        raise AssertionError(f"unexpected intrinsics shape: {tuple(intrinsics.shape)}")

    renderer = PbrMeshRenderer(
        rendering_options={
            "resolution": resolution,
            "near": max(0.01, camera["distance"] - 2.0),
            "far": camera["distance"] + 10.0,
            "ssaa": 1,
            "peel_layers": 8,
            "face_chunk_size": int(args.face_chunk_size),
        },
        device=str(device),
    )
    envmap = load_envmap("studio", device=device)
    live = mesh.to(device)
    manifest: Dict[str, Any] = {
        "format": "pixal3d_global4096_multiview_native_pbr_v1",
        "status": "in_progress",
        "mesh": str(mesh_path),
        "camera": str(camera_path),
        "camera_parameters": camera,
        "angles_deg": list(angles),
        "resolution": [resolution, resolution],
        "renderer": {
            "name": "pixal3d.renderers.PbrMeshRenderer",
            "face_chunk_size": int(args.face_chunk_size),
            "peel_layers": 8,
            "environment": "studio",
            "use_envmap_bg": False,
        },
        "runtime": _runtime(device),
        "views": [],
    }
    _atomic_json(manifest_path, manifest)

    try:
        with torch.inference_mode():
            for angle in angles:
                paths = _output_names(output_dir, angle)
                if all(path.is_file() for path in paths) and not args.force:
                    print(f"[multiview] view {angle}° cache hit")
                    manifest["views"].append(
                        {"angle_deg": angle, "status": "cached", "files": [str(p) for p in paths]}
                    )
                    continue
                view_started = time.time()
                print(f"[multiview] rendering view {angle}° at {resolution}x{resolution}")
                result = renderer.render(
                    live,
                    view_extrinsics[angle].to(device),
                    intrinsics.to(device),
                    envmap=envmap,
                    use_envmap_bg=False,
                )
                prefix = output_dir / f"view_{angle:03d}"
                _save_image(result["shaded"], Path(f"{prefix}_render_rgb_4096.png"))
                _save_image(result["mask"], Path(f"{prefix}_render_alpha_4096.png"), channels=1)
                _save_image(result["normal"], Path(f"{prefix}_render_normal_camera_4096.png"))

                encoded = result["normal"].detach().float().to(device)
                normal_cam = -(encoded * 2.0 - 1.0)
                rotation = view_extrinsics[angle].to(device)[:3, :3]
                normal_world = torch.matmul(
                    rotation.transpose(0, 1), normal_cam.reshape(3, -1)
                ).reshape_as(normal_cam)
                normal_world = -normal_world * 0.5 + 0.5
                _save_image(
                    normal_world,
                    Path(f"{prefix}_render_normal_world_4096.png"),
                )
                _save_image(
                    result["base_color"],
                    Path(f"{prefix}_pbr_base_color_4096.png"),
                )
                _save_image(
                    result["metallic"],
                    Path(f"{prefix}_pbr_metallic_4096.png"),
                    channels=1,
                )
                _save_image(
                    result["roughness"],
                    Path(f"{prefix}_pbr_roughness_4096.png"),
                    channels=1,
                )
                _save_image(
                    result["alpha"],
                    Path(f"{prefix}_pbr_alpha_4096.png"),
                    channels=1,
                )
                elapsed = time.time() - view_started
                manifest["views"].append(
                    {
                        "angle_deg": angle,
                        "status": "rendered",
                        "seconds": elapsed,
                        "files": [str(p) for p in paths],
                    }
                )
                _atomic_json(manifest_path, manifest)
                print(f"[multiview] view {angle}° complete in {elapsed:.1f}s")
                del result, encoded, normal_cam, normal_world
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
    finally:
        del live, envmap, renderer, mesh
        if device.type == "cuda":
            torch.cuda.empty_cache()

    contact_sheet = _make_contact_sheet(output_dir, angles)
    normal_camera_sheet = _make_contact_sheet(
        output_dir, angles,
        image_suffix="render_normal_camera_4096.png",
        output_name="multiview_normal_camera_contact_sheet.png",
        title="Camera normal",
    )
    normal_world_sheet = _make_contact_sheet(
        output_dir, angles,
        image_suffix="render_normal_world_4096.png",
        output_name="multiview_normal_world_contact_sheet.png",
        title="World normal",
    )
    manifest["contact_sheet"] = str(contact_sheet)
    manifest["normal_camera_contact_sheet"] = str(normal_camera_sheet)
    manifest["normal_world_contact_sheet"] = str(normal_world_sheet)
    manifest["status"] = "complete"
    manifest["seconds"] = time.time() - started
    _atomic_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--angles", default=",".join(map(str, DEFAULT_ANGLES)))
    parser.add_argument("--resolution", type=int, default=4096)
    parser.add_argument("--face-chunk-size", type=int, default=DEFAULT_FACE_CHUNK_SIZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    render(build_parser().parse_args())
