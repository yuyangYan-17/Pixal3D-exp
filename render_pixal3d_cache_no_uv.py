#!/usr/bin/env python3
"""Render and evaluate a Pixal3D cache in Blender without UV or GLB.

The script reuses the raw decoder correspondence
``vertices[i] <-> attr_volume[i]``. It applies the exact coordinate conversion
needed to reproduce the original GLB evaluator's aligned camera view:

    (x, y, z)_decoder -> (x, -z, y)_Blender

It then renders one or more light rigs, composites the transparent render onto
black, saves original/render/comparison images, and computes PSNR, SSIM, LPIPS.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import shlex
import shutil
import subprocess
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CACHE_VERSION = 1
PACKAGE_VERSION = 2
LIGHT_MODES = ("studio", "three_point", "softbox", "front", "uniform", "dramatic")

# Exact raw decoder -> aligned Blender basis conversion.
DECODER_TO_ALIGNED_BLENDER = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

BLENDER_HELPER_SOURCE = r'''
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

DECODER_TO_ALIGNED_BLENDER = Matrix((
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
))


def log(message):
    print("[no-uv-aligned-blender] %s" % message, flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--jobs-json", type=Path, required=True)
    parser.add_argument("--engine", choices=("cycles", "eevee"), default="cycles")
    parser.add_argument("--render-resolution", type=int, default=2048)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--material-mode", choices=("pbr", "basecolor"), default="pbr")
    parser.add_argument("--base-color-space", choices=("srgb", "linear"), default="srgb")
    parser.add_argument("--flat-shading", action="store_true")
    parser.add_argument("--use-alpha", action="store_true")
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights, bpy.data.images):
        for item in list(collection):
            try:
                collection.remove(item)
            except Exception:
                pass


def clear_lights():
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    for light in list(bpy.data.lights):
        if light.users == 0:
            try:
                bpy.data.lights.remove(light)
            except Exception:
                pass


def configure_cycles_gpu(scene):
    addon = bpy.context.preferences.addons.get("cycles")
    if addon is None:
        raise RuntimeError("Cycles addon is unavailable")
    preferences = addon.preferences
    errors = []
    selected_backend = None
    selected_device = None
    for backend in ("CUDA", "OPTIX"):
        try:
            preferences.compute_device_type = backend
            preferences.get_devices()
            devices = list(preferences.devices)
            candidates = [device for device in devices if device.type == backend]
            if not candidates:
                continue
            for device in devices:
                device.use = False
            selected_device = candidates[0]
            selected_device.use = True
            selected_backend = backend
            break
        except Exception as exc:
            errors.append("%s: %s" % (backend, exc))
    if selected_backend is None:
        raise RuntimeError("No usable CUDA/OptiX device: " + "; ".join(errors))
    scene.cycles.device = "GPU"
    log("cycles backend=%s device=%s" % (selected_backend, selected_device.name))


def configure_scene(args):
    scene = bpy.context.scene
    if args.engine == "cycles":
        scene.render.engine = "CYCLES"
        configure_cycles_gpu(scene)
        scene.cycles.samples = int(args.samples)
        scene.cycles.use_denoising = True
    else:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        try:
            scene.eevee.taa_render_samples = int(args.samples)
        except Exception:
            pass
    scene.render.resolution_x = int(args.render_resolution)
    scene.render.resolution_y = int(args.render_resolution)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.render.use_file_extension = True
    if hasattr(scene.render, "use_persistent_data"):
        scene.render.use_persistent_data = True
    try:
        scene.display_settings.display_device = "sRGB"
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    return scene


def set_world(scene, strength, color=(1.0, 1.0, 1.0, 1.0)):
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = color
        background.inputs["Strength"].default_value = float(strength)


def add_area(scene, name, location, energy, size, color=(1.0, 1.0, 1.0)):
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = float(energy)
    data.shape = "DISK"
    data.size = float(size)
    data.color = color
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = (-obj.location).to_track_quat("-Z", "Y").to_euler()
    return obj


def add_light_rig(scene, mode):
    if mode == "studio":
        set_world(scene, 0.22)
        add_area(scene, "Key", (-3.5, -4.5, 4.5), 850.0, 4.5)
        add_area(scene, "Fill", (4.0, -2.5, 2.0), 430.0, 5.0)
        add_area(scene, "Top", (0.0, 0.5, 5.0), 300.0, 4.0)
        add_area(scene, "Rim", (0.5, 4.0, 3.0), 300.0, 3.0)
    elif mode == "three_point":
        set_world(scene, 0.12)
        add_area(scene, "Key", (-3.0, -4.0, 3.5), 1000.0, 3.0)
        add_area(scene, "Fill", (3.5, -2.5, 1.5), 350.0, 4.0)
        add_area(scene, "Rim", (1.0, 4.0, 3.0), 650.0, 2.5)
    elif mode == "softbox":
        set_world(scene, 0.30)
        add_area(scene, "SoftboxLeft", (-3.5, -3.5, 3.0), 650.0, 6.0)
        add_area(scene, "SoftboxRight", (3.5, -3.0, 2.5), 600.0, 6.0)
        add_area(scene, "SoftboxTop", (0.0, 0.0, 5.0), 250.0, 5.0)
    elif mode == "front":
        set_world(scene, 0.15)
        add_area(scene, "Front", (0.0, -4.5, 0.8), 1000.0, 5.0)
        add_area(scene, "FrontTop", (0.0, -2.5, 4.0), 250.0, 4.0)
    elif mode == "uniform":
        set_world(scene, 0.35)
        positions = ((0.0, -4.0, 0.0), (0.0, 4.0, 0.0), (-4.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 0.0, 4.0), (0.0, 0.0, -4.0))
        for index, position in enumerate(positions):
            add_area(scene, "Uniform%02d" % index, position, 230.0, 4.5)
    elif mode == "dramatic":
        set_world(scene, 0.035)
        add_area(scene, "HardKey", (-3.0, -3.5, 4.5), 1250.0, 1.5)
        add_area(scene, "CoolRim", (2.5, 4.0, 3.5), 900.0, 2.0, (0.65, 0.75, 1.0))
        add_area(scene, "WarmFill", (3.5, -1.5, 0.5), 120.0, 3.0, (1.0, 0.72, 0.55))
    else:
        raise ValueError("unsupported light mode: %s" % mode)


def create_aligned_camera(scene, distance, fov_rad):
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.matrix_world = Matrix((
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, -float(distance)),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ))
    camera_data.type = "PERSP"
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = 32.0
    camera_data.sensor_height = 32.0
    camera_data.lens = 16.0 / math.tan(float(fov_rad) / 2.0)
    camera_data.clip_start = 0.01
    camera_data.clip_end = 100.0
    scene.camera = camera
    return camera


def srgb_to_linear(values):
    values = np.asarray(values, dtype=np.float32)
    low = values <= 0.04045
    output = np.empty_like(values)
    output[low] = values[low] / 12.92
    output[~low] = ((values[~low] + 0.055) / 1.055) ** 2.4
    return output


def make_material(mode, use_alpha):
    material = bpy.data.materials.new("Pixal3D_NoUV_PBR")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    color_node = nodes.new("ShaderNodeVertexColor")
    color_node.layer_name = "pixal_base_color"
    if mode == "basecolor":
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Strength"].default_value = 1.0
        links.new(color_node.outputs["Color"], emission.inputs["Color"])
        links.new(emission.outputs["Emission"], output.inputs["Surface"])
        return material
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    metallic_node = nodes.new("ShaderNodeAttribute")
    metallic_node.attribute_name = "pixal_metallic"
    roughness_node = nodes.new("ShaderNodeAttribute")
    roughness_node.attribute_name = "pixal_roughness"
    links.new(color_node.outputs["Color"], principled.inputs["Base Color"])
    links.new(metallic_node.outputs["Fac"], principled.inputs["Metallic"])
    links.new(roughness_node.outputs["Fac"], principled.inputs["Roughness"])
    if use_alpha:
        links.new(color_node.outputs["Alpha"], principled.inputs["Alpha"])
        try:
            material.surface_render_method = "DITHERED"
        except Exception:
            try:
                material.blend_method = "BLEND"
            except Exception:
                pass
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material


def create_mesh_object(name, vertices, faces, attrs, material, base_color_space, flat_shading):
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    attrs = np.asarray(attrs, dtype=np.float32)
    num_vertices = int(vertices.shape[0])
    num_faces = int(faces.shape[0])
    num_loops = num_faces * 3
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.vertices.add(num_vertices)
    mesh.loops.add(num_loops)
    mesh.polygons.add(num_faces)
    mesh.vertices.foreach_set("co", vertices.reshape(-1))
    mesh.loops.foreach_set("vertex_index", faces.reshape(-1))
    mesh.polygons.foreach_set("loop_start", np.arange(0, num_loops, 3, dtype=np.int32))
    mesh.polygons.foreach_set("loop_total", np.full(num_faces, 3, dtype=np.int32))
    if not flat_shading:
        mesh.polygons.foreach_set("use_smooth", np.ones(num_faces, dtype=np.bool_))
    rgba = np.empty((num_vertices, 4), dtype=np.float32)
    base_color = np.clip(attrs[:, 0:3], 0.0, 1.0)
    if base_color_space == "srgb":
        base_color = srgb_to_linear(base_color)
    rgba[:, 0:3] = base_color
    rgba[:, 3] = np.clip(attrs[:, 5], 0.0, 1.0)
    color_attribute = mesh.color_attributes.new(name="pixal_base_color", type="FLOAT_COLOR", domain="POINT")
    color_attribute.data.foreach_set("color", rgba.reshape(-1))
    metallic_attribute = mesh.attributes.new(name="pixal_metallic", type="FLOAT", domain="POINT")
    metallic_attribute.data.foreach_set("value", np.clip(attrs[:, 3], 0.0, 1.0))
    roughness_attribute = mesh.attributes.new(name="pixal_roughness", type="FLOAT", domain="POINT")
    roughness_attribute.data.foreach_set("value", np.clip(attrs[:, 4], 0.0, 1.0))
    try:
        mesh.update(calc_edges=False, calc_edges_loose=False)
    except TypeError:
        mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    # Critical correction omitted by the previous script.
    obj.matrix_world = DECODER_TO_ALIGNED_BLENDER
    return obj


def load_package(args):
    package_dir = args.package_dir.resolve()
    if not (package_dir / "READY").is_file():
        raise FileNotFoundError("incomplete package: %s" % package_dir)
    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    vertices = np.load(package_dir / manifest["vertices_file"], mmap_mode="r")
    attrs = np.load(package_dir / manifest["attrs_file"], mmap_mode="r")
    material = make_material(args.material_mode, args.use_alpha)
    total_vertices = 0
    total_faces = 0
    for shard_index, shard in enumerate(manifest["shards"], start=1):
        started = time.perf_counter()
        vertex_ids = np.load(package_dir / shard["vertex_ids_file"], mmap_mode="r")
        faces = np.load(package_dir / shard["faces_file"], mmap_mode="r")
        shard_vertices = np.ascontiguousarray(vertices[vertex_ids], dtype=np.float32)
        shard_attrs = np.ascontiguousarray(attrs[vertex_ids], dtype=np.float32)
        shard_faces = np.ascontiguousarray(faces, dtype=np.int32)
        create_mesh_object(
            "Pixal3D_NoUV_%05d" % (shard_index - 1),
            shard_vertices,
            shard_faces,
            shard_attrs,
            material,
            args.base_color_space,
            bool(args.flat_shading),
        )
        total_vertices += int(shard_vertices.shape[0])
        total_faces += int(shard_faces.shape[0])
        del vertex_ids, faces, shard_vertices, shard_attrs, shard_faces
        gc.collect()
        log("shard=%d/%d cumulative_vertices=%d cumulative_faces=%d seconds=%.3f" % (
            shard_index, len(manifest["shards"]), total_vertices, total_faces, time.perf_counter() - started
        ))
    log("axis transform applied: (x,y,z)->(x,-z,y)")
    return manifest


def write_status(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    jobs = json.loads(args.jobs_json.read_text(encoding="utf-8"))
    clear_scene()
    scene = configure_scene(args)
    manifest = load_package(args)
    camera = manifest["camera"]
    fov_rad = float(camera["camera_angle_x"])
    distance = float(camera["distance"]) * float(camera.get("mesh_scale", 1.0))
    create_aligned_camera(scene, distance, fov_rad)
    log("camera fov=%.12f distance=%.12f" % (fov_rad, distance))
    for index, job in enumerate(jobs, start=1):
        try:
            clear_lights()
            if args.material_mode == "pbr":
                add_light_rig(scene, job["light_mode"])
            else:
                set_world(scene, 0.0)
            output = Path(job["output_png"])
            output.parent.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(output)
            started = time.perf_counter()
            log("render %d/%d light=%s" % (index, len(jobs), job["light_mode"]))
            bpy.ops.render.render(write_still=True)
            elapsed = time.perf_counter() - started
            write_status(job["status_json"], {"status": "success", "output_png": str(output), "seconds": elapsed, "error": None})
            log("render seconds=%.3f" % elapsed)
        except Exception as exc:
            traceback.print_exc()
            write_status(job["status_json"], {
                "status": "failed",
                "output_png": str(job["output_png"]),
                "error": "%s: %s" % (type(exc).__name__, exc),
                "traceback": traceback.format_exc(),
            })


if __name__ == "__main__":
    main()
'''


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    keys = {key for row in rows for key in row}
    preferred = [
        "status", "image_name", "pipeline_resolution", "actual_grid_resolution", "seed", "light",
        "render_resolution", "metric_resolution", "psnr_db", "ssim", "lpips", "decoder_vertices",
        "decoder_faces", "selected_faces", "num_shards", "original_png", "render_png",
        "comparison_png", "metrics_json", "error",
    ]
    fields = [key for key in preferred if key in keys] + sorted(key for key in keys if key not in preferred)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def composite_on_black(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def load_black_image(path: Path, resolution: int) -> Image.Image:
    with Image.open(path) as image:
        result = composite_on_black(image)
    target = (int(resolution), int(resolution))
    if result.size != target:
        result = result.resize(target, Image.Resampling.LANCZOS)
    return result


def save_black_render(path: Path, resolution: int) -> Image.Image:
    result = load_black_image(path, resolution)
    result.save(path)
    return result


def save_comparison(original: Image.Image, rendered: Image.Image, output_path: Path) -> None:
    original = original.convert("RGB")
    rendered = rendered.convert("RGB")
    if rendered.size != original.size:
        rendered = rendered.resize(original.size, Image.Resampling.LANCZOS)
    width, height = original.size
    comparison = Image.new("RGB", (width * 2, height), (255, 255, 255))
    comparison.paste(original, (0, 0))
    comparison.paste(rendered, (width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(output_path)


def image_to_tensor(image: Image.Image, size: Tuple[int, int]) -> torch.Tensor:
    rgb = image.convert("RGB")
    if rgb.size != size:
        rgb = rgb.resize(size, Image.Resampling.LANCZOS)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def path_to_tensor(path: Path, size: Tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = composite_on_black(image)
    return image_to_tensor(rgb, size)


def psnr_metric(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    mse = float(F.mse_loss(prediction, reference).item())
    return float("inf") if mse <= 0.0 else float(10.0 * math.log10(1.0 / mse))


def gaussian_kernel(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coordinates = torch.arange(window_size, device=device, dtype=dtype)
    coordinates = coordinates - (window_size - 1) / 2.0
    kernel = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel[:, None] * kernel[None, :]


def ssim_metric(reference: torch.Tensor, prediction: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    x = reference.unsqueeze(0).float()
    y = prediction.unsqueeze(0).float()
    channels = int(x.shape[1])
    kernel = gaussian_kernel(window_size, sigma, x.device, x.dtype)[None, None].expand(channels, 1, window_size, window_size)
    padding = window_size // 2
    mu_x = F.conv2d(x, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=channels)
    mu_x_sq, mu_y_sq, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sigma_x_sq = F.conv2d(x * x, kernel, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=channels) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    value = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2) + 1e-12)
    return float(value.mean().item())


class LPIPSEvaluator:
    def __init__(self, network: str, device: torch.device):
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError("The lpips package is required: pip install lpips") from exc
        self.device = device
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = lpips.LPIPS(net=network).eval().to(device)

    @torch.inference_mode()
    def evaluate(self, reference: torch.Tensor, prediction: torch.Tensor) -> float:
        x = reference.unsqueeze(0).to(self.device)
        y = prediction.unsqueeze(0).to(self.device)
        return float(self.model(x * 2.0 - 1.0, y * 2.0 - 1.0).mean().item())


def load_torch_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def copy_tensor_to_npy(tensor: torch.Tensor, output_path: Path, rows_per_chunk: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(output_path, mode="w+", dtype=tensor.numpy().dtype, shape=tuple(int(v) for v in tensor.shape))
    total = int(tensor.shape[0])
    for start in range(0, total, rows_per_chunk):
        end = min(start + rows_per_chunk, total)
        array[start:end] = tensor[start:end].numpy()
        array.flush()
        print(f"[package] {output_path.name} rows={end:,}/{total:,}", flush=True)
    del array


def verify_alignment(vertices: torch.Tensor, coords: torch.Tensor, grid_size: int, samples: int) -> Dict[str, Any]:
    count = min(int(vertices.shape[0]), int(samples))
    if count <= 0:
        return {"checked": 0, "invalid": 0}
    indices = torch.linspace(0, int(vertices.shape[0]) - 1, steps=count, dtype=torch.float64).round().long()
    local = (vertices.index_select(0, indices).float() + 0.5) * float(grid_size) - coords.index_select(0, indices).float()
    invalid = ((local < -0.501) | (local > 1.501)).any(dim=1)
    return {"checked": count, "invalid": int(invalid.sum()), "local_min": float(local.min()), "local_max": float(local.max())}


def prepare_package(args: argparse.Namespace, cache_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    if (args.package_dir / "READY").is_file() and not args.overwrite_package:
        manifest = json.loads((args.package_dir / "manifest.json").read_text(encoding="utf-8"))
        print(f"[package] reuse={args.package_dir} faces={manifest.get('selected_faces'):,} shards={manifest.get('num_shards'):,}")
        return manifest
    vertices = load_torch_tensor(args.cache_dir / "vertices.pt")
    faces = load_torch_tensor(args.cache_dir / "faces.pt")
    attrs = load_torch_tensor(args.cache_dir / "attr_volume.pt")
    coords = load_torch_tensor(args.cache_dir / "coords.pt")
    if attrs.shape[0] != vertices.shape[0] or attrs.shape[1] < 6 or coords.shape != vertices.shape:
        raise ValueError("cache tensor shapes are inconsistent")
    alignment = verify_alignment(vertices, coords, int(cache_manifest["grid_size"]), args.alignment_samples)
    print(f"[alignment] checked={alignment['checked']:,} invalid={alignment['invalid']:,} range=[{alignment.get('local_min')}, {alignment.get('local_max')}]")
    if alignment["invalid"] and not args.allow_alignment_failure:
        raise RuntimeError("vertices/attrs index alignment check failed")
    temporary = args.package_dir.parent / f".{args.package_dir.name}.tmp-{os.getpid()}-{time.time_ns()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        copy_tensor_to_npy(vertices, temporary / "vertices.npy", args.copy_rows_per_chunk)
        copy_tensor_to_npy(attrs, temporary / "attrs.npy", args.copy_rows_per_chunk)
        total_faces = int(faces.shape[0])
        selected_faces = total_faces if args.max_faces <= 0 else min(total_faces, args.max_faces)
        (temporary / "shards").mkdir()
        shards = []
        faces_np = faces.numpy()
        for shard_index, start in enumerate(range(0, selected_faces, args.faces_per_shard)):
            end = min(start + args.faces_per_shard, selected_faces)
            global_faces = np.asarray(faces_np[start:end], dtype=np.int32)
            vertex_ids, inverse = np.unique(global_faces.reshape(-1), return_inverse=True)
            local_faces = inverse.reshape(-1, 3).astype(np.int32, copy=False)
            vertex_ids = vertex_ids.astype(np.int64, copy=False)
            vname = f"shards/shard_{shard_index:05d}_vertex_ids.npy"
            fname = f"shards/shard_{shard_index:05d}_faces.npy"
            np.save(temporary / vname, vertex_ids, allow_pickle=False)
            np.save(temporary / fname, local_faces, allow_pickle=False)
            shards.append({"index": shard_index, "num_faces": end-start, "num_vertices": int(vertex_ids.shape[0]), "vertex_ids_file": vname, "faces_file": fname})
            print(f"[package] shard={shard_index:05d} faces={end-start:,} vertices={vertex_ids.shape[0]:,}")
        camera = dict(cache_manifest.get("extra_metadata", {}).get("camera_params", {}))
        for key in ("camera_angle_x", "distance", "mesh_scale"):
            if key not in camera:
                raise KeyError(f"cache camera missing {key}")
        manifest = {
            "package_version": PACKAGE_VERSION,
            "source_cache": str(args.cache_dir),
            "vertices_file": "vertices.npy",
            "attrs_file": "attrs.npy",
            "source_vertices": int(vertices.shape[0]),
            "source_faces": total_faces,
            "selected_faces": selected_faces,
            "faces_per_shard": args.faces_per_shard,
            "num_shards": len(shards),
            "camera": camera,
            "decoder_to_aligned_blender": DECODER_TO_ALIGNED_BLENDER,
            "alignment_check": alignment,
            "shards": shards,
        }
        atomic_json(temporary / "manifest.json", manifest)
        (temporary / "READY").write_text("ok\n", encoding="utf-8")
        if args.package_dir.exists():
            shutil.rmtree(args.package_dir)
        temporary.replace(args.package_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        del vertices, faces, attrs, coords
        gc.collect()


def find_reference(cache_dir: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    current = cache_dir
    for _ in range(7):
        for name in ("metric_reference_rgb.png", "input_preprocessed_rgba.png"):
            candidate = current / name
            if candidate.is_file():
                return candidate.resolve()
        current = current.parent
    raise FileNotFoundError("reference image not found; pass --reference-image")


def infer_identity(cache_dir: Path, cache_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    extra = dict(cache_manifest.get("extra_metadata", {}))
    image_name, resolution_label, seed_label = "image", "resolution", "seed"
    for parent in cache_dir.parents:
        if parent.name.startswith("seed_"):
            seed_label = parent.name
        if parent.name.startswith("r") and parent.name[1:].isdigit():
            resolution_label = parent.name
            image_name = parent.parent.name
            break
    tensors = cache_manifest.get("tensor_files", {})
    return {
        "image_name": image_name,
        "resolution_label": resolution_label,
        "seed_label": seed_label,
        "pipeline_resolution": extra.get("pipeline_resolution"),
        "actual_grid_resolution": extra.get("actual_grid_resolution", cache_manifest.get("grid_size")),
        "seed": extra.get("seed"),
        "decoder_vertices": extra.get("decoder_vertices", tensors.get("vertices", {}).get("shape", [None])[0]),
        "decoder_faces": extra.get("decoder_faces", tensors.get("faces", {}).get("shape", [None])[0]),
    }


def experiment_paths(output_dir: Path, prefix: str, light: str) -> Dict[str, Path]:
    directory = output_dir / light
    base = f"{prefix}__{light}"
    return {
        "dir": directory,
        "original": directory / f"{base}__original.png",
        "render": directory / f"{base}__render.png",
        "comparison": directory / f"{base}__comparison.png",
        "metrics": directory / f"{base}__metrics.json",
        "status": directory / f"{base}__render_status.json",
    }


def run_blender(args: argparse.Namespace, jobs: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    helper = args.package_dir / "_pixal3d_no_uv_aligned_blender.py"
    helper.write_text(BLENDER_HELPER_SOURCE, encoding="utf-8")
    jobs_path = args.output_dir / "blender_jobs.json"
    atomic_json(jobs_path, list(jobs))
    command = [
        str(args.blender), "--background", "--factory-startup", "--python", str(helper), "--",
        "--package-dir", str(args.package_dir), "--jobs-json", str(jobs_path),
        "--engine", args.engine, "--render-resolution", str(args.render_resolution),
        "--samples", str(args.samples), "--material-mode", args.material_mode,
        "--base-color-space", args.base_color_space,
    ]
    if args.flat_shading:
        command.append("--flat-shading")
    if args.use_alpha:
        command.append("--use-alpha")
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    for key in ("PYTHONHOME", "PYTHONPATH"):
        environment.pop(key, None)
    if "conda" in environment.get("LD_LIBRARY_PATH", "").lower():
        environment.pop("LD_LIBRARY_PATH", None)
    print("[blender-command] " + shlex.join(command))
    log_path = args.output_dir / "blender.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, text=True, env=environment, check=False)
    results = {}
    for job in jobs:
        status_path = Path(job["status_json"])
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
        elif Path(job["output_png"]).is_file():
            status = {"status": "success", "output_png": job["output_png"], "error": None}
        else:
            status = {"status": "failed", "output_png": job["output_png"], "error": f"Blender exit code {process.returncode}"}
        results[str(job["output_png"])] = status
    print(f"[render] requested={len(jobs)} success={sum(v.get('status') == 'success' for v in results.values())}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reference-image", type=Path, default=None)
    parser.add_argument("--max-faces", type=int, default=0)
    parser.add_argument("--faces-per-shard", type=int, default=2_000_000)
    parser.add_argument("--copy-rows-per-chunk", type=int, default=5_000_000)
    parser.add_argument("--alignment-samples", type=int, default=100_000)
    parser.add_argument("--allow-alignment-failure", action="store_true")
    parser.add_argument("--overwrite-package", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--lights", nargs="+", choices=LIGHT_MODES, default=["studio"])
    parser.add_argument("--engine", choices=("cycles", "eevee"), default="cycles")
    parser.add_argument("--material-mode", choices=("pbr", "basecolor"), default="pbr")
    parser.add_argument("--base-color-space", choices=("srgb", "linear"), default="srgb")
    parser.add_argument("--flat-shading", action="store_true")
    parser.add_argument("--use-alpha", action="store_true")
    parser.add_argument("--render-resolution", type=int, default=2048)
    parser.add_argument("--metric-resolution", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--overwrite-renders", action="store_true")
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--metric-device", default="cuda")
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()
    args.cache_dir = args.cache_dir.resolve()
    args.package_dir = (args.cache_dir.parent / "no_uv_full_package") if args.package_dir is None else args.package_dir.resolve()
    args.output_dir = (args.cache_dir.parent / "no_uv_aligned_eval") if args.output_dir is None else args.output_dir.resolve()
    args.lights = list(dict.fromkeys(args.lights))
    if args.render_only and args.prepare_only:
        parser.error("--render-only and --prepare-only are mutually exclusive")
    if args.faces_per_shard <= 0 or args.render_resolution <= 0 or args.metric_resolution <= 0 or args.samples <= 0:
        parser.error("numeric arguments must be positive")
    return args


def main() -> int:
    args = parse_args()
    if not (args.cache_dir / "READY").is_file():
        raise FileNotFoundError(f"incomplete cache: {args.cache_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest = json.loads((args.cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if int(cache_manifest.get("cache_version", -1)) != CACHE_VERSION:
        raise RuntimeError("unsupported cache version")
    reference_path = find_reference(args.cache_dir, args.reference_image)
    identity = infer_identity(args.cache_dir, cache_manifest)
    prefix = f"{identity['image_name']}__{identity['resolution_label']}__{identity['seed_label']}__no_uv"
    print(f"[transform] raw decoder (x,y,z) -> aligned Blender (x,-z,y)")
    print(f"[reference] {reference_path}")
    if args.render_only:
        if not (args.package_dir / "READY").is_file():
            raise FileNotFoundError(args.package_dir)
        package_manifest = json.loads((args.package_dir / "manifest.json").read_text(encoding="utf-8"))
    else:
        package_manifest = prepare_package(args, cache_manifest)
    if args.prepare_only:
        return 0
    jobs = []
    by_light = {}
    for light in args.lights:
        paths = experiment_paths(args.output_dir, prefix, light)
        paths["dir"].mkdir(parents=True, exist_ok=True)
        by_light[light] = paths
        if args.overwrite_renders or not paths["render"].is_file():
            jobs.append({"light_mode": light, "output_png": str(paths["render"]), "status_json": str(paths["status"])})
    blender_results = run_blender(args, jobs) if jobs else {}
    reference_image = load_black_image(reference_path, args.render_resolution)
    metric_size = (args.metric_resolution, args.metric_resolution)
    reference_tensor = image_to_tensor(reference_image, metric_size)
    metric_device = torch.device(args.metric_device if str(args.metric_device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    lpips_evaluator = None
    rows = []
    for light in args.lights:
        paths = by_light[light]
        base = {
            "status": "failed",
            "image_name": identity["image_name"],
            "pipeline_resolution": identity["pipeline_resolution"],
            "actual_grid_resolution": identity["actual_grid_resolution"],
            "seed": identity["seed"],
            "light": light,
            "render_resolution": args.render_resolution,
            "metric_resolution": args.metric_resolution,
            "decoder_vertices": identity["decoder_vertices"],
            "decoder_faces": identity["decoder_faces"],
            "selected_faces": package_manifest.get("selected_faces"),
            "num_shards": package_manifest.get("num_shards"),
            "original_png": str(paths["original"]),
            "render_png": str(paths["render"]),
            "comparison_png": str(paths["comparison"]),
            "metrics_json": str(paths["metrics"]),
            "coordinate_transform": "(x,y,z)->(x,-z,y)",
        }
        status = blender_results.get(str(paths["render"]))
        if status is not None and status.get("status") != "success":
            row = {**base, "error": f"render failed: {status.get('error')}"}
            rows.append(row)
            atomic_json(paths["metrics"], row)
            continue
        if not paths["render"].is_file():
            row = {**base, "error": "render PNG is missing"}
            rows.append(row)
            atomic_json(paths["metrics"], row)
            continue
        try:
            rendered = save_black_render(paths["render"], args.render_resolution)
            reference_image.save(paths["original"])
            save_comparison(reference_image, rendered, paths["comparison"])
            prediction_tensor = path_to_tensor(paths["render"], metric_size)
            if not args.skip_lpips and lpips_evaluator is None:
                lpips_evaluator = LPIPSEvaluator(args.lpips_net, metric_device)
            metrics = {
                "psnr_db": psnr_metric(reference_tensor, prediction_tensor),
                "ssim": ssim_metric(reference_tensor, prediction_tensor),
                "lpips": None if args.skip_lpips else lpips_evaluator.evaluate(reference_tensor, prediction_tensor),
            }
            row = {**base, "status": "success", **metrics, "error": None}
            rows.append(row)
            atomic_json(paths["metrics"], row)
            print(f"[metrics] light={light} PSNR={metrics['psnr_db']:.4f} SSIM={metrics['ssim']:.6f} LPIPS={metrics['lpips']}")
            del rendered, prediction_tensor
        except Exception as exc:
            traceback.print_exc()
            row = {**base, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            rows.append(row)
            atomic_json(paths["metrics"], row)
    write_csv(args.output_dir / "metrics.csv", rows)
    atomic_json(args.output_dir / "metrics.json", {
        "config": vars(args),
        "metric_convention": {
            "reference": "Pixal3D preprocessed aligned condition image",
            "background": "black",
            "region": "full RGB canvas",
            "saved_render_resolution": args.render_resolution,
            "metric_resolution": args.metric_resolution,
            "ssim": "11x11 Gaussian-window RGB SSIM",
            "lpips_network": None if args.skip_lpips else args.lpips_net,
            "averages_included": False,
        },
        "coordinate_transform": {"mapping": "(x,y,z)->(x,-z,y)", "matrix": DECODER_TO_ALIGNED_BLENDER},
        "rows": rows,
    })
    atomic_json(args.output_dir / "run_config.json", {
        "config": vars(args),
        "reference_image": reference_path,
        "identity": identity,
        "package_dir": args.package_dir,
        "coordinate_transform": {"mapping": "(x,y,z)->(x,-z,y)", "matrix": DECODER_TO_ALIGNED_BLENDER},
        "comparison_layout": "aligned original left, no-UV render right",
    })
    successful = sum(row.get("status") == "success" for row in rows)
    print(f"[done] successful={successful} failed={len(rows)-successful} output={args.output_dir}")
    return 1 if successful != len(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())