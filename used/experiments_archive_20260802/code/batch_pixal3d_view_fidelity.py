#!/usr/bin/env python3
"""Batch Pixal3D view-fidelity benchmark using the official training protocol.

For each source GLB this script:
  1. Uses Blender's training-time scene normalization (unit cube, centered).
  2. Uses Pixal3D data_toolkit.utils.sphere_hammersley_sequence() for views.
  3. Saves the actual Blender camera-to-world matrix used for each condition image.
  4. Calls the official sphere_normalize_torch() and transform_mesh() functions.
  5. Derives mesh_scale exactly as training does: box_scale / sphere_radius.
  6. Feeds the full rendered canvas to pipeline.run(preprocess_image=False), without
     rembg/cropping/recentering it a second time.
  7. Saves and evaluates pipeline.run()'s raw decoder mesh. Exported GLB is only
     a visualization artifact and is never used for geometric metrics.
  8. Checks GT normal-render silhouette against the Blender input alpha mask and
     aborts a view when protocol IoU is below --protocol-min-iou.
  9. Undoes the complete o_voxel + Pixal3D GLB coordinate transform and checks
     exported-GLB silhouette against the raw decoder mesh via --export-min-iou.
 10. Runs one or more pipeline resolutions on exactly the same rendered views,
     camera matrices, and seeds, storing outputs under view_xxx/r<resolution>.
 11. Writes paired per-view resolution comparisons (for example 1536 - 1024).

Run from the Pixal3D repository root. Blender, CUDA, nvdiffrast and the local
Pixal3D checkpoints are required.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import random
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_cache.json"),
)
os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "1")

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image, ImageDraw, ImageFont



# -----------------------------------------------------------------------------
# Local checkpoints: kept consistent with the user-provided inference script.
# -----------------------------------------------------------------------------
MODEL_PATH = "/home/nvme04/yyyan/download/model/Pixal3D"
DINOV3_PATH = (
    "/home/nvme04/yyyan/download/model/"
    "dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m"
)

IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": DINOV3_PATH,
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": DINOV3_PATH,
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": DINOV3_PATH,
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": DINOV3_PATH,
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}

PIXAL3D_EXPORT_ROTATION = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# o_voxel.postprocess.to_glb() already converts decoder coordinates before
# constructing the trimesh object:
#     (x, y, z)_decoder -> (x, z, -y)_gltf
# Pixal3D then left-multiplies PIXAL3D_EXPORT_ROTATION before export.
# Therefore the complete decoder -> exported-GLTF transform is R @ S, not R.
OVOXEL_DECODER_TO_GLTF = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
PIXAL3D_DECODER_TO_EXPORTED_GLTF = (
    PIXAL3D_EXPORT_ROTATION @ OVOXEL_DECODER_TO_GLTF
)
PIXAL3D_EXPORTED_GLTF_TO_INTERNAL = np.linalg.inv(
    PIXAL3D_DECODER_TO_EXPORTED_GLTF
)

PAPER_ANGULAR_THRESHOLDS_DEG = (11.25, 22.5, 30.0)
PROTOCOL_VERSION = "pixal3d_official_training_view_v3_multires"
PRIMARY_METRIC_KEYS = (
    "normal_iou_percent",
    "normal_psnr_overlap_db",
    "normal_ssim_overlap",
    "normal_lpips_overlap",
    "normal_mean_angular_error_deg",
    "normal_median_angular_error_deg",
    "normal_boundary_mean_angular_error_deg",
    "normal_acc_11_25_percent",
    "normal_acc_22_5_percent",
    "normal_acc_30_percent",
    "rgb_iou_percent",
    "rgb_psnr_overlap_db",
    "rgb_ssim_overlap",
    "rgb_lpips_overlap",
    "protocol_gt_blender_vs_nvdiffrast_iou_percent",
    "protocol_raw_decoder_vs_exported_glb_iou_percent",
)


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------
def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "asset"


def asset_id(path: Path, root: Path) -> str:
    rel = path.resolve().relative_to(root.resolve())
    # Preserve the GLB stem as the directory name. Relative parent components
    # are prepended only when recursive discovery would otherwise collide.
    return "__".join(safe_name(part) for part in rel.with_suffix("").parts)


def discover_glbs(root: Path, recursive: bool) -> List[Path]:
    iterator = root.rglob("*.glb") if recursive else root.glob("*.glb")
    return sorted((p for p in iterator if p.is_file()), key=lambda p: p.as_posix().lower())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(data), file, ensure_ascii=False, indent=2, allow_nan=False)
    temp_path.replace(path)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def flatten_dict(value: Any, prefix: str = "") -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}__{key}" if prefix else str(key)
            output.update(flatten_dict(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}__{index}" if prefix else str(index)
            output.update(flatten_dict(item, child))
    elif isinstance(value, (str, int, float, bool)) or value is None:
        output[prefix] = value
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    preferred = [
        "asset_id",
        "mesh_name",
        "relative_input",
        "view_index",
        "yaw_deg",
        "elevation_deg",
        "seed",
        "pipeline_resolution",
        "status",
        *PRIMARY_METRIC_KEYS,
        "error",
    ]
    fields = [key for key in preferred if key in keys] + [key for key in keys if key not in preferred]
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def summarize_numeric_rows(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str] = PRIMARY_METRIC_KEYS,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"n_views": len(rows)}
    for key in keys:
        values = [number for row in rows if (number := numeric(row.get(key))) is not None]
        if not values:
            continue
        summary[key] = {
            "n": len(values),
            "mean": float(sum(values) / len(values)),
            "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
            "median": float(statistics.median(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }
    return summary


def resolution_pair_rows(
    rows: Sequence[Mapping[str, Any]],
    resolutions: Sequence[int],
) -> List[Dict[str, Any]]:
    """Build strictly paired per-view comparisons between pipeline resolutions."""
    successful = [
        row for row in rows
        if row.get("status") in {"success", "skipped"}
        and numeric(row.get("pipeline_resolution")) is not None
    ]
    grouped: Dict[Tuple[str, int, int], Dict[int, Mapping[str, Any]]] = {}
    for row in successful:
        key = (
            str(row.get("asset_id")),
            int(row.get("view_index", -1)),
            int(row.get("seed", -1)),
        )
        grouped.setdefault(key, {})[int(row["pipeline_resolution"])] = row

    ordered_resolutions = list(dict.fromkeys(int(value) for value in resolutions))
    output: List[Dict[str, Any]] = []
    for (identifier, view_index, seed), by_resolution in sorted(grouped.items()):
        for reference_index in range(len(ordered_resolutions)):
            for target_index in range(reference_index + 1, len(ordered_resolutions)):
                reference_resolution = ordered_resolutions[reference_index]
                target_resolution = ordered_resolutions[target_index]
                reference = by_resolution.get(reference_resolution)
                target = by_resolution.get(target_resolution)
                if reference is None or target is None:
                    continue
                pair: Dict[str, Any] = {
                    "asset_id": identifier,
                    "mesh_name": reference.get("mesh_name"),
                    "relative_input": reference.get("relative_input"),
                    "view_index": view_index,
                    "yaw_deg": reference.get("yaw_deg"),
                    "elevation_deg": reference.get("elevation_deg"),
                    "seed": seed,
                    "reference_resolution": reference_resolution,
                    "target_resolution": target_resolution,
                }
                for metric_name in PRIMARY_METRIC_KEYS:
                    reference_value = numeric(reference.get(metric_name))
                    target_value = numeric(target.get(metric_name))
                    if reference_value is None or target_value is None:
                        continue
                    pair[f"{metric_name}__r{reference_resolution}"] = reference_value
                    pair[f"{metric_name}__r{target_resolution}"] = target_value
                    pair[
                        f"{metric_name}__delta_r{target_resolution}_minus_r{reference_resolution}"
                    ] = target_value - reference_value
                output.append(pair)
    return output


def summarize_pair_deltas(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    delta_keys = sorted({
        key for row in rows for key in row
        if "__delta_r" in key and "_minus_r" in key
    })
    return summarize_numeric_rows(rows, keys=delta_keys)


# -----------------------------------------------------------------------------
# View generation and coordinate transforms
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ViewSpec:
    index: int
    yaw_rad: float
    pitch_rad: float
    yaw_deg: float
    elevation_deg: float
    initial_radius: float
    fov_rad: float


class Pixal3DTrainingProtocol:
    """Thin wrapper around the exact functions shipped in data_toolkit/utils.py."""

    def __init__(self, pixal3d_root: Path):
        self.root = pixal3d_root.resolve()
        utils_path = self.root / "data_toolkit" / "utils.py"
        if not utils_path.is_file():
            raise FileNotFoundError(
                f"Pixal3D training utility not found: {utils_path}. "
                "Run this script from the Pixal3D repository root or pass --pixal3d-root."
            )
        spec = importlib.util.spec_from_file_location(
            "_pixal3d_training_protocol_utils", utils_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load Pixal3D training utilities: {utils_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        required = (
            "sphere_hammersley_sequence",
            "sphere_normalize_torch",
            "transform_mesh",
        )
        missing = [name for name in required if not callable(getattr(module, name, None))]
        if missing:
            raise AttributeError(f"Missing official training functions in {utils_path}: {missing}")
        self.sphere_hammersley_sequence = module.sphere_hammersley_sequence
        self.sphere_normalize_torch = module.sphere_normalize_torch
        self.transform_mesh = module.transform_mesh
        self.utils_path = utils_path

    def make_views(
        self,
        num_views: int,
        max_elevation_deg: float,
        seed: int,
        fov_deg: float,
        camera_distance: Optional[float],
    ) -> Tuple[List[ViewSpec], Tuple[float, float]]:
        if num_views <= 0:
            raise ValueError("num_views must be positive")
        if not 0.0 <= max_elevation_deg < 89.0:
            raise ValueError("max_elevation_deg must be in [0, 89)")
        rng = np.random.RandomState(int(seed))
        offset = (float(rng.rand()), float(rng.rand()))
        fov_rad = math.radians(float(fov_deg))
        initial_radius = (
            float(camera_distance)
            if camera_distance is not None
            else float(math.sqrt(3.0) / 2.0 / math.sin(fov_rad / 2.0))
        )

        # The official sampler covers the whole sphere. Generate an official
        # Hammersley candidate set and retain the requested elevation band.
        candidate_count = max(64, num_views * 32)
        candidates: List[Tuple[float, float]] = []
        for candidate_index in range(candidate_count):
            yaw, pitch = self.sphere_hammersley_sequence(
                candidate_index, candidate_count, offset
            )
            yaw = float(yaw)
            pitch = float(pitch)
            if abs(math.degrees(pitch)) <= max_elevation_deg + 1e-9:
                candidates.append((yaw, pitch))
        if len(candidates) < num_views:
            raise RuntimeError(
                f"Official Hammersley sampler produced only {len(candidates)} views "
                f"inside ±{max_elevation_deg}°, need {num_views}"
            )

        # Spread selected indices across the retained band instead of taking a
        # contiguous prefix.
        selected_indices = np.linspace(0, len(candidates) - 1, num_views)
        selected_indices = np.round(selected_indices).astype(np.int64).tolist()
        views: List[ViewSpec] = []
        for index, candidate_index in enumerate(selected_indices):
            yaw, pitch = candidates[candidate_index]
            views.append(
                ViewSpec(
                    index=index,
                    yaw_rad=yaw,
                    pitch_rad=pitch,
                    yaw_deg=float(math.degrees(yaw) % 360.0),
                    elevation_deg=float(math.degrees(pitch)),
                    initial_radius=initial_radius,
                    fov_rad=fov_rad,
                )
            )
        return views, offset

    def view_align_mesh(
        self,
        normalized_mesh_npz: Path,
        frame: Mapping[str, Any],
        output_npz: Optional[Path] = None,
    ) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
        data = np.load(normalized_mesh_npz)
        vertices_np = np.ascontiguousarray(data["vertices"], dtype=np.float32)
        faces_np = np.ascontiguousarray(data["faces"], dtype=np.int64)
        vertices = torch.from_numpy(vertices_np).float().contiguous()
        faces = torch.from_numpy(faces_np).long().contiguous()

        # These are the exact calls used by data_toolkit/dual_grid_view.py.
        vertices_sphere, sphere_center, sphere_radius = self.sphere_normalize_torch(vertices)
        transformed_vertices = self.transform_mesh(vertices_sphere, dict(frame))
        abs_max = float(transformed_vertices.abs().max().item())
        if not math.isfinite(abs_max) or abs_max <= 0.0:
            raise ValueError(f"Invalid transformed abs_max: {abs_max}")
        box_scale = float(0.49999 / abs_max)
        transformed_normalized = transformed_vertices * box_scale
        total_scale = float(box_scale / float(sphere_radius.item()))

        out_vertices = np.ascontiguousarray(
            transformed_normalized.detach().cpu().numpy(), dtype=np.float32
        )
        out_faces = np.ascontiguousarray(faces.detach().cpu().numpy(), dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=out_vertices, faces=out_faces, process=False)
        mesh, integrity = sanitize_mesh(mesh, label="training_view_aligned_gt")
        if output_npz is not None:
            output_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_npz,
                vertices=np.ascontiguousarray(mesh.vertices, dtype=np.float32),
                faces=np.ascontiguousarray(mesh.faces, dtype=np.int64),
            )
        info = {
            "sphere_center": sphere_center.detach().cpu().tolist(),
            "sphere_radius": float(sphere_radius.item()),
            "post_transform_abs_max": abs_max,
            "box_scale": box_scale,
            "total_scale": total_scale,
            "training_utils": str(self.utils_path),
            "mesh_integrity": integrity,
        }
        return mesh, info


def load_npz_mesh(path: Path, label: str) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    data = np.load(path)
    mesh = trimesh.Trimesh(
        vertices=np.ascontiguousarray(data["vertices"], dtype=np.float32),
        faces=np.ascontiguousarray(data["faces"], dtype=np.int64),
        process=False,
    )
    return sanitize_mesh(mesh, label=label)

def sanitize_mesh(mesh: trimesh.Trimesh, label: str) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    vertices = np.ascontiguousarray(np.asarray(mesh.vertices, dtype=np.float64))
    faces_raw = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"{label}: vertices must be [V,3], got {vertices.shape}")
    if faces_raw.ndim != 2 or faces_raw.shape[1] != 3:
        raise ValueError(f"{label}: faces must be [F,3], got {faces_raw.shape}")

    finite_vertices = np.isfinite(vertices).all(axis=1)
    if np.issubdtype(faces_raw.dtype, np.floating):
        finite_faces = np.isfinite(faces_raw).all(axis=1)
        integral_faces = np.isclose(faces_raw, np.round(faces_raw), atol=0.0).all(axis=1)
        faces = np.round(np.nan_to_num(faces_raw, nan=-1)).astype(np.int64)
    else:
        finite_faces = np.ones(len(faces_raw), dtype=bool)
        integral_faces = np.ones(len(faces_raw), dtype=bool)
        faces = np.asarray(faces_raw, dtype=np.int64)

    if len(vertices) == 0:
        raise ValueError(f"{label}: mesh has no vertices")
    in_range = (faces >= 0).all(axis=1) & (faces < len(vertices)).all(axis=1)
    distinct = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    references_finite = np.zeros(len(faces), dtype=bool)
    safe_rows = finite_faces & integral_faces & in_range
    if safe_rows.any():
        references_finite[safe_rows] = finite_vertices[faces[safe_rows]].all(axis=1)
    valid = safe_rows & distinct & references_finite

    filtered_faces = np.ascontiguousarray(faces[valid], dtype=np.int64)
    if len(filtered_faces) == 0:
        raise ValueError(f"{label}: all faces are invalid")
    cleaned = trimesh.Trimesh(
        vertices=np.ascontiguousarray(vertices, dtype=np.float32),
        faces=filtered_faces,
        process=False,
    )
    cleaned.remove_unreferenced_vertices()
    stats = {
        "label": label,
        "input_vertices": int(len(vertices)),
        "input_faces": int(len(faces)),
        "output_vertices": int(len(cleaned.vertices)),
        "output_faces": int(len(cleaned.faces)),
        "dropped_faces": int((~valid).sum()),
        "nonfinite_face_rows": int((~finite_faces).sum()),
        "nonintegral_face_rows": int((~integral_faces).sum()),
        "out_of_range_face_rows": int((~in_range).sum()),
        "repeated_vertex_face_rows": int((~distinct).sum()),
        "nonfinite_vertex_reference_rows": int((safe_rows & ~references_finite).sum()),
    }
    return cleaned, stats


# -----------------------------------------------------------------------------
# Blender beauty/PBR renderer
# -----------------------------------------------------------------------------
BLENDER_HELPER_SOURCE = r'''
import argparse
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes, bpy.data.curves, bpy.data.materials,
        bpy.data.cameras, bpy.data.lights, bpy.data.images,
    ):
        for block in list(datablocks):
            try:
                datablocks.remove(block)
            except Exception:
                pass


def set_engine(scene):
    for engine in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
        try:
            scene.render.engine = engine
            return engine
        except Exception:
            continue
    scene.render.engine = 'CYCLES'
    return 'CYCLES'


def configure_render(resolution, transparent=True, samples=64):
    scene = bpy.context.scene
    engine = set_engine(scene)
    scene.render.resolution_x = int(resolution)
    scene.render.resolution_y = int(resolution)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = bool(transparent)
    scene.render.use_file_extension = True
    if engine.startswith('BLENDER_EEVEE') and hasattr(scene, 'eevee'):
        scene.eevee.taa_render_samples = int(samples)
    elif engine == 'CYCLES':
        scene.cycles.samples = int(samples)
        scene.cycles.use_denoising = True
    try:
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'Medium High Contrast'
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    world = scene.world or bpy.data.worlds.new('World')
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg is not None:
        bg.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs['Strength'].default_value = 0.8
    return scene


def import_glb(path):
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.gltf(filepath=str(path), merge_vertices=True, import_shading='NORMALS')
    except TypeError:
        bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj for obj in bpy.data.objects if obj not in before]
    roots = [obj for obj in imported if obj.parent is None]
    return imported, roots


def scene_bbox():
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    meshes = [obj for obj in bpy.context.scene.objects.values()
              if isinstance(obj.data, bpy.types.Mesh)]
    for obj in meshes:
        found = True
        for coord in obj.bound_box:
            coord = obj.matrix_world @ Vector(coord)
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError('no mesh objects in scene')
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene_training():
    # Exact normalization used by Pixal3D data_toolkit/blender_script/render_cond.py.
    roots = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(roots) > 1:
        scene_root = bpy.data.objects.new('ParentEmpty', None)
        bpy.context.scene.collection.objects.link(scene_root)
        for obj in roots:
            obj.parent = scene_root
    else:
        scene_root = roots[0]
    bbox_min, bbox_max = scene_bbox()
    scale = 1.0 / max(bbox_max - bbox_min)
    scene_root.scale = scene_root.scale * scale
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2.0
    scene_root.matrix_world.translation += offset
    bpy.context.view_layer.update()
    return float(scale), [float(offset.x), float(offset.y), float(offset.z)]


def dump_normalized_mesh(path):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices = []
    faces = []
    start = 0
    for obj in list(bpy.context.scene.objects):
        if obj.type != 'MESH':
            continue
        obj_eval = obj.evaluated_get(depsgraph)
        mesh_eval = obj_eval.to_mesh()
        bm = bmesh.new()
        try:
            bm.from_mesh(mesh_eval)
            bmesh.ops.triangulate(bm, faces=list(bm.faces))
            bm.verts.index_update()
            bm.faces.index_update()
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            world = obj_eval.matrix_world
            local_vertices = [world @ v.co for v in bm.verts]
            vertices.extend([[float(v.x), float(v.y), float(v.z)] for v in local_vertices])
            faces.extend([[start + int(v.index) for v in face.verts] for face in bm.faces])
            start += len(local_vertices)
        finally:
            bm.free()
            obj_eval.to_mesh_clear()
    if not vertices or not faces:
        raise RuntimeError('normalized scene contains no triangles')
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output),
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
    )


def get_transform_matrix(obj):
    pos, rotation, _ = obj.matrix_world.decompose()
    rotation = rotation.to_matrix()
    matrix = []
    for i in range(3):
        matrix.append([float(rotation[i][j]) for j in range(3)] + [float(pos[i])])
    matrix.append([0.0, 0.0, 0.0, 1.0])
    return matrix


def add_lights(scene):
    def add_area(name, location, energy, size):
        data = bpy.data.lights.new(name=name, type='AREA')
        data.energy = float(energy)
        data.shape = 'DISK'
        data.size = float(size)
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (-obj.location).to_track_quat('-Z', 'Y').to_euler()
    add_area('Key', (3.5, -4.0, 5.0), 900.0, 4.0)
    add_area('Fill', (-4.0, 1.5, 3.0), 500.0, 5.0)
    add_area('Rim', (1.0, 4.0, 2.5), 350.0, 3.0)


def make_track_camera(scene):
    camera_data = bpy.data.cameras.new('Camera')
    camera_data.type = 'PERSP'
    camera_data.sensor_fit = 'HORIZONTAL'
    camera_data.sensor_width = 32.0
    camera_data.sensor_height = 32.0
    camera_data.clip_start = 0.01
    camera_data.clip_end = 100.0
    camera = bpy.data.objects.new('Camera', camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    target = bpy.data.objects.new('CameraTarget', None)
    target.location = (0.0, 0.0, 0.0)
    scene.collection.objects.link(target)
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    constraint.target = target
    return camera


def mask_boundary_distance(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = image.size
        pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
        ys, xs = np.where(pixels[..., 3] > 0.0)
        if len(xs) == 0:
            return 0, max(width, height)
        distance = min(int(xs.min()), int(ys.min()),
                       int(width - 1 - xs.max()), int(height - 1 - ys.max()))
        return len(xs), distance
    finally:
        bpy.data.images.remove(image)


def render_training_source(job):
    clear_scene()
    scene = configure_render(job['resolution'], True, job.get('samples', 64))
    import_glb(job['input_glb'])
    scale, offset = normalize_scene_training()
    dump_normalized_mesh(job['normalized_mesh_npz'])
    camera = make_track_camera(scene)
    add_lights(scene)
    target_margin = int(round(130.0 * int(job['resolution']) / 1024.0))
    fit_boundary = bool(job.get('fit_boundary', True))
    for view in job['views']:
        radius = float(view['initial_radius'])
        retries = 0
        output = Path(view['output_png'])
        output.parent.mkdir(parents=True, exist_ok=True)
        while True:
            yaw = float(view['yaw_rad'])
            pitch = float(view['pitch_rad'])
            direction = np.asarray([
                math.cos(yaw) * math.cos(pitch),
                math.sin(yaw) * math.cos(pitch),
                math.sin(pitch),
            ], dtype=np.float64)
            camera.location = tuple(radius * direction)
            camera.data.lens = 16.0 / math.tan(float(view['fov_rad']) / 2.0)
            bpy.context.view_layer.update()
            scene.render.filepath = str(output)
            bpy.ops.render.render(write_still=True)
            _, margin = mask_boundary_distance(output)
            if not fit_boundary or retries >= 10:
                break
            if margin <= 0:
                radius *= 1.1
            elif margin > target_margin:
                radius *= 0.9
            else:
                break
            retries += 1
        bpy.context.view_layer.update()
        frame = {
            'file_path': output.name,
            'camera_angle_x': float(view['fov_rad']),
            'transform_matrix': get_transform_matrix(camera),
            'radius': float(radius),
            'original_radius': float(view['initial_radius']),
            'retries': int(retries),
            'yaw_rad': float(view['yaw_rad']),
            'pitch_rad': float(view['pitch_rad']),
            'yaw_deg': float(math.degrees(view['yaw_rad']) % 360.0),
            'elevation_deg': float(math.degrees(view['pitch_rad'])),
            'source_scene_scale': scale,
            'source_scene_offset': offset,
            'boundary_margin_pixels': int(margin),
        }
        camera_json = Path(view['camera_json'])
        camera_json.parent.mkdir(parents=True, exist_ok=True)
        camera_json.write_text(json.dumps(frame, indent=2), encoding='utf-8')


def apply_root_transform(roots, values):
    matrix_gltf = Matrix(values)
    gltf_to_blender = Matrix((
        (1.0, 0.0,  0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 1.0,  0.0, 0.0),
        (0.0, 0.0,  0.0, 1.0),
    ))
    matrix_blender = gltf_to_blender @ matrix_gltf @ gltf_to_blender.inverted()
    for root in roots:
        root.matrix_world = matrix_blender @ root.matrix_world


def render_fixed_glb(job):
    clear_scene()
    scene = configure_render(job['resolution'], True, job.get('samples', 64))
    _, roots = import_glb(job['input_glb'])
    apply_root_transform(roots, job['transform'])
    camera_data = bpy.data.cameras.new('Camera')
    camera = bpy.data.objects.new('Camera', camera_data)
    scene.collection.objects.link(camera)
    distance = float(job['distance'])
    camera.matrix_world = Matrix((
        (1.0, 0.0,  0.0, 0.0),
        (0.0, 0.0, -1.0, -distance),
        (0.0, 1.0,  0.0, 0.0),
        (0.0, 0.0,  0.0, 1.0),
    ))
    camera_data.type = 'PERSP'
    camera_data.sensor_fit = 'HORIZONTAL'
    camera_data.sensor_width = 32.0
    camera_data.lens = 16.0 / math.tan(float(job['fov_rad']) / 2.0)
    camera_data.clip_start = 0.01
    camera_data.clip_end = 100.0
    scene.camera = camera
    add_lights(scene)
    output = Path(job['output_png'])
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs-json', required=True)
    args = parser.parse_args(argv)
    jobs = json.loads(Path(args.jobs_json).read_text(encoding='utf-8'))
    for index, job in enumerate(jobs):
        print('[blender] job %d/%d kind=%s' %
              (index + 1, len(jobs), job.get('kind', 'fixed_glb')))
        if job.get('kind') == 'training_source':
            render_training_source(job)
        else:
            render_fixed_glb(job)


if __name__ == '__main__':
    main()
'''



def ensure_blender_helper(output_root: Path) -> Path:
    helper_path = output_root / "_blender_render_jobs.py"
    if not helper_path.is_file() or helper_path.read_text(encoding="utf-8") != BLENDER_HELPER_SOURCE:
        helper_path.write_text(BLENDER_HELPER_SOURCE, encoding="utf-8")
    return helper_path


def run_blender_jobs(
    blender_executable: str,
    helper_path: Path,
    jobs: Sequence[Mapping[str, Any]],
    output_root: Path,
    log_path: Path,
) -> None:
    if not jobs:
        return
    jobs_path = output_root / f"_blender_jobs_{int(time.time() * 1_000_000)}.json"
    atomic_json(jobs_path, list(jobs))
    command = [
        blender_executable,
        "--background",
        "--python",
        str(helper_path),
        "--",
        "--jobs-json",
        str(jobs_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[blender] {shlex.join(command)}")
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n# Blender command: {shlex.join(command)}\n")
        blender_env = os.environ.copy()
        for key in ("PYTHONHOME", "PYTHONPATH"):
            blender_env.pop(key, None)
        ld_library_path = blender_env.get("LD_LIBRARY_PATH", "")
        if "conda" in ld_library_path.lower() or "miniconda" in ld_library_path.lower():
            blender_env.pop("LD_LIBRARY_PATH", None)
        blender_env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

        process = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=blender_env,
        )
    try:
        jobs_path.unlink()
    except OSError:
        pass
    if process.returncode != 0:
        raise RuntimeError(
            f"Blender rendering failed with code {process.returncode}; see {log_path}"
        )
    missing: List[Path] = []
    for job in jobs:
        if job.get("kind") == "training_source":
            mesh_npz = Path(job["normalized_mesh_npz"])
            if not mesh_npz.is_file():
                missing.append(mesh_npz)
            for view in job["views"]:
                for key in ("output_png", "camera_json"):
                    candidate = Path(view[key])
                    if not candidate.is_file():
                        missing.append(candidate)
        else:
            candidate = Path(job["output_png"])
            if not candidate.is_file():
                missing.append(candidate)
    if missing:
        raise RuntimeError(f"Blender reported success but outputs are missing: {missing}")


# -----------------------------------------------------------------------------
# Pixal3D runner: load the complete pipeline once, then reuse it for every view.
# -----------------------------------------------------------------------------
def build_image_cond_model(config: Mapping[str, Any]):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        DinoV3ProjFeatureExtractor,
    )

    model = DinoV3ProjFeatureExtractor(**dict(config))
    model.eval()
    return model


def init_pipeline(model_path: str, device: str, low_vram: bool):
    from pixal3d.pipelines import Pixal3DImageTo3DPipeline

    print(f"[pipeline] loading Pixal3D from {model_path}")
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)
    pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"])
    pipeline.image_cond_model_shape_512 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_512"])
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_1024"])
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["tex_1024"])

    attributes = (
        "image_cond_model_ss",
        "image_cond_model_shape_512",
        "image_cond_model_shape_1024",
        "image_cond_model_tex_1024",
    )
    if low_vram:
        for attribute in attributes:
            model = getattr(pipeline, attribute, None)
            if model is not None and getattr(model, "use_naf_upsample", False):
                model._load_naf()
        pipeline._device = torch.device(device)
        pipeline.low_vram = True
        print("[pipeline] low-VRAM mode enabled")
    else:
        pipeline.low_vram = False
        if str(device).startswith("cuda"):
            pipeline.cuda()
        else:
            pipeline.to(device)
        for attribute in attributes:
            model = getattr(pipeline, attribute, None)
            if model is not None:
                if str(device).startswith("cuda"):
                    model.cuda()
                else:
                    model.to(device)
                if getattr(model, "use_naf_upsample", False):
                    model._load_naf()
        print("[pipeline] standard GPU-resident mode enabled")
    return pipeline


class Pixal3DRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.pipeline = init_pipeline(args.model_path, args.device, args.low_vram)
        self.ss_sampler = {
            "steps": args.ss_sampling_steps,
            "guidance_strength": args.ss_guidance_strength,
            "guidance_rescale": args.ss_guidance_rescale,
            "rescale_t": args.ss_rescale_t,
        }
        self.shape_sampler = {
            "steps": args.shape_sampling_steps,
            "guidance_strength": args.shape_guidance_strength,
            "guidance_rescale": args.shape_guidance_rescale,
            "rescale_t": args.shape_rescale_t,
        }
        self.texture_sampler = {
            "steps": args.texture_sampling_steps,
            "guidance_strength": args.texture_guidance_strength,
            "guidance_rescale": args.texture_guidance_rescale,
            "rescale_t": args.texture_rescale_t,
        }

    @torch.inference_mode()
    def generate(
        self,
        input_png: Path,
        preprocessed_png: Path,
        raw_mesh_npz: Path,
        output_glb: Path,
        seed: int,
        pipeline_resolution: int,
        camera_params: Mapping[str, float],
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        # Training condition renders are already segmented, framed, and camera
        # calibrated. Re-running preprocess_image() would crop/recenter the canvas
        # and invalidate the saved camera matrix.
        with Image.open(input_png) as source_image:
            rgba = source_image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        condition_image = background.convert("RGB")
        condition_image.save(preprocessed_png)

        # Pixal3DImageTo3DPipeline.run() expects one PIL image. The pipeline
        # itself wraps it as [image] before calling each projection-conditioned
        # DINO model. Passing a path or an already wrapped list produces:
        # "Image list should be list of PIL images".
        if not isinstance(condition_image, Image.Image):
            raise TypeError(
                f"condition_image must be PIL.Image.Image, got {type(condition_image)!r}"
            )

        set_seed(seed)
        pipeline_resolution = int(pipeline_resolution)
        if pipeline_resolution not in {1024, 1536}:
            raise ValueError(f"Unsupported pipeline resolution: {pipeline_resolution}")
        pipeline_type = f"{pipeline_resolution}_cascade"
        result = self.pipeline.run(
            condition_image,
            camera_params=dict(camera_params),
            seed=seed,
            sparse_structure_sampler_params=self.ss_sampler,
            shape_slat_sampler_params=self.shape_sampler,
            tex_slat_sampler_params=self.texture_sampler,
            preprocess_image=False,
            return_latent=True,
            pipeline_type=pipeline_type,
            max_num_tokens=self.args.max_num_tokens,
        )
        mesh_list, latent_bundle = result
        shape_slat, texture_slat, grid_resolution = latent_bundle
        mesh = mesh_list[0]

        def to_numpy(value: Any, dtype: np.dtype) -> np.ndarray:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            return np.ascontiguousarray(np.asarray(value), dtype=dtype)

        raw_mesh_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            raw_mesh_npz,
            vertices=to_numpy(mesh.vertices, np.float32),
            faces=to_numpy(mesh.faces, np.int64),
        )

        import o_voxel

        glb_scene = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=self.pipeline.pbr_attr_layout,
            grid_size=grid_resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=int(self.args.decimation_target),
            texture_size=int(self.args.texture_size),
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            use_tqdm=True,
        )
        glb_scene.apply_transform(PIXAL3D_EXPORT_ROTATION)
        output_glb.parent.mkdir(parents=True, exist_ok=True)
        glb_scene.export(str(output_glb), extension_webp=self.args.extension_webp)

        metadata = {
            "elapsed_seconds": float(time.perf_counter() - started),
            "pipeline_type": pipeline_type,
            "pipeline_resolution": pipeline_resolution,
            "grid_resolution": int(grid_resolution),
            "seed": int(seed),
            "camera_params": dict(camera_params),
            "raw_decoder_mesh": str(raw_mesh_npz),
            "metric_mesh_source": "pipeline.run raw MeshWithVoxel",
            "decimation_target": int(self.args.decimation_target),
            "texture_size": int(self.args.texture_size),
        }
        del mesh_list, mesh, shape_slat, texture_slat, latent_bundle, result, glb_scene
        gc.collect()
        torch.cuda.empty_cache()
        return metadata


# -----------------------------------------------------------------------------
# Exact face-chunked geometric normal renderer
# -----------------------------------------------------------------------------
class ChunkedNormalRenderer:
    def __init__(
        self,
        resolution: int,
        fov_deg: float,
        face_chunk_size: int,
        normal_mode: str,
        orient_to_camera: bool,
        device: str = "cuda",
    ):
        try:
            import nvdiffrast.torch as dr
        except ImportError as exc:
            raise RuntimeError("nvdiffrast is required for paper normal metrics") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by nvdiffrast")
        if face_chunk_size <= 0:
            raise ValueError("face_chunk_size must be positive")
        if normal_mode not in {"face", "vertex"}:
            raise ValueError(f"Unsupported normal_mode: {normal_mode}")
        self.dr = dr
        self.device = torch.device(device)
        if self.device.index is not None:
            torch.cuda.set_device(self.device)
        self.ctx = dr.RasterizeCudaContext()
        self.resolution = int(resolution)
        self.fov_deg = float(fov_deg)
        self.face_chunk_size = int(face_chunk_size)
        self.normal_mode = normal_mode
        self.orient_to_camera = bool(orient_to_camera)

    def view_matrix(self, distance: float) -> torch.Tensor:
        # Pixal3D internal fixed front camera:
        # camera at +Z, looking toward -Z; image right +X, image up +Y.
        matrix = torch.eye(4, dtype=torch.float32, device=self.device)
        matrix[2, 3] = -float(distance)
        return matrix

    def projection_matrix(self) -> torch.Tensor:
        near, far = 0.01, 100.0
        f = 1.0 / math.tan(math.radians(self.fov_deg) / 2.0)
        matrix = torch.zeros((4, 4), dtype=torch.float32, device=self.device)
        matrix[0, 0] = f
        matrix[1, 1] = f
        matrix[2, 2] = (far + near) / (near - far)
        matrix[2, 3] = 2.0 * far * near / (near - far)
        matrix[3, 2] = -1.0
        return matrix

    @staticmethod
    def _mesh_arrays(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cleaned, _ = sanitize_mesh(mesh, label="normal_renderer")
        vertices = np.ascontiguousarray(cleaned.vertices, dtype=np.float32)
        faces = np.ascontiguousarray(cleaned.faces, dtype=np.int64)
        vertex_normals = np.ascontiguousarray(cleaned.vertex_normals, dtype=np.float32)
        vertex_normals = np.nan_to_num(vertex_normals, nan=0.0, posinf=0.0, neginf=0.0)
        return vertices, faces, vertex_normals

    @torch.inference_mode()
    def render(self, mesh: trimesh.Trimesh, distance: float) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        vertices, faces, vertex_normals = self._mesh_arrays(mesh)
        height = width = self.resolution
        best_depth = torch.full(
            (height, width), float("inf"), dtype=torch.float32, device=self.device
        )
        best_normal = torch.zeros(
            (height, width, 3), dtype=torch.float32, device=self.device
        )
        best_mask = torch.zeros((height, width), dtype=torch.bool, device=self.device)

        view = self.view_matrix(distance)
        mvp = self.projection_matrix() @ view
        view_rotation = view[:3, :3]
        total_faces = int(len(faces))
        chunk_count = (total_faces + self.face_chunk_size - 1) // self.face_chunk_size
        max_local_vertices = 0

        for chunk_index, start in enumerate(range(0, total_faces, self.face_chunk_size)):
            stop = min(start + self.face_chunk_size, total_faces)
            global_faces = faces[start:stop]
            unique_vertices, inverse = np.unique(global_faces.reshape(-1), return_inverse=True)
            local_faces_np = np.ascontiguousarray(
                inverse.reshape(-1, 3).astype(np.int32, copy=False)
            )
            local_vertices_np = np.ascontiguousarray(vertices[unique_vertices])
            max_local_vertices = max(max_local_vertices, int(len(local_vertices_np)))

            vertex_tensor = torch.from_numpy(local_vertices_np).to(self.device)
            face_tensor = torch.from_numpy(local_faces_np).to(self.device)
            clip = torch.cat(
                [
                    vertex_tensor,
                    torch.ones(
                        (len(vertex_tensor), 1),
                        dtype=vertex_tensor.dtype,
                        device=self.device,
                    ),
                ],
                dim=1,
            ) @ mvp.T
            rast, _ = self.dr.rasterize(
                self.ctx,
                clip[None],
                face_tensor,
                resolution=[height, width],
            )
            mask = rast[0, ..., 3] > 0
            if bool(mask.any()):
                depth = rast[0, ..., 2]
                update = mask & ((~best_mask) | (depth < best_depth))
                if bool(update.any()):
                    if self.normal_mode == "vertex":
                        local_normals = torch.from_numpy(
                            np.ascontiguousarray(vertex_normals[unique_vertices])
                        ).to(self.device)
                        normal_image, _ = self.dr.interpolate(
                            local_normals[None], rast, face_tensor
                        )
                        normal_world = F.normalize(normal_image[0], dim=-1, eps=1e-8)
                        del local_normals, normal_image
                    else:
                        triangles = vertex_tensor[face_tensor.long()]
                        face_normals = F.normalize(
                            torch.cross(
                                triangles[:, 1] - triangles[:, 0],
                                triangles[:, 2] - triangles[:, 0],
                                dim=-1,
                            ),
                            dim=-1,
                            eps=1e-8,
                        )
                        triangle_ids = rast[0, ..., 3].long() - 1
                        safe_ids = triangle_ids.clamp(0, max(len(face_normals) - 1, 0))
                        normal_world = face_normals[safe_ids]
                        del triangles, face_normals, triangle_ids, safe_ids

                    normal_camera = normal_world @ view_rotation.T
                    normal_camera = F.normalize(normal_camera, dim=-1, eps=1e-8)
                    if self.orient_to_camera:
                        flip = normal_camera[..., 2:3] < 0
                        normal_camera = torch.where(flip, -normal_camera, normal_camera)
                    best_depth[update] = depth[update]
                    best_normal[update] = normal_camera[update]
                    best_mask[update] = True
                    del normal_world, normal_camera

            del vertex_tensor, face_tensor, clip, rast, mask
            if chunk_count > 1 and (
                chunk_index + 1 == chunk_count or (chunk_index + 1) % 25 == 0
            ):
                print(
                    f"[normal-render] chunk {chunk_index + 1}/{chunk_count}, "
                    f"faces={start:,}:{stop:,}, local_vertices={len(unique_vertices):,}"
                )

        # Match PIL/Blender top-left image origin.
        best_normal = torch.flip(best_normal, dims=[0])
        best_mask = torch.flip(best_mask, dims=[0])

        stats = {
            "vertices": int(len(vertices)),
            "faces": total_faces,
            "face_chunk_size": self.face_chunk_size,
            "chunk_count": int(chunk_count),
            "max_local_vertices": int(max_local_vertices),
            "normal_mode": self.normal_mode,
            "orient_to_camera": self.orient_to_camera,
            "mode": "exact_face_chunked_depth_composite",
        }
        return best_normal, best_mask, stats


# -----------------------------------------------------------------------------
# Image and paper metric functions
# -----------------------------------------------------------------------------
def normal_to_rgb(normal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(normal)
    output[mask] = normal[mask] * 0.5 + 0.5
    return output.clamp(0.0, 1.0)


def tensor_to_pil_rgb(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach().float().cpu().clamp(0.0, 1.0).numpy() * 255.0 + 0.5
    ).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def mask_to_pil(mask: torch.Tensor) -> Image.Image:
    array = (mask.detach().cpu().numpy().astype(np.uint8) * 255)
    return Image.fromarray(array, mode="L")


def load_rgba_tensor(path: Path, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(path).convert("RGBA")
    array = np.asarray(image, dtype=np.float32) / 255.0
    rgb = torch.from_numpy(array[..., :3]).to(device)
    mask = torch.from_numpy(array[..., 3] > 0.5).to(device)
    return rgb, mask


def masked_psnr(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return float("nan")
    mse = float(((first[mask] - second[mask]) ** 2).mean().item())
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def gaussian_kernel(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coordinates = torch.arange(window_size, device=device, dtype=dtype)
    coordinates = coordinates - (window_size - 1) / 2.0
    kernel_1d = torch.exp(-(coordinates ** 2) / (2.0 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d


def masked_ssim(
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    if not bool(mask.any()):
        return float("nan")
    x = first.permute(2, 0, 1)[None].float()
    y = second.permute(2, 0, 1)[None].float()
    channels = x.shape[1]
    kernel = gaussian_kernel(window_size, sigma, x.device, x.dtype)
    kernel = kernel[None, None].expand(channels, 1, window_size, window_size)
    padding = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=channels)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.conv2d(x * x, kernel, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=channels) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = (
        (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2) + 1e-12)
    )
    ssim_map = ssim_map.mean(dim=1)[0]

    mask_float = mask.float()[None, None]
    support = F.avg_pool2d(
        mask_float,
        kernel_size=window_size,
        stride=1,
        padding=padding,
    )[0, 0]
    valid = mask & (support > 0.999)
    if not bool(valid.any()):
        valid = mask
    return float(ssim_map[valid].mean().item())


class LPIPSEvaluator:
    def __init__(self, network: str, device: torch.device):
        self.network = network
        self.device = device
        self.model = None
        self.error: Optional[str] = None
        try:
            import lpips

            self.model = lpips.LPIPS(net=network).eval().to(device)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[warning] LPIPS disabled: {self.error}")

    @torch.inference_mode()
    def evaluate(self, first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
        if self.model is None or not bool(mask.any()):
            return None
        indices = torch.nonzero(mask, as_tuple=False)
        y0, x0 = indices.min(dim=0).values.tolist()
        y1, x1 = indices.max(dim=0).values.tolist()
        x = first[y0 : y1 + 1, x0 : x1 + 1].clone()
        y = second[y0 : y1 + 1, x0 : x1 + 1].clone()
        local_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
        x[~local_mask] = 0.0
        y[~local_mask] = 0.0
        x = x.permute(2, 0, 1)[None]
        y = y.permute(2, 0, 1)[None]
        min_side = min(x.shape[-2:])
        if min_side < 64:
            scale = 64.0 / max(min_side, 1)
            target = (
                max(64, int(round(x.shape[-2] * scale))),
                max(64, int(round(x.shape[-1] * scale))),
            )
            x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
            y = F.interpolate(y, size=target, mode="bilinear", align_corners=False)
        value = self.model(x * 2.0 - 1.0, y * 2.0 - 1.0)
        return float(value.mean().item())


def inner_boundary(mask: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return mask
    kernel_size = width * 2 + 1
    inverse = (~mask).float()[None, None]
    eroded = F.max_pool2d(
        inverse,
        kernel_size=kernel_size,
        stride=1,
        padding=width,
    )[0, 0] < 0.5
    return mask & (~eroded)


def angular_error_map(gt_normal: torch.Tensor, pred_normal: torch.Tensor) -> torch.Tensor:
    cosine = (gt_normal * pred_normal).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def normal_paper_metrics(
    gt_normal: torch.Tensor,
    pred_normal: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    lpips_evaluator: LPIPSEvaluator,
    boundary_width: int,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    intersection = gt_mask & pred_mask
    union = gt_mask | pred_mask
    iou = float(intersection.sum().item() / max(union.sum().item(), 1))
    gt_rgb = normal_to_rgb(gt_normal, gt_mask)
    pred_rgb = normal_to_rgb(pred_normal, pred_mask)

    errors = angular_error_map(gt_normal, pred_normal)
    overlap_errors = errors[intersection]
    if overlap_errors.numel() == 0:
        mean_error = median_error = boundary_error = float("nan")
        accuracies = {threshold: float("nan") for threshold in PAPER_ANGULAR_THRESHOLDS_DEG}
    else:
        mean_error = float(overlap_errors.mean().item())
        median_error = float(overlap_errors.median().item())
        boundary = inner_boundary(gt_mask, boundary_width) & pred_mask
        boundary_error = (
            float(errors[boundary].mean().item()) if bool(boundary.any()) else float("nan")
        )
        accuracies = {
            threshold: float((overlap_errors < threshold).float().mean().item() * 100.0)
            for threshold in PAPER_ANGULAR_THRESHOLDS_DEG
        }

    metrics = {
        "normal_iou": iou,
        "normal_iou_percent": iou * 100.0,
        "normal_psnr_overlap_db": masked_psnr(gt_rgb, pred_rgb, intersection),
        "normal_ssim_overlap": masked_ssim(gt_rgb, pred_rgb, intersection),
        "normal_lpips_overlap": lpips_evaluator.evaluate(gt_rgb, pred_rgb, intersection),
        "normal_mean_angular_error_deg": mean_error,
        "normal_median_angular_error_deg": median_error,
        "normal_boundary_mean_angular_error_deg": boundary_error,
        "normal_acc_11_25_percent": accuracies[11.25],
        "normal_acc_22_5_percent": accuracies[22.5],
        "normal_acc_30_percent": accuracies[30.0],
        "normal_overlap_pixels": int(intersection.sum().item()),
        "normal_union_pixels": int(union.sum().item()),
        "normal_gt_pixels": int(gt_mask.sum().item()),
        "normal_pred_pixels": int(pred_mask.sum().item()),
        "boundary_width_pixels": int(boundary_width),
    }
    return metrics, errors


def rgb_metrics(
    gt_rgb: torch.Tensor,
    pred_rgb: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    lpips_evaluator: LPIPSEvaluator,
) -> Dict[str, Any]:
    intersection = gt_mask & pred_mask
    union = gt_mask | pred_mask
    iou = float(intersection.sum().item() / max(union.sum().item(), 1))
    return {
        "rgb_iou": iou,
        "rgb_iou_percent": iou * 100.0,
        "rgb_psnr_overlap_db": masked_psnr(gt_rgb, pred_rgb, intersection),
        "rgb_ssim_overlap": masked_ssim(gt_rgb, pred_rgb, intersection),
        "rgb_lpips_overlap": lpips_evaluator.evaluate(gt_rgb, pred_rgb, intersection),
        "rgb_overlap_pixels": int(intersection.sum().item()),
    }


def angular_heatmap(errors: torch.Tensor, overlap: torch.Tensor, max_error_deg: float = 60.0) -> Image.Image:
    normalized = (errors / max_error_deg).clamp(0.0, 1.0)
    # Blue -> cyan -> yellow -> red, implemented without an external colormap.
    red = torch.clamp(2.0 * normalized, 0.0, 1.0)
    green = torch.clamp(2.0 - 2.0 * torch.abs(normalized - 0.5), 0.0, 1.0)
    blue = torch.clamp(2.0 * (1.0 - normalized), 0.0, 1.0)
    rgb = torch.stack([red, green, blue], dim=-1)
    rgb[~overlap] = 0.0
    return tensor_to_pil_rgb(rgb)


def composite_over_background(image: Image.Image, background=(24, 24, 24)) -> Image.Image:
    rgba = image.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (*background, 255))
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")


def save_comparison_sheet(
    path: Path,
    gt_beauty_path: Path,
    pred_beauty_path: Path,
    gt_normal_rgb: torch.Tensor,
    pred_normal_rgb: torch.Tensor,
    heatmap: Image.Image,
    metrics: Mapping[str, Any],
) -> None:
    gt_beauty = composite_over_background(Image.open(gt_beauty_path))
    pred_beauty = composite_over_background(Image.open(pred_beauty_path))
    gt_normal = tensor_to_pil_rgb(gt_normal_rgb)
    pred_normal = tensor_to_pil_rgb(pred_normal_rgb)
    images = [gt_beauty, pred_beauty, gt_normal, pred_normal, heatmap.convert("RGB")]
    labels = ["GT input render", "Pixal3D GLB render", "GT normal", "Pixal3D normal", "Angular error"]
    width, height = images[0].size
    label_height = 28
    footer_height = 76
    canvas = Image.new(
        "RGB",
        (width * len(images), height + label_height + footer_height),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (image, label) in enumerate(zip(images, labels)):
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        x = index * width
        canvas.paste(image, (x, label_height))
        draw.text((x + 8, 8), label, fill=(240, 240, 240), font=font)

    footer = (
        f"IoU={metrics.get('normal_iou_percent', float('nan')):.2f}%   "
        f"PSNR={metrics.get('normal_psnr_overlap_db', float('nan')):.3f} dB   "
        f"SSIM={metrics.get('normal_ssim_overlap', float('nan')):.4f}   "
        f"LPIPS={metrics.get('normal_lpips_overlap')}\n"
        f"Mean={metrics.get('normal_mean_angular_error_deg', float('nan')):.3f}°   "
        f"Median={metrics.get('normal_median_angular_error_deg', float('nan')):.3f}°   "
        f"Mean_B={metrics.get('normal_boundary_mean_angular_error_deg', float('nan')):.3f}°   "
        f"Acc@11.25/22.5/30="
        f"{metrics.get('normal_acc_11_25_percent', float('nan')):.2f}/"
        f"{metrics.get('normal_acc_22_5_percent', float('nan')):.2f}/"
        f"{metrics.get('normal_acc_30_percent', float('nan')):.2f}%"
    )
    draw.multiline_text(
        (8, label_height + height + 10),
        footer,
        fill=(240, 240, 240),
        font=font,
        spacing=8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


# -----------------------------------------------------------------------------
# Per-asset execution
# -----------------------------------------------------------------------------
def view_seed(base_seed: int, view_index: int, mode: str) -> int:
    if mode == "fixed":
        return int(base_seed)
    if mode == "offset":
        return int(base_seed + view_index)
    raise ValueError(mode)


def shared_view_paths(asset_dir: Path, index: int) -> Dict[str, Path]:
    """Artifacts shared by every pipeline resolution for one calibrated view."""
    view_dir = asset_dir / f"view_{index:03d}"
    return {
        "dir": view_dir,
        "camera": view_dir / "camera.json",
        "gt_render": view_dir / "gt_input_rgba.png",
        "preprocessed": view_dir / "gt_input_full_canvas_rgb.png",
        "gt_aligned_mesh": view_dir / "gt_training_view_aligned_mesh.npz",
        "gt_normal": view_dir / "gt_normal.png",
        "gt_mask": view_dir / "gt_mask.png",
    }


def resolution_paths(view_dir: Path, pipeline_resolution: int) -> Dict[str, Path]:
    """Prediction artifacts isolated by pipeline resolution."""
    resolution_dir = view_dir / f"r{int(pipeline_resolution)}"
    return {
        "dir": resolution_dir,
        "raw_mesh": resolution_dir / "pixal3d_raw_decoder_mesh.npz",
        "generated_glb": resolution_dir / "pixal3d_generated.glb",
        "generation": resolution_dir / "generation.json",
        "pred_render": resolution_dir / "pixal3d_render_rgba_internal_v2.png",
        "pred_normal": resolution_dir / "pixal3d_normal.png",
        "pred_mask": resolution_dir / "pixal3d_mask.png",
        "angular": resolution_dir / "angular_error.png",
        "comparison": resolution_dir / "comparison.png",
        "metrics": resolution_dir / "metrics.json",
    }


def source_blender_job(
    input_glb: Path,
    normalized_mesh_npz: Path,
    views: Sequence[ViewSpec],
    paths_by_view: Mapping[int, Mapping[str, Path]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "kind": "training_source",
        "input_glb": str(input_glb.resolve()),
        "normalized_mesh_npz": str(normalized_mesh_npz.resolve()),
        "resolution": int(args.render_resolution),
        "samples": int(args.blender_samples),
        "fit_boundary": bool(args.training_fit_boundary),
        "views": [
            {
                "index": view.index,
                "yaw_rad": view.yaw_rad,
                "pitch_rad": view.pitch_rad,
                "initial_radius": view.initial_radius,
                "fov_rad": view.fov_rad,
                "output_png": str(paths_by_view[view.index]["gt_render"].resolve()),
                "camera_json": str(paths_by_view[view.index]["camera"].resolve()),
            }
            for view in views
        ],
    }


def prediction_blender_job(
    input_glb: Path,
    output_png: Path,
    args: argparse.Namespace,
    grid_distance: float,
) -> Dict[str, Any]:
    return {
        "kind": "fixed_glb",
        "input_glb": str(input_glb.resolve()),
        "output_png": str(output_png.resolve()),
        "transform": PIXAL3D_EXPORTED_GLTF_TO_INTERNAL.tolist(),
        "resolution": int(args.render_resolution),
        "fov_rad": float(math.radians(args.fov_deg)),
        "distance": float(grid_distance),
        "samples": int(args.blender_samples),
    }


def silhouette_iou(first: torch.Tensor, second: torch.Tensor) -> float:
    union = first | second
    intersection = first & second
    return float(intersection.sum().item() / max(int(union.sum().item()), 1))


def metric_report_is_current(path: Path, pipeline_resolution: int) -> bool:
    if not path.is_file():
        return False
    report = read_json(path)
    protocol_info = report.get("metrics_protocol", {})
    return (
        protocol_info.get("protocol_version") == PROTOCOL_VERSION
        and int(protocol_info.get("pipeline_resolution", -1)) == int(pipeline_resolution)
        and isinstance(report.get("metrics"), Mapping)
    )


def process_asset(
    input_path: Path,
    input_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    protocol: Pixal3DTrainingProtocol,
    runner: Pixal3DRunner,
    normal_renderer: ChunkedNormalRenderer,
    lpips_evaluator: LPIPSEvaluator,
    blender_helper: Path,
) -> List[Dict[str, Any]]:
    identifier = asset_id(input_path, input_root)
    asset_dir = output_root / identifier
    asset_dir.mkdir(parents=True, exist_ok=True)
    log_path = asset_dir / "run.log"
    normalized_mesh_npz = asset_dir / "source_blender_training_normalized_mesh.npz"
    resolutions = [int(value) for value in args.pipeline_resolutions]
    print(f"\n[asset] {identifier}: {input_path}")

    # Views are sampled exactly once and are shared by all resolutions.
    views, hammersley_offset = protocol.make_views(
        num_views=args.num_views,
        max_elevation_deg=args.max_elevation_deg,
        seed=args.seed,
        fov_deg=args.fov_deg,
        camera_distance=args.camera_distance,
    )
    shared_by_view: Dict[int, Dict[str, Path]] = {}
    prediction_by_view_resolution: Dict[Tuple[int, int], Dict[str, Path]] = {}
    for view in views:
        shared = shared_view_paths(asset_dir, view.index)
        shared["dir"].mkdir(parents=True, exist_ok=True)
        shared_by_view[view.index] = shared
        for pipeline_resolution in resolutions:
            prediction = resolution_paths(shared["dir"], pipeline_resolution)
            prediction["dir"].mkdir(parents=True, exist_ok=True)
            prediction_by_view_resolution[(view.index, pipeline_resolution)] = prediction

    # One source render/camera set for all resolutions.
    source_outputs_exist = normalized_mesh_npz.is_file() and all(
        shared_by_view[view.index]["gt_render"].is_file()
        and shared_by_view[view.index]["camera"].is_file()
        for view in views
    )
    if args.overwrite or not source_outputs_exist:
        run_blender_jobs(
            args.blender,
            blender_helper,
            [source_blender_job(
                input_path, normalized_mesh_npz, views, shared_by_view, args
            )],
            output_root,
            log_path,
        )

    gt_meshes: Dict[int, trimesh.Trimesh] = {}
    camera_by_view: Dict[int, Dict[str, Any]] = {}
    alignment_by_view: Dict[int, Dict[str, Any]] = {}
    for view in views:
        shared = shared_by_view[view.index]
        frame = read_json(shared["camera"])
        gt_mesh, alignment = protocol.view_align_mesh(
            normalized_mesh_npz,
            frame,
            output_npz=shared["gt_aligned_mesh"],
        )
        distance = float(frame.get(
            "radius",
            np.linalg.norm(np.asarray(frame["transform_matrix"], dtype=np.float64)[:3, 3]),
        ))
        mesh_scale = float(alignment["total_scale"])
        grid_distance = float(distance * mesh_scale)
        camera_params = {
            "camera_angle_x": float(frame["camera_angle_x"]),
            "distance": distance,
            "mesh_scale": mesh_scale,
        }
        frame.update({
            "view_index": view.index,
            "seed": view_seed(args.seed, view.index, args.seed_mode),
            "camera_params": camera_params,
            "grid_space_camera_distance": grid_distance,
            "training_alignment": alignment,
            "pipeline_resolutions": resolutions,
            "protocol": "Pixal3D official sphere_normalize_torch + transform_mesh",
            "protocol_version": PROTOCOL_VERSION,
        })
        atomic_json(shared["camera"], frame)
        gt_meshes[view.index] = gt_mesh
        camera_by_view[view.index] = camera_params
        alignment_by_view[view.index] = alignment

    atomic_json(
        asset_dir / "asset.json",
        {
            "asset_id": identifier,
            "protocol_version": PROTOCOL_VERSION,
            "pipeline_resolutions": resolutions,
            "input": str(input_path.resolve()),
            "relative_input": input_path.relative_to(input_root).as_posix(),
            "training_utils": str(protocol.utils_path),
            "normalized_mesh_npz": str(normalized_mesh_npz),
            "hammersley_offset": list(hammersley_offset),
            "view_protocol": {
                "name": "official_sphere_hammersley_sequence_elevation_band",
                "num_views": args.num_views,
                "max_elevation_deg": args.max_elevation_deg,
                "source_normalization": "official Blender unit-cube normalization",
                "view_alignment": "official sphere_normalize_torch + transform_mesh",
                "mesh_scale": "box_scale / sphere_radius, derived per view",
                "shared_across_resolutions": [
                    "view selection", "GT render", "camera matrix", "camera params", "seed"
                ],
                "metric_mesh": "raw decoder mesh",
            },
        },
    )

    # Each resolution receives the exact same PIL condition image, camera and seed.
    for view in views:
        shared = shared_by_view[view.index]
        seed = view_seed(args.seed, view.index, args.seed_mode)
        for pipeline_resolution in resolutions:
            prediction = prediction_by_view_resolution[(view.index, pipeline_resolution)]
            need_generate = (
                args.overwrite
                or not prediction["generated_glb"].is_file()
                or not prediction["raw_mesh"].is_file()
                or not prediction["generation"].is_file()
            )
            if need_generate:
                print(
                    f"[generate] {identifier} view={view.index}/{len(views)-1} "
                    f"res={pipeline_resolution} yaw={view.yaw_deg:.2f} "
                    f"elev={view.elevation_deg:.2f} seed={seed}"
                )
                generation_info = runner.generate(
                    input_png=shared["gt_render"],
                    preprocessed_png=shared["preprocessed"],
                    raw_mesh_npz=prediction["raw_mesh"],
                    output_glb=prediction["generated_glb"],
                    seed=seed,
                    pipeline_resolution=pipeline_resolution,
                    camera_params=camera_by_view[view.index],
                )
                atomic_json(prediction["generation"], generation_info)
            else:
                print(
                    f"[generate] resume view={view.index} res={pipeline_resolution} "
                    f"raw={prediction['raw_mesh']}"
                )

    # Render all generated GLBs, still using the same calibrated camera per view.
    pred_jobs: List[Dict[str, Any]] = []
    for view in views:
        camera_params = camera_by_view[view.index]
        grid_distance = camera_params["distance"] * camera_params["mesh_scale"]
        for pipeline_resolution in resolutions:
            prediction = prediction_by_view_resolution[(view.index, pipeline_resolution)]
            if args.overwrite or not prediction["pred_render"].is_file():
                pred_jobs.append(prediction_blender_job(
                    prediction["generated_glb"],
                    prediction["pred_render"],
                    args,
                    grid_distance,
                ))
    run_blender_jobs(args.blender, blender_helper, pred_jobs, output_root, log_path)

    view_records: List[Dict[str, Any]] = []
    for view in views:
        shared = shared_by_view[view.index]
        seed = view_seed(args.seed, view.index, args.seed_mode)
        camera_params = camera_by_view[view.index]
        grid_distance = float(camera_params["distance"] * camera_params["mesh_scale"])

        pending_resolutions: List[int] = []
        for pipeline_resolution in resolutions:
            prediction = prediction_by_view_resolution[(view.index, pipeline_resolution)]
            base_record = {
                "asset_id": identifier,
                "mesh_name": input_path.stem,
                "relative_input": input_path.relative_to(input_root).as_posix(),
                "view_index": view.index,
                "yaw_deg": view.yaw_deg,
                "elevation_deg": view.elevation_deg,
                "seed": seed,
                "pipeline_resolution": pipeline_resolution,
                "input_glb": str(input_path.resolve()),
                "output_dir": str(prediction["dir"]),
                "generated_glb": str(prediction["generated_glb"]),
                "raw_decoder_mesh": str(prediction["raw_mesh"]),
            }
            if not args.overwrite:
                try:
                    if metric_report_is_current(prediction["metrics"], pipeline_resolution):
                        report = read_json(prediction["metrics"])
                        view_records.append({
                            **base_record,
                            "status": "skipped",
                            "error": None,
                            **report["metrics"],
                        })
                        print(
                            f"[metrics] resume view={view.index} "
                            f"res={pipeline_resolution} {prediction['metrics']}"
                        )
                        continue
                except Exception as exc:
                    print(
                        f"[metrics] invalid existing report view={view.index} "
                        f"res={pipeline_resolution}, recomputing: {exc}"
                    )
            pending_resolutions.append(pipeline_resolution)

        if not pending_resolutions:
            continue

        # Shared GT rasterization and protocol check are computed once per view.
        gt_mesh = gt_meshes[view.index]
        gt_normal, gt_mask, gt_render_stats = normal_renderer.render(gt_mesh, grid_distance)
        gt_beauty, gt_beauty_mask = load_rgba_tensor(
            shared["gt_render"], normal_renderer.device
        )
        protocol_iou = silhouette_iou(gt_mask, gt_beauty_mask)
        if protocol_iou < args.protocol_min_iou:
            error = (
                f"training protocol silhouette IoU {protocol_iou:.6f} is below "
                f"--protocol-min-iou={args.protocol_min_iou:.6f}"
            )
            for pipeline_resolution in pending_resolutions:
                prediction = prediction_by_view_resolution[(view.index, pipeline_resolution)]
                base_record = {
                    "asset_id": identifier,
                    "mesh_name": input_path.stem,
                    "relative_input": input_path.relative_to(input_root).as_posix(),
                    "view_index": view.index,
                    "yaw_deg": view.yaw_deg,
                    "elevation_deg": view.elevation_deg,
                    "seed": seed,
                    "pipeline_resolution": pipeline_resolution,
                    "input_glb": str(input_path.resolve()),
                    "output_dir": str(prediction["dir"]),
                    "generated_glb": str(prediction["generated_glb"]),
                    "raw_decoder_mesh": str(prediction["raw_mesh"]),
                }
                atomic_json(prediction["metrics"], {
                    **base_record, "status": "failed", "error": error
                })
                view_records.append({**base_record, "status": "failed", "error": error})
            if args.fail_fast:
                raise RuntimeError(error)
            del gt_normal, gt_mask, gt_beauty, gt_beauty_mask
            continue

        gt_normal_rgb = normal_to_rgb(gt_normal, gt_mask)
        tensor_to_pil_rgb(gt_normal_rgb).save(shared["gt_normal"])
        mask_to_pil(gt_mask).save(shared["gt_mask"])

        for pipeline_resolution in pending_resolutions:
            prediction = prediction_by_view_resolution[(view.index, pipeline_resolution)]
            base_record = {
                "asset_id": identifier,
                "mesh_name": input_path.stem,
                "relative_input": input_path.relative_to(input_root).as_posix(),
                "view_index": view.index,
                "yaw_deg": view.yaw_deg,
                "elevation_deg": view.elevation_deg,
                "seed": seed,
                "pipeline_resolution": pipeline_resolution,
                "input_glb": str(input_path.resolve()),
                "output_dir": str(prediction["dir"]),
                "generated_glb": str(prediction["generated_glb"]),
                "raw_decoder_mesh": str(prediction["raw_mesh"]),
            }
            started = time.perf_counter()
            try:
                pred_mesh, pred_integrity = load_npz_mesh(
                    prediction["raw_mesh"],
                    label=(
                        f"{identifier}/view_{view.index}/r{pipeline_resolution}/raw_decoder"
                    ),
                )
                pred_normal, pred_mask, pred_render_stats = normal_renderer.render(
                    pred_mesh, grid_distance
                )
                pred_normal_rgb = normal_to_rgb(pred_normal, pred_mask)
                paper_metrics, error_map = normal_paper_metrics(
                    gt_normal,
                    pred_normal,
                    gt_mask,
                    pred_mask,
                    lpips_evaluator,
                    args.boundary_width,
                )
                pred_beauty, pred_beauty_mask = load_rgba_tensor(
                    prediction["pred_render"], normal_renderer.device
                )
                export_iou = silhouette_iou(pred_mask, pred_beauty_mask)
                if export_iou < args.export_min_iou:
                    raise RuntimeError(
                        f"raw-decoder/exported-GLB silhouette IoU {export_iou:.6f} "
                        f"is below --export-min-iou={args.export_min_iou:.6f}"
                    )
                supplementary = rgb_metrics(
                    gt_beauty,
                    pred_beauty,
                    gt_beauty_mask,
                    pred_beauty_mask,
                    lpips_evaluator,
                )
                metrics = {
                    "protocol_gt_blender_vs_nvdiffrast_iou": protocol_iou,
                    "protocol_gt_blender_vs_nvdiffrast_iou_percent": protocol_iou * 100.0,
                    "protocol_raw_decoder_vs_exported_glb_iou": export_iou,
                    "protocol_raw_decoder_vs_exported_glb_iou_percent": export_iou * 100.0,
                    **paper_metrics,
                    **supplementary,
                }

                tensor_to_pil_rgb(pred_normal_rgb).save(prediction["pred_normal"])
                mask_to_pil(pred_mask).save(prediction["pred_mask"])
                overlap = gt_mask & pred_mask
                heatmap = angular_heatmap(error_map, overlap)
                heatmap.save(prediction["angular"])
                save_comparison_sheet(
                    prediction["comparison"],
                    shared["gt_render"],
                    prediction["pred_render"],
                    gt_normal_rgb,
                    pred_normal_rgb,
                    heatmap,
                    metrics,
                )

                report = {
                    **base_record,
                    "status": "success",
                    "camera": read_json(shared["camera"]),
                    "metrics_protocol": {
                        "protocol_version": PROTOCOL_VERSION,
                        "pipeline_resolution": pipeline_resolution,
                        "all_pipeline_resolutions": resolutions,
                        "paired_view_identity": {
                            "view_index": view.index,
                            "yaw_deg": view.yaw_deg,
                            "elevation_deg": view.elevation_deg,
                            "seed": seed,
                            "camera_json": str(shared["camera"]),
                            "condition_image": str(shared["gt_render"]),
                        },
                        "source_render": "official Blender normalization and saved c2w",
                        "gt_alignment": "official sphere_normalize_torch + transform_mesh",
                        "pipeline_preprocess_image": False,
                        "prediction_metric_mesh": "pipeline.run raw decoder mesh",
                        "grid_space_camera_distance": grid_distance,
                        "protocol_min_iou": args.protocol_min_iou,
                        "export_min_iou": args.export_min_iou,
                        "decoder_to_exported_gltf": (
                            PIXAL3D_DECODER_TO_EXPORTED_GLTF.tolist()
                        ),
                        "exported_gltf_to_internal": (
                            PIXAL3D_EXPORTED_GLTF_TO_INTERNAL.tolist()
                        ),
                        "angular_thresholds_deg": list(PAPER_ANGULAR_THRESHOLDS_DEG),
                        "lpips_network": args.lpips_net,
                    },
                    "metrics": metrics,
                    "mesh_integrity": {
                        "gt": alignment_by_view[view.index]["mesh_integrity"],
                        "pred": pred_integrity,
                    },
                    "mesh_stats": {
                        "gt_vertices": int(len(gt_mesh.vertices)),
                        "gt_faces": int(len(gt_mesh.faces)),
                        "pred_vertices": int(len(pred_mesh.vertices)),
                        "pred_faces": int(len(pred_mesh.faces)),
                    },
                    "normal_render_stats": {
                        "gt": gt_render_stats,
                        "pred": pred_render_stats,
                    },
                    "elapsed_seconds": float(time.perf_counter() - started),
                    "shared_artifacts": {
                        key: str(value) for key, value in shared.items() if key != "dir"
                    },
                    "prediction_artifacts": {
                        key: str(value) for key, value in prediction.items() if key != "dir"
                    },
                }
                atomic_json(prediction["metrics"], report)
                view_records.append({
                    **base_record,
                    "status": "success",
                    "error": None,
                    **metrics,
                })
                print(
                    f"[metrics] view={view.index} res={pipeline_resolution} "
                    f"protocol_IoU={protocol_iou*100:.2f}% "
                    f"export_IoU={export_iou*100:.2f}% "
                    f"pred_IoU={metrics['normal_iou_percent']:.2f}% "
                    f"Mean={metrics['normal_mean_angular_error_deg']:.3f}°"
                )
                del pred_mesh, pred_normal, pred_mask, pred_normal_rgb
                del pred_beauty, pred_beauty_mask, error_map, heatmap
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write("\n" + traceback.format_exc() + "\n")
                atomic_json(prediction["metrics"], {
                    **base_record,
                    "status": "failed",
                    "error": error,
                    "traceback": traceback.format_exc(),
                })
                view_records.append({**base_record, "status": "failed", "error": error})
                print(
                    f"[error] {identifier} view={view.index} "
                    f"res={pipeline_resolution}: {error}"
                )
                if args.fail_fast:
                    raise
            finally:
                gc.collect()
                torch.cuda.empty_cache()

        del gt_normal, gt_mask, gt_normal_rgb, gt_beauty, gt_beauty_mask
        gc.collect()
        torch.cuda.empty_cache()

    successful = [
        row for row in view_records if row.get("status") in {"success", "skipped"}
    ]
    by_resolution: Dict[int, List[Dict[str, Any]]] = {}
    for row in successful:
        by_resolution.setdefault(int(row["pipeline_resolution"]), []).append(row)

    summary = {
        "asset_id": identifier,
        "mesh_name": input_path.stem,
        "relative_input": input_path.relative_to(input_root).as_posix(),
        "pipeline_resolutions": resolutions,
        "successful_runs": len(successful),
        "failed_runs": len(view_records) - len(successful),
        "statistics_by_resolution": {
            str(pipeline_resolution): summarize_numeric_rows(rows)
            for pipeline_resolution, rows in sorted(by_resolution.items())
        },
    }
    paired_rows = resolution_pair_rows(successful, resolutions)
    summary["paired_delta_statistics"] = summarize_pair_deltas(paired_rows)
    atomic_json(asset_dir / "summary.json", summary)
    write_csv(asset_dir / "views_all_resolutions.csv", view_records)
    write_csv(asset_dir / "paired_resolution_comparison.csv", paired_rows)

    summary_rows: List[Dict[str, Any]] = []
    for pipeline_resolution, rows in sorted(by_resolution.items()):
        output: Dict[str, Any] = {
            "asset_id": identifier,
            "mesh_name": input_path.stem,
            "relative_input": input_path.relative_to(input_root).as_posix(),
            "pipeline_resolution": pipeline_resolution,
            "n_views": len(rows),
        }
        for metric_name, stats in summarize_numeric_rows(rows).items():
            if isinstance(stats, Mapping):
                for stat_name, value in stats.items():
                    output[f"{metric_name}__{stat_name}"] = value
        summary_rows.append(output)
    write_csv(asset_dir / "summary_by_resolution.csv", summary_rows)
    return view_records


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pixal3d-root", type=Path, default=Path("."))
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-meshes", type=int)
    parser.add_argument("--start-index", type=int, default=0)

    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--max-elevation-deg", type=float, default=30.0)
    parser.add_argument("--target-extent", type=float, default=1.0)
    parser.add_argument("--fov-deg", type=float, default=30.0)
    parser.add_argument("--camera-distance", type=float)
    parser.add_argument("--mesh-scale", type=float, default=None)
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument(
        "--training-fit-boundary",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--blender-samples", type=int, default=64)

    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low-vram", action="store_true")
    resolution_group = parser.add_mutually_exclusive_group()
    resolution_group.add_argument(
        "--pipeline-resolutions",
        type=int,
        nargs="+",
        choices=[1024, 1536],
        help="One or more resolutions evaluated on identical views, e.g. 1024 1536.",
    )
    resolution_group.add_argument(
        "--pipeline-resolution",
        type=int,
        choices=[1024, 1536],
        help="Backward-compatible single-resolution form.",
    )
    parser.add_argument("--max-num-tokens", type=int, default=49152)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-mode", choices=["fixed", "offset"], default="fixed")

    parser.add_argument("--ss-guidance-strength", type=float, default=7.5)
    parser.add_argument("--ss-guidance-rescale", type=float, default=0.7)
    parser.add_argument("--ss-sampling-steps", type=int, default=12)
    parser.add_argument("--ss-rescale-t", type=float, default=5.0)
    parser.add_argument("--shape-guidance-strength", type=float, default=7.5)
    parser.add_argument("--shape-guidance-rescale", type=float, default=0.5)
    parser.add_argument("--shape-sampling-steps", type=int, default=12)
    parser.add_argument("--shape-rescale-t", type=float, default=3.0)
    parser.add_argument("--texture-guidance-strength", type=float, default=1.0)
    parser.add_argument("--texture-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--texture-sampling-steps", type=int, default=12)
    parser.add_argument("--texture-rescale-t", type=float, default=3.0)
    parser.add_argument("--decimation-target", type=int, default=1_000_000)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--extension-webp", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--face-chunk-size", type=int, default=100_000)
    parser.add_argument("--normal-mode", choices=["face", "vertex"], default="vertex")
    parser.add_argument(
        "--orient-normals-to-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--boundary-width", type=int, default=5)
    parser.add_argument("--protocol-min-iou", type=float, default=0.95)
    parser.add_argument("--export-min-iou", type=float, default=0.95)
    parser.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze"], default="vgg")

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.pipeline_resolutions is None:
        args.pipeline_resolutions = [
            int(args.pipeline_resolution) if args.pipeline_resolution is not None else 1536
        ]
    args.pipeline_resolutions = list(dict.fromkeys(
        int(value) for value in args.pipeline_resolutions
    ))

    if args.num_views <= 0:
        parser.error("--num-views must be positive")
    if args.render_resolution <= 0:
        parser.error("--render-resolution must be positive")
    if args.face_chunk_size <= 0:
        parser.error("--face-chunk-size must be positive")
    if not math.isclose(args.target_extent, 1.0, rel_tol=0.0, abs_tol=1e-12):
        parser.error("training protocol requires --target-extent 1.0")
    if args.mesh_scale is not None:
        parser.error("do not set --mesh-scale; it is derived as box_scale / sphere_radius")
    if args.extend_pixel != 0:
        parser.error("training protocol requires --extend-pixel 0")
    if not 1.0 < args.fov_deg < 120.0:
        parser.error("--fov-deg must be in (1,120)")
    if args.camera_distance is not None and args.camera_distance <= 0.0:
        parser.error("--camera-distance must be positive")
    if args.boundary_width < 0:
        parser.error("--boundary-width must be non-negative")
    if not 0.0 <= args.protocol_min_iou <= 1.0:
        parser.error("--protocol-min-iou must be in [0,1]")
    if not 0.0 <= args.export_min_iou <= 1.0:
        parser.error("--export-min-iou must be in [0,1]")
    return args


def main() -> int:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.pixal3d_root = args.pixal3d_root.resolve()
    if not args.input_dir.is_dir():
        raise NotADirectoryError(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_glbs = discover_glbs(args.input_dir, args.recursive)
    glbs = all_glbs[args.start_index:]
    if args.max_meshes is not None:
        glbs = glbs[:args.max_meshes]
    if not glbs:
        raise RuntimeError(f"No GLB files found under {args.input_dir}")

    print(
        f"[setup] meshes={len(glbs)} views={args.num_views} "
        f"render_resolution={args.render_resolution} "
        f"pipeline_resolutions={args.pipeline_resolutions}"
    )
    protocol = Pixal3DTrainingProtocol(args.pixal3d_root)
    print(f"[protocol] official utilities: {protocol.utils_path}")
    blender_helper = ensure_blender_helper(args.output_dir)
    normal_renderer = ChunkedNormalRenderer(
        resolution=args.render_resolution,
        fov_deg=args.fov_deg,
        face_chunk_size=args.face_chunk_size,
        normal_mode=args.normal_mode,
        orient_to_camera=args.orient_normals_to_camera,
        device=args.device,
    )
    lpips_evaluator = LPIPSEvaluator(args.lpips_net, normal_renderer.device)
    runner = Pixal3DRunner(args)

    # Resolution is part of the identity; otherwise 1024 and 1536 overwrite rows.
    row_map: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    prior_path = args.output_dir / "all_views.json"
    if prior_path.is_file() and not args.overwrite:
        try:
            prior = read_json(prior_path)
            for row in prior.get("views", []):
                if not isinstance(row, Mapping):
                    continue
                pipeline_resolution = row.get("pipeline_resolution")
                if pipeline_resolution is None:
                    continue
                key = (
                    str(row.get("asset_id")),
                    int(row.get("view_index", -1)),
                    int(pipeline_resolution),
                )
                row_map[key] = dict(row)
        except Exception as exc:
            print(f"[warning] cannot restore previous all_views.json: {exc}")

    for index, input_path in enumerate(glbs, start=1):
        print(f"[progress] asset {index}/{len(glbs)}")
        try:
            rows = process_asset(
                input_path=input_path,
                input_root=args.input_dir,
                output_root=args.output_dir,
                args=args,
                protocol=protocol,
                runner=runner,
                normal_renderer=normal_renderer,
                lpips_evaluator=lpips_evaluator,
                blender_helper=blender_helper,
            )
            for row in rows:
                row_map[(
                    str(row["asset_id"]),
                    int(row["view_index"]),
                    int(row["pipeline_resolution"]),
                )] = row
        except Exception as exc:
            print(f"[asset-error] {input_path}: {type(exc).__name__}: {exc}")
            with (args.output_dir / "errors.log").open("a", encoding="utf-8") as file:
                file.write(f"\n# {input_path}\n{traceback.format_exc()}\n")
            if args.fail_fast:
                raise

        all_view_rows = [row_map[key] for key in sorted(row_map)]
        write_csv(args.output_dir / "all_views.csv", all_view_rows)
        atomic_json(
            args.output_dir / "all_views.json",
            {"config": vars(args), "views": all_view_rows},
        )
        paired_rows = resolution_pair_rows(all_view_rows, args.pipeline_resolutions)
        write_csv(args.output_dir / "paired_resolution_comparison.csv", paired_rows)
        atomic_json(
            args.output_dir / "paired_resolution_comparison.json",
            {
                "pipeline_resolutions": args.pipeline_resolutions,
                "pairs": paired_rows,
                "delta_statistics": summarize_pair_deltas(paired_rows),
            },
        )

    all_view_rows = [row_map[key] for key in sorted(row_map)]
    successful = [
        row for row in all_view_rows if row.get("status") in {"success", "skipped"}
    ]
    by_asset_resolution: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    by_resolution: Dict[int, List[Dict[str, Any]]] = {}
    for row in successful:
        identifier = str(row["asset_id"])
        pipeline_resolution = int(row["pipeline_resolution"])
        by_asset_resolution.setdefault(
            (identifier, pipeline_resolution), []
        ).append(row)
        by_resolution.setdefault(pipeline_resolution, []).append(row)

    asset_summary_rows: List[Dict[str, Any]] = []
    for (identifier, pipeline_resolution), rows in sorted(by_asset_resolution.items()):
        summary = summarize_numeric_rows(rows)
        output: Dict[str, Any] = {
            "asset_id": identifier,
            "mesh_name": rows[0].get("mesh_name"),
            "relative_input": rows[0].get("relative_input"),
            "pipeline_resolution": pipeline_resolution,
            "n_views": len(rows),
        }
        for metric_name, stats in summary.items():
            if isinstance(stats, Mapping):
                for stat_name, value in stats.items():
                    output[f"{metric_name}__{stat_name}"] = value
        asset_summary_rows.append(output)

    paired_rows = resolution_pair_rows(successful, args.pipeline_resolutions)
    write_csv(args.output_dir / "summary_by_asset_resolution.csv", asset_summary_rows)
    write_csv(args.output_dir / "paired_resolution_comparison.csv", paired_rows)
    atomic_json(
        args.output_dir / "summary_global.json",
        {
            "config": vars(args),
            "pipeline_resolutions": args.pipeline_resolutions,
            "total_assets": len({key[0] for key in by_asset_resolution}),
            "successful_runs": len(successful),
            "failed_runs": len(all_view_rows) - len(successful),
            "statistics_by_resolution": {
                str(pipeline_resolution): summarize_numeric_rows(rows)
                for pipeline_resolution, rows in sorted(by_resolution.items())
            },
            "paired_delta_statistics": summarize_pair_deltas(paired_rows),
            "lpips_error": lpips_evaluator.error,
        },
    )
    print(
        f"[done] assets={len({key[0] for key in by_asset_resolution})} "
        f"successful_runs={len(successful)} "
        f"failed_runs={len(all_view_rows) - len(successful)} "
        f"resolutions={args.pipeline_resolutions} output={args.output_dir}"
    )
    return 1 if len(all_view_rows) != len(successful) else 0


if __name__ == "__main__":
    raise SystemExit(main())
