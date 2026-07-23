#!/usr/bin/env python3
"""Single-image Pixal3D textured reconstruction and aligned-view evaluation.

Workflow
--------
1. Load one input image and run Pixal3D's official image preprocessing.
2. Estimate the aligned camera once with MoGe-2 (or use --fov-rad).
3. Generate one textured GLB for every resolution x seed combination.
4. Undo the complete Pixal3D/O-Voxel export transform in Blender.
5. Render the generated PBR GLB from the same camera under every selected light.
6. Compare every render with the aligned condition image using full-frame
   PSNR, SSIM and LPIPS.
7. Write per-render rows, per-seed light averages, per-resolution averages,
   and a final global-average row to metrics.csv and metrics.json.

Run this script from the Pixal3D repository root. It requires the Pixal3D
runtime, CUDA, MoGe-2, lpips and Blender with glTF import support.

Example
-------
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
python pixal3d_single_image_texture_eval.py \
  --image assets/images/1_img.png \
  --output-dir outputs/single_image_texture_eval \
  --resolutions 1024 1536 \
  --seeds 42 123 \
  --lights studio softbox front uniform \
  --low-vram

Important metric convention
---------------------------
Pixal3D generates a view-aligned object from pipeline.preprocess_image(image).
Therefore metrics are computed against metric_reference_rgb.png, which is the
white-composited, render-resolution version of that aligned condition image.
Comparing against an arbitrary uncropped original canvas would mix image
preprocessing/cropping error into the 3D reconstruction score.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import shlex
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Set backend-related variables before importing torch/Pixal3D.
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
from PIL import Image


# -----------------------------------------------------------------------------
# User-local defaults, matching the supplied inference script.
# -----------------------------------------------------------------------------
MODEL_PATH = "/home/nvme04/yyyan/download/model/Pixal3D"
MOGE_MODEL_NAME = "/home/nvme04/yyyan/download/model/moge-2-vitl/model.pt"
DINOV3_PATH = (
    "/home/nvme04/yyyan/download/model/"
    "dinov3-vitl16-pretrain-lvd1689m/facebook/"
    "dinov3-vitl16-pretrain-lvd1689m"
)

SUPPORTED_RESOLUTIONS = (1024, 1536)
LIGHT_MODES = (
    "studio",
    "three_point",
    "softbox",
    "front",
    "uniform",
    "dramatic",
)

# Official inference sampler values. These are intentionally fixed instead of
# exposing the large collection of generation knobs from the old benchmark.
SS_SAMPLER = {
    "steps": 12,
    "guidance_strength": 7.5,
    "guidance_rescale": 0.7,
    "rescale_t": 5.0,
}
SHAPE_SAMPLER = {
    "steps": 12,
    "guidance_strength": 7.5,
    "guidance_rescale": 0.5,
    "rescale_t": 3.0,
}
TEXTURE_SAMPLER = {
    "steps": 12,
    "guidance_strength": 1.0,
    "guidance_rescale": 0.0,
    "rescale_t": 3.0,
}

# Pixal3D applies this matrix after o_voxel.postprocess.to_glb().
PIXAL3D_EXPORT_ROTATION = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# o_voxel's decoder-space -> glTF-space conversion:
# (x, y, z)_decoder -> (x, z, -y)_gltf.
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


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(json_safe(value), file, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "row_type",
        "status",
        "image_name",
        "pipeline_resolution",
        "pipeline_type",
        "seed",
        "light",
        "psnr_db",
        "ssim",
        "lpips",
        "generated_glb",
        "render_png",
        "error",
    ]
    keys = {key for row in rows for key in row.keys()}
    fields = [key for key in preferred if key in keys]
    fields.extend(sorted(key for key in keys if key not in fields))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    output: Dict[str, Optional[float]] = {}
    for key in ("psnr_db", "ssim", "lpips"):
        values = [
            number
            for row in rows
            if (number := finite_number(row.get(key))) is not None
        ]
        output[key] = float(statistics.fmean(values)) if values else None
    return output


def composite_on_white(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def save_metric_reference(
    condition_image: Image.Image,
    output_path: Path,
    render_resolution: int,
) -> Image.Image:
    reference = composite_on_white(condition_image)
    target_size = (int(render_resolution), int(render_resolution))
    if reference.size != target_size:
        reference = reference.resize(target_size, Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference.save(output_path)
    return reference


# -----------------------------------------------------------------------------
# Pixal3D model and camera setup
# -----------------------------------------------------------------------------
def image_cond_configs(dino_model: str) -> Dict[str, Dict[str, Any]]:
    return {
        "ss": {
            "model_name": dino_model,
            "image_size": 512,
            "grid_resolution": 16,
        },
        "shape_512": {
            "model_name": dino_model,
            "image_size": 512,
            "grid_resolution": 32,
            "use_naf_upsample": True,
            "naf_target_size": 512,
        },
        "shape_1024": {
            "model_name": dino_model,
            "image_size": 1024,
            "grid_resolution": 64,
            "use_naf_upsample": True,
            "naf_target_size": 512,
        },
        "tex_1024": {
            "model_name": dino_model,
            "image_size": 1024,
            "grid_resolution": 64,
            "use_naf_upsample": True,
            "naf_target_size": 1024,
        },
    }


def build_image_cond_model(config: Mapping[str, Any]):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        DinoV3ProjFeatureExtractor,
    )

    model = DinoV3ProjFeatureExtractor(**dict(config))
    model.eval()
    return model


def init_pipeline(
    model_path: str,
    dino_model: str,
    device: str,
    low_vram: bool,
):
    from pixal3d.pipelines import Pixal3DImageTo3DPipeline

    print(f"[pipeline] loading from {model_path}")
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)
    configs = image_cond_configs(dino_model)
    pipeline.image_cond_model_ss = build_image_cond_model(configs["ss"])
    pipeline.image_cond_model_shape_512 = build_image_cond_model(configs["shape_512"])
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(configs["shape_1024"])
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(configs["tex_1024"])

    attributes = (
        "image_cond_model_ss",
        "image_cond_model_shape_512",
        "image_cond_model_shape_1024",
        "image_cond_model_tex_1024",
    )
    if low_vram:
        print("[pipeline] enabling low-VRAM mode")
        for attribute in attributes:
            model = getattr(pipeline, attribute, None)
            if model is not None and getattr(model, "use_naf_upsample", False):
                model._load_naf()
        pipeline._device = torch.device(device)
        pipeline.low_vram = True
    else:
        pipeline.low_vram = False
        if str(device).startswith("cuda"):
            pipeline.cuda()
        else:
            pipeline.to(device)
        for attribute in attributes:
            model = getattr(pipeline, attribute, None)
            if model is None:
                continue
            if str(device).startswith("cuda"):
                model.cuda()
            else:
                model.to(device)
            if getattr(model, "use_naf_upsample", False):
                model._load_naf()
    return pipeline


def load_moge_model(model_name: str, device: str):
    from moge.model.v2 import MoGeModel

    print(f"[moge] loading from {model_name}")
    model = MoGeModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model


def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / math.tan(float(camera_angle_x) / 2.0)
    return float(focal_length * float(resolution) / 32.0)


def distance_from_fov(
    camera_angle_x: float,
    grid_point: torch.Tensor,
    target_point: torch.Tensor,
    mesh_scale: float,
    image_resolution: int,
) -> Dict[str, float]:
    rotation_matrix = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    grid_rotated = grid_point.to(torch.float32) @ rotation_matrix.T
    grid_rotated = grid_rotated / float(mesh_scale) / 2.0
    x_world, y_world = grid_rotated[0].item(), grid_rotated[1].item()
    x_target = float(target_point[0].item())
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = x_target - float(image_resolution) / 2.0
    if abs(x_ndc) < 1e-8:
        raise ValueError("Cannot derive camera distance because x_ndc is zero")
    distance_x = f_pixels * x_world / x_ndc - y_world
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}


@torch.inference_mode()
def estimate_camera_with_moge(
    condition_image: Image.Image,
    moge_model: Any,
    device: str,
    mesh_scale: float,
    extend_pixel: int,
) -> Dict[str, float]:
    rgb = condition_image.convert("RGB")
    width, height = rgb.size
    if width != height:
        print(
            f"[warning] preprocessed image is {width}x{height}; "
            "the aligned Pixal3D camera normally expects a square canvas"
        )
    image_np = np.asarray(rgb, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    output = moge_model.infer(image_tensor)
    intrinsics = output["intrinsics"].squeeze().detach().cpu().numpy()
    fx_normalized = float(intrinsics[0, 0])
    fx = fx_normalized * float(width)
    if not math.isfinite(fx) or fx <= 0.0:
        raise ValueError(f"MoGe returned invalid focal length: {fx}")
    camera_angle_x = float(2.0 * math.atan(width / (2.0 * fx)))
    image_resolution = int(width)
    distance = distance_from_fov(
        camera_angle_x=camera_angle_x,
        grid_point=torch.tensor([-1.0, 0.0, 0.0]),
        target_point=torch.tensor(
            [0 - int(extend_pixel), image_resolution - 1 + int(extend_pixel)]
        ),
        mesh_scale=mesh_scale,
        image_resolution=image_resolution,
    )["distance_from_x"]
    return {
        "camera_angle_x": camera_angle_x,
        "distance": float(distance),
        "mesh_scale": float(mesh_scale),
        "source": "moge2",
        "condition_width": int(width),
        "condition_height": int(height),
        "fx_normalized": fx_normalized,
        "fx_pixels": float(fx),
    }


def manual_camera_params(
    fov_rad: float,
    condition_image: Image.Image,
    mesh_scale: float,
    extend_pixel: int,
) -> Dict[str, float]:
    width = int(condition_image.size[0])
    distance = distance_from_fov(
        camera_angle_x=float(fov_rad),
        grid_point=torch.tensor([-1.0, 0.0, 0.0]),
        target_point=torch.tensor([0 - int(extend_pixel), width - 1 + int(extend_pixel)]),
        mesh_scale=mesh_scale,
        image_resolution=width,
    )["distance_from_x"]
    return {
        "camera_angle_x": float(fov_rad),
        "distance": float(distance),
        "mesh_scale": float(mesh_scale),
        "source": "manual_fov",
        "condition_width": int(condition_image.size[0]),
        "condition_height": int(condition_image.size[1]),
    }


class Pixal3DGenerator:
    def __init__(self, pipeline: Any, args: argparse.Namespace):
        self.pipeline = pipeline
        self.args = args

    @staticmethod
    def _sharded_uv_unwrap(
        mesh,
        compute_charts_kwargs=None,
        xatlas_compute_charts_kwargs=None,
        xatlas_pack_charts_kwargs=None,
        return_vmaps=False,
        verbose=False,
        max_faces_per_shard=10_000,
    ):
        """Feed balanced, lossless face shards to xatlas.

        CuMesh normally calls ``Atlas.add_mesh`` once per GPU-generated chart.
        Raw Pixal3D meshes can contain hundreds of thousands of tiny charts,
        making those Python/C++ calls a dominant serial cost.  Adding every
        chart as one mesh also leaves a long single-core tail when one chart is
        much larger than the others.  Contiguous face shards keep all triangles
        while bounding the largest independent xatlas job so its native worker
        pool can process multiple shards concurrently.
        """
        from cumesh.xatlas import Atlas
        from tqdm import tqdm

        compute_charts_kwargs = dict(compute_charts_kwargs or {})
        xatlas_compute_charts_kwargs = dict(xatlas_compute_charts_kwargs or {})
        xatlas_pack_charts_kwargs = dict(xatlas_pack_charts_kwargs or {})
        xatlas_compute_charts_kwargs["verbose"] = bool(verbose)
        xatlas_pack_charts_kwargs["verbose"] = bool(verbose)

        mesh.remove_degenerate_faces()
        clustering_started = time.perf_counter()
        mesh.compute_charts(**compute_charts_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(
            f"[uv-sharded] gpu_clustering_seconds="
            f"{time.perf_counter() - clustering_started:.3f}"
        )

        new_vertices, _ = mesh.read()
        (
            num_charts,
            _,
            chart_vmap,
            chart_faces,
            _,
            chart_face_offset,
        ) = mesh.read_atlas_charts()
        face_counts = (chart_face_offset[1:] - chart_face_offset[:-1]).cpu()
        if face_counts.numel():
            quantiles = torch.quantile(
                face_counts.float(),
                torch.tensor([0.5, 0.9, 0.99]),
            ).tolist()
            print(
                f"[uv-sharded] clusters={int(num_charts):,} "
                f"faces_per_cluster median={quantiles[0]:.0f} "
                f"p90={quantiles[1]:.0f} p99={quantiles[2]:.0f} "
                f"max={int(face_counts.max()):,}"
            )

        transfer_started = time.perf_counter()
        new_vertices_cpu = new_vertices.cpu().contiguous()
        chart_faces = chart_faces.cpu().contiguous()
        chart_vmap = chart_vmap.cpu().contiguous()
        max_faces_per_shard = int(max_faces_per_shard)
        if max_faces_per_shard <= 0:
            raise ValueError("max_faces_per_shard must be positive")

        atlas = Atlas()
        input_vmaps = []
        num_faces = int(chart_faces.shape[0])
        shard_ranges = list(range(0, num_faces, max_faces_per_shard))
        for start in tqdm(
            shard_ranges,
            desc="Preparing balanced xatlas shards",
            disable=not verbose,
        ):
            end = min(start + max_faces_per_shard, num_faces)
            shard_chart_faces = chart_faces[start:end]
            chart_vertex_ids, inverse = torch.unique(
                shard_chart_faces.reshape(-1),
                sorted=True,
                return_inverse=True,
            )
            shard_faces = inverse.reshape(-1, 3).to(torch.int32).contiguous()
            shard_vmap = chart_vmap[chart_vertex_ids.long()].contiguous()
            shard_vertices = new_vertices_cpu[shard_vmap.long()].contiguous()
            atlas.add_mesh(shard_vertices * 1024.0, shard_faces)
            input_vmaps.append(shard_vmap)
        print(
            f"[uv-sharded] shards={len(input_vmaps):,} "
            f"max_faces_per_shard={max_faces_per_shard:,} "
            f"prepare_cpu_seconds={time.perf_counter() - transfer_started:.3f}"
        )

        xatlas_started = time.perf_counter()
        compute_started = time.perf_counter()
        atlas.compute_charts(**xatlas_compute_charts_kwargs)
        compute_seconds = time.perf_counter() - compute_started
        print(f"[uv-sharded] xatlas_compute_seconds={compute_seconds:.3f}")
        pack_started = time.perf_counter()
        atlas.pack_charts(**xatlas_pack_charts_kwargs)
        pack_seconds = time.perf_counter() - pack_started
        print(f"[uv-sharded] xatlas_pack_seconds={pack_seconds:.3f}")
        print(
            f"[uv-sharded] xatlas_seconds="
            f"{time.perf_counter() - xatlas_started:.3f}"
        )

        vmaps = []
        faces = []
        uvs = []
        vertex_offset = 0
        for index, input_vmap in enumerate(
            tqdm(
                input_vmaps,
                desc="Gathering balanced xatlas shards",
                disable=not verbose,
            )
        ):
            vmap, shard_faces, shard_uvs = atlas.get_mesh(index)
            vmaps.append(input_vmap[vmap.long()])
            faces.append(shard_faces + vertex_offset)
            uvs.append(shard_uvs)
            vertex_offset += int(vmap.shape[0])

        out_vmaps = torch.cat(vmaps, dim=0)
        out_faces = torch.cat(faces, dim=0)
        out_uvs = torch.cat(uvs, dim=0)
        if int(out_faces.shape[0]) != num_faces:
            raise RuntimeError(
                f"xatlas changed face count: expected {num_faces}, "
                f"got {int(out_faces.shape[0])}"
            )
        out_vertices = new_vertices_cpu[out_vmaps.long()]
        output = [out_vertices, out_faces, out_uvs]
        if return_vmaps:
            output.append(out_vmaps)
        return tuple(output)

    @torch.inference_mode()
    def generate(
        self,
        condition_image: Image.Image,
        camera_params: Mapping[str, float],
        resolution: int,
        seed: int,
        output_glb: Path,
    ) -> Dict[str, Any]:
        import o_voxel

        started = time.perf_counter()
        resolution = int(resolution)
        if resolution not in SUPPORTED_RESOLUTIONS:
            raise ValueError(f"Unsupported Pixal3D resolution: {resolution}")
        pipeline_type = f"{resolution}_cascade"
        set_seed(int(seed))
        print(
            f"[generate] pipeline_type={pipeline_type} seed={seed} "
            f"output={output_glb}"
        )
        pipeline_started = time.perf_counter()
        mesh_list, latent_bundle = self.pipeline.run(
            condition_image,
            camera_params=dict(camera_params),
            seed=int(seed),
            sparse_structure_sampler_params=dict(SS_SAMPLER),
            shape_slat_sampler_params=dict(SHAPE_SAMPLER),
            tex_slat_sampler_params=dict(TEXTURE_SAMPLER),
            preprocess_image=False,
            return_latent=True,
            pipeline_type=pipeline_type,
            max_num_tokens=int(self.args.max_num_tokens),
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pipeline_seconds = time.perf_counter() - pipeline_started
        shape_slat, tex_slat, grid_resolution = latent_bundle
        mesh = mesh_list[0]
        print(
            f"[decoder] vertices={int(mesh.vertices.shape[0]):,} "
            f"faces={int(mesh.faces.shape[0]):,} grid={int(grid_resolution)} "
            f"pipeline_seconds={pipeline_seconds:.3f}"
        )

        export_kwargs: Dict[str, Any] = {
            "vertices": mesh.vertices,
            "faces": mesh.faces,
            "attr_volume": mesh.attrs,
            "coords": mesh.coords,
            "attr_layout": self.pipeline.pbr_attr_layout,
            "grid_size": grid_resolution,
            "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            "decimation_target": int(self.args.decimation_target),
            "texture_size": int(self.args.texture_size),
            "remesh": bool(self.args.remesh),
            "use_tqdm": True,
            "verbose": True,
        }
        if self.args.remesh:
            export_kwargs.update(remesh_band=1, remesh_project=0)

        export_started = time.perf_counter()
        original_uv_unwrap = None
        if self.args.uv_mode == "sharded":
            import cumesh

            original_uv_unwrap = cumesh.CuMesh.uv_unwrap

            def sharded_uv_unwrap(mesh_instance, *unwrap_args, **unwrap_kwargs):
                return self._sharded_uv_unwrap(
                    mesh_instance,
                    *unwrap_args,
                    **unwrap_kwargs,
                    max_faces_per_shard=int(self.args.uv_shard_faces),
                )

            cumesh.CuMesh.uv_unwrap = sharded_uv_unwrap
        try:
            glb_scene = o_voxel.postprocess.to_glb(**export_kwargs)
        finally:
            if original_uv_unwrap is not None:
                cumesh.CuMesh.uv_unwrap = original_uv_unwrap
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        export_seconds = time.perf_counter() - export_started
        print(f"[export] to_glb_seconds={export_seconds:.3f}")
        glb_scene.apply_transform(PIXAL3D_EXPORT_ROTATION)
        output_glb.parent.mkdir(parents=True, exist_ok=True)
        glb_write_started = time.perf_counter()
        glb_scene.export(
            str(output_glb),
            extension_webp=bool(self.args.extension_webp),
        )
        glb_write_seconds = time.perf_counter() - glb_write_started
        print(f"[export] glb_write_seconds={glb_write_seconds:.3f}")

        metadata = {
            "status": "success",
            "pipeline_resolution": resolution,
            "pipeline_type": pipeline_type,
            "seed": int(seed),
            "grid_resolution": int(grid_resolution),
            "camera_params": dict(camera_params),
            "decoder_vertices": int(mesh.vertices.shape[0]),
            "decoder_faces": int(mesh.faces.shape[0]),
            "decimation_target": int(self.args.decimation_target),
            "texture_size": int(self.args.texture_size),
            "remesh": bool(self.args.remesh),
            "uv_mode": str(self.args.uv_mode),
            "uv_shard_faces": int(self.args.uv_shard_faces),
            "extension_webp": bool(self.args.extension_webp),
            "pipeline_seconds": float(pipeline_seconds),
            "to_glb_seconds": float(export_seconds),
            "glb_write_seconds": float(glb_write_seconds),
            "elapsed_seconds": float(time.perf_counter() - started),
            "output_glb": str(output_glb),
        }

        del glb_scene, mesh, mesh_list, latent_bundle, shape_slat, tex_slat
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metadata


# -----------------------------------------------------------------------------
# Blender PBR renderer with selectable light rigs
# -----------------------------------------------------------------------------
BLENDER_HELPER_SOURCE = r'''
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.images,
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


def configure_render(resolution, samples):
    scene = bpy.context.scene
    engine = set_engine(scene)
    scene.render.resolution_x = int(resolution)
    scene.render.resolution_y = int(resolution)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.film_transparent = True
    scene.render.use_file_extension = True
    if engine.startswith('BLENDER_EEVEE') and hasattr(scene, 'eevee'):
        if hasattr(scene.eevee, 'taa_render_samples'):
            scene.eevee.taa_render_samples = int(samples)
        if hasattr(scene.eevee, 'use_gtao'):
            scene.eevee.use_gtao = True
        if hasattr(scene.eevee, 'gtao_distance'):
            scene.eevee.gtao_distance = 3.0
        if hasattr(scene.eevee, 'gtao_factor'):
            scene.eevee.gtao_factor = 1.1
    elif engine == 'CYCLES':
        scene.cycles.samples = int(samples)
        scene.cycles.use_denoising = True
    try:
        scene.display_settings.display_device = 'sRGB'
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'
    except Exception:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    world = scene.world or bpy.data.worlds.new('World')
    scene.world = world
    world.use_nodes = True
    return scene


def set_world(scene, strength, color=(1.0, 1.0, 1.0, 1.0)):
    world = scene.world
    world.use_nodes = True
    background = world.node_tree.nodes.get('Background')
    if background is not None:
        background.inputs['Color'].default_value = color
        background.inputs['Strength'].default_value = float(strength)


def import_glb(path):
    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.gltf(
            filepath=str(path),
            merge_vertices=True,
            import_shading='NORMALS',
        )
    except TypeError:
        bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj for obj in bpy.data.objects if obj not in before]
    roots = [obj for obj in imported if obj.parent is None]
    if not any(obj.type == 'MESH' for obj in imported):
        raise RuntimeError('GLB contains no mesh objects')
    return imported, roots


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
    bpy.context.view_layer.update()


def add_area(scene, name, location, energy, size, color=(1.0, 1.0, 1.0)):
    data = bpy.data.lights.new(name=name, type='AREA')
    data.energy = float(energy)
    data.shape = 'DISK'
    data.size = float(size)
    data.color = color
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = (-obj.location).to_track_quat('-Z', 'Y').to_euler()
    return obj


def add_light_rig(scene, mode):
    if mode == 'studio':
        set_world(scene, 0.22)
        add_area(scene, 'Key', (-3.5, -4.5, 4.5), 850.0, 4.5)
        add_area(scene, 'Fill', (4.0, -2.5, 2.0), 430.0, 5.0)
        add_area(scene, 'Top', (0.0, 0.5, 5.0), 300.0, 4.0)
        add_area(scene, 'Rim', (0.5, 4.0, 3.0), 300.0, 3.0)
    elif mode == 'three_point':
        set_world(scene, 0.12)
        add_area(scene, 'Key', (-3.0, -4.0, 3.5), 1000.0, 3.0)
        add_area(scene, 'Fill', (3.5, -2.5, 1.5), 350.0, 4.0)
        add_area(scene, 'Rim', (1.0, 4.0, 3.0), 650.0, 2.5)
    elif mode == 'softbox':
        set_world(scene, 0.30)
        add_area(scene, 'SoftboxLeft', (-3.5, -3.5, 3.0), 650.0, 6.0)
        add_area(scene, 'SoftboxRight', (3.5, -3.0, 2.5), 600.0, 6.0)
        add_area(scene, 'SoftboxTop', (0.0, 0.0, 5.0), 250.0, 5.0)
    elif mode == 'front':
        set_world(scene, 0.15)
        add_area(scene, 'Front', (0.0, -4.5, 0.8), 1000.0, 5.0)
        add_area(scene, 'FrontTop', (0.0, -2.5, 4.0), 250.0, 4.0)
    elif mode == 'uniform':
        set_world(scene, 0.35)
        positions = (
            (0.0, -4.0, 0.0),
            (0.0, 4.0, 0.0),
            (-4.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (0.0, 0.0, 4.0),
            (0.0, 0.0, -4.0),
        )
        for index, position in enumerate(positions):
            add_area(scene, 'Uniform%02d' % index, position, 230.0, 4.5)
    elif mode == 'dramatic':
        set_world(scene, 0.035)
        add_area(scene, 'HardKey', (-3.0, -3.5, 4.5), 1250.0, 1.5)
        add_area(scene, 'CoolRim', (2.5, 4.0, 3.5), 900.0, 2.0, (0.65, 0.75, 1.0))
        add_area(scene, 'WarmFill', (3.5, -1.5, 0.5), 120.0, 3.0, (1.0, 0.72, 0.55))
    else:
        raise ValueError('unsupported light mode: %s' % mode)


def create_aligned_camera(scene, distance, fov_rad):
    camera_data = bpy.data.cameras.new('Camera')
    camera = bpy.data.objects.new('Camera', camera_data)
    scene.collection.objects.link(camera)
    # Pixal3D internal front view represented in Blender coordinates:
    # camera at (0, -distance, 0), looking toward +Y, with +Z as image up.
    camera.matrix_world = Matrix((
        (1.0, 0.0,  0.0, 0.0),
        (0.0, 0.0, -1.0, -float(distance)),
        (0.0, 1.0,  0.0, 0.0),
        (0.0, 0.0,  0.0, 1.0),
    ))
    camera_data.type = 'PERSP'
    camera_data.sensor_fit = 'HORIZONTAL'
    camera_data.sensor_width = 32.0
    camera_data.sensor_height = 32.0
    camera_data.lens = 16.0 / math.tan(float(fov_rad) / 2.0)
    camera_data.clip_start = 0.01
    camera_data.clip_end = 100.0
    scene.camera = camera
    return camera


def render_job(job):
    clear_scene()
    scene = configure_render(job['resolution'], job.get('samples', 64))
    _, roots = import_glb(job['input_glb'])
    apply_root_transform(roots, job['transform'])
    create_aligned_camera(scene, job['distance'], job['fov_rad'])
    add_light_rig(scene, job['light_mode'])
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
        print(
            '[blender] job %d/%d light=%s input=%s' % (
                index + 1,
                len(jobs),
                job['light_mode'],
                job['input_glb'],
            )
        )
        render_job(job)


if __name__ == '__main__':
    main()
'''


def ensure_blender_helper(output_dir: Path) -> Path:
    helper_path = output_dir / "_pixal3d_blender_texture_render.py"
    if (
        not helper_path.is_file()
        or helper_path.read_text(encoding="utf-8") != BLENDER_HELPER_SOURCE
    ):
        helper_path.write_text(BLENDER_HELPER_SOURCE, encoding="utf-8")
    return helper_path


def run_blender_jobs(
    blender_executable: str,
    helper_path: Path,
    jobs: Sequence[Mapping[str, Any]],
    output_dir: Path,
    log_path: Path,
) -> None:
    if not jobs:
        print("[render] all requested renders already exist")
        return
    jobs_path = output_dir / f"_blender_jobs_{int(time.time() * 1_000_000)}.json"
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
    print(f"[render] {shlex.join(command)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH"):
        environment.pop(key, None)
    ld_library_path = environment.get("LD_LIBRARY_PATH", "")
    if "conda" in ld_library_path.lower() or "miniconda" in ld_library_path.lower():
        environment.pop("LD_LIBRARY_PATH", None)
    environment.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n# Blender command: {shlex.join(command)}\n")
        process = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=environment,
        )
    try:
        jobs_path.unlink()
    except OSError:
        pass
    if process.returncode != 0:
        raise RuntimeError(
            f"Blender failed with exit code {process.returncode}; see {log_path}"
        )
    missing = [Path(job["output_png"]) for job in jobs if not Path(job["output_png"]).is_file()]
    if missing:
        raise RuntimeError(f"Blender finished but outputs are missing: {missing}")


# -----------------------------------------------------------------------------
# PSNR, SSIM and LPIPS
# -----------------------------------------------------------------------------
def load_metric_tensor(path: Path, size: Tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = composite_on_white(image)
    if rgb.size != size:
        rgb = rgb.resize(size, Image.Resampling.LANCZOS)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def psnr_metric(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    mse = float(F.mse_loss(prediction, reference).item())
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def gaussian_kernel(
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinates = torch.arange(window_size, device=device, dtype=dtype)
    coordinates = coordinates - (window_size - 1) / 2.0
    kernel_1d = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return kernel_1d[:, None] * kernel_1d[None, :]


def ssim_metric(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    x = reference.unsqueeze(0).float()
    y = prediction.unsqueeze(0).float()
    channels = int(x.shape[1])
    kernel = gaussian_kernel(window_size, sigma, x.device, x.dtype)
    kernel = kernel[None, None].expand(channels, 1, window_size, window_size)
    padding = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=channels)
    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.conv2d(x * x, kernel, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=channels) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = (
        (2.0 * mu_xy + c1)
        * (2.0 * sigma_xy + c2)
        / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2) + 1e-12)
    )
    return float(ssim_map.mean().item())


class LPIPSEvaluator:
    def __init__(self, network: str, device: torch.device):
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError(
                "The lpips package is required. Install it with: pip install lpips"
            ) from exc
        self.device = device
        self.model = lpips.LPIPS(net=network).eval().to(device)

    @torch.inference_mode()
    def evaluate(self, reference: torch.Tensor, prediction: torch.Tensor) -> float:
        x = reference.unsqueeze(0).to(self.device)
        y = prediction.unsqueeze(0).to(self.device)
        value = self.model(x * 2.0 - 1.0, y * 2.0 - 1.0)
        return float(value.mean().item())


def evaluate_render(
    reference_cpu: torch.Tensor,
    prediction_path: Path,
    lpips_evaluator: LPIPSEvaluator,
) -> Dict[str, float]:
    height, width = int(reference_cpu.shape[1]), int(reference_cpu.shape[2])
    prediction_cpu = load_metric_tensor(prediction_path, (width, height))
    # PSNR/SSIM are inexpensive and stay on CPU. LPIPS uses the selected device.
    metrics = {
        "psnr_db": psnr_metric(reference_cpu, prediction_cpu),
        "ssim": ssim_metric(reference_cpu, prediction_cpu),
        "lpips": lpips_evaluator.evaluate(reference_cpu, prediction_cpu),
    }
    del prediction_cpu
    return metrics


# -----------------------------------------------------------------------------
# Output paths and summaries
# -----------------------------------------------------------------------------
def run_paths(
    output_dir: Path,
    image_stem: str,
    resolution: int,
    seed: int,
) -> Dict[str, Path]:
    directory = output_dir / f"r{int(resolution)}" / f"seed_{int(seed)}"
    base = f"{image_stem}__r{int(resolution)}__seed{int(seed)}"
    return {
        "dir": directory,
        "glb": directory / f"{base}.glb",
        "generation": directory / "generation.json",
    }


def render_path(
    run_directory: Path,
    image_stem: str,
    resolution: int,
    seed: int,
    light: str,
) -> Path:
    # Starts with the light name, then image identity, resolution and seed.
    return run_directory / (
        f"{light}__{image_stem}__r{int(resolution)}__seed{int(seed)}.png"
    )


def build_summary_rows(
    render_rows: Sequence[Mapping[str, Any]],
    image_name: str,
    resolutions: Sequence[int],
    seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    successful = [
        row
        for row in render_rows
        if row.get("row_type") == "RENDER" and row.get("status") == "success"
    ]
    summaries: List[Dict[str, Any]] = []

    for resolution in resolutions:
        for seed in seeds:
            group = [
                row
                for row in successful
                if int(row["pipeline_resolution"]) == int(resolution)
                and int(row["seed"]) == int(seed)
            ]
            if not group:
                continue
            summaries.append(
                {
                    "row_type": "MEAN_LIGHTS",
                    "status": "success",
                    "image_name": image_name,
                    "pipeline_resolution": int(resolution),
                    "pipeline_type": f"{int(resolution)}_cascade",
                    "seed": int(seed),
                    "light": "AVERAGE_LIGHTS",
                    **mean_metrics(group),
                    "n_renders": len(group),
                }
            )

    for resolution in resolutions:
        group = [
            row
            for row in successful
            if int(row["pipeline_resolution"]) == int(resolution)
        ]
        if not group:
            continue
        summaries.append(
            {
                "row_type": "MEAN_RESOLUTION",
                "status": "success",
                "image_name": image_name,
                "pipeline_resolution": int(resolution),
                "pipeline_type": f"{int(resolution)}_cascade",
                "seed": "AVERAGE_SEEDS",
                "light": "AVERAGE_LIGHTS",
                **mean_metrics(group),
                "n_renders": len(group),
            }
        )

    # Deliberately append this last so the final CSV line is the requested mean.
    if successful:
        summaries.append(
            {
                "row_type": "GLOBAL_AVERAGE",
                "status": "success",
                "image_name": image_name,
                "pipeline_resolution": "AVERAGE_RESOLUTIONS",
                "pipeline_type": "AVERAGE_RESOLUTIONS",
                "seed": "AVERAGE_SEEDS",
                "light": "AVERAGE_LIGHTS",
                **mean_metrics(successful),
                "n_renders": len(successful),
            }
        )
    return summaries


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--moge-model", default=MOGE_MODEL_NAME)
    parser.add_argument("--dino-model", default=DINOV3_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low-vram", action="store_true")

    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        choices=SUPPORTED_RESOLUTIONS,
        default=[1024, 1536],
        help="Pixal3D cascade resolutions to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="One or more generation seeds.",
    )
    parser.add_argument(
        "--lights",
        nargs="+",
        choices=LIGHT_MODES,
        default=["studio"],
        help="One or more Blender PBR light rigs.",
    )

    parser.add_argument(
        "--fov-rad",
        type=float,
        default=-1.0,
        help="Manual horizontal FOV in radians. <=0 uses MoGe-2.",
    )
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=1_000_000,
        help="Maximum sparse tokens passed to pipeline.run.",
    )

    parser.add_argument("--decimation-target", type=int, default=1_000_000)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument(
        "--uv-mode",
        choices=("upstream", "sharded"),
        default="sharded",
        help="Use upstream per-chart xatlas calls or balanced lossless shards.",
    )
    parser.add_argument(
        "--uv-shard-faces",
        type=int,
        default=10_000,
        help="Maximum faces in each xatlas input when --uv-mode=sharded.",
    )
    parser.add_argument(
        "--remesh",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the official inference GLB remesh path by default.",
    )
    parser.add_argument(
        "--extension-webp",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--blender-samples", type=int, default=64)
    parser.add_argument(
        "--lpips-net",
        choices=["alex", "vgg", "squeeze"],
        default="vgg",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    args.resolutions = list(dict.fromkeys(int(value) for value in args.resolutions))
    args.seeds = list(dict.fromkeys(int(value) for value in args.seeds))
    args.lights = list(dict.fromkeys(str(value) for value in args.lights))

    if args.mesh_scale <= 0.0:
        parser.error("--mesh-scale must be positive")
    if args.render_resolution <= 0:
        parser.error("--render-resolution must be positive")
    if args.max_num_tokens <= 0:
        parser.error("--max-num-tokens must be positive")
    if args.decimation_target <= 0:
        parser.error("--decimation-target must be positive")
    if args.texture_size <= 0:
        parser.error("--texture-size must be positive")
    if args.uv_shard_faces <= 0:
        parser.error("--uv-shard-faces must be positive")
    if args.blender_samples <= 0:
        parser.error("--blender-samples must be positive")
    if args.fov_rad > 0.0 and not 0.01 < args.fov_rad < math.pi - 0.01:
        parser.error("--fov-rad must be a plausible angle in radians")
    return args


def main() -> int:
    args = parse_args()
    args.image = args.image.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "run.log"
    image_stem = args.image.stem

    print(
        f"[setup] image={args.image} resolutions={args.resolutions} "
        f"seeds={args.seeds} lights={args.lights}"
    )

    # Load the Pixal3D pipeline once and use its official preprocessing.
    pipeline = init_pipeline(
        model_path=args.model_path,
        dino_model=args.dino_model,
        device=args.device,
        low_vram=args.low_vram,
    )
    with Image.open(args.image) as source:
        source_copy = source.convert("RGBA")
    source_copy.save(args.output_dir / "input_original.png")
    condition_image = pipeline.preprocess_image(source_copy)
    condition_path = args.output_dir / "input_preprocessed_rgba.png"
    condition_image.save(condition_path)
    reference_path = args.output_dir / "metric_reference_rgb.png"
    reference_image = save_metric_reference(
        condition_image,
        reference_path,
        args.render_resolution,
    )

    # Estimate one camera and reuse it for every resolution, seed and light.
    if args.fov_rad > 0.0:
        camera_params = manual_camera_params(
            fov_rad=args.fov_rad,
            condition_image=condition_image,
            mesh_scale=args.mesh_scale,
            extend_pixel=args.extend_pixel,
        )
    else:
        moge_model = load_moge_model(args.moge_model, args.device)
        camera_params = estimate_camera_with_moge(
            condition_image=condition_image,
            moge_model=moge_model,
            device=args.device,
            mesh_scale=args.mesh_scale,
            extend_pixel=args.extend_pixel,
        )
        try:
            moge_model.cpu()
        except Exception:
            pass
        del moge_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # pipeline.run expects only the three numeric camera fields.
    pipeline_camera_params = {
        "camera_angle_x": float(camera_params["camera_angle_x"]),
        "distance": float(camera_params["distance"]),
        "mesh_scale": float(camera_params["mesh_scale"]),
    }
    grid_space_camera_distance = float(
        pipeline_camera_params["distance"] * pipeline_camera_params["mesh_scale"]
    )
    camera_record = {
        **camera_params,
        "pipeline_camera_params": pipeline_camera_params,
        "grid_space_camera_distance": grid_space_camera_distance,
        "fov_degrees": math.degrees(pipeline_camera_params["camera_angle_x"]),
        "reference_image": str(reference_path),
    }
    atomic_json(args.output_dir / "camera.json", camera_record)
    print(
        f"[camera] source={camera_params['source']} "
        f"fov={camera_record['fov_degrees']:.3f}deg "
        f"distance={pipeline_camera_params['distance']:.6f} "
        f"grid_distance={grid_space_camera_distance:.6f}"
    )

    generator = Pixal3DGenerator(pipeline, args)
    generation_records: List[Dict[str, Any]] = []
    run_index: Dict[Tuple[int, int], Dict[str, Path]] = {}

    for resolution in args.resolutions:
        for seed in args.seeds:
            paths = run_paths(args.output_dir, image_stem, resolution, seed)
            paths["dir"].mkdir(parents=True, exist_ok=True)
            run_index[(int(resolution), int(seed))] = paths
            if paths["glb"].is_file() and paths["generation"].is_file() and not args.overwrite:
                print(f"[generate] resume {paths['glb']}")
                with paths["generation"].open("r", encoding="utf-8") as file:
                    generation_records.append(json.load(file))
                continue
            try:
                metadata = generator.generate(
                    condition_image=condition_image,
                    camera_params=pipeline_camera_params,
                    resolution=resolution,
                    seed=seed,
                    output_glb=paths["glb"],
                )
                atomic_json(paths["generation"], metadata)
                generation_records.append(metadata)
            except Exception as exc:
                error_record = {
                    "status": "failed",
                    "pipeline_resolution": int(resolution),
                    "pipeline_type": f"{int(resolution)}_cascade",
                    "seed": int(seed),
                    "output_glb": str(paths["glb"]),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                atomic_json(paths["generation"], error_record)
                generation_records.append(error_record)
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write("\n" + traceback.format_exc() + "\n")
                print(f"[generation-error] {error_record['error']}")
                if args.fail_fast:
                    raise

    # Generation is complete; free the very large pipeline before LPIPS.
    del generator, pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    blender_helper = ensure_blender_helper(args.output_dir)
    render_jobs: List[Dict[str, Any]] = []
    for resolution in args.resolutions:
        for seed in args.seeds:
            paths = run_index[(int(resolution), int(seed))]
            if not paths["glb"].is_file():
                continue
            for light in args.lights:
                output_png = render_path(
                    paths["dir"], image_stem, resolution, seed, light
                )
                if output_png.is_file() and not args.overwrite:
                    continue
                render_jobs.append(
                    {
                        "input_glb": str(paths["glb"]),
                        "output_png": str(output_png),
                        "transform": PIXAL3D_EXPORTED_GLTF_TO_INTERNAL.tolist(),
                        "resolution": int(args.render_resolution),
                        "fov_rad": float(pipeline_camera_params["camera_angle_x"]),
                        "distance": grid_space_camera_distance,
                        "samples": int(args.blender_samples),
                        "light_mode": light,
                    }
                )

    try:
        run_blender_jobs(
            blender_executable=args.blender,
            helper_path=blender_helper,
            jobs=render_jobs,
            output_dir=args.output_dir,
            log_path=log_path,
        )
    except Exception:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("\n" + traceback.format_exc() + "\n")
        if args.fail_fast:
            raise
        print(f"[render-error] {traceback.format_exc().splitlines()[-1]}")

    reference_cpu = load_metric_tensor(
        reference_path,
        (args.render_resolution, args.render_resolution),
    )
    metric_device = torch.device(
        args.device
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    lpips_evaluator = LPIPSEvaluator(args.lpips_net, metric_device)

    render_rows: List[Dict[str, Any]] = []
    for resolution in args.resolutions:
        for seed in args.seeds:
            paths = run_index[(int(resolution), int(seed))]
            for light in args.lights:
                output_png = render_path(
                    paths["dir"], image_stem, resolution, seed, light
                )
                base_row: Dict[str, Any] = {
                    "row_type": "RENDER",
                    "image_name": args.image.name,
                    "pipeline_resolution": int(resolution),
                    "pipeline_type": f"{int(resolution)}_cascade",
                    "seed": int(seed),
                    "light": light,
                    "generated_glb": str(paths["glb"]),
                    "render_png": str(output_png),
                    "reference_png": str(reference_path),
                }
                if not paths["glb"].is_file():
                    render_rows.append(
                        {
                            **base_row,
                            "status": "failed",
                            "error": "generated GLB is missing",
                        }
                    )
                    continue
                if not output_png.is_file():
                    render_rows.append(
                        {
                            **base_row,
                            "status": "failed",
                            "error": "render PNG is missing",
                        }
                    )
                    continue
                try:
                    metrics = evaluate_render(
                        reference_cpu=reference_cpu,
                        prediction_path=output_png,
                        lpips_evaluator=lpips_evaluator,
                    )
                    row = {
                        **base_row,
                        "status": "success",
                        **metrics,
                        "error": None,
                    }
                    render_rows.append(row)
                    atomic_json(output_png.with_suffix(".metrics.json"), row)
                    print(
                        f"[metrics] r={resolution} seed={seed} light={light} "
                        f"PSNR={metrics['psnr_db']:.4f} "
                        f"SSIM={metrics['ssim']:.6f} "
                        f"LPIPS={metrics['lpips']:.6f}"
                    )
                except Exception as exc:
                    row = {
                        **base_row,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    render_rows.append(row)
                    atomic_json(output_png.with_suffix(".metrics.json"), row)
                    with log_path.open("a", encoding="utf-8") as log_file:
                        log_file.write("\n" + traceback.format_exc() + "\n")
                    print(f"[metric-error] {row['error']}")
                    if args.fail_fast:
                        raise

    summary_rows = build_summary_rows(
        render_rows=render_rows,
        image_name=args.image.name,
        resolutions=args.resolutions,
        seeds=args.seeds,
    )
    all_rows = [*render_rows, *summary_rows]
    write_csv(args.output_dir / "metrics.csv", all_rows)
    atomic_json(
        args.output_dir / "metrics.json",
        {
            "config": vars(args),
            "camera": camera_record,
            "metric_convention": {
                "reference": "Pixal3D preprocessed aligned condition image",
                "reference_path": str(reference_path),
                "background": "white",
                "region": "full RGB canvas",
                "render_resolution": int(args.render_resolution),
                "ssim": "11x11 Gaussian-window RGB SSIM",
                "lpips_network": args.lpips_net,
            },
            "generation": generation_records,
            "render_rows": render_rows,
            "summary_rows": summary_rows,
        },
    )
    atomic_json(
        args.output_dir / "run_config.json",
        {
            "config": vars(args),
            "input_original": str(args.output_dir / "input_original.png"),
            "condition_image": str(condition_path),
            "metric_reference": str(reference_path),
            "camera": camera_record,
            "decoder_to_exported_gltf": PIXAL3D_DECODER_TO_EXPORTED_GLTF,
            "exported_gltf_to_internal": PIXAL3D_EXPORTED_GLTF_TO_INTERNAL,
            "light_modes": list(args.lights),
        },
    )

    successful = sum(row.get("status") == "success" for row in render_rows)
    failed = len(render_rows) - successful
    global_average = next(
        (row for row in reversed(summary_rows) if row["row_type"] == "GLOBAL_AVERAGE"),
        None,
    )
    print(
        f"[done] successful_renders={successful} failed_renders={failed} "
        f"output={args.output_dir}"
    )
    if global_average is not None:
        print(
            "[global-average] "
            f"PSNR={global_average['psnr_db']:.4f} "
            f"SSIM={global_average['ssim']:.6f} "
            f"LPIPS={global_average['lpips']:.6f}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
