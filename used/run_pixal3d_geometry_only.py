#!/usr/bin/env python3
"""
Batch-run Pixal3D geometry-only image-to-3D over:
  input images x pipeline types x seeds

Supported pipeline types in the replacement geometry pipeline:
  - 512
  - 1024
  - 1024_cascade
  - 1536_cascade

This mirrors the TRELLIS.2 resolution-grid script, but keeps Pixal3D's
pixel-aligned/projection conditioning and per-image camera estimation.

Outputs are geometry-only:
  - raw_mesh.glb: vertices + faces only, no PBR texture/material
  - simplified_mesh.glb: optional simplified geometry-only GLB
  - normal_raw/normal_vXX.png and normal_simplified/normal_vXX.png previews
  - metadata.json and manifest.json

Texture/PBR SLat is not sampled or decoded. Use the provided replacement
pixal3d/pipelines/pixal3d_image_to_3d.py before running this script.
"""

import argparse
import copy
import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from PIL import Image

from pixal3d.pipelines import Pixal3DImageTo3DPipeline
from pixal3d.utils import render_utils


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SUPPORTED_PIPELINE_TYPES = ["512", "1024", "1024_cascade", "1536_cascade"]
DEFAULT_PIPELINE_TYPES = ["512", "1024", "1024_cascade", "1536_cascade"]
DEFAULT_RENDER_CHUNK_SIZE = 200_000
MIN_RENDER_CHUNK_SIZE = 25_000

MOGE_MODEL_NAME = "/home/nvme04/yyyan/download/model/moge-2-vitl/model.pt"
DEFAULT_MODEL_PATH = "/home/nvme04/yyyan/download/model/Pixal3D"
IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": "/home/nvme04/yyyan/download/model/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": "/home/nvme04/yyyan/download/model/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": "/home/nvme04/yyyan/download/model/dinov3-vitl16-pretrain-lvd1689m/facebook/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
}


def build_image_cond_model(config: dict):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor

    model = DinoV3ProjFeatureExtractor(**config)
    model.eval()
    return model


def maybe_preload_naf(models: Sequence[Any]) -> None:
    for model in models:
        if model is not None and getattr(model, "use_naf_upsample", False):
            model._load_naf()


def init_pipeline(model_path: str, device: str = "cuda", low_vram: bool = True) -> Pixal3DImageTo3DPipeline:
    print(f"[Pipeline] Loading from {model_path}...")
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)

    print("[ImageCond] Building DinoV3ProjFeatureExtractor models...")
    pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"])
    pipeline.image_cond_model_shape_512 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_512"])
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_1024"])
    pipeline.image_cond_model_tex_1024 = None

    cond_models = [
        pipeline.image_cond_model_ss,
        pipeline.image_cond_model_shape_512,
        pipeline.image_cond_model_shape_1024,
    ]

    if low_vram:
        print("[Pipeline] Low-VRAM mode: models stay on CPU and are loaded to GPU per stage.")
        print("[NAF] Pre-downloading NAF upsampler weights on CPU...")
        maybe_preload_naf(cond_models)
        pipeline._device = torch.device(device)
        pipeline.low_vram = True
    else:
        print("[Pipeline] Standard mode: models loaded to target device.")
        pipeline.low_vram = False
        pipeline.to(torch.device(device))
        for model in cond_models:
            model.to(torch.device(device))
        print("[NAF] Pre-loading NAF upsampler weights...")
        maybe_preload_naf(cond_models)

    return pipeline


def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    f_pixels = focal_length * resolution / 32.0
    return float(f_pixels.item())


def distance_from_fov(
    camera_angle_x: float,
    grid_point: torch.Tensor,
    target_point: torch.Tensor,
    mesh_scale: float,
    image_resolution: int,
) -> Dict[str, float]:
    rotation_matrix = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ])
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / float(mesh_scale) / 2.0
    xw, yw, _ = gp[0].item(), gp[1].item(), gp[2].item()
    xt, yt = float(target_point[0].item()), float(target_point[1].item())
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    _ = -(yt - image_resolution / 2.0)
    distance_x = f_pixels * xw / x_ndc - yw
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}


def load_moge_model(device: str = "cuda", model_name: str = MOGE_MODEL_NAME):
    from moge.model.v2 import MoGeModel

    moge_model = MoGeModel.from_pretrained(model_name)
    moge_model = moge_model.to(device)
    moge_model.eval()
    return moge_model


def get_camera_params_wild_moge(
    image_path: Path,
    moge_model,
    device: str = "cuda",
    mesh_scale: float = 1.0,
    extend_pixel: int = 0,
    image_resolution: int = 512,
) -> Dict[str, float]:
    pil_image = Image.open(image_path).convert("RGB")
    width, _ = pil_image.size
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    with torch.no_grad():
        output = moge_model.infer(image_tensor)
    intrinsics = output["intrinsics"].squeeze().cpu().numpy()
    fx_normalized = intrinsics[0, 0]
    fx = fx_normalized * width
    camera_angle_x = 2.0 * math.atan(width / (2.0 * fx))
    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    distance = distance_from_fov(
        camera_angle_x,
        grid_point,
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale,
        image_resolution,
    )["distance_from_x"]
    return {"camera_angle_x": float(camera_angle_x), "distance": float(distance), "mesh_scale": float(mesh_scale)}


def camera_params_from_manual_fov(
    manual_fov: float,
    mesh_scale: float,
    extend_pixel: int,
    image_resolution: int,
) -> Dict[str, float]:
    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    distance = distance_from_fov(
        float(manual_fov),
        grid_point,
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale,
        image_resolution,
    )["distance_from_x"]
    return {"camera_angle_x": float(manual_fov), "distance": float(distance), "mesh_scale": float(mesh_scale)}


def prepare_image_and_camera(
    pipeline: Pixal3DImageTo3DPipeline,
    image_path: Path,
    image_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Image.Image, Dict[str, float], Optional[str]]:
    image = Image.open(image_path)
    if args.preprocess_image:
        print(f"[preprocess] {image_path.name}")
        image_preprocessed = pipeline.preprocess_image(image)
    else:
        image_preprocessed = image.convert("RGB")

    camera_tmp_rel = None
    if args.manual_fov > 0:
        camera_params = camera_params_from_manual_fov(
            args.manual_fov,
            args.mesh_scale,
            args.extend_pixel,
            args.camera_image_resolution,
        )
        print(
            f"[camera] manual_fov={args.manual_fov:.6f} rad "
            f"({math.degrees(args.manual_fov):.2f} deg), distance={camera_params['distance']:.6f}"
        )
        return image_preprocessed, camera_params, camera_tmp_rel

    tmp_path = image_dir / "_tmp_preprocessed_for_moge.png"
    image_preprocessed.save(tmp_path)
    camera_tmp_rel = tmp_path.name

    moge_device = args.device
    print(f"[MoGe-2] Loading camera estimator on {moge_device}...")
    moge_model = load_moge_model(device=moge_device, model_name=args.moge_model_name)
    try:
        print("[MoGe-2] Estimating camera parameters...")
        camera_params = get_camera_params_wild_moge(
            tmp_path,
            moge_model,
            device=moge_device,
            mesh_scale=args.mesh_scale,
            extend_pixel=args.extend_pixel,
            image_resolution=args.camera_image_resolution,
        )
    finally:
        try:
            moge_model.cpu()
        except Exception:
            pass
        del moge_model
        cleanup_cuda_after_oom()

    print(
        f"[camera] camera_angle_x={camera_params['camera_angle_x']:.6f}, "
        f"distance={camera_params['distance']:.6f}, mesh_scale={camera_params['mesh_scale']:.4f}"
    )
    return image_preprocessed, camera_params, camera_tmp_rel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("/home/nvme04/yyyan/Pixal3D/assets/images"))
    parser.add_argument("--result-root", type=Path, default=Path("/home/nvme04/yyyan/Pixal3D/outputs/pixal3d_resolution_grid_geometry"))
    parser.add_argument("--num-images", type=int, default=0, help="<=0 means use all images in --input-dir.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--pipeline-types", nargs="+", default=DEFAULT_PIPELINE_TYPES, choices=SUPPORTED_PIPELINE_TYPES)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--low-vram", action="store_true", default=True)
    parser.add_argument("--no-low-vram", dest="low_vram", action="store_false")

    parser.add_argument("--preprocess-image", action="store_true", default=True)
    parser.add_argument("--no-preprocess-image", dest="preprocess_image", action="store_false")
    parser.add_argument("--manual-fov", type=float, default=-1.0, help="Manual horizontal FOV in radians. <=0 uses MoGe-2 per image.")
    parser.add_argument("--moge-model-name", type=str, default=MOGE_MODEL_NAME)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--extend-pixel", type=int, default=0)
    parser.add_argument("--camera-image-resolution", type=int, default=512)
    parser.add_argument("--max-num-tokens", type=int, default=49152, help="Cascade token cap. Set <=0 to disable auto-reducing cascade resolution.")

    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--ss-guidance-strength", type=float, default=7.5)
    parser.add_argument("--ss-guidance-rescale", type=float, default=0.7)
    parser.add_argument("--ss-rescale-t", type=float, default=5.0)
    parser.add_argument("--shape-steps", type=int, default=12)
    parser.add_argument("--shape-guidance-strength", type=float, default=7.5)
    parser.add_argument("--shape-guidance-rescale", type=float, default=0.5)
    parser.add_argument("--shape-rescale-t", type=float, default=3.0)

    parser.add_argument("--target-faces", type=int, default=1_000_000)
    parser.add_argument("--simplify-raw-mesh", action="store_true", default=True)
    parser.add_argument("--no-simplify-raw-mesh", dest="simplify_raw_mesh", action="store_false")
    parser.add_argument("--save-raw-mesh", action="store_true", default=True)
    parser.add_argument("--no-save-raw-mesh", dest="save_raw_mesh", action="store_false")

    parser.add_argument("--save-normal-views", action="store_true", default=True)
    parser.add_argument("--no-save-normal-views", dest="save_normal_views", action="store_false")
    parser.add_argument("--normal-render-res", type=int, default=1536)
    parser.add_argument("--normal-nviews", type=int, default=8)
    parser.add_argument("--normal-ssaa", type=int, default=1)
    parser.add_argument("--normal-fit-ratio", type=float, default=0.95)
    parser.add_argument("--render-radius", type=float, default=2.0)
    parser.add_argument("--render-fov", type=float, default=36.0)
    parser.add_argument("--render-pitch-deg", type=float, default=20.0)
    parser.add_argument("--render-yaw-offset-deg", type=float, default=-16.0)
    parser.add_argument("--render-chunk-size", type=int, default=DEFAULT_RENDER_CHUNK_SIZE)
    parser.add_argument("--max-render-faces", type=int, default=0, help="Deprecated/ignored; kept for compatibility.")

    parser.add_argument("--drop-image-on-oom", action="store_true", default=True)
    parser.add_argument("--no-drop-image-on-oom", dest="drop_image_on_oom", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def list_first_images(input_dir: Path, n: int) -> List[Path]:
    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not images:
        raise FileNotFoundError(f"No images found in {input_dir}")
    if n <= 0:
        return images
    return images[:n]


def safe_name(path: Path) -> str:
    keep = []
    for ch in path.stem:
        keep.append(ch if ch.isalnum() or ch in "-_" else "_")
    return "".join(keep)[:96]


def torch_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def mesh_arrays(mesh: Any) -> Tuple[np.ndarray, np.ndarray]:
    vertices = torch_to_numpy(mesh.vertices).astype(np.float32)
    faces = torch_to_numpy(mesh.faces).astype(np.int64)
    return vertices, faces


def export_mesh_geometry_glb(mesh: Any, path: Path) -> None:
    """Export vertices + faces only. No vertex colors, no face colors, no duplicated vertices."""
    import trimesh

    path.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces = mesh_arrays(mesh)
    tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Avoid calling trimesh processing/repair here. We want to inspect the raw decoded geometry.
    # glTF may still include a tiny default material wrapper, but no baked color attributes are written.
    tri.export(str(path), file_type="glb")


def clone_mesh(mesh: Any) -> Any:
    return copy.deepcopy(mesh)


def simplify_mesh(mesh: Any, target_faces: int) -> Any:
    simple = clone_mesh(mesh)
    if int(simple.faces.shape[0]) > target_faces:
        simple.simplify(int(target_faces))
    return simple


def mesh_stats(mesh: Any) -> Dict[str, int]:
    return {"vertices": int(mesh.vertices.shape[0]), "faces": int(mesh.faces.shape[0])}


def fit_image_to_mask(
    image_np: np.ndarray,
    mask_np: Optional[np.ndarray],
    fill_ratio: float = 0.95,
    background_value: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Center/crop/scale an image so the masked object fills fill_ratio of canvas.

    This is a post-process for normal render PNGs only. It does not alter mesh geometry.
    """
    if mask_np is None or fill_ratio <= 0 or fill_ratio >= 1.0 + 1e-6:
        return image_np, {"applied": False, "reason": "no_mask_or_disabled"}

    if mask_np.ndim == 3:
        mask = mask_np[..., 0] > 127
    else:
        mask = mask_np > 127

    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return image_np, {"applied": False, "reason": "empty_mask"}

    h, w = image_np.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bbox_w = max(1, x1 - x0)
    bbox_h = max(1, y1 - y0)
    bbox_max = max(bbox_w, bbox_h)
    crop_side = int(np.ceil(bbox_max / max(float(fill_ratio), 1e-6)))
    crop_side = max(crop_side, bbox_max, 1)

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    crop_x0 = int(np.floor(cx - crop_side / 2.0))
    crop_y0 = int(np.floor(cy - crop_side / 2.0))
    crop_x1 = crop_x0 + crop_side
    crop_y1 = crop_y0 + crop_side

    if image_np.ndim == 2:
        crop = np.full((crop_side, crop_side), background_value, dtype=image_np.dtype)
    else:
        crop = np.full((crop_side, crop_side, image_np.shape[2]), background_value, dtype=image_np.dtype)

    src_x0 = max(0, crop_x0)
    src_y0 = max(0, crop_y0)
    src_x1 = min(w, crop_x1)
    src_y1 = min(h, crop_y1)
    dst_x0 = src_x0 - crop_x0
    dst_y0 = src_y0 - crop_y0

    if src_x1 > src_x0 and src_y1 > src_y0:
        crop[dst_y0:dst_y0 + (src_y1 - src_y0), dst_x0:dst_x0 + (src_x1 - src_x0)] = image_np[src_y0:src_y1, src_x0:src_x1]

    pil = Image.fromarray(crop)
    pil = pil.resize((w, h), Image.Resampling.LANCZOS)
    framed = np.array(pil).astype(image_np.dtype)
    return framed, {
        "applied": True,
        "fill_ratio": float(fill_ratio),
        "source_bbox_xyxy": [x0, y0, x1, y1],
        "crop_xyxy": [crop_x0, crop_y0, crop_x1, crop_y1],
        "output_size": [w, h],
    }


def render_frames_with_chunk_retry(
    mesh: Any,
    extrinsics: Sequence[torch.Tensor],
    intrinsics: Sequence[torch.Tensor],
    options: Dict[str, Any],
    return_types: Sequence[str],
) -> Tuple[Dict[str, List[np.ndarray]], Optional[int], int]:
    """Render the exact input mesh with nvdiffrast chunking and retry on overflow."""
    requested_chunk_size = options.get("chunk_size", None)
    if requested_chunk_size is None or int(requested_chunk_size) <= 0:
        chunk_candidates: List[Optional[int]] = [None]
    else:
        chunk_candidates = []
        cur = int(requested_chunk_size)
        while cur >= MIN_RENDER_CHUNK_SIZE:
            chunk_candidates.append(cur)
            cur //= 2
        if chunk_candidates[-1] != MIN_RENDER_CHUNK_SIZE:
            chunk_candidates.append(MIN_RENDER_CHUNK_SIZE)

    last_err: Optional[BaseException] = None
    for attempt_idx, chunk_size in enumerate(chunk_candidates):
        try:
            trial_options = dict(options)
            if chunk_size is None:
                trial_options.pop("chunk_size", None)
                print("[render] exact mesh render with chunk_size=None")
            else:
                trial_options["chunk_size"] = int(chunk_size)
                print(f"[render] exact mesh render with chunk_size={int(chunk_size):,}")

            result = render_utils.render_frames(
                mesh,
                extrinsics,
                intrinsics,
                options=trial_options,
                verbose=True,
                return_types=list(return_types),
            )
            return result, chunk_size, attempt_idx + 1

        except RuntimeError as e:
            last_err = e
            msg = str(e)
            if "subtriangle count overflow" not in msg or chunk_size is None:
                raise
            print(f"[render] subtriangle overflow at chunk_size={int(chunk_size):,}; retrying smaller")
            torch.cuda.empty_cache()

            if "out of memory" in str(e).lower() or "cuda error" in str(e).lower():
                cleanup_cuda_after_oom()

    assert last_err is not None
    raise last_err


def render_normal_views(
    mesh: Any,
    out_dir: Path,
    resolution: int,
    nviews: int,
    radius: float,
    fov: float,
    pitch_deg: float,
    yaw_offset_deg: float,
    chunk_size: Optional[int],
    max_render_faces: int,
    normal_fit_ratio: float = 0.95,
    normal_ssaa: int = 1,
) -> Dict[str, Any]:
    """Render normal previews for the exact input mesh. This does not write mesh colors."""
    del max_render_faces

    out_dir.mkdir(parents=True, exist_ok=True)
    input_stats = mesh_stats(mesh)
    print(
        f"[render] exact mesh stats: vertices={input_stats['vertices']:,} "
        f"faces={input_stats['faces']:,}; resolution={resolution}; ssaa={normal_ssaa}; "
        f"chunk_size={chunk_size}"
    )

    yaws = np.linspace(0.0, 2.0 * np.pi, nviews, endpoint=False) + np.deg2rad(yaw_offset_deg)
    pitchs = np.full(nviews, np.deg2rad(pitch_deg), dtype=np.float32)
    extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws.tolist(), pitchs.tolist(), radius, fov
    )

    options: Dict[str, Any] = {
        "resolution": int(resolution),
        "near": 1,
        "far": 100,
        "ssaa": int(normal_ssaa),
    }
    if chunk_size is not None and int(chunk_size) > 0:
        options["chunk_size"] = int(chunk_size)

    result, used_chunk_size, retry_attempts = render_frames_with_chunk_retry(
        mesh,
        extrinsics,
        intrinsics,
        options=options,
        return_types=["normal", "mask"],
    )

    views: List[Dict[str, Any]] = []
    normal_imgs: List[Image.Image] = []
    for view_idx, normal_np in enumerate(result["normal"]):
        mask_np = result.get("mask", [None] * len(result["normal"]))[view_idx]
        framing: Dict[str, Any] = {"applied": False}
        if mask_np is not None:
            if mask_np.ndim == 3:
                mask = mask_np[..., :1] > 127
            else:
                mask = mask_np[..., None] > 127
            normal_np = np.where(mask, normal_np, 0).astype(np.uint8)
            normal_np, framing = fit_image_to_mask(
                normal_np,
                mask_np,
                fill_ratio=normal_fit_ratio,
                background_value=0,
            )
        img = Image.fromarray(normal_np)
        filename = f"normal_v{view_idx:02d}.png"
        img.save(out_dir / filename)
        normal_imgs.append(img)
        views.append({
            "view_idx": int(view_idx),
            "path": str(out_dir / filename),
            "yaw_deg": float(np.rad2deg(yaws[view_idx])),
            "pitch_deg": float(pitch_deg),
            "radius": float(radius),
            "fov_deg": float(fov),
            "framing": framing,
        })

    sheet_path = out_dir / "normal_contact_sheet.png"
    save_contact_sheet(normal_imgs, sheet_path)

    return {
        "views": views,
        "contact_sheet": str(sheet_path),
        "render_simplified_due_to_face_limit": False,
        "render_mesh_stats": input_stats,
        "render_mode": "exact_raw_mesh_chunked" if used_chunk_size is not None else "exact_raw_mesh_single_call",
        "render_chunk_size": used_chunk_size,
        "render_retry_attempts": int(retry_attempts),
        "camera": {
            "nviews": int(nviews),
            "resolution": int(resolution),
            "radius": float(radius),
            "fov_deg": float(fov),
            "pitch_deg": float(pitch_deg),
            "yaw_offset_deg": float(yaw_offset_deg),
            "normal_fit_ratio": float(normal_fit_ratio),
            "normal_ssaa": int(normal_ssaa),
        },
    }


def empty_normal_views() -> Dict[str, Any]:
    return {
        "views": [],
        "contact_sheet": None,
        "render_mode": "disabled",
        "render_mesh_stats": None,
        "render_chunk_size": None,
        "render_retry_attempts": 0,
        "camera": None,
    }


def save_contact_sheet(images: Sequence[Image.Image], path: Path) -> None:
    if not images:
        return
    w, h = images[0].size
    cols = min(4, len(images))
    rows = int(np.ceil(len(images) / cols))
    canvas = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    for i, img in enumerate(images):
        canvas.paste(img.convert("RGB"), ((i % cols) * w, (i // cols) * h))
    canvas.save(path)


def rel_to_root(path: str | Path, root: Path) -> str:
    return str(Path(path).resolve().relative_to(root.resolve()))


def relativize_views(d: Dict[str, Any], root: Path) -> Dict[str, Any]:
    out = dict(d)
    if out.get("contact_sheet"):
        out["contact_sheet"] = rel_to_root(out["contact_sheet"], root)
    out["views"] = [{**v, "path": rel_to_root(v["path"], root)} for v in out.get("views", [])]
    return out


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_existing_manifest(path: Path) -> Dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"schema_version": 3, "created_by": "run_pixal3d_resolution_grid_geometry.py", "images": [], "pipeline_types": DEFAULT_PIPELINE_TYPES, "seeds": []}


def add_or_replace_image_record(manifest: Dict[str, Any], record: Dict[str, Any]) -> None:
    manifest["images"] = [x for x in manifest.get("images", []) if x["image_id"] != record["image_id"]]
    manifest["images"].append(record)
    manifest["images"] = sorted(manifest["images"], key=lambda x: x["image_id"])


def should_skip_existing(
    metadata_path: Path,
    raw_glb: Path,
    simplified_glb: Path,
    save_raw_mesh: bool,
    simplify_raw_mesh: bool,
    overwrite: bool,
) -> bool:
    if overwrite:
        return False
    if not metadata_path.exists():
        return False
    if save_raw_mesh and not raw_glb.exists():
        return False
    if simplify_raw_mesh and not simplified_glb.exists():
        return False
    return True


def is_oom_exception(exc: BaseException) -> bool:
    """Return True only for CUDA/GPU allocation failures that should drop the whole image."""
    torch_oom_cls = getattr(torch, "OutOfMemoryError", None)
    cuda_oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if torch_oom_cls is not None and isinstance(exc, torch_oom_cls):
        return True
    if cuda_oom_cls is not None and isinstance(exc, cuda_oom_cls):
        return True

    msg = str(exc).lower()
    oom_markers = (
        "out of memory",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
        "cusparse_status_alloc_failed",
        "cudnn_status_alloc_failed",
        "failed to allocate",
        "memory allocation failed",
        "std::bad_alloc",
    )
    return any(marker in msg for marker in oom_markers)


def remove_image_record(manifest: Dict[str, Any], image_id: str) -> None:
    manifest["images"] = [x for x in manifest.get("images", []) if x.get("image_id") != image_id]


def clear_cuda_after_oom() -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass

def cleanup_cuda_after_oom():
    import gc
    import torch

    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    gc.collect()

def discard_image_outputs_after_oom(
    image_dir: Path,
    manifest: Dict[str, Any],
    image_id: str,
    manifest_path: Path,
) -> None:
    """Delete all partial results for this image and remove it from manifest."""
    print(f"[oom/drop] removing partial outputs for {image_id}: {image_dir}")
    shutil.rmtree(image_dir, ignore_errors=True)
    remove_image_record(manifest, image_id)
    write_json(manifest_path, manifest)



def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    result_root = args.result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)

    manifest_path = result_root / "manifest.json"
    if args.overwrite and result_root.exists():
        shutil.rmtree(result_root)
        result_root.mkdir(parents=True, exist_ok=True)

    images = list_first_images(args.input_dir, args.num_images)
    manifest = load_existing_manifest(manifest_path)
    manifest["schema_version"] = 3
    manifest["created_by"] = "run_pixal3d_resolution_grid_geometry.py"
    manifest["source_project"] = "TencentARC/Pixal3D"
    manifest["pipeline_types"] = args.pipeline_types
    manifest["supported_pipeline_types"] = SUPPORTED_PIPELINE_TYPES
    manifest["seeds"] = args.seeds
    manifest["result_root"] = str(result_root)
    manifest["input_dir"] = str(args.input_dir)
    manifest["model_path"] = str(args.model_path)
    manifest["low_vram"] = bool(args.low_vram)
    manifest["mesh_format"] = "geometry_glb"
    manifest["mesh_color"] = "none; GLB files contain geometry only. Viewer computes preview colors with a shader."
    manifest["raw_mesh_format"] = "glb"
    manifest["simplified_mesh_format"] = "glb" if args.simplify_raw_mesh else None
    manifest["simplify_raw_mesh"] = bool(args.simplify_raw_mesh)
    manifest["drop_image_on_oom"] = bool(args.drop_image_on_oom)
    manifest["normal_render_res"] = args.normal_render_res
    manifest["normal_nviews"] = args.normal_nviews
    manifest["normal_ssaa"] = args.normal_ssaa
    manifest["save_normal_views"] = bool(args.save_normal_views)
    manifest["render_chunk_size"] = args.render_chunk_size
    manifest["target_faces"] = args.target_faces
    manifest["normal_fit_ratio"] = args.normal_fit_ratio
    manifest["max_num_tokens"] = args.max_num_tokens
    manifest["texture"] = "disabled; texture/PBR SLat is not sampled or decoded"
    manifest["camera"] = {
        "manual_fov": args.manual_fov,
        "moge_model_name": args.moge_model_name,
        "mesh_scale": args.mesh_scale,
        "extend_pixel": args.extend_pixel,
        "camera_image_resolution": args.camera_image_resolution,
    }

    print(f"[setup] repo_root={repo_root}")
    print(f"[setup] result_root={result_root}")
    print(f"[setup] images={len(images)}, pipeline_types={args.pipeline_types}, seeds={args.seeds}")
    print(f"[setup] simplify_raw_mesh={args.simplify_raw_mesh}, drop_image_on_oom={args.drop_image_on_oom}")
    print(f"[setup] low_vram={args.low_vram}, texture=disabled")

    pipeline = init_pipeline(args.model_path, device=args.device, low_vram=args.low_vram)

    ss_sampler_params = {
        "steps": args.ss_steps,
        "guidance_strength": args.ss_guidance_strength,
        "guidance_rescale": args.ss_guidance_rescale,
        "rescale_t": args.ss_rescale_t,
    }
    shape_sampler_params = {
        "steps": args.shape_steps,
        "guidance_strength": args.shape_guidance_strength,
        "guidance_rescale": args.shape_guidance_rescale,
        "rescale_t": args.shape_rescale_t,
    }

    for image_idx, image_path in enumerate(images):
        image_id = f"image_{image_idx:03d}_{safe_name(image_path)}"
        image_dir = result_root / image_id
        image_dir.mkdir(parents=True, exist_ok=True)

        input_copy = image_dir / f"input{image_path.suffix.lower()}"
        shutil.copy2(image_path, input_copy)

        image_record: Dict[str, Any] = {
            "image_id": image_id,
            "input_name": image_path.name,
            "input_original_path": str(image_path),
            "input_image": rel_to_root(input_copy, result_root),
            "results": [],
            "errors": [],
        }

        image_failed_due_to_oom = False
        try:
            image_preprocessed, camera_params, camera_tmp_rel = prepare_image_and_camera(
                pipeline,
                image_path,
                image_dir,
                args,
            )
            preprocessed_copy = image_dir / "input_preprocessed.png"
            image_preprocessed.save(preprocessed_copy)
            image_record["input_preprocessed"] = rel_to_root(preprocessed_copy, result_root)
            image_record["camera_params"] = camera_params
            if camera_tmp_rel:
                image_record["camera_tmp"] = camera_tmp_rel
        except BaseException as e:
            if not is_oom_exception(e):
                raise
            cleanup_cuda_after_oom()
            err_record = {
                "image_id": image_id,
                "input_name": image_path.name,
                "status": "oom",
                "stage": "preprocess_or_camera",
                "error_type": type(e).__name__,
                "error": str(e),
                "drop_image_on_oom": bool(args.drop_image_on_oom),
            }
            image_record["errors"].append(err_record)
            if args.drop_image_on_oom:
                discard_image_outputs_after_oom(image_dir, manifest, image_id, manifest_path)
                print(f"[oom/drop] camera/preprocess failed for {image_id}; continuing next image", file=sys.stderr)
                continue
            add_or_replace_image_record(manifest, image_record)
            write_json(manifest_path, manifest)
            continue

        for seed in args.seeds:
            if image_failed_due_to_oom:
                break

            for pipeline_type in args.pipeline_types:
                if image_failed_due_to_oom:
                    break

                run_id = f"seed_{seed:010d}/{pipeline_type}"
                out_dir = image_dir / run_id
                metadata_path = out_dir / "metadata.json"
                raw_glb = out_dir / "raw_mesh.glb"
                simplified_glb = out_dir / "simplified_mesh.glb"

                if should_skip_existing(
                    metadata_path,
                    raw_glb,
                    simplified_glb,
                    args.save_raw_mesh,
                    args.simplify_raw_mesh,
                    args.overwrite,
                ):
                    print(f"[skip] {image_id} seed={seed} type={pipeline_type}")
                    with metadata_path.open("r", encoding="utf-8") as f:
                        meta = json.load(f)
                    image_record["results"].append(meta)
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)
                print(f"[run] image={image_path.name} seed={seed} type={pipeline_type}")
                start_time = time.time()

                try:
                    outputs, latents = pipeline.run(
                        image_preprocessed,
                        camera_params=camera_params,
                        seed=int(seed),
                        preprocess_image=False,
                        pipeline_type=pipeline_type,
                        return_latent=True,
                        max_num_tokens=args.max_num_tokens,
                        sparse_structure_sampler_params=ss_sampler_params,
                        shape_slat_sampler_params=shape_sampler_params,
                    )
                    mesh = outputs[0]
                    effective_resolution = int(latents[2]) if isinstance(latents, tuple) and len(latents) >= 3 else None

                    raw_stats = mesh_stats(mesh)
                    print(f"[mesh/raw] vertices={raw_stats['vertices']:,} faces={raw_stats['faces']:,}")

                    if args.save_raw_mesh:
                        export_mesh_geometry_glb(mesh, raw_glb)
                        print(f"[mesh/raw] saved geometry GLB: {raw_glb}")

                    if args.save_normal_views:
                        raw_normal = render_normal_views(
                            mesh,
                            out_dir / "normal_raw",
                            resolution=args.normal_render_res,
                            nviews=args.normal_nviews,
                            radius=args.render_radius,
                            fov=args.render_fov,
                            pitch_deg=args.render_pitch_deg,
                            yaw_offset_deg=args.render_yaw_offset_deg,
                            chunk_size=args.render_chunk_size,
                            max_render_faces=args.max_render_faces,
                            normal_fit_ratio=args.normal_fit_ratio,
                            normal_ssaa=args.normal_ssaa,
                        )
                    else:
                        raw_normal = empty_normal_views()

                    if args.simplify_raw_mesh:
                        simple_mesh = simplify_mesh(mesh, args.target_faces)
                        simple_stats = mesh_stats(simple_mesh)
                        print(f"[mesh/simple] vertices={simple_stats['vertices']:,} faces={simple_stats['faces']:,}")
                        export_mesh_geometry_glb(simple_mesh, simplified_glb)
                        print(f"[mesh/simple] saved geometry GLB: {simplified_glb}")

                        if args.save_normal_views:
                            simple_normal = render_normal_views(
                                simple_mesh,
                                out_dir / "normal_simplified",
                                resolution=args.normal_render_res,
                                nviews=args.normal_nviews,
                                radius=args.render_radius,
                                fov=args.render_fov,
                                pitch_deg=args.render_pitch_deg,
                                yaw_offset_deg=args.render_yaw_offset_deg,
                                chunk_size=args.render_chunk_size,
                                max_render_faces=args.max_render_faces,
                                normal_fit_ratio=args.normal_fit_ratio,
                                normal_ssaa=args.normal_ssaa,
                            )
                        else:
                            simple_normal = empty_normal_views()
                    else:
                        simple_stats = None
                        simple_normal = empty_normal_views()
                        print("[mesh/simple] skipped because --no-simplify-raw-mesh is enabled")

                    elapsed = time.time() - start_time
                    meta: Dict[str, Any] = {
                        "schema_version": 3,
                        "image_id": image_id,
                        "input_name": image_path.name,
                        "seed": int(seed),
                        "pipeline_type": pipeline_type,
                        "effective_resolution": effective_resolution,
                        "elapsed_sec": elapsed,
                        "raw_mesh_glb": rel_to_root(raw_glb, result_root) if args.save_raw_mesh else None,
                        "simplified_mesh_glb": rel_to_root(simplified_glb, result_root) if args.simplify_raw_mesh else None,
                        "raw_mesh": rel_to_root(raw_glb, result_root) if args.save_raw_mesh else None,
                        "simplified_mesh": rel_to_root(simplified_glb, result_root) if args.simplify_raw_mesh else None,
                        "mesh_format": "geometry_glb",
                        "mesh_color": "none; geometry-only GLB, viewer computes normal preview colors",
                        "texture": "disabled",
                        "raw_stats": raw_stats,
                        "simplified_stats": simple_stats,
                        "simplify_raw_mesh": bool(args.simplify_raw_mesh),
                        "normal_raw": relativize_views(raw_normal, result_root),
                        "normal_simplified": relativize_views(simple_normal, result_root),
                        "camera_params": camera_params,
                        "status": "ok",
                        "params": {
                            "target_faces": args.target_faces,
                            "simplify_raw_mesh": bool(args.simplify_raw_mesh),
                            "drop_image_on_oom": bool(args.drop_image_on_oom),
                            "save_raw_mesh": bool(args.save_raw_mesh),
                            "save_normal_views": bool(args.save_normal_views),
                            "normal_render_res": args.normal_render_res,
                            "normal_nviews": args.normal_nviews,
                            "normal_fit_ratio": args.normal_fit_ratio,
                            "normal_ssaa": args.normal_ssaa,
                            "render_chunk_size": args.render_chunk_size,
                            "ss_sampler_params": ss_sampler_params,
                            "shape_sampler_params": shape_sampler_params,
                            "max_num_tokens": args.max_num_tokens,
                            "low_vram": bool(args.low_vram),
                            "manual_fov": args.manual_fov,
                        },
                    }
                    write_json(metadata_path, meta)
                    image_record["results"].append(meta)
                    torch.cuda.empty_cache()
                    gc.collect()

                except BaseException as e:
                    if not is_oom_exception(e):
                        raise

                    cleanup_cuda_after_oom()
                    err_record = {
                        "image_id": image_id,
                        "input_name": image_path.name,
                        "seed": int(seed),
                        "pipeline_type": pipeline_type,
                        "run_id": run_id,
                        "status": "oom",
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "drop_image_on_oom": bool(args.drop_image_on_oom),
                    }
                    image_record["errors"].append(err_record)
                    print(
                        f"[oom] image={image_path.name} image_id={image_id} seed={seed} "
                        f"type={pipeline_type}; error={type(e).__name__}: {e}",
                        file=sys.stderr,
                    )

                    if args.drop_image_on_oom:
                        image_failed_due_to_oom = True
                        discard_image_outputs_after_oom(image_dir, manifest, image_id, manifest_path)
                        cleanup_cuda_after_oom()
                        break

                    print(f"[oom/keep-image] removing failed run only: {out_dir}", file=sys.stderr)
                    shutil.rmtree(out_dir, ignore_errors=True)
                    cleanup_cuda_after_oom()
                    continue

        if image_failed_due_to_oom:
            print(f"[oom/drop] skipped all remaining seeds/pipeline types for {image_id}; continuing with next image")
            continue

        add_or_replace_image_record(manifest, image_record)
        write_json(manifest_path, manifest)
        print(f"[manifest] updated {manifest_path}")

    print("[done]")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
