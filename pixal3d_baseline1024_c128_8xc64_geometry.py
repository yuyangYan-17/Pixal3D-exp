#!/usr/bin/env python3
"""Geometry-only C64 baseline -> global C128 support -> 8-way C64 flow.

The first three stages are the native Pixal3D 1024 cascade.  Instead of
decoding its C64 shape SLat, the shape decoder is used only as a learned
support upsampler.  Those C1024 candidate coordinates are requantized to a
global C128 support with the baseline round-to-(grid-1) convention.  The full
support is projected into the complete canonical 1024 input once, partitioned
into eight disjoint C64 cubes, packed as a real sparse batch, and passed
through one Shape1024 flow.  The assembled global shape SLat is decoded once
at resolution 2048.  No texture model or texture decoder is loaded or run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    str(Path(__file__).with_name("autotune_cache.json")),
)

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

import pixal3d_cascade512_1024_tiled2048_crop_condition as tiled
from pixal3d.modules.sparse import SparseTensor
from pixal3d.utils import render_utils


FORMAT = "pixal3d_baseline1024_c128_8xc64_geometry_v1"
GRID = 128
CUBE = 64
DECODE_RESOLUTION = 2048
DEFAULT_BASELINE = Path("outputs/baseline1024_raw_ovoxel_cuda4_0_img")


def save_sparse(path: Path, value: SparseTensor, normalized: bool) -> None:
    tiled.atomic_save(
        path,
        {
            "format": FORMAT,
            "coords": value.coords.detach().cpu().int(),
            "features": value.feats.detach().cpu(),
            "normalized": bool(normalized),
        },
    )


def load_sparse(path: Path, device: torch.device) -> SparseTensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return SparseTensor(payload["features"].to(device), payload["coords"].to(device).int())


def full_image_conditions(
    pipeline: Any,
    image: Image.Image,
    camera: Mapping[str, float],
    coords: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    out: Path,
) -> dict[int, dict[str, torch.Tensor]]:
    """Project global C128 once, then route aligned rows to the eight cubes."""
    cache = out / "conditions" / "shape_global_c128_full_image.pt"
    if cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if not torch.equal(payload["coords"].int(), coords.cpu().int()):
            raise RuntimeError("cached C128 condition coordinates do not match support")
        glob, proj = payload["global"], payload["proj"]
        reused = True
    else:
        if image.size != (1024, 1024):
            raise RuntimeError(f"shape condition must be 1024x1024, got {image.size}")
        cond = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_shape_1024,
            [image],
            coords.to(pipeline.device),
            camera_angle_x=float(camera["camera_angle_x"]),
            distance=float(camera["distance"]),
            mesh_scale=float(camera.get("mesh_scale", 1.0)),
            grid_resolution_override=GRID,
        )
        glob = cond["cond"]["global"].detach().cpu().contiguous()
        proj = cond["cond"]["proj"].feats.detach().cpu().contiguous()
        tiled.atomic_save(
            cache,
            {
                "format": FORMAT,
                "coords": coords.cpu().int(),
                "global": glob,
                "proj": proj,
                "source": "one get_proj_cond_shape call on complete canonical image_1024",
            },
        )
        del cond
        tiled.empty_cuda()
        reused = False
    if glob.shape[0] != 1 or proj.shape[0] != coords.shape[0]:
        raise RuntimeError("global C128 projected condition is not row-aligned")
    result: dict[int, dict[str, torch.Tensor]] = {}
    for rec in records:
        rows = rec["global_row_ids"].long()
        if rows.numel():
            result[int(rec["cube_id"])] = {
                "global": glob,
                "proj": proj.index_select(0, rows),
            }
    tiled.atomic_json(
        out / "conditions" / "manifest.json",
        {
            "format": FORMAT,
            "cache_reused": reused,
            "image_size": [1024, 1024],
            "grid_resolution_override": GRID,
            "get_proj_cond_shape_calls": 0 if reused else 1,
            "routing": "global projected rows gathered into eight disjoint C64 batches",
            "tokens": int(coords.shape[0]),
            "active_cubes": len(result),
        },
    )
    return result


def c128_support(pipeline: Any, shape_c64: SparseTensor) -> torch.Tensor:
    """Exact requested decoder.upsample + round-to-(grid-1) requantization."""
    decoder = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        decoder.to(pipeline.device)
        decoder.low_vram = True
    hr_coords = decoder.upsample(shape_c64, upsample_times=4)
    if pipeline.low_vram:
        decoder.cpu()
        decoder.low_vram = False
    lr_resolution = 1024
    grid_res = GRID
    quant_coords = torch.cat(
        [
            hr_coords[:, :1],
            ((hr_coords[:, 1:] + 0.5) / lr_resolution * (grid_res - 1)).round().int(),
        ],
        dim=1,
    )
    valid = (quant_coords[:, 1:] >= 0).all(1) & (quant_coords[:, 1:] < grid_res).all(1)
    coords = quant_coords[valid].unique(dim=0).int()
    if not coords.numel():
        raise RuntimeError("C64 decoder upsample produced an empty C128 support")
    return coords


def shade_normal(normal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n = normal.astype(np.float32) / 255.0 * 2.0 - 1.0
    # MeshRenderer orients visible camera-space normals toward +Z for this
    # projective camera convention, so use a slightly elevated camera light.
    light = np.asarray([0.25, -0.35, 1.0], dtype=np.float32)
    light /= np.linalg.norm(light)
    lambert = np.clip(np.sum(n * light[None, None], axis=2), 0.0, 1.0)
    gray = 0.22 + 0.72 * lambert
    rgb = np.repeat(gray[..., None], 3, axis=2)
    alpha = mask[..., :1].astype(np.float32) / 255.0
    return np.uint8(np.clip(rgb * alpha + (1.0 - alpha), 0.0, 1.0) * 255.0 + 0.5)


def contact_sheet(items: Sequence[tuple[str, Image.Image]], path: Path) -> None:
    if not items:
        return
    panel = items[0][1].width
    header = 36
    cols = min(3, len(items))
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * panel, rows * (panel + header)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(items):
        x, y = index % cols * panel, index // cols * (panel + header)
        sheet.paste(image.convert("RGB"), (x, y + header))
        draw.text((x + 8, y + 10), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


@torch.no_grad()
def render_geometry(
    mesh: Any,
    camera: Mapping[str, float],
    reference_path: Path,
    reference_mask_path: Path,
    out: Path,
    resolution: int,
    angles: Sequence[int],
    chunk_size: int,
) -> dict[str, Any]:
    device = mesh.device
    extrinsics, intrinsics, _ = __import__(
        "pixal3d_baseline1024_pbr_mesh_compare"
    )._make_camera_views(float(camera["camera_angle_x"]), float(camera["distance"]), angles)
    options = {
        "resolution": int(resolution),
        "near": 0.01,
        "far": float(camera["distance"]) + 10.0,
        "ssaa": 1,
        "chunk_size": int(chunk_size),
    }
    result = render_utils.render_frames(
        mesh,
        [extrinsics[a].to(device) for a in angles],
        [intrinsics.to(device) for _ in angles],
        options=options,
        return_types=["normal", "mask"],
        verbose=True,
    )
    render_dir = out / f"multiview_{resolution}"
    render_dir.mkdir(parents=True, exist_ok=True)
    normal_panels: list[tuple[str, Image.Image]] = []
    shaded_panels: list[tuple[str, Image.Image]] = []
    for index, angle in enumerate(angles):
        normal = np.asarray(result["normal"][index])
        mask = np.asarray(result["mask"][index])
        normal_image = Image.fromarray(normal).convert("RGB")
        shaded_image = Image.fromarray(shade_normal(normal, mask)).convert("RGB")
        normal_image.save(render_dir / f"view_{angle:03d}_camera_normal.png")
        shaded_image.save(render_dir / f"view_{angle:03d}_geometry_shaded.png")
        Image.fromarray(mask).convert("L").save(render_dir / f"view_{angle:03d}_mask.png")
        normal_panels.append((f"yaw {angle} normal", normal_image))
        shaded_panels.append((f"yaw {angle} geometry", shaded_image))
    normal_sheet = render_dir / "camera_normal_contact_sheet.png"
    shaded_sheet = render_dir / "geometry_shaded_contact_sheet.png"
    contact_sheet(normal_panels, normal_sheet)
    contact_sheet(shaded_panels, shaded_sheet)

    size = (resolution, resolution)
    reference = np.asarray(
        Image.open(reference_path).convert("RGB").resize(size, Image.Resampling.LANCZOS),
        dtype=np.float32,
    ) / 255.0
    prediction = np.asarray(shaded_panels[0][1], dtype=np.float32) / 255.0
    reference_mask = np.asarray(
        Image.open(reference_mask_path).convert("L").resize(size, Image.Resampling.NEAREST),
        dtype=np.float32,
    ) / 255.0 > 0.5
    prediction_mask = np.asarray(result["mask"][0])[..., 0] > 127
    diff = prediction - reference
    mse = float(np.mean(diff * diff))
    fg_mse = float(np.mean(diff[reference_mask] ** 2))
    _, ssim_map = structural_similarity(reference, prediction, data_range=1.0, channel_axis=2, full=True)
    intersection = int(np.logical_and(reference_mask, prediction_mask).sum())
    union = int(np.logical_or(reference_mask, prediction_mask).sum())
    metrics = {
        "metric_target": "geometry-only gray Lambert render compared with the input RGB view; proxy only",
        "is_ground_truth": False,
        "reference_kind": "input_rgb_conditioning_view",
        "reference_mask_kind": "provided_foreground_proxy",
        "reference": str(reference_path.resolve()),
        "reference_mask": str(reference_mask_path.resolve()),
        "input_view_yaw": int(angles[0]),
        "psnr_db": float(10.0 * math.log10(1.0 / max(mse, 1e-12))),
        "foreground_psnr_db": float(10.0 * math.log10(1.0 / max(fg_mse, 1e-12))),
        "ssim": float(np.mean(ssim_map)),
        "foreground_ssim": float(np.mean(ssim_map[reference_mask])),
        "mae": float(np.mean(np.abs(diff))),
        "foreground_mae": float(np.mean(np.abs(diff[reference_mask]))),
        "silhouette_iou": float(intersection / max(union, 1)),
    }
    tiled.atomic_json(render_dir / "input_view_metrics.json", metrics)
    return {
        "normal_contact_sheet": str(normal_sheet.resolve()),
        "geometry_contact_sheet": str(shaded_sheet.resolve()),
        "metrics": metrics,
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    from inference import init_pipeline

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.cuda_device):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}, expected {args.cuda_device}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=True)
    canonical = pipeline.preprocess_canonical_images(Image.open(args.image).convert("RGB"))
    camera = json.loads(args.camera.read_text(encoding="utf-8"))
    if "camera" in camera:
        camera = camera["camera"]
    for key in ("image_512", "image_1024"):
        canonical[key].save(out / f"canonical_{key}.png")
    tiled.atomic_json(out / "camera.json", camera)
    tiled.atomic_json(
        out / "config.json",
        {
            "format": FORMAT,
            "status": "running",
            "path": "baseline SS/C32 -> Shape512/C32 -> Shape1024/C64 -> upsample(4) -> quant C128 -> one full-image projection -> B=8 disjoint C64 Shape1024 flow -> one global geometry decode2048",
            "texture": "not sampled, not decoded",
            "args": vars(args),
        },
    )

    baseline_cache = out / "baseline" / "shape_c64_denormalized.pt"
    if baseline_cache.is_file() and args.resume:
        print("[baseline] reuse cached native C64 shape SLat", flush=True)
        shape_c64 = load_sparse(baseline_cache, device)
    else:
        print("[baseline 1/3] sparse structure -> C32", flush=True)
        cond_ss = pipeline.get_proj_cond_ss(
            [canonical["image_512"]],
            camera_angle_x=float(camera["camera_angle_x"]),
            distance=float(camera["distance"]),
            mesh_scale=float(camera.get("mesh_scale", 1.0)),
        )
        coords_c32 = pipeline.sample_sparse_structure(cond_ss, 32)
        del cond_ss
        tiled.empty_cuda()
        print("[baseline 2/3] Shape512 flow on C32", flush=True)
        cond_c32 = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_shape_512,
            [canonical["image_512"]],
            coords_c32,
            camera_angle_x=float(camera["camera_angle_x"]),
            distance=float(camera["distance"]),
            mesh_scale=float(camera.get("mesh_scale", 1.0)),
        )
        shape_c32 = pipeline.sample_shape_slat(
            cond_c32, pipeline.models["shape_slat_flow_model_512"], coords_c32
        )
        del cond_c32, coords_c32
        tiled.empty_cuda()
        print("[baseline 3/3] decoder support -> C64, Shape1024 flow", flush=True)
        coords_c64 = tiled.upsample_and_quantize(pipeline, shape_c32, 512, 64, True)
        cond_c64 = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_shape_1024,
            [canonical["image_1024"]],
            coords_c64,
            camera_angle_x=float(camera["camera_angle_x"]),
            distance=float(camera["distance"]),
            mesh_scale=float(camera.get("mesh_scale", 1.0)),
            grid_resolution_override=64,
        )
        shape_c64 = pipeline.sample_shape_slat(
            cond_c64, pipeline.models["shape_slat_flow_model_1024"], coords_c64
        )
        save_sparse(baseline_cache, shape_c64, normalized=False)
        del cond_c64, coords_c64, shape_c32
        tiled.empty_cuda()

    support_cache = out / "support" / "coords_c128.pt"
    if support_cache.is_file() and args.resume:
        coords_c128 = torch.load(support_cache, map_location="cpu", weights_only=False)["coords"].int()
    else:
        print("[support] decoder.upsample(..., 4), requested round quantization -> C128", flush=True)
        coords_c128 = c128_support(pipeline, shape_c64).cpu()
        tiled.atomic_save(support_cache, {"format": FORMAT, "coords": coords_c128})
    del shape_c64
    tiled.empty_cuda()

    # build_records needs camera only for its unused crop metadata.  The flow
    # condition below always comes from the complete 1024 image.
    records = tiled.build_records(coords_c128, camera)
    tiled.atomic_json(
        out / "support" / "cube_layout.json",
        {
            "global_grid": GRID,
            "cube_size": CUBE,
            "stride": CUBE,
            "coverage": "exactly one cube per token; no overlap",
            "cubes": [
                {"cube_id": r["cube_id"], "start": r["start"], "tokens": int(r["global_row_ids"].numel())}
                for r in records
            ],
        },
    )
    final_norm_cache = out / "shape" / "final_state_normalized.pt"
    if final_norm_cache.is_file() and args.resume:
        payload = torch.load(final_norm_cache, map_location="cpu", weights_only=False)
        if not torch.equal(payload["coords"].int(), coords_c128.int()):
            raise RuntimeError("cached final shape support mismatch")
        shape_norm = payload["features"]
        print("[flow] reuse cached 8-way C64 result", flush=True)
    else:
        conditions = full_image_conditions(
            pipeline, canonical["image_1024"], camera, coords_c128, records, out
        )
        old_steps = pipeline.shape_slat_sampler_params.get("steps")
        pipeline.shape_slat_sampler_params["steps"] = int(args.shape_steps)
        shape_norm = tiled.run_tiled_flow(
            pipeline, "shape", coords_c128, records, conditions,
            args.shape_seed, None, out,
        )
        if old_steps is not None:
            pipeline.shape_slat_sampler_params["steps"] = old_steps
        del conditions
    mean, std = tiled.normalization_tensors(
        pipeline.shape_slat_normalization, torch.device("cpu")
    )
    shape_raw = shape_norm * std + mean
    tiled.atomic_save(
        out / "shape" / "final_state_denormalized.pt",
        {"format": FORMAT, "coords": coords_c128, "features": shape_raw},
    )
    tiled.empty_cuda()

    print("[decode] assembled global C128 shape -> geometry-only decode at 2048", flush=True)
    shape_st = SparseTensor(shape_raw.to(device), coords_c128.to(device))
    meshes, _ = pipeline.decode_shape_slat(shape_st, DECODE_RESOLUTION)
    if len(meshes) != 1:
        raise RuntimeError(f"shape decoder returned B={len(meshes)}, expected 1")
    mesh = meshes[0]
    tiled.atomic_save(out / "final" / "geometry_mesh.pt", {"format": FORMAT, "mesh": mesh.cpu()})
    try:
        import trimesh

        tri = trimesh.Trimesh(
            vertices=mesh.vertices.detach().cpu().numpy(),
            faces=mesh.faces.detach().cpu().numpy(),
            process=False,
        )
        (out / "final").mkdir(parents=True, exist_ok=True)
        tri.export(out / "final" / "geometry_mesh.glb")
    except Exception as exc:
        tiled.atomic_json(out / "final" / "glb_export_error.json", {"error": repr(exc)})

    render = render_geometry(
        mesh,
        camera,
        args.reference,
        args.reference_mask,
        out,
        args.render_resolution,
        tuple(int(x) % 360 for x in args.angles.split(",")),
        args.render_chunk_size,
    )
    summary = {
        "format": FORMAT,
        "status": "complete",
        "texture_executed": False,
        "c128_tokens": int(coords_c128.shape[0]),
        "active_c64_batches": sum(bool(r["global_row_ids"].numel()) for r in records),
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "seconds": time.perf_counter() - started,
        "mesh_pt": str((out / "final" / "geometry_mesh.pt").resolve()),
        "mesh_glb": str((out / "final" / "geometry_mesh.glb").resolve()),
        **render,
    }
    tiled.atomic_json(out / "summary.json", summary)
    print(json.dumps(tiled._jsonable(summary), indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("assets/choose/0_img.png"))
    parser.add_argument("--camera", type=Path, default=DEFAULT_BASELINE / "global_camera.json")
    parser.add_argument("--reference", type=Path, default=DEFAULT_BASELINE / "canonical_1024.png")
    parser.add_argument("--reference-mask", type=Path, default=DEFAULT_BASELINE / "raw_ovoxel_render/alpha.png")
    parser.add_argument("--model-path", type=Path, default=Path(tiled.DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/baseline1024_c128_8xc64_geometry_cuda4"),
    )
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shape-seed", type=int, default=43)
    parser.add_argument("--shape-steps", type=int, default=12)
    parser.add_argument("--angles", default="0,60,120,180,240,300")
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--render-chunk-size", type=int, default=200_000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
