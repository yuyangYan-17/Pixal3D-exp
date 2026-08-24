#!/usr/bin/env python3
"""Render the completed CUDA4 mesh at the three input viewpoints.

The input composite is the native 3072x1024 Pixal3D three-view image.  This
utility renders the completed per-vertex PBR mesh at yaw 0/120/240 with the
project's native camera convention, saves input-matched 1024x1024 images, and
reports full-frame and reference-foreground PSNR/SSIM plus AlexNet LPIPS when
the installed LPIPS weights are available.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image

import pixal3d_baseline1024_pbr_mesh_compare as baseline_render
from pixal3d.representations import MeshWithVertexPbr


ANGLES = (0, 120, 240)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/global4096_multiview_joint_shape_tex_sr_cuda4/multiview_metrics"
)
DEFAULT_MESH = Path(
    "outputs/global4096_multiview_joint_shape_tex_sr_cuda4/final/"
    "final_per_vertex_pbr_mesh.pt"
)
DEFAULT_CAMERA = Path(
    "outputs/global4096_multiview_joint_shape_tex_sr_cuda4/global_camera.json"
)
DEFAULT_INPUT = Path("test_pic/mask_compare_output/image2_resized.png")
DEFAULT_ALPHA = Path("test_pic/mask_compare_output/image2_bg_removed.png")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _save_tensor_image(value: torch.Tensor, path: Path, channels: int = 3) -> None:
    data = value.detach().float().cpu()
    if data.ndim == 3 and data.shape[0] in (1, 3):
        data = data.permute(1, 2, 0)
    if data.ndim == 2:
        data = data[..., None]
    array = data.numpy()
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    array = np.clip(array, 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if channels == 1:
        Image.fromarray((array[..., 0] * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)
    else:
        Image.fromarray((array[..., :3] * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(path)


def _load_mesh(path: Path) -> MeshWithVertexPbr:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    mesh = payload.get("mesh", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(mesh, MeshWithVertexPbr):
        raise TypeError(f"expected MeshWithVertexPbr in {path}, got {type(mesh)!r}")
    return mesh.cpu()


def _load_camera(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload.get("camera"), Mapping):
        payload = payload["camera"]
    return {
        "camera_angle_x": float(payload["camera_angle_x"]),
        "distance": float(payload["distance"]),
        "mesh_scale": float(payload.get("mesh_scale", 1.0)),
    }


def _split_composite(path: Path, size: int) -> Dict[int, Image.Image]:
    with Image.open(path) as source:
        image = source.convert("RGB")
    if image.size != (3 * size, size):
        raise ValueError(f"expected a 3-view composite of {(3 * size, size)}, got {image.size}")
    return {
        angle: image.crop((index * size, 0, (index + 1) * size, size))
        for index, angle in enumerate(ANGLES)
    }


def _split_alpha(path: Path, size: int) -> Dict[int, Image.Image]:
    with Image.open(path) as source:
        if "A" in source.getbands():
            image = source.getchannel("A")
        else:
            image = source.convert("L")
    if image.size != (3 * size, size):
        raise ValueError(f"expected a 3-view foreground mask of {(3 * size, size)}, got {image.size}")
    return {
        angle: image.crop((index * size, 0, (index + 1) * size, size))
        for index, angle in enumerate(ANGLES)
    }


def _array_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _array_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0 > 0.5


def _psnr(mse: float) -> Optional[float]:
    if not math.isfinite(mse):
        return None
    if mse <= 0.0:
        return None
    return float(10.0 * math.log10(1.0 / mse))


def _make_boundary(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import binary_dilation, binary_erosion

        boundary = binary_dilation(mask, iterations=8) ^ binary_erosion(mask, iterations=8)
        return boundary if bool(boundary.any()) else mask
    except Exception:
        return mask


def _make_lpips() -> Tuple[Optional[Any], Optional[str]]:
    try:
        import lpips

        return lpips.LPIPS(net="alex", verbose=False).eval(), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _lpips_native_patches(
    metric: Any,
    reference: np.ndarray,
    prediction: np.ndarray,
    patch_size: int = 512,
) -> Optional[float]:
    if metric is None:
        return None
    values: List[float] = []
    try:
        with torch.no_grad():
            for y in range(0, reference.shape[0], patch_size):
                for x in range(0, reference.shape[1], patch_size):
                    ref_patch = (
                        torch.from_numpy(reference[y : y + patch_size, x : x + patch_size])
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        * 2.0
                        - 1.0
                    )
                    pred_patch = (
                        torch.from_numpy(prediction[y : y + patch_size, x : x + patch_size])
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        * 2.0
                        - 1.0
                    )
                    values.append(float(metric(pred_patch, ref_patch).mean()))
    except Exception:
        return None
    return float(np.mean(values)) if values else None


def _compute_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    reference_mask: np.ndarray,
    prediction_mask: np.ndarray,
    lpips_metric: Any,
) -> Dict[str, Any]:
    from skimage.metrics import structural_similarity

    diff = prediction - reference
    full_mse = float(np.mean(diff * diff))
    foreground = reference_mask
    foreground_mse = float(np.mean(diff[foreground] ** 2)) if bool(foreground.any()) else float("nan")
    boundary = _make_boundary(foreground)
    boundary_mse = float(np.mean(diff[boundary] ** 2)) if bool(boundary.any()) else float("nan")

    _, ssim_map = structural_similarity(
        reference, prediction, data_range=1.0, channel_axis=2, full=True
    )
    ssim_pixels = np.mean(ssim_map, axis=2) if ssim_map.ndim == 3 else ssim_map
    intersection = np.logical_and(reference_mask, prediction_mask).sum()
    union = np.logical_or(reference_mask, prediction_mask).sum()

    return {
        "psnr_db": _psnr(full_mse),
        "foreground_psnr_db": _psnr(foreground_mse),
        "boundary_band_psnr_db": _psnr(boundary_mse),
        "ssim": float(np.mean(ssim_map)),
        "foreground_ssim": float(np.mean(ssim_pixels[foreground])) if bool(foreground.any()) else None,
        "boundary_band_ssim": float(np.mean(ssim_pixels[boundary])) if bool(boundary.any()) else None,
        "lpips_alex_native_512_patch": _lpips_native_patches(
            lpips_metric, reference, prediction
        ),
        "mae": float(np.mean(np.abs(diff))),
        "foreground_mae": float(np.mean(np.abs(diff[foreground]))) if bool(foreground.any()) else None,
        "reference_foreground_fraction": float(foreground.mean()),
        "render_foreground_fraction": float(prediction_mask.mean()),
        "alpha_iou": float(intersection / max(1, union)),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metric_names = (
        "psnr_db",
        "foreground_psnr_db",
        "boundary_band_psnr_db",
        "ssim",
        "foreground_ssim",
        "boundary_band_ssim",
        "lpips_alex_native_512_patch",
        "mae",
        "foreground_mae",
        "alpha_iou",
    )
    result: Dict[str, Any] = {}
    for name in metric_names:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        result[name] = {
            "count": len(values),
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
            "min": float(np.min(values)) if values else None,
            "max": float(np.max(values)) if values else None,
        }
    return result


def _logical_device(cuda_device: int) -> Tuple[torch.device, str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.strip():
        # With CUDA_VISIBLE_DEVICES=4, physical GPU 4 is logical cuda:0.
        device = torch.device("cuda:0")
    else:
        device = torch.device(f"cuda:{int(cuda_device)}")
    return device, visible


def _render_views(
    mesh: MeshWithVertexPbr,
    camera: Mapping[str, float],
    output_dir: Path,
    resolution: int,
    device: torch.device,
    force: bool,
) -> Dict[int, Dict[str, str]]:
    from pixal3d.renderers import PbrMeshRenderer
    from render_pixal3d_raw_ovoxel import load_envmap

    output_dir.mkdir(parents=True, exist_ok=True)
    extrinsics, intrinsics, _ = baseline_render._make_camera_views(
        float(camera["camera_angle_x"]), float(camera["distance"]), ANGLES
    )
    renderer = PbrMeshRenderer(
        rendering_options={
            "resolution": int(resolution),
            "near": max(0.01, float(camera["distance"]) - 2.0),
            "far": float(camera["distance"]) + 10.0,
            "ssaa": 1,
            "peel_layers": 8,
            "face_chunk_size": 4_000_000,
        },
        device=str(device),
    )
    envmap = load_envmap("studio", device=device)
    live = mesh.to(device)
    records: Dict[int, Dict[str, str]] = {}
    try:
        for angle in ANGLES:
            view_dir = output_dir / f"view_{angle:03d}"
            rgb_path = view_dir / f"render_rgb_{resolution}.png"
            alpha_path = view_dir / f"render_alpha_{resolution}.png"
            normal_path = view_dir / f"render_normal_camera_{resolution}.png"
            base_color_path = view_dir / f"pbr_base_color_{resolution}.png"
            if not force and all(
                path.is_file() for path in (rgb_path, alpha_path, normal_path, base_color_path)
            ):
                print(f"[render] reuse yaw={angle}: {view_dir}", flush=True)
            else:
                print(f"[render] yaw={angle} resolution={resolution}", flush=True)
                result = renderer.render(
                    live,
                    extrinsics[angle].to(device),
                    intrinsics.to(device),
                    envmap=envmap,
                    use_envmap_bg=False,
                )
                _save_tensor_image(result["shaded"], rgb_path, channels=3)
                _save_tensor_image(result["mask"], alpha_path, channels=1)
                _save_tensor_image(result["normal"], normal_path, channels=3)
                _save_tensor_image(result["base_color"], base_color_path, channels=3)
                del result
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            records[angle] = {
                "rgb_path": str(rgb_path),
                "alpha_path": str(alpha_path),
                "normal_path": str(normal_path),
                "base_color_path": str(base_color_path),
            }
    finally:
        del live, envmap, renderer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    sheet = Image.new("RGB", (3 * resolution, resolution), "black")
    for index, angle in enumerate(ANGLES):
        with Image.open(records[angle]["rgb_path"]) as image:
            sheet.paste(image.convert("RGB"), (index * resolution, 0))
    sheet.save(output_dir / f"render_three_view_sheet_{resolution}.png")
    return records


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multi-view rendering")
    if args.resolution < 64:
        raise ValueError("resolution is unexpectedly small")
    device, visible = _logical_device(args.cuda_device)
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"requested logical device {device}, visible CUDA count is {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(device)

    references = _split_composite(args.input, args.input_size)
    masks = _split_alpha(args.alpha, args.input_size)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for angle in ANGLES:
        references[angle].save(output_dir / f"input_view_{angle:03d}.png")
        masks[angle].save(output_dir / f"input_mask_{angle:03d}.png")

    camera = _load_camera(args.camera)
    mesh = _load_mesh(args.mesh)
    print(
        f"[runtime] physical_cuda_requested={args.cuda_device} "
        f"CUDA_VISIBLE_DEVICES={visible or '<unset>'} logical_device={device} "
        f"vertices={mesh.vertices.shape[0]} faces={mesh.faces.shape[0]}",
        flush=True,
    )
    records = _render_views(
        mesh=mesh,
        camera=camera,
        output_dir=output_dir,
        resolution=args.resolution,
        device=device,
        force=args.force,
    )

    lpips_metric, lpips_error = _make_lpips()
    if lpips_metric is None:
        print(f"[metrics] LPIPS unavailable: {lpips_error}", flush=True)
    else:
        print("[metrics] AlexNet LPIPS loaded", flush=True)

    rows: List[Dict[str, Any]] = []
    for angle in ANGLES:
        rgb_path = Path(records[angle]["rgb_path"])
        alpha_path = Path(records[angle]["alpha_path"])
        reference = np.asarray(references[angle], dtype=np.float32) / 255.0
        prediction = _array_rgb(rgb_path)
        reference_mask = np.asarray(masks[angle], dtype=np.uint8) > 127
        prediction_mask = _array_mask(alpha_path)
        if prediction.shape != reference.shape:
            raise RuntimeError(
                f"yaw {angle}: rendered shape {prediction.shape} != reference {reference.shape}"
            )
        metric_row = _compute_metrics(
            reference, prediction, reference_mask, prediction_mask, lpips_metric
        )
        row = {
            "yaw_deg": angle,
            "resolution": [int(prediction.shape[1]), int(prediction.shape[0])],
            "reference_path": str(output_dir / f"input_view_{angle:03d}.png"),
            "reference_mask_path": str(output_dir / f"input_mask_{angle:03d}.png"),
            **records[angle],
            "metrics": metric_row,
        }
        rows.append(row)
        print(
            f"[metrics] yaw={angle}: PSNR={metric_row['psnr_db']:.4f} dB, "
            f"FG-PSNR={metric_row['foreground_psnr_db']:.4f} dB, "
            f"SSIM={metric_row['ssim']:.6f}, "
            f"LPIPS={metric_row['lpips_alex_native_512_patch']}",
            flush=True,
        )
    # Flatten per-view metrics at the top-level row for convenient aggregation.
    flat_rows = [dict(row["metrics"]) for row in rows]
    payload = {
        "format": "pixal3d_multiview_render_metrics_v1",
        "status": "complete",
        "cuda": {
            "physical_device_requested": int(args.cuda_device),
            "cuda_visible_devices": visible or None,
            "logical_device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "mesh": str(args.mesh.resolve()),
        "camera": dict(camera),
        "input_composite": str(args.input.resolve()),
        "input_alpha_composite": str(args.alpha.resolve()),
        "angles_deg": list(ANGLES),
        "render_resolution": int(args.resolution),
        "camera_convention": "baseline _make_camera_views; attached input panels correspond to yaw 0/120/240",
        "metric_definition": {
            "full_frame": "RGB PNG values in [0,1], data_range=1",
            "foreground": "reference alpha > 0.5 from image2_bg_removed.png",
            "boundary": "8-pixel dilation XOR erosion of the reference foreground mask",
            "lpips": "AlexNet LPIPS averaged over native 512x512 patches; no upsampling",
        },
        "lpips_error": lpips_error,
        "views": rows,
        "aggregate": _aggregate(flat_rows),
        "artifacts": {
            "three_view_sheet": str(output_dir / f"render_three_view_sheet_{args.resolution}.png"),
            "metrics": str(output_dir / "metrics.json"),
        },
    }
    _write_json(output_dir / "metrics.json", payload)
    print(f"[done] metrics={output_dir / 'metrics.json'}", flush=True)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--alpha", type=Path, default=DEFAULT_ALPHA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
