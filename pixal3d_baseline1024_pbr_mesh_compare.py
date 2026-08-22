#!/usr/bin/env python3
"""Pixal3D native 1024 PBR representation comparison.

This is intentionally a self-contained experiment driver.  It runs the
repository's native 1024 cascade once, keeps the decoder ``MeshWithVoxel``
unchanged, creates centroid-sampled per-face and vertex-sampled PBR views of
the same geometry, and sends all three representations through the native
``PbrMeshRenderer`` under identical cameras and lighting.

The script does not export GLB, bake UVs, create an atlas, or alter the mesh
topology.  The only material changes are the explicitly requested PBR
representations.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from pixal3d.representations import (
    MeshWithFacePbr,
    MeshWithVertexPbr,
    MeshWithVoxel,
)
from pixal3d.renderers import PbrMeshRenderer
from pixal3d.utils import render_utils


PIPELINE_TYPE = "1024_cascade"
ANGLES_DEG = (0, 120, 240)
DEFAULT_QUERY_CHUNK_SIZE = 262_144
DEFAULT_FACE_CHUNK_SIZE = 4_000_000
DEFAULT_MODEL_PATH = "/home/nvme04/yyyan/download/model/Pixal3D"
DEFAULT_MOGE_MODEL = "/home/nvme04/yyyan/download/model/moge-2-vitl/model.pt"
DEFAULT_IMAGE = "/home/nvme04/yyyan/Pixal3D/assets/choose/0_img.png"

EXPECTED_LAYOUT = {
    "base_color": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}
PBR_CHANNEL_NAMES = ("base_color", "metallic", "roughness", "alpha")
RENDER_MODES = ("shaded", "base_color", "normal", "metallic", "roughness", "alpha")


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


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _layout_to_json(layout: Mapping[str, slice]) -> Dict[str, Sequence[int]]:
    result: Dict[str, Sequence[int]] = {}
    for name in PBR_CHANNEL_NAMES:
        value = layout[name]
        result[name] = [int(value.start), int(value.stop)]
    return result


def _validate_layout(layout: Mapping[str, slice]) -> None:
    for name, expected in EXPECTED_LAYOUT.items():
        actual = layout.get(name)
        if not isinstance(actual, slice):
            raise AssertionError(f"layout[{name!r}] is not a slice: {actual!r}")
        if (actual.start, actual.stop, actual.step) != (
            expected.start,
            expected.stop,
            expected.step,
        ):
            raise AssertionError(
                f"unexpected native PBR layout for {name}: {actual!r}; "
                f"expected {expected!r}"
            )


def _finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all().item():
        raise AssertionError(f"{name} contains NaN or Inf")


@torch.no_grad()
def query_pbr(
    mesh: MeshWithVoxel,
    xyz: torch.Tensor,
    query_chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
) -> torch.Tensor:
    """Query the native sparse O-Voxel PBR field at raw mesh coordinates.

    The lookup body deliberately matches ``MeshWithVoxel.query_attrs`` and
    the live ``PbrMeshRenderer`` path.  Chunking only limits the number of
    query points in one CUDA call; it does not change interpolation.
    """
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must be [N, 3], got {tuple(xyz.shape)}")
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")
    from flex_gemm.ops.grid_sample import grid_sample_3d

    ov_coords = torch.cat(
        [torch.zeros_like(mesh.coords[..., :1]), mesh.coords],
        dim=-1,
    )
    chunks = []
    for xyz_chunk in xyz.split(int(query_chunk_size), dim=0):
        xyz_voxel = (xyz_chunk - mesh.origin) / mesh.voxel_size
        queried = grid_sample_3d(
            mesh.attrs,
            ov_coords,
            mesh.voxel_shape,
            xyz_voxel.reshape(1, -1, 3),
            mode="trilinear",
        )[0]
        chunks.append(queried)
    if not chunks:
        return mesh.attrs.new_empty((0, mesh.attrs.shape[-1]))
    result = torch.cat(chunks, dim=0)
    _finite_tensor("queried PBR", result)
    return result


@torch.no_grad()
def build_pbr_representations(
    raw_mesh: MeshWithVoxel,
    query_chunk_size: int,
) -> Tuple[MeshWithFacePbr, MeshWithVertexPbr]:
    _validate_layout(raw_mesh.layout)
    if raw_mesh.attrs.ndim != 2 or raw_mesh.attrs.shape[-1] != 6:
        raise AssertionError(
            "native Pixal3D PBR attrs must be [active_ovoxels, 6], "
            f"got {tuple(raw_mesh.attrs.shape)}"
        )
    _finite_tensor("raw vertices", raw_mesh.vertices)
    _finite_tensor("raw attrs", raw_mesh.attrs)

    faces = raw_mesh.faces
    vertices = raw_mesh.vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_centers = (v0 + v1 + v2) / 3.0
    face_attrs = query_pbr(raw_mesh, face_centers, query_chunk_size)
    vertex_attrs = query_pbr(raw_mesh, vertices, query_chunk_size)
    del v0, v1, v2, face_centers

    _finite_tensor("face attrs", face_attrs)
    _finite_tensor("vertex attrs", vertex_attrs)
    if face_attrs.shape != (faces.shape[0], 6):
        raise AssertionError(
            f"face_attrs must be [{faces.shape[0]}, 6], got {tuple(face_attrs.shape)}"
        )
    if vertex_attrs.shape != (vertices.shape[0], 6):
        raise AssertionError(
            f"vertex_attrs must be [{vertices.shape[0]}, 6], "
            f"got {tuple(vertex_attrs.shape)}"
        )

    per_face = MeshWithFacePbr(
        vertices=vertices,
        faces=faces,
        face_attrs=face_attrs,
        layout=raw_mesh.layout,
    )
    per_vertex = MeshWithVertexPbr(
        vertices=vertices,
        faces=faces,
        vertex_attrs=vertex_attrs,
        layout=raw_mesh.layout,
    )

    # Geometry and topology are shared exactly; this catches accidental
    # remeshing or an attribute/topology alignment error immediately.
    assert per_face.vertices.shape == raw_mesh.vertices.shape
    assert torch.equal(per_face.faces, raw_mesh.faces)
    assert per_vertex.vertices.shape == raw_mesh.vertices.shape
    assert torch.equal(per_vertex.faces, raw_mesh.faces)
    assert face_attrs.shape[0] == raw_mesh.faces.shape[0]
    assert vertex_attrs.shape[0] == raw_mesh.vertices.shape[0]
    return per_face, per_vertex


def _raw_checkpoint_payload(mesh: MeshWithVoxel) -> Dict[str, Any]:
    return {
        "vertices": mesh.vertices.detach().cpu(),
        "faces": mesh.faces.detach().cpu(),
        "coords": mesh.coords.detach().cpu(),
        "attrs": mesh.attrs.detach().cpu(),
        "origin": mesh.origin.detach().cpu(),
        "voxel_size": float(mesh.voxel_size),
        "voxel_shape": list(mesh.voxel_shape),
        "layout": dict(mesh.layout),
        "representation": "raw_ovoxel",
        "pipeline_type": PIPELINE_TYPE,
    }


def _face_checkpoint_payload(mesh: MeshWithFacePbr) -> Dict[str, Any]:
    return {
        "vertices": mesh.vertices.detach().cpu(),
        "faces": mesh.faces.detach().cpu(),
        "face_attrs": mesh.face_attrs.detach().cpu(),
        "layout": dict(mesh.layout),
        "representation": "per_face_pbr",
        "pipeline_type": PIPELINE_TYPE,
    }


def _vertex_checkpoint_payload(mesh: MeshWithVertexPbr) -> Dict[str, Any]:
    return {
        "vertices": mesh.vertices.detach().cpu(),
        "faces": mesh.faces.detach().cpu(),
        "vertex_attrs": mesh.vertex_attrs.detach().cpu(),
        "layout": dict(mesh.layout),
        "representation": "per_vertex_pbr",
        "pipeline_type": PIPELINE_TYPE,
    }


def _stats(attrs: torch.Tensor, layout: Mapping[str, slice]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for name in PBR_CHANNEL_NAMES:
        channel = attrs[..., layout[name]].float()
        result[name] = {
            "shape": list(channel.shape),
            "min": float(channel.amin().item()),
            "max": float(channel.amax().item()),
            "mean": float(channel.mean().item()),
        }
    return result


def _make_camera_views(
    camera_angle_x: float,
    distance: float,
    angles_deg: Iterable[int] = ANGLES_DEG,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    extr_front, intrinsics = render_utils.proj_camera_to_render_params(
        camera_angle_x=float(camera_angle_x),
        distance=float(distance),
    )
    if extr_front.shape != (4, 4) or intrinsics.shape != (3, 3):
        raise AssertionError("native camera matrices have unexpected shapes")
    _finite_tensor("front extrinsics", extr_front)
    _finite_tensor("intrinsics", intrinsics)

    extrinsics: Dict[int, torch.Tensor] = {}
    for angle_deg in angles_deg:
        angle = math.radians(float(angle_deg))
        c = math.cos(angle)
        s = math.sin(angle)
        r_y = torch.tensor(
            [
                [c, 0.0, s, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-s, 0.0, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=extr_front.dtype,
            device=extr_front.device,
        )
        # Exactly the rotation convention used by render_proj_aligned_video.
        r_y_inv = r_y.clone()
        r_y_inv[:3, :3] = r_y[:3, :3].T
        extrinsics[int(angle_deg)] = extr_front @ r_y_inv
        _finite_tensor(f"yaw {angle_deg} extrinsics", extrinsics[int(angle_deg)])
    return extrinsics, intrinsics, extr_front


def _tensor_to_hwc(value: torch.Tensor) -> np.ndarray:
    value = value.detach().float().cpu()
    if value.ndim == 3:
        return value.permute(1, 2, 0).numpy()
    if value.ndim == 2:
        return value.numpy()
    raise ValueError(f"unexpected render tensor shape: {tuple(value.shape)}")


def _image_from_array(array: np.ndarray) -> Image.Image:
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    array = np.clip(array, 0.0, 1.0)
    if array.ndim == 2:
        return Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), mode="L").convert("RGB")
    if array.ndim == 3 and array.shape[2] == 1:
        return Image.fromarray((array[..., 0] * 255.0 + 0.5).astype(np.uint8), mode="L").convert("RGB")
    if array.ndim == 3 and array.shape[2] == 3:
        return Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    raise ValueError(f"cannot save array with shape {array.shape} as image")


def _save_render(
    result: Mapping[str, torch.Tensor],
    output_dir: Path,
) -> Dict[str, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: Dict[str, np.ndarray] = {}
    for mode in RENDER_MODES + ("mask",):
        if mode not in result:
            raise KeyError(f"native renderer did not return {mode!r}")
        array = _tensor_to_hwc(result[mode])
        rendered[mode] = array.astype(np.float32, copy=False)
        _image_from_array(array).save(output_dir / f"{mode}.png")
    return rendered


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _contact_sheet(
    rendered: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    mode: str,
    output_path: Path,
    variants: Sequence[str],
    angles: Sequence[int],
) -> None:
    first = _image_from_array(rendered[variants[0]][angles[0]][mode])
    width, height = first.size
    left_margin, top_margin = 150, 40
    cell_gap = 4
    canvas = Image.new(
        "RGB",
        (
            left_margin + len(angles) * width + (len(angles) - 1) * cell_gap,
            top_margin + len(variants) * height + (len(variants) - 1) * cell_gap,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = _font()
    for col, angle in enumerate(angles):
        x = left_margin + col * (width + cell_gap)
        draw.text((x + 5, 10), f"yaw{angle:03d}", fill="black", font=font)
    for row, variant in enumerate(variants):
        y = top_margin + row * (height + cell_gap)
        draw.text((8, y + 8), variant, fill="black", font=font)
        for col, angle in enumerate(angles):
            x = left_margin + col * (width + cell_gap)
            canvas.paste(_image_from_array(rendered[variant][angle][mode]), (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _pbr_channel_sheet(
    rendered: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    output_path: Path,
    variants: Sequence[str],
    angles: Sequence[int],
) -> None:
    first = _image_from_array(rendered[variants[0]][angles[0]]["base_color"])
    width, height = first.size
    sub_width = max(1, width // 3)
    left_margin, top_margin, label_height, gap = 150, 40, 20, 4
    canvas = Image.new(
        "RGB",
        (
            left_margin + len(angles) * width + (len(angles) - 1) * gap,
            top_margin + len(variants) * (height + label_height) + (len(variants) - 1) * gap,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = _font()
    for col, angle in enumerate(angles):
        x = left_margin + col * (width + gap)
        draw.text((x + 5, 10), f"yaw{angle:03d}", fill="black", font=font)
    for row, variant in enumerate(variants):
        y = top_margin + row * (height + label_height + gap)
        draw.text((8, y + 8), variant, fill="black", font=font)
        for col, angle in enumerate(angles):
            x = left_margin + col * (width + gap)
            cell = Image.new("RGB", (width, height + label_height), "white")
            for index, mode in enumerate(("base_color", "metallic", "roughness")):
                tile = _image_from_array(rendered[variant][angle][mode]).resize(
                    (sub_width, height), Image.Resampling.BILINEAR
                )
                cell.paste(tile, (index * sub_width, label_height))
                draw_cell = ImageDraw.Draw(cell)
                draw_cell.text((index * sub_width + 3, 3), mode, fill="black", font=font)
            canvas.paste(cell, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _input_vs_raw(input_image: Image.Image, raw_shaded: np.ndarray, output_path: Path) -> None:
    raw_image = _image_from_array(raw_shaded)
    input_image = input_image.convert("RGB").resize(raw_image.size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (raw_image.width * 2, raw_image.height + 28), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font()
    draw.text((4, 4), "preprocessed Pixal3D input", fill="black", font=font)
    draw.text((raw_image.width + 4, 4), "raw_ovoxel yaw000 shaded", fill="black", font=font)
    canvas.paste(input_image, (0, 28))
    canvas.paste(raw_image, (raw_image.width, 28))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= (window_size - 1) / 2.0
    gaussian = torch.exp(-(coords.square()) / (2.0 * sigma**2))
    gaussian /= gaussian.sum()
    return torch.outer(gaussian, gaussian)


def _ssim_map(reference: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    channels = int(reference.shape[1])
    kernel = _gaussian_kernel().to(reference).expand(channels, 1, 11, 11).contiguous()
    mu_ref = F.conv2d(reference, kernel, padding=5, groups=channels)
    mu_pred = F.conv2d(prediction, kernel, padding=5, groups=channels)
    mu_ref_sq = mu_ref.square()
    mu_pred_sq = mu_pred.square()
    mu_cross = mu_ref * mu_pred
    sigma_ref = F.conv2d(reference.square(), kernel, padding=5, groups=channels) - mu_ref_sq
    sigma_pred = F.conv2d(prediction.square(), kernel, padding=5, groups=channels) - mu_pred_sq
    sigma_cross = F.conv2d(reference * prediction, kernel, padding=5, groups=channels) - mu_cross
    c1, c2 = 0.01**2, 0.03**2
    score = ((2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)) / (
        (mu_ref_sq + mu_pred_sq + c1) * (sigma_ref + sigma_pred + c2)
    )
    return torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)


def _metric_pair(
    reference: np.ndarray,
    prediction: np.ndarray,
    foreground: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    def as_tensor(array: np.ndarray) -> torch.Tensor:
        if array.ndim == 2:
            array = array[..., None]
        return torch.from_numpy(array.astype(np.float32, copy=False)).permute(2, 0, 1).unsqueeze(0)

    ref = as_tensor(reference)
    pred = as_tensor(prediction)
    diff = pred - ref
    mask = torch.from_numpy((foreground > 0.5).astype(np.float32))[None, None]
    mask_count = int(mask.sum().item())
    if mask_count < 1:
        raise AssertionError("raw reference foreground mask is empty")
    channel_count = int(ref.shape[1])
    full_mse = float(diff.square().mean().item())
    fg_mse = float((diff.square() * mask).sum().item() / (mask_count * channel_count))
    full_mae = float(diff.abs().mean().item())
    fg_mae = float((diff.abs() * mask).sum().item() / (mask_count * channel_count))
    full_psnr = float("inf") if full_mse <= 1e-12 else float(10.0 * math.log10(1.0 / full_mse))
    fg_psnr = float("inf") if fg_mse <= 1e-12 else float(10.0 * math.log10(1.0 / fg_mse))
    ssim_map = _ssim_map(ref, pred).mean(dim=1, keepdim=True)
    full_ssim = float(ssim_map.mean().item())
    fg_ssim = float((ssim_map * mask).sum().item() / mask_count)
    return {
        "full-image": {"psnr_db": full_psnr, "ssim": full_ssim, "mae": full_mae},
        "foreground-only": {"psnr_db": fg_psnr, "ssim": fg_ssim, "mae": fg_mae},
    }


def _compute_metrics(
    rendered: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    variants: Sequence[str],
    angles: Sequence[int],
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for variant in variants:
        if variant == "raw_ovoxel":
            continue
        for angle in angles:
            foreground = rendered["raw_ovoxel"][angle]["mask"]
            for mode in ("shaded", "base_color"):
                scores = _metric_pair(
                    rendered["raw_ovoxel"][angle][mode],
                    rendered[variant][angle][mode],
                    foreground,
                )
                for scope, values in scores.items():
                    rows.append(
                        {
                            "variant": variant,
                            "reference": "raw_ovoxel",
                            "yaw_deg": angle,
                            "mode": mode,
                            "scope": scope,
                            "foreground_pixels": int((foreground > 0.5).sum()),
                            **values,
                        }
                    )
    return rows


def _write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "variant",
        "reference",
        "yaw_deg",
        "mode",
        "scope",
        "foreground_pixels",
        "psnr_db",
        "ssim",
        "mae",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _native_camera_and_mesh(
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[MeshWithVoxel, Image.Image, Dict[str, float]]:
    """Run the exact native preprocessing, MoGe camera estimation, and 1024 run."""
    # Importing inference here keeps its original initialization code intact,
    # while allowing --cuda-device to select the current CUDA device first.
    from inference import (
        IMAGE_COND_CONFIGS,
        build_image_cond_model,
        distance_from_fov,
        get_camera_params_wild_moge,
        init_pipeline,
        load_moge_model,
    )
    from pixal3d.pipelines import Pixal3DImageTo3DPipeline  # noqa: F401

    pipeline = init_pipeline(
        model_path=str(args.model_path),
        device="cuda",
        low_vram=bool(args.low_vram),
    )
    input_image = Image.open(args.image)
    image_preprocessed = pipeline.preprocess_image(input_image)
    tmp_path = output_dir / "_tmp_preprocessed_for_moge.png"
    image_preprocessed.save(tmp_path)
    try:
        if args.fov > 0.0:
            camera_angle_x = float(args.fov)
            grid_point = torch.tensor([-1.0, 0.0, 0.0])
            distance = distance_from_fov(
                camera_angle_x,
                grid_point,
                torch.tensor([0.0, 512 - 1.0]),
                1.0,
                512,
            )["distance_from_x"]
            camera_params = {
                "camera_angle_x": camera_angle_x,
                "distance": distance,
                "mesh_scale": 1.0,
            }
            print(
                f"[Camera] manual FOV={math.degrees(camera_angle_x):.4f} deg "
                f"distance={distance:.6f}"
            )
        else:
            print("[MoGe-2] Loading model for native camera estimation...")
            moge_model = load_moge_model(
                device="cuda",
                model_name=str(args.moge_model),
            )
            camera_params = get_camera_params_wild_moge(
                tmp_path,
                moge_model,
                device="cuda",
                mesh_scale=1.0,
                extend_pixel=0,
                image_resolution=512,
            )
            print(
                f"[Camera] camera_angle_x={camera_params['camera_angle_x']:.8f} "
                f"distance={camera_params['distance']:.8f}"
            )
            moge_model.cpu()
            del moge_model
            torch.cuda.empty_cache()

        # Keep the sampler values identical to inference.py's native defaults.
        ss_sampler = {
            "steps": 12,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.7,
            "rescale_t": 5.0,
        }
        shape_sampler = {
            "steps": 12,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.5,
            "rescale_t": 3.0,
        }
        tex_sampler = {
            "steps": 12,
            "guidance_strength": 1.0,
            "guidance_rescale": 0.0,
            "rescale_t": 3.0,
        }
        torch.manual_seed(int(args.seed))
        print(f"[Inference] Running native pipeline_type={PIPELINE_TYPE}")
        mesh_list, (_, _, resolution) = pipeline.run(
            image_preprocessed,
            camera_params=camera_params,
            seed=int(args.seed),
            sparse_structure_sampler_params=ss_sampler,
            shape_slat_sampler_params=shape_sampler,
            tex_slat_sampler_params=tex_sampler,
            preprocess_image=False,
            return_latent=True,
            pipeline_type=PIPELINE_TYPE,
            max_num_tokens=int(args.max_num_tokens),
        )
        if int(resolution) != 1024:
            raise AssertionError(f"1024 cascade returned decoder resolution {resolution}")
        raw_mesh = mesh_list[0]
        if not isinstance(raw_mesh, MeshWithVoxel):
            raise TypeError(f"native decoder returned {type(raw_mesh)!r}, expected MeshWithVoxel")
        return raw_mesh, image_preprocessed, camera_params
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
        # The models are no longer needed after decode; retain only the mesh.
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path(DEFAULT_IMAGE))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--moge-model", type=Path, default=Path(DEFAULT_MOGE_MODEL))
    parser.add_argument("--cuda-device", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--ssaa", type=int, default=2)
    parser.add_argument("--peel-layers", type=int, default=8)
    parser.add_argument("--face-chunk-size", type=int, default=DEFAULT_FACE_CHUNK_SIZE)
    parser.add_argument("--query-chunk-size", type=int, default=DEFAULT_QUERY_CHUNK_SIZE)
    parser.add_argument("--envmap", type=str, default="studio")
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--fov",
        type=float,
        default=-1.0,
        help="manual horizontal FOV in radians; default uses native MoGe estimation",
    )
    parser.add_argument("--max-num-tokens", type=int, default=1_000_000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if args.cuda_device < 0:
        raise ValueError("--cuda-device must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native Pixal3D experiment")
    if args.render_resolution < 1 or args.ssaa < 1 or args.peel_layers < 1:
        raise ValueError("render-resolution, ssaa, and peel-layers must be positive")
    if args.face_chunk_size < 0 or args.query_chunk_size < 1:
        raise ValueError("face-chunk-size must be non-negative and query-chunk-size positive")
    if args.cuda_device >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {args.cuda_device} is unavailable; "
            f"device_count={torch.cuda.device_count()}"
        )

    torch.cuda.set_device(args.cuda_device)
    device = torch.device("cuda", args.cuda_device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[GPU] cuda:{args.cuda_device} {torch.cuda.get_device_name(args.cuda_device)}")
    print(f"[Output] {output_dir}")

    raw_mesh, preprocessed_image, camera_params = _native_camera_and_mesh(args, output_dir)
    if raw_mesh.device != device:
        raw_mesh = raw_mesh.to(device)
    _validate_layout(raw_mesh.layout)
    _finite_tensor("raw vertices", raw_mesh.vertices)
    _finite_tensor("raw faces", raw_mesh.faces)
    _finite_tensor("raw coords", raw_mesh.coords)
    _finite_tensor("raw attrs", raw_mesh.attrs)
    if raw_mesh.faces.ndim != 2 or raw_mesh.faces.shape[1] != 3:
        raise AssertionError("raw faces must be [M, 3]")
    if raw_mesh.vertices.ndim != 2 or raw_mesh.vertices.shape[1] != 3:
        raise AssertionError("raw vertices must be [N, 3]")
    if raw_mesh.coords.ndim != 2 or raw_mesh.coords.shape[1] != 3:
        raise AssertionError("raw coords must be [L, 3]")

    print(
        f"[Decoder mesh] vertices={raw_mesh.vertices.shape[0]:,} "
        f"faces={raw_mesh.faces.shape[0]:,} active_ovoxels={raw_mesh.coords.shape[0]:,}"
    )
    per_face, per_vertex = build_pbr_representations(raw_mesh, args.query_chunk_size)

    raw_path = output_dir / "raw_ovoxel_mesh.pt"
    face_path = output_dir / "per_face_pbr_mesh.pt"
    vertex_path = output_dir / "per_vertex_pbr_mesh.pt"
    _atomic_torch_save(raw_path, _raw_checkpoint_payload(raw_mesh))
    _atomic_torch_save(face_path, _face_checkpoint_payload(per_face))
    _atomic_torch_save(vertex_path, _vertex_checkpoint_payload(per_vertex))
    print(f"[Checkpoint] {raw_path}")
    print(f"[Checkpoint] {face_path}")
    print(f"[Checkpoint] {vertex_path}")

    angles = list(ANGLES_DEG)
    extrinsics, intrinsics, extr_front = _make_camera_views(
        camera_params["camera_angle_x"],
        camera_params["distance"],
        angles,
    )
    near = max(0.01, float(camera_params["distance"]) - 2.0)
    far = float(camera_params["distance"]) + 10.0
    if not (0.0 < near < far):
        raise AssertionError(f"invalid clipping range near={near} far={far}")

    from render_pixal3d_raw_ovoxel import load_envmap

    envmap = load_envmap(args.envmap, device=device)
    renderer = PbrMeshRenderer(
        rendering_options={
            "resolution": int(args.render_resolution),
            "near": near,
            "far": far,
            "ssaa": int(args.ssaa),
            "peel_layers": int(args.peel_layers),
            "face_chunk_size": int(args.face_chunk_size),
        },
        device=f"cuda:{args.cuda_device}",
    )
    variants = {
        "raw_ovoxel": raw_mesh,
        "per_face_pbr": per_face,
        "per_vertex_pbr": per_vertex,
    }
    rendered: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    for variant_name, mesh in variants.items():
        rendered[variant_name] = {}
        for angle in angles:
            print(f"[Render] {variant_name} yaw={angle} degrees")
            # The native SSAO kernel draws a per-pixel random hemisphere
            # sample.  Reset the same CUDA RNG state for every representation
            # at a given yaw so SSAO is identical; only PBR representation can
            # then change the comparison.
            render_seed = int(args.seed) + 100_000 + int(angle)
            torch.cuda.manual_seed_all(render_seed)
            result = renderer.render(
                mesh,
                extrinsics[angle],
                intrinsics,
                envmap=envmap,
                use_envmap_bg=False,
            )
            render_dir = output_dir / variant_name / f"yaw{angle:03d}"
            rendered[variant_name][angle] = _save_render(result, render_dir)
            del result
            torch.cuda.empty_cache()

    _contact_sheet(
        rendered,
        "shaded",
        output_dir / "comparison_shaded.png",
        list(variants),
        angles,
    )
    _contact_sheet(
        rendered,
        "base_color",
        output_dir / "comparison_base_color.png",
        list(variants),
        angles,
    )
    _pbr_channel_sheet(
        rendered,
        output_dir / "comparison_pbr_channels.png",
        list(variants),
        angles,
    )
    _input_vs_raw(
        preprocessed_image,
        rendered["raw_ovoxel"][0]["shaded"],
        output_dir / "input_vs_raw_yaw000.png",
    )

    metric_rows = _compute_metrics(rendered, list(variants), angles)
    _write_metrics_csv(output_dir / "metrics.csv", metric_rows)
    _atomic_json(
        output_dir / "metrics.json",
        {"reference": "raw_ovoxel", "rows": metric_rows},
    )

    summary = {
        "pipeline_type": PIPELINE_TYPE,
        "seed": int(args.seed),
        "cuda_device": int(args.cuda_device),
        "gpu": torch.cuda.get_device_name(args.cuda_device),
        "camera_angle_x": float(camera_params["camera_angle_x"]),
        "distance": float(camera_params["distance"]),
        "near": near,
        "far": far,
        "num_vertices": int(raw_mesh.vertices.shape[0]),
        "num_faces": int(raw_mesh.faces.shape[0]),
        "num_active_ovoxels": int(raw_mesh.coords.shape[0]),
        "face_attrs_shape": list(per_face.face_attrs.shape),
        "vertex_attrs_shape": list(per_vertex.vertex_attrs.shape),
        "angles_deg": angles,
        "render_resolution": int(args.render_resolution),
        "ssaa": int(args.ssaa),
        "peel_layers": int(args.peel_layers),
        "face_chunk_size": int(args.face_chunk_size),
        "query_chunk_size": int(args.query_chunk_size),
        "envmap": str(getattr(envmap, "source_path", args.envmap)),
        "camera_convention": {
            "front": "proj_camera_to_render_params(camera_angle_x, distance)",
            "front_camera_position": [0.0, 0.0, float(camera_params["distance"])],
            "target": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "rotation": "extr_front @ inverse(R_y(angle))",
        },
        "pbr_layout": _layout_to_json(raw_mesh.layout),
        "pbr_ranges": {
            "raw_attrs": _stats(raw_mesh.attrs, raw_mesh.layout),
            "face_attrs": _stats(per_face.face_attrs, per_face.layout),
            "vertex_attrs": _stats(per_vertex.vertex_attrs, per_vertex.layout),
        },
        "artifacts": {
            "raw_ovoxel_mesh": str(raw_path),
            "per_face_pbr_mesh": str(face_path),
            "per_vertex_pbr_mesh": str(vertex_path),
            "comparison_shaded": str(output_dir / "comparison_shaded.png"),
            "comparison_base_color": str(output_dir / "comparison_base_color.png"),
            "comparison_pbr_channels": str(output_dir / "comparison_pbr_channels.png"),
            "input_vs_raw_yaw000": str(output_dir / "input_vs_raw_yaw000.png"),
            "metrics_csv": str(output_dir / "metrics.csv"),
            "metrics_json": str(output_dir / "metrics.json"),
        },
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(f"[Done] summary={output_dir / 'summary.json'}")
    print(f"[Done] metrics={output_dir / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
