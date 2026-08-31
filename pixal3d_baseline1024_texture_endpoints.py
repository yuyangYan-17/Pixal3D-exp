#!/usr/bin/env python3
"""Run native Pixal3D 1024 flow and render every texture x0 endpoint."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import torch
from PIL import Image, ImageDraw

from pixal3d.representations import MeshWithVoxel
from pixal3d.renderers import PbrMeshRenderer
from pixal3d_baseline1024_pbr_mesh_compare import (
    DEFAULT_IMAGE,
    DEFAULT_MODEL_PATH,
    DEFAULT_MOGE_MODEL,
    _image_from_array,
    _make_camera_views,
    _save_render,
)


def _save_endpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _endpoint_sheet(paths: list[Path], output_path: Path, columns: int = 4) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    if not images:
        return
    width, height = images[0].size
    label_height = 24
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (path, image) in enumerate(zip(paths, images)):
        x = index % columns * width
        y = index // columns * (height + label_height)
        draw.text((x + 4, y + 4), path.parent.parent.name, fill="black")
        sheet.paste(image, (x, y + label_height))
    sheet.save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path(DEFAULT_IMAGE))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/baseline1024_texture_endpoints"),
    )
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--moge-model", type=Path, default=Path(DEFAULT_MOGE_MODEL))
    parser.add_argument("--cuda-device", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fov", type=float, default=-1.0)
    parser.add_argument("--max-num-tokens", type=int, default=1_000_000)
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--ssaa", type=int, default=2)
    parser.add_argument("--peel-layers", type=int, default=8)
    parser.add_argument("--face-chunk-size", type=int, default=4_000_000)
    parser.add_argument("--envmap", type=str, default="studio")
    parser.add_argument("--angles", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--low-vram", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


@torch.no_grad()
def main() -> int:
    args = build_parser().parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 <= args.cuda_device < torch.cuda.device_count():
        raise ValueError(f"invalid CUDA device {args.cuda_device}")
    torch.cuda.set_device(args.cuda_device)
    device = torch.device("cuda", args.cuda_device)
    output_dir = args.output_dir.expanduser().resolve()
    endpoint_dir = output_dir / "endpoint_latents"
    endpoint_dir.mkdir(parents=True, exist_ok=True)

    from inference import distance_from_fov, get_camera_params_wild_moge, init_pipeline, load_moge_model

    pipeline = init_pipeline(str(args.model_path), device="cuda", low_vram=args.low_vram)
    image = pipeline.preprocess_image(Image.open(args.image))
    tmp_image = output_dir / "_tmp_preprocessed_for_moge.png"
    image.save(tmp_image)
    try:
        if args.fov > 0:
            distance = distance_from_fov(
                float(args.fov),
                torch.tensor([-1.0, 0.0, 0.0]),
                torch.tensor([0.0, 511.0]),
                1.0,
                512,
            )["distance_from_x"]
            camera = {
                "camera_angle_x": float(args.fov),
                "distance": float(distance),
                "mesh_scale": 1.0,
            }
        else:
            moge = load_moge_model(device="cuda", model_name=str(args.moge_model))
            camera = get_camera_params_wild_moge(
                tmp_image,
                moge,
                device="cuda",
                mesh_scale=1.0,
                extend_pixel=0,
                image_resolution=512,
            )
            moge.cpu()
            del moge
            gc.collect()
            torch.cuda.empty_cache()

        endpoint_records: list[dict[str, Any]] = []

        def save_texture_endpoint(**payload: Any) -> None:
            step = int(payload["step_index"])
            endpoint = payload["endpoint"]
            x_t = payload["x_t"]
            velocity = payload["v"]
            t = float(payload["t"])
            expected = pipeline.tex_slat_sampler._pred_to_xstart(x_t, t, velocity)
            error = float((endpoint.feats - expected.feats).abs().max().item())
            path = endpoint_dir / f"step_{step:02d}.pt"
            _save_endpoint(
                path,
                {
                    "step": step,
                    "t": t,
                    "t_next": float(payload["t_next"]),
                    "coords": endpoint.coords.detach().cpu(),
                    "endpoint_normalized_feats": endpoint.feats.detach().cpu(),
                    "formula": "x0=(1-sigma_min)*x_t-(sigma_min+(1-sigma_min)*t)*v",
                    "formula_max_abs_error": error,
                },
            )
            endpoint_records.append(
                {
                    "step": step,
                    "t": t,
                    "t_next": float(payload["t_next"]),
                    "path": str(path),
                    "formula_max_abs_error": error,
                }
            )
            print(f"[texture endpoint] step={step:02d} t={t:.8f} saved={path}")

        common = {
            "steps": 12,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.7,
            "rescale_t": 5.0,
        }
        shape_params = {
            "steps": 12,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.5,
            "rescale_t": 3.0,
        }
        texture_params = {
            "steps": 12,
            "guidance_strength": 1.0,
            "guidance_rescale": 0.0,
            "rescale_t": 3.0,
            "endpoint_callback": save_texture_endpoint,
            "return_model_history": False,
        }
        mesh_list, (shape_slat, final_tex_slat, resolution) = pipeline.run(
            image,
            camera_params=camera,
            seed=args.seed,
            sparse_structure_sampler_params=common,
            shape_slat_sampler_params=shape_params,
            tex_slat_sampler_params=texture_params,
            preprocess_image=False,
            return_latent=True,
            pipeline_type="1024_cascade",
            max_num_tokens=args.max_num_tokens,
        )
        if resolution != 1024 or len(endpoint_records) != 12:
            raise RuntimeError(
                f"expected resolution=1024 and 12 endpoints, got {resolution}, {len(endpoint_records)}"
            )

        # Geometry is constant across texture endpoints, so decode it once.
        meshes, subs = pipeline.decode_shape_slat(shape_slat, resolution)
        tex_mean = torch.tensor(pipeline.tex_slat_normalization["mean"], device=device)[None]
        tex_std = torch.tensor(pipeline.tex_slat_normalization["std"], device=device)[None]
        extrinsics, intrinsics, _ = _make_camera_views(
            camera["camera_angle_x"], camera["distance"], args.angles
        )
        near = max(0.01, float(camera["distance"]) - 2.0)
        far = float(camera["distance"]) + 10.0
        from render_pixal3d_raw_ovoxel import load_envmap

        envmap = load_envmap(args.envmap, device=device)
        renderer = PbrMeshRenderer(
            rendering_options={
                "resolution": args.render_resolution,
                "near": near,
                "far": far,
                "ssaa": args.ssaa,
                "peel_layers": args.peel_layers,
                "face_chunk_size": args.face_chunk_size,
            },
            device=f"cuda:{args.cuda_device}",
        )
        sheet_paths: dict[int, list[Path]] = {angle: [] for angle in args.angles}
        for record in endpoint_records:
            saved = torch.load(record["path"], map_location="cpu", weights_only=False)
            if not torch.equal(saved["coords"], final_tex_slat.coords.cpu()):
                raise RuntimeError(f"endpoint coordinate mismatch at step {record['step']}")
            normalized = final_tex_slat.replace(
                feats=saved["endpoint_normalized_feats"].to(device)
            )
            raw_endpoint = normalized * tex_std + tex_mean
            tex_voxels = pipeline.decode_tex_slat(raw_endpoint, subs)
            decoded = MeshWithVoxel(
                meshes[0].vertices,
                meshes[0].faces,
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1 / resolution,
                coords=tex_voxels[0].coords[:, 1:],
                attrs=tex_voxels[0].feats,
                voxel_shape=torch.Size([*tex_voxels[0].shape, *tex_voxels[0].spatial_shape]),
                layout=pipeline.pbr_attr_layout,
            )
            step_dir = output_dir / "renders" / f"step_{record['step']:02d}_t_{record['t']:.6f}"
            for angle in args.angles:
                torch.cuda.manual_seed_all(args.seed + 100_000 + angle)
                result = renderer.render(
                    decoded,
                    extrinsics[angle],
                    intrinsics,
                    envmap=envmap,
                    use_envmap_bg=False,
                )
                yaw_dir = step_dir / f"yaw{angle:03d}"
                _save_render(result, yaw_dir)
                sheet_paths[angle].append(yaw_dir / "shaded.png")
                del result
            del normalized, raw_endpoint, tex_voxels, decoded
            torch.cuda.empty_cache()

        for angle, paths in sheet_paths.items():
            _endpoint_sheet(paths, output_dir / f"texture_endpoints_yaw{angle:03d}.png")
        summary = {
            "pipeline_type": "1024_cascade",
            "seed": args.seed,
            "resolution": resolution,
            "camera": {key: float(value) for key, value in camera.items()},
            "endpoint_formula": "x0=(1-sigma_min)*x_t-(sigma_min+(1-sigma_min)*t)*v",
            "endpoint_count": len(endpoint_records),
            "endpoints": endpoint_records,
            "angles": args.angles,
            "render_resolution": args.render_resolution,
            "final_native_mesh_vertices": int(mesh_list[0].vertices.shape[0]),
            "final_native_mesh_faces": int(mesh_list[0].faces.shape[0]),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[done] {output_dir}")
        return 0
    finally:
        tmp_image.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
