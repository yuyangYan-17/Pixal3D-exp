#!/usr/bin/env python3
"""Run the mapped crop support with a camera-consistent local frame.

The C32 coordinates are local, but the image crop was cut from the global
4096 canonical image.  This experiment passes a custom camera-to-world matrix
that maps the local C32 cube back to the global C128 crop before projection.
Consequently, a local token and the pixel feature sampled for that token use
the same perspective coordinate system.  It tests the optical-center/frame
hypothesis directly.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    str(Path(__file__).with_name("autotune_cache.json")),
)

import numpy as np
import torch
from PIL import Image

import pixal3d_c128_context32_hr_crop_local_geometry as local
import run_c32_block_distribution_ablation_cuda4 as ab
import run_c32_crop_support_equivalence_cuda4 as eq


ROOT = Path(__file__).resolve().parent
BLOCK_ROOT = ROOT / "outputs/c128_context32_local_cascade_4096_crop_geometry_cuda4"
SUPPORT_ROOT = ROOT / "outputs/c32_crop_support_equivalence_cuda4"
CANONICAL = ROOT / "outputs/c64_to_c128_two_stage_block_flow_geometry_cuda4/inputs/canonical_4096.png"
CAMERA_PATH = ROOT / "outputs/baseline1024_c128_8xc64_geometry_cuda4/camera.json"
STEPS = 12


def load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def record(xyz: torch.Tensor, grid: int) -> dict[str, Any]:
    xyz = xyz.int().cpu()
    if not len(xyz) or bool(((xyz < 0) | (xyz >= grid)).any()):
        raise ValueError(f"invalid C{grid} support")
    rows = torch.arange(len(xyz), dtype=torch.long)
    coords = torch.cat((torch.zeros((len(xyz), 1), dtype=torch.int32), xyz), dim=1)
    return {
        "cube_id": 0,
        "start": (0, 0, 0),
        "global_row_ids": rows,
        "local_xyz": xyz,
        "local_coords": coords,
        "projection_coords": coords.clone(),
        "owned_row_ids": rows,
        "tokens": int(len(xyz)),
    }


def global_camera_transform(camera: Mapping[str, Any]) -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -float(camera["distance"])],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def local_to_global_camera_transform(
    camera: Mapping[str, Any], affine_a: torch.Tensor, affine_b: torch.Tensor
) -> torch.Tensor:
    """Return T_local so inv(T_local) q_local = inv(T_global) L q_local.

    ProjGrid uses endpoint coordinates idx/(R-1)-1/2.  The affine therefore
    uses 31 and 127 here; using cell-center factors (32/128) would leave a
    systematic half-cell projection error.
    """
    # model C32 endpoint coordinate q_l -> source C128 endpoint coordinate q_g
    scale = 31.0 / (127.0 * affine_a.float())
    offset = (
        (15.5 - affine_b.float()) / (127.0 * affine_a.float()) - 0.5
    )
    q_to_global_model = torch.diag(scale)
    q_to_global_model = torch.cat(
        (q_to_global_model, offset[:, None]), dim=1
    )
    q_to_global_model = torch.cat(
        (q_to_global_model, torch.tensor([[0.0, 0.0, 0.0, 1.0]])), dim=0
    )
    # ProjGrid rotates model axes (x,y,z) -> world axes (x,-z,y).
    rotation = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    rotation4 = torch.eye(4, dtype=torch.float32)
    rotation4[:3, :3] = rotation
    local_world_to_global_world = rotation4 @ q_to_global_model @ rotation4.T
    return torch.linalg.inv(local_world_to_global_world) @ global_camera_transform(camera)


def project_with_transform(
    xyz: torch.Tensor,
    grid: int,
    camera: Mapping[str, Any],
    transform: torch.Tensor,
) -> torch.Tensor:
    one = torch.linspace(-1.0, 1.0, grid)
    model = one[xyz.int()] / 2.0
    world = torch.stack((model[:, 0], -model[:, 2], model[:, 1]), dim=1)
    ones = torch.ones((len(world), 1))
    camera_xyz = torch.cat((world, ones), 1) @ torch.linalg.inv(transform).T
    x, y, z = camera_xyz[:, 0], camera_xyz[:, 1], camera_xyz[:, 2]
    focal = 16.0 / np.tan(float(camera["camera_angle_x"]) / 2.0) * 512.0 / 32.0
    uv512 = torch.stack((focal * x / (-z + 1e-8) + 256.0, -focal * y / (-z + 1e-8) + 256.0), 1)
    return (uv512 + 0.5) * 8.0 - 0.5


def condition_payload(
    pipeline: Any,
    image: Image.Image,
    camera: Mapping[str, Any],
    transform: torch.Tensor,
    rec: Mapping[str, Any],
    stage: str,
    crop_box: list[int],
    out: Path,
) -> dict[str, Any]:
    if stage == "shape512":
        model = pipeline.image_cond_model_shape_512
        grid = 32
        input_size = 512
    elif stage == "shape1024":
        model = pipeline.image_cond_model_shape_1024
        grid = 64
        input_size = 1024
    else:
        raise ValueError(stage)
    x0, y0, x1, y1 = crop_box
    region = image.crop((x0, y0, x1, y1)).convert("RGB")
    feature_image = region if input_size == 1024 else region.resize((512, 512), Image.Resampling.LANCZOS)
    image_dir = out / "inputs" / f"{stage}_crop_1024_input_{input_size}"
    image_dir.mkdir(parents=True, exist_ok=True)
    feature_image.save(image_dir / "crop.png")
    crop_norm = [x0 / 4096.0, y0 / 4096.0, x1 / 4096.0, y1 / 4096.0]
    cond = pipeline.get_proj_cond_shape(
        model,
        [feature_image],
        rec["local_coords"].to(pipeline.device),
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera.get("mesh_scale", 1.0)),
        grid_resolution_override=grid,
        projection_crop_box=crop_norm,
        transform_matrix=transform,
        preserve_image_resolution=True,
    )
    global_feature = cond["cond"]["global"].detach().cpu().contiguous()
    projected_feature = cond["cond"]["proj"].feats.detach().cpu().contiguous()
    if projected_feature.shape[0] != len(rec["local_coords"]):
        raise RuntimeError(f"{stage}: condition row mismatch")
    payload = {
        "cubes": {
            0: {
                "global_row_ids": rec["global_row_ids"].clone(),
                "global": global_feature,
                "proj": projected_feature,
            }
        }
    }
    del cond, region, feature_image, global_feature, projected_feature
    empty_cuda()
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/c32_camera_aligned_crop_ablation_cuda4")
    p.add_argument("--model-path", type=Path, default=Path("/home/nvme04/yyyan/download/model/Pixal3D"))
    p.add_argument("--cuda-device", type=int, default=4)
    p.add_argument("--seed-c32", type=int, default=72101)
    p.add_argument("--seed-c64", type=int, default=72102)
    p.add_argument(
        "--support-source",
        choices=("mapped", "independent"),
        default="mapped",
        help="Use the mapped global C128 support or the native independent C32 template.",
    )
    p.add_argument("--max-flow-tokens", type=int, default=30000)
    p.add_argument("--render-resolution", type=int, default=1024)
    p.add_argument("--render-chunk-size", type=int, default=200000)
    p.add_argument("--angles", default="0,60,120,180,240,300")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.cuda_device):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}, expected {args.cuda_device}")
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    started = time.perf_counter()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    global_camera = read_json(CAMERA_PATH)
    crop_box = [int(v) for v in read_json(ROOT / "outputs/independent_subimage_baseline_compare_cuda4/inputs/back_headtop_inner4096_bbox1024.json")["selected_inner_crop_4096"]]
    support = load(SUPPORT_ROOT / "support/shape512_input.pt")
    if args.support_source == "mapped":
        xyz32 = support["local_coords"][:, 1:].int()
    else:
        xyz32 = load(
            ROOT
            / "outputs/independent_subimage_baseline_compare_cuda4/"
            "back_headtop_inner4096_baseline1024/support/coords_c32.pt"
        )["coords"][:, 1:].int()
    cfg = read_json(SUPPORT_ROOT / "config.json")
    affine_a = torch.tensor(cfg["affine_source_c128_to_model_c32"]["a"], dtype=torch.float32)
    affine_b = torch.tensor(cfg["affine_source_c128_to_model_c32"]["b"], dtype=torch.float32)
    transform = local_to_global_camera_transform(global_camera, affine_a, affine_b)
    if args.support_source == "mapped":
        source_projection = support["projection_coords"][:, 1:].int()
        projected_local = project_with_transform(xyz32, 32, global_camera, transform)
        source_projected = eq.project_c128_to_4096(
            torch.cat((torch.zeros((len(source_projection), 1), dtype=torch.int32), source_projection), 1),
            global_camera,
        )[0]
        projection_error = (projected_local - source_projected).norm(dim=1)
    else:
        projection_error = torch.empty(0)
    if len(projection_error):
        print(
            f"[projection] mapped local->global median={projection_error.median():.3f}px "
            f"p95={torch.quantile(projection_error, torch.tensor(.95)):.3f}px "
            f"max={projection_error.max():.3f}px",
            flush=True,
        )
    atomic_json(
        out / "config.json",
        {
            "format": "pixal3d_c32_camera_aligned_crop_ablation_cuda4_v1",
            "status": "running",
            "route": "mapped global C128 crop support -> local C32/C64 Flow with affine camera transform",
            "crop_box_4096": crop_box,
            "support_source": args.support_source,
            "support_c32_tokens": int(len(xyz32)),
            "affine_source_c128_to_model_c32": {"a": affine_a, "b": affine_b},
            "projection_error_pixels": (
                {
                    "median": float(projection_error.median()),
                    "p95": float(torch.quantile(projection_error, torch.tensor(.95))),
                    "max": float(projection_error.max()),
                }
                if len(projection_error)
                else None
            ),
            "global_camera": global_camera,
            "local_to_global_camera_transform": transform,
            "image_condition": "direct canonical_4096 1024 crop resized to 512 for Shape512; native 1024 for Shape1024",
        },
    )
    canonical = Image.open(CANONICAL).convert("RGB")
    rec32 = record(xyz32, 32)
    print("[model] loading shape/image pipeline", flush=True)
    pipeline = ab.cascade.init_shape_pipeline(args.model_path.resolve(), device)
    pipeline.shape_slat_sampler_params["steps"] = STEPS
    cond32 = condition_payload(pipeline, canonical, global_camera, transform, rec32, "shape512", crop_box, out)
    state32 = local.run_complete_local_flow(
        pipeline, [rec32], cond32, "shape_slat_flow_model_512", device,
        args.seed_c32, "shape512", out, args.max_flow_tokens, 1,
    )
    local64 = ab.decode_stage1_support(pipeline, rec32, state32, "round", device)
    rec64 = record(local64, 64)
    print(f"[decode] local C64={len(local64)}", flush=True)
    cond64 = condition_payload(pipeline, canonical, global_camera, transform, rec64, "shape1024", crop_box, out)
    state64 = local.run_complete_local_flow(
        pipeline, [rec64], cond64, "shape_slat_flow_model_1024", device,
        args.seed_c64, "shape1024", out, args.max_flow_tokens, 1,
    )
    mesh_local = ab.decode_mesh(pipeline, rec64, state64, device)
    mesh_global = eq.map_mesh_to_global(mesh_local, affine_a, affine_b)
    atomic_save(out / "final/geometry_mesh.pt", {"mesh": mesh_global, "mesh_local": mesh_local})
    angles = tuple(int(v) % 360 for v in args.angles.split(",") if v.strip())
    render_local = local.render_normals(mesh_local.to(device), global_camera, out / "final_local_frame", args.render_resolution, angles, args.render_chunk_size)
    render_global = local.render_normals(mesh_global.to(device), global_camera, out / "final_global", args.render_resolution, angles, args.render_chunk_size)
    summary = {
        "format": "pixal3d_c32_camera_aligned_crop_ablation_cuda4_v1",
        "status": "complete",
        "support_c32_tokens": int(len(xyz32)),
        "support_c64_tokens": int(len(local64)),
        "vertices": int(len(mesh_global.vertices)),
        "faces": int(len(mesh_global.faces)),
        "projection_error_pixels": (
            {
                "median": float(projection_error.median()),
                "p95": float(torch.quantile(projection_error, torch.tensor(.95))),
                "max": float(projection_error.max()),
            }
            if len(projection_error)
            else None
        ),
        "render_local_frame": render_local,
        "render_global": render_global,
        "seconds": time.perf_counter() - started,
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "config.json", {**read_json(out / "config.json"), "status": "complete"})
    print(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
