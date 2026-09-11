#!/usr/bin/env python3
"""Continue one saved C32-halo run through full-halo Shape1024 Flow.

The C32 halo run compresses a 64-cell C128 neighborhood into a 32^3 model
frame.  Its decoder C64 therefore represents one C128 cell per coordinate,
so the full halo must remain in the Shape1024 64^3 frame.  This script keeps
that full frame through Shape1024 Flow and only crops the decoded mesh at the
end.  It avoids rerunning the already completed C32 Flow.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

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


ROOT = Path(__file__).resolve().parent
DEFAULT_STAGE1_ROOT = ROOT / "outputs/c32_halo_block_cuda4"
DEFAULT_BLOCK_ROOT = ROOT / "outputs/c128_context32_local_cascade_4096_crop_geometry_cuda4"
DEFAULT_CANONICAL = ROOT / (
    "outputs/c64_to_c128_two_stage_block_flow_geometry_cuda4/inputs/canonical_4096.png"
)
DEFAULT_CAMERA = ROOT / "outputs/baseline1024_c128_8xc64_geometry_cuda4/camera.json"
GLOBAL_GRID = 128
STAGE2_GRID = 64
STEPS = 12


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def make_halo_c64_record(
    halo_c64: torch.Tensor,
    halo_start: Sequence[int],
    target_start: Sequence[int],
) -> dict[str, Any]:
    n = int(len(halo_c64))
    halo_start_t = torch.tensor(halo_start, dtype=torch.int32)
    global_c256 = halo_start_t[None] * 2 + halo_c64.int() * 2
    local_coords = torch.cat(
        (torch.zeros((n, 1), dtype=torch.int32), halo_c64.int()), dim=1
    )
    return {
        "cube_id": -3000,
        "start": tuple(int(v) for v in halo_start),
        "target_start": tuple(int(v) for v in target_start),
        "global_row_ids": torch.arange(n, dtype=torch.long),
        "local_xyz": halo_c64.int(),
        "local_coords": local_coords,
        "projection_coords": torch.cat(
            (torch.zeros((n, 1), dtype=torch.int32), global_c256), dim=1
        ),
        "owned_row_ids": torch.arange(n, dtype=torch.long),
        "tokens": n,
    }


def support_stats(coords: torch.Tensor, grid: int) -> dict[str, Any]:
    xyz = coords[:, 1:].int() if coords.ndim == 2 and coords.shape[1] == 4 else coords.int()
    if not len(xyz):
        return {"tokens": 0}
    lo = xyz.amin(0)
    hi = xyz.amax(0)
    distance = torch.minimum(xyz, (grid - 1) - xyz).amin(1)
    boundary = distance == 0
    return {
        "tokens": int(len(xyz)),
        "min": lo.tolist(),
        "max": hi.tolist(),
        "span": (hi - lo + 1).tolist(),
        "bbox_fill": float(len(xyz) / max(1, int(torch.prod(hi - lo + 1)))),
        "boundary_tokens": int(boundary.sum()),
        "boundary_fraction": float(boundary.float().mean()),
    }


def extract_fixed_shape1024_condition(
    pipeline: Any,
    canonical: Image.Image,
    camera: Mapping[str, Any],
    record: Mapping[str, Any],
    crop_info: Mapping[str, Any],
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use the target's 1024 region for all halo rows."""
    image_model = pipeline.image_cond_model_shape_1024
    projection_coords = record["projection_coords"].int().cpu()
    uv4096, depth, finite = local.project_condition_points(
        image_model, projection_coords, GLOBAL_GRID * 2, camera
    )
    box = tuple(int(v) for v in crop_info["crop_box_4096"])
    region = canonical.crop(box).convert("RGB")
    if region.size != (1024, 1024):
        raise RuntimeError(f"invalid Shape1024 target region: {region.size}")
    image_dir = output / "inputs/shape1024_region_1024_from_canonical4096"
    image_dir.mkdir(parents=True, exist_ok=True)
    region.save(image_dir / "target.png")
    condition = pipeline.get_proj_cond_shape(
        image_model,
        [region],
        projection_coords.to(pipeline.device),
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera.get("mesh_scale", 1.0)),
        grid_resolution_override=GLOBAL_GRID * 2,
        projection_crop_box=crop_info["projection_crop_box"],
        preserve_image_resolution=True,
    )["cond"]
    global_feature = condition["global"].detach().cpu().contiguous()
    projected_feature = condition["proj"].feats.detach().cpu().contiguous()
    if global_feature.shape[0] != 1 or projected_feature.shape[0] != len(projection_coords):
        raise RuntimeError("Shape1024 halo condition row mismatch")
    x0, y0, x1, y1 = [float(v) for v in crop_info["crop_box_4096"]]
    inside = (
        finite
        & (uv4096[:, 0] >= x0)
        & (uv4096[:, 0] < x1)
        & (uv4096[:, 1] >= y0)
        & (uv4096[:, 1] < y1)
    )
    payload = {
        "cubes": {
            int(record["cube_id"]): {
                "global_row_ids": record["global_row_ids"].clone(),
                "global": global_feature,
                "proj": projected_feature,
                "projection_coords": projection_coords,
                "crop": dict(crop_info),
                "source": "canonical_4096 target 1024 region for full C64 halo",
            }
        },
        "source": "target Shape1024 crop applied to full halo",
    }
    coverage = {
        "finite_projection_tokens": int(finite.sum()),
        "target_crop_covered_tokens": int(inside.sum()),
        "target_crop_coverage_fraction": float(inside[finite].float().mean()) if int(finite.sum()) else 0.0,
        "projected_uv_bbox_4096": [
            *uv4096[finite].amin(0).tolist(), *uv4096[finite].amax(0).tolist()
        ] if int(finite.sum()) else None,
    }
    del condition, region, global_feature, projected_feature
    empty_cuda()
    return payload, coverage


def map_halo_mesh(mesh: Any, halo_start: Sequence[int]) -> Any:
    vertices = mesh.vertices.float().cpu()
    start = torch.tensor(halo_start, dtype=torch.float32)
    center = -0.5 + (start + 32.0) / GLOBAL_GRID
    scale = 64.0 / GLOBAL_GRID
    return type(mesh)(center[None] + scale * vertices, mesh.faces.int().cpu())


def crop_mesh_to_target(mesh_global: Any, target_start: Sequence[int]) -> tuple[Any, Any]:
    """Keep faces fully inside the target block and provide target-local q."""
    vertices = mesh_global.vertices.float().cpu()
    start = torch.tensor(target_start, dtype=torch.float32)
    center = -0.5 + (start + 16.0) / GLOBAL_GRID
    scale = 32.0 / GLOBAL_GRID
    lower = center - scale / 2.0
    upper = center + scale / 2.0
    vertex_keep = ((vertices >= lower) & (vertices <= upper)).all(1)
    faces = mesh_global.faces.int().cpu()
    face_keep = vertex_keep[faces].all(1)
    kept_faces = faces[face_keep]
    if len(kept_faces):
        used, inverse = torch.unique(kept_faces, sorted=True, return_inverse=True)
        local_vertices = vertices.index_select(0, used)
        local_faces = inverse.reshape(-1, 3).int()
    else:
        local_vertices = vertices.new_empty((0, 3))
        local_faces = torch.empty((0, 3), dtype=torch.int32)
    global_mesh = type(mesh_global)(local_vertices, local_faces)
    local_mesh = type(mesh_global)((local_vertices - center[None]) / scale, local_faces)
    return global_mesh, local_mesh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block-id", type=int, default=57)
    p.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    p.add_argument("--block-root", type=Path, default=DEFAULT_BLOCK_ROOT)
    p.add_argument("--canonical-4096", type=Path, default=DEFAULT_CANONICAL)
    p.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--model-path", type=Path, default=Path("/home/nvme04/yyyan/download/model/Pixal3D"))
    p.add_argument("--cuda-device", type=int, default=4)
    p.add_argument("--seed-c64", type=int, default=72102)
    p.add_argument("--max-flow-tokens", type=int, default=30000)
    p.add_argument("--render-resolution", type=int, default=1024)
    p.add_argument("--render-chunk-size", type=int, default=200000)
    p.add_argument("--angles", default="0,60,120,180,240,300")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.cuda_device):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r}, expected physical CUDA {args.cuda_device}"
        )
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    started = time.perf_counter()

    stage1_root = args.stage1_root.resolve() / f"cube_{args.block_id:02d}"
    if args.output_dir is None:
        output = stage1_root.parent / f"cube_{args.block_id:02d}_stage2_full_halo"
    else:
        output = args.output_dir.resolve() / f"cube_{args.block_id:02d}"
    output.mkdir(parents=True, exist_ok=True)
    stage1_summary = read_json(stage1_root / "summary.json")
    halo = stage1_summary["halo"]
    target_start = tuple(int(v) for v in stage1_summary["target_start_c128"])
    halo_start = tuple(int(v) for v in halo["halo_start_c128"])
    halo_c64 = load(stage1_root / "support/shape1024_input.pt")["halo_local_c64"].int()
    state32_path = stage1_root / "flow/shape512_halo/final_state_normalized.pt"
    if not state32_path.is_file():
        raise FileNotFoundError(state32_path)
    block_root = args.block_root.resolve()
    stage2_crop = load(
        block_root / "conditions/shape1024" / f"cube_{args.block_id:02d}.pt"
    ).get("crop")
    if not stage2_crop:
        raise RuntimeError("target Shape1024 crop metadata is missing")
    camera_payload = read_json(args.camera)
    camera = dict(camera_payload.get("camera", camera_payload))
    camera.setdefault("mesh_scale", 1.0)
    canonical = Image.open(args.canonical_4096).convert("RGB")

    record64 = make_halo_c64_record(halo_c64, halo_start, target_start)
    atomic_json(
        output / "config.json",
        {
            "format": "pixal3d_c32_halo_stage2_full_cuda4_v1",
            "status": "running",
            "block_id": int(args.block_id),
            "target_start_c128": list(target_start),
            "halo_start_c128": list(halo_start),
            "route": "saved halo C32 Flow -> full halo C64 Shape1024 Flow -> mesh -> target crop",
            "stage1_source": str(stage1_root),
            "stage2_image_condition": "target block 1024 region from canonical_4096",
            "halo_c64": support_stats(record64["local_coords"], STAGE2_GRID),
            "stage2_crop": stage2_crop,
        },
    )
    print(
        f"[stage2 halo] cube={args.block_id} halo_start={halo_start} "
        f"C64={len(halo_c64)} target={target_start}",
        flush=True,
    )
    print("[model] loading shape/image pipeline", flush=True)
    pipeline = ab.cascade.init_shape_pipeline(args.model_path, device)
    pipeline.shape_slat_sampler_params["steps"] = STEPS
    condition64, coverage = extract_fixed_shape1024_condition(
        pipeline, canonical, camera, record64, stage2_crop, output
    )
    atomic_save(
        output / "support/shape1024_halo_input.pt",
        {
            "local_coords": record64["local_coords"],
            "projection_coords": record64["projection_coords"],
            "source_halo_c64": halo_c64,
        },
    )
    atomic_json(output / "support/projection_coverage.json", coverage)
    state64 = local.run_complete_local_flow(
        pipeline,
        [record64],
        condition64,
        "shape_slat_flow_model_1024",
        device,
        args.seed_c64,
        "shape1024_full_halo",
        output,
        args.max_flow_tokens,
        1,
    )
    mesh_local_halo = ab.decode_mesh(pipeline, record64, state64, device)
    mesh_global_halo = map_halo_mesh(mesh_local_halo, halo_start)
    mesh_global_target, mesh_local_target = crop_mesh_to_target(
        mesh_global_halo, target_start
    )
    atomic_save(
        output / "final/geometry_mesh.pt",
        {
            "format": "pixal3d_c32_halo_stage2_full_cuda4_v1",
            "mesh_halo_global": mesh_global_halo,
            "mesh_target_global": mesh_global_target,
            "mesh_target_local": mesh_local_target,
            "target_start_c128": list(target_start),
            "halo_start_c128": list(halo_start),
        },
    )
    glb_paths: dict[str, str | None] = {}
    try:
        import trimesh

        for name, mesh in {
            "halo": mesh_global_halo,
            "target": mesh_global_target,
        }.items():
            path = output / "final" / f"geometry_mesh_{name}.glb"
            tri = trimesh.Trimesh(
                vertices=mesh.vertices.numpy(),
                faces=mesh.faces.numpy(),
                process=False,
            )
            tri.export(path)
            glb_paths[name] = str(path.resolve())
    except Exception as exc:
        atomic_json(output / "final/glb_export_error.json", {"error": repr(exc)})
        glb_paths = {"halo": None, "target": None}

    angles = tuple(int(v) % 360 for v in args.angles.split(",") if v.strip())
    print("[render] full halo in global camera", flush=True)
    render_halo = local.render_normals(
        mesh_global_halo.to(device),
        camera,
        output / "final_halo_global",
        args.render_resolution,
        angles,
        args.render_chunk_size,
    )
    print("[render] cropped target in global camera", flush=True)
    render_target = local.render_normals(
        mesh_global_target.to(device),
        camera,
        output / "final_target_global",
        args.render_resolution,
        angles,
        args.render_chunk_size,
    )
    independent_camera_path = ROOT / (
        "outputs/independent_subimage_baseline_compare_cuda4/"
        "back_headtop_inner4096_baseline1024/camera.json"
    )
    independent_camera = read_json(independent_camera_path)
    print("[render] cropped target in independent frame", flush=True)
    render_local = local.render_normals(
        mesh_local_target.to(device),
        independent_camera,
        output / "final_target_independent_frame",
        args.render_resolution,
        angles,
        args.render_chunk_size,
    )
    summary = {
        "format": "pixal3d_c32_halo_stage2_full_cuda4_v1",
        "status": "complete",
        "block_id": int(args.block_id),
        "target_start_c128": list(target_start),
        "halo_start_c128": list(halo_start),
        "source_halo_c64": support_stats(halo_c64, STAGE2_GRID),
        "stage2_projection_coverage": coverage,
        "halo_mesh_vertices": int(len(mesh_global_halo.vertices)),
        "halo_mesh_faces": int(len(mesh_global_halo.faces)),
        "target_mesh_vertices": int(len(mesh_global_target.vertices)),
        "target_mesh_faces": int(len(mesh_global_target.faces)),
        "seconds": time.perf_counter() - started,
        "mesh_pt": str((output / "final/geometry_mesh.pt").resolve()),
        "mesh_glb": glb_paths,
        "render_halo_global": render_halo,
        "render_target_global": render_target,
        "render_target_independent_frame": render_local,
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "config.json", {**read_json(output / "config.json"), "status": "complete"})
    print(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
