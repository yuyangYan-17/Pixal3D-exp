#!/usr/bin/env python3
"""Separate local-support distribution from crop/camera conditioning.

Two C32 supports are run with the *same* local independent-subimage
canonical images and camera:

* ``mapped``: C128 points projected into the selected 4096 crop, then mapped
  into the observed interior C32 range;
* ``independent``: the native C32 support sampled by the independent crop
  baseline.

Both variants use local C32/C64 coordinates for image projection, exactly as
the native independent baseline does.  This avoids mixing a global C128
projection row with a reparameterized local coordinate, which would make the
image token and the Flow/RoPE position refer to different 3-D points.
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


ROOT = Path(__file__).resolve().parent
INDEPENDENT_ROOT = ROOT / (
    "outputs/independent_subimage_baseline_compare_cuda4/"
    "back_headtop_inner4096_baseline1024"
)
MAPPED_ROOT = ROOT / "outputs/c32_crop_support_equivalence_cuda4"
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


def empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def record(cube_id: int, xyz: torch.Tensor, grid: int) -> dict[str, Any]:
    xyz = xyz.int().cpu()
    if not len(xyz) or bool(((xyz < 0) | (xyz >= grid)).any()):
        raise ValueError(f"C{grid} support is empty or out of range")
    rows = torch.arange(len(xyz), dtype=torch.long)
    coords = torch.cat((torch.zeros((len(xyz), 1), dtype=torch.int32), xyz), dim=1)
    return {
        "cube_id": int(cube_id),
        "start": (0, 0, 0),
        "global_row_ids": rows,
        "local_xyz": xyz,
        "local_coords": coords,
        "projection_coords": coords.clone(),
        "owned_row_ids": rows,
        "tokens": int(len(xyz)),
    }


def support_stats(xyz: torch.Tensor, grid: int) -> dict[str, Any]:
    xyz = xyz[:, 1:].int() if xyz.ndim == 2 and xyz.shape[1] == 4 else xyz.int()
    lo, hi = xyz.amin(0), xyz.amax(0)
    d = torch.minimum(xyz, (grid - 1) - xyz).amin(1)
    return {
        "tokens": int(len(xyz)),
        "min": lo.tolist(),
        "max": hi.tolist(),
        "span": (hi - lo + 1).tolist(),
        "bbox_fill": float(len(xyz) / max(1, int(torch.prod(hi - lo + 1)))),
        "boundary_tokens": int((d == 0).sum()),
        "boundary_fraction": float((d == 0).float().mean()),
    }


@torch.no_grad()
def local_condition(
    pipeline: Any,
    image: Image.Image,
    camera: Mapping[str, Any],
    rec: Mapping[str, Any],
    stage: str,
) -> dict[str, Any]:
    if stage == "shape512":
        model = pipeline.image_cond_model_shape_512
        grid = 32
    elif stage == "shape1024":
        model = pipeline.image_cond_model_shape_1024
        grid = 64
    else:
        raise ValueError(stage)
    cond = pipeline.get_proj_cond_shape(
        model,
        [image],
        rec["local_coords"].to(pipeline.device),
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera.get("mesh_scale", 1.0)),
        grid_resolution_override=grid,
    )
    global_feature = cond["cond"]["global"].detach().cpu().contiguous()
    projected_feature = cond["cond"]["proj"].feats.detach().cpu().contiguous()
    if projected_feature.shape[0] != len(rec["local_coords"]):
        raise RuntimeError(f"{stage}: condition rows do not match local support")
    result = {
        "cubes": {
            int(rec["cube_id"]): {
                "global_row_ids": rec["global_row_ids"].clone(),
                "global": global_feature,
                "proj": projected_feature,
            }
        }
    }
    del cond, global_feature, projected_feature
    empty_cuda()
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/c32_support_condition_ablation_cuda4")
    p.add_argument("--model-path", type=Path, default=Path("/home/nvme04/yyyan/download/model/Pixal3D"))
    p.add_argument("--cuda-device", type=int, default=4)
    p.add_argument("--seed-c32", type=int, default=72101)
    p.add_argument("--seed-c64", type=int, default=72102)
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
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    independent_image_512 = Image.open(INDEPENDENT_ROOT / "inputs/canonical_512.png").convert("RGB")
    independent_image_1024 = Image.open(INDEPENDENT_ROOT / "inputs/canonical_1024.png").convert("RGB")
    independent_camera = read_json(INDEPENDENT_ROOT / "camera.json")
    independent_coords = load(INDEPENDENT_ROOT / "support/coords_c32.pt")["coords"].int()
    independent_xyz = independent_coords[:, 1:]
    mapped_payload = load(MAPPED_ROOT / "support/shape512_input.pt")
    mapped_xyz = mapped_payload["local_coords"][:, 1:].int()
    mapped_config = read_json(MAPPED_ROOT / "config.json")
    affine = mapped_config["affine_source_c128_to_model_c32"]
    affine_a = torch.tensor(affine["a"], dtype=torch.float32)
    affine_b = torch.tensor(affine["b"], dtype=torch.float32)
    variants = {"mapped": mapped_xyz, "independent": independent_xyz}
    atomic_json(
        output / "config.json",
        {
            "format": "pixal3d_c32_support_condition_ablation_cuda4_v1",
            "status": "running",
            "condition": "exact independent baseline canonical_512/canonical_1024 and camera; local coords projected at C32/C64",
            "variants": {k: support_stats(v, 32) for k, v in variants.items()},
            "independent_root": INDEPENDENT_ROOT,
            "mapped_root": MAPPED_ROOT,
            "affine_source_c128_to_model_c32": {"a": affine_a, "b": affine_b},
        },
    )
    print(
        f"[ablation] mapped C32={len(mapped_xyz)} independent C32={len(independent_xyz)} "
        "condition=exact independent canonical+camera",
        flush=True,
    )
    print("[model] loading shape/image pipeline", flush=True)
    pipeline = ab.cascade.init_shape_pipeline(args.model_path.resolve(), device)
    pipeline.shape_slat_sampler_params["steps"] = STEPS
    summaries: dict[str, Any] = {}
    for name, xyz in variants.items():
        variant_dir = output / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        rec32 = record(0, xyz, 32)
        print(f"\n[{name}] Shape512 C32 Flow tokens={len(xyz)}", flush=True)
        atomic_save(variant_dir / "support/shape512_input.pt", {"local_coords": rec32["local_coords"], "stats": support_stats(xyz, 32)})
        cond32 = local_condition(pipeline, independent_image_512, independent_camera, rec32, "shape512")
        state32 = local.run_complete_local_flow(
            pipeline, [rec32], cond32, "shape_slat_flow_model_512", device,
            args.seed_c32, "shape512", variant_dir, args.max_flow_tokens, 1,
        )
        local64 = ab.decode_stage1_support(pipeline, rec32, state32, "round", device)
        rec64 = record(0, local64, 64)
        atomic_save(variant_dir / "support/shape1024_input.pt", {"local_coords": rec64["local_coords"], "stats": support_stats(local64, 64)})
        print(f"[{name}] decoded local C64={len(local64)}; Shape1024 Flow", flush=True)
        cond64 = local_condition(pipeline, independent_image_1024, independent_camera, rec64, "shape1024")
        state64 = local.run_complete_local_flow(
            pipeline, [rec64], cond64, "shape_slat_flow_model_1024", device,
            args.seed_c64, "shape1024", variant_dir, args.max_flow_tokens, 1,
        )
        mesh_local = ab.decode_mesh(pipeline, rec64, state64, device)
        atomic_save(variant_dir / "final/geometry_mesh.pt", {"mesh": mesh_local})
        try:
            import trimesh
            trimesh.Trimesh(
                vertices=mesh_local.vertices.numpy(),
                faces=mesh_local.faces.numpy(),
                process=False,
            ).export(variant_dir / "final/geometry_mesh.glb")
        except Exception as exc:
            atomic_json(variant_dir / "final/glb_export_error.json", {"error": repr(exc)})
        angles = tuple(int(v) % 360 for v in args.angles.split(",") if v.strip())
        render = local.render_normals(
            mesh_local.to(device), independent_camera,
            variant_dir / "final", args.render_resolution, angles,
            args.render_chunk_size,
        )
        mesh_global = None
        if name == "mapped":
            mesh_global = __import__("run_c32_crop_support_equivalence_cuda4", fromlist=["map_mesh_to_global"]).map_mesh_to_global(
                mesh_local, affine_a, affine_b
            )
            atomic_save(variant_dir / "final/geometry_mesh_global.pt", {"mesh": mesh_global})
            render_global = local.render_normals(
                mesh_global.to(device),
                read_json(ROOT / "outputs/baseline1024_c128_8xc64_geometry_cuda4/camera.json"),
                variant_dir / "final_global", args.render_resolution, angles,
                args.render_chunk_size,
            )
        else:
            render_global = None
        summaries[name] = {
            "support_c32": support_stats(xyz, 32),
            "support_c64": support_stats(local64, 64),
            "vertices": int(len(mesh_local.vertices)),
            "faces": int(len(mesh_local.faces)),
            "render_independent_frame": render,
            "render_global": render_global,
        }
        atomic_json(variant_dir / "summary.json", summaries[name])
        del cond32, state32, cond64, state64, mesh_local, local64
        empty_cuda()
    summary = {
        "format": "pixal3d_c32_support_condition_ablation_cuda4_v1",
        "status": "complete",
        "condition": "exact independent baseline canonical_512/canonical_1024 and camera; local coords projected at C32/C64",
        "variants": summaries,
        "seconds": time.perf_counter() - started,
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "config.json", {**read_json(output / "config.json"), "status": "complete"})
    print(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
