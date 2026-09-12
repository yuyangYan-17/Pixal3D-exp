#!/usr/bin/env python3
"""纯几何 C64 baseline → global C128 support → 8 个 C64 分块 flow。

前 3 个阶段沿用 Pixal3D 原生 1024 cascade：

1. 从 canonical 图像生成 Sparse Structure C32；
2. 运行 Shape512/C32 flow；
3. 用 shape decoder 的 support upsample 得到 C64，再运行 Shape1024/C64 flow。

随后不直接解码原生 C64，而是把 C64 support 通过 decoder 上采样到 C1024
候选坐标，再按 baseline 的 ``round((x + 0.5) / 1024 * 127)`` 规则量化为
global C128。global C128 被严格拆成 8 个互不重叠的 local C64 cube；完整的
canonical 1024 图像只投影一次，条件特征按照 global row id 路由到各个 cube，
再作为一个真正的稀疏 batch 运行 Shape1024 flow。最后把 8 块结果按原始 row
顺序还原为 global C128，并一次性解码为 2048 geometry mesh。

本入口不运行 texture flow、texture decoder、PBR 渲染或局部 crop 实验。
"""
from __future__ import annotations

import argparse
import json
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
from PIL import Image

import pixal3d_cascade512_1024_tiled2048_crop_condition as tiled
from pixal3d.modules.sparse import SparseTensor


FORMAT = "pixal3d_baseline1024_c128_8xc64_geometry_v1"
GRID = 128
CUBE = 64
DECODE_RESOLUTION = 2048


def save_sparse(path: Path, value: SparseTensor, normalized: bool) -> None:
    """保存 SparseTensor 的坐标和特征，供 ``--resume`` 复用。"""
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
    """从 CPU checkpoint 读取 SparseTensor，并把特征移动到目标设备。"""
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
    """只对完整 1024 图像做一次投影，再按 row id 路由到 8 个 cube。

    ``global`` 是每个 batch 共用的全局图像 token，``proj`` 与 C128 support
    的行一一对应。这里不重新裁剪图像，也不改变局部坐标；局部化只发生在
    稀疏 batch 的坐标字段中。
    """
    cache = out / "conditions" / "shape_global_c128_full_image.pt"
    if cache.is_file():
        # cache 中的坐标必须与当前 support 完全一致，避免错误复用条件。
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if not torch.equal(payload["coords"].int(), coords.cpu().int()):
            raise RuntimeError("cached C128 condition coordinates do not match support")
        glob, proj = payload["global"], payload["proj"]
        reused = True
    else:
        if image.size != (1024, 1024):
            raise RuntimeError(f"shape condition must be 1024x1024, got {image.size}")
        # 一次性对完整 canonical image_1024 投影所有 global C128 token。
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
        # 每个 cube 只取属于自己的 global row；global token 保持同一个来源。
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
    """把原生 C64 support 上采样并按 baseline 规则量化成 global C128。

    decoder 输出的是 C1024 坐标。先加半个 voxel 中心偏移，再除以 1024，
    映射到 ``[0, 127]``，最后 round 并去重。
    """
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


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    """执行完整的 C64→C128→8×C64→2048 geometry cascade。"""
    from inference import init_pipeline

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.cuda_device):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r}, expected {args.cuda_device}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    # 设置 CUDA_VISIBLE_DEVICES 后，进程内可见卡会重新编号为 cuda:0。
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # low_vram 模式按阶段把模型搬到 GPU，避免同时常驻所有大模型。
    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=True)
    # 这里的 canonical 预处理包含前景分割、bbox padding、正方形居中和多尺度图像。
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

    # ------------------------- 原生 1024 baseline -------------------------
    baseline_cache = out / "baseline" / "shape_c64_denormalized.pt"
    if baseline_cache.is_file() and args.resume:
        print("[baseline] reuse cached native C64 shape SLat", flush=True)
        shape_c64 = load_sparse(baseline_cache, device)
    else:
        print("[baseline 1/3] sparse structure -> C32", flush=True)
        # SS 条件只负责确定 C32 的稀疏 support。
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
        # Shape512 使用 image_512 条件，在原生 C32 support 上生成 shape SLat。
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
        # decoder support upsample 只改变稀疏坐标，不是对 latent 特征做插值。
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

    # ------------------------- 构造 global C128 -------------------------
    support_cache = out / "support" / "coords_c128.pt"
    if support_cache.is_file() and args.resume:
        coords_c128 = torch.load(support_cache, map_location="cpu", weights_only=False)["coords"].int()
    else:
        print("[support] decoder.upsample(..., 4), requested round quantization -> C128", flush=True)
        coords_c128 = c128_support(pipeline, shape_c64).cpu()
        tiled.atomic_save(support_cache, {"format": FORMAT, "coords": coords_c128})
    del shape_c64
    tiled.empty_cuda()

    # 只切 support，不切图像；条件仍来自完整 canonical 1024 图像，
    # 后续由 global row id 精确路由到对应的 local cube。
    records = tiled.build_records(coords_c128)
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
    # ----------------------- 8×C64 Shape1024 flow -----------------------
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
    # flow 输出是归一化 latent，解码前恢复训练时的 mean/std。
    mean, std = tiled.normalization_tensors(
        pipeline.shape_slat_normalization, torch.device("cpu")
    )
    shape_raw = shape_norm * std + mean
    tiled.atomic_save(
        out / "shape" / "final_state_denormalized.pt",
        {"format": FORMAT, "coords": coords_c128, "features": shape_raw},
    )
    tiled.empty_cuda()

    # -------------------------- 全局一次性解码 --------------------------
    print("[decode] assembled global C128 shape -> geometry-only decode at 2048", flush=True)
    shape_st = SparseTensor(shape_raw.to(device), coords_c128.to(device))
    meshes, _ = pipeline.decode_shape_slat(shape_st, DECODE_RESOLUTION)
    if len(meshes) != 1:
        raise RuntimeError(f"shape decoder returned B={len(meshes)}, expected 1")
    mesh = meshes[0]
    tiled.atomic_save(out / "final" / "geometry_mesh.pt", {"format": FORMAT, "mesh": mesh.cpu()})
    # 同时保存原生 mesh checkpoint 和便于查看的 GLB；GLB 导出失败不影响 PT 结果。
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
    }
    tiled.atomic_json(out / "summary.json", summary)
    print(json.dumps(tiled._jsonable(summary), indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    """命令行参数；图像和相机文件显式传入，避免依赖历史 outputs。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path(tiled.DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/baseline1024_c128_8xc64_geometry_cuda4"),
    )
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shape-seed", type=int, default=43)
    parser.add_argument("--shape-steps", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
