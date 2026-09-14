#!/usr/bin/env python3
"""C32 分块 cascade 的纯函数实现。

本文件只定义主程序调用的具体函数：native C64 support → 8 个 local C32 →
global C128 → 64 个 local C32 → global C256 的坐标映射、条件路由、flow
batch、geometry 解码、normal 渲染和 checkpoint 保存。本文件不负责解析命令行、
也不作为独立程序运行；不包含 texture、UV、PBR 或图像 crop。

主程序只应该调用这里列出的具体函数，不应再从本文件导入旧实验别名或兼容
入口；所有函数都只服务于当前这一条几何生成路径。

一个 record 同时保存两套坐标：

* ``local_coords``：送进 flow 的局部坐标，范围是 ``[0, 31]`` 或
  ``[0, 63]``；
* ``global_coords``：同一行在完整 C64/C128 网格中的坐标，只用于相机投影和
  最终 row-id 回填。

这两套坐标不能混用。local 坐标给投影会把每个子块误认为完整物体，global
坐标给 local flow 又会破坏模型的局部窗口坐标约定。
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    str(Path(__file__).with_name("autotune_cache.json")),
)

# normal 渲染函数只在最终 mesh 完成后使用本地 nvdiffrast。
_NVDIFFRAST_ROOT = Path("/home/nvme04/yyyan/nvdiffrast")
if (_NVDIFFRAST_ROOT / "nvdiffrast").is_dir():
    sys.path.insert(0, str(_NVDIFFRAST_ROOT))

import numpy as np
import torch
from PIL import Image

from pixal3d.modules.sparse import SparseTensor


FORMAT = "pixal3d_local_c32_to_c64_c128_geometry_helpers_v2"
NATIVE_C64_GRID = 64
LOCAL_C32_SIZE = 32
FINAL_C128_GRID = 128
DECODE_RESOLUTION = 2048
DECODE_C256_RESOLUTION = 4096
CUBE_STARTS = (0, 32)
DEFAULT_MODEL_PATH = "/home/nvme04/yyyan/download/model/Pixal3D"


@dataclass
class RunState:
    """一次实验运行所需的共享状态；只由 helper 创建和消费。"""

    pipeline: Any
    canonical: Mapping[str, Image.Image]
    camera: Mapping[str, float]
    device: torch.device
    out: Path
    started: float

def jsonable(value: Any) -> Any:
    """把 Tensor、NumPy 类型和 Path 转成 JSON 可写的普通对象。"""
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


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """先写临时文件，再原子替换，避免中断时留下半个 manifest。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_save(path: Path, payload: Any) -> None:
    """以临时文件原子保存 torch checkpoint。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def empty_cuda() -> None:
    """回收 Python 引用并清空 CUDA allocator 缓存。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_sparse(path: Path, value: SparseTensor, normalized: bool) -> None:
    """把 SparseTensor 的坐标和特征保存到 CPU checkpoint。"""
    # 只保存当前实验真正需要的张量，不把 pipeline 或模型对象序列化进去。
    atomic_save(
        path,
        {
            "format": FORMAT,
            "coords": value.coords.detach().cpu().int(),
            "features": value.feats.detach().cpu(),
            "normalized": bool(normalized),
        },
    )


def load_payload(path: Path) -> dict[str, Any]:
    """读取本实验格式的 checkpoint，拒绝其它实验留下的缓存。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise RuntimeError(f"refusing checkpoint from another format: {path}")
    return payload


def denormalize_shape(pipeline: Any, features: torch.Tensor) -> torch.Tensor:
    """把 shape flow 的 normalized latent 恢复为 decoder 使用的原始分布。"""
    mean, std = normalization_tensors(
        pipeline.shape_slat_normalization,
        torch.device("cpu"),
    )
    return features.float() * std + mean


def write_cube_manifest(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    global_grid: int,
    local_grid: int,
    stage: str,
) -> None:
    """保存分块起点、token 数和 row 范围，供人工检查布局。"""
    atomic_json(
        path,
        {
            "format": FORMAT,
            "stage": stage,
            "global_grid": global_grid,
            "local_grid": local_grid,
            "cube_count": len(records),
            "stride": local_grid,
            "coverage": "half-open, disjoint cube partition",
            "cubes": [
                {
                    "cube_id": int(record["cube_id"]),
                    "start": tuple(int(v) for v in record["start"]),
                    "tokens": int(record["global_row_ids"].numel()),
                    "global_row_min": (
                        int(record["global_row_ids"].min())
                        if record["global_row_ids"].numel()
                        else None
                    ),
                    "global_row_max": (
                        int(record["global_row_ids"].max())
                        if record["global_row_ids"].numel()
                        else None
                    ),
                }
                for record in records
            ],
        },
    )


def save_canonical_images(canonical: Mapping[str, Image.Image], output_dir: Path) -> None:
    """保存前景预处理后供各阶段使用的 512/1024/4096 三个图像尺度。"""
    for key in ("image_512", "image_1024", "image_4096"):
        if key not in canonical:
            raise KeyError(f"canonical preprocessing did not return {key}")
        canonical[key].save(output_dir / f"{key}.png")


def save_row_id_map(
    path: Path,
    coords_name: str,
    coords: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """保存 global 坐标、local 坐标和每个 token 的 global row id。"""
    atomic_save(
        path,
        {
            "format": FORMAT,
            coords_name: coords,
            "records": [
                {
                    "cube_id": int(record["cube_id"]),
                    "start": tuple(int(v) for v in record["start"]),
                    "global_row_ids": record["global_row_ids"],
                    "global_coords": record["global_coords"],
                    "local_coords": record["local_coords"],
                }
                for record in records
            ],
        },
    )


def save_geometry_mesh(mesh: Any, output_dir: Path) -> tuple[Path, Path]:
    """保存无 UV 的 geometry mesh PT 和 GLB。"""
    import trimesh

    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_pt = output_dir / "geometry_mesh.pt"
    mesh_glb = output_dir / "geometry_mesh.glb"
    # PT 保存 Pixal3D 原生 Mesh；GLB 只写 vertices/faces，不创建 UV 或材质。
    atomic_save(mesh_pt, {"format": FORMAT, "mesh": mesh.cpu()})
    tri = trimesh.Trimesh(
        vertices=mesh.vertices.detach().cpu().numpy(),
        faces=mesh.faces.detach().cpu().numpy(),
        process=False,
    )
    tri.export(mesh_glb)
    return mesh_pt, mesh_glb


def _check_coords(coords: torch.Tensor, resolution: int, name: str) -> torch.Tensor:
    coords = torch.as_tensor(coords).int().cpu()
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"{name} must have shape [N,4], got {tuple(coords.shape)}")
    if bool((coords[:, 0] != 0).any()):
        raise ValueError(f"{name} must contain batch index zero only")
    xyz = coords[:, 1:]
    if bool((xyz < 0).any()) or bool((xyz >= resolution).any()):
        raise ValueError(f"{name} coordinates must lie in [0,{resolution - 1}]")
    return coords


def build_records(coords: torch.Tensor) -> list[dict[str, Any]]:
    """把 global C64 support 严格拆成 8 个 local C32 record。

    ``global_row_ids`` 指向传入 ``coords`` 的行；``global_coords`` 是这些行
    的原坐标，``local_coords`` 则减去 cube 起点后落在 ``[0,31]^3``。半开区间
    ``[start,start+32)`` 使边界 token 只属于一个 cube，并且 8 个 cube 覆盖
    完整 C64 网格。
    """
    coords = _check_coords(coords, NATIVE_C64_GRID, "global C64 coords")
    xyz = coords[:, 1:]
    writes = torch.zeros(coords.shape[0], dtype=torch.int16)
    records: list[dict[str, Any]] = []
    cube_id = 0
    for sx in CUBE_STARTS:
        for sy in CUBE_STARTS:
            for sz in CUBE_STARTS:
                start = torch.tensor((sx, sy, sz), dtype=torch.int32)
                rows = torch.where(
                    ((xyz >= start) & (xyz < start + LOCAL_C32_SIZE)).all(dim=1)
                )[0].long()
                writes.index_add_(0, rows, torch.ones_like(rows, dtype=torch.int16))
                global_part = coords.index_select(0, rows).clone()
                local_part = global_part.clone()
                local_part[:, 1:] -= start
                records.append(
                    {
                        "cube_id": cube_id,
                        "start": tuple(int(v) for v in start.tolist()),
                        "global_row_ids": rows,
                        "global_coords": global_part,
                        "local_coords": local_part,
                    }
                )
                cube_id += 1
    if not torch.all(writes == 1):
        raise RuntimeError("8 local C32 cubes must partition every global C64 token exactly once")
    return records


def build_grid_records(
    coords: torch.Tensor,
    global_grid: int,
    local_size: int,
) -> list[dict[str, Any]]:
    """把任意完整网格按等大的 local cube 切分并保存 row id。

    这个通用版本用于新的 C128→64×C32 层级；旧的 ``build_records`` 保持
    C64→8×C32 的固定实验语义。所有 cube 使用半开区间，local 坐标始终从 0
    开始，flow 只看到 local 坐标，投影只使用 global 坐标。
    """
    global_grid = int(global_grid)
    local_size = int(local_size)
    if global_grid <= 0 or local_size <= 0 or global_grid % local_size:
        raise ValueError("global_grid must be a positive multiple of local_size")
    coords = _check_coords(coords, global_grid, f"global C{global_grid} coords")
    xyz = coords[:, 1:]
    starts = tuple(range(0, global_grid, local_size))
    writes = torch.zeros(coords.shape[0], dtype=torch.int16)
    records: list[dict[str, Any]] = []
    cube_id = 0
    for sx in starts:
        for sy in starts:
            for sz in starts:
                start = torch.tensor((sx, sy, sz), dtype=torch.int32)
                rows = torch.where(
                    ((xyz >= start) & (xyz < start + local_size)).all(dim=1)
                )[0].long()
                writes.index_add_(0, rows, torch.ones_like(rows, dtype=torch.int16))
                global_part = coords.index_select(0, rows).clone()
                local_part = global_part.clone()
                local_part[:, 1:] -= start
                records.append(
                    {
                        "cube_id": cube_id,
                        "start": tuple(int(v) for v in start.tolist()),
                        "global_row_ids": rows,
                        "global_coords": global_part,
                        "local_coords": local_part,
                    }
                )
                cube_id += 1
    if not torch.all(writes == 1):
        raise RuntimeError(
            f"{len(records)} local C{local_size} cubes must partition every "
            f"global C{global_grid} token exactly once"
        )
    return records


@torch.no_grad()
def encode_mesh_latent(state, args, mesh, resolution, cache_path):
    """在固定世界坐标 AABB voxelize，再以 encoder posterior mean 编码并归一化。"""
    import o_voxel
    from pixal3d import models
    if args.resume and cache_path.is_file():
        payload = load_payload(cache_path)
        if payload["resolution"] != resolution:
            raise RuntimeError("encoded latent cache resolution mismatch")
        return payload["coords"], payload["features"]
    print(f"[encode] mesh -> voxelize {resolution} -> C{resolution // 16}", flush=True)
    indices, dual, intersections = o_voxel.convert.mesh_to_flexible_dual_grid(
        mesh.vertices.detach().float().cpu(), mesh.faces.detach().long().cpu(),
        grid_size=resolution, aabb=[[-0.5] * 3, [0.5] * 3], face_weight=1.0,
        boundary_weight=0.2, regularization_weight=1e-2, timing=True)
    if resolution == 4096 and getattr(args, "texture_encoder_path", None) is not None:
        # Preserve the actual pre-flow voxelization for material-field transfer.
        atomic_save(cache_path.with_name("voxel4096.pt"), dict(
            format=FORMAT, coords=indices.cpu(), dual=dual.cpu(), intersections=intersections.cpu(),
            resolution=4096, aabb=[[-0.5] * 3, [0.5] * 3]))
    vertices = SparseTensor(dual * resolution - indices,
                           torch.cat([torch.zeros_like(indices[:, :1]), indices], 1)).to(state.device)
    intersected = vertices.replace(intersections.to(state.device))
    encoder = models.from_pretrained(str(args.encoder_path)).eval().to(state.device)
    try:
        encoded = encoder(vertices, intersected, sample_posterior=False)
        coords = encoded.coords.int().cpu()
        raw = encoded.feats.float().cpu()
        mean, std = normalization_tensors(state.pipeline.shape_slat_normalization, raw.device)
        normalized = (raw - mean) / std
        if not torch.isfinite(normalized).all():
            raise RuntimeError("encoder returned nonfinite latent")
        _check_coords(coords, resolution // 16, "encoded latent")
        atomic_save(cache_path, dict(format=FORMAT, coords=coords, features=normalized,
                                    resolution=resolution, normalized=True,
                                    encoder=str(args.encoder_path), voxel_tokens=len(indices)))
    finally:
        encoder.cpu()
    del encoder, encoded, vertices, intersected
    empty_cuda()
    return coords, normalized


@torch.no_grad()
def prepare_baseline_encoded_s128(state, args):
    cache = state.out / "baseline" / "encoded_s128.pt"
    if args.resume and cache.is_file():
        payload = load_payload(cache)
        return payload["coords"], payload["features"]
    coords = run_native_prefix(state, args)
    latent = run_native_shape1024(state, args, coords)
    meshes, _ = state.pipeline.decode_shape_slat(latent, 1024)
    if len(meshes) != 1:
        raise RuntimeError("expected one native baseline mesh")
    mesh = meshes[0]
    save_geometry_mesh(mesh, state.out / "baseline" / "mesh1024")
    del latent
    empty_cuda()
    return encode_mesh_latent(state, args, mesh, 2048, cache)


@torch.no_grad()
def decode_c128_and_encode_s256(state, args, coords, normalized):
    """C128 latent 解码为 2048 mesh，再 voxel4096 + Encoder 得到 S256。"""
    cache = state.out / "bridge" / "encoded_s256.pt"
    if args.resume and cache.is_file():
        payload = load_payload(cache)
        if payload["resolution"] != 4096:
            raise RuntimeError("S256 bridge cache resolution mismatch")
        return payload["coords"], payload["features"]
    mesh_path = state.out / "bridge" / "mesh2048" / "geometry_mesh.pt"
    if args.resume and mesh_path.is_file():
        mesh = load_payload(mesh_path)["mesh"]
    else:
        raw = denormalize_shape(state.pipeline, normalized)
        slat = SparseTensor(raw.to(state.device), coords.to(state.device))
        print("[bridge] global C128 latent -> decode mesh2048", flush=True)
        meshes, _ = state.pipeline.decode_shape_slat(slat, 2048)
        if len(meshes) != 1:
            raise RuntimeError("expected one bridge mesh")
        mesh = meshes[0].cpu()
        save_geometry_mesh(mesh, mesh_path.parent)
        del slat, raw, meshes
        empty_cuda()
    return encode_mesh_latent(state, args, mesh, 4096, cache)


def save_sweep_contact_sheet(output_dir, mode, times, tile_size=512):
    """按 step 0..11 排列三行四列的最终输入相机法线图，未完成格留空。"""
    from PIL import ImageDraw
    sheet = Image.new("RGB", (4 * tile_size, 3 * (tile_size + 32)))
    draw = ImageDraw.Draw(sheet)
    for step in range(12):
        x, y = step % 4 * tile_size, step // 4 * (tile_size + 32)
        path = output_dir / mode / f"start_{step:02d}" / "final" / "geometry_normal.png"
        if path.is_file():
            with Image.open(path) as image:
                sheet.paste(image.resize((tile_size, tile_size)), (x, y))
        draw.text((x + 8, y + tile_size + 8),
                  f"{mode} start={step} t={times[step]:.5f}", fill="white")
    path = output_dir / f"{mode}_3x4.png"
    sheet.save(path)
    return path


def build_overlap_records(coords, global_grid, local_size, stride):
    """重叠 context；按体素中心距离选唯一 owner，等距取较小 cube_id。"""
    coords = _check_coords(coords, global_grid, "overlap coords")
    if stride <= 0 or stride > local_size or (global_grid - local_size) % stride:
        raise ValueError("invalid overlap stride/grid")
    records = []
    distance = torch.full((len(coords),), float("inf"))
    owner = torch.full((len(coords),), -1, dtype=torch.long)
    starts = range(0, global_grid - local_size + 1, stride)
    for sx in starts:
        for sy in starts:
            for sz in starts:
                start = torch.tensor([sx, sy, sz], dtype=torch.int32)
                rows = torch.where(((coords[:, 1:] >= start) &
                                    (coords[:, 1:] < start + local_size)).all(1))[0]
                local = coords[rows].clone()
                local[:, 1:] -= start
                d = ((local[:, 1:].float() + 0.5 - local_size / 2) ** 2).sum(1)
                better = d < distance[rows]
                cube_id = len(records)
                distance[rows[better]] = d[better]
                owner[rows[better]] = cube_id
                records.append(dict(cube_id=cube_id, start=(sx, sy, sz),
                                    global_row_ids=rows, global_coords=coords[rows],
                                    local_coords=local))
    if (owner < 0).any():
        raise RuntimeError("overlap windows do not cover all points")
    for record in records:
        record["owner_mask"] = owner[record["global_row_ids"]] == record["cube_id"]
    return records, owner


@torch.no_grad()
def run_synchronized_overlap_stage(state, args, coords, global_grid, local_size, stride,
                                   flow_model_key, stage_name, seed, initial_normalized=None,
                                   start_step=0, condition_mode="conditional"):
    """每一步从同一全局 x_t 取 context，收齐 owner 的 v 后统一 Euler 更新。"""
    if condition_mode not in ("conditional", "unconditional"):
        raise ValueError("invalid condition mode")
    if not 0 <= start_step < args.shape_steps:
        raise ValueError("start_step must select one of the nonzero schedule times")
    records, owner = build_overlap_records(coords, global_grid, local_size, stride)
    atomic_save(state.out / stage_name / "layout.pt",
                dict(format=FORMAT, coords=coords, records=records, owner=owner, stride=stride))
    atomic_json(state.out / stage_name / "layout.json", dict(
        global_grid=global_grid, context=local_size, stride=stride, cubes=len(records),
        active_cubes=sum(bool(r["global_row_ids"].numel()) for r in records),
        selection="nearest voxel center to cube center; ties use lowest cube_id"))
    cache = state.out / "shape" / f"{stage_name}_normalized.pt"
    if args.resume and cache.is_file():
        payload = load_payload(cache)
        if not torch.equal(payload["coords"], coords):
            raise RuntimeError("synchronized flow cache coordinate mismatch")
        if payload.get("experiment") != dict(start_step=start_step, condition_mode=condition_mode,
                                              steps=args.shape_steps, seed=seed):
            raise RuntimeError("synchronized flow cache experiment mismatch")
        return payload["features"], records
    conditions = project_local_conditions(
        state.pipeline, state.canonical["image_1024"], state.camera, coords,
        records, global_grid, state.out, stage_name, args.resume)
    model = state.pipeline.models[flow_model_key]
    sampler = state.pipeline.shape_slat_sampler
    params = dict(state.pipeline.shape_slat_sampler_params)
    params.pop("steps", None)
    times = sampler.timestep_schedule(args.shape_steps, params.pop("rescale_t", 1.0))
    noise = torch.randn((len(coords), int(model.in_channels)),
                    generator=torch.Generator().manual_seed(seed))
    if initial_normalized is None:
        if start_step:
            raise ValueError("partial trajectory requires an initial latent")
        x = noise
    else:
        if initial_normalized.shape != noise.shape:
            raise ValueError("initial latent must align with global support and channels")
        t_start = times[start_step]
        x = (1 - t_start) * initial_normalized.float().cpu() + (
            sampler.sigma_min + (1 - sampler.sigma_min) * t_start) * noise
    times = times[start_step:]
    active = [r for r in records if r["global_row_ids"].numel()]
    if state.pipeline.low_vram:
        model.to(state.device)
    started = time.perf_counter()
    try:
        for step, (t, t_next) in enumerate(zip(times, times[1:])):
            velocity = torch.empty_like(x)
            writes = torch.zeros(len(coords), dtype=torch.int16)
            for offset in range(0, len(active), args.flow_batch_size):
                batch = active[offset:offset + args.flow_batch_size]
                packed = pack_sparse(batch, x).to(state.device)
                cond = pack_conditions(batch, conditions, packed.coords, state.device)
                # Direct single model call: no CFG pair, extrapolation, rescale or interval.
                selected_cond = cond["cond" if condition_mode == "conditional" else "neg_cond"]
                timestep = torch.full((len(batch),), 1000 * t, device=state.device)
                pred = model(packed, timestep, selected_cond)
                if not torch.equal(pred.coords, packed.coords):
                    raise RuntimeError("flow changed context coordinates/order")
                # Streaming reduction retains the nearest-center candidate only.
                # x stays frozen until every context has predicted its velocity.
                feats = pred.feats.float().cpu()
                cursor = 0
                for record in batch:
                    rows = record["global_row_ids"]
                    mask = record["owner_mask"]
                    selected = rows[mask]
                    velocity[selected] = feats[cursor:cursor + len(rows)][mask]
                    writes[selected] += 1
                    cursor += len(rows)
                del packed, cond, pred, feats
            if not torch.all(writes == 1) or not torch.isfinite(velocity).all():
                raise RuntimeError("invalid synchronized velocity or owner coverage")
            x = x - (t - t_next) * velocity
            print(f"[{stage_name}/{condition_mode}] synchronized step {start_step + step + 1}/{args.shape_steps}, "
                  f"{len(active)} contexts, {len(coords):,} global points", flush=True)
    finally:
        if state.pipeline.low_vram:
            model.cpu()
    atomic_save(cache, dict(format=FORMAT, coords=coords, features=x, seed=seed,
                           flow_model=flow_model_key,
                           experiment=dict(start_step=start_step, condition_mode=condition_mode,
                                           steps=args.shape_steps, seed=seed)))
    atomic_json(state.out / stage_name / "flow_summary.json", dict(
        seconds=time.perf_counter() - started, steps=args.shape_steps,
        executed_steps=len(times) - 1,
        times=times, start_step=start_step, condition_mode=condition_mode, cfg=False,
        global_points=len(coords), contexts=len(records),
        active_contexts=len(active), update="global x_next = x_t - dt * nearest_center_v"))
    empty_cuda()
    return x, records


@torch.no_grad()
def upsample_global_latent_once(state, coords, normalized, target_grid):
    """完整全局 latent 上采样一次，直接保留 decoder support。"""
    raw = denormalize_shape(state.pipeline, normalized)
    atomic_save(state.out / "shape" / f"global_c{target_grid // 2}_shape_denormalized.pt",
                dict(format=FORMAT, coords=coords, features=raw))
    decoder = state.pipeline.models["shape_slat_decoder"]
    if state.pipeline.low_vram:
        decoder.to(state.device)
        decoder.low_vram = True
    try:
        slat = SparseTensor(raw.to(state.device), coords.to(state.device))
        result = decoder.upsample(slat, upsample_times=1).cpu().int().unique(dim=0)
    finally:
        if state.pipeline.low_vram:
            decoder.cpu()
            decoder.low_vram = False
    _check_coords(result, target_grid, "upsampled global support")
    atomic_save(state.out / "shape" / f"global_c{target_grid}_support.pt", dict(format=FORMAT, coords=result))
    empty_cuda()
    return result


@torch.no_grad()
def render_saved_normal_multiview(state, args):
    """首帧严格使用输入相机，其余七帧保持 0° 仰角绕 Y 轴渲染。"""
    import utils3d
    from PIL import ImageDraw
    from pixal3d.renderers import MeshRenderer
    from pixal3d.utils.render_utils import proj_camera_to_render_params
    mesh = load_payload(state.out / "final" / "geometry_mesh.pt")["mesh"].to(state.device)
    render_geometry_normal(mesh, state.camera, state.out / "final", state.device,
                           args.normal_resolution, args.normal_face_chunk_size)
    renderer = MeshRenderer(dict(resolution=args.normal_resolution, near=0.1, far=100,
                                 ssaa=1, chunk_size=args.normal_face_chunk_size,
                                 antialias=False), device=state.device)
    out = state.out / "multiview_normals"
    out.mkdir(exist_ok=True)
    size = args.normal_resolution
    sheet = Image.new("RGB", (4 * size, 2 * (size + 28)))
    input_extr, intr = proj_camera_to_render_params(
        float(state.camera["camera_angle_x"]), float(state.camera["distance"]))
    views = []
    for i, yaw in enumerate(range(0, 360, 45)):
        pitch_degrees = 0
        angle, pitch = np.deg2rad(yaw), np.deg2rad(pitch_degrees)
        eye = torch.tensor([np.sin(angle) * np.cos(pitch), np.sin(pitch),
                            np.cos(angle) * np.cos(pitch)], device=state.device,
                           dtype=torch.float32) * float(state.camera["distance"])
        extr = utils3d.torch.extrinsics_look_at(
            eye, torch.zeros(3, device=state.device),
            torch.tensor([0., 1., 0.], device=state.device))
        if i == 0:
            extr = input_extr
        rendered = renderer.render(mesh, extr, intr, return_types=["normal", "mask"])
        pixels = (rendered["normal"].clamp(0, 1) * rendered["mask"].reshape(1, size, size))
        img = Image.fromarray((pixels.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
        path = out / f"normal_yaw_{yaw:03d}.png"
        img.save(path)
        x0, y0 = (i % 4) * size, (i // 4) * (size + 28)
        sheet.paste(img, (x0, y0))
        label = "input camera (yaw 0, pitch 0)" if i == 0 else f"yaw {yaw}, pitch {pitch_degrees}"
        ImageDraw.Draw(sheet).text((x0 + 8, y0 + size + 5), label, fill="white")
        views.append(dict(yaw=yaw, pitch=pitch_degrees, input_camera=i == 0,
                          image=str(path), extrinsics=extr.cpu().tolist()))
    sheet.save(out / "normal_contact_sheet.png")
    atomic_json(out / "views.json", dict(views=views, intrinsics=intr.cpu().tolist()))


def build_refined_records(
    parent_records: Sequence[Mapping[str, Any]],
    local_c64_coords: torch.Tensor,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    """把每个 local C32 decoder 输出的 local C64 support 组成 global C128。

    ``local_c64_coords[:,0]`` 是 ``parent_records`` 中 active record 的 batch
    id。每个 local C64 voxel 映射为 ``2 * parent_start + local_xyz``，所以
    8 个 32³ parent cube 恰好对应 8 个不重叠的 64³ C128 区域。
    """
    local_c64_coords = torch.as_tensor(local_c64_coords).int().cpu()
    if local_c64_coords.ndim != 2 or local_c64_coords.shape[1] != 4:
        raise ValueError("local C64 coords must have shape [N,4]")
    active = [r for r in parent_records if r["global_row_ids"].numel()]
    if local_c64_coords.numel() and bool(
        (local_c64_coords[:, 0] < 0).any()
        or (local_c64_coords[:, 0] >= len(active)).any()
    ):
        raise ValueError("local C64 batch ids do not match active parent records")

    records: list[dict[str, Any]] = []
    global_chunks: list[torch.Tensor] = []
    next_row = 0
    for packed_id, parent in enumerate(active):
        rows = torch.where(local_c64_coords[:, 0] == packed_id)[0].long()
        local_part = local_c64_coords.index_select(0, rows).clone()
        local_part[:, 0] = 0
        start_c128 = 2 * torch.tensor(parent["start"], dtype=torch.int32)
        global_part = local_part.clone()
        if global_part.numel():
            global_part[:, 1:] += start_c128
        global_rows = torch.arange(
            next_row,
            next_row + global_part.shape[0],
            dtype=torch.long,
        )
        next_row += global_part.shape[0]
        records.append(
            {
                "cube_id": int(parent["cube_id"]),
                "parent_cube_id": int(parent["cube_id"]),
                "start": tuple(int(v) for v in start_c128.tolist()),
                "global_row_ids": global_rows,
                "global_coords": global_part,
                "local_coords": local_part,
            }
        )
        if global_part.numel():
            global_chunks.append(global_part)

    if not global_chunks:
        raise RuntimeError("all local C32 decoder supports are empty")
    global_coords = torch.cat(global_chunks, dim=0)
    _check_coords(global_coords, FINAL_C128_GRID, "global C128 coords")
    if torch.unique(global_coords, dim=0).shape[0] != global_coords.shape[0]:
        raise RuntimeError("refined local C64 cubes overlap after C128 mapping")
    return records, global_coords


def build_refined_records_grid(
    parent_records: Sequence[Mapping[str, Any]],
    local_c64_coords: torch.Tensor,
    target_grid: int,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    """把任意数量 parent C32 的 local C64 support 映射到目标 global 网格。"""
    local_c64_coords = torch.as_tensor(local_c64_coords).int().cpu()
    if local_c64_coords.ndim != 2 or local_c64_coords.shape[1] != 4:
        raise ValueError("local C64 coords must have shape [N,4]")
    active = [r for r in parent_records if r["global_row_ids"].numel()]
    if local_c64_coords.numel() and bool(
        (local_c64_coords[:, 0] < 0).any()
        or (local_c64_coords[:, 0] >= len(active)).any()
    ):
        raise ValueError("local C64 batch ids do not match active parent records")

    records: list[dict[str, Any]] = []
    global_chunks: list[torch.Tensor] = []
    next_row = 0
    for packed_id, parent in enumerate(active):
        rows = torch.where(local_c64_coords[:, 0] == packed_id)[0].long()
        local_part = local_c64_coords.index_select(0, rows).clone()
        local_part[:, 0] = 0
        start_target = 2 * torch.tensor(parent["start"], dtype=torch.int32)
        global_part = local_part.clone()
        if global_part.numel():
            global_part[:, 1:] += start_target
        global_rows = torch.arange(
            next_row,
            next_row + global_part.shape[0],
            dtype=torch.long,
        )
        next_row += global_part.shape[0]
        records.append(
            {
                "cube_id": int(parent["cube_id"]),
                "parent_cube_id": int(parent["cube_id"]),
                "start": tuple(int(v) for v in start_target.tolist()),
                "global_row_ids": global_rows,
                "global_coords": global_part,
                "local_coords": local_part,
            }
        )
        if global_part.numel():
            global_chunks.append(global_part)

    if not global_chunks:
        raise RuntimeError("all local C32 decoder supports are empty")
    global_coords = torch.cat(global_chunks, dim=0)
    _check_coords(global_coords, int(target_grid), f"global C{target_grid} coords")
    if torch.unique(global_coords, dim=0).shape[0] != global_coords.shape[0]:
        raise RuntimeError("refined local C64 cubes overlap after global mapping")
    return records, global_coords


@torch.no_grad()
def project_local_conditions(
    pipeline: Any,
    image_1024: Image.Image,
    camera: Mapping[str, float],
    physical_coords: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    physical_grid: int,
    out: Path,
    cache_name: str,
    resume: bool,
) -> dict[int, dict[str, torch.Tensor]]:
    """把完整 image_1024 的投影特征按 local cube 分发。

    ``physical_coords`` 永远是完整 C64/C128 网格坐标，用于相机投影；
    ``records[*]['local_coords']`` 只在后面的 local flow 中使用。两者必须
    保持行一一对应，不能直接拿 local 坐标去做全局相机投影。
    """
    physical_coords = torch.as_tensor(physical_coords).int().cpu()
    cache = out / "conditions" / f"{cache_name}.pt"
    reused = False
    if resume and cache.is_file():
        payload = load_payload(cache)
        if int(payload["physical_grid"]) != int(physical_grid):
            raise RuntimeError(f"cached condition grid mismatch: {cache}")
        if not torch.equal(payload["coords"].int(), physical_coords):
            raise RuntimeError(f"cached condition coordinates do not match: {cache}")
        global_features = payload["global"]
        projected_features = payload["proj"]
        reused = True
    else:
        if image_1024.size != (1024, 1024):
            raise RuntimeError(f"shape condition must be 1024x1024, got {image_1024.size}")
        # 这里使用 shape_1024 条件模型，但分别以 C64/C128 physical grid
        # 投影；这样 local C32 和 local C64 共享同一完整图像坐标系。
        cond = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_shape_1024,
            [image_1024],
            physical_coords.to(pipeline.device),
            camera_angle_x=float(camera["camera_angle_x"]),
            distance=float(camera["distance"]),
            mesh_scale=float(camera.get("mesh_scale", 1.0)),
            grid_resolution_override=int(physical_grid),
        )
        global_features = cond["cond"]["global"].detach().cpu().contiguous()
        projected_features = cond["cond"]["proj"].feats.detach().cpu().contiguous()
        atomic_save(
            cache,
            {
                "format": FORMAT,
                "coords": physical_coords,
                "physical_grid": int(physical_grid),
                "global": global_features,
                "proj": projected_features,
                "source": "complete canonical image_1024 projected at global coordinates",
            },
        )
        del cond
        empty_cuda()

    if global_features.shape[0] != 1:
        raise RuntimeError("image condition global token batch must be one")
    if projected_features.shape[0] != physical_coords.shape[0]:
        raise RuntimeError("projected feature rows are not aligned with physical coords")

    conditions: dict[int, dict[str, torch.Tensor]] = {}
    for record in records:
        rows = record["global_row_ids"].long()
        if rows.numel():
            conditions[int(record["cube_id"])] = {
                "global": global_features,
                "proj": projected_features.index_select(0, rows),
            }
    atomic_json(
        out / "conditions" / f"{cache_name}.json",
        {
            "format": FORMAT,
            "cache_reused": reused,
            "cache": str(cache),
            "image": "canonical image_1024 (complete image, no crop)",
            "physical_grid": int(physical_grid),
            "tokens": int(physical_coords.shape[0]),
            "active_cubes": len(conditions),
            "projection_calls": 0 if reused else 1,
        },
    )
    return conditions


def normalization_tensors(
    spec: Mapping[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """读取 shape flow 的训练归一化参数，并整理成可广播形状。"""
    mean = torch.as_tensor(spec["mean"], dtype=torch.float32, device=device)[None]
    std = torch.as_tensor(spec["std"], dtype=torch.float32, device=device)[None]
    return mean, std


@torch.no_grad()
def render_geometry_normal(
    mesh: Any,
    camera: Mapping[str, float],
    output_dir: Path,
    device: torch.device,
    resolution: int,
    face_chunk_size: int,
) -> tuple[Path, Path]:
    """用 nvdiffrast 按面分块渲染输入相机视角的 geometry normal。"""
    from pixal3d.renderers import MeshRenderer
    from pixal3d.utils.render_utils import proj_camera_to_render_params

    # 相机转换函数使用当前 CUDA device；主程序在调用前已经 set_device。
    extrinsics, intrinsics = proj_camera_to_render_params(
        float(camera["camera_angle_x"]),
        float(camera["distance"]),
    )
    mesh = mesh.to(device)
    renderer = MeshRenderer(
        {
            "resolution": int(resolution),
            "near": 0.1,
            "far": 100.0,
            "ssaa": 1,
            "chunk_size": int(face_chunk_size),
            "antialias": False,
        },
        device=device,
    )
    rendered = renderer.render(
        mesh,
        extrinsics,
        intrinsics,
        return_types=["normal", "mask"],
    )
    normal = rendered["normal"].detach().float().clamp(0, 1)
    mask = rendered["mask"].detach().float()
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    # 法向量编码为 RGB；背景置黑，mask 单独保存，便于区分背景与真实法向量。
    normal = normal * mask.unsqueeze(0)
    normal_image = (
        (normal.permute(1, 2, 0).cpu().numpy() * 255.0)
        .round()
        .astype(np.uint8)
    )
    mask_image = (mask.cpu().numpy() * 255.0).round().astype(np.uint8)
    output_dir.mkdir(parents=True, exist_ok=True)
    normal_path = output_dir / "geometry_normal.png"
    mask_path = output_dir / "geometry_normal_mask.png"
    Image.fromarray(normal_image, mode="RGB").save(normal_path)
    Image.fromarray(mask_image, mode="L").save(mask_path)
    del renderer, rendered, mesh
    empty_cuda()
    return normal_path, mask_path


def upsample_c32_to_c64(pipeline: Any, slat: SparseTensor) -> torch.Tensor:
    """把 C32 shape latent 解码成 C64 support，保留 batch 列。

    decoder 首先把 C32 latent support 上采样到 C512 坐标，再严格使用 baseline
    的端点量化规则 ``round((x+0.5)/512*63)`` 得到 C64 坐标。这个函数不负责
    生成 latent 特征，只生成下一阶段需要的稀疏 support。
    """
    decoder = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        decoder.to(pipeline.device)
        decoder.low_vram = True
    candidates = decoder.upsample(slat, upsample_times=4)
    if pipeline.low_vram:
        decoder.cpu()
        decoder.low_vram = False
    if candidates.shape[0] == 0:
        raise RuntimeError("shape decoder upsample produced no support candidates")

    scaled = (candidates[:, 1:].float() + 0.5) / 512.0
    xyz = (scaled * 63).round().int()
    coords = torch.cat((candidates[:, :1].int(), xyz), dim=1).unique(dim=0)
    if bool(((coords[:, 1:] < 0) | (coords[:, 1:] >= 64)).any()):
        raise RuntimeError("C32 decoder support escapes the C64 grid")
    return coords


def pack_sparse(
    active: Sequence[Mapping[str, Any]], features: torch.Tensor
) -> SparseTensor:
    """按 active cube 顺序拼接 feature，并把 batch 列改为 local cube id。"""
    feats: list[torch.Tensor] = []
    coords: list[torch.Tensor] = []
    for batch_id, record in enumerate(active):
        rows = record["global_row_ids"].long()
        local = record["local_coords"].clone()
        local[:, 0] = batch_id
        feats.append(features.index_select(0, rows))
        coords.append(local)
    if not feats:
        raise RuntimeError("all local cubes are empty")
    return SparseTensor(torch.cat(feats, 0), torch.cat(coords, 0))


def pack_conditions(
    active: Sequence[Mapping[str, Any]],
    conditions: Mapping[int, Mapping[str, torch.Tensor]],
    packed_coords: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """把每个 cube 的 global/proj 条件打包成 sampler 的 cond 字典。"""
    global_parts: list[torch.Tensor] = []
    proj_parts: list[torch.Tensor] = []
    for record in active:
        cube_id = int(record["cube_id"])
        if cube_id not in conditions:
            raise KeyError(f"missing projected conditions for cube {cube_id}")
        global_parts.append(conditions[cube_id]["global"])
        proj_parts.append(conditions[cube_id]["proj"])
    global_features = torch.cat(global_parts, dim=0).to(device)
    projected_features = torch.cat(proj_parts, dim=0).to(device)
    if projected_features.shape[0] != packed_coords.shape[0]:
        raise RuntimeError("projected condition rows are not aligned with local support")
    return {
        "cond": {
            "global": global_features,
            "proj": SparseTensor(projected_features, packed_coords),
        },
        "neg_cond": {
            "global": torch.zeros_like(global_features),
            "proj": SparseTensor(torch.zeros_like(projected_features), packed_coords),
        },
    }


def unpack_global(
    sampled: SparseTensor,
    active: Sequence[Mapping[str, Any]],
    global_rows: int,
) -> torch.Tensor:
    """把 local batch 输出按 global row id 一对一写回 CPU feature 表。"""
    output = torch.empty(
        (global_rows, sampled.feats.shape[1]),
        dtype=sampled.feats.dtype,
        device="cpu",
    )
    writes = torch.zeros(global_rows, dtype=torch.int16)
    for batch_id, record in enumerate(active):
        mask = sampled.coords[:, 0] == batch_id
        if not bool(mask.any()):
            raise RuntimeError(f"cube {record['cube_id']} disappeared during flow")
        local_coords = sampled.coords[mask].detach().cpu().clone()
        local_coords[:, 0] = 0
        if not torch.equal(local_coords, record["local_coords"]):
            raise RuntimeError(
                f"cube {record['cube_id']} support changed/order changed during flow"
            )
        rows = record["global_row_ids"].long()
        output.index_copy_(0, rows, sampled.feats[mask].detach().cpu())
        writes.index_add_(0, rows, torch.ones_like(rows, dtype=torch.int16))
    if not torch.all(writes == 1):
        raise RuntimeError("global flow result must have exactly one write per row")
    return output


@torch.no_grad()
def run_shape_flow(
    pipeline: Any,
    flow_model_key: str,
    stage_name: str,
    global_coords: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    conditions: Mapping[int, Mapping[str, torch.Tensor]],
    seed: int,
    output_dir: Path,
    steps: int | None = None,
) -> torch.Tensor:
    """以一个稀疏 batch 运行 geometry-only Shape flow。

    ``stage_name`` 只用于日志和 checkpoint 名称；模型由 ``flow_model_key``
    指定，因此同一个具体实现可分别运行 local C32/Shape512 和 local C64/Shape1024。
    sampler 的 cond/neg_cond 会原样传入，代码不执行 texture，也不做 UV 操作。
    """
    if not stage_name:
        raise ValueError("stage_name must be non-empty")
    global_coords = torch.as_tensor(global_coords).int().cpu()
    active = [record for record in records if record["global_row_ids"].numel()]
    if not active:
        raise RuntimeError(f"{stage_name}: all local cubes are empty")
    if sum(int(r["global_row_ids"].numel()) for r in active) != global_coords.shape[0]:
        raise RuntimeError(f"{stage_name}: row ids do not cover global support")

    model = pipeline.models[flow_model_key]
    # 两个 local shape stage 都从纯噪声开始；本流程不把旧 shape latent
    # 作为 concat condition，也不混入 texture flow 的输入。
    noise_channels = int(model.in_channels)
    noise = torch.randn(
        (global_coords.shape[0], noise_channels),
        generator=torch.Generator(device="cpu").manual_seed(int(seed)),
    )
    packed_noise = pack_sparse(active, noise).to(pipeline.device)
    packed_cond = pack_conditions(active, conditions, packed_noise.coords, pipeline.device)
    params = dict(pipeline.shape_slat_sampler_params)
    if steps is not None:
        params["steps"] = int(steps)
    if pipeline.low_vram:
        model.to(pipeline.device)
    started = time.perf_counter()
    sampled = pipeline.shape_slat_sampler.sample(
        model,
        packed_noise,
        **packed_cond,
        **params,
        verbose=True,
        tqdm_desc=f"Sampling {stage_name} as B={len(active)} local cubes",
    ).samples
    seconds = time.perf_counter() - started
    if pipeline.low_vram:
        model.cpu()

    features = unpack_global(sampled, active, global_coords.shape[0])
    atomic_save(
        output_dir / "shape" / f"{stage_name}_normalized.pt",
        {
            "format": FORMAT,
            "coords": global_coords,
            "features": features,
            "seed": int(seed),
            "flow_model": flow_model_key,
        },
    )
    atomic_json(
        output_dir / "shape" / f"{stage_name}_summary.json",
        {
            "format": FORMAT,
            "stage": stage_name,
            "flow_model": flow_model_key,
            "batch_size": len(active),
            "tokens": int(global_coords.shape[0]),
            "seconds": seconds,
            "sampler_params": params,
            "coordinate_rule": "local flow coordinates; global row-id writeback",
        },
    )
    del packed_noise, packed_cond, sampled, noise
    empty_cuda()
    return features


# ============================================================================
# 主程序调用的高层具体步骤：初始化、prefix、分块 stage、最终解码
# ============================================================================


def initialize_run(
    args: Any,
    path_description: str | None = None,
) -> RunState:
    """初始化一次纯几何实验，并写入 canonical 图像、相机和运行配置。"""
    from inference import init_pipeline

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() != str(args.cuda_device):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r}, expected {args.cuda_device}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    # CUDA_VISIBLE_DEVICES 指定物理卡后，进程内该卡的逻辑编号是 0。
    device = torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}")
    torch.cuda.set_device(device)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    pipeline = init_pipeline(str(args.model_path), device=str(device), low_vram=True)
    with Image.open(args.image) as source:
        source_image = source.copy()
    canonical = pipeline.preprocess_canonical_images(source_image)
    save_canonical_images(canonical, out)

    camera_payload = json.loads(Path(args.camera).read_text(encoding="utf-8"))
    camera = camera_payload.get("camera", camera_payload)
    for key in ("camera_angle_x", "distance"):
        if key not in camera:
            raise KeyError(f"camera JSON must contain {key!r}")
    atomic_json(out / "camera.json", camera)
    if path_description is None:
        path_description = (
            "image_512->SS C32->Shape512->native C64 support->8 local C32 "
            "(image_1024 physical projection)->local Shape512->local C64 "
            "->local Shape1024->global C128->decode2048"
        )
    atomic_json(
        out / "config.json",
        {
            "format": FORMAT,
            "status": "running",
            "path": path_description,
            "texture": "not sampled, not decoded",
            "uv": "not generated",
            "args": vars(args),
        },
    )
    return RunState(
        pipeline=pipeline,
        canonical=canonical,
        camera=camera,
        device=device,
        out=out,
        started=started,
    )


@torch.no_grad()
def run_native_prefix(state: RunState, args: Any) -> torch.Tensor:
    """从 image_512 重新生成 SS C32、Shape512 和 native C64 support。"""
    pipeline = state.pipeline
    prefix_coords_path = state.out / "prefix" / "native_c64_support_coords.pt"
    prefix_shape_path = state.out / "prefix" / "native_shape_c32_denormalized.pt"
    if args.resume and prefix_coords_path.is_file():
        payload = load_payload(prefix_coords_path)
        coords_c64 = payload["coords"].int()
        print(
            f"[prefix] reuse current-run native C64 support ({coords_c64.shape[0]:,} tokens)",
            flush=True,
        )
        return coords_c64

    camera = state.camera
    canonical = state.canonical
    print("[prefix 1/3] image_512 -> Sparse Structure C32", flush=True)
    cond_ss = pipeline.get_proj_cond_ss(
        [canonical["image_512"]],
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera.get("mesh_scale", 1.0)),
    )
    coords_c32 = pipeline.sample_sparse_structure(
        cond_ss,
        LOCAL_C32_SIZE,
        sampler_params={"steps": int(args.ss_steps)},
    )
    del cond_ss
    empty_cuda()

    print("[prefix 2/3] C32 -> Shape512 flow", flush=True)
    cond_shape_c32 = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_512,
        [canonical["image_512"]],
        coords_c32,
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera.get("mesh_scale", 1.0)),
    )
    shape_c32 = pipeline.sample_shape_slat(
        cond_shape_c32,
        pipeline.models["shape_slat_flow_model_512"],
        coords_c32,
        sampler_params={"steps": int(args.shape_steps)},
    )
    save_sparse(prefix_shape_path, shape_c32, normalized=False)
    del cond_shape_c32, coords_c32
    empty_cuda()

    print("[prefix 3/3] decoder upsample -> native global C64 support", flush=True)
    coords_c64 = upsample_c32_to_c64(pipeline, shape_c32).cpu()
    atomic_save(
        prefix_coords_path,
        {
            "format": FORMAT,
            "coords": coords_c64,
            "source": "fresh image_512 SS/Shape512 prefix",
        },
    )
    del shape_c32
    empty_cuda()
    return coords_c64


@torch.no_grad()
def run_native_shape1024(
    state: RunState,
    args: Any,
    coords_c64: torch.Tensor,
) -> SparseTensor:
    """按原生 baseline 的 Shape1024 运行 C64，返回反归一化 C64 latent。"""
    pipeline = state.pipeline
    camera = state.camera
    image_1024 = state.canonical["image_1024"]
    denormalized_path = state.out / "baseline" / "shape_c64_shape1024_denormalized.pt"

    if args.resume and denormalized_path.is_file():
        payload = load_payload(denormalized_path)
        cached_coords = payload["coords"].int()
        if not torch.equal(cached_coords, coords_c64.int()):
            raise RuntimeError("cached baseline Shape1024 C64 coordinates do not match")
        print(
            f"[baseline Shape1024] reuse current-run C64 latent "
            f"({cached_coords.shape[0]:,} tokens)",
            flush=True,
        )
        return SparseTensor(
            payload["features"].to(pipeline.device),
            cached_coords.to(pipeline.device),
        )

    print("[baseline Shape1024] project native C64 points from image_1024", flush=True)
    cond = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [image_1024],
        coords_c64.to(pipeline.device),
        camera_angle_x=float(camera["camera_angle_x"]),
        distance=float(camera["distance"]),
        mesh_scale=float(camera.get("mesh_scale", 1.0)),
        grid_resolution_override=NATIVE_C64_GRID,
    )
    print("[baseline Shape1024] native global C64 -> Shape1024 flow", flush=True)
    shape_c64 = pipeline.sample_shape_slat(
        cond,
        pipeline.models["shape_slat_flow_model_1024"],
        coords_c64.to(pipeline.device),
        sampler_params={"steps": int(args.shape_steps)},
    )
    save_sparse(denormalized_path, shape_c64, normalized=False)
    del cond
    empty_cuda()
    return shape_c64


@torch.no_grad()
def run_native_shape1024_to_c128(
    state: RunState,
    args: Any,
    coords_c64: torch.Tensor,
) -> torch.Tensor:
    """按原生 baseline 的 Shape1024 运行 C64，再用 upsample(1) 得到 C128。"""
    pipeline = state.pipeline
    support_path = state.out / "baseline" / "global_c128_support.pt"
    if args.resume and support_path.is_file():
        payload = load_payload(support_path)
        coords_c128 = payload["coords"].int()
        _check_coords(coords_c128, FINAL_C128_GRID, "baseline global C128 coords")
        print(
            f"[baseline Shape1024] reuse current-run C128 support "
            f"({coords_c128.shape[0]:,} tokens)",
            flush=True,
        )
        return coords_c128

    shape_c64 = run_native_shape1024(state, args, coords_c64)

    print("[baseline Shape1024] decoder.upsample(..., 1) -> global C128", flush=True)
    decoder = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        decoder.to(pipeline.device)
        decoder.low_vram = True
    coords_c128 = decoder.upsample(shape_c64, upsample_times=1).cpu().int().unique(dim=0)
    if pipeline.low_vram:
        decoder.cpu()
        decoder.low_vram = False
    _check_coords(coords_c128, FINAL_C128_GRID, "baseline global C128 coords")
    atomic_save(
        support_path,
        {
            "format": FORMAT,
            "coords": coords_c128,
            "source": "native C64 Shape1024 + decoder.upsample(1)",
        },
    )
    del shape_c64
    empty_cuda()
    return coords_c128


def split_native_c64(state: RunState, coords_c64: torch.Tensor) -> list[dict[str, Any]]:
    """把 native global C64 拆成 8 个 local C32，并保存 row-id 映射。"""
    records_c32 = build_records(coords_c64)
    write_cube_manifest(
        state.out / "local_c32" / "cube_layout.json",
        records_c32,
        global_grid=NATIVE_C64_GRID,
        local_grid=LOCAL_C32_SIZE,
        stage="native_global_c64_split_to_8_local_c32",
    )
    save_row_id_map(
        state.out / "local_c32" / "row_id_map.pt",
        "global_coords_c64",
        coords_c64,
        records_c32,
    )
    return records_c32


@torch.no_grad()
def run_local_c32_stage(
    state: RunState,
    args: Any,
    coords_c64: torch.Tensor,
    records_c32: Sequence[Mapping[str, Any]],
) -> torch.Tensor:
    """运行 local C32 Shape512，并把 decoder support 变为 local C64。"""
    print("[local 1/4] project native-C64 points into complete image_1024", flush=True)
    conditions_c32 = project_local_conditions(
        state.pipeline,
        state.canonical["image_1024"],
        state.camera,
        coords_c64,
        records_c32,
        physical_grid=NATIVE_C64_GRID,
        out=state.out,
        cache_name="local_c32_from_image1024",
        resume=bool(args.resume),
    )

    local_c32_norm_path = state.out / "shape" / "local_c32_shape512_normalized.pt"
    if args.resume and local_c32_norm_path.is_file():
        payload = load_payload(local_c32_norm_path)
        if not torch.equal(payload["coords"].int(), coords_c64.int()):
            raise RuntimeError("cached local C32 Shape512 support mismatch")
        local_c32_norm = payload["features"]
        print("[local 2/4] reuse current-run local Shape512 result", flush=True)
    else:
        local_c32_norm = run_shape_flow(
            state.pipeline,
            flow_model_key="shape_slat_flow_model_512",
            stage_name="local_c32_shape512",
            global_coords=coords_c64,
            records=records_c32,
            conditions=conditions_c32,
            seed=int(args.shape_seed),
            output_dir=state.out,
            steps=int(args.shape_steps),
        )
    del conditions_c32
    empty_cuda()

    local_c32_raw = denormalize_shape(state.pipeline, local_c32_norm)
    active_c32 = [record for record in records_c32 if record["global_row_ids"].numel()]
    local_c32_slat = pack_sparse(active_c32, local_c32_raw).to(state.device)
    print("[local 3/4] local Shape512 decoder upsample -> local C64 support", flush=True)
    local_c64_coords = upsample_c32_to_c64(state.pipeline, local_c32_slat).cpu()
    atomic_save(
        state.out / "local_c64" / "support_from_local_c32.pt",
        {
            "format": FORMAT,
            "coords": local_c64_coords,
            "source": "8 local C32 Shape512 denormalized latents",
        },
    )
    del local_c32_slat, local_c32_raw, local_c32_norm
    empty_cuda()
    return local_c64_coords


@torch.no_grad()
def run_local_c64_stage(
    state: RunState,
    args: Any,
    records_c32: Sequence[Mapping[str, Any]],
    local_c64_coords: torch.Tensor,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    """把 local C64 映射到 global C128，并运行 Shape1024。"""
    records_c64, coords_c128 = build_refined_records(records_c32, local_c64_coords)
    write_cube_manifest(
        state.out / "local_c64" / "cube_layout.json",
        records_c64,
        global_grid=FINAL_C128_GRID,
        local_grid=64,
        stage="8_local_c64_mapped_to_global_c128",
    )
    save_row_id_map(
        state.out / "local_c64" / "row_id_map.pt",
        "global_coords_c128",
        coords_c128,
        records_c64,
    )

    print("[local 4/4] project global C128 points and run local Shape1024", flush=True)
    conditions_c64 = project_local_conditions(
        state.pipeline,
        state.canonical["image_1024"],
        state.camera,
        coords_c128,
        records_c64,
        physical_grid=FINAL_C128_GRID,
        out=state.out,
        cache_name="local_c64_from_image1024",
        resume=bool(args.resume),
    )
    final_norm_path = state.out / "shape" / "local_c64_shape1024_normalized.pt"
    if args.resume and final_norm_path.is_file():
        payload = load_payload(final_norm_path)
        if not torch.equal(payload["coords"].int(), coords_c128.int()):
            raise RuntimeError("cached local C64 Shape1024 support mismatch")
        final_norm = payload["features"]
        print("[local] reuse current-run local Shape1024 result", flush=True)
    else:
        final_norm = run_shape_flow(
            state.pipeline,
            flow_model_key="shape_slat_flow_model_1024",
            stage_name="local_c64_shape1024",
            global_coords=coords_c128,
            records=records_c64,
            conditions=conditions_c64,
            seed=int(args.shape_seed) + 1,
            output_dir=state.out,
            steps=int(args.shape_steps),
        )
    del conditions_c64
    empty_cuda()
    return records_c64, coords_c128, final_norm


@torch.no_grad()
def run_local_c32_to_c64_stage(
    state: RunState,
    args: Any,
    global_coords: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    physical_grid: int,
    stage_name: str,
    condition_name: str,
    support_path: Path,
    seed: int,
) -> torch.Tensor:
    """对任意 global 网格中的 local C32 批量跑 Shape512 并 upsample 到 local C64。"""
    print(f"[{stage_name}] project complete image_1024 conditions", flush=True)
    conditions = project_local_conditions(
        state.pipeline,
        state.canonical["image_1024"],
        state.camera,
        global_coords,
        records,
        physical_grid=int(physical_grid),
        out=state.out,
        cache_name=condition_name,
        resume=bool(args.resume),
    )

    normalized_path = state.out / "shape" / f"{stage_name}_normalized.pt"
    if args.resume and normalized_path.is_file():
        payload = load_payload(normalized_path)
        if not torch.equal(payload["coords"].int(), global_coords.int()):
            raise RuntimeError(f"cached {stage_name} coordinates do not match support")
        normalized = payload["features"]
        print(f"[{stage_name}] reuse current-run Shape512 result", flush=True)
    else:
        normalized = run_shape_flow(
            state.pipeline,
            flow_model_key="shape_slat_flow_model_512",
            stage_name=stage_name,
            global_coords=global_coords,
            records=records,
            conditions=conditions,
            seed=int(seed),
            output_dir=state.out,
            steps=int(args.shape_steps),
        )
    del conditions
    empty_cuda()

    raw = denormalize_shape(state.pipeline, normalized)
    active = [record for record in records if record["global_row_ids"].numel()]
    packed = pack_sparse(active, raw).to(state.device)
    print(f"[{stage_name}] decoder upsample -> local C64 support", flush=True)
    local_c64_coords = upsample_c32_to_c64(state.pipeline, packed).cpu()
    atomic_save(
        state.out / support_path,
        {
            "format": FORMAT,
            "coords": local_c64_coords,
            "source": f"{stage_name} denormalized local C32 latents",
        },
    )
    del packed, raw, normalized
    empty_cuda()
    return local_c64_coords


@torch.no_grad()
def run_local_c64_shape1024_stage(
    state: RunState,
    args: Any,
    global_coords: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    physical_grid: int,
    stage_name: str,
    condition_name: str,
    seed: int,
) -> torch.Tensor:
    """对 local C64 support 批量跑 Shape1024，返回 global 网格行序的 normalized latent。"""
    print(f"[{stage_name}] project complete image_1024 conditions", flush=True)
    conditions = project_local_conditions(
        state.pipeline,
        state.canonical["image_1024"],
        state.camera,
        global_coords,
        records,
        physical_grid=int(physical_grid),
        out=state.out,
        cache_name=condition_name,
        resume=bool(args.resume),
    )
    normalized_path = state.out / "shape" / f"{stage_name}_normalized.pt"
    if args.resume and normalized_path.is_file():
        payload = load_payload(normalized_path)
        if not torch.equal(payload["coords"].int(), global_coords.int()):
            raise RuntimeError(f"cached {stage_name} coordinates do not match support")
        normalized = payload["features"]
        print(f"[{stage_name}] reuse current-run Shape1024 result", flush=True)
    else:
        normalized = run_shape_flow(
            state.pipeline,
            flow_model_key="shape_slat_flow_model_1024",
            stage_name=stage_name,
            global_coords=global_coords,
            records=records,
            conditions=conditions,
            seed=int(seed),
            output_dir=state.out,
            steps=int(args.shape_steps),
        )
    del conditions
    empty_cuda()
    return normalized


@torch.no_grad()
def decode_and_save(
    state: RunState,
    args: Any,
    coords_c128: torch.Tensor,
    records_c32: Sequence[Mapping[str, Any]],
    records_c64: Sequence[Mapping[str, Any]],
    final_norm: torch.Tensor,
) -> dict[str, Any]:
    """恢复 normalization、解码 2048 mesh、保存 mesh 和 normal，并写 summary。"""
    final_raw = denormalize_shape(state.pipeline, final_norm)
    atomic_save(
        state.out / "shape" / "global_c128_shape_denormalized.pt",
        {
            "format": FORMAT,
            "coords": coords_c128,
            "features": final_raw,
            "source": "local C64 Shape1024 row-id concatenation",
        },
    )
    del final_norm
    empty_cuda()

    print("[decode] global C128 -> geometry-only mesh at 2048", flush=True)
    shape_st = SparseTensor(final_raw.to(state.device), coords_c128.to(state.device))
    meshes, _ = state.pipeline.decode_shape_slat(shape_st, DECODE_RESOLUTION)
    if len(meshes) != 1:
        raise RuntimeError(f"shape decoder returned B={len(meshes)}, expected 1")
    mesh = meshes[0]

    final_dir = state.out / "final"
    mesh_pt, mesh_glb = save_geometry_mesh(mesh, final_dir)
    normal_path, normal_mask_path = render_geometry_normal(
        mesh,
        state.camera,
        final_dir,
        state.device,
        resolution=int(args.normal_resolution),
        face_chunk_size=int(args.normal_face_chunk_size),
    )

    native_c64_tokens = sum(int(r["global_row_ids"].numel()) for r in records_c32)
    summary = {
        "format": FORMAT,
        "status": "complete",
        "texture_executed": False,
        "uv_generated": False,
        "native_c64_tokens": native_c64_tokens,
        "local_c32_tokens": native_c64_tokens,
        "global_c128_tokens": int(coords_c128.shape[0]),
        "active_local_c32_cubes": sum(
            bool(r["global_row_ids"].numel()) for r in records_c32
        ),
        "active_local_c64_cubes": sum(
            bool(r["global_row_ids"].numel()) for r in records_c64
        ),
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "seconds": time.perf_counter() - state.started,
        "mesh_pt": str(mesh_pt.resolve()),
        "mesh_glb": str(mesh_glb.resolve()),
        "normal": str(normal_path.resolve()),
        "normal_mask": str(normal_mask_path.resolve()),
    }
    atomic_json(state.out / "summary.json", summary)
    atomic_json(
        state.out / "config.json",
        {
            "format": FORMAT,
            "status": "complete",
            "args": vars(args),
            "path": "native C64 -> 8 local C32 -> 8 local C64 -> global C128 -> mesh2048 + normal",
        },
    )
    print(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), flush=True)
    return summary


@torch.no_grad()
def decode_global_geometry(
    state: RunState,
    args: Any,
    coords_global: torch.Tensor,
    final_normalized: torch.Tensor,
    decode_resolution: int,
    global_grid: int,
    token_counts: Mapping[str, int],
    active_cube_counts: Mapping[str, int],
) -> dict[str, Any]:
    """按指定全局网格和分辨率解码 geometry-only mesh，不运行纹理。"""
    final_raw = denormalize_shape(state.pipeline, final_normalized)
    atomic_save(
        state.out / "shape" / f"global_c{int(global_grid)}_shape_denormalized.pt",
        {
            "format": FORMAT,
            "coords": coords_global,
            "features": final_raw,
            "source": "global shape latent in global coordinate row order",
        },
    )
    del final_normalized
    empty_cuda()

    print(
        f"[decode] global C{int(global_grid)} -> geometry-only mesh at {int(decode_resolution)}",
        flush=True,
    )
    shape_st = SparseTensor(
        final_raw.to(state.device),
        coords_global.to(state.device),
    )
    meshes, _ = state.pipeline.decode_shape_slat(shape_st, int(decode_resolution))
    if len(meshes) != 1:
        raise RuntimeError(f"shape decoder returned B={len(meshes)}, expected 1")
    mesh = meshes[0]
    mesh_pt, mesh_glb = save_geometry_mesh(mesh, state.out / "final")

    summary = {
        "format": FORMAT,
        "status": "complete",
        "texture_executed": False,
        "uv_generated": False,
        "decode_resolution": int(decode_resolution),
        "global_grid": int(global_grid),
        "token_counts": {str(k): int(v) for k, v in token_counts.items()},
        "active_cube_counts": {
            str(k): int(v) for k, v in active_cube_counts.items()
        },
        "vertices": int(mesh.vertices.shape[0]),
        "faces": int(mesh.faces.shape[0]),
        "seconds": time.perf_counter() - state.started,
        "mesh_pt": str(mesh_pt.resolve()),
        "mesh_glb": str(mesh_glb.resolve()),
    }
    atomic_json(state.out / "summary.json", summary)
    atomic_json(
        state.out / "config.json",
        {
            "format": FORMAT,
            "status": "complete",
            "args": vars(args),
            "path": json.loads((state.out / "config.json").read_text())["path"],
        },
    )
    print(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), flush=True)
    return summary
