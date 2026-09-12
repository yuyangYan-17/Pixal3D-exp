#!/usr/bin/env python3
"""原生 C128 → tiled 2048 几何 cascade 的最小 helper。

公开入口是 ``pixal3d_baseline1024_c128_8xc64_geometry.py``。本文件只保留
几何路径需要的可复用操作：

* C32/C64 support 的 decoder upsample 与坐标量化；
* global C128 support 拆成 8 个互不重叠的 local C64；
* 把 global row 对齐的图像条件打包成稀疏 batch；
* 运行 Shape1024 flow，并把 local 结果写回 global row 顺序。

texture flow、PBR 转换、图像 crop、渲染、局部重建和其他实验分支均不在
保留路径中。
"""
from __future__ import annotations

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

from pixal3d.modules.sparse import SparseTensor


FORMAT = "pixal3d_c128_tiled2048_geometry_helpers_v1"
# global support 是 128³ 网格；每个 local cube 是 64³，stride 同样为 64。
GLOBAL_GRID = 128
CUBE_SIZE = 64
CUBE_STARTS = (0, 64)
DEFAULT_MODEL_PATH = "/home/nvme04/yyyan/download/model/Pixal3D"


def _jsonable(value: Any) -> Any:
    """把 Tensor、NumPy 类型和 Path 转成可写入 JSON 的普通对象。"""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
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
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
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


def build_records(coords: torch.Tensor) -> list[dict[str, Any]]:
    """把一个 global C128 support 严格拆成 8 个 local C64 视图。

    每个 record 保存三种对应关系：cube 起点、该 cube 在 global support 中的
    row id，以及减去 cube 起点后的 local 坐标。``writes`` 检查保证每个 token
    恰好属于一个 cube，不允许遗漏或重叠。
    """
    coords = torch.as_tensor(coords).int().cpu()
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"global coords must have shape [N,4], got {tuple(coords.shape)}")
    if bool((coords[:, 0] != 0).any()):
        raise ValueError("global C128 support must contain batch index zero only")
    xyz = coords[:, 1:]
    if bool((xyz < 0).any()) or bool((xyz >= GLOBAL_GRID).any()):
        raise ValueError("global C128 coordinates must lie in [0,127]")

    writes = torch.zeros(coords.shape[0], dtype=torch.int16)
    records: list[dict[str, Any]] = []
    cube_id = 0
    for sx in CUBE_STARTS:
        for sy in CUBE_STARTS:
            for sz in CUBE_STARTS:
                start = torch.tensor((sx, sy, sz), dtype=torch.int32)
                # 半开区间 [start, start + 64) 保证边界 token 只有唯一归属。
                rows = torch.where(
                    ((xyz >= start) & (xyz < start + CUBE_SIZE)).all(dim=1)
                )[0].long()
                writes.index_add_(0, rows, torch.ones_like(rows, dtype=torch.int16))
                local = coords.index_select(0, rows).clone()
                local[:, 1:] -= start
                records.append(
                    {
                        "cube_id": cube_id,
                        "start": tuple(int(v) for v in start.tolist()),
                        "global_row_ids": rows,
                        "local_coords": local,
                    }
                )
                cube_id += 1
    if not torch.all(writes == 1):
        raise RuntimeError("C64 cubes must partition every global token exactly once")
    return records


def normalization_tensors(
    spec: Mapping[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """读取 shape flow 的训练归一化参数，并整理成可广播形状。"""
    mean = torch.as_tensor(spec["mean"], dtype=torch.float32, device=device)[None]
    std = torch.as_tensor(spec["std"], dtype=torch.float32, device=device)[None]
    return mean, std


def upsample_and_quantize(
    pipeline: Any,
    slat: SparseTensor,
    source_resolution: int,
    target_grid: int,
    baseline_rounding: bool,
) -> torch.Tensor:
    """调用原生 shape decoder support upsampler，再量化为目标网格坐标。

    ``baseline_rounding=True`` 使用训练 baseline 的 ``grid-1`` 端点规则；
    否则使用普通的 ``floor(x * grid)`` 规则。返回值始终是单 batch 的
    ``[batch, x, y, z]`` 整数坐标。
    """
    decoder = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        decoder.to(pipeline.device)
        decoder.low_vram = True
    candidates = decoder.upsample(slat, upsample_times=4)
    if pipeline.low_vram:
        decoder.cpu()
        decoder.low_vram = False

    # decoder 坐标表示 voxel index；加 0.5 后转为 voxel center 的归一化位置。
    scaled = (candidates[:, 1:].float() + 0.5) / float(source_resolution)
    if baseline_rounding:
        xyz = (scaled * (target_grid - 1)).round().int()
    else:
        xyz = (scaled * target_grid).int()
    coords = torch.cat((candidates[:, :1].int(), xyz), dim=1).unique(dim=0)
    if bool(((coords[:, 1:] < 0) | (coords[:, 1:] >= target_grid)).any()):
        raise RuntimeError(f"quantized support escapes C{target_grid}")
    return coords


def pack_sparse(
    active: Sequence[Mapping[str, Any]], features: torch.Tensor
) -> SparseTensor:
    """按 cube 顺序拼接 feature，并把 batch 列改为 local cube id。"""
    feats: list[torch.Tensor] = []
    coords: list[torch.Tensor] = []
    for batch_id, record in enumerate(active):
        rows = record["global_row_ids"].long()
        feats.append(features.index_select(0, rows))
        # 原始 global row 顺序通过 rows 保留，坐标本身改成 [0, 63]³ local view。
        local = record["local_coords"].clone()
        local[:, 0] = batch_id
        coords.append(local)
    if not feats:
        raise RuntimeError("all C64 cubes are empty")
    return SparseTensor(torch.cat(feats, 0), torch.cat(coords, 0))


def pack_conditions(
    active: Sequence[Mapping[str, Any]],
    conditions: Mapping[int, Mapping[str, torch.Tensor]],
    packed_coords: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """把每个 cube 的 global/proj 条件拼成 flow sampler 所需的 cond 字典。

    ``global`` 按 active cube 复制为 batch token；``proj`` 则与拼接后的稀疏
    坐标逐行对齐。``neg_cond`` 是无条件分支使用的全零条件。
    """
    global_features = torch.cat(
        [conditions[int(record["cube_id"])]["global"] for record in active], 0
    ).to(device)
    projected_features = torch.cat(
        [conditions[int(record["cube_id"])]["proj"] for record in active], 0
    ).to(device)
    if projected_features.shape[0] != packed_coords.shape[0]:
        raise RuntimeError("projected condition rows are not aligned with support")
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
    """把 local batch 输出按 global row id 写回一个 global feature 表。"""
    output = torch.empty(
        (global_rows, sampled.feats.shape[1]),
        dtype=sampled.feats.dtype,
        device="cpu",
    )
    writes = torch.zeros(global_rows, dtype=torch.int16)
    for batch_id, record in enumerate(active):
        # sampler 必须保持每个 cube 的 support 和行顺序不变。
        mask = sampled.coords[:, 0] == batch_id
        if not bool(mask.any()):
            raise RuntimeError(f"cube {record['cube_id']} disappeared during flow")
        local_coords = sampled.coords[mask].detach().cpu().clone()
        local_coords[:, 0] = 0
        if not torch.equal(local_coords, record["local_coords"]):
            raise RuntimeError(f"cube {record['cube_id']} changed support/order during flow")
        rows = record["global_row_ids"].long()
        output.index_copy_(0, rows, sampled.feats[mask].detach().cpu())
        writes.index_add_(0, rows, torch.ones_like(rows, dtype=torch.int16))
    if not torch.all(writes == 1):
        raise RuntimeError("global flow result must have exactly one write per row")
    return output


@torch.no_grad()
def run_tiled_flow(
    pipeline: Any,
    stage: str,
    global_coords: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    conditions: Mapping[int, Mapping[str, torch.Tensor]],
    seed: int,
    shape_normalized: torch.Tensor | None,
    output_dir: Path,
) -> torch.Tensor:
    """把 8 个 local C64 作为一个稀疏 batch 运行 geometry-only Shape1024 flow。

    这里的 ``global_coords`` 只用于生成 global noise 和最终回填；模型实际看到
    的坐标是每个 cube 内的 local ``0..63`` 坐标。由于 cubes 不重叠，回填是
    一对一的 ``global_row_ids`` 写入，不需要融合或平均边界速度。
    """
    if stage != "shape":
        raise ValueError("the retained cascade is geometry-only; stage must be 'shape'")
    active = [record for record in records if record["global_row_ids"].numel()]
    model = pipeline.models["shape_slat_flow_model_1024"]
    shape_channels = 0 if shape_normalized is None else int(shape_normalized.shape[1])
    noise_channels = int(model.in_channels) - shape_channels
    # 噪声先在 global row 顺序生成，再通过 pack_sparse 分发到各个 cube，
    # 这样 seed 与 cube 的排列无关，最终仍能确定性回填。
    noise = torch.randn(
        (global_coords.shape[0], noise_channels),
        generator=torch.Generator(device="cpu").manual_seed(int(seed)),
    )
    packed_noise = pack_sparse(active, noise).to(pipeline.device)
    packed_cond = pack_conditions(active, conditions, packed_noise.coords, pipeline.device)
    concat = (
        pack_sparse(active, shape_normalized).to(pipeline.device)
        if shape_normalized is not None
        else None
    )
    params = dict(pipeline.shape_slat_sampler_params)
    if pipeline.low_vram:
        model.to(pipeline.device)
    started = time.perf_counter()
    # sampler 只运行 shape flow；此 helper 不接受 texture stage。
    sampled = pipeline.shape_slat_sampler.sample(
        model,
        packed_noise,
        concat_cond=concat,
        **packed_cond,
        **params,
        verbose=True,
        tqdm_desc=f"Sampling tiled C128 shape as B={len(active)} C64 cubes",
    ).samples
    seconds = time.perf_counter() - started
    if pipeline.low_vram:
        model.cpu()

    # 把 local batch 的输出恢复成 global C128 row 顺序。
    features = unpack_global(sampled, active, global_coords.shape[0])
    atomic_save(
        output_dir / "shape" / "final_state_normalized.pt",
        {"coords": global_coords.cpu(), "features": features, "seed": int(seed)},
    )
    atomic_json(
        output_dir / "shape" / "flow_summary.json",
        {
            "stage": "shape",
            "mode": "one sparse batch of eight non-overlapping C64 cubes",
            "batch_size": len(active),
            "global_grid": GLOBAL_GRID,
            "context": CUBE_SIZE,
            "stride": CUBE_SIZE,
            "tokens": int(global_coords.shape[0]),
            "seconds": seconds,
            "sampler_params": params,
        },
    )
    del packed_noise, packed_cond, concat, sampled, noise
    empty_cuda()
    return features
