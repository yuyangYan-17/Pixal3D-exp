#!/usr/bin/env python3
"""重新生成 baseline 前置并运行 8×local-C32 geometry cascade。

执行路径严格是：

1. ``--image`` → rembg 前景分割、bbox padding、正方形居中，保存
   ``image_512``、``image_1024``、``image_4096``；
2. ``image_512`` → Sparse Structure C32 → Shape512 → decoder support C64；
3. global C64 按 ``(0,32)³`` 拆成 8 个 local C32，并保存 global row id；
4. 每个 local C32 用完整 ``image_1024`` 的投影条件跑 Shape512，decoder 后得到
   local C64，再跑 Shape1024；
5. local C64 用 ``global = 2 * cube_start + local`` 映射到 global C128，按 row
   id 一次性解码为 2048 geometry mesh，并用 nvdiffrast 分块渲染相机对齐 normal。

本文件从输入图像重新生成所有 baseline 前置 latent/support，不读取
``inference.py`` 产生的 latent、mesh 或旧 output。只跑几何，不运行 texture、UV、
PBR 或纹理贴图。

代码边界：本文件只保留命令行参数和 ``run()`` 的执行顺序；坐标拆分、条件
投影、checkpoint、flow batch、mesh 导出和 normal 渲染的具体实现全部在
``pixal3d_cascade512_1024_tiled2048_crop_condition.py`` 中。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pixal3d_cascade512_1024_tiled2048_crop_condition as geometry_ops

def run(args: argparse.Namespace) -> None:
    """只保留实验步骤顺序；每一步的具体实现都在 geometry_ops。"""
    state = geometry_ops.initialize_run(args)
    coords_c64 = geometry_ops.run_native_prefix(state, args)
    records_c32 = geometry_ops.split_native_c64(state, coords_c64)
    local_c64_coords = geometry_ops.run_local_c32_stage(
        state,
        args,
        coords_c64,
        records_c32,
    )
    records_c64, coords_c128, final_norm = geometry_ops.run_local_c64_stage(
        state,
        args,
        records_c32,
        local_c64_coords,
    )
    geometry_ops.decode_and_save(
        state,
        args,
        coords_c128,
        records_c32,
        records_c64,
        final_norm,
    )


def parse_args() -> argparse.Namespace:
    """命令行参数；相机参数显式传入，避免读取历史 output。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path(geometry_ops.DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/baseline_native_c64_local_c32_c128_geometry_cuda4"),
    )
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shape-seed", type=int, default=43)
    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--shape-steps", type=int, default=12)
    parser.add_argument("--normal-resolution", type=int, default=1024)
    parser.add_argument("--normal-face-chunk-size", type=int, default=200_000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
