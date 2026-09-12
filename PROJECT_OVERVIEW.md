# Pixal3D 最小生成路径

目录只保留原生 baseline 和 C128/2048 geometry-only 分块路径。历史实验、局部 crop、纹理/PBR sweep、渲染对比和运行产物不属于当前入口。

## 1. 原生 baseline

| 文件 | 用途 |
| --- | --- |
| `inference.py` | 官方单图推理入口：加载模型、前景预处理、MoGe 相机、原生 1024 cascade 和标准 mesh 解码。 |

## 2. C128 / 2048 几何与分块 Cascade

| 文件 | 用途 |
| --- | --- |
| `pixal3d_baseline1024_c128_8xc64_geometry.py` | 运行原生 SS/C32 → Shape512/C32 → Shape1024/C64；将 C64 decoder support 上采样并量化为 global C128，拆为 8 个互不重叠的 local C64，进行一次 Shape1024 geometry flow，最后一次性解码 2048 mesh。 |
| `pixal3d_cascade512_1024_tiled2048_crop_condition.py` | 上述路径的最小几何 helper：support 量化、8 块稀疏 batch 打包、完整 1024 图像条件路由、Shape flow 和 global row 回填；不包含 texture、crop、PBR 或渲染实验。 |

流程：

```text
输入图像
  -> canonical 预处理 + MoGe 相机
  -> SS/C32 -> Shape512/C32 -> Shape1024/C64
  -> decoder support upsample -> global C128
  -> 8 个 disjoint C64 Shape1024 flow
  -> global geometry decode -> 2048 mesh
```

运行示例：

```bash
CUDA_VISIBLE_DEVICES=4 python pixal3d_baseline1024_c128_8xc64_geometry.py \
  --image assets/choose/0_img.png \
  --camera /path/to/global_camera.json \
  --output-dir outputs/baseline1024_c128_8xc64_geometry_cuda4
```

## 3. baseline 与分块 flow 对照运行

先运行原生 baseline，同时保存 MoGe 相机；再使用同一张图像、相机和 seed
运行 C128/8×C64 geometry flow：

```bash
mkdir -p outputs/baseline_vs_tiled_0

CUDA_VISIBLE_DEVICES=4 python inference.py \
  --image assets/choose/0_img.png \
  --output outputs/baseline_vs_tiled_0/baseline_1024.glb \
  --camera-output outputs/baseline_vs_tiled_0/global_camera.json \
  --low_vram --resolution 1024 --seed 42

CUDA_VISIBLE_DEVICES=4 python pixal3d_baseline1024_c128_8xc64_geometry.py \
  --image assets/choose/0_img.png \
  --camera outputs/baseline_vs_tiled_0/global_camera.json \
  --output-dir outputs/baseline_vs_tiled_0/tiled_2048 \
  --cuda-device 4 --seed 42 --shape-seed 43 --shape-steps 12
```

第一条输出原生 1024 baseline GLB；第二条输出
`tiled_2048/final/geometry_mesh.glb` 和 `geometry_mesh.pt`。第二条会从图像重新
跑 SS/C32、Shape512/C32、原生 Shape1024/C64，再继续 C128 分块 flow；它不会从
第一条生成的 GLB 反推 latent。`--resume` 只复用第二条输出目录中的中间 checkpoint。

## 4. 目录边界

- `pixal3d/`：模型、pipeline、sampler、稀疏算子、representation 和 trainer 等底层实现。
- `configs/`：生成配置。
- `assets/`：输入图片和模型运行所需静态资源。
- `data_toolkit/`：仓库基础工具，未作为实验入口删除。
- `outputs/`：运行时由脚本按需创建，不提交、不保留历史产物。
