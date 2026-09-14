# Pixal3D 最小生成路径

目录只保留原生 baseline 和 C128/2048 geometry-only 分块路径。历史实验、局部 crop、纹理/PBR sweep、渲染对比和运行产物不属于当前入口。

## 1. 原生 baseline

| 文件 | 用途 |
| --- | --- |
| `inference.py` | 官方单图推理入口：加载模型、前景预处理、MoGe 相机、原生 1024 cascade 和标准 mesh 解码。 |

## 2. C128 / 2048 几何与分块 Cascade

| 文件 | 用途 |
| --- | --- |
| `pixal3d_baseline1024_c128_8xc64_geometry.py` | 纯入口：只保留 `run()` 的执行顺序和 CLI 参数，不放坐标、flow、解码或渲染实现。 |
| `pixal3d_baseline1024_c256_64xc32_geometry.py` | 新的两级纯几何 cascade 入口：baseline C64→Shape1024→decoder upsample(1)→global C128→64×C32→global C256→Shape1024→4096 mesh。 |
| `pixal3d_cascade512_1024_tiled2048_crop_condition.py` | 纯实现：初始化、C64→8×C32 分块、local→global 坐标映射、support 量化、条件路由、Shape flow、解码、mesh/normal 保存；不包含 CLI、不作为独立程序运行，也不包含 texture、UV、crop 或 PBR。 |

代码阅读边界：先看入口文件即可得到完整实验顺序；需要检查某一步的具体算法时，再到 helper 文件查看对应函数。两者之间只通过明确的 `geometry_ops.*` 函数调用连接，不保留旧实验别名或兼容入口。

流程：

```text
输入图像
  -> canonical 预处理 + MoGe 相机
  -> SS/C32 -> Shape512/C32 -> native C64 support
  -> baseline Shape1024/C64 -> decoder upsample(1) -> global C128
  -> 64 个 local C32 -> local Shape512 -> local C64
  -> 64 个 local Shape1024 -> global C256 latent
  -> global geometry decode -> 4096 mesh
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
运行 native C64 → 8×local C32 → global C128 geometry flow：

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
`tiled_2048/final/geometry_mesh.glb`、`geometry_mesh.pt` 和
`geometry_normal.png`。第二条会从图像重新跑 SS/C32、Shape512/C32、native C64
support，再跑 8 个 local C32→C64→Shape1024；它不会从
第一条生成的 GLB 反推 latent。`--resume` 只复用第二条输出目录中的中间 checkpoint。

## 4. 目录边界

- `pixal3d/`：模型、pipeline、sampler、稀疏算子、representation 和 trainer 等底层实现。
- `configs/`：生成配置。
- `assets/`：输入图片和模型运行所需静态资源。
- `data_toolkit/`：仓库基础工具，未作为实验入口删除。
- `outputs/`：运行时由脚本按需创建，不提交、不保留历史产物。
