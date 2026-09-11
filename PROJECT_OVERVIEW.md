# Pixal3D 根目录代码概述

本文档记录仓库根目录中当前保留的可执行脚本及其关系，方便后续查找和复用代码。内容基于 Git 提交 `9647e63` 整理；模型主体、数据集、训练器和渲染器的底层实现位于 `pixal3d/`，本文不逐文件展开。

## 1. 主流程概览

当前代码主要围绕以下几条执行路径展开：

```text
输入图像
  -> 去背景、规范化裁剪、图像金字塔
  -> MoGe 相机参数估计（或手动 FOV）
  -> C32 Sparse Structure
  -> 512 Shape SLat
  -> 1024 Shape SLat
  -> 可选的 C128/C256 高分辨率分块 Shape/Texture Flow
  -> Shape 与 Texture Decoder
  -> MeshWithVoxel / MeshWithVertexPbr
  -> 原生 PBR 渲染、对比和指标计算
```

常用基础模块：

- `inference.py`：模型加载、DINO 投影条件、MoGe 相机估计以及标准推理封装。
- `pixal3d/pipelines/pixal3d_image_to_3d.py`：实际的 Pixal3D cascade pipeline；支持 `1024_cascade`、实验性的 `1536_cascade` 和 `2048_cascade`，并包含后期 texture patch/tile flow。
- `render_pixal3d_raw_ovoxel.py`：保留原生 O-Voxel PBR 表示的加载、渲染、保存和指标工具。

## 2. 标准 1024 基线

| 文件 | 用途与可复用内容 |
| --- | --- |
| `inference.py` | 标准单图到 3D 入口。可复用 `init_pipeline`、图像条件模型构造、MoGe 相机估计和 `run_inference`。模型路径目前是本机绝对路径，迁移环境时需要调整。 |
| `run_baseline1024_raw_ovoxel.py` | 运行一次原生 1024 cascade，并直接把 `MeshWithVoxel` 送入原生 PBR renderer；适合作为回归基线。 |
| `render_pixal3d_raw_ovoxel.py` | 渲染已保存的 `MeshWithVoxel` 或局部烘焙的 vertex-PBR mesh；不经过 GLB、UV atlas 或 Blender。也提供 PSNR/SSIM 和 comparison sheet 工具。 |
| `pixal3d_baseline1024_pbr_mesh_compare.py` | 在同一几何上比较原生 O-Voxel、逐面 PBR 和逐顶点 PBR 三种表示，保持相机和光照一致。 |

## 3. C128 / 2048 几何与分块 Cascade

| 文件 | 用途与可复用内容 |
| --- | --- |
| `pixal3d_baseline1024_c128_8xc64_geometry.py` | 原生 C64 shape support 经 decoder 上采样并量化成 global C128，拆成 8 个互不重叠的 local C64 cube，批量跑 Shape1024 flow，最终全局解码到 2048；不运行 texture。 |
| `pixal3d_cascade512_1024_tiled2048_crop_condition.py` | 完整的 512 -> 1024 -> tiled 2048 路径。每个 C64 cube 使用独立、对齐的图像 crop condition，依次运行 shape 和 texture flow。包含 crop、support 量化、稀疏 batch 打包及全局回填等通用函数。 |
| `crop_c128_head_4096.py` | 将保存的 global C128 support 投影到 4096 图像，提取一个 1024 像素的头部 ROI。 |
| `run_head_crop_baseline1024_geometry.py` | 对头部 crop 运行原生、geometry-only 的 1024 cascade，主要作为局部几何基线。 |

## 4. Global C256 / Local C64 分块 Flow

| 文件 | 用途与可复用内容 |
| --- | --- |
| `pixal3d_global_c256_cube_owner_flow_singleview.py` | 当前 global C256 分块实验的核心。唯一状态是 global sparse row table，C64 cubes 只预测局部速度；支持 nearest-centre hard owner 和重叠区域 Gaussian 融合。大量后续脚本复用其 cube layout、坐标变换、条件构造、flow、checkpoint 和 decode 工具。 |
| `pixal3d_global_c256_encdown_context1024_flow_singleview.py` | 用 C4096 shape encoder 的下采样结果构造 global C256 support，以完整 1024 图像作为条件，再复用 C256 owner-flow 和单次 global decode。 |
| `pixal3d_baseline1024_c128_8xc64_geometry.py` | 可作为 disjoint cube sparse batching 的较小规模参考实现。 |

## 5. Global 4096 同步与 Shared-SLat

| 文件 | 用途与可复用内容 |
| --- | --- |
| `pixal3d_global4096_tile_endpoint_rollout_sync.py` | 以稳定 master ID 管理 global 4096 irregular support。每个 tile 是 master rows 的 local C64 视图，通过 Jacobi barrier、完整 endpoint rollout 和 FP32 endpoint 融合推进全局状态。也是多套 4096 实验的底层工具库。 |
| `pixal3d_global4096_tile_x0_consensus_sync.py` | endpoint-rollout 的替代同步方案：每一步收集 tile 的即时 `pred_x_0`，按 master ID 和二维 Gaussian 权重做 consensus，再通过官方公式完成一次 Euler 更新。 |
| `pixal3d_global4096_singleview_shared_slat_shape_tex_sr.py` | 单视图 shared-SLat 4096 Shape + Texture SR 主入口。shape 与 texture 共用一份 support；texture 可依据可见性在前视图和反面材质观测之间路由逐点投影特征。 |
| `pixal3d_singleview_shared_slat_support.py` | shared-SLat 的辅助模块：tile box、基线三角形可见性、反面材质观测，以及逐行 front/back projection-feature routing。它不会创建第二份 latent state。 |
| `pixal3d_render_global4096_multiview.py` | 使用原生 `PbrMeshRenderer` 对完成的 4096 vertex-PBR mesh 做多视角渲染和 contact sheet。 |

三条同步路径的定位：

```text
endpoint rollout     较早的完整余程 rollout 同步实验
instant x0 consensus 每个 Euler step 直接同步 pred_x_0
shared-SLat          在统一 support 上继续做单视图 shape/texture SR
```

## 6. C64 Attention Topology -> C256 Value 实验

| 文件 | 用途与可复用内容 |
| --- | --- |
| `pixal3d_c64_topology_c256_value_probe.py` | 最小 probe：让 baseline C64 和 tiled C256 同步运行到同一 timestep/block，抓取 RMSNorm、RoPE 后真正参与 attention 的 Q/K，并测试 C64 topology 对 C256 value message 的影响；不做最终解码。 |
| `pixal3d_c64_topology_c256_online_intervention.py` | 在 texture flow 的 step 6、block 20 进行一次在线 intervention，继续完整 trajectory，随后解码、渲染并计算 seam/input-view/multiview 指标。 |

底层 attention hook、coarse/fine correspondence 和 routing 实现在 `pixal3d/experiments/global_attention_routing.py`。

## 7. Texture 与 PBR 专项实验

| 文件 | 用途与可复用内容 |
| --- | --- |
| `pixal3d_baseline1024_uniform_texture_endpoints.py` | 保存 12-step baseline C64 texture flow 各个 uniform timestep 的 `x0` endpoint，并解码成原生 1024 O-Voxel。 |
| `pixal3d_baseline1024_tex_slat_renoise_uncond_back_sweep.py` | 先跑完整 1024/C64 baseline，再将最终 texture SLat 按 12 个起始时间重新加噪，仅用无图像条件的 texture flow 去噪；固定几何并输出背面前景指标、error map 和 3×4 汇总图。 |
| `sweep_baseline1024_tex_unconditional_start_step.py` | 扫描 texture flow 从第几步开始移除图像条件，用于判断条件信息在 trajectory 中的有效区间。 |
| `compare_baseline_tex_image_unconditional.py` | 对 baseline 与 image-unconditional texture 结果计算和排版 PSNR、SSIM 等对比。 |
| `pixal3d_c256_uniform_endpoint_prefix_uncond_sweep.py` | 在 global C256 上使用 baseline endpoint 引导前缀、image-unconditional 后缀，扫描前缀长度。 |
| `pixal3d_c128_baseline_endpoint_renoise_uncond_2048.py` | C2048/C128 endpoint 重加噪实验入口：把 1024 baseline 几何 voxelize 成 C2048 flexible dual-grid，以官方 shape encoder 得到固定 C128 support；查询并编码 12 个 baseline O-Voxel 材质 endpoint，再以 8 个 disjoint C64 context 扫描 n=1…12 的无图条件 texture-flow 后缀，统一解码到 2048，并生成正/背面 1K 渲染、error map 和 PSNR/SSIM。支持逐阶段断点复用。 |
| `pixal3d_c128_baseline_endpoint_renoise_cond_2048.py` | 上述 C2048/C128 sweep 的严格 image-conditional 对照：复用相同 endpoint、fixed shape、C64 layout 和固定噪声，只把剩余 12-n 步换成完整 1024 输入图的 texture condition，并输出 conditional/unconditional 同轴指标曲线。 |
| `render_c256_endpoint_prefix_back_grid.py` | 批量渲染上述 C256 sweep 的背面视图并生成网格。 |
| `pixal3d_texture_pbr_degradation_experiment.py` | 固定局部 shape/support，只研究 texture。将 baseline PBR field 编码为指导 endpoint，再从该 endpoint 加噪并运行原生 texture flow；包含 PBR 低频投影及 SLat/PBR degradation 分析。 |
| `pixal3d_texture_visibility_guided_pbr_flow.py` | training-free、fixed-shape 的可见性引导 texture endpoint 修正：HR `pred_x0` 解码到 PBR、按可见性融合，再编码回 SLat 并转回官方 flow velocity。 |

此外，主 pipeline 内还保留了以下 texture 实验组件：

- 2048 overlap 3D patch flow；
- 4096 高分辨率 image-tile texture flow；
- 4096 image tile 与 C64 3D patch 配对；
- `paired_block_fusion`、`target_context_hard` 和 `membership_velocity_fusion`；
- Haar/wavelet guidance、patch velocity 融合和 flow trace 保存。

## 8. Local C1024 Tile 编解码

| 文件 | 用途与可复用内容 |
| --- | --- |
| `pixal3d_tile_c1024_local_slat_and_local_decode_return_global.py` | 从 baseline 4096 投影区域出发，为每个 1024 image crop 建立独立 local C1024 dual-grid，运行本地 shape/PBR encoder、flow 和 decoder，再精确变换回 global object space。最终 patch 只拼接，不焊点、不重网格。 |

该文件是相机 global/local round-trip、tile layout、投影、local support 构造、PBR 查询以及局部 mesh 回到全局空间等功能的主要复用来源。

## 9. Fresh Support / Head Patch 路径

这组脚本用于判断：直接切 global support 得到的 Cartesian cube 是否偏离模型见过的局部物体 support 分布，以及从局部图像重新生成 fresh support 能否改善细节。

| 文件 | 用途与可复用内容 |
| --- | --- |
| `run_stride1024_fresh_points.py` | 对 4096 图像的 4x4 个互不重叠 1024 crops 独立生成 fresh C64 support；保持 crop 像素不变，并从全图相机推导统一局部相机，不再次使用 MoGe 或前景居中。 |
| `analyze_head_cube_occupancy.py` | 比较 Cartesian global-C256 C64 cut 与正常 local-object support 的占用率、拓扑和投影分布，诊断 OOD 原因。 |
| `run_head_similarity_experiment.py` | 可恢复的头部 similarity 实验。Phase A 比较已有 mesh；Phase B/C 在同一 metric local cube 中分别运行 inherited support 与 fresh support。 |
| `run_gated_fresh_support.py` | 使用 fresh sparse-structure interior，同时在颈部接口保留少量 inherited C32 context，测试局部连接连续性。 |
| `combine_cube_with_fresh_patch.py` | 原型式 mesh 组合：global Cartesian cube 决定输出区域，fresh local decode 提供内部几何。 |
| `partition_fresh_patch_to_global_cubes.py` | 将一个连续 fresh local decode 按 global C256/C64 owner 空间重新分区，并检查三角形一致性。 |
| `render_full_global_fresh_cube.py` | 将一个 Cartesian cube 替换为 fresh local decode 后，渲染完整 global C256 结果。 |

相关分析结论和实验记录保存在：

- `occupancy_report.md`
- `cube_fresh_integration_report.md`

## 10. 渲染与评估辅助脚本

| 文件 | 用途与可复用内容 |
| --- | --- |
| `render_encdown_result_metrics.py` | 在输入视角渲染 encoder-downsample 实验，并标注图像指标。 |
| `render_encdown_six_views.py` | 渲染 encoder-downsample 结果的前、后、左、右、上、下六个视图。 |
| `render_baseline1024_tex_slat_renoise_uncond_back_2k.py` | 复用已保存的 baseline/variant SLat，以真实 2048×2048 分辨率重渲染背面、重算 error map 和前景指标，并生成 2K 子图的 3×4 汇总图。 |
| `pixal3d_c128_final_baseline_renoise_visibility_gated_2048.py` | 固定最终 baseline E12，对每个 n 直接按 t=1−n/12 加同一份噪声，从 n+1 步执行可见比例 >10% 条件分流；C128 shape 固定，2048 decode 与 1024×1024 前后渲染。输出至 `outputs/c128_final_baseline_renoise_visibility_gated_2048_cuda4/`。 |
| `render_c128_final_baseline_renoise_errors.py` | 从上述 1024 渲染生成统一色标的前后 error map、前景指标 CSV 与折线图。 |
| `pixal3d_c128_visibility_gated_endpoint_prefix_2048.py` | 复用已校验的 12 步均匀 baseline/C128 endpoints；从相同噪声逐步执行 endpoint 1…n 引导 Euler 更新，余步按 C64 可见活跃体素比例 >10% 切换图像条件（否则 global/proj 置零），保留 shape。CUDA4 输出 12 组 2048 decode、前后 3×4 渲染及输入前景指标。 |
| `render_c128_baseline_visibility.py` | 使用 C128 support 的直接来源 baseline 1024 mesh，在生成相机下以 4096 triangle-ID 深度缓冲和面中心首交射线计算可见性；C128 体素中心通过最近三角面表面继承标签。输出斜上方原空间/分块展开总览、8 块细节、逐面及逐体素标签和统计到 `outputs/c128_baseline_visibility/`。 |
| `render_c128_c64_cube_wireframe_on_final_pbr.py` | 将 C128 support 的 2×2×2 个 C64 分块边界生成为真实三维圆柱 mesh，并与最终 2048 PBR mesh 共用生成时的固定正面相机（global_camera.json）；禁止旋转或拉远相机适配边框。保持像素焦距扩展画布展示完整空间框，中心裁剪输出 1024×1024 原视野图，同时输出 canonical 参考图/PBR/边框三联对照。用实际 ProjGrid 投影校验渲染相机，并记录内外参和误差。 |
| `render_c256_endpoint_prefix_back_grid.py` | C256 endpoint-prefix sweep 的背面批量渲染器。 |
| `pixal3d_render_global4096_multiview.py` | 4096 native PBR mesh 的多视角渲染器。 |
| `render_pixal3d_raw_ovoxel.py` | 最通用的原生 O-Voxel/vertex-PBR 渲染和指标工具。 |

## 11. 复用时的推荐入口

按任务选择代码时，优先从以下入口开始：

| 目标 | 推荐入口 |
| --- | --- |
| 跑标准单图 1024 结果 | `inference.py` 或 `run_baseline1024_raw_ovoxel.py` |
| 加载一次 pipeline 后自行组合阶段 | `inference.init_pipeline` |
| 原生 O-Voxel 渲染与指标 | `render_pixal3d_raw_ovoxel.py` |
| C128、8 个 disjoint C64 cubes | `pixal3d_baseline1024_c128_8xc64_geometry.py` |
| 1024 endpoint -> C2048/C128 fixed-shape texture sweep | `pixal3d_c128_baseline_endpoint_renoise_uncond_2048.py` |
| 每块独立 crop condition 的 2048 cascade | `pixal3d_cascade512_1024_tiled2048_crop_condition.py` |
| Global C256 + local C64 flow | `pixal3d_global_c256_cube_owner_flow_singleview.py` |
| Stable master-ID 的 4096 tile 同步 | `pixal3d_global4096_tile_endpoint_rollout_sync.py` |
| 每步 `pred_x0` 共识 | `pixal3d_global4096_tile_x0_consensus_sync.py` |
| Shared support 的 4096 shape/texture SR | `pixal3d_global4096_singleview_shared_slat_shape_tex_sr.py` |
| Global/local 相机、投影和局部 mesh 回填 | `pixal3d_tile_c1024_local_slat_and_local_decode_return_global.py` |
| C64/C256 attention routing | `pixal3d/experiments/global_attention_routing.py` |
| Fresh support 与 inherited support 对照 | `run_head_similarity_experiment.py` |

## 12. 目录边界

- `pixal3d/`：当前底层模型、pipeline、sampler、稀疏算子、representation、renderer 和 trainer。
- `configs/`：生成与微调配置。
- `data_toolkit/`：数据下载、metadata、渲染、voxelize 和 latent encode 工具。
- `tests/`：当前主线功能的单元/一致性测试。
- `used/`：旧实验、废弃路线和历史快照；可以参考，但不应默认作为当前入口。
- `outputs/`：实验产物，不是源码。
- `record/`：历史记录和图片，不是运行依赖。
- `assets/`：输入图片、GLB、HDRI 和小型数据集。

## 13. 使用注意事项

- 多数实验默认 CUDA，并且部分脚本带有特定 GPU、显存分配策略或本机模型绝对路径，复用前先检查文件顶部常量和 CLI 默认值。
- C64、C128、C256 通常指 sparse SLat/support 网格分辨率，不等同于最终图像或 mesh 分辨率。
- 高分辨率实验经常区分 global coordinate、tile-local `0..63` coordinate 和 stable master ID；不要直接用 local coordinate 跨 tile 合并。
- `MeshWithVoxel` 是首选的原生 PBR 输出。只有明确需要 patch 拼接或逐顶点材质时，才转成 `MeshWithVertexPbr`。
- 复用实验 helper 时优先调用公开的纯函数；大型入口脚本中存在对输出目录、缓存 schema 和中间 checkpoint 的严格假设。
- 修改主 flow、attention 或坐标映射后，优先运行 `tests/` 中对应的一致性测试，再启动高显存端到端实验。
