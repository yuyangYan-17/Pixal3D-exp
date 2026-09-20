# Pixal3D 单图超分

当前仅维护一条 SR 流程及原生 baseline1024 对比流程。默认使用物理 GPU4；模型、DINO、MoGe 和 encoder 路径沿用当前机器配置（`inference.py`、`sr_tools/baseline.py`）。Python 环境为 `/home/nvme04/yyyan/miniconda3/envs/pixal3d/bin/python`。

```bash
# 一张输入图：baseline + 几何/纹理 SR + 指标 + 多视角渲染
python sr.py --image assets/images/0_img.png --output-dir outputs/sr_0 --gpu 4

# 接口 smoke：真实 baseline；每个 SR flow 一步；小范围解码
python sr.py --image assets/images/s_20_img.webp --output-dir outputs/smoke --gpu 4 --smoke

# 只运行 baseline1024，包括渲染和输入视角指标
python inference.py --image assets/images/0_img.png --output-dir outputs/baseline_0 --gpu 4

python -m unittest discover -s tests -v
```

建议使用 GPU UUID 防止设备编号重排；省略 `--gpu` 即绑定当前物理 GPU4 的 UUID。重复命令会从本次输出目录恢复，输入或代码版本不匹配会拒绝复用。不能将旧实验输出直接当作新流程缓存。

## 流程

1. 一次前景分割/裁剪，生成对齐的 512、1024、4096 图像及 mask；MoGe 估计共享输入相机。
2. `run_baseline1024()`：SS512 → Shape512 → 原生 C64 support → Shape1024 → Texture1024 → 带材质的 1024 mesh。各 flow 12 步，几何 seed42、纹理 seed46。
3. baseline mesh voxelize2048 → encoder posterior mean → 全局 normalized C128。
4. 4096 图按不重叠 2048 core 切 4 图，各边 padding32，再 resize512；Z context32/stride16。全局 C128 加一次噪声，所有块共享冻结状态，12 步同步 Shape512。
5. 全局 decoder.upsample(...,1) → C256；4096 图按不重叠 1024 core 切 16 图，各边 padding32，再 resize1024；Z context64/stride32，12 步同步 Shape1024。
6. 在完全相同的最终 C256 支持和分块上，使用最终 shape 特征作条件，12 步同步 Texture1024。
7. 全局几何解码4096，保存 subdivision；纹理解码复用同一 subdivision，组合成高分辨率带材质 mesh。
8. 用同一输入相机渲染原生1024 base color，计算共享 GT 前景内 PSNR、SSIM、LPIPS；两个模型均保存6个纹理视角、几何法向量视角，每张2048²，拼图12288×2048。

XY 仅平移并 localize 索引；Z 索引减块起点。padding 点只提供上下文，不能取得更新所有权。同一 owner 图像的 Z 重叠 velocity 求平均，每一步只更新一次全局状态。显存估计只测最大点数块的一次 forward，按增量占用和剩余显存线性估计 batch，预留 max(8GiB, 10%)；运行时 OOM 递归拆 batch。每个样本的 Linear 运算尺寸保持一致，避免 batch 改变数值路径。

## 代码

| 文件 | 职责 |
| --- | --- |
| `sr.py` | 单图 SR 入口与完整流程编排 |
| `inference.py` | 模型初始化、相机估计、baseline1024 命令入口 |
| `sr_tools/baseline.py` | 可复用 baseline1024 函数与 C128 编码入口 |
| `sr_tools/common.py` | 原生 baseline 采样阶段、编码、原子缓存 IO |
| `sr_tools/shape.py` | 坐标映射、图像条件、owner reduction、几何同步 flow |
| `sr_tools/texture.py` | 纹理条件与同步 flow |
| `sr_tools/batching.py` / `capacity.py` | batch 数值一致性与显存估计 |
| `sr_tools/rendering.py` | 几何/材质解码与相机对齐的多视角渲染 |
| `sr_tools/metrics.py` | 1024 输入视角前景指标 |

`pixal3d/pipelines/pixal3d_image_to_3d.py` 只保留上述流程需要的模型接口。旧多路径 `pipeline.run(...)`、wavelet 和实验分块入口已经移除；用两个命令入口或 `run_baseline1024()` 调用。解码器保留分块 LayerNorm/output 修复，防止超大 mesh 尾部退化为碎点。

## 输出

- `baseline/textured_mesh.pt`：baseline 原生 MeshWithVoxel，含 PBR 属性。
- `sr/final_mesh/geometry_mesh.{pt,glb}`：SR 几何；`topology.pt` 用于纹理解码。
- `sr/texture/textured_mesh.pt`：SR 原生 MeshWithVoxel，含体素材质；没有执行 UV 展开/材质 GLB 烘焙。
- `sr/evaluation_1024/`：reference、foreground、baseline、SR、对比图、`metrics.json`。
- `baseline/views_2048/`、`sr/texture/views_2048/`：对齐的纹理/法向量子图；其父目录保存完整分辨率拼图。
- `result.json`、`status.json`：汇总和运行状态。`--smoke` 写入独立 `smoke/` 和 `smoke.json`，其指标不能用于质量对比。

指标只使用输入前景 mask：RGB MSE→PSNR；Gaussian11/sigma1.5 的 SSIM map 前景均值；预训练 AlexNet LPIPS v0.1 spatial map 前景均值。没有图像配准或按预测前景缩小评价区域。

超大完整解码仍受 GPU 显存限制；flow 的 batch 拆分不能解决全局 decoder 的 OOM。smoke 的小范围解码不证明完整4096解码一定能装入显存。

旧实验代码及训练实验目录已归档到仓库外 `/home/nvme04/yyyan/Pixal3D_legacy_20260919_100929`；原有 `outputs/`、输入数据和模型权重保留。

## 本次验证

GPU4 上已通过新输入完整 baseline、SR 各 flow 一步及断点恢复 smoke；1024 前景指标、36 张原生2K子图与相机对齐均验证通过。三个 flow 的单/多 batch 误差均为0；保存并复用 subdivision 与直接解码的顶点、面、体素坐标、材质属性逐项完全相同。另通过4项CPU回归检查与静态检查。

验证记录：`outputs/consolidated_sr_smoke_20260919/validation.json`。本次整理没有重跑全量12步SR质量实验；smoke 解码只使用小范围支持。

## Z 边界加权对比

当前两阶段几何 flow 使用体素中心三角窗：`w(z)=1-|2*(z_local+0.5)/context-1|`，先累加 `w*v` 再除以每点权重和。保留 XY owner 和 padding 约束；纹理 flow 暂保持等权，以单独验证几何改变。新输出签名版本3，不允许读取旧等权几何 checkpoint。`outputs/sr_0_img_z_tent/` 对照 `outputs/sr_0_img/`，复用相同 baseline，GPU4 上先 smoke，成功后自动开始完整实验；日志在新目录的 `smoke.log` 和 `run.log`。

## 基于可见性的 baseline 纹理引导实验

```bash
python sr_visibility_sweep.py --source outputs/sr_0_img_z_tent \
  --output-dir outputs/sr_0_img_visibility_guidance
```

默认 GPU4，入口先跑 GPU smoke，再准备共享几何/引导，最后依次跑 n0（全 conditional 对照）、n1–n4。复用 source 已完成的 Shape512 endpoint，替换桥接为 `C128 decode mesh2048 → voxel4096 → shape encoder C256`，再执行 Shape1024 和全局解码。所有变体共用该几何。

输入相机在4096分辨率进行遮挡光栅化；可见三角面集合使用同一 voxelizer 再次体素化，与完整4096体素支持求交。C256 的四级空间下采样祖先中任一来源面可见，就标记可见。保存可见面ID、可见面体素集合、每个完整体素的可见标记和 C256 父行号以供追溯。小于一个光栅像素的面可能没有命中，这是此可见性定义的采样限制。

块内全部点（包括 padding）严格超过30%可见时保持 conditional，否则前 n 次更新从对应的 baseline clean endpoint 构造 velocity，第 n+1 次开始图像 unconditional；保留 shape concat condition。n0全部 conditional。使用同一全局初始噪声、同一12步时间表、原有 XY owner 和纹理 Z 等权融合。批处理默认使用 `guide_steps=4`、`guide_threshold=1.0`、`hidden_mode=unconditional`；只有 baseline 查询完整支持的隐藏点接受 endpoint 引导，避免把无效查询的归一化零特征送入 flow。

baseline 1024 纹理前四步按同一 seed46 与时间表重放，用 sampler 含 sigma_min 的原公式预测 clean endpoint，解码并缓存材质场；在新4096体素 dual vertex 的连续世界坐标上查询1024稀疏材质场的最近三个激活点，按距离反比加权，不再把无支持点零填充（`guides/coverage_*.json` 记录距离统计）。四份材质使用与 shape 相同的4096支持编码，严格校验并重排到同一 C256 行序。

- `baseline_fields/`：四步 endpoint、时间和解码材质场。
- `shared/bridge/`：4096体素、C256支持和可见性来源记录。
- `shared/block_visibility.json`：逐块可见比例与分支。
- `guides/`：四份对齐的纹理 C256 latent、查询有效覆盖率。
- `n0/`–`n4/`：各自 flow、带材质 mesh、1024前景三指标和每张2K的多视角图。
- `summary.json`、`status.json`：对比汇总和状态；阶段日志 `prepare.log`、`n*.log`。

批量执行每个阶段使用独立进程释放显存；smoke 或公共准备失败会停止，某个变体失败会记录后继续其余变体。失败不计入完整结果。

## 不可见纹理探索与批处理

`sr_texture_explore.py` 在已经完成的 `shared/final_mesh` 上只重跑 Texture1024，几何不会改变。每个空间/深度块按 4096 可见性选择唯一的一条路线：块可见比例严格超过30%走 image-conditional；隐藏块在前四步且最近三点查询都有材质支持时整块走对应 clean endpoint，否则整块走 `unconditional`。一个原始块不会被拆成 conditional 与 unconditional 两个 forward；所有块的 velocity 在全局 C256 上归并后再做一次 Euler 更新。

单个固定几何的探索示例：

```bash
python sr_texture_explore.py --source outputs/sr_0_img_visibility_guidance \
  --output-dir outputs/sr_0_img_texture_explore/full_g4_unconditional_seed46 \
  --mode full --guide-steps 4 --guide-threshold 1.0 \
  --hidden-mode unconditional --gpu 4
```

批处理入口会先为每张图执行 baseline1024 和几何 SR（几何只生成一次），再缓存 C128→C256 可见性、四个 baseline endpoint 和所有纹理条件，最后按多个种子重跑纹理、解码 4096 材质、渲染六张 2048² 视图，并在输入视角 1024 前景上计算 PSNR、SSIM、LPIPS：

```bash
python sr_batch_experiment.py --phase batch --seeds 46,47,48 \
  --hidden-mode unconditional --guide-steps 4 --guide-threshold 1.0 --gpu 4 \
  --output-root outputs/sr_texture_visibility_final_20260919
```

`batch.json`、`summary.json` 和每张图的 `full_timing.json` 同时保存 baseline、几何、准备、纹理和完整流程耗时；后台运行时 `batch.pid`、`batch_command.txt` 和 `run.log` 位于同一输出目录。每个阶段完成后会清掉可由 endpoint 重算的 flow tensor、已编码的临时 tile 和中间 2048 bridge，保留最终 mesh、endpoint、指标、JSON 和 2K 视图。
