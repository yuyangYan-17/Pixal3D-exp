# Global C256 / Local C64 Cube Owner Flow：逐 Cube 4096 Local Condition

## 结论

这套构造在坐标和同步更新上是自洽的，可以作为实验方案运行。实现保持唯一的 global C256 `X_t`，只把每个 cube 的图像条件改成独立 local condition；默认 `--velocity-fusion owner` 下，每个 global row 仍只采用三维中心最近 cube 的 velocity，并在收齐全部 proposal 后做一次 Jacobi/Euler 同步更新。

主要实验风险不是坐标错误，而是条件分布偏移：DINO/NAF 和 flow model 主要在固定 1024 全图条件上训练，现在输入是尺寸可变的局部 crop，局部 CLS/register token 不再表达完整物体。代码不掩盖这个差异，并在输出 manifest 中记录每个 crop。

## 固定的三维状态与 Cube 划分

- C4096 occupancy 以 `floor(coord / 16)` 降采样并去重，得到唯一 global C256 sparse row table。
- C4096 的 `1024 / stride 512` cube 等价为 C256 的 `64 / stride 32`，各轴 start 为 `0,32,...,192`，总数 `7³ = 343`。
- 每个 cube 只对 state/concat 的坐标做 `global_xyz - start`，得到 local C64 坐标；row 身份和值始终来自同一份 global table。
- cube membership 使用半开区间 `[start, start + 64)`；重叠只产生多个 velocity proposal，不产生多份可独立演化的 `X_t`。

对应实现位于 `pixal3d_global_c256_cube_owner_flow_singleview.py` 的 `build_cube_records`、`center_translate_scale_to_local_c64` 和 `build_owner_map`。

## 每个 Cube 的 Local Token 构造

### 1. 投影物理 Cube，而不是重用全图 feature

对每个 C256 cube 使用八个物理边界角点：

```text
boundary = start + {0, 64}³
q_global = 2 * boundary / 256 - 1
```

角点使用与 `ProjGrid` 相同的 global centered camera 投影。为严格匹配模型的 pixel-center 约定，先在模型的 C1024 投影平面得到 `uv`，再用 `(uv + 0.5) / 1024` 转成 normalized pixel-edge 坐标，最后映射到 canonical 4096。这样不会把直接 C4096 投影与 `ProjGrid` 的 `+0.5` 约定混用而产生两像素偏移。

实现：`physical_boundary_q`、`cube_projection_crop`。

### 2. Bbox 只向外扩展，并对齐 DINO patch 16

对八个投影角点取 axis-aligned bbox：

- 先与 canonical 4096 图像求交；越界 cube 使用所有仍在图内的像素。
- left/top 向下取整并向外对齐到 16。
- right/bottom 按右下排他的 pixel-edge 约定向上取整并向外对齐到 16。
- 最终 crop 的 width/height 均为 16 的整数倍，且不缩小有效投影区域。

实现：`align_projected_crop_box`、`attach_cube_projection_crops`。完整 343-cube bbox、原始/裁剪范围和尺寸会写入 `conditions/projection_diagnostics.json`。

### 3. 保持 Crop 的原始长宽比运行 DINOv3

不能先对齐到 16、随后又强制拉伸到 `1024×1024`，否则“16 倍数”约束失去意义，非方形 crop 也会发生几何扭曲。因此 `DinoV3ProjFeatureExtractor` 新增 `preserve_input_resolution=True` 路径：

- PIL crop 不 resize。
- 检查原始 height/width 都可被 patch size 16 整除。
- DINO patch grid 动态使用 `(height / 16, width / 16)`，不再假设固定 `64×64`。
- 默认调用仍保持原来的固定输入行为，不影响其他 pipeline。

接口通过 `Pixal3DImageTo3DPipeline.get_proj_cond_shape(..., preserve_image_resolution=True)` 暴露。

### 4. NAF 保持原模型倍率并支持矩形输出

NAF 的 target height/width 分别按 crop 尺寸缩放：

```text
target_h = round(crop_h * nominal_naf_h / 1024)
target_w = round(crop_w * nominal_naf_w / 1024)
```

- Shape 1024 condition 的 nominal NAF target 是 512，因此保持约 `1/2` 尺度。
- Texture 1024 condition 的 nominal NAF target 是 1024，因此保持约 `1:1` 尺度。
- DINO low-resolution projected feature 与 NAF high-resolution projected feature仍按 channel concat，输出 channel 契约不变。

### 5. Projected token 与 Global token 都改为局部来源

每个 flow-active cube 独立调用一次图像编码器：

- `proj`：用该 cube 的 global C256 row 坐标做 global-camera 投影，再用 `projection_crop_box` 映射到 crop-local feature coordinates。
- `global`：使用同一次 local crop DINO forward 产生的 CLS + register tokens，不再复制全图 global token。
- 进入 flow model 前，`proj.feats` 保持该 cube 自己的 row 顺序，SparseTensor coords 改为对应 local C64 coords。

因此，同一个 global row 出现在两个重叠 cube 中时，可以具有不同的 local `proj` 和 `global` 条件；它的 frozen `x_t` 值仍完全相同。

实现：`build_condition`、`_validate_cube_condition`、`_pack_condition`。

## Owner Velocity 与 Global Jacobi 更新

Owner 规则没有改：

```text
owner(row) = argmin over containing cubes
             || global_cell_center(row) - cube_center ||²
```

若距离相同，按遍历顺序保留最小 `cube_id`。每一步执行：

1. 冻结唯一 global `X_t`。
2. 每个 flow-active cube gather 自己的 membership rows，使用 local C64 coords 和该 cube 的 local condition 计算一次 `pred_v`。
3. 默认 owner 模式只 scatter `owner(row)` 对应的 velocity；其他重叠 cube 对该 row 的 proposal 丢弃，不平均。
4. 收齐并验证每个 global row 恰好写入一次后，统一执行 `X_next = X_t - (t - t_next) * V_owner`。

核心实现仍是 `validate_owner_scatter`、`jacobi_update` 和 `run_flow`。如显式选择 `--velocity-fusion gaussian`，则会对所有包含 cube 的 velocity 做三维 Gaussian 融合；这不是本方案要求的默认语义。

## Cache 与运行范围

- 默认 owner 模式只提取有 owned row 的 cube condition；没有 owner row 的 cube 永远不会贡献 velocity，提取其图像 feature 没有运行意义。
- Gaussian 模式会为全部 nonempty cube 提取 condition。
- Shape/Texture 分开缓存在 `conditions/{stage}_local_cubes/cube_XXX.pt`，每个文件保存 crop、row IDs、local `global` 和 local `proj`，并用图像、support、camera、crop 和 fusion mode 指纹拒绝错误复用。
- 每个 stage 完成 flow 后释放其整套 CPU local condition，再构建下一 stage，避免 Shape 与 Texture 的大体积 projected feature 同时驻留。

默认运行保留：

```bash
CUDA_VISIBLE_DEVICES=4 python pixal3d_global_c256_cube_owner_flow_singleview.py \
  --device cuda:0 \
  --physical-cuda 4 \
  --velocity-fusion owner \
  --condition-image-4096 outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024/inputs/canonical_foreground_rgb_4096.png
```

旧的 `--condition-image` 名称仍作为同一参数的兼容 alias，但现在该路径必须是 canonical `4096×4096` 图像。

## 已知风险边界

- Variable-size local DINO/NAF condition 对 flow model 是分布外输入；代码正确不等于生成质量一定提升。
- Local CLS/register token 缺少完整物体语义，不同 cube 的 global token 会跳变；这是本实验明确要求的行为。
- 物理 cube bbox 是保守区域，会包含背景、遮挡面投影和其他深度上的像素；单视图无法提供真实背面信息。
- 投影超出 4096 时 crop 会裁到图像边界，crop 外 query 延续已有 `grid_sample(..., padding_mode="border")` 行为。
- Hard owner 保证确定、无平均，但可能在三维 Voronoi owner 边界形成 velocity/latent 不连续。
- Global support 固定来自 baseline mesh 的 C4096 voxelization，flow 无法在 support 外自由创建新拓扑。

## 验证

`tests/test_global_c256_cube_owner_flow_singleview.py` 覆盖：

- C4096/C256/C64 坐标与 343-cube membership。
- 最近三维中心 owner、tie-break、每 row 恰好一次 scatter。
- crop bbox 向外 16 对齐及 normalized crop mapping。
- 不同 cube 的 local `proj` 与 local `global` 打包，不再退化为全局 gather。
- 矩形 native-resolution DINO patch layout 与按比例 NAF target。
- Jacobi cube-order independence、Gaussian 显式分支和 Shape/Texture concat 对齐。

当前验证结果：完整 `tests/` 为 `102 passed`；真实 Shape DINOv3+NAF 权重也已用 `928×912` local crop 冒烟，得到有限的 `[1,5,1024]` global token 和 `[1,8,2048]` projected features。
