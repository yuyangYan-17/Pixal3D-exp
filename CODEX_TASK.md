# CODEX TASK：基于高分辨率图像分块的 Texture Flow

在现有 `pixal3d_directory_texture_eval.py`、`pixal3d/pipelines/pixal3d_image_to_3d.py`、图像投影模块及 sampler 代码基础上实现。只修改 texture 阶段，shape 生成流程暂时保持不变。

## 一、修改前备份

开始修改前必须备份当前代码，不允许覆盖已有备份。

1. 记录：

   * `git status --short`
   * 当前 commit hash
   * `git diff`
2. 对所有准备修改的文件创建带时间戳的副本，例如：

   * `pixal3d_directory_texture_eval.py.bak_YYYYMMDD_before_hr_image_tile_texflow`
   * `pixal3d_image_to_3d.py.bak_YYYYMMDD_before_hr_image_tile_texflow`
3. 将备份文件列表和当前 commit 写入：

   * `HR_IMAGE_TILE_BACKUP.md`
4. 不执行 `git reset`、`git checkout -- .` 等可能丢失现有修改的操作。

## 二、目标流程

```text
在 # ---- Stage 4: Texture (proj) ---- 之后：
高分辨率原图(处理完前景与边缘扩展)
    ├── resize 到正常 1024 输入
    │       └── 完整执行一次原始 texture flow
    │               └── 保存 13 个 latent states、
    │                   12 个最终 Euler velocities 和时间间隔
    │
    └── 按图像空间切成多个 1024×1024 tile
            └── 每个有效 tile 独立：
                    重新运行 DINO
                    重新运行 NAF
                    三维找二维粗糙图找二维精细图，找到属于该 tile 的 shape_slat.coords
                    构造该 tile 自己的 cond_tex
                    在当前时间 t 上调用 texture flow
                    得到该 tile 的最终 CFG velocity
```

增强采样使用：

```text
前 6 步：原始全局 texture flow
后 6 步：图像 tile 独立预测 velocity
         → 重叠区域加权融合
         → 得到唯一 global velocity
         → 每个时间步只更新一次完整 texture latent
```

保存的 `global_state` 的flow相关数值，局部flow时候直接恢复指定时间点

## 三、保留高分辨率原图

当前预处理可能先把图像缩小。修改数据流，同时保留：

* `global_image`：按照原预处理流程得到的正方形图像，供正常 1024 texture condition 使用；
* `hr_image`：使用完全相同的前景 bbox、扩边、padding 和正方形坐标系，但保留原始高分辨率；
* `foreground_mask_hr`：与 HR 正方形图严格对齐的 alpha/前景 mask；
* 从 global square 到 HR square 的精确线性坐标变换。

不能让 global 和 HR 分支分别重新抠图或重新计算 bbox。

## 四、图像分块

第一版使用无重叠规则分块：

```text
tile_size = 1024
tile_stride = 1024
```

例如 4096×4096 图像得到 4×4 共 16 个 tile。

根据 `foreground_mask_hr` 判断 tile 是否有效，不要根据 RGB 是否为黑色判断。纯背景 tile 直接跳过。

保存每个 tile 的：

* tile 编号；
* HR 图上的 `(x0, y0, x1, y1)`；
* 前景占比；
* 是否启用；
* tile 图像；
* 对应的 sparse token 数量。

接口应允许以后修改 tile size、stride 和 overlap。

## 五、从 shape_slat.coords 分配 token

texture flow 的全局稀疏坐标始终使用：

```python
shape_slat.coords
```

坐标集合和全局 token 顺序不可改变。

对全部 `shape_slat.coords` 使用 Pixal3D 当前相同的相机参数和坐标变换，投影到 global square 的二维归一化坐标，再线性映射到 HR square。

对每个 tile，找到二维投影落入其 crop 范围且前景 mask 有效的 token，得到：

```python
tile_global_indices
tile_coords = shape_slat.coords[tile_global_indices]
```

禁止根据二维像素反向生成新体素；只能从现有三维 shape_slat.coords 投影到二维后进行筛选。

保存调试可视化：

* 所有 coords 在 global image 上的投影；
* 在 HR image 上的投影；
* tile 边界；
* 不同 tile 对应 token 的颜色；
* 每个 tile 的 token 数量。

## 六、每个 tile 独立重跑图像条件

每个有效 tile 都必须独立调用 texture 图像条件模型，重新执行：

* DINOv3；
* NAF；
* pixel-aligned projection。

不能先提取整图 feature map 后直接裁 feature，也不能把局部 feature 预先写回一个全局 `cond_tex`。

实现 tile-aware 的 `get_proj_cond_shape`，例如：

```python
tile_cond_tex = self.get_proj_cond_shape(
    image_cond_model=self.image_cond_model_tex_1024,
    image=[tile_image],
    coords=tile_coords,
    camera_angle_x=camera_angle_x,
    distance=distance,
    mesh_scale=mesh_scale,
    grid_resolution_override=...,
    projection_crop_box=tile_box_normalized,
)
```

关键要求：

现有三维点首先按全局相机投影到完整正方形图像。随后必须把完整图上的二维坐标变换为 tile 内部坐标，再从该 tile 的 DINO/NAF feature map 采样。

不能把 crop 当成新的完整相机图像，然后仍使用原始投影坐标，否则三维 token 和局部图像内容会错位。

保持该 tile 的：

* `cond["global"]`；
* `cond["proj"]`；
* `neg_cond["global"]`；
* `neg_cond["proj"]`

均与 `tile_coords` 和 tile latent 的 token 顺序严格一致。

第一版沿用原始 zero negative condition 和原始 texture CFG 语义。

## 七、全局基线 trajectory

先使用正常 resize 到 1024 的完整图像构造原始：

```python
global_cond_tex = self.get_proj_cond_shape(...)
```

在全部 `shape_slat.coords` 上完整执行一次原始 12-step texture flow，并保存：

* `states[0:13]`；
* `velocities[0:12]`；
* 原始和非线性映射后的时间；
* 每步 Euler interval；
* 每步最终实际用于更新的 velocity；
* CFG strength、interval、rescale 等参数。

这里的 velocity 必须是完成 conditional/unconditional CFG、guidance interval 和 guidance rescale 后的最终 velocity。

保存全局结果作为 baseline，并确保关闭新功能时输出与当前代码一致。

## 八、后六步 tile texture flow

从：

```python
x_global = saved_global_states[6]
```

开始执行步骤 6～11。

每个时间步 `t_i`：

1. 遍历所有有效 tile；
2. 使用 `tile_global_indices` 从当前完整 texture latent 中提取 tile latent：

   ```python
   x_tile = x_global[tile_global_indices]
   ```
3. 同步切出与 texture flow 对齐的 shape condition；
4. 使用该 tile 自己重新运行 DINO/NAF 得到的 `tile_cond_tex`；
5. 使用与原始 Pixal3D 完全相同的：

   * 非线性时间映射；
   * conditional/unconditional forward；
   * CFG interval；
   * guidance strength；
   * guidance rescale；
   * velocity/x0 转换；
6. 得到该 tile 在当前 `t_i` 上最终用于 Euler 更新的：

   ```python
   v_tile
   ```
7. 使用全局索引把 `v_tile` scatter 回完整 token 空间；
8. 多个 tile 覆盖同一 token 时进行加权平均；
9. 得到唯一的：

   ```python
   v_merged_global
   ```
10. 对完整 texture latent 只执行一次：

```python
x_global = x_global - dt_i * v_merged_global
```

禁止让每个 tile 独立完成后六步以后再拼 latent。

禁止修改 `shape_slat.coords`、全局 token 数量或 token 顺序。

## 九、未覆盖 token 的处理

代码逻辑是三维查二维，二维特征图是稠密的，肯定有对应点，查不到说明代码错了，报错中断。

## 十、重叠融合

即使第一版 stride 等于 tile size，也把融合逻辑写成支持重叠 tile。

每个 tile 使用二维中心权重或 separable tent weight。对 token 的融合为：

```python
velocity_sum[index] += weight * v_tile
weight_sum[index] += weight
```

最终：

```python
v_local = velocity_sum / weight_sum.clamp_min(eps)
```

对于 `weight_sum == 0` 的 token，使用指定的 global fallback velocity。

## 十一、诊断与保存

每个后半程步骤保存：

* 当前 step、原始 t、映射后的 t 和 dt；
* 有效 tile 数；
* 各 tile token 数；
* 被局部 tile 覆盖的全局 token 比例；
* overlap token 比例；
* `v_merged_global` 与保存的 global baseline velocity：

  * cosine similarity；
  * mean token cosine；
  * MSE；
  * relative L2；
  * norm ratio；
* 每个 tile velocity 的 norm；
* 未覆盖 token 数；
* global fallback 模式；
* 峰值显存和运行时间。

保存最终：

* 全局 baseline texture latent；
* tile-enhanced texture latent；
* flow trace；
* tile metadata；
* projection 可视化；
* 完整运行配置。

将关键信息写入当前实验日志格式，但不要伪造实验指标；代码修改完成后只记录“待用户运行”。

## 十二、CLI

至少增加：

```text
--hr-image-tile-texture-flow
--hr-image-tile-size 1024
--hr-image-tile-stride 1024
--hr-image-tile-start-step 6
--hr-image-tile-min-foreground-ratio
--hr-image-tile-fallback saved_global|current_global
--hr-image-tile-weight tent|uniform
--hr-image-tile-save-debug
```

默认关闭新功能。关闭时必须完全走当前 pipeline。

开始 step 的定义要清楚：

```text
start_step=6 表示 states[6] 作为起点，
执行 velocity steps 6,7,8,9,10,11。
```

## 十三、实现约束

* 暂时不要修改 sparse structure flow；
* 暂时不要修改 shape LR 或 shape HR flow；
* 暂时不加入 Haar、HiWave frequency CFG 或 skip residual；
* 暂时不训练模型；
* 不改变模型权重和通道维度；
* 不改变 texture decoder；
* 不改变相机估计、渲染和指标计算；
* tile 图像必须真正重新运行 DINO 和 NAF；
* 所有 image condition、texture latent、shape condition、coords 的 token 顺序必须显式断言一致；
* 空 tile 必须跳过；
* 尽量批量处理 tile，但优先保证语义正确和显存安全；
* 不删除或破坏当前已有的 shape/texture patch-flow 实验代码。

## 十四、完成标准

修改完成后先不要运行完整生成实验，只进行：

1. Python 语法检查；
2. import 检查；
3. 小规模 synthetic coordinate 测试；
4. global-to-HR-to-tile 坐标映射 round-trip 测试；
5. scatter/merge token 顺序测试；
6. overlap velocity 融合测试；
7. 关闭功能时配置回归检查。

最后给出：

* 修改文件列表；
* 备份文件列表；
* 主要数据流；
* 新增 CLI；
* 用户应运行的完整命令；
* 预计显存热点；
* 尚未验证的风险。

最重要的验证目标是：

> 高分辨率图像 tile 必须分别重新运行 DINO、NAF 和 texture-flow model；最终融合的是每个 tile 在相同时间步预测出的 velocity，而不是预先融合图像特征。
