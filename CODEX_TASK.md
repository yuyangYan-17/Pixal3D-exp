# CODEX_TASK.md

## 任务名称

实现 Pixal3D 2048 Texture Flow 的：

* 单次统一高质量图像预处理；
* 4K 重叠图像 tile DINO/NAF 条件；
* 每个 3D token 对应多个成对的 `(tile global, tile proj)`；
* Transformer block 内 paired multi-tile 条件融合；
* 128-grid 上的 64³、stride 32 三维 patch flow；
* 无任何 global velocity fallback 的严格完整覆盖更新。

---

# 1. 仓库与执行要求

工作目录预计为：

```text
/home/nvme04/yyyan/Pixal3D
```

主要执行 GPU：

```text
CUDA_VISIBLE_DEVICES=4
```

离线模型模式：

```text
HF_HUB_OFFLINE=1
```

本任务不是只写方案。必须：

1. 阅读当前仓库最新代码；
2. 直接修改实现；
3. 添加测试；
4. 运行测试；
5. 修复测试暴露的问题；
6. 至少执行一次 identity 验证；
7. 条件允许时执行首次完整实验；
8. 更新 `EXPERIMENT_LOG.md`；
9. 最终报告修改文件、测试结果、准确运行命令、实验结果和已知风险。

不要假设下面列出的函数名与当前仓库完全一致。先定位当前实际实现，再按现有架构完成修改。

---

# 2. 背景与已确认结果

当前 2048 clean global baseline：

```text
PSNR  = 17.1593
SSIM  = 0.666374
LPIPS = 0.218942
```

当前按 2D image tile 反查 token、再把不规则 token 子集送给 Texture Flow 的结果：

```text
Global shape + image-tile texture flow, step 6:
PSNR  = 16.9282
SSIM  = 0.653563
LPIPS = 0.225339
```

全程 image-tile texture flow、tile 1024、stride 512、step 0：

```text
PSNR  = 16.6391
SSIM  = 0.659533
LPIPS = 0.228252
```

旧 image-tile flow 的根本问题：

1. Flow token 集合由二维图像 tile 决定；
2. 同一个 tile 反查出的 token 在三维空间中可能相距很远；
3. Flow 输入不是一个完整连续的 64-grid 三维局部块；
4. token 仍使用残缺的 128-grid global coordinate subset；
5. 每个 tile 的 DINO global 不同；
6. 旧实现将 tile global 与该 tile token 子集绑定，但丢失完整三维 self-attention 上下文；
7. 存在未覆盖 token 和 global velocity fallback；
8. 重叠 tile 的 velocity 融合不能解决上述结构问题。

本任务必须废弃“2D tile 决定 Flow token 集合”的逻辑。

---

# 3. 本任务的核心设计

## 3.1 两个分块空间必须解耦

### 2D 图像 tile

只负责：

* 从 canonical 4096 图中裁取局部 1024 crop；
* 每个 crop 独立运行 DINO 和 NAF；
* 产生该 crop 的：

  * DINO global `[1, 5, 1024]`；
  * DINO/NAF projected features；
* 建立每个 3D token 与一个或多个重叠 tile 的条件关系。

### 3D Flow patch

只负责：

* 决定一次 Flow forward 处理哪些 token；
* 在 global 128-grid 中按欧氏坐标切 64³ patch；
* stride 32；
* 将 global coord 平移为 `[0,63]³` local coord；
* 保留完整三维 patch 内的 self-attention。

禁止让 2D tile 决定哪些 token 一起进入 Flow。

---

## 3.2 一个 token 可以有多个 tile 条件

Canonical image：

```text
4096×4096
```

图像 tile：

```text
tile_size   = 1024
tile_stride = 512
```

固定起点：

```text
[0, 512, 1024, 1536, 2048, 2560, 3072]
```

总数：

```text
7 × 7 = 49 tiles
```

图像内部的一个投影点最多同时属于 4 个 tile。

对全局第 `i` 个 3D token，保存：

```text
tile_ids[i]       [M]
tile_weights[i]   [M]
tile_proj[i]      [M, 2048]
```

其中：

```text
1 <= M <= 4
```

每个有效 slot 都表示一对严格匹配的条件：

```text
(
    tile_global_bank[tile_id],
    tile_proj_from_the_same_tile
)
```

禁止出现：

```text
global 来自 tile A
proj 来自 tile B
```

---

## 3.3 Flow block 内进行 paired condition fusion

当前 Project Attention 的语义大致是：

```text
global_out = CrossAttention(hidden, global)
proj_out   = Linear(proj)
image_out  = global_out + proj_out
```

新语义必须是：

[
c_i^{(b)}
=========

\sum_{j\in\mathcal T(i)}
\alpha_{ij}
\left[
\operatorname{CrossAttn}*b(h_i,G_j)
+
W_b P*{ij}
\right]
]

其中：

* (i)：当前 3D token；
* (j)：覆盖该 token 的 image tile；
* (b)：Flow Transformer block；
* (G_j)：tile `j` 的 DINO CLS/register；
* (P_{ij})：token `i` 从 tile `j` 查询到的 DINO/NAF proj；
* (\alpha_{ij})：归一化二维 tile 权重；
* 每个 token 的有效权重和必须严格为 1。

完整 3D self-attention 只运行一次。

不要为每个 tile 重跑整个 Flow block 或整个 Flow model。

---

# 4. 必须保持的 Flow 结构

以一个包含 `K` 个 sparse token 的 64³ patch 为例。

输入：

```text
texture latent:
    feats  [K, 32]
    coords [K, 4]

shape concat:
    feats  [K, 32]
    coords [K, 4]

拼接后:
    feats [K, 64]

projected condition:
    fused proj [K, 2048]

tile global bank:
    [T, 5, 1024]

token tile ids:
    [K, M]

token tile weights:
    [K, M]
```

Flow block 顺序必须仍然是：

```text
full 3D self-attention over all K tokens
    ↓
paired multi-tile global/proj condition fusion
    ↓
MLP
```

输出：

```text
patch velocity [K, 32]
```

不得把 patch 再按照 tile ID 切成残缺的 Flow token 子集。

---

# 5. 统一图像预处理

## 5.1 预处理只允许执行一次

原始图像进入 pipeline 后，只允许执行一次：

* 判断有效 alpha；
* 或运行一次 rembg；
* 计算 foreground bbox；
* 将 bbox 映射回源图分辨率；
* 从源图 RGB/alpha 构造前景正方形；
* padding；
* 黑背景合成；
* 生成 canonical image pyramid。

后续所有阶段不得再次：

```text
调用 rembg
重新计算 bbox
重新 crop
重新 padding
调用 preprocess_image
调用 preprocess_image_with_hr
```

应新增或重构为明确的单次接口，例如：

```python
preprocess_canonical_images(...)
```

建议返回 dataclass 或结构化字典：

```python
{
    "image_4096": PIL.Image.Image,
    "image_1024": PIL.Image.Image,
    "image_512": PIL.Image.Image,
    "foreground_mask_4096": PIL.Image.Image,
    "metadata": {...},
}
```

---

## 5.2 Alpha 与 rembg

规则：

1. 输入是 RGBA 且 alpha 不是全 255：

   * 使用原 alpha；
   * 不调用 rembg。

2. 输入没有有效 alpha：

   * 最长边缩到不超过 1024，仅用于一次 rembg；
   * 将得到的 alpha 和 bbox 映射回源图坐标；
   * 最终 RGB 必须来自原始高分辨率源图，不得来自 rembg proxy RGB。

3. 如果 alpha mask 为空：

   * 直接报错；
   * 不允许回退为整图 bbox。

---

## 5.3 前景、margin、正方形和 padding

保留现有 1.1 倍前景 margin 语义：

```python
side = max(foreground_width, foreground_height)
side = side * 1.1
```

要求：

* 在源图坐标中确定最终 square；
* square 超出源图边界时显式 padding；
* padding 先以透明 RGBA 表示；
* 在高分辨率 square 上合成黑色背景；
* 不允许直接 `RGBA.convert("RGB")` 丢弃 alpha；
* 不允许透明区域的隐藏 RGB 污染 DINO 输入。

最终得到一个高质量 source-square RGB。

---

## 5.4 Canonical resize 链

必须：

```python
image_4096 = source_square.resize(
    (4096, 4096),
    Image.Resampling.LANCZOS,
)

image_1024 = image_4096.resize(
    (1024, 1024),
    Image.Resampling.LANCZOS,
)

image_512 = image_4096.resize(
    (512, 512),
    Image.Resampling.LANCZOS,
)
```

禁止：

```text
原图 → 1024 → crop → 1024
1024 → 512
不同阶段各自重新处理
不同阶段重新找前景
```

512 和 1024 都必须直接从 canonical 4096 resize。

---

## 5.5 各阶段图像来源

固定：

```text
Sparse Structure:
    image_512

Shape 512:
    image_512

Shape 1024:
    image_1024

Global Texture baseline:
    image_1024

Multi-tile HR Texture:
    image_4096 切出的 1024 tile
```

传给 extractor 的图像应已经是准确尺寸，使 extractor 内部同尺寸 resize 成为 identity，不再改变布局。

---

# 6. 4K 重叠 tile 构造

固定第一版配置：

```text
canonical image = 4096
tile size       = 1024
tile stride     = 512
tile count      = 49
```

起点只能是：

```python
starts = list(range(0, 4096 - 1024 + 1, 512))
```

即：

```text
0, 512, 1024, 1536, 2048, 2560, 3072
```

所有 tile：

```text
RGB
1024×1024
无尾部 sliver
无额外 padding
```

不得继续使用：

```python
range(0, image_extent, stride)
```

产生超出 canonical image 的尾部 tile。

---

# 7. 3D token 到重叠 image tile 的分配

## 7.1 先投影全部 global coords

输入：

```text
global_coords [N, 4]
xyz ∈ [0,127]
```

使用当前 Pixal3D 相机投影得到：

```text
global_uv [N, 2]
```

投影使用 global 128-grid coord，不能使用后续的 local 64-grid coord。

---

## 7.2 Assignment 坐标与 sampling 坐标分开

对于 tile membership：

```python
assignment_uv = raw_uv.clamp(
    minimum_image_edge,
    maximum_image_edge,
)
```

原因：

* 所有有限 token 必须至少属于一个 tile；
* 原始 Pixal3D feature sampling 使用 border padding 语义；
* image 外部投影应映射到边界 tile，而不是变成 uncovered。

但是 tile-local feature sampling 仍应使用原始投影语义，由现有 `projection_crop_box` 和 `grid_sample` border 行为处理。

要求：

```python
assert torch.isfinite(raw_uv).all()
assert torch.isfinite(assignment_uv).all()
```

非有限值直接报错。

---

## 7.3 半开区间规则

Tile 覆盖使用：

```text
[x0, x1)
[y0, y1)
```

最右侧和最下侧 image boundary 必须归入最后一列/行 tile。

每个 token：

```text
membership count >= 1
membership count <= 4
```

不允许根据以下信息过滤 token：

```text
foreground mask
foreground ratio
projection valid mask
tile 是否几乎全黑
```

前景 mask只用于日志和 debug，不用于 token 覆盖。

---

# 8. Tile 权重

每个 token 在每个覆盖 tile 中有局部归一化坐标：

```text
x, y ∈ [0,1]
```

使用二维 tent：

[
w_x = 1-|2x-1|
]

[
w_y = 1-|2y-1|
]

[
w = w_xw_y
]

为避免 image 外边界或精确 tile 边界产生全零：

```python
w = w.clamp_min(1e-3)
```

再对每个 token 的所有 tile 权重归一化：

[
\alpha_{ij}
===========

\frac{w_{ij}}{\sum_m w_{im}}
]

硬断言：

```python
valid_count = (tile_ids >= 0).sum(dim=1)

assert torch.all(valid_count >= 1)
assert torch.all(valid_count <= 4)
assert torch.isfinite(tile_weights).all()
assert torch.all(tile_weights >= 0)
assert torch.allclose(
    tile_weights.sum(dim=1),
    torch.ones(N),
    atol=1e-6,
    rtol=1e-6,
)
```

无效 slot：

```text
tile_id = -1
weight  = 0
```

同一 token 中不得出现重复 tile ID。

---

# 9. 每个 tile 独立提取 DINO 与 NAF

对每个有 token membership 的 tile：

```python
tile_condition = get_proj_cond_shape(
    image_cond_model=image_cond_model_tex_1024,
    image=[tile_image],
    coords=global_coords[tile_global_indices],
    grid_resolution_override=128,
    projection_crop_box=tile_crop_box_normalized,
    ...
)
```

必须得到：

```text
tile global:
    [1, 5, 1024]

tile proj:
    [Kt, 2048]
```

其中 `tile proj` 与该 tile 的 `tile_global_indices` 行序严格一致。

每个 tile 的：

```text
global
DINO LR proj
NAF HR proj
```

都必须来自同一个 1024 crop。

不得引用：

```text
full-image DINO global
pipeline_global_1024 global
其他 tile 的 global
多个 tile global 的 raw 平均
```

---

# 10. 全局 multi-tile condition 表示

建议建立明确的数据结构，例如：

```python
@dataclass
class MultiTileProjCondition:
    global_bank: torch.Tensor
    proj: SparseTensor
    tile_ids: torch.Tensor
    tile_weights: torch.Tensor
```

或兼容当前字典接口：

```python
{
    "mode": "multi_tile_paired",
    "global_bank": Tensor[T, 5, 1024],
    "proj": SparseTensor(
        feats=Tensor[N, 2048],
        coords=Tensor[N, 4],
    ),
    "tile_ids": LongTensor[N, 4],
    "tile_weights": Tensor[N, 4],
}
```

要求：

```text
global_bank:
    [T,5,1024]

tile_ids:
    [N,4]

tile_weights:
    [N,4]

proj.feats:
    [N,2048]

proj.coords:
    [N,4]
```

---

# 11. Proj 的融合方式

对同一 token 的各 tile projected feature：

```text
P_i1
P_i2
...
```

使用与 global branch 完全相同的归一化权重：

[
\bar P_i
========

\sum_j \alpha_{ij}P_{ij}
]

得到：

```text
fused_proj [N,2048]
```

可以在进入 Flow 前融合，因为每个 block 的 `proj_linear` 是线性的，且每个 token 的权重和为 1：

[
W\left(\sum_j\alpha_jP_j\right)+b
=================================

\sum_j\alpha_j(WP_j+b)
]

必须添加单元测试验证该等价关系，包括 Linear bias。

仍建议在 CPU debug/trace 中保存每个 token 的 raw slot proj，至少可选保存：

```text
slot_proj [N,4,2048]
```

正式 Flow 不要求长期把该大张量留在 GPU。

---

# 12. Global 不能在输入前直接平均

禁止：

```python
global_avg = sum(
    weight * tile_global
)
```

再调用一次 cross-attention。

原因：

```text
CrossAttention(hidden, average(global))
```

不等于：

```text
average(
    CrossAttention(hidden, global_j)
)
```

必须在每个 Transformer block 中：

1. 分别对每个 tile global 做 cross-attention；
2. 再按照 token 的 tile 权重融合 cross-attention 输出。

---

# 13. 修改 SLatFlowModel 与 SparseProjectAttention

主要检查：

```text
pixal3d/models/structured_latent_flow.py
pixal3d/modules/sparse/attention/proj_attention.py
pixal3d/modules/sparse/attention/modules.py
pixal3d/modules/sparse/transformer/...
```

## 13.1 保持 legacy path 完全兼容

旧条件：

```python
{
    "global": Tensor[B,5,1024],
    "proj": SparseTensor,
}
```

必须保持原行为和数值结果。

新 multi-tile path 只能在：

```text
mode == "multi_tile_paired"
```

或检测到 `global_bank/tile_ids/tile_weights` 时启用。

不要改变 checkpoint 参数名称、shape 或权重。

---

## 13.2 Multi-tile global cross-attention

对于当前 block hidden：

```text
hidden.feats [K,C]
```

以及：

```text
global_bank  [T,5,1024]
tile_ids     [K,4]
tile_weights [K,4]
```

计算：

[
g_i =
\sum_j
\alpha_{ij}
\operatorname{CrossAttn}(h_i,G_j)
]

第一版必须实现一个容易验证的 reference path。

### Reference 实现建议：按 tile 分组 query

```python
global_sum = zeros([K, C])

for tile_id in unique_valid_tile_ids:
    rows, slots = find rows using tile_id

    query_subset = hidden[rows]
    tile_global = global_bank[tile_id:tile_id+1]

    output_subset = existing_cross_attn(
        query_subset,
        tile_global,
    )

    weights = tile_weights[rows, slots]

    global_sum.index_add_(
        0,
        rows,
        output_subset.feats * weights[:, None],
    )
```

注意：

* 这里只拆 global cross-attention query；
* 不拆 3D self-attention；
* query 之间在 cross-attention 中本来不互相 attention；
* 所以按 tile 分组不会破坏三维 token context；
* 3D context 已经由前面的 full self-attention处理。

硬断言：

```python
assert every valid membership is processed exactly once
assert every token accumulated weight == 1
assert output shape == hidden.feats.shape
assert all finite
```

---

## 13.3 性能优化

Reference grouped path 正确后，可实现 vectorized path。

可选方向：

* 将所有有效 `(token row, tile slot)` 展平；
* gather 每个 membership 的 query；
* gather 对应 tile 的 `[5,1024]` context；
* 将每个 membership 视为一个独立 batch；
* 一次 batched attention；
* 按 token row 和权重 scatter 回 `[K,C]`。

必须保留 reference path用于测试。

Optimized path 必须与 reference path满足：

```text
cosine > 0.999999
relative L2 < 1e-5
max abs 在 dtype 合理容差内
```

不要为了优化速度先绕过正确性验证。

---

## 13.4 Image condition 输出

每个 block：

```python
global_out = multi_tile_global_cross_attention(...)
proj_out = self.proj_linear(fused_proj.feats)

combined = global_out + proj_out
```

然后按当前模型原有残差、AdaLN、gate、MLP 顺序继续。

不得改变 self-attention、MLP 或模型权重。

---

# 14. CFG Negative Condition

Positive condition：

```text
global_bank = tile DINO globals
proj        = fused tile DINO/NAF projected features
tile_ids    = membership
weights     = membership weights
```

Negative condition：

```text
global_bank = zeros_like(positive global_bank)
proj        = zeros_like(positive fused proj)
tile_ids    = 与 positive 相同
weights     = 与 positive 相同
```

不得对 negative condition重新计算 membership。

CFG 的 conditional/unconditional 两次模型调用必须使用完全相同的：

```text
coords
tile_ids
tile_weights
```

只有 feature tensor 为零。

---

# 15. Texture Flow 按 3D 坐标切 patch

全局 grid：

```text
128³
```

固定：

```text
patch_size   = 64
patch_stride = 32
patch_starts = [0,32,64]
patch_count  = 27
```

优先复用当前 shape patch flow 已验证的 3D patch builder。

每个 patch 选择：

```python
mask = (
    (xyz[:, 0] >= sx)
    & (xyz[:, 0] < sx + 64)
    & (xyz[:, 1] >= sy)
    & (xyz[:, 1] < sy + 64)
    & (xyz[:, 2] >= sz)
    & (xyz[:, 2] < sz + 64)
)
```

禁止按 image tile 切 Flow token。

---

# 16. Local 64-grid coordinates

对于 patch：

```python
global_indices = mask.nonzero().flatten()

local_coords = global_coords[global_indices].clone()
local_coords[:, 1] -= sx
local_coords[:, 2] -= sy
local_coords[:, 3] -= sz
```

硬断言：

```python
assert local_coords[:, 1:].amin() >= 0
assert local_coords[:, 1:].amax() <= 63
```

构造：

```python
x_patch = SparseTensor(
    feats=x_step_start.feats[global_indices],
    coords=local_coords,
)

shape_patch = SparseTensor(
    feats=shape_concat_cond.feats[global_indices],
    coords=local_coords,
)

proj_patch = SparseTensor(
    feats=full_fused_proj[global_indices],
    coords=local_coords,
)

patch_tile_ids = global_tile_ids[global_indices]
patch_tile_weights = global_tile_weights[global_indices]
```

所有 tensor 的第 0 维必须完全对齐。

图像 projected feature 的计算使用 global coord；送给 Flow 时才替换成 local coord。

---

# 17. 每个 Flow step 的正确执行方式

每个 step 开始：

```python
x_step_start = x_global
```

所有 27 个 patch 必须读取同一份 `x_step_start`。

禁止：

```text
patch 0 更新 x_global
patch 1 再读更新后的 x_global
```

对每个 patch：

```python
patch_cond = {
    "mode": "multi_tile_paired",
    "global_bank": tile_global_bank,
    "proj": proj_patch,
    "tile_ids": patch_tile_ids,
    "tile_weights": patch_tile_weights,
}

patch_neg_cond = {
    "mode": "multi_tile_paired",
    "global_bank": zeros_like(tile_global_bank),
    "proj": zero_proj_patch,
    "tile_ids": patch_tile_ids,
    "tile_weights": patch_tile_weights,
}
```

调用现有 Texture sampler/CFG prediction：

```python
_, _, patch_velocity = sampler._get_model_prediction(
    flow_model,
    x_patch,
    mapped_t,
    patch_cond,
    neg_cond=patch_neg_cond,
    concat_cond=shape_patch,
    **prediction_kwargs,
)
```

不得在本任务中加入：

```text
HiWave
Haar guidance
skip residual
新的 CFG strength
额外 velocity correction
```

本任务只测试 multi-tile condition 与 3D patch flow。

---

# 18. 3D patch velocity 融合

复用 shape patch flow 的三维 overlap window。

建议：

```python
patch_weight = wx * wy * wz
patch_weight = patch_weight.clamp_min(eps)
```

scatter：

```python
velocity_sum.index_add_(
    0,
    global_indices,
    patch_velocity.feats * patch_weight[:, None],
)

velocity_weight_sum.index_add_(
    0,
    global_indices,
    patch_weight[:, None],
)

coverage_count.index_add_(
    0,
    global_indices,
    ones,
)
```

全部 patch完成后：

```python
assert torch.all(coverage_count >= 1)
assert torch.all(velocity_weight_sum > 0)
assert torch.isfinite(velocity_sum).all()
assert torch.isfinite(velocity_weight_sum).all()

merged_velocity = (
    velocity_sum / velocity_weight_sum
)
```

对于当前配置，预期：

```text
coverage min >= 1
coverage max <= 8
```

打印 coverage histogram。

---

# 19. 严禁 Velocity Fallback

新路径中不得存在：

```text
saved_global fallback
current_global fallback
global velocity for uncovered tokens
torch.where(covered, local, global)
```

Global velocity只允许用于诊断：

```text
cosine
relative L2
MSE
norm ratio
```

不能参与最终 velocity。

如果任何 token：

```text
没有 image tile condition
没有 3D patch velocity
weight sum 为 0
存在 NaN/Inf
行序不匹配
```

立即 `RuntimeError`，并输出：

```text
token global index
global coord
projected UV
tile memberships
tile weights
3D patch memberships
```

---

# 20. 每个 step 只更新一次全局 latent

全部 27 个 patch velocity融合后：

```python
x_global = x_global.replace(
    x_step_start.feats
    - dt * merged_velocity
)
```

每个 step 只能有一次 global Euler update。

必须使用 global trajectory 保存的准确：

```text
state[start_step]
mapped_t
mapped_t_next
dt
```

不要重新近似 timestep。

---

# 21. Global Texture trajectory

保持现有逻辑：

1. 先运行完整 12-step global texture flow；
2. 保存 13 个 states；
3. 保存 12 个 global velocities；
4. 从指定 `start_step` 的 state 恢复；
5. 执行 `start_step..11` 的 multi-tile 3D patch flow；
6. Global velocity只作为诊断。

当：

```text
start_step = 12
```

必须：

* 直接返回 global final state；
* 不运行 tile DINO/NAF；
* 不构建 multi-tile condition；
* 不运行 3D patch Flow；
* 最终 latent 与 `global_flow.states[12]` 完全一致。

---

# 22. 推荐 CLI

新增清晰参数，避免继续复用旧 image-tile-flow 名称：

```text
--texture-multitile-3d-patch-flow
--no-texture-multitile-3d-patch-flow

--texture-multitile-start-step

--texture-canonical-image-size
--texture-image-tile-size
--texture-image-tile-stride

--texture-3d-patch-size
--texture-3d-patch-stride

--texture-multitile-global-mode
--texture-multitile-save-debug
```

第一版固定或默认：

```text
texture_canonical_image_size = 4096
texture_image_tile_size      = 1024
texture_image_tile_stride    = 512

texture_3d_patch_size        = 64
texture_3d_patch_stride      = 32

texture_multitile_global_mode = paired_block_fusion
```

不得提供或保留新路径的：

```text
--hr-image-tile-fallback
```

旧：

```text
--hr-image-tile-texture-flow
```

可以保留作为 legacy experiment，但：

* 不得默认启用；
* 必须在 help 中注明 legacy 2D-token-subset flow；
* 新旧实验 key 和 cache identity 必须区分。

---

# 23. Resume/cache identity

实验 identity、trace metadata 和 resume key 必须包含：

```text
canonical preprocessing version
canonical image size
image tile size
image tile stride
tile count
multi-tile fusion mode
3D patch size
3D patch stride
texture start step
condition format version
```

避免错误复用旧：

```text
image-tile-token-subset flow trace
global clean trace
不同 preprocessing 版本
```

---

# 24. Debug 与日志

## 24.1 Preprocessing

```text
[canonical-preprocess]
source=...
alpha_source=rgba|rembg
rembg_calls=0|1
rembg_input=...
foreground_bbox_source=...
square_extent_source=...
padding=left,right,top,bottom
image_4096=4096x4096
image_1024=1024x1024
image_512=512x512
```

---

## 24.2 2D image tiles

```text
[texture-image-tiles]
canonical=4096x4096
tile=1024
stride=512
count=49
active_tiles=...
tokens=N
membership_min=1
membership_max=4
membership_histogram=...
weight_sum_min=...
weight_sum_max=...
```

每个 tile：

```text
[texture-tile-cond]
tile=...
box=(x0,y0,x1,y1)
tokens=...
foreground_ratio=...
global_shape=(1,5,1024)
proj_shape=(Kt,2048)
seconds=...
```

---

## 24.3 3D patches

```text
[texture-3d-patches]
grid=128
patch=64
stride=32
count=27
tokens=N
coverage_min=1
coverage_max=8
coverage_histogram=...
```

每个 patch：

```text
[texture-3d-patch]
patch=...
start=(sx,sy,sz)
tokens=K
local_min=0
local_max=63
image_tiles_used=...
membership_histogram=...
```

---

## 24.4 每个 Flow step

```text
[texture-multitile-3d-flow]
step=...
raw_t=...->...
mapped_t=...->...
patches=27
tokens=N
image_membership_min=1
image_membership_max=4
velocity_coverage=1.000000
velocity_coverage_min=1
velocity_coverage_max=8
cos_vs_global=...
rel_l2_vs_global=...
mse_vs_global=...
seconds=...
```

新路径日志中绝不允许出现：

```text
uncovered > 0
fallback=saved_global
fallback=current_global
```

---

# 25. Debug 产物

建议输出：

```text
canonical_preprocess/
├── source.png
├── source_alpha.png
├── source_square_rgba.png
├── source_square_black_rgb.png
├── image_4096.png
├── image_1024.png
├── image_512.png
└── metadata.json

texture_multitile/
├── tile_000.png
├── ...
├── tile_048.png
├── tile_metadata.json
├── tile_global_bank.pt
├── token_tile_ids.pt
├── token_tile_weights.pt
├── fused_proj.pt
├── optional_slot_proj.pt
└── token_projection_debug.png

texture_3d_patch/
├── patch_metadata.json
├── coverage_count.pt
├── per_step_metrics.json
└── final_trace.pt
```

---

# 26. 必须实现的测试

## Test 1：Single preprocessing

验证：

* 有 alpha 时 rembg 调用次数为 0；
* 无 alpha 时 rembg 调用次数为 1；
* bbox 正确映射回 source；
* RGB 来自 source，不来自 1024 proxy；
* padding 后为正方形；
* 透明区域与黑色正确合成；
* 输出精确为 4096、1024、512；
* 1024 和 512 都直接从 4096 resize；
* 后续 pipeline 不再次 preprocessing。

---

## Test 2：Tile layout

验证：

```text
4096 image
1024 tile
512 stride
49 tiles
```

所有 box 位于 4096 内，无尾部 tile。

---

## Test 3：Tile membership coverage

使用 synthetic UV：

* 角落；
* image 边界；
* tile 边界；
* overlap 中心；
* image 外投影；
* 正常内部投影。

验证：

```text
membership count 1..4
无重复 tile id
权重有限且非负
每个 token 权重和为1
无 uncovered
```

---

## Test 4：Global/proj pairing

给每个 tile 的 global 和 proj 编入唯一 tile ID。

验证每个 token slot：

```text
tile_ids[i,j]
global bank 来源
proj 来源
```

严格一致。

任何交叉错配必须使测试失败。

---

## Test 5：Legacy shared-global identity

构造：

```text
所有 token 只属于 tile 0
weight = 1
tile 0 global = legacy global
fused proj = legacy proj
```

比较：

```text
旧 SparseProjectAttention
新 multi-tile SparseProjectAttention
```

要求：

```text
cosine > 0.999999
relative L2 < 1e-5
max abs 在 dtype 容差内
```

---

## Test 6：Duplicate identical context identity

给每个 token 两份完全相同条件：

```text
global_0 == global_1
proj_0   == proj_1
weights  = [0.3,0.7]
```

结果必须等于单条件模式。

这能验证：

* global output 融合；
* proj bias；
* 权重归一化；
* scatter；
* membership 行对齐。

---

## Test 7：Proj linear fusion identity

随机生成：

```text
P [K,M,2048]
weights [K,M]
Linear(2048,C), including bias
```

验证：

```text
Linear(sum(weight * P))
```

与：

```text
sum(weight * Linear(P))
```

一致。

---

## Test 8：Multi-global grouped reference

实现一个慢速逐 token/逐 membership reference。

比较 grouped-by-tile 实现：

```text
output cosine > 0.999999
relative L2 < 1e-5
```

如果增加 vectorized path，还要分别与 reference 对比。

---

## Test 9：Negative CFG condition

验证：

* negative global bank 全零；
* negative proj 全零；
* tile IDs 和 weights 与 positive 完全相同；
* CFG 调用不修改 membership。

---

## Test 10：3D patch coverage

在 synthetic 128-grid 上验证：

```text
27 patches
local coords 0..63
所有 token coverage >=1
coverage max <=8
无 fallback
```

---

## Test 11：Sparse row alignment

给全局 token feature 写入 global row index。

对每个 patch 验证：

```text
x_patch
shape_patch
proj_patch
tile_ids
tile_weights
local_coords
```

第 0 维始终对应相同 global token。

---

## Test 12：Velocity merge

构造可预测 patch velocity：

* overlap merge 与手工计算一致；
* patch 遍历顺序反转后结果一致；
* 每个 step 只执行一次 global update；
* 人为破坏 coverage 后直接 `RuntimeError`。

---

## Test 13：start_step=12 identity

启用新 flag，但：

```text
start_step=12
```

验证：

```text
final latent == global states[12]
tile DINO call count == 0
NAF call count == 0
3D patch flow call count == 0
```

---

## Test 14：旧路径兼容

新 flag关闭时：

* clean global 输出不变；
* legacy `cond["global"] + cond["proj"]` 行为不变；
* Shape Flow 不受 multi-tile condition 修改影响；
* checkpoint 可正常加载；
* 不新增 trainable 参数。

---

# 27. 实验顺序

## Experiment A：新 preprocessing + clean global

目的：

```text
只验证 canonical preprocessing 的影响
不执行 multi-tile 3D patch flow
```

命令应类似：

```bash
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
python pixal3d_directory_texture_eval.py \
  --image-dir assets/choose \
  --output-dir outputs/pixal3d_2048_canonical_global_control \
  --resolutions 2048 \
  --seeds 42 \
  --lights studio \
  --shape-start-step 12 \
  --texture-mode global_original \
  --no-texture-multitile-3d-patch-flow \
  --render-resolution 2048 \
  --metric-resolution 1024 \
  --fail-fast
```

Codex 必须根据最终 CLI 给出准确命令。

比较现有 baseline：

```text
17.1593 / 0.666374 / 0.218942
```

---

## Experiment B：Multi-tile paired fusion，step 6

首次正式实验：

```bash
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
python pixal3d_directory_texture_eval.py \
  --image-dir assets/choose \
  --output-dir outputs/pixal3d_2048_multitile_paired_3dpatch_step6 \
  --resolutions 2048 \
  --seeds 42 \
  --lights studio \
  --shape-start-step 12 \
  --texture-mode global_original \
  --texture-multitile-3d-patch-flow \
  --texture-multitile-start-step 6 \
  --texture-canonical-image-size 4096 \
  --texture-image-tile-size 1024 \
  --texture-image-tile-stride 512 \
  --texture-3d-patch-size 64 \
  --texture-3d-patch-stride 32 \
  --texture-multitile-global-mode paired_block_fusion \
  --texture-multitile-save-debug \
  --render-resolution 2048 \
  --metric-resolution 1024 \
  --fail-fast
```

如果最终参数名变化，最终报告中必须给出能直接运行的准确版本。

---

## Experiment C：保守 step 10

如果 Experiment B 明显偏离 global trajectory，再运行：

```text
start_step = 10
```

用于判断：

* paired multi-tile condition 是否只适合最后两步细节修正；
* 还是整个条件形式本身仍然有问题。

---

# 28. 指标与可视化

除了全图：

```text
PSNR
SSIM
LPIPS
```

尽可能增加：

```text
头部 ROI
龟壳 ROI
tile boundary band
非 boundary region
```

重点观察：

* 乌龟头部纹理是否恢复；
* 眼睛、嘴、鳞片是否更清楚；
* tile 交界是否有颜色或纹理不连续；
* 远离可见表面的区域是否异常着色；
* 全局颜色是否漂移；
* Texture latent 是否逐步偏离 global trajectory。

---

# 29. 实验记录

创建或更新：

```text
EXPERIMENT_LOG.md
```

每次实验格式：

```markdown
## Experiment XXX

### Goal
本次实验只验证什么。

### Code
commit/hash 或工作区状态。
修改文件。

### Configuration
完整命令和关键参数。

### Preprocessing
source、bbox、padding、4096/1024/512尺寸、rembg次数。

### 2D Condition Diagnostics
tile数量、membership histogram、权重范围、global bank shape、proj shape。

### 3D Flow Diagnostics
patch数量、local coord范围、velocity coverage、每步cosine和relative L2。

### Metrics
PSNR、SSIM、LPIPS、ROI指标。

### Visual Findings
头部、龟壳、边界、颜色、噪声和伪影。

### Conclusion
该实验支持或否定了什么。

### Next Action
下一步明确修改或实验。
```

不得只记录“成功运行”。

---

# 30. 完成标准

任务只有同时满足以下条件才算完成：

* 图像 preprocessing 只执行一次；
* 最终 RGB 来自源分辨率图像；
* canonical 4096、1024、512来自同一个正方形；
* 4K 使用 1024 tile、stride 512、共 49 块；
* 每个 tile 独立运行 DINO 和 NAF；
* 每个 token保留 1～4 个重叠 tile membership；
* 每个 `(global, proj)` 严格来自同一个 tile；
* Proj 使用相同权重提前融合；
* Global 在每个 Flow block 的 cross-attention 输出层融合；
* 不直接平均 raw global；
* 不为每个 tile 重跑整个 Flow；
* 完整 64³ patch self-attention只运行一次；
* Flow token 按 3D coord 切分；
* coords 平移到 `[0,63]³`；
* 27 个 3D patch、stride 32；
* 每个 Flow step统计所有 patch velocity后只更新一次；
* image condition覆盖所有 token；
* velocity覆盖所有 token；
* 不存在任何 global velocity fallback；
* 任何未覆盖、NaN、Inf、错位立即报错；
* legacy shared-global path 数值兼容；
* Shape Flow 不受影响；
* identity tests 全部通过；
* 更新实验记录；
* 给出准确运行命令；
* 最终报告已知性能瓶颈和风险。

---

# 31. 已知风险

最终报告必须明确讨论：

1. 训练时所有 token共享一组完整图 global；现在 global 变为 token-dependent multi-crop context，属于 inference-time condition distribution shift。
2. 不同 tile global 经过每个 block 的 cross-attention 后融合，虽然保持了 `(global,proj)` 成对一致，但模型没有在该形式上训练。
3. 49 个 tile 的 DINO/NAF 成本较高。
4. 每个 block 按 tile 分组 global cross-attention 可能产生大量小 kernel，需要后续 vectorization。
5. 同一条相机射线上的前后体素仍可能获得相似图像条件，这是 Pixal3D 投影机制本身的性质，本任务暂不增加 visibility/depth gating。
6. 本任务首先验证局部 crop 的 global/proj 是否能在保留完整 3D self-attention 时恢复高清纹理，不同时加入其他超分或 guidance 方法。

---

# 32. 最终回复格式

完成后必须回复：

```markdown
# Implementation Summary

## Modified Files
...

## Canonical Preprocessing
...

## Multi-Tile Condition Representation
...

## Flow Block Fusion
...

## 3D Patch Flow
...

## No-Fallback Guarantees
...

## Tests
测试名称、命令、结果。

## Experiments
完整命令、指标、关键日志。

## Comparison
与 clean global 和旧 image-tile flow 对比。

## Known Risks
...

## Next Recommended Experiment
...
```

不要只回复代码 diff 摘要。必须明确说明是否真正运行了测试和完整实验；没有运行的部分必须如实说明。
