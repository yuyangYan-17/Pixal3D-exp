请基于当前 Pixal3D / TRELLIS.2 开源代码，设计并实现一个 training-free 的最小验证实验，测试以下假设：

**C256 高分辨率 support 分块 flow 效果差的核心原因之一，是每个 local C64-sized tile 只能做块内 self-attention，丢失了原始 baseline C64 flow 中完整物体范围的 self-attention 全局纠错能力。尤其在 texture flow 中，逐点 proj condition 一旦给某些点注入错误或片面的颜色/材质信息，分块后没有跨块 self-attention 把它纠正回来。**

实验目标不是直接把 C64 attention 替换到 C256，也不是直接给整个 C256 做 full attention，而是维护两条同步分支：

1. **Global C64 branch**
   - 完全按照 Pixal3D 原始 baseline 跑完整 C64 flow。
   - 不改 support、不改坐标、不改 RoPE、不改模型权重。
   - 在指定 timestep / Transformer block 中提取原生 self-attention 的 `Q64 / K64`。
   - 这条 branch 负责提供可信的整物体 global communication topology。

2. **Fine C256 tiled branch**
   - 使用现有 C256 support。
   - 按当前项目已有方法切成多个 local C64-sized spatial tiles。
   - 每个 tile 内坐标仍重新映射到 `0~63`，保持 Pixal3D 原生 local RoPE 输入分布。
   - 每个 tile 继续使用原始 pretrained flow 网络。
   - 每个 fine point 除 local coordinate 外，额外维护其唯一的 global C256 coordinate，以及映射到 baseline C64 canonical support 的 coarse parent / global correspondence。
   - 禁止直接用不同 tile 的 local `0~63` RoPE 做跨块 QK，因为不同 tile 的相同 local coordinate 并不表示相同 global position。

第一阶段优先只测试 **texture flow**，固定 shape support 和 shape latent，先把变量减少到最少。如果 texture 验证成立，再扩展到 shape flow。

---

## 一、先确认并记录当前网络结构

请先阅读实际源码，确认以下位置和张量尺寸，不要凭猜测：

- `structured_latent_flow.py`
- sparse transformer block
- sparse self-attention
- RoPE
- texture concat shape latent 的位置
- flow sampler / scheduler
- 当前项目中 C256 tiled flow 的实现

打印并记录一个正常 baseline C64 texture flow 中：

- timestep
- block id
- hidden shape
- Q shape
- K shape
- V shape
- attention output shape
- coords min/max
- local/global support point 数

预期 self-attention 大致是：

```text
H       : [N, 1536]
Q/K/V   : [N, 12, 128]
```

但必须以实际代码为准。

---

## 二、建立 Global C64 ↔ Global C256 的唯一 correspondence

不要简单默认 `fine_coord // 4` 一定正确。

根据当前项目实际 C64/C256 support 的生成、upsample、voxelize/downsample 方式，建立并验证：

```text
fine C256 point
    ↓
对应哪个 baseline C64 coarse region / coarse point
```

需要输出以下统计：

- C256 总点数
- C64 总点数
- 每个 fine point 是否唯一对应一个 coarse parent
- 有多少 C64 parent 没有 fine child
- 每个 parent 的 child 数量分布：min / median / mean / max
- 是否存在重复 fine point
- 是否存在 tile overlap 导致同一 global fine point 多次出现

所有后续 global aggregation 都必须在 **unique global C256 support** 上完成，不能让 overlap tile 中同一个点重复计数。

定义：

```text
R: Global C256 -> C64 aggregation
P: C64 -> Global C256 broadcast
```

第一版先使用 parent 内简单 mean aggregation，但代码结构要方便后续换成其他 aggregation。

---

## 三、先做最小实验：只在一个 timestep、一个 Transformer block 注入 global communication

不要一开始改 30 层 × 12 timestep。

先选择：

- 一个中间 timestep，例如 12-step sampler 的中间 step；
- 一个 Transformer block，例如 block 20 或 block 30。

同时跑：

### A. Global C64 baseline branch

正常进入该 timestep / block，得到：

```text
Q64_base
K64_base
V64_base
```

Q/K 必须是**实际参与 self-attention 前的最终 Q/K**，即如果源码中有：

```text
to_qkv
→ q/k normalization
→ RoPE
→ attention
```

则记录 RoPE 后、真正送入 attention kernel 的 Q/K。

不要只 hook `to_qkv` 原始输出。

### B. C256 tiled branch

所有 tile 必须同步运行到同一个 timestep、同一个 Transformer block。

不能：

```text
tile1 跑完整 30 层
tile2 再跑完整 30 层
```

至少在实验指定 block，需要：

```text
所有 tile 先运行到该 block
→ 收集所有 tile 的当前 V
→ 做 global synchronization
→ 再继续该 block 后续部分
```

每个 tile 正常计算：

```text
Q_local
K_local
V_local
Y_local = local self-attention output
```

这里继续使用该 tile 的 local `0~63` RoPE。

---

## 四、把当前 C256 的 V 聚合到 C64

把所有 tile 的当前 `V_local` 按 global point ID 写回唯一 Global C256 support：

```text
V256_global
```

然后使用 correspondence 聚合：

```text
V64_from_fine = R(V256_global)
```

注意：

```text
V64_from_fine
```

的位置粒度是 C64，但内容来自当前 C256 tiled branch。

然后使用 baseline C64 的通信关系：

```text
A64_base = softmax(Q64_base @ K64_base^T / sqrt(d))
Y64_global = A64_base @ V64_from_fine
```

如果直接显式构建完整 `A64_base` 显存过大，可以复用当前 attention kernel，输入：

```text
Q = Q64_base
K = K64_base
V = V64_from_fine
```

得到：

```text
Y64_global
```

这里不要使用 baseline 自己的 `V64_base`，因为实验目标是：

**baseline 提供通信关系，当前 C256 提供被搬运的信息。**

---

## 五、把 global message 广播回 C256

通过 correspondence：

```text
Y256_global = P(Y64_global)
```

这样每个 C256 point 都拿到自己 coarse parent 对应的 global message。

不要直接简单：

```text
Y_final = Y_local + lambda * Y256_global
```

第一版测试以下两个版本。

### Variant 1：直接 Global Replace

完全使用：

```text
Y_final = Y256_global
```

目的是观察“完全由 baseline global communication 控制”会发生什么。

这个版本主要作为诊断，不一定是最终方法。

### Variant 2：Global coarse + Local residual

推荐重点测试：

```text
Y_local_coarse = P(R(Y_local))
Y_local_residual = Y_local - Y_local_coarse

Y_final = Y256_global + Y_local_residual
```

即：

```text
Y_final
=
global coarse message
+
local fine residual
```

数学形式：

```text
Y_final = P(A64_base R(V256)) + (I - PR)Y_local
```

它的目标是：

- coarse / 跨区域 communication：由完整 baseline C64 决定；
- 同一个 coarse region 内的 fine difference：由 local C256 tile 决定。

这里不要把 `(I-PR)` 直接称为“高频”，报告中称为：

```text
within-parent fine residual
```

或：

```text
fine-scale residual relative to C64 grouping
```

除非后续有额外实验能够证明频率意义。

---

## 六、注意 attention 模块插入的位置

请严格确认 Pixal3D Transformer block 的实际顺序。

目标是替换 / 修改 **self-attention 的 message output**，而不是：

- 修改 proj condition；
- 修改 global image cross-attention；
- 修改 MLP；
- 修改 flow scheduler；
- 修改 pretrained weight。

理想流程是：

```text
hidden
→ norm/modulation
→ Q/K/V
→ local self-attention
→ [这里加入 global synchronization / 替换 self-attention output]
→ 原始 residual / to_out
→ image condition
→ MLP
```

但必须按照实际源码判断 `to_out` 是 attention kernel 内还是外。

如果 `Y_local` 是 head-space：

```text
[N, heads, dim_head]
```

则 global message 也必须在同一空间融合后再经过原始 `to_out`。

如果源码已经返回：

```text
[N, 1536]
```

则需要保持两路处于相同 representation space。

不要混用不同空间的 tensor。

---

## 七、PE / RoPE 约束

必须明确区分两套坐标：

### Local coordinate

每个 tile 内：

```text
0~63
```

只用于 local tile 自己的原生 self-attention / RoPE。

### Global coordinate

每个 C256 point 保留真实 global C256 coordinate，并建立到 C64 baseline support 的 correspondence。

第一版 global communication **不要计算**：

```text
Q256_local @ K64_global
```

也不要：

```text
tile A Q_local @ tile B K_local
```

因为 local RoPE 坐标系不同。

第一版直接使用 baseline C64 已经计算完成的：

```text
Q64_base / K64_base
```

做 global routing，从而完全绕开跨尺度 RoPE 混算问题。

---

## 八、第一阶段实验矩阵

保持：

- 相同输入图像
- 相同 C256 support
- 相同 shape latent
- 相同 texture noise
- 相同 seed
- 相同 scheduler
- 相同 timestep
- 相同 decoder
- 不训练
- 不修改权重

比较：

```text
A. baseline C64 texture flow
B. 原始 C256 tiled texture flow
C. tiled + 单 block Global Replace
D. tiled + 单 block Global coarse + Local residual
```

然后再测试：

```text
E. block 10 / 20 / 30 做同步
F. 每 5 层做一次同步
G. 每层同步
```

先不要同时改 Shape flow。

---

## 九、必须做 attention / message 诊断

不要只看最终 render。

选取几个代表性的 C256 points：

- 正面可见点
- 背面点
- tile 边界点
- proj condition 明显可能错误的点

记录它们对应的 C64 parent。

对于 baseline C64 attention，统计：

```text
outside_tile_attention_ratio
=
该 coarse query 对当前 tile 之外 coarse points 的 attention 总和
```

需要画图 / 保存表格：

- 每层 outside-tile attention ratio
- 每 timestep outside-tile attention ratio
- front / back 的差异
- global message norm
- local message norm
- local residual norm
- global correction norm

重点验证：

**原始 tiled flow 中，一个背面点原本只能拿到本 tile 的 V；加入 global routing 后，它是否真正从其他 tile 对应的 coarse region 获取到了当前 C256 的 V 信息。**

这比单纯 PSNR 更重要。

---

## 十、最终结果输出

保存：

1. baseline C64 render
2. 原始 C256 tiled render
3. Global Replace render
4. Global coarse + Local residual render

保持固定相机，至少保存：

- 输入视角
- 左侧
- 右侧
- 背面
- 顶部或斜后方

Texture 第一阶段重点观察：

- 背面是否仍灰 / 颜色错误
- tile seam
- 同一材质区域跨 tile 是否一致
- front detail 是否被 baseline 粗结果压回去
- 是否出现新的颜色污染

另外输出一个 `report.md`，必须包含：

### 1. 实现位置

具体修改了哪些文件、哪些类、哪些函数。

### 2. 张量维度

给出真实运行时：

```text
Q64/K64
V256
V64_from_fine
Y64_global
Y_local
Y_local_residual
Y_final
```

的 shape。

### 3. correspondence 统计

C64/C256 parent-child mapping 是否严格成立。

### 4. attention 诊断

尤其是 outside-tile attention ratio。

### 5. 定量指标

如果现有项目已经有：

- PSNR
- SSIM
- LPIPS
- seam metric
- front/back metric

则全部输出。

### 6. 明确结论

最后只回答：

```text
Hypothesis:
Does restoring baseline-derived global self-attention routing improve C256 tiled texture flow?

Result:
Supported / Partially supported / Not supported
```

并说明证据。

---

## 十一、实现原则

整个测试阶段严格遵守：

- training-free
- 不改模型权重
- 不训练 LoRA
- 不调 guidance scale
- 不人工 front/back mask
- 不人工可见性 gating
- 不修改 DINO image condition
- 不把 global image token 注入 local proj
- 不修改 decoder
- 不用最终 PBR fusion 掩盖 flow 本身问题
- 不先做经验 lambda 加权
- 不把不同 tile 的 local PE 当 global PE
- 所有实验 seed、noise、scheduler 严格一致

优先实现**最小可证伪实验**。

如果工程上一次完整实现成本过高，先只跑：
- 1 个 object
- texture flow
- 1 个 timestep
- 1 个 block
- 抓取 tensor 并离线计算 `Global Replace` / `Global coarse + Local residual`

先验证张量映射、attention communication 和数值行为，再接入完整 12-step flow。

请先阅读代码并给出实现方案，然后直接实现并运行最小测试，不要只停留在理论分析。