# Pixal3D 4096 输入下的训练外三维超分辨率方案

## 1. 目标与基本结论

输入是一张：

[
I_{4K}\in\mathbb R^{4096\times4096\times3}
]

现有 Pixal3D 正常推理路径使用：

```text
4096 输入
→ resize 到 1024
→ C16
→ SS32
→ Shape512
→ C64
→ Shape1024
→ Texture1024
→ O-Voxel / Mesh
```

同时，将原始 4096 图按：

```text
tile_size = 1024
stride    = 512
```

切成：

[
7\times7=49
]

个 1024 tile。

Pixal3D 官方实现本身采用 Sparse Structure、Shape、Texture 三阶段级联，并逐步提升稀疏结构、几何和纹理分辨率；官方 shape/texture 模型最终训练到 1024 分辨率。这里的设计不修改模型权重，而是利用 exact camera transform，把多个 1024 local generation process 同步到统一 global canonical space。

完整方案的核心不是：

[
\text{average}(V_g,V_l)
]

而是：

[
\boxed{
\text{Global 提供唯一低频主轨迹}
+
\text{Local 提供零低频的高频 residual}
+
\text{Local consensus 提供新 topology}
}
]

整个系统分为四条相互独立但共享空间映射的链路：

1. **空间链路**：global/local token、latent、O-Voxel 精确映射到同一 global 空间；
2. **噪声链路**：global/local 使用同一空间白噪声 realization；
3. **Flow 链路**：同步 clean endpoint residual，不直接平均 raw velocity；
4. **解码链路**：在高分辨率统一 support 上融合 geometry 和 PBR material。

---

# 2. 4096 超分到底意味着什么

## 2.1 XY 获得真实四倍分辨率

一个 1024 tile 只覆盖 4096 图像宽度的四分之一。

你的 exact camera derivation 中：

[
\frac{d_ls_l}{d_gs_g}\approx4
]

因此，一个 local C64 token 映射回 global (q) 空间后，在 (x/y) 方向的 footprint 约为 global C64 token 的四分之一。

所以：

```text
local C64
≈ global C256 的 XY 采样尺度
```

类似地：

```text
local Shape512
≈ global C2048 的 XY 采样尺度

local Shape1024
≈ global C4096 的 XY 采样尺度
```

## 2.2 Z 不会自动获得四倍信息

你的变换严格保持：

[
q_{g,z}=q_{l,z}
]

所以 local C64 在 (z) 方向仍然只有 C64 采样能力；local Shape1024 在 (z) 方向仍然只有 C1024 采样能力。

因此当前方案实际获得的是：

[
\boxed{
XY:\ 1024\rightarrow4096
}
]

[
\boxed{
Z:\ 1024\rightarrow1024
}
]

为了工程统一，可以使用一个逻辑上的稀疏 C4096³ 网格，但 local 证据在 (z) 方向应被视为低通的：一个 local C1024 cell 会覆盖大约四个 C4096 z-cell，不能伪造不存在的 z 高频。

最终表示更准确地写成：

[
4096_x\times4096_y\times1024_z
]

或者使用逻辑 C4096³ sparse grid，并明确约束 z-bandwidth。

---

# 3. 总体生成架构

完整流程如下：

```text
                          ┌──────────────────────────────┐
4096 RGB ────────────────→│ Global image resize to 1024  │
                          └──────────────┬───────────────┘
                                         │
                                         ▼
                              Global Pixal3D baseline
                       C16 → SS32 → Shape512 → C64
                              → Shape1024 → Texture1024
                                         │
                                  Global master state
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
        ▼                                ▼                                ▼
   Tile 0, 1024                     Tile 1, 1024                    Tile 48, 1024
 exact local camera                exact local camera              exact local camera
        │                                │                                │
   Local Pixal3D                     Local Pixal3D                    Local Pixal3D
        │                                │                                │
        └──────── exact local→global canonical synchronization ──────────┘
                                         │
                                         ▼
                              Unified high-resolution state
                           C256 → C2048 → C4096 equivalent
                                         │
                                         ▼
                              Sparse high-resolution O-Voxel
                                         │
                                         ▼
                              Geometry + PBR material fusion
                                         │
                                         ▼
                                  Final global GLB
```

Global 和 local 模型都保持原有 Pixal3D 网络、图像条件和相机条件。

不修改：

* 模型结构；
* 网络权重；
* cross-attention；
* pixel back-projection conditioning；
* 原始 ODE/flow 模型。

新增的是：

* exact support mapping；
* shared spatial noise；
* canonical endpoint synchronization；
* hierarchical topology fusion；
* high-resolution O-Voxel assembly。

---

# 4. 全局与局部相机

对每个 tile (k)，定义 exact map：

[
T_k:
q_l\rightarrow q_g
]

根据已推导公式：

[
T_k(x_l,y_l,z)
==============

\left(
a_x(z)x_l+b_{x,k}(z),
a_y(z)y_l+b_{y,k}(z),
z
\right)
]

其中：

[
a_x(z)=
\frac{s_gD_g(z)f_{l,x}}
{s_lD_l(z)r_xf_{g,x}}
]

[
a_y(z)=
\frac{s_gD_g(z)f_{l,y}}
{s_lD_l(z)r_yf_{g,y}}
]

当前配置近似有：

[
a_x(z)=a_y(z)\approx\frac14
]

但所有实际映射必须使用 (z)-dependent exact formula，不能写成固定除以 4。

该映射用于：

* global support → local support；
* local support → global high-resolution support；
* token footprint 变换；
* noise overlap；
* flow synchronization；
* local O-Voxel → global O-Voxel；
* mesh vertex transform；
* decoded normal transform。

几何点使用 (T_k)，法线使用 inverse-transpose Jacobian：

[
n_g
===

\frac{
J_{T_k}^{-T}n_l
}{
|J_{T_k}^{-T}n_l|
}
]

32 维 latent feature 或 velocity 不做向量旋转；它们是定义在空间位置上的 feature field，不是 XYZ 物理速度。

---

# 5. 多层统一空间

每一个 Pixal3D 分辨率层都建立对应的 global high-resolution canonical space。

| Local 层级          | Local grid | 4096 等效 global XY 层级 |             建议统一空间 |
| ----------------- | ---------: | -------------------: | -----------------: |
| 初始 sparse noise   |        C16 |                  C64 |           (H_{16}) |
| Sparse Structure  |        C32 |                 C128 |           (H_{32}) |
| Shape512          |       C512 |                C2048 |          (H_{512}) |
| Upsampled support |        C64 |                 C256 |           (H_{64}) |
| Shape1024         |      C1024 |                C4096 | (H_{1024}^{shape}) |
| Texture1024       |      C1024 |                C4096 |   (H_{1024}^{tex}) |

为了保持各向同性索引，可以把 (H_R) 表示为逻辑：

[
(4R)\times(4R)\times(4R)
]

但 local cell 在 z 方向应覆盖约四个 logical cell。

更准确的表示是：

[
(4R)\times(4R)\times R
]

本方案推荐使用**自适应公共 cell complex**，而不是实际分配 dense tensor。

---

# 6. 公共空间原子

## 6.1 定义

Global token cell 和所有 mapped local token cell 会在 global (q) 空间形成一组相互交叉的空间划分。

将空间递归细分为一组互不重叠的原子：

[
\mathcal A^{(R)}
================

{A_r^{(R)}}
]

每个原子满足：

1. 完全位于某个 global token cell 内；
2. 对于每个 tile，要么完全位于一个 mapped local token cell 内，要么不在该 tile support 内；
3. 不跨越当前层级任何 token footprint 边界；
4. 原子之间不重叠；
5. 原子的并集覆盖参与生成的有效物体空间。

原子保存：

```text
atom_id
global_parent_id
local_parent_id[tile]
volume
center
bounding_box
active_tile_mask
```

## 6.2 构造方式

精确参考实现：

1. 以 global canonical cube 为根节点；
2. 使用 octree 或 rectilinear subdivision；
3. 对任一 cell，检查它是否被 global/local token 边界切割；
4. 若被切割，则继续细分；
5. 在当前目标层级达到所需空间精度后停止；
6. 仅构造物体 active support 和 tile 投影覆盖附近的节点。

C64 阶段的最小 XY cell 尺寸对齐 global C256；

Shape1024 阶段的最小 XY cell 尺寸对齐 global C4096。

不应构造 dense：

[
4096^3
]

只构造稀疏表面附近和 active token 所涉及的 cell。

## 6.3 交叠矩阵

定义：

[
\Omega_{kji}^{(R)}
==================

\left|
T_k(L_{k,j}^{(R)})
\cap
G_i^{(R)}
\right|
]

也可以通过公共原子表示：

[
\Omega_{kji}^{(R)}
==================

\sum_{
r:
A_r\subset T_k(L_{k,j}),
A_r\subset G_i
}
|A_r|
]

这是噪声、场传递和覆盖率的共同几何基础。

---

# 7. 阶段一：Global baseline

首先只使用 resize 后的 1024 global image：

[
I_g=\operatorname{Resize}(I_{4K},1024)
]

完整运行原始路径：

```text
global C16 noise
→ global SS32
→ global Shape512
→ global C64
→ global Shape1024
→ global Texture1024
→ global O-Voxel
```

保存：

```text
global SS32 support
global C64 support
global Shape512 states / endpoints
global Shape1024 states / endpoints
global Texture1024 states / endpoints
global decoded O-Voxels
global mesh
```

Global baseline 有三个作用：

1. 提供整体 topology；
2. 提供唯一低频 flow 主轨迹；
3. 提供最终几何和材质的 fallback。

Global 分支在后续同步过程中原则上不被 local 修改。

即：

[
x_g^{next}
==========

\operatorname{ODEStep}(x_g,v_g)
]

Local 只影响高分辨率统一状态和 local trajectory。

---

# 8. 阶段二：49 个 tile 与 local 相机

对 4096 输入按：

```text
tile_size = 1024
stride = 512
```

生成 49 个 tile。

每个 tile 保存：

```text
tile_id
crop box
local FOV
local distance
local mesh_scale
exact T_global_to_local
exact T_local_to_global
```

每个 tile 使用自己的 1024 原始 crop：

[
I_k=I_{4K}[B_k]
]

而不是从 global 1024 图再次裁剪。

Tile condition feature、DINO/image encoder feature 和 pixel back-projection 都使用 local camera：

[
(\theta_{l,k},d_{l,k},s_l)
]

不得混用 global camera。

---

# 9. 阶段三：Local support 构造

Local support 分成两部分：

[
S_k=S_k^{anchor}\cup S_k^{candidate}
]

## 9.1 Global anchor support

将 global C32/C64 support 的 token center 和 cell footprint 精确变换到 local：

[
q_l=T_k^{-1}(q_g)
]

然后量化到 local grid。

量化前：

* 不做 clamp；
* 超出 ([-1,1]^3) 的点丢弃；
* 统计 out-of-bounds；
* 多个 global token 映射到同一 local key 时只保留一个 key，但记录来源。

这一部分叫：

[
S_k^{anchor}
]

它强制 local trajectory 包含 global 已确认的主体 topology。

## 9.2 Local-native candidate support

同时让 tile 独立运行正常 SS：

```text
local C16
→ local SS32
→ local C64 candidate
```

得到：

[
S_k^{native}
]

去掉已经存在于 (S_k^{anchor}) 中的 token 后：

[
S_k^{candidate}
===============

S_k^{native}\setminus S_k^{anchor}
]

Candidate 不立即进入最终 topology，只参与 local flow 和后续一致性验证。

## 9.3 为什么必须保留两路

仅使用 projected global support：

* 稳定；
* 与 global 对齐；
* 但无法增加真实新 topology。

仅使用 native local support：

* 能生成局部结构；
* 但与 global topology 差异很大；
* 多 tile 之间也不稳定。

所以正确结构是：

```text
Projected global support = mandatory anchor
Local native support      = topology candidate
```

---

# 10. 阶段四：共享空间噪声

## 10.1 目标

Global 和 local 不能分别调用无关的：

```python
torch.randn(N_global, 32)
torch.randn(N_local, 32)
```

它们描述同一个物理空间，应共享同一个随机 realization。

对于任意 token footprint (D)，定义：

[
\epsilon(D)
===========

\frac1{\sqrt{|D|}}
\int_DdW
]

其中 (W) 是 global canonical space 中定义的 32 通道 Gaussian white-noise measure。

Global token：

[
\epsilon_i^g
============

\frac1{\sqrt{|G_i|}}
\int_{G_i}dW
]

Local token：

[
\epsilon_{k,j}^l
================

\frac1{\sqrt{|T_k(L_{k,j})|}}
\int_{T_k(L_{k,j})}dW
]

因此：

[
\operatorname{Cov}
(\epsilon_{k,j}^l,\epsilon_i^g)
===============================

\frac{
|T_k(L_{k,j})\cap G_i|
}{
\sqrt{
|T_k(L_{k,j})||G_i|
}
}
]

即：

[
\boxed{
C_{kji}^{noise}
===============

\frac{\Omega_{kji}}
{\sqrt{V_{k,j}^lV_i^g}}
}
]

## 10.2 使用公共原子生成

对每个公共原子生成：

[
\xi_r\sim\mathcal N(0,I_{32})
]

使用 stateless key：

```text
(seed, stage, resolution, atom_id, channel)
```

Global token noise：

[
\boxed{
\epsilon_i^g
============

\frac{
\sum_{r\subset G_i}
\sqrt{|A_r|}\xi_r
}{
\sqrt{|G_i|}
}
}
]

Local token noise：

[
\boxed{
\epsilon_{k,j}^l
================

\frac{
\sum_{r\subset T_k(L_{k,j})}
\sqrt{|A_r|}\xi_r
}{
\sqrt{|T_k(L_{k,j})|}
}
}
]

这样：

* 同一物理空间自动共享 noise；
* 到 global token 距离相同但位置不同的 local token 使用不同原子；
* 相同位置、相同 footprint、不同 tile 会得到相同 noise；
* 部分重叠只共享重叠部分；
* 不重叠区域独立；
* 每个 tile 内互不重叠 token 仍是 iid Gaussian；
* 同中心不同 footprint 会共享部分而不完全相同。

## 10.3 各层独立 noise field

不同 flow stage 使用不同 noise namespace：

```text
noise/ss16
noise/shape512
noise/shape1024
noise/texture1024
```

不能让 shape 与 texture 误用相同 noise。

同一 stage 内 global 和所有 local 必须使用同一 seed 和同一 atom field。

Coordinate-aware shared noise 用来给跨视图或跨窗口生成过程建立空间相关性是合理方向，但它只能解决初始随机条件一致性，不能替代后续 canonical synchronization。

---

# 11. 阶段五：SS32 topology 融合

Sparse Structure 是离散 support，不适合用普通 high-pass velocity 直接处理。

它使用 consensus 规则。

## 11.1 将 candidate 映射到 global high-resolution support

每个 local candidate token：

[
q_g=T_k(q_l)
]

映射到统一 (H_{32})：

[
H_{32}\approx C128_{xy}
]

记录：

```text
canonical key
tile id
local confidence
projected pixel
depth
distance to tile border
global anchor distance
```

## 11.2 Topology candidate 接受条件

一个 candidate 被接受为 high-resolution topology，应满足至少一种条件：

### 多 tile 共识

同一 canonical high-resolution cell 被两个或更多重叠 tile 支持：

[
#\text{supporting tiles}\ge2
]

并且它们的 global-space center 和 depth 足够一致。

### Global surface narrow band

candidate 靠近 global anchor surface：

[
d(q_{candidate},S_g)
<
\tau_{\text{surface}}
]

适合细分现有表面。

### 强单 tile 证据

图像边界区域可能只被一个 tile 覆盖，这时要求：

* 高 foreground probability；
* 深度与 global camera projection 一致；
* 与已有 support 连通；
* 不生成独立漂浮组件。

## 11.3 禁止条件

以下 candidate 直接拒绝：

* 变换后超出 global canonical cube；
* 与 global depth ordering 冲突；
* 形成 front/back double shell；
* 只有一个低置信度边缘 tile 支持；
* 与主体没有局部连通关系；
* 多 tile 给出相反表面法线或相差过大的深度。

## 11.4 最终 C64 local support

对 tile (k)，最终 local support：

[
S_k^{flow}
==========

S_k^{anchor}
\cup
S_k^{accepted}
]

在一次 ODE/flow 内 support 固定，不在中间 timestep 随机增删 token。

---

# 12. 阶段六：Flow 同步原则

Flow Matching 学习的是由概率路径定义的时间相关向量场；不同 condition、不同 support、不同状态上的 raw vector field 不能被视为可直接平均的同一个物理速度。

因此禁止：

[
v^{sync}
========

\alpha v_g+(1-\alpha)v_l
]

也禁止：

[
v_g\leftarrow Rv_l
]

本方案同步的是模型预测的 clean endpoint。

这种“多个 instance space 独立去噪，在 canonical space 同步 clean estimate”的策略与 SyncTweedies 的主要结论一致；其工作发现同步 Tweedie/clean outputs 通常比同步 noisy states 更可靠。

---

# 13. 从 velocity 恢复 clean endpoint

设当前概率路径：

[
x_t=\alpha(t)x_0+\beta(t)x_1
]

模型输出：

[
v_t=
\dot\alpha(t)x_0+
\dot\beta(t)x_1
]

定义：

[
\Delta(t)
=========

## \alpha(t)\dot\beta(t)

\dot\alpha(t)\beta(t)
]

则：

[
\boxed{
\hat x_1
========

\frac{
-\dot\alpha(t)x_t+\alpha(t)v_t
}{
\Delta(t)
}
}
]

对于线性 rectified flow：

[
x_t=(1-t)x_0+tx_1
]

则：

[
\boxed{
\hat x_1=x_t+(1-t)v_t
}
]

实现时必须读取 Pixal3D 当前 sampler 的真实时间方向和 path 参数，不能根据变量名猜测。

---

# 14. 每个 flow timestep 的联合过程

对 Shape512、Shape1024 和 Texture1024 分别执行以下同步。

## 14.1 Global 独立预测

[
v_g=F_g(x_g,I_g,t)
]

计算：

[
\hat x_{1,g}
]

Global trajectory 继续使用原始 (v_g) 更新。

## 14.2 每个 tile 独立预测

[
v_{l,k}=F_l(x_{l,k},I_k,t)
]

计算：

[
\hat x_{1,l,k}
]

## 14.3 提升到公共高分辨率原子空间

Global endpoint 在其 cell 内广播：

[
g_r=
\hat x_{1,g,i(r)}
]

Local endpoint 在 mapped local cell 内广播：

[
l_{r,k}
=======

\hat x_{1,l,k,j_k(r)}
]

## 14.4 Local 相对 global residual

[
\boxed{
d_{r,k}=l_{r,k}-g_r
}
]

不能融合 local 绝对 endpoint，因为不同 tile 的绝对颜色、材质和 shape latent 存在系统偏移。

只融合：

[
\text{local}-\text{global}
]

residual。

---

# 15. 多 tile residual 融合

stride 512 使图像内部区域通常被四个 tile 覆盖。

同一物理原子 (A_r) 可能存在多个 residual：

[
d_{r,1},d_{r,2},\ldots
]

先计算每个 tile 的基础权重：

[
w_{r,k}
=======

w_{image}
w_{visibility}
w_{support}
w_{agreement}
]

## 15.1 Tile image window

使用 tile 坐标中的 separable raised-cosine window：

[
w_{image}(u,v)
==============

\sin^2\left(\frac{\pi u}{W_t}\right)
\sin^2\left(\frac{\pi v}{H_t}\right)
]

tile 边缘接近 0，中心接近 1。

这只解决 crop 边缘 condition 不完整的问题，不参与 noise 方差计算。

## 15.2 Visibility

将当前 global surface 或 global depth 投影到 tile。

若原子：

* 被遮挡；
* 在相机背面；
* 超出 tile；
* 与 tile depth 不一致；

则：

[
w_{visibility}=0
]

## 15.3 Support confidence

Anchor support 权重高于 single-tile native candidate。

多 tile 共同支持的 topology 权重高于单 tile support。

## 15.4 Robust merge

不使用普通平均，使用 32 维 weighted geometric median 或 Huber estimator：

[
\bar d_r
========

\operatorname{RobustMerge}
{d_{r,k},w_{r,k}}
]

若 residual 方向互相冲突，例如：

[
\cos(d_{r,k},\bar d_r)<0
]

则拒绝该 tile 的 residual，而不是把相反方向平均成模糊结果。

MultiDiffusion 的核心也是用共享 canonical constraint 将多个局部生成过程绑定起来，而不是在最后直接拼接独立样本。

---

# 16. 高频与低频分解

## 16.1 Global 可表示子空间

对于当前 global resolution (R)，global token 在其 cell 内只能表示常量。

定义 global→atom 广播：

[
(P_gx_g)*r=x*{g,i(r)}
]

定义 atom→global 体积 restriction：

[
(R_gx_H)_i
==========

\frac{
\sum_{r\subset G_i}|A_r|x_{H,r}
}{
|G_i|
}
]

满足：

[
R_gP_g=I
]

因此：

[
\Pi_g=P_gR_g
]

是 global resolution 可表示子空间上的体积正交投影。

## 16.2 低频

[
d^{low}=P_gR_gd
]

即每个 global cell 内 residual 的体积平均。

## 16.3 高频

[
\boxed{
d^{high}
========

(I-P_gR_g)d
}
]

满足：

[
\boxed{
R_gd^{high}=0
}
]

因此 local 高频不会改变 global 低频。

---

# 17. 部分覆盖情况下的高通

实际 tile 只覆盖 global cell 的一部分。

对 global cell (G_i)，设原子 coverage 为：

[
c_r\in[0,1]
]

先计算已覆盖区域 residual 平均：

[
\mu_i^{cov}
===========

\frac{
\sum_{r\subset G_i}
|A_r|c_r\bar d_r
}{
\sum_{r\subset G_i}
|A_r|c_r
}
]

再定义写入的高频：

[
\boxed{
h_r
===

c_r
\left(
\bar d_r-\mu_{i(r)}^{cov}
\right)
}
]

于是：

[
\sum_{r\subset G_i}|A_r|h_r=0
]

即使 local 只覆盖 global cell 的 10%，也不会改变该 global cell 的平均 endpoint。

这是避免 tile 局部颜色、几何偏移污染 global 的关键约束。

---

# 18. 构造统一 high-resolution endpoint

当前 resolution 的统一 endpoint：

[
\boxed{
\hat x_{1,H}
============

P_g\hat x_{1,g}
+
h
}
]

它严格满足：

[
\boxed{
R_g\hat x_{1,H}
===============

\hat x_{1,g}
}
]

因此：

* global endpoint 是唯一低频；
* local 只能改变 global cell 内的分布；
* local detail 降回 global resolution 后严格消失。

这与 HiWave“保留 base low frequency，只从 patch process 获取 high-frequency detail”的思想相似，但这里不能直接使用二维小波，因为 sparse 3D support、相机空间和 topology 均不规则；本方案使用由 global cell partition 定义的精确体积投影代替固定小波。

---

# 19. 将统一 endpoint 投回 local

对 local token (j)：

[
\hat x_{1,l,k,j}^{sync}
=======================

\frac{
\sum_{
r\subset T_k(L_{k,j})
}
|A_r|\hat x_{1,H,r}
}{
|T_k(L_{k,j})|
}
]

然后由当前 path 恢复同步后的 velocity。

一般形式：

[
x_0
===

\frac{
x_t-\beta\hat x_1^{sync}
}{
\alpha
}
]

[
\boxed{
v_t^{sync}
==========

\dot\alpha x_0+
\dot\beta\hat x_1^{sync}
}
]

线性 rectified flow：

[
\boxed{
v_t^{sync}
==========

\frac{
\hat x_1^{sync}-x_t
}{
1-t
}
}
]

Local 使用 (v_t^{sync}) 更新：

[
x_{l,k}^{next}
==============

\operatorname{ODEStep}
(x_{l,k},v_{l,k}^{sync})
]

Global 继续使用原始 (v_g)。

---

# 20. Shape512 阶段

Shape512 的作用是形成中尺度 geometry latent。

统一空间目标约为：

[
H_{512}
\approx
2048_x\times2048_y\times512_z
]

过程：

1. Global Shape512 正常推理；
2. 每个 tile Shape512 正常预测；
3. 使用 shared Shape512 noise；
4. 每步恢复 clean endpoint；
5. local endpoint 映射到 (H_{512})；
6. 多 tile robust residual merge；
7. 对 global Shape512 cell 做零均值 high-pass；
8. 投回 local；
9. 更新 local Shape512 trajectory。

Shape512 结束后：

* global 提供中尺度主体；
* local 提供 2048-equivalent XY detail；
* topology 仍由 anchor + accepted candidate 控制。

---

# 21. C64 support 扩展

从 Shape512/C32 上采样或 expansion 到 C64 时：

## Global

按正常 Pixal3D 路径得到：

[
S_g^{64}
]

## Local

Local C64 support 包含：

[
S_{k}^{64}
==========

\operatorname{Upsample}
(S_k^{anchor}\cup S_k^{accepted})
]

如果 upsample 过程中模型生成新 token，这些 token仍需要：

1. mapped 到 global (H_{64})；
2. 标记 anchor-derived 或 local-born；
3. 进行多 tile topology consistency；
4. 拒绝不一致的孤立 support。

C64 统一空间约为：

[
H_{64}
\approx
256_x\times256_y\times64_z
]

或逻辑 C256³，其中 local cell 在 z 方向跨越约四个 logical cell。

---

# 22. Shape1024 阶段

Shape1024 是最终高分辨率 geometry latent 阶段。

统一空间目标：

[
H_{1024}^{shape}
\approx
4096_x\times4096_y\times1024_z
]

每个 timestep 执行相同的 endpoint synchronization：

[
\hat x_{1,H}^{shape}
====================

P_g\hat x_{1,g}^{shape}
+
Q_g
\operatorname{RobustMerge}
\left(
\hat x_{1,l,k}^{shape}
----------------------

P_{l,k}\hat x_{1,g}^{shape}
\right)
]

其中 (Q_g) 是相对于 global Shape1024 cell 的零均值高通，而不是相对于 C64。

必须逐层使用不同 projector：

```text
Shape512 residual
→ 对 global Shape512 cell 去低频

C64 residual
→ 对 global C64 cell 去低频

Shape1024 residual
→ 对 global Shape1024 cell 去低频
```

不能始终只相对 global C64 去低频，否则 Shape1024 local residual 会破坏 global Shape1024 中已有的中频结构。

---

# 23. 最终 high-resolution geometry support

Shape1024 结束后，将 global 与所有 local support 映射到 (H_{1024}^{shape})。

## 23.1 Global anchor

Global Shape1024 token 对应一个低频 parent cell。

它被广播或细分到 high-resolution children，但不凭空增加 geometry detail。

## 23.2 Local geometry cells

每个 local Shape1024 cell：

[
L_{k,j}^{1024}
]

通过 exact (T_k) 映射到 global。

不能只变换 center 后最近邻量化；应变换：

* center；
* 8 个 cell corners；
* 或使用 exact z-dependent cell boundary。

然后计算它对 global high-resolution cells 的覆盖。

## 23.3 支持融合

最终 support 分三类：

```text
ANCHOR
global baseline 明确存在

DETAIL
local 对 anchor surface 的细分

BIRTH
local 新增的 topology
```

ANCHOR 永久保留。

DETAIL 允许进入最终 support。

BIRTH 只有通过 topology consensus 后才保留。

---

# 24. Geometry latent 融合

对同一 high-resolution atom，不平均 local absolute shape latent。

先计算：

[
d_{r,k}^{shape}
===============

## x_{r,k}^{shape}

x_{r}^{global-base}
]

再 robust merge：

[
d_r^{shape}
===========

\operatorname{RobustMerge}*k
(d*{r,k}^{shape})
]

并执行相对于 global Shape1024 的 high-pass：

[
h_r^{shape}
===========

Q_{1024}^{global}d_r^{shape}
]

最终：

[
\boxed{
x_{r}^{shape-final}
===================

x_r^{global-base}
+
h_r^{shape}
}
]

对于 topology birth cell，没有 global base latent，应使用：

* 多 tile consensus latent；
* 或距离最近的 global surface latent作为 coarse reference；
* 再加入 local residual。

不能将不存在 global anchor 的 token 当成零 latent直接相减，因为零未必是模型 latent 的中性值。

---

# 25. Geometry 解码策略

优先顺序如下。

## 方案 A：统一 latent 后单次解码

Pixal3D decoder 能接受任意 sparse normalized coordinate 和 high-resolution support：

```text
unified H4096 sparse shape latent
→ one decoder
→ one global high-resolution O-Voxel field
```

这是最理想方案，因为所有 tile 在解码前已经统一。

---

# 26. Texture1024 阶段

Texture 必须在最终 geometry support 固定后运行。

输入：

```text
final high-resolution geometry support
global 1024 image condition
49 local tile conditions
```

## 26.1 Global texture master

Global Texture1024 产生：

[
\hat x_{1,g}^{tex}
]

它提供：

* 全局颜色；
* 全局材质风格；
* 不可见区域 fallback；
* tile seam 的低频基准。

## 26.2 Local texture residual

每个 tile 产生：

[
\hat x_{1,l,k}^{tex}
]

在统一 high-resolution atom 上计算：

[
d_{r,k}^{tex}
=============

## \hat x_{1,l,k}^{tex}

\hat x_{1,g}^{tex}
]

多 tile robust merge 后：

[
h_r^{tex}
=========

Q_{global,1024}^{tex}
d_r^{tex}
]

最终：

[
\boxed{
\hat x_{1,H}^{tex}
==================

P_g\hat x_{1,g}^{tex}
+
h_r^{tex}
}
]

## 26.3 不平均绝对材质

不同 local tile 可能对相同位置生成不同 base color 和 roughness。

禁止：

[
m^{final}
=========

\sum_kw_km_k^{local}
]

建议：

[
m^{final}
=========

m^{global}
+
\operatorname{RobustMerge}
(m_k^{local}-m^{global})
]

PBR channel 分开处理：

* base color：在线性 RGB 或 logit space 融合；
* roughness：标量 residual；
* metallic：概率/logit residual；
* normal：先变换到 global frame，再球面平均；
* opacity：以 global geometry occupancy 为硬约束。

---

# 27. 多 tile seam 消除

Seam 不应通过最后对 RGB blur 解决。

需要在三个层面消除。

## Noise 层

重叠 tile 使用同一 canonical spatial noise。

## Flow 层

重叠 tile 的 clean endpoint 在同一原子空间 robust merge。

## Material 层

同一 surface atom 使用唯一 unified material latent，不保留 tile-specific material copy。

Tile window 只用于降低边缘 condition 可信度，不是最终 seam blender。

---

# 28. Topology 与 detail 必须分离

完整生成时维护两个不同变量：

[
\mathcal S_H
]

表示高分辨率 support/topology；

[
x_H
]

表示 support 上的 continuous latent。

Topology 不能通过 velocity average 隐式产生。

建议流程：

```text
SS32:
决定 coarse topology candidates

Shape512:
验证中尺度 candidate

C64 expansion:
建立高分辨率 candidate children

Shape1024:
输出最终 geometry latent

Final consensus:
决定哪些 candidate 真正进入 O-Voxel
```

Candidate 在多个阶段持续不稳定时应删除。

---

# 29. 建议的完整执行顺序

## Pass 1：Global baseline

```text
I4096 → resize I1024
→ global camera
→ global C16
→ global SS32
→ global Shape512
→ global C64
→ global Shape1024
→ global Texture1024
→ baseline GLB
```

## Pass 2：Tile camera 与 support

对 49 个 tile：

```text
crop I4096
→ derive exact centered local camera
→ project global SS32/C64 anchors into local
→ run local native SS
→ build anchor + candidate support
```

## Pass 3：Shared-noise local generation

```text
canonical spatial noise
→ global and all tiles read same realization
```

## Pass 4：Shape512 coupled flow

```text
global endpoint
local endpoints
→ canonical H512 residual
→ robust merge
→ high-pass against global Shape512
→ project back to local
```

## Pass 5：C64 support refinement

```text
anchor upsample
+ accepted local candidates
→ canonical H64 support
→ topology consensus
```

## Pass 6：Shape1024 coupled flow

```text
global endpoint
local endpoints
→ canonical H1024 shape residual
→ robust merge
→ high-pass against global Shape1024
→ project back to local
```

## Pass 7：Final geometry support

```text
global anchors
+ local surface detail
+ accepted local topology birth
→ unified sparse H4096-equivalent support
```

## Pass 8：Texture1024 coupled flow

```text
geometry fixed
→ global texture master
→ local texture residual
→ canonical material merge
→ high-pass against global texture
```

## Pass 9：Decode and mesh

```text
unified high-resolution geometry latent
+ unified material latent
→ O-Voxel
→ mesh
→ global GLB
```

## Pass 10：4096 rendering evaluation

```text
render final model using global 4096 camera
compare against original I4096
and evaluate all 49 tile crops
```

---

# 30. 核心伪代码

```python
# ---------------------------------------------------------
# 1. Global baseline
# ---------------------------------------------------------
global_result = run_global_pixal3d(
    image=resize(image_4096, 1024),
    camera=global_camera,
    seed=seed,
    save_all_states=True,
)

# ---------------------------------------------------------
# 2. Build tiles and exact transforms
# ---------------------------------------------------------
tiles = []
for tile_box in make_tiles(4096, tile_size=1024, stride=512):
    tile_camera = derive_exact_centered_tile_camera(
        global_camera,
        tile_box,
    )

    tiles.append(
        TileContext(
            image=crop(image_4096, tile_box),
            camera=tile_camera,
            T_g2l=build_exact_global_to_local(global_camera, tile_camera),
            T_l2g=build_exact_local_to_global(global_camera, tile_camera),
        )
    )

# ---------------------------------------------------------
# 3. Support
# ---------------------------------------------------------
for tile in tiles:
    tile.anchor_support = project_global_support_to_local(
        global_result.support,
        tile.T_g2l,
        no_clamp=True,
    )

    tile.native_support = run_local_sparse_structure(
        tile.image,
        tile.camera,
        shared_spatial_noise,
    )

candidate_graph = map_all_candidates_to_global(
    tiles,
    exact_transforms=True,
)

accepted_candidates = topology_consensus(candidate_graph)

for tile in tiles:
    tile.flow_support = (
        tile.anchor_support
        | accepted_candidates.for_tile(tile.id)
    )

# ---------------------------------------------------------
# 4. Coupled flow for each continuous stage
# ---------------------------------------------------------
for stage in ["shape512", "shape1024", "texture1024"]:
    atom_space = build_common_atom_space(
        global_support=global_result.support_at(stage),
        local_supports=[t.flow_support_at(stage) for t in tiles],
        transforms=[t.T_l2g for t in tiles],
        target_resolution=target_resolution(stage),
    )

    global_state = init_global_state_from_spatial_noise(
        atom_space,
        stage,
    )

    local_states = [
        init_local_state_from_same_spatial_noise(
            atom_space,
            tile,
            stage,
        )
        for tile in tiles
    ]

    for t0, t1 in sampler_steps(stage):
        v_global = global_model(
            global_state,
            global_condition,
            t0,
        )
        x1_global = velocity_to_clean_endpoint(
            global_state,
            v_global,
            t0,
            path,
        )

        local_x1 = []
        for tile, state in zip(tiles, local_states):
            v_local = local_model(
                state,
                tile.condition,
                t0,
            )
            local_x1.append(
                velocity_to_clean_endpoint(
                    state,
                    v_local,
                    t0,
                    path,
                )
            )

        global_atoms = lift_global_to_atoms(
            x1_global,
            atom_space,
        )

        residuals = []
        for tile, x1_local in zip(tiles, local_x1):
            local_atoms = lift_local_to_atoms(
                x1_local,
                tile,
                atom_space,
            )
            residuals.append(local_atoms - global_atoms)

        merged_residual = robust_merge_per_atom(
            residuals,
            tile_windows=True,
            visibility=True,
            agreement=True,
        )

        high_residual = coverage_aware_zero_mean_highpass(
            merged_residual,
            atom_space,
            global_parent_resolution=stage,
        )

        x1_unified = global_atoms + high_residual

        new_local_states = []
        for tile, state in zip(tiles, local_states):
            x1_sync = restrict_atoms_to_local(
                x1_unified,
                tile,
                atom_space,
            )

            v_sync = clean_endpoint_to_velocity(
                state,
                x1_sync,
                t0,
                path,
            )

            new_local_states.append(
                ode_step(state, v_sync, t0, t1)
            )

        local_states = new_local_states
        global_state = ode_step(
            global_state,
            v_global,
            t0,
            t1,
        )

    save_unified_stage_state(
        stage,
        x1_unified,
        atom_space,
    )
```

---

# 31. 必须满足的数值不变量

## 相机变换

[
q_g\rightarrow q_l\rightarrow q_g
]

和：

[
q_l\rightarrow q_g\rightarrow q_l
]

要求：

[
\max|\Delta q|<2\times10^{-5}
]

## Cell 质量守恒

对每个 local cell：

[
\sum_i\Omega_{kji}
==================

|T_k(L_{k,j})\cap[-1,1]^3|
]

对完全位于 cube 内的 cell：

[
\sum_i\Omega_{kji}
==================

|T_k(L_{k,j})|
]

## Noise 边缘分布

[
\operatorname{mean}(\epsilon_g)\approx0
]

[
\operatorname{std}(\epsilon_g)\approx1
]

[
\operatorname{mean}(\epsilon_l)\approx0
]

[
\operatorname{std}(\epsilon_l)\approx1
]

## Noise 交叉协方差

[
\operatorname{Cov}
(\epsilon_{k,j}^l,\epsilon_i^g)
\approx
\frac{\Omega_{kji}}
{\sqrt{V_{k,j}^lV_i^g}}
]

## Field operator

[
R_gP_g=I
]

[
P_g\mathbf1=\mathbf1
]

[
R_g\mathbf1=\mathbf1
]

## 高频

[
R_gh=0
]

## Unified endpoint

[
R_g\hat x_{1,H}
===============

\hat x_{1,g}
]

只要最后一条不成立，就意味着 local detail 已经污染了 global low-frequency trajectory。

---

# 32. 评估设计

不能只看最终整图 PSNR。

## Global 4096 render

计算：

* PSNR；
* SSIM；
* LPIPS；
* silhouette IoU；
* foreground-only metrics。

## 49 个 tile render

对每个 tile 使用 exact local camera 渲染：

* reference tile；
* global baseline crop；
* local-only；
* unified SR。

记录每 tile：

```text
PSNR
SSIM
LPIPS
silhouette
depth consistency
normal consistency
```

## Overlap consistency

对相邻 tile overlap 区域计算：

[
E_{seam}
========

|R_k-R_{k'}|
]

包括：

* RGB seam；
* depth seam；
* normal seam；
* material seam。

## Geometry

比较同一物理区域的多个 local surface：

* point-to-point；
* point-to-plane；
* normal cosine；
* occupancy agreement；
* connected components；
* floating geometry count；
* double-shell distance。

---

# 33. 实施顺序

不要一次实现全部系统。按以下顺序推进。

## 第一阶段：固定 topology，只验证 noise

* 只用 projected global support；
* global/local shared noise；
* 不做 endpoint synchronization；
* 验证协方差和 tile consistency。

## 第二阶段：固定 topology，Shape512 endpoint sync

* 不增加 local support；
* 不做 texture；
* 只验证 high-pass endpoint；
* 检查：

[
R_g\hat x_{1,H}=\hat x_{1,g}
]

## 第三阶段：Shape1024 endpoint sync

* 固定 topology；
* 验证局部 geometry detail；
* 检查 global render 不下降。

## 第四阶段：local topology candidate

* 加入 native candidate；
* 先只允许 surface subdivision；
* 暂不允许独立 topology birth。

## 第五阶段：多 tile topology birth

* 多 tile consensus；
* visibility/depth 检查；
* connectivity 检查。

## 第六阶段：Texture1024

* geometry 固定；
* global material base；
* local material residual；
* robust merge。

## 第七阶段：统一 O-Voxel 和最终 GLB

* 单次 meshing；
* global camera 4096 render；
* 49 tile exact camera render。

---

# 34. 最终方案的本质

整个方法可以压缩成一个主公式。

对于每个连续生成阶段：

[
\boxed{
\hat x_{1,H}
============

P_g\hat x_{1,g}
+
Q_g
\operatorname{RobustMerge}*k
\left[
P*{H\leftarrow l_k}\hat x_{1,l,k}
---------------------------------

P_{H\leftarrow g}\hat x_{1,g}
\right]
}
]

其中：

[
Q_g=I-P_gR_g
]

并保证：

[
R_gQ_g=0
]

噪声则使用：

[
\boxed{
\epsilon(D)
===========

\frac1{\sqrt{|D|}}
\int_DdW
}
]

Topology 独立使用：

[
\boxed{
S_H
===

S_g^{anchor}
\cup
\operatorname{Consensus}
\left(
\bigcup_kT_k(S_k^{native})
\right)
}
]

最终组合为：

```text
Global:
决定低频结构、整体拓扑、不可见区域和材质基准

Local:
利用 4096 原图提供局部高频几何和纹理

Exact camera transform:
决定所有 token 的真实物理位置与 footprint

Shared spatial noise:
防止 local/global 从无关随机轨迹开始

Clean-endpoint synchronization:
避免直接平均不兼容的 raw velocity

Zero-mean high-pass:
保证 local detail 不改变 global 低频

Topology consensus:
允许真实新结构进入，但拒绝单 tile 幻觉

Unified O-Voxel:
最终只生成一个连续、无 tile 副本的 3D 资产
```

这不是把 49 个独立模型拼起来，而是把 49 个 local Pixal3D flow 当作同一个 global high-resolution latent field 的局部观测器。
