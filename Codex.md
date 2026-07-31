请按照仓库中的 `Codex.md` 执行本实验。使用 CUDA 4，基于当前 Pixal3D-exp、官方 TencentARC/Pixal3D 代码和本地 Hugging Face 模型权重，实现并完整验证以下研究想法：

# Visibility-Routed Per-Token Image Conditioning for Pixal3D

## 一、研究目标

当前 independent local HR tile flow 的优点是输入视图纹理质量高，但存在：

* 不同 tile 独立生成；
* 相邻 tile 之间缺乏联系；
* 明显纹理边界；
* 背面和遮挡区域被 local crop 错误重写；
* 多视角结果不稳定。

已有 latent posterior 实验表明：

* shape latent posterior 能明显改善几何一致性；
* texture latent posterior 会显著损害输入视图保真度；
* 继续直接融合 texture latent 或 velocity 的意义较弱。

本实验不再融合 SLat、clean latent 或最终 velocity，而是在 Pixal3D Transformer 的每个 block 内，对不同 SLat token 使用不同的图像条件：

[
\boxed{
\text{可见前表面 token 使用 local HR 图像条件}
}
]

[
\boxed{
\text{遮挡/背面 token 使用 global 图像语义条件，
但不使用 local projected pixel feature}
}
]

所有 SLat token 始终位于同一条 flow 轨迹、同一个 Transformer hidden state 中，并通过每层 self-attention 继续交换信息。

主方法定义为：

[
C_n^\ell
========

m_n
\left[
A_\ell(h_n^\ell;g_H)
+
W_{\ell,p}p_{H,n}
\right]
+
(1-m_n)
A_\ell(h_n^\ell;g_G),
]

其中：

* (n) 是 SLat token；
* (\ell) 是 Transformer block；
* (m_n\in{0,1}) 是固定的 baseline-geometry visibility mask；
* (g_H) 是 local HR tile 的完整 global DINO context；
* (p_{H,n}) 是与第 (n) 个 SLat token 对齐的 local HR projected feature；
* (g_G) 是完整 global 1024 图像的 global DINO context；
* 后景 projected feature 使用官方意义下的 zero feature branch；
* (A_\ell) 是该 block 的 global cross-attention；
* (W_{\ell,p}) 是该 block 的 projected-feature linear layer。

必须验证：

1. 逐 token 条件路由在代码和 tensor 语义上正确；
2. 前景 local HR 信息是否通过后续 self-attention传播到后景；
3. 后景使用 global semantic context 是否减少错误重建；
4. 是否降低 tile texture seam；
5. 是否保持输入视图 PSNR、SSIM、LPIPS；
6. 是否改善多视图几何和材质表现。

---

# 二、先检查官方代码、配置和权重

正式改代码前，必须检查并在报告中记录以下事实，不允许仅根据提示词假设。

检查官方及本地对应文件：

```text
pixal3d/modules/sparse/attention/proj_attention.py
pixal3d/modules/sparse/transformer/modulated.py
pixal3d/modules/attention/
pixal3d/pipelines/pixal3d_image_to_3d.py
pixal3d/pipelines/samplers/
```

检查本地 Hugging Face checkpoint 配置和实际 `state_dict`：

* shape 1024 flow；
* texture 1024 flow。

确认并记录：

1. shape/texture 模型是否使用：

```text
image_attn_mode = "proj"
```

2. 每个 block 的执行顺序是否为：

```text
self-attention
→ image attention / projected feature injection
→ MLP
```

3. `SparseProjectAttention` 当前是否计算：

```python
global_out = cross_attn(x, global_context)
proj_out = proj_linear(proj_context.feats)
combined = global_out.feats + proj_out
```

4. `proj_context` 是否与 SLat token：

* coordinates 完全一致；
* token 数完全一致；
* token order 完全一致。

5. `global_context` 是否在一个样本内被所有 SLat query 共享。

6. 从真实 checkpoint `state_dict` 检查：

* `to_q`；
* `to_kv`；
* `to_out`；
* `proj_linear`

是否带 bias。

7. 确认官方 CFG negative condition 到底是：

```python
zeros_like(global_feature)
zeros_like(projected_feature)
```

还是其他实现。

8. 确认 shape 默认 CFG 和 texture 默认 guidance 路径。

检查完成后，在：

```text
outputs/visibility_routed_conditioning/ARCHITECTURE_AUDIT.md
```

记录真实代码路径、类名、函数名、配置值、tensor shape 和权重 bias 信息。

如果实际实现与本提示词不符，以官方实际代码为准，但必须在报告中说明调整原因。

---

# 三、严格区分三种条件

对每个 tile 构造三套完整图像条件。

## 3.1 Local HR 条件 (H)

图像为真正的 4096 输入图中对应的 1024 tile。

完整调用官方条件提取流程，得到：

[
g_H,\qquad p_H.
]

必须保证：

* `global` 和 `proj` 都来自同一张 local HR tile；
* 使用当前 local camera；
* 使用当前 C64 sparse coordinates；
* 不混用其他图像的 projected feature。

## 3.2 Global 语义条件 (G)

图像为 global Pixal3D 使用的完整 canonical 1024 图像。

调用官方图像 encoder，得到：

[
g_G.
]

主方法中只使用其 global context。

后景 projected condition 不使用 global 图像的 projected volume，而使用官方 zero projected feature：

[
p_B=0.
]

因此后景条件为：

[
(g_G,0).
]

不得把 global full image 的 projected feature 与 local camera/SLat coords强行混合。

## 3.3 官方 zero condition (0)

严格复用官方 CFG negative condition 构造方式：

[
g_0=0,\qquad p_0=0.
]

注意：

* zero feature 不等于输入一张黑图；
* 不要把黑图送入 DINO；
* 必须直接构造与官方 negative condition一致的 zero feature tensor。

---

# 四、不要错误实现 per-token global condition

当前 global cross-attention 中，所有 SLat query通常共享同一组 image context。

禁止通过以下方式伪造逐 token global context：

```python
global_context.shape = [N_slat, N_image_tokens, C]
```

除非官方 attention API 原生支持这一 batch/query 语义，并经过严格验证。

主实现必须在每个 image-attention block 内分别计算两次：

[
a_H^\ell
========

A_\ell(h^\ell;g_H),
]

[
a_G^\ell
========

A_\ell(h^\ell;g_G),
]

然后按 SLat token mask 路由：

[
a_n^{\ell,*}
============

m_na_{H,n}^\ell
+
(1-m_n)a_{G,n}^\ell.
]

因为 cross-attention 的每个 query row 对给定 context 独立计算，所以对二值 (m_n) 而言，该路由严格对应：

[
A_\ell(h_n;g_H)
\quad\text{或}\quad
A_\ell(h_n;g_G).
]

---

# 五、修改 attention 模块，而不是最终 velocity

新建 routed attention 模块或在原模块中增加显式 mode，不允许破坏官方默认路径。

建议新增：

```text
VisibilityRoutedSparseProjectAttention
```

或者：

```text
SparseProjectAttention.forward_routed(...)
```

输入至少包括：

```python
x
global_front_context
proj_front_context
global_back_context
proj_back_context
token_visibility
```

核心逻辑应保持官方 residual、normalization、modulation 和 dtype 行为，仅替换 image-condition contribution。

伪代码：

```python
global_front_out = self.cross_attn_block(
    x,
    global_front_context,
)

global_back_out = self.cross_attn_block(
    x,
    global_back_context,
)

proj_front_feats = self.proj_linear(
    proj_front_context.feats,
)

proj_back_feats = self.proj_linear(
    proj_back_context.feats,
)

front_condition = (
    global_front_out.feats
    + proj_front_feats
)

back_condition = (
    global_back_out.feats
    + proj_back_feats
)

mask = token_visibility.to(
    device=front_condition.device,
    dtype=front_condition.dtype,
).reshape(-1, 1)

routed_condition = (
    mask * front_condition
    + (1.0 - mask) * back_condition
)

return x.replace(routed_condition)
```

这里 `proj_back_context.feats` 在主方法中必须是：

```python
torch.zeros_like(proj_front_context.feats)
```

但仍然要调用：

```python
self.proj_linear(zero_feats)
```

不能直接假设结果为零，因为真实 checkpoint 中 linear layer 可能有 bias。

同理，官方 zero global condition必须经过完整 cross-attention模块，不能直接把 global cross-attention输出手工设为零。

---

# 六、主条件路由公式

## 6.1 前景/可见 token

[
C_{F,n}^\ell
============

A_\ell(h_n^\ell;g_H)
+
W_{\ell,p}p_{H,n}.
]

## 6.2 后景/遮挡 token

主方法：

[
C_{B,n}^\ell
============

A_\ell(h_n^\ell;g_G)
+
W_{\ell,p}0.
]

因此：

[
\boxed{
C_n^\ell
========

m_nC_{F,n}^\ell
+
(1-m_n)C_{B,n}^\ell
}
]

这里：

* 前景由 local HR pixel-aligned feature重建；
* 后景不接受 local projected feature；
* 后景保留完整 global object semantic context；
* 前景信息通过下一层 self-attention继续传播给后景；
* 所有 token 使用同一个 hidden state 和同一条 flow trajectory。

---

# 七、baseline geometry visibility mask

这里的“前景/后景”指：

* 相对于当前 tile camera 的 first-hit visible surface；
* 被该 first-hit surface 遮挡的表面或背面。

不是二维背景分割，也不是图像 alpha/foreground mask。

## 7.1 Mask 来源

只允许使用当前输入经过 ordinary global Pixal3D 1024 得到的 baseline geometry。

禁止使用：

* ground-truth mesh；
* ground-truth depth；
* novel-view GT；
* 手工标记；
* 输入图像语义分割。

## 7.2 先在 baseline 几何空间判断 visibility

使用 global baseline：

* mesh；
* 或 O-Voxel surface；

在当前 local tile camera 下执行 z-buffer/first-hit visibility。

对 baseline surface O-Voxel 或 surface primitive标记：

[
v_q=
\begin{cases}
1,&q\text{ 是某个 camera ray 的 first-hit surface},\
0,&q\text{ 被遮挡或属于背面}.
\end{cases}
]

不能简单使用法向量正负判断，因为：

* 凹面会产生自遮挡；
* front-facing 不等于 visible；
* 多层表面需要 first-hit depth。

## 7.3 将 visibility 严格映射到 C64 SLat coords

不得只根据 SLat token center 与 depth map做随意阈值判断。

应优先使用模型真实的稀疏坐标下采样映射：

1. 从 global baseline O-Voxel 精确变换到 local C1024；
2. 得到每个 local source O-Voxel 的 visible/occluded标记；
3. 沿 shape encoder 的真实 sparse coordinate downsampling路径，追踪每个 source O-Voxel 对最终 C64 token 的对应关系；
4. 使用真实 sparse-convolution indice map、coordinate manager或等价的精确坐标映射；
5. 不允许未经确认直接假设：

[
\operatorname{coord}_{64}
=========================

\lfloor
\operatorname{coord}_{1024}/16
\rfloor.
]

只有在通过代码和实际 coords验证完全一致后才能使用该简化。

对每个最终 C64 token，统计其 encoder receptive support 中：

* visible baseline surface O-Voxel 数；
* 总 baseline surface O-Voxel 数。

定义：

[
r_n
===

\frac{
N_{n,\mathrm{visible}}
}{
N_{n,\mathrm{surface}}
}.
]

主 hard mask：

[
m_n=
\mathbf 1[
N_{n,\mathrm{visible}}>0
].
]

这里不使用手调可见性概率阈值。

同时保存连续 visibility ratio (r_n)，仅用于 soft-routing 消融。

## 7.4 无 baseline surface evidence 的 token

如果某个 active C64 token 没有可追踪的 baseline surface O-Voxel evidence：

* 主方法将其标记为 back/occluded：

[
m_n=0.
]

原因是不能证明它是当前视角的可见表面。

必须记录此类 token 的数量和比例。

## 7.5 Mask 固定

Shape flow 的 12 个 Euler steps 内，mask 固定：

[
m_n(t)=m_n.
]

禁止每一步根据当前生成 geometry 重新计算 hard visibility。

---

# 八、Shape 与 texture 的执行顺序

## 8.1 Shape flow

使用 global baseline geometry构造：

[
m_S.
]

运行完整 routed shape flow：

[
x_S(1)\rightarrow x_S(0).
]

Shape flow 中：

* front 使用 local HR (g_H^S,p_H^S)；
* back 使用 full global (g_G^S,0)；
* 使用官方 shape CFG；
* 不进行 latent posterior；
* 不进行 velocity fusion。

得到：

[
z_S^{route}.
]

## 8.2 Texture flow

Texture flow 必须使用：

```python
concat_cond = z_shape_route_norm
```

LR、HR、global、zero 等所有 texture condition 分支使用同一个最终 routed normalized shape SLat。

主实现优先基于最终 routed shape geometry重新计算一次 texture visibility mask：

[
m_T.
]

流程：

```text
routed final shape latent
→ model-native shape decode / geometry decode
→ 当前 local camera first-hit visibility
→ 严格映射回相同 C64 token coords
→ 得到固定 m_T
```

如果官方代码无法在 texture 前独立从 shape latent获得可靠 geometry，允许使用：

[
m_T=m_S,
]

但必须：

* 在报告中明确说明；
* 实现为显式 fallback；
* 不得静默发生；
* 将“重算 texture mask”和“复用 shape mask”作为消融。

Texture flow 中：

* front 使用 local HR texture condition；
* back 使用 full global image的 texture global context；
* back projected texture feature为 zero；
* `concat_cond` 始终为同一个 routed shape；
* 默认保持官方 texture guidance strength，不人为加入 CFG。

---

# 九、CFG 的正确使用

先在模型内部构造 routed conditional prediction：

[
v_{\mathrm{route}}
==================

f_\theta^{route}
(x,t;g_H,p_H,g_G,0,m).
]

再按官方 sampler 的原始逻辑做 CFG。

Shape：

[
v_{\mathrm{CFG}}
================

v_0
+
s(t)
\left(
v_{\mathrm{route}}-v_0
\right),
]

其中 (v_0) 必须由官方 full-zero negative condition获得。

禁止：

* 在每个 block 内分别对 front/back 做 CFG；
* 对 cross-attention block输出做 CFG；
* 将 global-back branch 当作 CFG negative branch；
* 手工修改官方 guidance rescale公式。

Texture：

* 使用官方默认 guidance；
* 如果 guidance strength 为 1，则保持只运行 routed conditional prediction；
* zero-back configuration只是方法消融，不代表 texture CFG。

---

# 十、必须实现的消融配置

所有配置使用：

* 完全相同的 global baseline；
* 完全相同的 tile layout；
* 完全相同的 sparse support；
* 完全相同的 seeds；
* 完全相同的 decode/mesh-return/ownership/welding/render路径。

## A. Local HR baseline

所有 token 都使用：

[
(g_H,p_H).
]

这是现有 independent local tile baseline。

## B. Proj-only visibility routing

所有 token共享 local HR global context：

[
C_n
===

A(h_n;g_H)
+
m_nW_pp_{H,n}
+
(1-m_n)W_p0.
]

该配置仅关闭背面 local projected feature，是最接近训练分布的最小修改。

## C. H/G blockwise routing，主方法

[
C_n
===

m_n
\left[
A(h_n;g_H)+W_pp_{H,n}
\right]
+
(1-m_n)
\left[
A(h_n;g_G)+W_p0
\right].
]

## D. H/zero blockwise routing

[
C_n
===

m_n
\left[
A(h_n;g_H)+W_pp_{H,n}
\right]
+
(1-m_n)
\left[
A(h_n;0)+W_p0
\right].
]

用于检验完全 unconditional backside completion。

## E. Final-velocity token routing

分别完整运行：

[
v_H=f_\theta(x,t;g_H,p_H),
]

[
v_G=f_\theta(x,t;g_G,0),
]

最后才做：

[
v_n^*
=====

m_nv_{H,n}
+
(1-m_n)v_{G,n}.
]

这是负对照，用来验证 blockwise hidden-state routing 是否优于最终 velocity拼接。

## F. Soft blockwise routing

使用连续 (r_n)：

[
C_n
===

r_nC_{F,n}
+
(1-r_n)C_{B,n}.
]

只作为消融，不作为默认主方法。

## G. Shape-only routing

* shape 使用主 H/G blockwise routing；
* texture 使用 ordinary local HR；
* texture `concat_cond` 使用 routed shape。

## H. Texture-only routing

* shape 使用 ordinary local HR；
* texture 使用 H/G blockwise routing。

## I. Joint routing

* shape 与 texture 都使用 H/G blockwise routing。

主配置为 I。

---

# 十一、实现开关

建议实现：

```text
--conditioning-mode local
--conditioning-mode proj_mask
--conditioning-mode hg_block
--conditioning-mode hzero_block
--conditioning-mode hg_velocity
--conditioning-mode hg_soft
```

分别实现：

```text
--route-shape
--no-route-shape
--route-texture
--no-route-texture
```

Texture mask：

```text
--texture-mask-source routed_shape
--texture-mask-source baseline_shape
```

默认主方法：

```text
--conditioning-mode hg_block
--route-shape
--route-texture
--texture-mask-source routed_shape
```

---

# 十二、单元测试和数学极限测试

正式 GPU 运行前必须完成以下测试。

## 12.1 全前景等价测试

当：

[
m=\mathbf1
]

时，routed model 输出必须与官方 local HR model输出一致：

[
\max
|v_{\mathrm{route}}-v_H|
<10^{-5}
]

或在 bf16/attention backend数值误差下给出合理阈值和实测误差。

需要在：

* 单个 attention block；
* 完整 shape flow model；
* 完整 texture flow model；

分别测试。

## 12.2 全后景 global 等价测试

当：

[
m=\mathbf0
]

时，H/G routed model必须等价于：

[
f_\theta(x,t;g_G,0).
]

## 12.3 全后景 zero 等价测试

H/zero mode，(m=0) 时必须等价于官方 full-zero condition model。

## 12.4 Token row routing测试

构造少量 token：

```text
front, back, front, back
```

验证每个 output row严格来自对应 front/back attention contribution。

## 12.5 Projected zero bias测试

验证：

```python
proj_linear(torch.zeros_like(proj))
```

的实际输出。

若不为零，必须保留真实 bias，不得手工置零。

## 12.6 坐标对齐

每一步断言：

```python
assert torch.equal(x.coords, proj_front.coords)
assert torch.equal(x.coords, proj_back.coords)
assert torch.equal(x.coords, mask_coords)
assert mask.shape == (x.feats.shape[0],)
```

## 12.7 不改变官方默认路径

在不启用 routed mode 时，修改后的代码必须与原 baseline逐项一致。

---

# 十三、需要记录的 flow 诊断

对 shape 和 texture每一步记录：

* timestep；
* token 总数；
* visible token 数和比例；
* back token 数和比例；
* no-surface-evidence token 数；
* front global-attention output RMS；
* back global-attention output RMS；
* front projected output RMS；
* zero projected output RMS；
* routed condition RMS；
* front/back condition cosine；
* routed velocity RMS；
* local HR velocity RMS；
* routed/local velocity cosine；
* front token velocity difference RMS；
* back token velocity difference RMS；
* 前景和后景 hidden-state RMS；
* 每个 block 前景/后景 hidden cosine；
* 相邻 block 中 front→back self-attention influence proxy。

对于最后一项，可以记录 self-attention matrix或 attention统计允许的近似指标，例如：

[
\frac{
\sum_{n\in B,m\in F}
A^{SA}*{nm}
}{
\sum*{n\in B,m}
A^{SA}_{nm}
}
]

如果使用 Flash Attention无法直接得到 attention matrix，则：

* 不允许关闭默认 backend影响主实验；
* 可以在小 token smoke test中使用可观测 backend单独分析；
* 主实验记录 hidden-state intervention proxy；
* 在报告中明确限制。

---

# 十四、边界和多视图评价

沿用现有实验完全相同的 final decode、local-to-global、triangle ownership、vertex weld、PBR sampling 和 renderer。

禁止通过修改 mesh merge 方法掩盖 condition-routing效果。

## 14.1 输入视图

报告：

* PSNR；
* SSIM；
* LPIPS。

## 14.2 几何

报告：

* connected components；
* backside components；
* overlap Chamfer；
* overlap normal consistency；
* low-frequency Chamfer to global；
* largest component ratio。

## 14.3 材质

报告：

* overlap PBR latent RMSE；
* base-color difference；
* roughness difference；
* metallic difference；
* existing multiview material flicker proxy。

## 14.4 Tile boundary image metric

增加专门的纹理边界评价。

在 canonical input-view render中，根据固定 7×7 tile layout构造：

* tile-boundary band；
* tile-interior region。

边界 band 宽度应由 tile overlap和像素几何确定，不用最终指标调参。

分别报告：

[
\operatorname{PSNR}*{boundary},
\quad
\operatorname{LPIPS}*{boundary},
]

[
\operatorname{PSNR}*{interior},
\quad
\operatorname{LPIPS}*{interior}.
]

同时计算沿边界两侧的梯度跳变：

[
E_{\mathrm{seam}}
=================

\frac1{|\mathcal B|}
\sum_{u\in\mathcal B}
|
\nabla I_{\mathrm{left}}(u)
---------------------------

\nabla I_{\mathrm{right}}(u)
|_1.
]

必须同时在：

* rendered RGB；
* base color；
* roughness；
* metallic；

上计算物理对应位置的 boundary discontinuity。

## 14.5 多视图

至少渲染：

```text
yaw = 0, -45, 45, -90, 90, 180
```

相同：

* FOV；
* camera distance；
* pitch；
* lighting；
* environment map；
* resolution。

若无 novel-view ground truth，不得将这些视图称为 novel-view PSNR。

必须生成统一 comparison sheet：

```text
Local HR
Proj-only mask
H/G blockwise
H/zero blockwise
H/G velocity routing
Joint routed
```

---

# 十五、运行顺序

## 第一阶段：架构审计

完成：

```text
ARCHITECTURE_AUDIT.md
```

并通过所有不加载大模型的合成测试。

## 第二阶段：单 block 与完整模型等价测试

随机或真实小 SparseTensor：

* all-front；
* all-back；
* mixed mask。

验证 shape/texture 模型。

## 第三阶段：单 tile smoke test

优先使用：

```text
tile 24
```

运行：

* local；
* proj_mask；
* hg_block；
* hzero_block；
* hg_velocity。

先 `--no-decode`，检查完整 12 steps。

## 第四阶段：7 tile 联合测试

使用：

```text
16,17,23,24,25,31,32
```

完成 shape + texture + decode/render。

重点查看：

* visible/back token比例；
* tile boundary；
* 前景 fidelity；
* 后景生成。

## 第五阶段：完整 seed 42

对完整 48 个成功 tile运行全部主要配置：

```text
local
proj_mask
hg_block
hzero_block
hg_velocity
shape_only_hg
texture_only_hg
joint_hg
```

## 第六阶段：多 seed

如果 `joint_hg` 或 `proj_mask` 相比 Local HR：

* 输入视图没有明显崩溃；
* boundary seam有改善；
* 多视图没有明显退化；

则对最好的两个 routed配置运行：

```text
seed = 42,43,44,45
```

不要只报告最好 seed。

---

# 十六、输出目录

```text
outputs/visibility_routed_conditioning/
```

建议结构：

```text
ARCHITECTURE_AUDIT.md
EXPERIMENT_REPORT.md
summary.json

local_hr/
proj_mask/
hg_block/
hzero_block/
hg_velocity/
shape_only_hg/
texture_only_hg/
joint_hg/

statistics/
  mask/
  shape_steps/
  texture_steps/
  block_diagnostics/

plots/
  render_metrics.png
  boundary_metrics.png
  geometry_metrics.png
  material_metrics.png
  front_back_condition_energy.png
  front_to_back_attention_proxy.png
  multiseed_metrics.png

multiview_comparison/
```

每个 tile 至少保存：

```text
tile_XX/
  visibility_mask.pt
  visibility_summary.json
  visible_ovoxels.png
  c64_token_visibility.png
  shape_trace.json
  texture_trace.json
```

`visibility_mask.pt` 保存：

```python
{
    "coords": c64_coords.cpu(),
    "hard_visibility": hard_mask.cpu(),
    "visibility_ratio": ratio.cpu(),
    "visible_surface_counts": visible_counts.cpu(),
    "total_surface_counts": total_counts.cpu(),
    "source": "global baseline first-hit visibility",
}
```

---

# 十七、成功判定

主方法 `joint_hg` 不要求在所有 consistency 指标上超过强 global anchor，但必须证明 condition routing不是另一种保真度—一致性简单折中。

相对 Local HR，建议严格判定：

## 输入视图保持

[
\Delta\operatorname{PSNR}\ge-0.3\text{ dB},
]

[
\Delta\operatorname{LPIPS}\le+0.01.
]

## Texture boundary

至少满足一个主要边界指标显著改善，并且 interior fidelity不明显下降：

* boundary PSNR提高；
* seam gradient energy下降；
* overlap base-color difference下降；
* overlap PBR RMSE下降。

## 多视图

* 不出现比 Local HR 更明显的新纹理块；
* 后景不再复制错误 local projected texture；
* H/G blockwise 应优于 H/G final velocity routing。

## 机制验证

必须观察到：

* all-front 与原 local HR 数值等价；
* blockwise routing内前景信息能够进入后续 back hidden state；
* 后景 global branch 与前景 local branch在同一隐藏轨迹中工作；
* 不需要融合最终 latent 或 velocity。

只有满足这些条件，才能宣称 per-token condition routing 有效。

---

# 十八、报告必须明确回答

最终 `EXPERIMENT_REPORT.md` 必须回答：

1. Pixal3D 官方代码是否原生支持 per-token projected feature？
2. global context 是否在样本内共享？
3. 实际 checkpoint 中 attention/proj linear是否有 bias？
4. 官方 zero condition 的精确定义是什么？
5. baseline geometry visibility 如何映射到 C64 SLat tokens？
6. 前景、后景、无表面证据 token各占多少？
7. Proj-only mask是否已经足以改善背面和边界？
8. H/G blockwise是否优于 H/zero？
9. H/G blockwise是否优于最终 velocity token routing？
10. Shape-only、texture-only、joint routing各自贡献是什么？
11. 输入视图是否保持在 Local HR 附近？
12. tile boundary是否真实减弱？
13. 多视图后景是否更合理？
14. 前景 HR detail是否被保留？
15. 该方法是否值得替代 texture latent posterior？
16. 是否存在明确的训练分布外失效现象？

不要因为代码可运行就宣称数学或生成质量成功。

该方法属于 test-time conditional intervention：

* 不更新权重；
* 不训练 mapper；
* 不融合 SLat；
* 不融合最终 velocity；
* 不读取未来 endpoint；
* 不使用 ground-truth geometry；
* 不修改最终 mesh merge逻辑。

最终结论必须由完整输入视图、边界、多视图、几何和材质结果共同决定。
