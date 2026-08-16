```markdown
# 任务：Final-PureHR Shared-Coarse Oracle 实验

## 0. 实验目的

本实验只回答一个问题：

> PureHR 相对 Global 的 coarse-scale PBR change 中，是否存在能够被多个独立 tile 重复生成的 shared coarse component？

当前 hidden Sparse-MRA 使用：

\[
Y_i^{null}
=
G_i+(I-P_iA_i)(H_i-G_i),
\]

等价于删除全部 coarse difference：

\[
P_iA_i(H_i-G_i).
\]

新的假设是：

> coarse 本身不代表错误。  
> 应删除的是 tile-specific coarse hallucination，而不是所有 coarse HR change。

因此定义：

\[
\boxed{
Y_i^{shared}
=
G_i
+
C_i^{shared}
+
D_i
}
\]

其中：

\[
C_i=P_iA_i(H_i-G_i),
\]

\[
D_i=(I-P_iA_i)(H_i-G_i),
\]

而 \(C_i^{shared}\) 是多个 overlapping tile 对同一 canonical surface region 产生的共同 coarse increment。

本轮只验证这个假设。

---

# 1. 最重要的限制

## 禁止

本轮禁止：

- 重新跑 texture flow；
- 重新跑 shape flow；
- per-step guidance；
- `_xstart_to_pred`；
- Euler；
- 将 oracle target encode 回 SLat；
- timestep weight；
- G/H scalar blending；
- 新的手工权重；
- Gaussian center-distance weighting用于 hidden consensus；
- clamp PBR 到 `[0,1]`；
- 修改 geometry；
- 修改 visibility；
- 修改 P/A 定义。

## 允许

只允许：

1. 读取已经生成好的最终 PureHR endpoint / final PBR field；
2. 如最终 PBR field 未缓存，可 decode final PureHR endpoint **一次**；
3. 使用已有 stable float64 direct-LSMR projector；
4. 在 PBR field 中进行 coarse/shared/private decomposition；
5. 直接生成 Mesh/PBR render。

---

# 2. 必须使用完全相同的 tile ensemble

正式 full oracle 必须与之前 PureHR / current Gaussian 实验使用同一套：

```text
tile_size = 1024
stride = 512
canonical = 4096
7 × 7 layout
```

先检查历史有效 tile set。

之前正式实验为 48 valid tiles，而最近 full batch 出现 47 active tiles并缺少 Tile35。

在 oracle full run 前必须：

```text
assert exact tile layout
assert tile id set
```

若 Tile35 缺失：

必须先查清原因。

禁止在 tile ensemble 不一致的情况下比较最终指标。

---

# 3. 数据定义

对每个 tile \(i\)：

读取：

### Global local reference

\[
G_i
\]

即 aligned local C1024 Global PBR field。

### Final PureHR

\[
H_i
\]

必须来自**未经过 MRA guidance 的 final PureHR endpoint**。

不要使用之前 MRA flow 的 step11 `H_t`。

这是非常重要的。

---

# 4. 使用 stable projector

继续使用当前已经验证的：

\[
A_i=P_i^\dagger
\]

计算上：

\[
A_iX
=
\operatorname{LSMR}(P_i,X).
\]

使用 direct float64 LSMR。

禁止：

\[
P^TP
\]

normal equation。

本轮是离线 oracle，优先 correctness。

---

# 5. 第一层：support-scale decomposition

对 hidden rows：

\[
\Delta_i
=
H_i-G_i.
\]

计算：

\[
\boxed{
c_i=A_i\Delta_i
}
\]

以及：

\[
\boxed{
C_i=P_ic_i
}
\]

和：

\[
\boxed{
D_i=\Delta_i-C_i.
}
\]

其中：

- \(C_i\)：C256-expressible coarse increment；
- \(D_i\)：C256-unrepresentable residual。

验证：

\[
P_i^TD_i\approx0.
\]

---

# 6. 构建完整 coarse field

当前 \(P_i\) 是 hidden rows × pure-hidden coarse basis。

构造：

```text
coarse_full_i
```

大小与完整 local C1024 field 相同。

规则：

```text
hidden rows  = C_i
observed rows = 0
```

这是合法的，因为 pure-hidden coarse basis 被定义为：

> prolongation support 不触碰任何 observed row。

因此 embedded pure-hidden coarse field 在 observed rows 理论上就是 0。

不要重新定义 mixed basis。

---

# 7. 将 coarse field 包装成可连续查询的 PBR field

利用每个 tile 已有 fixed C1024 geometry：

```text
geometry.coords
geometry.vertices
geometry.faces
```

建立临时：

```text
MeshWithVoxel
```

其：

```text
coords = tile C1024 coords
attrs  = coarse_full_i
```

这个 MeshWithVoxel **不是最终材质 mesh**。

它只用来实现：

\[
T_{j\rightarrow i}C_j
\]

即：

> 在 target tile \(i\) 的 canonical surface point 上查询 donor tile \(j\) 的 coarse increment。

复用已有：

```text
local -> global
global -> donor local
MeshWithVoxel.query_attrs
```

路径。

---

# 8. Canonical donor query

对于 target tile \(i\) 的每个 hidden C1024 query point \(x_i^r\)：

得到它对应的 canonical/global 3D point：

\[
p_i^r.
\]

然后对所有 overlapping donor tile \(j\)：

\[
p_i^r
\rightarrow
x_j(p_i^r)
\rightarrow
C_j(x_j).
\]

只有以下条件同时满足才视为 valid donor：

1. canonical point 落入 donor tile 的有效 local domain；
2. donor MeshWithVoxel continuous query 有有效 support；
3. query finite。

本轮 hidden consensus：

```text
不使用 tile-center Gaussian weight
不使用 visible weight
不使用 facing weight
不使用 G weight
```

所有有效 tile 一票相同。

---

# 9. 不允许把不同 tile 直接按 row index 对齐

严禁：

```text
C_i[row] ↔ C_j[row]
```

因为不同 tile local support 不同。

必须经过：

\[
\boxed{
\text{canonical/global surface correspondence}
}
\]

进行 donor query。

---

# 10. 定义 parameter-free shared coarse consensus

对于 target hidden row \(r\)，收集：

\[
\mathcal D_i(r)
=
\{
T_{j\rightarrow i}C_j(r)
\}.
\]

包括 self tile \(i\)。

但是 shared coarse 必须至少有：

```text
self + 1 independent donor
```

即：

\[
|\mathcal D_i(r)|\ge2.
\]

否则没有“cross-tile reproducibility evidence”。

定义：

\[
S_i(r)
=
\begin{cases}
\displaystyle
\arg\min_s
\sum_{j\in\mathcal D_i(r)}
\|s-T_{j\to i}C_j(r)\|_2^2,
&
|\mathcal D_i(r)|\ge2,
\\[8pt]
0,
&
|\mathcal D_i(r)|<2.
\end{cases}
\]

L2 consensus 有闭式：

\[
\boxed{
S_i(r)
=
\frac1{|\mathcal D_i(r)|}
\sum_{j\in\mathcal D_i(r)}
T_{j\to i}C_j(r)
}
\]

这里只使用数学上的 least-squares consensus。

不要添加人为权重。

---

# 11. 注意：\(S_i\) 还不一定属于 target tile 的 coarse range

因此再次投影：

\[
\boxed{
c_i^{shared}
=
A_iS_i
}
\]

然后：

\[
\boxed{
C_i^{shared}
=
P_ic_i^{shared}.
}
\]

这样：

\[
C_i^{shared}\in\operatorname{range}(P_i).
\]

定义：

\[
\boxed{
C_i^{private}
=
C_i-C_i^{shared}.
}
\]

检查：

\[
C_i
=
C_i^{shared}
+
C_i^{private}.
\]

---

# 12. 新 oracle hidden target

构造：

\[
\boxed{
Y_i^{shared}
=
G_i
+
C_i^{shared}
+
D_i.
}
\]

由于：

\[
H_i
=
G_i+C_i+D_i,
\]

因此也必须数值验证：

\[
\boxed{
Y_i^{shared}
=
H_i-C_i^{private}.
}
\]

两个表达必须：

```text
max_abs < numerical tolerance
```

这是本实验最重要的公式。

---

# 13. Old null-only oracle

同时构造旧方法的 oracle：

\[
\boxed{
Y_i^{null}
=
G_i+D_i.
}
\]

即：

\[
Y_i^{null}
=
H_i-C_i.
\]

注意：

这是 final-PureHR field oracle。

不是之前 per-step MRA 结果。

---

# 14. Hidden-isolation render

为了只测试 hidden 方法：

### observed rows

所有 oracle variant 都保持：

\[
Y_i(x)=H_i(x)
\]

即不修改 PureHR observed rows。

### hidden rows

#### Null-only

\[
Y_i(x)=Y_i^{null}(x)
\]

#### Shared-coarse

\[
Y_i(x)=Y_i^{shared}(x).
\]

因此：

```text
PureHR
Null-only oracle
Shared-coarse oracle
```

在 observed rows 完全相同。

所有差异只来自 hidden branch。

这是本轮主要比较。

---

# 15. 可选 production-composition render

在 hidden-isolation 测试成功后，再额外构建一个：

```text
shared_coarse_plus_final_gaussian
```

其中：

### observed

使用 final PureHR decoded fields 做一次现有 H-H Gaussian cross-tile consensus。

### hidden

使用：

\[
Y_i^{shared}.
\]

注意：

这仍然只是 final-field oracle。

禁止重新 flow。

这不是主要结论，只是看未来 production composition 的视觉上限。

---

# 16. 必须输出 cross-tile coarse agreement diagnostics

这是本实验最关键的数据之一。

对于每个 target hidden row：

记录：

```text
donor_count
```

输出 histogram：

```text
0/1 donor
2 donors
3 donors
4 donors
5+ donors
```

---

# 17. Pairwise coarse agreement

在 donor_count ≥ 2 的 query rows 上：

对于 donor coarse increments：

\[
q_j=T_{j\to i}C_j
\]

计算 pairwise：

### cosine

\[
\cos(q_j,q_k)
\]

### relative disagreement

\[
\frac{
\|q_j-q_k\|
}{
\frac12(\|q_j\|+\|q_k\|)+\epsilon
}
\]

分别统计：

```text
RGB
metallic
roughness
alpha
```

输出：

```text
mean
median
p10
p50
p90
```

目标不是设 threshold。

只是回答：

> final PureHR coarse increment 到底有没有 cross-tile reproducibility？

---

# 18. Shared/private energy decomposition

每 tile、每 PBR group 计算：

\[
r_{shared}
=
\frac{
\|C_i^{shared}\|
}{
\|C_i\|+\epsilon
}
\]

和：

\[
r_{private}
=
\frac{
\|C_i^{private}\|
}{
\|C_i\|+\epsilon
}.
\]

以及：

\[
E_{shared}
=
\frac{
\|C_i^{shared}\|^2
}{
\|C_i\|^2+\epsilon
}.
\]

特别输出：

```text
RGB shared coarse energy fraction
metallic shared coarse energy fraction
roughness shared coarse energy fraction
alpha shared coarse energy fraction
```

---

# 19. 与 Global / PureHR 的距离

对 hidden field：

### Old null oracle

\[
d_G^{null}
=
\|Y_i^{null}-G_i\|
\]

\[
d_H^{null}
=
\|Y_i^{null}-H_i\|.
\]

### Shared oracle

\[
d_G^{shared}
=
\|Y_i^{shared}-G_i\|
\]

\[
d_H^{shared}
=
\|Y_i^{shared}-H_i\|.
\]

分别 RGB / metallic / roughness / alpha。

目标是验证：

> shared oracle 是否确实比 old null-only 保留更多 PureHR variation，而不是机械回到 Global。

---

# 20. PBR physical-domain diagnostics

严禁 clamp。

分别检查：

```text
G
PureHR
Null-only oracle
Shared-coarse oracle
```

的：

```text
min
max
out_of_[0,1]_ratio
```

按：

```text
RGB
metallic
roughness
alpha
```

统计。

这是为了回答：

> 保留 shared coarse 后，PBR overshoot 是否比 null-only 减少、增加还是基本不变？

不要人为修。

---

# 21. Boundary / uncovered-row diagnostics

继续检查：

```text
P_h uncovered hidden rows
```

对这些 row 标记：

```text
covered_by_projector
uncovered_by_projector
```

同时统计 shared donor coverage。

生成：

```text
boundary_uncovered_stats.json
```

检查：

- uncovered projector rows 是否集中在 observed/hidden boundary；
- shared consensus 是否也在这些区域缺 donor；
- 是否产生明显 field discontinuity。

---

# 22. 第一阶段先跑局部 3×3 neighborhood

为了验证 implementation correctness，先不要 full 48。

以 Tile26/27 为中心，使用其 stride=512 overlap neighborhood。

优先尝试：

```text
18,19,20,
25,26,27,
32,33,34
```

如果实际 tile layout 对应关系不同，以真实 7×7 layout 自动查询 26/27 的一环 overlap neighbors。

不要硬编码错误邻居。

---

# 23. Phase-A correctness

局部 neighborhood 必须检查：

### P/A

\[
P^TD\approx0.
\]

### shared range

\[
P^T(S-P A S)\approx0.
\]

### algebraic identity

\[
G+C^{shared}+D
=
H-(C-C^{shared}).
\]

### self transport

对于 donor=self：

\[
T_{i\to i}C_i
\approx
C_i
\]

必须成立。

这是 canonical query pipeline 最重要的 correctness test。

### donor order invariant

改变 donor tile 遍历顺序，结果必须一致。

---

# 24. Phase-A 可视化

至少输出 Tile26/27：

```text
G
PureHR H
Delta = H-G
C = PA Delta
D = (I-PA) Delta
S = raw cross-tile consensus
C_shared = PA S
C_private = C-C_shared
Y_null
Y_shared
```

生成：

```text
RGB channel sheet
metallic sheet
roughness sheet
alpha sheet
```

另外生成：

```text
front/back comparison
```

---

# 25. Phase-B full oracle

只有 Phase-A correctness 全部通过后：

对与正式 baseline 完全一致的全部 valid tiles 运行。

本阶段仍然：

```text
NO flow
NO encode
NO per-step guidance
```

---

# 26. Stitch 与 render

必须复用之前正式实验完全相同的：

```text
tile -> global
overlap ownership
stitching
render
```

逻辑。

禁止因为 oracle 实验修改 stitching。

Geometry 对所有 variants 必须完全相同。

验证：

```text
vertices identical
faces identical
```

仅 PBR attrs 改变。

---

# 27. Full variants

至少生成：

```text
1. Global baseline
2. PureHR
3. Null-only final-field oracle
4. Shared-coarse final-field oracle
```

如果已有 current Gaussian：

```text
5. current Gaussian
```

只作为 reference 展示。

不要重新生成它。

---

# 28. Render 输出

全部使用同一 camera / envmap / renderer。

生成：

### aligned input view

```text
4096 × 4096
```

### fixed multiview

```text
front
back
left
right
top
bottom
```

### turntable

```text
24 frames
```

### contact sheets

必须有：

```text
back_view_4variants.png
front_back_4variants.png
RGB_front_back.png
roughness_front_back.png
metallic_front_back.png
```

---

# 29. 图像指标

Aligned input view 对 canonical reference：

```text
PSNR
SSIM
LPIPS
```

对：

```text
Global
PureHR
Null-only oracle
Shared-coarse oracle
```

统一计算。

注意：

本实验主要看 hidden/back。

输入视角指标不是唯一判断依据。

---

# 30. 最重要的成功判据

实验最终必须明确回答：

## Q1

Final PureHR 的 coarse increment：

\[
C_i=P_iA_i(H_i-G_i)
\]

是否存在明显 cross-tile agreement？

---

## Q2

有多少 coarse energy 是 shared：

\[
\frac{\|C^{shared}\|^2}{\|C\|^2}?
\]

特别是 RGB。

---

## Q3

Old null-only：

\[
G+D
\]

为什么视觉接近 Global？

是否可以直接从删除的 coarse energy 得到解释？

---

## Q4

新的：

\[
G+C^{shared}+D
\]

是否能够比：

\[
G+D
\]

明显保留 PureHR 背面的颜色/材质改善？

---

## Q5

它是否恢复了原来 PureHR 的随机灰化 / 材质漂移？

如果没有，则支持：

> tile-private coarse difference 才是主要 hallucination。

---

## Q6

Shared-coarse 是否减少 PBR OOB？

如果没有：

明确记录，后续再单独研究 constrained PBR projection。

本轮禁止用 clamp 修。

---

# 31. 必须给出的最终数学统计

汇总全部 tile：

\[
\boxed{
R_{shared}^{RGB}
=
\frac{
\sum_i\|C_{i}^{shared}\|^2
}{
\sum_i\|C_i\|^2
}
}
\]

\[
\boxed{
R_{private}^{RGB}
=
\frac{
\sum_i\|C_i-C_i^{shared}\|^2
}{
\sum_i\|C_i\|^2
}
}
\]

其它 PBR channels 同样。

另外计算：

\[
\boxed{
R_{preserve}
=
\frac{
\|Y^{shared}-G\|
}{
\|H-G\|+\epsilon
}
}
\]

与：

\[
\boxed{
R_{null}
=
\frac{
\|Y^{null}-G\|
}{
\|H-G\|+\epsilon
}
}
\]

用于量化：

> 新方法实际保留了多少 HR-G variation。

---

# 32. 本轮不要做 fixed-point encoder transport

即使 shared oracle 成功，也不要马上：

```text
encode
decode
flow
```

本轮只判断：

\[
\boxed{
\text{shared-coarse field target 本身是否正确}
}
\]

只有 oracle 明显优于 null-only 后，下一实验才测试：

\[
z^{k+1}
=
z^k+
E(Y^\star)-E(D(z^k)).
\]

不要把两个研究问题混在一次实验里。

---

# 33. 输出目录

建议：

```text
outputs/pbr_shared_coarse_oracle/
```

输出：

```text
SHARED_COARSE_ORACLE_REPORT.md
summary.json

coarse_agreement.json
shared_private_energy.json
pbr_domain_stats.json
boundary_uncovered_stats.json

phaseA_neighborhood/
full_oracle/

fields/
  tile_XX/
    H.pt
    G.pt
    Delta.pt
    coarse.pt
    detail.pt
    raw_consensus.pt
    shared_coarse.pt
    private_coarse.pt
    null_target.pt
    shared_target.pt

renders/
metrics.csv
multiview_metrics.json
```

---

# 34. 最终报告不要只写“效果更好/更差”

必须给出因果分析：

### 如果 shared oracle 明显优于 null-only

结论：

\[
\boxed{
\text{“all coarse difference is unreliable” 假设被否定}
}
\]

支持：

\[
\boxed{
H-G
=
C^{shared}
+
C^{private}
+
D
}
\]

并只删除：

\[
C^{private}.
\]

---

### 如果 shared oracle 和 null-only 几乎一样

说明：

\[
C^{shared}
\]

本身很小。

即 final PureHR coarse change 缺少 cross-tile reproducibility。

那么 shared-coarse 不是主要解法。

---

### 如果 shared oracle 接近 PureHR 但重新出现灰化/乱材质

说明：

cross-tile L2 consensus 不足以区分：

```text
shared semantic improvement
vs
shared systematic hallucination
```

下一步才考虑更强的 robust / semantic constraint。

---

### 如果 field oracle 很好，但未来 encode/flow 后变差

则问题已经明确定位为：

\[
\boxed{
\text{PBR-field → latent endpoint transport}
}
\]

而不是 MRA decomposition。

---

# 35. 最核心原则

本实验不是为了立即获得新的最终 SOTA 数字。

它是一个 hypothesis test：

\[
\boxed{
\text{HR coarse variation 是否可以通过 cross-tile reproducibility 判断可信度？}
}
\]

整个实验必须做到：

\[
\boxed{
\text{NO FLOW}
+
\text{NO ENCODE}
+
\text{NO HAND-TUNED WEIGHTS}
}
\]

只验证：

\[
\boxed{
Y_i^{shared}
=
H_i-
P_i(c_i-c_i^{shared})
}
\]

是否比：

\[
\boxed{
Y_i^{null}
=
H_i-P_ic_i
}
\]

更符合我们想要的 training-free 3D PBR super-resolution 定义。
```
