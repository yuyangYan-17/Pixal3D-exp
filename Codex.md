# 任务目标

修改并运行当前代码库完成一个新的、独立的 **Global C1024 Common-Field POD Diagnostic**。

本轮只回答：

\[
D_i=H_i-G
\]

这些不同 PureHR difference fields，在**同一套 Global baseline C1024 active O-Voxel query support**上，是否存在跨 tile 的共同低维变化方向。

本轮禁止：

- Wavelet；
- C256；
- \(P^\dagger\)；
- LSMR projector；
- range/null；
- MRA；
- Gaussian fusion；
- PBR averaging；
- flow guidance；
- re-encode；
- Euler；
- 修改任何生成轨迹。

这是纯 final-field diagnostic。

---

# 1. 不要修改旧方法语义

不要继续往：

`pixal3d_shared_coarse_oracle.py`

中塞新 POD 方法。

新建：

```text
pixal3d_global_c1024_common_field_pod.py
```

新建测试：

```text
tests/test_global_c1024_common_field_pod.py
```

可以复用旧代码中已经验证的：

- official tile layout；
- `PHASE_A_TILE_IDS`；
- PureHR candidate discovery/provenance validation；
- `TileCameraTransform`；
- global↔local transform；
- `MeshWithVoxel.query_attrs`；
- final PureHR endpoint decode；
- fixed-shape context loading。

但不要调用：

- `build_prolongation()` 做 C256 projector；
- `solve_direct_lsmr()`；
- coarse/detail decomposition；
- `C_shared/C_private` 旧逻辑。

---

# 2. 实验 tile

第一轮严格使用：

```python
PHASE_A_TILE_IDS = {
    18, 19, 20,
    25, 26, 27,
    32, 33, 34,
}
```

先 preflight。

PureHR 输入只能是：

1. 合法 cached final PureHR PBR field；
2. 或合法 final `pure_HR_endpoint.pt` decode 一次得到的 final field。

禁止使用：

- range-null；
- MRA；
- projector；
- guided；
- per-step corrected endpoint；
- trajectory intermediate；
- Gaussian field。

如果某个 Phase-A tile 缺合法 PureHR：

1. 先检查所有现有 output roots；
2. 按 `tile_camera.json.box` 匹配 legacy tile id；
3. 仍缺失时，使用现有 PureHR generation route 只重新生成缺失 tile；
4. seed 和模型配置必须与现有 PureHR baseline 一致；
5. 生成后重新跑 provenance preflight。

不能用其它 variant 替代。

---

# 3. 找到真正的 Global baseline C1024 O-Voxel

必须找到本实验对应的 **Global baseline final C1024 `MeshWithVoxel`**。

它必须至少包含：

```text
coords
attrs
origin
voxel_size
voxel_shape
layout
```

其中：

```text
resolution = 1024
attrs channels = 6
```

禁止把某个 tile 的：

```text
global_pbr_reference.pt
```

误认为统一 global support。

`global_pbr_reference.pt` 只是已有 tile-local query reference，可以用于 correctness cross-check。

如果找不到真正的 global baseline C1024 O-Voxel cache：

- 优先从现有 global baseline final artifact 恢复；
- 必要时仅重新执行 global baseline decode；
- 不允许重新生成一个不同 seed / 不同配置的 G。

在报告中记录 global artifact 的绝对路径、hash、coord count 和 provenance。

---

# 4. Global C1024 统一 query support

设 global active voxel coords 为：

\[
c_n\in\mathbb Z^3.
\]

根据 `MeshWithVoxel` 自己的：

```text
origin
voxel_size
```

得到每个 active voxel center：

\[
x_n
=
origin
+
(c_n+0.5)\,voxel\_size.
\]

得到：

\[
X_G=\{x_n\}_{n=1}^{N_G}.
\]

这就是本轮唯一允许的 analysis support。

保存：

```text
global_support/coords.pt
global_support/points.pt
global_support/G.pt
global_support/meta.json
```

其中：

```text
coords.pt   [N_G,3] int32
points.pt   [N_G,3] float32
G.pt        [N_G,6] float32
```

`G.pt` 必须通过 global `MeshWithVoxel.query_attrs(points)` 得到，而不是直接假定等于 `attrs`。

---

# 5. Global support self-query correctness

必须验证：

```python
G_query = global_mesh.query_attrs(global_points)
```

与 global coefficient `attrs` 在自身 voxel centers 上是否一致。

输出：

```text
max_abs
mean_abs
relative_l2
```

如果不一致：

不要偷偷使用 `attrs` 替代。

报告真实差异，并继续统一使用：

```text
G_query
```

作为场值定义。

---

# 6. 将同一批 global points 映射到每个 tile

对每个 tile \(i\)：

首先将：

\[
x_n
\]

从 global O-Voxel canonical coordinate 转换到现有 camera transform 所要求的 global-q coordinate。

严格遵守仓库当前 transform convention。

当前已有代码中 global mesh vertex 进入：

```python
_global_q_to_local_q(...)
```

前使用了与 `mesh_scale` 一致的 q scaling。

不要自行猜坐标单位。

必须复用并测试现有：

```text
_global_q_to_local_q
_local_q_to_global_q
```

得到：

```text
global baseline C1024 point
    -> q_global
    -> q_local
    -> local canonical query point
```

对抽样点做 round-trip：

```text
global -> local -> global
```

输出：

```text
max_abs
mean_abs
relative_l2
```

Tile26、Tile27 必须单独列出。

---

# 7. 不允许仅靠 local cube 判断 coverage

每个 global baseline point 对 tile \(i\) 是否有效，必须依据 tile 自己的 **C1024 sparse interpolation support**。

实现一个新的、语义明确的 helper：

```python
build_sparse_query_matrix(
    active_coords,
    query_points,
    resolution=1024,
)
```

它实现的只是：

\[
\text{active O-Voxel coefficients}
\rightarrow
\text{指定 query points 的 trilinear field value}.
\]

要求：

- 8-neighbor trilinear；
- missing-neighbor renormalization；
- 与 `MeshWithVoxel.query_attrs` 完全一致；
- 不求逆；
- 不做 projector；
- 不涉及 C256。

返回：

```text
Q_i       scipy CSR [N_G, N_i]
valid_i   bool [N_G]
metadata
```

其中：

\[
M_i(n)=valid_i(n)
\]

表示 global baseline voxel \(n\) 是否真正能够由 tile \(i\) 的 sparse C1024 field query。

没有 valid support 的位置是 missing：

```text
M_i = 0
```

不是 PBR=0。

---

# 8. Query operator correctness

至少对 Tile26、Tile27：

随机抽取不少于 10k 个 `valid_i=True` 的 common global support points。

比较：

```python
Q_i @ attrs_i
```

与：

```python
tile_mesh.query_attrs(local_points)
```

分别对：

- RGB；
- metallic；
- roughness；
- alpha

报告：

```text
max_abs
mean_abs
relative_l2
```

目标是 float32 interpolation precision 量级。

若明显不一致，停止 POD，不得继续。

---

# 9. 构造统一 H、G、Delta

对每个 Phase-A tile：

\[
H_i(n)
=
Q_i C_i,
\]

其中：

- \(C_i\)：该 PureHR tile C1024 PBR coefficients；
- \(Q_i\)：刚才的 sparse query matrix。

Global：

\[
G(n)
\]

来自统一 global baseline field。

定义：

\[
D_i(n)=H_i(n)-G(n).
\]

只在：

\[
M_i(n)=1
\]

处有效。

保存：

```text
tiles/tile_XX/H_on_global.pt
tiles/tile_XX/Delta_on_global.pt
tiles/tile_XX/valid_mask.pt
tiles/tile_XX/query_meta.json
```

不要给 invalid rows 填有意义的数值。

如果 tensor 存储需要 placeholder，可使用 NaN，并且所有后续分析必须显式 mask。

禁止使用 0 作为 missing sentinel。

---

# 10. observed / hidden mask 也投影到同一 support

复用现有 local：

```text
observed_mask.pt
hidden_mask.pt
```

通过同一个 `Q_i` 查询。

定义：

\[
o_i(n)=Q_i\,m_i^{obs},
\]

\[
h_i(n)=Q_i\,m_i^{hid}.
\]

不要直接用 0.5 threshold。

使用严格分类：

```text
observed: observed_fraction >= 1 - eps
hidden:   hidden_fraction   >= 1 - eps
mixed:    其它 valid point
invalid:  no sparse support
```

`eps` 只允许使用浮点容差，例如 `1e-6`。

保存：

```text
observed_mask_global.pt
hidden_mask_global.pt
mixed_mask_global.pt
```

并验证：

```text
observed / hidden / mixed / invalid
```

计数。

---

# 11. 先输出 coverage statistics

构造：

\[
C(n)=\sum_i M_i(n).
\]

保存：

```text
coverage_count.pt
coverage_histogram.json
pairwise_overlap.csv
```

`pairwise_overlap.csv` 对每个：

\[
(i,j)
\]

输出：

```text
tile_i
tile_j
common_valid_voxels
common_hidden_voxels
common_observed_voxels
```

再输出 9×9 heatmap：

```text
pairwise_overlap_all.png
pairwise_overlap_hidden.png
pairwise_overlap_observed.png
```

这一步是 POD 之前的 mandatory preflight。

---

# 12. 第一轮 POD 不做 gappy completion

不要：

- missing fill 0；
- EM-PCA；
- matrix completion；
- gappy reconstruction；
- overlap-chain propagation。

第一轮只分析真正的 direct common support。

主要四个 quartet：

```python
QUARTETS = [
    (18, 19, 25, 26),
    (19, 20, 26, 27),
    (25, 26, 32, 33),
    (26, 27, 33, 34),
]
```

但 quartet 的 2D layout 只用于定义 candidate。

真实 analysis support 是：

\[
M_R
=
\bigwedge_{i\in R}M_i.
\]

只有 `M_R=True` 的 global baseline C1024 O-Voxels 才能进入该 quartet 的 POD。

输出每个 quartet：

```text
common_valid_count
common_hidden_count
common_observed_count
mixed_count
```

如果某个 region 样本太少：

只报告 N/A。

不要伪造结果。

---

# 13. POD 数学定义

分别对：

```text
RGB       channels 0:3
metallic  channel 3
roughness channel 4
alpha     channel 5
```

独立分析。

不要首先构造 all-6 headline metric。

对 quartet \(R\) 和 channel group \(g\)：

只取共同 support：

\[
X_R=\{n:M_R(n)=1\}.
\]

对 tile \(i\in R\)：

\[
x_i
=
\operatorname{vec}
\left(
D_i[X_R,g]
\right).
\]

组成：

\[
X=
[x_1,x_2,x_3,x_4].
\]

形状：

```text
[num_common_voxels * group_channels, 4]
```

禁止 mean centering。

做：

\[
X=U\Sigma V^\top.
\]

计算：

\[
\rho_1
=
\frac{\sigma_1^2}{\sum_r\sigma_r^2},
\]

\[
\rho_{1:2}
=
\frac{\sigma_1^2+\sigma_2^2}
{\sum_r\sigma_r^2}.
\]

保存完整：

```text
sigma
energy_ratio
cumulative_energy_ratio
V
```

---

# 14. 同时做 directional POD

为了避免一个 amplitude 特别大的 tile 独自支配 POD，再构造：

\[
\hat x_i
=
\frac{x_i}{\|x_i\|_2}.
\]

只有 norm 非零时参与。

定义：

\[
\hat X=
[\hat x_1,\hat x_2,\hat x_3,\hat x_4].
\]

再次做 uncentered SVD。

分别报告：

```text
raw_rho1
raw_rho12
directional_rho1
directional_rho12
```

注意：

directional POD 只用于判断：

```text
same structure / different amplitude
```

不能替代 raw energy POD。

---

# 15. Pairwise cosine 是 mandatory

对每个 quartet/channel：

\[
C_{ij}
=
\frac{x_i^\top x_j}
{\|x_i\|_2\|x_j\|_2}.
\]

输出 4×4 cosine matrix。

生成：

```text
cosine_RGB.png
cosine_metallic.png
cosine_roughness.png
cosine_alpha.png
```

并保存原始 CSV/JSON。

这是判断类似：

\[
0.4u,\ 0.7u,\ 0.5u
\]

是否存在的最直接指标。

---

# 16. all / hidden / observed 三套分析

每个 quartet 都至少做：

```text
ALL_VALID
ALL_HIDDEN
ALL_OBSERVED
```

定义：

### ALL_VALID

所有四个 tile 均 valid。

### ALL_HIDDEN

所有四个 tile 对该 global baseline O-Voxel 都严格 classified hidden。

### ALL_OBSERVED

所有四个 tile 都严格 classified observed。

mixed visibility 不放进 hidden/observed POD。

如果 hidden/observed 样本数不足，报告 N/A。

不要降低条件来凑数据。

---

# 17. Leave-One-Tile-Out stability

每个 4-tile POD 再做一次 LOTO。

例如：

```text
26,27,33,34
```

分别计算：

```text
omit26 -> POD(27,33,34)
omit27 -> POD(26,33,34)
omit33 -> POD(26,27,34)
omit34 -> POD(26,27,33)
```

比较 full POD 与 LOTO 的：

```text
rho1
rho12
```

以及 tile-space 第一 eigenvector / right singular vector 的稳定性：

\[
s_{\rm LOTO}
=
|v_1^{full\top}v_1^{LOTO}|.
\]

如果维度不同，则把 full vector restrict 到对应剩余 tile 后重新 normalize 再比较。

这个测试用于排除：

```text
rho1 很高只是被某一个异常 tile 支配
```

的情况。

不要设置人为 pass/fail 阈值，只报告数据。

---

# 18. 数值实现建议

由于 quartet 只有 4 columns，没有必要对巨大矩阵直接做 full SVD。

优先累计：

\[
K=X^\top X
\]

得到 4×4 Gram matrix。

然后：

```python
torch.linalg.eigh(K)
```

得到：

\[
\sigma_r^2.
\]

要求：

- accumulation 使用 float64；
- 输入 field 可以 float32；
- Gram/eigenanalysis 使用 float64；
- 不需要 GPU 才能完成 POD 本身。

若需要保存 spatial PC1：

\[
u_1
=
\frac{Xv_1}{\sigma_1}.
\]

只计算前两个 mode：

```text
PC1
PC2
```

不要求完整 U。

---

# 19. 可视化

每个 quartet/domain/channel 至少输出：

### 1. Scree plot

```text
pod_spectrum.png
```

显示：

```text
rho1
rho2
rho3
rho4
```

### 2. cumulative energy

```text
pod_cumulative.png
```

### 3. pairwise cosine heatmap

```text
pairwise_cosine.png
```

### 4. per-tile amplitude

显示：

\[
\|x_i\|_2
\]

和第一 POD coefficient。

### 5. global C1024 support coverage

至少提供：

```text
coverage_count
quartet mask
hidden mask
observed mask
```

的 physical O-Voxel visualization。

不要把 POD signed mode 当成合法 PBR 渲染。

如果可视化 PC1，只显示：

```text
signed scalar/component magnitude
```

并明确标注：

```text
POD mode, not a PBR material
```

---

# 20. 必须加入的测试

`tests/test_global_c1024_common_field_pod.py`

至少包含：

## Test A — common support

两个不同 sparse active supports query 到同一批 points，输出 row order 必须完全一致。

## Test B — missing is not zero

无 support row：

```text
valid=False
```

不能因为输出 tensor placeholder=0 就进入 POD。

## Test C — query matrix

synthetic sparse C1024 support 上：

```python
Q @ attrs
```

与手工 sparse trilinear + missing-neighbor renormalization 一致。

## Test D — own-center query

active voxel center 上 query 的行为与 `MeshWithVoxel.query_attrs` 一致。

## Test E — uncentered POD

构造：

\[
x_1=0.4u,\quad
x_2=0.7u,\quad
x_3=0.5u.
\]

必须得到：

```text
rho1 ≈ 1
directional_rho1 ≈ 1
pairwise cosine ≈ 1
```

## Test F — private direction

构造：

\[
x_1=u,\quad
x_2=u,\quad
x_3=v,\quad u\perp v.
\]

POD 不得错误返回 rank-1 perfect commonality。

## Test G — no mean centering

验证 implementation 不会自动执行：

```python
X -= X.mean(...)
```

---

# 21. Correctness gate

正式 Phase-A POD 前，必须先在 Tile26/27 做：

```text
global/local roundtrip
Q @ attrs vs query_attrs
G reference consistency
mask counts
pairwise overlap count
```

把结果写入：

```text
correctness_gate.json
```

只有 correctness gate 通过后才继续 quartet POD。

如果失败：

停止实验并报告原因。

---

# 22. 正式运行

使用 CUDA4 对需要 decode/query 的部分运行。

POD/eigendecomposition 可以转 CPU float64。

保持当前实验 seed=42。

输出目录：

```text
outputs/global_c1024_common_field_pod_phaseA_cuda4/
```

不要覆盖任何旧实验。

---

# 23. 输出目录结构

最终至少：

```text
outputs/global_c1024_common_field_pod_phaseA_cuda4/
├── preflight.json
├── correctness_gate.json
├── global_support/
│   ├── coords.pt
│   ├── points.pt
│   ├── G.pt
│   └── meta.json
├── tiles/
│   ├── tile_18/
│   ├── ...
│   └── tile_34/
├── coverage/
│   ├── coverage_count.pt
│   ├── coverage_histogram.json
│   ├── pairwise_overlap.csv
│   └── *.png
├── quartets/
│   ├── 18_19_25_26/
│   ├── 19_20_26_27/
│   ├── 25_26_32_33/
│   └── 26_27_33_34/
├── pod_metrics.csv
├── pairwise_cosine.csv
├── loto_stability.csv
├── summary.json
└── GLOBAL_C1024_COMMON_FIELD_POD_REPORT.md
```

---

# 24. 报告必须回答的问题

最终报告不要只写“实验跑通”。

必须逐项回答：

1. Global baseline C1024 active O-Voxel 一共有多少？
2. 每个 Phase-A tile 能覆盖其中多少？
3. 9×9 direct physical overlap matrix 是什么？
4. 四个 quartet 各自真正有多少共同 C1024 samples？
5. hidden / observed 分别有多少共同 samples？
6. RGB 的 pairwise cosine 分布是什么？
7. metallic / roughness / alpha 分别怎样？
8. 各 quartet：

\[
\rho_1,\qquad
\rho_{1:2}
\]

是多少？
9. directional POD 与 raw POD 差异多大？
10. LOTO 后 dominant mode 是否稳定？
11. hidden region 是否明显存在低维 common \(H-G\) structure？
12. observed 与 hidden 的 POD spectrum 有什么差别？
13. 是否存在：

\[
H_i-G \approx a_i u
\]

这种“共享 direction、不同 amplitude”的现象？
14. 哪些 tile 是明显 outlier？
15. 当前证据是否足以进入下一阶段 Wavelet × POD？

不要预设结论。

---

# 25. 下一阶段的进入条件

本轮不设置类似：

```text
rho1 > 0.7 => success
```

这种人为阈值。

只根据完整数据判断。

如果结果呈现：

- 多个 quartet；
- 尤其 hidden region；
- pairwise cosine 稳定为正且较高；
- \(\rho_1\) 或 \(\rho_{1:2}\) 显著集中；
- LOTO 后仍稳定；

则报告：

```text
存在值得进一步做 multiscale analysis 的跨场 common subspace evidence
```

下一阶段才开始 Wavelet × POD。

如果 spectrum 平坦、pairwise directions 不稳定、LOTO 崩溃：

则明确报告：

```text
当前 final PureHR hidden difference fields 没有稳定的 direct-overlap common low-rank structure
```

此时不要擅自实现 wavelet/fusion。

---

# 26. 测试与最终汇报

依次执行：

```bash
pytest -q tests/test_global_c1024_common_field_pod.py
```

然后：

```bash
pytest -q tests/test_shared_coarse_oracle.py \
          tests/test_global_c1024_common_field_pod.py
```

最后运行仓库全量 tests。

报告：

```text
passed
failed
historical/unrelated failure
```

必须严格区分。

最终回复我：

1. 修改文件列表；
2. 测试结果；
3. 实际运行命令；
4. GPU；
5. PureHR provenance；
6. global baseline provenance；
7. correctness gate 数值；
8. coverage statistics；
9. 四个 quartet 的 POD 表格；
10. hidden/observed 对比；
11. pairwise cosine；
12. LOTO stability；
13. 关键可视化路径；
14. `summary.json`；
15. `GLOBAL_C1024_COMMON_FIELD_POD_REPORT.md`；
16. 最后用一段话判断下一步是否值得上 Wavelet。

本轮不要实现任何 fusion。