# Training-free 方法候选与预注册失败判据

日期：2026-07-27

共同约束：

- C64 flow support 使用 endpoint coordinate；
- decoded C1024 O-Voxel 使用 center coordinate
  `q=2*(coord+0.5)/1024-1`；
- 只用已验证的逐点 global↔local camera transform；
- 禁止 bbox/centroid normalization、clamp、dense `1024^3`；
- SparseTensor row number 不代表 correspondence；
- 固定 tiles 24/26/27，先小实验，再扩展 tile/输入图像；
- global geometry、material、topology 的改动必须能单独开关。

## 1. Strict matched decoded O-Voxel material residual

### 假设

global-guided local decoder 已含有价值的局部 base color。早期彩色点主要来自
partial material lattice 替换完整 global lattice、support 未严格匹配及
local alpha 被错误带入。保留完整 global field，只修改严格对应行，可以
直接检验 local material 是否可用于统一 global 模型。

### 数据流

```text
global MeshWithVoxel
  (Vg, Fg, Sg1024, Ag)

tile MeshWithVoxel
  local O-Voxel center
  → exact local-to-global camera inverse
  → continuous global q
  → global C1024 hash
  → strict matched local base color

Cf[g] = Cg[g] + alpha_tex * (Ct[l] - Cg[g])
```

不变项：

- `Vg/Fg/Sg1024`；
- unmatched global attrs；
- metallic、roughness、alpha；
- 首轮只改 `base_color[0:3]`。

同一 global row 有多个 tile candidate 时先做 tile-interior
winner-take-all，不先平均颜色。完整 global lattice 始终存在，因此不会再用
normal fallback 填补 material holes。

### 复杂度

- coordinate hash/sort-search：`O(Ng+Nt)` 或 `O(N log N)`；
- 不构造 `Ng×Nt` distance matrix；
- 每 tile 逐个加载/映射/释放；
- 主要显存仍是现有 decoder 约 61 GB；融合本身可离线在 CPU 完成。

### 最小顺序消融

先冻结 strict exact C1024 match、tile-margin WTA：

| 组 | `alpha_tex` |
|---|---:|
| E0 | 0 |
| E1 | .25 |
| E2 | .50 |
| E3 | 1.00 |

再对最佳非零 alpha 测：

- exact quantized center + continuous distance `<=.25 voxel`；
- `<=.5 voxel`；
- `<=1.0 voxel`。

负对照：将 matched global row 故意平移一个 voxel。若它与正确 mapping 一样，
说明材质没有真正进入 renderer 或 source map 有 bug。

### 必存诊断

- matched pair、continuous q、distance、tile ID、tile margin；
- exact/multi-to-one/rejected 数；
- tiles 26/27 的 color disagreement；
- `Cglobal/Clocal/residual/Cfused`；
- source-ID、residual norm、visible coverage render；
- geometry/support/alpha before-after hash；
- global、三块 crop、轻微新视角；
- timing、CPU/GPU peak memory。

### 实现错误判据

- global vertices/faces/coords 或 alpha 任一改变；
- alpha=0 不能复现 control attrs/render；
- non-matched attr 或 non-base-color channel 改变；
- source-ID 与实际 RGB 变化区域不一致。

### 方法停止判据

- 所有非零 alpha 相对 alpha=0 的 mean 三项中至少两项变差；
- 任一 tile 损失超过 `.2 dB` 或 LPIPS 增加超过 `.01`，且无 coverage 原因；
- 只靠明显模糊提升 PSNR/SSIM；
- overlap seam 或新视角 texture sliding。

只有统一 global 结果超过 global control，才能称为统一方法成功；超过 local
Baseline D 是更高一级目标。

## 2. Multi-tile agreement + narrow-band controlled topology birth

### 假设

可信 local-only geometry 应靠近 global surface、被至少两个重叠 tile 提出，
并在 position/normal/depth 上一致。单 tile 边缘独有且远离 global surface
的点更可能是 floater 或 double shell。

### 数据流

```text
local decoded surface
  → per-vertex exact local-to-global transform
  → global sparse hash
  → distance/normal/depth/multi-tile gates
  → accepted connected components
  → sparse narrow-band TSDF/surfel patch
  → local remesh + boundary weld
```

只在 accepted component 建 sparse band，组件外 global mesh bitwise 保留。
第一轮固定 global material，只评价 clay/depth/normal/silhouette；几何通过后
才用方法 1 给新表面上色。

### Gate

- local-only；
- `0 < distance_to_global_surface <= r_birth`；
- tiles 26/27 至少两个来源在 `r_agree=1 voxel` 内；
- normal cosine `>=.9`；
- visible depth order 一致；
- tile edge margin `>=128 px`；
- 26-neighbor component 与 global/matched surface 相连。

tile 24 没有第二个重叠观测时必须为 no-birth control。

### 最小实验

- agreement count `>=1` vs `>=2`；
- `r_birth∈{1,2,4}` global C1024 voxels；
- 只在 26/27 overlap 开启。

### 失败判据

- accepted disconnected floaters `>1%`；
- 未修改区 silhouette 改变 `>.5%` pixels；
- peel double-surface ratio 增加 `>1%`；
- depth/normal 只在输入视角改善；
- patch boundary crack；
- 资源不能扩展到所有有效 tile。

## 3. Fixed-support differentiable material TTO

### 假设

即使 local sparse support 不易直接对应，固定 global surface 的可见 O-Voxel
base color 仍可通过 tile render loss 更新。变量绑定在 3D support，不是
二维 overlay。

### 数据流

```text
fixed global V/F/S1024/alpha
  → visible global material rows
  → optimize only delta_base_color
  → differentiable sparse trilinear query + nvdiffrast
  → joint tiles 24/26/27 RGB loss
```

建议 loss：

- foreground Charbonnier/L1；
- 小权重 SSIM；
- delta-to-global anchor；
- sparse surface graph TV；
- tile-center weighting。

所有 Pixal3D 权重冻结，不优化 metallic/roughness/alpha/geometry。

### 最小实验

1. one-step gradient smoke，确认 sparse material query 对 attrs 有梯度；
2. 512 resolution、SSAA1、30 steps；
3. global init vs 方法 1 strict-graft init；
4. 固定 10% foreground pixel 作同视角 holdout；
5. 最终用统一 renderer 设置和轻微新视角评价。

### 失败判据

- attrs 无梯度或梯度只到异常少的 rows；
- train pixels 上升但 holdout 下降；
- 更新颜色超过 20% 饱和到 0/1；
- 新视角出现 view-baked sliding texture；
- 额外显存 `>10 GB` 或三 tile `>20 min` 且无明显收益。

## 4. Occupancy-aware RAHT material residual

### 假设

local 高频应定义在不规则 occupied global surface 上；dense Haar 把空节点当
零，主要响应 sparse mask 边界。RAHT 可在同一 absolute occupied octree 上
分离 material residual 的 coarse/fine scale。

### 数据流

```text
strict matched delta_color on global S1024
  → occupied-coordinate Morton/octree
  → RAHT
  → keep global low-pass
  → inject reliable fine local coefficients
  → inverse RAHT on original global support
```

RAHT 不负责 correspondence，只消费方法 1 已验证的 matched residual。

### 最小实验

- direct residual；
- only finest high-pass；
- finest two high-pass levels；
- subtree observation coverage threshold `.75`；
- 使用相同 alpha 和 tile WTA。

### 失败判据

- forward/inverse roundtrip max error `>1e-6`；
- `>50%` high-pass energy 落在 observation/tile-mask boundary；
- ringing/seam；
- 不优于相同 alpha 的 direct residual；
- octree 构建资源不可扩展。

## 5. Time-aligned texture-flow residual

### 假设

只有在相同 `x_t`、noise、timestep、C64 support 和 row order 上分别计算
global/tile condition velocity，`delta_v` 才可被解释为 condition residual。
旧实验的独立 trajectory velocity average 不满足该条件。

### 数据流

```text
shared projected C64, x_t, noise, t, shape concat
  → v_global_condition
  → v_tile_condition
  → delta_v
  → v_global + alpha_flow(t) * gate * clip(delta_v)
  → same sampler update
```

首轮只动 texture flow、最后 1–2 step，geometry 固定。

### 最小实验

- `alpha_flow∈{.1,.25,.5}`；
- residual norm clip 到 `.25*max(norm(v_global),eps)`；
- 保存 timestep norm/cosine/support hash。

### 失败判据

- 任一步 support hash/order 不同；
- `norm(delta_v)/norm(v_global)>2` 的 token 超过 25%；
- velocity cosine `<0` 的 token 超过 25%；
- texture-only 改变 depth/silhouette；
- 成本近翻倍且不优于 decoded material residual。

该方案后置到 C1024 correspondence 和 direct material residual 均验证之后。

## 6. 第一步实现选择

选择方法 1，并进一步缩小为：

> local decoded base color → C1024 center exact camera inverse → existing
> global C1024 key → tile-margin WTA → 完整 global material lattice 上的
> base-color residual。

禁止同时加入 topology birth、flow、RAHT、颜色对齐或 differentiable
optimization。该实验能以最低风险回答：

1. local material 是否真正绑定 global surface；
2. 彩色点是否来自 partial lattice；
3. local→global 量化的 coverage/collision 是否已使信息不可用；
4. local material source 质量本身是否优于 global；
5. 若失败，下一步应改 support resolution、source selection，还是停止材质线。
