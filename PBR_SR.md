# Global C4096 Carrier + Local C1024 多视角/多 Tile 同步高斯 PBR Flow

## 0. 方案结论

本方案保留已经验证的 local geometry 路径和每个 context 独立的图像条件，但恢复旧流程中真正有效的逐步 PBR 共识：

```text
所有 context 独立 flow forward
→ 所有 predicted endpoint 解码为 local C1024 PBR
→ 仅在 PBR donor 查询阶段使用精确可见性
→ 在 global C4096 carrier 上做多 view/tile Gaussian PBR fusion
→ 将 global 共识连续查询回每个 context 的完整 local C1024 support
→ official PBR encoder re-encode
→ 恢复为 normalized texture SLat endpoint
→ native _xstart_to_pred
→ 所有 context 同步 Euler 更新
```

最重要的边界是：

1. flow model 本身永远接收当前 tile 中所有投影 mesh 形成的完整 local support，包括正面、背面、遮挡面；不使用 visibility 删除、冻结、置零或 mask SLat；
2. visibility 只控制某个 decoded PBR 值能不能作为 donor，不控制 flow support，也不控制 target 能不能接收其他视角的共识；
3. 每个 context 只使用自己的 full-view global DINO、自己的 tile projected DINO、自己的 shape SLat 和自己的 texture-flow state；
4. 不拼接或平均不同 context 的 DINO token，不融合 texture SLat，不平均 velocity；跨 context 只交换已经 decode 到 PBR 空间的值；
5. texture flow state 和 flow endpoint 始终位于模型规定的 normalized SLat 空间；decode 前必须 inverse-normalize，PBR encode 后必须 normalize，再交回 flow sampler；
6. 所有 step 使用 Jacobi barrier：任何 context 都不能读到同一步中另一个 context 已更新后的状态。

---

## 1. 固定输入、几何和相机

### 1.1 唯一 baseline

几何、baseline PBR 和相机统一来自：

```text
/home/nvme04/yyyan/Pixal3D/outputs/baseline1024_pbr_mesh_compare
```

使用：

```text
raw_ovoxel_mesh.pt
summary.json
```

固定相机：

```text
camera_angle_x = 0.517371749106554
distance       = 1.889538288116455
mesh_scale     = 1.0
angles_deg     = [0, 120, 240]
```

最终 geometry 必须逐元素等于该 baseline decoded mesh。texture flow、PBR fusion 和 global carrier 均不得改变 vertices/faces。

### 1.2 输入视图

输入 composite 拆成三个原生 `1024×1024` view：

```text
view_000
view_120
view_240
```

每个 view 使用：

```text
source tile size   = 256
source tile stride = 128
model tile size    = 1024
```

tile 图像是原生 `256×256` crop 后直接 resize 到 `1024×1024`。不得先把整张 view resize 到 4096 再裁图。

---

## 2. Context 定义与完整 flow support

一个 context 定义为：

```text
c = (angle, tile_id)
```

每个 context 独立保存：

```text
full_view_image
tile_image
global_dino_c
projected_dino_c
local_camera_c
local_mesh_c
local_C1024_geometry_c
shape_slat_raw_c
shape_slat_norm_c
texture_slat_reference_raw_c
texture_slat_reference_norm_c
noise_c
x_t_c
condition_c
```

### 2.1 Mesh 选择不使用可见性

对每个 context：

1. 把 baseline mesh 旋转到该 view；
2. 投影 triangle bbox；
3. 选择 bbox 与 tile rectangle 相交的所有 triangle；
4. 不使用 centroid、front-facing、back-facing、z-buffer、face visibility 或 vertex visibility 过滤；
5. compact submesh；
6. 使用精确 centered local camera 变换；
7. 在 local 规范空间独立 voxelize 到原生 C1024 O-Voxel；
8. 使用完整 local O-Voxel support 做 shape/PBR encode 和 texture flow。

因此，一个 tile 内投影到该 tile 的背面或遮挡 geometry 仍属于 flow support。Pixal3D 的 flow model通过 shape、projected condition 和模型自身机制处理前后关系；外部 visibility 不介入 model forward。

### 2.2 禁止的 flow visibility 操作

正式 flow 路径禁止：

```text
visible O-Voxel → SLat inheritance
nearest visible mesh vertex → SLat bit
decoded C1024 parent=floor(coord/16) → visibility bit
删除 invisible SLat
冻结 invisible SLat
把 invisible SLat feature 置零
只对 visible SLat 加噪
给 model forward 额外乘 visibility mask
根据 visibility 修改 DINO/proj token
```

visibility overlay 可以保留为诊断，但其结果不得进入 flow model 输入。

---

## 3. Global C4096 PBR carrier

### 3.1 作用

global C4096 是逐 step 和最终输出共同使用的 world-space PBR join key。它不是 local geometry support，也不直接送入 Pixal3D flow model。

对 baseline mesh 做 conservative surface voxelization：

```text
global_coords_g        [Ng, 3], int32, C4096
surface_point_g        [Ng, 3], centered object space
baseline_pbr_g         [Ng, 6]
global_id_g            stable sorted ID
```

禁止创建 `4096³` dense tensor。

### 3.2 不建立离散 C4096→C1024 映射

global C4096 与 local C1024 之间只使用连续坐标变换和 sparse field query：

```text
global surface point
→ yaw view transform
→ exact global camera projection
→ exact local tile camera transform
→ continuous local point
```

以及逆向：

```text
local C1024 surface point
→ exact local-to-global camera transform
→ inverse yaw
→ continuous global object point
```

禁止：

```text
global_coord // 4
global_coord / 4 后 round
bbox normalization
nearest integer cell join
把 local C1024 coord 当成 global C4096 ID
```

---

## 4. 可见性与两类查询表

### 4.1 View-level direct visibility

对每个输入 view 只预计算一次 baseline mesh 的 triangle/depth z-buffer。每个 global C4096 surface point 使用精确相机投影和 depth comparison 得到：

```text
visible[angle, global_id] ∈ {0, 1}
```

该 bit 是 global surface point 的直接可见性，不从 mesh vertex、local O-Voxel 或 SLat 继承。

visibility 只在 decoded PBR donor 查询时使用，不进入 flow forward。

### 4.2 Coverage table：不带 visibility

对每个 context 预计算：

```text
coverage_table_c:
    global_id
    p_global
    p_local_c
    uv_full
    uv_tile_1024
    projection_finite
    inside_tile
    local_coordinate_valid
```

coverage table 包含所有投影进当前 tile、且连续 global→local 坐标有效的 global carrier point，不因可见/不可见而删除。

它用于：

- 诊断一个 tile 在 world space 覆盖了哪些 global carrier；
- 允许一个在当前 context 中不可见的 target 接收其他 view 的 PBR 共识；
- 验证 global↔local round-trip。

### 4.3 Donor table：查询 PBR 时才应用 visibility

```text
donor_table_c = coverage_table_c
                ∩ visible[angle_c]
```

只有 donor table 中的点会查询该 context decoded PBR，并成为 global Gaussian fusion 的候选。

进一步的 donor validity 条件：

```text
projection finite
inside current tile
direct visible in current view
continuous p_local finite and inside local cube
decoded sparse PBR query denominator > eps
decoded PBR finite
```

不使用额外的 normal-facing、view confidence、foreground probability 或 SLat visibility 权重。

---

## 5. DINO、condition 与 context 隔离

每个 view 只计算一次自己的 full-view global DINO：

```text
global_dino[angle] = encode(full 1024 view)
```

每个 tile 单独计算自己的 local projected DINO：

```text
projected_dino[c] = encode(resize(crop(view_angle, tile_box), 1024), local_geometry_c)
```

每个 context 的原生 condition：

```text
condition_c = {
    cond: {
        global: global_dino[angle_c],
        proj:   projected_dino[c],
    },
    neg_cond: {
        global: zeros_like(global_dino[angle_c]),
        proj:   zeros_like(projected_dino[c]),
    },
}
```

严格不允许：

```text
把多个 view 的 global DINO 拼成一个 sample
把多个 tile 的 projected DINO 拼成一个 sample
跨 context attention/token fusion
跨 context SLat union
跨 context latent averaging
跨 context velocity averaging
```

可以为性能把多个 context pack 成 batch dimension，但每个 sample 的 sparse coords、global/proj DINO 和 batch index 必须独立。正式运行直接采用分阶段 physical sparse batch；B=1 只作为数值等价 reference，不作为全量执行配置。

---

## 6. SLat 归一化契约

这是实现中的硬约束，不能依靠变量名猜测。

### 6.1 Texture SLat

只使用模型加载后的：

```python
pipeline.tex_slat_normalization["mean"]
pipeline.tex_slat_normalization["std"]
```

定义：

```text
N_tex(z_raw) = (z_raw - mean_tex) / std_tex
D_tex(z_norm) = z_norm * std_tex + mean_tex
```

变量空间必须显式写入名字：

```text
tex_raw
tex_norm
x_t_norm
x0_pred_norm
x0_corrected_norm
```

不允许一个模糊的 `tex` 或 `endpoint` 在 normalized/raw 之间隐式切换。

### 6.2 Shape SLat

shape encoder 输出 raw shape SLat：

```text
shape_raw_c
shape_norm_c = N_shape(shape_raw_c)
```

其中 `N_shape` 使用：

```python
pipeline.shape_slat_normalization
```

用途固定为：

```text
flow model concat_cond  使用 shape_norm_c
decode_latent           使用 shape_raw_c / D_shape(shape_norm_c)
```

### 6.3 各阶段空间

| 阶段 | 输入/输出空间 |
|---|---|
| PBR encoder 输入 | physical PBR 先做 `P*2-1` |
| PBR encoder 输出 | raw texture SLat |
| flow 初始 clean endpoint | normalized texture SLat |
| noise | normalized texture SLat feature space，同 support |
| `x_t` | normalized texture SLat |
| model predicted `x0` | normalized texture SLat |
| `_xstart_to_pred` 的 endpoint | normalized texture SLat |
| texture decoder 输入 | raw texture SLat，必须先 `D_tex` |
| PBR re-encode 输出交回 flow | raw encoder output 后必须 `N_tex` |

### 6.4 强制断言

每个 context、每个 step 都断言：

```text
coords(x_t_norm) == coords(x0_pred_norm)
coords(x_t_norm) == coords(x0_corrected_norm)
coords(shape_norm) == coords(texture_norm)
all features finite
std_tex > 0
N_tex(D_tex(x)) ≈ x
D_tex(N_tex(z)) ≈ z
```

禁止：

```text
normalized texture SLat 直接 decode
raw PBR encoder SLat 直接传入 flow
对 normalization mean/std 使用手写常数
对 PBR encode/decode 结果做 latent residual cancellation
```

---

## 7. 初始化

对每个 context：

1. 用完整 local C1024 geometry 查询 baseline PBR；
2. official shape encoder 得到 `shape_raw_c`；
3. official PBR encoder、`sample_posterior=False` 得到 `tex_reference_raw_c`；
4. 使用模型参数得到 `shape_norm_c` 和 `tex_reference_norm_c`；
5. 在完整 texture SLat support 上生成独立 noise；
6. 使用 native noise endpoint 公式初始化 normalized state。

```text
sigma(t) = sigma_min + (1 - sigma_min) * t
x_t_norm = (1 - t) * tex_reference_norm
           + sigma(t) * noise_strength * epsilon
```

默认：

```text
noise_timestep = 1.0
noise_strength = 1.0
num_steps      = 12
seed           = 42
```

当 `t=1` 时 clean coefficient 为 0，状态是完整 noise；这是预期行为。所有 SLat 都加噪，不按可见性修改 noise。

---

## 8. 每一步同步多 context Gaussian PBR flow

设当前 sampler step 为：

```text
t → t_next
```

所有 context 在该 step 开始时的快照为：

```text
X_t = {x_t_norm_c}
```

### Barrier A：所有 context 独立 flow forward

对每个 context：

```text
x0_pred_norm_c, v_native_c = model_prediction(
    x_t_norm_c,
    t,
    concat_cond=shape_norm_c,
    condition=condition_c,
)
```

要求：

- model forward 使用完整 local SLat support；
- 不读取 visibility；
- 不读取其他 context 的 DINO、shape SLat、texture state 或 velocity；
- 即使物理执行为 batch，也必须等价于 singleton forward；
- 所有 `x0_pred_norm_c` 完成并冻结后才进入 decode。

native `v_native_c` 只保存为诊断；正式更新使用融合后 endpoint 重新计算的 `v_corrected_c`。

### Barrier B：所有 predicted endpoint decode

对每个冻结 endpoint：

```text
x0_pred_raw_c = D_tex(x0_pred_norm_c)

decoded_mesh_c = decode_latent(
    shape_raw_c,
    x0_pred_raw_c,
    resolution=1024,
)
```

从 decoded sparse PBR field 查询两类值：

1. `P_self_c`：在完整 local C1024 geometry support 上查询，行数与 local O-Voxel coords 完全一致；
2. `P_donor_c`：只在 `donor_table_c` 的 continuous `p_local_c` 上查询。

稀疏 query 必须使用 masked trilinear：同时查询 PBR numerator 和 support denominator；denominator 小于 `eps` 的点无效，不能把零值当成真实 PBR。

所有 context 的 endpoint 都是同一个旧状态集合 `X_t` 的结果。可以分 microbatch decode 并 offload，但在逻辑上必须形成只读 snapshot，不能边 decode 边更新任何 `x_t`。

### Barrier C：构造 global C4096 visible donor candidates

对每个 context `c` 和每个有效 donor row：

```text
candidate = (
    global_id = g,
    context_id = c,
    angle,
    tile_id,
    pbr = P_donor_c,
    uv_tile,
    distance_to_tile_center,
)
```

再次断言：

```text
visible[angle_c, g] == True
```

不可见 global point 不执行 local PBR donor query，也不进入 reducer。

### Barrier D：global C4096 Gaussian PBR fusion

tile-center 距离使用 resize 后的 local/model `1024×1024` 像素坐标：

```text
center = (511.5, 511.5)
d(c,g) = ||uv_tile(c,g) - center||_2
w(c,g) = exp(-d(c,g)^2 / (2*sigma^2))
```

默认沿用已验证参数：

```text
sigma = 256 model pixels
      = 64 source-view pixels
```

对每个 global carrier ID：

```text
Numerator[g]   = Σ_c w(c,g) * P(c,g)
Denominator[g] = Σ_c w(c,g)
DonorCount[g]  = Σ_c 1
```

只对 direct-visible、query-valid candidates 求和。所有 view/tile 使用相同公式，不添加 angle 优先级或 same-view bonus。

融合所有六个 PBR channel：

```text
base_color RGB
metallic
roughness
alpha
```

Gaussian weight 对六个 channel 相同。正式路径不做逐 channel hand-tuned weight，也不对 decoded PBR 先做 ad-hoc clamp。

为了连续 global→local 查询，不能只保存已经相除的 `P_global`。必须把稀疏字段保存为：

```text
global_numerator_field   [Ng, 6]
global_denominator_field [Ng, 1]
```

随后在任意 continuous global point `p` 上使用：

```text
P_consensus(p) = query(global_numerator_field, p)
                 / query(global_denominator_field, p)
```

只有 query denominator `> eps` 才有效。这样可避免 invalid/zero cell 在 trilinear interpolation 中污染颜色。

### Barrier E：global consensus 连续查询回完整 local C1024 target

对每个 context 的每一个 local C1024 O-Voxel surface point：

1. 使用 exact local→view→world 变换得到 continuous global point；
2. 查询 global numerator/denominator fields；
3. 不检查 target context 对这个点是否可见；
4. 若 global denominator 有效，用 global consensus 替换 `P_self_c`；
5. 若没有任何 visible donor，保留该 context 自己的 `P_self_c`。

```text
P_corrected_c[i] =
    P_consensus(p_global_c[i])    if global denominator > eps
    P_self_c[i]                   otherwise
```

这一规则非常重要：

- 当前 view 不可见的 target 可以从其他 view 的可见 donor 接收 PBR；
- 所有 view 都不可见的 local point 继续由 Pixal3D 自己的 predicted endpoint 演化；
- flow 中不把 unseen point 强行 anchor 到 baseline；
- baseline fallback 只在最终 global export 时执行。

### Barrier F：official PBR re-encode 并恢复 normalized endpoint

构造与原始 local C1024 geometry 完全相同 support 的 encoder 输入：

```text
pbr_encoder_input_c = SparseTensor(
    feats = P_corrected_c * 2 - 1,
    coords = local_C1024_coords_c,
)
```

然后：

```text
x0_corrected_raw_c = PBR_Encoder(
    pbr_encoder_input_c,
    sample_posterior=False,
)

x0_corrected_norm_c = N_tex(x0_corrected_raw_c)
```

必须断言：

```text
coords(x0_corrected_norm_c) == coords(x0_pred_norm_c)
features finite
```

如果 encoder 输出 support 不一致，立即失败；禁止 nearest-token 对齐、补零或重新排序后静默继续。

### Barrier G：native endpoint bridge 和同步 Euler 更新

对每个 context：

```text
v_corrected_c = sampler._xstart_to_pred(
    x_t_norm_c,
    t,
    x0_corrected_norm_c,
)

x_next_norm_c = x_t_norm_c
                - (t - t_next) * v_corrected_c
```

先计算并验证所有 `x_next_norm_c`，然后一次性替换全体状态：

```text
X_t = {x_t_norm_c}
→ atomic state replacement
X_t_next = {x_next_norm_c}
```

任何 context 不得在同一步读取另一个 context 的 `x_next_norm`。这是同步 Jacobi，不是 Gauss-Seidel。

---

## 9. visibility 的精确语义

visibility 只回答：

```text
这个 context decoded 出来的 PBR 值，能否作为该 global surface point 的 image-observed donor？
```

visibility 不回答：

```text
这个点是否属于 flow support？
这个 SLat 是否应该存在？
这个 SLat 是否应该加噪？
这个 target 是否能接收其他视角的 PBR？
模型是否应该生成背面？
```

因此：

```text
donor invisible → 不能贡献 PBR
target invisible + another-view donor visible → 可以接收共识
target invisible + all donors absent → 保留自己的 predicted PBR
```

不能重新引入旧的 visibility ancestry：

```text
mesh vertex visibility
→ nearest local O-Voxel
→ parent SLat
→ decoded child O-Voxel
```

正式 donor gate 直接使用 global C4096 surface point 的 view-level depth visibility。

---

## 10. 最终 PBR 输出

最后一个 Euler step 完成后：

1. inverse-normalize final normalized texture SLat；
2. decode 所有 context final local PBR fields；
3. 使用与逐 step 相同的 direct-visible donor tables 查询 final candidates；
4. 对每个 global C4096 ID，只允许 direct-visible、query-valid candidate；
5. 默认选择距离 tile center 最近的 candidate；
6. 所有 view 均无有效 visible candidate 时使用 `baseline_pbr_g`；
7. 把 global final PBR field 查询到 baseline mesh vertices 和 face centroids；
8. 输出固定 geometry 的 per-vertex/per-face PBR mesh。

最终默认使用 hard nearest-center，而不是再做一次 Gaussian average。原因是逐 step Gaussian 已经让各 context 的 endpoint 收敛到一致 PBR；最终 hard selection 可避免额外模糊，并保留清晰 donor provenance。

必须保存：

```text
owner_context
owner_angle
owner_tile
owner_distance
visible_candidate_count
baseline_fallback_mask
```

确定性 tie-break：

```text
distance
→ angle ascending
→ tile_id ascending
→ context_id ascending
```

可以额外生成 final Gaussian render 作为诊断，但不得混淆为正式结果。

---

## 11. 完整伪代码

```python
# Immutable geometry and camera.
mesh, baseline_field, camera = load_baseline1024()
global_carrier = build_or_load_global_c4096(mesh)
global_carrier.base_pbr = baseline_field.query(global_carrier.surface_points)

# Direct point/depth visibility; never converted into an SLat mask.
for angle in (0, 120, 240):
    zbuffer[angle] = render_geometry_zbuffer(mesh, camera, angle, resolution=4096)
    visible[angle] = direct_point_visibility(global_carrier, zbuffer[angle])

# Each context uses all projected geometry, regardless of visibility.
contexts = []
for angle in (0, 120, 240):
    global_dino = encode_global_dino(view[angle])
    for tile in overlapping_tiles(size=256, stride=128):
        local_geometry = cut_transform_voxelize_all_projected_mesh(
            mesh, camera, angle, tile, resolution=1024,
            visibility_filter=None,
        )
        condition = build_isolated_condition(
            global_dino=global_dino,
            projected_dino=encode_tile_dino(view[angle], tile, local_geometry),
        )
        shape_raw = shape_encoder(local_geometry, sample_posterior=False)
        shape_norm = normalize(shape_raw, pipeline.shape_slat_normalization)
        pbr_reference = query_baseline_pbr(local_geometry)
        tex_raw = pbr_encoder(pbr_reference * 2 - 1, sample_posterior=False)
        tex_norm = normalize(tex_raw, pipeline.tex_slat_normalization)
        x_t_norm = native_noised_endpoint(tex_norm, independent_noise(angle, tile))
        coverage = build_all_projected_global_query_table(global_carrier, angle, tile)
        donor = coverage[visible[angle, coverage.global_id]]
        contexts.append(Context(...))

for t, t_next in native_schedule:
    # A. Independent native forwards. No visibility in this function.
    predicted = {}
    for c in contexts:
        predicted[c] = native_flow_forward(
            x_t_norm[c], shape_norm[c], condition[c], t
        ).x0_norm

    # B/C. Frozen endpoint decode and visible donor extraction.
    self_pbr = {}
    global_num = zeros_on_sparse_global_carrier(channels=6)
    global_den = zeros_on_sparse_global_carrier(channels=1)
    for c in contexts:
        tex_raw = denormalize(predicted[c], pipeline.tex_slat_normalization)
        decoded = decode_latent(shape_raw[c], tex_raw, resolution=1024)
        self_pbr[c] = masked_query(decoded, c.full_local_support_points)
        donor_pbr, valid = masked_query(decoded, c.donor_table.local_points)
        ids = c.donor_table.global_ids[valid]
        uv = c.donor_table.uv_tile[valid]
        w = exp(-squared_distance(uv, (511.5, 511.5)) / (2 * 256**2))
        global_num.index_add(ids, w[:, None] * donor_pbr[valid])
        global_den.index_add(ids, w[:, None])

    # D/E/F. Query global consensus to every local target, then re-encode.
    corrected = {}
    for c in contexts:
        num = sparse_query(global_num, c.full_local_support_world_points)
        den = sparse_query(global_den, c.full_local_support_world_points)
        valid = den > eps
        pbr = self_pbr[c].clone()
        pbr[valid] = num[valid] / den[valid]
        raw = pbr_encoder(pbr * 2 - 1, sample_posterior=False)
        corrected[c] = normalize(raw, pipeline.tex_slat_normalization)

    # G. Native endpoint bridge; replace all states only after all are ready.
    next_states = {}
    for c in contexts:
        velocity = sampler._xstart_to_pred(x_t_norm[c], t, corrected[c])
        next_states[c] = x_t_norm[c] - (t - t_next) * velocity
    x_t_norm = atomic_replace(next_states)

# Final export uses visible nearest-center donor; all-unseen uses baseline PBR.
final_decoded = decode_all(denormalize_all(x_t_norm))
final_candidates = query_direct_visible_candidates(final_decoded)
global_final_pbr = nearest_tile_center_visible_candidate_or_baseline(final_candidates)
final_mesh = assign_global_pbr_to_unchanged_baseline_mesh(global_final_pbr)
```

---

## 12. 内存与执行策略

### 12.1 Jacobi barrier 是逻辑约束，不要求同时驻留 GPU

允许：

- 同一 barrier 内把多个 context 沿 batch 维组成一次真实 physical sparse batch；
- flow、endpoint decode、PBR re-encode 使用各自独立的 batch size；
- frozen predicted endpoints offload 到 CPU；
- endpoint decode 按 microbatch 执行；
- decoded local field 查询完成后释放 GPU mesh；
- global numerator/denominator 分 chunk 累积；
- PBR re-encode 按 microbatch 执行。

不允许：

- 某 context re-encode 后立即更新，并让后续 context 读取该新状态；
- donor candidates 尚未全部完成就生成部分 global consensus 并更新 target；
- 因内存不足改变数学顺序为 Gauss-Seidel。

因此，“所有 context 先完成 flow forward”不表示 144 次串行 singleton call。它表示若 flow 分为若干 physical batch，则这些 batch 全部读取同一个 `X_t`，全部完成后才进入 decode barrier。decode 和 re-encode 同理。

正式默认 stage batch：

```text
flow forward      = 44
endpoint decode   = 12
PBR re-encode     = 13
```

每次切换 barrier 前释放上一阶段模型和临时 GPU tensor，并执行同步与 cache cleanup。global Gaussian accumulator 不得与 endpoint decoder 或 PBR encoder 的峰值 workspace 同时常驻 GPU；需要 streaming/chunking 或 CPU offload。

### 12.2 不保存 `Ng × Ncontext` dense candidate tensor

预计算每个 context 的稀疏 donor table，逐 context/chunk streaming：

```text
global_num [Ng, 6]
global_den [Ng, 1]
global_count [Ng]
```

`Ng≈65M` 时只维护 O(Ng) accumulator，不维护 `65M×144` 表。

为确定性，正式 correctness 运行按固定：

```text
angle → tile_id → global_id
```

顺序累积；如果改用并行 atomic reduction，必须先通过 traversal-invariance 容差测试。

### 12.3 Cache

可跨 step 复用：

```text
global carrier geometry
view visibility bitsets
context coverage/donor tables
local C1024 geometry
shape_raw/shape_norm
global/local DINO
local support world points
Gaussian distance/weight
```

每 step 必须重算：

```text
model predicted endpoint
decoded PBR
visible donor PBR values
global numerator/denominator
corrected local PBR
PBR re-encoded endpoint
Euler state
```

---

## 13. 实现映射

建议以当前：

```text
pixal3d_global_c4096_visible_local_flow.py
```

为主入口，替换 `_run_independent_flow`，不要另起一条无法复用现有 carrier/visibility 的实验路径。

复用：

```text
pixal3d_multiview_fixed_geometry_pbr_gaussian_sr.py
    _predict_flow_batch
    _decode_snapshots_batched 的 normalization/decode contract
    _encode_fused_batch 的 official PBR encode contract
    Jacobi barriers

pixal3d_cross_tile_pbr_perstep.py
    _normalize_slat
    _denormalize_slat
    _native_noised_endpoint
    _native_schedule
    _schedule_start
    _sampler_step_kwargs

pixal3d_global_c4096_visible_local_flow.py
    global C4096 carrier
    direct point/depth visibility
    local context construction
    continuous global/local projection
    final fixed-geometry output
```

新增核心函数建议：

```text
_build_all_projected_coverage_tables
_build_direct_visible_donor_tables
_decode_endpoint_snapshot
_accumulate_global_gaussian_pbr
_query_global_consensus_to_local_support
_encode_corrected_endpoints
_run_joint_gaussian_flow
_final_visible_hard_assignment
```

不得复用旧版错误的：

```text
nearest baseline vertex visibility inheritance
local O-Voxel visibility → SLat visibility
SLat parent visibility → decoded PBR visibility
```

---

## 14. 正确性测试

### 14.1 Flow support 不受 visibility 影响

对同一个 context，人工翻转 visibility bitset，验证：

```text
flow input coords identical
shape_norm identical
x_t_norm identical
condition/DINO identical
model predicted x0 before PBR fusion identical
```

只有 donor candidate 数量和 corrected endpoint 可以变化。

### 14.2 DINO isolation

逐 context 保存 global/proj DINO hash。断言：

```text
context c 只引用 angle_c 的 global DINO
context c 只引用自己的 projected DINO
```

B=1 与 physical B>1 的相同 context 输出必须在规定容差内一致。该测试验证 batch packing 不会混合 DINO 或 sparse batch index；通过后正式运行采用第 19 节的分阶段 batch 配置。

### 14.3 Normalization contract

测试：

```text
N_tex(D_tex(x_norm)) ≈ x_norm
D_tex(N_tex(x_raw)) ≈ x_raw
decode(normalized endpoint) 必须被测试捕获为错误调用
re-encode raw output 未 normalize 必须被测试捕获
```

每 step 保存：

```text
x_t_norm range
x0_pred_norm range
x0_pred_raw range
x0_corrected_raw range
x0_corrected_norm range
```

### 14.4 Jacobi barrier

正序、逆序遍历 contexts，最终 step 输出必须在浮点容差内一致。任何明显 traversal-order 依赖都视为失败。

### 14.5 Gaussian analytic test

两个 donor：

```text
P1, distance=0
P2, distance=512
sigma=256
```

验证 normalized weights 和解析值一致。不可见 donor 的 weight 必须严格为 0。

### 14.6 Target/Donor visibility semantics

至少测试：

```text
A：target visible，自身 donor valid
B：target invisible，另一个 view donor visible
C：target invisible，所有 view donor absent
D：donor projected inside tile 但被 z-buffer 遮挡
```

预期：

```text
A → 可参与 Gaussian
B → target 接收另一个 view 的共识
C → flow step 保留 target self prediction；final export baseline fallback
D → 不能进入 global accumulator
```

### 14.7 Masked sparse query

验证 numerator/denominator query，不得把 sparse 缺失 corner 的零值平均进 PBR。denominator=0 时必须回到 self prediction。

### 14.8 Continuous coordinate round-trip

覆盖：

```text
local mesh vertices
local C1024 surface points
global C4096 donor points
```

验证：

```text
global → local → global max_abs_error < 2e-5
local → global → local max_abs_error < 2e-5
```

### 14.9 Final owner invariants

```text
owner >= 0
→ candidate direct visible
→ query valid
→ owner distance 是全部合法 candidate 的最小值

owner == baseline
→ visible valid candidate count == 0
→ final PBR == baseline PBR
```

### 14.10 Geometry identity

最终 vertices/faces 与 baseline 逐元素相同，同时保存 hash。只比较数量不够。

---

## 15. 诊断与输出

每个 step 至少保存：

```text
step index, t, t_next
context count
flow/decode/fusion/encode/Euler timings
global carrier donor_count histogram
visible candidate count per view/tile
global denominator zero/nonzero count
cross-view donor count
same-view overlap count
PBR donor variance: RGB/M/R/A
P_self vs P_corrected MAE/median/p95/max
x0_pred_norm vs x0_corrected_norm correction norm
normalization ranges and finite checks
Jacobi barrier flags
```

每个 context 至少保存：

```text
all projected mesh faces
local C1024 O-Voxel count
shape/texture SLat count
coverage table count
direct-visible donor table count
valid/invalid decoded donor queries
received global consensus local O-Voxel count
kept self prediction local O-Voxel count
cross-view receipt count
```

全局至少保存：

```text
config.json
global_camera.json
input_hashes.json
normalization.json
global_c4096_carrier.pt
global_c4096_visibility.pt
coverage_tables/
donor_tables/
flow_steps/step_XX_summary.json
correction_norms.csv
pbr_disagreement.csv
cross_view_coverage_stats.json
final_owner.pt
global_c4096_final_pbr.pt
final_per_vertex_pbr_mesh.pt
final_per_face_pbr_mesh.pt
geometry_identity.json
metrics_1024.json
renders/
summary.json
```

可视化：

```text
每个 view 的 direct-visible/invisible global C4096 overlay
每个 tile 的 all-covered carrier overlay
每个 tile 的 visible donor overlay
每个 view 的 final owner-angle/owner-tile overlay
baseline fallback overlay
逐 step PBR correction heatmap（至少选定 ROI/tile）
```

---

## 16. 执行顺序

### Phase 1：静态 preflight

1. baseline/camera/hash；
2. global carrier；
3. 三视角 direct visibility；
4. local geometry 不使用 visibility 的断言；
5. coverage/donor table；
6. normalization round-trip；
7. Gaussian analytic test；
8. DINO isolation。

### Phase 2：小规模 flow correctness

先选择能覆盖主体且互相重叠的少量 tile，至少包含两个 view；不能只选单 view，否则无法验证 cross-view receipt。

运行完整 12 steps，并保存全部 step diagnostics。必须观察到：

```text
cross-view donor > 0
x0 correction norm > 0
target-invisible/other-view-visible receipt > 0
no visibility change to model-forward support
```

### Phase 3：全量 3-view × all active tiles

只有 Phase 2 全部通过后再运行全量。正式 correctness 首轮使用：

```text
flow batch size       = 44
endpoint decode batch = 12
PBR re-encode batch   = 13
full-strength Gaussian correction at every step
sigma = 256 model pixels
all 6 PBR channels fused
final hard nearest-center assignment
```

### Phase 4：评估

用精确 baseline camera 渲染：

```text
yaw000
yaw120
yaw240
```

以原生 `1024×1024` 输入计算：

```text
full-image PSNR/SSIM/LPIPS
input-foreground PSNR/SSIM
silhouette IoU
```

同时报告 baseline 和 independent-flow ablation，避免只报告新方法绝对值。

---

## 17. 验收条件

启动正式全量实验前必须满足：

1. flow support 包含全部投影 geometry，没有 visibility pruning；
2. 每个 context 的 DINO/condition/state 完全隔离；
3. texture flow 的所有状态和 endpoint 均为 model-parameter normalized SLat；
4. decode 前 inverse-normalize，PBR encode 后 normalize；
5. donor visibility 来自 global C4096 direct point/depth test；
6. visibility 只作用于 PBR donor query；
7. target invisible 时仍可接收其他 view 的 visible PBR；
8. no-donor 的 flow target 保留 self prediction；
9. 每步严格执行 forward/decode/fuse/re-encode/xstart/Euler Jacobi barrier；
10. 不存在 DINO、SLat 或 velocity 跨 context 融合；
11. global/local 只做 continuous transform/query，不做离散 ID 缩放；
12. final all-unseen region 严格 baseline fallback；
13. final geometry 与 baseline bitwise 相同；
14. 1024 三视角指标、逐 step correction 和 donor coverage 均完整保存。

---

## 18. 已确认决策

以下默认值与旧 Gaussian 流程和当前 global-C4096 目标一致，可直接实施：

```text
逐 step correction strength = 1.0（无额外时间 schedule）
Gaussian sigma              = 256 model pixels
Gaussian channel            = 全部 6 个 PBR channels
donor visibility            = binary direct C4096 depth visibility
flow visibility             = none
no-donor during flow        = keep self predicted endpoint
all-unseen final            = baseline PBR
final production assignment = visible nearest tile center
formal flow batch size      = 44
formal decode batch size    = 12
formal PBR encode batch     = 13
```

以下三项已经确认：

1. 最终正式输出使用 hard nearest-center；
2. Gaussian 融合全部 6 个 PBR channel；
3. 每一步使用 full-strength correction，不添加 time schedule。

除此之外不需要改变已经确定的核心语义。

---

## 19. A800 80GB 显存预检与分阶段 batching

### 19.1 测试对象和方法

全量旧运行的 `context_summary.json` 覆盖三视图全部 144 个有效 context。按 local C1024 O-Voxel 数和 texture SLat token 数检查全部 context 后，最大者是：

```text
view             = 240
tile_id          = 32
local O-Voxel    = 5,516,511
texture SLat     = 20,156 tokens
GPU              = NVIDIA A800-SXM4-80GB
PyTorch capacity = 79.33 GiB
```

先对该真实最大 context 测 B=1，再把它重复为一个 conservative worst-case physical sparse batch。重复最大 context 比普通 context 混合更严格：正式批次中不可能每个 sample 都同时超过这个 support。

测试路径使用真实：

```text
official PBR encoder
native tex_slat flow model + isolated per-sample DINO condition
inverse-normalized endpoint + pipeline.decode_latent(..., 1024)
```

原始预检 output 已在整理目录时清理；下面的实测数值已经固化在本节，
正式运行参数和最终产物保存在当前实验目录
`outputs/global_c4096_visible_local_flow_cuda4_batch44_12_13_2048/`。

### 19.2 B=1 最大 tile 峰值

| 阶段 | total peak allocated | peak reserved | 状态 |
|---|---:|---:|---|
| flow forward | 4.391 GiB | 4.619 GiB | ok |
| endpoint decode + local PBR query | 6.612 GiB | 8.383 GiB | ok |
| PBR re-encode | 6.189 GiB | 8.520 GiB | ok |

初始 context build 的 shape encode 与 baseline-PBR encode 峰值分别为 6.836 GiB 和 9.251 GiB；它们不在每个 Euler step 重复执行。

### 19.3 实测 batch 边界

| 阶段 | 最大成功 batch | 成功峰值 | 首个确认失败 batch | 失败方式 | 正式 batch |
|---|---:|---:|---:|---|---:|
| flow forward | 46 | 77.387 GiB | 48 | CUDA OOM | 44 |
| endpoint decode | 13 | 73.413 GiB | 14 | CUDA OOM | 12 |
| PBR re-encode | 14 | 76.958 GiB | 15 | CUDA OOM | 13 |

补充实测点：

```text
flow:   B32=54.677, B40=67.654, B44=74.143, B46=77.387 GiB
decode: B10=56.793, B12=67.945, B13=73.413 GiB
encode: B10=54.571, B12=65.323, B13=70.849, B14=76.958 GiB
```

正式值都低于硬边界。`decode B13` 曾在一次连续 sweep 中出现 non-finite mesh，虽然独立重测成功，因此正式值保持 B12；不能把“没有 OOM”当作唯一正确性标准。

### 19.4 调度规则

144 个 context 在每个 step 的 barrier 调度为：

```text
Barrier A / flow:      physical batches up to 44 contexts
Barrier B / decode:    physical batches up to 12 contexts
Barrier F / re-encode: physical batches up to 13 contexts
Barrier G / Euler:     可按 flow batch 或纯 tensor token budget 批处理
```

每个阶段把 context 按该阶段成本从大到小排序：

```text
flow cost   = texture SLat token count
decode cost = local C1024 O-Voxel count + texture SLat token count
encode cost = local C1024 O-Voxel count
```

除 batch-count cap 外，再使用 worst-case resource budget：

```text
flow token budget   = 44 * 20,156    = 886,864 SLat tokens
decode voxel budget = 12 * 5,516,511 = 66,198,132 O-Voxels
encode voxel budget = 13 * 5,516,511 = 71,714,643 O-Voxels
```

对当前 144 contexts，仅按正式 count cap 会得到：

```text
flow      44 + 44 + 44 + 12 = 4 physical calls / step
decode    12 * 12           = 12 physical calls / step
re-encode 13 * 11 + 1       = 12 physical calls / step
```

batch 只是物理执行分组，不改变 context 的条件归属，也不改变 Jacobi 数学顺序。所有 flow batches 完成前不 decode；所有 decode/donor extraction 完成前不 fusion/re-encode；所有 corrected endpoints 完成前不 Euler/state replace。
