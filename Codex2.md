# 任务：第一视角唯一稠密 SLat 上的 Shape + Texture 跨 Tile / 多视图联合超分

## 0. 方案结论

本方案只维护一套 SLat：

```text
第一视角 GT 4096
→ 49 个 HR tile
→ local C1024 geometry encode
→ 只取 activation positions
→ 二维 first-owner 稠密化
→ 唯一 global master_id / master_q_world
```

后续所有 view 和 tile 都只能读取、预测和更新这套 master ID，不能新增、删除、重排或另建 support。

Shape 和 Texture 都在这套 support 上做跨 tile、多视图超分。每个 context 有自己的 condition、noise 和 noisy `x_t`；全局只融合当前 timestep 的 clean endpoint。融合后的 endpoint 回到各 context，再根据各自 `x_t` 调用官方 `_xstart_to_pred`，所以不同 context 的 velocity 可以不同。

每个 master ID 的 endpoint 融合固定为：

```text
有 1 个直接可见 proposal：直接使用
有多个直接可见 proposal：按距离 tile 图像中心的 Gaussian 权重融合
没有直接可见 proposal：使用预先缓存的 baseline endpoint
```

Texture 继续复用已经跑通的：

```text
predicted endpoint
→ decode 到 PBR
→ visible donor Gaussian PBR fusion
→ query 回各 tile
→ official PBR encoder
→ corrected texture endpoint
→ 在唯一 master 上做 visible/Gaussian/fallback 融合
→ 回到各 tile 计算 velocity
```

第一版先把闭环跑通，不做动态 support、动态 visibility、复杂置信度或新的几何合并算法。

## 1. 参考结果和建议入口

保留的两份正式结果：

```text
outputs/global4096_tile_x0_consensus_sync_cuda5
outputs/global_c4096_visible_local_flow_cuda5_cycle
```

分别复用：

- 第一份的第一视角 dense master support、camera、tile mapping 和 direct `pred_x_0` flow；
- 第二份的多视图 condition、直接可见性、PBR decode/fusion/re-encode 和真实 multi-batch。

建议新入口：

```text
pixal3d_global4096_multiview_joint_shape_tex_sr.py
```

建议新输出：

```text
outputs/global4096_multiview_joint_shape_tex_sr_cuda5/
```

不得覆盖或 resume 两份已有正式结果。已放弃的 geometry-only O-Voxel/QEF 路线位于：

```text
used/abandoned_geometry_ovoxel_sr_20260823/
```

新实现不得重新依赖它。

## 2. 固定输入、视图和 tile

第一视角 GT：

```text
/home/nvme04/yyyan/Pixal3D/assets/images/0_img.png
```

多视图 composite：

```text
/home/nvme04/yyyan/Pixal3D/test_pic/mask_compare_output/image2_resized.png
```

按现有多视图路径先拆成：

```text
view_000
view_120
view_240
```

固定相机：

```text
camera_angle_x = 0.517371749106554
distance       = 1.889538288116455
mesh_scale     = 1.0
yaw            = [0, 120, 240]
```

第一视角 GT 是原生 4096，使用：

```text
tile size   = 1024
tile stride = 512
layout      = 7 × 7
```

每个多视图 panel 是原生 1024，使用等价布局：

```text
source tile   = 256
source stride = 128
model input   = 1024
virtual tile  = 1024 on virtual 4096 canvas
virtual stride = 512
```

source 256 crop 直接 resize 到 1024。不得先放大整张 view，也不得做 `1024→512→1024`。

## 3. 唯一 dense master support

### 3.1 只由第一视角建立

唯一 master 完全沿用当前第一视角正式路径：

1. baseline 1024 生成 geometry 和 PBR field；
2. baseline mesh 投影到第一视角 canonical 4096；
3. 每个 HR tile 建 local C1024 O-Voxel；
4. official shape encoder 只提供 local C64 activation positions；
5. encoded feature 不用于初始化 flow；
6. 第一视角 49 tile 内按二维 first-owner 稠密化；
7. 保存唯一 `master_id`、`master_q_world` 和 front-tile owner。

输入、camera、baseline mesh 和 schema hash 一致时可以直接复用：

```text
outputs/global4096_tile_x0_consensus_sync_cuda5/support/master_support.pt
```

附加 view/tile 只能建立到这些 ID 的 view mapping，不允许产生新的 activation ID。

### 3.2 master schema

```text
master_id                 int64   [N]
master_q_world            float32 [N,3]
owner_front_tile_id       int16   [N]
owner_front_local_coord   int32   [N,3]
front_uv_4096             float32 [N,2]
baseline_nearest_face_id  int64   [N]
baseline_nearest_point    float32 [N,3]
baseline_nearest_bary     float32 [N,3]
baseline_face_distance    float32 [N]
```

Shape 和 Texture 使用完全相同的 `N`、ID 顺序和 world position：

```text
shape_endpoint_global     [N,C_shape]
texture_endpoint_global   [N,C_texture]
```

### 3.3 其他 view/tile 的 mapping

对 context `c=(yaw,tile_id)`：

```text
master_q_world
→ R_y(yaw)
→ view camera projection
→ tile crop camera
→ centered local q
→ local C64 coord
```

只保留：

- 投影位于当前 tile；
- local q 在有效范围；
- local coord 位于该 context 的 native geometry support。

每个 local sparse row 必须一对一对应一个 master ID。附加 view 量化后若多个 master ID 撞到同一 local coord，第一版只保留 round-trip error 最小者；其他 ID 不从这个 context 接 proposal，但仍存在于 global master，可由其他 context 或 baseline fallback 覆盖。固定 tie-break 使用较小 `master_id`。

这一规则只解决模型输入坐标唯一性，不改变 global master identity，也不创建 collision group 或额外 feature fusion。

## 4. “直接被 tile 图看到”的固定定义

SLat activation 是 volumetric token，不能仅凭它落在二维 tile 内就宣称可见。可见性严格由 baseline 1024 生成的 mesh 三角面提供，并且必须对每个多视图 tile 单独保存，不能把一个 view-level bit 广播给该 view 的所有 tile。

### 4.1 每个 master SLat 固定最近三角面

对每个 `master_id=m`，在 canonical world 中执行一次 point-to-triangle 最近邻：

```text
master_q_world[m]
→ closest point on baseline-1024 mesh triangles
→ nearest_face_id[m]
→ nearest_point[m]
→ nearest_bary[m]
→ face_distance[m]
```

最近邻对象必须是三角面，不是 mesh vertex，也不是 O-Voxel parent。距离相同使用较小 `face_id` 稳定打破平局。这组对应在 shape/texture 全程冻结。

### 4.2 每个 `(view,tile)` 独立三角面可见性

对每个 context `c=(yaw,tile_id)`，使用该 context 自己的 exact tile camera，对 baseline mesh 生成 tile-local face-ID/depth buffer，并得到：

```text
face_visible[c, face_id] ∈ {0,1}
```

同一个 yaw 下的 49 个 tile 分别计算和保存自己的 `face_visible`；不得只渲染一次 full view 后把同一个 face bit 无条件复制给所有 tile。实现可以复用 full-view raster buffer做精确 crop/坐标换算，但最终仍必须物化为每个 context 独立的 visibility table。

对当前 context 已经拿走的 master SLat：

```text
visible[c,m] =
    mapping[c,m] valid
    AND face_visible[c, nearest_face_id[m]] == 1
```

`mapping[c,m] valid` 已经表示该 tile 确实拿走了这个 SLat；其可见 bit 直接查最近三角面在当前 tile 的独立 visibility。可额外保存 nearest point 的 projected depth 与 tile depth buffer 误差用于诊断，但不得用另一套 vertex/O-Voxel visibility 覆盖这个 face bit。

保存：

```text
visible[context_id, master_id] ∈ {0,1}
nearest_face_id[master_id]
nearest_point[master_id]
face_visible[context_id, face_id]
uv[context_id, master_id]
depth_error[context_id, master_id]
tile_center_distance[context_id, master_id]
```

visibility 在整次 shape/texture flow 中不随 timestep 更新。它只决定 endpoint/PBR donor 能否贡献，不删除 support、不冻结 state、不修改 condition。

如果最近三角面可见但 decoded PBR query 无效，该 PBR proposal 同样视为无效；不能把零值当作可见结果。

## 5. 一次性建立 baseline fallback endpoint

Dense master 与 official baseline C64 SLat 不对应，所以不能直接查 baseline SLat row。Fallback 必须在正式 flow 前按 dense master/context mapping 预计算一次。

### 5.1 Texture baseline reference

先由 baseline 1024 得到连续 PBR field。处理单位必须是一个完整子图/context，禁止以单个 master ID 为单位调用 encoder。

对每个 context：

1. 收集这个子图拿走的全部 dense master ID 和它的完整 local C1024 geometry support；
2. 对完整 local support 的所有 query point 一次性分 chunk 查询 baseline PBR field，得到 base color、metallic、roughness、alpha；
3. 将全部查询结果共同组装成该子图的一张完整 sparse local C1024 PBR field；
4. 把这张完整 field 作为一个 sparse batch row，调用一次 official PBR encoder、`sample_posterior=False`；
5. 得到该子图完整 texture SLat 后统一 normalize；
6. 再按该 context 的 local-row→master-ID mapping 拆回并保存 baseline endpoint。

物理实现可以把多个完整子图组成真实 `[B,...]` encoder batch，但每个 batch row 必须始终是一整个子图的 sparse field，不能把一个子图拆成逐点 encoder 调用。

得到：

```text
tex_baseline_ref[c,m]
tex_baseline_valid[c,m]
```

同一 master ID 可能从一个或多个 context 得到 baseline texture endpoint。先按 tile-center Gaussian 做一次固定融合：

\[
tex\_baseline\_global[m]=
\frac{\sum_c valid_{c,m}w_{c,m}tex\_baseline\_ref[c,m]}
{\sum_c valid_{c,m}w_{c,m}}
\]

这里必须先“每个子图的完整 support 一起 query baseline PBR → 整个子图 SLat 一起 encode”，再平均 encoded endpoint；不能逐 master 点 encode，不能先平均不同子图的 encoder 输入，也不能用 official baseline SLat 硬匹配 dense ID。

### 5.2 Shape baseline reference

Shape 也需要同样定义零可见 proposal 的 fallback，否则隐藏 master ID 在 shape flow 中没有来源。

对每个 context：

1. baseline geometry 变到该 context 的 local C1024 O-Voxel；
2. official shape encoder、`sample_posterior=False` 编码一次；
3. 映射到该 context 覆盖的 dense master ID；
4. normalize 成 shape flow endpoint 空间。

得到：

```text
shape_baseline_ref[c,m]
shape_baseline_valid[c,m]
```

同样按固定 tile-center Gaussian 预融合：

```text
shape_baseline_global[m]
```

Encoder feature 在这里仅作为明确的 baseline fallback endpoint，不作为 noise、初始 state 或新 support，也不参与有可见 proposal 时的融合。

### 5.3 fallback cache 要求

缓存必须记录：

- baseline geometry/PBR hash；
- master support hash；
- context mapping 和 camera hash；
- nearest-triangle mapping 和每个 context 的 face visibility hash；
- encoder checkpoint 与 normalization；
- 每个 master ID 的 reference count、sum weight 和来源 context。

`shape_baseline_global` 或 `tex_baseline_global` 任一 master ID 没有有效 reference 时，正式运行直接停止；第一版不做 nearest-token 补洞。

## 6. 独立 noise、local state 与 global endpoint

每个 context 独立采样：

```text
epsilon_shape_c   ~ N(0,I)
epsilon_texture_c ~ N(0,I)
```

seed 由 `(global_seed,stage,yaw,tile_id)` 决定即可。禁止 shared noise by master ID，也不要求同一 ID 在不同 context 中初始 feature 相同。

每个 timestep 保存的是：

```text
local noisy states:
    shape_x_t[c]
    texture_x_t[c]

global clean endpoints:
    shape_x0_global[m]
    texture_x0_global[m]
```

全局不维护共享 noisy `z_global(t)`，也不融合 velocity。

对融合后的 endpoint：

```text
local_x0_c = gather(global_x0, context_master_ids_c)
local_v_c  = sampler._xstart_to_pred(local_x_t_c, local_x0_c, t)
local_x_next_c = euler(local_x_t_c, local_v_c, t, t_next)
```

同一 global endpoint 回到不同 context 后得到不同 `local_v_c` 是正确结果。

## 7. 真实 multi-batch

训练本身就是 `[B,...]`，正式实现直接使用真实 batch：

```text
flow_batch_size       = 44
decode_batch_size     = 12
pbr_encode_batch_size = 13
```

要求：

- SparseTensor coords 的 batch column 是真实 `0..B-1`；
- state、shape concat、`cond.global`、`cond.proj` 和 negative condition row 完全对齐；
- unpack 后按 `context_id` 恢复；
- tail batch 是较小的真实 `B`；
- 不写 B=1 Python fallback；
- 不新增 batch/serial 对照路线。

每个 timestep 跨所有 physical batches 保留一个 Jacobi barrier：所有 active context 的 direct endpoint 完成前，不更新任何 local state。

## 8. Shape flow

### 8.1 当前步 prediction

在 timestep `t_k → t_next`，冻结全部 `shape_x_t[c]`，真实 batch 执行：

```text
pred_shape_x0[c] = shape_model_prediction(
    x_t=shape_x_t[c],
    t=t_k,
    condition=shape_condition[c],
).pred_x_0
```

每个 context 每个 timestep 只做一次当前 prediction。不得 suffix rollout 到 0。

### 8.2 visible/Gaussian/fallback 融合

对 master ID `m`，取本步所有有效 proposal：

```text
C_visible(m) = {
    c | mapping[c,m] valid AND visible[c,m] == 1
}
```

Gaussian：

\[
w_{c,m}=\exp\left(-\frac{\|uv_{c,m}-center_c\|^2}{2\sigma^2}\right)
\]

默认 virtual-4096 `sigma=256 px`。

融合：

```text
len(C_visible)==1:
    shape_x0_global[m] = 唯一 visible pred_shape_x0[c,m]

len(C_visible)>1:
    shape_x0_global[m] = Gaussian weighted mean of visible endpoints

len(C_visible)==0:
    shape_x0_global[m] = shape_baseline_global[m]
```

不可见 proposal 不进入 numerator/denominator。累加使用 FP32。

### 8.3 回到各 context 更新

```text
shape_x0_local[c] = gather(shape_x0_global, mapping[c])
shape_v_eff[c] = _xstart_to_pred(shape_x_t[c], shape_x0_local[c], t_k)
shape_x_next[c] = euler(shape_x_t[c], shape_v_eff[c], t_k, t_next)
```

最后一步的 `shape_x0_global` 定义为唯一：

```text
shape_global_final
```

## 9. Shape decode

在第一视角 49 个 canonical tile mapping 上 gather `shape_global_final`，用 official local C1024 shape decoder 解码并 exact local→world。

第一版继续使用当前单视图正式路线的二维 face ownership 合并：

```text
face centroid 投影到第一视角 4096
→ 覆盖该位置的 tile 中选择 Gaussian 权重最大者
→ concatenate owned faces
```

禁止把多个 mesh revoxelize 到 global O-Voxel、重新做 Hermite/QEF 或让 mesh merge 改写 master support。

Texture stage 的 shape concat 始终来自同一个 `shape_global_final`，不是各 context 自己的 final shape state。

## 10. Texture flow

### 10.1 当前步 texture endpoint

每个 context gather：

```text
shape_concat[c] = gather(shape_global_final, mapping[c])
```

冻结全部 `texture_x_t[c]`，真实 batch 执行：

```text
pred_tex_x0[c] = texture_model_prediction(
    x_t=texture_x_t[c],
    t=t_k,
    concat_cond=shape_concat[c],
    condition=texture_condition[c],
).pred_x_0
```

### 10.2 decode 到 PBR 并做 visible donor fusion

真实 decode batch：

```text
pred_tex_raw[c] = inverse_normalize(pred_tex_x0[c])
decoded_pbr[c] = decode_latent(shape_concat_raw[c], pred_tex_raw[c], C1024)
```

对每个 dense master 的 `baseline_nearest_point[m]` 查询各 context decoded PBR。只有：

```text
mapping valid
AND visible[c,m] == 1
AND masked trilinear denominator valid
AND PBR finite
```

才能成为 donor。

物理 PBR 使用相同三态规则：

```text
1 个 visible donor：直接使用
多个 visible donor：tile-center Gaussian fusion
0 个 visible donor：使用 baseline 1024 PBR field 在 baseline_nearest_point[m] 的查询值
```

把 fused global PBR 连续查询回每个 context 的完整 local C1024 support；无有效 global query 的 local row 保留该 row 的 baseline PBR reference，不能填零。

### 10.3 batch PBR re-encode

沿用已完成 `cuda5_cycle` 路径：

```text
E_fused[c] = normalize(PBR_encoder(P_fused[c] * 2 - 1))
E_self[c]  = normalize(PBR_encoder(P_self[c]  * 2 - 1))

tex_x0_corrected[c] = pred_tex_x0[c] + (E_fused[c] - E_self[c])
```

cycle coefficient 第一版固定为 1。这样 PBR 没有改变时，不把普通 encode/decode drift 当作多视图 correction。

### 10.4 corrected endpoint 的 visible/Gaussian/fallback 融合

对 master ID `m`，仍使用同一份冻结 `visible[c,m]`：

```text
len(C_visible)==1:
    texture_x0_global[m] = 唯一 visible tex_x0_corrected[c,m]

len(C_visible)>1:
    texture_x0_global[m] = Gaussian weighted mean of visible corrected endpoints

len(C_visible)==0:
    texture_x0_global[m] = tex_baseline_global[m]
```

这里的 `tex_baseline_global[m]` 正是第 5.1 节“各子图从 baseline PBR query 后分别 encode，再做 Gaussian 平均”的缓存。

### 10.5 回到各 context 更新

```text
tex_x0_local[c] = gather(texture_x0_global, mapping[c])
tex_v_eff[c] = _xstart_to_pred(texture_x_t[c], tex_x0_local[c], t_k)
texture_x_next[c] = euler(texture_x_t[c], tex_v_eff[c], t_k, t_next)
```

最后一步的 `texture_x0_global` 定义为唯一：

```text
texture_global_final
```

## 11. 联合 Shape + Texture 执行顺序

“联合超分”第一版按 Pixal3D 原生依赖执行，不做 shape/texture timestep 交错：

```text
第一视角建立唯一 dense master
→ 预计算 visibility
→ 预计算 shape/texture baseline fallback endpoint
→ 多 tile/多视图 shape flow
→ shape_global_final
→ decode 高分辨率 geometry
→ 多 tile/多视图 texture flow
→ texture_global_final
→ joint shape+texture decode
→ 4096 geometry / normal / PBR / relit RGB
```

这已经同时超分 shape 和 texture；不是旧路线的 fixed-baseline-geometry texture-only SR。

## 12. 最小主循环

```python
master = load_or_build_first_view_dense_master()
contexts = map_all_view_tiles_to_master(master)
nearest_faces = map_master_to_nearest_baseline_triangles(master)
visibility = precompute_per_context_tile_face_visibility(
    nearest_faces,
    contexts,
)

shape_base = precompute_shape_baseline_endpoints(master, contexts)
tex_base = precompute_texture_baseline_endpoints_from_pbr(master, contexts)

shape_state = init_independent_noise(contexts, stage="shape")
for t, t_next in shape_schedule:
    frozen = snapshot(shape_state)
    pred = predict_shape_in_real_batches(frozen, contexts, t)
    shape_x0_global = visible_gaussian_or_fallback(
        pred,
        visibility,
        fallback=shape_base,
    )
    shape_state = local_velocity_euler_batches(
        frozen,
        gather(shape_x0_global),
        t,
        t_next,
    )

shape_global_final = shape_x0_global
final_geometry = decode_first_view_tiles(shape_global_final)

texture_state = init_independent_noise(contexts, stage="texture")
shape_concat = gather_for_contexts(shape_global_final)
for t, t_next in texture_schedule:
    frozen = snapshot(texture_state)
    pred = predict_texture_in_real_batches(
        frozen,
        shape_concat,
        contexts,
        t,
    )
    pbr_self = decode_pbr_in_real_batches(shape_concat, pred)
    pbr_fused = visible_gaussian_pbr_or_baseline(
        pbr_self,
        visibility,
    )
    corrected = cycle_reencode_in_real_batches(
        pred,
        pbr_fused,
        pbr_self,
    )
    texture_x0_global = visible_gaussian_or_fallback(
        corrected,
        visibility,
        fallback=tex_base,
    )
    texture_state = local_velocity_euler_batches(
        frozen,
        gather(texture_x0_global),
        t,
        t_next,
    )

texture_global_final = texture_x0_global
final = decode_joint(shape_global_final, texture_global_final)
```

## 13. 第一版必须保存

### Master / mapping

```text
support/master_support.pt
support/context_mapping.pt
support/context_mapping_stats.json
support/master_nearest_triangle.pt
support/face_visibility_per_context.pt
support/frozen_visibility.pt
support/visibility_stats.json
```

### Baseline fallback

```text
baseline/shape_endpoint_per_context.pt
baseline/shape_endpoint_global.pt
baseline/texture_endpoint_per_context.pt
baseline/texture_endpoint_global.pt
baseline/reference_count_and_weights.json
```

### 每个 timestep

```text
shape/step_XX/global_endpoint.pt
shape/step_XX/visible_count.pt
shape/step_XX/fallback_mask.pt
shape/step_XX/stats.json

texture/step_XX/global_endpoint.pt
texture/step_XX/visible_count.pt
texture/step_XX/fallback_mask.pt
texture/step_XX/pbr_fusion_stats.json
texture/step_XX/cycle_stats.json
```

### 最终结果

```text
final/shape_global_final.pt
final/texture_global_final.pt
final/final_per_vertex_pbr_mesh.pt
final/final_per_face_pbr_mesh.pt
final/render_rgb_4096.png
final/render_alpha_4096.png
final/render_normal_camera_4096.png
final/render_normal_world_4096.png
final/pbr_base_color_4096.png
final/pbr_metallic_4096.png
final/pbr_roughness_4096.png
final/pbr_alpha_4096.png
final/depth_4096.pt
```

## 14. 运行时硬约束

正式运行中直接断言：

1. 全程只有第一视角建立的 master ID；
2. shape/texture 的 master ID、顺序和 mapping 完全一致；
3. 附加 view 没有创建 support；
4. 每个 master ID 通过三角面最近邻绑定 baseline face，不使用 nearest vertex/O-Voxel parent；
5. 每个 `(view,tile)` 都有独立 face visibility，未广播 view-level bit；
6. visibility 在 flow 前冻结且只用于 donor/fusion；
7. 每个 context/timestep 只有一次 direct prediction；
8. 所有 batch prediction 完成前不更新 state；
9. 一个 visible proposal 原样返回；
10. 多个 visible proposal 只按 tile-center Gaussian 融合；
11. 零 visible proposal 必须命中对应 baseline endpoint；
12. baseline endpoint 覆盖全部 master ID；
13. 每个 texture fallback encoder row 是完整子图 SLat，不存在逐点 encode；
14. PBR query denominator 和 finite gate 生效；
15. local state 独立，不存在 shared noise/global noisy state；
16. 融合 endpoint 回到 local 后才计算 velocity；
17. 不存在 velocity averaging；
18. flow/decode/PBR encode 都是真实 `[B,...]`；
19. 所有输出 feature/PBR finite；
20. 最终图真实为 4096×4096。

不需要新增测试或 smoke 路线，只保留这些在线断言、checkpoint 和诊断。

## 15. 第一版明确不做

为了先跑通，暂不加入：

- 侧面/背面 view 扩充 master support；
- 每 timestep 根据当前 shape 重新渲染 visibility；
- disagreement/confidence gating；
- learnable view 权重；
- 动态 Gaussian sigma；
- shape/texture timestep 交错；
- global O-Voxel/Hermite/QEF mesh merge；
- hidden token nearest-neighbor 补洞；
- B=1 或 serial 对照分支。

后续如果第一版已经完整运行，再单独评估 dynamic visibility 或更强的 hidden-region prior；不能在第一版同时引入。

## 16. 一句话定义

```text
第一视角 GT 决定唯一 dense master support；
其他 view/tile 只对这套 ID 提 direct endpoint proposal；
直接可见者按 tile-center Gaussian 融合；
无人直接看见者使用各子图从 baseline geometry/PBR 编码后得到的 global fallback endpoint；
融合 endpoint 回到每个独立 local x_t 计算自己的 velocity；
先完成 shape，再在同一 master 上完成 PBR decode/fusion/re-encode 的 texture flow。
```
