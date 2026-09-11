# C32 分块几何超分失败实验记录

## 目标与判定标准

目标是验证：

```text
baseline C64
  -> decoder.upsample 得到全局 C128
  -> context=32、stride=32 切块
  -> 每块使用 local C32 坐标做 Flow
  -> decoder 得到 C64
  -> 后续 Shape1024 Flow / decoder
```

是否能够达到切块区域的点投影回原输入4096图上crop，然后适当扩展投影区域bbox到1024子图的独立子图直接跑 baseline 的几何质量，尤其是乌龟头部、眼睛、颈部和背甲上的细节。

这里的 normal contact sheet 是生成 mesh 的法向量渲染图，用于视觉比较；没有 GT 法向量，因此没有把它当作定量 GT 指标。

## 结论先行

目前的主要问题已经从“图像 crop 不对”收敛到 **support 分布不对**：全局 C128 直接切出的点，即使经过以下处理：

- 重新映射到 `0..31`；
- 把点的 bbox 压到与独立子图相似的范围；
- 使用同一份独立子图 `canonical_512/canonical_1024` 和 camera；
- 修正 local cube 到 global camera 的透视变换；
- 加 halo，补充邻域点；

仍然不能复现独立子图 baseline 的质量。

最有区分度的实验是：在完全相同的图像和 camera 条件下，只替换 C32 support。

| 版本 | C32 token | C32 范围 | C32 bbox fill | C32 边界比例 | 结果 |
|---|---:|---|---:|---:|---|
| 全局 C128 support 映射到 local C32 | 2702 | `[1,3,1]..[29,31,26]` | 0.1236 | 0.30% | 仍然粗糙，失败 |
| 独立子图 native C32 support | 2413 | `[1,3,1]..[29,31,26]` | 0.1104 | 0.58% | 细节明显恢复 |

两者 token 数、外接 bbox 和索引范围都接近，但结果差异很大。由此可见，问题不是简单的 token 数，也不是只要把索引合法化就能解决。

## 参考 baseline

独立子图实验使用的是从 4096 图内部截取的背甲加头顶区域：

- crop metadata：`outputs/independent_subimage_baseline_compare_cuda4/inputs/back_headtop_inner4096_bbox1024.json`
- 独立 baseline 输出：`outputs/independent_subimage_baseline_compare_cuda4/back_headtop_inner4096_baseline1024/`
- 独立 baseline normal：
  `outputs/independent_subimage_baseline_compare_cuda4/back_headtop_inner4096_baseline1024/multiview_1024/camera_normal_contact_sheet.png`
- native C32 support：
  `outputs/independent_subimage_baseline_compare_cuda4/back_headtop_inner4096_baseline1024/support/coords_c32.pt`

独立子图的 C32 support 是 2413 个点，范围为 `[1,3,1]..[29,31,26]`。它没有把局部几何大面积压在立方体边界上。

## 问题一：全局 C128 切块得到的是被截断的表面

对全局 C128 support 直接按 `32^3` 分块时，局部块表示的是 global object space 中一个轴对齐小立方体内的点。它不等价于“把对应 2D 子图重新放入一个局部 canonical cube 后生成的 C32 support”。

两个典型块的投影诊断如下。投影 bbox 是映射到 512 图像后的坐标：

| 块 | token | 投影 bbox | 投影面积跨度 | 64x64 occupancy fill |
|---|---:|---|---|---:|
| block 54 | 2446 | `[46.9,1.8]..[459.2,511.4]` | `[412.2,509.6]` | 0.2693 |
| block 57 | 545 | `[184.9,109.1]..[333.6,410.4]` | `[148.7,301.3]` | 0.0483 |

block 57 的点只占据 local cube 的一小部分，且大量落在边缘。对应的原始 C32 统计为：

```text
token              = 545
min/max            = [0,0,9] .. [7,19,31]
boundary fraction  = 31.0%
```

这说明切块得到的是连续表面的一段截面：表面继续延伸到相邻 global block，但邻居已经被删掉。Flow 和 decoder 看到的不是完整的局部对象，而是带有人工边界的开放结构。

### 尝试的解决办法：只做 local index 重编号

做法是减去 block start，把每块坐标改成 `0..31`，再送进 C32 Flow。

这个步骤是必要的，因为 C32 的网格分辨率和 RoPE 需要 local 坐标；但它只改变坐标标签，不改变：

- 点在 local cube 中是否贴边；
- 点之间的 3D 邻接关系；
- 被相邻 block 截断的表面；
- support 的稀疏拓扑。

因此它没有解决问题。

## 问题二：只匹配 bbox 或占据比例仍然不够

我把选中的全局点重新压缩到独立子图 C32 的 observed bbox `[1,3,1]..[29,31,26]`，使两者的外接范围相同。然后分别使用：

```text
同一份 independent canonical_512.png
同一份 independent canonical_1024.png
同一个 independent camera
同一套 C32/C64 Flow 和 decoder
```

实验脚本：`run_c32_support_condition_ablation_cuda4.py`

输出：

- 映射 support，失败：
  `outputs/c32_support_condition_ablation_cuda4/mapped/final/multiview_1024/camera_normal_contact_sheet.png`
- native support，质量恢复：
  `outputs/c32_support_condition_ablation_cuda4/independent/final/multiview_1024/camera_normal_contact_sheet.png`
- 完整统计：`outputs/c32_support_condition_ablation_cuda4/summary.json`

两者的 3D support 集合并不相同。映射 support 与 native support 的 3D Jaccard 约为 `0.1783`；最近邻距离也明显不为零。也就是说，它们虽然在投影图上都能拿到正确颜色，例如眼睛区域仍然对应眼睛颜色，但 Flow 看到的 3D 邻接图和局部形状先验已经不同。

### 尝试的解决办法：按 2D bbox 重新缩放 support

这个办法改善了“整个点云只占 cube 一侧”的问题，但它把一个被截断的 global support 做了非刚性压缩，并没有生成独立子图 baseline 会生成的局部 occupancy。结果仍然粗糙，说明 bbox fill 不是充分条件。

## 问题三：local 3D 坐标与图像投影坐标可能不一致

把 global 点减去 block start 后，Flow 使用的是 local C32 坐标；如果图像条件仍然使用原 global projection row，就会出现：

```text
Flow/RoPE 认为 token 位于 local cube 的位置 q_local
图像投影却认为 token 位于 global cube 的位置 q_global
```

这种情况下，点的颜色可能仍看起来正确，但点的 3D 位置、射线和局部图像特征不是同一个物理位置。

### 尝试的解决办法：传入 local-to-global camera transform

实验脚本：`run_c32_camera_aligned_crop_ablation_cuda4.py`

该实验用 affine 把 local C32 endpoint 坐标映射回 global C128，再把这段变换合并到 camera-to-world 矩阵中，使 `ProjGrid` 使用 local 坐标时仍能采样 global crop 的正确透视位置。投影自检误差约为：

```text
median = 14.35 px
p95    = 25.39 px
max    = 35.02 px
```

输出：

`outputs/c32_camera_aligned_crop_ablation_cuda4/final_local_frame/multiview_1024/camera_normal_contact_sheet.png`

结果仍然粗糙，和 bbox 映射版本没有实质改善。因此光心偏移和透视变换确实是实现中必须处理的问题，但它不是当前质量差的唯一主因；修正投影不能修复错误的 support 拓扑。

## 问题四：第一轮 crop 实验存在图像预处理变量

独立 baseline 的 `preprocess_canonical_images()` 会对前景做 alpha/bbox 处理后再生成 `canonical_1024.png` 和 `canonical_512.png`。而第一轮映射实验直接从完整 `canonical_4096.png` 截取 raw 1024 crop，再 resize 到网络输入尺寸。

这使第一轮实验不能单独证明 support 是唯一变量。

后来在 `run_c32_support_condition_ablation_cuda4.py` 中已经改成严格复用独立 baseline 的：

- `canonical_512.png`；
- `canonical_1024.png`；
- camera；
- local C32/C64 projection 坐标。

映射 support 仍失败，native support 仍明显更好。因此图像 crop 预处理不是主要解释。

## 问题五：halo 只能补邻域，不能生成正确的局部 support

### 尝试的解决办法：给 block 加 halo

block 57 的直接 support 只有 545 个点，边界比例 31.0%。我向外扩展 C128 halo 后压到 C32：

- halo C32：2466 个点；
- 边界比例降到 11.9%；
- 但它投影到目标 crop 的 token 只有 951 个，覆盖率 38.6%；
- 目标区域单独截取后的 mesh 只有 6330 个顶点，仍然很稀疏。

输出：

`outputs/c32_halo_block_cuda4/cube_57/final_global/multiview_1024/camera_normal_contact_sheet.png`

继续让整个 halo 走第二阶段 C64 Flow：

- halo C64：10210 个点；
- 目标 crop 只覆盖 4244 个点，覆盖率 41.6%；
- 目标 mesh：151958 个顶点、300838 个面；
- 结果仍然是粗糙的块状局部。

输出：

`outputs/c32_halo_stage2_full_cuda4/cube_57/final_target_independent_frame/multiview_1024/camera_normal_contact_sheet.png`

halo 解决了“边界没有邻居”的一部分症状，但引入了更大的非目标区域，而且没有把目标局部重新组织成 native C32 的分布，所以没有达到独立子图质量。

## 问题六：当前模型没有被训练成“global C128 切块后独立完成 C32”

Shape C32 Flow 学到的是训练数据中的 C32 sparse structure 分布。独立子图 baseline 给出的 native C32 support 正好落在这个分布附近；global C128 切块得到的 support 即使 token 数接近，也可能是：

- 位置偏向 cube 边缘；
- 表面被硬截断；
- 邻接关系缺失；
- 深度方向的局部占据不完整；
- 与独立子图的 support posterior 不同。

所以 C32 Flow 不会自动把它“补成”独立子图的 native support。它只会在给定 sparse index 集合上更新 latent/features；decoder 也不会凭空恢复被 support 删除的邻域。

## 目前最强的对照：直接使用 native C32 support

在 `run_c32_camera_aligned_crop_ablation_cuda4.py --support-source independent` 中，使用独立子图 native C32 support，仍走同样的局部 Flow 和 crop 条件：

```text
outputs/c32_camera_aligned_crop_template_cuda4/
```

normal 输出：

`outputs/c32_camera_aligned_crop_template_cuda4/final_local_frame/multiview_1024/camera_normal_contact_sheet.png`

这个结果可以恢复眼睛环、背甲层次和局部几何纹理，明显接近独立子图 baseline。它是一个诊断性 oracle，因为 native support 是从独立 baseline 得到的，不能直接作为最终全局分块方案；但它证明了局部 C32 Flow 和图像条件本身有能力产生高质量局部几何，真正缺的是可用的 native-like C32 support。

## 失败方案汇总

| 方案 | 目的 | 结果 | 失败原因 |
|---|---|---|---|
| 直接切全局 C128 | 保持 baseline 几何并局部 Flow | 头部、颈部细节弱 | support 是硬截断表面 |
| 坐标减 block start，改成 `0..31` | 修正 local index / RoPE 范围 | 无明显改善 | 只改标签，不改分布和邻接 |
| bbox 重映射到独立 C32 范围 | 消除边缘占据 | 仍失败 | bbox 相同不代表 3D topology 相同 |
| 复用独立图像、camera、local projection | 排除图像条件变量 | mapped 仍差，native 好 | 直接证明 support 是主变量 |
| local-to-global camera transform | 修正光心偏移和透视 | mapped 仍差 | 投影问题是次要问题，不能恢复拓扑 |
| halo 扩展邻域 | 减少边界截断 | 稀疏/块状，仍失败 | 邻域不等于 native local support |
| 先 C32->C64 再第二阶段 C64 Flow | 增加后续几何细化 | 仍未达到子图质量 | 第一阶段 support 分布差异已传到 C64 |

## 当前可保留的解决方向

最终分块代码不能只做：

```text
global C128 points
    -> crop / slice
    -> local index 0..31
    -> C32 Flow
```

需要先建立“局部 canonical generation space”，再得到接近 native C32 prior 的 support：

1. 根据 global camera、目标 crop 和局部物理范围确定 local-to-global similarity transform；
2. 将 global 几何投影到这个 local frame，并用局部 support 生成策略重新组织 C32 occupancy，而不是只截取 global C128 token；
3. Flow、`ProjGrid` 和 RoPE 全部使用 local C32/C64 坐标；图像条件使用同一变换采样；
4. 完成 C32 Flow、decoder 到 C64、C64 Flow 后，再把 mesh 通过 inverse transform 放回 global frame；
5. 最后才处理相邻块的 overlap / anchor / consistency。

目前已经验证第 3 步和局部 Flow 可以工作，也验证了 native support 会恢复细节；尚未解决的是：**如何只从全局 baseline support 稳定地构造 native-like 的局部 C32 support**。bbox、reindex、camera transform 和 halo 四种办法都不足以完成这一步。

## 复现实验

所有实验使用 CUDA4。脚本语法检查已通过：

```bash
CUDA_VISIBLE_DEVICES=4 python -m py_compile \
  run_c32_crop_support_equivalence_cuda4.py \
  run_c32_support_condition_ablation_cuda4.py \
  run_c32_camera_aligned_crop_ablation_cuda4.py
```

主要实验入口：

```bash
CUDA_VISIBLE_DEVICES=4 python run_c32_support_condition_ablation_cuda4.py
CUDA_VISIBLE_DEVICES=4 python run_c32_camera_aligned_crop_ablation_cuda4.py
CUDA_VISIBLE_DEVICES=4 python run_c32_camera_aligned_crop_ablation_cuda4.py \
  --support-source independent
```
