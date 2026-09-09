# Cartesian cube × 子图 baseline 融合尝试

## 最终结论

可行的结构不是“每个 global C64 cube 独立进入 Shape512”，而是：

> **camera-compatible local cube 是 generation domain；global C64 cubes 只做 output ownership / storage domain。**

即：围绕头部和相机视锥运行一次完整 fresh SS baseline，按严格 similarity 返回 global，然后把同一连续 mesh 的 triangle 分配给它跨越的 global cubes。不要在 Shape512 前混入 inherited support，也不要先把 mesh 沿六个 cube 平面裁断。

## 三个实际尝试

### 1. Fresh local decode → 精确 Cartesian cube 裁剪

将上一轮 B fresh mesh 放回 global 后，裁到 C256 cube `(192,64,128)`，global bounds 为：

```text
[0.25,-0.25,0.00] → [0.50,0.00,0.25]
```

得到 `737,086 vertices / 1,469,744 faces`。cube 内机械眼环和鳞甲细节保留，证明“fresh local generation + cube ownership”方向成立。

但沿六个平面裁 mesh 会产生开放矩形切面。给 cube 加 global C256 薄壳的 naive union 虽使正面法向高频提高，但多视角出现明显薄带和缺口，不能作为最终实现。

产物：

- [fresh exact cube 六视角](outputs/head_cube_fresh_patch_prototype_cuda4/phase_bc/fixed_multiview_global_coordinates/fresh_exact_cube_gray_contact_sheet.png)
- [global C256 exact cube 六视角](outputs/head_cube_fresh_patch_prototype_cuda4/phase_bc/fixed_multiview_global_coordinates/global_C256_exact_cube_gray_contact_sheet.png)
- [naive shell hybrid 六视角](outputs/head_cube_fresh_patch_prototype_cuda4/phase_bc/fixed_multiview_global_coordinates/hybrid_fresh_interior_global_shell_gray_contact_sheet.png)
- [prototype metrics](outputs/head_cube_fresh_patch_prototype_cuda4/metrics.json)

### 2. Fresh C32 + gated inherited interface support

保留全部 2,729 个 fresh C32，仅加入与 fresh 距离不超过 `sqrt(2)` cell 且位于 global `X<0.25` 颈部侧的 inherited token：

```text
fresh C32       2729
added unique     521
gated C32       3250
fresh C64      11103
gated C64      13598
```

实验保持共同 C32 行顺序不变并重置 seed 4202，所以共同 fresh rows 的初始 Gaussian 完全一致；C64 也优先按 baseline 共同坐标顺序排列并重置 seed 4203。

结果仍明显退化：机械眼同心环变钝，背面和头顶出现额外碎片。正视角 normal-gradient mean 从 fresh 的 `0.08248` 降至 `0.06877`。这说明 Shape512 的 fresh occupancy posterior 对少量 support 插入也敏感；**不要在 flow 前做 support union，即使只加接口邻居。**

- [fresh baseline 六视角](outputs/head_gated_fresh_support_cuda4/phase_bc/fixed_multiview_global_coordinates/fresh_baseline_gray_contact_sheet.png)
- [gated support 六视角](outputs/head_gated_fresh_support_cuda4/phase_bc/fixed_multiview_global_coordinates/gated_fresh_interface_gray_contact_sheet.png)
- [gated metrics](outputs/head_gated_fresh_support_cuda4/metrics.json)

### 3. 完整 global C256 中替换一个 fresh cube

从 global C256 删除完全位于目标 cube 内的 faces，保留跨界 triangles 防止立即开缝，再插入 fresh exact-cube faces。组合场景为 `124,436,855 faces`。

正面和主要侧视角能看到 fresh 眼环/鳞甲进入 global 场景，证明坐标和尺度正确；但斜后视角仍暴露轴对齐 ownership 边界。因此单 cube replacement 只能作为 proof of concept。

- [完整 global + fresh cube 六视角](outputs/head_full_global_fresh_cube_cuda4/phase_bc/fixed_multiview_global_coordinates/full_global_C256_fresh_cube_gray_contact_sheet.png)
- [组合规则](outputs/head_full_global_fresh_cube_cuda4/manifest.json)

## 推荐实现：一次生成，八个 cube 持有

fresh head patch 在 global 中实际跨越 8 个 C64 cube，而不是一个：

```text
38 (128, 64,128)  1,076,616 faces
39 (128, 64,192)    435,697
42 (128,128,128)    884,072
43 (128,128,192)    227,448
54 (192, 64,128)  1,473,120
55 (192, 64,192)    723,471
58 (192,128,128)    811,817
59 (192,128,192)    306,001
```

已实现以下 partition：

1. local camera-compatible cube 中跑一次 fresh SS → Shape512 → C64 → Shape1024 → native decode；
2. 通过 `X_global=b+sR^Tq` 无畸变返回 global；
3. 每个 triangle 根据 global centroid 指派唯一 C64 owner；
4. triangle 本身不裁剪，即使跨越 cube 边界也由单个 owner 完整持有；
5. 重组时拼接 8 个 owner mesh；重复的边界 vertex 允许仅做零位移 index weld。

验证结果：

```text
source faces        5,938,242
partition faces     5,938,242
coverage exactly once  true
original SHA256     f633b873...99ae386
reassembled SHA256  f633b873...99ae386
geometry byte-identical true
```

因此每个 global cube 都能持有子图 baseline 质量的几何片段，而全部 cube 重组后与原 fresh local native decode **逐字节几何等价**，不会因为 cube 划分再次损失细节。

- [partition manifest](outputs/head_fresh_patch_global_cube_partition/manifest.json)
- [partition script](partition_fresh_patch_to_global_cubes.py)

## 下一步接口原则

global 与 fresh patch 的最终接口应只在 decode 后处理：

- local generation 必须带 halo；
- fresh patch 内部完全冻结；
- 只在颈部的窄 transition band 建立 global/fresh 表面对应；
- 在该窄带做边界对齐和 index weld/remesh；
- 不把 global/inherited support 送进 fresh Shape512/Shape1024 flow。

当前实验已经排除了两个错误方向：单 cube 硬裁薄壳、flow 前 support union。可工作的核心已经验证为“一次 fresh local generation，之后按 global cube owner 分发”。
