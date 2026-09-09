# Global C256 切块与正常子图 C32 的占据空间诊断

## 结论

你的判断基本正确，而且不是一个很弱的相关性：**质量差异的重要来源就是 support 在 canonical cube 中的占据位置和截断方式不一致。** `0..63` 只说明索引合法，并不说明它与正常子图生成处在同一种分布中。

但需要把“靠边”拆成三个同时发生的问题：

1. **位置偏置**：Cartesian global cube 切出的头部只落在 local cube 一侧；
2. **开放截断**：很多边界 token 的几何在相邻 global cube 中仍连续，切块把连续表面硬截断了；
3. **相机/视锥语义不一致**：正常子图的 canonical cube 是围绕前景和相机视锥定义的完整局部物体空间，而 global C256 的一个轴对齐 C64 是 global 物体空间的四分之一，不天然等于一张子图对应的视锥体积。

因此，`global C256 → 切 C64 → down C32 → Shape512 → C64 → Shape1024` 虽然后半段调用和 baseline 一样，**进入 Shape512 的随机变量并不服从 baseline C32 support 的训练分布**。

## 直接证据

### 1. 2D 头部 crop 不是一个 Cartesian C64 cube

把 global C256 support 投影到原 4096 图，按头部 ROI `(3008,1632)-(4032,2656)` 统计，命中最多的 cube 是：

```text
start = (192, 64, 128)
cube tokens = 10,755
投影进头部 ROI = 8,560
```

但同一 ROI 还大量命中：

```text
(192,128,128): 4,413
(192, 64, 64): 3,927
(192, 64,192): 3,474
(192,128, 64): 1,961
(192,128,192): 1,354
```

也就是说，一张 2D 头部图对应的是沿视线穿过多个 Y/Z Cartesian cube 的视锥区域。拿单个 C64 cube 不能完整代表这张子图的三维生成域。

### 2. token 数几乎相同，但空间分布完全不同

| C32 support | token | XYZ 范围 | canonical centroid | boundary ratio | 中央 75% cube 内比例 |
|---|---:|---|---|---:|---:|
| 正常 `crop + MoGe` SS | 2,511 | `[2..26, 0..30, 0..28]` | `[-0.027,+0.003,+0.025]` | 1.04% | 84.51% |
| global-compatible fresh SS | 2,729 | `[0..26, 0..31, 6..28]` | `[-0.082,-0.023,+0.071]` | 2.46% | 85.56% |
| metric inherited | 4,131 | `[0..26, 0..31, 0..31]` | `[-0.117,-0.040,-0.003]` | 9.90% | 65.36% |
| legacy best C64 cube down C32 | 2,519 | `[0..15, 0..31, 0..31]` | `[-0.281,+0.135,-0.076]` | 15.13% | 40.17% |

最有诊断意义的是第一行与第四行：token 数是 `2511` 对 `2519`，几乎一样；但 legacy support 的 X 只占半个 cube，质心明显偏向 `-X`，中央 50% cube 中只有 `4.64%` token，而正常子图有 `24.17%`。所以质量差不是简单由 token 少造成的。

可视化见 [四种 C32 occupancy 最大投影](outputs/head_cube_occupancy_analysis/occupancy_max_projection.png)，完整数字见 [metrics.json](outputs/head_cube_occupancy_analysis/metrics.json)。

### 3. legacy cube 是切断的表面，不是局部完整对象

对 `(192,64,128)` cube 的六个边界，检查边界 token 在相邻 global C256 cell 是否仍有 6-neighbor：

| cube face | 边界 token | 在 cube 外继续连接 | 比例 |
|---|---:|---:|---:|
| x=0 | 272 | 204 | 75.0% |
| y=0 | 79 | 67 | 84.8% |
| y=63 | 167 | 118 | 70.7% |
| z=0 | 176 | 142 | 80.7% |
| z=63 | 97 | 78 | 80.4% |

这说明 decoder/flow 看到大量“走到 cube 边界突然消失”的表面。正常 SS 的 support 是根据居中条件图生成的局部 occupancy posterior，不会把一块更大物体的任意切面误当成独立对象边界。

## 为什么后半段一样，结果仍差很多

Shape512 并不是只检查坐标是否在 `0..31`：

- 模型配置的 resolution 是 C32，并使用 3D RoPE；真实 sparse coordinates 参与 attention；
- sparse convolution 的邻接图直接由 support 决定；切断跨 cube 邻居后，边界处的 receptive field 已改变；
- projected image feature 根据每个绝对 sparse coordinate、FOV 和 distance 采样；同一张子图下，偏在 cube 左侧的 support 会查询与居中 support 不同的像素/射线；
- `shape_slat_decoder.upsample()` 的 child support 是 Shape512 latent 的函数，因此 C32 的占据偏置会继续传给 C64，而不是被 upsample 自动“洗掉”。

上一轮同 camera/condition 的 B/C 实验也支持这一因果链：重叠坐标 condition token 严格零差，但 fresh 与 inherited 的 C32 Jaccard 只有 `0.1787`；Shape512 endpoint relative L2 已达 `1.3425`；到 C64 Jaccard 又降至 `0.0964`。差异在 512→C64 之前就已经很大。

## 最准确的判断

**是，占据空间是主要原因之一；更准确地说，是“非居中的占据 + Cartesian 硬截断 + 与子图视锥不等价”共同造成的。**

单纯把切出的坐标都减去块起点，只完成了 index relabel，没有完成“把 global 局部三维区域规范化成模型熟悉的 local generation space”。同样，把 legacy support 平移到中心也不够，因为：

- 它仍是被切开的表面；
- 平移后相机投影和 image token 对应会改变；
- 单个 Cartesian cube 仍不能覆盖 2D crop 沿深度对应的多个 cube。

真正公平的路径应是上一轮 Phase B/C 所采用的：先用同一个物理局部中心、各向同性尺度和朝向定义 local cube，再把 global support 通过这一 metric similarity 变换重体素化。该实验中 inherited 已比 legacy 切块更居中，但仍明显差于 fresh SS，说明**消除边缘位置偏置是必要的，却不是充分的；fresh SS 学到的局部 occupancy posterior 本身也很关键。**
