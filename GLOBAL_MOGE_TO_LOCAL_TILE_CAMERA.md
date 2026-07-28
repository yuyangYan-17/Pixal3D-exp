# Global MoGe 相机到 Local Tile 相机与点坐标变换

本文档对应 `pixal3d_projective_tile_generation_eval.py` 中的实现，目标是：

1. 在 canonical 1024 图上用 MoGe-2 推断 global 相机。
2. 把相机内参连续缩放到 canonical 4096。
3. 在 4096 图上按 `tile_size=1024, stride=512` 切 tile。
4. 为每个 tile 构造 Pixal3D 可使用的、光心位于 tile 中心的 local 相机。
5. 在需要复用 global 支持点时，将 global `q` 点严格变换到 local tile `q`。
6. 将 local tile 生成的点严格反变换回 global 相机坐标。

这里不使用点云质心、点云包围盒或 min-max 归一化。X/Y 的移动和缩放完全由相机焦距、crop、像素位置和点深度决定。

## 1. 坐标约定

Pixal3D 的规范坐标记为：

```text
q = (qx, qy, qz), q ∈ [-1, 1]³
```

相机位于规范物体前方，camera-space 中看向 `-Z`。`q` 到 camera-space 点的变换为：

```text
C(d) = (0, 0, -d)
P(q; d, s) = C(d) + q / (2s)
```

其中：

- `d` 是 Pixal3D camera distance；
- `s` 是 `mesh_scale`；
- 点的正深度是：

```text
D = -Pz = d - qz / (2s)
```

像素投影约定为：

```text
u = fx X / D + cx
v = -fy Y / D + cy
```

负号来自图像坐标的 Y 轴向下。

## 2. Global MoGe 相机从 1024 缩放到 4096

MoGe-2 在 canonical 1024 图上给出 global 水平 FOV：

```text
theta_g
```

Pixal3D 再根据 FOV 计算 `distance=d_g`。设：

```text
W_g = H_g = 1024
f_g,1024 = W_g / (2 tan(theta_g / 2))
c_g,1024 = (512, 512)
```

canonical 4096 与 canonical 1024 是同一张图的连续缩放，因此：

```text
a_x = 4096 / 1024 = 4
a_y = 4096 / 1024 = 4

f_g,4096,x = a_x f_g,1024
f_g,4096,y = a_y f_g,1024
c_g,4096   = (2048, 2048)
```

相机 FOV、`distance` 和 `mesh_scale` 不因图像 resize 改变；只有以像素为单位的 `f`、`cx`、`cy` 随分辨率缩放。

## 3. 4096 tile 布局

使用：

```text
tile_size = 1024
tile_stride = 512
```

每个轴的起点为：

```text
0, 512, 1024, 1536, 2048, 2560, 3072
```

所以一共是 `7 × 7 = 49` 个 tile。

一个 tile 的 4096 crop box 记为：

```text
B = (x0, y0, x1, y1)
W_c = x1 - x0
H_c = y1 - y0
```

crop 后送入流程的尺寸记为 `W_t × H_t`，当前是 `1024 × 1024`：

```text
r_x = W_t / W_c
r_y = H_t / H_c
```

当前 `W_c=H_c=W_t=H_t=1024`，所以 `r_x=r_y=1`。公式仍保留 resize 系数，避免以后改变 tile 输出大小时出错。

## 4. Exact off-axis crop 相机

如果不移动 global 几何，只想直接渲染 global 相机的 crop，那么 tile 内参是：

```text
fx_off = f_g,4096,x r_x
fy_off = f_g,4096,y r_y

cx_off = (c_g,4096,x - x0) r_x
cy_off = (c_g,4096,y - y0) r_y
```

`cx_off/cy_off` 可以位于 tile 外。例如当前数据：

```text
tile 26, box=(2560,1536,3584,2560): cx_off=-512
tile 27, box=(3072,1536,4096,2560): cx_off=-1024
```

这是正确的离轴 crop 相机，适合“global 几何不变，只换渲染窗口”的情况。

## 5. Pixal3D centered local tile 相机

Pixal3D 的现有投影条件只接受中心光心，因此 local 相机使用：

```text
fx_l = fx_off
fy_l = fy_off
cx_l = W_t / 2
cy_l = H_t / 2
```

对应 FOV：

```text
theta_l,x = 2 atan(W_t / (2 fx_l))
theta_l,y = 2 atan(H_t / (2 fy_l))
```

仅修改 FOV、保留 global distance 是错误的。因为中心 local 相机面对的是一个完整的 local 规范立方体，必须重新计算其 Pixal3D distance：

```text
d_l = distance_from_fov(
    theta_l,x,
    grid_point=(-1,0,0),
    target_point=(-extend_pixel, W_t-1+extend_pixel),
    mesh_scale=s_l,
    image_resolution=W_t,
)
```

在 `extend_pixel=0` 时近似满足：

```text
d_l s_l = fx_l / W_t
```

本项目的 1024/4096/tile 配置下：

```text
f_g,1024 = 1934.887
d_g      = 1.889538

fx_l     = 7739.549
d_l      = 7.558153

(d_l s_l) / (d_g s_g) = 4
```

这个 4 倍就是 4096 上 1024 crop 相对于整幅图的重规范化倍率。

## 6. 为什么 1024 和 512 使用同一组 FOV/distance

tile 最初从 4096 上裁成 1024，shape512 阶段再把 tile resize 到 512。resize 不改变视锥，只改变像素内参：

```text
f_l,512 = f_l,1024 × 512 / 1024
c_l,512 = (256, 256)
```

FOV、`distance`、`mesh_scale` 保持不变。因此 1024 和 512 阶段应传同一组：

```text
(camera_angle_x=theta_l, distance=d_l, mesh_scale=s_l)
```

不能在 512 阶段再次按 512/1024 修改 distance。

## 7. Global q 到 local tile q

### 7.1 Global q 投影到 4096

对 global 点 `q_g`：

```text
P_g = (0,0,-d_g) + q_g / (2s_g)
D_g = d_g - q_g,z / (2s_g)
```

投影到 4096：

```text
u_4096 = f_g,4096,x P_g,x / D_g + 2048
v_4096 = -f_g,4096,y P_g,y / D_g + 2048
```

### 7.2 4096 crop 坐标变成 tile 像素

```text
u_t = (u_4096 - x0) r_x
v_t = (v_4096 - y0) r_y
```

### 7.3 保留规范深度，不保留 global 物理深度

定义：

```text
q_l,z = q_g,z
D_l = d_l - q_l,z / (2s_l)
```

必须使用 local depth `D_l` 做 local 反投影。若错误地沿用 `D_g`，在 `d_l≈4d_g` 时会把 local `q_z` 推到规范立方体之外。

### 7.4 从 tile 像素反投影 local X/Y

```text
X_l = (u_t - cx_l) D_l / fx_l
Y_l = -(v_t - cy_l) D_l / fy_l
Z_l = -D_l

q_l = 2s_l (P_l - (0,0,-d_l))
```

按构造有：

```text
q_l,z = q_g,z
```

其 X/Y 闭式写法为：

```text
tile_center_4096,x = x0 + cx_l / r_x
tile_center_4096,y = y0 + cy_l / r_y

q_l,x = 2s_l D_l [
    q_g,x / (2s_g D_g)
    + (c_g,4096,x - tile_center_4096,x) / f_g,4096,x
]

q_l,y = 2s_l D_l [
    q_g,y / (2s_g D_g)
    + (tile_center_4096,y - c_g,4096,y) / f_g,4096,y
]
```

可以看到：

- 光心偏移项乘以点深度；
- 缩放项包含 `D_l / D_g`；
- 这不是固定平移或简单 bbox normalization。

## 8. Local tile q 到 global q

反向变换同样使用投影/反投影。

先把 local q 投影到 tile：

```text
P_l = (0,0,-d_l) + q_l / (2s_l)
D_l = d_l - q_l,z / (2s_l)

u_t = fx_l P_l,x / D_l + cx_l
v_t = -fy_l P_l,y / D_l + cy_l
```

恢复 4096 像素：

```text
u_4096 = u_t / r_x + x0
v_4096 = v_t / r_y + y0
```

保留规范深度：

```text
q_g,z = q_l,z
D_g = d_g - q_g,z / (2s_g)
```

用 global 深度和 global 内参反投影：

```text
X_g = (u_4096 - 2048) D_g / f_g,4096,x
Y_g = -(v_4096 - 2048) D_g / f_g,4096,y
Z_g = -D_g

q_g = 2s_g (P_g - (0,0,-d_g))
```

因为 `D_l/D_g` 随 `q_z` 变化，这个逆变换一般不是单个 4×4 affine matrix。导出 global GLB 时必须逐顶点执行上述变换。

## 9. 两种代码路径如何使用

### 9.1 当前可执行的 tile-only 路径

该路径不把 global 点传进 tile：

```text
tile image
→ local SS32
→ local shape512/C64
→ local shape1024
→ local texture1024
```

它只使用从 global 相机和 crop 推导出的 centered local 相机：

```text
(theta_l, d_l, s_l)
```

生成出的几何天然位于 local `q`。只有在需要拼回 global 相机空间或导出 global GLB 时，才使用第 8 节的逐顶点逆变换。

### 9.2 文件后半段的注释版 projective-support 路径

该路径会先选出投影落在 tile 内的 global 支持点，再按第 7 节变到 local `q`，之后量化到 local C32/C64/C1024。

量化前不做 clamp。由于透视规范立方体的远平面投影比中心平面窄，少量落在 tile 边缘的 global 点可能自然落到 local `[-1,1]³` 外；这些点应丢弃并统计，不能压到边界。

## 10. 实现对应关系

代码中的主要函数：

```text
_derive_tile_camera
    global MoGe camera + 4096 crop -> centered local camera

_project_global_q_to_1024_and_4096
    global q -> global pixels

_global_q_to_centered_tile_q
    global q -> crop/resize -> local q

_centered_tile_q_to_global_q
    local q -> inverse crop -> global q/global camera point

_decoder_vertices_to_global_camera
    decoder vertex [-0.5,0.5] -> local q -> global camera XYZ
```

数值测试要求：

```text
global q -> local q -> global q
max_abs_error < 2e-5

local q -> global q -> local q
max_abs_error < 2e-5
```

当前 float32 随机点测试的典型误差约为 `1e-6`，像素往返误差低于 `4e-4 px`。

## 11. 常见错误

不要执行以下操作：

1. tile FOV 缩小 4 倍，但保留 global `distance × mesh_scale`。
2. 把 exact off-axis `cx_off/cy_off` 写进 JSON，却让生成阶段继续使用未变换的 global/local 点。
3. global→local 反投影时直接沿用 global 物理深度。
4. 对 tile 点云做质心或 bbox 归一化代替相机变换。
5. local→global 使用固定 affine shear；重规范化后的正确变换依赖 `q_z`。
6. 在 1024 tile resize 到 512 时再次修改 FOV 或 distance。
7. 同时对点做 local recanonicalization、又在 local 相机上重复施加 off-axis shift；这会重复计算光心偏移。
