请修改 Pixal3D 当前的 global→tile 相机与三维坐标变换。目标不是简单做 2D crop，也不是保持 global depth，而是为每个 4096 图像 tile 构造一个真正对应局部三维区域的、各向同性的 local generation space。

整个实现必须严格遵循下面的数学定义。不要使用 bbox normalization、点云质心归一化、固定 XYZ 缩放、`q_l,z=q_g,z`、depth clamp 等旧方案。

# 一、目标

现有流程中：

1. Pixal3D global generation space 对应完整 global 图像；
2. 将 canonical 图像放大到 4096；
3. 在 4096 上按 `tile_size=1024, stride=512` 切 tile；
4. 每个 tile 对应更小 FOV，因此 local generation space 不能继续使用 global camera distance；
5. tile 中心射线原本通常不在 global camera Z 轴上，因此还需要重新定义 local camera frame；
6. 最终每个 tile 都应对应一个新的：

$$
g_l\in[-0.5,0.5]^3
$$

local generation cube；

7. 该 cube 对应 global 空间中的一个以 tile 中心表面点为中心、沿 tile center viewing direction 定向的局部三维 cube；
8. XYZ 三轴必须使用同一个尺度，不能出现原方案中 `XY × 4、Z × 1` 的各向异性；
9. 原始高清 tile 图像不要 warp；
10. local 3D 点查询 DINO / projected image feature 时，应 inverse 回 global，再投影到原始 4096 图像，最后换算到 raw tile pixel。

---

# 二、坐标约定

为了避免 Pixal3D 当前 `Z` 正负号造成混淆，统一区分：

## 2.1 generation/world coordinate

定义：

$$
g=(x,y,z)
$$

其中：

$$
g\in[-0.5,0.5]^3.
$$

相机位于 generation/world space：

$$
C=(0,0,d)
$$

并朝向：

$$
-Z
$$

观察原点。

注意：

如果现有代码使用：

$$
q\in[-1,1]^3
$$

并通过：

$$
g=\frac{q}{2s}
$$

变成实际 generation coordinate，则必须显式完成该转换。

不要在代码中混用 `q∈[-1,1]` 和 `g∈[-0.5,0.5]`。

如果 `s=1`：

$$
g=q/2.
$$

---

## 2.2 OpenCV camera coordinates

为了方便投影，定义：

$$
p^{cv}=(X,Y,Z)
$$

满足：

* X：图像右；
* Y：图像下；
* Z：相机前方；
* \(Z>0\)。

global generation point：

$$
g_g=(x_g,y_g,z_g)
$$

变成 global camera coordinate：

$$
\boxed{
p_g^{cv}
=
\begin{bmatrix}
x_g\\
-y_g\\
d_g-z_g
\end{bmatrix}
}
$$

因此：

$$
D_g=d_g-z_g.
$$

投影为：

$$
\boxed{
u_g=f_x\frac{X_g}{Z_g}+c_x
}
$$

$$
\boxed{
v_g=f_y\frac{Y_g}{Z_g}+c_y.
}
$$

这与原 Pixal3D：

$$
v=-f_yY_{\text{world}}/D+c_y
$$

完全等价。

---

# 三、global 1024 相机变成 global 4096 相机

MoGe / Pixal3D 已经得到 global FOV 和：

$$
d_g.
$$

1024 → 4096 只是连续 resize。

因此：

$$
f_{4096,x}=4f_{1024,x}
$$

$$
f_{4096,y}=4f_{1024,y}
$$

以及：

$$
c_{4096}=(2048,2048).
$$

FOV、`distance=d_g`、`mesh_scale` 不变。

实现中形成：

```python
K_global_4096 = [
    [fx4096, 0,      cx4096],
    [0,      fy4096, cy4096],
    [0,      0,      1],
]
```

---

# 四、global 三维点先投影到 global 4096 图像

对于任意 global generation point：

$$
g_g
$$

先转换：

$$
p_g^{cv}
=
[x_g,-y_g,d_g-z_g]^T.
$$

要求：

$$
Z_g>0.
$$

投影：

$$
u_{4096}
=
f_xX_g/Z_g+c_x
$$

$$
v_{4096}
=
f_yY_g/Z_g+c_y.
$$

这一步用于：

1. 判断 global point 是否投影到当前 tile；
2. 后续查询 raw tile 图像 feature；
3. debug correspondence。

一个 tile：

$$
B=(x_0,y_0,x_1,y_1)
$$

使用 half-open membership：

$$
x_0\le u<x_1
$$

$$
y_0\le v<y_1.
$$

不要用固定 global XYZ bbox 替代投影 membership。

---

# 五、计算 tile center ray

tile 中心像素：

$$
u_c=\frac{x_0+x_1}{2}
$$

$$
v_c=\frac{y_0+y_1}{2}.
$$

注意这里是 global 4096 pixel coordinate。

通过 global intrinsics：

$$
r_c
=
K_g^{-1}
\begin{bmatrix}
u_c\\v_c\\1
\end{bmatrix}.
$$

归一化：

$$
\boxed{
\hat r_c
=
\frac{r_c}{\|r_c\|}
}
$$

这是 OpenCV camera coordinate 中的 tile center ray。

---

# 六、求 tile center 对应的三维 anchor

必须为每个 tile 找一个真实的三维中心：

$$
P_c^{cv}.
$$

优先使用 baseline Pixal3D mesh。

从 global camera origin：

$$
O=(0,0,0)
$$

沿：

$$
\hat r_c
$$

进行 ray-mesh intersection。

取第一可见交点。

于是：

$$
\boxed{
P_c^{cv}
=
\rho_c\hat r_c
}
$$

其中：

$$
\boxed{
\rho_c=\|P_c^{cv}\|
}
$$

是 tile center surface point 到 global camera 的真实 ray distance。

不要直接将 MoGe metric depth 与 Pixal3D canonical `d_g` 混用。

必须保证：

$$
P_c,\rho_c,d_g
$$

处于同一个 Pixal3D generation/camera scale。

如果 center ray 没有击中 baseline mesh：

1. 在 tile 中心附近的小窗口寻找有效 foreground ray intersection；
2. 使用其 robust median depth；
3. 再沿真正的 center ray 构造：

$$
P_c=\rho_c\hat r_c.
$$

必须记录 fallback。

如果整个 tile 没有有效 foreground，则跳过 tile。

---

# 七、构造 local camera frame

现在需要把：

$$
\hat r_c
$$

变成 local camera 的正 Z 轴。

在 OpenCV camera coordinates 中定义：

$$
z_l^{(g)}=\hat r_c.
$$

为了尽量保持图像方向，使用 global camera 的 image-down axis：

$$
y_g=(0,1,0).
$$

计算：

$$
x_l^{(g)}
=
\frac{
y_g\times z_l^{(g)}
}{
\|y_g\times z_l^{(g)}\|
}.
$$

然后：

$$
y_l^{(g)}
=
z_l^{(g)}\times x_l^{(g)}.
$$

构造：

$$
\boxed{
R=
\begin{bmatrix}
(x_l^{(g)})^T\\
(y_l^{(g)})^T\\
(z_l^{(g)})^T
\end{bmatrix}
}
$$

必须满足：

$$
RR^T=I
$$

$$
\det R=1
$$

以及：

$$
\boxed{
R\hat r_c=(0,0,1)^T.
}
$$

不要使用会产生 reflection 的矩阵。

---

# 八、根据 tile 实际角度求 local FOV

不要简单使用：

```python
theta_local = theta_global / 4
```

中心 tile 只有在小角度近似下才接近这种关系。

正确做法是使用 tile 四个 corner ray。

四个 corner：

$$
(u_i,v_i).
$$

计算：

$$
r_i=
\operatorname{normalize}
\left(
K_g^{-1}
[u_i,v_i,1]^T
\right).
$$

变换到 local camera frame：

$$
r_i'=Rr_i.
$$

对于每个 corner：

$$
s_{x,i}=\frac{r'_{i,x}}{r'_{i,z}}
$$

$$
s_{y,i}=\frac{r'_{i,y}}{r'_{i,z}}.
$$

计算：

$$
t_x=\max_i|s_{x,i}|
$$

$$
t_y=\max_i|s_{y,i}|.
$$

如果 Pixal3D 只能接受 square / 单一 `camera_angle_x`，使用：

$$
\boxed{
t=\max(t_x,t_y)
}
$$

从而：

$$
\boxed{
\theta_l=2\arctan(t).
}
$$

如果代码允许独立 fx/fy，则分别使用：

$$
\theta_{l,x}=2\arctan(t_x)
$$

$$
\theta_{l,y}=2\arctan(t_y).
$$

注意：

边缘 tile 的左右角度一般不完全对称。

如果要求 centered symmetric Pixal3D camera，就只能使用最大绝对角度，使整个 tile ray bundle 都被包含。

因此某一侧可能存在少量 margin，这是 centered symmetric camera 本身造成的，不要人为再做非均匀 XYZ scale。

---

# 九、FOV 变小后重新计算 local camera distance

local generation cube：

$$
[-0.5,0.5]^3.
$$

半边长：

$$
h=0.5.
$$

理想 pinhole 关系：

$$
\boxed{
d_l=
\frac{h}{\tan(\theta_l/2)}
}
$$

即：

$$
\boxed{
d_l=
\frac{0.5}{\tan(\theta_l/2)}.
}
$$

中心 tile、4096→1024 crop 时，应该近似满足：

$$
d_l\approx4d_g.
$$

但是实际代码优先使用 Pixal3D 已有：

```python
distance_from_fov(...)
```

保持和原模型 camera convention 完全一致。

需要打印并验证：

```text
theta_global
theta_local
d_global
d_local
d_local / d_global
```

中心 tile 应接近 4。

不要因为 tile 之后 resize 到 512 再重新修改 `d_l`。

1024 tile resize 到 512 只改变像素焦距，不改变 FOV 和三维 camera distance。

---

# 十、global → local 的统一三维尺度

tile center 在 global camera 中距离：

$$
\rho_c.
$$

在新的 local generation space 中，我们希望：

tile center：

$$
\rightarrow(0,0,0)
$$

而 global camera：

$$
\rightarrow
\text{距 local origin }d_l.
$$

因此唯一自然的 uniform scale：

$$
\boxed{
k=
\frac{d_l}{\rho_c}.
}
$$

global → local 是放大：

$$
k.
$$

local → global 是：

$$
\boxed{
k^{-1}
=
\frac{\rho_c}{d_l}.
}
$$

这是 XYZ 三个方向共同使用的尺度。

禁止：

```text
scale_x != scale_y
scale_z = 1
```

之类的各向异性做法。

---

# 十一、global 三维点变成 local generation point

先在 OpenCV camera coordinate 中计算相对于 anchor 的向量：

$$
\Delta p_g^{cv}
=
p_g^{cv}-P_c^{cv}.
$$

旋转并统一缩放：

$$
\boxed{
t_l^{cv}
=
kR
\left(
p_g^{cv}-P_c^{cv}
\right).
}
$$

其中：

$$
t_l^{cv}
=
(\Delta X_l,\Delta Y_l,\Delta Z_l)
$$

使用的是：

* X：右；
* Y：下；
* Z：朝相机观察方向的 forward。

local generation coordinate 则是：

$$
\boxed{
g_l=
\begin{bmatrix}
t_{l,x}\\
-t_{l,y}\\
-t_{l,z}
\end{bmatrix}.
}
$$

也就是说：

$$
\boxed{
g_{l,x}=t_{l,x}
}
$$

$$
\boxed{
g_{l,y}=-t_{l,y}
}
$$

$$
\boxed{
g_{l,z}=-t_{l,z}.
}
$$

这里的负号来自 Pixal3D generation/world Z 与 OpenCV camera-forward Z 方向相反。

必须统一处理，不允许代码中随意切换符号。

最终：

$$
\boxed{
g_l\in[-0.5,0.5]^3.
}
$$

如果内部 flow 使用：

$$
q_l\in[-1,1]^3
$$

则最后转换：

$$
\boxed{
q_l=2s_lg_l.
}
$$

---

# 十二、这个公式的几何意义

核心公式：

$$
\boxed{
t_l^{cv}
=
\frac{d_l}{\rho_c}
R
(p_g^{cv}-P_c^{cv})
}
$$

等价于：

```text
1. 以 tile center 对应的真实三维 surface point 为原点；
2. 把 tile center ray 旋转成新的 local Z axis；
3. 根据 local FOV 对三维 XYZ 统一放大；
4. 得到新的 isotropic local generation cube。
```

它不是 heuristic。

它是一个严格的：

$$
\boxed{
rotation
+
translation
+
uniform\ scale
}
$$

similarity transform。

---

# 十三、深度公式必须满足

如果一个 global point 到 global camera 的距离为：

$$
\rho_p
$$

它与 tile center ray 的夹角为：

$$
\gamma
$$

那么 rotation 后，其 axial depth 为：

$$
\bar Z_p
=
\rho_p\cos\gamma.
$$

相对于 center point：

$$
\Delta Z
=
\rho_p\cos\gamma-\rho_c.
$$

因此 local camera-frame depth offset：

$$
\boxed{
t_{l,z}
=
d_l
\left(
\frac{\rho_p}{\rho_c}\cos\gamma-1
\right).
}
$$

对应 generation Z：

$$
\boxed{
g_{l,z}
=
-d_l
\left(
\frac{\rho_p}{\rho_c}\cos\gamma-1
\right).
}
$$

必须通过数值实验验证矩阵版本和该显式公式一致。

禁止继续使用：

$$
q_{l,z}=q_{g,z}.
$$

---

# 十四、local cube 在 global 空间中的真实含义

因为：

$$
t_l
=
kR(p_g-P_c)
$$

所以：

$$
g_l\in[-0.5,0.5]^3
$$

对应 global 中一个 oriented cube。

其 global 边长为：

$$
\boxed{
L_g=
\frac{\rho_c}{d_l}.
}
$$

半边长：

$$
\boxed{
h_g=
0.5\frac{\rho_c}{d_l}.
}
$$

中心 tile 通常：

$$
\rho_c\approx d_g
$$

且：

$$
d_l\approx4d_g
$$

因此：

$$
L_g\approx0.25.
$$

也就是：

```text
global 中约边长 0.25 的局部三维区域
        ↓
local 中边长 1.0 的完整 generation cube
```

因此理论上：

$$
C256_{\text{global}}
\rightarrow
C64_{\text{local}}
$$

可以在 XYZ 三个方向上都接近一格对一格。

这正是本次修改的主要目标。

---

# 十五、support 选择

旧代码可能采用：

```text
只要 global point 投影落入 tile
→ 就认为属于 tile
```

现在要改为两级判断。

第一层：

global point 必须满足：

$$
Z_g>0
$$

且 global projection 落入 tile：

$$
x_0\le u_g<x_1
$$

$$
y_0\le v_g<y_1.
$$

第二层：

global point 变换到 local：

$$
g_l=T(g_g)
$$

以后，还需要满足：

$$
\boxed{
|g_{l,x}|\le0.5
}
$$

$$
\boxed{
|g_{l,y}|\le0.5
}
$$

$$
\boxed{
|g_{l,z}|\le0.5.
}
$$

不要 clamp。

越界点直接统计并丢弃。

必须分别统计：

```text
num_projected_into_tile
num_inside_local_cube
num_outside_local_x
num_outside_local_y
num_outside_local_z
```

---

# 十六、local → global 必须有严格解析逆

已知：

$$
g_l.
$$

先恢复 local CV-relative coordinate：

$$
\boxed{
t_l^{cv}
=
\begin{bmatrix}
g_{l,x}\\
-g_{l,y}\\
-g_{l,z}
\end{bmatrix}.
}
$$

然后：

$$
\boxed{
p_g^{cv}
=
P_c^{cv}
+
\frac{\rho_c}{d_l}
R^Tt_l^{cv}.
}
$$

这就是严格 inverse。

然后从 global camera CV coordinate 恢复 generation coordinate：

$$
\boxed{
x_g=X_g
}
$$

$$
\boxed{
y_g=-Y_g
}
$$

$$
\boxed{
z_g=d_g-Z_g.
}
$$

如果最终需要 `q_g∈[-1,1]`：

$$
q_g=2s_gg_g.
$$

---

# 十七、非常重要：raw tile 图像不要 warp

不要执行：

$$
H=K_lRK_g^{-1}
$$

去 warp 高清 tile。

本项目保留：

```text
global 4096 image
→ raw crop [x0:x1, y0:y1]
→ 原始高清 tile
```

不改变原始 GT 图像。

原因是：

local generation space 是人为重新 canonicalize 的三维坐标系；

但图像 condition 表示的仍然是原 global camera 中真实的观测 ray。

因此需要将：

```text
geometry coordinate
```

和：

```text
image feature sampling coordinate
```

拆开。

---

# 十八、local 3D 点查询图像 feature 的正确方法

禁止直接：

```text
local g_l
→ 使用 local centered K_l / d_l
→ 投影到 raw tile
→ sample feature
```

因为纯 rotation 后：

$$
\Pi_l(g_l)
$$

通常不等于原始 crop 中对应 pixel。

正确方法是：

$$
\boxed{
g_l
\rightarrow
T^{-1}
\rightarrow
p_g^{cv}
\rightarrow
\Pi_g
\rightarrow
global4096 pixel
\rightarrow
raw tile pixel.
}
$$

完整定义：

$$
\boxed{
F_{\mathrm{proj}}(g_l)
=
F_{\mathrm{tile}}
\left(
\operatorname{Crop}
\left[
\Pi_g
\left(
T^{-1}(g_l)
\right)
\right]
\right).
}
$$

---

# 十九、local point → raw tile pixel

首先：

$$
t_l^{cv}
=
[g_{l,x},-g_{l,y},-g_{l,z}]^T.
$$

inverse：

$$
p_g^{cv}
=
P_c^{cv}
+
\frac{\rho_c}{d_l}
R^Tt_l^{cv}.
$$

设：

$$
p_g^{cv}=(X_g,Y_g,Z_g).
$$

投影回 global 4096：

$$
\boxed{
u_g=f_xX_g/Z_g+c_x
}
$$

$$
\boxed{
v_g=f_yY_g/Z_g+c_y.
}
$$

raw tile pixel：

$$
\boxed{
u_t=(u_g-x_0)r_x
}
$$

$$
\boxed{
v_t=(v_g-y_0)r_y.
}
$$

当前：

$$
r_x=r_y=1.
$$

所以：

$$
u_t=u_g-x_0
$$

$$
v_t=v_g-y_0.
$$

然后按照现有 DINO preprocessing 的真实 resize / pixel-center convention，把：

$$
(u_t,v_t)
$$

换算为 DINO feature-map sampling coordinate。

必须沿用当前 `grid_sample(..., align_corners=False)` 的 convention。

如果是直接 resize：

$$
W_t\rightarrow W_f
$$

则 pixel-center 映射应使用：

$$
u_f
=
(u_t+0.5)\frac{W_f}{W_t}-0.5
$$

而不是简单：

$$
u_f=u_tW_f/W_t
$$

后直接忽略 pixel center。

优先复用 Pixal3D 当前 feature sampling helper。

---

# 二十、不要 border clamp 图像 feature

如果 inverse 后：

$$
u_t<0
$$

或：

$$
u_t\ge W_t
$$

或：

$$
v_t<0
$$

或：

$$
v_t\ge H_t
$$

则：

$$
\boxed{
proj\_feature\_valid=False.
}
$$

不要使用：

```python
padding_mode="border"
```

把越界点强行吸到 tile 边缘。

越界点应该：

1. 使用 null / zero proj feature；
2. 或只保留 global image token；
3. 同时输出 valid mask。

具体选择保持和当前模型 architecture 最兼容，但绝对不要伪造边缘 correspondence。

---

# 二十一、需要修改 Pixal3D 当前 projection condition

当前 Pixal3D 很可能默认：

```text
local 3D coordinate
→ local camera projection
→ feature map sample
```

现在需要拆开。

新的逻辑：

```text
local sparse/grid coordinate
        ↓
local generation g_l
        ↓
T^{-1}
        ↓
global camera CV point
        ↓
K_global_4096 projection
        ↓
global 4096 pixel
        ↓
subtract tile origin
        ↓
raw tile pixel
        ↓
DINO preprocessing coordinate
        ↓
grid_sample feature
```

local camera：

```text
theta_l
d_l
```

仍然用于：

* local Pixal3D generation camera condition；
* local generation-space definition；
* local decoder；
* local flow。

但它不再负责 raw tile DINO feature correspondence。

---

# 二十二、建议新增函数

请尽量把实现拆成独立函数，例如：

```python
generation_to_camera_cv(
    g,
    distance,
)

camera_cv_to_generation(
    p_cv,
    distance,
)

project_camera_cv_to_image(
    p_cv,
    K,
)

derive_tile_center_ray(
    tile_box,
    K_global,
)

raycast_tile_anchor(
    baseline_mesh,
    center_ray,
    global_camera,
)

build_tile_rotation(
    center_ray,
)

derive_local_fov_from_corner_rays(
    tile_box,
    K_global,
    R,
)

derive_local_distance(
    theta_local,
    mesh_scale,
)

global_generation_to_local_generation(
    g_global,
    tile_transform,
)

local_generation_to_global_generation(
    g_local,
    tile_transform,
)

local_generation_to_raw_tile_pixel(
    g_local,
    tile_transform,
    K_global,
    tile_box,
)

sample_tile_projected_feature(
    feature_map,
    tile_pixel,
    valid_mask,
)
```

并定义一个明确的数据结构：

```python
TileProjectiveTransform:
    tile_box
    K_global
    center_pixel
    center_ray
    anchor_camera_cv
    rho_center
    R_globalcv_to_localcv
    theta_local
    distance_local
    scale_global_to_local
```

不要在不同函数内部重新独立计算这些量。

---

# 二十三、必须保存 debug 信息

每个 tile 保存：

```text
tile_id
tile_box
tile_center_pixel

center_ray
anchor_global_generation
anchor_camera_cv
rho_center

theta_global
theta_local

distance_global
distance_local

scale_global_to_local = d_l / rho_c
scale_local_to_global = rho_c / d_l

R
det(R)
orthogonality_error

num_projected_points
num_inside_local_cube
num_outside_local_cube

feature_valid_ratio
```

同时建议导出少量可视化：

1. global 4096 上画 tile；
2. 画 tile center ray；
3. 标记 baseline anchor；
4. local cube point cloud；
5. local XYZ axis；
6. inverse 回 global 后的 point cloud；
7. raw tile 上画 projected feature lookup pixels。

---

# 二十四、必须做的数值测试

## Test 1：rotation

必须满足：

$$
\|R\hat r_c-e_z\|<10^{-6}
$$

$$
\|RR^T-I\|<10^{-6}
$$

$$
|\det R-1|<10^{-6}.
$$

---

## Test 2：tile anchor

必须满足：

$$
T(P_c)=0.
$$

误差：

```text
< 1e-6
```

---

## Test 3：camera distance

global camera origin 在 local CV-relative frame 中：

$$
t_{\mathrm{camera}}
=
\frac{d_l}{\rho_c}
R(0-P_c).
$$

必须得到：

$$
\boxed{
t_{\mathrm{camera}}
=
(0,0,-d_l).
}
$$

误差：

```text
< 1e-6
```

---

## Test 4：global→local→global round trip

随机采样至少 100000 个点：

$$
g_g
\rightarrow
g_l
\rightarrow
\hat g_g.
$$

要求：

```text
float64 max_abs_error < 1e-9
float32 max_abs_error < 2e-5
```

---

## Test 5：local→global→local round trip

同样：

$$
g_l
\rightarrow
g_g
\rightarrow
\hat g_l.
$$

要求：

```text
max_abs_error < 2e-5
```

---

## Test 6：显式 depth 公式

随机点验证：

矩阵计算的 local CV Z：

$$
t_{l,z}
$$

和：

$$
d_l
\left(
\frac{\rho_p}{\rho_c}\cos\gamma-1
\right)
$$

一致。

误差：

```text
< 1e-6
```

---

## Test 7：raw tile feature correspondence

对于原本来自 global point 的：

$$
P_g
$$

先直接 global project：

$$
P_g
\rightarrow
(u_g,v_g)
\rightarrow
(u_t,v_t).
$$

另一条路径：

$$
P_g
\rightarrow
g_l
\rightarrow
T^{-1}(g_l)
\rightarrow
(u'_g,v'_g)
\rightarrow
(u'_t,v'_t).
$$

必须满足：

$$
\boxed{
u_t=u'_t,\quad
v_t=v'_t.
}
$$

要求：

```text
pixel roundtrip max error < 1e-4 px
```

---

## Test 8：中心 tile 尺度

中心 tile 应验证：

```text
d_local / d_global ≈ 4
```

以及：

```text
global local-ROI side length
≈ rho_center / d_local
≈ 0.25
```

在物体中心深度接近 `d_global` 时应成立。

---

## Test 9：XYZ isotropic

构造 local：

```text
(+eps,0,0)
(0,+eps,0)
(0,0,+eps)
```

inverse 到 global。

三条方向对应的欧氏长度必须都约为：

$$
\epsilon\frac{\rho_c}{d_l}.
$$

即：

$$
\boxed{
L_x=L_y=L_z
}
$$

不能再出现：

```text
X/Y ×4
Z ×1
```

---

# 二十五、禁止事项

不要再执行以下操作：

1. `q_l,z = q_g,z`
2. local XYZ 使用不同缩放倍率
3. bbox min-max normalization
4. point-cloud centroid normalization
5. clamp local point 到 cube 边界
6. 仅仅因为 pixel 落入 tile 就认为 3D point 属于 local cube
7. tile resize 1024→512 时再次修改 FOV/distance
8. 用 local centered camera 直接投影 raw tile feature
9. 为了匹配 local camera 去 warp 原始高清 tile
10. `grid_sample(padding_mode="border")` 伪造越界 feature
11. 把 MoGe metric depth 直接和 Pixal3D canonical camera distance 混用
12. 在不同模块中重复计算不同版本的 tile center / FOV / R / distance

---

# 二十六、最终应形成的数学闭环

整个 pipeline 必须严格实现：

```text
global generation point g_g
        ↓
global camera CV coordinate
        ↓
global K_4096
        ↓
global 4096 pixel
        ↓
tile membership / raw tile crop
        ↓
derive tile center ray
        ↓
baseline mesh ray intersection → P_c, rho_c
        ↓
R * center_ray = local Z
        ↓
tile corner rays → exact local FOV
        ↓
FOV → d_l
        ↓
k = d_l / rho_c
        ↓
t_l = k R (p_g - P_c)
        ↓
local generation coordinate g_l
        ↓
local flow / local sparse lattice / decoder
```

图像 condition 单独走：

```text
local generation point g_l
        ↓
T^{-1}
        ↓
global camera point
        ↓
global K_4096
        ↓
global pixel
        ↓
subtract tile origin
        ↓
raw tile pixel
        ↓
DINO feature sampling
```

最终最重要的两个公式必须作为代码注释保留。

### Global → Local

$$
\boxed{
t_l^{cv}
=
\frac{d_l}{\rho_c}
R
\left(
p_g^{cv}-P_c^{cv}
\right)
}
$$

以及：

$$
\boxed{
g_l=
(t_{l,x},-t_{l,y},-t_{l,z}).
}
$$

### Local → Global

$$
\boxed{
p_g^{cv}
=
P_c^{cv}
+
\frac{\rho_c}{d_l}
R^T
(g_{l,x},-g_{l,y},-g_{l,z})^T.
}
$$

这两个公式必须成为唯一的 global/local 三维变换来源。

不要继续维护旧的 projective-depth-preserving 路径作为主路径。

实现完成后，请输出：

1. 修改文件列表；
2. 每个新增/修改函数及职责；
3. 完整数学公式与代码变量对应关系；
4. 所有数值测试结果；
5. 中心 tile、边缘 tile 各选至少一个给出 debug 数值；
6. round-trip error；
7. pixel correspondence error；
8. local cube occupancy；
9. feature valid ratio；
10. 是否仍存在任何旧 `q_l,z=q_g,z` 或 anisotropic scaling 路径。
