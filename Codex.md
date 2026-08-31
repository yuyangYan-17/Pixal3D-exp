# Codex 提示词：第二轮 SLAT 稀疏冲激频谱实验——分离 Support Spectrum 与 Feature Spectrum

## 一、实验目标

基于上一轮 `Sparse Fourier Projection` 实验继续研究，但本轮不要继续把主要精力放在 least-squares SFP 上。

上一轮已经证明：

* fixed-support 上的 Fourier projection 数学实现正确；
* dense grid 可以精确复现 FFT；
* sparse samples 上可以恢复已知 band-limited function；
* 但 decoder coarse/fine 语义没有得到支持；
* 原因之一是当前实验始终保留完整 sparse coordinates，只改变 `[N,32]` features；
* 另外当前 `rho=1.0` 实际只对应 `max_radius=5`，并不是 C64 的完整频率范围。

本轮直接回到最初的信号定义：

$$
\boxed{
\mu_c(x)
=
\sum_{i=1}^{N}
f_{ic}\delta(x-p_i),
\qquad c=1,\dots,32
}
$$

也就是：

> 每一个 active SLAT point 是一个三维冲激，坐标 \(p_i\) 是冲激位置，第 \(c\) 个 feature 是该冲激在 channel \(c\) 上的幅值。

本轮要回答三个问题：

1. **整个 sparse impulse SLAT 的频谱是什么？**
2. **物体的低频主要来自 active-point support，还是来自 32-D feature variation？**
3. **把真正的完整 C64 Fourier band 从低到高逐步加入后，decoder 是否存在 coarse-to-fine 规律？**

本轮禁止：

* graph Laplacian；
* octree；
* RAHT；
* PCA；
* decoder Jacobian；
* channel-frequency 假设。

---

# 二、首先修正 frequency 定义

当前真实 SLAT 是 C64 lattice。

假设坐标：

$$
x,y,z\in\{0,\dots,63\}
$$

归一化为长度为 1 的空间：

$$
[-0.5,0.5)^3.
$$

那么 lattice spacing：

$$
h=\frac1{64}.
$$

每个坐标轴的 Nyquist frequency：

$$
\boxed{
k_N=32
}
$$

cycles / object-space unit。

完整三维 DFT frequency 应来自：

```python
torch.fft.fftfreq(64, d=1/64)
```

即大约：

```text
0,1,...,31,-32,-31,...,-1
```

不要再用：

```text
max_radius = 5
rho * max_radius
```

作为完整频率定义。

---

# 三、使用真实完整 C64 DFT frequency cube

构造：

$$
k=(k_x,k_y,k_z)
$$

其中：

$$
k_x,k_y,k_z
\in
\{-32,\dots,31\}.
$$

对于每个 frequency：

$$
r(k)
=
\sqrt{k_x^2+k_y^2+k_z^2}.
$$

记录：

```text
axis Nyquist = 32
maximum cube radius ≈ sqrt(3)*32 ≈ 55.43
```

不要把：

$$
r=32
$$

错误叫成完整三维最大频率。

为了方便解释，本轮直接使用实际 cutoff：

```text
k_c =
1
2
4
8
12
16
20
24
28
32
40
48
full
```

其中：

```text
full
```

表示保留完整 DFT cube，不再使用 radial mask。

输出文件名称直接使用：

```text
kc_01
kc_02
kc_04
...
kc_32
kc_40
kc_48
full
```

不要再用容易误导的：

```text
rho_0.80
rho_1.00
```

---

# 四、第一条分解：完整 Sparse Impulse Signal

对于每个 channel：

$$
c=1,\dots,32,
$$

创建 dense C64 signal：

$$
X_c[x,y,z]
=
\begin{cases}
f_{ic},
&(x,y,z)=p_i,\\
0,
&\text{otherwise}.
\end{cases}
$$

注意：

这里 inactive voxel = 0 **不是插值假设**。

因为当前理论定义本身就是：

$$
\sum_i f_i\delta_{p_i}.
$$

没有冲激的位置，其离散 impulse amplitude 就是 0。

因此：

$$
\boxed{
\text{zero-filled C64 FFT}
}
$$

在这个定义下不是 baseline approximation，而是这个离散 sparse impulse signal 的**精确 DFT**。

这一点必须在代码和 report 中明确说明。

---

# 五、FFT 得到完整 SLAT spectrum

对：

$$
X\in\mathbb R^{64\times64\times64\times32}
$$

执行：

$$
\hat X
=
FFT3D(X).
$$

FFT 只作用 XYZ：

```python
torch.fft.fftn(X, dim=(0,1,2))
```

得到：

$$
\hat X(k)
\in\mathbb C^{32}.
$$

定义总 spectral power：

$$
\boxed{
S_{\rm total}(k)
=
\|\hat X(k)\|_2^2
=
\sum_{c=1}^{32}
|\hat X_c(k)|^2
}
$$

---

# 六、必须把 Support 与 Feature Modulation 分开

这是本轮最重要的实验。

定义 occupancy/support signal：

$$
M[x,y,z]
=
\begin{cases}
1,&\text{active voxel}\\
0,&\text{otherwise}
\end{cases}
$$

计算 32-D feature 平均：

$$
\bar f
=
\frac1N
\sum_i f_i
\in\mathbb R^{32}.
$$

然后把完整 sparse SLAT 拆为：

$$
\boxed{
X
=
X_{\rm carrier}
+
X_{\rm modulation}
}
$$

其中：

### Support carrier

$$
\boxed{
X_{\rm carrier}(x)
=
M(x)\bar f
}
$$

含义：

> active points 在哪里，但所有 active points 都使用完全相同的 32-D feature。

它只保留：

$$
\boxed{\text{support geometry + constant feature carrier}}
$$

---

### Feature modulation

$$
\boxed{
X_{\rm modulation}(p_i)
=
f_i-\bar f
}
$$

其它位置为 0。

也就是：

$$
X_{\rm modulation}
=
X-X_{\rm carrier}.
$$

含义：

> 在已经存在这些 active points 的基础上，各 point 的 32-D feature 相对于平均 feature 如何变化。

---

# 七、频域中必须验证精确关系

计算：

$$
\hat X
$$

$$
\hat X_C
=
FFT(X_{\rm carrier})
$$

$$
\hat X_M
=
FFT(X_{\rm modulation}).
$$

验证：

$$
\boxed{
\hat X
=
\hat X_C+\hat X_M
}
$$

relative error 应接近浮点误差。

然后分别定义：

$$
S_C(k)
=
\|\hat X_C(k)\|^2
$$

$$
S_M(k)
=
\|\hat X_M(k)\|^2.
$$

另外不能忽略 cross term。

因为：

$$
\|A+B\|^2
=
\|A\|^2+\|B\|^2
+
2\operatorname{Re}\langle A,B\rangle.
$$

因此计算：

$$
\boxed{
S_{\rm cross}(k)
=
2\operatorname{Re}
\left[
\hat X_C(k)^H
\hat X_M(k)
\right]
}
$$

验证：

$$
\boxed{
S_{\rm total}
=
S_C+S_M+S_{\rm cross}
}
$$

达到浮点误差。

---

# 八、画真正的 radial spectrum

按：

$$
r=\|k\|
$$

将 frequencies 分到 radial shells。

至少输出两种统计。

### Shell total energy

$$
E_{\rm sum}(r)
=
\sum_{k\in shell(r)}
S(k)
$$

### Shell mean energy

$$
\boxed{
E_{\rm mean}(r)
=
\frac{
E_{\rm sum}(r)
}{
\#shell(r)
}
}
$$

因为三维高频 shell 中 frequency mode 数量更多，所以不能只看 sum。

同一张图画：

```text
Total SLAT
Support carrier
Feature modulation
Cross term
```

输出：

```text
support_feature_spectrum_sum.png
support_feature_spectrum_mean.png
support_feature_spectrum_log.png
support_feature_spectrum.csv
```

CSV 至少包含：

```text
radius
num_modes
total_sum
carrier_sum
modulation_sum
cross_sum
total_mean
carrier_mean
modulation_mean
cross_mean
```

---

# 九、必须计算累计 spectral energy

定义：

$$
C(r)
=
\frac{
\sum_{\|k\|\le r}S(k)
}{
\sum_kS(k)
}.
$$

分别对：

```text
total
carrier
modulation
```

计算。

输出：

```text
cumulative_spectral_energy.png
```

并报告达到：

```text
50% energy
75% energy
90% energy
95% energy
```

分别需要多大的 frequency radius。

例如最终报告：

```text
carrier:
50% energy radius = ...
90% energy radius = ...

modulation:
50% energy radius = ...
90% energy radius = ...
```

这是判断：

> support 是否比 feature variation 更偏低频

最直接的指标。

不要提前假定结果。

---

# 十、做一个明确的 decoder control：Constant Feature

上一轮 `rho=0.05/0.10` 实际非常接近 DC，但这一轮不要再依赖间接结果。

明确创建：

$$
\boxed{
F_{\rm constant}
=
\mathbf 1_N\bar f^T
}
$$

coordinates 完全使用原始：

$$
P.
$$

decode：

$$
D(P,F_{\rm constant}).
$$

命名：

```text
decoder_controls/constant_feature
```

这个实验回答：

> 只知道完整 sparse support，再给所有点相同的 32-D feature，decoder 可以恢复多少物体？

---

# 十一、Feature Modulation Only

构造：

$$
\boxed{
F_{\rm modulation}
=
F-\bar f
}
$$

coordinates 仍然完整。

decode：

$$
D(P,F_{\rm modulation}).
$$

输出：

```text
decoder_controls/modulation_only
```

注意：

这可能是 decoder OOD 输入，因此只作为 diagnostic。

不要因为效果差就得出强结论。

---

# 十二、Feature Shuffle Control

这是非常重要的 control。

随机 permutation：

$$
\pi(i).
$$

构造：

$$
\boxed{
F_i^{shuffle}
=
\bar f
+
(F_{\pi(i)}-\bar f)
}
$$

也就是：

* sparse support 完全相同；
* feature 的整体分布完全相同；
* 每个 channel 的统计量基本相同；
* 但 feature 和 spatial location 的对应关系被破坏。

至少测试：

```text
5 random permutations
```

decode：

$$
D(P,F^{shuffle}).
$$

输出：

```text
decoder_controls/shuffle_seed_*
```

如果 shuffled feature 仍然恢复完整 global object：

说明 global object 很大程度来自：

$$
P
$$

而不是 feature 的空间组织。

如果结构明显崩坏：

说明：

$$
F(p_i)
$$

的空间排列也承载重要几何信息。

不要提前假定结果。

---

# 十三、真正的 Full-Impulse Low/High Filtering

对于每个 cutoff：

$$
k_c
$$

定义 radial low-pass：

$$
H_L(k)
$$

第一版做两种。

### A. Ideal cutoff

$$
H_L(k)
=
\mathbf1[\|k\|\le k_c]
$$

### B. Butterworth

参考图像方法使用平滑 cutoff：

$$
\boxed{
H_L(k)
=
\frac{
1
}{
1+
(\|k\|/k_c)^{2n}
}
}
$$

默认：

$$
n=4.
$$

对应：

$$
H_H=1-H_L.
$$

然后：

$$
X_L
=
IFFT(
H_L\hat X
)
$$

$$
X_H
=
IFFT(
H_H\hat X
).
$$

必须验证完整 dense field：

$$
\boxed{
X_L+X_H=X
}
$$

达到浮点误差。

---

# 十四、区别两个概念：数学 filtered field 与 decoder-compatible SLAT

FFT low-pass 后：

$$
X_L
$$

一般会在 inactive voxel 上产生非零值。

这完全正常。

因为 low-pass 后冲激会变成空间中展开的平滑波。

因此必须保存两个版本。

### Version A：完整 filtered dense field

```text
dense_filtered/
```

保存：

$$
X_L,X_H
$$

用于数学分析。

### Version B：重新采样到原 active coordinates

取：

$$
\boxed{
F_L^{active}
=
X_L[P]
}
$$

$$
\boxed{
F_H^{active}
=
X_H[P]
}
$$

有：

$$
F_L^{active}+F_H^{active}=F.
$$

这是唯一能够在：

```text
coords 不变
```

条件下重新送回 Pixal3D decoder 的版本。

报告中必须明确写：

> `active-sampled low/high` 是完整 filtered continuous/dense field 在原 SLAT support 上重新采样后的 decoder-compatible approximation，不等价于完整 filtered field。

---

# 十五、使用真正的 cutoff 做 low/high decoder

至少测试：

```text
kc = 1
kc = 2
kc = 4
kc = 8
kc = 16
kc = 24
kc = 32
kc = 40
kc = 48
```

以及：

```text
full
```

但 `full` 不需要 decode low/high，因为：

$$
low=F,\quad high=0.
$$

对每个 cutoff 分别 decode：

```text
low_only
high_only
```

所有实验：

* coordinates 完全相同；
* decoder 完全相同；
* normalization 完全相同；
* camera/render 完全相同。

输出 contact sheets。

文件名不要叫 rho：

```text
low_only/kc_01
low_only/kc_02
...
high_only/kc_48
```

---

# 十六、Band decomposition

定义真正的 bands：

$$
B_1:0<r\le2
$$

$$
B_2:2<r\le4
$$

$$
B_3:4<r\le8
$$

$$
B_4:8<r\le16
$$

$$
B_5:16<r\le24
$$

$$
B_6:24<r\le32
$$

$$
B_7:32<r\le40
$$

$$
B_8:40<r\le48
$$

$$
B_9:r>48.
$$

在 frequency domain：

$$
\hat X_{B_j}
=
M_{B_j}\hat X.
$$

然后：

$$
X_{B_j}
=
IFFT(\hat X_{B_j}).
$$

取：

$$
F_{B_j}=X_{B_j}[P].
$$

分别 decode。

不要再出现上一轮：

```text
B5_high_gt_0.80
```

这种把绝大多数频率都叫“最高频”的命名。

---

# 十七、加一个“条件 feature frequency”对照

完整 impulse spectrum 会把：

```text
support density
+
feature variation
```

混在一起。

因此额外实现一个简单 conditional feature smoothing，只作为对照。

使用 Gaussian frequency low-pass：

$$
H_\sigma(k)
=
e^{-\sigma^2\|k\|^2/2}.
$$

分别 low-pass：

$$
M(x)F(x)
$$

和：

$$
M(x).
$$

得到：

$$
N_\sigma
=
LPF(MF)
$$

$$
D_\sigma
=
LPF(M).
$$

然后在 active points 定义：

$$
\boxed{
F_{\rm cond-low}(p_i)
=
\frac{
N_\sigma(p_i)
}{
D_\sigma(p_i)+\epsilon
}
}
$$

这相当于：

> 去除 sampling/support density 的影响，只研究 feature 在当前 support 上的平滑变化。

然后：

$$
F_{\rm cond-high}
=
F-F_{\rm cond-low}.
$$

注意：

这个方法不是正交 Fourier decomposition。

报告中只能称：

```text
support-normalized / conditional feature low-pass
```

不能和 full impulse FFT 混为一谈。

---

# 十八、Rotation 测试不要再使用有限 Cartesian basis reconstruction

上一轮：

$$
rotation\ error=0.568
$$

很大程度来自 finite Cartesian frequency basis。

本轮 rotation invariance 只测试：

$$
\boxed{\text{radial spectrum}}
$$

不要测试 least-squares reconstruction。

做法：

对实际 coordinates：

$$
p_i
$$

直接计算 NUFT：

$$
\hat\mu(ru)
=
\sum_i
f_i
e^{-j2\pi r u^Tp_i}.
$$

其中：

$$
u\in S^2.
$$

每个 radius 使用 Fibonacci sphere 均匀采样方向。

至少测试：

```text
num_directions =
64
128
256
512
```

radii：

```text
1,2,4,8,12,16,24,32
```

对每个 radius：

$$
\boxed{
P(r)
=
\frac1M
\sum_{m=1}^M
\|
\hat\mu(ru_m)
\|^2
}
$$

然后对 coordinates 做若干随机 3D rotations：

$$
p_i'=Rp_i
$$

feature 不变。

比较 rotation 前后的：

$$
P(r).
$$

输出：

```text
rotation_invariant_radial_spectrum.png
rotation_error_vs_num_directions.png
```

如果方向采样增加后 rotation error 明显下降，说明上一轮 0.568 是 finite Cartesian discretization 问题，而不是 Fourier theory 本身的问题。

---

# 十九、不要再使用 SFP condition number 判断完整频谱是否存在

上一轮：

$$
\kappa(A)\approx5.09\times10^5
$$

说明：

> 在 fixed support 上用越来越多低频函数做 least-squares projection 会病态。

但本轮完整 impulse FFT：

$$
X\rightarrow FFT3D(X)
$$

不存在这个 design matrix。

因此本轮必须明确区分：

### Previous SFP

问题：

```text
给定 sparse sample values，
拟合一个 band-limited continuous function
```

### Current impulse FFT

问题：

```text
把 sparse points 本身定义成离散 impulse signal，
直接分析该 signal 的 spectrum
```

两者数学问题不同。

本轮不要因为上一轮 SFP condition number 大，就限制 impulse FFT 的最大频率。

---

# 二十、decoder 结果至少比较以下六组

最终必须产生一张总 contact sheet：

### 1. Original

$$
D(P,F)
$$

### 2. Constant feature / carrier-only

$$
D(P,\bar f)
$$

### 3. Feature modulation only

$$
D(P,F-\bar f)
$$

### 4. Shuffled features

至少 5 seeds。

### 5. Full-impulse low-only

多个真实 cutoff。

### 6. Full-impulse high-only

多个真实 cutoff。

并在每张图下面标清：

```text
actual cutoff cycles/unit
axis Nyquist = 32
```

---

# 二十一、重点观察而不是只算 mesh 数量

对 fixed view render 同时计算：

### Low-frequency image difference

对 render 做 2D low-pass 后比较 baseline。

### High-frequency image difference

对 render 做 high-pass 后比较。

另外记录：

* silhouette IoU；
* bbox；
* connected components；
* vertex count；
* face count；
* Chamfer（如果 mesh correspondence 可计算）；
* normal difference；
* LPIPS / image residual；
* render FFT radial spectrum。

目的不是用一个 metric 决定结论，而是辅助观察：

```text
global object
local geometry
texture/detail
```

分别什么时候出现。

---

# 二十二、特别关注四种可能结果

## 情况 A

`constant feature` 已经几乎恢复完整 global shape。

同时 support carrier spectrum 高度集中在低频。

那么支持：

$$
\boxed{
\text{SLAT global structure 很大部分编码在 active support 中。}
}
$$

---

## 情况 B

shuffle feature 后 global shape 仍然基本不变，但细节明显下降。

那么支持：

$$
\boxed{
P
\text{主要决定 global structure，}
\quad
F(p)
\text{主要 refinement。}
}
$$

---

## 情况 C

shuffle 后 geometry 严重崩坏。

说明：

$$
\boxed{
\text{feature 与坐标之间的空间对应本身包含重要 geometry information。}
}
$$

---

## 情况 D

真正使用完整 Nyquist 范围以后，high-only 在：

```text
kc = 24 / 28 / 32 / 40 / 48
```

仍然稳定保留 global object。

这时才真正有依据说：

> SLAT amplitude 的 Euclidean high-frequency component 也携带 global semantic information，简单 Fourier coarse/fine 对应关系不成立。

上一轮 `high > 4` 不能支持这个结论。

---

# 二十三、最重要的输出图

必须生成：

```text
01_support_feature_spectrum_mean.png
02_support_feature_spectrum_sum.png
03_cumulative_spectral_energy.png
04_rotation_invariant_radial_spectrum.png
05_constant_modulation_shuffle_contact_sheet.png
06_full_impulse_low_only_contact_sheet.png
07_full_impulse_high_only_contact_sheet.png
08_frequency_bands_contact_sheet.png
09_conditional_feature_low_high_contact_sheet.png
```

---

# 二十四、最终 report.md 必须回答

1. C64 真正的 frequency range 是多少？

2. support carrier 的 radial spectrum 是什么形状？

3. feature modulation 的 spectrum 是什么形状？

4. carrier 和 modulation 各自在什么 radius 达到 50%、90% cumulative energy？

5. cross spectral term 是否显著？

6. constant feature + original support 能恢复多少 global structure？

7. modulation-only 能恢复什么？

8. feature shuffle 后 global geometry 是否保持？

9. 使用真正的 \(k_c=1\rightarrow48\) 后，low-only 是否出现更清晰的 coarse-to-fine progression？

10. 真正接近 Nyquist 的 high-only 是否仍保留完整 global object？

11. 哪些 individual frequency bands 保留 global structure？

12. full impulse frequency 与 conditional feature frequency 的 decoder 语义是否不同？

13. spherical NUFT radial spectrum 的 rotation error 是否随着 direction 数增加而下降？

14. 当前证据更支持哪一种模型：

### Model A

$$
\text{low frequency}
\leftrightarrow
\text{coarse},
\qquad
\text{high frequency}
\leftrightarrow
\text{detail}
$$

### Model B

$$
\text{support}
\leftrightarrow
\text{global geometry},
\qquad
\text{feature frequency}
\leftrightarrow
\text{refinement}
$$

### Model C

support 与 feature 强耦合，无法简单分开。

### Model D

Euclidean spatial frequency 和 decoder semantic scale 没有稳定对应。

不要提前选择结论。

---

# 二十五、本轮最重要的理论区分

最终报告必须明确区分以下三个对象：

$$
\boxed{
\textbf{1. Full sparse impulse frequency}
}
$$

$$
\mu(x)=\sum_i f_i\delta(x-p_i)
$$

它同时包含 support 和 amplitude。

---

$$
\boxed{
\textbf{2. Support/carrier frequency}
}
$$

$$
\nu(x)=\sum_i\delta(x-p_i)
$$

描述 active points 的空间结构。

---

$$
\boxed{
\textbf{3. Conditional feature frequency}
}
$$

描述：

> 已经知道 support \(P\) 的条件下，\(f_i\) 在这些点上变化得快还是慢。

上一轮 SFP 更接近第 3 类。

本轮重点研究第 1 和第 2 类，并把三者放在同一个实验中比较。

最终不要再简单把：

```text
high-only 还能看到物体
```

作为否定 Fourier 理论的依据。

必须先明确：

```text
high 的真实 cutoff 到底是多少
support 是否仍完整存在
分析的是 full impulse 还是 conditional feature
```

再给结论。
