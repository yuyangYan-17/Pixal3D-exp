# 正交纹理 flow 可见性路由实验

日期：2026-09-20  
测试图：`assets/images/0_img.png`  ；flow 使用 CUDA:1，解码/1024 渲染和指标统计使用 CUDA:4。  
输出目录：`outputs/sr_point_visibility_20260919/`

## 规则

- 可见点：每一步使用 full conditional flow。
- 不可见点：前 `n` 步使用 baseline endpoint guide，后 `12-n` 步使用 unconditional flow。
- 混合 block：conditional 和 hidden 分支分别推理；同一点在多个重叠子块中的 velocity 做平均后同步更新。
- `hglobal_only` 是诊断分支：后段不可见点保留全局图像条件，只清零局部几何对应条件。
- `anchor_dilation` 是 decoder 耦合诊断：把可见点附近的 C256 点固定为 n=0 的 full-conditional latent。

## 参考阈值

原始 baseline 1024 前景指标为：

```text
PSNR 13.916996 dB, SSIM 0.203488, LPIPS 0.359764
```

因此用户要求的 `+1 dB` 阈值为 **14.916996 dB**。

## n=1--4（seed=46）

`back` 不是真实背面 GT，只是与 baseline 背面渲染的 agreement，不能当作背面真实质量。

| variant | flow s | render s | front PSNR | SSIM | LPIPS | back agreement PSNR |
|---|---:|---:|---:|---:|---:|---:|
| n=0 full conditional | 700.733 | 6.211 | 14.424193 | 0.353219 | 0.250245 | 14.717752 |
| n=1 | 1145.177 | 3.573 | 14.312702 | 0.336166 | 0.261162 | 17.081302 |
| n=2 | 926.881 | 3.792 | 14.395956 | 0.338059 | 0.258558 | 18.294391 |
| n=3 | 885.314 | 3.678 | 14.443661 | 0.339369 | 0.257328 | 19.151030 |
| n=4 | 841.889 | 3.547 | 14.485152 | 0.340568 | 0.256263 | 19.863805 |

n 增大确实提高了与 baseline 背面的相似度，但前景 SSIM/LPIPS 相对 n=0 变差；只有 PSNR 小幅上升，且距离 +1 dB 仍有 0.432 dB。

## 多 seed 对比（n=0 vs n=4）

| seed | n=0 front PSNR / SSIM / LPIPS | n=4 front PSNR / SSIM / LPIPS | n=4 - n=0 PSNR |
|---:|---|---|---:|
| 46 | 14.424193 / 0.353219 / 0.250245 | 14.485152 / 0.340568 / 0.256263 | +0.060958 dB |
| 47 | 14.607603 / 0.353724 / 0.253532 | 14.599699 / 0.336659 / 0.260380 | -0.007904 dB |
| 48 | 14.886539 / 0.365564 / 0.239105 | 14.904325 / 0.352673 / 0.252836 | +0.017785 dB |

seed=47 的 PSNR 下降，三个 seed 的 n=4 SSIM 都低于对应 n=0，因此没有达到“正面不劣”的稳定要求。seed=48 的 raw n=4 已是 14.904325 dB，仍比 +1 dB 阈值低 0.012671 dB。

补充的 n=0 搜索：seed=49 为 14.592996 dB，seed=50 为 14.812195 dB，均未过阈值。

## 诊断迭代

### global-only

seed=48、n=4：

```text
front: 14.904768 dB / 0.352802 / 0.252310
back agreement: 19.642440 dB / 0.662409 / 0.226756
flow: 848.823 s, render: 3.931 s
```

相对 raw n=4 仅提升 0.000443 dB；因此问题不是简单的“后段完全 unconditional 导致全局条件丢失”。

### visible anchor

seed=46、n=4、固定可见点附近半径 4 的 C256 点：

```text
visible/fixed points: 208,554 / 251,447
front: 14.424349 dB / 0.353229 / 0.250191
back agreement: 15.097800 dB / 0.600951 / 0.287764
```

它相对同 seed n=0 的前景几乎不变（PSNR +0.000156 dB），说明 decoder 的空间耦合会让“只改变不可见 latent”影响前景；但它没有带来可靠的背面收益。

### hidden latent 软混合

为排除硬路由造成的跳变，固定 seed=48 的可见点为 n=0，隐藏点取
`x_n0 + alpha * (x_n4 - x_n0)`：

| alpha | render s | front PSNR | SSIM | LPIPS | back agreement PSNR |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 3.893 | 14.889061 | 0.365152 | 0.238864 | 15.645976 |
| 0.50 | 2.241 | 14.887909 | 0.364159 | 0.238893 | 16.699916 |
| 0.75 | 2.313 | 14.881508 | 0.362626 | 0.239610 | 17.780276 |

软混合没有跨过 +1 dB 阈值；alpha 越大，背面 agreement 越高但前面越差，说明仅在最终 latent 上做连续插值也不能解决 decoder 的空间耦合。

## 结论

本轮设计在单个输入上没有通过验收条件：

1. 多 seed 下 n=4 不能稳定保持前景 full-conditional 的 SSIM/LPIPS/PSNR。
2. 背面没有 GT，当前只能报告与 baseline 的 agreement，不能声称背面真实质量超过 baseline；agreement 变高也可能只是更接近 baseline 的模糊纹理。
3. 最好的正式前景 PSNR 是 `14.904768 dB`，距离 `14.916996 dB` 还差 `0.012228 dB`，所以没有继续对剩余图片做扩展测试。
4. 下一步若要真正满足目标，应改 decoder/texture flow 的空间耦合或训练可见性条件，而不是继续枚举 n；当前实验已经足以排除“只调 n=1--4”以及“对 n=0/n=4 endpoint 做简单软混合”这两个方向。

逐项结果也保存在 [`summary.json`](../outputs/sr_point_visibility_20260919/summary.json) 和 [`blend_summary.json`](../outputs/sr_point_visibility_20260919/blend_summary.json)，实验代码入口为 [`run_point_visibility_texture.py`](run_point_visibility_texture.py)，软混合诊断入口为 [`run_texture_latent_blend.py`](run_texture_latent_blend.py)，核心实现为 [`sr_point_visibility_texture.py`](../sr_point_visibility_texture.py)。
