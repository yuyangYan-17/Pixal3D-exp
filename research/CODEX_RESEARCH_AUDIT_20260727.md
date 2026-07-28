# Pixal3D global/local 高分辨率研究审计

日期：2026-07-27  
仓库：`/home/nvme04/yyyan/Pixal3D`  
审计时 HEAD：`6e14ed0490cc6a636292a41fbe31cfb00725766d`

## 1. 当前结论

1. 相机的 global↔tile 逐点透视变换已经通过数值检查。100,000 个随机
   local 点在 tiles 24/26/27 上的 local→global→local 最大 `q` 误差为
   `1.1921e-6`，像素逆变换最大误差不超过 `3.57e-4 px`；当前生成结果的
   support round-trip 最大误差为 `8.63e-5 px`。
2. 当前最可靠的 local 质量来源不是 global/local latent 或 velocity
   平均，而是：global 只提供稀疏 topology anchor，tile 图像重新执行
   shape/texture flow。
3. 对固定 tiles 24/26/27，纯 projected-C64 消融三项均值为
   `14.1466 / .4370 / .4784`；带 constrained native support 的 v7 为
   `14.3413 / .4333 / .4737`。v7 的 PSNR、LPIPS 均值更好，但 SSIM 略差；
   tile 24 的 PSNR/SSIM 也低于纯 projected-C64。因此不存在对三块、
   三项指标都占优的单一 Baseline D。
4. 旧的统一 global velocity replacement/average 已被成对实验明确证伪：
   projective tile super-resolution 从
   `17.0683 / .6622 / .2116` 退化到
   `14.3229 / .5871 / .3257`，并出现全局彩色马赛克。
5. decoded C1024 support 的严格 correspondence 已完成。三块 local
   support 与 global tile-region support 的 Jaccard 仅为
   `6.33% / 13.99% / 14.49%`，exact-match 材质伪 PSNR 仅为
   `13.07 / 11.84 / 11.72 dB`。这支持下一步只做 fixed-support、
   exact-key、小残差材质实验；继续 average latent/velocity、union 全部
   local topology 或 densify `1024^3` 仍没有证据支持。
6. fixed-global-support 材质 residual 已按上述诊断完成。`alpha=.25`
   在 tiles 24/26/27 上逐 tile 的 PSNR、SSIM、LPIPS 全部改善，三块均值
   增益为 `+.01865 dB / +.001598 / LPIPS -.002382`；全局图也小幅改善。
   `alpha=1` 则三块 SSIM 全部下降并出现局部颜色块，因此该路线目前只证明
   保守材质迁移有效，不代表已经达到 independent local Baseline D 的质量。

指标顺序均为 `PSNR / SSIM / LPIPS`，LPIPS 越低越好。

## 2. 仓库状态与审计边界

开始审计时 worktree 已包含用户改动及未跟踪研究脚本。未重置、覆盖或删除
这些内容。特别是：

- `CODEX_TASK.md`、renderer 相关文件已有修改；
- 多个 `pixal3d_projective_*.py` 脚本未被 git 跟踪；
- 一些历史 summary 内的绝对路径在结果移动到 `outputs/` 后已经失效；
- 历史目录采用不同 renderer、metric resolution、环境光或相机实现时，
  不做横向排名。

基础验证：

- Python 环境：`/home/nvme04/yyyan/miniconda3/envs/pixal3d`
- GPU：NVIDIA A800-SXM4-80GB，实验固定 `CUDA_VISIBLE_DEVICES=4`
- 单元测试：`26 passed`
- 生成过程中观察到的 decoder 峰值显存约 `60.8–61.1 GB`

## 3. 实际代码路径图

### 3.1 官方 1024 global 路径

```text
RGBA / RGB input
  │
  ├─ canonical preprocessing
  │    alpha reuse or one rembg call
  │    foreground bbox + 1.1 square padding
  │    └─ image_4096 ─LANCZOS→ image_1024 / image_512
  │
  ├─ MoGe camera helper
  │    仅使用 focal/FOV，不使用 MoGe depth/point cloud
  │    └─ global FOV + distance_from_fov
  │
  ├─ image_512 + DINO condition
  │    └─ sparse-structure flow → C32 coordinates
  │
  ├─ image_512 + NAF512 + C32
  │    └─ shape512 flow → shape SLat(C32)
  │       └─ learned occupancy upsample → C64 support
  │
  ├─ image_1024 + NAF512 + C64
  │    └─ shape1024 flow → shape SLat(C64, 32 channels)
  │
  ├─ image_1024 + NAF1024 + same C64 support
  │    └─ texture1024 flow(shape-conditioned)
  │       → texture SLat(C64, 32 channels)
  │
  ├─ shape decoder
  │    └─ mesh + guide_subs:
  │       C64→C128→C256→C512→C1024 parent/child hierarchy
  │
  ├─ texture decoder(texture SLat, guide_subs)
  │    └─ C1024 sparse O-Voxel material attributes
  │
  └─ MeshWithVoxel
       vertices/faces + C1024 coords/attrs
       └─ native PbrMeshRenderer
          interpolated surface position
          + sparse trilinear O-Voxel query
```

`shape SLat` 与 `texture SLat` 在官方路径中使用完全相同的 C64 support。
`guide_subs` 不是各层 leaf coordinate list，而是“当前 parent coordinate +
8 个 child 的 active decision”。行号相同不代表不同 SparseTensor 间有语义
对应；只有 support 和排序都经过验证时，才能逐行操作。

### 3.2 当前 v7 global-guided local 路径

```text
global decoded C1024 support
  └─ global q → full 4096 pixel → tile pixel → centered local q
     ├─ outside local cube: drop, never clamp
     ├─ independently quantize projected local C32
     └─ independently quantize projected local C64

tile image
  └─ tile-native sparse structure → native C32
     ├─ alpha-foreground gate
     └─ projected-C32 radius-1 neighborhood gate
        └─ constrained native C32
           └─ tile shape512 → learned native C64

projected global C64 ∪ native C64 (deduplicate)
  └─ tile image + tile camera shape1024 flow
     └─ tile image + regenerated tile shape SLat texture1024 flow
        └─ ordinary shape/texture decoder + native local render
```

Global 在该路径中只介入 sparse support/topology。它不直接混入 global
condition、latent、noise、velocity 或 decoded material。

### 3.3 当前 modified-material 路径

当前 `pixal3d_projective_tile_generation_eval_projected_c64_only_copy.py`
中的修改路线实际执行：

```text
global C1024 leaves
  → 投影并降采样成 tile C64
  → tile image 重新生成 tile shape SLat
  → tile image + tile shape SLat 生成 tile texture SLat
  → projected global guide_subs 强制 texture decode
  → decoded attrs 按 leaf correspondence 回填 global support
```

该路线比早期“点状上色”版本连续，但 tile shape feature 与强制 global
subdivision hierarchy 存在分布错配。`global_5` 中 PSNR/SSIM 局部提高，
LPIPS 明显恶化且视觉变糊，因此不能视为已解决的 fixed-geometry material
融合。

## 4. 三种坐标语义

必须区分：

| 对象 | 到归一化 `q` 的公式 | 用途 |
|---|---|---|
| C32/C64 flow lattice | `q = 2*i/(R-1)-1` | sparse flow/projected support endpoint |
| decoded C1024 O-Voxel | `q = 2*(i+0.5)/R-1` | material voxel center |
| decoder surface vertex | `q = 2*vertex` | 连续 surface point |

把 decoded C1024 material coordinate 当 endpoint 会产生半 voxel 偏差。该偏差
小于旧相机的几十像素错误，但在 strict support correspondence 和材质绑定中
不可忽略。

禁止：

- bbox 或 centroid normalization；
- 把越界点 clamp 到立方体边界；
- 把 local tensor row index 当成 global correspondence；
- 把不存在的 sparse node 当作数值为零的 dense voxel。

## 5. 固定阶段 1 协议

| 项 | 值 |
|---|---|
| 输入 | `assets/choose/0_img.png` |
| base seed | 42 |
| tiles | 24（主体中心）、26（复杂结构）、27（右侧边缘） |
| tile size / stride | 1024 / 512 |
| global canonical size | 4096 |
| global model input | 1024 |
| global baseline render | 4096，再按 canonical tile box 精确 crop |
| tile render / metric | 1024 / 1024 |
| renderer | native `PbrMeshRenderer` |
| HDRI / background | studio / black |
| SSAA / peel layers | 2 / 8 |
| metrics | full-frame PSNR, SSIM, LPIPS-VGG |
| exposure/color alignment | 无 |

高分辨率 global render 的精确 crop 与对应 off-axis crop camera 的 ray
等价。当前采用 crop 是为了避免把 256-pixel global crop 上采样为 1024。

共享的是 base seed 42，不是 C/D 完全相同的 latent noise。脚本按 tile 和
route 派生 seed；例如 tile 24 的 projected route 和 direct route 分别使用
24043 与 124043。结果可确定性复现，但严格 causal noise-matched 对照尚未
完成。

## 6. Baseline A–D

### 6.1 A/B/C 与 topology-locked D0

产物：

- `outputs/codex_baseline_abcd_0img_seed42_tiles24_26_27_20260727`
- `effective_run_config.json`
- `summary.json`
- `aggregate_metrics.csv`
- `all_tiles_all_comparison_routes_00.png`

Baseline A（完整 global）：

| PSNR | SSIM | LPIPS | C32 | C64 | C1024 |
|---:|---:|---:|---:|---:|---:|
| 14.338614 | .622458 | .234705 | 3,187 | 13,504 | 3,956,833 |

局部结果：

| Tile | B: global crop | C: local-only | D0: projected-global C64 |
|---:|---:|---:|---:|
| 24 | 9.9711 / .2912 / .6259 | 7.4132 / .1425 / .6266 | 14.4343 / .4068 / .5647 |
| 26 | 11.0785 / .2961 / .5854 | 12.1036 / .2924 / .5255 | 12.5956 / .3166 / .5390 |
| 27 | 12.9284 / .5274 / .3972 | 13.5159 / .5499 / .3587 | 15.4098 / .5877 / .3314 |
| Mean | 11.3260 / .3716 / .5362 | 11.0109 / .3283 / .5036 | 14.1466 / .4370 / .4784 |

D0 相对 B 为 `+2.82055 dB / +.065475 / -.057791`，三个 tile 的三项指标
均优于 B。D0 跳过 tile SS、shape512、native C64，只用 global C1024
投影得到的 local C64 运行 tile shape1024/texture1024，是干净的
topology-locked 消融。

### 6.2 复现当前 v7 Baseline D

产物：

- `outputs/codex_baseline_d_best_v7_0img_seed42_tiles24_26_27_20260727`
- `effective_run_config.json`
- `run.log`
- `summary.json`
- `aggregate_metrics.csv`
- `all_tiles_baseline_vs_tile_00.png`

| Tile | B: global crop | D-v7 | projected/native/fused C64 |
|---:|---:|---:|---:|
| 24 | 9.9711 / .2912 / .6259 | 14.0540 / .3841 / .5599 | 12,040 / 995 / 12,823 |
| 26 | 11.0785 / .2961 / .5854 | 13.5531 / .3346 / .5246 | 16,397 / 6,509 / 21,209 |
| 27 | 12.9284 / .5274 / .3972 | 15.4169 / .5813 / .3365 | 11,406 / 3,033 / 13,375 |
| Mean | 11.3260 / .3716 / .5362 | 14.3413 / .4333 / .4737 | — |

D-v7 相对 B 为 `+3.01530 dB / +.061770 / -.062467`。它相对 D0：

- mean PSNR `+0.19475 dB`；
- mean SSIM `-0.00371`；
- mean LPIPS `-0.00468`；
- tile 24 的 PSNR/SSIM 下降，表明 native topology 不是普遍有益。

因此后续实验应同时保留 D0 和 D-v7；不能只报告对复杂/边缘 tile 有利的
结果。

## 7. Baseline E 与强控制组

### 暂定 E：target-context-hard step10

`outputs/pixal3d_2048_target_context_hard_step10`

- 49 个 2D image tiles；
- active tile 独立 DINO/NAF condition；
- hard center-owner routing；
- 在单一 global latent 上更新；
- `expert_flow_calls=938`；
- 指标 `17.446677 / .670896 / .216202`；
- 同协议 canonical global control：
  `17.386317 / .670299 / .217115`。

提升只有 `+.06036 dB / +.000598 / -.000913`，目前是一张图、一个 seed，
且缺少 local-camera、depth/normal 和新视角验证，只能标记为暂定 E。

### 强 whole-model 控制：uniform CFG 3.0

`BEST_CONFIG.md` 的 EXP-010/011 使用 shape conditional-only、texture
uniform CFG strength 3、step 6。五 seed 均值：

`17.380263±.162508 / .677013±.003448 / .213781±.001325`

它比 archived original-local-CFG 稳定，但属于 fixed global 3D patch CFG，
不是 high-resolution image-tile global/local 融合，不能替代 D 或 E。

## 8. 历史实验复盘

| 路线 | 证据 | 结论 | 可复用部分 |
|---|---|---|---|
| v7 projected + constrained native support | `tile_26_27_test_2` 与本次复现 | 当前最可信 global-guided local；仍是独立 local mesh | 正确相机、foreground/neighborhood gate、C64 union |
| projected-C64 only | `tile_26_27_projected_c64_only` 与本次 D0 | 稳定干净 topology anchor；不能新增结构 | topology-locked control |
| modified global material `_4` | `tile_26_27_projected_c64_only_global_4` | 点状颜色、材质与连续表面 lookup 不对应 | 失败诊断 |
| modified global material `_5` | `..._global_5` | 连续性改善但模糊、黑点；LPIPS 恶化 | exact leaf correspondence、normal fallback |
| global C128 tile velocity residual | `...projected_c64_only.py` v2 | `14.2011/.6290/.2418`，明显低于 ordinary global | sparse transport、step trace |
| projective tile super-resolution | `projective_tile_superres_2048` | 17.0683→14.3229 PSNR；全局马赛克 | CSR provenance、paired control |
| joint tile cascade | `joint_tile_*` | 多对一 transport、旧 clamp 相机；未稳定胜出 | late-step schedule、master support |
| fixed global patch uniform CFG | EXP-010/011 | 五 seed 稳定改善，但无 local topology | synchronous global update |
| Haar / wavelet | EXP-003/007 | 未胜 matched uniform CFG；收益非频率特异 | FP32 transform self-check |
| HR 2D tile condition | `globalshape_hrtile_step0/6` | 低于 clean controls | condition cache/coverage trace |
| target-context-hard | step10 | 小幅三项全胜，但仅单图单 seed | hard owner/membership |
| old global shape/geometry prior | `global_*prior_tile_eval` | 约 23–25 px 投影误差，旧 distance/FOV 不一致 | 仅保留 subdivision diagnostics |
| independent tile MoGe camera | `tile_camera_independent_*` | PSNR 约 4–9，严重不一致 | 不再作为相机 |

另有参数/记录异常：

- `pixal3d_batch_eval_2048/metrics.csv` 实际记录
  `pipeline_resolution=1536`；
- `joint_tile_1024_all` 的 `replace_last_n=0`、coverage=0，融合未启用；
- `--decimation-target` 只记录 requested 值，并未真正 decimate；
- `sym4` 与 `hiwave_reserved` 是显式拒绝，不是可用实现；
- 用户会话中的 79GB tex-decoder OOM 未保存到历史 outputs，不能从 outputs
  推断“OOM 从未发生”。

## 9. 主要失败原因分类

### 9.1 Support/拓扑不对应

- global/local SparseTensor 行序没有物理语义；
- C64→C1024 decoder 会产生数量不同的 support；
- direct union 造成点数、面数和显存爆炸；
- nearest/average 容易把前后表面或双壳绑定在一起。

### 9.2 坐标语义混用

- 旧路线把 tile FOV 缩小却保留 global distance；
- 旧路线 clamp 越界点；
- material center 与 flow endpoint 混用；
- local camera view 与 global crop camera view 混淆。

### 9.3 Flow/latent 不是视角等变物理量

global 与 tile 的 condition、support 和上下文都不同。同一 timestep 不保证
latent/velocity 语义可逐点平均。历史的 replace/average 已表现为暗化、
彩色污染和马赛克。

### 9.4 Decoder hierarchy 分布错配

tile shape SLat 配合强制 global `guide_subs` 时，texture decoder 被要求在
与其生成 shape feature 不一致的 hierarchy 上解码，导致模糊或黑点。

### 9.5 材质查询与几何表面脱节

renderer 在连续 triangle surface position 上做 sparse trilinear material
query。只有零散 attrs 命中而没有连续邻域时，会看到“一个个颜色点”；几何
正常不代表材质 lookup 有效。

### 9.6 指标与视觉协议混排

- full-frame black background 会放大背景贡献；
- 1024/2048 metric、4096/1024 render 不能直接排名；
- Blender 与 native renderer 不能直接排名；
- 只看 PSNR 可能奖励模糊；
- 单图/单 seed 小增量不足以宣称成功。

## 10. 可复现命令

### A/B/C/D0

```bash
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/nvme04/yyyan/miniconda3/envs/pixal3d/bin/python \
pixal3d_projective_tile_generation_eval_projected_c64_only_copy.py \
  --image assets/choose/0_img.png \
  --output-dir outputs/codex_baseline_abcd_0img_seed42_tiles24_26_27_20260727 \
  --model-path /home/nvme04/yyyan/download/model/Pixal3D \
  --seed 42 --tile-ids 24,26,27 \
  --enable-direct-tile-comparison \
  --no-enable-modified-material-comparison \
  --render-resolution 1024 \
  --modified-global-render-resolution 1024 \
  --baseline-render-resolution 4096 \
  --metric-resolution 1024 \
  --render-ssaa 2 --render-peel-layers 8 \
  --envmap studio --no-use-envmap-bg \
  --lpips-net vgg --metric-device cuda --no-skip-lpips
```

### D-v7

```bash
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/nvme04/yyyan/miniconda3/envs/pixal3d/bin/python \
pixal3d_projective_tile_generation_eval.py \
  --image assets/choose/0_img.png \
  --output-dir outputs/codex_baseline_d_best_v7_0img_seed42_tiles24_26_27_20260727 \
  --model-path /home/nvme04/yyyan/download/model/Pixal3D \
  --seed 42 --tile-ids 24,26,27 \
  --tile-size 1024 --tile-stride 512 \
  --min-tile-tokens 1000 --max-num-tokens 100000000 \
  --boundary-epsilon 1e-5 --max-outside-fraction .10 \
  --render-resolution 1024 --baseline-render-resolution 4096 \
  --metric-resolution 1024 --render-ssaa 2 --render-peel-layers 8 \
  --envmap studio --no-use-envmap-bg \
  --lpips-net vgg --metric-device cuda --no-skip-lpips
```

完整 sampler 参数和派生设备信息保存在各目录
`effective_run_config.json`，运行 stdout 保存在 `run.log`。

## 11. Support 分析状态与尚未完成内容

固定三块的 decoded local C1024→global strict correspondence、nearest
support/global vertex-proxy distance、center/mid/edge margin 分布，以及
tiles 26/27 overlap 的多 tile geometry/material agreement 已完成。完整
结论、数表、限制和复现命令见
`research/SUPPORT_ANALYSIS_20260727.md`；机器可读结果位于：

`outputs/codex_support_v7_0img_seed42_tiles24_26_27_20260727/support_correspondence_analysis`

fixed-global-support material residual 的配对 4096 render、三块 crop 指标、
强度消融和 bitwise 几何/属性不变量也已完成，见：

`research/FIXED_SUPPORT_MATERIAL_EXPERIMENT_20260727.md`

仍未完成、不得提前宣称：

- depth、normal、silhouette 的统一定量；
- 轻微新视角；
- 多输入图像和多 seed；
- 完整统一模型优于 global control。

这些内容完成前，当前结果只说明 local 生成在给定 crop view 上有质量潜力，
不说明已经得到更好的统一 global 3D。
