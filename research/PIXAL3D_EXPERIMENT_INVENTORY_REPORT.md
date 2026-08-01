# Pixal3D 实验整理与结果报告

更新日期：2026-08-01  
数据范围：本项目 `outputs/`、`research/` 中已落盘的实验摘要、指标和可视化；默认以 seed=42、`assets/choose/0_img.png` 为主线。所有结果均来自项目实际运行记录，不把 smoke/诊断结果当作正式结论。

## 1. 结论摘要

目前最可靠的结论有四条：

1. **高分辨率图像条件确实有收益，但必须保持 latent 的全局坐标身份。** 原始 Global-C64 行 gather + HR tile condition 在 4096 评测上从 PSNR 13.9680 提升到 15.7870 dB（+1.8190 dB），SSIM 基本持平（0.680013 → 0.678919）。按照用户最终修正的流程，从官方 Global-1024 生成 C1024，再固定量化到 C256，使用一份全局噪声、投影分块、同一全局行的 shape/texture velocity 做平均并只更新一次，PSNR 进一步达到 16.5046 dB（相对官方 Global-1024 +2.5366 dB），但 SSIM 下降 0.00525，且解码结果达到 55.8M 顶点/107.3M 面，代价很高。
2. **把不同局部坐标系的 latent/velocity 直接运输、拼接或平均是不可靠的。** 多个 projective/local-return/global-merge 实验产生明显回退、碎片化和背面伪影；旧的统一 global velocity replacement/average 由 17.0683/0.6622/0.2116 回退到 14.3229/0.5871/0.3257（PSNR/SSIM/LPIPS）。
3. **固定全局几何 support、只做保守材质残差是稳定的小幅改进。** alpha=0.25 是测试值中唯一对三个 tile 都一致获胜的设置；平均增益约 +0.01865 dB PSNR、+0.001598 SSIM、LPIPS -0.002382，说明 support 固定比盲目融合更重要。
4. **真正的瓶颈不是 camera round-trip。** 相机正逆变换误差很小；主要问题是 global/local support 的离散化、碰撞和 token 对应关系。C1024→local C64 的匹配率/Jaccard 很低，不能假设“投影到同一像素”就等价于“同一 latent”。

## 2. 官方基线与测量校验

官方路径保持 Pixal3D 原始 pipeline：`global SS C32 → Shape512 → learned global C64 → Shape1024 → Texture1024 → decoder`。多个目录重复验证了相同的 camera、sparse support、sampler、noise capture 和 decoder 行为；正式全图基线通常约为：

| 记录 | PSNR | SSIM | LPIPS | 备注 |
|---|---:|---:|---:|---|
| `global_c64_hr_tile_condition_ablation` control | 13.96796 | 0.68001 | — | 4096 shaded render，官方 1024 decoder |
| `codex_baseline_abcd...` / `...d_best_v7...` | 14.33861 | 0.62246 | 0.23470 | 另一套历史 renderer/metric convention，不能与上一行直接混排 |
| `joint_online_canonical_posterior` global baseline | 14.18089 | 0.62055 | 0.24061 | joint-posterior suite 的基线 |

由于历史实验使用过不同 render resolution、SSAA、peel layers、renderer 和 camera 导出版本，跨 suite 的绝对分数不能直接排序；报告中的“增益”只在同一 `summary.json` 内进行。

## 3. 实验主线 A：全局 support 上直接使用 HR tile condition

目录：[`outputs/global_c64_hr_tile_condition_ablation/seed_42`](../outputs/global_c64_hr_tile_condition_ablation/seed_42/summary.json)

方法：从官方调用中捕获原始 global C64 support 和初始 shape/texture noise；4096 canonical image 切成 1024 tile、stride=512；根据投影位置对 global C64 行做 subset；不做 local 坐标变换、不重新量化、不重编码、不做 latent transport；每个 tile 单独跑完整 shape/texture flow，再按相同 global row 对 velocity 做算术平均并执行一次全局更新。

校验：49 个 tile 中 48 个 active；13,566/13,566 行覆盖且全部进入 overlap；12 个 shape + 12 个 texture steps；immutable coordinates/order、full coverage、zero fallback 均通过。

结果：

| 方法 | PSNR | SSIM | 几何 |
|---|---:|---:|---:|
| 官方 global 1024 | 13.96796 | 0.68001 | 4.08M vertices / 8.23M faces |
| HR tile condition | 15.78701 | 0.67892 | 3.77M vertices / 7.46M faces |
| 差值 | **+1.81904 dB** | -0.00109 | 更少但更贴近可见区域 |

视觉上输入视图的局部纹理更清晰；多视图中正面改善明显，背面没有 ground truth，能保持整体形状但存在材质泄漏/高频噪声。该实验证明“同一全局 latent 行只换图像 condition”本身有效。

## 4. 实验主线 B：Global-1024 → C256 的正确全局空间 velocity-average

目录：[`outputs/global1024_to_c256_hr_tile_velocity_average/seed_42`](../outputs/global1024_to_c256_hr_tile_velocity_average/seed_42/summary.json)

这是最终纠正后的 C256 实验，不是早先误跑的“C512→C256 后直接 decode2048”路径。流程是：先正常生成官方 Global-1024；Shape1024 decoder subdivision 得到 4,082,509 个 C1024 source rows；固定量化到 244,514 个 global C256 rows；一次性生成 shape seed=443、texture seed=543 的全局噪声；按 4096 图像投影将行分到 49 个 1024/512 tile；所有 flow 都在同一 global C256 空间中进行，shape 和 texture 的 overlap velocity 分别平均，且每一步只更新一个 global state。

不变量全部满足：244,514/244,514 行覆盖，全部有 overlap，最大 membership=4；没有 local transform、re-quantization、re-encoding 或 latent transport。C256 native 4096 decoder 成功。

| 方法 | PSNR | SSIM | 几何 |
|---|---:|---:|---:|
| 官方 Global-1024 | 13.96802 | 0.68001 | 4.08M vertices / 8.23M faces |
| 正确 C256 HR tile velocity-average | 16.50458 | 0.67476 | 55.76M vertices / 107.27M faces |
| 差值 | **+2.53656 dB** | -0.00525 | 约 13.7× vertices、13.0× faces |

总耗时约 1,943 s（约 32.4 min，A800-SXM4-80GB，CUDA device 4）。输入视图中 shell、面部、四肢和城堡细节更亮更锐；背面形状完整，但 C256 的碎片/噪声更多，前景蓝绿材质有向背面泄漏。该实验的直接结果是：全局坐标 velocity averaging 带来明显单视图 PSNR 收益，同时增加了几何规模、碎片化和背面材质泄漏。

## 5. 实验主线 C：局部坐标、support 回收与 velocity 同步

代表目录：

- `outputs/c1024_overlap_velocity_sync_geometry_fusion`
- `outputs/c1024_direct_local_return_global`
- `outputs/c1024_direct_local_return_global_2048`
- `outputs/tile_slat_return_global`
- `outputs/tile_slat_global_merge_with_localdecode_compare`
- `outputs/tile_slat_object_correct_spatial_merge`

尝试过：将 global C1024 投影到 tile 后 recanonicalize 到 local camera、量化为 local C64、用 exact global C1024 source key 建图、逐 Euler step 平均 linked velocity；或完整 local decode 后再回收到 global C64/C1024；以及 local decode geometry 的 ownership/halo merge。

结果整体不稳定：

- `c1024_overlap_velocity_sync_geometry_fusion` 的代表值为 baseline 14.33861/0.62246/0.23470，fusion 16.34872/0.62403/0.32311；PSNR 有提升但 LPIPS 变差，且只成功 45/49 tiles。
- `c1024_direct_local_return_global` 的 global-return 路线约 14.32339/0.57369/0.33276，而独立 local decode merge 可达 18.68581/0.76757/0.16608；两者不是同一个输出域，说明 local decode 质量不能反推 global latent merge 正确。
- `tile_slat_return_global`、`tile_slat_global_merge_with_localdecode_compare`、`object_correct_spatial_merge` 多数出现 9–12 dB 级回退或明显碎片化。

support 分析显示：global C1024 support 约 3.96M 行，tile local support 的碰撞约 86–88%；C1024↔local C64 的 Jaccard 只有约 6.33/13.99/14.49%（典型 tile）。因此直接把 local token 当成 global token，或按近邻/像素做平均，语义上是不成立的。

## 6. 实验主线 D：projective tile SR、encoded query/noise 与 canonical endpoint

### Projective tile SR

目录：`outputs/projective_tile_superres_2048`、`projective_tile_camera_eval_v2/v3/v4`。这些实验验证了 2048/4096 HR crop、projective camera 和 tile condition 的可行性，但旧的统一 global velocity replacement/average 反而回退：17.06828/0.66216/0.21161 → 14.32287/0.58709/0.32571。结论是不能把不同 tile 的预测 velocity 无条件覆盖同一 global state。

### Encoded query/noise

目录：`outputs/codex_encoded_query_noise_global_c4096_*`、`codex_encoded_query_noise_overlap_smoke_gpu4`、`encoded_query_noise_overlap_full`、`encoded_query_noise_global_c8192_full`。尝试把 HR image query/noise 编码到更高 C4096/C8192 support，并在 overlap 中保持一致噪声。smoke 可运行，full overlap 曾得到 19.01871/0.75832/0.17426，但其 support、renderer 和输出域与主线不同，不能作为全局 C64/C256 结论；主要价值是证明“query/noise 一致性”值得保留。

### Canonical endpoint SR

目录：`research/CANONICAL_ENDPOINT_SR_EXPERIMENT_20260727.md`、`outputs/canonical_endpoint_sr_tile24_12step_decode2048_20260727`。同步 C256/C128 endpoint 的一 tile smoke 通过，但 C256→4096 全解码在 80GB A800 上先达到约 74.24GB、随后 normalization 额外申请约 25.18GiB 而 OOM。C128→2048 的 12-step ablation 反而回退：15.20315/0.62947/0.23501 → 13.00467/0.57304/0.31599；tile24 也从 11.04526/0.34590/0.63866 降至 10.18073/0.24248/0.66287。该结果支持“endpoint 设计必须与全局 support/decoder 共同验证”。

## 7. 实验主线 E：固定 support 的材质残差

目录：`outputs/codex_fixed_global_support_material_alpha025_winner_20260727`，配套说明见 `research/FIXED_SUPPORT_MATERIAL_EXPERIMENT_20260727.md`。

策略是固定官方 global geometry/support，只允许 conservative base-color residual；不新增几何、不改变坐标、不做盲目多候选平均。alpha 测试了 0.1、0.25、0.5、1.0；alpha=0.25 是三个 tile 都一致获胜的值。

汇总平均增益：PSNR +0.01865 dB，SSIM +0.001598，LPIPS -0.002382；global 汇总约 +0.00608/+0.000829/-0.000620。幅度不大，但稳定、可解释，适合作为后处理或 winner-selection 组件，而不是主 SR 方法。

## 8. 实验主线 F：2048 pipeline、patch/wavelet/texture sweep

目录族：`outputs/pixal3d_2048_*`、`outputs/pixal3d_batch_eval_1024/1536/2048*`。

覆盖了：global clean control、canonical global control、global-shape + HR-tile texture、local/original texture、uniform CFG（1.5/2/2.5/3）、wavelet/Haar、patch start step 6/10/12、global-coordinate/local-coordinate patch、多 tile paired 3D patch、multi-seed（42/123/2024/3407/9999）等。它们主要回答“在哪个 flow step 注入 HR condition、texture guidance 强度和 wavelet 形式如何影响 2048 输出”。

结果模式：texture-only 和 late-step patch 通常比同时改 shape 更稳定；较强 uniform CFG 或直接 global regression 容易出现颜色过饱和、局部破碎；multi-seed 用来测方差而非寻找单一最高分。由于这些 metrics 使用独立的 2048 suite（render=2048、metric=1024，且 camera/environment 与 4096 主线不同），只应在各自 `metrics.json` 内比较，不能与本报告第 3–5 节的 4096 分数直接合并排名。

## 9. 实验主线 G：joint posterior 与 visibility routing

### Joint online canonical posterior

目录：`outputs/joint_online_canonical_posterior`。这是 training-free、0 gradient update 的 CCA/ridge-style posterior correction，对比 local baseline、old anchor、shape posterior、texture posterior、joint fixed/per-step：

| 配置 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| local_hr_baseline | 18.47197 | 0.75596 | 0.16769 |
| old_anchor | 14.61858 | 0.62599 | 0.25248 |
| shape_posterior | **18.72780** | **0.76363** | **0.16693** |
| texture_posterior | 16.75833 | 0.67990 | 0.21709 |
| joint_posterior_fixed | 17.74677 | 0.69607 | 0.21220 |
| joint_posterior_per_step | 16.88153 | 0.66470 | 0.23102 |

该 suite 说明 shape posterior 可能带来可观收益，但 joint/per-step correction 会破坏稳定性；不能把 shape 的收益简单外推到 texture 或全局 fusion。

### Visibility-routed conditioning

目录：`outputs/visibility_routed_conditioning`。按可见性/投影置信度选择 condition，代表值约 18.45847/0.75304/0.17017，接近 joint suite 的 local baseline。它支持“先做可见性路由、再决定哪些 condition 参与”这一方向，但仍属于独立 local suite，尚未证明能安全替代全局 absolute-row averaging。

## 10. 已确认无效或需排除的记录

- `outputs/global_c256_hr_tile_condition_decode2048/seed_42` 是早期错误路径：直接从较低阶段 C512/C256 改路径后 decode2048，**不是**“先官方 Global-1024、再 C256、再全局噪声和 velocity averaging”。该目录已写入 `SUPERSEDED.md`，不应纳入方法结论。
- 只跑少量 tile 的 smoke、direct-mesh smoke、camera roundtrip、O-Voxel roundtrip 只能用于工程校验，不能与完整多 tile 质量比较。
- 不同 suite 的分数若 renderer、camera、SSAA、metric resolution 或 environment 不同，不做跨目录绝对排名。

## 11. `outputs/` 完整实验台账

下面逐目录登记所有一级输出目录。`摘要数` 是该目录递归找到的 `summary.json/metrics.json` 文件数；“无摘要”表示只有中间文件、日志或图像，不能从机器摘要恢复单一数值。目录中更细的 tile、seed 和阶段结果仍以对应 JSON 为准。

### 基线、相机、渲染与 support 校验

| 输出目录 | 尝试 | 结果/状态 |
|---|---|---|
| `GPT_SR` | 早期 SR/mesh 基线与对比 | 有多组 PSNR 14.936/16.943 等历史结果；与后期官方 renderer 不同 |
| `pixal3d_original_1024` | 原始 1024 pipeline | 无摘要，保留原始中间产物 |
| `pixal3d_global_original_2048` | Global 2048 原始 CFG | 有 metrics，作为 2048 global control |
| `pixal3d_batch_eval_1024` | 1024 batch baseline | 有 metrics；单图/多 seed 统计在 JSON |
| `pixal3d_batch_eval_1536` | 1536 batch baseline | 有 metrics；用于分辨率对照 |
| `pixal3d_batch_eval_2048` | 2048 batch baseline | 有 metrics；render 2048、metric 1024 |
| `pixal3d_batch_eval_2048_global_multiseed` | 2048 global 多 seed | seed 42/123/2024/3407/9999，测随机方差 |
| `pixal3d_batch_eval_2048_local_multiseed` | 2048 local 多 seed | 同上，local 路线 |
| `moge_vs_derived_camera_multi` | MoGe camera 与 derived crop camera | 有 camera 对比摘要；round-trip 误差小，差异来自 camera 来源/裁剪 |
| `global_derived_crop_camera_eval` | derived crop camera 评测 | 有 summary，代表 PSNR 16.2281；不与主线直接横比 |
| `dino_global_compare` | DINO global source 比较 | 无摘要，只有比较产物 |
| `codex_direct_mesh_smoke_gpu4` | direct mesh/PBR smoke | smoke；代表指标约 10.82/0.435/—，非正式质量结论 |
| `codex_direct_mesh_local_pbr_smoke_gpu4` | local PBR direct mesh smoke | 无摘要，仅工程 smoke |
| `codex_vertex_pbr_multiview_unit_gpu4` | vertex PBR multiview 单元测试 | 无摘要，单元/可视化产物 |
| `codex_vertex_pbr_realmesh_smoke_gpu4` | real mesh vertex PBR smoke | success，非完整 SR benchmark |
| `ovoxel_tile_roundtrip` | O-Voxel tile round-trip | 148 个摘要；主要验证坐标/材质往返 |
| `ovoxel_tile_roundtrip_2` | round-trip 重复实验 | baseline 14.3872/0.6311，验证 renderer/坐标 |
| `ovoxel_tile_roundtrip_3` | round-trip 第三版 | 有 54 个摘要，工程校验 |
| `ovoxel_tile_global_anchor_detail_4` | O-Voxel global anchor detail | 15 个摘要；global anchor/support 细节检查，非独立方法结论 |
| `ovoxel_tile_global_anchor_hr_lr_fullsupport_4` | O-Voxel global anchor HR/LR full support | 54 个摘要；baseline 14.1806，anchor 14.6179/0.6259/0.2521，属于 anchor 对照 |
| `ovoxel_tile_global_anchor_smoke_tile24` | O-Voxel global anchor tile24 smoke | 6 个摘要；smoke/局部诊断 |
| `codex_ovoxel_tile_roundtrip_smoke_gpu4` | GPU4 O-Voxel smoke | smoke |
| `codex_ovoxel_tile_roundtrip_smoke_gpu4_v2` | GPU4 O-Voxel smoke v2 | smoke |

### Global→local、projective tile、SLat 回收和融合

| 输出目录 | 尝试 | 结果/状态 |
|---|---|---|
| `codex_baseline_abcd_0img_seed42_tiles24_26_27_20260727` | global 1024 与 tile cascade 基线 | global 14.3386/0.6225/0.2347；tile 子实验分数因 tile/merge 不同而波动 |
| `codex_baseline_d_best_v7_0img_seed42_tiles24_26_27_20260727` | v7 官方 renderer/材质路径 | global 同为 14.3386/0.6225/0.2347；tile support 仍有大量碰撞 |
| `codex_support_v7_0img_seed42_tiles24_26_27_20260727` | v7 support correspondence | global 14.9333/0.6793；local support match/碰撞统计用于诊断，不代表全局提升 |
| `global_shape1024_prior_tile_eval` | global shape1024 作为 tile prior | 有 20 个摘要，比较 geometry prior 与 local tile |
| `global_local_latent_relationship` | global/local latent relationship full analysis | 有 summary；统计 correspondence、坐标与 latent 差异 |
| `global_local_latent_relationship_smoke` | global/local latent relationship smoke | 无摘要，诊断中间产物 |
| `global_geometry1024_prior_tile_eval` | global geometry prior | 有 20 个摘要；global prior 保形但局部材质不能直接回收 |
| `global_geometry1024_prior_tile_eval_v2` | geometry prior v2 | 同类复现实验，记录 camera/support/decoder 差异 |
| `projective_tile_camera_eval_v2` | projective tile camera v2 | 代表结果含 16.7977/0.6319/0.3373，tile 结果从 9.37 到 21.13，方差很大 |
| `projective_tile_camera_eval_v3` | projective tile camera v3 | global 17.0193/0.6385/0.3284；局部 tile 指标不稳定 |
| `projective_tile_camera_eval_v4` | projective tile camera v4 | 107 个摘要；多 camera/merge 组合，未形成稳定统一收益 |
| `projective_tile_superres_2048` | projective tile superresolution | baseline 17.0683/0.6622/0.2116；旧 unified velocity replacement 14.3229/0.5871/0.3257，明确回退 |
| `tile_26_27_projected_c64_only` | 只投影 global C64 到 tile 26/27 | tile/local 结果约 14.00/0.452 或更低，证明只按投影筛选不足 |
| `tile_26_27_projected_c64_only_global` | projected C64 global 回收 | 有完整 global summary，feature/coordinate return 不稳定 |
| `tile_26_27_projected_c64_only_global_2` | projected C64 global v2 | global baseline 14.2011/0.6290，未得到稳定 SR |
| `tile_26_27_projected_c64_only_global_3` | projected C64 global v3 | local tile 约 14.00/0.452/0.435，回收有退化 |
| `tile_26_27_projected_c64_only_global_4` | projected C64 global v4 | 多 tile/support 选择，指标波动 |
| `tile_26_27_projected_c64_only_global_5` | projected C64 global v5 | 与 v4 类似，未解决碰撞 |
| `tile_26_27_test` | tile 26/27 基础测试 | global 16.4106/0.6213/0.2439；局部分数明显低 |
| `tile_26_27_test_2` | 基础测试 v2 | global 14.3386；local 14.4850/0.4580 和 13.5531/0.3346 |
| `tile_26_27_test_3` | 基础测试 v3 | global 13.8730/0.6213；local 14.8848/0.4560、13.8474/0.3309 |
| `tile_c32_c64_native_fusion_tiles_26_27` | native C32/C64 fusion | baseline 16.4106/0.6213；tile local 11.80/0.35 级别，融合失败 |
| `tile_c32_c64_native_fusion_tile12_smoke` | tile12 native fusion smoke | smoke；局部可达 17.7814/0.8456，但不是全图结果 |
| `tile_c32_cascade_global_camera` | global camera cascade | 无摘要，保留中间 cascade |
| `tile_slat_return_global` | tile SLat return global | global 14.3386；返回结果 11.6602/0.5502/0.3620，明显退化 |
| `tile_slat_global_merge_with_localdecode_compare` | global merge vs local decode | 14.3386 baseline；return/merge 11.8691 与 9.9764 级别 |
| `tile_slat_object_correct_spatial_merge` | object-correct spatial merge | 14.3386 baseline；merge 11.8691/0.5497、10.2773/0.5438 |
| `c1024_direct_local_return_global` | C1024 local decode 后回 global C64 | global 14.3386；global return 14.3234/0.5737/0.3328；独立 local decode 18.6858/0.7676/0.1661，说明两个输出域不可混同 |
| `c1024_direct_local_return_global_2048` | 同路线 2048 | 有完整 2048 summary；global-return 与 local-decode 仍有明显域差异 |
| `c1024_overlap_velocity_sync_geometry_fusion` | exact global C1024 key 的 local velocity sync | 45/49 tiles 成功；fusion 16.3487/0.6240/0.3231，PSNR 增但 LPIPS 变差 |
| `tile_three_way_tile24` | tile24 三路坐标比较 | 有 8 个摘要，比较 global/local/recanonicalized |
| `tile_three_way_coord_v2_tile24` | 三路坐标 v2 | tile24 代表 16.7319/0.6571/0.2158 |
| `tile_official_camera_compare_26_27` | 官方 camera vs tile camera | baseline 16.4106；tile 14.80/0.427、13.85/0.297 |
| `tile_official_camera_compare_26_27_2` | camera compare v2 | 同类结论，camera 不匹配会造成明显退化 |
| `tile_camera_independent_smoke` | 独立 tile camera smoke | global 16.9654；局部结果 4–6 dB，明确不适合直接拼接 |
| `tile_camera_recanonicalized_diagnostic` | recanonicalized camera diagnostic | success；用于确认 transform，不是质量实验 |

### Encoded query/noise、C4096/C8192 和材质残差

| 输出目录 | 尝试 | 结果/状态 |
|---|---|---|
| `codex_encoded_query_noise_global_c4096_smoke_gpu4` | C4096 encoded query/noise smoke | 15.1545/0.6166 与低质量 tile smoke；只证明可运行 |
| `codex_encoded_query_noise_global_c4096_smoke_gpu4_v2` | C4096 smoke v2 | 与 v1 同类 |
| `codex_encoded_query_noise_global_c4096_two_tile_smoke_gpu4` | C4096 两 tile smoke | 15.6372/0.6495；局部 11.2132/0.4900 |
| `codex_encoded_query_noise_overlap_smoke_gpu4` | overlap encoded query/noise smoke | 15.1545/0.6166；非完整结果 |
| `encoded_query_noise_overlap_full` | overlap full | 19.0187/0.7583/0.1743，但为独立 support/renderer suite，不能与 C64/C256 主线直排 |
| `encoded_query_noise_global_c8192_full` | global C8192 full | success；高分辨率 encoded support full run，资源/输出域不同 |
| `codex_fixed_global_support_material_dryrun_winner_20260727` | material winner mapping dry-run | `dry_run_mapping_complete`，无生成质量 |
| `codex_fixed_global_support_material_dryrun_weighted_gated_20260727` | weighted/gated mapping dry-run | `dry_run_mapping_complete` |
| `codex_fixed_global_support_material_alpha025_winner_20260727` | fixed support material alpha=.25 | 14.33862→14.34470；全套三个 tile 平均 +0.01865 dB、SSIM +0.001598、LPIPS -0.002382 |
| `codex_fixed_global_support_material_alpha050_winner_20260727` | alpha=.50 | 14.33865→14.34896；不如 alpha=.25 的跨 tile 稳定性 |
| `codex_fixed_global_support_material_alpha100_winner_20260727` | alpha=1.0 | 14.33869→14.35236；局部/跨 tile 一致性更差 |

### 正确 C64/C256、旧错误 C256、canonical endpoint

| 输出目录 | 尝试 | 结果/状态 |
|---|---|---|
| `global_c64_hr_tile_condition_ablation` | absolute global C64 row gather + HR tile condition | 13.96796/0.68001→15.78701/0.67892，+1.81904 dB；49 tiles/48 active，全部行覆盖 |
| `global1024_to_c256_hr_tile_velocity_average` | 官方 Global-1024→C1024→C256；全局一次噪声；absolute global C256；shape/texture velocity overlap 平均 | 13.96802/0.68001→16.50458/0.67476，+2.53656 dB；244,514 rows；55.76M verts/107.27M faces；completed |
| `global_c256_hr_tile_condition_decode2048` | 早期错误的 C512/C256→decode2048 路径 | 已由 `SUPERSEDED.md` 标记；不代表用户最终指定流程 |
| `pixal3d_c256_local_c64_haar_velocity_0img_seed42_20260728` | 早期 local C256/C64 Haar velocity | 13.7742/0.6034/0.3080，低于正确 global-space C256 路径 |
| `canonical_endpoint_sr_smoke_tile24_seed42_20260727` | endpoint SR tile24 smoke | 无摘要，只有 smoke 中间结果 |
| `canonical_endpoint_sr_smoke_tile24_seed42_20260727_v2` | endpoint SR smoke v2 | 无摘要 |
| `canonical_endpoint_sr_smoke_tile24_seed42_20260727_v3` | endpoint SR smoke v3 | 有摘要，smoke 成功 |
| `canonical_endpoint_2048_c128_smoke_tile24_seed42_20260727` | C128→2048 tile24 smoke | 无摘要 |
| `canonical_endpoint_2048_c128_smoke_tile24_seed42_20260727_gpu4` | GPU4 版本 | 无摘要 |
| `canonical_endpoint_2048_c128_smoke_tile24_seed42_ss4_20260727` | SS4 C128→2048 smoke | 有摘要，smoke 成功 |
| `canonical_endpoint_sr_tile24_12step_decode2048_20260727` | C128→2048 12-step endpoint ablation | 15.20315/0.62947/0.23501→13.00467/0.57304/0.31599，回退；C256→4096 full 另有 74GB 后 OOM 记录 |

### 2048 patch、wavelet、texture CFG 与多 seed

| 输出目录 | 尝试 | 结果/状态 |
|---|---|---|
| `pixal3d_2048_canonical_global_control` | canonical global 2048 control | metrics.json；global control |
| `pixal3d_2048_global_clean_control` | clean global 2048 control | metrics.json；无 HR tile 注入 |
| `pixal3d_2048_globalshape_hrtile_step0` | global shape + HR tile texture，从 step0 | metrics.json；最早注入版本 |
| `pixal3d_2048_globalshape_hrtile_step6` | global shape + HR tile texture，从 step6 | metrics.json；晚期注入版本 |
| `pixal3d_2048_haar_s1_control` | Haar patch strength=1 control | metrics.json |
| `pixal3d_2048_haar_s2_smoke` | Haar strength=2 smoke | metrics.json，smoke |
| `pixal3d_2048_original_cfg_regression` | original CFG patch regression | metrics.json；验证高强度 patch 回退 |
| `pixal3d_2048_target_context_hard_step10` | target-context hard step10 | metrics.json；测试 late hard conditioning |
| `pixal3d_2048_multitile_paired_3dpatch_step6` | multi-tile paired 3D patch step6 | metrics.json |
| `pixal3d_2048_multitile_paired_3dpatch_step10` | multi-tile paired 3D patch step10 | metrics.json；和 step6 对照 |
| `pixal3d_2048_texture_exp004_global_regression` | global texture regression | metrics.json |
| `pixal3d_2048_texture_exp005_local_original_step6` | local original texture CFG step6 | metrics.json |
| `pixal3d_2048_texture_exp006_uniform_g1p5_step6` | uniform CFG 1.5 | metrics.json |
| `pixal3d_2048_texture_exp007_haar_g1p5_step6` | Haar CFG 1.5 | metrics.json |
| `pixal3d_2048_texture_exp008_uniform_g2_step6` | uniform CFG 2 | metrics.json |
| `pixal3d_2048_texture_exp009_uniform_g2p5_step6` | uniform CFG 2.5 | metrics.json |
| `pixal3d_2048_texture_exp010_uniform_g3_step6` | uniform CFG 3 | metrics.json |
| `pixal3d_2048_texture_exp011_multiseed_g3_step6` | uniform CFG 3，多 seed | seed 123/2024/3407/9999；用于稳定性，不与单 seed 绝对排名 |
| `pixal3d_hr_image_tile_texture_flow_step6` | HR image tile texture flow | metrics.json；texture-only 注入 |
| `pixal3d_membership_velocity_step6` | membership velocity step6 | metrics.json；记录 tile membership/velocity |
| `pixal3d_membership_velocity_step11` | membership velocity step11 | metrics.json；记录 late-step membership |
| `pixal3d_batch_eval_2048_patch_global_coord` | 2048 global-coordinate patch | metrics.json |
| `pixal3d_batch_eval_2048_patch_local` | 2048 local-coordinate patch | metrics.json；与 global-coordinate 对照 |

### Joint posterior、tile cascade、visibility

| 输出目录 | 尝试 | 结果/状态 |
|---|---|---|
| `joint_online_canonical_posterior` | training-free shape/texture/joint posterior | local baseline 18.47197/0.75596/0.16769；shape posterior 18.72780/0.76363/0.16693；texture 16.75833；joint fixed 17.74677；joint per-step 16.88153 |
| `joint_online_canonical_posterior_smoke_7tiles` | 7 tile posterior smoke | 有 summary，非全图 |
| `joint_online_canonical_posterior_smoke_tile24` | tile24 posterior smoke | 10.5811/0.5579/0.3472 等，非全图 |
| `joint_tile_1024_all` | 1024 joint tile 全量 | 有 traces summary |
| `joint_tile_1024_smoke` | 1024 joint smoke | 有 traces summary |
| `joint_tile_2048_smoke` | 2048 joint smoke | 有 traces summary |
| `joint_tile_2048_smoke_v2` | 2048 joint smoke v2 | 有 traces summary |
| `joint_tile_2048_tile24_replace6` | 2048 tile24 replace6 | 无摘要 |
| `joint_tile_master_union_all` | joint tile master union | 有 2 个摘要，union/coverage |
| `joint_tile_master_union_all6` | master union all6 | 有 2 个摘要 |
| `joint_tile_projective_tile24_replace6` | projective tile24 replace6 | 有 2 个摘要 |
| `joint_tile_v5_tile24_replace6` | joint v5 tile24 replace6 | 有 2 个摘要 |
| `visibility_routed_conditioning` | visibility-routed condition | 代表 18.45847/0.75304/0.17017；接近 local baseline，但为独立 local suite |

## 12. 项目中对应的实验代码/测试登记

本项目根目录中与上述实验直接相关的改动脚本包括：

| 代码 | 对应内容 |
|---|---|
| `pixal3d_global_c64_hr_tile_condition_ablation.py` | 正确的 global C64 absolute-row HR tile condition；shape/texture 全 flow、velocity 平均 |
| `pixal3d_global1024_to_c256_hr_tile_velocity_average.py` | 最终纠正的 Global-1024→C256 全局空间实验 |
| `pixal3d_projective_tile_generation_eval.py` | projective tile generation/merge 主实验 |
| `pixal3d_projective_tile_generation_eval_projected_c64_only.py` | projected global C64-only 分支 |
| `pixal3d_projective_tile_superresolution.py` | 2048 projective tile SR 与 unified velocity 路径 |
| `pixal3d_projective_tile_wavelet_fusion_c256.py` | C256 wavelet/fusion 分支 |
| `pixal3d_tile_c1024_overlap_velocity_sync_local_decode_merge.py` | C1024 exact-source velocity sync + local decode merge |
| `pixal3d_tile_c1024_local_slat_and_local_decode_return_global.py` | local C64 return global C64；closest-center 选择 |
| `pixal3d_tile_slat_return_to_global_closest_center_stride1024.py` | SLat 回 global 的 closest-center 路径 |
| `pixal3d_tile_slat_return_to_global_closest_center_stride1024_with_localdecode_compare.py` | global return 与 local decode 对照 |
| `pixal3d_tile_encoded_query_noise_flow_overlap_render.py` | encoded query/noise overlap flow |
| `pixal3d_global_local_latent_relationship_analysis.py` | global/local latent correspondence、camera/support 统计 |
| `pixal3d_support_correspondence_analysis.py` | support match、碰撞、Jaccard、round-trip 分析 |
| `pixal3d_fixed_global_support_tile_material_fusion.py` | fixed-support material residual alpha/winner/gated 实验 |
| `pixal3d_canonical_tile_superresolution.py` | canonical endpoint/C128/C256 tile SR |
| `pixal3d_joint_tile_cascade.py` | joint tile cascade 与 union |
| `pixal3d_tile_online_canonical_posterior_shape_texture.py` | online posterior shape/texture/joint |
| `pixal3d_online_canonical_posterior_report.py` | posterior 结果汇总与报告 |
| `pixal3d_visibility_routed_conditioning.py` | visibility-routed conditioning |
| `pixal3d_visibility_routed_equivalence.py` | visibility routing 等价性检查 |
| `pixal3d_tile_camera_independent_test.py` | tile 独立 camera 诊断 |
| `pixal3d_tile_three_way_compare_test.py` | tile 三种坐标/融合路线比较 |
| `pixal3d_directory_texture_eval*.py`、`pixal3d_single_image_texture_eval.py` | 2048/texture/材质渲染评测 |
| `batch_pixal3d_view_fidelity.py`、`used/summarize_pixal3d_metrics.py` | batch fidelity 与 metrics 汇总 |

配套测试文件：`tests/test_global_c64_hr_tile_condition_ablation.py`、`tests/test_global1024_to_c256_hr_tile_velocity_average.py`、`tests/test_canonical_tile_sync.py`、`tests/test_overlap_velocity_sync.py`、`tests/test_multitile_paired_3d_flow.py`、`tests/test_hr_image_tile_texture_flow.py`、`tests/test_target_context_hard_flow.py`、`tests/test_visibility_routed_attention.py`、`tests/test_pbr_mesh_renderer_face_chunking.py`。最近一次完整 pytest 结果为 **50 passed**。

## 13. 关键原始记录

- [`research/CODEX_RESEARCH_AUDIT_20260727.md`](CODEX_RESEARCH_AUDIT_20260727.md)
- [`research/SUPPORT_ANALYSIS_20260727.md`](SUPPORT_ANALYSIS_20260727.md)
- [`research/FIXED_SUPPORT_MATERIAL_EXPERIMENT_20260727.md`](FIXED_SUPPORT_MATERIAL_EXPERIMENT_20260727.md)
- [`research/CANONICAL_ENDPOINT_SR_EXPERIMENT_20260727.md`](CANONICAL_ENDPOINT_SR_EXPERIMENT_20260727.md)
- [`outputs/global_c64_hr_tile_condition_ablation/seed_42/EXPERIMENT_REPORT.md`](../outputs/global_c64_hr_tile_condition_ablation/seed_42/EXPERIMENT_REPORT.md)
- [`outputs/global1024_to_c256_hr_tile_velocity_average/seed_42/EXPERIMENT_REPORT.md`](../outputs/global1024_to_c256_hr_tile_velocity_average/seed_42/EXPERIMENT_REPORT.md)
