# HR tile velocity mapper 小样本实验

日期：2026-08-02

## 实验协议

- 数据：`assets/small_glb_dataset_full` 中 10 个完整物体。
- 划分：按物体隔离，8 个训练物体、2 个测试物体；不存在 view/tile 泄漏。
- 设备：physical `cuda:4`，NVIDIA A800-SXM4-80GB。
- 分支：shape C64 与 texture C64；texture 注入 GT normalized shape SLat。
- 冻结模型：Pixal3D shape/texture 1024 flow。
- 可训练部分：逐 token `LowRankVelocityMapper(G, H-L)`，约束 `Phi(G,0)=0`，零初始化。
- 训练：每个分支 20 个在线随机 `t/noise` step；每次都从完整 global support 生成 `x_t`，再 gather local tile。
- 测试：两个未见物体，在 5 个时间区间中心分别测试，共 10 个测试项/分支。
- tile：每物体一个 view，选择 owner token 最多的有效 tile；local flow 使用完整 `Ck`，loss 只使用唯一 owner `Ak`。

## 不变量核验

- sparse support、坐标和行顺序不变。
- local 不量化、不重建 support。
- `x_t_tile == gather(x_t_global)` 的最大误差为 `0`。
- 每个 global row 最多一个 owner。
- global/local image projection 使用真实渲染相机与 canonical/tile 嵌套 crop；坐标闭合最大误差约 `3.25e-4` pixel。
- HR tile 4 倍下采样与 canonical-1024 对应 crop 的平均像素绝对误差为 `0.0060/255`；最大误差来自 Lanczos crop 边界支持。

## 未见物体测试结果

| Branch | G | H | L | G+(H-L) | G+Phi | Phi vs G |
|---|---:|---:|---:|---:|---:|---:|
| Shape | 0.32355135 | 0.34414807 | 0.34485361 | 0.32357375 | 0.32355450 | -0.00098% |
| Texture | 0.54265809 | 0.55942059 | 0.56016882 | 0.54299076 | 0.54264332 | +0.00272% |

其中正的 `Phi vs G` 表示 MSE 下降。

- HR local 相比 LR local 略好：shape `0.205%`，texture `0.134%`。因此 HR/LR 分辨率差并非严格为零。
- 但是 local HR 本身比 global G 更差：shape `6.37%`，texture `3.09%`。
- 直接相加在两个分支都退化：shape `0.0069%`，texture `0.0613%`。
- learned mapper 在 shape 上略退化；texture 只改善 `0.0027%`，10 个测试项中 8 个变好，但幅度远小于可作为质量提升证据的水平。
- 分时间看，shape 有正有负；texture 五个区间均为微小正值，最大改善出现在 `t=0.9`，为 `0.0186%`。

## 结论

该小样本实验没有支持“仅凭 `(G,H-L)` 的逐 token 通道映射即可产生有意义超分增益”这一强假设。结果更接近：现有 frozen flow 中确实存在很弱的分辨率残差信号，但它相对 global velocity 误差过小或方向不够稳定，mapper 的最优行为接近零修正。

这不是继续微调 learning rate、rank 或 hidden width 的理由。更有论文价值的下一步是直接验证信号瓶颈：统计 `H-L` 与所需修正 `v*-G` 的幅值和 cosine/oracle linear upper bound，并检查当前按任意训练相机渲染、但只编码一次 global SLat 的数据是否偏离原 Pixal3D 的 view-front-aligned latent 训练分布。只有确认 `H-L` 对目标修正有稳定可预测性后，才值得做完整 solver rollout 和 decoded mesh/texture 对照。

## 产物

- 汇总：`outputs/hr_tile_velocity_mapper_small/summary.json`
- 数据/坐标审计：`outputs/hr_tile_velocity_mapper_small/data_audit.json`
- Shape checkpoint/指标：`outputs/hr_tile_velocity_mapper_small/shape/`
- Texture checkpoint/指标：`outputs/hr_tile_velocity_mapper_small/texture/`
- 图：`outputs/hr_tile_velocity_mapper_small/test_flow_mse.png`

复现命令：

```bash
python pixal3d_hr_tile_velocity_mapper.py \
  --dataset-root assets/small_glb_dataset_full \
  --output-dir outputs/hr_tile_velocity_mapper_small \
  --cuda-device 4 \
  --branches shape,texture \
  --views-per-object 1 \
  --tiles-per-view 1 \
  --train-steps 20 \
  --eval-items-per-bin 2 \
  --seed 20260802
```
