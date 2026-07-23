# CODEX_TASK.md

## 任务目标

在当前 Pixal3D 2048 推理实验代码基础上，继续进行 **training-free 高分辨率纹理增强实验**。

核心目标：

- 主要优化最终输入视角渲染的 **PSNR**，目标是明显超过当前约 17 dB 的水平；
- 同时记录 SSIM、LPIPS、运行时间、显存、mesh/texture 规模等辅助指标；
- **texture 是主要优化对象，shape 只作为几何条件和辅助分支**；
- 不使用法向量相关指标作为本阶段主要判断依据；
- 使用 GPU 4 运行实验：
  ```bash
  CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1
  ```

不能假设一定能达到目标。每轮实验必须根据真实结果判断，并自主调整后续路线。

仅供参考，生成当前版本的代码的提示词“修改pixal3d_directory_texture_eval.py及其相关文件

先只考虑增强 2048 分辨率：按原流程完成 512 坐标系候选 `hr_coords`，将其重新量化到 128-grid，得到对应 2048 分辨率的固定稀疏坐标。然后在这些坐标上初始化 (N\times32) 随机特征，按照 Pixal3D 原始的 12 步、非线性时间映射、CFG interval 和 guidance rescale 完整执行一次全局 flow，得到一份对应 2048 高分辨率 base latent，同时保存全部时间状态、每一步实际用于 Euler 更新的最终 velocity 以及对应时间间隔。由于这些 velocity 已经包含 conditional/unconditional CFG、时间区间控制和 rescale，因此后续不需要 DDIM inversion，也不需要重新调用模型做反向 ODE 积分，而是可以利用保存的 velocity 对 Euler 更新进行代数逆运算，精确回退到选定的中间时间点；也可以直接读取当时保存的全局中间状态。获得统一的全局中间 latent 后，再依据三维坐标将其划分为 27 个相互重叠的 (64^3) 空间 patch（如果全空就跳过），例如 patch stride 取 32，使每个 patch 的空间尺度接近模型在 1024 分辨率下见过的 64-grid 输入，同时保持原始全局坐标、token 顺序和投影条件不变。第二遍重采样时，不让各 patch 独立完成全部 flow 后再拼接，而是在每一个 flow 时间步中，让所有 patch 分别计算 conditional 和 unconditional velocity prediction；前期先不加入小波，只验证 patch-wise prediction 经过重叠区域加权融合后是否能够近似原始全局 flow。考虑到原始pixal3d 的flow自带cfg与非线性时间映射，所以要patch单独推理时候也要对应上。把每一步的合并后的速度与之前保存的速度计算一个相似度。修改好后我来跑代码。

验证通过后，再将每个 patch 的 velocity 按三维坐标排列到空间网格上进行 3D 小波或其他空间低高频分解，而不是直接对无空间顺序的 (N\times32) feature 行序列做变换；低频部分一直保留 conditional prediction，以维持原有几何结构，高频部分使用 conditional 和 unconditional prediction 构造增强后的 CFG，从而主要修改局部高频几何特征。所有 patch 的引导 velocity 在重叠区域加权融合成唯一的全局 velocity，每个时间步只对全局 latent 更新一次，随后将更新后的 feature 与第一遍全局 flow 在同一时间点保存的 trajectory feature 做逐步衰减的 skip residual 融合，以限制 patch 上下文缺失造成的结构漂移。完成剩余 flow 步骤后，得到新的 128-grid 稀疏 shape latent，其坐标集合及 token 对齐关系始终不变，只更新对应的 (N\times32) feature，最后进行反归一化并使用设置为 2048 分辨率的 shape decoder 解码。
”

---

## 当前代码与已有路线

当前 2048 shape 实验已经完成以下能力：

1. 从 512 坐标候选重新量化到固定 128-grid。
2. 在固定坐标上运行完整 12 步 global shape flow，并保存：
   - 13 个 trajectory states；
   - 12 个实际 Euler velocity；
   - 非线性时间表和 interval。
3. 从指定中间 step 恢复状态。
4. 将 128-grid 划分成 27 个重叠 `64^3` local-coordinate patch。
5. 每个时间步分别计算 patch velocity，tent merge 后只更新一次 global latent。
6. 已实现 shape velocity 的一级 3D Haar frequency CFG：
   - `LLL` 使用 conditional prediction；
   - 其余七个高频子带使用：
     \[
     v_u+s_h(v_c-v_u)
     \]
7. 当前没有启用 HiWave skip residual。
8. 配置、trace、render、postprocess cache 已按 experiment tag 隔离。
9. Haar DWT/iDWT 和真实 sparse patch round-trip 已通过数值检查，wavelet 路径使用 FP32。

---

## 已有实验结果

同一输入、seed 42、studio light：

| Shape HR 后六步配置 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| local original CFG | 17.2825 | 0.667071 | 0.218230 |
| Haar：low=1, high=2 | 17.3507 | 0.668342 | 0.219087 |
| conditional-only：low=1, high=1 | 17.3659 | 0.669138 | 0.219073 |

初步结论：

- shape-only Haar 路线工程实现正确；
- 当前提升很小；
- `high=1` 略优于 `high=2`，说明现有改善更可能来自削弱/取消原始强 CFG，而不是高频增强本身；
- 继续只在 shape 上调参，预计难以把 PSNR 显著推高；
- 后续应把主要精力转向 **texture flow、texture condition 和纹理生成路径**。

---

## 建议的本轮主要任务（允许自助决策）

### 1. 将 local patch / frequency guidance 扩展到 texture flow

参考已经验证的 shape 实现，为 2048 texture flow 增加独立、可关闭的实验路径。

基本要求：

- texture 先完整运行原始 12 步 global flow并保存 trajectory；
- 支持从中间 step 恢复；
- 按同一固定 sparse coordinates 划分 27 个重叠 `64^3` local patch；
- 每一步对所有 patch 预测 velocity，重叠区域融合后只更新一次 global texture latent；
- texture patch 必须同步切片并对齐：
  - texture latent；
  - image `proj` condition；
  - negative image condition；
  - shape `concat_cond`；
- conditional 和 unconditional texture 分支必须保留同一份 shape condition，只改变图像条件；
- 支持 texture 的：
  - original/local baseline；
  - conditional-only；
  - uniform CFG；
  - 3D Haar frequency CFG；
- shape 和 texture 的所有实验开关、strength、interval、rescale、start step 必须独立。

不要把 texture latent channel 解释成最终 RGB/PBR 通道，也不要按 latent channel 人为分组。小波仍然只沿三维空间轴处理。

### 2. 优先研究 texture，而不是继续堆叠 shape 复杂度

初始阶段建议固定一个稳定的 shape 配置，例如：

- shape local conditional-only；或
- 当前最优的 shape 配置。

然后依次研究：

1. 原始 texture baseline；
2. texture local patch baseline；
3. texture conditional-only；
4. texture uniform weak CFG；
5. texture Haar：低频 conditional，高频弱 CFG；
6. texture guidance interval / start step；
7. shape + texture 联合配置。

HiWave skip residual、`sym4`、更复杂边界处理可以根据实验结果再决定，不要求立即实现。

---

## 自主实验与迭代要求

Codex 可以根据实验结果自行决定下一轮参数和代码修改，但必须遵循：

1. 先建立能解释因果的对照，不要同时修改过多变量。
2. 单 seed smoke test 通过后，再扩大到多 seed。
3. 不得覆盖已有 baseline；每个配置使用唯一 experiment tag 和独立输出目录。
4. 任何新路径必须保留关闭开关，关闭时不得改变原始结果。
5. 发现实现问题时先修复并做回归测试，再继续调参。
6. 优先寻找能明显提高 PSNR 的 texture 侧方案，不要长期停留在只能带来 `0.01~0.08 dB` 的 shape 微调。
7. 可以根据结果自主尝试：
   - texture patch start step；
   - original interval / all remaining；
   - weak CFG strength；
   - low/high 频率 strength；
   - guidance rescale；
   - texture trajectory residual；
   - shape/texture 联合配置；
   - 其他合理的 training-free 纹理增强或条件融合方式。
8. 不要为了追求 PSNR 静默改变输入图像、相机、metric reference、render pipeline 或 metric 实现。任何评价协议变化必须单独记录并重新建立 baseline。

---

## 每次实验必须记录

在仓库根目录维护：

```text
EXPERIMENT_LOG.md
```

每次真实运行后追加一条，不得只记录成功实验。

建议格式：

```markdown
## EXP-XXX：实验名

- 日期：
- Git commit / 代码版本：
- GPU：4
- 目标 / 假设：
- 相对上一轮改动：
- 完整命令：
- 输入：
- seeds：
- experiment tag：
- 输出目录：
- 是否通过数值检查：
- 是否 OOM / 异常：
- PSNR：
- SSIM：
- LPIPS：
- pipeline time：
- texture/mesh 规模：
- 关键 velocity / wavelet diagnostics：
- 与 baseline 的差值：
- 结果解释：
- 当前结论：
- 下一轮决策：
```

同时维护一个便于排序的：

```text
EXPERIMENT_RESULTS.csv
```

至少包含：

```text
experiment_id
date
commit
image
seed
shape_mode
texture_mode
shape_start_step
texture_start_step
shape_strength
texture_strength
shape_interval
texture_interval
psnr
ssim
lpips
status
output_dir
notes
```

对于多 seed 实验，记录每个 seed 的单独结果，并额外记录 mean/std。

---

## 当前基线

首要对照至少保留：

```text
original_cfg_step6
haar_s1_original_interval_rescale_off_step6
haar_s2_original_interval_rescale_off_step6
```

当前 seed 42 的参考值：

```text
Original local CFG:
PSNR  = 17.2825
SSIM  = 0.667071
LPIPS = 0.218230

Shape conditional-only:
PSNR  = 17.3659
SSIM  = 0.669138
LPIPS = 0.219073
```

后续所有“提升”必须明确说明相对于哪个 baseline。

---

## 运行与安全要求

默认命令前缀：

```bash
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
python pixal3d_directory_texture_eval.py ...
```

要求：

- 首次新代码路径使用单图、单 seed、`--fail-fast`；
- 先确认 trace、experiment tag、cache、render 和 metrics 对应同一配置；
- 检查 NaN、Inf、round-trip error、token/coords 对齐和 OOM；
- 每轮修改前保留备份或提交 Git；
- 不删除已有实验结果；
- 大规模实验前先估算磁盘占用；
- 失败实验也必须写入日志，并说明失败阶段。

---

## 最终交付

持续迭代直到出现以下任一情况：

1. 找到明显优于当前约 17 dB baseline 的稳定 texture/shape 联合配置；
2. 多轮合理尝试后确认当前路线收益有限，并给出证据和下一条建议路线；
3. 出现必须由用户决定的重大方法分叉。

每个阶段结束时更新：

```text
EXPERIMENT_LOG.md
EXPERIMENT_RESULTS.csv
BEST_CONFIG.md
```

`BEST_CONFIG.md` 应包含：

- 当前最佳配置；
- 完整执行命令；
- 单 seed 和多 seed 结果；
- 相对 baseline 提升；
- 主要方法解释；
- 已知问题；
- 下一步建议。

执行过程中自行做实验、读取日志、比较结果、修改代码并继续迭代，不必在每个小步骤等待用户确认。
