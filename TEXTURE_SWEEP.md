# C256 几何与材质级联扫描（v2）

入口 `pixal3d_baseline1024_c256_64xc32_geometry.py` 在最终几何后调用
`pixal3d_sweep_texture.py`。`--geometry-source` 可以只读复用已有 24 组几何，
新结果写入独立 `--output-dir`。不传此参数则完整执行几何和材质级联。

```text
Shape C128 -> mesh2048 -> voxel4096 -> Shape Encoder C256
                                           -> start_t 加噪 -> C64 Shape flow -> final Shape C256
baseline Shape C64 -> 12步 Tex1024 -> 带材质 O-voxel1024
                                         |
                         在上述 voxel4096 的 dual 世界坐标上三线性查询
                                         |
                 6通道材质 [0,1] -> [-1,1] -> Texture Encoder posterior mean
                                         |
                         Texture C256 -> 归一化 -> start_t 加噪
                                         |
              12步 C64 Texture flow（concat=归一化 final Shape C256）
                                         |
                 一次全局 Texture Decoder -> O-voxel4096 材质
```

baseline 材质按 conditional/unconditional 分别生成，各模式共享其 baseline
1024 材质场。查询使用 `MeshWithVoxel.query_attrs()`，位置为 voxelizer 返回的
dual 坐标减 0.5，恢复到固定世界 AABB `[-0.5,0.5]^3`。Texture Encoder 和
Shape Encoder 对同一 voxel4096 support 做四次下采样，强制检查 C256 坐标
集合完全相同并重排为 Shape 行顺序。旧几何未保留 voxel4096 时，从其 bridge
mesh2048 用完全相同的 QEF 设置重新体素化；不会从最终 Shape mesh 生成初始材质。

纹理在 C256 support 上使用 C64 窗口、stride=32（最多 343 个），每步冻结全局
状态，各窗口预测后由最近中心 owner 写回速度，统一 Euler 更新。使用专用
`image_cond_model_tex_1024` 在全局坐标投影，以及对应几何的归一化 latent
作为 concat condition。conditional 使用图像条件，unconditional 使用零图像
条件；两者保留几何条件，均不做 CFG。默认 tex_seed=46。

`start_t` 取几何 12 步时间表的 `start_step`。纹理初始化为
`x_t=(1-t)*encoded_texture+(sigma_min+(1-sigma_min)*t)*noise`。
将纹理原生 12 步非线性时间表乘以 `start_t`，从该时间到 0 执行完整 12 步
Euler；不是截取末尾剩余步数。`start_step=0` 时 t=1，自然退化为纯噪声。
每组 `texture/schedule.json` 记录实际全部时间点。

shape decoder 返回的 subdivision 引导 texture decoder，得到 4096 材质体素。
`texture/textured_mesh.pt` 保存 MeshWithVoxel（几何和 PBR 体素材质），
`texture/latent.pt` 保存归一化纹理 latent；不做 UV 烘焙。
`baseline_texture/<mode>/material1024.pt` 保存 baseline 场；
`texture/baseline_material4096.pt` 保存 bridge 上查询到的材质，
`texture/initial_c256.pt` 保存加噪前编码结果，`initial_summary.json`
记录查询有效比例和 Shape/Texture 支持集对齐结果。缓存版本为 2，不复用旧纯噪声纹理。

输入视角使用相同 camera JSON，渲染未打光的 base_color。默认在 1024×1024
计算 RGB [0,1] 上的前景 MSE / PSNR：参考图为同一 canonical 变换后的输入图，
mask 为 canonical alpha > 0.5，分母为输入前景像素数 × 3。未被模型覆盖的
输入前景按黑色渲染计入误差，不取输入和渲染 mask 的交集。PSNR 使用保存 PNG
之前的 float 图；完全匹配时 JSON 的 PSNR 为 null，perfect_match=true（数学上 +∞）。

每组输出 `texture/input_view_texture.png`、`input_reference.png`、
`foreground_mask.png`、`metrics.json`。根目录 `texture_psnr.json` 汇总 24 组结果，
`conditional_texture_3x4.png` 和 `unconditional_texture_3x4.png` 显示图像与 PSNR。

```bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8 \
  /home/nvme04/yyyan/miniconda3/envs/pixal3d/bin/python -u \
  pixal3d_baseline1024_c256_64xc32_geometry.py \
  --image assets/choose/0_img.png --camera outputs/run_camera_turtle.json \
  --geometry-source outputs/baseline_encoded_s128_s256_sweep_cuda4 \
  --output-dir outputs/baseline_encoded_s128_s256_material_cascade_v2_cuda4 \
  --cuda-device 4 --resume --tex-steps 12
```

上述命令增加 `--smoke-test --tex-steps 1` 可对已有 conditional/unconditional
start_11 几何各做一次材质级联 smoke：baseline 仍跑 12 步，保留完整 4096
材质查询、Texture Encoder、全窗口单步 flow、4096 解码和输入视角渲染。
结果单独写入 `texture_smoke/`，不写入正式实验的纹理缓存；baseline1024
材质缓存可以共享。需要完整 12 步 smoke 时保留 `--tex-steps 12`。
`test_material_cascade.py` 另检查所有起点的 12 步时间表、加噪公式、真实 CUDA
三线性插值坐标和权重。单步 smoke 的 PSNR 不代表正式 12 步结果。
