# Training-free global/local 3D refinement 文献审查

日期：2026-07-27

本审查只采用论文原文、作者项目页或官方代码。分类：

- **TF**：冻结预训练模型，不训练/微调新权重；
- **Analytic**：解析算法；
- **TTO**：只对当前样本做 test-time optimization；
- **Trained**：原方法需要额外训练，只能借鉴结构思想。

## 1. 规则图像域的 training-free SR

| 工作 | 类型 | 原表示与假设 | 对本项目可迁移部分 | 不能直接迁移的原因 |
|---|---|---|---|---|
| [HiFlow](https://arxiv.org/abs/2504.06232) | TF | 同一规则 2D 域上的低/高分辨率 rectified-flow；初始化、方向、加速度对齐 | 只在 exact-matched C64 上测试 late-step 弱 direction residual | global/tile condition 与 active support 不同，同 timestep feature 不保证同构 |
| [HiWave](https://arxiv.org/abs/2506.20452) | TF | 规则 2D latent，base 低频 + patch 高频 | “global 保低频、local 只提 residual”的原则 | sparse 空节点不是数值零；dense Haar 会把 support mask 边界当高频 |
| [MultiDiffusion](https://arxiv.org/abs/2302.08113) | TF | 所有 crop 是同一规则 canvas 的线性 restriction，每步最小二乘共识 | strict absolute support 上聚合多 tile material candidate | 独立 local sparse latent 不是 global latent 的线性 crop |
| [DemoFusion](https://arxiv.org/abs/2311.16973) | TF | progressive scale、skip residual、dilated global path | projected global C64 只作结构锚点，tile condition 重跑 shape/texture | dilated sampling 不能直接定义在动态 sparse topology |

结论：这些方法支持“global anchor + local residual”，不支持不同 sparse
support 上按 row index 或 nearest latent 做平均。

## 2. Point / sparse voxel / octree refinement

| 工作 | 类型 | Support/topology | 可借鉴部分 | 风险 |
|---|---|---|---|---|
| [PU-Net](https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_PU-Net_Point_Cloud_CVPR_2018_paper.html) | Trained | kNN point patches，可增加点 | local-only candidate 的 surface-distance 与防聚集 gate | 点增密不保持 O-Voxel parent/child hierarchy |
| [SnowflakeNet](https://openaccess.thecvf.com/content/ICCV2021/papers/Xiang_SnowflakeNet_Point_Cloud_Completion_by_Snowflake_Point_Deconvolution_With_Skip-Transformer_ICCV_2021_paper.pdf) | Trained | 显式 parent→children point split | topology birth 必须挂到可信 global parent | 分裂方向依赖训练，输出也不是 Pixal3D subs |
| [Adaptive O-CNN](https://arxiv.org/abs/1809.07917) | Trained | adaptive octree + local planar patch | 按 local complexity/normal variance 决定 subdivision | 其 decoder 不能替换 Pixal3D decoder |
| [Dual Octree Graph Networks](https://arxiv.org/abs/2205.02825) | Trained | 跨层 octree dual graph | 在固定 O-Voxel support 上定义邻接/Laplacian material regularizer | 空间邻近可能错误连接前后双层表面 |

这些网络本体不满足当前“冻结 Pixal3D、不额外训练”的约束，只能借鉴
parent-child lineage、邻接与 candidate gate。

## 3. Local-to-global geometry fusion

### Curless–Levoy TSDF

[A Volumetric Method for Building Complex Models from Range Images](https://graphics.stanford.edu/papers/volrange/)

- 类型：Analytic；
- 在统一物理坐标中累计带方向权重的 signed-distance observation；
- 明确区分 empty 与 unobserved；
- 可迁移：只在 global surface narrow band 内评估 tile geometry residual，
  global 提供高基础置信度，tile 只提交 observation/candidate；
- 风险：这里的 tile depth 来自同一个生成模型，不是独立测量，误差高度相关；
  相机稍错就会形成双壳；不得 densify `1024^3`。

### ElasticFusion

[论文](https://doi.org/10.1177/0278364916669237)，
[官方代码](https://github.com/mp3guy/elasticfusion)

- 类型：Analytic optimization；
- 用 position/normal/color/radius/confidence 维护 surfel map；
- 显式 projective association、active/inactive 与 candidate birth；
- 可迁移：把 O-Voxel 分为 matched/global-only/local-only；
  matched 只更新 material/confidence，local-only 需多 tile 确认；
- 风险：重叠 image tiles 不是独立相机帧，agreement 不能被当成独立测量次数。

### NeuralRecon

[论文](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_NeuralRecon_Real-Time_Coherent_3D_Reconstruction_From_Monocular_Video_CVPR_2021_paper.html)

- 类型：Trained；
- 核心思想是 global persistent sparse state + local incremental update；
- 可迁移：用 global-coordinate hash 保存 occupancy、material、confidence 与
  tile provenance；
- 不能直接使用其 learned recurrent fusion。

## 4. Geometry-aware material fusion

### Let There Be Color!

[论文](https://download.hrz.tu-darmstadt.de/pub/FB20/GCC/paper/Waechter-2014-LTB.pdf)

- 类型：Analytic optimization；
- 固定 mesh，对每个 face 选择来源 view；
- unary 使用 visibility、投影 footprint、清晰度和 photo-consistency；
- pairwise 平滑的是来源 label，而不是先平均颜色；
- 对本项目最直接的迁移：
  - 固定 global geometry/support；
  - candidate label 为 global material 或覆盖该 voxel 的 tile material；
  - 先 winner-take-all，保留高频；
  - overlap seam 只做独立、可关闭的低频校正；
- 风险：输入 tiles 来自同一张图，hidden surface 无新证据；visibility 错误会
  把清晰颜色绑到错误表面。

## 5. Differentiable test-time refinement

### nvdiffrast

[Modular Primitives for High-Performance Differentiable Rendering](https://arxiv.org/abs/2011.03277)，
[官方代码](https://github.com/NVlabs/nvdiffrast)

- 类型：TTO 工具；
- rasterization、attribute interpolation、filtered lookup、antialias 均可微；
- 最低风险实验：固定 global vertices/faces/O-Voxel coords，只优化 strict
  matched support 的 base-color residual；
- loss：tile RGB + global retention + surface-graph TV；
- 必须先验证 sparse O-Voxel query 到 RGB 的梯度没有被离散索引截断；
- 单视角极易过拟合，不可一开始同时优化 geometry/topology。

### 3D Gaussian Splatting

[论文](https://arxiv.org/abs/2308.04079)，
[官方代码](https://github.com/graphdeco-inria/gaussian-splatting)

- 类型：TTO；
- clone/split/prune 提供“由 render residual 触发容量变化”的范式；
- 只适合借鉴 candidate birth/death 顺序；
- Gaussian 是软体积表示，不能直接替代 Pixal3D surface O-Voxel。

## 6. Sparse multiresolution material

### RAHT

[Region-Adaptive Hierarchical Transform](https://www.microsoft.com/en-us/research/publication/compression-of-3d-point-clouds-using-a-region-adaptive-hierarchical-transform/)

- 类型：Analytic；
- 在 occupied octree/Morton order 上对颜色做按子树数量加权的 hierarchical
  transform；
- 比 dense Haar 更符合 sparse O-Voxel；
- 最小实验：只在 fixed global C1024 material support 上处理 decoded RGB，
  保留 global coarse coefficients，只替换 tile 中心、strict matched subtree
  的 fine coefficients；
- 风险：RAHT 本身不解决 correspondence；必须先有 absolute coordinate、
  valid mask 和 provenance；不应先对 learned texture SLat 假定线性颜色语义。

## 7. 对当前路线的研究排序

可直接做 training-free 最小实验：

1. fixed global support 上的 material source selection/residual；
2. strict matched support 的 winner-take-all 与 weighted mean 对照；
3. RAHT decoded-color fine residual；
4. fixed-support differentiable material refinement；
5. surface narrow-band + multi-tile agreement 的 controlled topology birth。

后置：

- HiFlow 式 velocity residual：只能在 correspondence 已验证后做；
- geometry birth：只能在 fixed-support material/geometry residual 已排除错误后做；
- dense Haar、global/local latent average、row-index velocity average：不再作为
  首选方向。

共同原则是先建立统一物理坐标、显式 correspondence、visibility 和
confidence，再决定 material update 或 topology birth。
