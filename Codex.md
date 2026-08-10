请直接修改当前 Pixal3D tile encode/decode 实验代码，实现以下完整流程。

先正常运行 Pixal3D `1024_cascade`，得到 global baseline `MeshWithVoxel`：

* `vertices/faces`：global mesh；
* `coords/attrs`：global 连续 PBR 属性场；
* baseline shape/texture SLat。

4096 canonical image 按 `1024 tile、512 stride` 切成 49 个 tile。不要额外构建 halo；相邻 tile 的 50% 重叠区域自然提供上下文。

对每个有效 tile：

1. 根据三角形投影包围盒与 tile 矩形是否相交选择 global mesh faces，不再使用 face centroid。保留投影进入 tile 的前表面、背面和遮挡面。
2. 将选中三角形的 global 顶点通过现有精确 global→local camera mapping 变换到 local object space，保持 faces connectivity，得到 local mesh。
3. 对 local mesh 重新调用 `mesh_to_flexible_dual_grid`，生成正确的 local 1024 geometry O-Voxel：

   * `local_coords`
   * `local_dual_vertices`
   * `local_intersected`
4. 禁止把 global PBR O-Voxel 坐标前向变换并 round 到 local index。使用最准确的材质重采样方式：

   * 对每个 `local_coords` 计算 local voxel center；
   * 找到穿过该 voxel 的 local triangles；
   * 将 voxel center 投影到这些三角形，计算重心坐标及距离权重；
   * 得到对应的 local surface points；
   * 将 surface points 精确逆变换回 global object space；
   * 调用 global baseline `MeshWithVoxel.query_attrs()` 三线性查询 `base_color/metallic/roughness/alpha`；
   * 多个三角形贡献时按距离权重平均；
   * 最终生成与 `local_coords` 完全相同 support 的 `local_attrs`。
5. 使用：

   * `(local_coords, local_dual_vertices, local_intersected)` 输入 shape encoder；
   * `(local_coords, local_attrs)` 输入 PBR encoder；
     得到 local reference `shape_slat` 和 `tex_slat`。禁止继续使用 latent support intersection 静默丢点；检查并记录两个 encoder 输出坐标是否一致。
6. 按 Pixal3D 现有训练或 sampler 代码中的原生 flow-matching convention，对 reference shape/texture SLat 重新加噪，不要自行猜测公式。使用完全相同的 seed、timestep、CFG 和 sampler 参数：

   * tile 1024 图重新计算图像条件；
   * 从加噪后的 reference shape SLat 运行 shape flow；
   * texture flow 使用生成后的 shape SLat support，并按原生代码将 shape SLat 作为条件；
   * 联合 decode 得到带 PBR 材质的 local `MeshWithVoxel`。
7. 每个 tile 保存并渲染：

   * local reference mesh；
   * reference SLat decode；
   * tile-flow 后的 local mesh；
   * tile 原图；
   * shape/texture token 数、坐标一致性、PBR 属性范围和运行时间。

随后将所有成功 tile 的 flow 后结果返回 global：

1. 在 local mesh 的每个 face corner 位置，调用该 tile 的 local `MeshWithVoxel.query_attrs()`，取得连续的 base color、metallic、roughness、alpha。
2. 将 local vertices 精确逆变换回 global object space。
3. 不做 welding、去重、remesh、seam repair、face 删除或 overlap 融合；直接将所有 tile patch 的 vertices/faces 拼接，保留重叠几何。
4. 每个 face corner 保留独立 PBR 属性。新增 nvdiffrast 渲染路径，在三角形内部对 face-corner PBR 属性做重心插值，并使用项目现有环境光与 PBR shading。
5. 面数过多时必须按 face chunk 分块 rasterize/shade，并正确进行深度合成，不能因为显存不足减少 mesh 或只渲染部分 tile。
6. 输出一个未修复、未合并的 global tiled mesh/GLB 场景；每个 tile patch 可作为独立 primitive 保存。GLB 至少保存几何和 base color，完整 PBR 评价以 nvdiffrast 渲染为准。

最终生成统一对比结果：

* 原始输入图；
* ordinary global 1024 baseline render；
* 拼回 global 的全部 local tile render；
* 三图横向对比；
* overlap/seam 可视化，可使用 tile ID、覆盖次数或随机 tile 标识显示重叠区域；
* 正面视角计算 PSNR、SSIM，并在现有依赖可用时计算 LPIPS；
* baseline 对原图、stitched local 对原图、stitched local 对 baseline 三组指标；
* 使用固定 global 相机轨迹渲染多视角，包括正面、左右侧面、背面、俯视、仰视和 turntable；
* baseline 与 stitched local 必须使用完全相同的相机、环境光、分辨率、SSAA 和背景；
* 保存每个视角的 baseline/local 并排图和多视角汇总视频或 GIF。

删除或停用旧流程：

* global O-Voxel center → local O-Voxel index；
* local index 冲突后保留最近项；
* remap 后直接复制 global attrs；
* shape/PBR latent support intersection；
* decoded local O-Voxel index → global O-Voxel index。

保留现有相机 round-trip 检查、错误隔离、JSON summary、低显存模式和命令行参数。先用单 tile 验证 reference mesh → local O-Voxel → SLat → decode 的几何与材质闭环，再运行全部 49 个 tile。
