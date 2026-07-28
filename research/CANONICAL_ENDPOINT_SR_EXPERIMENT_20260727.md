# Canonical clean-endpoint tile synchronization experiment

Date: 2026-07-27

## Outcome

The canonical synchronization implementation is complete enough to run the
real Pixal3D SS32, Shape512, Shape1024 and Texture1024 models.  It no longer
uses the old global/tile raw-velocity average.

The full C256 latent path was validated with a one-tile real-model smoke run,
but a single C256 -> 4096 decoder pass is larger than one 80 GB A800 for this
input.  A separate, explicitly labelled C128 -> 2048 decode ablation rendered
successfully through the native Pixal3D `MeshWithVoxel` renderer.  That
ablation regressed, so it is retained as a negative result and must not be
presented as a successful 4096 result.

## Implementation

New files:

- `pixal3d/pipelines/canonical_tile_sync.py`
- `pixal3d_canonical_tile_superresolution.py`
- `tests/test_canonical_tile_sync.py`

Small compatible extension:

- `Pixal3DImageTo3DPipeline.sample_sparse_structure` accepts optional explicit
  dense noise, allowing global/local SS flows to share the same spatial noise
  realization.

The new path implements:

1. exact per-point global/local camera transforms from the projective
   evaluation route;
2. sparse common target cells and fractional overlap incidence;
3. stateless noise namespaces for SS16, Shape512, Shape1024 and Texture1024;
4. partial-overlap covariance-preserving local noise restriction;
5. global trajectory updates using only the ordinary global velocity;
6. local clean-endpoint residuals, robust Huber merge and opposite-direction
   rejection;
7. coverage-aware zero-global-parent-mean high-pass;
8. synchronized local velocity recovered with Pixal3D's actual
   `FlowEulerSampler._xstart_to_pred` path;
9. mandatory projected anchors and gated local topology candidates;
10. multi-tile/global-narrow-band topology acceptance;
11. one unified shape/material support and native decoder/renderer path;
12. exact face-chunk raster merge for meshes too large for one nvdiffrast
    submission.

No modified global-geometry/tile-material route from
`pixal3d_projective_tile_generation_eval_projected_c64_only_copy.py` is called.

## Tests

Command:

```bash
python -m pytest -q \
  tests/test_canonical_tile_sync.py \
  tests/test_hr_image_tile_texture_flow.py \
  tests/test_multitile_paired_3d_flow.py \
  tests/test_target_context_hard_flow.py \
  pixal3d_tile_camera_independent_test.py \
  pixal3d_tile_three_way_compare_test.py
```

Result:

```text
32 passed in 12.44s
```

The canonical tests cover:

- z-dependent transform round trip;
- sparse atom construction;
- `R_g P_g = I`;
- constant-field preservation;
- partial-coverage `R_g h = 0`;
- unified endpoint restriction;
- noise mean/std;
- global/local overlap covariance;
- stateless atom ordering/subset identity;
- independent noise stage namespaces;
- endpoint versus decoded-cell-center coordinate semantics.

## Real-model C256 smoke

Output:

```text
outputs/canonical_endpoint_sr_smoke_tile24_seed42_20260727_v3
```

Configuration:

- input `assets/choose/0_img.png`;
- seed 42;
- tile 24;
- SS 4 steps;
- Shape512/Shape1024/Texture1024 1 step each;
- final logical C256 support;
- no decode.

Results:

| Stage | atoms | covered atoms | max `|R_g h|` | max `|R_g x_H-x_g|` |
|---|---:|---:|---:|---:|
| Shape512 | 206,560 | 5,534 | 8.31e-9 | 8.58e-6 |
| Shape1024 | 883,802 | 38,980 | 8.14e-8 | 6.68e-6 |
| Texture1024 | 883,802 | 38,980 | 2.18e-8 | 2.15e-6 |

Final support:

- global decoded C1024 rows: 4,304,512;
- global C256 anchors: 277,951;
- local covered detail atoms: 35,662;
- final unique C256 tokens: 303,083;
- effective bandwidth: `256 x 256 x 64`;
- 7,997 learned global anchor cells outside the original active C64 Voronoi
  partition were retained as global-only nearest-parent fallback, not dropped.

## Direct 4096 decode attempt

Result:

```text
outputs/canonical_endpoint_sr_smoke_tile24_seed42_20260727_v3/
  unified_sr_4096_decode_attempt/decode_result.json
```

The decoder reached the approximately `2^24` sparse-row level.  Peak allocated
memory was 74.24 GB; the following normalization required another 25.18 GiB
and failed.  Runtime before failure was 262.94 seconds.

This establishes a concrete resource boundary:

```text
303,083 C256 input tokens -> 4096 shape decoder
```

does not fit one A800-80GB with the current decoder implementation.  Face
chunking cannot fix this because the failure occurs before meshing/rendering.

## 12-step rendered 2048 decode ablation

Output:

```text
outputs/canonical_endpoint_sr_tile24_12step_decode2048_20260727
```

This run uses 12 steps for every Pixal3D flow.  The C256 synchronized latent is
preserved in `traces/unified_latents.pt`; only the decode input is restricted
to C128 so a render can be produced on current hardware.

Render protocol:

- native `MeshWithVoxel`;
- `render_utils.render_frames` / `PbrMeshRenderer`;
- studio HDRI on black;
- 2048 render;
- metric resolution 1024;
- SSAA 2;
- 8 peel layers;
- exact 1,000,000-face chunks for the 25,362,388-face unified mesh;
- PSNR, SSIM, LPIPS-VGG and silhouette IoU;
- no modified material route.

Full-image metrics:

| Route | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| ordinary global 1024 model | 15.20315 | 0.629466 | 0.235013 |
| synchronized, C128/2048 decode ablation | 13.00467 | 0.573039 | 0.315988 |
| delta / reduction | -2.19847 | -0.056427 | -0.080975 |

Tile 24 exact canonical crop:

| Route | PSNR | SSIM | LPIPS | silhouette IoU |
|---|---:|---:|---:|---:|
| ordinary global crop | 11.04526 | 0.345904 | 0.638662 | 0.999935 |
| synchronized ablation | 10.18073 | 0.242475 | 0.662867 | 0.956596 |
| delta / reduction | -0.86453 | -0.103430 | -0.024205 | -0.043339 |

The contact sheet shows repeated/double surface structure and locally
inconsistent color.  The result is a failure.  The likely causes are:

1. C256 -> C128 occupied-cell averaging destroys the high-pass scale on which
   synchronization was constrained;
2. broadcasting one C64 global base across a much finer decoder support does
   not equal a model-generated C128 global master;
3. one tile cannot provide multi-view consensus for most of the final support.

## Commands

C256 latent smoke:

```bash
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
python pixal3d_canonical_tile_superresolution.py \
  --image assets/choose/0_img.png \
  --output-dir outputs/canonical_endpoint_sr_smoke_tile24_seed42_20260727_v3 \
  --seed 42 --cuda-device 0 --tile-ids 24 \
  --fov 0.517371749106554 \
  --ss-steps 4 --shape-steps 1 --texture-steps 1 \
  --no-decode --skip-lpips
```

12-step rendered ablation:

```bash
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
python pixal3d_canonical_tile_superresolution.py \
  --image assets/choose/0_img.png \
  --output-dir outputs/canonical_endpoint_sr_tile24_12step_decode2048_20260727 \
  --seed 42 --cuda-device 0 --tile-ids 24 \
  --fov 0.517371749106554 \
  --ss-steps 12 --shape-steps 12 --texture-steps 12 \
  --decode-resolution 2048 \
  --render-resolution 2048 --metric-resolution 1024 \
  --render-ssaa 2 --render-peel-layers 8 \
  --render-face-chunk-size 1000000
```

## Decision

Keep the canonical noise and clean-endpoint high-pass operators: their
invariants pass both synthetic and real-model tests.

Do not claim final 4096 success.  Do not use the C128 decode ablation as a
quality route.  The next engineering requirement is a decoder that can process
the unified C256 field with exact spatial chunk halos or a unified decoded
O-Voxel assembly/meshing path.  Until that exists, expanding to all 49 tiles
would spend substantially more compute but still stop at the same decoder
capacity boundary.
