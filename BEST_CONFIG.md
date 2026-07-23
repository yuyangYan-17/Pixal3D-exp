# Current Best 2048 Texture Configuration

The best validated configuration is EXP-010/EXP-011: shape local conditional-only plus texture local uniform image CFG at strength 3.0. It is training-free and preserves the fixed sparse coordinates throughout both flows.

## Configuration

- Resolution: Pixal3D 2048 cascade; fixed 128-grid sparse coordinates
- Shape: 27 overlapping local `64^3` patches, restored from saved step 6, conditional-only, guidance rescale 0, no skip residual
- Texture: record the complete original 12-step global trajectory, restore saved step 6, then run 27 aligned local patches
- Texture conditioning: slice texture latent, positive image projection, negative image projection, and shape `concat_cond` with the same token indices; conditional and unconditional image branches share the exact shape condition
- Texture guidance: uniform CFG strength 3.0 during the pretrained `original_interval` only (steps 6, 7, and 8); guidance is inactive for steps 9–11; rescale 0
- Patch merge: overlap-weight all patch velocities, then update the global latent once per flow step
- Render/evaluation: studio light, 2048 render, 1024 metric resolution, black full-frame reference, VGG LPIPS

## Reproduction command

Fresh five-seed run:

```bash
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
python pixal3d_directory_texture_eval.py \
  --image-dir assets/choose \
  --output-dir outputs/pixal3d_2048_texture_best_g3_step6 \
  --resolutions 2048 \
  --seeds 42 123 2024 3407 9999 \
  --lights studio \
  --shape-mode conditional_only \
  --shape-start-step 6 \
  --shape-start-source saved_state \
  --texture-mode uniform_cfg \
  --texture-start-step 6 \
  --texture-start-source saved_state \
  --texture-strength 3.0 \
  --texture-interval original_interval \
  --texture-rescale 0 \
  --render-resolution 2048 \
  --metric-resolution 1024
```

The preserved seed-42 artifact is in `outputs/pixal3d_2048_texture_exp010_uniform_g3_step6`. Seeds 123/2024/3407/9999 are in `outputs/pixal3d_2048_texture_exp011_multiseed_g3_step6`.

## Results

Single-seed tuning result (seed 42):

| Metric | Best | Direct conditional-shape/global-texture control (EXP-004) | Delta | Original local-CFG baseline (EXP-001) | Delta |
|---|---:|---:|---:|---:|---:|
| PSNR | 17.6772044103 dB | 17.4693047768 dB | +0.207900 dB | 17.2824859944 dB | +0.394718 dB |
| SSIM | 0.6823697686 | 0.6694514751 | +0.012918 | 0.6670711040 | +0.015299 |
| LPIPS | 0.2149969339 | 0.2210101634 | -0.006013 | 0.2182299644 | -0.003233 |

Frozen five-seed validation (population mean ± std):

| Metric | Best configuration | Archived original local-CFG baseline | Paired mean delta | Seed wins |
|---|---:|---:|---:|---:|
| PSNR | 17.3802628185 ± 0.1625079399 dB | 17.1112277096 ± 0.2430351531 dB | +0.2690351089 dB | 4/5 |
| SSIM | 0.6770132661 ± 0.0034482318 | 0.6630744696 ± 0.0048925807 | +0.0139387965 | 5/5 |
| LPIPS | 0.2137809604 ± 0.0013252310 | 0.2185184568 ± 0.0011862797 | -0.0047374964 | 5/5 |

Per-seed PSNR: 42 `17.6772`, 123 `17.2900`, 2024 `17.1917`, 3407 `17.3719`, 9999 `17.3705`. Seed 9999 is the only PSNR loss versus its archived baseline (`-0.0566` dB), although its SSIM and LPIPS both improve.

## Evidence and cost

- All five seeds passed coordinate/order checks, 27/27 patch coverage, finite checks, and algebraic inverse checks; maximum inverse errors stayed below `7.16e-7`.
- Every texture conditional/unconditional pair used the same shape `concat_cond`; only the image condition changed.
- Every trace retained 13 global states and 12 Euler velocities. Texture uniform guidance was active only at steps 6–8.
- Active merged/global texture-velocity cosine ranges across five seeds: step 6 `0.960886–0.966403`, step 7 `0.954594–0.959816`, step 8 `0.950200–0.954925`.
- The archived baseline and new runs have identical preprocessed-input and metric-reference hashes and identical camera/evaluation settings.
- Added-seed pipeline time: 323.7–335.7 s; material bake: 360.3–400.3 s; total generation/bake: 720.2–754.6 s.
- Maximum observed CUDA memory: 36,350,530,560 allocated bytes and 46,554,677,248 reserved bytes in EXP-011. Seed 42 used 36,555,000,320 allocated bytes.
- Across five seeds: 58,218–59,995 latent tokens, 15,537,994–16,148,025 decoder vertices, 30,895,528–32,172,326 faces, and a 4096 texture bake.

## Why this configuration

Patch context alone was PSNR-neutral and worsened SSIM/LPIPS (EXP-005). Uniform texture CFG produced a monotonic seed-42 sweep from strengths 1.5 to 3.0 and outperformed matched Haar high-band guidance, showing that useful input-image signal exists in both low- and high-frequency texture velocity. Strength 3.0 was frozen before multi-seed evaluation; no parameters were changed after observing those four additional seeds.

## Known limitations

- Validation covers five seeds but only one source image. Cross-image generalization remains unmeasured.
- The configuration does not strictly improve PSNR on every seed; seed 9999 loses 0.0566 dB.
- Strength 3.0 still gained 0.0129 dB over 2.5 on seed 42, so the upper optimum was not fully bounded. The sweep stopped to avoid further single-seed overfitting.
- The five-seed comparison against the archived original baseline includes both the conditional-only shape change and texture guidance. The texture-specific causal control was run on seed 42 only (EXP-004 versus EXP-010).
- Full-resolution mesh baking dominates runtime and each retained experiment consumes roughly 1.6 GB per seed/configuration.

## Recommended next work

1. Validate the frozen strength-3.0 configuration on held-out source images before treating it as a general default.
2. On a held-out validation set, compare fixed strengths 2.0/2.5/3.0 or an adaptive cap based on merged/global velocity cosine to reduce the seed-9999-type PSNR regression.
3. Run paired direct conditional-shape/global-texture controls on any new image/seed set when a texture-only causal estimate is required.
4. Reuse shape/global trajectories and optimize the UV bake in future sweeps; model generation is no longer the dominant cost.

Full experimental history and exact commands are in `EXPERIMENT_LOG.md`; sortable rows are in `EXPERIMENT_RESULTS.csv`.
