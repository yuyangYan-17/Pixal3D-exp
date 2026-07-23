# Pixal3D 2048 Training-Free Texture Experiments

Evaluation protocol is fixed across the entries below: input `assets/choose/0_img.png`, seed 42, studio light, 2048 render, 1024 metric resolution, black full-frame reference, original camera estimation/render/metric code, GPU 4. Historical entries were reconstructed from their preserved `run_config.json`, `generation.json`, traces, and `metrics.csv`.

## EXP-001: Shape local original CFG baseline

- Date: 2026-07-22
- Git commit / code version: `cdbb2bb` plus the preserved dirty pre-texture implementation
- GPU: 4
- Goal / hypothesis: establish the original six-step local shape CFG control while leaving texture global and unchanged
- Relative change: baseline
- Full command: `CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 python pixal3d_directory_texture_eval.py --image-dir assets/choose --output-dir outputs/pixal3d_2048_original_cfg_regression --resolutions 2048 --seeds 42 --lights studio --patch-start-step 6 --patch-guidance-mode original_cfg --render-resolution 2048 --metric-resolution 1024 --fail-fast`
- Input: `assets/choose/0_img.png`
- Seeds: 42
- Experiment tag: `original_cfg_step6`
- Output directory: `outputs/pixal3d_2048_original_cfg_regression`
- Numerical checks: passed; algebraic inverse max absolute error `7.15e-7`; 13 states and 12 velocities saved
- OOM / exception: no
- PSNR: 17.2824859944 dB
- SSIM: 0.6670711040
- LPIPS: 0.2182299644
- Pipeline time: 249.162 s; total generation/bake time 696.872 s
- Texture/mesh scale: 2048 decode; 16,295,612 vertices; 32,490,382 faces; postprocess tensors 1,172,073,960 bytes
- Key diagnostics: step-6 merged/global velocity cosine 0.952092; relative L2 0.307592
- Delta vs baseline: 0 dB
- Interpretation/conclusion: valid shape-local baseline; texture used the original global conditional flow
- Next decision: test weaker shape guidance while preserving texture

## EXP-002: Shape Haar high-frequency CFG 2.0

- Date: 2026-07-22
- Git commit / code version: `cdbb2bb` plus the preserved dirty pre-texture implementation
- GPU: 4
- Goal / hypothesis: modest high-frequency shape guidance may improve view fidelity
- Relative change: shape mode only, original CFG to Haar low=conditional/high=2.0
- Full command: `CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 python pixal3d_directory_texture_eval.py --image-dir assets/choose --output-dir outputs/pixal3d_2048_haar_s2_smoke --resolutions 2048 --seeds 42 --lights studio --patch-start-step 6 --patch-guidance-mode wavelet_cfg --wavelet-high-strength 2 --wavelet-guidance-interval original_interval --wavelet-guidance-rescale off --render-resolution 2048 --metric-resolution 1024 --fail-fast`
- Input / seeds: `assets/choose/0_img.png`; 42
- Experiment tag: `haar_s2_original_interval_rescale_off_step6`
- Output directory: `outputs/pixal3d_2048_haar_s2_smoke`
- Numerical checks: passed; synthetic and real sparse Haar round trips passed; inverse max error `7.15e-7`
- OOM / exception: no
- PSNR / SSIM / LPIPS: 17.3507143631 / 0.6683423519 / 0.2190874517
- Pipeline time: 277.609 s; total generation/bake time 695.469 s
- Texture/mesh scale: 16,180,293 vertices; 32,247,550 faces; postprocess tensors 1,163,624,664 bytes
- Key diagnostics: step-6 cosine 0.947619; high-frequency RMS amplification 1.005003
- Delta vs EXP-001: +0.068228 dB PSNR, +0.001271 SSIM, +0.000857 LPIPS (worse)
- Interpretation/conclusion: gain is small and unlikely to come from meaningful high-frequency amplification
- Next decision: test high strength 1.0 / conditional-only

## EXP-003: Shape conditional-only control

- Date: 2026-07-22
- Git commit / code version: `cdbb2bb` plus the preserved dirty pre-texture implementation
- GPU: 4
- Goal / hypothesis: removing strong shape CFG may preserve input-view geometry better
- Relative change: EXP-002 high-band strength 2.0 to 1.0, making every band conditional
- Full command: `CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 python pixal3d_directory_texture_eval.py --image-dir assets/choose --output-dir outputs/pixal3d_2048_haar_s1_control --resolutions 2048 --seeds 42 --lights studio --patch-start-step 6 --patch-guidance-mode wavelet_cfg --wavelet-high-strength 1 --wavelet-guidance-interval original_interval --wavelet-guidance-rescale off --render-resolution 2048 --metric-resolution 1024 --fail-fast`
- Input / seeds: `assets/choose/0_img.png`; 42
- Experiment tag: `haar_s1_original_interval_rescale_off_step6`
- Output directory: `outputs/pixal3d_2048_haar_s1_control`
- Numerical checks: passed; real sparse Haar round trip and inverse checks passed
- OOM / exception: no
- PSNR / SSIM / LPIPS: 17.3658843008 / 0.6691375375 / 0.2190725952
- Pipeline time: 276.411 s; total generation/bake time 679.656 s
- Texture/mesh scale: 16,148,484 vertices; 32,173,516 faces; postprocess tensors 1,161,209,424 bytes
- Key diagnostics: step-6 cosine 0.938684; high-frequency RMS amplification exactly 1.0
- Delta vs EXP-001: +0.083398 dB PSNR, +0.002066 SSIM, +0.000843 LPIPS (worse)
- Interpretation/conclusion: current single-seed best; benefit is cancellation of strong shape CFG, not frequency enhancement
- Next decision: hold this shape configuration fixed and move the intervention to texture flow

## EXP-004: Direct shape conditional-only + global texture regression

- Date: 2026-07-23
- Git commit / code version: `cdbb2bb` dirty working tree; pipeline SHA-256 `810da4829ebceccc`; evaluator SHA-256 `ac05aa42283846`
- GPU: 4 (A800 80 GB)
- Goal / hypothesis: verify the new texture trajectory recorder with local texture disabled and establish a direct conditional-only shape baseline
- Relative change: replace the prior Haar-strength-1 proxy with a direct one-pass conditional shape prediction; record but do not alter the original global texture flow
- Full command: `CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 python pixal3d_directory_texture_eval.py --image-dir assets/choose --output-dir outputs/pixal3d_2048_texture_exp004_global_regression --resolutions 2048 --seeds 42 --lights studio --shape-mode conditional_only --shape-start-step 6 --shape-start-source saved_state --texture-mode global_original --texture-start-step 6 --texture-start-source saved_state --render-resolution 2048 --metric-resolution 1024 --fail-fast`
- Input / seeds: `assets/choose/0_img.png`; 42
- Experiment tag: `s-cond-saved-step6-g1-original_interval-r0-whaar__t-global-saved-step6-g1-original_interval-r0-whaar`
- Output directory: `outputs/pixal3d_2048_texture_exp004_global_regression`
- Numerical checks: passed; both traces contain 13 states and 12 velocities; shape inverse max/mean error `7.15e-7` / `2.11e-8`; texture latent/image projections/negative projections/shape concat condition all have 59,223 aligned tokens; conditional and unconditional texture branches share the shape condition by construction
- Protocol checks: preprocessed input and metric-reference SHA-256 hashes exactly match EXP-003; camera parameters match exactly
- OOM / exception: no
- PSNR: 17.4693047768 dB
- SSIM: 0.6694514751
- LPIPS: 0.2210101634
- Pipeline time: 223.421 s; material bake 414.854 s; total generation/bake 678.944 s
- Peak CUDA memory: 36,554,999,808 allocated bytes; 45,799,702,528 reserved bytes
- Texture/mesh scale: shape latent 59,223 x 32; texture latent 59,223 x 32; 16,148,025 vertices; 32,172,326 faces; 4096 texture bake
- Key diagnostics: shape step-6 merged/global cosine 0.938684 and relative L2 0.344962; texture trace status `global_only_complete`; shape trace 243,542,980 bytes; texture trace 190,471,330 bytes
- Delta vs EXP-001 original local CFG: +0.186819 dB PSNR, +0.002380 SSIM, +0.002780 LPIPS (worse)
- Delta vs EXP-003 Haar-strength-1 proxy: +0.103420 dB PSNR, +0.000314 SSIM, +0.001938 LPIPS (worse)
- Interpretation: the improvement is attributable to direct conditional-only shape execution versus the previous Haar proxy; texture is still the untouched global baseline. The texture recording path itself completed without changing coordinates or conditions.
- Current conclusion: use EXP-004 as the causal baseline for texture-local experiments
- Next decision: compare texture local original/conditional-only from step 6 while holding every other setting fixed

## EXP-005: Texture local original/conditional-only control

- Date: 2026-07-23
- Git commit / code version: `cdbb2bb` dirty working tree; pipeline SHA-256 `810da4829ebceccc`; evaluator SHA-256 `ac05aa42283846`
- GPU: 4 (A800 80 GB)
- Goal / hypothesis: determine whether local 64-cube texture context and overlap fusion improve fidelity without any guidance amplification
- Relative change: EXP-004 global texture output replaced by texture local patch flow from step 6; pretrained texture strength remains 1.0, so original CFG is exactly conditional-only
- Full command: `CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 python pixal3d_directory_texture_eval.py --image-dir assets/choose --output-dir outputs/pixal3d_2048_texture_exp005_local_original_step6 --resolutions 2048 --seeds 42 --lights studio --shape-mode conditional_only --shape-start-step 6 --shape-start-source saved_state --texture-mode original_cfg --texture-start-step 6 --texture-start-source saved_state --render-resolution 2048 --metric-resolution 1024 --fail-fast`
- Input / seeds / tag: `assets/choose/0_img.png`; 42; `s-cond-saved-step6-g1-original_interval-r0-whaar__t-original-saved-step6-g1-original_interval-r0-whaar`
- Output directory: `outputs/pixal3d_2048_texture_exp005_local_original_step6`
- Numerical checks: passed; 27/27 patches active; all 59,223 texture/positive image/negative image/shape-condition tokens aligned; no uncovered tokens; inverse max/mean error `7.15e-7` / `1.92e-8`; 13 global states and 12 global velocities retained
- OOM / exception: no
- PSNR / SSIM / LPIPS: 17.4697072554 / 0.6674870849 / 0.2235724032
- Pipeline time: 274.682 s; material bake 402.914 s; total generation/bake 710.755 s
- Peak CUDA memory: 36,555,000,320 allocated bytes; 46,261,075,968 reserved bytes
- Texture/mesh scale: shape and texture latents 59,223 x 32; 16,148,025 vertices; 32,172,326 faces; topology exactly matches EXP-004
- Key diagnostics: texture merged/global cosine by step `[0.995934, 0.995194, 0.993831, 0.991661, 0.988698, 0.986383]`; relative L2 `[0.090856, 0.098692, 0.111563, 0.129409, 0.150313, 0.164797]`
- Delta vs EXP-004 global texture: +0.000402 dB PSNR, -0.001964 SSIM, +0.002562 LPIPS (worse)
- Delta vs EXP-001 original local shape baseline: +0.187221 dB PSNR
- Interpretation: local context/fusion alone is PSNR-neutral and worsens both perceptual metrics; its high velocity similarity confirms the implementation is a conservative perturbation
- Current conclusion: do not treat local conditional texture as an improvement
- Next decision: keep the same local flow and test weak uniform image CFG strength 1.5 only in the original texture interval

## EXP-006: Texture local uniform CFG 1.5

- Date: 2026-07-23
- Git commit / code version: `cdbb2bb` dirty working tree; pipeline SHA-256 `810da4829ebceccc`; evaluator SHA-256 `ac05aa42283846`
- GPU: 4 (A800 80 GB)
- Goal / hypothesis: weak image CFG during the original texture interval may improve input-view color/texture fidelity
- Relative change: EXP-005 texture mode conditional/original strength 1.0 to uniform CFG strength 1.5; only steps 6, 7, and 8 are guided
- Full command: `CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 python pixal3d_directory_texture_eval.py --image-dir assets/choose --output-dir outputs/pixal3d_2048_texture_exp006_uniform_g1p5_step6 --resolutions 2048 --seeds 42 --lights studio --shape-mode conditional_only --shape-start-step 6 --shape-start-source saved_state --texture-mode uniform_cfg --texture-start-step 6 --texture-start-source saved_state --texture-strength 1.5 --texture-interval original_interval --texture-rescale 0 --render-resolution 2048 --metric-resolution 1024 --fail-fast`
- Input / seeds / tag: `assets/choose/0_img.png`; 42; `s-cond-saved-step6-g1-original_interval-r0-whaar__t-uniform-saved-step6-g1p5-original_interval-r0-whaar`
- Output directory: `outputs/pixal3d_2048_texture_exp006_uniform_g1p5_step6`
- Numerical checks: passed; 27/27 patches, all 59,223 tokens aligned/covered, finite latents, inverse error within `7.15e-7`, 13 states/12 velocities retained
- OOM / exception: no
- PSNR / SSIM / LPIPS: 17.5719230321 / 0.6728637218 / 0.2202381045
- Pipeline time: 333.240 s; material bake 407.602 s; total generation/bake 773.914 s
- Peak CUDA memory: 36,555,000,320 allocated bytes; 45,799,702,528 reserved bytes
- Texture/mesh scale: shape/texture latents 59,223 x 32; 16,148,025 vertices; 32,172,326 faces; geometry unchanged
- Key diagnostics: active-step merged/global cosine `[0.993046, 0.991693, 0.989402]`; active-step merged/plain-local cosine `[0.997830, 0.997925, 0.998172]`; later steps correctly report guidance inactive
- Delta vs EXP-004 global texture: +0.102618 dB PSNR, +0.003412 SSIM, -0.000772 LPIPS (better)
- Delta vs EXP-005 local conditional: +0.102216 dB PSNR, +0.005377 SSIM, -0.003334 LPIPS (better)
- Delta vs EXP-001 original local shape baseline: +0.289437 dB PSNR, +0.005793 SSIM, +0.002008 LPIPS (worse)
- Interpretation: weak texture CFG yields a real improvement across all three metrics relative to the direct global-texture baseline; the gain is much larger than patch context alone
- Current conclusion: best completed configuration so far; texture guidance is the promising lever
- Next decision: compare matched Haar high-band strength 1.5 to identify whether the gain is frequency-specific

## EXP-007: Texture Haar high-band CFG 1.5

- Date: 2026-07-23
- Git commit / code version: `cdbb2bb` dirty working tree; pipeline SHA-256 `810da4829ebceccc`; evaluator SHA-256 `ac05aa42283846`
- GPU: 4 (A800 80 GB)
- Goal / hypothesis: retain conditional low frequencies while guiding only high-frequency texture velocity
- Relative change: EXP-006 uniform strength 1.5 replaced with Haar low=conditional/high=1.5; interval/start/rescale unchanged
- Full command: `CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 python pixal3d_directory_texture_eval.py --image-dir assets/choose --output-dir outputs/pixal3d_2048_texture_exp007_haar_g1p5_step6 --resolutions 2048 --seeds 42 --lights studio --shape-mode conditional_only --shape-start-step 6 --shape-start-source saved_state --texture-mode wavelet_cfg --texture-start-step 6 --texture-start-source saved_state --texture-strength 1.5 --texture-interval original_interval --texture-rescale 0 --render-resolution 2048 --metric-resolution 1024 --fail-fast`
- Input / seeds / tag: `assets/choose/0_img.png`; 42; `s-cond-saved-step6-g1-original_interval-r0-whaar__t-haar-saved-step6-g1p5-original_interval-r0-whaar`
- Output directory: `outputs/pixal3d_2048_texture_exp007_haar_g1p5_step6`
- Numerical checks: passed; synthetic Haar round trip `2.38e-7`; uniform-band CFG equivalence `7.15e-7`; real sparse patch round trip `1.43e-6` (<`1e-5`); inverse/alignment/coverage/finite checks passed
- OOM / exception: no
- PSNR / SSIM / LPIPS: 17.5522768251 / 0.6720808148 / 0.2206130773
- Pipeline time: 334.177 s; material bake 371.990 s; total generation/bake 739.182 s
- Peak CUDA memory: 36,555,000,320 allocated bytes; 45,799,702,528 reserved bytes
- Texture/mesh scale: 59,223 x 32 shape and texture latents; 16,148,025 vertices; 32,172,326 faces; geometry unchanged
- Key diagnostics: active-step high-frequency RMS amplification `[1.010373, 1.011253, 1.011406]`; merged/global cosine `[0.993806, 0.992594, 0.990493]`
- Delta vs EXP-004 global texture: +0.082972 dB PSNR, +0.002629 SSIM, -0.000397 LPIPS (better)
- Delta vs EXP-005 local conditional: +0.082570 dB PSNR, +0.004594 SSIM, -0.002959 LPIPS (better)
- Delta vs EXP-006 matched uniform 1.5: -0.019646 dB PSNR, -0.000783 SSIM, +0.000375 LPIPS (worse)
- Delta vs EXP-001 original local shape baseline: +0.269791 dB PSNR
- Interpretation: high-band guidance is beneficial, but the matched uniform mode is better on every metric; conditional low frequencies discard part of the useful image-guidance signal
- Current conclusion: prioritize uniform texture CFG; retain Haar as a verified optional path, not the current best
- Next decision: test uniform strength 2.0 with the same interval and start step

## EXP-008: Texture local uniform CFG 2.0

- Date: 2026-07-23
- Git commit / code version: `cdbb2bb` dirty working tree; pipeline SHA-256 `810da4829ebceccc`; evaluator SHA-256 `ac05aa42283846`
- GPU: 4 (A800 80 GB)
- Goal / hypothesis: determine whether the uniform texture-guidance gain continues above strength 1.5
- Relative change: EXP-006 texture strength 1.5 to 2.0; shape, start step, interval, patch geometry, rescale, render, and metric protocol unchanged
- Full command: `CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 python pixal3d_directory_texture_eval.py --image-dir assets/choose --output-dir outputs/pixal3d_2048_texture_exp008_uniform_g2_step6 --resolutions 2048 --seeds 42 --lights studio --shape-mode conditional_only --shape-start-step 6 --shape-start-source saved_state --texture-mode uniform_cfg --texture-start-step 6 --texture-start-source saved_state --texture-strength 2.0 --texture-interval original_interval --texture-rescale 0 --render-resolution 2048 --metric-resolution 1024 --fail-fast`
- Input / seeds / tag: `assets/choose/0_img.png`; 42; `s-cond-saved-step6-g1-original_interval-r0-whaar__t-uniform-saved-step6-g2-original_interval-r0-whaar`
- Output directory: `outputs/pixal3d_2048_texture_exp008_uniform_g2_step6`
- Numerical checks: passed; 27/27 patches, all 59,223 tokens aligned/covered, conditional and unconditional branches share the sliced shape condition, finite latents, inverse max/mean error `7.15e-7` / `1.92e-8`, and 13 states/12 velocities retained
- OOM / exception: no
- PSNR / SSIM / LPIPS: 17.6319890655 / 0.6769221425 / 0.2177441120
- Pipeline time: 333.868 s; material bake 388.354 s; total generation/bake 755.556 s
- Peak CUDA memory: 36,555,000,320 allocated bytes; 45,799,702,528 reserved bytes
- Texture/mesh scale: shape/texture latents 59,223 x 32; 16,148,025 vertices; 32,172,326 faces; geometry unchanged; postprocess tensors 1,161,173,112 bytes
- Key diagnostics: active-step merged/global cosine `[0.985950, 0.983293, 0.979746]`; active-step merged/plain-local cosine `[0.991360, 0.991723, 0.992899]`; later steps correctly report guidance inactive
- Delta vs EXP-004 global texture: +0.162684 dB PSNR, +0.007471 SSIM, -0.003266 LPIPS (better)
- Delta vs EXP-006 uniform strength 1.5: +0.060066 dB PSNR, +0.004058 SSIM, -0.002494 LPIPS (better)
- Delta vs EXP-001 original local shape baseline: +0.349503 dB PSNR, +0.009851 SSIM, -0.000486 LPIPS (better)
- Interpretation: raising uniform texture CFG from 1.5 to 2.0 improves every metric, including crossing below the original baseline's LPIPS while preserving exactly the same geometry
- Current conclusion: new best configuration and the first result that dominates the original baseline on PSNR, SSIM, and LPIPS
- Next decision: probe uniform strength 2.5 at the same start and interval to locate the guidance-strength optimum before multi-seed validation

## EXP-009: Texture local uniform CFG 2.5

- Date: 2026-07-23
- Git commit / code version: `cdbb2bb` dirty working tree; pipeline SHA-256 `810da4829ebceccc`; evaluator SHA-256 `ac05aa42283846`
- GPU: 4 (A800 80 GB)
- Goal / hypothesis: continue the monotonic strength sweep and test whether 2.5 improves fidelity without oversteering the local texture velocity
- Relative change: EXP-008 texture strength 2.0 to 2.5; every other generation and evaluation parameter fixed
- Full command: `CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 python pixal3d_directory_texture_eval.py --image-dir assets/choose --output-dir outputs/pixal3d_2048_texture_exp009_uniform_g2p5_step6 --resolutions 2048 --seeds 42 --lights studio --shape-mode conditional_only --shape-start-step 6 --shape-start-source saved_state --texture-mode uniform_cfg --texture-start-step 6 --texture-start-source saved_state --texture-strength 2.5 --texture-interval original_interval --texture-rescale 0 --render-resolution 2048 --metric-resolution 1024 --fail-fast`
- Input / seeds / tag: `assets/choose/0_img.png`; 42; `s-cond-saved-step6-g1-original_interval-r0-whaar__t-uniform-saved-step6-g2p5-original_interval-r0-whaar`
- Output directory: `outputs/pixal3d_2048_texture_exp009_uniform_g2p5_step6`
- Numerical checks: passed; 27/27 patches, 59,223 aligned/covered tokens, shared shape condition across texture branches, finite outputs, inverse max/mean error `7.15e-7` / `1.92e-8`, and 13 states/12 velocities retained
- OOM / exception: no
- PSNR / SSIM / LPIPS: 17.6643472390 / 0.6800096631 / 0.2162539065
- Pipeline time: 333.468 s; material bake 371.831 s; total generation/bake 739.439 s
- Peak CUDA memory: 36,555,000,320 allocated bytes; 45,799,702,528 reserved bytes
- Texture/mesh scale: shape/texture latents 59,223 x 32; 16,148,025 vertices; 32,172,326 faces; geometry unchanged; postprocess tensors 1,161,173,112 bytes
- Key diagnostics: active-step merged/global cosine `[0.975062, 0.970639, 0.966283]`; active-step merged/plain-local cosine `[0.981057, 0.981961, 0.984960]`; guidance inactive after step 8
- Delta vs EXP-004 global texture: +0.195042 dB PSNR, +0.010558 SSIM, -0.004756 LPIPS (better)
- Delta vs EXP-008 uniform strength 2.0: +0.032358 dB PSNR, +0.003088 SSIM, -0.001490 LPIPS (better)
- Delta vs EXP-001 original local shape baseline: +0.381861 dB PSNR, +0.012939 SSIM, -0.001976 LPIPS (better)
- Interpretation: strength 2.5 still improves all three metrics and preserves identical geometry, although the incremental PSNR gain is diminishing relative to the 1.5-to-2.0 step
- Current conclusion: new single-seed best; the optimum has not yet been bounded above
- Next decision: run one final uniform strength 3.0 probe, then freeze the better of 2.5/3.0 for multi-seed validation

## EXP-010: Texture local uniform CFG 3.0

- Date: 2026-07-23
- Git commit / code version: `cdbb2bb` dirty working tree; pipeline SHA-256 `810da4829ebceccc`; evaluator SHA-256 `ac05aa42283846`
- GPU: 4 (A800 80 GB)
- Goal / hypothesis: provide an upper sweep point and decide the fixed configuration for multi-seed validation
- Relative change: EXP-009 texture strength 2.5 to 3.0; all other generation/evaluation settings unchanged
- Full command: `CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 python pixal3d_directory_texture_eval.py --image-dir assets/choose --output-dir outputs/pixal3d_2048_texture_exp010_uniform_g3_step6 --resolutions 2048 --seeds 42 --lights studio --shape-mode conditional_only --shape-start-step 6 --shape-start-source saved_state --texture-mode uniform_cfg --texture-start-step 6 --texture-start-source saved_state --texture-strength 3.0 --texture-interval original_interval --texture-rescale 0 --render-resolution 2048 --metric-resolution 1024 --fail-fast`
- Input / seeds / tag: `assets/choose/0_img.png`; 42; `s-cond-saved-step6-g1-original_interval-r0-whaar__t-uniform-saved-step6-g3-original_interval-r0-whaar`
- Output directory: `outputs/pixal3d_2048_texture_exp010_uniform_g3_step6`
- Numerical checks: passed; 27/27 patches, all 59,223 tokens aligned/covered, shared texture shape condition, finite decoded attributes, inverse max/mean error `7.15e-7` / `1.92e-8`, and 13 states/12 velocities retained
- OOM / exception: no
- PSNR / SSIM / LPIPS: 17.6772044103 / 0.6823697686 / 0.2149969339
- Pipeline time: 333.581 s; material bake 394.941 s; total generation/bake 761.837 s
- Peak CUDA memory: 36,555,000,320 allocated bytes; 45,799,702,528 reserved bytes
- Texture/mesh scale: shape/texture latents 59,223 x 32; 16,148,025 vertices; 32,172,326 faces; geometry unchanged; postprocess tensors 1,161,173,112 bytes
- Key diagnostics: active-step merged/global cosine `[0.960886, 0.954594, 0.950200]`; active-step merged/plain-local cosine `[0.967423, 0.969338, 0.974954]`; guidance inactive after step 8
- Delta vs EXP-004 global texture: +0.207900 dB PSNR, +0.012918 SSIM, -0.006013 LPIPS (better)
- Delta vs EXP-009 uniform strength 2.5: +0.012857 dB PSNR, +0.002360 SSIM, -0.001257 LPIPS (better)
- Delta vs EXP-001 original local shape baseline: +0.394718 dB PSNR, +0.015299 SSIM, -0.003233 LPIPS (better)
- Interpretation: 3.0 is the best tested strength on every metric, but its incremental PSNR advantage over 2.5 is only 0.0129 dB, indicating PSNR saturation even as SSIM/LPIPS still improve
- Current conclusion: freeze strength 3.0 as the multi-seed candidate; do not spend more single-seed trials extending the strength sweep
- Next decision: evaluate the frozen configuration on seeds 123, 2024, 3407, and 9999, then combine those rows with seed 42 for a five-seed mean/std and paired original-baseline deltas

## EXP-011: Frozen texture uniform CFG 3.0 multi-seed validation

- Date: 2026-07-23
- Git commit / code version: `cdbb2bb` dirty working tree; pipeline SHA-256 `810da4829ebceccc`; evaluator SHA-256 `ac05aa42283846`
- GPU: 4 (A800 80 GB)
- Goal / hypothesis: test whether the seed-42 winner is a stable improvement rather than a single-seed tuning artifact
- Relative change: no parameter tuning; freeze EXP-010 exactly and add seeds 123, 2024, 3407, and 9999. Combine them with the already completed seed-42 EXP-010 row for five-seed statistics.
- Full command: `CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 python pixal3d_directory_texture_eval.py --image-dir assets/choose --output-dir outputs/pixal3d_2048_texture_exp011_multiseed_g3_step6 --resolutions 2048 --seeds 123 2024 3407 9999 --lights studio --shape-mode conditional_only --shape-start-step 6 --shape-start-source saved_state --texture-mode uniform_cfg --texture-start-step 6 --texture-start-source saved_state --texture-strength 3.0 --texture-interval original_interval --texture-rescale 0 --render-resolution 2048 --metric-resolution 1024`
- Input / seeds / tag: `assets/choose/0_img.png`; 42, 123, 2024, 3407, 9999; `s-cond-saved-step6-g1-original_interval-r0-whaar__t-uniform-saved-step6-g3-original_interval-r0-whaar`
- Output directories: seed 42 in `outputs/pixal3d_2048_texture_exp010_uniform_g3_step6`; added seeds in `outputs/pixal3d_2048_texture_exp011_multiseed_g3_step6`
- Numerical checks: passed for every seed; 27/27 patches; 58,218 to 59,995 aligned/covered texture, image-positive, image-negative, and shape-condition tokens; shape inverse max error `4.77e-7` to `7.15e-7`; texture inverse max error `5.36e-7` to `7.15e-7`; finite decoded attributes; each trace has 13 states and 12 velocities
- Protocol checks: archived baseline and new runs use identical preprocessed-input SHA-256 `69beebef07637e1d...`, metric-reference SHA-256 `6f73a5f01d2cc6a...`, camera FOV/distance, 2048 render, 1024 metric resolution, studio light, and metric implementation. Every saved per-seed original PNG is byte-identical to the fixed metric reference.
- OOM / exception: no; four requested renders succeeded and none failed

| Seed | PSNR | SSIM | LPIPS | PSNR delta vs archived original local CFG | SSIM delta | LPIPS delta |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 17.6772044103 | 0.6823697686 | 0.2149969339 | +0.394716 | +0.015298 | -0.003235 |
| 123 | 17.2899820855 | 0.6763852239 | 0.2125864327 | +0.579251 | +0.019973 | -0.007538 |
| 2024 | 17.1917041478 | 0.6715292931 | 0.2151205987 | +0.156085 | +0.013167 | -0.004213 |
| 3407 | 17.3719244775 | 0.6773381233 | 0.2118444741 | +0.271707 | +0.012801 | -0.004760 |
| 9999 | 17.3704989712 | 0.6774439216 | 0.2143563628 | -0.056584 | +0.008455 | -0.003940 |

- Five-seed result (population mean ± std): PSNR `17.3802628185 ± 0.1625079399`; SSIM `0.6770132661 ± 0.0034482318`; LPIPS `0.2137809604 ± 0.0013252310`
- Archived original-local-CFG baseline (same five seeds, population mean ± std): PSNR `17.1112277096 ± 0.2430351531`; SSIM `0.6630744696 ± 0.0048925807`; LPIPS `0.2185184568 ± 0.0011862797`
- Paired delta mean ± population std: PSNR `+0.2690351089 ± 0.2148592730` dB; SSIM `+0.0139387965 ± 0.0037476919`; LPIPS `-0.0047374964 ± 0.0014839083` (better)
- Win counts versus archived original local CFG: PSNR 4/5; SSIM 5/5; LPIPS 5/5
- Pipeline time across added seeds: mean 329.848 s, range 323.719 to 335.697 s; material bake mean 373.435 s, range 360.264 to 400.312 s; total generation/bake mean 731.991 s, range 720.200 to 754.616 s
- Peak CUDA memory across added seeds: maximum 36,350,530,560 allocated bytes; 46,554,677,248 reserved bytes
- Texture/mesh scale: 32-channel shape and texture latents; 58,218 to 59,995 tokens; 15,537,994 to 16,148,025 decoder vertices and 30,895,528 to 32,172,326 faces across all five seeds; 4096 texture bake
- Key diagnostics: active step-6 merged/global texture cosine spans `0.960886` to `0.966403`; step 7 spans `0.954594` to `0.959816`; step 8 spans `0.950200` to `0.954925`. Guidance is inactive after step 8 on every seed.
- Visual sanity check: seed 42 and seed 9999 show coherent textures without patch seams, wavelet ringing, or texture collapse. A viewer anomaly initially made one comparison panel appear black, but SHA-256 and pixel-exact crop checks proved the saved reference/comparison artifacts are correct.
- Interpretation: the frozen joint configuration provides a stable average gain, reduces PSNR variance, and improves both perceptual metrics for every seed. It does not strictly dominate PSNR per seed: seed 9999 regresses by 0.0566 dB, which is the main robustness limitation.
- Current conclusion: the task's stable-improvement stopping condition is met. Uniform image CFG in the texture latent is the useful mechanism; matched Haar guidance is weaker, while patch context alone is neutral.
- Next decision: finalize `BEST_CONFIG.md` with the five-seed evidence. A future robustness study should tune or adapt texture strength on held-out inputs/seeds rather than extending the present seed-42 strength sweep.

## EXP-012: High-resolution image-tile texture velocity flow (pending user run)

- Date: 2026-07-23
- Git commit / code version: base commit `2898139c04f92aceef34199f0dd12b49d92c395e`; dirty working tree backed up in `HR_IMAGE_TILE_BACKUP.md`
- GPU: intended GPU 4; not run by Codex
- Goal / hypothesis: preserve the original global 1024 texture trajectory for steps 0–5, then use independently extracted HR image-tile DINOv3+NAF conditions for synchronized per-step texture velocities during steps 6–11
- Relative change: texture-only optional path; shape flow, model weights, texture decoder, camera, renderer, and metrics remain unchanged
- Full command: `CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 python pixal3d_directory_texture_eval.py --image-dir assets/choose --output-dir outputs/pixal3d_hr_image_tile_texture_flow_step6 --resolutions 2048 --seeds 42 --lights studio --texture-mode global_original --hr-image-tile-texture-flow --hr-image-tile-size 1024 --hr-image-tile-stride 1024 --hr-image-tile-start-step 6 --hr-image-tile-min-foreground-ratio 0 --hr-image-tile-fallback saved_global --hr-image-tile-weight tent --hr-image-tile-save-debug --render-resolution 2048 --metric-resolution 1024 --low-vram --fail-fast`
- Input / seeds / tag: resolved by the user run; the tag gains an `hrtile` suffix containing tile size, stride, start step, foreground threshold, fallback, and weight
- Output directory: `outputs/pixal3d_hr_image_tile_texture_flow_step6`
- Numerical checks: Python syntax, import, synthetic projection/token assignment, global→HR→tile round trip, scatter/token-order, overlap-weighted velocity merge, and disabled-config regression passed
- OOM / exception: pending user run
- PSNR / SSIM / LPIPS: pending user run; no metrics fabricated
- Pipeline time / peak CUDA memory: pending user run
- Texture/mesh scale: pending user run
- Key diagnostics to inspect: active tile/token counts, covered/overlap/uncovered ratios, per-tile velocity norms, merged-vs-global velocity cosine/MSE/relative-L2/norm ratio, fallback use, DINOv3/NAF per-tile flags, step runtime, and CUDA peaks
- Current conclusion: implementation complete; full model generation intentionally not run
- Next decision: user runs the command above, checks `texture_flow_2048_trace.pt` and `hr_image_tile_debug/`, then records real metrics or failures here

## EXP-013: Canonical preprocessing + clean global control

### Goal

Isolate the effect of the single canonical high-quality preprocessing pyramid. No multi-tile texture condition or second texture pass is enabled. Shape `start_step=12` also supplies a real saved-state identity check.

### Code

- Date: 2026-07-24
- Base commit: `cfe73ac`, dirty working tree
- Pipeline / sparse project attention / structured flow / evaluator SHA-256 prefixes: `ffb4e9a7a526`, `54059f0c07fc`, `e8dd94ed23b9`, `31314f102f41`
- Main modified files: `pixal3d/pipelines/pixal3d_image_to_3d.py`, `pixal3d/modules/sparse/attention/proj_attention.py`, `pixal3d/models/structured_latent_flow.py`, `pixal3d_directory_texture_eval.py`

### Configuration

```bash
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
python pixal3d_directory_texture_eval.py \
  --image-dir assets/choose \
  --output-dir outputs/pixal3d_2048_canonical_global_control \
  --resolutions 2048 \
  --seeds 42 \
  --lights studio \
  --shape-start-step 12 \
  --texture-mode global_original \
  --no-texture-multitile-3d-patch-flow \
  --render-resolution 2048 \
  --metric-resolution 1024 \
  --fail-fast
```

### Preprocessing

- Source: `assets/choose/0_img.png`
- Alpha source: existing RGBA alpha; rembg calls: 0
- Source foreground bbox: `(115, 414, 3897, 3867)`
- Source square extent: `(-75, 60, 4086, 4221)`
- Padding `(left,right,top,bottom)`: `(75, 0, 0, 125)`
- Canonical outputs: exactly 4096, 1024, and 512; 1024/512 directly resized from 4096

### 2D Condition Diagnostics

Not applicable: the multi-tile flag was disabled.

### 3D Flow Diagnostics

- Shape fixed grid: 128; tokens: 59,234
- Shape identity: `states[12]` restored, zero model steps, max/mean algebraic identity error `0 / 0`
- Texture: unchanged 12-step global trajectory, 13 saved states and 12 velocities

### Metrics

- PSNR: `17.3863170053`
- SSIM: `0.6702985168`
- LPIPS: `0.2171153277`
- Pipeline: `158.142 s`
- Peak CUDA allocated/reserved: `36,414,627,328 / 45,925,531,648` bytes
- Geometry: 59,234 latent tokens; 16,020,838 decoder vertices; 31,974,354 faces

### Visual Findings

The rendered turtle is coherent, with stable global color and no patch seams because no texture patch pass ran. The saved control comparison panel displayed a viewer-side black reference half, but metrics used `metric_reference_rgb.png`; the rendered half itself is valid.

### Conclusion

Canonical preprocessing improves over the task's historical clean global baseline (`17.1593 / 0.666374 / 0.218942`) by `+0.2270 dB / +0.003925 / -0.001827`. This is the correct same-preprocessing control for EXP-014.

### Next Action

Run paired multi-tile 3D patch flow from texture step 6 with every other setting fixed.

## EXP-014: Paired multi-tile block fusion + 3D patch texture flow, step 6

### Goal

Test whether independently extracted overlapping 4K tile global/proj pairs recover local texture while preserving full 64³ 3D patch self-attention and strict no-fallback coverage.

### Code

Same working tree as EXP-013. The first attempt correctly extracted all conditions but stopped before its first patch prediction because model-dtype conversion changed float32 normalized weights to fp16. The implementation was corrected to preserve float32 membership weights through validation/accumulation and cast only attention contributions; the combined suite then passed 22 tests before this successful retry.

### Configuration

```bash
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
python pixal3d_directory_texture_eval.py \
  --image-dir assets/choose \
  --output-dir outputs/pixal3d_2048_multitile_paired_3dpatch_step6 \
  --resolutions 2048 \
  --seeds 42 \
  --lights studio \
  --shape-start-step 12 \
  --texture-mode global_original \
  --texture-multitile-3d-patch-flow \
  --texture-multitile-start-step 6 \
  --texture-canonical-image-size 4096 \
  --texture-image-tile-size 1024 \
  --texture-image-tile-stride 512 \
  --texture-3d-patch-size 64 \
  --texture-3d-patch-stride 32 \
  --texture-multitile-global-mode paired_block_fusion \
  --texture-multitile-save-debug \
  --render-resolution 2048 \
  --metric-resolution 1024 \
  --fail-fast
```

### Preprocessing

Identical to EXP-013: existing RGBA alpha, rembg calls 0, source bbox `(115,414,3897,3867)`, square `(-75,60,4086,4221)`, padding `(75,0,0,125)`, canonical 4096/1024/512.

### 2D Condition Diagnostics

- Fixed layout: 49 tiles, 1024 size, stride 512; 48 tiles had token membership
- Tokens: 59,234
- Membership min/max: 1 / 4
- Membership histogram: `{1: 1, 2: 4,694, 4: 54,539}`
- Weight-sum min/max: `0.99999988 / 1.00000012`
- Every active crop independently ran DINO and NAF; extraction was about `0.57–0.62 s/tile`
- Global bank: `[48,5,1024]`; fused proj: `[59234,2048]`
- Debug artifacts include global bank, IDs, weights, fused proj, and optional raw slot proj

### 3D Flow Diagnostics

- Fixed 128 grid, 64³ patches, stride 32, 27 patches
- Patch coverage min/max: 2 / 8; histogram `{2: 8,025, 4: 33,720, 8: 17,489}`
- Local coordinate range: `[0,63]`
- Every step read one immutable `x_step_start`, merged all patch velocities, and made one global Euler update
- No global velocity fallback exists or ran

| Step | cosine vs global | relative L2 | MSE | seconds |
|---:|---:|---:|---:|---:|
| 6 | 0.99079573 | 0.13578466 | 0.02404319 | 31.077 |
| 7 | 0.98944414 | 0.14532241 | 0.02605185 | 30.896 |
| 8 | 0.98723418 | 0.15963548 | 0.02955591 | 30.881 |
| 9 | 0.98378837 | 0.17965209 | 0.03439710 | 31.004 |
| 10 | 0.97924823 | 0.20298506 | 0.03934061 | 30.996 |
| 11 | 0.97582603 | 0.21897635 | 0.03950640 | 30.991 |

### Metrics

- PSNR: `17.0416468268`
- SSIM: `0.6577093005`
- LPIPS: `0.2198133171`
- Pipeline: `375.663 s`
- Peak CUDA allocated/reserved: `36,414,627,840 / 45,946,503,168` bytes
- Material bake: `372.266 s`
- Geometry is identical to EXP-013: 59,234 tokens, 16,020,838 vertices, 31,974,354 faces

### Visual Findings

The result is coherent with no obvious axis-aligned 2D tile seam, texture collapse, or geometry change. The head eye/red/blue markings and shell remain recognizable, but the render does not show a convincing fine-detail recovery relative to the canonical control; global shell coloring is slightly shifted and perceptual metrics regress.

### Conclusion

Against the canonical global control, paired step 6 changes PSNR/SSIM/LPIPS by `-0.344670 dB / -0.012589 / +0.002698` (worse). Against the historical legacy 2D image-tile step-6 result (`16.9282 / 0.653563 / 0.225339`), it is `+0.113447 dB / +0.004146 / -0.005526` (better), but it does not beat clean global. The structural defects of the old token-subset/fallback path are removed; the remaining issue is likely inference-time condition distribution shift or applying local globals too early.

### Next Action

Run the conservative paired experiment at `--texture-multitile-start-step 10`. This directly tests whether tile-dependent global/proj context is useful only for the last two detail steps. Vectorize membership attention before larger sweeps; the reference grouped implementation costs about 31 seconds per paired texture step.

## EXP-015: Paired multi-tile block fusion + 3D patch texture flow, step 10

### Goal

Test the task's conservative hypothesis: use the canonical global trajectory through step 9, then apply paired tile-dependent context only for the last two texture-detail steps.

### Code

Same verified working tree and condition format as EXP-014.

### Configuration

```bash
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
python pixal3d_directory_texture_eval.py \
  --image-dir assets/choose \
  --output-dir outputs/pixal3d_2048_multitile_paired_3dpatch_step10 \
  --resolutions 2048 \
  --seeds 42 \
  --lights studio \
  --shape-start-step 12 \
  --texture-mode global_original \
  --texture-multitile-3d-patch-flow \
  --texture-multitile-start-step 10 \
  --texture-canonical-image-size 4096 \
  --texture-image-tile-size 1024 \
  --texture-image-tile-stride 512 \
  --texture-3d-patch-size 64 \
  --texture-3d-patch-stride 32 \
  --texture-multitile-global-mode paired_block_fusion \
  --texture-multitile-save-debug \
  --render-resolution 2048 \
  --metric-resolution 1024 \
  --fail-fast
```

### Preprocessing

Identical to EXP-013/014: one RGBA-alpha preprocessing operation, rembg calls 0, source bbox `(115,414,3897,3867)`, square `(-75,60,4086,4221)`, padding `(75,0,0,125)`, canonical 4096/1024/512.

### 2D Condition Diagnostics

Identical token projection and assignment to EXP-014: fixed 49-tile layout, 48 used tiles, 59,234 tokens, membership histogram `{1:1, 2:4694, 4:54539}`, normalized float32 weight sums within `[0.99999988,1.00000012]`.

### 3D Flow Diagnostics

- 27 fixed 64³ patches, stride 32; coverage `{2:8025, 4:33720, 8:17489}`
- No uncovered token and no velocity fallback
- Step 10: cosine `0.99361449`, relative L2 `0.11331707`, MSE `0.01226053`, `31.416 s`
- Step 11: cosine `0.99498975`, relative L2 `0.10062289`, MSE `0.00834188`, `31.387 s`
- One synchronized global update per step

### Metrics

- PSNR: `17.3391828638`
- SSIM: `0.6688802838`
- LPIPS: `0.2155238390`
- Pipeline: `254.381 s`
- Peak CUDA allocated/reserved: `36,414,627,840 / 45,946,503,168` bytes
- Material bake: `410.247 s`
- Geometry unchanged: 59,234 latent tokens; 16,020,838 vertices; 31,974,354 faces

### Visual Findings

The late-only result remains coherent with no visible tile seam or geometry change. It stays much closer to the global control in global shell color/structure than EXP-014 while retaining the recognizable eye, neck, and shell markings.

### Conclusion

Versus canonical global, step 10 changes PSNR/SSIM/LPIPS by `-0.047134 dB / -0.001418 / -0.001591`: small PSNR/SSIM regressions but a perceptual LPIPS improvement. Versus step 6 it improves by `+0.297536 dB / +0.011171 / -0.004289`. This supports the hypothesis that paired crop context is safer as a last-two-step detail correction, but it still does not strictly dominate the canonical control on all metrics.

### Next Action

Treat step 10 as the paired-path candidate. Before multi-seed evaluation, implement and validate vectorized membership cross-attention, and add ROI/tile-boundary metrics to determine whether the LPIPS gain corresponds to useful head/shell detail rather than global color shift.
