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
