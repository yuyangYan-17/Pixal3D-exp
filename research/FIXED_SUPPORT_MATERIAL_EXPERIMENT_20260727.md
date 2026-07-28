# Fixed-global-support local material residual experiment

Date: 2026-07-27

## 1. Conclusion first

The conservative material route is **partially effective**.  It proves that
tile-decoded color can enter the unified global `MeshWithVoxel` correctly
without normal-color fallback, point birth, geometry changes, or face-color
misalignment.

At residual strength `alpha=0.25`, every one of tiles 24/26/27 improves in all
three paired RGB metrics.  The mean change is:

- PSNR: `+0.01865 dB`;
- SSIM: `+0.001598`;
- LPIPS: `-0.002382`.

The full global render also improves by
`+0.00608 dB / +0.000829 / -0.000620`.

The gain is real but small.  It remains far below the independent local
Baseline D crop quality, so it is not yet a high-resolution unified 3D
solution.  Stronger replacement is not uniformly better: `alpha=0.5` loses
SSIM on tile 24, while `alpha=1.0` loses SSIM on all three tiles and produces
visible patchy/dark local-color discontinuities.  Therefore `alpha=0.25` is
the only tested setting with a uniform per-tile, per-metric win.

## 2. Hypothesis

The earlier modified-material render looked like isolated colored points
because it discarded the complete global material field and populated only a
small, discontinuous set of transformed tile keys.  The hypothesis tested
here is narrower:

> Keep the complete global geometry and material field, then apply a small
> local base-color residual only at exact existing global C1024 support keys.

This differs from failed global/local latent or velocity averaging:

- it operates after decode, on explicit O-Voxel material rows;
- correspondence is established by absolute camera geometry and sparse key
  equality, not row order;
- local points cannot create topology;
- unmatched global material remains present;
- no global/local feature distribution is averaged.

## 3. Implementation

New entry point:

`pixal3d_fixed_global_support_tile_material_fusion.py`

Inputs are decoded-support checkpoints created by
`pixal3d_projective_tile_generation_eval.py --save-decoded-support`.

Data flow:

```text
local decoded O-Voxel coord i_l
  -> q_l = 2 * (i_l + .5) / 1024 - 1
  -> exact centered-tile-to-global camera inverse
  -> continuous global C1024 index
  -> round without clamp
  -> reject unless key already exists in immutable global support
  -> within-tile: retain minimum continuous center error
  -> cross-tile: winner with maximum image-center tent confidence
  -> C_new = C_global + alpha * (C_tile - C_global)
```

Defaults and switches:

- `--fusion-mode winner_center`;
- `--confidence-mode tent`;
- `--blend-alpha 0.25`;
- `--min-confidence 1e-6`;
- optional foreground and approximate front-surface gates are off;
- `--dry-run` performs correspondence/fusion diagnostics without attrs or
  rendering;
- only the serialized `base_color` slice is modified.

No clamp, bbox/centroid normalization, dense `1024^3`, color alignment,
smoothing, PLY, GLB, or image-space overlay is used.

## 4. Correspondence and invariants

The immutable global model contains `3,956,833` O-Voxel rows.  After
tile-internal collision resolution and cross-tile winner selection:

| Item | Count |
|---|---:|
| tile 24 exact existing keys | 47,167 |
| tile 26 exact existing keys | 154,125 |
| tile 27 exact existing keys | 110,283 |
| union of modified global keys | 276,960 |
| fraction of global support | 6.9995% |
| cross-tile conflict keys | 34,615 |
| unchanged global rows | 3,679,873 |

Maximum camera round-trip error is below `8.35e-7` in normalized coordinates
and `0.000387 px`.

For all three alpha runs:

- vertices SHA-256 is unchanged;
- faces SHA-256 is unchanged;
- O-Voxel coordinate SHA-256 is unchanged;
- unmatched attrs are bitwise unchanged;
- metallic, roughness, alpha, and every other non-base-color channel are
  bitwise unchanged;
- `alpha=0` reconstructs the control attrs exactly.

In the `alpha=0.25` paired render, control/fused `normal.png`, `alpha.png`,
`metallic.png`, and `roughness.png` have identical SHA-256 hashes.  Only
base-color and shaded RGB differ.  This is direct evidence that the material
was queried on the original global faces rather than displayed as detached
normal-colored points.

## 5. Metrics

All numbers use the native Pixal3D PBR route, studio environment on black,
SSAA 2, peel 8, a 4096 full global render, native 1024 tile crops, metric
resolution 1024, and VGG LPIPS.  Deltas are paired within each run to avoid
mistaking small renderer repeatability differences for method gain.

### Alpha 0.25

| Tile | Control PSNR | Fused PSNR | ΔPSNR | Control SSIM | Fused SSIM | ΔSSIM | Control LPIPS | Fused LPIPS | LPIPS reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 9.97120 | 9.98883 | +0.01764 | .291196 | .291736 | +.000540 | .625915 | .624736 | .001180 |
| 26 | 11.07848 | 11.09987 | +0.02139 | .296209 | .298627 | +.002419 | .585676 | .582258 | .003418 |
| 27 | 12.92835 | 12.94527 | +0.01692 | .527507 | .529343 | +.001836 | .397355 | .394807 | .002549 |
| Mean | 11.32601 | 11.34466 | +0.01865 | .371637 | .373235 | +.001598 | .536315 | .533933 | .002382 |
| Median | 11.07848 | 11.09987 | +0.01764 | .296209 | .298627 | +.001836 | .585676 | .582258 | .002549 |

Relative to the control means, this is approximately `+0.165%` PSNR,
`+0.430%` SSIM, and `0.444%` LPIPS reduction.

### Strength ablation

| Alpha | Mean ΔPSNR | Mean ΔSSIM | Mean LPIPS reduction | Full-global ΔPSNR | Full-global ΔSSIM | Full-global LPIPS reduction |
|---:|---:|---:|---:|---:|---:|---:|
| .25 | +.01865 | +.001598 | .002382 | +.00608 | +.000829 | .000620 |
| .50 | +.02912 | +.001171 | .005157 | +.01031 | +.001549 | .001300 |
| 1.00 | +.02823 | -.003891 | .006411 | +.01367 | +.002530 | .002172 |

`alpha=0.5` has one per-tile regression: tile-24 SSIM `-0.000332`.
`alpha=1.0` regresses SSIM on all tiles:
`-0.003980 / -0.004126 / -0.003568`.

The complete per-tile/mean/median/global table is
`research/fixed_support_material_ablation_20260727.csv`.

## 6. Visual analysis

At `alpha=0.25`, changes are subtle and remain on the original continuous
surface.  The circular aperture and adjacent red/cyan bands in tiles 26/27
receive small local-color corrections; no new holes, detached points,
silhouette change, or normal change is visible.  This matches the identical
normal and geometry hashes.

At `alpha=0.5`, the local color contribution is clearer around the central
ring.  PSNR and LPIPS improve more, but tile 24 begins to lose local structural
agreement.

At `alpha=1.0`, dark/sharp patches appear around the central rings and small
high-contrast local regions.  These are consistent with the measured strong
local/global color disagreement and sparse 7% update coverage.  The loss of
SSIM on all three tiles rules out interpreting the stronger PSNR/LPIPS result
as a uniform win.

No obvious new tile seam is visible at `alpha=0.25` in the fixed global input
view.  A slight novel-view render was not included in this minimal experiment,
so texture stability away from the input view remains unproven.

## 7. Runtime and resource use

The offline mapping plus two 4096 renders and all full/crop metrics took about
`97–101 s` per alpha run.  `nvidia-smi dmon` measured peak framebuffer use of
`28,396 MiB` on the A800 for alpha 0.5 and 1.0.  Each complete experiment
directory is approximately `270 MiB`.

This is a post-generation cost.  The earlier v7 checkpoint-generation decoder
peak was about `61 GB`; this experiment does not rerun diffusion or decode.

## 8. Commands and outputs

Representative command:

```bash
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
python pixal3d_fixed_global_support_tile_material_fusion.py \
  --run-dir outputs/codex_support_v7_0img_seed42_tiles24_26_27_20260727 \
  --output-dir outputs/codex_fixed_global_support_material_alpha025_winner_20260727 \
  --tile-ids 24,26,27 \
  --fusion-mode winner_center \
  --confidence-mode tent \
  --blend-alpha 0.25 \
  --mapping-device cuda \
  --cuda-device 0 \
  --envmap studio \
  --render-resolution 4096 \
  --metric-resolution 1024 \
  --ssaa 2 \
  --peel-layers 8 \
  --no-use-envmap-bg \
  --lpips-net vgg \
  --metric-device cuda \
  --no-skip-lpips
```

Primary output:

`outputs/codex_fixed_global_support_material_alpha025_winner_20260727`

Useful artifacts:

- `summary.json`: full config, mapping counts, invariants, render metrics;
- `mapping_summary.json`: per-tile mapping/collision/conflict diagnostics;
- `fusion_provenance.pt`: every surviving candidate and winner;
- `global_control/` and `global_fused/`: shaded/base-color/normal/PBR maps;
- `tile_crop_metrics.csv`: paired per-tile metrics;
- `tile_24_26_27_contact_sheet.png`: reference/control/fused comparison;
- `tiles/tile_*/candidate_disagreement_overlay.png`: spatial candidate
  provenance/disagreement;
- `global_modified_attrs.pt`: modified attrs plus exact source-row provenance.

The alpha 0.5 and 1.0 outputs use the same directory naming with
`alpha050` and `alpha100`.

## 9. Decision

Classification: **partially effective; retain as a validated material-transfer
primitive, but do not claim the high-resolution objective is solved**.

What is established:

- the absolute coordinate/material binding is correct;
- preserving the complete global material removes the “normal-only holes”;
- conservative local color residual can improve a single unified global model
  on all fixed tiles.

What is not established:

- improvement over the much stronger independent local Baseline D;
- novel-view stability;
- multi-image or multi-seed robustness;
- useful geometry super-resolution;
- whether a distance/conflict gate can make stronger residuals safe.

The next justified experiment is a pre-registered center-distance/conflict
gate at `alpha=0.25`, not topology union or global/local velocity averaging.

