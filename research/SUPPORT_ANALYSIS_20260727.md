# Fixed-three-tile support correspondence analysis

Date: 2026-07-27

## Scope and protocol

This report evaluates the support correspondence that is required before any
global/local material fusion is attempted.  The protocol is fixed to:

- input: `assets/choose/0_img.png`
- base seed: `42`
- tiles: `24` (center), `26` (complex/mid), and `27` (edge)
- route: v7, namely native tile C32 plus projected global support, native tile
  shape, projected/native C64 union, then tile shape/texture decode
- global reference: ordinary one-step global C1024 decode

The generation checkpoint is:

`outputs/codex_support_v7_0img_seed42_tiles24_26_27_20260727`

The complete machine-readable analysis is:

`outputs/codex_support_v7_0img_seed42_tiles24_26_27_20260727/support_correspondence_analysis/summary.json`

Coordinates follow the decoder conventions exactly:

- decoded C1024 material voxels use centers
  `q = 2 * (coord + 0.5) / 1024 - 1`;
- learned/projected C32 and C64 shape support uses endpoint coordinates
  `q = 2 * coord / (R - 1) - 1`;
- camera conversion uses the exact documented global-to-centered-tile mapping
  and its analytic inverse;
- global C1024 re-quantization uses
  `round((q_global + 1) * 1024 / 2 - 0.5)`, without clamp, bbox
  normalization, centroid normalization, or coordinate compression.

The analysis uses sparse hashes and sampled KD-tree queries.  It never
constructs a dense `1024^3` tensor and exports no PLY/GLB.

## Coordinate validation

For 100,000 random local points on each fixed tile:

- local -> global -> local normalized-coordinate maximum absolute error:
  `1.192e-6`;
- inverse pixel round-trip maximum error:
  tile 24 `0.000173 px`, tile 26 `0.000341 px`, tile 27 `0.000357 px`;
- forward pixel round-trip maximum error: `4.316e-5 px`;
- analytic inverse versus the documented closed form maximum error:
  `9.537e-7`.

The decoded-support run itself reports local-material camera round-trip errors
below `7.153e-7` in normalized coordinates and `0.000358 px`.  Consequently,
the observed support mismatch is not explained by numerical camera inversion.

## Shape-support projection

The ordinary global decode contains `3,956,833` unique C1024 O-Voxels,
`3,956,833` decoded vertices, and `7,918,394` faces.  The learned global shape
support contains `3,187` C32 and `13,504` C64 tokens.

| Tile | Selected global C1024 rows | Kept in local cube | Projected C32 | Projected C64 | Native C64 | Fused C64 | C1024→C64 collision |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 382,437 | 365,932 | 2,869 | 12,040 | 995 | 12,823 | 96.710% |
| 26 | 567,525 | 561,523 | 3,725 | 16,397 | 6,509 | 21,209 | 97.080% |
| 27 | 395,800 | 394,028 | 2,496 | 11,406 | 3,033 | 13,375 | 97.106% |

The approximately 97% reduction is expected when millions of C1024 samples
are quantized onto C64, but it is an important information bottleneck: a
projected C64 token is a coarse shape prior, not an identity-preserving
material correspondence.

## Decoded C1024 correspondence

Each local decode has several million rows.  Transforming their voxel centers
to the global camera and quantizing to global C1024 creates substantial
many-to-one fan-in:

| Tile | Local rows | Unique global keys | Collision rows | Collision rate | Max fan-in |
|---:|---:|---:|---:|---:|---:|
| 24 | 3,577,518 | 413,172 | 3,164,346 | 88.451% | 25 |
| 26 | 4,868,002 | 678,168 | 4,189,834 | 86.069% | 25 |
| 27 | 3,756,203 | 470,387 | 3,285,816 | 87.477% | 25 |

After tile-internal deduplication, exact equality with the immutable global
C1024 support is still sparse:

| Tile | Matched | Global-only in tile region | Local-only | Jaccard | Match/local | Match/global-region |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 47,167 | 331,681 | 366,005 | 6.332% | 11.416% | 12.449% |
| 26 | 154,192 | 424,298 | 523,976 | 13.986% | 22.736% | 26.652% |
| 27 | 110,289 | 290,619 | 360,098 | 14.493% | 23.446% | 27.507% |

This directly explains the earlier “colored points over normal color” result:
assigning only exact local-derived material keys colors a small and spatially
discontinuous subset of the existing global O-Voxel field.  It is not a
renderer face-coloring bug.  The native renderer continuously interpolates
surface position and queries sparse O-Voxel material, but sparse isolated
updates cannot provide a continuous replacement material field.

## Spatial-distance diagnosis

Nearest-neighbor distances were measured from 200,000 transformed local
material centers per tile to the complete global O-Voxel set.  The following
values are expressed in global C1024 voxel widths (`q * 512`):

| Tile | O-Voxel p50 | O-Voxel p95 | O-Voxel p99 | Surface-vertex proxy p50 | proxy p95 | proxy p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 2.635 | 15.735 | 42.908 | 3.127 | 16.237 | 43.265 |
| 26 | 1.602 | 20.755 | 35.480 | 2.006 | 21.308 | 36.082 |
| 27 | 1.326 | 13.863 | 23.916 | 1.737 | 14.363 | 24.486 |

The vertex figure is only a nearest-decoded-vertex proxy, not an exact
point-to-triangle distance.  Even so, the long tails show that unrestricted
nearest-neighbor transfer would cross real geometry gaps and cannot be
treated as a safe correspondence rule.

## Margin bands

The representative local keys were grouped by their projected distance to
the 1024-pixel tile boundary: edge `<128 px`, transition `128–256 px`, and
center `>=256 px`.

| Tile | Edge matched/all | Transition matched/all | Center matched/all |
|---:|---:|---:|---:|
| 24 | 15,917 / 148,037 (10.75%) | 19,843 / 157,034 (12.64%) | 11,407 / 108,049 (10.56%) |
| 26 | 43,428 / 231,864 (18.73%) | 49,773 / 246,571 (20.19%) | 60,877 / 199,315 (30.54%) |
| 27 | 36,965 / 142,371 (25.96%) | 38,774 / 156,825 (24.72%) | 34,544 / 171,152 (20.18%) |

Mismatch is therefore not confined to tile borders.  Tile 26 benefits from
its center, whereas tiles 24 and 27 do not show a monotonic center advantage.
A fixed “discard the outer band” rule would remove evidence without resolving
the underlying support disagreement.

## Material agreement on exact keys

Even where support identity is exact, independently decoded base color differs
strongly:

| Tile | Matched keys | Mean absolute error | RMSE | Pseudo-PSNR | RGB L2 p50 | RGB L2 p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 47,167 | 0.17145 | 0.22200 | 13.073 dB | 0.2434 | 0.7584 |
| 26 | 154,192 | 0.20684 | 0.25592 | 11.838 dB | 0.3342 | 0.8344 |
| 27 | 110,289 | 0.21202 | 0.25957 | 11.715 dB | 0.3487 | 0.8295 |

Thus exact coordinate identity is necessary for a conservative test, but it
does not establish photometric consistency.

Tiles 26 and 27 overlap in canonical image space at
`[3072,1536,3584,2560]`.  Within that overlap:

- tile-26 unique support: `372,038`;
- tile-27 unique support: `365,337`;
- exact shared keys: `92,977`;
- union: `644,398`;
- Jaccard: `14.429%`;
- agreement over the smaller support: `25.450%`;
- shared-key base-color MAE/RMSE: `0.14329 / 0.19011`;
- shared-key pseudo-PSNR: `14.420 dB`.

Because two local decoders disagree both topologically and photometrically,
blind averaging is not a justified default.  Winner selection plus a small
residual update is the safer first intervention.

## Consequence and pre-registered experiment

The diagnosis supports only a narrow first experiment:

1. freeze global vertices, faces, and C1024 O-Voxel coordinates bitwise;
2. transform local decoded C1024 centers with the exact camera inverse;
3. retain only keys that already exist in the global sparse support;
4. resolve within-tile collisions by minimum continuous center distance;
5. resolve cross-tile conflicts by image-center confidence;
6. modify only base color, leaving every unmatched and non-base attribute
   bitwise unchanged;
7. apply a small residual blend, with alpha zero as an exact control.

The experiment is deliberately a material-quality test, not a topology-fusion
claim.  With only about 7% of the full global support reachable after strict
matching across the three selected tiles, a large full-image gain is not
expected.  Failure is defined as any geometry/support hash change, alpha-zero
non-identity, visible new discontinuities, or degradation of the fixed-tile
mean metrics beyond ordinary render repeatability.

## Reproduction

Generate the checkpoints:

```bash
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 \
python pixal3d_projective_tile_generation_eval.py \
  --image assets/choose/0_img.png \
  --output-dir outputs/codex_support_v7_0img_seed42_tiles24_26_27_20260727 \
  --tile-ids 24,26,27 \
  --base-seed 42 \
  --render-resolution 256 \
  --modified-global-render-resolution 256 \
  --baseline-render-resolution 1024 \
  --metric-resolution 256 \
  --skip-lpips \
  --save-decoded-support
```

Run the CPU postprocessor:

```bash
python pixal3d_support_correspondence_analysis.py \
  --run-dir outputs/codex_support_v7_0img_seed42_tiles24_26_27_20260727 \
  --output-dir outputs/codex_support_v7_0img_seed42_tiles24_26_27_20260727/support_correspondence_analysis \
  --tile-ids 24,26,27 \
  --distance-samples 200000 \
  --workers -1
```

