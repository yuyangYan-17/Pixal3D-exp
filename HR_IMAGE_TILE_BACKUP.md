# HR image-tile texture-flow backup

- Backup timestamp: `20260723T143051` (Asia/Shanghai)
- Current commit before implementation: `2898139c04f92aceef34199f0dd12b49d92c395e`

## Pre-change `git status --short`

```text
 M CODEX_TASK.md
 D assets/choose/0_img.png
?? assets/0_img.png
?? assets/choose/0_img_part.png
?? export_pixal3d_cache_to_glb.py
```

## Pre-change `git diff`

The complete binary-safe pre-change diff is recorded without truncation in:

- `HR_IMAGE_TILE_PRECHANGE_20260723T143051.patch`

The corresponding diff stat was:

```text
 CODEX_TASK.md           | 496 +++++++++++++++++++++++++++---------------------
 assets/choose/0_img.png | Bin 14005813 -> 0 bytes
 2 files changed, 277 insertions(+), 219 deletions(-)
```

## Source backups

- `pixal3d_directory_texture_eval.py.bak_20260723T143051_before_hr_image_tile_texflow`
- `pixal3d/pipelines/pixal3d_image_to_3d.py.bak_20260723T143051_before_hr_image_tile_texflow`
- `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py.bak_20260723T143051_before_hr_image_tile_texflow`
- `EXPERIMENT_LOG.md.bak_20260723T143051_before_hr_image_tile_texflow`

All backups were created with no-clobber semantics. No reset or checkout operation
was used.
