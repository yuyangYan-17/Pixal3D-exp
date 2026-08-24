# Abandoned geometry-only O-Voxel SR route

Archived on 2026-08-23 after the project chose the global dense-SLat route that jointly super-resolves geometry and texture.

This directory preserves the unfinished geometry-only experiment that attempted to collapse independently decoded local meshes into one global C4096 O-Voxel/mesh. It is historical code, not an active entry point, and its generated artifacts were intentionally removed from `outputs/`.

The progression was:

1. `root/pixal3d_ovoxel_hermite_qef_sr.py`: baseline-backed Hermite/QEF prototype.
2. `root/pixal3d_ovoxel_local_only_overlap.py`: local-only overlapping tile collection.
3. `root/pixal3d_ovoxel_local_only_topology_qef_merge.py`: decoder-native topology/QEF merge.
4. `root/pixal3d_ovoxel_global_mesh_revoxelize_merge.py`: mesh-first global revoxelization and one final O-Voxel extraction.
5. `root/pixal3d_tile26_27_codex_adapter.py`: two-tile raw-mesh/provenance adapter for the revoxelizer.
6. `tests/test_global_mesh_revoxelize_merge.py`: small placement/key tests for that route.

`supporting_core_snapshot/` contains the corresponding decoder hook that exported raw O-Voxel fields and provenance. The active `pixal3d/models/sc_vaes/fdg_vae.py` was restored to its default decoder API because neither retained formal route needs this hook.

The archived scripts still contain their original output paths and external TRELLIS.2 O-Voxel extension assumptions. They are preserved for audit only; do not treat them as supported or resume-compatible.
