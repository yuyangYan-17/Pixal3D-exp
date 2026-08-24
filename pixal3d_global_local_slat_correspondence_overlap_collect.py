#!/usr/bin/env python3
"""Run the legacy local-SLat collector with the documented 7x7 overlap layout.

The repository core currently defaults to a 4x4 disjoint quick-experiment
layout. This wrapper changes only the collector's tile-layout function in
process; it does not modify the shared core module or any model parameters.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pixal3d_tile_c1024_local_slat_and_local_decode_return_global as core


ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / "used" / "pixal3d_global_local_slat_correspondence.py"
_original_layout = core._tile_layout


def _overlap_layout(
    canonical_size: int = core.CANONICAL_IMAGE_SIZE,
    tile_size: int = core.TILE_SIZE,
    stride: int = 512,
):
    del stride
    return _original_layout(canonical_size, tile_size, 512)


core.TILE_STRIDE = 512
core._tile_layout = _overlap_layout

spec = importlib.util.spec_from_file_location("_legacy_overlap_collector", LEGACY)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {LEGACY}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.core.TILE_STRIDE = 512
module.core._tile_layout = _overlap_layout
module.main()
