#!/usr/bin/env python3
from __future__ import annotations

import ast
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "pixal3d/pipelines/pixal3d_image_to_3d.py"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected one exact match, found {count}"
        )
    return source.replace(old, new, 1)


def main() -> None:
    source = PIPELINE.read_text(encoding="utf-8")

    source = replace_once(
        source,
        '        base_condition: Mapping[str, Any],\n'
        '        grid_resolution: int = 128,\n',
        '        base_condition: Mapping[str, Any] = None,\n'
        '        grid_resolution: int = 128,\n',
        "optional base_condition signature",
    )

    source = replace_once(
        source,
        '''        base_global = base_condition.get("global")
        base_proj = base_condition.get("proj")
        if not isinstance(base_global, torch.Tensor):
            raise TypeError("base_condition['global'] must be a tensor")
        if not isinstance(base_proj, SparseTensor):
            raise TypeError("base_condition['proj'] must be a SparseTensor")
        if base_global.ndim != 3 or base_global.shape[0] != 1:
            raise ValueError(
                "base full-image global must have shape [1, L, C]"
            )
        if base_proj.feats.shape[0] != global_coords.shape[0]:
            raise RuntimeError("base projected feature is not token aligned")
        if not torch.equal(base_proj.coords, global_coords):
            raise RuntimeError("base projected coordinates changed token order")
        if not torch.isfinite(base_global).all():
            raise RuntimeError("base global contains NaN/Inf")
        if not torch.isfinite(base_proj.feats).all():
            raise RuntimeError("base projected feature contains NaN/Inf")
''',
        '''        base_condition_available = base_condition is not None
        base_global = None
        base_proj = None
        if base_condition_available:
            base_global = base_condition.get("global")
            base_proj = base_condition.get("proj")
            if not isinstance(base_global, torch.Tensor):
                raise TypeError(
                    "base_condition['global'] must be a tensor"
                )
            if not isinstance(base_proj, SparseTensor):
                raise TypeError(
                    "base_condition['proj'] must be a SparseTensor"
                )
            if base_global.ndim != 3 or base_global.shape[0] != 1:
                raise ValueError(
                    "base full-image global must have shape [1, L, C]"
                )
            if base_proj.feats.shape[0] != global_coords.shape[0]:
                raise RuntimeError(
                    "base projected feature is not token aligned"
                )
            if not torch.equal(base_proj.coords, global_coords):
                raise RuntimeError(
                    "base projected coordinates changed token order"
                )
            if not torch.isfinite(base_global).all():
                raise RuntimeError("base global contains NaN/Inf")
            if not torch.isfinite(base_proj.feats).all():
                raise RuntimeError(
                    "base projected feature contains NaN/Inf"
                )
''',
        "optional base_condition validation",
    )

    source = replace_once(
        source,
        '''        fused_proj = torch.zeros_like(base_proj.feats)
        owner_proj = torch.zeros_like(base_proj.feats)
''',
        '''        fused_proj = None
        owner_proj = None
''',
        "lazy projected-feature allocation",
    )

    source = replace_once(
        source,
        '''            if tile_proj.shape[1] != base_proj.feats.shape[1]:
                raise RuntimeError(
                    "local and base projected channel counts differ"
                )
''',
        '''            if (
                base_proj is not None
                and tile_proj.shape[1] != base_proj.feats.shape[1]
            ):
                raise RuntimeError(
                    "local and base projected channel counts differ"
                )
            if fused_proj is None:
                fused_proj = torch.zeros(
                    global_coords.shape[0],
                    tile_proj.shape[1],
                    device=tile_proj.device,
                    dtype=tile_proj.dtype,
                )
                owner_proj = torch.zeros_like(fused_proj)
''',
        "lazy projected-feature initialization",
    )

    source = replace_once(
        source,
        '''        if not global_bank_parts:
            raise RuntimeError("no image tile condition was extracted")
        if torch.any(owner_write_count != 1):
''',
        '''        if not global_bank_parts:
            raise RuntimeError("no image tile condition was extracted")
        if fused_proj is None or owner_proj is None:
            raise RuntimeError(
                "tile extraction did not initialize projected features"
            )
        if not base_condition_available:
            # Legacy paired-fusion tests historically called this internal
            # helper without a full-image condition. These placeholders keep
            # that old preparation path numerically unchanged. The
            # target_context_hard runner rejects them explicitly below.
            base_global = torch.zeros_like(
                global_bank_parts[0]
            ).unsqueeze(0)
            base_proj = SparseTensor(
                feats=torch.zeros_like(fused_proj),
                coords=global_coords,
            )
        if torch.any(owner_write_count != 1):
''',
        "legacy base placeholders",
    )

    source = replace_once(
        source,
        '''            "base_global": base_global,
            "base_proj": base_proj,
''',
        '''            "base_condition_available": base_condition_available,
            "base_global": base_global,
            "base_proj": base_proj,
''',
        "base availability marker",
    )

    source = replace_once(
        source,
        '''        if fusion_mode == "target_context_hard":
            required = {
''',
        '''        if fusion_mode == "target_context_hard":
            if not condition.get("base_condition_available", False):
                raise RuntimeError(
                    "target_context_hard requires the canonical "
                    "full-image base condition"
                )
            required = {
''',
        "target mode base-condition guard",
    )

    ast.parse(source, filename=str(PIPELINE))

    backup = PIPELINE.with_suffix(
        PIPELINE.suffix + ".before_optional_base_fix"
    )
    if not backup.exists():
        shutil.copy2(PIPELINE, backup)
        print(f"[backup] {backup}")

    PIPELINE.write_text(source, encoding="utf-8")
    print(f"[patch] wrote {PIPELINE}")
    print("[done] optional base-condition compatibility fix applied")


if __name__ == "__main__":
    main()
