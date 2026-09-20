"""Homogeneous forwards, followed by visibility-based velocity selection."""

import torch

VERSION = "block_threshold_point_selection_v2"


def route_blocks(data, visible, coverage, guide_step, threshold=0.30,
                 guide_threshold=1.0, all_conditional=False):
    if not 0 <= threshold <= 1:
        raise ValueError("visibility threshold must be in [0,1]")
    result = {name: [] for name in ("conditional", "unconditional", "guided")}
    stats = dict(high_blocks=0, mixed_blocks=0, hidden_blocks=0,
                 hidden_conditional_owner_occurrences=0)

    def append(block, name, mask):
        # Retain the identical full sparse context for both forward calls.
        view = dict(block)
        view["owned"] = block["owned"] & mask
        view["branch"] = name
        view["source_block_id"] = block["block_id"]
        result[name].append(view)

    for block in data["blocks"]:
        ids = block["global_ids"]
        if not len(ids):
            continue
        vis = visible[ids].bool()
        all_rows = torch.ones_like(vis)
        fraction = int(vis.sum()) / len(vis)
        if all_conditional or fraction > threshold:
            stats["high_blocks"] += 1
            stats["hidden_conditional_owner_occurrences"] += int((~vis & block["owned"]).sum())
            append(block, "conditional", all_rows)
            continue
        mixed = bool(vis.any())
        stats["mixed_blocks" if mixed else "hidden_blocks"] += 1
        if mixed:
            append(block, "conditional", vis)
        if guide_step is not None:
            # Never silently replace requested baseline guidance by another route.
            if not bool((coverage[ids][~vis] >= guide_threshold).all()):
                raise ValueError("baseline guide lacks hidden support; rebuild with mesh-face fallback")
            append(block, "guided", ~vis)
            if mixed:
                # User contract: two independent full-block forwards every step.
                # During the guide prefix, unconditional v is evaluated but unused.
                append(block, "unconditional", torch.zeros_like(vis))
        else:
            append(block, "unconditional", ~vis)
    return result, stats
