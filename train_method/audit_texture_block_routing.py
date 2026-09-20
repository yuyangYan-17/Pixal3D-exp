"""CPU audit of saved visibility, independent of GPU flow and rendering."""
import argparse
import json
from pathlib import Path

import torch
from train_method.texture_block_routing import route_blocks, VERSION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    data = torch.load(args.root / "shared/mapping.pt", map_location="cpu", weights_only=False)
    encoded = torch.load(args.root / "shared/bridge/encoded.pt", map_location="cpu", weights_only=False)
    assert torch.equal(data["coords"], encoded["coords"])
    visible = encoded["visible"].bool()
    routes, stats = route_blocks(data, visible, torch.ones(len(visible)), None)
    counts = torch.zeros(len(visible), dtype=torch.int32)
    hidden_conditional = torch.zeros(len(visible), dtype=torch.bool)
    for branch in ("conditional", "unconditional", "guided"):
        for block in routes[branch]:
            ids = block["global_ids"][block["owned"]]
            counts.index_add_(0, ids, torch.ones(len(ids), dtype=torch.int32))
            if branch == "conditional":
                hidden_conditional[ids[~visible[ids]]] = True
    assert torch.equal(counts, data["counts"])
    result = dict(version=VERSION, root=str(args.root), tiles=len(data["tiles"]),
                  candidate_blocks=len(data["blocks"]), points=len(visible),
                  hidden_points=int((~visible).sum()),
                  hidden_points_updated_by_high_conditional_blocks=int(hidden_conditional.sum()),
                  counts_match=True, **stats,
                  note="Visibility/ownership audit only; guide coverage and model output not evaluated")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
