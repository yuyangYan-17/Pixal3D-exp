from __future__ import annotations

import torch


def test_target_context_hard_owner_is_unique() -> None:
    tile_ids = torch.tensor(
        [
            [0, 1, 7, 8],
            [4, 5, -1, -1],
            [9, -1, -1, -1],
        ],
        dtype=torch.long,
    )
    weights = torch.tensor(
        [
            [0.1, 0.4, 0.2, 0.3],
            [0.75, 0.25, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    scores = weights.masked_fill(tile_ids < 0, -torch.inf)
    slots = scores.argmax(dim=1)
    owners = tile_ids.gather(1, slots[:, None])[:, 0]
    assert owners.tolist() == [1, 4, 9]
    assert torch.all(owners >= 0)


def test_target_context_expert_has_one_condition_per_row() -> None:
    token_count = 6
    target_rows = torch.tensor([1, 4])
    tile_ids = torch.full((token_count, 4), -1, dtype=torch.long)
    tile_weights = torch.zeros((token_count, 4), dtype=torch.float32)
    tile_ids[:, 0] = 0
    tile_ids[target_rows, 0] = 1
    tile_weights[:, 0] = 1.0

    assert torch.equal(
        (tile_ids[:, 0] == 1),
        torch.tensor([False, True, False, False, True, False]),
    )
    assert torch.all((tile_ids >= 0).sum(1) == 1)
    assert torch.allclose(tile_weights.sum(1), torch.ones(token_count))


def test_target_context_projected_rows_are_selected_not_blended() -> None:
    base = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    local = base + 1000
    target_rows = torch.tensor([1, 4])
    mixed = base.clone()
    mixed[target_rows] = local[target_rows]

    assert torch.equal(mixed[0], base[0])
    assert torch.equal(mixed[1], local[1])
    assert torch.equal(mixed[4], local[4])
    assert torch.equal(mixed[5], base[5])
