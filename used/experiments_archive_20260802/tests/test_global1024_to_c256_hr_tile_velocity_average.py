from __future__ import annotations

import torch

import pixal3d_global1024_to_c256_hr_tile_velocity_average as experiment


def test_fresh_noise_is_global_row_aligned_and_seeded() -> None:
    coords = torch.tensor(
        [[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]],
        dtype=torch.int32,
    )

    first = experiment._fresh_global_noise(
        coords=coords,
        channels=4,
        seed=443,
        device=torch.device("cpu"),
    )
    replay = experiment._fresh_global_noise(
        coords=coords,
        channels=4,
        seed=443,
        device=torch.device("cpu"),
    )
    texture_namespace = experiment._fresh_global_noise(
        coords=coords,
        channels=4,
        seed=543,
        device=torch.device("cpu"),
    )

    assert torch.equal(first.coords, coords)
    assert torch.equal(first.feats, replay.feats)
    assert not torch.equal(first.feats, texture_namespace.feats)
