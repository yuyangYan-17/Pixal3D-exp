from types import SimpleNamespace

import torch

import pixal3d_tile_c1024_overlap_velocity_sync_local_decode_merge as experiment


def _tile(
    tile_id: int,
    local_token_count: int,
    source_xyz: list[tuple[int, int, int]],
    source_to_local: list[int],
) -> SimpleNamespace:
    local_coords = torch.zeros((local_token_count, 4), dtype=torch.int32)
    local_coords[:, 1] = torch.arange(local_token_count, dtype=torch.int32)
    source_coords = torch.tensor(
        [[0, x, y, z] for x, y, z in source_xyz],
        dtype=torch.int32,
    )
    return SimpleNamespace(
        tile_id=tile_id,
        mapping=SimpleNamespace(
            local_coords64=local_coords,
            source_global_coords1024=source_coords,
            source_to_local_index=torch.tensor(
                source_to_local,
                dtype=torch.int64,
            ),
        ),
    )


def test_exact_global_c1024_links_average_each_step_velocity() -> None:
    tiles = [
        _tile(
            0,
            3,
            [(1, 1, 1), (3, 3, 3), (5, 5, 5)],
            [0, 1, 2],
        ),
        _tile(
            1,
            2,
            [(1, 1, 1), (3, 3, 3), (4, 4, 4)],
            [0, 1, 1],
        ),
        _tile(2, 1, [(1, 1, 1)], [0]),
    ]
    correspondence = experiment._build_velocity_correspondence(tiles)
    raw_velocity = torch.tensor(
        [[0.0], [10.0], [99.0], [3.0], [20.0], [6.0]]
    )

    synchronized, metrics = experiment._average_corresponding_velocities(
        raw_velocity,
        correspondence,
    )

    torch.testing.assert_close(
        synchronized[:, 0],
        torch.tensor([3.0, 15.0, 99.0, 3.0, 15.0, 3.0]),
    )
    assert correspondence.stats["shared_global_c1024_sources"] == 2
    assert correspondence.stats["linked_local_c64_nodes"] == 5
    assert correspondence.stats["unlinked_local_c64_nodes"] == 1
    assert metrics["linked_node_count"] == 5.0


def test_many_global_sources_on_one_local_token_mean_consensus_proposals() -> None:
    source_a = (10, 20, 30)
    source_c = (40, 50, 60)
    tiles = [
        _tile(0, 1, [source_a, source_c], [0, 0]),
        _tile(1, 1, [source_a], [0]),
        _tile(2, 1, [source_c], [0]),
    ]
    correspondence = experiment._build_velocity_correspondence(tiles)
    raw_velocity = torch.tensor([[0.0], [6.0], [12.0]])

    synchronized, _ = experiment._average_corresponding_velocities(
        raw_velocity,
        correspondence,
    )

    # Source A proposes (0 + 6) / 2 = 3 to tile 0; source C proposes
    # (0 + 12) / 2 = 6. Tile 0 receives their mean once: 4.5.
    torch.testing.assert_close(
        synchronized[:, 0],
        torch.tensor([4.5, 3.0, 6.0]),
    )
    torch.testing.assert_close(
        correspondence.linked_node_group_counts,
        torch.tensor([2.0, 1.0, 1.0]),
    )


def test_unlinked_tokens_keep_their_own_velocity() -> None:
    tiles = [
        _tile(0, 1, [(100, 100, 100)], [0]),
        _tile(1, 1, [(200, 200, 200)], [0]),
    ]
    correspondence = experiment._build_velocity_correspondence(tiles)
    raw_velocity = torch.tensor([[1.25, -2.0], [7.0, 8.0]])

    synchronized, metrics = experiment._average_corresponding_velocities(
        raw_velocity,
        correspondence,
    )

    torch.testing.assert_close(synchronized, raw_velocity)
    assert correspondence.shared_global_coords1024.shape == (0, 4)
    assert metrics["linked_node_count"] == 0.0


def test_lockstep_stage_uses_consensus_before_each_euler_update() -> None:
    source_a = (7, 8, 9)
    tiles = [
        _tile(0, 2, [source_a, (1, 2, 3)], [0, 1]),
        _tile(1, 2, [source_a, (4, 5, 6)], [0, 1]),
    ]
    correspondence = experiment._build_velocity_correspondence(tiles)

    class FakeSampler:
        @staticmethod
        def timestep_schedule(
            steps: int,
            _rescale_t: float,
        ) -> list[float]:
            assert steps == 2
            return [1.0, 0.5, 0.0]

        @staticmethod
        def sample_once(
            _model: torch.nn.Module,
            state: experiment.SparseTensor,
            _t: float,
            _t_next: float,
            *,
            cond: dict,
            **_kwargs: object,
        ) -> SimpleNamespace:
            marker = float(cond["global"].item())
            velocity = state.replace(
                torch.full_like(state.feats, marker)
            )
            return SimpleNamespace(pred_v=velocity)

    packed_conditions = []
    for marker in (0.0, 4.0):
        branch = {
            "global": torch.tensor([[[marker]]]),
            "proj": torch.zeros((2, 1)),
        }
        packed_conditions.append(
            {
                "cond": branch,
                "neg_cond": {
                    "global": torch.zeros((1, 1, 1)),
                    "proj": torch.zeros((2, 1)),
                },
            }
        )
    seeds = [11, 22]
    initial = [
        experiment._randn(
            2,
            1,
            device=torch.device("cpu"),
            seed=seed,
        )
        for seed in seeds
    ]
    pipeline = SimpleNamespace(
        device=torch.device("cpu"),
        low_vram=False,
    )

    result = experiment._run_correspondence_velocity_synced_stage(
        pipeline=pipeline,
        sampler=FakeSampler(),
        model=torch.nn.Identity(),
        tiles=tiles,
        packed_conditions=packed_conditions,
        correspondence=correspondence,
        params={
            "steps": 2,
            "rescale_t": 1.0,
            "guidance_strength": 3.0,
        },
        seeds=seeds,
        noise_channels=1,
        stage="test",
    )

    # Shared token 0 uses mean(0, 4) = 2 in both tiles. The unlinked token
    # keeps tile 0's own velocity 0 or tile 1's own velocity 4.
    torch.testing.assert_close(
        result.normalized_features[0],
        initial[0] - torch.tensor([[2.0], [0.0]]),
    )
    torch.testing.assert_close(
        result.normalized_features[1],
        initial[1] - torch.tensor([[2.0], [4.0]]),
    )
    assert len(result.step_records) == 2
