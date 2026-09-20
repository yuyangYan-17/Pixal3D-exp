import unittest

import torch

from train_method.texture_block_routing import route_blocks


class BlockRouting(unittest.TestCase):
    def block(self, count=10, block_id=0):
        return dict(block_id=block_id, global_ids=torch.arange(count),
                    owned=torch.ones(count, dtype=torch.bool), rows=torch.arange(count))

    def test_n_zero_through_four_all_twelve_steps(self):
        for visible_count in (0, 1, 3, 4, 10):
            visible = torch.arange(10) < visible_count
            # Two overlapping z blocks must contribute exactly twice per point.
            data = dict(blocks=[self.block(), self.block(block_id=1)])
            for n in range(5):
                for step in range(12):
                    routes, stats = route_blocks(data, visible, torch.ones(10),
                                                 step if step < n else None)
                    total = torch.zeros(10)
                    counts = torch.zeros(10, dtype=torch.long)
                    for branch, value in (("conditional", 2), ("unconditional", -2), ("guided", 7)):
                        for block in routes[branch]:
                            self.assertEqual(len(block["global_ids"]), 10)
                            ids = block["global_ids"][block["owned"]]
                            counts.index_add_(0, ids, torch.ones_like(ids))
                            total.index_add_(0, ids, torch.full((len(ids),), float(value)))
                    torch.testing.assert_close(counts, torch.full_like(counts, 2))
                    expected = torch.full((10,), float(7 if step < n else -2))
                    expected[visible] = 2
                    if visible_count > 3:
                        expected[:] = 2
                    torch.testing.assert_close(total / counts, expected)
                    if 0 < visible_count <= 3:
                        self.assertEqual(len(routes["conditional"]), 2)
                        self.assertEqual(len(routes["unconditional"]), 2)
                    if visible_count == 0:
                        self.assertFalse(routes["conditional"])

    def test_partial_guide_does_not_silently_change_route(self):
        with self.assertRaisesRegex(ValueError, "baseline guide"):
            route_blocks(dict(blocks=[self.block()]), torch.zeros(10, dtype=torch.bool),
                         torch.zeros(10), 0)

    def test_explicit_all_conditional_control(self):
        routes, _ = route_blocks(dict(blocks=[self.block()]), torch.zeros(10, dtype=torch.bool),
                                 torch.ones(10), None, all_conditional=True)
        self.assertEqual(len(routes["conditional"]), 1)
        self.assertFalse(routes["unconditional"])

    def test_first_stage_3_by_3_by_7(self):
        from sr import mapping
        from pathlib import Path
        import tempfile
        coords = torch.tensor([[0, 64, 64, z] for z in range(128)], dtype=torch.int32)
        with tempfile.TemporaryDirectory() as tmp:
            data = mapping(coords, 512, dict(distance=2.1, camera_angle_x=0.48), Path(tmp), None)
        self.assertEqual(len(data["tiles"]), 9)
        self.assertEqual(len(data["blocks"]), 63)
        self.assertEqual(sorted(set(b["z_start"] for b in data["blocks"])), list(range(0, 97, 16)))


if __name__ == "__main__":
    unittest.main()
