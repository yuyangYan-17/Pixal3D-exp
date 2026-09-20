"""Regression contracts for ownership, global synchronization and padding."""

import unittest
import torch
from sr_tools import shape


class GlobalSynchronization(unittest.TestCase):
    def test_padding_adds_context_without_changing_owners(self):
        for stage, grid in ((512, 128), (1024, 256)):
            xyz = torch.cartesian_prod(
                torch.arange(grid // 4, 3 * grid // 4, grid // 16),
                torch.arange(grid // 4, 3 * grid // 4, grid // 16),
                torch.tensor([0, grid // 4 - 1, grid // 4, grid - 1]),
            )
            coords = torch.cat(
                (torch.zeros(len(xyz), 1, dtype=torch.long), xyz), 1
            ).int()
            camera = dict(distance=2.1, camera_angle_x=0.48)
            core = shape.construct(coords, stage, camera, padding=0)
            padded = shape.construct(coords, stage, camera, padding=32)
            self.assertTrue(torch.equal(core["owner"], padded["owner"]))
            self.assertGreater(
                sum(len(t["rows"]) for t in padded["tiles"]),
                sum(len(t["rows"]) for t in core["tiles"]),
            )
            # Synthetic velocities distinguish owner predictions from halo predictions.
            summed = torch.zeros(len(coords), 1)
            counts = torch.zeros(len(coords), dtype=torch.int32)
            values = [
                torch.full((len(b["rows"]), 1), float(b["tile_id"] + 1))
                for b in padded["blocks"]
            ]
            shape.reduce_predictions(summed, counts, padded["blocks"], values)
            self.assertTrue(torch.equal(counts, padded["counts"]))
            torch.testing.assert_close(
                summed[:, 0] / counts, padded["owner"].float() + 1
            )

    def test_batches_do_not_change_velocity_reduction(self):
        coords = torch.tensor([[0, 64, 64, z] for z in range(128)], dtype=torch.int32)
        data = shape.construct(coords, 512, dict(distance=2.1, camera_angle_x=0.48))
        outputs = []
        for size in (1, 3, 28):
            total, count = torch.zeros(128, 2), torch.zeros(128, dtype=torch.int32)
            for group in shape.groups(data["blocks"], size, 100000):
                velocities = [
                    torch.full((len(b["rows"]), 2), float(b["z_start"])) for b in group
                ]
                shape.reduce_predictions(total, count, group, velocities)
            outputs.append(total / count[:, None])
        for output in outputs[1:]:
            torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
