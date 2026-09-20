import unittest
import torch
from sr_tools import shape


class DepthBlend(unittest.TestCase):
    def test_positive_partition_and_continuous_crossfade(self):
        for stage, grid, width in ((512, 128, 32), (1024, 256, 64)):
            coords = torch.tensor(
                [[0, grid // 2, grid // 2, z] for z in range(grid)], dtype=torch.int32
            )
            data = shape.construct(
                coords, stage, dict(distance=2.1, camera_angle_x=0.48)
            )
            stride = width // 2
            self.assertTrue((data["depth_weight_sums"] > 0).all())
            torch.testing.assert_close(
                data["depth_weight_sums"][stride:-stride], torch.ones(grid - 2 * stride)
            )
            results = []
            for batch in (1, 3, 112):
                total = torch.zeros(grid, 1)
                count = torch.zeros(grid, dtype=torch.int32)
                weights = torch.zeros(grid)
                for group in shape.groups(data["blocks"], batch, 100000):
                    # Neighboring blocks deliberately disagree by one unit.
                    values = [
                        torch.full((len(b["rows"]), 1), float(b["z_start"] / stride))
                        for b in group
                    ]
                    shape.reduce_predictions(total, count, group, values, weights)
                torch.testing.assert_close(count, data["counts"])
                result = total[:, 0] / weights
                self.assertLessEqual(
                    float(result.diff().abs().max()), 1 / stride + 1e-6
                )
                results.append(result)
            for result in results[1:]:
                torch.testing.assert_close(result, results[0], rtol=0, atol=0)

    def test_halo_never_contributes_and_constant_velocity_is_preserved(self):
        coords = torch.tensor([[0, 64, 64, z] for z in range(128)], dtype=torch.int32)
        data = shape.construct(coords, 512, dict(distance=2.1, camera_angle_x=0.48))
        total = torch.zeros(128, 2)
        count = torch.zeros(128, dtype=torch.int32)
        weights = torch.zeros(128)
        values = []
        for b in data["blocks"]:
            v = torch.full((len(b["rows"]), 2), 1e6)
            v[b["owned"]] = 3.25
            values.append(v)
        shape.reduce_predictions(total, count, data["blocks"], values, weights)
        torch.testing.assert_close(total / weights[:, None], torch.full((128, 2), 3.25))


if __name__ == "__main__":
    unittest.main()
