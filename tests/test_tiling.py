import unittest
import torch
from sr_tools import shape as s
from sr_tools import texture as tex


class BatchContracts(unittest.TestCase):
    def test_xy_partition_z_overlap_and_texture_mapping(self):
        for stage, grid, context, width in (
            (512, 128, 2048, 32),
            (1024, 256, 1024, 64),
        ):
            coords = torch.tensor(
                [
                    [0, grid // 2, grid // 2, z]
                    for z in (0, width // 2, width, grid - 1)
                ],
                dtype=torch.int32,
            )
            data = s.construct(
                coords,
                stage,
                {"distance": 2.1, "camera_angle_x": 0.48},
                padding=32,
                image_stride=context,
                depth_stride=width // 2,
            )
            self.assertEqual(len(data["blocks"]), 28 if stage == 512 else 112)
            self.assertEqual(data["counts"].tolist(), [1, 2, 2, 1])
            tex.validate_geometry(
                dict(coords=coords, features=torch.ones(4, 2), normalized=True), data
            )
            self.assertTrue(
                all(
                    not (
                        b["owned"] & ~data["tiles"][b["tile_id"]]["core"][b["rows"]]
                    ).any()
                    for b in data["blocks"]
                )
            )


if __name__ == "__main__":
    unittest.main()
