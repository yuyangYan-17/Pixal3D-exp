import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pixal3d_cascade512_1024_tiled2048_crop_condition as impl


def test_square_aligned_box_is_square_aligned_and_contains_clipped_bbox():
    points = torch.tensor([[101.2, 209.8], [378.1, 330.4]])
    box, stats = impl.square_aligned_box(points)
    x0, y0, x1, y1 = box
    assert x1 - x0 == y1 - y0
    assert (x1 - x0) % 16 == 0
    assert x0 <= 101.2 and y0 <= 209.8
    assert x1 >= 378.1 and y1 >= 330.4
    assert stats["square"] is True


def test_square_aligned_box_clamps_at_image_edge():
    points = torch.tensor([[-10.0, 3900.0], [200.0, 4200.0]])
    box, _ = impl.square_aligned_box(points)
    x0, y0, x1, y1 = box
    assert x0 == 0
    assert y1 == 4096
    assert x1 - x0 == y1 - y0
    assert (x1 - x0) % 16 == 0


def test_nonoverlap_cube_partition_writes_each_row_once(monkeypatch):
    monkeypatch.setattr(impl, "cube_crop", lambda start, camera: {"start": tuple(start)})
    xyz = torch.tensor(
        [[0, 0, 0], [63, 63, 63], [64, 64, 64], [127, 127, 127], [10, 70, 90]],
        dtype=torch.int32,
    )
    coords = torch.cat((torch.zeros((xyz.shape[0], 1), dtype=torch.int32), xyz), 1)
    records = impl.build_records(coords, {})
    rows = torch.cat([rec["global_row_ids"] for rec in records])
    assert len(records) == 8
    assert torch.equal(torch.sort(rows).values, torch.arange(coords.shape[0]))
    for rec in records:
        local = rec["local_coords"][:, 1:]
        assert not local.numel() or bool(((local >= 0) & (local < 64)).all())
