import torch

from pixal3d.models.sc_vaes.sparse_unet_vae import subdivision_to_child_coords
from pixal3d_guided_endpoint_sr import _support_subdivisions


def test_support_subdivisions_round_trip_leaf_support_and_c256_parent():
    leaf = torch.tensor(
        [[0, 32, 48, 64], [0, 33, 48, 64], [0, 47, 63, 79]],
        dtype=torch.int32,
    )
    subs = _support_subdivisions(leaf, levels=4)
    assert len(subs) == 4
    assert torch.equal(subdivision_to_child_coords(subs[-1]), leaf)
    expected_c256 = torch.unique(
        torch.cat([leaf[:, :1], leaf[:, 1:] // 16], dim=1), dim=0, sorted=True
    )
    assert torch.equal(subs[0].coords, expected_c256)
