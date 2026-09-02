import pytest
import torch

from pixal3d.models.sc_vaes.sparse_unet_vae import (
    align_sparse_tensor_to_coords,
    subdivision_to_child_coords,
)
from pixal3d.modules.sparse import SparseTensor
from pixal3d.modules.sparse import SparseDownsample, SparseSpatial2Channel


def _sparse(feats, coords):
    return SparseTensor(
        torch.tensor(feats, dtype=torch.float32),
        torch.tensor(coords, dtype=torch.int32),
    )


def test_subdivision_to_child_coords_uses_decoder_bit_order():
    sub = _sparse([[1, -1, -1, -1, -1, -1, -1, 1]], [[0, 3, 4, 5]])
    children = subdivision_to_child_coords(sub)
    assert torch.equal(
        children,
        torch.tensor([[0, 6, 8, 10], [0, 7, 9, 11]], dtype=torch.int32),
    )


def test_align_sparse_tensor_follows_target_order_and_drops_extras():
    source = _sparse(
        [[10], [20], [30]],
        [[0, 1, 1, 1], [0, 2, 2, 2], [0, 3, 3, 3]],
    )
    target = torch.tensor([[0, 3, 3, 3], [0, 1, 1, 1]], dtype=torch.int32)
    aligned, stats = align_sparse_tensor_to_coords(source, target)
    assert torch.equal(aligned.coords, target)
    assert torch.equal(aligned.feats[:, 0], torch.tensor([30.0, 10.0]))
    assert stats == {
        "source": 3, "target": 2, "present": 2, "missing": 0, "dropped": 1,
    }


def test_align_sparse_tensor_missing_policy_is_explicit():
    source = _sparse([[10]], [[0, 1, 1, 1]])
    target = torch.tensor([[0, 1, 1, 1], [0, 9, 9, 9]], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="missing=1"):
        align_sparse_tensor_to_coords(source, target)
    aligned, stats = align_sparse_tensor_to_coords(source, target, missing="zeros")
    assert torch.equal(aligned.feats[:, 0], torch.tensor([10.0, 0.0]))
    assert stats["missing"] == 1


def test_c4096_downsampling_uses_int64_linear_keys_without_batch_overflow():
    coords = torch.tensor(
        [[0, 4094, 4092, 4090], [0, 4095, 4093, 4091]], dtype=torch.int32
    )
    sparse = SparseTensor(torch.arange(2, dtype=torch.float32)[:, None], coords)
    expected = torch.tensor([[0, 2047, 2046, 2045]], dtype=torch.int32)
    spatial = SparseSpatial2Channel(2)(sparse)
    pooled = SparseDownsample(2)(sparse)
    assert torch.equal(spatial.coords, expected)
    assert torch.equal(pooled.coords, expected)
