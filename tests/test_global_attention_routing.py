import torch

from pixal3d.experiments.global_attention_routing import (
    ParentCorrespondence,
    global_replace_and_residual,
    scatter_unique_rows,
)


def test_reduce_broadcast_and_variants():
    corr = ParentCorrespondence(torch.tensor([0, 0, 1, 1], dtype=torch.long), 2)
    fine = torch.tensor([[1.0], [3.0], [10.0], [14.0]])
    coarse, counts = corr.reduce_mean(fine)
    assert torch.equal(counts, torch.tensor([2, 2]))
    assert torch.equal(coarse, torch.tensor([[2.0], [12.0]]))
    replace, combined, residual = global_replace_and_residual(
        fine, torch.tensor([[20.0], [30.0]]), corr
    )
    assert torch.equal(replace, torch.tensor([[20.0], [20.0], [30.0], [30.0]]))
    assert torch.equal(residual, torch.tensor([[-1.0], [1.0], [-2.0], [2.0]]))
    assert torch.equal(combined, replace + residual)


def test_scatter_unique_rows_reorders_and_rejects_overlap():
    packed = torch.tensor([[30.0], [10.0], [20.0]])
    result = scatter_unique_rows(packed, torch.tensor([2, 0, 1]), 3)
    assert torch.equal(result[:, 0], torch.tensor([10.0, 20.0, 30.0]))
    try:
        scatter_unique_rows(packed, torch.tensor([0, 0, 2]), 3)
    except ValueError as exc:
        assert "duplicates=1" in str(exc)
        assert "missing=1" in str(exc)
    else:
        raise AssertionError("duplicate global rows must be rejected")


def test_attention_runtime_processor_replaces_pre_to_out_head_space():
    from pixal3d.modules.sparse import SparseTensor, config
    from pixal3d.modules.sparse.attention.modules import SparseMultiHeadAttention

    previous = config.ATTN
    config.ATTN = "sdpa"
    try:
        torch.manual_seed(7)
        attention = SparseMultiHeadAttention(8, num_heads=2, use_rope=False)
        coords = torch.tensor(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]], dtype=torch.int32
        )
        value = SparseTensor(torch.randn(3, 8), coords)
        ordinary = attention(value).feats

        attention.runtime_attention_processor = (
            lambda _module, _qkv, local: local.replace(local.feats.clone())
        )
        identity = attention(value).feats
        assert torch.equal(identity, ordinary)
        assert torch.count_nonzero(
            attention.runtime_last_intervention["head_delta"].feats
        ) == 0

        attention.runtime_attention_processor = (
            lambda _module, _qkv, local: local.replace(torch.zeros_like(local.feats))
        )
        replaced = attention(value).feats
        assert not torch.equal(replaced, ordinary)
        # Zero head output followed by pretrained to_out leaves exactly its bias.
        assert torch.allclose(replaced, attention.to_out.bias.expand_as(replaced))
        assert attention.runtime_last_intervention["to_out_delta"].shape == (1, 8)
    finally:
        config.ATTN = previous
