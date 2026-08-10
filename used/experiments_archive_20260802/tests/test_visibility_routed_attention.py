from __future__ import annotations

import torch
import torch.nn as nn

from pixal3d.models.structured_latent_flow import SLatFlowModel
from pixal3d.modules.sparse import SparseTensor
from pixal3d.modules.sparse.attention.proj_attention import (
    SparseProjectAttention,
)
import pixal3d.modules.sparse.config as sparse_attention_config


class _SharedContextBlock(nn.Module):
    """Deterministic cross-attention stand-in with shared batch context."""

    def forward(
        self,
        x: SparseTensor,
        context: torch.Tensor,
    ) -> SparseTensor:
        shared = context.mean(dim=1).index_select(
            0, x.coords[:, 0].to(torch.long)
        )
        return x.replace(x.feats + shared)


def _coords(rows: int = 4) -> torch.Tensor:
    return torch.tensor(
        [[0, index, index + 1, index + 2] for index in range(rows)],
        dtype=torch.int32,
    )


def _routed_context(
    *,
    coords: torch.Tensor,
    global_front: torch.Tensor,
    proj_front: SparseTensor,
    global_back: torch.Tensor,
    proj_back: SparseTensor,
    visibility: torch.Tensor,
    diagnostics: bool = False,
) -> dict:
    return {
        "mode": "visibility_routed",
        "global_front": global_front,
        "proj_front": proj_front,
        "global_back": global_back,
        "proj_back": proj_back,
        "token_visibility": visibility,
        "mask_coords": coords,
        "routing_kind": "hard",
        "record_diagnostics": diagnostics,
        "record_self_attention_intervention": diagnostics,
    }


def test_sparse_project_attention_routing_limits_and_rows() -> None:
    torch.manual_seed(7)
    coords = _coords()
    x = SparseTensor(torch.randn(4, 4), coords)
    proj_front = SparseTensor(torch.randn(4, 3), coords)
    proj_back = proj_front.replace(torch.zeros_like(proj_front.feats))
    global_front = torch.randn(1, 3, 4)
    global_back = torch.randn(1, 3, 4)
    module = SparseProjectAttention(_SharedContextBlock(), 4, 3)
    with torch.no_grad():
        module.proj_linear.weight.copy_(
            torch.tensor(
                [
                    [0.1, 0.2, 0.3],
                    [-0.3, 0.2, 0.1],
                    [0.4, -0.2, 0.5],
                    [0.7, 0.1, -0.4],
                ]
            )
        )
        # The zero-projected branch must retain this learned bias.
        module.proj_linear.bias.copy_(
            torch.tensor([0.25, -0.5, 0.75, -1.0])
        )

    local = module(
        x, {"global": global_front, "proj": proj_front}
    ).feats
    back = module(
        x, {"global": global_back, "proj": proj_back}
    ).feats
    zero = module(
        x,
        {
            "global": torch.zeros_like(global_front),
            "proj": proj_back,
        },
    ).feats

    all_front = module(
        x,
        _routed_context(
            coords=coords,
            global_front=global_front,
            proj_front=proj_front,
            global_back=global_back,
            proj_back=proj_back,
            visibility=torch.ones(4, dtype=torch.bool),
        ),
    ).feats
    assert torch.equal(all_front, local)

    all_back = module(
        x,
        _routed_context(
            coords=coords,
            global_front=global_front,
            proj_front=proj_front,
            global_back=global_back,
            proj_back=proj_back,
            visibility=torch.zeros(4, dtype=torch.bool),
        ),
    ).feats
    assert torch.equal(all_back, back)

    all_zero = module(
        x,
        _routed_context(
            coords=coords,
            global_front=global_front,
            proj_front=proj_front,
            global_back=torch.zeros_like(global_front),
            proj_back=proj_back,
            visibility=torch.zeros(4, dtype=torch.bool),
        ),
    ).feats
    assert torch.equal(all_zero, zero)
    assert torch.equal(
        module.proj_linear(torch.zeros_like(proj_front.feats))[0],
        module.proj_linear.bias,
    )

    mixed_mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    mixed = module(
        x,
        _routed_context(
            coords=coords,
            global_front=global_front,
            proj_front=proj_front,
            global_back=global_back,
            proj_back=proj_back,
            visibility=mixed_mask,
            diagnostics=True,
        ),
    ).feats
    expected = torch.where(mixed_mask[:, None], local, back)
    assert torch.equal(mixed, expected)
    assert module.last_routing_tensors is not None


def test_sparse_project_attention_rejects_coordinate_misalignment() -> None:
    coords = _coords()
    bad_coords = coords.clone()
    bad_coords[1, 1] += 1
    x = SparseTensor(torch.randn(4, 4), coords)
    front = SparseTensor(torch.randn(4, 3), coords)
    back = SparseTensor(torch.zeros(4, 3), bad_coords)
    module = SparseProjectAttention(_SharedContextBlock(), 4, 3)
    context = _routed_context(
        coords=coords,
        global_front=torch.randn(1, 2, 4),
        proj_front=front,
        global_back=torch.randn(1, 2, 4),
        proj_back=back,
        visibility=torch.ones(4, dtype=torch.bool),
    )
    try:
        module(x, context)
    except ValueError as error:
        assert "proj_back coordinates" in str(error)
    else:
        raise AssertionError("misaligned projected coordinates were accepted")


def _small_flow_model(in_channels: int) -> SLatFlowModel:
    torch.manual_seed(11 + in_channels)
    model = SLatFlowModel(
        resolution=64,
        in_channels=in_channels,
        model_channels=16,
        cond_channels=8,
        out_channels=32,
        num_blocks=2,
        num_heads=4,
        mlp_ratio=2.0,
        pe_mode="rope",
        dtype="float32",
        image_attn_mode="proj",
        proj_in_channels=12,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(101 + in_channels)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(
                torch.randn(
                    parameter.shape,
                    generator=generator,
                    dtype=parameter.dtype,
                )
                * 0.03
            )
    return model.eval()


def _full_model_limit_check(
    model: SLatFlowModel,
    *,
    texture: bool,
) -> None:
    coords = _coords()
    state_channels = 32
    state = SparseTensor(torch.randn(4, state_channels), coords)
    concat = (
        SparseTensor(torch.randn(4, 32), coords) if texture else None
    )
    global_front = torch.randn(1, 5, 8)
    global_back = torch.randn(1, 5, 8)
    proj_front = SparseTensor(torch.randn(4, 12), coords)
    proj_back = proj_front.replace(torch.zeros_like(proj_front.feats))
    timestep = torch.tensor([500.0])
    kwargs = {"concat_cond": concat} if concat is not None else {}

    local = model(
        state,
        timestep,
        {"global": global_front, "proj": proj_front},
        **kwargs,
    )
    routed_front = model(
        state,
        timestep,
        _routed_context(
            coords=coords,
            global_front=global_front,
            proj_front=proj_front,
            global_back=global_back,
            proj_back=proj_back,
            visibility=torch.ones(4, dtype=torch.bool),
        ),
        **kwargs,
    )
    assert torch.equal(local.coords, routed_front.coords)
    assert float(
        (local.feats - routed_front.feats).abs().max().item()
    ) < 1e-6

    back = model(
        state,
        timestep,
        {"global": global_back, "proj": proj_back},
        **kwargs,
    )
    routed_back = model(
        state,
        timestep,
        _routed_context(
            coords=coords,
            global_front=global_front,
            proj_front=proj_front,
            global_back=global_back,
            proj_back=proj_back,
            visibility=torch.zeros(4, dtype=torch.bool),
        ),
        **kwargs,
    )
    assert float(
        (back.feats - routed_back.feats).abs().max().item()
    ) < 1e-6

    mixed = model(
        state,
        timestep,
        _routed_context(
            coords=coords,
            global_front=global_front,
            proj_front=proj_front,
            global_back=global_back,
            proj_back=proj_back,
            visibility=torch.tensor([1, 0, 1, 0], dtype=torch.bool),
            diagnostics=True,
        ),
        **kwargs,
    )
    assert torch.isfinite(mixed.feats).all()
    assert all(block.last_routing_tensors for block in model.blocks)
    assert all(
        "front_to_back_self_attention_intervention_rms"
        in block.last_routing_tensors
        for block in model.blocks
    )


def test_full_shape_and_texture_models_obey_routing_limits() -> None:
    previous_backend = sparse_attention_config.ATTN
    sparse_attention_config.ATTN = "sdpa"
    try:
        _full_model_limit_check(_small_flow_model(32), texture=False)
        _full_model_limit_check(_small_flow_model(64), texture=True)
    finally:
        sparse_attention_config.ATTN = previous_backend
