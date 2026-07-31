from typing import *
import torch
import torch.nn as nn
from ..basic import VarLenTensor, SparseTensor
from ..attention import SparseMultiHeadAttention, SparseProjectAttention, SparseGatedProjectAttention
from ...norm import LayerNorm32
from .blocks import SparseFeedForwardNet


class ModulatedSparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels ** 0.5)

    def _forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    
    Supports two image attention modes:
    - "cross": Standard cross-attention with image features
    - "proj": Projection-based attention with view-aligned features
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        image_attn_mode: Literal["cross", "proj", "gated_proj"] = "cross",
        proj_in_channels: Optional[int] = None,
        vae_in_channels: Optional[int] = None,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.image_attn_mode = image_attn_mode
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        
        # Build cross attention based on mode
        if image_attn_mode == "cross":
            self.cross_attn = SparseMultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                type="cross",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )
        elif image_attn_mode == "proj":
            _proj_in = proj_in_channels if proj_in_channels is not None else ctx_channels
            cross_attn_block = SparseMultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                type="cross",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )
            self.cross_attn = SparseProjectAttention(cross_attn_block, channels, _proj_in)
        elif image_attn_mode == "gated_proj":
            _dino_in = proj_in_channels if proj_in_channels is not None else ctx_channels
            _vae_in = vae_in_channels if vae_in_channels is not None else 16
            cross_attn_block = SparseMultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                type="cross",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )
            self.cross_attn = SparseGatedProjectAttention(cross_attn_block, channels, _dino_in, _vae_in)
        else:
            raise ValueError(f"Unknown image attention mode: {image_attn_mode}")
            
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels ** 0.5)
        self.last_routing_tensors: Optional[Dict[str, torch.Tensor]] = None

    @staticmethod
    def _masked_rms(
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        selected = value[mask]
        if selected.numel() == 0:
            return torch.zeros((), device=value.device, dtype=torch.float32)
        return selected.float().square().mean().sqrt()

    @staticmethod
    def _masked_mean_cosine(
        value: torch.Tensor,
        front: torch.Tensor,
    ) -> torch.Tensor:
        back = ~front
        if not bool(front.any()) or not bool(back.any()):
            return torch.zeros((), device=value.device, dtype=torch.float32)
        front_mean = value[front].float().mean(dim=0)
        back_mean = value[back].float().mean(dim=0)
        denominator = (
            torch.linalg.vector_norm(front_mean)
            * torch.linalg.vector_norm(back_mean)
        ).clamp_min(1e-12)
        return torch.dot(front_mean, back_mean) / denominator

    def _forward(self, x: SparseTensor, mod: torch.Tensor, context: Union[torch.Tensor, VarLenTensor]) -> SparseTensor:
        routed = (
            isinstance(context, dict)
            and context.get("mode") == "visibility_routed"
        )
        record_diagnostics = routed and bool(
            context.get("record_diagnostics", False)
        )
        front_mask = None
        if routed:
            front_mask = context["token_visibility"].to(x.device) > 0.5
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        self_attention_input = h
        h = self.self_attn(h)
        intervention_back_delta = None
        if (
            record_diagnostics
            and bool(context.get("record_self_attention_intervention", False))
            and front_mask is not None
            and bool(front_mask.any())
            and bool((~front_mask).any())
        ):
            intervened_input = self_attention_input.replace(
                self_attention_input.feats
                * (~front_mask).to(self_attention_input.dtype)[:, None]
            )
            intervened_output = self.self_attn(intervened_input)
            intervention_back_delta = self._masked_rms(
                h.feats - intervened_output.feats,
                ~front_mask,
            )
        h = h * gate_msa
        x = x + h
        after_self = x
        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context)
        x = x + h
        after_image = x
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        if record_diagnostics and front_mask is not None:
            back_mask = ~front_mask
            diagnostics = {
                "front_hidden_after_self_rms": self._masked_rms(
                    after_self.feats, front_mask
                ).detach(),
                "back_hidden_after_self_rms": self._masked_rms(
                    after_self.feats, back_mask
                ).detach(),
                "front_back_after_self_mean_cosine": (
                    self._masked_mean_cosine(after_self.feats, front_mask)
                    .detach()
                ),
                "front_hidden_after_image_rms": self._masked_rms(
                    after_image.feats, front_mask
                ).detach(),
                "back_hidden_after_image_rms": self._masked_rms(
                    after_image.feats, back_mask
                ).detach(),
                "front_back_after_image_mean_cosine": (
                    self._masked_mean_cosine(after_image.feats, front_mask)
                    .detach()
                ),
                "front_hidden_after_block_rms": self._masked_rms(
                    x.feats, front_mask
                ).detach(),
                "back_hidden_after_block_rms": self._masked_rms(
                    x.feats, back_mask
                ).detach(),
                "front_back_after_block_mean_cosine": (
                    self._masked_mean_cosine(x.feats, front_mask).detach()
                ),
            }
            if intervention_back_delta is not None:
                diagnostics[
                    "front_to_back_self_attention_intervention_rms"
                ] = intervention_back_delta.detach()
            attention_diagnostics = getattr(
                self.cross_attn, "last_routing_tensors", None
            )
            if attention_diagnostics:
                diagnostics.update(attention_diagnostics)
            self.last_routing_tensors = diagnostics
        # A following official CFG-negative pass uses an ordinary context.
        # Preserve the immediately preceding routed-positive diagnostics so
        # the sampler caller can collect them after the complete CFG call.
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: Union[torch.Tensor, VarLenTensor]) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, use_reentrant=False)
        else:
            return self._forward(x, mod, context)
