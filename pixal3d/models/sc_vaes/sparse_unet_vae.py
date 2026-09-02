from typing import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from ...modules.utils import convert_module_to_f16, convert_module_to_f32, zero_module
from ...modules import sparse as sp
from ...modules.norm import LayerNorm32


def subdivision_to_child_coords(subdivision: sp.SparseTensor) -> torch.Tensor:
    """Expand one decoder subdivision tensor into its selected child coords."""
    if not isinstance(subdivision, sp.SparseTensor):
        raise TypeError("subdivision must be a SparseTensor")
    if subdivision.coords.ndim != 2 or subdivision.coords.shape[1] != 4:
        raise ValueError("subdivision coords must have shape [N, 4]")
    if subdivision.feats.ndim != 2 or subdivision.feats.shape[1] != 8:
        raise ValueError("3-D subdivision feats must have shape [N, 8]")
    active = (subdivision.feats > 0).nonzero(as_tuple=False)
    if active.numel() == 0:
        raise ValueError("guided subdivision selects no child coordinates")
    parent_rows = active[:, 0]
    child_index = active[:, 1]
    coords = subdivision.coords.index_select(0, parent_rows).clone()
    coords[:, 1:] *= 2
    coords[:, 1] += child_index % 2
    coords[:, 2] += (child_index // 2) % 2
    coords[:, 3] += (child_index // 4) % 2
    return coords


def align_sparse_tensor_to_coords(
    x: sp.SparseTensor,
    target_coords: torch.Tensor,
    *,
    missing: Literal['error', 'zeros'] = 'error',
) -> Tuple[sp.SparseTensor, Dict[str, int]]:
    """Reindex a sparse tensor onto an exact caller-selected support.

    Extra source rows are discarded. Missing target rows either fail loudly
    or are materialized with zero features. A fresh tensor prevents stale
    convolution/spatial caches from leaking into the guided support.
    """
    if not isinstance(x, sp.SparseTensor):
        raise TypeError("x must be a SparseTensor")
    if missing not in {'error', 'zeros'}:
        raise ValueError("missing must be 'error' or 'zeros'")
    target_coords = target_coords.to(device=x.coords.device, dtype=x.coords.dtype)
    if target_coords.ndim != 2 or target_coords.shape[1] != x.coords.shape[1]:
        raise ValueError("target_coords must have the same coordinate rank as x.coords")
    if target_coords.shape[0] == 0:
        raise ValueError("target_coords must not be empty")

    if torch.equal(x.coords, target_coords):
        count = int(x.coords.shape[0])
        return x, {
            'source': count,
            'target': count,
            'present': count,
            'missing': 0,
            'dropped': 0,
        }

    # A row-wise torch.unique join becomes both memory-heavy and unreliable at
    # tens of millions of 4-D coordinates. Encode each coordinate into one
    # collision-free int64 key and use a sorted search join instead.
    max_coord = torch.maximum(
        x.coords.amax(dim=0), target_coords.amax(dim=0)
    ).to(torch.int64)
    base = int(max_coord[1:].amax().item()) + 1

    def coordinate_keys(coords: torch.Tensor) -> torch.Tensor:
        values = coords.to(torch.int64)
        key = values[:, 0]
        for axis in range(1, values.shape[1]):
            key = key * base + values[:, axis]
        return key

    source_keys = coordinate_keys(x.coords)
    target_keys = coordinate_keys(target_coords)
    sorted_keys, order = torch.sort(source_keys)
    if sorted_keys.numel() > 1 and bool((sorted_keys[1:] == sorted_keys[:-1]).any().item()):
        raise ValueError("source sparse coordinates contain duplicates")
    sorted_target_keys = torch.sort(target_keys).values
    if sorted_target_keys.numel() > 1 and bool(
        (sorted_target_keys[1:] == sorted_target_keys[:-1]).any().item()
    ):
        raise ValueError("guided target coordinates contain duplicates")
    del sorted_target_keys
    positions = torch.searchsorted(sorted_keys, target_keys)
    inside = positions < sorted_keys.shape[0]
    safe_positions = positions.clamp_max(sorted_keys.shape[0] - 1)
    present = inside & (sorted_keys.index_select(0, safe_positions) == target_keys)
    source_rows = torch.full_like(positions, -1)
    source_rows[present] = order.index_select(0, positions[present])
    missing_count = int((~present).sum().item())
    if missing_count and missing == 'error':
        raise RuntimeError(
            "guided encoder support is absent from its input: "
            f"missing={missing_count} target={target_coords.shape[0]}"
        )

    feats = torch.zeros(
        (target_coords.shape[0], *x.feats.shape[1:]),
        device=x.feats.device, dtype=x.feats.dtype,
    )
    if bool(present.any().item()):
        feats[present] = x.feats.index_select(0, source_rows[present])
    out = sp.SparseTensor(feats=feats, coords=target_coords)
    out._scale = x._scale
    present_count = int(present.sum().item())
    stats = {
        'source': int(x.coords.shape[0]),
        'target': int(target_coords.shape[0]),
        'present': present_count,
        'missing': missing_count,
        'dropped': int(x.coords.shape[0] - present_count),
    }
    return out, stats


class SparseResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
        resample_mode: Literal['nearest', 'spatial2channel'] = 'nearest',
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        self.resample_mode = resample_mode
        self.use_checkpoint = use_checkpoint
        
        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        if resample_mode == 'nearest':
            self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        elif resample_mode =='spatial2channel' and not self.downsample:
            self.conv1 = sp.SparseConv3d(channels, self.out_channels * 8, 3)
        elif resample_mode =='spatial2channel' and self.downsample:
            self.conv1 = sp.SparseConv3d(channels, self.out_channels // 8, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        if resample_mode == 'nearest':
            self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        elif resample_mode =='spatial2channel' and self.downsample:
            self.skip_connection = lambda x: x.replace(x.feats.reshape(x.feats.shape[0], out_channels, channels * 8 // out_channels).mean(dim=-1))
        elif resample_mode =='spatial2channel' and not self.downsample:
            self.skip_connection = lambda x: x.replace(x.feats.repeat_interleave(out_channels // (channels // 8), dim=1))
        self.updown = None
        if self.downsample:
            if resample_mode == 'nearest':
                self.updown = sp.SparseDownsample(2)
            elif resample_mode =='spatial2channel':
                self.updown = sp.SparseSpatial2Channel(2)
        elif self.upsample:
            self.to_subdiv = sp.SparseLinear(channels, 8)
            if resample_mode == 'nearest':
                self.updown = sp.SparseUpsample(2)
            elif resample_mode =='spatial2channel':
                self.updown = sp.SparseChannel2Spatial(2)

    def _updown(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.downsample:
            x = self.updown(x)
        elif self.upsample:
            x = self.updown(x, subdiv.replace(subdiv.feats > 0))
        return x

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        subdiv = None
        if self.upsample:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        if self.resample_mode == 'spatial2channel':
            h = self.conv1(h)
        h = self._updown(h, subdiv)
        x = self._updown(x, subdiv)
        if self.resample_mode == 'nearest':
            h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        if self.upsample:
            return h, subdiv
        return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockDownsample3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        self.updown = sp.SparseDownsample(2)

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.updown(h)
        x = self.updown(x)
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockUpsample3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.pred_subdiv = pred_subdiv
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        if self.pred_subdiv:
            self.to_subdiv = sp.SparseLinear(channels, 8)
        self.updown = sp.SparseUpsample(2)

    def _forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.pred_subdiv and subdiv is None:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        subdiv_binarized = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        h = self.updown(h, subdiv_binarized)
        x = self.updown(x, subdiv_binarized)
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        if self.pred_subdiv:
            return h, subdiv
        else:
            return h
    
    def forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, subdiv, use_reentrant=False
            )
        else:
            return self._forward(x, subdiv)


def print_gpu_mem(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        print(
            f"[GPU MEM] {tag} | "
            f"allocated={allocated:.2f} GB | "
            f"reserved={reserved:.2f} GB | "
            f"max_allocated={max_allocated:.2f} GB"
        )

class SparseResBlockS2C3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels // 8, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = lambda x: x.replace(x.feats.reshape(x.feats.shape[0], out_channels, channels * 8 // out_channels).mean(dim=-1))
        self.updown = sp.SparseSpatial2Channel(2)

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = self.updown(h)
        x = self.updown(x)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockC2S3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.pred_subdiv = pred_subdiv
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels * 8, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = lambda x: x.replace(x.feats.repeat_interleave(out_channels // (channels // 8), dim=1))
        if pred_subdiv:
            self.to_subdiv = sp.SparseLinear(channels, 8)
        self.updown = sp.SparseChannel2Spatial(2)

    def _forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.pred_subdiv and subdiv is None:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        subdiv_binarized = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        h = self.updown(h, subdiv_binarized)
        x = self.updown(x, subdiv_binarized)     
        print_gpu_mem("After updown")
        low_peak = (
            os.environ.get("PIXAL3D_LOW_MEMORY_DECODER", "0") == "1"
            and not torch.is_grad_enabled()
            and h.feats.numel() >= 64 * 1024 * 1024
        )
        if low_peak:
            # Earlier-resolution neighbor maps are carried through the sparse
            # tensor cache but cannot be reused after this C2S scale change.
            # Detach this resolution from those caches before the final SubM
            # convolution builds its own neighbor map.
            h.clear_spatial_cache()
            x.clear_spatial_cache()
            # LayerNorm32 otherwise materializes the complete activation in
            # FP32.  The final 4096 stage can make that temporary tens of GB.
            # Norm2 has no affine parameters here, so row chunks are exactly
            # equivalent and can safely overwrite h before conv2 consumes it.
            rows = max(1, (128 * 1024 * 1024) // max(1, h.feats.shape[1] * 4))
            for begin in range(0, h.feats.shape[0], rows):
                end = min(h.feats.shape[0], begin + rows)
                normalized = F.layer_norm(
                    h.feats[begin:end].float(), self.norm2.normalized_shape,
                    None, None, self.norm2.eps)
                h.feats[begin:end].copy_(normalized.to(h.feats.dtype))
                del normalized
        else:
            h = h.replace(self.norm2(h.feats))
        print_gpu_mem("After norm2")
        h = h.replace(F.silu(h.feats, inplace=low_peak))
        h = self.conv2(h)
        if low_peak:
            repeat = self.out_channels // (self.channels // 8)
            rows = max(1, (128 * 1024 * 1024) // max(1, self.out_channels * h.feats.element_size()))
            for begin in range(0, h.feats.shape[0], rows):
                end = min(h.feats.shape[0], begin + rows)
                residual = x.feats[begin:end].repeat_interleave(repeat, dim=1)
                h.feats[begin:end].add_(residual)
                del residual
        else:
            h = h + self.skip_connection(x)
        if self.pred_subdiv:
            return h, subdiv
        else:
            return h
    
    def forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, subdiv, use_reentrant=False)
        else:
            return self._forward(x, subdiv)
        
    
class SparseConvNeXtBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.use_checkpoint = use_checkpoint
        
        self.norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.conv = sp.SparseConv3d(channels, channels, 3)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.SiLU(),
            zero_module(nn.Linear(int(channels * mlp_ratio), channels)),
        )

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.conv(x)
        h = h.replace(self.norm(h.feats))
        h = h.replace(self.mlp(h.feats))
        return h + x
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseUnetVaeEncoder(nn.Module):
    """
    Sparse Swin Transformer Unet VAE model.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        down_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = sp.SparseLinear(in_channels, model_channels[0])
        self.to_latent = sp.SparseLinear(model_channels[-1], 2 * latent_channels)
        
        self.blocks = nn.ModuleList([])
        for i in range(len(num_blocks)):
            self.blocks.append(nn.ModuleList([]))
            for j in range(num_blocks[i]):
                self.blocks[-1].append(
                    globals()[block_type[i]](
                        model_channels[i],
                        **block_args[i],
                    )
                )
            if i < len(num_blocks) - 1:
                self.blocks[-1].append(
                    globals()[down_block_type[i]](
                        model_channels[i],
                        model_channels[i+1],
                        **block_args[i],
                    )
                )
                
        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(
        self,
        x: sp.SparseTensor,
        sample_posterior=False,
        return_raw=False,
        guide_subs: Optional[List[sp.SparseTensor]] = None,
        guide_missing: Literal['error', 'zeros'] = 'error',
        return_guide_diagnostics: bool = False,
    ):
        """Encode, optionally forcing every level onto decoder topology.

        ``guide_subs`` must be the coarse-to-fine list returned by the shape
        decoder. The final encoded coordinates then exactly equal the original
        coarse SLat coordinates in ``guide_subs[0]``.
        """
        guide_diagnostics = None
        if guide_subs is not None:
            expected_levels = len(self.blocks) - 1
            if len(guide_subs) != expected_levels:
                raise ValueError(
                    f"guide_subs must contain {expected_levels} levels, "
                    f"got {len(guide_subs)}"
                )
            for level, sub in enumerate(guide_subs):
                if not isinstance(sub, sp.SparseTensor):
                    raise TypeError(f"guide_subs[{level}] is not a SparseTensor")
            leaf_coords = subdivision_to_child_coords(guide_subs[-1])
            x, input_stats = align_sparse_tensor_to_coords(
                x, leaf_coords, missing=guide_missing
            )
            guide_diagnostics = {
                'mode': 'decoder_subdivision_support',
                'missing_policy': guide_missing,
                'input_leaf_alignment': input_stats,
                'downsample_alignments': [],
            }
        h = self.input_layer(x)
        h = h.type(self.dtype)
        for i, res in enumerate(self.blocks):
            for j, block in enumerate(res):
                h = block(h)
                if (
                    guide_subs is not None
                    and i < len(self.blocks) - 1
                    and j == len(res) - 1
                ):
                    guide_level = len(guide_subs) - 1 - i
                    h, level_stats = align_sparse_tensor_to_coords(
                        h, guide_subs[guide_level].coords, missing=guide_missing
                    )
                    level_stats['encoder_stage'] = int(i)
                    level_stats['guide_level'] = int(guide_level)
                    guide_diagnostics['downsample_alignments'].append(level_stats)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.to_latent(h)
        
        # Sample from the posterior distribution
        mean, logvar = h.feats.chunk(2, dim=-1)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        z = h.replace(z)
            
        if guide_subs is not None and not torch.equal(
            z.coords, guide_subs[0].coords.to(z.coords.device)
        ):
            raise RuntimeError("guided encoder failed to preserve the requested SLat support")
        result = (z, mean, logvar) if return_raw else z
        if return_guide_diagnostics:
            return result, guide_diagnostics
        return result
    
    
class SparseUnetVaeDecoder(nn.Module):
    """
    Sparse Swin Transformer Unet VAE model.
    """
    def __init__(
        self,
        out_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.use_fp16 = use_fp16
        self.pred_subdiv = pred_subdiv
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.low_vram = False
        
        self.output_layer = sp.SparseLinear(model_channels[-1], out_channels)
        self.from_latent = sp.SparseLinear(latent_channels, model_channels[0])
        
        self.blocks = nn.ModuleList([])
        for i in range(len(num_blocks)):
            self.blocks.append(nn.ModuleList([]))
            for j in range(num_blocks[i]):
                self.blocks[-1].append(
                    globals()[block_type[i]](
                        model_channels[i],
                        **block_args[i],
                    )
                )
            if i < len(num_blocks) - 1:
                self.blocks[-1].append(
                    globals()[up_block_type[i]](
                        model_channels[i],
                        model_channels[i+1],
                        pred_subdiv=pred_subdiv,
                        **block_args[i],
                    )
                )
                    
        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
            
    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: sp.SparseTensor, guide_subs: Optional[List[sp.SparseTensor]] = None, return_subs: bool = False) -> sp.SparseTensor:
        assert return_subs == False or self.pred_subdiv == True, "Only decoders with pred_subdiv=True can be used with return_subs"
        if guide_subs is not None and len(guide_subs) != len(self.blocks) - 1:
            raise ValueError(
                f"guide_subs must contain {len(self.blocks) - 1} levels, "
                f"got {len(guide_subs)}"
            )
        
        h = self.from_latent(x)
        h = h.type(self.dtype)
        subs_gt = []
        subs = []
        for i, res in enumerate(self.blocks):
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    if self.pred_subdiv:
                        if self.training:
                            subs_gt.append(h.get_spatial_cache('subdivision'))
                        forced_sub = guide_subs[i] if guide_subs is not None else None
                        if forced_sub is not None:
                            forced_sub, _ = align_sparse_tensor_to_coords(
                                forced_sub.to(h.device), h.coords, missing='error'
                            )
                        h, sub = block(h, subdiv=forced_sub)
                        subs.append(sub)
                    else:
                        h = block(h, subdiv=guide_subs[i] if guide_subs is not None else None)
                else:
                    h = block(h)
            if (
                os.environ.get("PIXAL3D_LOW_MEMORY_DECODER", "0") == "1"
                and not torch.is_grad_enabled()
                and i < len(self.blocks) - 1
            ):
                # Completed decoder stages are never revisited.  Offloading
                # their weights leaves more room for the final C4096 neighbor
                # map while preserving the single global forward pass.
                res.cpu()
                torch.cuda.empty_cache()
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.output_layer(h)
        if self.training and self.pred_subdiv:
            return h, subs_gt, subs
        else:
            if return_subs:
                return h, subs
            else:
                return h
    
    def upsample(self, x: sp.SparseTensor, upsample_times: int) -> torch.Tensor:
        assert self.pred_subdiv == True, "Only decoders with pred_subdiv=True can be used with upsampling"
        
        h = self.from_latent(x)
        h = h.type(self.dtype)
        for i, res in enumerate(self.blocks):
            if i == upsample_times:
                return h.coords
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    h, sub = block(h)
                else:
                    h = block(h)
