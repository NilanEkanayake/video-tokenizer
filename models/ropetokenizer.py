import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from einops import einsum, rearrange, reduce

"""
References:
https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_ltx.py
https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_lumina2.py
https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py
"""

def apply_rotary_emb(x, freqs_cis):
    with torch.autocast(x.device.type, enabled=False):
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(-2) # unsqueeze head dim -> [L, H, D]
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(-2)

    return x_out.type_as(x)


def get_1d_rotary_pos_embed(dim, pos, theta=10000.0, freqs_dtype=torch.float64): 
    assert dim % 2 == 0

    if type(pos) is int:
        pos = torch.arange(pos)

    start = 1.0
    end = theta

    freqs = theta ** torch.linspace(
        math.log(start, theta), # 0.0?
        math.log(end, theta), # 1.0?
        dim//2,
        device=pos.device,
        dtype=freqs_dtype,
    )
    freqs = freqs * math.pi / 2.0
    freqs = freqs * pos.unsqueeze(-1)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis
    

class Lumina2RotaryPosEmbed(nn.Module):
    def __init__(
            self,
            theta: float = 10000.0,
            axes_dim: List[int] = (24, 20, 20),
            axes_lens: Optional[List[int]] = None,
        ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.axes_lens = axes_lens

        if self.axes_lens is not None:
            self.freqs_cis = self._precompute_freqs_cis(axes_dim, axes_lens, theta)

    def _precompute_freqs_cis(self, axes_dim: List[int], axes_lens: List[int], theta: int) -> List[torch.Tensor]:
        freqs_cis = []
        for d, e in zip(axes_dim, axes_lens):
            emb = get_1d_rotary_pos_embed(d, e, theta=theta)
            freqs_cis.append(emb)
        return freqs_cis

    def _get_precomputed_freqs_cis(self, ids: torch.Tensor) -> torch.Tensor:
        device = ids.device

        result = []
        for i in range(len(self.axes_dim)):
            freqs = self.freqs_cis[i].to(ids.device)
            index = ids[:, i : i + 1].repeat(1, freqs.shape[-1])
            result.append(torch.gather(freqs, dim=0, index=index))
        return torch.cat(result, dim=-1).to(device)
    
    def _get_freqs_cis(self, ids: torch.Tensor) -> torch.Tensor:
        device = ids.device

        result = []
        for i in range(len(self.axes_dim)):
            freqs = get_1d_rotary_pos_embed(self.axes_dim[i], ids[:, i], theta=self.theta).to(ids.device)
            result.append(freqs)
        return torch.cat(result, dim=-1).to(device)


    def forward(self, in_grid: Tuple[int, int, int], num_tokens: int, device: str):
        frames, height, width = in_grid
        seq_len = math.prod(in_grid) + num_tokens

        # Create position IDs -> [L, 3], all zeros. 3 = [frames, height, width]. Tokens are packed into THW dims like orig m-rope.
        position_ids = torch.zeros(seq_len, len(in_grid), dtype=torch.int64, device=device)
        position_ids[:num_tokens] = torch.arange(num_tokens, dtype=torch.int64, device=device).unsqueeze(-1) # assign to THW dims.

        # add THW position ids
        position_ids[num_tokens:, 0] = ( # frames
            torch.arange(frames, dtype=torch.int64, device=device)
            .view(-1, 1, 1)
            .repeat(1, height, width)
            .flatten()
        )

        position_ids[num_tokens:, 1] = ( # height
            torch.arange(height, dtype=torch.int64, device=device)
            .view(1, -1, 1)
            .repeat(frames, 1, width)
            .flatten()
        )

        position_ids[num_tokens:, 2] = ( # width
            torch.arange(width, dtype=torch.int64, device=device)
            .view(1, 1, -1)
            .repeat(frames, height, 1)
            .flatten()
        )

        position_ids[num_tokens:] += num_tokens # offset THW to increment from 1D enc

        if self.axes_lens is not None and self.training: # instead, check if in_grid less than the max precomputed/axes lens?
            freqs_cis = self._get_precomputed_freqs_cis(position_ids) # use precomputed
        else:
            freqs_cis = self._get_freqs_cis(position_ids)

        return freqs_cis


class RoPE(nn.Module):
    def __init__(
            self,
            theta=10000.0,
            head_dim=64,
            max_grid=[16, 64, 64],
            max_tokens=2048,
        ):
        super(RoPE, self).__init__()
        axes_dim = head_dim/len(max_grid)
        axes_dim = [int(axes_dim - (axes_dim % 2))] * len(max_grid)
        axes_dim[0] += head_dim - sum(axes_dim) # add remainder to T dim

        axes_lens = [x + max_tokens for x in max_grid]

        self.pos_emb = Lumina2RotaryPosEmbed(theta, axes_dim, axes_lens)


    def forward(self, grids, token_counts, device):
        with torch.autocast(device.type, enabled=False):
            freqs = []
            for grid, token_count in zip(grids, token_counts):
                freqs.append(self.pos_emb(grid, token_count, device))

            freqs = torch.cat(freqs, dim=0) # [B*L, C]

        return freqs
    


from flash_attn import flash_attn_varlen_func
from einops import rearrange


"""
Modified from: https://github.com/westlake-repl/LeanVAE/blob/master/LeanVAE/modules/backbones.py
"""

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x
    
    
def ffd(dim, mult=4, mult_of=32, dropout=0.):
    inner_dim = int(mult * (2 / 3) * dim)
    inner_dim = mult_of * ((inner_dim + mult_of - 1) // mult_of)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
        # nn.LayerNorm(dim), # another LN to fix instability
    )

class Attn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.q_heads, self.kv_heads = heads
        self.head_dim = dim//self.q_heads
        self.gqa_dim = self.head_dim * self.kv_heads

        self.pre_ln = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, self.gqa_dim * 2 + dim, bias=False)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs, cu_seqlens, max_seqlen):
        x = self.pre_ln(x)
        q, k, v = self.to_qkv(x).split([self.dim, self.gqa_dim, self.gqa_dim], dim=-1)

        q = q.unflatten(-1, (self.q_heads, self.head_dim)) # [L, H_Q, D_H]
        k = k.unflatten(-1, (self.kv_heads, self.head_dim)) # [L, H_KV, D_H]
        v = v.unflatten(-1, (self.kv_heads, self.head_dim))

        q = self.q_norm(q).to(v) # why is the cast needed?
        k = self.k_norm(k).to(v)

        q = apply_rotary_emb(q, freqs)
        k = apply_rotary_emb(k, freqs)

        x = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen)

        return self.out_proj(x.flatten(-2)) # flatten to [L, D]


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            embed_dim=512,
            heads=[8, 2],
            mlp_ratio=4,
            num_layer=2,
        ): 
        super(ResidualAttentionBlock, self).__init__()
        self.num_layer = num_layer
        self.attn_layer = nn.Sequential()
        self.ffd_layer = nn.Sequential()
        for _ in range(num_layer):
            self.attn_layer.append(Attn(embed_dim, heads))
            self.ffd_layer.append(ffd(embed_dim, mlp_ratio)) 
   
    def forward(self, x, freqs, cu_seqlens, max_seqlen):
        for i in range(self.num_layer):
            x = x + self.attn_layer[i](x.contiguous(), freqs, cu_seqlens, max_seqlen)
            x = x + self.ffd_layer[i](x.contiguous()) 
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
import math


def get_model_dims(model_size='tiny', head_dim=64, mlp_ratio=4.0):
    if model_size.endswith('_thin'): # https://arxiv.org/pdf/2505.20802
        model_size = model_size[:-5]
        layers = {
            "tiny": 2,
            "small": 5,
            "base": 7,
            "large": 8,
        }[model_size]
        heads = {
            "tiny": [8, 2], # Q heads, KV heads | GQA
            "small": [12, 4],
            "base": [16, 4],
            "large": [32, 8],
        }[model_size]
        mlp_ratio = mlp_ratio/2
    else:
        layers = {
            "tiny": 4,
            "small": 8,
            "base": 12,
            "large": 24,
        }[model_size]
        heads = {
            "tiny": [4, 2],
            "small": [8, 2],
            "base": [12, 4],
            "large": [16, 4],
        }[model_size]

    width = int(head_dim*heads[0])

    return width, layers, heads, mlp_ratio


def init_weights(module):
    if isinstance(module, nn.Linear): # SNLinear has internal init.
        module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
        if module.weight is not None:
            nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)
        
    
class TiTokEncoder(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            patch_size=(4, 8, 8),
            in_channels=3,
            out_channels=5,
            max_grid=(32, 256, 256),
            max_tokens=2048,
        ):
        super().__init__()
        self.patch_size = patch_size
        self.token_size = out_channels
        self.in_channels = in_channels

        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        self.rope = RoPE(
            head_dim=self.width//self.heads[0],
            max_grid=[x//y for x, y in zip(max_grid, patch_size)],
            max_tokens=max_tokens,
        )

        self.mask_token = nn.Parameter(scale * torch.randn(1, self.width)) # LC
        self.proj_in = nn.Linear(in_features=in_channels*math.prod(patch_size), out_features=self.width)

        self.model_layers = ResidualAttentionBlock(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers
        )

        self.ln_post = nn.LayerNorm(self.width)
        self.proj_out = nn.Linear(self.width, self.token_size, bias=True)

    def forward(self, x, token_counts):
        device = x[0].device
        grids = [[dim//ps for dim, ps in zip(vid.shape[1:], self.patch_size)] for vid in x]
        grid_sizes = [math.prod(g) for g in grids]
        seq_lens = [g + t for g, t in zip(grid_sizes, token_counts)]

        # cu_seqlens contains the indices of the packed samples.
        cu_seqlens = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seq_lens), dim=0)]).to(device, dtype=torch.int32)

        x = [rearrange(vid, 'c (nt pt) (nh ph) (nw pw) -> (nt nh nw) (c pt ph pw)', pt=self.patch_size[0], ph=self.patch_size[1], pw=self.patch_size[2]) for vid in x] # patching
        x = torch.cat(x, dim=0) # LC tensor
        x = self.proj_in(x) # returns LC

        x = torch.split(x, grid_sizes, dim=0)
        masked_tokens = [self.mask_token.repeat(num, 1) for num in token_counts]
        x = [torch.cat([tokens, vid], dim=0) for tokens, vid in zip(masked_tokens, x)]
        x = torch.cat(x, dim=0)

        freqs = self.rope(grids, token_counts, device)
        x = self.model_layers(x, freqs, cu_seqlens, max(seq_lens))

        x = torch.split(x, seq_lens, dim=0)
        x = [tokens[:num] for tokens, num in zip(x, token_counts)]
        x = torch.cat(x, dim=0)

        x = self.ln_post(x)
        x = self.proj_out(x)
        
        return x # packed LC, unpack or leave as-is for FSQ?


class TiTokDecoder(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            patch_size=(4, 8, 8),
            in_channels=5,
            out_channels=3,
            max_grid=(32, 256, 256),
            max_tokens=2048,
        ):
        super().__init__()
        self.patch_size = patch_size
        self.token_size = in_channels
        self.out_channels = out_channels

        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        self.rope = RoPE(
            head_dim=self.width//self.heads[0],
            max_grid=[x//y for x, y in zip(max_grid, patch_size)],
            max_tokens=max_tokens,
        )

        self.mask_token = nn.Parameter(scale * torch.randn(1, self.width)) # LC
        self.proj_in = nn.Linear(self.token_size, self.width, bias=True)
        self.ln_pre = nn.LayerNorm(self.width)

        self.model_layers = ResidualAttentionBlock(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers
        )

        self.proj_out = nn.Linear(in_features=self.width, out_features=out_channels*math.prod(patch_size))


    def forward(self, x, token_counts, grids):
        device = x.device
        grids = [[dim//ps for dim, ps in zip(grid, self.patch_size)] for grid in grids]
        grid_sizes = [math.prod(grid) for grid in grids]
        seq_lens = [g + t for g, t in zip(grid_sizes, token_counts)]
        cu_seqlens = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seq_lens), dim=0)]).to(device, dtype=torch.int32)

        x = self.proj_in(x) # LC in?

        x = torch.split(x, token_counts, dim=0)
        mask_tokens = [self.mask_token.repeat(grid_size, 1) for grid_size in grid_sizes]
        x = [torch.cat([tokens, masked_vid], dim=0) for tokens, masked_vid in zip(x, mask_tokens)]
        x = torch.cat(x, dim=0)

        x = self.ln_pre(x)

        freqs = self.rope(grids, token_counts, device)
        x = self.model_layers(x, freqs, cu_seqlens, max(seq_lens))

        x = torch.split(x, seq_lens, dim=0)
        x = [tokens[num:] for tokens, num in zip(x, token_counts)]
        x = torch.cat(x, dim=0)

        x = self.proj_out(x)
        x = torch.split(x, grid_sizes, dim=0)

        x = [rearrange(
                vid,
                '(nt nh nw) (c pt ph pw) -> c (nt pt) (nh ph) (nw pw)',
                nt=grid[0], nh=grid[1], nw=grid[2],
                pt=self.patch_size[0], ph=self.patch_size[1], pw=self.patch_size[2],
            ) for vid, grid in zip(x, grids)
        ]
        
        return x # list of video tensors in (CTHW) out


from __future__ import annotations
from functools import wraps, partial
from contextlib import nullcontext
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int32
from torch.amp import autocast

from einops import rearrange, pack, unpack

import random

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# tensor helpers

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

# main class

class FSQ(Module):
    def __init__(
        self,
        levels: List[int],
        dim: int | None = None,
    ):
        super().__init__()

        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent = False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent = False)

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        self.dim = default(dim, len(_levels))

        self.codebook_size = self._levels.prod().item()
        implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
        self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)

    def bound(self, z, eps: float = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """
        bounded = self.bound(z)
        quantized = round_ste(bounded)
        half_width = self._levels // 2 # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        # assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        """ Inverse of `codes_to_indices`. """
        assert exists(indices)
        codes = self._indices_to_codes(indices)
        return codes
    
    @torch.compiler.disable()
    def forward(self, z): # (B*L)C in

        with torch.autocast(z.device.type, enabled=False):
            orig_dtype = z.dtype
            z = z.float()

            codes = self.quantize(z)
            indices = self.codes_to_indices(codes)

            codes = codes.to(orig_dtype)

        return codes, {'indices': indices}
    
