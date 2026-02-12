import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_design.base.rope import apply_rotary_emb
from flash_attn import flash_attn_func
from einops import rearrange
import math


class GEGLU(nn.Module):
    """GEGLU 激活函数"""
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


class RMSNorm(nn.Module):
    """RMSNorm - 比 LayerNorm 更高效，适合大模型"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).to(x.dtype) * self.weight


def make_ffn(dim, mult=4, mult_of=32):
    """SwiGLU FFN with Pre-Norm"""
    inner_dim = int(mult * (2 / 3) * dim)
    inner_dim = mult_of * ((inner_dim + mult_of - 1) // mult_of)
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


# ============================================================
# 改进1: 更好的 Self-Attention（带 Pre-Norm + Gated Output）
# ============================================================

class SelfAttention(nn.Module):
    """
    改进点：
    1. Pre-RMSNorm
    2. QK-Norm 使用 RMSNorm（与 Gemma2 一致）
    3. Gated output（与 Qwen3 一致）
    4. 支持可选的 attention mask 用于因果或分段注意力
    """
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_gate = nn.Linear(dim, dim, bias=False)  # 独立的 gate projection
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs):
        x_norm = self.norm(x)
        q, k, v = self.to_qkv(x_norm).chunk(3, dim=-1)
        gate = self.to_gate(x_norm)

        q = q.unflatten(-1, (self.heads, self.head_dim))
        k = k.unflatten(-1, (self.heads, self.head_dim))
        v = v.unflatten(-1, (self.heads, self.head_dim))

        q = self.q_norm(q.contiguous()).to(q.dtype)
        k = self.k_norm(k.contiguous()).to(k.dtype)

        q = apply_rotary_emb(q, freqs)
        k = apply_rotary_emb(k, freqs)

        out = flash_attn_func(q, k, v)
        out = out.flatten(-2).contiguous()
        out = out * torch.sigmoid(gate)
        return self.out_proj(out)


# ============================================================
# 改进2: Cross-Attention（Decoder 用于接收 Condition）
# ============================================================

class CrossAttention(nn.Module):
    """
    用于 Decoder 中，让 mask queries 和 main latents 
    显式地从 condition tokens 中提取信息。
    
    这比简单拼接 condition 到序列中更有效，因为：
    1. Condition 不会占用 self-attention 的序列长度
    2. 可以对 condition 使用独立的 KV projection
    3. Query 可以选择性地关注 condition 的不同部分
    """
    def __init__(self, dim, heads, context_dim=None):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        context_dim = context_dim or dim

        self.norm_q = RMSNorm(dim)
        self.norm_kv = RMSNorm(context_dim)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim * 2, bias=False)
        self.to_gate = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, context):
        """
        x: [B, N, D] - query tokens
        context: [B, M, D] - condition tokens (第一帧 latents)
        """
        x_norm = self.norm_q(x)
        ctx_norm = self.norm_kv(context)

        q = self.to_q(x_norm)
        k, v = self.to_kv(ctx_norm).chunk(2, dim=-1)
        gate = self.to_gate(x_norm)

        q = q.unflatten(-1, (self.heads, self.head_dim))
        k = k.unflatten(-1, (self.heads, self.head_dim))
        v = v.unflatten(-1, (self.heads, self.head_dim))

        q = self.q_norm(q.contiguous()).to(q.dtype)
        k = self.k_norm(k.contiguous()).to(k.dtype)

        # Cross attention 不使用 RoPE（因为 query 和 key 来自不同空间）
        out = flash_attn_func(q, k, v)
        out = out.flatten(-2).contiguous()
        out = out * torch.sigmoid(gate)
        return self.out_proj(out)


# ============================================================
# 改进3: Transformer Block（支持 Self-Attn + Cross-Attn + FFN）
# ============================================================

class TransformerBlock(nn.Module):
    """
    单个 Transformer 块，支持：
    - Self-Attention (必须)
    - Cross-Attention (可选，用于 condition)
    - FFN (必须)
    
    残差连接使用 LNS（Layer-wise Normalized Scaling）
    """
    def __init__(self, dim, heads, mlp_ratio=4, has_cross_attn=False, layer_idx=0, total_layers=1):
        super().__init__()
        self.self_attn = SelfAttention(dim, heads)
        self.ffn = make_ffn(dim, mlp_ratio)
        self.has_cross_attn = has_cross_attn
        
        if has_cross_attn:
            self.cross_attn = CrossAttention(dim, heads)
        
        # LNS: 每层有一个可学习的残差缩放因子，初始化为 1/sqrt(2*layer_idx+1)
        # 乘以 2 是因为每层有两个（或三个）残差连接
        init_scale = 1.0 / math.sqrt(2 * layer_idx + 1)
        self.res_scale_sa = nn.Parameter(torch.tensor(init_scale))
        self.res_scale_ffn = nn.Parameter(torch.tensor(init_scale))
        if has_cross_attn:
            self.res_scale_ca = nn.Parameter(torch.tensor(init_scale))

    def forward(self, x, freqs, context=None):
        # Self-Attention
        x = x + self.res_scale_sa * self.self_attn(x, freqs)
        
        # Cross-Attention (if applicable)
        if self.has_cross_attn and context is not None:
            x = x + self.res_scale_ca * self.cross_attn(x, context)
        
        # FFN
        x = x + self.res_scale_ffn * self.ffn(x)
        
        return x


class TransformerStack(nn.Module):
    """
    Transformer 块的堆叠。
    
    改进点：
    1. 每个 block 是独立的 TransformerBlock
    2. 支持可选的 Cross-Attention
    3. 正确的 LNS 实现
    4. 输出前有 final norm
    """
    def __init__(self, embed_dim=512, heads=8, mlp_ratio=4, num_layers=2, has_cross_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, 
                heads=heads, 
                mlp_ratio=mlp_ratio, 
                has_cross_attn=has_cross_attn,
                layer_idx=i,
                total_layers=num_layers,
            )
            for i in range(num_layers)
        ])
        self.final_norm = RMSNorm(embed_dim)

    def forward(self, x, freqs, context=None):
        for layer in self.layers:
            x = layer(x, freqs, context=context)
        return self.final_norm(x)