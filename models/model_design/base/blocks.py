import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_design.base.transformer import TransformerStack, RMSNorm
from models.model_design.base.utils import get_model_dims, init_weights
from models.model_design.base.rope import get_freqs

from einops.layers.torch import Rearrange
from einops import rearrange
import math
        
    
class LearnedQueryTokens(nn.Module):
    """
    改进点：
    - 不再使用单个标量 expand 到所有位置
    - 使用 per-position 可学习的 query tokens
    - 可选地加入位置信息初始化
    
    这让每个 query token 有独立的初始表示，
    模型不完全依赖位置编码来区分不同 token。
    """
    def __init__(self, num_tokens, dim):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, dim) * (dim ** -0.5))

    def forward(self, batch_size):
        return self.tokens.expand(batch_size, -1, -1)


# ============================================================
# 改进5: 更好的 Encoder（Perceiver-style compression）
# ============================================================

class Encoder(nn.Module):
    """
    改进点：
    1. Learned query tokens 替代 scalar mask token
    2. Pre-norm / Post-norm 完整
    3. 输入 patch embedding 使用卷积（更好的局部特征提取）
    """
    def __init__(
        self,
        model_size="tiny",
        patch_size=(4, 8, 8),
        in_channels=3,
        out_channels=5,
        in_grid=(32, 256, 256),
        out_tokens=2048,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.token_size = out_channels
        self.in_channels = in_channels
        self.out_tokens = out_tokens
        self.grid = [x // y for x, y in zip(in_grid, patch_size)]
        self.grid_size = math.prod(self.grid)
        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)

        # 改进: 使用 3D 卷积做 patch embedding（比 linear projection 更好地捕获局部特征）
        self.patch_embed = nn.Conv3d(
            in_channels, self.width,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # 改进: Learned query tokens
        self.query_tokens = LearnedQueryTokens(out_tokens, self.width)

        # RoPE
        self.freqs = get_freqs(out_tokens, self.grid, head_dim=self.width // self.heads)

        # Transformer
        self.transformer = TransformerStack(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layers=self.num_layers,
            has_cross_attn=False,  # Encoder 只用 self-attention
        )

        # Output projection
        self.proj_out = nn.Linear(self.width, self.token_size, bias=True)
        self.apply(init_weights)

    def forward(self, x):
        """
        x: [B, C, T, H, W]
        returns: [B, out_tokens, token_size]
        """
        B = x.shape[0]
        device = x.device

        # Patch embedding via Conv3d
        x = self.patch_embed(x)  # [B, width, T', H', W']
        x = rearrange(x, 'b c t h w -> b (t h w) c')  # [B, grid_size, width]

        # Prepend learned query tokens
        queries = self.query_tokens(B)  # [B, out_tokens, width]
        x = torch.cat([queries, x], dim=1)  # [B, out_tokens + grid_size, width]

        # Transformer
        x = self.transformer(x, freqs=self.freqs.to(device))

        # Extract query outputs
        x = x[:, :self.out_tokens]
        x = self.proj_out(x)
        return x


# ============================================================
# 改进6: 第一帧编码器（共享 backbone 的轻量版本）
# ============================================================

class FirstFrameEncoder(nn.Module):
    """
    专门处理第一帧的编码器。
    
    改进点：
    1. 使用 2D patch embedding（第一帧没有时间维度）
    2. 独立但结构相似的 transformer
    3. 可以选择共享部分参数（这里暂不实现，但结构兼容）
    """
    def __init__(
        self,
        model_size="tiny",
        patch_size_hw=(8, 8),  # 只有空间维度
        in_channels=3,
        out_channels=5,
        in_hw=(128, 128),
        out_tokens=256,
    ):
        super().__init__()
        self.patch_size_hw = patch_size_hw
        self.token_size = out_channels
        self.out_tokens = out_tokens
        self.grid_hw = [x // y for x, y in zip(in_hw, patch_size_hw)]
        self.grid_size = math.prod(self.grid_hw)
        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)

        # 2D patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, self.width,
            kernel_size=patch_size_hw,
            stride=patch_size_hw,
        )

        # Learned query tokens
        self.query_tokens = LearnedQueryTokens(out_tokens, self.width)

        # RoPE: 对于 2D，用 grid=[1, H', W'] 来生成频率
        self.freqs = get_freqs(out_tokens, [1] + self.grid_hw, head_dim=self.width // self.heads)

        # Transformer (较少的层数，因为第一帧信息量较小)
        # 使用一半的层数
        first_frame_layers = max(self.num_layers // 2, 2)
        self.transformer = TransformerStack(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layers=first_frame_layers,
            has_cross_attn=False,
        )

        self.proj_out = nn.Linear(self.width, self.token_size, bias=True)
        self.apply(init_weights)

    def forward(self, x):
        """
        x: [B, C, 1, H, W] or [B, C, H, W]
        returns: [B, out_tokens, token_size]
        """
        B = x.shape[0]
        device = x.device

        # 处理可能的时间维度
        if x.dim() == 5:
            x = x.squeeze(2)  # [B, C, H, W]

        x = self.patch_embed(x)  # [B, width, H', W']
        x = rearrange(x, 'b c h w -> b (h w) c')

        queries = self.query_tokens(B)
        x = torch.cat([queries, x], dim=1)

        x = self.transformer(x, freqs=self.freqs.to(device))

        x = x[:, :self.out_tokens]
        x = self.proj_out(x)
        return x


# ============================================================
# 改进7: 统一 Decoder（带 Cross-Attention 的条件生成）
# ============================================================

class UnifiedDecoder(nn.Module):
    """
    改进点：
    1. 主路径用 Self-Attention 处理 [main_latents, mask_queries]
    2. Cross-Attention 层接收第一帧 condition
    3. 第一帧 condition 不占用主序列长度（更高效）
    4. Learned query tokens 替代 scalar mask token
    5. 可选的多尺度输出（未实现，但结构兼容）
    """
    def __init__(
        self,
        model_size="tiny",
        patch_size=(4, 8, 8),
        in_channels=5,
        out_channels=3,
        in_tokens=2048,
        cond_tokens=0,
        out_grid=(32, 256, 256),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.token_size = in_channels
        self.out_channels = out_channels
        self.in_tokens = in_tokens
        self.cond_tokens = cond_tokens
        self.grid = [x // y for x, y in zip(out_grid, patch_size)]
        self.grid_size = math.prod(self.grid)
        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)

        # Input projection for main latents
        self.proj_in = nn.Linear(self.token_size, self.width, bias=True)

        # Condition projection (独立的投影，因为 condition 和 main latents 语义不同)
        if self.cond_tokens > 0:
            self.proj_cond = nn.Linear(self.token_size, self.width, bias=True)
            # 额外的 condition 处理：一个轻量的 MLP 来适配 condition 特征
            self.cond_adapter = nn.Sequential(
                RMSNorm(self.width),
                nn.Linear(self.width, self.width, bias=False),
                nn.SiLU(),
                nn.Linear(self.width, self.width, bias=False),
            )

        # Learned query tokens for output grid
        self.query_tokens = LearnedQueryTokens(self.grid_size, self.width)

        # RoPE: 只需要为 self-attention 的序列生成
        # 序列 = [main_latents, mask_queries]
        self.freqs = get_freqs(in_tokens, self.grid, head_dim=self.width // self.heads)

        # Transformer with Cross-Attention
        self.transformer = TransformerStack(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layers=self.num_layers,
            has_cross_attn=(cond_tokens > 0),  # 只在有 condition 时启用
        )

        # Output projection
        self.proj_out = nn.Linear(self.width, out_channels * math.prod(patch_size))
        self.apply(init_weights)

    def forward(self, x, cond=None):
        """
        x: [B, in_tokens, token_size] - 主视频 latent tokens
        cond: [B, cond_tokens, token_size] - 第一帧 latent tokens (可选)
        returns: [B, C, T, H, W]
        """
        B = x.shape[0]
        device = x.device

        # 投影主 latents
        x = self.proj_in(x)  # [B, in_tokens, width]

        # 处理 condition
        context = None
        if self.cond_tokens > 0:
            if cond is None:
                raise ValueError(
                    f"Model initialized with cond_tokens={self.cond_tokens}, "
                    f"but no cond provided in forward()."
                )
            context = self.proj_cond(cond)  # [B, cond_tokens, width]
            context = context + self.cond_adapter(context)  # 残差适配

        # 准备 mask query tokens
        queries = self.query_tokens(B)  # [B, grid_size, width]

        # 拼接 self-attention 序列: [main_latents, mask_queries]
        # Condition 通过 cross-attention 注入，不拼接到序列中
        x = torch.cat([x, queries], dim=1)  # [B, in_tokens + grid_size, width]

        # Transformer forward
        x = self.transformer(x, freqs=self.freqs.to(device), context=context)

        # 提取 mask query 输出
        x = x[:, self.in_tokens:]  # [B, grid_size, width]

        # 输出投影和重排
        x = self.proj_out(x)
        x = rearrange(
            x, 'b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)',
            t=self.grid[0], h=self.grid[1], w=self.grid[2],
            pt=self.patch_size[0], ph=self.patch_size[1], pw=self.patch_size[2],
        )
        return x