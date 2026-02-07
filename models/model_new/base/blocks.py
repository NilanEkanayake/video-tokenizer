import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_new.base.transformer import ResidualAttentionBlock
from models.model_new.base.utils import get_model_dims, init_weights
from models.model_new.base.rope import get_freqs

from einops.layers.torch import Rearrange
from einops import rearrange
import math
        
    
class Encoder(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            patch_size=(4, 8, 8),
            in_channels=3, # RGB
            out_channels=5, # len(fsq_levels)
            in_grid=(32, 256, 256),
            out_tokens=2048,
        ):
        super().__init__()
        self.patch_size = patch_size
        self.token_size = out_channels
        self.in_channels = in_channels
        self.out_tokens = out_tokens
        self.grid = [x//y for x, y in zip(in_grid, patch_size)]
        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5
        
        self.proj_in = nn.Linear(in_features=in_channels*math.prod(patch_size), out_features=self.width)
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, 1))
        self.freqs = get_freqs(out_tokens, self.grid, head_dim=self.width//self.heads)

        self.model_layers = ResidualAttentionBlock(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers
        )

        self.proj_out = nn.Linear(self.width, self.token_size, bias=True)
        self.apply(init_weights)

    def forward(self, x):
        B = x.shape[0]
        device = x.device

        x = rearrange(
            x, 'b c (t pt) (h ph) (w pw) -> b (t h w) (pt ph pw c)',
            pt=self.patch_size[0], ph=self.patch_size[1], pw=self.patch_size[2]
        )
        x = self.proj_in(x) # returns BLC

        mask_tokens = self.mask_token.expand(B, self.out_tokens, self.width)
        x = torch.cat([mask_tokens, x], dim=1)

        x = self.model_layers(x, freqs=self.freqs.to(device))

        x = x[:, :self.out_tokens]
        x = self.proj_out(x)
        return x


class Decoder(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            patch_size=(4, 8, 8),
            in_channels=5,
            out_channels=3,
            in_tokens=2048,
            out_grid=(32, 256, 256),
        ):
        super().__init__()
        self.patch_size = patch_size
        self.token_size =in_channels
        self.in_channels = out_channels
        self.in_tokens = in_tokens
        self.grid = [x//y for x, y in zip(out_grid, patch_size)]
        self.grid_size = math.prod(self.grid)
        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        self.proj_in = nn.Linear(self.token_size, self.width, bias=True)
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, 1))
        self.freqs = get_freqs(in_tokens, self.grid, head_dim=self.width//self.heads)

        self.model_layers = ResidualAttentionBlock(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers
        )

        self.proj_out = nn.Linear(in_features=self.width, out_features=out_channels*math.prod(patch_size))
        self.apply(init_weights)

    def forward(self, x): # unlike the encoder, 'x' is the quantized latent tokens
        B = x.shape[0]
        device = x.device

        x = self.proj_in(x)

        mask_tokens = self.mask_token.expand(B, self.grid_size, self.width)
        x = torch.cat([x, mask_tokens], dim=1)

        x = self.model_layers(x, freqs=self.freqs.to(device))

        x = x[:, self.in_tokens:]
        x = self.proj_out(x)
        x = rearrange(
            x, 'b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)',
            t=self.grid[0], h=self.grid[1], w=self.grid[2],
            pt=self.patch_size[0], ph=self.patch_size[1], pw=self.patch_size[2],
        )

        return x

import torch
import torch.nn as nn
import math
from einops import rearrange

class Decoder_unify(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            patch_size=(4, 8, 8),
            in_channels=5,
            out_channels=3,
            in_tokens=2048,
            cond_tokens=0,    # <--- 新增：Condition token 数量
            out_grid=(32, 256, 256),
        ):
        super().__init__()
        self.patch_size = patch_size
        self.token_size = in_channels
        self.in_channels = out_channels
        self.in_tokens = in_tokens
        self.cond_tokens = cond_tokens  # 记录 condition 长度
        
        self.grid = [x//y for x, y in zip(out_grid, patch_size)]
        self.grid_size = math.prod(self.grid)
        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        # 1. Main Latent Projector
        self.proj_in = nn.Linear(self.token_size, self.width, bias=True)
        
        # 2. Condition Projector (如果有 condition)
        if self.cond_tokens > 0:
            self.proj_cond = nn.Linear(self.token_size, self.width, bias=True)
        
        # 3. Mask Token (Query)
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, 1)) # [1, 1, width]
        
        # 4. RoPE Freqs Calculation
        # 注意：这里假设 get_freqs 的第一个参数是 "Context Length" (非 Grid 部分的长度)。
        # 原代码是 in_tokens，现在变成了 in_tokens + cond_tokens。
        # 这样 RoPE 才能为前面的 cond 和 x 生成 1D 位置编码，为后面的 grid 生成 3D 位置编码。
        total_context_len = in_tokens + cond_tokens
        self.freqs = get_freqs(total_context_len, self.grid, head_dim=self.width//self.heads)

        self.model_layers = ResidualAttentionBlock(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers
        )

        self.proj_out = nn.Linear(in_features=self.width, out_features=out_channels*math.prod(patch_size))
        self.apply(init_weights)

    def forward(self, x, cond=None): 
        # x: Main latents [B, in_tokens, C]
        # cond: First frame latents [B, cond_tokens, C] (Optional)
        B = x.shape[0]
        device = x.device

        # --- 1. 处理主 Latents ---
        x = self.proj_in(x)
        
        tokens_list = []
        
        # --- 2. 处理 Condition (如果有) ---
        if self.cond_tokens > 0 and cond is not None:
            cond = self.proj_cond(cond) # [B, cond_tokens, W]
            tokens_list.append(cond)
        elif self.cond_tokens > 0 and cond is None:
            # 如果定义了 cond_tokens 但 forward 没传，可能需要报错或补零，这里建议报错
            raise ValueError(f"Model initialized with cond_tokens={self.cond_tokens}, but no cond provided.")

        # --- 3. 拼接序列 ---
        # 顺序建议：[Condition, Main Latents, Mask Queries]
        tokens_list.append(x)
        
        # 准备 Mask Tokens
        mask_tokens = self.mask_token.expand(B, self.grid_size, self.width)
        tokens_list.append(mask_tokens)
        
        # 拼接
        x_full = torch.cat(tokens_list, dim=1) # [B, cond+in+grid, W]

        # --- 4. Transformer Forward ---
        # 传入 freqs 进行旋转位置编码
        x_out = self.model_layers(x_full, freqs=self.freqs.to(device))

        # --- 5. 切片获取输出 ---
        # 我们只需要 Mask Tokens 对应的部分进行解码
        # 前缀长度 = cond_tokens + in_tokens
        prefix_len = self.in_tokens + (self.cond_tokens if cond is not None else 0)
        
        x_out = x_out[:, prefix_len:] 

        # --- 6. 输出投影与重排 ---
        x_out = self.proj_out(x_out)
        x_out = rearrange(
            x_out, 'b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)',
            t=self.grid[0], h=self.grid[1], w=self.grid[2],
            pt=self.patch_size[0], ph=self.patch_size[1], pw=self.patch_size[2],
        )

        return x_out