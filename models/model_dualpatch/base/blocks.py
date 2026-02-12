import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_dualpatch.base.transformer import ResidualAttentionBlock
from models.model_dualpatch.base.utils import get_model_dims, init_weights

from einops.layers.torch import Rearrange
from einops import rearrange
import math


class Encoder(nn.Module):
    def __init__(
            self,
            model_size="small_thin",
            spatial_patch_size=(8, 8),
            first_frame_temporal_patch=1,
            rest_temporal_patch=3,
            in_channels=3,
            out_channels=6,
            num_frames=16,
            spatial_size=(128, 128),
            out_tokens=1024,
        ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.first_frame_temporal_patch = first_frame_temporal_patch
        self.rest_temporal_patch = rest_temporal_patch
        self.token_size = out_channels
        self.in_channels = in_channels
        self.out_tokens = out_tokens
        self.num_frames = num_frames
        self.spatial_size = spatial_size

        ph, pw = spatial_patch_size

        # Grid sizes
        self.spatial_grid = (spatial_size[0] // ph, spatial_size[1] // pw)
        nh, nw = self.spatial_grid

        self.first_temporal_grid = 1
        self.first_num_tokens = self.first_temporal_grid * nh * nw

        self.rest_frame_count = num_frames - first_frame_temporal_patch
        assert self.rest_frame_count % rest_temporal_patch == 0
        self.rest_temporal_grid = self.rest_frame_count // rest_temporal_patch
        self.rest_num_tokens = self.rest_temporal_grid * nh * nw

        self.total_patch_tokens = self.first_num_tokens + self.rest_num_tokens

        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        # 3D Conv patchify: first frame (t=1, h=ph, w=pw)
        self.first_patch_embed = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self.width,
            kernel_size=(first_frame_temporal_patch, ph, pw),
            stride=(first_frame_temporal_patch, ph, pw),
            bias=True
        )

        # 3D Conv patchify: rest frames (t=3, h=ph, w=pw)
        self.rest_patch_embed = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self.width,
            kernel_size=(rest_temporal_patch, ph, pw),
            stride=(rest_temporal_patch, ph, pw),
            bias=True
        )

        # Positional embeddings
        self.first_pos_embed = nn.Parameter(scale * torch.randn(1, self.first_num_tokens, self.width))
        self.rest_pos_embed = nn.Parameter(scale * torch.randn(1, self.rest_num_tokens, self.width))

        # Learnable latent query tokens
        self.latent_queries = nn.Parameter(scale * torch.randn(1, out_tokens, self.width))

        # Transformer
        self.model_layers = ResidualAttentionBlock(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers
        )

        # Output projection: width -> token_size
        self.proj_out = nn.Linear(self.width, self.token_size, bias=True)
        self.apply(init_weights)

    def forward(self, x):
        """
        x: (B, C, T, H, W) e.g. (B, 3, 16, 256, 256)
        Returns: (B, out_tokens, token_size)
        """
        B = x.shape[0]
        pt_first = self.first_frame_temporal_patch

        # Split temporally
        x_first = x[:, :, :pt_first, :, :]   # (B, C, 1, H, W)
        x_rest = x[:, :, pt_first:, :, :]     # (B, C, 15, H, W)

        # 3D Conv patchify
        # first: (B, C, 1, H, W) -> (B, width, 1, nh, nw)
        f_first = self.first_patch_embed(x_first)
        f_first = rearrange(f_first, 'b d t h w -> b (t h w) d')
        f_first = f_first + self.first_pos_embed

        # rest: (B, C, 15, H, W) -> (B, width, 5, nh, nw)
        f_rest = self.rest_patch_embed(x_rest)
        f_rest = rearrange(f_rest, 'b d t h w -> b (t h w) d')
        f_rest = f_rest + self.rest_pos_embed

        # Concatenate patch tokens
        patch_tokens = torch.cat([f_first, f_rest], dim=1)  # (B, total_patch_tokens, width)

        # Prepend latent queries
        latent_queries = self.latent_queries.expand(B, -1, -1)
        tokens = torch.cat([latent_queries, patch_tokens], dim=1)
        # (B, out_tokens + total_patch_tokens, width)

        # Self-attention
        tokens = self.model_layers(tokens)

        # Extract latent tokens only
        latent = tokens[:, :self.out_tokens]
        latent = self.proj_out(latent)  # (B, out_tokens, token_size)
        return latent



    

class Decoder(nn.Module):
    def __init__(
            self,
            model_size="small_thin",
            spatial_patch_size=(8, 8),
            first_frame_temporal_patch=1,
            rest_temporal_patch=3,
            in_channels=6,
            out_channels=3,
            in_tokens=1024,
            num_frames=16,
            spatial_size=(128, 128),
        ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.first_frame_temporal_patch = first_frame_temporal_patch
        self.rest_temporal_patch = rest_temporal_patch
        self.token_size = in_channels
        self.out_channels = out_channels
        self.in_tokens = in_tokens
        self.num_frames = num_frames
        self.spatial_size = spatial_size

        ph, pw = spatial_patch_size

        # Grid sizes (same as encoder)
        self.spatial_grid = (spatial_size[0] // ph, spatial_size[1] // pw)
        nh, nw = self.spatial_grid

        self.first_temporal_grid = 1
        self.first_num_tokens = self.first_temporal_grid * nh * nw

        self.rest_frame_count = num_frames - first_frame_temporal_patch
        self.rest_temporal_grid = self.rest_frame_count // rest_temporal_patch
        self.rest_num_tokens = self.rest_temporal_grid * nh * nw

        self.total_patch_tokens = self.first_num_tokens + self.rest_num_tokens

        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        # Input projection: token_size -> width
        self.proj_in = nn.Linear(self.token_size, self.width, bias=True)

        # Positional embedding for latent tokens
        self.latent_pos_embed = nn.Parameter(scale * torch.randn(1, in_tokens, self.width))

        # Learnable patch query tokens
        self.first_patch_queries = nn.Parameter(scale * torch.randn(1, self.first_num_tokens, self.width))
        self.rest_patch_queries = nn.Parameter(scale * torch.randn(1, self.rest_num_tokens, self.width))

        # Transformer
        self.model_layers = ResidualAttentionBlock(
            embed_dim=self.width,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            num_layer=self.num_layers
        )

        # 3D ConvTranspose unpatchify: first frame
        self.first_unpatch = nn.ConvTranspose3d(
            in_channels=self.width,
            out_channels=out_channels,
            kernel_size=(first_frame_temporal_patch, ph, pw),
            stride=(first_frame_temporal_patch, ph, pw),
            bias=True
        )

        # 3D ConvTranspose unpatchify: rest frames
        self.rest_unpatch = nn.ConvTranspose3d(
            in_channels=self.width,
            out_channels=out_channels,
            kernel_size=(rest_temporal_patch, ph, pw),
            stride=(rest_temporal_patch, ph, pw),
            bias=True
        )

        self.apply(init_weights)

    def forward(self, x):
        """
        x: (B, in_tokens, token_size)
        Returns: (B, C, T, H, W) e.g. (B, 3, 16, 256, 256)
        """
        B = x.shape[0]
        nh, nw = self.spatial_grid

        # Project latent tokens
        x = self.proj_in(x)
        x = x + self.latent_pos_embed

        # Append patch query tokens
        first_queries = self.first_patch_queries.expand(B, -1, -1)
        rest_queries = self.rest_patch_queries.expand(B, -1, -1)
        tokens = torch.cat([x, first_queries, rest_queries], dim=1)
        # (B, in_tokens + total_patch_tokens, width)

        # Self-attention
        tokens = self.model_layers(tokens)

        # Extract patch query outputs
        patch_tokens = tokens[:, self.in_tokens:]
        first_tokens = patch_tokens[:, :self.first_num_tokens]
        rest_tokens = patch_tokens[:, self.first_num_tokens:]

        # Reshape to 3D feature maps for ConvTranspose3d
        # first: (B, first_num_tokens, width) -> (B, width, 1, nh, nw)
        first_feat = rearrange(
            first_tokens, 'b (t h w) d -> b d t h w',
            t=self.first_temporal_grid, h=nh, w=nw
        )

        # rest: (B, rest_num_tokens, width) -> (B, width, rest_temporal_grid, nh, nw)
        rest_feat = rearrange(
            rest_tokens, 'b (t h w) d -> b d t h w',
            t=self.rest_temporal_grid, h=nh, w=nw
        )

        # 3D ConvTranspose unpatchify
        x_first = self.first_unpatch(first_feat)   # (B, C, 1, H, W)
        x_rest = self.rest_unpatch(rest_feat)       # (B, C, 15, H, W)

        # Concatenate along temporal dimension
        out = torch.cat([x_first, x_rest], dim=2)  # (B, C, 16, H, W)
        return out


    


class Decoder_unify(nn.Module):
    def __init__(
            self,
            model_size="tiny",
            patch_size=(4, 8, 8),
            in_channels=5,
            out_channels=3,
            in_tokens=2048,   # 主视频的 tokens 数量
            cond_tokens=0,    # <--- 新增：Condition (第一帧) 的 tokens 数量
            out_grid=(32, 256, 256),
        ):
        super().__init__()
        self.patch_size = patch_size
        self.token_size = in_channels
        self.in_channels = out_channels
        self.in_tokens = in_tokens
        self.cond_tokens = cond_tokens # 记录 condition 长度
        
        self.grid = [x//y for x, y in zip(out_grid, patch_size)]
        self.width, self.num_layers, self.heads, mlp_ratio = get_model_dims(model_size)
        scale = self.width ** -0.5

        # 1. Main Latent Projector
        self.proj_in = nn.Linear(self.token_size, self.width, bias=True)
        # Main Positional Embedding
        self.positional_embedding = nn.Parameter(scale * torch.randn(1, in_tokens, self.width))
        
        # 2. Condition (First Frame) Projector & Embedding
        if self.cond_tokens > 0:
            self.proj_cond = nn.Linear(self.token_size, self.width, bias=True)
            self.cond_positional_embedding = nn.Parameter(scale * torch.randn(1, cond_tokens, self.width))

        # 3. Output Mask (Learnable Queries for Pixels)
        # 这些是最终生成像素的 Query Token
        self.patch_token_mask = nn.Parameter(scale * torch.randn(1, math.prod(self.grid), self.width)) 

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
        # cond: First frame latents [B, cond_tokens, C]
        B = x.shape[0]
        # --- 处理主 Latents ---
        x = self.proj_in(x)
        x = x + self.positional_embedding # [B, in_tokens, Width]
        tokens_list = []
        # --- 处理 Condition (第一帧) ---
        if self.cond_tokens > 0 and cond is not None:
            cond = self.proj_cond(cond)
            cond = cond + self.cond_positional_embedding # [B, cond_tokens, Width]
            tokens_list.append(cond) # 放在最前面作为前缀
        # --- 拼接 ---
        # 顺序: [第一帧Token, 主LatentToken, 目标MaskToken]
        # 这样 Attention 可以让 MaskToken 看到前面所有的信息
        tokens_list.append(x)
        # --- 准备 Output Mask ---
        patch_token_mask = self.patch_token_mask.expand(B, -1, -1)
        tokens_list.append(patch_token_mask)
        # 拼接所有 tokens
        x_full = torch.cat(tokens_list, dim=1)
        # --- Transformer 处理 ---
        x_out = self.model_layers(x_full)
        prefix_len = self.in_tokens + (self.cond_tokens if cond is not None else 0)
        
        # 取出最后部分的 tokens 进行解码
        x_out = x_out[:, prefix_len:] 
        
        x_out = self.proj_out(x_out)
        
        # Rearrange pixel shuffle
        x_out = rearrange(
            x_out, 'b (nt nh nw) (c pt ph pw) -> b c (nt pt) (nh ph) (nw pw)',
            nt=self.grid[0], nh=self.grid[1], nw=self.grid[2],
            pt=self.patch_size[0], ph=self.patch_size[1], pw=self.patch_size[2],
        )

        return x_out
