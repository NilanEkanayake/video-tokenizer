import itertools
import os

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from easydict import EasyDict as edict

import models
import utils
from models import register

from .embed import (PatchEmbed3D, VideoPatchEmbed,
                    get_1d_sincos_pos_embed_from_grid, get_3d_sincos_pos_embed)



def get_orig_module(module):
    if hasattr(module, 'module'):
        module = module.module
    if hasattr(module, '_orig_mod'):
        module = module._orig_mod
    return module

import torch
import torch.nn as nn
import numpy as np
import itertools
import einops
# 假设其他依赖项 (models, PatchEmbed3D, get_3d_sincos_pos_embed 等) 在上下文环境中已存在

@register('cosmos_larp_tokenizer_unified')
class CosmosLARPTokenizerUnified(nn.Module, PyTorchModelHubMixin):
    output_format = 'bcthw'
    def __init__(
        self, 
        bottleneck,
        prior_model,
        # --- Token 数量配置 ---
        num_latent_tokens=1024,    # 原本是 128+256，现在统一为一个总数
        
        # --- 尺寸配置 ---
        input_size=128,
        frame_num=16,             # 统一处理的帧数
        temporal_patch_size=4,    # 时间维度的 Patch Size bottleneck_token_num
        patch_size=8,
        decoder_temporal_patch_size=4,
        decoder_patch_size=8,
        in_channels=3,

        # --- 模型配置 ---
        transformer_name='transformer_encoder_parallel',
        encoder_name=None,
        decoder_name=None,
        encoder_hidden_size=768,
        decoder_hidden_size=768,
        
        encoder_num_heads=12,
        decoder_num_heads=12,
        encoder_depth=6,
        decoder_depth=6,

        # --- Embedding 配置 ---
        latent_pe_scale_factor=10000,
        query_init_std=0.02,
        encoder_query_gaussian_init=True,
        
        # --- Boolean Flags ---
        learned_decoder_latent_pe=False,
        
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.frame_num = frame_num 
        
        self.num_latent_tokens = num_latent_tokens
        self.bottleneck_token_num = num_latent_tokens

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        
        self.decoder_patch_size = decoder_patch_size
        self.decoder_temporal_patch_size = decoder_temporal_patch_size

        # =========================================================
        # 1. Embedder (统一处理所有帧)
        # =========================================================
        self.token_h = self.token_w = input_size // patch_size
        
        assert frame_num % temporal_patch_size == 0, "frame_num must be divisible by temporal_patch_size"
        
        # 统一的 3D Patch Embedder
        self.x_embedder = PatchEmbed3D(
            input_size, frame_num, patch_size, temporal_patch_size, in_channels, encoder_hidden_size, bias=True
        )
        
        self.token_t = self.x_embedder.num_temporal_patches
        self.num_patches = self.x_embedder.num_spatial_patches * self.x_embedder.num_temporal_patches

        # Encoder Patch PE
        self.register_buffer('encoder_patch_pe', torch.zeros(1, self.num_patches, encoder_hidden_size))
        self.get_encoder_patch_pe = lambda: self.encoder_patch_pe

        # Encoder Latent Queries (Learnable Queries for Perceiver IO)
        self.encoder_latent_query_embed = nn.Parameter(torch.zeros(num_latent_tokens, encoder_hidden_size), requires_grad=True)
        self.encoder_query = lambda: self.encoder_latent_query_embed.unsqueeze(0)
        
        # Init Latent Queries
        query_embed = torch.randn(self.num_latent_tokens, self.encoder_hidden_size) * query_init_std
        self.encoder_latent_query_embed.data.copy_(query_embed)

        # =========================================================
        # 2. Encoder & Decoder Models
        # =========================================================
        if encoder_name is None or encoder_name.lower() in ['none', 'no', 'null', '']:
            encoder_name = transformer_name
        if decoder_name is None or decoder_name.lower() in ['none', 'no', 'null', '']:
            decoder_name = transformer_name

        encoder_args = {
            'name': encoder_name,
            'args': {
                'dim': encoder_hidden_size,
                'depth': encoder_depth,
                'n_head': encoder_num_heads,
                'head_dim': encoder_hidden_size // encoder_num_heads,
            },
        }
        self.encoder = models.make(encoder_args)

        # 瓶颈层 (VQ)
        self.bottleneck_dim = bottleneck['args']['bottleneck_dim']
        bottleneck_args = {
            'token_nums': self.bottleneck_token_num, 
            'input_dim': encoder_hidden_size, 
            'output_dim': decoder_hidden_size
        }
        self.bottleneck = models.make(bottleneck, args=bottleneck_args)
        self.codebook_size = bottleneck['args']['regularizer']['args']['codebook_size']

        decoder_args = {
            'name': decoder_name,
            'args': {
                'dim': decoder_hidden_size,
                'depth': decoder_depth,
                'n_head': decoder_num_heads,
                'head_dim': decoder_hidden_size // decoder_num_heads,
            }, 
        }
        self.decoder = models.make(decoder_args)

        # =========================================================
        # 3. Output Layer & Decoder Embeddings
        # =========================================================
        self.final_layer = OutputLayer(decoder_hidden_size, decoder_temporal_patch_size, decoder_patch_size, self.out_channels)

        # 计算 Decoder 需要恢复的 patch 数量
        recon_t = frame_num // decoder_temporal_patch_size
        recon_hw = (input_size // decoder_patch_size)**2
        
        # Decoder Spatial-Temporal Query Embeddings (用于告知 Decoder 每个位置是哪里)
        self.register_buffer('decoder_patch_query_embed', torch.zeros(1, recon_t * recon_hw, decoder_hidden_size))
        self.get_decoder_patch_query_embed_raw = lambda: self.decoder_patch_query_embed
        self.decoder_patch_query_token_type_embed = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size), requires_grad=True)
        self.get_decoder_patch_query_embed = lambda: self.get_decoder_patch_query_embed_raw() + self.decoder_patch_query_token_type_embed

        # Decoder Latent PE (加在 Latent Codes 上)
        self.learned_decoder_latent_pe = learned_decoder_latent_pe
        self.register_buffer('decoder_latent_pe', torch.zeros(1, self.num_latent_tokens, decoder_hidden_size))
        self.get_decoder_latent_pe = lambda: self.decoder_latent_pe

        self.prior_model = None 
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 1. Encoder Positional Embeddings (3D Sincos)
        encoder_pos_embed = get_3d_sincos_pos_embed(self.encoder_hidden_size, self.token_h, self.token_t)
        self.encoder_patch_pe.data.copy_(torch.from_numpy(encoder_pos_embed).float().reshape_as(self.encoder_patch_pe))

        # 2. Decoder Latent Embeddings (1D Sincos for latents)
        decoder_token_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, np.arange(self.num_latent_tokens), 10000)
        decoder_token_embed = torch.from_numpy(decoder_token_embed).float().reshape(1, self.num_latent_tokens, self.decoder_hidden_size)
        self.decoder_latent_pe.data.copy_(decoder_token_embed)

        # 3. Decoder Query Embeddings (3D Sincos for output grid)
        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_hidden_size, self.token_h, self.token_t) # Assuming decode dim matches
        self.decoder_patch_query_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().reshape_as(self.decoder_patch_query_embed))
        
        decoder_patch_query_token_type_embed = torch.randn(1, 1, self.decoder_hidden_size) * .02
        self.decoder_patch_query_token_type_embed.data.copy_(decoder_patch_query_token_type_embed)

        # 4. Patch Embedder Projections
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # 5. Output Layer
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def decoder_parameters(self):
        decoder_params = itertools.chain(
            self.decoder.parameters(),
            self.final_layer.parameters(),
            [self.decoder_patch_query_token_type_embed],
        )
        return decoder_params

    def decoder_requires_grad_(self, requires_grad):
        for param in self.decoder_parameters():
            param.requires_grad_(requires_grad)

    def others_parameters(self):
        decoder_params_set = set(self.decoder_parameters())
        return (p for p in self.parameters() if p not in decoder_params_set)

    def others_requires_grad_(self, requires_grad):
        for param in self.others_parameters():
            param.requires_grad_(requires_grad)

    def encode(self, x):
        # x: [B, C, T, H, W] (例如 [B, 3, 16, 128, 128])
        B = x.shape[0]
        
        # 1. Patch Embed
        # [B, N_patch, D]
        feat = self.x_embedder(x) + self.get_encoder_patch_pe() 
        
        # 2. Prepare Latent Queries
        # [B, N_latent, D]
        q = self.encoder_query().expand(B, -1, -1) 
        
        # 3. Encoder (Perceiver IO: Cross Attn -> Self Attn layers)
        # [B, N_latent, D]
        z = self.encoder(feat, q) 

        # 4. VQ Bottleneck
        bottleneck_out = self.bottleneck(z)
        z_quantized = bottleneck_out.pop('output') # [B, N_latent, D]

        return {
            'encoded': z_quantized, 
            **bottleneck_out
        }

    def unpatchify(self, x):
        """
        x: (b, n, t_patch_size * s_patch_size**2 * c)
        videos: (b, c, t, h, w)
        注意：这里的 n = t_patches * h_patches * w_patches
        """
        c = self.out_channels
        pt = self.decoder_temporal_patch_size # 使用 decoder 的配置
        p = self.decoder_patch_size
        h = w = self.token_h # 假设输入输出空间尺寸一致
        
        # 计算时间维度的 patch 数
        # n = t_grid * h_grid * w_grid
        t_grid = x.size(1) // (h * w)

        x = x.reshape(-1, t_grid, h, w, pt, p, p, c)
        # 重新排列为视频格式: [b, c, t, h, w]
        x = einops.rearrange(x, 'b t h w pt p1 p2 c -> b c (t pt) (h p1) (w p2)')
        return x

    def decode(self, z):
        # z: [B, N_latent, D] (Quantized Latent)
        B = z.shape[0]

        # 1. Add Decoder Latent Positional Embedding
        decoder_token_embed = self.get_decoder_latent_pe()
        z = z + decoder_token_embed 

        # 2. Prepare Decoder Spatial-Temporal Queries (Target Grid)
        # [B, N_grid, D]
        decoder_pos_embed = self.get_decoder_patch_query_embed().expand(B, -1, -1)
        
        # 3. Decoder (Perceiver IO or Transformer Decoder)
        # Cross attend latent z to grid decoder_pos_embed
        out = self.decoder(z, decoder_pos_embed)
        
        # 4. Final Projection
        out = self.final_layer(out)
        
        # 5. Unpatchify to Video
        pred_video = self.unpatchify(out)

        return pred_video

    def forward(self, data, **kwargs):
        # data: [B, C, Frame_Num, H, W]
        encode_output = self.encode(data)
        pred_frames = self.decode(encode_output['encoded']).contiguous()
        
        return_dict = {'pred_frames': pred_frames, **encode_output}
        return return_dict





class TokenInteractionLayer(nn.Module):
    """
    专门用于两组 Token 之间的 Cross Attention 交互
    Query: Rest Tokens (256)
    Key/Value: First Frame Tokens (128)
    """
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, context):
        # x: [B, N_rest, C] -> Queries
        # context: [B, N_first, C] -> Keys, Values
        
        # 1. Cross Attention
        residual = x
        x = self.norm(x)
        # attn_output, _ = self.cross_attn(query=x, key=context, value=context)
        # 现在的 PyTorch MultiheadAttention batch_first=True 输入为 (B, L, E)
        attn_output, _ = self.cross_attn(x, context, context)
        x = residual + attn_output
        
        # 2. FFN
        residual = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = residual + x
        return x



@register('cosmos_larp_tokenizer')
class CosmosLARPTokenizer(nn.Module, PyTorchModelHubMixin):
    output_format = 'bcthw'
    def __init__(
        self, 
        bottleneck,
        prior_model,
        # --- Token 数量配置 ---
        token_num_first=128,      # 第一帧的 Token 数
        token_num_rest=256,       # 后续帧的 Token 数
        
        # --- 尺寸配置 ---
        input_size=128,
        frame_num=16,             # 总帧数 1+16
        temporal_patch_size=4,    # 仅用于后续帧
        patch_size=8,
        decoder_temporal_patch_size=4,
        decoder_patch_size=8,
        in_channels=3,

        # --- 模型配置 ---
        transformer_name='transformer_encoder_parallel',
        encoder_name=None,
        decoder_name=None,
        encoder_hidden_size=768,
        decoder_hidden_size=768,
        
        # 为了简化演示，这里假设 encoder/decoder 参数大部分共享配置
        encoder_num_heads=12,
        decoder_num_heads=12,
        encoder_depth=6,
        decoder_depth=6,

        # --- Embedding 配置 ---
        latent_pe_scale_factor=10000,
        query_init_std=0.02,
        encoder_query_gaussian_init=True,
        
        # --- Boolean Flags ---
        learned_encoder_patch_pe=False,
        learned_encoder_latent_query_embed=True,
        learned_decoder_latent_pe=False,
        learned_decoder_patch_query_embed=False,
        
        # 忽略部分复杂 flag 以简化代码...
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.frame_num = frame_num # 应为 17
        
        self.token_num_first = token_num_first
        self.token_num_rest = token_num_rest
        self.bottleneck_token_num = token_num_first + token_num_rest # 总 Codebook 索引数

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size # 针对 rest frames
        
        self.decoder_patch_size = decoder_patch_size
        self.decoder_temporal_patch_size = decoder_temporal_patch_size

        # =========================================================
        # 1. Embedders (切片层)
        # =========================================================
        self.token_h =self.token_w =token_h=token_w = input_size // patch_size
        # A. 第一帧 Embedder (T=1, 使用 2D 逻辑或 3D T=1)
        # 这里强制 temporal_patch_size=1 因为只有一帧
        self.x_embedder_first = PatchEmbed3D(
            input_size, 4, patch_size, temporal_patch_size, in_channels, encoder_hidden_size, bias=True
        )
        # B. 后续帧 Embedder (T=16)
        print("Frame num:", frame_num)
        rest_frames = frame_num - 4
        print("Rest frames:", rest_frames)
        print("Temporal patch size:", temporal_patch_size)
        assert rest_frames % temporal_patch_size == 0
        self.x_embedder_rest = PatchEmbed3D(
            input_size, rest_frames, patch_size, temporal_patch_size, in_channels, encoder_hidden_size, bias=True
        )
        self.token_t_first = token_t_first = self.x_embedder_first.num_temporal_patches
        self.token_t_rest = token_t_rest = self.x_embedder_rest.num_temporal_patches
        # 计算 Patch 数量用于 PE
        self.num_patches_first = self.x_embedder_first.num_spatial_patches*self.x_embedder_first.num_temporal_patches
        self.num_patches_rest = self.x_embedder_rest.num_spatial_patches * self.x_embedder_rest.num_temporal_patches

        self.register_buffer('encoder_patch_pe_first', torch.zeros(1, self.num_patches_first, encoder_hidden_size))
        self.get_encoder_patch_pe_raw_first = lambda: self.encoder_patch_pe_first
        self.get_encoder_patch_pe_first = self.get_encoder_patch_pe_raw_first
        self.register_buffer('encoder_patch_pe_rest', torch.zeros(1, self.num_patches_rest, encoder_hidden_size))
        self.get_encoder_patch_pe_raw_rest = lambda: self.encoder_patch_pe_rest
        self.get_encoder_patch_pe_rest = self.get_encoder_patch_pe_raw_rest


        self.encoder_latent_query_embed_first = nn.Parameter(torch.zeros(token_num_first, encoder_hidden_size), requires_grad=True)
        self.encoder_query_first = lambda: self.encoder_latent_query_embed_first.unsqueeze(0)
        self.encoder_latent_query_embed_rest = nn.Parameter(torch.zeros(token_num_rest, encoder_hidden_size), requires_grad=True)
        self.encoder_query_rest = lambda: self.encoder_latent_query_embed_rest.unsqueeze(0)
        query_embed_first = torch.randn(self.token_num_first, self.encoder_hidden_size) * query_init_std
        self.encoder_latent_query_embed_first.data.copy_(query_embed_first)

        query_embed_rest = torch.randn(self.token_num_rest, self.encoder_hidden_size) * query_init_std
        self.encoder_latent_query_embed_rest.data.copy_(query_embed_rest)


        decoder_h = input_size // decoder_patch_size
        decoder_w = input_size // decoder_patch_size
        # 编码器：可以使用同一个 Transformer 权重处理两组数据（只要 Dim 相同），也可以分开。
        # 这里为了节省显存，使用共享权重的 Encoder (Perceiver IO 结构)
        if encoder_name is None or encoder_name.lower() in ['none', 'no', 'null', '']:
            encoder_name = transformer_name
        if decoder_name is None or decoder_name.lower() in ['none', 'no', 'null', '']:
            decoder_name = transformer_name

        encoder_args = {
            'name': encoder_name,
            'args': {
                'dim': encoder_hidden_size,
                'depth': encoder_depth,
                'n_head': encoder_num_heads,
                'head_dim': encoder_hidden_size // encoder_num_heads,
            }, # the args can be redundant, but redundant args will be filtered out in models.make
        }
        self.encoder_first = models.make(encoder_args)
        self.encoder_rest = models.make(encoder_args)
        #self.decoder = models.make(decoder_args)

        # 瓶颈层 (VQ)
        self.bottleneck_dim = bottleneck['args']['bottleneck_dim']
        bottleneck_args = {
            'token_nums': self.bottleneck_token_num, # 128 + 256
            'input_dim': encoder_hidden_size, 
            'output_dim': decoder_hidden_size
        }
        self.bottleneck = models.make(bottleneck, args=bottleneck_args)
        self.codebook_size = bottleneck['args']['regularizer']['args']['codebook_size']

        # 交互层：Rest Tokens -> First Tokens
        self.interaction_layer = TokenInteractionLayer(decoder_hidden_size, num_heads=decoder_num_heads)
        decoder_args = {
            'name': decoder_name,
            'args': {
                'dim': decoder_hidden_size,
                'depth': decoder_depth,
                'n_head': decoder_num_heads,
                'head_dim': decoder_hidden_size // decoder_num_heads,
            }, # the args can be redundant, but redundant args will be filtered out in models.make
        }
        self.decoder_first = models.make(decoder_args)
        self.decoder_rest = models.make(decoder_args)

        self.final_layer_rest = OutputLayer(decoder_hidden_size, decoder_temporal_patch_size, decoder_patch_size, self.out_channels)
        self.final_layer_first = OutputLayer(decoder_hidden_size, decoder_temporal_patch_size, decoder_patch_size, self.out_channels)

        recon_t_rest = rest_frames // decoder_temporal_patch_size
        recon_hw_rest = (input_size // decoder_patch_size)**2

        self.register_buffer('decoder_patch_query_embed_first', torch.zeros(1, 1 * recon_hw_rest, decoder_hidden_size))
        self.get_decoder_patch_query_embed_raw_first = lambda: self.decoder_patch_query_embed_first
        self.decoder_patch_query_token_type_embed_first = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size), requires_grad=True)
        self.get_decoder_patch_query_embed_first = lambda: self.get_decoder_patch_query_embed_raw_first() + self.decoder_patch_query_token_type_embed_first


        self.register_buffer('decoder_patch_query_embed_rest', torch.zeros(1, recon_t_rest * recon_hw_rest, decoder_hidden_size))
        self.get_decoder_patch_query_embed_raw_rest = lambda: self.decoder_patch_query_embed_rest
        self.decoder_patch_query_token_type_embed_rest = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size), requires_grad=True)
        self.get_decoder_patch_query_embed_rest= lambda: self.get_decoder_patch_query_embed_raw_rest() + self.decoder_patch_query_token_type_embed_rest


        # decoder latent PE
        self.learned_decoder_latent_pe = learned_decoder_latent_pe

        self.register_buffer('decoder_latent_pe_first', torch.zeros(1, self.token_num_first, decoder_hidden_size))
        self.get_decoder_latent_pe_first = lambda: self.decoder_latent_pe_first


        self.register_buffer('decoder_latent_pe_rest', torch.zeros(1, self.token_num_rest, decoder_hidden_size))
        self.get_decoder_latent_pe_rest = lambda: self.decoder_latent_pe_rest

        self.prior_model = None 
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        token_h, token_w = self.token_h, self.token_w

        encoder_pos_embed_first = get_3d_sincos_pos_embed(self.encoder_hidden_size, token_h, self.token_t_first)
        self.encoder_patch_pe_first.data.copy_(torch.from_numpy(encoder_pos_embed_first).float().reshape_as(self.encoder_patch_pe_first))

        encoder_pos_embed_rest = get_3d_sincos_pos_embed(self.encoder_hidden_size, token_h, self.token_t_rest)
        self.encoder_patch_pe_rest.data.copy_(torch.from_numpy(encoder_pos_embed_rest).float().reshape_as(self.encoder_patch_pe_rest))



        decoder_token_embed_first = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, np.arange(self.token_num_first), 10000)
        decoder_token_embed_first = torch.from_numpy(decoder_token_embed_first).float().reshape(1, self.token_num_first, self.decoder_hidden_size)
        self.decoder_latent_pe_first.data.copy_(decoder_token_embed_first)
        decoder_token_embed_rest = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, np.arange(self.token_num_rest), 10000)
        decoder_token_embed_rest = torch.from_numpy(decoder_token_embed_rest).float().reshape(1, self.token_num_rest, self.decoder_hidden_size)
        self.decoder_latent_pe_rest.data.copy_(decoder_token_embed_rest)

        decoder_pos_embed_first = get_3d_sincos_pos_embed(self.decoder_hidden_size, self.token_h, self.token_t_first)
        self.decoder_patch_query_embed_first.data.copy_(torch.from_numpy(decoder_pos_embed_first).float().reshape_as(self.decoder_patch_query_embed_first))
        decoder_patch_query_token_type_embed_first = torch.randn(1, 1, self.decoder_hidden_size) * .02
        self.decoder_patch_query_token_type_embed_first.data.copy_(decoder_patch_query_token_type_embed_first)

        decoder_pos_embed_rest = get_3d_sincos_pos_embed(self.decoder_hidden_size, self.token_h, self.token_t_rest)
        self.decoder_patch_query_embed_rest.data.copy_(torch.from_numpy(decoder_pos_embed_rest).float().reshape_as(self.decoder_patch_query_embed_rest))
        decoder_patch_query_token_type_embed_rest = torch.randn(1, 1, self.decoder_hidden_size) * .02
        self.decoder_patch_query_token_type_embed_rest.data.copy_(decoder_patch_query_token_type_embed_rest)
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder_first.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder_first.proj.bias, 0)

        w1 = self.x_embedder_rest.proj.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        nn.init.constant_(self.x_embedder_rest.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer_first.linear.weight, 0)
        nn.init.constant_(self.final_layer_first.linear.bias, 0)
        nn.init.constant_(self.final_layer_rest.linear.weight, 0)
        nn.init.constant_(self.final_layer_rest.linear.bias, 0)
    def decoder_parameters(self):
        decoder_params = itertools.chain(
            self.decoder_first.parameters(),
            self.decoder_rest.parameters(),
            self.final_layer_first.parameters(),
            self.final_layer_rest.parameters(),
            self.decoder_patch_query_token_type_embed_first, 
            self.decoder_patch_query_token_type_embed_rest,
            self.interaction_layer.parameters(),
        )
        return decoder_params

    def decoder_requires_grad_(self, requires_grad):
        for param in self.decoder_parameters():
            param.requires_grad_(requires_grad)

    def others_parameters(self):
        decoder_params_set = set(self.decoder_parameters())
        return (p for p in self.parameters() if p not in decoder_params_set)

    def others_requires_grad_(self, requires_grad):
        for param in self.others_parameters():
            param.requires_grad_(requires_grad)
    def encode(self, x):
        B = x.shape[0]
        x_first = x[:, :, 0:4, :, :]  # [B, C, 1, H, W]
        x_rest = x[:, :, 4:, :, :]   
        feat_first = self.x_embedder_first(x_first) + self.get_encoder_patch_pe_first() # [B, N_patch_1, D]
        feat_rest = self.x_embedder_rest(x_rest) + self.get_encoder_patch_pe_rest()    # [B, N_patch_rest, D]
        q_first = self.encoder_query_first().expand(B, -1, -1) # [B, 128, D]
        q_rest = self.encoder_query_rest().expand(B, -1, -1)   # [B, 256, D]
        z_first = self.encoder_first(feat_first, q_first) # [B, 128, D]
        z_rest = self.encoder_rest(feat_rest, q_rest)    # [B, 256, D]

        # 4. Concatenate for Bottleneck
        z_combined = torch.cat([z_first, z_rest], dim=1) # [B, 384, D]

        # 5. VQ Bottleneck
        bottleneck_out = self.bottleneck(z_combined)
        z_quantized = bottleneck_out.pop('output') # [B, 384, D]

        return {
            'encoded': z_quantized, 
            **bottleneck_out
        }

    def unpatchify(self, x):
        """
        x: (b, n, t_patch_size * s_patch_size**2 * c)
        videos: (b, c, t, h, w)
        """
        c = self.out_channels
        pt = self.temporal_patch_size
        p = self.patch_size
        h = w = self.token_h
        t = x.size(1) // (h * w)

        x = x.reshape(-1, t, h, w, pt, p, p, c)
        x = einops.rearrange(x, 'b t h w pt p1 p2 c -> b c (t pt) (h p1) (w p2)')
        return x

    def decode(self, z):
        # z: [B, 384, D] (Quantized Latent)
        B = z.shape[0]

        # 1. Split Latent back
        z_first = z[:, :self.token_num_first, :] # [B, 128, D]
        z_rest = z[:, self.token_num_first:, :]  # [B, 256, D]
        z_rest = self.interaction_layer(x=z_rest, context=z_first)
        decoder_token_embed_first = self.get_decoder_latent_pe_first()
        z_first = z_first + decoder_token_embed_first 
        decoder_pos_embed_first = self.get_decoder_patch_query_embed_first().expand(B, -1, -1)
        out_first = self.decoder_first(z_first, decoder_pos_embed_first)
        out_first = self.final_layer_first(out_first)
        out_first = self.unpatchify(out_first)

        decoder_token_embed_rest = self.get_decoder_latent_pe_rest()
        z_rest = z_rest + decoder_token_embed_rest 
        decoder_pos_embed_rest = self.get_decoder_patch_query_embed_rest().expand(B, -1, -1)
        out_rest = self.decoder_rest(z_rest, decoder_pos_embed_rest)
        out_rest = self.final_layer_rest(out_rest)
        out_rest = self.unpatchify(out_rest)

        # 5. Final Concatenation
        pred_video = torch.cat([out_first, out_rest], dim=2)

        return pred_video
    def forward(self, data, **kwargs):
        # data: [B, C, 17, H, W]
        encode_output = self.encode(data)
        pred_frames = self.decode(encode_output['encoded']).contiguous()
        
        return_dict = {'pred_frames': pred_frames, **encode_output}
        return return_dict






class OutputLayer(nn.Module):
    def __init__(self, hidden_size, temporal_patch_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, temporal_patch_size * patch_size * patch_size * out_channels, bias=True)

    def forward(self, x):
        # x: [b, n, c]
        x = self.norm_final(x)
        x = self.linear(x)
        return x


@register('larp_tokenizer')
class LARPTokenizer(nn.Module, PyTorchModelHubMixin):
    output_format = 'bcthw'
    def __init__(
        self, 
        bottleneck,
        prior_model,
        bottleneck_token_num=1024,
        input_size=128,
        frame_num=16,
        temporal_patch_size=4,
        patch_size=8,
        decoder_temporal_patch_size=4,
        decoder_patch_size=8,
        in_channels=3,

        transformer_name='transformer_encoder_parallel',
        encoder_name=None,
        decoder_name=None,
        latent_pe_scale_factor=10000,
        query_init_std=0.02,
        encoder_hidden_size=768,
        decoder_hidden_size=768,
        encoder_num_heads=12,
        decoder_num_heads=12,
        encoder_depth=6,
        decoder_depth=6,

        learned_encoder_patch_pe=False,
        learned_encoder_latent_query_embed=True,
        learned_decoder_latent_pe=False,
        learned_decoder_patch_query_embed=False,

        use_encoder_patch_token_type_embed=False,
        use_encoder_latent_query_token_type_embed=False,
        use_decoder_latent_token_type_embed=False,
        use_decoder_patch_query_token_type_embed=False,

        encoder_query_gaussian_init=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.frame_num = frame_num
        self.bottleneck_token_num = bottleneck_token_num
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.decoder_temporal_patch_size = decoder_temporal_patch_size
        self.decoder_patch_size = decoder_patch_size
        self.decoder_latent_len = bottleneck_token_num

        self.encoder_hidden_size = encoder_hidden_size = int(encoder_hidden_size)
        self.decoder_hidden_size = decoder_hidden_size = int(decoder_hidden_size)
        self.encoder_num_heads = encoder_num_heads = int(encoder_num_heads)
        self.decoder_num_heads = decoder_num_heads = int(decoder_num_heads)

        self.latent_pe_scale_factor = latent_pe_scale_factor
        self.query_init_std = query_init_std

        if temporal_patch_size == 1:
            self.x_embedder = VideoPatchEmbed(input_size, patch_size, in_channels, encoder_hidden_size, bias=True, frame_num=frame_num)
        else:
            assert temporal_patch_size > 1
            self.x_embedder = PatchEmbed3D(input_size, frame_num, patch_size, temporal_patch_size, in_channels, encoder_hidden_size, bias=True)
        self.token_h = token_h = self.token_w = token_w = int(self.x_embedder.num_spatial_patches ** 0.5)
        self.token_t = token_t = self.x_embedder.num_temporal_patches
        self.video_token_num = video_token_num = self.x_embedder.num_spatial_patches * token_t
        assert input_size % decoder_patch_size == 0, "input_size must be divisible by decoder_patch_size"
        self.decoder_token_t = decoder_token_t = frame_num // decoder_temporal_patch_size
        decoder_token_h = decoder_token_w = input_size // decoder_patch_size
        recon_num_patches_per_frame = decoder_token_h * decoder_token_w
        self.decoder_token_h = self.decoder_token_w = decoder_token_h
        self.recon_video_token_num = recon_video_token_num = recon_num_patches_per_frame * decoder_token_t


        # encoder patch PE
        self.learned_encoder_patch_pe = learned_encoder_patch_pe
        if self.learned_encoder_patch_pe:
            self.encoder_h_embed = nn.Parameter(torch.zeros(1, 1, token_h, 1, encoder_hidden_size), requires_grad=True)
            self.encode_w_embed = nn.Parameter(torch.zeros(1, 1, 1, token_w, encoder_hidden_size), requires_grad=True)
            self.encoder_t_embed = nn.Parameter(torch.zeros(1, token_t, 1, 1, encoder_hidden_size), requires_grad=True)
            self.get_encoder_patch_pe_raw = lambda: (self.encoder_h_embed + self.encode_w_embed + self.encoder_t_embed).reshape(1, video_token_num, encoder_hidden_size)
        else:
            self.register_buffer('encoder_patch_pe', torch.zeros(1, video_token_num, encoder_hidden_size))
            self.get_encoder_patch_pe_raw = lambda: self.encoder_patch_pe
        self.use_encoder_patch_token_type_embed = use_encoder_patch_token_type_embed
        if self.use_encoder_patch_token_type_embed:
            self.encoder_patch_token_type_embed = nn.Parameter(torch.zeros(1, 1, encoder_hidden_size), requires_grad=True)
            self.get_encoder_patch_pe = lambda: self.get_encoder_patch_pe_raw() + self.encoder_patch_token_type_embed
        else:
            self.get_encoder_patch_pe = self.get_encoder_patch_pe_raw

        # encoder latent query embed        learned_encoder_latent_query_embed: true
        self.learned_encoder_latent_query_embed = learned_encoder_latent_query_embed
        self.encoder_query_gaussian_init = encoder_query_gaussian_init
        if self.learned_encoder_latent_query_embed:
            self.encoder_latent_query_embed = nn.Parameter(torch.zeros(bottleneck_token_num, encoder_hidden_size), requires_grad=True)
        else:
            self.register_buffer('encoder_latent_query_embed', torch.zeros(bottleneck_token_num, encoder_hidden_size))
            assert not encoder_query_gaussian_init, "encoder_query_gaussian_init requires learned_encoder_latent_query_embed to be True"
        self.use_encoder_latent_query_token_type_embed = use_encoder_latent_query_token_type_embed
        if self.use_encoder_latent_query_token_type_embed:
            self.encoder_latent_query_token_type_embed = nn.Parameter(torch.zeros(1, 1, encoder_hidden_size), requires_grad=True)
            self.get_encoder_latent_query_embed = lambda: self.encoder_latent_query_embed.unsqueeze(0) + self.encoder_latent_query_token_type_embed
        else:
            self.get_encoder_latent_query_embed = lambda: self.encoder_latent_query_embed.unsqueeze(0)

        # decoder latent PE
        self.learned_decoder_latent_pe = learned_decoder_latent_pe
        if self.learned_decoder_latent_pe:
            self.decoder_latent_pe = nn.Parameter(torch.zeros(1, self.decoder_latent_len, decoder_hidden_size), requires_grad=True)
        else: 
            self.register_buffer('decoder_latent_pe', torch.zeros(1, self.decoder_latent_len, decoder_hidden_size))
        self.use_decoder_latent_token_type_embed = use_decoder_latent_token_type_embed
        if self.use_decoder_latent_token_type_embed:
            self.decoder_latent_token_type_embed = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size), requires_grad=True)
            self.get_decoder_latent_pe = lambda: self.decoder_latent_pe + self.decoder_latent_token_type_embed
        else: 
            self.get_decoder_latent_pe = lambda: self.decoder_latent_pe

        # decoder patch query embed
        self.learned_decoder_patch_query_embed = learned_decoder_patch_query_embed
        if self.learned_decoder_patch_query_embed:
            self.decoder_h_embed = nn.Parameter(torch.zeros(1, 1, decoder_token_h, 1, decoder_hidden_size), requires_grad=True)
            self.decoder_w_embed = nn.Parameter(torch.zeros(1, 1, 1, decoder_token_w, decoder_hidden_size), requires_grad=True)
            self.decoder_t_embed = nn.Parameter(torch.zeros(1, decoder_token_t, 1, 1, decoder_hidden_size), requires_grad=True)
            self.get_decoder_patch_query_embed_raw = lambda: (self.decoder_h_embed + self.decoder_w_embed + self.decoder_t_embed).reshape(1, recon_video_token_num, decoder_hidden_size)
        else:
            self.register_buffer('decoder_patch_query_embed', torch.zeros(1, recon_video_token_num, decoder_hidden_size))
            self.get_decoder_patch_query_embed_raw = lambda: self.decoder_patch_query_embed
        self.use_decoder_patch_query_token_type_embed = use_decoder_patch_query_token_type_embed
        if self.use_decoder_patch_query_token_type_embed:
            self.decoder_patch_query_token_type_embed = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size), requires_grad=True)
            self.get_decoder_patch_query_embed = lambda: self.get_decoder_patch_query_embed_raw() + self.decoder_patch_query_token_type_embed
        else: 
            self.get_decoder_patch_query_embed = self.get_decoder_patch_query_embed_raw


        # Build encoder, decoder, and bottleneck
        if encoder_name is None or encoder_name.lower() in ['none', 'no', 'null', '']:
            encoder_name = transformer_name
        if decoder_name is None or decoder_name.lower() in ['none', 'no', 'null', '']:
            decoder_name = transformer_name

        encoder_args = {
            'name': encoder_name,
            'args': {
                'dim': encoder_hidden_size,
                'depth': encoder_depth,
                'n_head': encoder_num_heads,
                'head_dim': encoder_hidden_size // encoder_num_heads,
            }, # the args can be redundant, but redundant args will be filtered out in models.make
        }

        decoder_args = {
            'name': decoder_name,
            'args': {
                'dim': decoder_hidden_size,
                'depth': decoder_depth,
                'n_head': decoder_num_heads,
                'head_dim': decoder_hidden_size // decoder_num_heads,
            }, # the args can be redundant, but redundant args will be filtered out in models.make
        }

        self.encoder = models.make(encoder_args)
        self.decoder = models.make(decoder_args)

        self.bottleneck_dim = bottleneck['args']['bottleneck_dim']
        bottleneck_args = {'token_nums': self.bottleneck_token_num, 'input_dim': encoder_hidden_size, 'output_dim': decoder_hidden_size}
        self.bottleneck = models.make(bottleneck, args=bottleneck_args)
        self.codebook_size = bottleneck['args']['regularizer']['args']['codebook_size']
        self.final_layer = OutputLayer(decoder_hidden_size, decoder_temporal_patch_size, decoder_patch_size, self.out_channels)


        # Build prior model
        prior_model = edict(prior_model)
        # if prior_model.get('name', '').lower() in ['none', 'no', 'null', '']:
        self.prior_model = None
        # else:
        #     prior_model_additional_args = {'n_ind': self.bottleneck_dim, 'n_classes': self.codebook_size}
        #     if prior_model.get('no_dropout', False):
        #         prior_model_additional_args['embd_pdrop'] = 0.0
        #         prior_model_additional_args['resid_pdrop'] = 0.0
        #         prior_model_additional_args['attn_pdrop'] = 0.0
        #         print(f"Warning: prior_loss is using no dropout")


        #     self.prior_model = models.make(prior_model, args=prior_model_additional_args)
        #     self.prior_n_rounds = prior_model.n_rounds
        #     self.prior_no_grad_before_last_round = prior_model.no_grad_before_last_round
        #     self.prior_avg_loss_over_rounds = prior_model.avg_loss_over_rounds
        #     self.use_mix_ss = prior_model.use_mix_ss
        #     self.mix_ss_max_ratio = prior_model.mix_ss_max_ratio
        #     self.mix_ss_peak_steps_ratio = prior_model.mix_ss_peak_steps_ratio
        #     self.prior_latent_ce_temperature = prior_model.latent_ce_temperature
        
        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        token_h, token_w = self.token_h, self.token_w

        # Initialize encoder patch PE
        if self.learned_encoder_patch_pe:
            h_embed = get_1d_sincos_pos_embed_from_grid(self.encoder_hidden_size, np.arange(token_h))
            w_embed = get_1d_sincos_pos_embed_from_grid(self.encoder_hidden_size, np.arange(token_w))
            t_embed = get_1d_sincos_pos_embed_from_grid(self.encoder_hidden_size, np.arange(self.token_t))
            self.encoder_h_embed.data.copy_(torch.from_numpy(h_embed).float().reshape_as(self.encoder_h_embed))
            self.encode_w_embed.data.copy_(torch.from_numpy(w_embed).float().reshape_as(self.encode_w_embed))
            self.encoder_t_embed.data.copy_(torch.from_numpy(t_embed).float().reshape_as(self.encoder_t_embed))
        else:
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            encoder_pos_embed = get_3d_sincos_pos_embed(self.encoder_hidden_size, token_h, self.token_t)
            self.encoder_patch_pe.data.copy_(torch.from_numpy(encoder_pos_embed).float().reshape_as(self.encoder_patch_pe))
        if self.use_encoder_patch_token_type_embed:
            encoder_patch_token_type_embed = torch.randn(1, 1, self.encoder_hidden_size) * .02
            self.encoder_patch_token_type_embed.data.copy_(encoder_patch_token_type_embed)

        # Initialize encoder latent query embed
        if self.learned_encoder_latent_query_embed:
            if self.encoder_query_gaussian_init:
                # from timm vision_transformer.py
                # https://github.com/huggingface/pytorch-image-models/blob/70ccf00c95a2d78a166cca24ef6adbca46f47c2a/timm/models/vision_transformer.py#L495
                query_embed = torch.randn(self.bottleneck_token_num, self.encoder_hidden_size) * self.query_init_std
            else:
                query_embed = get_1d_sincos_pos_embed_from_grid(self.encoder_hidden_size, np.arange(self.bottleneck_token_num))
                query_embed = torch.from_numpy(query_embed).float().reshape(self.bottleneck_token_num, self.encoder_hidden_size)
        else:
            query_embed = get_1d_sincos_pos_embed_from_grid(self.encoder_hidden_size, np.arange(self.bottleneck_token_num), self.latent_pe_scale_factor)
            query_embed = torch.from_numpy(query_embed).float().reshape(self.bottleneck_token_num, self.encoder_hidden_size)
        self.encoder_latent_query_embed.data.copy_(query_embed)
        if self.use_encoder_latent_query_token_type_embed:
            encoder_latent_query_token_type_embed = torch.randn(1, 1, self.encoder_hidden_size) * .02
            self.encoder_latent_query_token_type_embed.data.copy_(encoder_latent_query_token_type_embed)

        # initialize decoder latent PE
        if self.learned_decoder_latent_pe:
            decoder_token_embed = torch.randn(1, self.decoder_latent_len, self.decoder_hidden_size) * .02
            self.decoder_latent_pe.data.copy_(decoder_token_embed)
        else:
            decoder_token_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, np.arange(self.decoder_latent_len), self.latent_pe_scale_factor)
            decoder_token_embed = torch.from_numpy(decoder_token_embed).float().reshape(1, self.decoder_latent_len, self.decoder_hidden_size)
            self.decoder_latent_pe.data.copy_(decoder_token_embed)
        if self.use_decoder_latent_token_type_embed:
            decoder_latent_token_type_embed = torch.randn(1, 1, self.decoder_hidden_size) * .02
            self.decoder_latent_token_type_embed.data.copy_(decoder_latent_token_type_embed)

        # initialize decoder patch query PE
        if self.learned_decoder_patch_query_embed:
            h_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, np.arange(self.decoder_token_h))
            w_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, np.arange(self.decoder_token_w))
            t_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, np.arange(self.decoder_token_t))
            self.decoder_h_embed.data.copy_(torch.from_numpy(h_embed).float().reshape_as(self.decoder_h_embed))
            self.decoder_w_embed.data.copy_(torch.from_numpy(w_embed).float().reshape_as(self.decoder_w_embed))
            self.decoder_t_embed.data.copy_(torch.from_numpy(t_embed).float().reshape_as(self.decoder_t_embed))
        else:
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_hidden_size, self.decoder_token_h, self.decoder_token_t)
            self.decoder_patch_query_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().reshape_as(self.decoder_patch_query_embed))
        if self.use_decoder_patch_query_token_type_embed:
            decoder_patch_query_token_type_embed = torch.randn(1, 1, self.decoder_hidden_size) * .02
            self.decoder_patch_query_token_type_embed.data.copy_(decoder_patch_query_token_type_embed)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def get_last_layer(self):
        return self.final_layer.linear.weight

    def set_vq_eval_deterministic(self, deterministic=True):
        self.bottleneck.regularizer.set_eval_deterministic(deterministic)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def decoder_parameters(self):
        decoder_params = itertools.chain(
            self.decoder.parameters(),
            self.final_layer.parameters()
        )

        if self.learned_decoder_patch_query_embed:
            decoder_params = itertools.chain(
                decoder_params,
                [self.decoder_h_embed, self.decoder_w_embed, self.decoder_t_embed]
            )

        if self.learned_decoder_latent_pe:
            decoder_params = itertools.chain(
                decoder_params,
                [self.decoder_latent_pe]
            )

        return decoder_params

    def decoder_requires_grad_(self, requires_grad):
        for param in self.decoder_parameters():
            param.requires_grad_(requires_grad)

    def others_parameters(self):
        decoder_params_set = set(self.decoder_parameters())
        return (p for p in self.parameters() if p not in decoder_params_set)

    def others_requires_grad_(self, requires_grad):
        for param in self.others_parameters():
            param.requires_grad_(requires_grad)

    @classmethod
    def from_checkpoint(cls, ckpt, load_state_dict=True, version='sd'):
        if isinstance(ckpt, str):
            assert os.path.exists(ckpt), f"checkpoint {ckpt} does not exist"
            ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
        else:
            assert isinstance(
                ckpt, dict
            ), f"checkpoint must be a dict or a path to a checkpoint"

        kwargs = ckpt["model"]["args"]
        model = cls(**kwargs)
        if load_state_dict:
            if version == 'sd':
                sd = ckpt["model"]["sd"]
            elif version.startswith('ema'):
                assert '_' in version, "ema version must be in the format 'ema_{alpha}'"
                alpha = float(version.split('_')[1])
                sd = ckpt["model"]['ema_sd'][alpha]
            else:
                raise ValueError(f"Unknown version: {version}")
            model.load_state_dict(sd, strict=True)
        return model

    def encode(self, x):
        x = self.x_embedder(x) + self.get_encoder_patch_pe() # (b, n, d)
        #print("x_embedder output shape:", x.shape)
        b = x.shape[0]
        q_emb = self.get_encoder_latent_query_embed().repeat(b, 1, 1) # (b, n, d)
        z = self.encoder(x, q_emb)
        bottleneck_out = self.bottleneck(z)
        z = bottleneck_out.pop('output')
        return {'encoded': z, **bottleneck_out}

    def encode_eval(self, x):
        x_tokens = self.x_embedder(x)
        num_x_tokens = x_tokens.size(1)
        x = x_tokens + self.get_encoder_patch_pe()[:, :num_x_tokens, :] # (b, n, d) # can encode fewer frames
        b = x.shape[0]
        q_emb = self.get_encoder_latent_query_embed().repeat(b, 1, 1) # (b, n, d)
        z = self.encoder(x, q_emb)
        bottleneck_out = self.bottleneck(z)
        z = bottleneck_out.pop('output')
        return {'encoded': z, **bottleneck_out, 'num_x_tokens': num_x_tokens}

    def unpatchify(self, x):
        """
        x: (b, n, t_patch_size * s_patch_size**2 * c)
        videos: (b, c, t, h, w)
        """
        c = self.out_channels
        pt = self.temporal_patch_size
        p = self.patch_size
        h = w = self.token_h
        t = x.size(1) // (h * w)

        x = x.reshape(-1, t, h, w, pt, p, p, c)
        x = einops.rearrange(x, 'b t h w pt p1 p2 c -> b c (t pt) (h p1) (w p2)')
        return x

    def decode(self, z):
        # z: (b, n, d)
        b = z.size(0)

        decoder_token_embed = self.get_decoder_latent_pe()
        z = z + decoder_token_embed 
        decoder_pos_embed = self.get_decoder_patch_query_embed().expand(b, -1, -1)
        x = self.decoder(z, decoder_pos_embed)
        x = self.final_layer(x)
        x = self.unpatchify(x)
        return x

    def decode_eval(self, z, num_x_tokens=None):
        # z: (b, n, d)
        b = z.size(0)
        decoder_token_embed = self.get_decoder_latent_pe()
        z = z + decoder_token_embed 
        decoder_pos_embed = self.get_decoder_patch_query_embed().expand(b, -1, -1)
        if num_x_tokens is not None:
            decoder_pos_embed = decoder_pos_embed[:, :num_x_tokens, :]
        x = self.decoder(z, decoder_pos_embed)
        x = self.final_layer(x)
        x = self.unpatchify(x)
        return x

    def decode_from_bottleneck(self, bottleneck_rep):
        # This method is only used when this module is used as a first-stage model
        z = self.bottleneck.decode(bottleneck_rep) # (b, n, c)
        return self.decode(z)

    def forward(self, data, **kwargs):
        # data: video in shape (b, c, t, h, w)
        B = data.size(0)
        encode_output = self.encode(data)
        pred_frames = self.decode(encode_output['encoded']).contiguous() # [b, c, t, h, w]
        return_dict = {'pred_frames': pred_frames, **encode_output}

        if self.prior_model is not None:
            results = self.calculate_prior_loss_with_pred(encode_output, **kwargs)
            return_dict.update(results)

        return return_dict

    def get_emb(self):
        emb = self.bottleneck.regularizer.get_emb()
        emb = emb.detach()
        return emb

    def calculate_prior_loss_with_pred(self, encode_output, **kwargs):
        return_dict = {}
        B = encode_output['encoded'].size(0)
        ar_input = encode_output['regularized_z'] # (b, n, d=16) normalized

        labels = encode_output['bottleneck_rep'][:, 1:].contiguous() # (b, n - 1)
        logits_all_rounds, ar_pred_cont, regularized_z_ss = self.prior_ar_predict_n_rounds_ss(ar_input, **kwargs) # regularized_z_ss: (b, n, d=16)
        labels_all_rounds = labels.unsqueeze(0).expand(logits_all_rounds.size(0), -1, -1).contiguous() # (n_rounds or 1, b, n - 1)
        
        loss_latent_ce = F.cross_entropy(logits_all_rounds.view(-1, self.codebook_size), labels_all_rounds.view(-1))
        return_dict['loss_latent_ce'] = loss_latent_ce
        topk_accuracies = utils.calculate_topk_accuracy(logits_all_rounds[0], labels, topk=(1, 5), prepend='prior_')
        return_dict.update(topk_accuracies)

        return return_dict


    def logits_to_token_embedding_with_ss(self, logits, ar_input_staring_from_idx_1, mask=None, **kwargs):
        # logits: (b, n - 1, codebook_size), sequence index from 1 to n-1 (inclusive)
        # ar_input_staring_from_idx_1: (b, n - 1, d=16), requires_grad=True
        if mask is None:
            b, n_minus_1, _ = logits.size()
            if self.use_mix_ss:
                ss_ratio = (kwargs['global_step'] / (kwargs['max_steps'] * self.mix_ss_peak_steps_ratio )) * self.mix_ss_max_ratio
                ss_ratio = min(ss_ratio, self.mix_ss_max_ratio)
            else:
                ss_ratio = 1.0

            mask = torch.rand(b, n_minus_1, 1, device=self.device) < ss_ratio
            mask = mask.expand(-1, -1, self.bottleneck_dim) # (b, n - 1, d=16)

        with torch.autocast(device_type='cuda', enabled=False):
            logits = logits.float()
            probs = F.softmax(logits, dim=-1) # (b, n - 1, codebook_size)
            indices = torch.multinomial(probs.view(-1, self.codebook_size), 1).view(*probs.size()[:-1]) # (b, n - 1)
        token_embedding = F.embedding(indices, self.get_emb()) # (b, n - 1, d=16)
        token_embedding = torch.where(mask, token_embedding, ar_input_staring_from_idx_1)

        return token_embedding

    def calculate_logits_and_ar_pred_cont(self, prior_model_output):
        ar_pred_cont = prior_model_output # (b, n, d=16)
        logits = F.linear(prior_model_output, self.get_emb())[:, 1:]
        logits = logits.mul_(1 / self.prior_latent_ce_temperature)
        logits = logits.contiguous() # (b, n - 1, codebook_size)
        return logits, ar_pred_cont

    def prior_ar_predict_n_rounds_ss(self, ar_input, **kwargs):
        prior_model = self.prior_model
        n_rounds = self.prior_n_rounds
        no_grad_before_last_round = self.prior_no_grad_before_last_round

        b, n, _ = ar_input.size()
        n_minus_1 = n - 1
        if self.use_mix_ss:
            global_step = kwargs['global_step']
            max_steps = kwargs['max_steps']
            peak_steps_ratio = torch.tensor(self.mix_ss_peak_steps_ratio, dtype=torch.float32)
            max_ratio = torch.tensor(self.mix_ss_max_ratio, dtype=torch.float32)

            ss_ratio = (global_step / (max_steps * peak_steps_ratio)) * max_ratio
            ss_ratio = torch.min(ss_ratio, max_ratio)
        else:
            ss_ratio = torch.tensor(1.0, dtype=torch.float32)

        mask_ss = torch.rand(b, n_minus_1, 1, device=self.device) < ss_ratio
        mask_ss = mask_ss.expand(-1, -1, self.bottleneck_dim) # (b, n - 1, d=16)

        logits_all_rounds = []
        next_ar_input = ar_input # (b, n, d=16)
        for i in range(n_rounds):
            if no_grad_before_last_round and i < n_rounds - 1:
                # we can not use "with torch.no_grad()" here due to a pytorch's bug!
                # https://github.com/pytorch/pytorch/issues/112583
                prior_model.requires_grad_(False)
                prior_model_output = prior_model.ar_predict(next_ar_input.detach()) # (b, n - 1, codebook_size)
                logits, ar_pred_cont = self.calculate_logits_and_ar_pred_cont(prior_model_output)
                prior_model.requires_grad_(True)
            else:
                prior_model_output = prior_model.ar_predict(next_ar_input) # (b, n - 1, codebook_size) or (b, n, d=16)
                logits, ar_pred_cont = self.calculate_logits_and_ar_pred_cont(prior_model_output)
                logits_all_rounds.append(logits)


            if i < n_rounds - 1:
                token_embedding = self.logits_to_token_embedding_with_ss(logits, ar_input[:, 1:], mask=mask_ss, **kwargs) # (b, n - 1, d=16)
                next_ar_input = torch.cat([ar_input[:, :1], token_embedding], dim=1) # (b, n, d=16)

        if self.prior_avg_loss_over_rounds:
            logits_all_rounds = torch.stack(logits_all_rounds, dim=0) # (n_rounds, b, n - 1, codebook_size)

        else:
            logits_all_rounds = torch.stack([logits_all_rounds[-1]], dim=0) # (1, b, n - 1, codebook_size)

        return logits_all_rounds, ar_pred_cont, next_ar_input # here the next_ar_input is actually the last round's ar_input

def test_cosmos_larp_tokenizer():
    input_size = 128
    # 注意：这里 frame_num 必须是 1 + (N * temporal_patch_size)
    # 例如：1 (首帧) + 16 (后续帧，能被4整除) = 17 帧
    frame_num = 16
    token_num_first = 128
    token_num_rest = 256
    total_tokens = token_num_first + token_num_rest # 384
    
    # 1. Translate YAML config to Python Dict
    config = {
        'bottleneck': {
            'name': 'bottleneck',
            'args': {
                'bottleneck_dim': 16,
                'norm': 'none',
                'regularizer': {
                    'name': 'vq',
                    'args': {
                        'codebook_size': 8192,
                        'commitment_loss_weight': 0.25,
                        'codebook_loss_weight': 1.0,
                        'entropy_loss_weight': 0.0,
                        'entropy_loss_temperature': 0.01,
                        'l2_normalized': True,
                        'stochastic': True,
                        'stochastic_temperature': 0.03
                    }
                }
            }
        },
        'prior_model': {'name': 'no'},
        
        # Cosmos 特有的配置
        'token_num_first': token_num_first,
        'token_num_rest': token_num_rest,
        
        'transformer_name': 'transformer_encoder_parallel',
        'encoder_name': 'none',
        'decoder_name': 'none',
        
        # 虽然类内部会重新计算，但保持一致是个好习惯
        'bottleneck_token_num': total_tokens, 
        
        'input_size': input_size,
        'frame_num': frame_num,
        'temporal_patch_size': 4, # 仅作用于后续的 16 帧
        'patch_size': 8,
        'decoder_temporal_patch_size': 4,
        'decoder_patch_size': 8,
        'in_channels': 3,
        'encoder_hidden_size': 768,
        'decoder_hidden_size': 768,
        'encoder_num_heads': 12,
        'decoder_num_heads': 12,
        'encoder_depth': 6,  # 测试时稍微改小一点深度，跑得快
        'decoder_depth': 6,
        
        'learned_encoder_patch_pe': False,
        'learned_encoder_latent_query_embed': True,
        'learned_decoder_latent_pe': False,
        'learned_decoder_patch_query_embed': False,
        
        # 'use_encoder_latent_query_token_type_embed': True, # 启用我们新加的特性
        # 'use_decoder_latent_token_type_embed': True,       # 启用我们新加的特性
        
        'encoder_query_gaussian_init': True,
        'latent_pe_scale_factor': 10000,
        'query_init_std': 0.02
    }

    # 2. Instantiate Model
    print(f"Initializing CosmosLARPTokenizer with {frame_num} frames...")
    print(f"Structure: 1 Frame ({token_num_first} tokens) + {frame_num-1} Frames ({token_num_rest} tokens)")
    
    # 假设你已经定义了 CosmosLARPTokenizer 类
    model = CosmosLARPTokenizer(**config)
    
    # 移动到 GPU 如果可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model moved to {device}")

    # 3. Create dummy input: (Batch, Channels, Time, Height, Width)
    # 注意这里 Time = 17
    dummy_input = torch.randn(1, 3, frame_num, input_size, input_size).to(device)
    print(f"Input shape: {dummy_input.shape}")

    # 4. Run Forward
    print("Running forward pass...")
    with torch.no_grad(): # 测试不需要梯度
        output = model(dummy_input)

    # 5. Assertions and Checks
    pred_frames = output['pred_frames']
    print(f"Prediction shape: {pred_frames.shape}")
    
    # Check 1: Input/Output Resolution match
    assert pred_frames.shape == dummy_input.shape, \
        f"Shape mismatch! Input {dummy_input.shape}, Output {pred_frames.shape}"
    
    # Check 2: Total Bottleneck Token Count (128 + 256 = 384)
    encoded_z = output['encoded'] # Should be (B, N_total, D)
    print(f"Encoded shape: {encoded_z.shape}")
    assert encoded_z.shape[1] == total_tokens, \
        f"Expected {total_tokens} tokens (128+256), got {encoded_z.shape[1]}"

    # Check 3: Check Codebook Indices
    indices = output['bottleneck_rep']
    print(f"Codebook indices shape: {indices.shape}")
    assert indices.shape == (1, total_tokens)
    
    # Check 4: Check if Interaction Layer works (Implicit check via forward success)
    # 如果 Interaction Layer 维度对不上，Forward 过程中就会报错
    
    print("\n✅ Test Passed: CosmosLARPTokenizer (First/Rest Split) initialized and ran successfully.")

if __name__ == "__main__":
    test_cosmos_larp_tokenizer()

# def test_larp_tokenizer():
#     input_size = 128
#     frame_num = 16
    
#     # 1. Translate YAML config to Python Dict
#     config = {
#         'bottleneck': {
#             'name': 'bottleneck',
#             'args': {
#                 'bottleneck_dim': 16,
#                 'norm': 'none',
#                 'regularizer': {
#                     'name': 'vq',
#                     'args': {
#                         'codebook_size': 8192,
#                         'commitment_loss_weight': 0.25,
#                         'codebook_loss_weight': 1.0,
#                         'entropy_loss_weight': 0.0,
#                         'entropy_loss_temperature': 0.01,
#                         'l2_normalized': True,
#                         'stochastic': True,
#                         'stochastic_temperature': 0.03
#                     }
#                 }
#             }
#         },
#         'prior_model': {'name': 'no'},
#         'transformer_name': 'transformer_encoder_parallel',
#         'encoder_name': 'none',
#         'decoder_name': 'none',
#         'bottleneck_token_num': 1024,
#         'input_size': input_size,
#         'frame_num': frame_num,
#         'temporal_patch_size': 4,
#         'patch_size': 8,
#         'decoder_temporal_patch_size': 4,
#         'decoder_patch_size': 8,
#         'in_channels': 3,
#         'encoder_hidden_size': 768,
#         'decoder_hidden_size': 768,
#         'encoder_num_heads': 12,
#         'decoder_num_heads': 12,
#         'encoder_depth': 12,
#         'decoder_depth': 12,
#         'learned_encoder_patch_pe': False,
#         'learned_encoder_latent_query_embed': True,
#         'learned_decoder_latent_pe': False,
#         'learned_decoder_patch_query_embed': False,
#         'use_encoder_patch_token_type_embed': False,
#         'use_encoder_latent_query_token_type_embed': False,
#         'use_decoder_latent_token_type_embed': False,
#         'use_decoder_patch_query_token_type_embed': True,
#         'encoder_query_gaussian_init': True,
#         'latent_pe_scale_factor': 10000,
#         'query_init_std': 0.02
#     }

#     # 2. Instantiate Model
#     print("Initializing LARPTokenizer...")
#     model = LARPTokenizer(**config)
    
#     # 3. Create dummy input: (Batch, Channels, Time, Height, Width)
#     dummy_input = torch.randn(1, 3, frame_num, input_size, input_size)
#     print(f"Input shape: {dummy_input.shape}")

#     # 4. Run Forward
#     print("Running forward pass...")
#     output = model(dummy_input)

#     # 5. Assertions and Checks
#     pred_frames = output['pred_frames']
#     print(f"Prediction shape: {pred_frames.shape}")
    
#     # Check 1: Input/Output Resolution match
#     assert pred_frames.shape == dummy_input.shape, \
#         f"Shape mismatch! Input {dummy_input.shape}, Output {pred_frames.shape}"
    
#     # Check 2: Bottleneck Token Count
#     encoded_z = output['encoded'] # Should be (B, N, D)
#     print(f"Encoded shape: {encoded_z.shape}")
#     assert encoded_z.shape[1] == config['bottleneck_token_num'], \
#         f"Expected {config['bottleneck_token_num']} tokens, got {encoded_z.shape[1]}"

#     # Check 3: Check Codebook Indices
#     indices = output['bottleneck_rep']
#     print(f"Codebook indices shape: {indices.shape}")
#     assert indices.shape == (1, config['bottleneck_token_num'])

#     print("\n✅ Test Passed: LARPTokenizer initialized and ran successfully with provided config.")

# if __name__ == "__main__":
#     test_larp_tokenizer()