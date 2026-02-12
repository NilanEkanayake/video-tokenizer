import torch
import torch.nn as nn
from models.model_cnnvit.base.blocks import Encoder, Decoder ,Decoder_unify
from models.model_cnnvit.quantizer.fsq import FSQ
from models import register


@register('autoencoder_cnnvit')
class AutoEncoder(nn.Module):
    def __init__(self, 
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
        
        **kwargs):
        super().__init__()

        in_grid =  [16, 128, 128]
        token_size = 6

        self.encoder = Encoder(
            model_size='small_thin',
            patch_size=[4, 8, 8],
            in_channels=3,
            out_channels=token_size,
            in_grid=in_grid,
            out_tokens=1024,
        )
        self.quantize = FSQ(levels=[8, 8,8, 5, 5, 5])
        self.decoder = Decoder(
            model_size='small_thin',
            patch_size=[4, 8, 8],
            in_channels=token_size,
            out_channels=3,
            in_tokens=1024,
            out_grid=in_grid,
        )
        self.prior_model = None 
    def encode(self, data, **kwargs):
        x_first = data[:, :, 0:1, :, :]  # [B, C, 1, H, W]
        x = self.encoder(data)
        x_q, x_dict = self.quantize(x)
        return x_q, x_dict
    
    def decode(self, x):
        x = self.decoder(x)
        return x
    
    def decode_indices(self, indices):
        x_q = self.quantize.indices_to_codes(indices).to(indices.device, next(self.decoder.parameters()).dtype)
        return self.decoder(x_q)
    
    def forward(self, x):
        x_q, out_dict = self.encode(x)
        x = self.decode(x_q)
        return_dict = {'pred_frames': x.contiguous()}
        return return_dict

@register('autoencoder_large')
class AutoEncoder(nn.Module):
    def __init__(self, 
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
        
        **kwargs):
        super().__init__()

        in_grid =  [16, 128, 128]
        token_size = 6

        self.encoder = Encoder(
            model_size='large',
            patch_size=[4, 8, 8],
            in_channels=3,
            out_channels=token_size,
            in_grid=in_grid,
            out_tokens=1024,
        )
        self.quantize = FSQ(levels=[8, 8,8, 5, 5, 5])
        self.decoder = Decoder(
            model_size='large',
            patch_size=[4, 8, 8],
            in_channels=token_size,
            out_channels=3,
            in_tokens=1024,
            out_grid=in_grid,
        )
        self.prior_model = None 
    def encode(self, data, **kwargs):
        x_first = data[:, :, 0:1, :, :]  # [B, C, 1, H, W]
        x = self.encoder(data)
        x_q, x_dict = self.quantize(x)
        return x_q, x_dict
    
    def decode(self, x):
        x = self.decoder(x)
        return x
    
    def decode_indices(self, indices):
        x_q = self.quantize.indices_to_codes(indices).to(indices.device, next(self.decoder.parameters()).dtype)
        return self.decoder(x_q)
    
    def forward(self, x):
        x_q, out_dict = self.encode(x)
        x = self.decode(x_q)
        return_dict = {'pred_frames': x}
        return return_dict



@register('autoencoder_first_token_f256t1024')
class AutoEncoder_first_token(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        in_grid = [16, 128, 128]
        token_size = 6
        
        # 主编码器：处理整个视频流（或者侧重于时间变化）
        self.encoder = Encoder(
            model_size='base',
            patch_size=[4, 8, 8],
            in_channels=3,
            out_channels=token_size,
            in_grid=in_grid,
            out_tokens=1024,
        )
        
        # 第一帧编码器：专门提供高保真的空间参考
        # out_tokens=256 (1x128x128 grid -> patch 1x8x8 -> 16x16 tokens = 256)
        self.encoder1 = Encoder(
            model_size='base',
            patch_size=[1, 8, 8],
            in_channels=3,
            out_channels=token_size,
            in_grid=[1, 128, 128],
            out_tokens=256, 
        )
        
        self.quantize = FSQ(levels=[8, 8, 8,5, 5, 5])
        
        # 这里的 cond_tokens 对应 encoder1 的 out_tokens
        self.decoder = Decoder_unify(
            model_size='base',
            patch_size=[4, 8, 8],
            in_channels=token_size,
            out_channels=3,
            in_tokens=1024,
            cond_tokens=256,  # <--- 新增：告诉Decoder有多少个第一帧的参考token
            out_grid=in_grid,
        )
        self.prior_model = None 

    def encode(self, data, **kwargs):
        # data: [B, C, T, H, W]
        x_first = data[:, :, 0:1, :, :]  # 取第一帧 [B, C, 1, H, W]
        # 1. 编码主视频流
        x = self.encoder(data)
        # 2. 编码第一帧
        first_token = self.encoder1(x_first)
        # 3. 量化
        # 这里有两个选择：
        # A. 共享量化器 (如果First Token也需要被压缩成离散码) - 推荐
        # B. 仅量化主视频，第一帧保持连续特征 (作为Condition)
        # 这里演示方案A，共享量化器，方便后续训练Transformer
        x_q, x_dict = self.quantize(x)
        first_q, first_dict = self.quantize(first_token) 
        return x_q, first_q
    
    def decode(self, x, first_token):
        # x: 主视频 latent tokens
        # first_token: 第一帧 latent tokens (作为参考)
        x = self.decoder(x, first_token)
        return x
    
    def decode_indices(self, indices, first_indices):
        # 从索引恢复视频
        x_q = self.quantize.indices_to_codes(indices).to(indices.device, next(self.decoder.parameters()).dtype)
        first_q = self.quantize.indices_to_codes(first_indices).to(first_indices.device, next(self.decoder.parameters()).dtype)
        return self.decoder(x_q, first_q)
    
    def forward(self, x):
        # x: [B, 3, 16, 128, 128]
        x_q, first_q = self.encode(x)
        
        # Decoder 接收两部分输入
        pred_frames = self.decode(x_q, first_q)
        return_dict = {'pred_frames': pred_frames}   
        return return_dict
    


@register('autoencoder_first_token_f256t768')
class AutoEncoder_first_token(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        in_grid = [16, 128, 128]
        token_size = 6
        
        # 主编码器：处理整个视频流（或者侧重于时间变化）
        self.encoder = Encoder(
            model_size='base',
            patch_size=[4, 8, 8],
            in_channels=3,
            out_channels=token_size,
            in_grid=in_grid,
            out_tokens=768,
        )
        
        # 第一帧编码器：专门提供高保真的空间参考
        # out_tokens=256 (1x128x128 grid -> patch 1x8x8 -> 16x16 tokens = 256)
        self.encoder1 = Encoder(
            model_size='base',
            patch_size=[1, 8, 8],
            in_channels=3,
            out_channels=token_size,
            in_grid=[1, 128, 128],
            out_tokens=256, 
        )
        
        self.quantize = FSQ(levels=[8, 8, 8,5, 5, 5])
        
        # 这里的 cond_tokens 对应 encoder1 的 out_tokens
        self.decoder = Decoder_unify(
            model_size='base',
            patch_size=[4, 8, 8],
            in_channels=token_size,
            out_channels=3,
            in_tokens=768,
            cond_tokens=256,  # <--- 新增：告诉Decoder有多少个第一帧的参考token
            out_grid=in_grid,
        )
        self.prior_model = None 

    def encode(self, data, **kwargs):
        # data: [B, C, T, H, W]
        x_first = data[:, :, 0:1, :, :]  # 取第一帧 [B, C, 1, H, W]
        # 1. 编码主视频流
        x = self.encoder(data)
        # 2. 编码第一帧
        first_token = self.encoder1(x_first)
        # 3. 量化
        # 这里有两个选择：
        # A. 共享量化器 (如果First Token也需要被压缩成离散码) - 推荐
        # B. 仅量化主视频，第一帧保持连续特征 (作为Condition)
        # 这里演示方案A，共享量化器，方便后续训练Transformer
        x_q, x_dict = self.quantize(x)
        first_q, first_dict = self.quantize(first_token) 
        return x_q, first_q
    
    def decode(self, x, first_token):
        # x: 主视频 latent tokens
        # first_token: 第一帧 latent tokens (作为参考)
        x = self.decoder(x, first_token)
        return x
    
    def decode_indices(self, indices, first_indices):
        # 从索引恢复视频
        x_q = self.quantize.indices_to_codes(indices).to(indices.device, next(self.decoder.parameters()).dtype)
        first_q = self.quantize.indices_to_codes(first_indices).to(first_indices.device, next(self.decoder.parameters()).dtype)
        return self.decoder(x_q, first_q)
    
    def forward(self, x):
        # x: [B, 3, 16, 128, 128]
        x_q, first_q = self.encode(x)
        
        # Decoder 接收两部分输入
        pred_frames = self.decode(x_q, first_q)
        return_dict = {'pred_frames': pred_frames}   
        return return_dict


@register('autoencoder_first_token_f256t512')
class AutoEncoder_first_token(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        in_grid = [16, 128, 128]
        token_size = 6
        
        # 主编码器：处理整个视频流（或者侧重于时间变化）
        self.encoder = Encoder(
            model_size='base',
            patch_size=[4, 8, 8],
            in_channels=3,
            out_channels=token_size,
            in_grid=in_grid,
            out_tokens=512,
        )
        
        # 第一帧编码器：专门提供高保真的空间参考
        # out_tokens=256 (1x128x128 grid -> patch 1x8x8 -> 16x16 tokens = 256)
        self.encoder1 = Encoder(
            model_size='base',
            patch_size=[1, 8, 8],
            in_channels=3,
            out_channels=token_size,
            in_grid=[1, 128, 128],
            out_tokens=256, 
        )
        
        self.quantize = FSQ(levels=[8, 8, 8,5, 5, 5])
        
        # 这里的 cond_tokens 对应 encoder1 的 out_tokens
        self.decoder = Decoder_unify(
            model_size='base',
            patch_size=[4, 8, 8],
            in_channels=token_size,
            out_channels=3,
            in_tokens=512,
            cond_tokens=256,  # <--- 新增：告诉Decoder有多少个第一帧的参考token
            out_grid=in_grid,
        )
        self.prior_model = None 

    def encode(self, data, **kwargs):
        # data: [B, C, T, H, W]
        x_first = data[:, :, 0:1, :, :]  # 取第一帧 [B, C, 1, H, W]
        # 1. 编码主视频流
        x = self.encoder(data)
        # 2. 编码第一帧
        first_token = self.encoder1(x_first)
        # 3. 量化
        # 这里有两个选择：
        # A. 共享量化器 (如果First Token也需要被压缩成离散码) - 推荐
        # B. 仅量化主视频，第一帧保持连续特征 (作为Condition)
        # 这里演示方案A，共享量化器，方便后续训练Transformer
        x_q, x_dict = self.quantize(x)
        first_q, first_dict = self.quantize(first_token) 
        return x_q, first_q
    
    def decode(self, x, first_token):
        # x: 主视频 latent tokens
        # first_token: 第一帧 latent tokens (作为参考)
        x = self.decoder(x, first_token)
        return x
    
    def decode_indices(self, indices, first_indices):
        # 从索引恢复视频
        x_q = self.quantize.indices_to_codes(indices).to(indices.device, next(self.decoder.parameters()).dtype)
        first_q = self.quantize.indices_to_codes(first_indices).to(first_indices.device, next(self.decoder.parameters()).dtype)
        return self.decoder(x_q, first_q)
    
    def forward(self, x):
        # x: [B, 3, 16, 128, 128]
        x_q, first_q = self.encode(x)
        
        # Decoder 接收两部分输入
        pred_frames = self.decode(x_q, first_q)
        return_dict = {'pred_frames': pred_frames}   
        return return_dict
