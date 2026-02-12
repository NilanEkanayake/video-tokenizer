import torch
import torch.nn as nn
from models.model_design.base.blocks import Encoder, FirstFrameEncoder ,UnifiedDecoder
from models.model_design.quantizer.fsq import FSQ
from models import register


@register('autoencoder_design')
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
            model_size='small',
            patch_size=(4, 8, 8),
            in_channels=3,
            out_channels=token_size,
            in_grid=in_grid,
            out_tokens=1024,
        )

        # 第一帧编码器：专门的 2D 编码器
        self.first_frame_encoder = FirstFrameEncoder(
            model_size='small',
            patch_size_hw=(8, 8),
            in_channels=3,
            out_channels=token_size,
            in_hw=(128, 128),
            out_tokens=256,
        )

        # 量化器（共享）
        self.quantize = FSQ(levels=[8, 8, 8, 5, 5, 5])

        # 解码器（带 Cross-Attention 条件注入）
        self.decoder = UnifiedDecoder(
            model_size='small',
            patch_size=(4, 8, 8),
            in_channels=token_size,
            out_channels=3,
            in_tokens=1024,
            cond_tokens=256,
            out_grid=in_grid,
        )

        self.prior_model = None

    def encode(self, data, **kwargs):
        """
        data: [B, 3, T, H, W]
        returns: (main_quantized, first_quantized, main_dict, first_dict)
        """
        # 编码完整视频
        main_tokens = self.encoder(data)  # [B, 1024, token_size]

        # 编码第一帧
        first_frame = data[:, :, 0:1, :, :]  # [B, 3, 1, H, W]
        first_tokens = self.first_frame_encoder(first_frame)  # [B, 256, token_size]

        # 量化（共享量化器）
        main_q, main_indices = self.quantize(main_tokens)
        first_q, first_indices = self.quantize(first_tokens)

        return main_q, first_q, main_indices, first_indices

    def decode(self, main_q, first_q):
        """
        main_q: [B, 1024, token_size] - 量化后的主视频 tokens
        first_q: [B, 256, token_size] - 量化后的第一帧 tokens
        returns: [B, 3, T, H, W]
        """
        return self.decoder(main_q, cond=first_q)

    def decode_from_indices(self, main_indices, first_indices):
        """从离散索引重建视频"""
        dtype = next(self.decoder.parameters()).dtype
        device = main_indices.device

        main_q = self.quantize.indices_to_codes(main_indices).to(device=device, dtype=dtype)
        first_q = self.quantize.indices_to_codes(first_indices).to(device=device, dtype=dtype)

        return self.decode(main_q, first_q)

    def forward(self, x):
        """
        训练时的完整前向传播。
        x: [B, 3, T, H, W]
        """
        # Encode
        main_q, first_q, main_indices, first_indices = self.encode(x)

        # Decode
        pred_frames = self.decode(main_q, first_q)

        return {
            'pred_frames': pred_frames,
        }