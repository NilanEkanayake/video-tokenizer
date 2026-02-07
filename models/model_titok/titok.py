"""This file contains the model definition of TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.
"""
import torch
import torch.nn as nn
from models.model_titok.base.blocks import TiTokEncoder, TiTokDecoder, init_weights
from models.model_titok.quantizer.fsq import FSQ
from models import register
@register('titok')
class TiTok(nn.Module):
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

        self.encoder = TiTokEncoder(
            model_size='base',
            patch_size=[4, 8, 8],
            in_channels=3,
            out_channels=token_size,
            max_grid= [16, 128, 128],
            max_tokens=1024,
        )
        self.quantize = FSQ(levels=[8, 8,8, 5, 5, 5])
        self.decoder = TiTokDecoder(
            model_size='base',
            patch_size=[4, 8, 8],
            in_channels=token_size,
            out_channels=3,
            max_grid=in_grid,
            max_tokens=1024,
        )
        self.prior_model = None 
        self.apply(init_weights)

    def encode(self, x, token_counts):
        x = self.encoder(x, token_counts)
        x_q, x_dict = self.quantize(x)
        x_dict['indices'] = torch.split(x_dict['indices'], token_counts, dim=0) # [B*L] -> [B, L]
        return x_q, x_dict
    
    def decode(self, x, token_counts, grids):
        x = self.decoder(x, token_counts, grids)
        return x
    
    def decode_indices(self, indices, grids):
        token_counts = [x.shape[0] for x in indices]
        x_q = self.quantize.indices_to_codes(indices).to(indices.device, next(self.decoder.parameters()).dtype) # expects B*L in
        return self.decode(x_q, token_counts, grids)
    
    def forward(self, x):
        token_counts= [1024 for vid in x] # B,
         # 计算每个视频的 token 数量
        grids = [vid.shape[1:] for vid in x] # c|THW|
        x_q, out_dict = self.encode(x, token_counts)
        x = self.decode(x_q, token_counts, grids)
        x = torch.stack(x, dim=0)
        return_dict = {'pred_frames': x}
        return return_dict