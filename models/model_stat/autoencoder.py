import torch
import torch.nn as nn
from models.model_stat.base.blocks import Encoder, Decoder ,Decoder_unify
from models.model_stat.quantizer.fsq import FSQ
from models import register


@register('autoencoder_stat')
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
            model_size='base_thin',
            patch_size=[4, 8, 8],
            in_channels=3,
            out_channels=token_size,
            in_grid=in_grid,
            out_tokens=1024,
        )
        self.quantize = FSQ(levels=[8, 8,8, 5, 5, 5])
        self.decoder = Decoder(
            model_size='base_thin',
            patch_size=[4, 8, 8],
            in_channels=token_size,
            out_channels=3,
            in_tokens=1024,
            out_grid=in_grid,
        )
        self.prior_model = None 
    def get_mask_with_ste(self, probs):
        """
        生成掩码并应用 Straight-Through Estimator (STE)。
        前向传播：使用采样的 mask (0 或 1)。
        反向传播：梯度直接传给 probs。
        """
        if self.training:
            # # 伯努利采样
            # # 生成 [0, 1] 均匀分布随机数
            # noise = torch.rand_like(probs)
            # # 比较生成 mask: noise < probs -> 1 (Keep), else 0 (Drop)
            # mask = (noise < probs).float()
            # # STE 技巧: (mask - probs).detach() + probs
            # # 前向值: mask
            # # 梯度: d(probs)
            # mask_ste = (mask - probs).detach() + probs
            mask = torch.bernoulli(probs)
            mask = (mask - probs).detach() + probs
        else:
            mask = (probs > 0.5).to(probs)
        return mask

    def get_stage(self, current_epoch):
            if current_epoch < 0:
                return "vanilla"  # 阶段 1: 全量训练
            elif current_epoch < 0:
                return "random_drop" # 阶段 2: 随机截断
            else:
                return "adaptive" # 阶段 3: 概率预测 (STAT)

    def encode(self, data, current_epoch=0, **kwargs):
        # x: [B, N, D_token], probs: [B, N, 1]
        x, probs = self.encoder(data)
        B, N, _ = x.shape
        stage = self.get_stage(current_epoch)
        #mask = torch.ones_like(probs).to(x.device) # 默认全 1 (全保留)
        mask = torch.ones_like(probs).to(x) # 默认全 1 (全保留)
        if self.training:
            if stage == "vanilla":
                pass 
                
            elif stage == "random_drop":
                min_keep = 800
                max_keep = 1024
                K = torch.randint(min_keep, max_keep + 1, (B,), device=x.device)
                seq_idx = torch.arange(N, device=x.device).unsqueeze(0) # [1, N]
                #mask = (seq_idx < K.unsqueeze(1)).float().unsqueeze(-1) # [B, N, 1]
                mask = (seq_idx < K.unsqueeze(1)).float()
                
            elif stage == "adaptive":
                mask = self.get_mask_with_ste(probs)
        else:
            if stage == "adaptive":
                 #mask = (probs >= 0.5).float()
                 mask = (probs > 0.5).to(x)
            else:
                 pass # 全量重建
        # 应用 Mask 到量化前的特征
        #x_masked = x * mask
        x_masked = x * mask.to(x).unsqueeze(-1)
        # 量化
        x_q, x_dict = self.quantize(x_masked)
        
        # 存入字典供 Loss 使用
        x_dict['mask'] = mask
        x_dict['probs'] = probs
        x_dict['stage'] = stage # 记录当前阶段
        
        return x_q, x_dict


    def decode(self, x):
        x = self.decoder(x)
        return x
    def forward(self, x, current_epoch=0): # [修改] 增加 current_epoch 参数
        x_q, out_dict = self.encode(x, current_epoch=current_epoch)
        x_recon = self.decode(x_q)
        
        return_dict = {
            'pred_frames': x_recon,
            'probs': out_dict['probs'],
            'mask': out_dict['mask'],
            'stage': out_dict['stage']
        }
        return return_dict


    


