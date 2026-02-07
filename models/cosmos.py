import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from collections import namedtuple
from models import register
import os
# ==========================================
# 1. 基础工具与组件 (Utils & Basic Blocks)
# ==========================================
from typing import List, Optional, Tuple, NamedTuple
from einops import rearrange, pack, unpack

# ==========================================
# 1. 基础工具与占位符 (请替换为你原有的完整 Encoder/Decoder 实现)
# ==========================================

# 这里的函数和类仅为占位符，确保代码结构完整，请使用你之前的 Encoder/Decoder 代码
def default(val, d): return val if val is not None else d
def round_ste(x): return (x.round() - x).detach() + x
def nonlinearity(x):
    return x * torch.sigmoid(x)

def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)

def is_odd(n):
    return (n % 2) == 1

def time2batch(x):
    b, c, t, h, w = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    return x, b

def batch2time(x, b):
    _, c, h, w = x.shape
    x = x.reshape(b, -1, c, h, w).permute(0, 2, 1, 3, 4)
    return x

def space2batch(x):   #_replication_pad
    b, c, t, h, w = x.shape
    x = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, t)
    return x, b, h

def batch2space(x, b, h):
    bhw, c, t = x.shape
    w = bhw // (b * h)
    x = x.reshape(b, h, w, c, t).permute(0, 3, 4, 1, 2)
    return x

class CausalNormalize(nn.Module):
    def __init__(self, in_channels, num_groups=1):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    def forward(self, x):
        return self.norm(x)

class CausalConv3d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size=3, pad_mode="constant", **kwargs):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        
        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)
        time_stride = kwargs.pop("time_stride", 1)
        time_dilation = kwargs.pop("time_dilation", 1)
        padding = kwargs.pop("padding", 0)
        
        self.pad_mode = pad_mode
        time_pad = time_dilation * (time_kernel_size - 1) + (1 - time_stride)
        self.time_pad = max(0, time_pad)
        self.spatial_pad = (padding, padding, padding, padding)
        
        stride = (time_stride, stride, stride)
        dilation = (time_dilation, dilation, dilation)
        self.conv3d = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def _replication_pad(self, x):
        if self.time_pad > 0:
            x_prev = x[:, :, :1, ...].repeat(1, 1, self.time_pad, 1, 1)
            x = torch.cat([x_prev, x], dim=2)
        if sum(self.spatial_pad) > 0:
            x = F.pad(x, self.spatial_pad, mode=self.pad_mode, value=0.0)
        return x

    def forward(self, x):
        x = self._replication_pad(x)
        return self.conv3d(x)

class CausalResnetBlockFactorized3d(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout=0.0, num_groups=1):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = CausalNormalize(in_channels, num_groups=num_groups)
        self.conv1 = nn.Sequential(
            CausalConv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=1, padding=1),
            CausalConv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=0),
        )
        self.norm2 = CausalNormalize(out_channels, num_groups=num_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Sequential(
            CausalConv3d(out_channels, out_channels, kernel_size=(1, 3, 3), stride=1, padding=1),
            CausalConv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=0),
        )
        self.nin_shortcut = (
            CausalConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels else nn.Identity()
        )
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        x = self.nin_shortcut(x)
        return x + h

class CausalAttnBlock(nn.Module):
    def __init__(self, in_channels, num_groups=1):
        super().__init__()
        self.norm = CausalNormalize(in_channels, num_groups=num_groups)
        self.q = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        q, b = time2batch(q)
        k, _ = time2batch(k)
        v, _ = time2batch(v)
        B, C, H, W = q.shape
        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        k = k.reshape(B, C, H * W)
        w_ = torch.bmm(q, k) * (int(C) ** (-0.5))
        w_ = F.softmax(w_, dim=2)
        v = v.reshape(B, C, H * W)
        h_ = torch.bmm(v, w_.permute(0, 2, 1)).reshape(B, C, H, W)
        h_ = batch2time(h_, b)
        return x + self.proj_out(h_)

class CausalTemporalAttnBlock(nn.Module):
    def __init__(self, in_channels, num_groups=1):
        super().__init__()
        self.norm = CausalNormalize(in_channels, num_groups=num_groups)
        self.q = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        if x.shape[2] <= 1: return x
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        q, b, height = space2batch(q)
        k, _, _ = space2batch(k)
        v, _, _ = space2batch(v)
        bhw, c, t = q.shape
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        w_ = torch.bmm(q, k.permute(0, 2, 1)) * (int(c) ** (-0.5))
        mask = torch.tril(torch.ones(t, t, device=q.device))
        w_ = w_.masked_fill(mask == 0, float("-inf"))
        w_ = F.softmax(w_, dim=2)
        h_ = torch.bmm(w_, v).permute(0, 2, 1).reshape(bhw, c, t)
        h_ = batch2space(h_, b, height)
        return x + self.proj_out(h_)

class CausalHybridDownsample3d(nn.Module):
    def __init__(self, in_channels, spatial_down=True, temporal_down=False):
        super().__init__()
        self.spatial_down = spatial_down
        self.temporal_down = temporal_down
        if spatial_down:
            self.conv_s1 = CausalConv3d(in_channels, in_channels, kernel_size=(1, 3, 3), stride=2, time_stride=1, padding=0)
        if temporal_down:
            self.conv_t1 = CausalConv3d(in_channels, in_channels, kernel_size=(3, 1, 1), stride=1, time_stride=2, padding=0)
        self.conv_mix = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if not self.spatial_down and not self.temporal_down: return x
        out = x
        if self.spatial_down:
            pad = (0, 1, 0, 1, 0, 0)
            out = F.pad(out, pad, mode="constant", value=0)
            out = self.conv_s1(out)
        if self.temporal_down:
            out = self.conv_t1(out)
        return self.conv_mix(out)

# class CausalHybridUpsample3d(nn.Module):
#     def __init__(self, in_channels, spatial_up=True, temporal_up=False):
#         super().__init__()
#         self.spatial_up = spatial_up
#         self.temporal_up = temporal_up
#         self.conv_mix = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
#         if spatial_up:
#             self.conv_s = CausalConv3d(in_channels, in_channels, kernel_size=(1, 3, 3), stride=1, padding=1)
#         if temporal_up:
#             self.conv_t = CausalConv3d(in_channels, in_channels, kernel_size=(3, 1, 1), stride=1, padding=0)

#     def forward(self, x):
#         if not self.spatial_up and not self.temporal_up: return x
        
#         if self.temporal_up:
#             x = x.repeat_interleave(2, dim=2)
#             x = self.conv_t(x)

#         if self.spatial_up:
#             x = F.interpolate(x, scale_factor=(1, 2, 2), mode='nearest')
#             x = self.conv_s(x)
            
#         return self.conv_mix(x)
class CausalHybridUpsample3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        spatial_up: bool = True,
        temporal_up: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv3d(
            in_channels,
            in_channels,
            kernel_size=(3, 1, 1),
            stride=1,
            time_stride=1,
            padding=0,
        )
        self.conv2 = CausalConv3d(
            in_channels,
            in_channels,
            kernel_size=(1, 3, 3),
            stride=1,
            time_stride=1,
            padding=1,
        )
        self.conv3 = CausalConv3d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            time_stride=1,
            padding=0,
        )
        self.spatial_up = spatial_up
        self.temporal_up = temporal_up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.spatial_up and not self.temporal_up:
            return x

        # hybrid upsample temporally.
        if self.temporal_up:
            time_factor = 1.0 + 1.0 * (x.shape[2] > 1)
            if isinstance(time_factor, torch.Tensor):
                time_factor = time_factor.item()
            x = x.repeat_interleave(int(time_factor), dim=2)
            #x = x[..., int(time_factor - 1) :, :, :]
            x = self.conv1(x) + x

        # hybrid upsample spatially.
        if self.spatial_up:
            x = x.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
            x = self.conv2(x) + x

        # final 1x1x1 conv.
        x = self.conv3(x)
        return x
# ==========================================
# 2. 核心组件: 空间交叉注意力 (Injection)
# ==========================================

class SpatialCrossAttnBlock(nn.Module):
    def __init__(self, in_channels, num_groups=1):
        super().__init__()
        self.norm = CausalNormalize(in_channels, num_groups=num_groups)
        self.q = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_motion, x_ref):
        h_mot = self.norm(x_motion)
        h_ref = self.norm(x_ref)
        q = self.q(h_mot) 
        k = self.k(h_ref)
        v = self.v(h_ref)
        b, c, t, h, w = q.shape
        q = q.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c)
        k = k.squeeze(2).permute(0, 2, 3, 1).reshape(b, h * w, c)
        v = v.squeeze(2).permute(0, 2, 3, 1).reshape(b, h * w, c)
        k = k.repeat_interleave(t, dim=0) 
        v = v.repeat_interleave(t, dim=0)
        attn = torch.bmm(q, k.transpose(1, 2)) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)
        out = out.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        return x_motion + self.proj_out(out)

# ==========================================
# 3. 编码器 (Encoder)
# ==========================================_replication_pad

class CosmosDualSharedEncoder(nn.Module):
    def __init__(self, in_channels=3, channels=64, channels_mult=[1, 2, 4, 8, 8], num_res_blocks=2, attn_resolutions=[], dropout=0.0, z_channels=1024, ref_target_stride=16, motion_target_stride=32, motion_temporal_down_count=2):
        super().__init__()
        self.conv_in = nn.Sequential(
            CausalConv3d(in_channels, channels, kernel_size=(1, 3, 3), stride=1, padding=1),
            CausalConv3d(channels, channels, kernel_size=(3, 1, 1), stride=1, padding=0),
        )
        ref_steps = int(math.log2(ref_target_stride))
        mot_steps = int(math.log2(motion_target_stride))
        max_steps = max(ref_steps, mot_steps)
        time_schedule = [False] * max_steps
        for i in range(motion_temporal_down_count):
            if i < max_steps: time_schedule[i] = True
        self.layers = nn.ModuleList()
        curr_res = 1
        curr_ch = channels
        for i in range(max_steps):
            mult = channels_mult[i] if i < len(channels_mult) else channels_mult[-1]
            out_ch = channels * mult
            blocks, attns = nn.ModuleList(), nn.ModuleList()
            tmp_ch = curr_ch
            for _ in range(num_res_blocks):
                blocks.append(CausalResnetBlockFactorized3d(in_channels=tmp_ch, out_channels=out_ch, dropout=dropout))
                tmp_ch = out_ch
                attns.append(nn.Sequential(CausalAttnBlock(out_ch), CausalTemporalAttnBlock(out_ch)) if curr_res in attn_resolutions else nn.Identity())
            ref_down = CausalHybridDownsample3d(out_ch, spatial_down=True, temporal_down=False) if i < ref_steps else None
            mot_down = CausalHybridDownsample3d(out_ch, spatial_down=True, temporal_down=time_schedule[i]) if i < mot_steps else None
            layer = nn.Module()
            layer.blocks, layer.attns, layer.ref_down, layer.mot_down = blocks, attns, ref_down, mot_down
            self.layers.append(layer)
            curr_ch = out_ch
            curr_res *= 2
        self.ref_out_ch = channels * channels_mult[ref_steps-1]
        self.mot_out_ch = channels * channels_mult[mot_steps-1] if mot_steps <= len(channels_mult) else channels * channels_mult[-1]
        self.ref_head = self._make_head(self.ref_out_ch, z_channels, dropout)
        self.mot_head = self._make_head(self.mot_out_ch, z_channels, dropout)
    def _make_head(self, ch, z_ch, drop):
        return nn.ModuleDict({
            'mid_block1': CausalResnetBlockFactorized3d(in_channels=ch, dropout=drop),
            'mid_attn': nn.Sequential(CausalAttnBlock(ch), CausalTemporalAttnBlock(ch)),
            'mid_block2': CausalResnetBlockFactorized3d(in_channels=ch, dropout=drop),
            'norm': CausalNormalize(ch),
            'conv_out': nn.Sequential(CausalConv3d(ch, z_ch, kernel_size=(1, 3, 3), stride=1, padding=1), CausalConv3d(z_ch, z_ch, kernel_size=(3, 1, 1), stride=1, padding=0))
        })
    def run_head(self, x, head_dict):
        h = head_dict['mid_block1'](x)
        h = head_dict['mid_attn'](h)
        h = head_dict['mid_block2'](h)
        h = nonlinearity(head_dict['norm'](h))
        return head_dict['conv_out'](h)
    def forward(self, x):
        x_ref, x_mot = x[:, :, 0:1, :, :], x[:, :, 1:, :, :]
        h_ref, h_mot = self.conv_in(x_ref), (self.conv_in(x_mot) if x_mot.shape[2] > 0 else None)
        #print(f"h_ref shape: {h_ref.shape}")
        #print(f"h_mot shape: {h_mot.shape}" if h_mot is not None else "h_mot is None")
        for layer in self.layers:
            if h_ref is not None and layer.ref_down is not None:
                for b, a in zip(layer.blocks, layer.attns): h_ref = a(b(h_ref))
                h_ref = layer.ref_down(h_ref)
            if h_mot is not None and layer.mot_down is not None:
                for b, a in zip(layer.blocks, layer.attns): h_mot = a(b(h_mot))
                h_mot = layer.mot_down(h_mot)
            #print(f"After layer: h_ref shape: {h_ref.shape}" if h_ref is not None else "h_ref is None")
            #print(f"After layer: h_mot shape: {h_mot.shape}" if h_mot is not None else "h_mot is None")
        z_ref = self.run_head(h_ref, self.ref_head)
        z_mot = self.run_head(h_mot, self.mot_head) if h_mot is not None else None
        #print(f"z_ref shape: {z_ref.shape}")
        #print(f"z_mot shape: {z_mot.shape}" if z_mot is not None else "z_mot is None")
        return z_ref, z_mot



class CosmosDualSharedDecoder(nn.Module):
    def __init__(self, out_channels=3, channels=64, channels_mult=[1, 2, 4, 8, 8], num_res_blocks=2, attn_resolutions=[], dropout=0.0, resolution=256, z_channels=1024, spatial_compression=16, motion_spatial_compression=32, motion_temporal_compression=4, cross_attn_resolutions=[16, 8]):
        super().__init__()
        self.cross_attn_resolutions = cross_attn_resolutions
        ref_level_idx = int(math.log2(spatial_compression)) - 1
        mot_level_idx = int(math.log2(motion_spatial_compression)) - 1
        block_in_ref = channels * channels_mult[ref_level_idx]
        block_in_mot = channels * channels_mult[mot_level_idx]
        
        # Motion Adapter
        self.mot_conv_in = nn.Sequential(CausalConv3d(z_channels, block_in_mot, kernel_size=1), CausalConv3d(block_in_mot, block_in_mot, kernel_size=1))
        self.motion_adapter = nn.ModuleList()
        adapter_levels = range(mot_level_idx, ref_level_idx, -1)
        curr_ch = block_in_mot
        for i_level in adapter_levels:
            target_ch = channels * channels_mult[i_level - 1]
            blocks = nn.ModuleList([CausalResnetBlockFactorized3d(in_channels=curr_ch, out_channels=curr_ch, dropout=dropout) for _ in range(num_res_blocks)])
            upsample = CausalHybridUpsample3d(curr_ch, spatial_up=True, temporal_up=False)
            self.motion_adapter.append(nn.ModuleDict({'blocks': blocks, 'up': upsample}))
            if curr_ch != target_ch:
                self.motion_adapter.append(CausalConv3d(curr_ch, target_ch, kernel_size=1))
                curr_ch = target_ch
        # self.mot_time_ups = nn.ModuleList()
        # for _ in range(int(math.log2(motion_temporal_compression)) - len(adapter_levels)):
        #     self.mot_time_ups.append(CausalHybridUpsample3d(curr_ch, spatial_up=False, temporal_up=False))

        # Ref Adapter
        self.ref_conv_in = CausalConv3d(z_channels, block_in_ref, kernel_size=3, padding=1)
        self.ref_mid = nn.Sequential(CausalResnetBlockFactorized3d(in_channels=block_in_ref, dropout=dropout), CausalAttnBlock(block_in_ref), CausalResnetBlockFactorized3d(in_channels=block_in_ref, dropout=dropout))

        # Backbone
        self.cross_injections = nn.ModuleDict()
        self.up_layers = nn.ModuleList()
        block_in = block_in_ref
        
        # 修复点：确保循环包含 Level 0，并且 Level 0 也执行 Upsample
        for i_level in reversed(range(ref_level_idx + 1)):
            current_scale = 2 ** (i_level + 1)
            #print(f"Setting up Decoder Level {i_level} at scale {current_scale}")
            if current_scale in self.cross_attn_resolutions:
                self.cross_injections[f"scale_{current_scale}"] = SpatialCrossAttnBlock(block_in)
            
            block_out = channels * channels_mult[i_level - 1] if i_level > 0 else channels
            blocks, attns = nn.ModuleList(), nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(CausalResnetBlockFactorized3d(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                attns.append(nn.Sequential(CausalAttnBlock(block_in), CausalTemporalAttnBlock(block_in)) if current_scale in attn_resolutions else nn.Identity())
            layer = nn.Module()
            # 关键修改：移除 if i_level != 0 判断，所有层都进行上采样
            # 因为 Encoder 每一层都做了下采样， Decoder 每一层都要上采样才能恢复到 f1
            temporal_up=True if (current_scale ==8 or current_scale ==4) else False
            #upsample = CausalHybridUpsample3d(block_in, spatial_up=True, temporal_up=temporal_up)
            layer.upsample_mot = CausalHybridUpsample3d(block_in, spatial_up=True, temporal_up=temporal_up)

            # 2. 给 Reference 用的 (强制关闭时间上采样，temporal_up=False)
            # 这样就不需要依赖 x.shape[2] > 1 的判断了，结构是静态的
            layer.upsample_ref = CausalHybridUpsample3d(block_in, spatial_up=True, temporal_up=False)
            #layer = nn.Module()
            layer.blocks, layer.attns= blocks, attns
            self.up_layers.append(layer)
        #print(self.cross_injections.keys())
        self.norm_out = CausalNormalize(block_in)
        self.conv_out = CausalConv3d(block_in, out_channels, kernel_size=3, padding=1)

    def forward(self, z_ref, z_mot):
        h_mot = self.mot_conv_in(z_mot)
        for layer in self.motion_adapter:
            #print(f"h_mot shape  {h_mot.shape}")
            if isinstance(layer, CausalConv3d): h_mot = layer(h_mot)
            else:
                for b in layer['blocks']: h_mot = b(h_mot)
                h_mot = layer['up'](h_mot)
        #for layer in self.mot_time_ups: h_mot = layer(h_mot)
        #print(f"h_mot shape before fusion: {h_mot.shape}")
        h_ref = self.ref_mid(self.ref_conv_in(z_ref))
        #print(f"h_ref shape before fusion: {h_ref.shape}")

        if "scale_8" in self.cross_injections:
            h_mot = self.cross_injections["scale_8"](h_mot, h_ref)
            #print('yes')
        #h = torch.cat([h_ref, h_mot], dim=2)
        
        current_scale = 8
        for i, layer in enumerate(self.up_layers):
            for j in range(len(layer.blocks)):
                h_ref = layer.attns[j](layer.blocks[j](h_ref))#16
                h_mot = layer.attns[j](layer.blocks[j](h_mot))

            # if i < len(self.up_layers) - 1:
            #     for j in range(len(layer.blocks)):
            #         h_ref = layer.attns[j](layer.blocks[j](h_ref))
            h_ref = layer.upsample_ref(h_ref)
            h_mot = layer.upsample_mot(h_mot)
            #if i < len(self.up_layers) - 1: h_ref = layer.upsample(h_ref)
            #print(f"After upsample at scale {current_scale}, h_mot shape: {h_mot.shape}, h_ref shape: {h_ref.shape}")
            current_scale //= 2
            if f"scale_{current_scale}" in self.cross_injections:
                #print(f"Injecting at scale {current_scale}")
                h_mot = self.cross_injections[f"scale_{current_scale}"](h_mot, h_ref)
        h_final = torch.cat([h_ref, h_mot], dim=2)
        return self.conv_out(nonlinearity(self.norm_out(h_final)))
    
class FSQuantizer(nn.Module):
    """
    Finite Scalar Quantization (FSQ).  After layer
    支持自定义 levels，例如 [8, 8, 8, 5, 5, 5]。
    """
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        dtype=torch.float32,
        **ignore_kwargs,
    ):
        super().__init__()
        self.dtype = dtype
        
        # 注册 levels 和 basis 到 buffer
        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)

        # 计算 basis
        _basis = torch.cumprod(
            torch.tensor([1] + levels[:-1], dtype=torch.int32), dim=0
        )
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale
        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim
        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim
        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, codebook_dim * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim)
            if has_projections
            else nn.Identity()
        )
        self.has_projections = has_projections
        self.codebook_size = self._levels.prod().item()

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        levels = self._levels.to(z.device)
        half_l = (levels - 1) * (1 + eps) / 2
        offset = torch.where(levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z.float() + shift).tanh() * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        quantized = round_ste(self.bound(z))
        half_width = (self._levels.to(z.device) / 2).float()
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        half_width = (self._levels.to(zhat_normalized.device) / 2).float()
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        half_width = (self._levels.to(zhat.device) / 2).float()
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat).float()
        basis = self._basis.to(zhat.device)
        indices = (zhat.round() * basis).sum(dim=-1).to(torch.int32)
        return indices

    def indices_to_codes(self, indices: torch.Tensor, project_out=True) -> torch.Tensor:
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        indices = indices.unsqueeze(-1) 
        levels = self._levels.to(indices.device).float()
        basis = self._basis.to(indices.device).float()
        codes_non_centered = (indices / basis) % levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim and self.num_codebooks > 1:
            codes = rearrange(codes, "... c d -> ... (c d)")
        else:
             codes = codes.squeeze(-2)

        if project_out:
            codes = self.project_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes.to(self.dtype)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[Optional[torch.Tensor], torch.Tensor]]:
        is_img_or_video = z.ndim >= 4
        original_shape = z.shape
        
        # 1. 维度调整
        if is_img_or_video:
            z_permuted = rearrange(z, "b c ... -> b ... c")
            z_packed, ps = pack([z_permuted], "b * c")
        else:
            z_permuted = rearrange(z, "b c d -> b d")
            z_packed, ps = pack([z_permuted], "b * c")
            
        assert z_packed.shape[-1] == self.dim

        # 2. 投影到 FSQ 维度
        z_projected = self.project_in(z_packed)

        # 3. Reshape for FSQ
        z_reshaped = rearrange(z_projected, "b n (c d) -> b n c d", c=self.num_codebooks)
        
        # 4. 量化
        codes = self.quantize(z_reshaped)
        indices = self.codes_to_indices(codes)

        # 5. 投影回原始维度
        codes_reshaped = rearrange(codes, "b n c d -> b n (c d)")
        out = self.project_out(codes_reshaped)

        # 6. 恢复原始形状
        if is_img_or_video:
            # === 修改点 1：使用 [var] = ... 进行解包 ===
            [out_reconstructed] = unpack(out, ps, "b * c")
            out_reconstructed = rearrange(out_reconstructed, "b ... c -> b c ...")
            
            # === 修改点 2：使用 [var] = ... 进行解包 ===
            [indices_reshaped] = unpack(indices, ps, "b * c")
            if self.num_codebooks == 1:
                indices_reshaped = indices_reshaped.squeeze(-1)
            
            # 恢复为 (B, T, H, W) 或 (B, H, W)
            spatial_dims = original_shape[2:] 
            indices_final = indices_reshaped.view(original_shape[0], *spatial_dims)
        else:
            # === 修改点 3：非视频情况也同样修改 ===
            [out_reconstructed] = unpack(out, ps, "b * c")
            # 这里的 indices 是 unpack 后列表的第一个元素
            indices_final = unpack(indices, ps, "b * c")[0].squeeze(-1)

        dummy_loss = torch.zeros((1,), device=z.device)

        return out_reconstructed.to(self.dtype), dummy_loss, (None, indices_final)

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        return self.indices_to_codes(indices, project_out=True)
# ==========================================
# 3. VideoTokenizer (应用修改: 指定 8,8,8,5,5,5)
# ==========================================
@register('cosmos_fsq')
class FSQ_VideoTokenizer(nn.Module):
    """
    Unified Tokenizer for Video using Cosmos architecture with FSQuantizer.
    配置为 FSQ levels: [8, 8, 8, 5, 5, 5]
    """
    def __init__(
        self,
        # 架构参数
        in_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: List[int] = [1, 2, 4, 4],
        
        # Latent 维度
        latent_dim: int = 256,   # Encoder输出和Decoder输入的通道数
        
        # FSQ 配置 (无需传入 codebook_size，由 fsq_levels 决定)
        # 这里的默认值被修改为您要求的版本
        fsq_levels: List[int] = [8, 8, 8, 5, 5, 5], 
        
        # 压缩目标
        ref_stride: int = 8,
        mot_stride: int = 16,
        mot_time_down: int = 2,
        
        dropout: float = 0.0,
        fsq_dtype = torch.float32
    ):
        super().__init__()
        
        # 计算总 Codebook 大小: 8*8*8*5*5*5 = 64000
        codebook_size = math.prod(fsq_levels)
        #print(f"Initializing VideoTokenizer with FSQ levels: {fsq_levels}")
        #print(f"Effective Codebook Size: {codebook_size}")
        #print(f"Latent Dim: {latent_dim} -> Quantization Dim: {len(fsq_levels)}")
        self.prior_model = None 
        # 1. 实例化 Encoder (保持不变)
        # 注意：Encoder 的输出头通道数为 latent_dim (1024)
        self.encoder = CosmosDualSharedEncoder(
            in_channels=in_channels,
            channels=base_channels,
            channels_mult=channel_multipliers,
            z_channels=latent_dim, 
            ref_target_stride=ref_stride,
            motion_target_stride=mot_stride,
            motion_temporal_down_count=mot_time_down,
            dropout=dropout
        )
        
        # 2. 实例化 FSQuantizer (使用 8,8,8,5,5,5)
        # FSQ 会自动创建 Linear(1024, 6) 和 Linear(6, 1024)
        self.quantizer = FSQuantizer(
            levels=fsq_levels,
            dim=latent_dim,             # 输入维度 1024
            num_codebooks=1,
            dtype=fsq_dtype
        )
        
        # 3. 实例化 Decoder (保持不变)
        # Decoder 接收通道数为 latent_dim (1024)
        self.decoder = CosmosDualSharedDecoder(
            out_channels=in_channels,
            channels=base_channels,
            channels_mult=channel_multipliers,
            z_channels=latent_dim, 
            spatial_compression=ref_stride,
            motion_spatial_compression=mot_stride,
            motion_temporal_compression=(2**mot_time_down),
            cross_attn_resolutions=[2, 4, 8],
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[Optional[Tuple], Optional[Tuple]]]:
        """Training Forward Pass"""
        # 1. Encode
        z_ref, z_mot = self.encoder(x)
        
        # 2. Quantize (FSQ)
        # z_ref_q: Quantized latent (近似连续值, B, 1024, T, H, W)
        z_ref_q, loss_ref, info_ref = self.quantizer(z_ref)
        z_mot_q, loss_mot, info_mot = self.quantizer(z_mot)
        
        # FSQ 通常不需要显式的 Commitment Loss (loss_ref/loss_mot 通常为 0)
        # 但为了接口兼容保留相加
        vq_loss = loss_ref + loss_mot
        
        # 3. Decode
        recon = self.decoder(z_ref_q, z_mot_q)
        return {
            'pred_frames': recon,
            'loss_vq': vq_loss,
            'info_ref': info_ref,
            'info_mot': info_mot
        }
        #return recon, vq_loss, (info_ref, info_mot)

    def encode_indices(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Inference: Get discrete indices (INT32)"""
        z_ref, z_mot = self.encoder(x)
        
        _, _, (_, ind_ref) = self.quantizer(z_ref)
        ind_mot = None
        if z_mot is not None:
            _, _, (_, ind_mot) = self.quantizer(z_mot)
            
        return ind_ref, ind_mot

    def decode_indices(self, ind_ref: torch.Tensor, ind_mot: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Inference: Decode indices to Video"""
        z_ref_q = self.quantizer.get_codebook_entry(ind_ref)
        z_mot_q = None
        if ind_mot is not None:
            z_mot_q = self.quantizer.get_codebook_entry(ind_mot)
            
        return self.decoder(z_ref_q, z_mot_q)

# ==========================================
# 3. 测试脚本 (验证 1024 维度)
# ==========================================

LossBreakdown = namedtuple('LossBreakdown', ['perceptual_loss', 'nll_loss', 'commitment_loss', 'reconstruction_loss'])

class SimVQ(nn.Module):

    def __init__(self, n_e: int, e_dim: int, beta: float = 0.25, legacy: bool = True):

        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        # 1. 定义固定的 Codebook (Anchor Points)
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # 初始化为单位球表面或者是标准正态分布
        nn.init.normal_(self.embedding.weight, mean=0, std=self.e_dim**-0.5)
        
        # 关键：锁死 Codebook，不进行梯度更新
        self.embedding.weight.requires_grad = False
        
        # 2. 定义可学习的投影层 (The only learnable part in VQ)
        # 它负责调整 Codebook 的位置去适应 Encoder 的输出
        self.embedding_proj = nn.Linear(self.e_dim, self.e_dim)

    def forward(self, z):
        # 1. 维度适配与扁平化
        is_video = z.ndim == 5
        input_shape = z.shape
        
        if is_video:
            # (B, C, T, H, W) -> (B, T, H, W, C)
            z_permuted = rearrange(z, 'b c t h w -> b t h w c').contiguous()
        else:
            # (B, C, H, W) -> (B, H, W, C)
            z_permuted = rearrange(z, 'b c h w -> b h w c').contiguous()
            
        z_flattened = z_permuted.view(-1, self.e_dim)

        # 2. 投影 Codebook
        # SimVQ 核心: 使用 Linear 层变换固定的 Embedding
        # 使得 Codebook 能够动态适应 z 的分布，而不是让 z 强行适应固定的点
        quant_codebook = self.embedding_proj(self.embedding.weight)

        # 3. 计算距离 (Squared Euclidean Distance)
        # (z - e)^2 = z^2 + e^2 - 2ze
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(quant_codebook**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(quant_codebook, 'n d -> d n'))

        # 4. 寻找最近邻 (Nearest Neighbor)
        min_encoding_indices = torch.argmin(d, dim=1)
        
        # 5. 取出量化向量 (Quantize)
        z_q = F.embedding(min_encoding_indices, quant_codebook).view(z_permuted.shape)
        
        # 6. 计算 Loss (Commitment Loss)
        # 让 Encoder 的输出 z 不要离量化向量 z_q 太远
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z_permuted)**2) + \
                   torch.mean((z_q - z_permuted.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z_permuted)**2) + self.beta * \
                   torch.mean((z_q - z_permuted.detach()) ** 2)

        # 7. Straight-Through Estimator (STE)
        # 前向传播用 z_q，反向传播梯度传给 z
        z_q = z_permuted + (z_q - z_permuted).detach()

        # 8. 恢复形状 (Reshape back to B, C, ...)
        if is_video:
            z_q = rearrange(z_q, 'b t h w c -> b c t h w').contiguous()
            # Indices shape: (B, T, H, W)
            min_encoding_indices = min_encoding_indices.view(input_shape[0], input_shape[2], input_shape[3], input_shape[4])
        else:
            z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
            # Indices shape: (B, H, W)
            min_encoding_indices = min_encoding_indices.view(input_shape[0], input_shape[2], input_shape[3])

        # 9. 计算 Perplexity (可选指标，用于监控 Codebook 利用率)
        # e_mean = torch.mean(min_encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        perplexity = None # 简化计算

        # 返回格式适配常见的 Training Loop
        return z_q, loss, (perplexity, min_encoding_indices)

    def get_codebook_entry(self, indices):
        """
        推理阶段使用：根据索引获取量化向量
        Args:
            indices: (B, T, H, W) or (B, H, W)
        """
        # 确保使用投影后的 Codebook
        quant_codebook = self.embedding_proj(self.embedding.weight)
        
        # 获取向量
        z_q = F.embedding(indices, quant_codebook)
        
        # 调整为 Channel-first 格式以便 Decoder 使用
        # Current: (B, T, H, W, C) -> Target: (B, C, T, H, W)
        if z_q.ndim == 5:
            z_q = rearrange(z_q, 'b t h w c -> b c t h w').contiguous()
        elif z_q.ndim == 4:
            z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
            
        return z_q
    

@register('cosmos')
class VideoTokenizer(nn.Module):
    """
    统一管理 Encoder, Decoder 和 VQ，确保维度一致性。
    """
    def __init__(
        self,
        # 架构参数
        
        in_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: List[int] = [1, 2, 4, 4], # f2, f4, f8, f16, f32
        
        # Latent 维度 (关键修改点)
        latent_dim: int = 256,   # z_channels / e_dim
        codebook_size: int = 16384, # n_e
        
        # 压缩目标
        ref_stride: int = 8,
        mot_stride: int = 16,
        mot_time_down: int = 2, # time / 4
        
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # 1. 实例化 Encoder
        # 注意：Encoder 的输出头会将通道数投影到 latent_dim (1024)
        self.prior_model = None 
        self.encoder = CosmosDualSharedEncoder(
            in_channels=in_channels,
            channels=base_channels,
            channels_mult=channel_multipliers,
            z_channels=latent_dim, # <--- 传入 1024
            ref_target_stride=ref_stride,
            motion_target_stride=mot_stride,
            motion_temporal_down_count=mot_time_down,
            dropout=dropout
        )
        
        # 2. 实例化 SimVQ
        # 注意：e_dim 必须等于 latent_dim (1024)
        self.quantizer = SimVQ(
            n_e=codebook_size,
            e_dim=latent_dim,      # <--- 传入 1024
            beta=0.25
        )
        
        # 3. 实例化 Decoder
        # 注意：Decoder 的输入头会接受 latent_dim (1024) 并投影回内部通道
        self.decoder = CosmosDualSharedDecoder(
            out_channels=in_channels,
            channels=base_channels,
            channels_mult=channel_multipliers,
            z_channels=latent_dim, # <--- 传入 1024
            spatial_compression=ref_stride,
            motion_spatial_compression=mot_stride,
            motion_temporal_compression=(2**mot_time_down),
            cross_attn_resolutions=[8, 4,2],
            dropout=dropout
        )
    
    def forward(self, x):
        """
        End-to-end forward pass (Training)
        """
        # 1. Encode
        z_ref, z_mot = self.encoder(x)
        #print(f"Encoder Output Shapes: z_ref {z_ref.shape}, z_mot {z_mot.shape}")
        # 2. Quantize
        z_ref_q, loss_ref, info_ref = self.quantizer(z_ref)
        z_mot_q, loss_mot, info_mot = self.quantizer(z_mot)
        
        vq_loss = loss_ref + loss_mot
        
        # 3. Decode
        recon = self.decoder(z_ref_q, z_mot_q)
        #print(f"Decoder Output Shape: recon {recon.shape}")
        return {
            'pred_frames': recon,
            'loss_vq': vq_loss,
            'info_ref': info_ref,
            'info_mot': info_mot
        }
        # return recon, vq_loss, (info_ref, info_mot)

    def encode_indices(self, x):
        """Inference: Get indices"""
        z_ref, z_mot = self.encoder(x)
        _, _, (_, ind_ref) = self.quantizer(z_ref)
        _, _, (_, ind_mot) = self.quantizer(z_mot)
        return ind_ref, ind_mot

    def decode_indices(self, ind_ref, ind_mot):
        """Inference: Indices to Video"""
        z_ref_q = self.quantizer.get_codebook_entry(ind_ref)
        z_mot_q = self.quantizer.get_codebook_entry(ind_mot)
        return self.decoder(z_ref_q, z_mot_q)


if __name__ == "__main__":

    
    print("=== Testing High-Dimensional (1024) SimVQ Video Tokenizer ===")
    
    # 配置参数
    LATENT_DIM = 256 # 你想要的维度
    BASE_CH = 64       # 基础通道 (稍微调小以防笔记本显存爆炸)
    
    # 实例化 Tokenizer
    model = VideoTokenizer(
        in_channels=3,
        base_channels=64,
        channel_multipliers=[1, 2, 4, 4], # f2, f4, f8, f16, f32
        latent_dim=256, # <--- 关键：设置为 1024
        #codebook_size=16384,
        ref_stride=8,
        mot_stride=16,
        mot_time_down=2
    )
    
    # 打印参数量，看看 1024 维度增加了多少参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params} M")
    
    # 构造输入: 17 帧 (1 Ref + 16 Motion)
    # 使用较小的 spatial size 进行测试
    x = torch.randn(1, 3, 17, 128, 128)
    print(f"Input shape: {x.shape}")
    
    # Forward Pass
    try:
        result= model(x)
        
        print("\n--- Forward Successful ---")
        print(f"Reconstruction shape: {result['pred_frames'].shape}")
        print(f"VQ Loss: {result['loss_vq'].item()}")
        
        # 检查 Latent 形状
        # SimVQ 内部没有打印，我们可以手动跑一下 Encoder 看看出来的维度
        z_ref, z_mot = model.encoder(x)
        print(f"\nInternal Latent Check:")
        print(f"Ref Latent (Unquantized): {z_ref.shape}") # Expect (1, 1024, 1, H/16, W/16)
        print(f"Mot Latent (Unquantized): {z_mot.shape}") # Expect (1, 1024, 4, H/32, W/32)
        
        if z_ref.shape[1] == 256 and z_mot.shape[1] == 256:
            print("✅ Latent Dimension is confirmed to be 1024.")
        else:
            print(f"❌ Error: Latent Dimension is {z_ref.shape[1]}")

        assert result['pred_frames'].shape == x.shape, "Shape mismatch detected!"
        print("✅ End-to-End Shape Test Passed.")

    except RuntimeError as e:
        print(f"\n❌ Runtime Error: {e}")
        if "out of memory" in str(e):
            print("提示: 1024 维度的 3D Tensor 显存占用极大，尝试减小 batch_size 或 input resolution。")
        else:
            print("提示: 请检查 Encoder/Decoder 里的 CausalConv3d 定义是否正确使用了 z_channels 参数。")