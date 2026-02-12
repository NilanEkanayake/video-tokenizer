import os
import time
from copy import deepcopy

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pytorch_msssim import ssim
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import ConcatDataset, Subset

import models
import utils
from trainers import register

from .base_trainer import BaseTrainer

import random
optimizer_dict = {
    'sgd': SGD,
    'adam': Adam,
    'adamw': AdamW
}


class STATLoss(nn.Module):
    def __init__(self, 
                 target_sparsity=0.7, # 目标平均保留比例
                 lambda_content=0.1,  # 内容自适应 Loss 权重
                 lambda_decrease=0.01, # 单调递减 Loss 权重
                 lambda_sparse=0.01    # 稀疏性 Loss 权重
                 ):
        super().__init__()
        self.target_sparsity = random.uniform(0.85, 0.99) 
        self.lambda_content = lambda_content
        self.lambda_decrease = lambda_decrease
        self.lambda_sparse = lambda_sparse

    # def compute_correlation(self, x, y):
    #     """计算两个向量 x 和 y 的皮尔逊相关系数"""
    #     vx = x - torch.mean(x)
    #     vy = y - torch.mean(y)
    #     cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    #     return cost

    def forward(self, probs, lpips_scores):
        self.target_sparsity = random.uniform(0.85, 0.99) 
        # """
        # probs: [B, N, 1] - 模型预测的保留概率
        # lpips_scores: [B] - 当前 Batch 中每张图/视频的 LPIPS 重建误差 (Ground Truth vs Reconstruction)
        # """
        # #print(f"STATLoss: probs shape={probs.shape}, lpips_scores shape={lpips_scores.shape}")
        # #print(lpips_scores)
        # B, N, _ = probs.shape
        # probs = probs.squeeze(-1) # [B, N]
        # #print(f"Probs: {probs}")
        
        # # 1. 计算期望 Token 数量 (Expected Token Count)
        # # T_expected = sum(probs) per sample
        # expected_token_counts = probs.sum(dim=1) # [B]
        # print(f"Expected token counts: {expected_token_counts}")
        
        # # --- Loss 1: Content-Adaptive Prior (内容自适应先验) ---
        # # 目标：让 Expected Token Count 与 LPIPS Score 正相关
        # # 如果 LPIPS 高 (重建难)，Token 数应该多。
        # # 我们最大化相关性 -> 最小化 (1 - Correlation)^2
        
        # # 归一化 LPIPS scores (可选，有助于稳定训练)
        # lpips_norm = (lpips_scores - lpips_scores.mean()) / (lpips_scores.std() + 1e-8)
        
        # correlation = self.compute_correlation(expected_token_counts, lpips_norm.detach())
        # loss_content = (1 - correlation) ** 2
        
        # # --- Loss 2: Decreasing Importance Prior (单调递减先验) ---
        # # 目标：probs[i] >= probs[i+1]
        # # 惩罚所有 probs[i+1] > probs[i] 的情况
        # # calculate diff: p[i+1] - p[i]
        # diff = probs[:, 1:] - probs[:, :-1] # [B, N-1]
        # # 我们只惩罚正值 (即后面比前面大的情况)
        # loss_decrease = torch.sum(torch.relu(diff)) / B
        
        # # --- Loss 3: Sparsity Prior (稀疏性/压缩率先验) ---
        # # 目标：整个 Batch 的平均保留率接近 target_sparsity
        # avg_prob = probs.mean()
        # print(f"Average prob: {avg_prob.item():.4f}, Target sparsity: {self.target_sparsity}")
        # # 使用 KL 散度或者简单的 MSE
        # # Paper 中使用的是 Binary KL divergence, 这里用简单的 MSE 演示，效果通常类似
        # loss_sparse = (avg_prob - self.target_sparsity) ** 2
        B = probs.shape[0]
        probs = probs.view(B, -1)
        with torch.autocast(enabled=False, device_type=probs.device.type): # loss calcs are untested
            s_probs = probs.mean(-1) # [B]
            corr_matrix = torch.corrcoef(torch.stack((lpips_scores.clone().detach().float(), s_probs.float()))) # detach needed?
            content_loss = 1.0 - (corr_matrix[0, 1] ** 2).to(s_probs).mean()
            if torch.isnan(content_loss):
                 content_loss = torch.tensor(0.0, device=probs.device)
        #sparsity_loss = F.binary_cross_entropy(s_probs, torch.full_like(s_probs, self.target_sparsity), reduction='mean')
        sample_avg_prob = probs.mean(dim=1) # [B]
        
        # 使用 MSE Loss
        sparsity_loss = F.mse_loss(
            sample_avg_prob, 
            torch.full_like(sample_avg_prob, self.target_sparsity)
        )

        diff = probs[:, 1:] - probs[:, :-1] # [B, N-1]
        loss_decrease = torch.relu(diff).mean()



        # 总 Loss
        total_stat_loss = (
            self.lambda_content * content_loss +
            self.lambda_decrease * loss_decrease +
            self.lambda_sparse * sparsity_loss
        )
        prob_var = probs.var(dim=1).mean()  # 每个样本内probs的方差
        # 目标方差：如果target_sparsity=0.5，理想的均匀分布方差约0.083
        # 我们用一个简单的损失鼓励方差不要太小
        diversity_loss = 1.0 / (prob_var + 1e-4)  # 方差越大loss越小
        # 归一化到合理范围
        diversity_loss = diversity_loss * 0.0005
        total_loss = total_stat_loss +  diversity_loss
        #print(f"STATLoss: loss_content={loss_content.item():.4f}, loss_decrease={loss_decrease.item():.4f}, loss_sparse={loss_sparse.item():.4f}, diversity_loss={diversity_loss.item():.4f}, total_loss={total_loss.item():.4f}")
        #print(f"STATLoss: content_loss={content_loss.item():.4f}, loss_decrease={loss_decrease.item():.4f}, sparsity_loss={sparsity_loss.item():.4f}, diversity_loss={diversity_loss.item():.4f}, total_stat_loss={total_stat_loss.item():.4f}, total_loss={total_loss.item():.4f}")
        
        return total_loss, {
            "loss_content": content_loss.item(),
            "loss_decrease": loss_decrease.item(),
            "loss_sparse": sparsity_loss.item(),
            "avg_tokens": (probs > 0.5).float().sum(-1).mean().item(),
            'diversity_loss': diversity_loss.item(),
        }



def get_optimizer_parameter_set(optimizer):
    return set([p for group in optimizer.param_groups for p in group['params']])


@register('larp_tokenizer_trainer_stat')
class LARPTokenizerTrainer(BaseTrainer):
    def __init__(self, rank, cfg):
        super().__init__(rank, cfg)
        loss_q_starting_ratio, loss_q_warmup_epochs = cfg['loss_q_warmup'].split('_')
        self.loss_q_starting_ratio = float(loss_q_starting_ratio)
        self.loss_q_warmup_epochs = int(loss_q_warmup_epochs)
        self.loss_q_weight = float(cfg['loss_q_weight'])
        self.clip_grad_max_norm = float(cfg['clip_grad_max_norm'])
        self.loss_latent_ce_weight = float(cfg.get('loss_latent_ce_weight', 0.0))

        self.kl_decay_epoch = cfg.get('kl_decay_epoch', -1)
        self.base_kl_weight = float(self.cfg['loss_kl_weight'])

        sqt_start_end_epoch = cfg.get('sqt_start_end_epoch', '0.0_0.0_0')
        sqt_start, sqt_end, sqt_epoch = sqt_start_end_epoch.split('_')  # start, end, epoch    decode     17
        self.sqt_start = float(sqt_start)
        self.sqt_end = float(sqt_end)
        self.sqt_epoch = int(sqt_epoch)
        self.stat_loss_fn=STATLoss()
        if self.sqt_start <= 0.0 or self.sqt_end <= 0.0 or self.sqt_epoch <= 0:
            self.set_sqt_every_step = False
        else:
            self.set_sqt_every_step = True
            self.log(f'sqt_start={self.sqt_start}, sqt_end={self.sqt_end}, sqt_epoch={self.sqt_epoch}')


    @staticmethod
    def get_exp_name(base_exp_name, cfg, args):
        exp_name = f"{base_exp_name}/"
        exp_name += f"b{args.batch_size}_"

        if float(cfg.optimizer.args.lr) != 0.0001:
            exp_name += f"lr{cfg.optimizer.args.lr}_"
        model_args = cfg.model.args
        exp_name += f"btn{model_args.bottleneck_token_num}_"

        if 'bottleneck' in cfg.model.args:
            bottleneck = cfg.model.args.bottleneck
            regularizer = bottleneck.args.regularizer
            exp_name += f"{regularizer.name}_"
            if 'codebook_size' in regularizer.args and regularizer.name.lower() not in ['no', 'none']:
                if int(regularizer.args.codebook_size) != 1024:
                    exp_name += f"rcs{regularizer.args.codebook_size}_"

        exp_name += f'_{args.tag}'
        return exp_name


    def make_model(self, model_spec=None, load_sd=False):
        super().make_model(model_spec, load_sd)


    def get_loss_q_weight(self):
        loss_q_weight = self.loss_q_weight
        if self.epoch < self.loss_q_warmup_epochs:
            ratio = self.loss_q_starting_ratio + (1 - self.loss_q_starting_ratio) * (self.epoch - 1) / (self.loss_q_warmup_epochs - 1)
            loss_q_weight = ratio * loss_q_weight
        return loss_q_weight
    

    def get_current_kl_weight(self):
        if self.kl_decay_epoch <= 0:
            return self.base_kl_weight
        else:
            cutoff_step = self.kl_decay_epoch * self.n_steps_per_epoch
            current_step = self.global_step
            # kl_weight linearly decays from base_kl_weight to 0 from first step to cutoff_step
            if current_step < cutoff_step:
                return self.base_kl_weight * (1 - current_step / cutoff_step)
            else:
                return 0.0
            
    def get_sqt_weight(self):
        assert self.set_sqt_every_step
        cutoff_step = self.sqt_epoch * self.n_steps_per_epoch
        current_step = self.global_step
        if current_step < cutoff_step:
            return self.sqt_start + (self.sqt_end - self.sqt_start) * current_step / cutoff_step
        else:
            return self.sqt_end


    def make_loss(self, loss_spec=None, load_sd=False):
        loss = loss_spec
        if loss is None:
            loss = self.cfg['loss']
        assert loss['name'].lower() not in ['', 'none', 'no', 'null', 'false'], 'loss not specified'

        loss = models.make(loss, load_sd=load_sd).to(self.device)
        loss.set_training_mode(True)
        loss.set_perceptual_eval()

        if self.distributed:
            loss = nn.SyncBatchNorm.convert_sync_batchnorm(loss)

        if hasattr(loss, 'discriminator'):
            disc_size = utils.compute_num_params(loss.discriminator, text=False)
            disc_size_str = utils.text2str(disc_size)
            self.log(f'Discriminator: #params={disc_size_str}')

        if self.compile:
            self.log(f'compiling loss with mode {self.cfg["compile_mode"]}')
            loss_compiled = torch.compile(loss, mode=self.cfg['compile_mode'])
        else:
            loss_compiled = loss

        if self.distributed:
            loss_ddp = nn.parallel.DistributedDataParallel(loss_compiled, device_ids=[self.rank])
        else:
            loss_ddp = loss_compiled

        self.loss = loss_compiled
        self.loss_ddp = loss_ddp

    def make_datasets(self):
        super().make_datasets()
        def get_vislist(dataset, n_vis=32):
            ids = torch.arange(n_vis) * (len(dataset) // n_vis)
            return Subset(dataset, ids.tolist())

        if hasattr(self, 'train_loader'):
            self.vislist_train = get_vislist(self.train_loader.dataset)
        if hasattr(self, 'test_loader_dict'):
            vislist_test = []
            for k, test_loader in self.test_loader_dict.items():
                vislist_test.append(get_vislist(test_loader.dataset))
            self.vislist_test = ConcatDataset(vislist_test)

    def configure_optimizers(self, config, load_sd=False):
        if isinstance(self.loss, nn.Module):
            model_params = self.orig_model.parameters()
            loss_params = self.loss.trainable_parameters()
            prior_name = 'prior_model'

            if (getattr(self.orig_model, prior_name) is not None):
                config_args_no_lr = deepcopy(config['args'])
                config_args_no_lr.pop('lr')
                if config.get('emb_lr_mult', 1.0) != 1.0:
                    emb_new_lr_param_dict = {pn: p for pn, p in self.orig_model.named_parameters() if '.' not in pn}
                else:
                    emb_new_lr_param_dict = {}

                model_other_param_dict = {pn: p for pn, p in self.orig_model.named_parameters() if not pn.startswith(prior_name) and pn not in emb_new_lr_param_dict}
                prior_param_dict = {pn: p for pn, p in getattr(self.orig_model, prior_name).named_parameters()}
                emb_new_lr_param = [emb_new_lr_param_dict[pn] for pn in sorted(list(emb_new_lr_param_dict.keys()))]
                model_other_param = [model_other_param_dict[pn] for pn in sorted(list(model_other_param_dict.keys()))]
                prior_param = [prior_param_dict[pn] for pn in sorted(list(prior_param_dict.keys()))]
                assert len(emb_new_lr_param) + len(model_other_param) + len(prior_param) == len(list(self.orig_model.named_parameters()))

                model_optimizer_groups = [
                    {'params': prior_param, 'lr': config['args']['lr'] * config['prior_lr_mult']},
                    {'params': model_other_param, 'lr': config['args']['lr']}
                ]

                if len(emb_new_lr_param) > 0:
                    model_optimizer_groups.append({'params': emb_new_lr_param, 'lr': config['args']['lr'] * config['emb_lr_mult']})

                model_optimizer = optimizer_dict[config['name']](model_optimizer_groups, **config_args_no_lr)
            else:
                model_optimizer = optimizer_dict[config['name']](model_params, **config['args'])

            if load_sd:
                model_optimizer.load_state_dict(config['sd'][0])    

            if 'loss_name' in config:
                loss_optimizer_name = config['loss_name']
            else:
                loss_optimizer_name = config['name'] 

            loss_optimizer = optimizer_dict[loss_optimizer_name](loss_params, **config['loss_args'])
            if load_sd:
                loss_optimizer.load_state_dict(config['sd'][1])

            self.optimizer = [model_optimizer, loss_optimizer]

        else: # loss is a function, so only one optimizer is needed
            params = self.orig_model.parameters()

            self.optimizer = optimizer_dict[config['name']](params, **config['args'])
            if load_sd:
                self.optimizer.load_state_dict(config['sd'])

    def configure_scalers(self, sd=None, load_sd=False):
        if load_sd:
            assert sd is not None, "ckpts must be provided to load state_dict"
        if isinstance(self.loss, nn.Module):
            model_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)
            if load_sd:
                model_scaler.load_state_dict(sd[0])
            if self.amp_dtype == torch.bfloat16:
                assert not model_scaler.is_enabled(), 'GradScaler should be disabled when using bfloat16'
                print('GradScaler is disabled because bfloat16 is used')
            discriminator_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)
            if load_sd:
                discriminator_scaler.load_state_dict(sd[1])
            self.scaler = [model_scaler, discriminator_scaler]
        else:
            super().configure_scalers(sd, load_sd)


    def _iter_step(self, data, is_train):
        start = time.time()
        data = data['gt'].to(self.device) 
        B = data.shape[0]

        # if self.set_sqt_every_step:
        #     sqt_weight = self.get_sqt_weight()
        #     self.orig_model.bottleneck.regularizer.set_stochastic_temperature(sqt_weight)
        # else:
        #     sqt_weight = self.orig_model.bottleneck.regularizer.default_stochastic_temperature 
        sqt_weight=0
        # generate fake frames
        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            model_output = self.model_ddp(
                data,
                current_epoch=self.epoch
                # global_step=self.global_step,
                # max_steps=self.max_steps,
            )
            assert isinstance(model_output, dict)
            pred_frames = model_output['pred_frames']
            current_stage = model_output['stage']
            if not is_train: # only calculate FVD and/or FID for validation
                if pred_frames.shape[2] >= 10: # FVD calc requires at least 10 frames
                    self.fake_stats = self.fvd_calculator.get_feature_stats_for_batch(pred_frames, self.fake_stats)
                    self.running_real_stats = self.fvd_calculator.get_feature_stats_for_batch(data, self.running_real_stats)
                elif self.fid_calculator is not None and pred_frames.shape[2] == 1: # calculate FID for images (single frame videos)
                    self.img_fake_stats = self.fid_calculator.get_feature_stats_for_batch(pred_frames, self.img_fake_stats)
                    self.img_running_real_stats = self.fid_calculator.get_feature_stats_for_batch(data, self.img_running_real_stats)

        info_dict = {}

        # training discriminator if needed
        if self.epoch >= self.loss.discriminator_self_start and (
            not is_train or self.global_step % self.loss.d_update_freq == 0
        ):
            if is_train:
                self.loss.trainable_requires_grad_(True)

            pred_frames_detached = pred_frames.detach()
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                disc_loss, disc_info_dict,_= self.loss_ddp( 
                    data, 
                    pred_frames_detached,
                    global_step=self.epoch,
                    for_discriminator=True, 
                    last_layer=None
                )
                info_dict.update({k: v for k, v in disc_info_dict.items() if k not in model_output})

            if is_train:
                self.optimizer[1].zero_grad()
                if disc_loss.item() > self.loss.d_update_loss_threshold:
                    self.scaler[1].scale(disc_loss).backward()
                    if self.clip_grad_max_norm > 0.0:
                        self.scaler[1].unscale_(self.optimizer[1])
                        torch.nn.utils.clip_grad_norm_(self.loss.trainable_parameters(), self.clip_grad_max_norm)

                    self.scaler[1].step(self.optimizer[1])
                    self.scaler[1].update()

        # training the generator (LARP encoder)
        self.loss.trainable_requires_grad_(False)

        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):     #perceptual loss calculation
            loss_obj, g_info_dict,lpips_per_sample = self.loss( 
                data, 
                pred_frames,
                global_step=self.epoch, 
                for_discriminator=False, 
                last_layer=None
            )
            info_dict.update(g_info_dict)




            

            if isinstance(loss_obj, dict): # seperate rp_loss adn g_loss
                loss: torch.Tensor = loss_obj['rp_loss']
                g_loss: torch.Tensor = loss_obj['g_loss']
            else:
                assert isinstance(loss_obj, torch.Tensor)
                loss: torch.Tensor = loss_obj
                g_loss: torch.Tensor = loss.new_zeros(1)

            mses = ((pred_frames - data)**2).reshape(B, -1).mean(dim=-1)
            psnr = -10 * torch.log10(mses).mean()
            ssim_v = ssim(pred_frames.flatten(end_dim=1), data.flatten(end_dim=1), data_range=1)
            info_dict['psnr'] = psnr.item()
            info_dict['ssim'] = ssim_v.item()
            info_dict['sqt_weight'] = sqt_weight

            if current_stage == "adaptive" and is_train:
                # 直接使用从 self.loss 返回的 lpips_per_sample
                # 它是 [B] 大小的 Tensor，且已经 detach 过了 (通常 LPIPS 不需要梯度传给 STAT 参照)
                # 哪怕没有显式 detach，lpips_per_sample 本身也是从计算图中来的
                # 为了安全起见，传入 STATLoss 时可以使用 .detach()
                probs = model_output['probs']
                # 这里的 lpips_per_sample 就是我们需要的 "Batch LPIPS"
                loss_stat, stat_logs = self.stat_loss_fn(probs, lpips_per_sample.detach())
                loss = loss + 0.1*loss_stat 
                info_dict.update(stat_logs)


            if 'loss_kl' in model_output:
                loss_kl = model_output.pop('loss_kl')
                current_kl_weight = self.get_current_kl_weight()
                loss = loss + loss_kl * current_kl_weight
                info_dict['loss_kl'] = loss_kl.item()
                info_dict['kl_weight'] = current_kl_weight

            if 'loss_q' in model_output:
                loss_q = model_output.pop('loss_q')
                loss = loss + loss_q * self.get_loss_q_weight()
                info_dict['loss_q'] = loss_q.item()

                with torch.no_grad():
                    codebook_size = self.model.codebook_size
                    used_indices = model_output['bottleneck_rep'][0].flatten() # only use the first instance in the batch
                    index_counts = torch.bincount(used_indices, minlength=codebook_size)
                    index_usage = utils.index_usage_percentage(index_counts)

                    used_indices_batch = model_output['bottleneck_rep'].flatten() # all instances in the batch
                    index_counts_batch = torch.bincount(used_indices_batch, minlength=codebook_size)
                    index_usage_batch = utils.index_usage_percentage(index_counts_batch)

                    kl_uni = utils.kl_divergence_from_uniform(index_counts)
                    encodings = F.one_hot(used_indices, codebook_size).float().reshape(-1, codebook_size)
                    avg_probs = encodings.mean(0)
                    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp().item()
                    info_dict['index_usage'] = index_usage
                    info_dict['index_usage_batch'] = index_usage_batch
                    info_dict['perplexity'] = perplexity
                    info_dict['kl_uni'] = kl_uni

            if 'loss_latent_ce' in model_output:
                loss_latent_ce = model_output.pop('loss_latent_ce')
                loss = loss + loss_latent_ce * self.loss_latent_ce_weight
                info_dict['loss_latent_ce'] = loss_latent_ce.item()
            
            for k, v in model_output.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    info_dict[k] = v.item()
                elif isinstance(v, float) or isinstance(v, int):
                    info_dict[k] = v

            info_dict['loss'] = loss.item()

        if is_train:
            self.optimizer[0].zero_grad()
            if g_loss.requires_grad : # seperate rp_loss adn g_loss
                raise NotImplementedError('not implemented yet')
            else:
                self.scaler[0].scale(loss).backward()
            if self.clip_grad_max_norm > 0.0:
                self.scaler[0].unscale_(self.optimizer[0])
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_max_norm)
            self.scaler[0].step(self.optimizer[0])
            self.scaler[0].update()
            for ema_decay, ema_model in self.ema_model_dict.items():
                self.update_ema(ema_model, decay=ema_decay)

        fps = B / (time.time() - start)
        info_dict['fps'] = fps

        return info_dict

    def train_epoch(self):
        self.model.requires_grad_(True)
        return super().train_epoch()

    def train_step(self, data):
        return self._iter_step(data, is_train=True)

    def evaluate_step(self, data):
        with torch.no_grad():
            return self._iter_step(data, is_train=False)

    @torch.no_grad()
    def _gen_vis_result(self, tag, vislist):
        self.model_ddp.eval()
        out_dir = os.path.join(self.cfg['env']['save_dir'], 'visualize')
        os.makedirs(out_dir, exist_ok=True)
        vid_dir, img_dir = os.path.join(out_dir, 'vid'), os.path.join(out_dir, 'img')
        if self.is_master:
            for cur_dir in [vid_dir, img_dir]:
                if not os.path.exists(cur_dir):
                    os.makedirs(cur_dir)

        res = []

        for batch_id, data in enumerate(vislist):
            data = {k: data[k].unsqueeze(0).cuda() for k in ['gt']}
            gt = data['gt'][0] # [c, t, h, w]
            model_input = {
                'data': data['gt'], 
                'global_step': self.global_step,
                'max_steps': self.max_steps,
            }
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.model_ddp(**model_input)
                pred = output['pred_frames'] # [b, c, t, h, w]
                pred = pred.clamp(0., 1.)
                if hasattr(self.model, 'output_format') and self.model.output_format == 'bcthw':
                    pred = einops.rearrange(pred, 'b c t h w -> t b c h w').contiguous()
                pred = pred[:, 0] # [t, c, h, w]
            res.append(gt.permute(1,0,2,3)) # [t, c, h, w]
            res.append(pred.float())

        res = torch.stack(res)
        res = res.detach().cpu()

        if self.enable_tb:
            self.writer.add_video(tag, res, self.epoch)
        if self.enable_wandb:
            wandb.log({tag: wandb.Video(res*255., format='mp4')}, step=self.epoch)

    def visualize_epoch(self):
        pass
        # if hasattr(self, 'vislist_train'):
        #     self._gen_vis_result('vis_train_dataset', self.vislist_train)
        # if hasattr(self, 'vislist_test'):
        #     self._gen_vis_result('vis_test_dataset', self.vislist_test)

