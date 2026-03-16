from typing import Optional, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import einops
import lpips
import math
from omegaconf import DictConfig
from torch.autograd.functional import jvp as jvp
from torch.nn.attention import sdpa_kernel, SDPBackend
from beartype import beartype

from src.structs import EasyDict, TensorGroup
from src.utils import misc
from src.utils.training_utils import sample_frames_masks, cut_dct2d_high_freqs, reg_dc_dct2d_high_freqs, compute_annealed_weight, compute_scheduled_weight
from src.training.perceptual_loss import PerceptualPyramidLoss
from src.structs import TokenType, LossPhase, TensorLike, BaseLoss
from src.training.network_utils import load_snapshot

#----------------------------------------------------------------------------
# Some constants.

SOBEL_FILTER = [
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
] # [2, 3, 3]

#----------------------------------------------------------------------------

class DiffusionLoss(BaseLoss):
    def __init__(self, cfg: EasyDict):
        super().__init__()
        self.cfg = cfg
        self.teacher = None

        if self.cfg.use_single_step_denoising:
            self._init_lpips_models()

    def _get_sigma_shape(self, input_shape: torch.Size) -> torch.Size:
        if self.cfg.per_pixel_noise_level:
            return torch.Size(input_shape[:2]) + torch.Size([1]) + torch.Size(input_shape[3:]) # [5]. Shape is [b, t, 1, h, w]
        else:
            return torch.Size(input_shape[:1]) + torch.Size([1] * (len(input_shape) - 1)) # [5]. Shape is [b, 1, 1, 1, 1]

    def _init_lpips_models(self):
        self.lpips_pyr = PerceptualPyramidLoss(
            scales=self.cfg.perceptual_loss.scales,
            loss_weights=self.cfg.perceptual_loss.weights,
            replace_maxpool_with_avgpool=self.cfg.perceptual_loss.replace_maxpool_with_avgpool,
            downsample_to_native=self.cfg.perceptual_loss.downsample_to_native,
        ).requires_grad_(False) if self.cfg.perceptual_loss.weight > 0 else None
        self.lpips_alex = lpips.LPIPS(net='alex').requires_grad_(False) if self.cfg.perceptual_loss_alex.weight > 0 else None
        self.lpips_vgg = lpips.LPIPS(net='vgg').requires_grad_(False) if self.cfg.perceptual_loss_vgg.weight > 0 else None
        self.lpips_squeeze = lpips.LPIPS(net='squeeze').requires_grad_(False) if self.cfg.perceptual_loss_squeeze.weight > 0 else None

    def sample_sigma(self, batch_size, device, cfg=None) -> torch.Tensor:
        raise NotImplementedError

    def compute_loss_weight(self, sigma: torch.Tensor, logvar: torch.Tensor=None) -> torch.Tensor:
        raise NotImplementedError

    def apply_noise(self, videos: torch.Tensor, noise_scaled: torch.Tensor, sigma: torch.Tensor, net: torch.nn.Module) -> torch.Tensor: # pylint: disable=unused-argument
        raise NotImplementedError

    def compute_x_denoised(self, net_output: torch.Tensor, videos_noised: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """x_denoised for single-step trajectory solving"""
        raise NotImplementedError

    @torch.no_grad()
    def _maybe_compute_sc_latents(self, net, videos_aug_noised, sigma, cond, **kwargs) -> torch.Tensor:
        if self.cfg.model.self_cond_probability > 0.0:
            sc_latents = net(videos_aug_noised, sigma, cond, return_extra_output=True, **kwargs)[1]['sc_latents'] # [b, lt, c, lh, lw] or [b, num_latents, lat_dim]
        else:
            sc_latents = None
        return sc_latents

    @torch.no_grad()
    def _maybe_encode(self, net, x: TensorLike, cond) -> TensorLike:
        if misc.unwrap_module(net).cfg.is_lgm:
            return misc.unwrap_module(net).encode(x, cond=cond, only_normalize=self.cfg.model.use_precomputed_latents)
        else:
            return x

    def compute_kl_loss(self, loss_total: torch.Tensor, ctx: EasyDict, cur_step: int=None, loss_weight: torch.Tensor=None) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.cfg.model.is_vae:
            loss_kl = ctx['kl'] # [b]
            loss_kl_diffusion_weighted = (loss_weight * loss_kl) if (loss_weight is not None and self.cfg.apply_denoising_loss_weight_to_reg) else loss_kl # [b]
            raw_kl_loss_weight: float = self.get_cur_kl_loss_weight(cur_step) # [1]
            if self.cfg.kl_weight_relative:
                loss_kl_weighted = reweigh_supp_loss(loss_total, loss_kl_diffusion_weighted, raw_kl_loss_weight) # [b]
            else:
                loss_kl_weighted = raw_kl_loss_weight * loss_kl_diffusion_weighted # [b]
        else:
            loss_kl = loss_kl_weighted = None

        return EasyDict(kl=loss_kl, kl_weighted=loss_kl_weighted)

    def maybe_apply_reg(self, loss_total: torch.Tensor, ctx: EasyDict, cur_step: int=None, **kwargs) -> tuple[torch.Tensor, EasyDict[str, Optional[torch.Tensor]]]:
        kl_losses = self.compute_kl_loss(loss_total, ctx, cur_step, **kwargs) # <str, [b]>
        if kl_losses.kl_weighted is not None:
            loss_total = loss_total + kl_losses.kl_weighted # [b]
        reg_losses = EasyDict(**kl_losses) # <str, [b]>
        return loss_total, reg_losses

    def compute_rec_losses(self, videos_pred: torch.Tensor, targets: torch.Tensor, frames_mask: torch.Tensor=None, loss_weights_overrides: dict=None) -> EasyDict[str, torch.Tensor]:
        loss_weights = EasyDict(
            lpips_pyr=self.cfg.perceptual_loss.weight,
            lpips_alex=self.cfg.perceptual_loss_alex.weight,
            lpips_vgg=self.cfg.perceptual_loss_vgg.weight,
            lpips_squeeze=self.cfg.perceptual_loss_squeeze.weight,
            mse=self.cfg.mse_loss.weight,
            mae=self.cfg.mae_loss.weight,
            pseudo_huber=self.cfg.pseudo_huber_loss.weight,
            img_grad=self.cfg.img_grad_loss.weight,
            freq2d=self.cfg.freq2d_loss.weight,
            freq3d=self.cfg.freq3d_loss.weight,
            random_conv_l2=self.cfg.random_conv_l2_loss.weight,
            video_random_conv_l2=self.cfg.video_random_conv_l2_loss.weight,
        )
        loss_weights = EasyDict(**{**loss_weights, **loss_weights_overrides}) if loss_weights_overrides is not None else loss_weights
        loss_fns = EasyDict(
            lpips_pyr = self.lpips_pyr, # [b, t, 1, 1, 1]
            lpips_alex = lambda x_rec, x_gt: compute_video_lpips(x_rec, x_gt, self.lpips_alex, self.cfg.perceptual_loss_alex), # [b, t, 1, 1, 1]
            lpips_vgg = lambda x_rec, x_gt: compute_video_lpips(x_rec, x_gt, self.lpips_vgg, self.cfg.perceptual_loss_vgg), # [b, t, 1, 1, 1]
            lpips_squeeze = lambda x_rec, x_gt: compute_video_lpips(x_rec, x_gt, self.lpips_squeeze, self.cfg.perceptual_loss_squeeze), # [b, t, 1, 1, 1]
            mse = lambda x_rec, x_gt: ((x_rec - x_gt) ** 2), # [b, t, c, h, w]
            mae = lambda x_rec, x_gt: (x_rec - x_gt).abs(), # [b, t, c, h, w]
            pseudo_huber = lambda x_rec, x_gt: ((x_rec - x_gt) ** 2 + self.cfg.pseudo_huber_loss.breadth_coef ** 2).sqrt() - self.cfg.pseudo_huber_loss.breadth_coef, # [b, t, c, h, w]
            img_grad = compute_img_grad_loss_per_frame, # [b, t, 1, 1, 1]
            freq2d = compute_video_freq2d_loss, # [b, t, 1, 1, 1]
            freq3d = compute_video_freq3d_loss, # [b, t, 1, 1, 1]
            random_conv_l2 = lambda x_rec, x_gt: compute_framewise_video_random_conv_l2(x_rec, x_gt, self.cfg.random_conv_l2_loss), # [b, t, 1, 1, 1]
            video_random_conv_l2 = lambda x_rec, x_gt: compute_video_random_conv_l2(x_rec, x_gt, self.cfg.video_random_conv_l2_loss), # [b, t, 1, 1, 1]
        )
        losses_all = EasyDict(**{k: loss_fn(videos_pred, targets) if loss_weights[k] > 0 else None for k, loss_fn in loss_fns.items()})
        losses_all = misc.filter_nones(losses_all)
        losses_all_filtered = EasyDict(**{k: maybe_filter_loss_by_mask(l, frames_mask) for k, l in losses_all.items()}) # [b, t, c, h, w]
        losses_all_filtered_agg = EasyDict(**{k: l.reshape(len(l), -1).mean(dim=1, keepdim=True) for k, l in losses_all_filtered.items()}) # [b]
        losses_all_filtered_agg_weighted = EasyDict(**{k: (l * loss_weights[k]) for k, l in losses_all_filtered_agg.items()}) # [b]
        loss_total_rec = sum(l for l in losses_all_filtered_agg_weighted.values()) # [b]
        return EasyDict(rec=loss_total_rec, **losses_all_filtered_agg) # Returning unweighted losses for logging purposes.

    def get_cur_kl_loss_weight(self, cur_step: int) -> float:
        if self.cfg.model.is_vae:
            if len(self.cfg.kl_weight_schedule) > 0:
                assert self.cfg.kl_weight == 0.0, f"Expected kl_weight to be 0.0, but got {self.cfg.kl_weight} instead."
                cur_kl_weight: float = compute_scheduled_weight(cur_step, self.cfg.kl_weight_schedule)
            elif self.cfg.kl_weight_anneal_steps is not None:
                cur_kl_weight = compute_annealed_weight(cur_step, start_weight=0.0, end_weight=self.cfg.kl_weight, annealing_steps=self.cfg.kl_weight_anneal_steps)
            else:
                cur_kl_weight = self.cfg.kl_weight
        else:
            cur_kl_weight = 0.0
        return cur_kl_weight

    # This functon shouldn't be overridden. Instead, override the methods.
    def forward(self, net, x: TensorLike, cond: TensorGroup, augment_pipe=None, phase=LossPhase.Gen, cur_step=None, force_sigma_val: Optional[float]=None) -> EasyDict[str, torch.Tensor]:
        _ = phase # Unused.
        assert isinstance(x, TensorGroup) or x.ndim == 5, f"Expected tensor x to be videos and have 5 dimensions: [b, t, c, h, w], got {x.shape} instead."
        x, augment_labels = maybe_augment_videos(x, augment_pipe) # [b, t, c, h, w], [b, augment_dim] or None
        x = self._maybe_encode(net, x, cond) # [b, t, c, h, w]
        sigma = self.sample_sigma(x.shape, x.device) # [b, ...]
        if force_sigma_val is not None:
            sigma.fill_(force_sigma_val)
        noise_unscaled = misc.randn_like(x) # [b, t, c, h, w]
        noise_scaled = noise_unscaled * sigma # [b, t, c, h, w]
        videos_aug_noised = self.apply_noise(x, noise_scaled, sigma, net) # [b, t, c, h, w]
        sc_latents = self._maybe_compute_sc_latents(net, videos_aug_noised, sigma, cond, augment_labels=augment_labels) # [b, num_latents, lat_dim] or None
        net_output, ctx = net(videos_aug_noised, sigma, cond, augment_labels=augment_labels, sc_latents=sc_latents, return_extra_output=True) # [b, t, c, h, w]
        net_output = TensorGroup(net_output) if isinstance(net_output, dict) else net_output # [b, t, c, h, w]
        targets = self.compute_targets(x, noise_unscaled, sigma) # [b, t, c, h, w]
        if isinstance(net_output, TensorGroup) and 'audio' in net_output:
            targets.audio = targets.audio[:, :net_output.audio.shape[1], :] # [b, t_a, c_a, 1, 1]
        loss_diffusion = (net_output - targets) ** 2 # [b, t, c, h, w]
        loss_rec = self.cfg.denoising_loss_weight * loss_diffusion # [b, t, c, h, w]
        if self.cfg.use_single_step_denoising:
            videos_denoised = self.compute_x_denoised(net_output, videos_aug_noised, sigma) # [b, t, c, h, w]
            single_step_rec_losses = self.compute_rec_losses(videos_denoised, x) # [b]
            single_step_rec_losses.rec_single_step = single_step_rec_losses.pop('rec') # [b]
            loss_rec = loss_rec + single_step_rec_losses.rec_single_step # [b, t, c, h, w]
        else:
            single_step_rec_losses = EasyDict({})
        loss_weight = self.compute_loss_weight(sigma, logvar=ctx.get('logvar')) # [b, ...]
        loss_total = loss_weight * loss_rec + (ctx['logvar'] if ctx.get('logvar') is not None else 0.0) # [b, t, c, h, w]
        loss_total, reg_losses_dict = self.maybe_apply_reg(loss_total, ctx, cur_step=cur_step, loss_weight=loss_weight) # [b], <str, [b]>
        loss_dict = misc.filter_nones(_maybe_convert_tensor_group(EasyDict(
            total=loss_total, # [b]
            logvar_uncertainty=ctx.get('logvar'), # [b, tl, c, hl, wl]
            rec=loss_rec * loss_weight, # [b]
            rec_unweighted=loss_rec, # [b]
            diffusion=loss_weight * loss_diffusion, # [b]
            diffusion_unwheighted=loss_diffusion, # [b]
            **reg_losses_dict, **single_step_rec_losses)))
        return loss_dict

#----------------------------------------------------------------------------

class EDMLoss(DiffusionLoss):
    def sample_sigma(self, shape: torch.Size, device, cfg=None) -> torch.Tensor:
        cfg = self.cfg if cfg is None else cfg
        rnd_normal = torch.randn(self._get_sigma_shape(shape), device=device) # [b, ...]
        return (rnd_normal * cfg.P_std + self.cfg.P_mean).exp() # [b, ...]

    def apply_noise(self, videos: torch.Tensor, noise_scaled: torch.Tensor, sigma: torch.Tensor, net: torch.nn.Module) -> torch.Tensor: # pylint: disable=unused-argument
        return videos + noise_scaled

    def compute_targets(self, videos_gt: torch.Tensor, noise_unscaled: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor: # pylint: disable=unused-argument
        return videos_gt

    def compute_loss_weight(self, sigma: torch.Tensor, logvar: torch.Tensor=None) -> torch.Tensor:
        if self.cfg.model.sampling.sigma_max == 0.0:
            return misc.ones_like(sigma)
        return (sigma ** 2 + self.cfg.model.sigma_data ** 2) / (sigma * self.cfg.model.sigma_data) ** 2 / (logvar.exp() if logvar is not None else 1.0)

#----------------------------------------------------------------------------

class RecFlowLoss(DiffusionLoss):
    def sample_sigma(self, shape: torch.Size, device, cfg=None) -> torch.Tensor:
        cfg = self.cfg.time_sampling if cfg is None else cfg
        shape = self._get_sigma_shape(shape)
        if cfg.custom_t_steps is None:
            return sample_logit_norm_time(cfg, shape, device=device) # [b, ...]
        else:
            return sample_custom_t_steps(cfg.custom_t_steps, shape, device=device) # [b, ...]

    def compute_loss_weight(self, sigma: TensorLike, logvar: TensorLike=None) -> TensorLike:
        return misc.ones_like(sigma) / (logvar.exp() if logvar is not None else 1.0) # [b, 1, 1, 1, 1]

    def apply_noise(self, videos: TensorLike, noise_scaled: TensorLike, sigma: TensorLike, net: torch.nn.Module) -> TensorLike:
        # Noise has already been scaled by sigma, the only extra scaling for it is sigma_noise.
        return (1 - sigma) * videos + self.cfg.model.sampling.sigma_noise * noise_scaled # [b, t, c, h, w]

    def compute_targets(self, videos_gt: TensorLike, noise_unscaled: TensorLike, sigma: TensorLike) -> TensorLike:
        _ = sigma # Unused.
        return noise_unscaled * self.cfg.model.sampling.sigma_noise - videos_gt # [b, t, c, h, w]

    def compute_x_denoised(self, net_output: TensorLike, videos_noised: TensorLike, sigma: TensorLike) -> TensorLike:
        # Sometimes we want to do kinda one-step diffusion, where the target is the clean video.
        # For rectified flows:
        #     - net_output is the velocity: v = noise - x_0 (independent of the time step) of going from x_0 to x_1.
        #     - sigma is time t (the distance betwen noise = z_0 and z_t)
        #     - the starting point z_t = (1 - t) * x_0 + t * noise.
        # We do a single large step with the predicted velocity: x_0^hat = z_t - t * v
        return videos_noised - sigma * net_output # [b, t, c, h, w]

#----------------------------------------------------------------------------

class AutoEncoderLoss(EDMLoss):
    def __init__(self, cfg: EasyDict):
        super().__init__(cfg)
        self._init_lpips_models()
        self.teacher = None if all(w == 0 for w in self.cfg.teacher.weights.values()) else load_snapshot(self.cfg.teacher.ckpt, verbose=False)[0].train().requires_grad_(False)

    def compute_ae_loss(self, videos_pred, videos_gt, noise_unscaled, sigma, ctx, cur_step, frames_mask) -> tuple[EasyDict, torch.Tensor, torch.Tensor]:
        targets = self.compute_targets(videos_gt, noise_unscaled, sigma) # [b, t, c, h, w]
        loss_weight = self.compute_loss_weight(sigma) # [b, 1, 1, 1, 1]
        losses_rec_all = self.compute_rec_losses(videos_pred, targets, frames_mask=frames_mask)
        loss_total = loss_weight.reshape(losses_rec_all.rec.shape[0]) * losses_rec_all.rec # [b]
        loss_total, reg_losses_dict = self.maybe_apply_reg(loss_total, ctx, cur_step, loss_weight=loss_weight) # [1], <str, [b]>

        return misc.filter_nones(EasyDict(total=loss_total, **reg_losses_dict, **losses_rec_all))

    def maybe_compute_lipschitz_reg(self, losses: EasyDict, latents: torch.Tensor, videos_pred: torch.Tensor, cur_step: int) -> EasyDict[str, torch.Tensor]:
        _ = cur_step # Unused.
        if self.cfg.dec_lipschitz_reg.r1_weight == 0:
            return losses
        dec_r1_reg = compute_r1_reg(latents, videos_pred, reduce_op='mean') # [b]
        dec_r1_reg_weighted = self.cfg.dec_lipschitz_reg.r1_weight * self.cfg.dec_lipschitz_reg.r1_freq * dec_r1_reg # [b]
        loss_total_new = losses.total + dec_r1_reg_weighted # [b]
        return EasyDict.init_recursively({**losses, 'total': loss_total_new, 'decoder_r1_reg': dec_r1_reg, 'dec_r1_reg_weighted': dec_r1_reg_weighted})


    def maybe_compute_scale_equiv_reg(self, net, losses: EasyDict, latents: torch.Tensor, videos_gt: torch.Tensor, cond, augment_labels) -> EasyDict:
        if self.cfg.scale_equiv_reg.weight == 0:
            return losses

        assert len(self.cfg.scale_equiv_reg.scale_factors) > 0 or self.cfg.scale_equiv_reg.cut_dct2d_high_freqs.max_cut_ratio > 0, "Can only use one of the downsampling methods, but got both."
        assert len(self.cfg.scale_equiv_reg.scale_factors) == 0 or self.cfg.scale_equiv_reg.cut_dct2d_high_freqs.max_cut_ratio == 0, "Expected only one of the downsampling methods to be used, but got both."
        if len(self.cfg.scale_equiv_reg.scale_factors) > 0:
            scale_factor: float = np.random.choice(self.cfg.scale_equiv_reg.scale_factors) # Randomly choose the scale factor.
            if self.cfg.scale_equiv_reg.resample_strategy == 'mean':
                inv_scale_factor = 1.0 / scale_factor
                assert inv_scale_factor.is_integer(), f"Expected inv_scale_factor to be an integer, but got {inv_scale_factor} instead."
                interp_kwargs = dict(pattern='b t c (h fh) (w fw) -> b t c h w', reduction='mean', fh=int(inv_scale_factor), fw=int(inv_scale_factor))
                videos_gt_down, latents_down = [einops.reduce(x, **interp_kwargs) for x in (videos_gt, latents)] # [b, t, c, h/sh, w/sw], [b, lt, c, lh/sh, lw/sw]
            elif self.cfg.scale_equiv_reg.resample_strategy == 'bil':
                interp_kwargs = dict(scale_factor=scale_factor, mode='bilinear', align_corners=True)
                videos_gt_down, latents_down = [F.interpolate(x.flatten(0, 1), **interp_kwargs).unflatten(dim=0, sizes=(len(x), -1)) for x in (videos_gt, latents)] # [b, t, c, h, w]
            else:
                raise NotImplementedError(f"Unknown resampling strategy: {self.cfg.scale_equiv_reg.resample_strategy}")

            # Handling temporal compression separately.
            temporal_scale_factor: float = float(1 / np.random.choice(self.cfg.scale_equiv_reg.temporal_scale_factors)) if len(self.cfg.scale_equiv_reg.temporal_scale_factors) > 0 else 1.0
            if temporal_scale_factor > 1:
                # We resize all the frames except the first one to maintain causality.
                interp_kwargs = dict(pattern='b (t ft) c h w -> b (t ft) c h w', reduction='mean', ft=int(temporal_scale_factor))
                videos_gt_down, latents_down = [torch.cat([x[:, :1], einops.reduce(x[:, 1:], **interp_kwargs)], dim=1) for x in (videos_gt_down, latents_down)] # [b, 1 + t / ft, c, h/sh, w/sw], [b, 1 + lt / ft, c, lh/sh, lw/sw]
        else:
            assert len(self.cfg.scale_equiv_reg.temporal_scale_factors) == 0, f"Expected temporal_scale_factors to be empty, but got {self.cfg.scale_equiv_reg.temporal_scale_factors} instead."
            cut_kwargs = EasyDict(
                cut_ratio=np.random.rand() * self.cfg.scale_equiv_reg.cut_dct2d_high_freqs.max_cut_ratio,
                block_size=self.cfg.scale_equiv_reg.cut_dct2d_high_freqs.block_size,
                zigzag=self.cfg.scale_equiv_reg.cut_dct2d_high_freqs.zigzag,
            )
            # Cut the high frequencies for GT videos and return them together with the cut_kwargs.
            latents_down = cut_dct2d_high_freqs(latents, block_size=cut_kwargs.block_size, zigzag=cut_kwargs.zigzag, cut_ratio=cut_kwargs.cut_ratio) # [b, t, c, h, w]
            rgb_block_size = cut_kwargs.block_size * misc.unwrap_module(net).compression_rate[-1] # [1]
            videos_gt_down = cut_dct2d_high_freqs(videos_gt, block_size=rgb_block_size, zigzag=cut_kwargs.zigzag, cut_ratio=cut_kwargs.cut_ratio) # [b, t, c, h, w]

        videos_pred_down = net(None, None, cond, augment_labels=augment_labels, encode=False, decode=True, latents=latents_down) # [b, t, c, h / sh, w / sw]
        loss_weights_overrides = dict(lpips_pyr=0.0, lpips_vgg=0.0, lpips_alex=0.0, lpips_squeeze=0.0) if self.cfg.scale_equiv_reg.ignore_lpips else None
        ae_down_losses = self.compute_rec_losses(videos_pred_down, videos_gt_down, loss_weights_overrides=loss_weights_overrides) # <str, [b]>
        losses.total = losses.total + self.cfg.scale_equiv_reg.weight * ae_down_losses.rec # [b]
        ae_down_losses = EasyDict({f'latdown_{k}': l for k, l in ae_down_losses.items()}) # <str, [b]>
        return EasyDict.init_recursively({**losses, **ae_down_losses}) # <str, [b]>

    def maybe_compute_highfreq_reg(self, losses: EasyDict, latents: torch.Tensor) -> EasyDict:
        if self.cfg.high_freq_reg.weight == 0:
            return losses

        high_freq_reg = reg_dc_dct2d_high_freqs(latents, self.cfg.high_freq_reg.block_size, self.cfg.high_freq_reg.power) # [batch_size]
        losses.total = losses.total + self.cfg.high_freq_reg.weight * high_freq_reg # [b]
        return EasyDict(**losses, high_freq_reg=high_freq_reg) # <str, [b]>

    def compute_video_pred(self, net, videos_gt, cond, augment_labels, phase) -> tuple[torch.Tensor, torch.Tensor | None, EasyDict]:
        sigma = self.sample_sigma(videos_gt.shape, videos_gt.device) # [b, ...]
        noise_unscaled = torch.randn_like(videos_gt) # [b, t, c, h, w]
        noise_scaled = noise_unscaled * sigma # [b, t, c, h, w]
        videos_aug_noised = self.apply_noise(videos_gt, noise_scaled, sigma, net=net) # [b, t, c, h, w]
        enable_latents_grad = (phase == LossPhase.GenAll and self.cfg.dec_lipschitz_reg.r1_weight > 0) or phase == LossPhase.GenLipReg
        videos_pred, ctx = net(videos_aug_noised, sigma, cond, augment_labels=augment_labels, return_extra_output=True, encode=True, decode=True, enable_latents_grad=enable_latents_grad) # [b, t, c, h, w]
        latents = ctx['latents'] if 'latents' in ctx else None

        return videos_pred, videos_gt, latents, ctx, noise_unscaled, sigma

    def forward(self, net, x: torch.Tensor, cond: TensorGroup, augment_pipe=None,  phase=LossPhase.GenAll, cur_step: int=None, force_sigma_val: Optional[float]=None) -> EasyDict[str, torch.Tensor]:
        assert x.ndim == 5, f"Expected x to be videos and have 5 dimensions: [b, t, c, h, w], got {x.shape} instead."
        _ = force_sigma_val # Unused.
        x, augment_labels = maybe_augment_videos(x, augment_pipe) # [b, t, c, h, w], [b, augment_dim] or None

        if self.cfg.model.is_masked_ae:
            cond.frames_mask = sample_frames_masks(x, self.cfg.model.mask_sampling) # [b, t]
            cond.x_cond = x # [b, t, c, h, w]

        with torch.set_grad_enabled(False if phase == LossPhase.Discr else (self.cfg.dec_lipschitz_reg.r1_weight > 0 or torch.is_grad_enabled())):
            x_pred, x, latents, ctx, noise_unscaled, sigma = self.compute_video_pred(net, x, cond, augment_labels, phase)
            if phase != LossPhase.GenLipReg:
                losses = self.compute_ae_loss(x_pred, x, noise_unscaled, sigma, ctx, cur_step, frames_mask=cond.get('frames_mask')) # [b]
            else:
                losses = EasyDict(total=torch.zeros(len(x), device=x.device)) # <str, [b]>
            losses = self.maybe_compute_scale_equiv_reg(net, losses, latents, x, cond, augment_labels) # <str, [b]>
            losses = self.maybe_compute_highfreq_reg(losses, latents) # <str, [b]>
            if phase in (LossPhase.GenAll, LossPhase.GenLipReg):
                losses = self.maybe_compute_lipschitz_reg(losses, latents, x_pred, cur_step) # <str, [b]>

        return losses

#----------------------------------------------------------------------------

class AlphaFlowLoss(RecFlowLoss):
    @beartype
    def sample_timestep(self, sampling_cfg: EasyDict, cur_step: int, batch_size: int, device: torch.device, upper_truncated: Optional[float] = None) -> torch.Tensor:
        if sampling_cfg.timestep_distrib_type == "logit_norm":
            return sample_logit_norm_time(sampling_cfg, batch_size, device=device)
        elif sampling_cfg.timestep_distrib_type == "truncated_logit_norm":
            assert upper_truncated is not None
            return sample_truncated_logit_norm_time(sampling_cfg, batch_size, device=device, upper_truncated = upper_truncated)
        elif sampling_cfg.timestep_distrib_type == "adaptive_beta":
            # Linearly interpolate alpha and beta from initial to end values between init_steps and end_steps
            if cur_step < sampling_cfg.init_steps:
                alpha, beta = sampling_cfg.initial_alpha, sampling_cfg.initial_beta
            elif cur_step > sampling_cfg.end_steps:
                alpha, beta = sampling_cfg.end_alpha, sampling_cfg.end_beta
            else:
                progress = (cur_step - sampling_cfg.init_steps) / max(1, (sampling_cfg.end_steps - sampling_cfg.init_steps))
                alpha = sampling_cfg.initial_alpha + (sampling_cfg.end_alpha - sampling_cfg.initial_alpha) * progress
                beta = sampling_cfg.initial_beta + (sampling_cfg.end_beta - sampling_cfg.initial_beta) * progress
            beta_distrib = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([beta]))
            return beta_distrib.sample((batch_size,), device=device)
        elif sampling_cfg.timestep_distrib_type == "uniform":
            return torch.rand((batch_size,), device=device) * (sampling_cfg.max - sampling_cfg.min) + sampling_cfg.min
        elif sampling_cfg.timestep_distrib_type == "constant":
            return torch.ones((batch_size,), device=device) * sampling_cfg.scale
        elif sampling_cfg.timestep_distrib_type == "arctan":
            sigma = torch.exp(torch.randn((batch_size,), device=device) * sampling_cfg.scale + sampling_cfg.location)
            return 2 / math.pi * torch.atan(sigma)

    @beartype
    def get_ratio(self, cfg: EasyDict, cur_step: int) -> float:
        if cfg.scheduler == "constant":
            current_ratio = cfg.initial_value
        elif cfg.scheduler == "step":
            assert cfg.change_init_steps == cfg.change_end_steps, "For step scheduler, change_init_steps and change_end_steps must be equal."
            current_ratio = cfg.initial_value if cur_step < cfg.change_init_steps else cfg.end_value
        elif cfg.scheduler in ["linear", "exponential", "log", "sigmoid"]:
            if cur_step < cfg.change_init_steps:
                current_ratio = cfg.initial_value
            elif cur_step > cfg.change_end_steps:
                current_ratio = cfg.end_value
            else:
                if cfg.scheduler in ["linear", "exponential", "log"]:
                    progress = (cur_step - cfg.change_init_steps) / (cfg.change_end_steps - cfg.change_init_steps)
                elif cfg.scheduler == "sigmoid":
                    middle_step = cfg.change_init_steps + (cfg.change_end_steps - cfg.change_init_steps) / 2
                    progress = (cur_step - middle_step) / (cfg.change_end_steps - cfg.change_init_steps)

                if cfg.scheduler == "linear":
                    current_ratio = cfg.initial_value + (cfg.end_value - cfg.initial_value) * progress
                elif cfg.scheduler == "exponential":
                    progress = progress ** cfg.gamma
                    current_ratio = cfg.initial_value * ((cfg.end_value / cfg.initial_value) ** progress)
                elif cfg.scheduler == "log":
                    log_progress = math.log(1 + progress * 9) / math.log(10)
                    current_ratio = cfg.initial_value + (cfg.end_value - cfg.initial_value) * log_progress
                elif cfg.scheduler == "sigmoid":
                    current_ratio = cfg.initial_value + (cfg.end_value - cfg.initial_value) * (1 / (1 + math.exp(-progress * cfg.gamma)))
        else:
            raise NotImplementedError(f"Unknown scheduler type: {cfg.scheduler}")

        if current_ratio < cfg.clamp_value:
            current_ratio = 0.0
            if "discrete_training" in cfg and cfg.discrete_training:
                current_ratio = cfg.clamp_value
        elif current_ratio > 1 - cfg.clamp_value or (cfg.up_clamp_value is not None and current_ratio > cfg.up_clamp_value):
            current_ratio = 1.0
        return current_ratio
      
    def sample_timesteps_mf(self, cfg, cur_step, batch_size, device):
        if cfg.type == "truncated":
            t = self.sample_timestep(cfg.time_sampling_mf_t, cur_step, batch_size, device=device)
            t_next = self.sample_timestep(cfg.time_sampling_mf_t_next, cur_step, batch_size, device=device, upper_truncated = t)
        elif cfg.type in ["minmax", "min", "r_in_t_range"]:
            t_1 = self.sample_timestep(cfg.time_sampling_mf_t, cur_step, batch_size, device=device)
            t_2 = self.sample_timestep(cfg.time_sampling_mf_t_next, cur_step, batch_size, device=device)
            if cfg.type == "minmax":
                t = torch.maximum(t_1, t_2)
                t_next = torch.minimum(t_1, t_2)
            elif cfg.type == "min":
                t = t_1
                t_next = torch.minimum(t_1, t_2)
            elif cfg.type == "r_in_t_range":
                t = t_1
                t_next = t_2 * t_1
        else:
            raise NotImplementedError(f"Unknown meanflow distribution type: {cfg.type}")
        return t, t_next

    def sample_traj_params(self, batch_size, cur_step, device):
        ratio_fm = self.get_ratio(self.cfg.ratio_fm, cur_step)
        alpha = self.get_ratio(self.cfg.alpha, cur_step)
        batch_size_fm = int(batch_size * ratio_fm)
        batch_size_mf = batch_size - batch_size_fm

        t_fm = t_next_fm = self.sample_timestep(self.cfg.time_sampling_fm, cur_step, batch_size_fm, device=device)
        dt_fm = torch.zeros_like(t_next_fm)
        t_mf, t_next_mf = self.sample_timesteps_mf(self.cfg.distrib_t_t_next_mf, cur_step, batch_size_mf, device)
        dt_mf = alpha * (t_mf - t_next_mf)

        t = torch.cat([t_fm, t_mf], dim=0)
        t_next = torch.cat([t_next_fm, t_next_mf], dim=0)
        dt = torch.cat([dt_fm, dt_mf], dim=0)

        return t.view(batch_size, 1, 1, 1, 1), t_next.view(batch_size, 1, 1, 1, 1), dt.view(batch_size, 1, 1, 1, 1), alpha

    @torch.no_grad()
    def _compute_velocity_cfg(self, velocity, x_t, t, cond, augment_labels, net, batch_size):
        # Create classifier free guidance mask for t: True where self.cfg.cfg_params.t_min < t < self.cfg.cfg_params.t_max
        t_flat = t.view(batch_size, -1)
        omega = self.cfg.cfg_params.omega
        kappa = self.cfg.cfg_params.kappa
        mask = (t_flat > self.cfg.cfg_params.t_min) & (t_flat < self.cfg.cfg_params.t_max)
        cfg_mask_idx = mask.view(batch_size).bool()  # [b]
        # Only apply classifier free guidance t is in the interval and when scale != 1.0
        velocity_cfg = velocity.clone() # [b, t, c, h, w]

        # Drop labels, with probability self.cfg.label_dropout, set cond.label[...] to zero vector
        if self.cfg.model.label_dropout > 0.0:
            label_drop_mask_idx = (torch.rand(cond.label.shape[0], device=cond.label.device) < self.cfg.model.label_dropout)
            drop_mask = cfg_mask_idx & label_drop_mask_idx
            cond.label[drop_mask] = torch.zeros_like(cond.label[drop_mask])

        if 1 - omega - kappa != 0.0:
            videos_u_t_t_uncond = net(
                x_t, sigma_next=t, sigma=t,
                cond=None, augment_labels=augment_labels, return_extra_output=False
            ) # [b, t, c, h, w]
        else:
            videos_u_t_t_uncond = torch.zeros_like(x_t) # [b, t, c, h, w]

        if kappa != 0:
            videos_u_t_t_cond = net(
                x_t, sigma_next=t, sigma=t,
                cond=cond, augment_labels=augment_labels, return_extra_output=False
            ) # [b, t, c, h, w]
        else:
            videos_u_t_t_cond = torch.zeros_like(x_t) # [b, t, c, h, w]

        guided = omega * velocity + kappa * videos_u_t_t_cond + (1 - omega - kappa) * videos_u_t_t_uncond # [b, t, c, h, w]
        velocity_cfg[cfg_mask_idx] = guided[cfg_mask_idx] # [b, t, c, h, w]
        return velocity_cfg

    @torch.no_grad()
    def _compute_mean_velocity_c(self, x_t, t_next, t, velocity_cfg, cond, augment_labels, net):
        if x_t.shape[0] == 0:
            return torch.empty((0, *velocity_cfg.shape[1:]), device=velocity_cfg.device)
        t = t.flatten() # [b]
        t_next = t_next.flatten() # [b]

        mask_mf = ~torch.isclose(t_next, t) # [b]
        batch_size_mf = mask_mf.sum().item()
        mean_velocity = velocity_cfg.clone() # [b, t, c, h, w]

        def wrap_net(x_t, t_next, t):
            return net(x_t, sigma_next=t_next, sigma=t, cond=cond[mask_mf], augment_labels=augment_labels, return_extra_output=False)
        x_t_mf, t_mf, t_next_mf, velocity_cfg_mf = x_t[mask_mf], t[mask_mf], t_next[mask_mf], velocity_cfg[mask_mf]

        if batch_size_mf == 0:
            return mean_velocity
        
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            _, videos_dudt_mf = jvp(
                wrap_net,
                (x_t_mf, t_next_mf, t_mf),
                (velocity_cfg_mf, torch.zeros_like(t_next_mf), torch.ones_like(t_mf)),
            )
        mean_velocity_mf = (velocity_cfg_mf -  (t_mf - t_next_mf).view(batch_size_mf, 1, 1, 1, 1) * videos_dudt_mf) # [b_mf, t, c, h, w]
        mean_velocity[mask_mf] = mean_velocity_mf
        return mean_velocity # [b, t, c, h, w]

    @torch.no_grad()
    def _compute_mean_velocity_d(self, x_t, velocity_cfg, t_next, t, dt, cond, augment_labels, net):
        if x_t.shape[0] == 0:
            return torch.empty((0, *velocity_cfg.shape[1:]), device=velocity_cfg.device)
        x_t_minus_dt = x_t - dt * velocity_cfg # [b, t, c, h, w]

        if torch.isclose(1 - dt/(t - t_next), torch.zeros_like(t)).all():
            mean_velocity_next = torch.zeros_like(x_t_minus_dt) # [b, t, c, h, w]
        else:
            mean_velocity_next, _ = net(x_t_minus_dt, sigma_next = t_next, sigma = t - dt, cond=cond, augment_labels=augment_labels, return_extra_output=True) # [b, t, c, h, w]

        mean_velocity = (dt * velocity_cfg + (t - dt - t_next) * mean_velocity_next) / (t - t_next) # [b, t, c, h, w]
        mean_velocity = torch.clip(mean_velocity, min=-self.cfg.clamp_utgt, max=self.cfg.clamp_utgt) # [b, t, c, h, w]
        return mean_velocity # [b, t, c, h, w]


    def calculate_per_delta(self, batch_size, t, t_next, loss):
        dt_stats = {}
        if self.cfg.loss_per_delta:
            dt_flat = (t - t_next).flatten()  # [b]

            perscent = (20, 40, 60, 80, 100)
            qs = [1.0 - math.sqrt(1.0 - p / 100.0) for p in perscent]
            bins = [(lo, hi, f"q{p:03d}") for (lo, hi, p) in zip([1e-6] + qs[:-1], qs, perscent)]
            
            metrics = {'total': loss}
            
            for name, vec in metrics.items():
                mask_fm = (dt_flat == 0)
                batch_size_fm = int(mask_fm.sum().item())
                
                if batch_size_fm > 0:
                    dt_stats[f"{name}_dt_0"] = mask_fm.to(vec.dtype) * vec * (batch_size / batch_size_fm)
                else:
                    dt_stats[f"{name}_dt_0"] = mask_fm.to(vec.dtype) * vec
                
            # dt bins (discrete)
            for lo, hi, tag in bins:
                mask = (dt_flat > lo) & (dt_flat <= hi)
                batch_size_mask = int(mask.sum().item())

                for name, vec in metrics.items():
                    if batch_size_mask == 0:
                        dt_stats[f"{name}_dt_{tag}"] = mask * vec
                    else:
                        scale = batch_size / batch_size_mask
                        mask = mask.to(vec.dtype)
                        dt_stats[f"{name}_dt_{tag}"] = mask * vec * scale
        
        return dt_stats

    def forward(self, net, x, cond, augment_pipe=None, phase=LossPhase.Gen, cur_step=None, force_t_val: Optional[float]=None, force_t_r_dt_val=None, align_ctx=None, compute_decomposed_loss=False) -> EasyDict[str, torch.Tensor]:
        _ = force_t_val ## Unused variable
        assert x.ndim == 5, f"Expected x to have 5 dimensions: [b, t, c, h, w], got {x.shape} instead."
        batch_size = len(x) # [1]
        x_aug, augment_labels = maybe_augment_videos(x, augment_pipe) # [b, t, c, h, w], [b, augment_dim] or None
        assert augment_labels is None, "Augment labels are not supported for joint training"
        x_aug = self._maybe_encode(net, x_aug, cond) # [b, t, c, h, w]

        # Sample t, t_next and alpha, dt = (t - t_next) * alpha
        t, t_next, dt, alpha = self.sample_traj_params(x_aug.shape[0], cur_step, x_aug.device) # [b, 1, 1, 1, 1], [b, 1, 1, 1, 1]
        noise_unscaled = torch.randn_like(x_aug) # [b, t, c, h, w]
        noise_scaled = noise_unscaled * t # [b, t, c, h, w]
        x_t = self.apply_noise(x_aug, noise_scaled, t, net) # [b, t, c, h, w]
        velocity = noise_unscaled - x_aug # [b, t, c, h, w]

        # Apply cfg training
        velocity_cfg = self._compute_velocity_cfg(velocity=velocity, x_t=x_t, t=t, cond=cond, augment_labels=augment_labels, net=net, batch_size=batch_size) # [b, t, c, h, w]

        # Split batch for continuous (alpha == 1 or r == t) and discrete training (0 < alpha <= 1)
        mask_c = (dt == 0).flatten() # [b]
        mask_d = ~mask_c # [b]
        batch_size_c, batch_size_d = mask_c.sum().item(), mask_d.sum().item() # [1], [1]
        velocity_cfg_c, x_t_c, t_c, t_next_c, cond_c, = velocity_cfg[mask_c], x_t[mask_c], t[mask_c], t_next[mask_c], cond[mask_c]
        x_t_d, velocity_cfg_d, t_d, t_next_d, dt_d, cond_d, = x_t[mask_d], velocity_cfg[mask_d], t[mask_d], t_next[mask_d], dt[mask_d], cond[mask_d]

        # Calculate u_tgt when alpha == 1 or r == t
        mean_velocity_c = self._compute_mean_velocity_c(x_t_c, t_next_c, t_c, velocity_cfg_c, cond_c, augment_labels, net) # [b_c, t, c, h, w]

        # Calculate u_tgt when 0 < alpha <= 1
        mean_velocity_d = self._compute_mean_velocity_d(x_t_d, velocity_cfg_d, t_next_d, t_d, dt_d, cond_d, augment_labels, net) # [b_d, t, c, h, w]

        mean_velocity = torch.cat([mean_velocity_c, mean_velocity_d], dim=0) # [b, t, c, h, w]

        pred_mean_velocity, ctx = net(
            x_t,
            sigma_next=t_next,
            sigma=t,
            cond=cond,
            augment_labels=augment_labels,
            return_extra_output=True
        ) # [b, t, c, h, w]

        ## Adaptive loss
        loss_unscaled = ((pred_mean_velocity - mean_velocity) ** 2).flatten(1).mean(1) # [b]
        weight_c = torch.ones(batch_size_c, device=velocity.device) # [b_c]
        weight_d = torch.ones(batch_size_d, device=velocity.device) * alpha # [b_d]
        weight = torch.cat([weight_c, weight_d], dim=0) # [b]
        if self.cfg.use_adaptive_loss:
            weight = weight / (loss_unscaled.detach() + self.cfg.adaptive_loss_weight_eps) # [b]
        loss = weight * loss_unscaled # [b]

        ## Compute trajectory flow matching loss
        loss_tfm = ((pred_mean_velocity - velocity_cfg) ** 2).flatten(1).mean(1) # [b]

        ## Compute consistency flow matching loss
        loss_tcc = (2 * (velocity_cfg - mean_velocity) * pred_mean_velocity).flatten(1).mean(1) # [b]
        loss_tfm_plus_tcc = loss_tfm + loss_tcc # [b]
        
        ## Loss per delta
        dt_stats = self.calculate_per_delta(batch_size, t, t_next, loss)

        loss_dict = misc.filter_nones(EasyDict(
            total                      = loss, # [b]
            trajectory_FM              = loss_tfm, # [b]
            trajectory_consistency     = loss_tcc, # [b]
            trajectory_sum             = loss_tfm_plus_tcc, # [b]
            **dt_stats,
            ))
        return loss_dict

class ImprovedMeanFlow(AlphaFlowLoss):
    def sample_timesteps_mf(self, cfg, cur_step, batch_size, device):
        if cfg.time_sampling_mf_t_next.timestep_distrib_type == "fixed_delta":
            fixed_delta = cfg.time_sampling_mf_t_next.delta_size
            t = self.sample_timestep(cfg.time_sampling_mf_t, cur_step, batch_size, device=device)
            t_next = torch.clamp(t - fixed_delta, min = 0.0)
            return t, t_next
        
        if cfg.type == "truncated":
            t = self.sample_timestep(cfg.time_sampling_mf_t, cur_step, batch_size, device=device)
            t_next = self.sample_timestep(cfg.time_sampling_mf_t_next, cur_step, batch_size, device=device, upper_truncated = t)
        elif cfg.type in ["minmax", "min", "r_in_t_range"]:
            t_1 = self.sample_timestep(cfg.time_sampling_mf_t, cur_step, batch_size, device=device)
            t_2 = self.sample_timestep(cfg.time_sampling_mf_t_next, cur_step, batch_size, device=device)
            if cfg.type == "minmax":
                t = torch.maximum(t_1, t_2)
                t_next = torch.minimum(t_1, t_2)
            elif cfg.type == "min":
                t = t_1
                t_next = torch.minimum(t_1, t_2)
            elif cfg.type == "r_in_t_range":
                t = t_1
                t_next = t_2 * t_1
        elif cfg.type == "incremental_curriculum":
            curriculum_cfg = cfg.curriculum
            s = (cur_step + 0.5) / float(curriculum_cfg.total_step)
            alpha = torch.tanh(torch.tensor(curriculum_cfg.steepness * (2.0 * s - 1.0), device=device, dtype=torch.float32))
            t, t_next = self.get_incremental_curriculum(curriculum_cfg, alpha, batch_size, device, cur_step)
        else:
            raise NotImplementedError(f"Unknown meanflow distribution type: {cfg.type}")
        return t, t_next
    
    def _compute_inst_velocity(self, x_t, t, t_next, cond, augment_labels, net):
        """Return pred_mean + (t - t_next) * dudt, where dudt is computed via JVP.

        Tangent vector for the JVP: net(x_t, σ_next=t, σ=t) with stop_grad.
        Only MF samples (t != t_next) get a non-zero dudt contribution;
        FM samples (t == t_next) leave dudt = 0 and the term vanishes naturally.

        Args:
            x_t, t, t_next, cond: continuous-sample subset  [b, ...]
            net:     network (used with stop_grad for the tangent)

        Returns:
            pred_mean          = net(x_t, σ_next=t_next, σ=t)      [b, T, C, H, W]
            pred_inst_velocity = pred_mean + (t - t_next) * dudt  [b, T, C, H, W]
        """
        # Prediction: net(x_t, sigma_next=t_next, sigma=t) with gradients.
        pred_mean, _ = net(
            x_t, sigma_next=t_next, sigma=t,
            cond=cond, augment_labels=augment_labels, return_extra_output=True,
        )  # [b, T, C, H, W]

        mask_mf = ~torch.isclose(t_next.flatten(), t.flatten())  # [b]
        dudt = torch.zeros(pred_mean.shape, device=pred_mean.device, dtype=pred_mean.dtype)
        
        if mask_mf.any():
            x_t_mf    = x_t[mask_mf]
            t_mf      = t[mask_mf]
            t_next_mf = t_next[mask_mf]
            cond_mf   = cond[mask_mf]

            with torch.no_grad():
                # Tangent: net(x_t, σ_next=t, σ=t) — stop_grad
                pred_inst = net(
                    x_t_mf, sigma_next=t_mf, sigma=t_mf,
                    cond=cond_mf, augment_labels=augment_labels, return_extra_output=False,
                )  # [b_mf, T, C, H, W]

                def wrap_net(x_, tn_, t_):
                    return net(x_, sigma_next=tn_, sigma=t_,
                               cond=cond_mf, augment_labels=augment_labels, return_extra_output=False)

                with sdpa_kernel(backends=[SDPBackend.MATH]):
                    _, dudt_mf = jvp(
                        wrap_net,
                        (x_t_mf, t_next_mf, t_mf),
                        (pred_inst, torch.zeros_like(t_next_mf), torch.ones_like(t_mf)),
                    )  # [b_mf, T, C, H, W]

            dudt[mask_mf] = dudt_mf  # already no_grad

        return pred_mean, pred_mean + (t - t_next) * dudt  # [b, T, C, H, W], [b, T, C, H, W]

    def forward(self, net, x, cond, augment_pipe=None, phase=LossPhase.Gen, cur_step=None, force_t_val: Optional[float]=None, force_t_r_dt_val=None, align_ctx=None, compute_decomposed_loss=False) -> EasyDict[str, torch.Tensor]:
        assert x.ndim == 5, f"Expected [b, T, C, H, W], got {x.shape}."
        batch_size = len(x)
        x_aug, augment_labels = maybe_augment_videos(x, augment_pipe)
        assert augment_labels is None, "Augment labels are not supported."
        x_aug = self._maybe_encode(net, x_aug, cond)

        t, t_next, dt, alpha = self.sample_traj_params(batch_size, cur_step, x_aug.device)
        noise_unscaled = torch.randn_like(x_aug)
        x_t = self.apply_noise(x_aug, noise_unscaled * t, t, net)
        velocity = noise_unscaled - x_aug
        velocity_cfg = self._compute_velocity_cfg(
            velocity=velocity, x_t=x_t, t=t, cond=cond,
            augment_labels=augment_labels, net=net, batch_size=batch_size,
        )  # [b, T, C, H, W]

        # pred_mean + (t - t_next) * dudt,  dudt via JVP with stop_grad tangent.
        pred_mean, pred_inst_velocity = self._compute_inst_velocity(
            x_t, t, t_next, cond, augment_labels, net,
        )  # [b, T, C, H, W], [b, T, C, H, W]

        # Loss: (pred_mean + (t - t_next) * dudt - v_cfg)^2
        loss_unscaled = ((pred_inst_velocity - velocity_cfg) ** 2).flatten(1).mean(1)  # [b]

        # Adaptive weighting (same eps as AlphaFlowLoss).
        weight = torch.ones(batch_size, device=x_t.device)  # [b]
        if self.cfg.use_adaptive_loss:
            weight = weight / (loss_unscaled.detach() + self.cfg.adaptive_loss_weight_eps)
        loss = weight * loss_unscaled  # [b]

        ## Trajectory FM loss and consistency loss (same formula as AlphaFlowLoss)
        loss_tfm = ((pred_inst_velocity - velocity_cfg) ** 2).flatten(1).mean(1)  # [b]
        loss_tcc = (2 * (velocity_cfg - pred_mean) * pred_inst_velocity).flatten(1).mean(1)  # [b]
        loss_tfm_plus_tcc = loss_tfm + loss_tcc  # [b]

        return EasyDict(
            total                  = loss,              # [b]
            trajectory_FM          = loss_tfm,          # [b]
            trajectory_consistency = loss_tcc,          # [b]
            trajectory_sum         = loss_tfm_plus_tcc, # [b]
        )


class MeanFlowAnalysis_WithTeacher(AlphaFlowLoss):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.teacher = None if self.cfg.teacher.ckpt is None else load_snapshot(self.cfg.teacher.ckpt, verbose=False)[0].train().requires_grad_(False)

    @torch.no_grad()
    def _compute_mean_velocity_teacher(self, x_t, t, t_next, cond, augment_labels, step_size: float) -> torch.Tensor:
        """Multi-step Euler integration of teacher ODE from t to t_next with a fixed step size.

        Each sample in the batch may have a different interval (t - t_next), so the number of
        iterations is determined by the largest interval: ceil(max_dt / step_size).
        Per-sample step h = min(step_size, t_cur - t_next) handles the last partial step and
        samples that have already reached t_next naturally (h -> 0, x_cur stops moving).

        Returns:
            mean_velocity_teacher = (x_t - x_{t_next}) / (t - t_next)  [b, T, C, H, W]
        """
        dt_total = t - t_next  # [b, 1, 1, 1, 1], positive since t > t_next
        max_dt = dt_total.max().item()
        if max_dt <= 0:
            return torch.zeros_like(x_t)

        num_iters = math.ceil(max_dt / step_size)
        x_cur = x_t.clone()
        t_cur = t.clone()  # [b, 1, 1, 1, 1]

        for _ in range(num_iters):
            # Per-sample h: full step_size, shrunk to remaining distance at the last step.
            h = (t_cur - t_next).clamp(min=0.0, max=step_size)  # [b, 1, 1, 1, 1]
            if h.max().item() <= 0:
                break
            t_next_step = (t_cur - h).clamp(min=0.0)  # [b, 1, 1, 1, 1]
            v = self.teacher(
                x_cur, sigma_next=t_next_step, sigma=t_cur,
                cond=cond, augment_labels=augment_labels, return_extra_output=False,
            )  # [b, T, C, H, W]
            x_cur = x_cur - h * v  # samples with h=0 do not move
            t_cur = t_next_step

        safe_dt = dt_total.clamp(min=1e-8)
        return (x_t - x_cur) / safe_dt  # [b, T, C, H, W]

    @torch.no_grad()
    def forward(self, net, x, cond, augment_pipe=None, phase=LossPhase.Gen, cur_step=None, force_t_val: Optional[float]=None, force_t_r_dt_val=None, align_ctx=None, compute_decomposed_loss=False) -> EasyDict[str, torch.Tensor]:
        assert self.teacher is not None, "Teacher model is not loaded. Set loss.teacher.ckpt.snapshot_path."
        assert x.ndim == 5, f"Expected x to have 5 dimensions [b, T, C, H, W], got {x.shape}."
        batch_size = len(x)
        x_aug, augment_labels = maybe_augment_videos(x, augment_pipe)
        assert augment_labels is None, "Augment labels are not supported."
        x_aug = self._maybe_encode(net, x_aug, cond)  # [b, T, C, H, W]

        # Sample (t, t_next) using the same sampler as AlphaFlowLoss.
        t, t_next, dt, alpha = self.sample_traj_params(batch_size, cur_step, x_aug.device)
        noise_unscaled = torch.randn_like(x_aug)
        x_t = self.apply_noise(x_aug, noise_unscaled * t, t, net)  # [b, T, C, H, W]
        velocity = noise_unscaled - x_aug                           # [b, T, C, H, W]
        velocity_cfg = self._compute_velocity_cfg(
            velocity=velocity, x_t=x_t, t=t, cond=cond,
            augment_labels=augment_labels, net=net, batch_size=batch_size,
        )  # [b, T, C, H, W]

        # Focus on mean-flow samples where t != t_next (t == t_next cases carry no interval).
        mask_mf = ~torch.isclose(t_next.flatten(), t.flatten())  # [b]
        x_t_mf      = x_t[mask_mf]
        t_mf        = t[mask_mf]
        t_next_mf   = t_next[mask_mf]
        v_cfg_mf    = velocity_cfg[mask_mf]
        cond_mf     = cond[mask_mf]

        # --- Method 1: JVP-based mean velocity (existing AlphaFlow approach) ---
        mean_vel_jvp = self._compute_mean_velocity_c(
            x_t_mf, t_next_mf, t_mf, v_cfg_mf, cond_mf, augment_labels, net,
        )  # [b_mf, T, C, H, W]

        # --- Method 2: teacher multi-step Euler integration ---
        step_size = getattr(self.cfg, 'step_size', 0.1)
        mean_vel_teacher = self._compute_mean_velocity_teacher(
            x_t_mf, t_mf, t_next_mf, cond_mf, augment_labels, step_size,
        )  # [b_mf, T, C, H, W]

        # --- Difference metrics ---
        diff = mean_vel_jvp - mean_vel_teacher  # [b_mf, T, C, H, W]
        diff_mse = (diff ** 2).flatten(1).mean(1)                        # [b_mf]
        diff_mae = diff.abs().flatten(1).mean(1)                         # [b_mf]
        diff_cos = F.cosine_similarity(
            mean_vel_jvp.flatten(1), mean_vel_teacher.flatten(1), dim=1,
        )  # [b_mf]

        # Pad back to full batch (zeros for FM samples where t == t_next).
        def _pad(x_mf: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(batch_size, device=x_mf.device, dtype=x_mf.dtype)
            out[mask_mf] = x_mf
            return out

        return EasyDict(
            mean_velocity_diff_mse  = _pad(diff_mse),  # [b]
            mean_velocity_diff_mae  = _pad(diff_mae),  # [b]
            mean_velocity_cosine_sim= _pad(diff_cos),  # [b]
        )

class MeanFlowAnalysis(AlphaFlowLoss):
    def get_incremental_curriculum(self, cfg, alpha, batch_size, device, cur_step):
        t = torch.empty((batch_size,), device=device, dtype=torch.float32)
        t_next = torch.empty((batch_size,), device=device, dtype=torch.float32)

        c = 0.5
        max_rounds = 50
        oversample = 4
        kappa = cfg.kappa
        beta = cfg.beta
        
        filled = 0
        for _ in range(max_rounds):
            need = batch_size - filled
            if need <= 0:
                break

            m = max(oversample * need, 32)

            u1 = torch.rand((m,), device=device, dtype=torch.float32)
            u2 = torch.rand((m,), device=device, dtype=torch.float32)

            hi = torch.maximum(u1, u2)
            lo = torch.minimum(u1, u2)
            delta = hi - lo

            u = 2.0 * delta - delta * delta
            h = torch.tanh(beta * (u - 0.5))
            a = c + kappa * alpha * h
            a = a.clamp(0.0, 1.0)

            keep = torch.rand((m,), device=device, dtype=torch.float32) < a
            idx = torch.nonzero(keep, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue

            take = min(idx.numel(), need)
            sel = idx[:take]

            t[filled:filled + take] = hi[sel]
            t_next[filled:filled + take] = lo[sel]
            filled += take

        if filled < batch_size:
            need = batch_size - filled
            u1 = torch.rand((need,), device=device, dtype=torch.float32)
            u2 = torch.rand((need,), device=device, dtype=torch.float32)
            t[filled:] = torch.maximum(u1, u2)
            t_next[filled:] = torch.minimum(u1, u2)
            print("not filled!!")
            
        if cur_step % 100 == 0 :
            delta = (t - t_next).detach()
            qs = torch.quantile(
                delta,
                torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], device=delta.device, dtype=delta.dtype),
            ).cpu().tolist()
            print(
                f"alpha: {alpha:.4f} / delta percentiles | 10%: {qs[0]:.4f}, 30%: {qs[1]:.4f}, 50%: {qs[2]:.4f}, 70%: {qs[3]:.4f}, 90%: {qs[4]:.4f}"
            )
        return t, t_next
    
    def sample_timesteps_mf(self, cfg, cur_step, batch_size, device):
        if cfg.time_sampling_mf_t_next.timestep_distrib_type == "fixed_delta":
            fixed_delta = cfg.time_sampling_mf_t_next.delta_size
            t = self.sample_timestep(cfg.time_sampling_mf_t, cur_step, batch_size, device=device)
            t_next = torch.clamp(t - fixed_delta, min = 0.0)
            return t, t_next
        
        if cfg.type == "truncated":
            t = self.sample_timestep(cfg.time_sampling_mf_t, cur_step, batch_size, device=device)
            t_next = self.sample_timestep(cfg.time_sampling_mf_t_next, cur_step, batch_size, device=device, upper_truncated = t)
        elif cfg.type in ["minmax", "min", "r_in_t_range"]:
            t_1 = self.sample_timestep(cfg.time_sampling_mf_t, cur_step, batch_size, device=device)
            t_2 = self.sample_timestep(cfg.time_sampling_mf_t_next, cur_step, batch_size, device=device)
            if cfg.type == "minmax":
                t = torch.maximum(t_1, t_2)
                t_next = torch.minimum(t_1, t_2)
            elif cfg.type == "min":
                t = t_1
                t_next = torch.minimum(t_1, t_2)
            elif cfg.type == "r_in_t_range":
                t = t_1
                t_next = t_2 * t_1
        elif cfg.type == "incremental_curriculum":
            curriculum_cfg = cfg.curriculum
            s = (cur_step + 0.5) / float(curriculum_cfg.total_step)
            alpha = torch.tanh(torch.tensor(curriculum_cfg.steepness * (2.0 * s - 1.0), device=device, dtype=torch.float32))
            t, t_next = self.get_incremental_curriculum(curriculum_cfg, alpha, batch_size, device, cur_step)
        else:
            raise NotImplementedError(f"Unknown meanflow distribution type: {cfg.type}")
        return t, t_next

# MeanFlow with incremental-Consistency Curriculum
class MeanFlow_iC2(AlphaFlowLoss):
    """MeanFlow with incremental-Consistency Curriculum (iC2).

    Maintains a per-t-bin maximum delta (Δ = t − r) table.  The table is
    updated online every `consistency_curriculum.update_freq` steps by checking whether
    extending the current max delta by `consistency_curriculum.step_size` still satisfies:

        ||f(anchor) + f(step) − f(anchor + step)||² < consistency_threshold

    where
        f(anchor)       = u(x_t,       t → r_anchor)   [net mean velocity]
        f(step)         = u(x_r_anchor, r_anchor → r_step)
        f(anchor + step)= u(x_t,       t → r_step)

    Required cfg fields (under cfg.consistency_curriculum):
        num_bins              : int   – number of t-bins covering [0, 1]
        init_max_delta        : float – initial max Δ for every bin
        step_size             : float – extension step tried each update
        consistency_threshold : float – MSE threshold to accept extension
        update_freq           : int   – update every this many iterations
        probe_batch_size      : int   – samples used per-bin consistency probe
    """

    def __init__(self, cfg: EasyDict):
        super().__init__(cfg)
        consistency_curriculum = cfg.consistency_curriculum
        # Per-bin max Δ — saved/loaded with checkpoint via register_buffer.
        self.register_buffer(
            'max_delta_per_bin',
            torch.full((consistency_curriculum.num_bins,), float(consistency_curriculum.init_max_delta), dtype=torch.float32),
        )
        self._last_update_step  = -1
        self._last_analysis_step = -1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bin_idx(self, t: torch.Tensor) -> torch.Tensor:
        """Map t ∈ [0, 1] → bin index ∈ [0, num_bins-1]."""
        n = self.cfg.consistency_curriculum.num_bins
        return (t.float() * n).long().clamp(0, n - 1)

    # ------------------------------------------------------------------
    # Timestep sampling: constrain delta by per-bin max
    # ------------------------------------------------------------------

    def sample_timesteps_mf(self, cfg, cur_step, batch_size, device):
        use_curriculum_sampling = getattr(self.cfg.consistency_curriculum, 'use_curriculum_sampling', True)
        if not use_curriculum_sampling:
            return super().sample_timesteps_mf(cfg, cur_step, batch_size, device)
        bin_mode = getattr(self.cfg.consistency_curriculum, 'bin_mode', 't')
        if bin_mode == 't_next':
            # Bin by t_next: sample t_next first, then derive t = t_next + delta
            t_next    = self.sample_timestep(cfg.time_sampling_mf_t_next, cur_step, batch_size, device=device)  # [b]
            bin_idx   = self._bin_idx(t_next)                    # [b]
            max_delta = self.max_delta_per_bin[bin_idx]          # [b]  per-sample cap
            delta     = torch.rand_like(t_next) * max_delta      # [b]  uniform in [0, max_delta]
            t         = (t_next + delta).clamp(max=1.0)          # [b]
        else:  # bin_mode == 't'
            t         = self.sample_timestep(cfg.time_sampling_mf_t, cur_step, batch_size, device=device)  # [b]
            bin_idx   = self._bin_idx(t)                         # [b]
            max_delta = self.max_delta_per_bin[bin_idx]          # [b]  per-sample cap
            delta     = torch.rand_like(t) * max_delta           # [b]  uniform in [0, max_delta]
            t_next    = (t - delta).clamp(min=0.0)               # [b]
        return t, t_next

    # ------------------------------------------------------------------
    # Online max-delta update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _check_and_extend_bin(
        self,
        bin_idx: int,
        net,
        x_t: torch.Tensor,   # [b, T, C, H, W]  noisy sample at t_center
        t_val: float,
        cond,
        augment_labels,
    ) -> float:
        """Probe bin_idx and extend max_delta by step_size if consistency holds.

        Consistency (per user spec):
            error = ||f(anchor) + f(step) − f(anchor+step)||²  < threshold
        """
        consistency_curriculum = self.cfg.consistency_curriculum
        device = x_t.device
        current_max = self.max_delta_per_bin[bin_idx].item()
        step        = float(consistency_curriculum.step_size)
        r_anchor    = t_val - current_max
        r_step      = r_anchor - step

        if r_anchor < 0.0 or r_step < 0.0:
            return float('nan')  # already at t=0 boundary; cannot extend

        b = x_t.shape[0]
        bc = (b, 1, 1, 1, 1)
        t_t      = torch.full(bc, t_val,    device=device, dtype=torch.float32)
        t_anchor = torch.full(bc, r_anchor, device=device, dtype=torch.float32)
        t_step   = torch.full(bc, r_step,   device=device, dtype=torch.float32)

        # f(anchor): mean velocity u(x_t, t → r_anchor)
        f_anchor = net(
            x_t, sigma_next=t_anchor, sigma=t_t,
            cond=cond, augment_labels=augment_labels, return_extra_output=False,
        )  # [b, T, C, H, W]

        # x at r_anchor via f(anchor)
        x_at_anchor = x_t - current_max * f_anchor  # [b, T, C, H, W]

        # f(step): mean velocity u(x_at_anchor, r_anchor → r_step)
        f_step = net(
            x_at_anchor, sigma_next=t_step, sigma=t_anchor,
            cond=cond, augment_labels=augment_labels, return_extra_output=False,
        )  # [b, T, C, H, W]

        # f(anchor+step): mean velocity u(x_t, t → r_step)
        f_full = net(
            x_t, sigma_next=t_step, sigma=t_t,
            cond=cond, augment_labels=augment_labels, return_extra_output=False,
        )  # [b, T, C, H, W]

        error_space = getattr(consistency_curriculum, "error_space", "x_space")
        if error_space == "x_space":
            x_t_next_1 = x_t - (current_max + step) * f_full
            x_t_next_2 = x_t - current_max * f_anchor - step * f_step
            diff  = (x_t_next_2 - x_t_next_1).flatten(1)
            denom = x_t_next_2.flatten(1).pow(2).mean(1).sqrt() + 1e-12
            error = (diff.pow(2).mean(1).sqrt() / denom).mean().item()
        else:  # v_space
            residual_sq = (f_anchor + f_step - f_full).pow(2).flatten(1).mean(1).mean()
            f_full_sq   = f_full.pow(2).flatten(1).mean(1).mean().clamp(min=1e-8)
            error = (residual_sq / f_full_sq).item()
        return error

    @torch.no_grad()
    def _check_and_extend_bin_t_next(
        self,
        bin_idx: int,
        net,
        x_t: torch.Tensor,    # [b, T, C, H, W]  noisy sample at t_val = t_next + current_delta + step
        t_next_val: float,    # bin center (t_next)
        t_val: float,         # = t_next_val + current_delta + step  (starting point)
        cond,
        augment_labels,
    ) -> float:
        """t_next-bin mode counterpart to _check_and_extend_bin.

        Given t_next is fixed (bin center), check if extending delta by step_size is
        consistent.  The total interval is (step + delta), so the two-hop order is:
            hop-1 (size step)  : t_val    → r_anchor  (= t_next + current_delta)
            hop-2 (size delta) : r_anchor → t_next_val
        compared against the direct hop of size (step + delta).
        """
        consistency_curriculum = self.cfg.consistency_curriculum
        device        = x_t.device
        current_delta = self.max_delta_per_bin[bin_idx].item()
        step          = float(consistency_curriculum.step_size)
        r_anchor      = t_next_val + current_delta   # = t_val - step
        r_step        = t_next_val                   # end = t_next (bin center)

        # t_val > 1.0 guard is handled before calling this function
        b  = x_t.shape[0]
        bc = (b, 1, 1, 1, 1)
        t_t      = torch.full(bc, t_val,    device=device, dtype=torch.float32)
        t_anchor = torch.full(bc, r_anchor, device=device, dtype=torch.float32)
        t_end    = torch.full(bc, r_step,   device=device, dtype=torch.float32)

        # f_anchor: hop of size `step` from t_val → r_anchor
        f_anchor = net(
            x_t, sigma_next=t_anchor, sigma=t_t,
            cond=cond, augment_labels=augment_labels, return_extra_output=False,
        )
        x_at_anchor = x_t - step * f_anchor

        # f_step: hop of size `current_delta` from r_anchor → t_next
        f_step = net(
            x_at_anchor, sigma_next=t_end, sigma=t_anchor,
            cond=cond, augment_labels=augment_labels, return_extra_output=False,
        )

        # f_full: direct hop of size (step + current_delta) from t_val → t_next
        f_full = net(
            x_t, sigma_next=t_end, sigma=t_t,
            cond=cond, augment_labels=augment_labels, return_extra_output=False,
        )

        error_space = getattr(consistency_curriculum, "error_space", "x_space")
        if error_space == "x_space":
            x_t_next_1 = x_t - (step + current_delta) * f_full
            x_t_next_2 = x_t - step * f_anchor - current_delta * f_step
            diff  = (x_t_next_2 - x_t_next_1).flatten(1)
            denom = x_t_next_2.flatten(1).pow(2).mean(1).sqrt() + 1e-12
            error = (diff.pow(2).mean(1).sqrt() / denom).mean().item()
        else:  # v_space
            residual_sq = (f_anchor + f_step - f_full).pow(2).flatten(1).mean(1).mean()
            f_full_sq   = f_full.pow(2).flatten(1).mean(1).mean().clamp(min=1e-8)
            error = (residual_sq / f_full_sq).item()
        return error

    @torch.no_grad()
    def update_max_delta(self, net, x_encoded: torch.Tensor, cond, augment_labels, cur_step: int):
        """Every `update_freq` steps, probe all bins and extend where consistent."""
        consistency_curriculum = self.cfg.consistency_curriculum
        if (cur_step % consistency_curriculum.update_freq != 0) or (cur_step == 0):
            return
        start = getattr(consistency_curriculum, 'start_max_delta_iter', None)
        end   = getattr(consistency_curriculum, 'end_max_delta_iter', None)
        if start is not None and cur_step < start:
            return
        if end is not None and cur_step >= end:
            return
        if self._last_update_step == cur_step:
            return
        self._last_update_step = cur_step

        device  = x_encoded.device
        probe_n = min(int(consistency_curriculum.probe_batch_size), x_encoded.shape[0])
        x_probe = x_encoded[:probe_n]
        cond_probe = cond[:probe_n]

        # Each rank computes errors independently (no buffer writes yet).
        num_bins = int(consistency_curriculum.num_bins)
        bin_mode = getattr(consistency_curriculum, 'bin_mode', 't')
        step     = float(consistency_curriculum.step_size)
        bin_errors = []
        for i in range(num_bins):
            if bin_mode == 't_next':
                t_next_val    = (i + 1) / num_bins
                current_delta = self.max_delta_per_bin[i].item()
                t_val         = t_next_val + current_delta + step
                if t_val > 1.0:
                    bin_errors.append(float('nan'))
                    continue
                noise = torch.randn_like(x_probe)
                x_t   = x_probe + noise * t_val
                err   = self._check_and_extend_bin_t_next(i, net, x_t, t_next_val, t_val, cond_probe, augment_labels)
            else:  # bin_mode == 't'
                t_boundary = (i + 1) / num_bins
                noise = torch.randn_like(x_probe)
                x_t   = x_probe + noise * t_boundary
                err   = self._check_and_extend_bin(i, net, x_t, t_boundary, cond_probe, augment_labels)
            bin_errors.append(err)

        # All-reduce errors across ranks (sum valid, average), then apply extension once.
        is_distributed = torch.distributed.is_initialized()
        is_rank0       = (not is_distributed) or (torch.distributed.get_rank() == 0)

        err_tensor  = torch.tensor(bin_errors, device=device, dtype=torch.float32)
        valid_mask  = ~torch.isnan(err_tensor)
        err_sum     = err_tensor.nan_to_num(0.0)
        valid_count = valid_mask.float()
        if is_distributed:
            torch.distributed.all_reduce(err_sum,     op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(valid_count, op=torch.distributed.ReduceOp.SUM)
        err_avg = torch.where(valid_count > 0, err_sum / valid_count.clamp(min=1.0),
                              torch.full_like(err_sum, float('nan')))

        # Apply extension based on averaged errors — happens identically on all ranks.
        thresh = float(consistency_curriculum.consistency_threshold)
        for i in range(num_bins):
            e = err_avg[i].item()
            if e != e:  # nan → boundary, skip
                continue
            if e < thresh:
                if bin_mode == 't_next':
                    t_next_val = (i + 1) / num_bins
                    max_cap    = 1.0 - t_next_val   # t = t_next + delta ≤ 1.0
                else:
                    max_cap = (i + 1) / num_bins    # t_next = t - delta ≥ 0.0
                self.max_delta_per_bin[i] = min(self.max_delta_per_bin[i].item() + step, max_cap)

        # Linear floor: regardless of error, increase max_delta towards 1.0
        use_linear_floor = getattr(consistency_curriculum, 'use_linear_floor', True)
        linear_floor = None
        if start is not None and end is not None and end > start:
            progress     = max(0.0, min(1.0, (cur_step - start) / (end - start)))
            linear_floor = float(consistency_curriculum.init_max_delta) + progress * (1.0 - float(consistency_curriculum.init_max_delta))
            if use_linear_floor:
                if bin_mode == 't_next':
                    # Respect per-bin physical cap (delta ≤ 1 - t_next_val)
                    for j in range(num_bins):
                        t_next_val_j = (j + 1) / num_bins
                        cap_j = 1.0 - t_next_val_j
                        self.max_delta_per_bin[j] = max(
                            self.max_delta_per_bin[j].item(),
                            min(linear_floor, cap_j),
                        )
                else:
                    self.max_delta_per_bin.clamp_(min=linear_floor)

        if is_rank0:
            err_str = " ".join(f"[{i}]{err_avg[i].item():.6f}" for i in range(num_bins))
            print(f"[consistency] step={cur_step} errors: {err_str}")
            if linear_floor is not None:
                print(f"[consistency] linear_floor: {linear_floor:.4f}")
            delta_str = " ".join(f"[{i}]{v:.6f}" for i, v in enumerate(self.max_delta_per_bin.tolist()))
            print(f"[consistency] step={cur_step} max_delta: {delta_str}")

    # ------------------------------------------------------------------
    # Consistency analysis sweep (analysis mode only, rank-0 print)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def analyze_consistency(self, net, x_encoded: torch.Tensor, cond, augment_labels, cur_step: int):
        """Sweep delta values [delta_step … 1.0] for every t-bin and print consistency
        errors.  Useful for calibrating consistency_threshold before enabling the
        curriculum.

        For each (t_bin, delta) pair, mirrors _check_and_extend_bin:
            r_anchor = t_val - delta          (treat delta as the current max)
            r_step   = r_anchor - step_size   (candidate one-step extension)
            f_anchor = net(x_t, r_anchor, t)
            f_step   = net(x_{r_anchor}, r_step, r_anchor)
            f_full   = net(x_t, r_step, t)
            error (x_space): ||x_two - x_one|| / ||x_two||
                where x_one = x_t - (delta+step_size)*f_full
                      x_two = x_t - delta*f_anchor - step_size*f_step
            error (v_space): ||f_anchor + f_step - f_full||² / ||f_full||²
        """
        cc = self.cfg.consistency_curriculum
        analysis_cfg = getattr(cc, 'analysis', None)
        if analysis_cfg is None or not getattr(analysis_cfg, 'enabled', False):
            return

        freq = int(getattr(analysis_cfg, 'freq', 1000))
        if (cur_step % freq != 0) or (cur_step == 0):
            return
        if self._last_analysis_step == cur_step:
            return
        self._last_analysis_step = cur_step

        is_distributed = torch.distributed.is_initialized()
        is_rank0 = (not is_distributed) or (torch.distributed.get_rank() == 0)
        if not is_rank0:
            return

        device    = x_encoded.device
        probe_n   = min(int(getattr(analysis_cfg, 'probe_batch_size', 256)), x_encoded.shape[0])
        x_probe   = x_encoded[:probe_n]
        cond_probe = cond[:probe_n] if cond is not None else None
        aug_probe  = augment_labels[:probe_n] if augment_labels is not None else None

        num_bins    = int(cc.num_bins)
        step_size   = float(cc.step_size)
        bin_mode    = getattr(cc, 'bin_mode', 't')
        delta_step  = float(getattr(analysis_cfg, 'delta_step', 0.05))
        stride      = int(getattr(analysis_cfg, 'num_bins_stride', 1))
        error_space = getattr(cc, 'error_space', 'x_space')

        n_steps      = max(1, round(1.0 / delta_step))
        delta_values = [round((k + 1) * delta_step, 8) for k in range(n_steps)]

        # ---- header ----
        col_w        = 10  # must match f"{e:>10.5f}" below
        bin_label    = 't_next' if bin_mode == 't_next' else 't'
        delta_header = "  ".join(f"{'d='+f'{d:.2f}':>{col_w}}" for d in delta_values)
        print(f"\n[analysis] step={cur_step}  bins={num_bins}  stride={stride}  bin_mode={bin_mode}  step_size={step_size}  error_space={error_space}")
        print(f"[analysis] {'bin':>4}  {bin_label:>6}  {delta_header}")

        for i in range(0, num_bins, stride):
            anchor_val = (i + 1) / num_bins   # t_val (t-mode) or t_next_val (t_next-mode)

            bin_errors = []
            for delta in delta_values:
                if bin_mode == 't_next':
                    # t_next fixed; t_val = t_next + delta + step_size
                    t_next_val = anchor_val
                    t_val      = t_next_val + delta + step_size
                    r_anchor   = t_next_val + delta   # = t_val - step_size
                    r_step     = t_next_val

                    if t_val > 1.0:
                        bin_errors.append(float('nan'))
                        continue

                    noise = torch.randn_like(x_probe)
                    x_t   = x_probe + noise * t_val
                    b, bc = x_t.shape[0], (x_t.shape[0], 1, 1, 1, 1)
                    t_t      = torch.full(bc, t_val,    device=device, dtype=torch.float32)
                    t_anchor = torch.full(bc, r_anchor, device=device, dtype=torch.float32)
                    t_end    = torch.full(bc, r_step,   device=device, dtype=torch.float32)

                    # hop-1 (step_size): t_val → r_anchor
                    f_anchor = net(x_t, sigma_next=t_anchor, sigma=t_t,
                                   cond=cond_probe, augment_labels=aug_probe, return_extra_output=False)
                    x_at_anchor = x_t - step_size * f_anchor
                    # hop-2 (delta): r_anchor → t_next
                    f_step = net(x_at_anchor, sigma_next=t_end, sigma=t_anchor,
                                 cond=cond_probe, augment_labels=aug_probe, return_extra_output=False)
                    # direct (step_size + delta): t_val → t_next
                    f_full = net(x_t, sigma_next=t_end, sigma=t_t,
                                 cond=cond_probe, augment_labels=aug_probe, return_extra_output=False)

                else:  # bin_mode == 't'
                    # t fixed; r_anchor = t - delta, r_step = r_anchor - step_size
                    t_val    = anchor_val
                    r_anchor = t_val - delta
                    r_step   = r_anchor - step_size

                    if r_anchor < 0.0 or r_step < 0.0:
                        bin_errors.append(float('nan'))
                        continue

                    noise = torch.randn_like(x_probe)
                    x_t   = x_probe + noise * t_val
                    b, bc = x_t.shape[0], (x_t.shape[0], 1, 1, 1, 1)
                    t_t      = torch.full(bc, t_val,    device=device, dtype=torch.float32)
                    t_anchor = torch.full(bc, r_anchor, device=device, dtype=torch.float32)
                    t_end    = torch.full(bc, r_step,   device=device, dtype=torch.float32)

                    # hop-1 (delta): t → r_anchor
                    f_anchor = net(x_t, sigma_next=t_anchor, sigma=t_t,
                                   cond=cond_probe, augment_labels=aug_probe, return_extra_output=False)
                    x_at_anchor = x_t - delta * f_anchor
                    # hop-2 (step_size): r_anchor → r_step
                    f_step = net(x_at_anchor, sigma_next=t_end, sigma=t_anchor,
                                 cond=cond_probe, augment_labels=aug_probe, return_extra_output=False)
                    # direct (delta + step_size): t → r_step
                    f_full = net(x_t, sigma_next=t_end, sigma=t_t,
                                 cond=cond_probe, augment_labels=aug_probe, return_extra_output=False)

                # ---- shared error computation ----
                if error_space == 'x_space':
                    if bin_mode == 't_next':
                        x_one = x_t - (step_size + delta) * f_full
                        x_two = x_t - step_size * f_anchor - delta * f_step
                    else:
                        x_one = x_t - (delta + step_size) * f_full
                        x_two = x_t - delta * f_anchor - step_size * f_step
                    diff  = (x_two - x_one).flatten(1)
                    denom = x_two.flatten(1).pow(2).mean(1).sqrt() + 1e-12
                    error = (diff.pow(2).mean(1).sqrt() / denom).mean().item()
                else:  # v_space
                    res_sq    = (f_anchor + f_step - f_full).pow(2).flatten(1).mean(1).mean()
                    f_full_sq = f_full.pow(2).flatten(1).mean(1).mean().clamp(min=1e-8)
                    error     = (res_sq / f_full_sq).item()

                bin_errors.append(error)

            err_str = "  ".join(
                f"{e:>10.5f}" if e == e else f"{'nan':>10}"
                for e in bin_errors
            )
            print(f"[analysis] {i:>4}  {anchor_val:>6.4f}  {err_str}")

        print(f"[analysis] step={cur_step} done.\n")

    # ------------------------------------------------------------------
    # Forward — identical to AlphaFlowLoss except for the CAC update call
    # ------------------------------------------------------------------

    def forward(self, net, x, cond, augment_pipe=None, phase=LossPhase.Gen, cur_step=None,
                force_t_val=None, force_t_r_dt_val=None, align_ctx=None,
                compute_decomposed_loss=False) -> EasyDict:
        _ = force_t_val
        assert x.ndim == 5, f"Expected [b, T, C, H, W], got {x.shape}."
        batch_size = len(x)
        x_aug, augment_labels = maybe_augment_videos(x, augment_pipe)
        assert augment_labels is None, "Augment labels are not supported for joint training."
        x_aug = self._maybe_encode(net, x_aug, cond)

        # CAC: periodically update per-bin max delta (no-grad, no side effect on loss)
        cc = self.cfg.consistency_curriculum
        analysis_enabled = getattr(getattr(cc, 'analysis', None), 'enabled', False)
        if not analysis_enabled:
            self.update_max_delta(net, x_aug, cond, augment_labels, cur_step)
        self.analyze_consistency(net, x_aug, cond, augment_labels, cur_step)

        # -- Identical to AlphaFlowLoss.forward from here --
        t, t_next, dt, alpha = self.sample_traj_params(x_aug.shape[0], cur_step, x_aug.device)
        noise_unscaled = torch.randn_like(x_aug)
        noise_scaled   = noise_unscaled * t
        x_t            = self.apply_noise(x_aug, noise_scaled, t, net)
        velocity       = noise_unscaled - x_aug

        velocity_cfg = self._compute_velocity_cfg(
            velocity=velocity, x_t=x_t, t=t, cond=cond,
            augment_labels=augment_labels, net=net, batch_size=batch_size,
        )

        mask_c = (dt == 0).flatten()
        mask_d = ~mask_c
        batch_size_c, batch_size_d = mask_c.sum().item(), mask_d.sum().item()
        velocity_cfg_c, x_t_c, t_c, t_next_c, cond_c = (
            velocity_cfg[mask_c], x_t[mask_c], t[mask_c], t_next[mask_c], cond[mask_c],
        )
        x_t_d, velocity_cfg_d, t_d, t_next_d, dt_d, cond_d = (
            x_t[mask_d], velocity_cfg[mask_d], t[mask_d], t_next[mask_d], dt[mask_d], cond[mask_d],
        )

        mean_velocity_c = self._compute_mean_velocity_c(
            x_t_c, t_next_c, t_c, velocity_cfg_c, cond_c, augment_labels, net,
        )
        mean_velocity_d = self._compute_mean_velocity_d(
            x_t_d, velocity_cfg_d, t_next_d, t_d, dt_d, cond_d, augment_labels, net,
        )
        mean_velocity = torch.cat([mean_velocity_c, mean_velocity_d], dim=0)

        pred_mean_velocity, ctx = net(
            x_t, sigma_next=t_next, sigma=t,
            cond=cond, augment_labels=augment_labels, return_extra_output=True,
        )

        loss_unscaled = ((pred_mean_velocity - mean_velocity) ** 2).flatten(1).mean(1)
        weight_c = torch.ones(batch_size_c, device=velocity.device)
        weight_d = torch.ones(batch_size_d, device=velocity.device) * alpha
        weight   = torch.cat([weight_c, weight_d], dim=0)
        if self.cfg.use_adaptive_loss:
            weight = weight / (loss_unscaled.detach() + self.cfg.adaptive_loss_weight_eps)
        loss = weight * loss_unscaled

        loss_tfm          = ((pred_mean_velocity - velocity_cfg) ** 2).flatten(1).mean(1)
        loss_tcc          = (2 * (velocity_cfg - mean_velocity) * pred_mean_velocity).flatten(1).mean(1)
        loss_tfm_plus_tcc = loss_tfm + loss_tcc

        dt_stats = self.calculate_per_delta(batch_size, t, t_next, loss)

        return misc.filter_nones(EasyDict(
            total                  = loss,
            trajectory_FM          = loss_tfm,
            trajectory_consistency = loss_tcc,
            trajectory_sum         = loss_tfm_plus_tcc,
            **dt_stats,
        ))


#----------------------------------------------------------------------------
# Various reconstruction loss functions.

def reweigh_supp_loss(loss_main: torch.Tensor, loss_supp: torch.Tensor, relative_weight: float, eps: float=1e-6) -> torch.Tensor:
    # Re-adjusts the loss_supp to have the magnitude of relative_weight * loss_main.
    assert relative_weight > 0, f"Expected relative_weight to be positive, but got {relative_weight} instead."
    return loss_supp * relative_weight * (loss_main.abs().mean().item() / (loss_supp.abs().mean().item() + eps)) # [<loss_supp shape>]

def compute_image_loss_per_frame(x_rec: torch.Tensor, x_gt: torch.Tensor, loss_fn: Callable, *args, **kwargs) -> torch.Tensor:
    assert x_rec.ndim == 5, f"Expected x to have 5 dimensions: [b, t, c, h, w], got {x_rec.shape} instead."
    assert x_rec.shape == x_gt.shape, f"Expected x_rec and x_gt to have the same shape, but got {x_rec.shape} and {x_gt.shape} instead."
    b, t = x_rec.shape[:2] # (1, 1)
    x_rec = einops.rearrange(x_rec, 'b t c h w -> (b t) c h w') # [b * t, c, h, w]
    x_gt = einops.rearrange(x_gt, 'b t c h w -> (b t) c h w') # [b * t, c, h, w]
    loss = loss_fn(x_rec, x_gt, *args, **kwargs) # [b * t]
    loss = loss.view(-1, 1, 1, 1) if loss.ndim == 1 else loss # [b * t, 1, 1, 1]
    loss = einops.rearrange(loss, '(b t) 1 1 1 -> b t 1 1 1', b=b, t=t) # [b, t, 1, 1, 1]
    return loss # [b, t, 1, 1, 1]

def compute_image_lpips(x_rec: torch.Tensor, x_gt: torch.Tensor, lpips_fn: Callable, loss_cfg: DictConfig) -> torch.Tensor:
    num_tiles = 1
    if loss_cfg.downsample_to_native:
        if loss_cfg.downsample_tiled:
            # Split the image into tiles, compute LPIPS for each tile, and average the results.
            x_rec, x_gt = [pad_to_divisible(x, 224) for x in (x_rec, x_gt)] # 2 x [b, c, h' * 224, w' * 224]
            x_rec, x_gt = [F.unfold(x, kernel_size=224, stride=224) for x in (x_rec, x_gt)] # 2 x [b, c * 224 * 224, num_tiles]
            num_tiles = x_rec.shape[-1] # (1)
            x_rec, x_gt = [einops.rearrange(x, 'b (c h w) n -> (b n) c h w', c=3, h=224, w=224) for x in (x_rec, x_gt)]
        else:
            x_rec = F.interpolate(x_rec, size=(224, 224), mode='area') # [b, c, 224, 224]
            x_gt = F.interpolate(x_gt, size=(224, 224), mode='area') # [b, c, 224, 224]
    loss_total, losses_per_layer = lpips_fn(x_rec, x_gt, retPerLayer=True) # [b, 1, 1, 1]
    loss = loss_total if loss_cfg.num_first_layers is None else torch.stack(losses_per_layer[:loss_cfg.num_first_layers]).sum(dim=0) # [b, 1, 1, 1]

    if loss_cfg.downsample_to_native and loss_cfg.downsample_tiled:
        loss = einops.rearrange(loss, '(b n) 1 1 1 -> b n 1 1 1', n=num_tiles).mean(dim=1) # [b, 1, 1, 1]
    return loss

def compute_video_lpips(x_rec: torch.Tensor, x_gt: torch.Tensor, lpips_fn: Callable, *args, **kwargs) -> torch.Tensor:
    return compute_image_loss_per_frame(x_rec, x_gt, compute_image_lpips, lpips_fn, *args, **kwargs) # [b, t, 1, 1, 1]

def compute_img_grad_loss_per_frame(x_rec: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    return compute_image_loss_per_frame(x_rec, x_gt, compute_img_grad_loss) # [b, t, 1, 1, 1]

def compute_img_grad_loss(x_rec: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    x_rec_grad = compute_img_grad_magnitude(x_rec) # [b, h, w]
    x_gt_grad = compute_img_grad_magnitude(x_gt) # [b, h, w]
    loss = (x_rec_grad - x_gt_grad).pow(2).mean(dim=[1, 2]).sqrt() # [b]
    return loss

def compute_img_grad_magnitude(x: torch.Tensor, filter_pt: torch.Tensor=None) -> torch.Tensor:
    assert x.ndim == 4, f"Expected x to have 4 dimensions: [b, c, h, w], got {x.shape} instead."
    filter_pt = torch.tensor(SOBEL_FILTER, dtype=torch.float32, device=x.device).reshape(2, 1, 3, 3) if filter_pt is None else filter_pt # [2, 1, 3, 3]
    batch_size, num_channels = x.shape[0], x.shape[1] # (1)
    x = einops.rearrange(x, 'b c h w -> (b c) 1 h w') # [b * c, 1, h, w]
    grad = F.conv2d(x, filter_pt, padding=1) # [b * c, 2, h, w], pylint: disable=not-callable
    grad_magnitude = grad.pow(2).sum(dim=1).sqrt() # [b * c, h, w]
    grad_magnitude = einops.rearrange(grad_magnitude, '(b c) h w -> b c h w', b=batch_size, c=num_channels) # [b, c, h, w]
    grad_magnitude = grad_magnitude.mean(dim=1) # [b, h, w]

    return grad_magnitude

def compute_video_freq3d_loss(x_rec: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    assert x_rec.ndim == 5, f"Expected x to have 5 dimensions: [b, t, c, h, w], got {x_rec.shape} instead."
    assert x_rec.shape == x_gt.shape, f"Expected x_rec and x_gt to have the same shape, but got {x_rec.shape} and {x_gt.shape} instead."
    return compute_freq_loss(x_rec, x_gt, dim=(1, 3, 4)).view(-1, 1, 1, 1, 1) # [b, 1, 1, 1, 1]

def compute_video_freq2d_loss(x_rec: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    return compute_image_loss_per_frame(x_rec, x_gt, compute_img_freq_loss) # [b, t, 1, 1, 1]

def compute_img_freq_loss(x_rec: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    return compute_freq_loss(x_rec, x_gt, dim=(2, 3)).view(-1, 1, 1, 1) # [b, 1, 1, 1]

def compute_freq_loss(x_rec, x_gt, dim: tuple[int]):
    # Transform both inputs to the frequency domain using FFT
    fft_rec = torch.fft.fftn(x_rec, dim=dim) # pylint: disable=not-callable
    fft_gt = torch.fft.fftn(x_gt, dim=dim) # pylint: disable=not-callable

    # Shift the zero frequency component to the center (TODO: we don't really need that?)
    fft_shifted_rec = torch.fft.fftshift(fft_rec, dim=dim) # pylint: disable=not-callable
    fft_shifted_gt = torch.fft.fftshift(fft_gt, dim=dim) # pylint: disable=not-callable

    # Compute magnitude and/or phase
    magnitude_rec = torch.abs(fft_shifted_rec) # [b, c, d1, ..., dn]
    magnitude_gt = torch.abs(fft_shifted_gt) # [b, c, d1, ..., dn]

    # Compute phase
    # phase_rec = torch.angle(gen_fft_shift)
    # phase_gt = torch.angle(gt_fft_shift)

    loss = (magnitude_rec - magnitude_gt).pow(2).view(magnitude_rec.shape[0], -1).mean(dim=1).sqrt() # [b]

    return loss

def compute_image_random_conv_l2(x_rec: torch.Tensor, x_gt: torch.Tensor, loss_cfg: DictConfig) -> torch.Tensor:
    p: int = loss_cfg.patch_size # [1]
    weight = torch.randn(loss_cfg.embed_dim, x_rec.shape[1], p, p, device=x_rec.device, dtype=x_rec.dtype) / (x_rec.shape[1] * p ** 2) # [embed_dim, c, kh, kw]
    x_rec, x_gt = [einops.rearrange(F.conv2d(x, weight, stride=p // 2, padding=0), 'b c h w -> b (h w) c') for x in (x_rec, x_gt)] # pylint: disable=not-callable
    return (x_rec - x_gt).pow(2).flatten(start_dim=1).mean(dim=1).sqrt().view(-1, 1, 1, 1) # [b, 1, 1, 1]

def compute_video_random_conv_l2(x_rec: torch.Tensor, x_gt: torch.Tensor, loss_cfg: DictConfig) -> torch.Tensor:
    assert x_rec.shape == x_gt.shape, f"Expected x_rec and x_gt to have the same shape, but got {x_rec.shape} and {x_gt.shape} instead."
    x_rec, x_gt = [einops.rearrange(x, 'b t c h w -> b c t h w') for x in (x_rec, x_gt)] # 2 x [b, c, t, h, w]
    t, h, w = x_rec.shape[2:] # 3x (1,)
    kernel_size = tuple(min(r, loss_cfg.patch_size) for r in (t, h, w)) # [3]
    stride = tuple(max(1, k // 2) for k in kernel_size) # [3]
    padding = tuple(k // 2 for k in kernel_size) # [3]
    weight = torch.randn(loss_cfg.embed_dim, x_rec.shape[1], *kernel_size, device=x_rec.device, dtype=x_rec.dtype) / (x_rec.shape[1] * np.prod(kernel_size)) # [embed_dim, c, kt, kh, kw]
    x_rec, x_gt = [F.conv3d(x, weight, stride=stride, padding=padding) for x in (x_rec, x_gt)] # 2 x [b, embed_dim, t', h', w']; pylint: disable=not-callable
    loss = (x_rec - x_gt).pow(2).flatten(start_dim=1).mean(dim=1).sqrt().view(-1, 1, 1, 1, 1) # [b, 1, 1, 1, 1]
    return loss

def compute_framewise_video_random_conv_l2(x_rec: torch.Tensor, x_gt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return compute_image_loss_per_frame(x_rec, x_gt, compute_image_random_conv_l2, *args, **kwargs) # [b, t, 1, 1, 1]

#----------------------------------------------------------------------------
# Diffusion training utility functions.

def sample_logit_norm_time(time_sampling_cfg: EasyDict, shape: torch.Size | tuple, device: torch.device=None) -> torch.Tensor:
    """
    Time Samples following the Logit Normal distribution of Stable Diffusion 3
    Produces times in [0, 1-eps] following "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
    """
    randn = misc.randn(shape, device=device) * time_sampling_cfg.scale + time_sampling_cfg.location # [b, ...]
    logit_normal = randn.sigmoid() # [b, ...]
    logit_normal_rescaled = logit_normal * (1 - time_sampling_cfg.eps) # [b, ...]. Rescales between [0, 1-eps]

    return logit_normal_rescaled

def sample_truncated_logit_norm_time(time_sampling_cfg: EasyDict, batch_size: int, device: torch.device=None, upper_truncated = None) -> torch.Tensor:
    """
    Time Samples following the Logit Normal distribution of Stable Diffusion 3
    Produces times in [0, 1-eps] following "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
    """
    from torchrl.modules.distributions import TruncatedNormal
    assert upper_truncated is not None, "upper_truncated must be specified"
    loc = torch.ones(batch_size, device=device) * time_sampling_cfg.location # [b]
    scale = torch.ones(batch_size, device=device) * time_sampling_cfg.scale # [b]
    trunc_randn = TruncatedNormal(loc=loc, scale=scale, low=-float('Inf'), high=torch.logit(upper_truncated)).sample() # [b]
    logit_normal = torch.sigmoid(trunc_randn) # [b]

    return logit_normal

def sample_custom_t_steps(t_steps: list[float], shape: torch.Size | int, device) -> torch.Tensor:
    """Sampling uniformly from a list of t_steps."""
    assert len(t_steps) > 0, f"Expected t_steps to be non-empty, but got {t_steps} instead."
    assert all(0.0 <= t <= 1.0 for t in t_steps), f"Expected all t_steps to be in [0, 1], but got {t_steps} instead."
    t_steps_sampled = np.random.choice(t_steps, size=tuple(shape), replace=True) # [b, ...]
    t_steps_sampled = torch.from_numpy(t_steps_sampled).float().to(device) # [b, ...]

    return t_steps_sampled

#----------------------------------------------------------------------------
# Utility functions.

def _maybe_convert_tensor_group(losses: dict[str, TensorLike], sum_key='total', default_modality='video') -> EasyDict:
    """We can't backprop through TensorGroup losses, so we convert/rename them to tensors here."""
    out = {}
    for k, v in losses.items():
        if isinstance(v, TensorGroup):
            if k == sum_key:
                out[sum_key] = torch.stack(list(v.flatten(1).mean(1).values()), dim=1).sum(dim=1) # [b]
            else:
                for modality in v.keys():
                    if modality == default_modality:
                        out[k] = v.video
                    out[f"{k}_{modality}"] = v[modality]

        else:
            out[k] = v
    return EasyDict(out)

def maybe_filter_loss_by_mask(loss_rec: torch.Tensor, frames_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    assert loss_rec.ndim == 5, f"Expected loss_rec to have 5 dimensions: [b, t, c, h, w], got {loss_rec.shape} instead."
    assert frames_mask is None or frames_mask.ndim == 2, f"Expected frames_mask to have 2 dimensions: [b, t], got {frames_mask.shape} instead."
    assert frames_mask is None or len(loss_rec) == len(frames_mask), f"Expected loss_rec and frames_mask to have the same batch size, but got {len(loss_rec)} and {len(frames_mask)} instead."
    if frames_mask is not None:
        # Do not compute the loss for MISSING and CONTEXT frames.
        assert frames_mask.shape[:2] == loss_rec.shape[:2], f"Expected frames_mask and loss to have the same shape, but got {frames_mask.shape} and {loss_rec.shape} instead."
        keep_mask = frames_mask == TokenType.QUERY.value # [b, t]
        loss_rec = [l[m] for l, m in zip(loss_rec, keep_mask)] # (batch_size, [<any>, c, h, w])
        assert all(l.shape[0] > 0 for l in loss_rec), f"Expected all losses to have at least one frame, but got {list(map(len, loss_rec))} frames instead."
        loss_rec = torch.stack([l.mean(dim=0) for l in loss_rec]).unsqueeze(1) # [b, 1, c, h, w]
    return loss_rec

def pad_to_divisible(x: torch.Tensor, divisor: int=224) -> torch.Tensor:
    _b, _c, h, w = x.shape
    pad_h_total = (divisor - h % divisor) % divisor
    pad_w_total = (divisor - w % divisor) % divisor
    pad_top, pad_left = pad_h_total // 2, pad_w_total // 2
    pad_bottom, pad_right = pad_h_total - pad_top, pad_w_total - pad_left # Residual padding
    x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom)) # [b, c, h', w']

    return x_padded

def maybe_augment_videos(videos: torch.Tensor, augment_pipe: Optional[torch.nn.Module]):
    if not augment_pipe is None:
        num_frames = videos.shape[1]
        videos = einops.rearrange(videos, 'b t c h w -> b (t c) h w') # [b, t * c, h, w]
        videos_aug, augment_labels = augment_pipe(videos, num_frames=num_frames) # [b, t * c, h, w], [b, augment_dim]
        videos_aug = einops.rearrange(videos_aug, 'b (t c) h w -> b t c h w', t=num_frames) # [b, t, c, h, w]
    else:
        videos_aug, augment_labels = (videos, None)
    return videos_aug, augment_labels

#----------------------------------------------------------------------------
# Lipschitz regularization utilities.

def compute_r1_reg(x: torch.Tensor, y: torch.Tensor, reduce_op: str='sum') -> torch.Tensor:
    assert x.ndim == 5, f"Expected x to have 5 dimensions: [b, t, c, h, w], got {x.shape} instead."
    assert x.requires_grad, "Expected x to require gradients."
    r1_grads = torch.autograd.grad(outputs=[y.sum()], inputs=[x], create_graph=True, only_inputs=True)[0] # [b * np, t, c, h, w]
    r1_penalty = r1_grads.square() # [b * np, t, c, h, w]
    r1_penalty = einops.reduce(r1_penalty, 'b t c h w -> b', reduce_op) # [b * np]

    return r1_penalty

#----------------------------------------------------------------------------
