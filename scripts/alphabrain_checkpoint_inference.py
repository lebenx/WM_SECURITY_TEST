#!/usr/bin/env python3
"""Checkpoint-exact AlphaBrain CosmosPolicy inference for clean task evaluation.

The historical probe remains unchanged.  This loader uses the native WanVAE_
weights AND normalization buffers embedded in the policy checkpoint, and marks
the four observed latent frames as clean at every denoising step.  Its query
signature is compatible with pilot_alphabrain_cosmos_libero_short_control.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/AlphaBrain/cosmos_policy_libero/model.safetensors"


def _source_path():
    source = str(ROOT / "vendor/AlphaBrain")
    if source not in sys.path:
        sys.path.insert(0, source)


class CheckpointWanVAE(torch.nn.Module):
    """Native VAE; all learned tensors and scale buffers come from checkpoint."""

    def __init__(self, checkpoint=MODEL, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        _source_path()
        from AlphaBrain.model.modules.world_model.cosmos.wan_vae_arch import WanVAE_

        with torch.device("meta"):
            self.vae = WanVAE_(
                dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                attn_scales=[], temperal_downsample=[False, True, True],
                dropout=0.0, temporal_window=16,
            )
            for name in ("latents_mean", "latents_std", "_inv_std"):
                self.register_buffer(name, torch.empty(1, 16, 1, 1, 1))
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            state = {
                key.removeprefix("vae."): handle.get_tensor(key)
                for key in handle.keys() if key.startswith("vae.")
            }
        # Strict checking catches incompatible architectures and absent buffers.
        self.load_state_dict(state, strict=True, assign=True)
        self.to(device=device, dtype=dtype).eval().requires_grad_(False)
        if not torch.isfinite(self.latents_mean).all() or not (self.latents_std > 0).all():
            raise ValueError("Invalid checkpoint VAE normalization buffers")
        if not torch.isfinite(self._inv_std).all() or not (self._inv_std > 0).all():
            raise ValueError("Invalid checkpoint VAE reciprocal scale")
        self.checkpoint = str(checkpoint)
        self._is_pth_vae = True
        self.z_dim = 16
        self.spatial_compression_factor = 8
        self.temporal_compression_factor = 4

    @torch.inference_mode()
    def encode(self, video):
        # Native normalization multiplies the saved bf16 reciprocal.  Recomputing
        # from JSON, or dividing by std, need not be numerically equivalent.
        return self.vae.encode(
            video.to(device=self.latents_mean.device, dtype=self.latents_mean.dtype),
            [self.latents_mean, self._inv_std],
        ).to(video.dtype)

    @torch.inference_mode()
    def decode(self, latent):
        return self.vae.decode(
            latent.to(device=self.latents_mean.device, dtype=self.latents_mean.dtype),
            [self.latents_mean, self._inv_std],
        ).to(latent.dtype)

    def get_latent_num_frames(self, num_pixel_frames):
        return 1 + (num_pixel_frames - 1) // 4

    def get_pixel_num_frames(self, num_latent_frames):
        return 4 * (num_latent_frames - 1) + 1


def load_model(checkpoint=MODEL, device="cuda"):
    """Return DiT, checkpoint-exact VAE, official sampler, official noise RNG."""
    if os.environ.get("ALPHABRAIN_USE_NATIVE_TE") != "1":
        import pilot_alphabrain_cosmos_policy_goal_swap as legacy
        legacy.install_transformer_engine_stub()
    _source_path()
    from AlphaBrain.model.modules.world_model.cosmos.official_dit.minimal_v1_lvg_dit import MinimalV1LVGDiT
    from AlphaBrain.model.modules.world_model.cosmos.cosmos_sampler import CosmosPolicySampler
    from AlphaBrain.model.modules.world_model.cosmos.noise_utils import arch_invariant_rand

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # Preserve the released architecture, changing only the attention backend
    # needed by this host.  Every learned parameter must load successfully.
    with torch.device("meta"):
        model = MinimalV1LVGDiT(
            max_img_h=240, max_img_w=240, max_frames=128, in_channels=16,
            out_channels=16, patch_spatial=2, patch_temporal=1,
            concat_padding_mask=True, model_channels=2048, num_blocks=28,
            num_heads=16, crossattn_emb_channels=1024, use_crossattn_projection=False,
            atten_backend="torch", pos_emb_cls="rope3d", mlp_ratio=4.0,
            use_adaln_lora=True, adaln_lora_dim=256, use_wan_fp32_strategy=False,
            pos_emb_learnable=True, rope_enable_fps_modulation=False,
            rope_h_extrapolation_ratio=3.0, rope_w_extrapolation_ratio=3.0,
        ).to(dtype=torch.bfloat16)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        state = {
            key.removeprefix("dit."): handle.get_tensor(key)
            for key in handle.keys() if key.startswith("dit.")
        }
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing or any("_extra_state" not in key for key in unexpected):
        raise RuntimeError(f"DiT checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    del state
    vae = CheckpointWanVAE(checkpoint, device=device)
    return model, vae, CosmosPolicySampler(), arch_invariant_rand


def denoise(model, xt, sigma, crossattn_emb, fps, padding_mask, gt_frames):
    """Match released CosmosPolicy._denoise, with sigma_conditional=0, data=1."""
    batch, _, frames, height, width = xt.shape
    n_cond = gt_frames.shape[2]
    sigma5 = sigma.float().view(batch, 1, 1, 1, 1)
    t = sigma5 / (sigma5 + 1.0)
    net_input = xt.float() * (1.0 - t)
    net_input[:, :, :n_cond] = gt_frames.float()
    frame_noise = t.view(batch, 1).expand(batch, frames).clone()
    frame_noise[:, :n_cond] = 0.0
    cond = torch.zeros(batch, 1, frames, height, width, device=xt.device, dtype=xt.dtype)
    cond[:, :, :n_cond] = 1.0
    ts_dtype = torch.float32 if model.use_wan_fp32_strategy else torch.bfloat16
    out = model(
        x_B_C_T_H_W=net_input.to(torch.bfloat16),
        timesteps_B_T=frame_noise.to(ts_dtype),
        crossattn_emb=crossattn_emb.to(torch.bfloat16),
        condition_video_input_mask_B_C_T_H_W=cond.to(torch.bfloat16),
        fps=fps.to(torch.bfloat16),
        padding_mask=padding_mask.to(torch.bfloat16),
    )
    x0 = (1.0 - t) * xt.float() - t * out.float()
    x0[:, :, :n_cond] = gt_frames
    return x0


def normalize_action(action, stats):
    """Official inverse scaling clips normalized actions before min-max mapping."""
    low = np.asarray(stats["actions_min"], dtype=np.float32)
    high = np.asarray(stats["actions_max"], dtype=np.float32)
    values = np.clip(action.detach().float().cpu().numpy(), -1.0, 1.0)
    return 0.5 * (values + 1.0) * (high - low) + low


@torch.inference_mode()
def query(model, vae, sampler, rand, obs, task_embedding, stats, num_steps=5,
          *, return_latent=False, image_transform="vertical"):
    import pilot_alphabrain_cosmos_libero_short_control as legacy
    _source_path()
    from AlphaBrain.model.modules.world_model.cosmos.latent_utils import replace_latent_with_proprio

    device = next(model.parameters()).device
    if image_transform not in {"vertical", "rot180", "none"}:
        raise ValueError(f"Unsupported image transform: {image_transform}")
    def transform(image):
        if image_transform == "vertical":
            image = image[::-1]
        elif image_transform == "rot180":
            image = image[::-1, ::-1]
        return legacy._official_preprocess_image(np.ascontiguousarray(image))
    primary = transform(obs["agentview_image"])
    wrist = transform(obs["robot0_eye_in_hand_image"])
    p = torch.from_numpy(primary).permute(2, 0, 1).unsqueeze(1).expand(3, 4, 224, 224)
    w = torch.from_numpy(wrist).permute(2, 0, 1).unsqueeze(1).expand(3, 4, 224, 224)
    blank = torch.zeros_like(p)
    video = torch.cat([blank[:, :1], blank, w, p, blank, blank, w, p, blank], dim=1).unsqueeze(0)
    video = (video.to(device=device, dtype=torch.bfloat16) / 127.5 - 1.0).to(memory_format=torch.channels_last_3d)
    latent = vae.encode(video).to(torch.bfloat16)
    state = np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]]).astype(np.float32)
    p_min, p_max = (np.asarray(stats[key], dtype=np.float32) for key in ("proprio_min", "proprio_max"))
    state = 2.0 * (state - p_min) / np.where(np.abs(p_max - p_min) < 1e-8, 1.0, p_max - p_min) - 1.0
    latent = replace_latent_with_proprio(
        latent, torch.tensor(state[None], device=device, dtype=torch.bfloat16),
        torch.tensor([1], device=device),
    )
    if latent.shape != (1, 16, 9, 28, 28):
        raise ValueError(f"Unexpected policy latent shape: {tuple(latent.shape)}")
    gt = latent[:, :, :4].clone()
    padding = torch.zeros(1, 1, 224, 224, device=device, dtype=torch.bfloat16)
    fps = torch.tensor([16], device=device, dtype=torch.float32)
    noise = rand(tuple(latent.shape), seed=1).to(device=device, dtype=torch.float32) * 80.0

    def x0_fn(xt, sigma):
        return denoise(model, xt, sigma, task_embedding, fps, padding, gt)

    sampled = sampler.forward(
        x0_fn=x0_fn, x_sigma_max=noise, num_steps=num_steps, sigma_min=4.0, sigma_max=80.0,
    )
    action = legacy.action_from_latent(sampled[:, :, 4]).detach()
    return (action, sampled.detach()) if return_latent else action
