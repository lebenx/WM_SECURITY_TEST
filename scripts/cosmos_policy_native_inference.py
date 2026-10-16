#!/usr/bin/env python3
"""Lightweight adapter for the official NVIDIA Cosmos Policy LIBERO checkpoint.

Implements the released policy -> fixed-action forecast -> future-conditioned
value calls, with no new score or planner. The DiT uses the available torch
backend port; its 570 model tensors load strictly from NVIDIA's weights. The
VAE core and data codecs are extracted unchanged from the pinned NVlabs source.
The full upstream distributed/Transformer-Engine stack is not reproduced here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import alphabrain_checkpoint_inference as alpha
import cosmos_policy_codecs as codecs
from cosmos_policy_wan_vae import WanVAE_

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/CosmosPolicy/Cosmos-Policy-LIBERO-Predict2-2B.pt"
TOKENIZER = MODEL.parent / "tokenizer/tokenizer.pth"
MODEL_SHA256 = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2"
TOKENIZER_SHA256 = "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
TEXT_SHA256 = "8a03499676c6c196127c577144b9fd09bb02c30f9caf058adbcef09bb99ad8f5"
SOURCE_REVISION = "18a2accadf4e7a3531e56754102af5a24d2316da"
CHECKPOINT_REVISION = "cb689ec0e3347c13667d70a78a3447388f5c3bb8"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def verify_artifact(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {actual}")
    return actual


class OfficialWanVAE(torch.nn.Module):
    """Official tokenizer tensors, inherited window=16, source bf16 scales.

    The policy tokenizer registration defaults to 4, but the inherited
    Stage-c/Index-102 experiment's later _self_ explicitly sets 16.
    """

    def __init__(self, checkpoint=TOKENIZER, device="cuda", verify=True):
        super().__init__()
        if verify:
            verify_artifact(checkpoint, TOKENIZER_SHA256)
        with torch.device("meta"):
            self.vae = WanVAE_(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4],
                               num_res_blocks=2, attn_scales=[],
                               temperal_downsample=[False, True, True],
                               dropout=0.0, temporal_window=16)
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True, mmap=True)
        self.vae.load_state_dict(state, strict=True, assign=True)
        # NVlabs CosmosPolicyWanVAE constants; load_mean_std=False. Its interface
        # explicitly sets is_amp=False and casts all parameters to bf16.
        mean = [-.7571, -.7089, -.9113, .1075, -.1745, .9653, -.1517, 1.5508,
                .4134, -.0715, .5517, -.3632, -.1922, -.9497, .2503, -.2921]
        std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
               3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
        self.register_buffer("latents_mean", torch.tensor(mean, dtype=torch.bfloat16))
        self.register_buffer("latents_std", torch.tensor(std, dtype=torch.bfloat16))
        self.register_buffer("inv_std", 1.0 / self.latents_std)
        self.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, video):
        return self.vae.encode(video.to(self.latents_mean),
                               [self.latents_mean, self.inv_std]).to(video.dtype)

    @torch.inference_mode()
    def decode(self, latent):
        return self.vae.decode(latent.to(self.latents_mean),
                               [self.latents_mean, self.inv_std]).to(latent.dtype)


def load_model(checkpoint=MODEL, device="cuda", verify=True):
    """Load independent NVIDIA weights; never substitute Alpha checkpoint VAE."""
    if verify:
        verify_artifact(checkpoint, MODEL_SHA256)
    if os.environ.get("ALPHABRAIN_USE_NATIVE_TE") != "1":
        import cosmos_policy_torch_compat as compatibility
        compatibility.install_transformer_engine_stub()
    alpha._source_path()
    from AlphaBrain.model.modules.world_model.cosmos.official_dit.minimal_v1_lvg_dit import MinimalV1LVGDiT
    from AlphaBrain.model.modules.world_model.cosmos.cosmos_sampler import CosmosPolicySampler
    from AlphaBrain.model.modules.world_model.cosmos.noise_utils import arch_invariant_rand

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
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
    raw = torch.load(str(checkpoint), map_location="cpu", weights_only=True, mmap=True)
    counters = {"accum_image_sample_counter", "accum_iteration", "accum_train_in_hours",
                "accum_video_sample_counter"}
    state, omitted = {}, []
    for key, value in raw.items():
        if not key.startswith("net."):
            raise RuntimeError(f"Unexpected official checkpoint namespace: {key}")
        name = key.removeprefix("net.")
        if name in counters or name.endswith("._extra_state"):
            omitted.append(key)
        else:
            state[name] = value
    if len(state) != 570:
        raise RuntimeError(f"Expected 570 DiT tensors, found {len(state)}")
    model.load_state_dict(state, strict=True, assign=True)
    model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    model.checkpoint_adapter_metadata = {"loaded_tensors": len(state), "omitted_metadata": omitted}
    del state, raw
    return model, OfficialWanVAE(device=device, verify=verify), CosmosPolicySampler(), arch_invariant_rand


def load_stats(path=MODEL.parent / "libero_dataset_statistics.json"):
    # Upstream load_dataset_stats uses np.array without a dtype override.
    return {key: np.array(value) for key, value in json.loads(Path(path).read_text()).items()}


def preprocess_observation(obs):
    cfg = SimpleNamespace(use_jpeg_compression=True, trained_with_image_aug=True)
    images = [np.ascontiguousarray(obs[key][::-1])
              for key in ("robot0_eye_in_hand_image", "agentview_image")]
    return codecs.prepare_images_for_model(images, cfg)


@torch.inference_mode()
def encode_observation(vae, obs, stats):
    """Encode the upstream 33-frame sequence and inject observed proprio."""
    wrist, primary = preprocess_observation(obs)
    blank = np.zeros_like(primary)
    groups = [blank[None]] + [np.repeat(x[None], 4, axis=0)
                             for x in (blank, wrist, primary, blank, blank, wrist, primary, blank)]
    raw = np.concatenate(groups, axis=0)
    video = torch.from_numpy(raw).permute(3, 0, 1, 2)[None]
    video = (video.to(vae.latents_mean) / 127.5 - 1.0).to(memory_format=torch.channels_last_3d)
    original = vae.encode(video)
    proprio = np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]))
    proprio = codecs.rescale_proprio(proprio, stats)
    latent = codecs.replace_latent_with_proprio(
        original.clone(), torch.from_numpy(proprio)[None].to(vae.latents_mean),
        torch.tensor([1], device=video.device))
    if latent.shape != (1, 16, 9, 28, 28):
        raise ValueError(f"Unexpected official latent shape: {tuple(latent.shape)}")
    return latent, original


@torch.inference_mode()
def sample_conditioned(model, sampler, rand, latent, text, n_cond, seed=1, num_steps=5):
    """Released sampling with prefix n_cond=4 policy, 5 forecast, or 8 value."""
    if n_cond not in (4, 5, 8) or num_steps not in (1, 5):
        raise ValueError("This verified adapter supports n_cond=4/5/8 and official 1/5 denoising steps")
    device = latent.device
    text = torch.as_tensor(text, device=device, dtype=torch.bfloat16)
    if text.ndim == 2:
        text = text[None]
    if text.shape[0] != latent.shape[0]:
        raise ValueError("Text and latent batch sizes differ")
    fps = torch.full((latent.shape[0],), 16, device=device, dtype=torch.bfloat16)
    padding = torch.zeros(latent.shape[0], 1, 224, 224, device=device, dtype=torch.bfloat16)
    gt = latent[:, :, :n_cond].clone()
    noise = rand(tuple(latent.shape), seed=seed).to(device=device, dtype=torch.float32) * 80.0
    sampled = sampler.forward(
        x0_fn=lambda x, sigma: alpha.denoise(model, x, sigma, text, fps, padding, gt),
        x_sigma_max=noise, num_steps=num_steps, sigma_min=4.0, sigma_max=80.0,
    )
    if not torch.equal(sampled[:, :, :n_cond], gt.float()):
        raise RuntimeError("Native conditioning drifted during denoising")
    if not torch.isfinite(sampled).all():
        raise RuntimeError("Nonfinite native sampler output")
    return sampled


def action_from_latent(latent):
    indices = torch.full((latent.shape[0],), 4, device=latent.device, dtype=torch.int64)
    return codecs.extract_action_chunk_from_latent_sequence(latent, (16, 7), indices)


def normalize_action(action, stats):
    """NVIDIA's exact inverse min-max. It does not clip or binarize gripper."""
    return codecs.unnormalize_actions(action.detach().float().cpu().numpy(), stats)


@torch.inference_mode()
def get_action(model, sampler, rand, observed_latent, text, seed=1, num_steps=5):
    sample = sample_conditioned(model, sampler, rand, observed_latent, text, 4, seed, num_steps)
    return action_from_latent(sample), sample


@torch.inference_mode()
def get_future_state_prediction(model, sampler, rand, action_latent, text, seed=1, num_steps=1):
    return sample_conditioned(model, sampler, rand, action_latent, text, 5, seed, num_steps)


@torch.inference_mode()
def get_value_prediction(model, sampler, rand, future_latent, text, seed=1, num_steps=1):
    sample = sample_conditioned(model, sampler, rand, future_latent, text, 8, seed, num_steps)
    indices = torch.full((sample.shape[0],), -1, device=sample.device, dtype=torch.int64)
    value = codecs.extract_value_from_latent_sequence(sample, indices)
    return torch.clamp((value + 1.0) / 2.0, min=0.0, max=1.0), sample


@torch.inference_mode()
def query(model, vae, sampler, rand, obs, task_embedding, stats, num_steps=5,
          *, return_latent=False, image_transform="vertical", seed=1):
    if image_transform != "vertical":
        raise ValueError("This adapter fixes the official LIBERO vertical image transform")
    observed, _ = encode_observation(vae, obs, stats)
    action, sample = get_action(model, sampler, rand, observed, task_embedding, seed, num_steps)
    return (action, sample) if return_latent else action


@torch.inference_mode()
def best_of_n(model, vae, sampler, rand, obs, text, stats, n=2, seed=1):
    """Native depth-one best-of-N with AR future and AR value, sequentially.

    Text remains canonical for all calls. Each candidate uses seed+index for all
    three calls, as in run_libero_eval.py. No learned or synthetic utility added.
    """
    if n < 1:
        raise ValueError("n must be positive")
    observed, original = encode_observation(vae, obs, stats)
    candidates = []
    for index in range(n):
        candidate_seed = seed + index
        actions, action_latent = get_action(model, sampler, rand, observed, text, candidate_seed, 5)
        future = get_future_state_prediction(model, sampler, rand, action_latent, text, candidate_seed, 1)
        value, value_latent = get_value_prediction(model, sampler, rand, future, text, candidate_seed, 1)
        candidates.append({"index": index, "seed": candidate_seed, "normalized_actions": actions,
                           "actions": normalize_action(actions, stats), "action_latent": action_latent,
                           "future_latent": future, "value": float(value.item())})
        del value_latent
    selected = max(range(n), key=lambda i: candidates[i]["value"])
    return {"selected_index": selected, "candidates": candidates,
            "observed_latent": observed, "original_latent": original}


@torch.inference_mode()
def decode_future_images(vae, future_latent, original_latent):
    """Official decode restores the nonimage placeholders before VAE decoding."""
    sample = future_latent.clone()
    for index in (0, 1, 4, 5):
        sample[:, :, index] = original_latent[:, :, index]
    images = ((vae.decode(sample) + 1.0) * 127.5).clamp(0, 255)
    images = images.permute(0, 2, 3, 4, 1).to(torch.uint8).cpu().numpy()
    return {"wrist": images[:, 21], "primary": images[:, 25]}
