#!/usr/bin/env python3
"""Native/edge-cut AlphaBrain CosmosPolicy LIBERO control probe.

It uses the released Cosmos sampler and VAE, executes the sampled actions in
LIBERO, and compares the native goal-bus path with the matched world-row cut.
The task id, suite, and horizon are configurable for short or full-horizon
capability gates; it is not a task success-rate benchmark.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/AlphaBrain/cosmos_policy_libero/model.safetensors"
TEXT = ROOT / "models/AlphaBrain/cosmos_policy_libero/libero_t5_embeddings.pkl"
STATS = ROOT / "models/AlphaBrain/cosmos_policy_libero/libero_dataset_statistics.json"
PRETRAINED = ROOT / "models/AlphaBrain/Cosmos-Predict2-2B-Video2World"
OUT = Path(os.environ.get(
    "ALPHABRAIN_LIBERO_CONTROL_OUT",
    str(ROOT / "runs/alphabrain_cosmos_libero_short_control_v0/summary.json"),
))
LIBERO = ROOT / "vendor/LIBERO"


def load_model():
    if os.environ.get("ALPHABRAIN_USE_NATIVE_TE") != "1":
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "scripts"))
        import pilot_alphabrain_cosmos_policy_goal_swap as fixed
        fixed.install_transformer_engine_stub()
    sys.path.insert(0, str(ROOT / "vendor/AlphaBrain"))
    from AlphaBrain.model.modules.world_model.cosmos.official_dit.minimal_v1_lvg_dit import MinimalV1LVGDiT
    from AlphaBrain.model.modules.world_model.cosmos.wan_vae import WanVAEWrapper
    from AlphaBrain.model.modules.world_model.cosmos.cosmos_sampler import CosmosPolicySampler
    from AlphaBrain.model.modules.world_model.cosmos.noise_utils import arch_invariant_rand
    from safetensors.torch import load_file

    model = MinimalV1LVGDiT(
        max_img_h=240, max_img_w=240, max_frames=128, in_channels=16,
        out_channels=16, patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True, model_channels=2048, num_blocks=28,
        num_heads=16, crossattn_emb_channels=1024, use_crossattn_projection=False,
        atten_backend="torch", pos_emb_cls="rope3d", mlp_ratio=4.0,
        use_adaln_lora=True, adaln_lora_dim=256, use_wan_fp32_strategy=False,
        pos_emb_learnable=True, rope_enable_fps_modulation=False,
        rope_h_extrapolation_ratio=3.0, rope_w_extrapolation_ratio=3.0,
    ).cuda().to(dtype=torch.bfloat16).eval()
    sd = load_file(str(MODEL), device="cpu")
    sd = {k.removeprefix("dit."): v for k, v in sd.items() if k.startswith("dit.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Backend metadata buffers differ between the torch fallback and
    # Transformer-Engine builds.  Learned checkpoint parameters are all
    # present; missing learned keys remain a hard failure.
    if missing:
        raise RuntimeError(f"checkpoint mismatch missing={len(missing)} unexpected={len(unexpected)}")
    del sd
    return model, WanVAEWrapper(str(PRETRAINED), dtype=torch.bfloat16, device="cuda", temporal_window=16).eval(), CosmosPolicySampler(), arch_invariant_rand


def install_cut(model):
    sys.path.insert(0, str(ROOT / "scripts"))
    import pilot_alphabrain_cosmos_policy_goal_swap as fixed
    # Keep the action carrier available to the world branch.  The historical
    # full cut also removed self-attention from action/value rows and is kept
    # only as a legacy over-cut comparator in the report.
    fixed.install_fixed_action_world_cut(model)


def action_from_latent(action_latent):
    b = action_latent.shape[0]
    flat = action_latent.reshape(b, -1)
    chunk = 16 * 7
    n = flat.shape[1] // chunk
    return flat[:, :n * chunk].reshape(b, n, 16, 7).mean(dim=1)


def normalize_action(action, stats):
    lo = np.asarray(stats["actions_min"], dtype=np.float32)
    hi = np.asarray(stats["actions_max"], dtype=np.float32)
    a = action.detach().float().cpu().numpy()
    return np.clip(0.5 * (a + 1.0) * (hi - lo) + lo, -1.0, 1.0)


def _official_preprocess_image(image):
    """Match CosmosPolicy._preprocess_image used by the released server."""
    import io
    from PIL import Image as PILImage
    import torchvision.transforms.functional as F_tv
    image = np.asarray(image, dtype=np.uint8)
    pil = PILImage.fromarray(image)
    buf = io.BytesIO(); pil.save(buf, format="JPEG", quality=95); buf.seek(0)
    pil = PILImage.open(buf).resize((224, 224))
    tensor = torch.from_numpy(np.array(pil)).permute(2, 0, 1)
    crop_size = int(224 * 0.9 ** 0.5)
    tensor = F_tv.center_crop(tensor, crop_size)
    tensor = F_tv.resize(tensor, [224, 224], antialias=True)
    return tensor.permute(1, 2, 0).numpy().astype(np.uint8)


def query(model, vae, sampler, rand, obs, task_embedding, stats, num_steps=5):
    primary = np.ascontiguousarray(obs["agentview_image"][::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    primary = _official_preprocess_image(primary)
    wrist = _official_preprocess_image(wrist)
    p = torch.from_numpy(primary).permute(2, 0, 1).unsqueeze(1).expand(3, 4, 224, 224)
    w = torch.from_numpy(wrist).permute(2, 0, 1).unsqueeze(1).expand(3, 4, 224, 224)
    blank = torch.zeros_like(p)
    video = torch.cat([blank[:, :1], blank, w, p, blank, blank, w, p, blank], dim=1).unsqueeze(0).cuda()
    with torch.inference_mode():
        encoded = vae.encode(video.to(torch.bfloat16) / 127.5 - 1.0)
    latent = encoded.to(torch.bfloat16).clone()
    state = np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]]).astype(np.float32)
    p_min = np.asarray(stats["proprio_min"], dtype=np.float32)
    p_max = np.asarray(stats["proprio_max"], dtype=np.float32)
    state = 2.0 * (state - p_min) / np.where(np.abs(p_max - p_min) < 1e-8, 1.0, p_max - p_min) - 1.0
    # Cosmos proprio is injected by tiling the 9-vector into latent frame 1.
    from AlphaBrain.model.modules.world_model.cosmos.latent_utils import replace_latent_with_proprio
    latent = replace_latent_with_proprio(latent, torch.tensor(state[None], device="cuda", dtype=torch.bfloat16), torch.tensor([1], device="cuda"))
    gt = latent[:, :, :4].float().clone()
    cond = torch.zeros((1, 1, 9, 28, 28), device="cuda", dtype=torch.bfloat16)
    cond[:, :, :4] = 1
    padding = torch.zeros((1, 1, 224, 224), device="cuda", dtype=torch.bfloat16)
    fps = torch.tensor([16], device="cuda", dtype=torch.float32)
    noise = rand(tuple(latent.shape), seed=1).to(device="cuda", dtype=torch.float32) * 80.0

    def x0_fn(xt, sigma):
        sigma = sigma.float()
        sigma5 = sigma.view(-1, 1, 1, 1, 1)
        t = sigma5 / (sigma5 + 1.0)
        net_input = xt.float() * (1.0 - t)
        net_input[:, :, :4] = gt
        frame_noise = (sigma / (sigma + 1.0)).view(-1, 1).expand(-1, 9)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(net_input.to(torch.bfloat16), frame_noise.to(torch.bfloat16), task_embedding, cond, fps=fps, padding_mask=padding)
        x0 = (1.0 - t) * xt.float() - t * out.float()
        x0[:, :, :4] = gt
        return x0

    with torch.inference_mode():
        denoised = sampler.forward(x0_fn=x0_fn, x_sigma_max=noise, num_steps=num_steps, sigma_min=4.0, sigma_max=80.0)
    return action_from_latent(denoised[:, :, 4]).detach()


def run_condition(model, vae, sampler, rand, task, suite_name, condition, embedding, stats, max_chunks=2, num_steps=5):
    from libero.libero.envs import OffScreenRenderEnv
    bddl = LIBERO / "libero/libero/bddl_files" / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256, render_gpu_device_id=0)
    env.seed(0); env.reset(); env.task_id = 0; env.task_name = task.name; env.task_suite_name = suite_name
    states = torch.load(LIBERO / "libero/libero/init_files" / task.problem_folder / task.init_states_file, map_location="cpu", weights_only=False)
    obs = env.set_init_state(states[0])
    for _ in range(5): obs, *_ = env.step(np.zeros(7, dtype=np.float32))
    success = False; steps = 0; norms = []
    while steps < max_chunks * 16:
        norm_action = query(model, vae, sampler, rand, obs, embedding, stats, num_steps=num_steps)
        actions = normalize_action(norm_action, stats)[0]
        norms.append(float(np.linalg.norm(actions)))
        for action in actions:
            obs, reward, done, info = env.step(action.tolist())
            steps += 1
            if done:
                success = True; break
        if success: break
    env.close()
    return {"suite": suite_name, "task": task.language, "condition": condition, "success": success, "environment_steps": steps, "action_norm_mean": float(np.mean(norms))}


def main():
    if not torch.cuda.is_available(): raise SystemExit("CUDA required")
    started = time.monotonic()
    # Import after paths are set so the vendored LIBERO package is used.
    sys.path.insert(0, str(LIBERO))
    from libero.libero import benchmark
    model, vae, sampler, rand = load_model()
    text = pickle.load(open(TEXT, "rb"))
    stats = json.loads(STATS.read_text())
    suite_name = os.environ.get("ALPHABRAIN_LIBERO_SUITE", "libero_spatial")
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task_id = int(os.environ.get("ALPHABRAIN_LIBERO_TASK_ID", "0"))
    task = suite.get_task(task_id)
    embedding = text[task.language].cuda().to(torch.bfloat16)
    max_chunks = int(os.environ.get("ALPHABRAIN_LIBERO_MAX_CHUNKS", "2"))
    num_steps = int(os.environ.get("ALPHABRAIN_LIBERO_NUM_STEPS", "5"))
    if max_chunks < 1:
        raise ValueError("ALPHABRAIN_LIBERO_MAX_CHUNKS must be positive")
    results = [run_condition(model, vae, sampler, rand, task, suite_name, "native", embedding, stats, max_chunks=max_chunks, num_steps=num_steps)]
    install_cut(model)
    results.append(run_condition(model, vae, sampler, rand, task, suite_name, "fixed_action_world_cut", embedding, stats, max_chunks=max_chunks, num_steps=num_steps))
    out = {"status":"completed_alphabrain_libero_short_control", "suite":suite_name, "task_id":task_id, "task":task.language, "conditions":["native","fixed_action_world_cut"], "cut_mode":"fixed_action_world", "max_chunks":max_chunks, "sampler_num_steps":num_steps, "results":results, "elapsed_seconds":time.monotonic()-started, "scope":"One initial state; official image preprocessing, proprio normalization, and sampler step count; corrected world-branch cut; short or full-horizon control evaluation depending on max_chunks."}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n"); print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
