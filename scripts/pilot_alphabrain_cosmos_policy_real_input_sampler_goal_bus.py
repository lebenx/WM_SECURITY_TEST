#!/usr/bin/env python3
"""Two-step-sampler real-image AlphaBrain goal-bus gate.

This is the next gate after ``pilot_alphabrain_cosmos_policy_goal_swap.py``.
It encodes one frozen LIBERO image/state window with the released Cosmos VAE,
then runs the released CosmosPolicy DiT twice with two legal task embeddings.
The image, state, fixed action slot, noise/timestep and condition mask are
identical.  A row-wise goal-bus cut is applied to the same forward pass.

The public AlphaBrain policy checkpoint is available locally, but NVIDIA's
Cosmos VAE is gated.  When that VAE is absent this script writes a structured
preflight record and exits without pretending that random-latent evidence is a
real-input result.  Once the VAE is supplied, the same command becomes the
real-input probe without changing the measurement code.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/AlphaBrain/cosmos_policy_libero/model.safetensors"
TEXT = ROOT / "models/AlphaBrain/cosmos_policy_libero/libero_t5_embeddings.pkl"
DATA = Path(os.environ.get(
    "ALPHABRAIN_REAL_INPUT_DATA_FILE",
    str(ROOT / "data/worldagen_goal_probe_extra/data/chunk-000/file-001.parquet"),
))
if not DATA.is_absolute():
    DATA = ROOT / DATA
OUT = Path(os.environ.get(
    "ALPHABRAIN_REAL_INPUT_SAMPLER_OUT",
    str(ROOT / "runs/alphabrain_cosmos_policy_real_input_goal_bus_v0/summary.json"),
))
PRETRAINED = Path(os.environ.get(
    "ALPHABRAIN_COSMOS_PRETRAINED_DIR",
    str(ROOT / "models/AlphaBrain/Cosmos-Predict2-2B-Video2World"),
))
VAE_CANDIDATES = [
    PRETRAINED / "tokenizer/tokenizer.pth",
    PRETRAINED / "vae/diffusion_pytorch_model.safetensors",
]


def write_result(result: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    available_vae = next((p for p in VAE_CANDIDATES if p.is_file()), None)
    if available_vae is None:
        write_result({
            "status": "preflight_blocked_missing_cosmos_vae",
            "checkpoint": str(MODEL.relative_to(ROOT)),
            "pretrained_dir": str(PRETRAINED.relative_to(ROOT)) if PRETRAINED.is_relative_to(ROOT) else str(PRETRAINED),
            "expected_files": [str(p) for p in VAE_CANDIDATES],
            "data_file_present": DATA.is_file(),
            "scope": "No model forward was run; random-latent AlphaBrain evidence is not upgraded to real-input evidence.",
            "next_action": "Supply tokenizer.pth or a compatible diffusers VAE under the configured pretrained directory, then rerun this script.",
        })
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the real-input gate")

    import pyarrow.parquet as pq

    # Reuse the audited direct-DiT implementation and its native-TE/stub
    # compatibility layer.  Importing it keeps the edge-cut semantics exactly
    # aligned with the fixed-latent gate.
    sys.path.insert(0, str(ROOT / "scripts"))
    import pilot_alphabrain_cosmos_policy_goal_swap as fixed

    if os.environ.get("ALPHABRAIN_USE_NATIVE_TE") != "1":
        fixed.install_transformer_engine_stub()
    sys.path.insert(0, str(ROOT / "vendor/AlphaBrain"))
    from AlphaBrain.model.modules.world_model.cosmos.latent_utils import (
        replace_latent_with_action_chunk,
        replace_latent_with_proprio,
    )
    from AlphaBrain.model.modules.world_model.cosmos.wan_vae import WanVAEWrapper
    from AlphaBrain.model.modules.world_model.cosmos.cosmos_sampler import CosmosPolicySampler
    from AlphaBrain.model.modules.world_model.cosmos.noise_utils import arch_invariant_rand
    from AlphaBrain.model.modules.world_model.cosmos.official_dit.minimal_v1_lvg_dit import MinimalV1LVGDiT
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
    if missing or unexpected:
        raise RuntimeError(f"state mismatch missing={len(missing)} unexpected={len(unexpected)}")
    del sd

    rows = pq.read_table(DATA).to_pylist()
    target_episode = int(os.environ.get("ALPHABRAIN_REAL_INPUT_EPISODE", "-1"))
    target_frame = int(os.environ.get("ALPHABRAIN_REAL_INPUT_FRAME", "30"))
    candidates = [r for r in rows if r["frame_index"] >= target_frame]
    if target_episode >= 0:
        candidates = [r for r in candidates if int(r["episode_index"]) == target_episode]
    if not candidates:
        raise RuntimeError(f"no row for episode={target_episode}, frame>={target_frame}")
    row = min(candidates, key=lambda r: (int(r["episode_index"]), int(r["frame_index"])))
    # Use the recorded executed action chunk rather than a zero placeholder.
    # This keeps the physical action condition fixed across the two goals and
    # matches the threat model's fixed-state/fixed-action intervention.
    episode_rows = sorted(
        (r for r in rows if int(r["episode_index"]) == int(row["episode_index"])),
        key=lambda r: int(r["frame_index"]),
    )
    action_rows = [r for r in episode_rows if int(r["frame_index"]) >= int(row["frame_index"])][:16]
    if not action_rows:
        raise RuntimeError("no executed actions for selected episode/frame")
    action_chunk = np.asarray([r["action"] for r in action_rows], dtype=np.float32)
    action_padded = len(action_rows) < 16
    if action_padded:
        action_chunk = np.concatenate(
            [action_chunk, np.repeat(action_chunk[-1:, :], 16 - len(action_rows), axis=0)], axis=0
        )
    # A second recorded action chunk from the same episode is used only by
    # the optional action-sensitivity control. It is never used in the goal
    # attack aggregate.
    action_rows_alt = [
        r for r in episode_rows
        if int(r["frame_index"]) >= int(row["frame_index"]) + 16
    ][:16]
    if len(action_rows_alt) < 1:
        raise RuntimeError("no alternate recorded action chunk for control")
    action_chunk_alt = np.asarray([r["action"] for r in action_rows_alt], dtype=np.float32)
    action_alt_padded = len(action_rows_alt) < 16
    if action_alt_padded:
        action_chunk_alt = np.concatenate(
            [action_chunk_alt, np.repeat(action_chunk_alt[-1:, :], 16 - len(action_rows_alt), axis=0)], axis=0
        )
    task_rows = pq.read_table(DATA.parents[2] / "meta/tasks.parquet").to_pylist()
    task_names = [r.get("task", r.get("__index_level_0__")) for r in task_rows]
    task_a = task_names[int(row["task_index"])]
    task_offset = int(os.environ.get("ALPHABRAIN_REAL_INPUT_GOAL_OFFSET", "5"))
    task_b = task_names[(int(row["task_index"]) + task_offset) % len(task_names)]

    def image(key):
        return Image.open(BytesIO(row[key]["bytes"])).convert("RGB").transpose(Image.Transpose.ROTATE_180)

    primary = np.array(image("observation.images.image").resize((224, 224)), copy=True)
    wrist = np.array(image("observation.images.wrist_image").resize((224, 224)), copy=True)
    # Match CosmosPolicy's 33-pixel-frame layout: one blank plus eight groups
    # of four repeated frames.  Future visual slots are held equal to current
    # images, so the only intervention is task text.
    primary = torch.from_numpy(primary).permute(2, 0, 1).unsqueeze(1).expand(3, 4, 224, 224)
    wrist = torch.from_numpy(wrist).permute(2, 0, 1).unsqueeze(1).expand(3, 4, 224, 224)
    blank = torch.zeros_like(primary)
    video = torch.cat([blank[:, :1], blank, wrist, primary, blank, blank, wrist, primary, blank], dim=1).unsqueeze(0).cuda()
    vae = WanVAEWrapper(str(PRETRAINED), dtype=torch.bfloat16, device="cuda", temporal_window=16).eval()
    with torch.inference_mode():
        encoded = vae.encode(video.to(dtype=torch.bfloat16) / 127.5 - 1.0)
    # Clone outside inference_mode so the fixed action/proprio injections are
    # ordinary tensors and can be written in place by latent_utils.
    latent = encoded.to(torch.bfloat16).clone()
    raw_state = np.asarray(row["observation.state"], dtype=np.float32)
    # LIBERO parquet state is [eef_pos(3), axis-angle(3), gripper(2)].
    # CosmosPolicy conditions on [gripper(2), eef_pos(3), quaternion(4)].
    quat = Rotation.from_rotvec(raw_state[3:6]).as_quat().astype(np.float32)
    cosmos_state = np.concatenate([raw_state[6:8], raw_state[:3], quat])
    proprio = torch.tensor(np.asarray([cosmos_state]), device="cuda", dtype=torch.bfloat16)
    action = torch.tensor(action_chunk, device="cuda", dtype=torch.bfloat16).unsqueeze(0)
    latent = replace_latent_with_proprio(latent, proprio, torch.tensor([1], device="cuda"))
    latent = replace_latent_with_action_chunk(latent, action, torch.tensor([4], device="cuda"))
    latent_alt = encoded.to(torch.bfloat16).clone()
    latent_alt = replace_latent_with_proprio(latent_alt, proprio, torch.tensor([1], device="cuda"))
    action_alt = torch.tensor(action_chunk_alt, device="cuda", dtype=torch.bfloat16).unsqueeze(0)
    latent_alt = replace_latent_with_action_chunk(latent_alt, action_alt, torch.tensor([4], device="cuda"))
    cond = torch.zeros((1, 1, 9, 28, 28), device="cuda", dtype=torch.bfloat16)
    cond[:, :, :4] = 1
    timestep = torch.full((1, 9), 0.45, device="cuda", dtype=torch.bfloat16)
    fps = torch.tensor([16], device="cuda", dtype=torch.bfloat16)
    padding = torch.zeros((1, 1, 224, 224), device="cuda", dtype=torch.bfloat16)
    text = pickle.load(open(TEXT, "rb"))
    a, b = text[task_a].cuda().to(torch.bfloat16), text[task_b].cuda().to(torch.bfloat16)
    zero = torch.zeros_like(a)
    sampler = CosmosPolicySampler()
    # Hold the recorded action latent fixed as well as the four observation
    # condition slots.  This makes the sampler probe a true fixed-state /
    # fixed-action counterfactual rather than a jointly regenerated action.
    gt_frames = latent[:, :, :5, :, :].float().clone()
    cond[:, :, :5] = 1
    x_sigma_max = arch_invariant_rand(tuple(latent.shape), seed=1).to(device="cuda", dtype=torch.float32) * 80.0

    def sampled(ctx, latent_in=latent):
        gt_frames_local = latent_in[:, :, :5, :, :].float().clone()
        def x0_fn(xt, sigma):
            sigma = sigma.float()
            sigma5 = sigma.view(-1, 1, 1, 1, 1)
            t = sigma5 / (sigma5 + 1.0)
            net_input = xt.float() * (1.0 - t)
            net_input[:, :, :5] = gt_frames_local
            frame_noise = (sigma / (sigma + 1.0)).view(-1, 1).expand(-1, 9)
            cond_mask = cond.to(dtype=torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                net_output = model(
                    net_input.to(torch.bfloat16), frame_noise.to(torch.bfloat16), ctx,
                    cond_mask, fps=fps, padding_mask=padding
                )
            x0 = (1.0 - t) * xt.float() - t * net_output.float()
            x0[:, :, :5] = gt_frames_local
            return x0
        return sampler.forward(x0_fn=x0_fn, x_sigma_max=x_sigma_max,
                               num_steps=2, sigma_min=4.0, sigma_max=80.0)

    with torch.inference_mode():
        ya = sampled(a)
        yb = sampled(b)
        yz = sampled(zero)
        action_control = os.environ.get("ALPHABRAIN_ACTION_CONTROL", "0") == "1"
        native_action_alt = sampled(a, latent_alt) if action_control else None
        cut_mode = os.environ.get("ALPHABRAIN_GOAL_BUS_CUT_MODE", "full")
        if cut_mode == "world_text_only":
            fixed.install_world_text_only_cut(model)
        elif cut_mode == "fixed_action_world":
            fixed.install_fixed_action_world_cut(model)
        else:
            fixed.install_goal_bus_cut(model)
        ca = sampled(a)
        cb = sampled(b)
        cz = sampled(zero)
        cut_action_alt = sampled(a, latent_alt) if action_control else None
    native = (ya.float() - yb.float()).abs().mean(dim=(0, 1, 3, 4))
    native_zero = (ya.float() - yz.float()).abs().mean(dim=(0, 1, 3, 4))
    cut = (ca.float() - cb.float()).abs().mean(dim=(0, 1, 3, 4))
    cut_zero = (ca.float() - cz.float()).abs().mean(dim=(0, 1, 3, 4))
    action_control_fields = {}
    if native_action_alt is not None:
        native_action_delta = (ya.float() - native_action_alt.float()).abs().mean(dim=(0, 1, 3, 4))
        cut_action_delta = (ca.float() - cut_action_alt.float()).abs().mean(dim=(0, 1, 3, 4))
        action_control_fields = {
            "action_control_enabled": True,
            "alternate_action_source": "same_episode_recorded_chunk_start_plus_16",
            "alternate_action_chunk_frame_start": int(action_rows_alt[0]["frame_index"]),
            "alternate_action_chunk_frame_end": int(action_rows_alt[-1]["frame_index"]),
            "native_action_world_delta_mean": float(native_action_delta[5:8].mean()),
            "cut_action_world_delta_mean": float(cut_action_delta[5:8].mean()),
            "native_action_per_slot_mean_abs_delta": native_action_delta.tolist(),
            "cut_action_per_slot_mean_abs_delta": cut_action_delta.tolist(),
        }
    write_result({
        "status": "completed_alphabrain_real_input_sampler_goal_bus_gate",
        "runtime_backend": "native_transformer_engine" if os.environ.get("ALPHABRAIN_USE_NATIVE_TE") == "1" else "local_transformer_engine_compatibility_stub",
        "cut_mode": os.environ.get("ALPHABRAIN_GOAL_BUS_CUT_MODE", "full"),
        "vae_source": str(available_vae.relative_to(ROOT)) if available_vae.is_relative_to(ROOT) else str(available_vae),
        "task_a": task_a, "task_b": task_b, "task_offset": task_offset,
        "episode": int(row["episode_index"]), "frame": int(row["frame_index"]),
        "data_file": str(DATA.relative_to(ROOT)),
        "action_source": "recorded_executed_action_chunk",
        "action_chunk_frame_start": int(action_rows[0]["frame_index"]),
        "action_chunk_frame_end": int(action_rows[-1]["frame_index"]),
        "action_chunk_padded_last_value": action_padded,
        "action_chunk_l2": float(np.linalg.norm(action_chunk)),
        "latent_shape": list(latent.shape), "native_per_slot_mean_abs_delta": native.tolist(),
        "cut_per_slot_mean_abs_delta": cut.tolist(),
        "native_future_world_delta_mean": float(native[5:8].mean()),
        "cut_future_world_delta_mean": float(cut[5:8].mean()),
        "cut_policy_delta_mean": float(cut[[4, 8]].mean()),
        "native_zero_context_future_world_delta_mean": float(native_zero[5:8].mean()),
        "native_zero_context_policy_delta_mean": float(native_zero[[4, 8]].mean()),
        "cut_zero_context_future_world_delta_mean": float(cut_zero[5:8].mean()),
        "cut_zero_context_policy_delta_mean": float(cut_zero[[4, 8]].mean()),
        **action_control_fields,
        "scope": "One real LIBERO image/state window; observation and recorded executed action slots fixed across goals; two-step Cosmos sampler, no environment rollout. Optional action control compares two recorded same-episode chunks at the same current state.",
    })


if __name__ == "__main__":
    main()
