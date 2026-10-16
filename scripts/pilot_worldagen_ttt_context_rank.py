#!/usr/bin/env python3
"""Minimal native WorldAgen/LIBERO TTT context-rank gate.

This is deliberately a short, real-environment screen. It uses the released
WorldAgen model/checkpoint, real LIBERO observations, and the same action
self-supervision used by the repository's LIBERO training loop. The upstream
checkout currently lacks ``test_time_training/ttt_libero.py``, so this script
does the small equivalent adaptation directly on the Qwen backbone.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from models.model import WorldAgen
import clip


ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda:0"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def task_env(task, suite_name: str, image_size: int = 128):
    bddl = ROOT / "vendor" / "LIBERO" / "libero" / "libero" / "bddl_files" / task.problem_folder / task.bddl_file
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=image_size,
        camera_widths=image_size,
        render_gpu_device_id=0,
    )


def init_env(env, task, exp_id: int):
    env.reset()
    init_path = ROOT / "vendor" / "LIBERO" / "libero" / "libero" / "init_files" / task.problem_folder / task.init_states_file
    states = torch.load(init_path, map_location="cpu")
    obs = env.set_init_state(states[exp_id])
    for _ in range(5):
        env.step(np.zeros(7, dtype=np.float32))
    return obs


def make_model():
    model = WorldAgen(
        clip_device=DEVICE,
        vit_checkpoint_path=str(ROOT / "models" / "WorldAgen" / "mae_pretrain_vit_base.pth"),
        sequence_length=2,
        image_sequence_length=1,
        state_sequence_length=1,
        action_sequence_length=3,
        num_resampler_query=6,
        num_obs_token_per_image=9,
        calvin_input_image_size=224,
        patch_size=16,
        transformer_layers=24,
        hidden_dim=384,
        transformer_heads=12,
        phase="evaluate",
        gripper_width=False,
        pred_state=False,
        action_pred_steps=1,
        pred_image=True,
        atten_goal=0,
        use_qwen=True,
    )
    checkpoint = torch.load(ROOT / "models" / "WorldAgen" / "libero-38.pth", map_location="cpu")
    state = {k.removeprefix("module."): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE)
    model.clip_model.requires_grad_(False)
    model.vision_encoder.requires_grad_(False)
    model._init_model_type()
    return model.eval()


def obs_tensors(model, obs, task_text: str):
    proc = model.image_processor
    primary = proc(Image.fromarray(obs["agentview_image"]))
    wrist = proc(Image.fromarray(obs["robot0_eye_in_hand_image"]))
    euler = Rotation.from_quat(obs["robot0_eef_quat"]).as_euler("xyz")
    state = np.concatenate([obs["robot0_eef_pos"], euler, np.array([0.0], dtype=np.float32)])
    text = clip.tokenize([task_text], truncate=True)[0]
    return primary, wrist, torch.from_numpy(state).float(), text


def legal_probe_action(step: int, shuffled: bool = False) -> np.ndarray:
    # Small Cartesian/orientation oscillations are legal robosuite actions;
    # gripper remains open and no reward or task predicate is used.
    phase = step if not shuffled else (step * 7 + 3) % 10
    action = np.zeros(7, dtype=np.float32)
    action[0] = 0.35 * (1.0 if phase % 2 == 0 else -1.0)
    action[1] = 0.20 * np.sin(phase * np.pi / 2.0)
    action[3] = 0.15 * np.cos(phase * np.pi / 2.0)
    action[6] = -1.0
    return action


def pack_window(model, obs_list, action_list, task_text: str):
    # sequence_length=2, action_sequence_length=3: two image/state tokens and
    # six observed actions, matching the WorldAgen LIBERO training shape.
    values = [obs_tensors(model, obs, task_text) for obs in obs_list]
    primary = torch.stack([v[0] for v in values]).unsqueeze(0).to(DEVICE)
    wrist = torch.stack([v[1] for v in values]).unsqueeze(0).to(DEVICE)
    state = torch.stack([v[2] for v in values]).unsqueeze(0).to(DEVICE)
    text = torch.stack([v[3] for v in values]).unsqueeze(0).to(DEVICE)
    actions = torch.from_numpy(np.asarray(action_list, dtype=np.float32)).unsqueeze(0).to(DEVICE)
    return primary, wrist, state, text, actions


def forward_loss(model, batch, capture=False):
    primary, wrist, state, text, actions = batch
    hidden = []

    def hook(_module, _inputs, output):
        value = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        hidden.append(value.detach().float().mean(dim=1))

    handle = model.transformer_backbone.register_forward_hook(hook) if capture else None
    arm, grip, _image, _state_arm, _state_grip = model(
        image_primary=primary,
        image_wrist=wrist,
        state=state,
        text_token=text,
        action=actions,
    )
    if handle is not None:
        handle.remove()
    labels = actions.view(1, 2, 3, 7)
    loss_arm = F.smooth_l1_loss(arm, labels[..., :6])
    # Actions use {-1, +1}; the WorldAgen decoder predicts Bernoulli gripper.
    loss_grip = F.binary_cross_entropy(grip, ((labels[..., 6:] + 1.0) / 2.0).clamp(0, 1))
    return loss_arm + 0.01 * loss_grip, hidden[0] if hidden else None


def collect_probe(model, task, suite_name: str, exp_id: int, n_steps: int, shuffled: bool):
    env = task_env(task, suite_name)
    obs = init_env(env, task, exp_id)
    observations = [copy.deepcopy(obs)]
    actions = []
    rewards = []
    for i in range(n_steps):
        action = legal_probe_action(i, shuffled=shuffled)
        nxt, reward, done, _info = env.step(action)
        actions.append(action)
        rewards.append(float(reward))
        observations.append(copy.deepcopy(nxt))
        if done:
            break
    env.close()
    return observations, actions, rewards


def build_windows(model, observations, actions, task_text):
    windows = []
    # six actions correspond to two 3-action prediction chunks; use two
    # overlapping windows when the 10-step probe permits it.
    for start in range(max(1, len(actions) - 6)):
        if start + 6 > len(actions) or start + 2 > len(observations):
            continue
        windows.append(pack_window(model, observations[start : start + 2], actions[start : start + 6], task_text))
    return windows


def adapt(model, windows, steps: int, lr: float):
    # Native TTT in the released scripts adapts the causal backbone with LoRA;
    # this short gate uses the backbone itself, which is the conservative
    # equivalent for a deterministic screen and keeps the loss auditable.
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.transformer_backbone.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.transformer_backbone.parameters(), lr=lr, weight_decay=0.01)
    losses = []
    model.train()
    for i in range(steps):
        batch = windows[i % len(windows)]
        optimizer.zero_grad(set_to_none=True)
        loss, _ = forward_loss(model, batch, capture=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.transformer_backbone.parameters(), 0.1)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    return losses


def spectral_entropy(features: np.ndarray) -> float:
    if features.shape[0] < 2:
        return 0.0
    x = features - features.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    p = (s * s) / max(float((s * s).sum()), 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(max(2, len(p))))


def latent_metrics(model, windows):
    feats = []
    losses = []
    with torch.no_grad():
        for batch in windows:
            loss, hidden = forward_loss(model, batch, capture=True)
            losses.append(float(loss.cpu()))
            feats.append(hidden.cpu().numpy().reshape(-1))
    return {"loss": float(np.mean(losses)), "spectral_entropy": spectral_entropy(np.asarray(feats))}


def policy_action(model, obs, task_text):
    primary, wrist, state, text = obs_tensors(model, obs, task_text)
    primary = primary.unsqueeze(0).repeat(1, 2, 1, 1, 1).to(DEVICE)
    wrist = wrist.unsqueeze(0).repeat(1, 2, 1, 1, 1).to(DEVICE)
    state = state.unsqueeze(0).repeat(1, 2, 1).to(DEVICE)
    text = text.unsqueeze(0).repeat(1, 2, 1).to(DEVICE)
    actions = torch.zeros(1, 6, 7, device=DEVICE)
    with torch.no_grad():
        arm, grip, _image, _state_arm, _state_grip = model(primary, wrist, state, text, actions)
        out = torch.cat([arm[0, 0], torch.where(grip[0, 0] > 0.5, 1.0, -1.0)], dim=-1)
    return out.detach().cpu().numpy().astype(np.float32)


def evaluate(model, task, suite_name: str, exp_id: int, max_steps: int):
    env = task_env(task, suite_name)
    obs = init_env(env, task, exp_id)
    total_reward = 0.0
    success = 0
    task_text = task.language
    n = 0
    while n < max_steps:
        action = policy_action(model, obs, task_text)
        for a in action.reshape(3, 7):
            obs, reward, done, _info = env.step(np.clip(a, -1, 1))
            total_reward += float(reward)
            n += 1
            if done:
                success = 1
                break
        if success:
            break
    env.close()
    return {"success": success, "reward": total_reward, "steps": n}


def run_condition(task, suite_name, exp_id, mode, args):
    seed_all(args.seed)
    model = make_model()
    task_text = task.language
    probe_obs, probe_actions, probe_rewards = collect_probe(
        model, task, suite_name, exp_id, args.probe_steps, shuffled=(mode == "shuffled_probe")
    )
    windows = build_windows(model, probe_obs, probe_actions, task_text)
    if not windows:
        raise RuntimeError("probe did not produce a valid six-action native TTT window")
    pre = latent_metrics(model, windows)
    train_losses = []
    if mode == "oscillatory_ttt":
        train_losses = adapt(model, windows, args.ttt_steps, args.learning_rate)
    post = latent_metrics(model, windows)
    evaluation = evaluate(model, task, suite_name, exp_id, args.eval_steps)
    return {
        "mode": mode,
        "pre": pre,
        "post": post,
        "train_losses": train_losses,
        "probe_reward_sum": float(sum(probe_rewards)),
        "probe_action_l2": float(np.mean(np.square(np.asarray(probe_actions)))),
        "evaluation": evaluation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--exp-id", type=int, default=0)
    parser.add_argument("--probe-steps", type=int, default=10)
    parser.add_argument("--ttt-steps", type=int, default=3)
    parser.add_argument("--eval-steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/worldagen_ttt_context_rank_v0/status.json")
    args = parser.parse_args()
    suite_name = "libero_10"
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(args.task_id)
    started = time.time()
    results = []
    for mode in ("baseline", "oscillatory_ttt", "shuffled_probe"):
        print(f"[C1] running {mode}", flush=True)
        results.append(run_condition(task, suite_name, args.exp_id, mode, args))
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    attack = next(x for x in results if x["mode"] == "oscillatory_ttt")
    baseline = next(x for x in results if x["mode"] == "baseline")
    one_step_ratio = attack["post"]["loss"] / max(baseline["pre"]["loss"], 1e-12)
    entropy_ratio = attack["post"]["spectral_entropy"] / max(attack["pre"]["spectral_entropy"], 1e-12)
    status = {
        "schema_version": 1,
        "candidate": "C1_TTT_Context_Rank_Parasitism",
        "task": {"suite": suite_name, "task_id": args.task_id, "name": task.name, "language": task.language, "exp_id": args.exp_id},
        "config": vars(args),
        "results": results,
        "gate_metrics": {
            "one_step_loss_ratio_post_vs_baseline_pre": one_step_ratio,
            "latent_spectral_entropy_ratio_post_vs_attack_pre": entropy_ratio,
            "entropy_drop_fraction": 1.0 - entropy_ratio,
            "completion_drop_vs_baseline": float(baseline["evaluation"]["success"] - attack["evaluation"]["success"]),
        },
        "gates": {
            "one_step_loss_le_1_15": bool(one_step_ratio <= 1.15),
            "spectral_entropy_drop_gt_50pct": bool(entropy_ratio < 0.5),
            "baseline_success_observed": bool(baseline["evaluation"]["success"]),
            "completion_drop_gt_60pct": bool(baseline["evaluation"]["success"] and not attack["evaluation"]["success"]),
        },
        "runtime_seconds": time.time() - started,
        "limitations": [
            "Upstream WorldAgen checkout lacks test_time_training/ttt_libero.py; direct backbone adaptation is used for this screen.",
            "One LIBERO task and one initial state are a falsification screen, not a cross-task claim.",
        ],
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
