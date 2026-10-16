#!/usr/bin/env python3
"""Short native LIBERO task baseline for the WorldAgen goal-bus gate.

Runs one fixed initial state from each of three released LIBERO task suites
under the native mask, the nine-edge goal-bus cut, and an empty instruction.
This is a task-level sanity check for the frozen policy; it is not an attack
success claim and uses no training or environment-side intervention.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import types
import importlib.machinery

_omegaconf = types.ModuleType("omegaconf")
_omegaconf.DictConfig = dict
_omegaconf.__spec__ = importlib.machinery.ModuleSpec("omegaconf", loader=None)
sys.modules.setdefault("omegaconf", _omegaconf)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "LIBERO"))
sys.path.insert(0, str(ROOT / "vendor" / "WorldAgen"))

from libero.libero import benchmark  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from PIL import Image  # noqa: E402
from models.model import WorldAgen  # noqa: E402
import utils.calvin_data_utils as _data_utils  # noqa: E402
sys.modules.setdefault("utils.data_utils", _data_utils)
from utils.eval_utils_libero import (  # noqa: E402
    WorldAgenLiberoModelWrapper,
    get_robot_info_from_multi_step_env_libero,
    quaternion_to_euler,
)
from utils.data_utils import preprocess_image, preprocess_text_calvin  # noqa: E402
import clip  # noqa: E402

OUT = ROOT / "runs/worldagen_libero_closed_loop_v0/summary.json"
MODEL_DIR = ROOT / "models/WorldAgen"
LIBERO_ROOT = ROOT / "vendor/LIBERO"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model() -> WorldAgen:
    original = clip.load

    def local_clip_load(_name, device, **kwargs):
        return original(str(MODEL_DIR / "ViT-B-32.pt"), device=device, **kwargs)

    from unittest.mock import patch
    with patch.object(clip, "load", side_effect=local_clip_load):
        model = WorldAgen(
            clip_device=DEVICE,
            vit_checkpoint_path=str(MODEL_DIR / "mae_pretrain_vit_base.pth"),
            sequence_length=2,
            image_sequence_length=1,
            state_sequence_length=1,
            action_sequence_length=3,
            num_resampler_query=6,
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
    checkpoint = torch.load(MODEL_DIR / "libero-38.pth", map_location="cpu", weights_only=True)
    saved = {k.removeprefix("module."): v for k, v in checkpoint["model_state_dict"].items()}
    loaded = model.load_state_dict(saved, strict=False)
    expected_missing = {k for k in model.state_dict() if k.startswith(("vision_encoder.", "clip_model."))}
    expected_missing.add("image_decoder_position_embedding")
    if loaded.unexpected_keys or set(loaded.missing_keys) != expected_missing:
        raise RuntimeError(f"checkpoint mismatch: {loaded.missing_keys}, {loaded.unexpected_keys}")
    model = model.to(DEVICE).float().eval().requires_grad_(False)
    model._init_model_type()
    return model


def masks(model: WorldAgen):
    native = model.attention_mask.detach().clone()
    cut = native.clone()
    actions = torch.tensor([2, 3, 4, 42, 43, 44])
    goals = torch.tensor([0, 40])
    cut[actions[:, None], goals[None, :]] = -float("inf")
    if int((torch.isfinite(native) & ~torch.isfinite(cut)).sum()) != 9:
        raise RuntimeError("unexpected attention mask layout")
    return native, cut


def make_args(max_chunks: int):
    return SimpleNamespace(
        sequence_length=2, image_sequence_length=1, state_sequence_length=1,
        action_sequence_length=3, libero_eval_max_steps=max_chunks * 3,
    )


def run_episode(model, task, suite_name: str, condition: str, mask: torch.Tensor, max_chunks: int, seed: int):
    model.attention_mask.copy_(mask)
    bddl = LIBERO_ROOT / "libero/libero/bddl_files" / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=224, camera_widths=224,
        render_gpu_device_id=0,
    )
    env.task_id = 0
    env.task_name = task.name
    env.task_suite_name = suite_name
    env.reset()
    env.seed(seed)
    init_path = LIBERO_ROOT / "libero/libero/init_files" / task.problem_folder / task.init_states_file
    init_states = torch.load(init_path, map_location="cpu", weights_only=False)
    obs = env.set_init_state(init_states[0])
    for _ in range(5):
        env.step(np.zeros(7, dtype=np.float32))

    image_fn = lambda ims: preprocess_image(ims, image_processor=model.image_processor)
    text_fn = lambda txt: preprocess_text_calvin(txt, tokenizer=clip)
    wrapper = WorldAgenLiberoModelWrapper(
        make_args(max_chunks), model, clip, model.image_processor, torch.float32,
        history_len=2, libero_eval_max_steps=max_chunks * 3,
    )
    wrapper.reset()
    goal = task.language if condition != "empty_goal" else ""
    text = text_fn([goal])
    primary = image_fn([Image.fromarray(obs["agentview_image"])])
    wrist = image_fn([Image.fromarray(obs["robot0_eye_in_hand_image"])])
    ori = quaternion_to_euler(obs["robot0_eef_quat"])
    state_np = np.concatenate([obs["robot0_eef_pos"], ori, wrapper.gripper_state])
    state = torch.from_numpy(state_np[None, None, :]).float()
    primary = primary.unsqueeze(1)
    wrist = wrist.unsqueeze(1)
    actions = torch.zeros(1, 3, 7)
    steps = 0
    success = 0
    action_norms = []
    while steps < max_chunks * 3:
        if steps == 0:
            action = wrapper.step_agent(text, state, primary, wrist, actions)
        else:
            action = wrapper.step_agent(text, state, primary, wrist)
        action_norms.append(float(action.float().norm()))
        # Match the released evaluator: append this executed chunk to the
        # wrapper's action history and run the frozen world head before the
        # environment step.  The prediction itself is not used for control.
        wrapper.step_world_model(action)
        primary, wrist, state, _info, success = get_robot_info_from_multi_step_env_libero(
            env, action, image_fn
        )
        steps += int(action.shape[1])
        if success or primary.shape[1] == 0:
            break
        # image_sequence_length=1: retain the last observation in the chunk.
        primary = primary[:, -1].unsqueeze(1)
        wrist = wrist[:, -1].unsqueeze(1)
        state = state[:, -1].unsqueeze(1)
        wrapper.gripper_state = np.array([action[0, -1, -1].item()])
    env.close()
    return {
        "suite": suite_name, "task": task.language, "condition": condition,
        "success": bool(success), "environment_steps": steps,
        "action_norm_mean": float(np.mean(action_norms)) if action_norms else None,
        "action_norm_max": float(np.max(action_norms)) if action_norms else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chunks", type=int, default=12)
    parser.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal"])
    parser.add_argument("--conditions", nargs="+", default=["native", "edge_cut", "empty_goal"])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.monotonic()
    model = build_model()
    native, cut = masks(model)
    suites = args.suites
    allowed_conditions = {"native": native, "edge_cut": cut, "empty_goal": native}
    unknown = set(args.conditions) - set(allowed_conditions)
    if unknown:
        raise SystemExit(f"unknown conditions: {sorted(unknown)}")
    results = []
    for suite_name in suites:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        task = suite.get_task(0)
        for condition in args.conditions:
            mask = allowed_conditions[condition]
            print(json.dumps({"suite": suite_name, "condition": condition}), flush=True)
            results.append(run_episode(model, task, suite_name, condition, mask, args.max_chunks, 31))
    summary = {
        "status": "completed_worldagen_short_libero_closed_loop_baseline",
        "model": "WorldAgen released LIBERO-38",
        "checkpoint": str(MODEL_DIR / "libero-38.pth"),
        "conditions": args.conditions,
        "suites": suites, "one_initial_state_per_suite": True,
        "max_chunks": args.max_chunks, "results": results,
        "training_updates": 0, "api_requests": 0,
        "scope": "Short frozen LIBERO policy baseline; not an attack-success claim or ICLR-scale benchmark.",
        "elapsed_seconds": time.monotonic() - started,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
