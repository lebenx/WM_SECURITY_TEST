#!/usr/bin/env python3
"""Probe Goal invariance under meaning-preserving text surface edits.

This is a small control for the physical-target audit: images, state, recorded
actions, masks, and RNG seeds are fixed while only whitespace/case/punctuation
of the same task instruction changes.  It does not claim that arbitrary
paraphrases preserve model semantics.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import clip
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/WorldAgen"))
from models.model import WorldAgen

MODEL = ROOT / "models/WorldAgen"
DATA = ROOT / "data/worldagen_goal_probe"
RUN = ROOT / "runs/worldagen_semantic_goal_invariance_v0"


def diff(a, b):
    delta = (a.double() - b.double())
    return {
        "relative_l2": float(delta.norm() / a.double().norm().clamp_min(1e-12)),
        "rmse": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
        "exactly_equal": bool(torch.equal(a, b)),
    }


def digest_model(model):
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_model():
    original = clip.load

    def local_clip_load(_name, device, **kwargs):
        return original(str(MODEL / "ViT-B-32.pt"), device=device, **kwargs)

    with patch.object(clip, "load", side_effect=local_clip_load):
        model = WorldAgen(
            clip_device="cpu", vit_checkpoint_path=str(MODEL / "mae_pretrain_vit_base.pth"),
            sequence_length=2, image_sequence_length=1, state_sequence_length=1,
            action_sequence_length=3, num_resampler_query=6, transformer_layers=24,
            hidden_dim=384, transformer_heads=12, phase="evaluate",
            gripper_width=False, pred_state=False, action_pred_steps=1,
            pred_image=True, atten_goal=0, use_qwen=True,
        )
    ckpt = torch.load(MODEL / "libero-38.pth", map_location="cpu", weights_only=True)
    saved = {k.removeprefix("module."): v for k, v in ckpt["model_state_dict"].items()}
    loaded = model.load_state_dict(saved, strict=False)
    assert not loaded.unexpected_keys
    model.float().eval().requires_grad_(False)
    model._init_model_type()
    return model


def main():
    start = time.monotonic()
    RUN.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model = load_model()
    before = digest_model(model)
    native = model.attention_mask.clone()
    cut = native.clone()
    actions = torch.tensor([2, 3, 4, 42, 43, 44])
    goals = torch.tensor([0, 40])
    cut[actions[:, None], goals[None, :]] = -float("inf")
    table = pq.read_table(DATA / "data/chunk-000/file-000.parquet").to_pylist()
    tasks = pq.read_table(DATA / "meta/tasks.parquet").to_pylist()
    task_names = [r.get("task", r.get("__index_level_0__")) for r in tasks]
    rows = []
    all_metrics = []
    for ep in [0, 1, 2]:
        records = {r["frame_index"]: r for r in table if r["episode_index"] == ep}
        base = 30
        observations = [records[base + i] for i in [0, 3]]
        images = []
        for key in ["observation.images.image", "observation.images.wrist_image"]:
            ims = [Image.open(BytesIO(r[key]["bytes"])).convert("RGB").transpose(Image.Transpose.ROTATE_180)
                   for r in observations]
            images.append(torch.stack([model.image_processor(im) for im in ims]).unsqueeze(0))
        primary, wrist = images
        states = []
        for offset, record in zip([0, 3], observations):
            s = record["observation.state"]
            previous = records[max(0, base + offset - 1)]["action"][-1]
            states.append([*s[:3], *Rotation.from_rotvec(s[3:6]).as_euler("xyz"), previous])
        state = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
        action = torch.tensor([records[base + i]["action"] for i in range(6)], dtype=torch.float32).unsqueeze(0)
        task_index = observations[0]["task_index"]
        canonical = task_names[task_index]
        variants = {
            "canonical": canonical,
            "trailing_period": canonical if canonical.endswith(".") else canonical + ".",
            "double_spaces": canonical.replace(" ", "  "),
            "lowercase": canonical.lower(),
        }
        outputs = {}
        for condition, mask in [("native", native), ("bus_cut", cut)]:
            model.attention_mask.copy_(mask)
            for name, prompt in variants.items():
                tokens = clip.tokenize([prompt]).unsqueeze(1).repeat(1, 2, 1)
                torch.manual_seed(20261009)
                with torch.inference_mode():
                    arm, grip, world, _, _ = model(primary, wrist, state, tokens, action)
                outputs[condition, name] = (world.clone(), torch.cat((arm, grip), dim=-1).clone())
        for condition in ["native", "bus_cut"]:
            base_world, base_policy = outputs[condition, "canonical"]
            for name in variants:
                if name == "canonical":
                    continue
                m = {"episode": ep, "task_index": int(task_index), "condition": condition,
                     "variant": name, "prompt": variants[name],
                     "world_vs_canonical": diff(base_world, outputs[condition, name][0]),
                     "policy_vs_canonical": diff(base_policy, outputs[condition, name][1])}
                all_metrics.append(m)
        rows.append({"episode": ep, "task_index": int(task_index), "canonical": canonical,
                     "variants": variants})
    model.attention_mask.copy_(native)
    after = digest_model(model)
    assert before == after
    result = {
        "status": "completed_semantic_goal_surface_control",
        "episodes": rows,
        "metrics": all_metrics,
        "protocol": {
            "fixed": ["image", "wrist image", "state", "recorded action", "attention mask", "RNG seed"],
            "surface_edits": list(next(iter(rows))["variants"])[1:],
            "native_gripper_mode": "binary supported mode",
        },
        "parameter_digest_unchanged": True,
        "native_mask_restored": bool(torch.equal(model.attention_mask, native)),
        "training_updates": 0, "environment_steps": 0, "api_requests": 0,
        "elapsed_s": time.monotonic() - start,
        "scope_limits": [
            "Surface edits are a control, not a semantic paraphrase benchmark.",
            "Nonzero surface sensitivity shows textual fragility, not physical task harm.",
            "The probe uses three real input windows and the released binary-gripper path.",
        ],
    }
    (RUN / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "metrics": len(all_metrics),
                      "elapsed_s": result["elapsed_s"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
