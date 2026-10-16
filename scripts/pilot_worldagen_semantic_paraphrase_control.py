#!/usr/bin/env python3
"""Meaning-preserving paraphrase control for the Goal-Bus forecast probe.

Images, state, executed actions, model weights, RNG and attention masks are
fixed.  Only the wording of the same task is changed.  This is a semantic
surface control, not evidence that arbitrary language is physically harmless.
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
import pyarrow.parquet as pq
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/WorldAgen"))
from models.model import WorldAgen

MODEL = ROOT / "models/WorldAgen"
DATA = ROOT / "data/worldagen_goal_probe"


def digest_model(model):
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def diff(a, b):
    d = a.double() - b.double()
    return {
        "relative_l2": float(d.norm() / a.double().norm().clamp_min(1e-12)),
        "rmse": float(d.square().mean().sqrt()),
        "max_abs": float(d.abs().max()),
        "exactly_equal": bool(torch.equal(a, b)),
    }


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
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/worldagen_semantic_paraphrase_v0"
    out.mkdir(parents=True, exist_ok=False)
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
    paraphrases = {
        0: [
            "place the white mug on the plate at the left, and place the yellow and white mug on the plate at the right",
            "move the white mug to the left plate and the yellow and white mug to the right plate",
        ],
        1: [
            "place the white mug on the plate, then put the chocolate pudding to the right of that plate",
            "move the white mug onto the plate and position the chocolate pudding on its right",
        ],
        2: [
            "place the yellow and white mug inside the microwave, then close the microwave door",
            "put the yellow-white mug into the microwave and shut it",
        ],
    }
    rows = []
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
        task_index = int(observations[0]["task_index"])
        canonical = task_names[task_index]
        prompts = [canonical, *paraphrases[task_index]]
        outputs = {}
        for condition, mask in [("native", native), ("bus_cut", cut)]:
            model.attention_mask.copy_(mask)
            for idx, prompt in enumerate(prompts):
                tokens = clip.tokenize([prompt]).unsqueeze(1).repeat(1, 2, 1)
                torch.manual_seed(20261012)
                with torch.inference_mode():
                    arm, grip, world, _, _ = model(primary, wrist, state, tokens, action)
                outputs[condition, idx] = (world.clone(), torch.cat((arm, grip), dim=-1).clone())
        metrics = []
        for condition in ["native", "bus_cut"]:
            base_world, base_policy = outputs[condition, 0]
            for idx in [1, 2]:
                metrics.append({
                    "condition": condition, "paraphrase_index": idx,
                    "prompt": prompts[idx],
                    "world_vs_canonical": diff(base_world, outputs[condition, idx][0]),
                    "policy_vs_canonical": diff(base_policy, outputs[condition, idx][1]),
                })
        rows.append({"episode": ep, "task_index": task_index, "canonical": canonical,
                     "paraphrases": prompts[1:], "metrics": metrics})
    model.attention_mask.copy_(native)
    after = digest_model(model)
    assert before == after
    all_metrics = [m for row in rows for m in row["metrics"]]
    native_world = [m["world_vs_canonical"]["relative_l2"] for m in all_metrics if m["condition"] == "native"]
    cut_world = [m["world_vs_canonical"]["relative_l2"] for m in all_metrics if m["condition"] == "bus_cut"]
    result = {
        "status": "completed_semantic_paraphrase_control",
        "rows": rows,
        "aggregate": {
            "native_world_relative_l2_mean": sum(native_world) / len(native_world),
            "native_world_relative_l2_max": max(native_world),
            "bus_cut_world_relative_l2_max": max(cut_world),
            "native_nonzero_count": sum(x > 0 for x in native_world),
            "bus_cut_exact_zero_count": sum(x == 0 for x in cut_world),
            "n_paraphrase_pairs": len(native_world),
        },
        "protocol": {
            "fixed": ["image", "wrist image", "state", "recorded action", "attention mask", "RNG seed"],
            "task_ids": [0, 1, 2],
            "meaning_preserving": True,
        },
        "parameter_digest_unchanged": True,
        "native_mask_restored": bool(torch.equal(model.attention_mask, native)),
        "training_updates": 0, "environment_steps": 0, "api_requests": 0,
        "elapsed_s": time.monotonic() - start,
        "scope_limits": [
            "Three real input windows and two hand-written paraphrases per task.",
            "Nonzero paraphrase sensitivity is textual fragility, not physical task harm.",
            "This is a control for task identity versus surface form, not a paraphrase benchmark.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"],
                      "elapsed_s": result["elapsed_s"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
