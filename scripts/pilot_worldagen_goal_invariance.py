"""Bounded CPU probe of released WorldAgen weights and indirect goal influence.

This calls the unchanged native forward, in its supported binary-gripper mode.
The released LIBERO shell wrapper instead sets gripper_width=True while passing
7D actions to a 2D gripper encoder; that incompatible wrapper is not reproduced.
This is a conditional-output experiment, not a benchmark or task-harm result.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import clip
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/WorldAgen"))
from models.model import WorldAgen

RUN = ROOT / "runs/worldagen_goal_invariance_v0"
MODEL = ROOT / "models/WorldAgen"
DATA = ROOT / "data/worldagen_goal_probe"


def write(name, obj):
    (RUN / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def parameter_digest(model):
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def difference(a, b):
    delta = a.double() - b.double()
    return {"rmse": float(delta.square().mean().sqrt()),
            "max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / a.double().norm().clamp_min(1e-12)),
            "exactly_equal": torch.equal(a, b)}


def main():
    start = time.monotonic()
    RUN.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    plan = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(ROOT / "vendor/WorldAgen"), "rev-parse", "HEAD"], text=True).strip(),
        "model_provenance": json.loads((MODEL / "provenance.json").read_text()),
        "data_provenance": json.loads((DATA / "provenance.json").read_text()),
        "selection": "First three episodes in downloaded shard; frame 30 in each; fixed before inference.",
        "input_offsets": {"images_states": [0, 3], "actions": [0, 1, 2, 3, 4, 5]},
        "goal_variants": ["recorded task", "next task cyclically among first three tasks", "empty string"],
        "controls": ["identical prompt with identical RNG", "only goal-to-input-action edges cut"],
        "preprocessing": {
            "images": "Rotate both cameras 180 degrees to raw LIBERO convention, then native CLIP transform.",
            "state": "EEF xyz; rotvec to Euler xyz; previous recorded gripper command (first action at episode start).",
            "action": "Recorded 7D action, no optimized or synthetic action.",
            "gripper_mode": "Native gripper_width=False; internally one-hot encodes recorded binary command.",
            "compatibility_limit": "Released eval shell sets gripper_width=True but its wrapper supplies 7D actions; native encoder requires 2 gripper dimensions in that mode. This experiment explicitly uses the implemented binary mode. It is not a released LIBERO evaluation reproduction.",
            "data_limit": "LeRobot conversion and state conventions are not byte-identical to author training data. Fixed across goal comparisons; no forecast-accuracy benchmark claim.",
        },
        "device": "cpu", "dtype": "float32", "seed_per_forward": 20261009,
        "training_updates": 0, "environment_steps": 0, "api_requests": 0,
        "not_claimed": ["task harm", "attack success", "generality", "novelty", "beneficial repair accuracy"],
    }
    write("plan.json", plan)

    # Only redirect the native CLIP loader to the verified local official file.
    original_clip_load = clip.load
    def local_clip_load(_name, device, **kwargs):
        return original_clip_load(str(MODEL / "ViT-B-32.pt"), device=device, **kwargs)

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
    del ckpt
    mask_matches = torch.equal(model.attention_mask, saved["attention_mask"])
    assert mask_matches, "Saved attention mask differs from native code/configuration"
    # The author excludes frozen encoder and sinusoidal position parameters.
    expected_missing = {k for k in model.state_dict() if k.startswith(("vision_encoder.", "clip_model."))}
    expected_missing.add("image_decoder_position_embedding")
    loaded = model.load_state_dict(saved, strict=False)
    assert not loaded.unexpected_keys, loaded.unexpected_keys
    assert set(loaded.missing_keys) == expected_missing, loaded.missing_keys
    assert all(torch.equal(model.state_dict()[k], v) for k, v in saved.items())
    del saved
    model.float().eval().requires_grad_(False)
    model._init_model_type()
    before = parameter_digest(model)
    native_mask = model.attention_mask.clone()
    cut_mask = native_mask.clone()
    actions = torch.tensor([2, 3, 4, 42, 43, 44])
    goals = torch.tensor([0, 40])
    cut_mask[actions[:, None], goals[None, :]] = -float("inf")
    removed = int((torch.isfinite(native_mask) & ~torch.isfinite(cut_mask)).sum())
    assert removed == 9
    write("load_validation.json", {
        "saved_mask_matches_native": mask_matches, "missing_keys": loaded.missing_keys,
        "unexpected_keys": loaded.unexpected_keys, "all_checkpoint_tensors_exactly_loaded": True,
        "parameter_sha256": before, "removed_edges": removed,
        "initialization_elapsed_s": time.monotonic() - start,
    })
    print("Released learned tensors loaded exactly; frozen encoders loaded separately.", flush=True)

    table = pq.read_table(DATA / "data/chunk-000/file-000.parquet")
    rows = table.to_pylist()
    tasks = pq.read_table(DATA / "meta/tasks.parquet").to_pylist()
    task_names = [r.get("task", r.get("__index_level_0__")) for r in tasks]
    assert all(isinstance(x, str) for x in task_names), tasks
    result_rows = []
    predictions = {}
    calls = 0
    for ep in [0, 1, 2]:
        records = {r["frame_index"]: r for r in rows if r["episode_index"] == ep}
        base = 30
        observations = [records[base + offset] for offset in [0, 3]]
        def images(key):
            ims = [Image.open(BytesIO(r[key]["bytes"])).convert("RGB").transpose(Image.Transpose.ROTATE_180)
                   for r in observations]
            return torch.stack([model.image_processor(im) for im in ims]).unsqueeze(0)
        primary, wrist = images("observation.images.image"), images("observation.images.wrist_image")
        states = []
        for offset, record in zip([0, 3], observations):
            s = record["observation.state"]
            previous = records[max(0, base + offset - 1)]["action"][-1]
            states.append([*s[:3], *Rotation.from_rotvec(s[3:6]).as_euler("xyz"), previous])
        state = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
        action = torch.tensor([records[base + i]["action"] for i in range(6)], dtype=torch.float32).unsqueeze(0)
        task_index = observations[0]["task_index"]
        prompts = {"recorded": task_names[task_index], "other_task": task_names[(task_index + 1) % 3], "empty": ""}
        outputs = {}
        for condition, mask in [("native", native_mask), ("cut", cut_mask)]:
            model.attention_mask.copy_(mask)
            for goal_name in ["recorded", "other_task", "empty", "repeat"]:
                prompt = prompts["recorded" if goal_name == "repeat" else goal_name]
                tokens = clip.tokenize([prompt]).unsqueeze(1).repeat(1, 2, 1)
                # MAE shuffles even with zero masking: reset its RNG to remove that confound.
                torch.manual_seed(20261009)
                call_start = time.monotonic()
                with torch.inference_mode():
                    arm, grip, world, _, _ = model(primary, wrist, state, tokens, action)
                assert all(torch.isfinite(x).all() for x in [arm, grip, world])
                policy = torch.cat((arm, grip), dim=-1)
                outputs[condition, goal_name] = (world.clone(), policy.clone())
                key = f"ep{ep}_{condition}_{goal_name}"
                predictions[key + "_world"] = world.numpy()
                predictions[key + "_policy"] = policy.numpy()
                calls += 1
                print(json.dumps({"episode": ep, "condition": condition, "goal": goal_name,
                                  "elapsed_s": time.monotonic() - call_start}), flush=True)
        metrics = {}
        for condition in ["native", "cut"]:
            base_world, base_policy = outputs[condition, "recorded"]
            for goal_name in ["other_task", "empty", "repeat"]:
                world, policy = outputs[condition, goal_name]
                metrics[f"{condition}_{goal_name}"] = {
                    "world": difference(base_world, world), "policy": difference(base_policy, policy)}
            assert metrics[f"{condition}_repeat"]["world"]["exactly_equal"]
            assert metrics[f"{condition}_repeat"]["policy"]["exactly_equal"]
        metrics["repair_shift_on_recorded_goal"] = {
            "world": difference(outputs["native", "recorded"][0], outputs["cut", "recorded"][0]),
            "policy": difference(outputs["native", "recorded"][1], outputs["cut", "recorded"][1]),
        }
        result_rows.append({"episode": ep, "frame": base, "task_index": task_index,
                            "prompts": prompts, "metrics": metrics})
        write("partial_results.json", result_rows)
    model.attention_mask.copy_(native_mask)
    after = parameter_digest(model)
    assert before == after
    np.savez_compressed(RUN / "predictions.npz", **predictions)
    summary = {
        "status": "completed_conditional_output_probe", "results": result_rows,
        "parameter_digest_unchanged": before == after,
        "native_attention_mask_restored": torch.equal(model.attention_mask, native_mask),
        "native_forward_calls": calls, "pretrained_model_loads": 1,
        "training_updates": 0, "environment_steps": 0, "api_requests": 0, "gpu_use": False,
        "elapsed_s": time.monotonic() - start,
        "scope": "Released weights with unchanged native forward in supported binary-gripper mode; recorded LIBERO inputs; conditional prediction dependence only. The released eval wrapper has an unresolved gripper-mode incompatibility. No task-harm, clean-performance, accuracy-improvement, or novelty claim.",
    }
    write("summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if RUN.exists():
            write("failure.json", {"type": type(exc).__name__, "error": str(exc)})
        raise
