"""Short wrapper-boundary screen for stale WorldAgen task text.

The released LIBERO wrapper fills ``text_queue`` only on its first call and
does not replace it when a caller supplies a new task.  This screen uses the
same released weights and recorded LIBERO observations to compare a fresh
target prompt with the wrapper's stale first prompt.  It is a conditional
output screen; it makes no task-success or attack-success claim.
"""
from __future__ import annotations

import hashlib
import json
from io import BytesIO
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

RUN = ROOT / "runs/worldagen_stale_goal_queue_v0"
MODEL = ROOT / "models/WorldAgen"
DATA = ROOT / "data/worldagen_goal_probe"


def write(name, obj):
    (RUN / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def diff(a, b):
    d = a.double() - b.double()
    return {
        "rmse": float(d.square().mean().sqrt()),
        "max_abs": float(d.abs().max()),
        "relative_l2": float(d.norm() / a.double().norm().clamp_min(1e-12)),
        "exactly_equal": bool(torch.equal(a, b)),
    }


def build_model():
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
    loaded = model.load_state_dict(saved, strict=False)
    expected_missing = {k for k in model.state_dict() if k.startswith(("vision_encoder.", "clip_model."))}
    expected_missing.add("image_decoder_position_embedding")
    assert not loaded.unexpected_keys and set(loaded.missing_keys) == expected_missing
    model.float().eval().requires_grad_(False)
    model._init_model_type()
    return model


def episode_input(model, rows, ep, task_names):
    records = {r["frame_index"]: r for r in rows if r["episode_index"] == ep}
    observations = [records[30], records[33]]
    ims = []
    for key in ["observation.images.image", "observation.images.wrist_image"]:
        ims.append(torch.stack([
            model.image_processor(Image.open(BytesIO(r[key]["bytes"])).convert("RGB").transpose(Image.Transpose.ROTATE_180))
            for r in observations
        ]).unsqueeze(0))
    states = []
    for offset, record in zip([0, 3], observations):
        s = record["observation.state"]
        prev = records[max(0, 30 + offset - 1)]["action"][-1]
        states.append([*s[:3], *Rotation.from_rotvec(s[3:6]).as_euler("xyz"), prev])
    state = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
    action = torch.tensor([records[30 + i]["action"] for i in range(6)], dtype=torch.float32).unsqueeze(0)
    task_index = observations[0]["task_index"]
    return ims[0], ims[1], state, action, task_names[task_index]


def infer(model, primary, wrist, state, action, goal):
    tokens = clip.tokenize([goal]).unsqueeze(1).repeat(1, 2, 1)
    torch.manual_seed(20261010)
    with torch.inference_mode():
        arm, grip, world, _, _ = model(primary, wrist, state, tokens, action)
    return world.clone(), torch.cat((arm, grip), dim=-1).clone()


def main():
    start = time.monotonic()
    RUN.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    rows = pq.read_table(DATA / "data/chunk-000/file-000.parquet").to_pylist()
    tasks = pq.read_table(DATA / "meta/tasks.parquet").to_pylist()
    task_names = [r.get("task", r.get("__index_level_0__")) for r in tasks]
    model = build_model()
    records = []
    for ep in [0, 1, 2]:
        primary, wrist, state, action, goal_a = episode_input(model, rows, ep, task_names)
        goal_b = task_names[(task_names.index(goal_a) + 1) % 3]
        fresh_a = infer(model, primary, wrist, state, action, goal_a)
        fresh_b = infer(model, primary, wrist, state, action, goal_b)
        # The released wrapper's second task call still feeds goal_a, because
        # it only initializes text_queue when the queue is empty.
        stale_b = infer(model, primary, wrist, state, action, goal_a)
        records.append({
            "episode": ep, "first_goal": goal_a, "requested_second_goal": goal_b,
            "metrics": {
                "stale_vs_fresh_first": {"world": diff(stale_b[0], fresh_a[0]), "policy": diff(stale_b[1], fresh_a[1])},
                "stale_vs_fresh_requested": {"world": diff(stale_b[0], fresh_b[0]), "policy": diff(stale_b[1], fresh_b[1])},
                "fresh_first_vs_requested": {"world": diff(fresh_a[0], fresh_b[0]), "policy": diff(fresh_a[1], fresh_b[1])},
            },
        })
    summary = {
        "status": "completed_wrapper_stale_goal_screen",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "upstream_commit": subprocess.check_output(["git", "-C", str(ROOT / "vendor/WorldAgen"), "rev-parse", "HEAD"], text=True).strip(),
        "model_provenance": json.loads((MODEL / "provenance.json").read_text()),
        "data_provenance": json.loads((DATA / "provenance.json").read_text()),
        "results": records,
        "wrapper_condition": "text_queue initialized on first step and not replaced on requested task switch",
        "controls": ["fresh first goal", "fresh requested second goal", "stale second call with first goal"],
        "training_updates": 0, "environment_steps": 0, "api_requests": 0, "gpu_use": False,
        "claims_excluded": ["task harm", "attack success", "benchmark success", "generality", "novelty"],
        "elapsed_s": time.monotonic() - start,
    }
    write("summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if RUN.exists():
            write("failure.json", {"type": type(exc).__name__, "error": str(exc)})
        raise
