#!/usr/bin/env python3
"""Replicate WorldAgen's indirect goal-to-world causal bus on more windows.

The probe keeps images, proprioception, and recorded actions fixed.  It swaps
only the instruction text, then repeats the comparison after cutting the
native goal-to-input-action attention edges.  The cut must remove goal
dependence from the world prediction while preserving goal dependence of the
policy output.  This is a frozen-model mechanism gate, not a benchmark score.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
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
sys.path.insert(0, str(ROOT / "vendor" / "WorldAgen"))
from models.model import WorldAgen  # noqa: E402

MODEL = ROOT / "models" / "WorldAgen"
DATA = ROOT / "data" / "worldagen_goal_probe"
DATA_ROOTS = [Path(x) for x in os.environ.get("WORLDAGEN_DATA_ROOTS", str(DATA)).split(",") if x]
OUT = Path(os.environ.get("WORLDAGEN_OUT", str(ROOT / "runs" / "worldagen_goal_bus_replicate_v1" / "summary.json")))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model() -> WorldAgen:
    original_clip_load = clip.load

    def local_clip_load(_name, device, **kwargs):
        return original_clip_load(str(MODEL / "ViT-B-32.pt"), device=device, **kwargs)

    with patch.object(clip, "load", side_effect=local_clip_load):
        model = WorldAgen(
            clip_device=DEVICE,
            vit_checkpoint_path=str(MODEL / "mae_pretrain_vit_base.pth"),
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
    checkpoint = torch.load(MODEL / "libero-38.pth", map_location="cpu", weights_only=True)
    saved = {k.removeprefix("module."): v for k, v in checkpoint["model_state_dict"].items()}
    loaded = model.load_state_dict(saved, strict=False)
    expected_missing = {k for k in model.state_dict() if k.startswith(("vision_encoder.", "clip_model."))}
    expected_missing.add("image_decoder_position_embedding")
    if loaded.unexpected_keys or set(loaded.missing_keys) != expected_missing:
        raise RuntimeError(f"checkpoint mismatch: missing={loaded.missing_keys}, unexpected={loaded.unexpected_keys}")
    model = model.to(DEVICE).float().eval().requires_grad_(False)
    model._init_model_type()
    return model


def diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | bool]:
    delta = (a.double() - b.double())
    return {
        "rmse": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / a.double().norm().clamp_min(1e-12)),
        "exactly_equal": bool(torch.equal(a, b)),
    }


def make_window(model: WorldAgen, records: dict[int, dict], base: int, task_text: str):
    observations = [records[base], records[base + 3]]
    primary = torch.stack([
        model.image_processor(Image.open(BytesIO(r["observation.images.image"]["bytes"])).convert("RGB").transpose(Image.Transpose.ROTATE_180))
        for r in observations
    ]).unsqueeze(0)
    wrist = torch.stack([
        model.image_processor(Image.open(BytesIO(r["observation.images.wrist_image"]["bytes"])).convert("RGB").transpose(Image.Transpose.ROTATE_180))
        for r in observations
    ]).unsqueeze(0)
    states = []
    for offset, record in zip([0, 3], observations):
        s = record["observation.state"]
        previous = records[max(0, base + offset - 1)]["action"][-1]
        states.append([*s[:3], *Rotation.from_rotvec(s[3:6]).as_euler("xyz"), previous])
    state = torch.tensor(states, dtype=torch.float32).unsqueeze(0)
    actions = torch.tensor([records[base + i]["action"] for i in range(6)], dtype=torch.float32).unsqueeze(0)
    text = clip.tokenize([task_text]).unsqueeze(1).repeat(1, 2, 1)
    return primary.to(DEVICE), wrist.to(DEVICE), state.to(DEVICE), text.to(DEVICE), actions.to(DEVICE)


def patchify(images: torch.Tensor) -> torch.Tensor:
    """Match the released training loss: patchify and normalize each patch."""
    n, c, h, w = images.shape
    p = 16
    x = images.reshape(n, c, h // p, p, w // p, p)
    x = torch.einsum("nchpwq->nhwpqc", x).reshape(n, (h // p) * (w // p), p * p * c)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    return (x - mean) / (var + 1e-6).sqrt()


def target_patches(model: WorldAgen, records: dict[int, dict], base: int):
    # With image_sequence_length=1 and action_sequence_length=3, the native
    # labels are the last frame of action chunks [0:3] and [3:6].
    frames = [records[base + 3], records[base + 6]]
    primary = torch.stack([
        model.image_processor(Image.open(BytesIO(r["observation.images.image"]["bytes"])).convert("RGB").transpose(Image.Transpose.ROTATE_180))
        for r in frames
    ])
    wrist = torch.stack([
        model.image_processor(Image.open(BytesIO(r["observation.images.wrist_image"]["bytes"])).convert("RGB").transpose(Image.Transpose.ROTATE_180))
        for r in frames
    ])
    return patchify(primary.to(DEVICE)), patchify(wrist.to(DEVICE))


def infer(model: WorldAgen, batch, mask: torch.Tensor, targets=None):
    primary, wrist, state, text, actions = batch
    model.attention_mask.copy_(mask)
    torch.manual_seed(20261011)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(20261011)
    with torch.inference_mode():
        arm, grip, world, _, _ = model(primary, wrist, state, text, actions)
    world = world.detach().clone()
    policy = torch.cat((arm, grip), dim=-1).detach().clone()
    result = (world, policy)
    if targets is not None:
        target_primary, target_wrist = targets
        pred_primary = world[:, 0]
        pred_wrist = world[:, 1]
        result = result + (float((pred_primary - target_primary).square().mean()),
                           float((pred_wrist - target_wrist).square().mean()))
    return result


def main() -> None:
    started = time.monotonic()
    torch.set_num_threads(4)
    model = build_model()
    files = []
    for root in DATA_ROOTS:
        files.extend(sorted((root / "data/chunk-000").glob("file-*.parquet")))
    if not files:
        raise RuntimeError(f"no parquet files under {DATA_ROOTS}")
    rows = []
    for file in files:
        rows.extend(pq.read_table(file).to_pylist())
    tasks = pq.read_table(DATA_ROOTS[0] / "meta/tasks.parquet").to_pylist()
    task_names = [r.get("task", r.get("__index_level_0__")) for r in tasks]
    records_by_episode = {}
    for row in rows:
        records_by_episode.setdefault(row["episode_index"], {})[row["frame_index"]] = row

    native_mask = model.attention_mask.detach().clone()
    cut_mask = native_mask.clone()
    actions = torch.tensor([2, 3, 4, 42, 43, 44])
    goals = torch.tensor([0, 40])
    # The released sequence has two chunks of stride 40.  Cut only the
    # native goal-to-input-action edges; some cross-chunk entries are already
    # masked by the upstream causal construction.
    cut_mask[actions[:, None], goals[None, :]] = -float("inf")
    if int((torch.isfinite(native_mask) & ~torch.isfinite(cut_mask)).sum()) != 9:
        raise RuntimeError("unexpected native mask layout")

    records = []
    goal_offset = int(os.environ.get("WORLDAGEN_GOAL_OFFSET", "1"))
    if not 1 <= goal_offset < len(task_names):
        raise ValueError(f"WORLDAGEN_GOAL_OFFSET must be in [1,{len(task_names)-1}]")
    for episode in sorted(records_by_episode):
        ep = records_by_episode[episode]
        base_list = [b for b in (30, 60, 90) if all(b + i in ep for i in range(6))]
        for base in base_list:
            recorded_goal = task_names[ep[base]["task_index"]]
            other_goal = task_names[(ep[base]["task_index"] + goal_offset) % len(task_names)]
            empty_goal = ""
            batch_a = make_window(model, ep, base, recorded_goal)
            batch_b = make_window(model, ep, base, other_goal)
            batch_e = make_window(model, ep, base, empty_goal)
            targets = target_patches(model, ep, base)
            native_a, native_policy_a, native_a_primary_mse, native_a_wrist_mse = infer(model, batch_a, native_mask, targets)
            native_b, native_policy_b, native_b_primary_mse, native_b_wrist_mse = infer(model, batch_b, native_mask, targets)
            native_e, native_policy_e, native_e_primary_mse, native_e_wrist_mse = infer(model, batch_e, native_mask, targets)
            cut_a, cut_policy_a, cut_a_primary_mse, cut_a_wrist_mse = infer(model, batch_a, cut_mask, targets)
            cut_b, cut_policy_b, cut_b_primary_mse, cut_b_wrist_mse = infer(model, batch_b, cut_mask, targets)
            cut_e, cut_policy_e, cut_e_primary_mse, cut_e_wrist_mse = infer(model, batch_e, cut_mask, targets)
            records.append({
                "episode": int(episode), "base_frame": int(base),
                "recorded_goal": recorded_goal, "other_goal": other_goal,
                "native": {
                    "other_world": diff(native_a, native_b),
                    "empty_world": diff(native_a, native_e),
                    "other_policy": diff(native_policy_a, native_policy_b),
                    "empty_policy": diff(native_policy_a, native_policy_e),
                    "recorded_primary_forecast_mse": native_a_primary_mse,
                    "other_primary_forecast_mse": native_b_primary_mse,
                    "empty_primary_forecast_mse": native_e_primary_mse,
                    "recorded_wrist_forecast_mse": native_a_wrist_mse,
                    "other_wrist_forecast_mse": native_b_wrist_mse,
                    "empty_wrist_forecast_mse": native_e_wrist_mse,
                },
                "cut_repair": {
                    "other_world": diff(cut_a, cut_b),
                    "empty_world": diff(cut_a, cut_e),
                    "other_policy": diff(cut_policy_a, cut_policy_b),
                    "empty_policy": diff(cut_policy_a, cut_policy_e),
                    "recorded_world_shift": diff(native_a, cut_a),
                    "recorded_primary_forecast_mse": cut_a_primary_mse,
                    "other_primary_forecast_mse": cut_b_primary_mse,
                    "empty_primary_forecast_mse": cut_e_primary_mse,
                    "recorded_wrist_forecast_mse": cut_a_wrist_mse,
                    "other_wrist_forecast_mse": cut_b_wrist_mse,
                    "empty_wrist_forecast_mse": cut_e_wrist_mse,
                },
            })
            print(json.dumps({"episode": episode, "base": base}, ensure_ascii=False), flush=True)
    model.attention_mask.copy_(native_mask)
    native_other = [r["native"]["other_world"]["relative_l2"] for r in records]
    native_empty = [r["native"]["empty_world"]["relative_l2"] for r in records]
    cut_other = [r["cut_repair"]["other_world"]["relative_l2"] for r in records]
    cut_empty = [r["cut_repair"]["empty_world"]["relative_l2"] for r in records]
    native_policy = [r["native"]["other_policy"]["relative_l2"] for r in records]
    cut_policy = [r["cut_repair"]["other_policy"]["relative_l2"] for r in records]
    native_a_mse = [r["native"]["recorded_primary_forecast_mse"] for r in records]
    native_b_mse = [r["native"]["other_primary_forecast_mse"] for r in records]
    cut_a_mse = [r["cut_repair"]["recorded_primary_forecast_mse"] for r in records]
    result = {
        "status": "completed_worldagen_goal_bus_replication",
        "model": "WorldAgen released LIBERO-38",
        "checkpoint": str(MODEL / "libero-38.pth"),
        "data_roots": [str(x) for x in DATA_ROOTS],
        "data_files": [str(x) for x in files],
        "goal_offset": goal_offset,
        "device": DEVICE,
        "episodes": sorted(records_by_episode),
        "base_frames": [30, 60, 90],
        "windows": len(records),
        "cut_edges": 3,
        "aggregate": {
            "native_other_world_relative_l2_mean": float(np.mean(native_other)),
            "native_empty_world_relative_l2_mean": float(np.mean(native_empty)),
            "cut_other_world_relative_l2_max": float(np.max(cut_other)),
            "cut_empty_world_relative_l2_max": float(np.max(cut_empty)),
            "native_other_policy_relative_l2_mean": float(np.mean(native_policy)),
            "cut_other_policy_relative_l2_mean": float(np.mean(cut_policy)),
            "native_world_nonzero_all": all(x > 1e-7 for x in native_other),
            "cut_world_zero_all": all(x <= 1e-7 for x in cut_other + cut_empty),
            "cut_policy_remains_sensitive_all": all(x > 1e-7 for x in cut_policy),
            "native_recorded_primary_forecast_mse_mean": float(np.mean(native_a_mse)),
            "native_swapped_primary_forecast_mse_mean": float(np.mean(native_b_mse)),
            "cut_recorded_primary_forecast_mse_mean": float(np.mean(cut_a_mse)),
        },
        "records": records,
        "scope": "Frozen native forward; recorded LIBERO windows; no training or environment rollouts. The gate tests a goal-to-world causal invariant and its edge-cut repair.",
        "elapsed_seconds": time.monotonic() - started,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
