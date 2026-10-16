#!/usr/bin/env python3
"""Activation mediation test for the WorldAgen goal-to-world bus.

For each fixed real LIBERO window, run the recorded goal (A) and a legal
swapped goal (B) with identical image/state/action inputs.  At selected
Transformer layers, replace only B's contextualized input-action token hidden
states with A's hidden states, then measure whether the future-world output
moves toward A while the action/policy output remains B-sensitive.  This is a
frozen-weight mediation probe; it does not train or roll out the policy.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import clip
import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "WorldAgen"))
from models.model import WorldAgen  # noqa: E402

DATA_ROOT = Path(os.environ.get("WORLDAGEN_ACTIVATION_PATCH_DATA_ROOT", str(ROOT / "data" / "worldagen_goal_probe_extra")))
OUT = Path(os.environ.get(
    "WORLDAGEN_ACTIVATION_PATCH_OUT",
    str(ROOT / "runs" / "worldagen_goal_bus_activation_patch_v0" / "summary.json"),
))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model() -> WorldAgen:
    original_clip_load = clip.load

    def local_clip_load(_name, device, **kwargs):
        return original_clip_load(str(ROOT / "models/WorldAgen/ViT-B-32.pt"), device=device, **kwargs)

    with patch.object(clip, "load", side_effect=local_clip_load):
        model = WorldAgen(
            clip_device=DEVICE,
            vit_checkpoint_path=str(ROOT / "models/WorldAgen/mae_pretrain_vit_base.pth"),
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
    checkpoint = torch.load(ROOT / "models/WorldAgen/libero-38.pth", map_location="cpu", weights_only=True)
    saved = {k.removeprefix("module."): v for k, v in checkpoint["model_state_dict"].items()}
    loaded = model.load_state_dict(saved, strict=False)
    expected_missing = {k for k in model.state_dict() if k.startswith(("vision_encoder.", "clip_model."))}
    expected_missing.add("image_decoder_position_embedding")
    if loaded.unexpected_keys or set(loaded.missing_keys) != expected_missing:
        raise RuntimeError(f"checkpoint mismatch: missing={loaded.missing_keys}, unexpected={loaded.unexpected_keys}")
    model = model.to(DEVICE).float().eval().requires_grad_(False)
    model._init_model_type()
    return model


def make_window(model: WorldAgen, records: dict[int, dict], base: int, task_text: str):
    from io import BytesIO
    from PIL import Image
    from scipy.spatial.transform import Rotation

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


def run(model, batch, mask, capture_layers=(), patch_layer=None, source=None):
    model.attention_mask.copy_(mask)
    handles = []
    captured = {}
    action_positions = []
    block = int(mask.shape[0] // model.sequence_length)
    for chunk in range(model.sequence_length):
        start = chunk * block
        action_positions.extend(range(start + 1 + model.state_sequence_length,
                                      start + 1 + model.state_sequence_length + model.action_sequence_length))

    def make_hook(layer_idx):
        def hook(_module, _inputs, output):
            hidden = output[0]
            if layer_idx in capture_layers:
                captured[layer_idx] = hidden.detach().clone()
            if patch_layer == layer_idx:
                if source is None:
                    raise RuntimeError("patch source is missing")
                patched = hidden.clone()
                patched[:, action_positions, :] = source[:, action_positions, :]
                return (patched,) + tuple(output[1:])
            return output
        return hook

    layer_indices = sorted(set(capture_layers) | ({patch_layer} if patch_layer is not None else set()))
    for idx in layer_indices:
        handles.append(model.transformer_backbone.layers[idx].register_forward_hook(make_hook(idx)))
    torch.manual_seed(20261011)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(20261011)
    primary, wrist, state, text, actions = batch
    with torch.inference_mode():
        arm, grip, world, _, _ = model(primary, wrist, state, text, actions)
    for handle in handles:
        handle.remove()
    policy = torch.cat((arm, grip), dim=-1).detach().clone()
    return world.detach().clone(), policy, captured


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-12))


def bootstrap_ci(values: list[float], seed: int, draws: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=float)
    sampled = rng.choice(x, size=(draws, len(x)), replace=True).mean(axis=1)
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


def main() -> None:
    started = time.monotonic()
    torch.set_num_threads(4)
    model = build_model()
    files = sorted((DATA_ROOT / "data/chunk-000").glob("file-*.parquet"))
    rows = []
    for file in files:
        rows.extend(pq.read_table(file).to_pylist())
    tasks = pq.read_table(DATA_ROOT / "meta/tasks.parquet").to_pylist()
    task_names = [r.get("task", r.get("__index_level_0__")) for r in tasks]
    by_episode = {}
    for row in rows:
        by_episode.setdefault(int(row["episode_index"]), {})[int(row["frame_index"])] = row
    offset = int(os.environ.get("WORLDAGEN_ACTIVATION_PATCH_OFFSET", "5"))
    requested_layers = os.environ.get("WORLDAGEN_ACTIVATION_PATCH_LAYERS", "0,4,8,12,16,20,23")
    layers = [int(x) for x in requested_layers.split(",") if x.strip()]
    if any(layer < 0 or layer >= len(model.transformer_backbone.layers) for layer in layers):
        raise ValueError(f"layer out of range: {layers}")
    max_episodes = int(os.environ.get("WORLDAGEN_ACTIVATION_PATCH_MAX_EPISODES", "7"))
    base_frames = [int(x) for x in os.environ.get("WORLDAGEN_ACTIVATION_PATCH_BASE_FRAMES", "30").split(",") if x.strip()]
    native_mask = model.attention_mask.detach().clone()
    records = []
    for episode in sorted(by_episode)[:max_episodes]:
        ep = by_episode[episode]
        for base_frame in base_frames:
            if not all(base_frame + i in ep for i in range(6)):
                continue
            task_a = task_names[int(ep[base_frame]["task_index"])]
            task_b = task_names[(int(ep[base_frame]["task_index"]) + offset) % len(task_names)]
            batch_a = make_window(model, ep, base_frame, task_a)
            batch_b = make_window(model, ep, base_frame, task_b)
            world_a, policy_a, source_by_layer = run(model, batch_a, native_mask, capture_layers=layers)
            world_b, policy_b, _ = run(model, batch_b, native_mask)
            native_world = rel(world_b, world_a)
            native_policy = rel(policy_b, policy_a)
            layer_rows = []
            for layer in layers:
                patched_world, patched_policy, _ = run(
                    model, batch_b, native_mask, patch_layer=layer, source=source_by_layer[layer]
                )
                layer_rows.append({
                    "layer": layer,
                    "patched_world_to_recorded_rel_l2": rel(patched_world, world_a),
                    "patched_policy_to_recorded_rel_l2": rel(patched_policy, policy_a),
                    "world_residual_fraction": rel(patched_world, world_a) / max(native_world, 1e-12),
                    "policy_residual_fraction": rel(patched_policy, policy_a) / max(native_policy, 1e-12),
                })
            records.append({
                "episode": episode,
                "base_frame": base_frame,
                "task_a": task_a,
                "task_b": task_b,
                "native_world_to_recorded_rel_l2": native_world,
                "native_policy_to_recorded_rel_l2": native_policy,
                "layers": layer_rows,
            })
            print(json.dumps({"episode": episode, "base_frame": base_frame, "native_world": native_world, "native_policy": native_policy}), flush=True)
    if not records:
        raise RuntimeError("no valid windows")
    summary = {}
    for layer in layers:
        vals = [row["world_residual_fraction"] for rec in records for row in rec["layers"] if row["layer"] == layer]
        pol = [row["policy_residual_fraction"] for rec in records for row in rec["layers"] if row["layer"] == layer]
        summary[str(layer)] = {
            "world_residual_fraction_mean": float(np.mean(vals)),
            "world_residual_fraction_cluster_bootstrap_ci95": bootstrap_ci(vals, 5400 + layer),
            "world_residual_fraction_median": float(np.median(vals)),
            "policy_residual_fraction_mean": float(np.mean(pol)),
            "policy_residual_fraction_cluster_bootstrap_ci95": bootstrap_ci(pol, 6400 + layer),
            "world_moves_toward_recorded_all": bool(all(v < 1.0 for v in vals)),
        }
    result = {
        "status": "completed_worldagen_goal_bus_activation_mediation",
        "checkpoint": "models/WorldAgen/libero-38.pth",
        "device": DEVICE,
        "episodes": sorted({r["episode"] for r in records}),
        "base_frames": base_frames,
        "windows": len(records),
        "goal_offset": offset,
        "layers": layers,
        "action_positions_patched_per_chunk": model.action_sequence_length,
        "aggregate_by_layer": summary,
        "records": records,
        "interpretation": "Patching recorded-goal action-token hidden states into swapped-goal forwards tests mediation through the shared action-token carrier; it is a frozen forecast-level mechanism probe, not task-harm evidence.",
        "elapsed_seconds": time.monotonic() - started,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "windows", "aggregate_by_layer")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
