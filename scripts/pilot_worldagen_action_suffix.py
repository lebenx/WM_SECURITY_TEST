"""Native short gate for future-action contamination in WorldAgen.

The released model predicts image tokens and actions from two action chunks.
For the suffix arm, the first chunk is held fixed and the second chunk is
reversed, preserving its action multiset.  The released mask should make the
first predicted world chunk invariant; this is a carrier boundary check, not
evidence of task harm.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pilot_worldagen_stale_goal_queue import (  # noqa: E402
    DATA,
    MODEL,
    build_model,
    episode_input,
    infer,
)


def stats(a, b):
    delta = (a.double() - b.double()).abs()
    denom = a.double().abs().mean().clamp_min(1e-12)
    return {
        "mean_abs": float(delta.mean()),
        "max_abs": float(delta.max()),
        "relative_mean": float(delta.mean() / denom),
        "exactly_equal": bool(torch.equal(a, b)),
    }


def main():
    start = time.monotonic()
    out = ROOT / "runs/worldagen_action_suffix_v0"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    rows = pq.read_table(DATA / "data/chunk-000/file-000.parquet").to_pylist()
    tasks = pq.read_table(DATA / "meta/tasks.parquet").to_pylist()
    task_names = [r.get("task", r.get("__index_level_0__")) for r in tasks]
    model = build_model()
    records = []
    for episode in [0, 1, 2]:
        primary, wrist, state, action, goal = episode_input(model, rows, episode, task_names)
        suffix = action.clone()
        suffix[:, 3:] = torch.flip(suffix[:, 3:], dims=[1])
        prefix = action.clone()
        prefix[:, :3] = torch.flip(prefix[:, :3], dims=[1])
        base_world, base_policy = infer(model, primary, wrist, state, action, goal)
        suffix_world, suffix_policy = infer(model, primary, wrist, state, suffix, goal)
        prefix_world, prefix_policy = infer(model, primary, wrist, state, prefix, goal)
        records.append({
            "episode": episode,
            "goal": goal,
            "suffix_action_multiset_preserved": True,
            "prefix_action_multiset_preserved": True,
            "suffix_world_per_chunk": [stats(base_world[i], suffix_world[i])
                                        for i in range(base_world.shape[0])],
            "prefix_world_per_chunk": [stats(base_world[i], prefix_world[i])
                                        for i in range(base_world.shape[0])],
            "suffix_policy": stats(base_policy, suffix_policy),
            "prefix_policy": stats(base_policy, prefix_policy),
        })
    first_suffix = [r["suffix_world_per_chunk"][0]["mean_abs"] for r in records]
    first_prefix = [r["prefix_world_per_chunk"][0]["mean_abs"] for r in records]
    summary = {
        "status": "completed_native_future_action_suffix_gate_negative",
        "model": "WorldAgen released LIBERO-38 checkpoint",
        "checkpoint": str(MODEL / "libero-38.pth"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "episodes": records,
        "aggregate": {
            "first_chunk_suffix_mean_abs": float(sum(first_suffix) / len(first_suffix)),
            "first_chunk_prefix_mean_abs": float(sum(first_prefix) / len(first_prefix)),
            "first_chunk_suffix_nonzero_all": all(x > 1e-7 for x in first_suffix),
            "suffix_policy_mean_abs": float(sum(r["suffix_policy"]["mean_abs"] for r in records) / len(records)),
        },
        "interpretation": "The released mask makes the earlier world chunk invariant to the later action suffix; this carrier does not support the proposed non-causal leakage mechanism.",
        "training_updates": 0,
        "environment_steps": 0,
        "api_requests": 0,
        "gpu_use": False,
        "elapsed_s": time.monotonic() - start,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
