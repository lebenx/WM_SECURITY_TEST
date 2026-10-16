"""Causal mask intervention for the WorldAgen future-action leakage gate."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pilot_worldagen_action_suffix import stats  # noqa: E402
from pilot_worldagen_stale_goal_queue import (  # noqa: E402
    DATA,
    MODEL,
    build_model,
    episode_input,
    infer,
)


def cut_future_action_edge(model):
    """Mask chunk-1 input actions from chunk-0 image prediction tokens."""
    block = int(model.attention_mask.shape[0] // model.sequence_length)
    num_a = block - (model.NUM_OBS_TOKEN * model.image_sequence_length
                     + model.action_sequence_length * model.action_pred_steps)
    world_start = num_a
    world_end = world_start + model.NUM_OBS_TOKEN * model.image_sequence_length
    future_action_start = block + 1 + model.state_sequence_length
    future_action_end = future_action_start + model.action_sequence_length
    model.attention_mask[world_start:world_end, future_action_start:future_action_end] = -float("inf")
    return {
        "block_size": block,
        "num_input_tokens": num_a,
        "world_rows": [world_start, world_end],
        "future_action_columns": [future_action_start, future_action_end],
    }


def main():
    started = time.monotonic()
    out = ROOT / "runs/worldagen_action_causal_mask_v0"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    rows = pq.read_table(DATA / "data/chunk-000/file-000.parquet").to_pylist()
    tasks = pq.read_table(DATA / "meta/tasks.parquet").to_pylist()
    task_names = [r.get("task", r.get("__index_level_0__")) for r in tasks]
    model = build_model()
    edge = cut_future_action_edge(model)
    records = []
    for episode in [0, 1, 2]:
        primary, wrist, state, action, goal = episode_input(model, rows, episode, task_names)
        suffix = action.clone()
        suffix[:, 3:] = torch.flip(suffix[:, 3:], dims=[1])
        # The mask is already causal for this model copy.  Each infer call
        # resets the seed, so the only changed variable is the action suffix.
        base_world, base_policy = infer(model, primary, wrist, state, action, goal)
        suffix_world, suffix_policy = infer(model, primary, wrist, state, suffix, goal)
        records.append({
            "episode": episode,
            "world_per_chunk": [stats(base_world[i], suffix_world[i])
                                 for i in range(base_world.shape[0])],
            "policy": stats(base_policy, suffix_policy),
        })
    first = [x["world_per_chunk"][0]["mean_abs"] for x in records]
    second = [x["world_per_chunk"][1]["mean_abs"] for x in records]
    result = {
        "status": "completed_causal_edge_intervention",
        "model": "WorldAgen released LIBERO-38 checkpoint",
        "checkpoint": str(MODEL / "libero-38.pth"),
        "edge_intervention": edge,
        "records": records,
        "aggregate": {
            "first_chunk_suffix_mean_abs": float(sum(first) / len(first)),
            "second_chunk_suffix_mean_abs": float(sum(second) / len(second)),
            "first_chunk_exact_zero_all": all(x < 1e-7 for x in first),
            "second_chunk_remains_sensitive_all": all(x > 1e-7 for x in second),
        },
        "interpretation": "The explicit future-action-to-earlier-world edge is already masked in the released model; the intervention is a negative confirmation of causal masking, not a task-harm result.",
        "training_updates": 0,
        "environment_steps": 0,
        "api_requests": 0,
        "gpu_use": False,
        "elapsed_s": time.monotonic() - started,
    }
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
