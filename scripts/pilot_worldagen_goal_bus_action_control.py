#!/usr/bin/env python3
"""WorldAgen fixed-goal alternate-action control.

The image, state, instruction, weights, and mask are fixed while the six-step
recorded action chunk is replaced by another real chunk from the same episode.
The probe compares native and goal-bus-cut world sensitivity; it is a frozen
mechanism control, not a closed-loop task experiment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pilot_worldagen_goal_bus_replicate as base  # noqa: E402

OUT = Path(os.environ.get(
    "WORLDAGEN_ACTION_CONTROL_OUT",
    str(ROOT / "runs" / "worldagen_goal_bus_action_control_v0" / "summary.json"),
))
MAX_EPISODES = int(os.environ.get("WORLDAGEN_ACTION_CONTROL_MAX_EPISODES", "0"))
BASE_FRAMES = [int(x) for x in os.environ.get(
    "WORLDAGEN_ACTION_CONTROL_BASE_FRAMES", "30,60,90"
).split(",") if x]


def main() -> None:
    started = time.monotonic()
    torch.set_num_threads(4)
    model = base.build_model()
    files = []
    for root in base.DATA_ROOTS:
        files.extend(sorted((root / "data/chunk-000").glob("file-*.parquet")))
    rows = []
    for file in files:
        rows.extend(pq.read_table(file).to_pylist())
    tasks = pq.read_table(base.DATA_ROOTS[0] / "meta/tasks.parquet").to_pylist()
    task_names = [r.get("task", r.get("__index_level_0__")) for r in tasks]
    records_by_episode = {}
    for row in rows:
        records_by_episode.setdefault(row["episode_index"], {})[row["frame_index"]] = row

    native_mask = model.attention_mask.detach().clone()
    cut_mask = native_mask.clone()
    actions = torch.tensor([2, 3, 4, 42, 43, 44])
    goals = torch.tensor([0, 40])
    cut_mask[actions[:, None], goals[None, :]] = -float("inf")
    if int((torch.isfinite(native_mask) & ~torch.isfinite(cut_mask)).sum()) != 9:
        raise RuntimeError("unexpected native mask layout")

    episode_ids = sorted(records_by_episode)
    if MAX_EPISODES > 0:
        episode_ids = episode_ids[:MAX_EPISODES]
    records = []
    for episode in episode_ids:
        ep = records_by_episode[episode]
        for base_frame in BASE_FRAMES:
            alt_frame = base_frame + 6
            if not all(base_frame + i in ep for i in range(6)) or not all(alt_frame + i in ep for i in range(6)):
                continue
            task_text = task_names[ep[base_frame]["task_index"]]
            current = base.make_window(model, ep, base_frame, task_text)
            alternate = base.make_window(model, ep, alt_frame, task_text)
            # Keep image/state/text from the current window and replace only actions.
            alternate = (current[0], current[1], current[2], current[3], alternate[4])
            native_current, native_policy_current = base.infer(model, current, native_mask)
            native_alt, native_policy_alt = base.infer(model, alternate, native_mask)
            cut_current, cut_policy_current = base.infer(model, current, cut_mask)
            cut_alt, cut_policy_alt = base.infer(model, alternate, cut_mask)
            row = {
                "episode": int(episode),
                "base_frame": int(base_frame),
                "alternate_action_frame": int(alt_frame),
                "task": task_text,
                "native_action_world": base.diff(native_current, native_alt),
                "cut_action_world": base.diff(cut_current, cut_alt),
                "native_action_policy": base.diff(native_policy_current, native_policy_alt),
                "cut_action_policy": base.diff(cut_policy_current, cut_policy_alt),
            }
            records.append(row)
            print(json.dumps({"episode": episode, "base": base_frame}, ensure_ascii=False), flush=True)

    if not records:
        raise RuntimeError("no eligible action-control windows")
    native_world = [r["native_action_world"]["relative_l2"] for r in records]
    cut_world = [r["cut_action_world"]["relative_l2"] for r in records]
    native_policy = [r["native_action_policy"]["relative_l2"] for r in records]
    cut_policy = [r["cut_action_policy"]["relative_l2"] for r in records]
    result = {
        "status": "completed_worldagen_goal_bus_action_control",
        "model": "WorldAgen released LIBERO-38",
        "data_roots": [str(x) for x in base.DATA_ROOTS],
        "episodes": episode_ids,
        "base_frames": BASE_FRAMES,
        "windows": len(records),
        "cut_edges": 9,
        "native_action_world_relative_l2_mean": float(np.mean(native_world)),
        "cut_action_world_relative_l2_mean": float(np.mean(cut_world)),
        "cut_action_world_nonzero_all": bool(all(x > 1e-12 for x in cut_world)),
        "native_action_policy_relative_l2_mean": float(np.mean(native_policy)),
        "cut_action_policy_relative_l2_mean": float(np.mean(cut_policy)),
        "rows": records,
        "elapsed_seconds": time.monotonic() - started,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in (
        "windows", "native_action_world_relative_l2_mean",
        "cut_action_world_relative_l2_mean", "cut_action_world_nonzero_all",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
