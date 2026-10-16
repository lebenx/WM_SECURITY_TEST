#!/usr/bin/env python3
"""Matched edge-cut controls for the WorldAgen goal-bus probe.

The primary cut removes the nine native goal-to-input-action edges.  The
policy-only control removes the same number of finite goal-to-future-action
edges, leaving the input-action carrier untouched.  If only the primary cut
restores world invariance, the result is specific to the proposed bus rather
than an artifact of deleting nine attention edges.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pilot_worldagen_goal_bus_replicate as base  # noqa: E402

OUT = Path(os.environ.get(
    "WORLDAGEN_EDGE_CONTROL_OUT",
    str(ROOT / "runs" / "worldagen_goal_bus_edge_cut_controls_v0" / "summary.json"),
))
MAX_EPISODES = int(os.environ.get("WORLDAGEN_EDGE_CONTROL_MAX_EPISODES", "0"))


def main() -> None:
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
    bus_cut = native_mask.clone()
    bus_cut[torch.tensor([2, 3, 4, 42, 43, 44])[:, None], torch.tensor([0, 40])[None, :]] = -float("inf")
    # These are the nine finite goal-to-future-action edges in the released
    # mask: policy-output rows, not the input-action carrier rows.
    policy_only_cut = native_mask.clone()
    policy_rows = torch.tensor([37, 38, 39, 77, 78, 79, 77, 78, 79])
    policy_cols = torch.tensor([0, 0, 0, 0, 0, 0, 40, 40, 40])
    policy_only_cut[policy_rows, policy_cols] = -float("inf")
    if int((torch.isfinite(native_mask) & ~torch.isfinite(bus_cut)).sum()) != 9:
        raise RuntimeError("primary bus cut is not nine edges")
    if int((torch.isfinite(native_mask) & ~torch.isfinite(policy_only_cut)).sum()) != 9:
        raise RuntimeError("policy-only control is not nine edges")

    episodes = sorted(records_by_episode)
    if MAX_EPISODES > 0:
        episodes = episodes[:MAX_EPISODES]
    records = []
    for episode in episodes:
        ep = records_by_episode[episode]
        for base_frame in (30, 60, 90):
            if not all(base_frame + i in ep for i in range(6)):
                continue
            recorded = task_names[ep[base_frame]["task_index"]]
            other = task_names[(ep[base_frame]["task_index"] + 1) % len(task_names)]
            current = base.make_window(model, ep, base_frame, recorded)
            swapped = base.make_window(model, ep, base_frame, other)
            row = {"episode": int(episode), "base_frame": int(base_frame), "recorded_goal": recorded, "other_goal": other}
            for name, mask in (("native", native_mask), ("bus_cut", bus_cut), ("policy_only_cut", policy_only_cut)):
                a_world, a_policy = base.infer(model, current, mask)
                b_world, b_policy = base.infer(model, swapped, mask)
                row[name] = {"world": base.diff(a_world, b_world), "policy": base.diff(a_policy, b_policy)}
            records.append(row)
            print(json.dumps({"episode": episode, "base": base_frame}, ensure_ascii=False), flush=True)

    if not records:
        raise RuntimeError("no eligible windows")
    result = {
        "status": "completed_worldagen_matched_edge_cut_controls",
        "model": "WorldAgen released LIBERO-38",
        "episodes": episodes,
        "base_frames": [30, 60, 90],
        "windows": len(records),
        "primary_cut": "nine goal-to-input-action edges",
        "policy_only_control": "nine goal-to-future-action edges (rows 37/77-79)",
        "native_world_relative_l2_mean": float(np.mean([r["native"]["world"]["relative_l2"] for r in records])),
        "bus_cut_world_relative_l2_max": float(np.max([r["bus_cut"]["world"]["relative_l2"] for r in records])),
        "bus_cut_policy_relative_l2_mean": float(np.mean([r["bus_cut"]["policy"]["relative_l2"] for r in records])),
        "policy_only_cut_world_relative_l2_mean": float(np.mean([r["policy_only_cut"]["world"]["relative_l2"] for r in records])),
        "policy_only_cut_world_nonzero_all": bool(all(r["policy_only_cut"]["world"]["relative_l2"] > 1e-12 for r in records)),
        "policy_only_cut_policy_relative_l2_mean": float(np.mean([r["policy_only_cut"]["policy"]["relative_l2"] for r in records])),
        "rows": records,
        "scope": "Matched frozen mask control; same real image/state/action inputs and legal goal swap across 30 windows.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ("windows", "native_world_relative_l2_mean", "bus_cut_world_relative_l2_max", "policy_only_cut_world_relative_l2_mean", "policy_only_cut_world_nonzero_all")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
