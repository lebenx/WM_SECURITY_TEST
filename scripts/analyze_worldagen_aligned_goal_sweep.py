#!/usr/bin/env python3
"""Analyze legal-goal offset sweep for the aligned long-replay planner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summary(x):
    a = np.asarray(x, dtype=float)
    return {"n": int(len(a)), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--scorer", default="reference_scalar")
    args = ap.parse_args()
    data = json.loads(args.input.read_text())
    rows = data["rows"]
    clean_util, clean_idx = {}, {}
    for row in rows:
        tid = int(row["task_id"])
        clean_idx[tid] = int(row["selected"]["clean"][args.scorer])
        clean_util[tid] = float(row["candidate_replays"][clean_idx[tid]]["eef_approach_progress"])
    offsets = sorted({int(c.split("_")[1]) for row in rows for c in row["scores"] if c.startswith("swap_") and c.count("_") == 1})
    records = []
    for off in offsets:
        for row in rows:
            tid = int(row["task_id"])
            c = f"swap_{off}"
            bc = f"{c}_bus_cut"
            utilities = np.asarray([x["eef_approach_progress"] for x in row["candidate_replays"]], dtype=float)
            best = int(np.argmax(utilities))
            idx = int(row["selected"][c][args.scorer])
            bidx = int(row["selected"][bc][args.scorer])
            records.append({
                "task_id": tid, "offset": off,
                "clean_selected": clean_idx[tid], "selected": idx, "bus_cut_selected": bidx,
                "selected_changed": idx != clean_idx[tid],
                "bus_cut_changed": bidx != clean_idx[tid],
                "clean_eef": clean_util[tid],
                "eef": float(utilities[idx]), "bus_cut_eef": float(utilities[bidx]),
                "eef_delta": float(utilities[idx] - clean_util[tid]),
                "bus_cut_eef_delta": float(utilities[bidx] - clean_util[tid]),
                "regret": float(utilities[best] - utilities[idx]),
                "bus_cut_regret": float(utilities[best] - utilities[bidx]),
                "clean_action_recovered": idx == clean_idx[tid],
                "bus_cut_clean_action_recovered": bidx == clean_idx[tid],
                "bus_cut_best_candidate": bidx == best,
            })
    by_offset = {}
    for off in offsets:
        r = [x for x in records if x["offset"] == off]
        by_offset[str(off)] = {
            "n": len(r),
            "eef_delta_vs_clean": summary([x["eef_delta"] for x in r]),
            "bus_cut_eef_delta_vs_clean": summary([x["bus_cut_eef_delta"] for x in r]),
            "regret": summary([x["regret"] for x in r]),
            "bus_cut_regret": summary([x["bus_cut_regret"] for x in r]),
            "selected_changed_rate": float(np.mean([x["selected_changed"] for x in r])),
            "bus_cut_clean_action_recovery_rate": float(np.mean([x["bus_cut_clean_action_recovered"] for x in r])),
            "bus_cut_best_candidate_rate": float(np.mean([x["bus_cut_best_candidate"] for x in r])),
        }
    result = {
        "status": "completed_aligned_legal_goal_sweep_analysis",
        "input": str(args.input), "scorer": args.scorer,
        "task_ids": sorted({int(r["task_id"]) for r in rows}),
        "offsets": offsets, "replay_horizon": data["replay_horizon"],
        "forecast_horizon": data["forecast_horizon"],
        "by_offset": by_offset, "records": records,
        "inference_limits": [
            "two tasks and one seed; descriptive dose-response only",
            "true utility is 30-step simulator EEF progress; sparse task success is not claimed",
            "Bus-Cut best-candidate rate is not Clean-action recovery",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
