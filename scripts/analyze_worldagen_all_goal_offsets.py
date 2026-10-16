#!/usr/bin/env python3
"""Analyze the legal-goal sweep without selecting a goal by its outcome."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def mean_sd(xs):
    a = np.asarray(xs, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    rows = []
    for path in args.inputs:
        rows.extend(json.loads(path.read_text())["rows"])
    offsets = sorted({int(k.split("_")[3]) for r in rows for k in r["replays"] if k.startswith("goal_swap_offset_") and k.count("_") == 3})
    per_offset = {}
    for offset in offsets:
        label = f"goal_swap_offset_{offset}"
        bus = label + "_bus_cut"
        control = label + "_control_cut"
        deltas = [r["replays"][label]["eef_approach_progress"] - r["replays"]["clean"]["eef_approach_progress"] for r in rows]
        bus_deltas = [r["replays"][bus]["eef_approach_progress"] - r["replays"]["clean"]["eef_approach_progress"] for r in rows]
        regrets = [r["replays"][label].get("eef_approach_regret_vs_candidate_best") for r in rows]
        bus_regrets = [r["replays"][bus].get("eef_approach_regret_vs_candidate_best") for r in rows]
        flips = [r["ranking_flip_rate"][label] for r in rows]
        per_offset[str(offset)] = {
            "native_eef_progress_delta_vs_clean": mean_sd(deltas),
            "bus_cut_eef_progress_delta_vs_clean": mean_sd(bus_deltas),
            "native_harm_count_delta_lt_zero": int(sum(x < 0 for x in deltas)),
            "native_benefit_count_delta_gt_zero": int(sum(x > 0 for x in deltas)),
            "bus_cut_harm_count_delta_lt_zero": int(sum(x < 0 for x in bus_deltas)),
            "native_regret_vs_candidate_best": mean_sd([x for x in regrets if x is not None]) if any(x is not None for x in regrets) else None,
            "bus_cut_regret_vs_candidate_best": mean_sd([x for x in bus_regrets if x is not None]) if any(x is not None for x in bus_regrets) else None,
            "ranking_flip_rate": mean_sd(flips),
            "bus_cut_selected_matches_clean_rate": float(np.mean([r["decision_recovery"][bus] for r in rows])),
            "control_cut_selected_matches_clean_rate": float(np.mean([r["decision_recovery"][control] for r in rows])),
            "control_cut_score_max_abs_diff": float(max(
                max(abs(a - b) for a, b in zip(r["scores"][label], r["scores"][control])) for r in rows
            )),
        }
    all_native = [v["native_eef_progress_delta_vs_clean"]["mean"] for v in per_offset.values()]
    all_bus = [v["bus_cut_eef_progress_delta_vs_clean"]["mean"] for v in per_offset.values()]
    summary = {
        "n_tasks": len(rows),
        "n_legal_offsets": len(offsets),
        "offsets": offsets,
        "per_offset": per_offset,
        "worst_native_offset_by_mean_eef_delta": offsets[int(np.argmin(all_native))],
        "worst_native_mean_eef_delta": float(min(all_native)),
        "best_native_offset_by_mean_eef_delta": offsets[int(np.argmax(all_native))],
        "best_native_mean_eef_delta": float(max(all_native)),
        "worst_bus_cut_offset_by_mean_eef_delta": offsets[int(np.argmin(all_bus))],
        "worst_bus_cut_mean_eef_delta": float(min(all_bus)),
        "interpretation": "all offsets are evaluated; worst/best labels are descriptive post-hoc summaries, not a pre-registered attack selection rule",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
