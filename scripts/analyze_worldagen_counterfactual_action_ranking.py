#!/usr/bin/env python3
"""Summarize the frozen-candidate counterfactual ranking experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    result = json.loads(args.input.read_text())
    rows = result["rows"]
    conditions = result.get("conditions") or ["clean", "goal_swap", "goal_swap_bus_cut", "goal_swap_control_cut"]
    nonclean = [c for c in conditions if c != "clean"]
    stats = {"n_tasks": len(rows), "n_candidates": sorted({r["candidate_count"] for r in rows})}
    stats["ranking_flip_rate"] = {
        c: mean_sd([r["ranking_flip_rate"].get(c, 0.0) for r in rows])
        for c in nonclean
    }
    stats["forecast_error_mse"] = {
        c: mean_sd([r["replays"][c]["forecast_error_mse"] for r in rows])
        for c in conditions
    }
    stats["forecast_error_delta_vs_clean"] = {
        c: mean_sd([
            r["replays"][c]["forecast_error_mse"] - r["replays"]["clean"]["forecast_error_mse"]
            for r in rows
        ])
        for c in nonclean
    }
    if all("forecast_relative_l2_to_clean" in r for r in rows):
        stats["forecast_relative_l2_to_clean"] = {
            c: mean_sd([
                value for r in rows for value in r["forecast_relative_l2_to_clean"].get(c, [])
            ])
            for c in conditions
        }
    stats["selected_action_utility"] = {
        c: {
            "cumulative_reward": mean_sd([r["replays"][c]["cumulative_reward"] for r in rows]),
            "max_reward": mean_sd([r["replays"][c]["max_reward"] for r in rows]),
            "success_count": int(sum(r["replays"][c]["success"] for r in rows)),
            "success_rate": float(np.mean([r["replays"][c]["success"] for r in rows])),
        }
        for c in conditions
    }
    stats["utility_delta_vs_clean"] = {
        c: mean_sd([
            r["replays"][c]["cumulative_reward"] - r["replays"]["clean"]["cumulative_reward"]
            for r in rows
        ])
        for c in nonclean
    }
    stats["true_environment_progress"] = {
        c: {
            "delta": mean_sd([r["replays"][c]["true_progress_delta"] for r in rows]),
            "final": mean_sd([r["replays"][c]["true_progress_final"] for r in rows]),
        }
        for c in conditions
    }
    stats["true_progress_delta_vs_clean"] = {
        c: mean_sd([
            r["replays"][c]["true_progress_delta"] - r["replays"]["clean"]["true_progress_delta"]
            for r in rows
        ])
        for c in nonclean
    }
    stats["eef_approach_progress"] = {
        c: mean_sd([r["replays"][c]["eef_approach_progress"] for r in rows])
        for c in conditions
    }
    stats["eef_approach_progress_delta_vs_clean"] = {
        c: mean_sd([
            r["replays"][c]["eef_approach_progress"] - r["replays"]["clean"]["eef_approach_progress"]
            for r in rows
        ])
        for c in nonclean
    }
    repeated = [
        r for r in rows
        if any("repeat_results" in replay for replay in r.get("replays", {}).values())
    ]
    if repeated:
        stats["repeat_consistency"] = {}
        for c in conditions:
            within_task_sd = []
            max_abs_pairwise = []
            repeat_counts = []
            for row in repeated:
                values = [x["eef_approach_progress"] for x in row["replays"][c]["repeat_results"]]
                repeat_counts.append(len(values))
                within_task_sd.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
                max_abs_pairwise.append(float(max(values) - min(values)) if values else 0.0)
            stats["repeat_consistency"][c] = {
                "tasks_with_repeats": len(within_task_sd),
                "repeat_counts": sorted(set(repeat_counts)),
                "mean_within_task_eef_sd": float(np.mean(within_task_sd)),
                "max_within_task_eef_sd": float(max(within_task_sd)),
                "max_within_task_eef_range": float(max(max_abs_pairwise)),
            }
    bus_names = [c for c in conditions if c.endswith("_bus_cut")]
    control_names = [c for c in conditions if c.endswith("_control_cut")]
    repair_names = [c for c in conditions if c.endswith("_bus_cut") or c.endswith("_control_cut")
                    or "activation_patch" in c]
    stats["decision_recovery"] = {
        c: {
            "matches_clean_count": int(sum(r["decision_recovery"][c] for r in rows)),
            "matches_clean_rate": float(np.mean([r["decision_recovery"][c] for r in rows])),
        }
        for c in repair_names
    }
    stats["control_cut_score_identity"] = {
        "max_abs_goal_swap_vs_control_cut": float(max(
            max(abs(a - b) for a, b in zip(r["scores"][c.removesuffix("_control_cut")], r["scores"][c]))
            for c in control_names for r in rows
        )) if control_names else None,
        "interpretation": "matched policy-only cut leaves the independent planner's world score unchanged",
    }
    stats["inference_limits"] = [
        f"n={len(rows)} tasks; report descriptive paired statistics only, no significance claim",
        f"replay environment steps observed: {sorted({r['replays']['clean']['environment_steps'] for r in rows})}; all sparse rewards were zero and all replays failed",
        "planner score is fixed CLIP similarity on a deterministic pseudo-image reconstruction of normalized WorldAgen patches",
    ]
    result["statistics"] = stats
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
