#!/usr/bin/env python3
"""Aggregate AlphaBrain Goal effects on future physical-output slots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def cluster_summary(rows: list[dict], key: str, draws: int = 20000, seed: int = 20261012) -> dict:
    episodes = sorted({int(r["episode"]) for r in rows})
    values = np.asarray([float(r[key]) for r in rows], dtype=float)
    episode_means = np.asarray([
        values[[int(r["episode"]) == e for r in rows]].mean() for e in episodes
    ])
    rng = np.random.default_rng(seed)
    boot = episode_means[rng.integers(0, len(episode_means), size=(draws, len(episode_means)))].mean(axis=1)
    return {
        "n_windows": int(values.size),
        "episode_count": len(episodes),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "episode_cluster_bootstrap_ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "positive_fraction": float(np.mean(values > 0)),
        "episode_means": {str(e): float(m) for e, m in zip(episodes, episode_means)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.input.read_text())
    rows = data["rows"]
    result = {
        "status": "completed_alphabrain_physical_slot_goal_audit",
        "source": str(args.input),
        "windows": len(rows),
        "episodes": sorted({int(r["episode"]) for r in rows}),
        "slots": {"5": "future_proprio", "6": "future_wrist", "7": "future_primary"},
        "native_goal_effect_abs_delta": {},
        "native_goal_target_mse_delta": {},
        "cut_goal_effect_abs_delta": {},
        "cut_goal_target_mse_delta": {},
        "fixed_action_cut_clean_target_mse_cost": {},
        "interpretation": "Goal changes the frozen fixed-action forecast in all physical-output slots; target-error direction is reported separately and is not equated with forecast divergence.",
        "scope_limits": [
            "Future proprio is a model output slot compared with the observed +16-frame state target; it is not a native action-selection result.",
            "The corrected cut is exact in latent space under the fixed sampler, so cut target-MSE deltas are zero by construction.",
            "Partial observability and goal-relative semantics remain validity boundaries for a universal physical-invariance claim.",
        ],
    }
    for name in ("future_proprio", "future_wrist", "future_primary"):
        result["native_goal_effect_abs_delta"][name] = cluster_summary(rows, f"native_goal_{name}_abs_delta")
        result["native_goal_target_mse_delta"][name] = cluster_summary(rows, f"native_goal_{name}_mse_delta")
        result["cut_goal_effect_abs_delta"][name] = cluster_summary(rows, f"cut_goal_{name}_abs_delta")
        result["cut_goal_target_mse_delta"][name] = cluster_summary(rows, f"cut_goal_{name}_mse_delta")
        # Cost of the invariant branch relative to the native Clean forecast.
        cost_rows = [
            {"episode": r["episode"], "value": r[f"cut_task_a_{name}_mse"] - r[f"native_task_a_{name}_mse"]}
            for r in rows
        ]
        result["fixed_action_cut_clean_target_mse_cost"][name] = cluster_summary(cost_rows, "value")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
