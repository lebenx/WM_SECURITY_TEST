#!/usr/bin/env python3
"""Aggregate all legal task-offset goal-bus probes over the same real windows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/worldagen_goal_bus_offset_sweep_v1/summary_stats.json"


def ci(values, seed=20261011, draws=20000):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(draws, len(x)))
    means = x[idx].mean(axis=1)
    return [float(q) for q in np.quantile(means, [0.025, 0.975])]


def cluster_ci(values, n_offsets=9, n_windows=30, seed=20261011, draws=20000):
    """Bootstrap physical windows, preserving the repeated offset measures."""
    x = np.asarray(values, dtype=float).reshape(n_offsets, n_windows)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_windows, size=(draws, n_windows))
    means = x[:, idx].mean(axis=(0, 2))
    return [float(q) for q in np.quantile(means, [0.025, 0.975])]


def read_records(path):
    return json.loads(path.read_text())


def main():
    sources = {1: ROOT / "runs/worldagen_goal_bus_replicate_v2/summary.json",
               5: ROOT / "runs/worldagen_goal_bus_far_goal_v3/summary.json"}
    for offset in [2, 3, 4, 6, 7, 8, 9]:
        sources[offset] = ROOT / f"runs/worldagen_goal_bus_offset_sweep_v1/offset_{offset}.json"
    all_rows = []
    by_offset = {}
    for offset, path in sorted(sources.items()):
        data = read_records(path)
        rows = data["records"]
        all_rows.extend([{**r, "goal_offset": offset} for r in rows])
        a = data["aggregate"]
        by_offset[str(offset)] = {
            "windows": len(rows),
            "native_world_relative_l2_mean": a["native_other_world_relative_l2_mean"],
            "native_policy_relative_l2_mean": a["native_other_policy_relative_l2_mean"],
            "cut_world_relative_l2_max": a["cut_other_world_relative_l2_max"],
            "cut_policy_relative_l2_mean": a["cut_other_policy_relative_l2_mean"],
            "native_swapped_primary_mse_mean": a["native_swapped_primary_forecast_mse_mean"],
        }
    world = [r["native"]["other_world"]["relative_l2"] for r in all_rows]
    policy = [r["cut_repair"]["other_policy"]["relative_l2"] for r in all_rows]
    mse_delta = [r["native"]["other_primary_forecast_mse"] - r["native"]["recorded_primary_forecast_mse"] for r in all_rows]
    cut_world = [r["cut_repair"]["other_world"]["relative_l2"] for r in all_rows]
    result = {
        "sources": {str(k): str(v.relative_to(ROOT)) for k, v in sorted(sources.items())},
        "offsets": sorted(sources),
        "windows_per_offset": 30,
        "paired_rows": len(all_rows),
        "native_world_relative_l2_mean": float(np.mean(world)),
        "native_world_relative_l2_cluster_bootstrap_95ci": cluster_ci(world),
        "native_world_relative_l2_min": float(np.min(world)),
        "native_world_nonzero_fraction": float(np.mean(np.asarray(world) > 1e-7)),
        "cut_world_exact_zero_fraction": float(np.mean(np.asarray(cut_world) == 0.0)),
        "cut_policy_relative_l2_mean": float(np.mean(policy)),
        "cut_policy_relative_l2_cluster_bootstrap_95ci": cluster_ci(policy),
        "cut_policy_sensitive_fraction": float(np.mean(np.asarray(policy) > 1e-7)),
        "swapped_minus_recorded_primary_mse_mean": float(np.mean(mse_delta)),
        "swapped_minus_recorded_primary_mse_cluster_bootstrap_95ci": cluster_ci(mse_delta),
        "positive_swapped_mse_fraction": float(np.mean(np.asarray(mse_delta) > 0)),
        "by_offset": by_offset,
        "scope": "Paired descriptive aggregate over legal goal offsets and expanded real LIBERO windows; frozen inference only.",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
