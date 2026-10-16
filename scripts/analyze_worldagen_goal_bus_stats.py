#!/usr/bin/env python3
"""Paired uncertainty summary for the expanded frozen WorldAgen gate."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("WORLDAGEN_STATS_SOURCE", str(ROOT / "runs/worldagen_goal_bus_replicate_v2/summary.json")))
OUT = Path(os.environ.get("WORLDAGEN_STATS_OUT", str(ROOT / "runs/worldagen_goal_bus_replicate_v2/statistics.json")))


def bootstrap_ci(values: np.ndarray, seed: int = 20261011, draws: int = 20000):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[idx].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def summarize(values):
    x = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
        "std": float(x.std(ddof=1)),
        "bootstrap_95ci_mean": bootstrap_ci(x),
    }


def main():
    data = json.loads(SOURCE.read_text())
    rows = data["records"]
    native_other = [r["native"]["other_world"]["relative_l2"] for r in rows]
    native_empty = [r["native"]["empty_world"]["relative_l2"] for r in rows]
    cut_other = [r["cut_repair"]["other_world"]["relative_l2"] for r in rows]
    cut_empty = [r["cut_repair"]["empty_world"]["relative_l2"] for r in rows]
    cut_policy = [r["cut_repair"]["other_policy"]["relative_l2"] for r in rows]
    mse_delta = [r["native"]["other_primary_forecast_mse"] - r["native"]["recorded_primary_forecast_mse"] for r in rows]
    repair_cost = [r["cut_repair"]["recorded_primary_forecast_mse"] - r["native"]["recorded_primary_forecast_mse"] for r in rows]
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "windows": len(rows),
        "native_other_world_relative_l2": summarize(native_other),
        "native_empty_world_relative_l2": summarize(native_empty),
        "cut_other_world_relative_l2": summarize(cut_other),
        "cut_empty_world_relative_l2": summarize(cut_empty),
        "cut_policy_relative_l2": summarize(cut_policy),
        "native_swapped_minus_recorded_primary_mse": summarize(mse_delta),
        "edge_cut_recorded_primary_mse_cost": summarize(repair_cost),
        "all_native_other_world_nonzero": bool(np.all(np.asarray(native_other) > 1e-7)),
        "all_cut_world_exact_zero": bool(np.all(np.asarray(cut_other + cut_empty) == 0.0)),
        "all_cut_policy_sensitive": bool(np.all(np.asarray(cut_policy) > 1e-7)),
        "positive_swapped_mse_fraction": float(np.mean(np.asarray(mse_delta) > 0)),
        "bootstrap_seed": 20261011,
        "bootstrap_draws": 20000,
        "scope": "Paired descriptive uncertainty for frozen model windows; no new forward pass or task-success claim.",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
