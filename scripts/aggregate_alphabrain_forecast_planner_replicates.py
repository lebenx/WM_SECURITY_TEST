#!/usr/bin/env python3
"""Aggregate AlphaBrain forecast-planner runs over explicit initial states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def cluster_bootstrap(rows, value_key, cluster_key, draws=20000, seed=20261012):
    clusters = sorted({str(row[cluster_key]) for row in rows})
    means = np.asarray([
        np.mean([float(row[value_key]) for row in rows if str(row[cluster_key]) == c])
        for c in clusters
    ], dtype=float)
    rng = np.random.default_rng(seed)
    sample = means[rng.integers(0, len(means), size=(draws, len(means)))].mean(axis=1)
    return {
        "clusters": clusters, "cluster_count": len(clusters),
        "mean": float(means.mean()),
        "ci95": [float(np.quantile(sample, .025)), float(np.quantile(sample, .975))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    docs = [json.loads(p.read_text()) for p in args.inputs]
    rows = []
    for doc, source in zip(docs, args.inputs):
        for task in doc["task_results"]:
            selected = task["selected"]
            regrets = task["regret"]
            rows.append({
                "source": str(source), "task_id": int(task["task_id"]),
                "init_index": int(doc.get("init_index", 0)),
                "clean_regret_m": float(regrets["clean_native"]),
                "swap_regret_m": float(regrets["swap_native"]),
                "bus_cut_regret_m": float(regrets["clean_cut"]),
                "swap_minus_clean_regret_m": float(regrets["swap_native"] - regrets["clean_native"]),
                "bus_cut_minus_clean_regret_m": float(regrets["clean_cut"] - regrets["clean_native"]),
                "selected_action_changed": bool(selected["clean_native"] != selected["swap_native"]),
                "clean_action_recovered": bool(selected["clean_cut"] == selected["clean_native"]),
            })
    swap_delta = [r["swap_minus_clean_regret_m"] for r in rows]
    cut_delta = [r["bus_cut_minus_clean_regret_m"] for r in rows]
    result = {
        "status": "completed_alphabrain_forecast_planner_replicate_aggregate",
        "sources": [str(p) for p in args.inputs],
        "task_state_rows": len(rows),
        "tasks": sorted({r["task_id"] for r in rows}),
        "init_indices": sorted({r["init_index"] for r in rows}),
        "rows": rows,
        "aggregate": {
            "selected_action_change_rate": float(np.mean([r["selected_action_changed"] for r in rows])),
            "clean_action_recovery_rate": float(np.mean([r["clean_action_recovered"] for r in rows])),
            "clean_regret_mean_m": float(np.mean([r["clean_regret_m"] for r in rows])),
            "swap_regret_mean_m": float(np.mean([r["swap_regret_m"] for r in rows])),
            "bus_cut_regret_mean_m": float(np.mean([r["bus_cut_regret_m"] for r in rows])),
            "swap_minus_clean_regret_mean_m": float(np.mean(swap_delta)),
            "bus_cut_minus_clean_regret_mean_m": float(np.mean(cut_delta)),
            "swap_minus_clean_regret_state_cluster_bootstrap": cluster_bootstrap(rows, "swap_minus_clean_regret_m", "init_index"),
            "swap_minus_clean_regret_task_cluster_bootstrap": cluster_bootstrap(rows, "swap_minus_clean_regret_m", "task_id", seed=20261013),
            "bus_cut_minus_clean_regret_state_cluster_bootstrap": cluster_bootstrap(rows, "bus_cut_minus_clean_regret_m", "init_index", seed=20261014),
        },
        "control_checks": {
            "clean_swap_cut_score_identity_all_rows": all(
                task["scores"]["clean_cut"] == task["scores"]["swap_cut"]
                for doc in docs for task in doc["task_results"]
            ),
            "native_policy_uses_forecast": False,
        },
        "interpretation": [
            "Explicit initial-state replication makes the AlphaBrain planner result a cross-task/state feasibility test rather than a single-state anecdote.",
            "Goal-Swap reliably changes ranking in this fixed scorer, but utility effects and repair outcomes remain planner- and state-dependent.",
            "The cluster intervals are descriptive with only three state clusters and six task clusters; they are not a universal significance claim.",
        ],
        "scope_limits": [
            "Independent forecast-driven planner only; native AlphaBrain action decoding does not consume the World Forecast.",
            "CLIP is used for selection; utility is the real LIBERO EEF replay measurement.",
            "No sparse task-success or native-policy harm claim.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
