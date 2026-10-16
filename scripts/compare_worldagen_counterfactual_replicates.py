#!/usr/bin/env python3
"""Aggregate independent counterfactual ranking replicates descriptively."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def mean_sd(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def task_cluster_bootstrap(values_by_task: dict[int, list[float]],
                           draws: int = 10000, seed: int = 0) -> dict[str, Any]:
    """Percentile interval after resampling physical tasks as clusters."""
    task_ids = sorted(values_by_task)
    if not task_ids:
        return {"n_tasks": 0, "ci95": [None, None]}
    task_means = np.asarray([np.mean(values_by_task[tid]) for tid in task_ids], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(task_means), size=(draws, len(task_means)))
    boot = task_means[sampled].mean(axis=1)
    return {
        "n_tasks": len(task_ids),
        "task_means": {str(tid): float(values_by_task[tid][0]) if len(values_by_task[tid]) == 1
                       else float(np.mean(values_by_task[tid])) for tid in task_ids},
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("candidate_set_frozen_across_conditions") is not True:
        raise ValueError(f"{path}: candidate set was not frozen")
    if data.get("selection_target_fixed_to_original_goal") is not True:
        raise ValueError(f"{path}: score target was not fixed to original goal")
    if data.get("statistics") is None:
        sidecar = path.with_name(path.stem + "_statistics.json")
        if sidecar.exists():
            data["statistics"] = json.loads(sidecar.read_text()).get("statistics")
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    runs = [load(path) for path in args.inputs]
    task_sets = [tuple(sorted(int(row["task_id"]) for row in run["rows"])) for run in runs]
    if len(set(task_sets)) != 1:
        raise ValueError(f"task mismatch across replicates: {task_sets}")
    conditions = list(runs[0]["conditions"])
    if any(list(run["conditions"]) != conditions for run in runs[1:]):
        raise ValueError("condition mismatch across replicates")

    flattened: dict[str, dict[str, list[float]]] = {
        c: {"ranking_flip_rate": [], "eef_delta_vs_clean": [], "eef_progress": []}
        for c in conditions if c != "clean"
    }
    clustered: dict[str, dict[str, dict[int, list[float]]]] = {
        c: {"ranking_flip_rate": {}, "eef_delta_vs_clean": {}}
        for c in conditions if c != "clean"
    }
    per_replicate = []
    selected_agreement: dict[str, list[bool]] = {c: [] for c in conditions}
    reference_selected = None
    for run_idx, run in enumerate(runs):
        row_map = {int(row["task_id"]): row for row in run["rows"]}
        per_replicate.append({
            "input": str(args.inputs[run_idx]),
            "seed": run["rows"][0].get("candidate_seed"),
            "statistics": run.get("statistics"),
        })
        if reference_selected is None:
            reference_selected = {c: {tid: row_map[tid]["selected"][c] for tid in row_map} for c in conditions}
        for condition in conditions:
            if condition != "clean":
                for row in run["rows"]:
                    flattened[condition]["ranking_flip_rate"].append(row["ranking_flip_rate"][condition])
                    flattened[condition]["eef_progress"].append(row["replays"][condition]["eef_approach_progress"])
                    flattened[condition]["eef_delta_vs_clean"].append(
                        row["replays"][condition]["eef_approach_progress"]
                        - row["replays"]["clean"]["eef_approach_progress"]
                    )
                    clustered[condition]["ranking_flip_rate"].setdefault(int(row["task_id"]), []).append(
                        row["ranking_flip_rate"][condition]
                    )
                    clustered[condition]["eef_delta_vs_clean"].setdefault(int(row["task_id"]), []).append(
                        row["replays"][condition]["eef_approach_progress"]
                        - row["replays"]["clean"]["eef_approach_progress"]
                    )
            for task_id, row in row_map.items():
                selected_agreement[condition].append(
                    row["selected"][condition] == reference_selected[condition][task_id]
                )

    out: dict[str, Any] = {
        "inputs": [str(x) for x in args.inputs],
        "n_replicates": len(runs),
        "task_ids": list(task_sets[0]),
        "conditions": conditions,
        "per_replicate": per_replicate,
        "pooled_task_replicate_metrics": {
            condition: {metric: mean_sd(values) for metric, values in metrics.items()}
            for condition, metrics in flattened.items()
        },
        "physical_task_cluster_bootstrap": {
            condition: {
                metric: task_cluster_bootstrap(values)
                for metric, values in metrics.items()
            }
            for condition, metrics in clustered.items()
        },
        "selected_action_agreement_to_first_replicate": {
            condition: mean_sd([float(x) for x in values])
            for condition, values in selected_agreement.items()
        },
        "control_cut_score_identity": [
            run.get("statistics", {}).get("control_cut_score_identity") for run in runs
        ],
        "interpretation": (
            "Descriptive aggregation over task-by-seed replicates. The candidate action chunks, "
            "original-goal scoring target, and model checkpoint are fixed; only the LIBERO seed "
            "changes. No significance claim is made."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
