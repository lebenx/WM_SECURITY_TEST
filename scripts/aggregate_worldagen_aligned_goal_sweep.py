#!/usr/bin/env python3
"""Aggregate legal Goal-offset sweeps across independent seeds.

The physical LIBERO task is the resampling unit.  Seed repetitions are kept
paired within a task, so the table does not turn repeated initializations into
independent task observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summary(values):
    x = np.asarray(values, dtype=float)
    return {
        "n": int(x.size),
        "mean": float(x.mean()) if x.size else None,
        "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0,
    }


def cluster_bootstrap(values_by_task, n_boot=10000, seed=31):
    task_means = np.asarray(
        [np.mean(v) for v in values_by_task.values() if len(v)], dtype=float
    )
    if not len(task_means):
        return {"n_tasks": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    draws = rng.choice(task_means, size=(n_boot, len(task_means)), replace=True).mean(axis=1)
    return {
        "n_tasks": int(len(task_means)),
        "mean": float(task_means.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def ranking_flip_rate(reference, other):
    """Fraction of non-tied candidate pairs whose order changes."""
    flips = total = 0
    for i in range(len(reference)):
        for j in range(i + 1, len(reference)):
            a = np.sign(reference[i] - reference[j])
            b = np.sign(other[i] - other[j])
            if a == 0 or b == 0:
                continue
            total += 1
            flips += int(a != b)
    return float(flips / total) if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path, help="raw sweep summary.json files")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--scorer", default="reference_scalar")
    args = ap.parse_args()

    runs = [json.loads(p.read_text()) for p in args.inputs]
    if not runs:
        raise ValueError("at least one sweep is required")
    if any(r.get("status") != "completed" for r in runs):
        raise ValueError("all inputs must be completed runs")
    offsets = sorted({int(o) for r in runs for o in r.get("goal_offsets", [])})
    task_ids = sorted({int(row["task_id"]) for r in runs for row in r["rows"]})
    records = []
    control_diffs = {str(off): [] for off in offsets}
    forecast_errors = {"clean": [], "clean_bus_cut": []}
    for off in offsets:
        for condition in (f"swap_{off}", f"swap_{off}_bus_cut", f"swap_{off}_control_cut"):
            forecast_errors[condition] = []
    for run, path in zip(runs, args.inputs):
        for row in run["rows"]:
            tid = int(row["task_id"])
            utility = np.asarray(
                [x["eef_approach_progress"] for x in row["candidate_replays"]], dtype=float
            )
            clean_score = np.asarray(row["scores"]["clean"][args.scorer], dtype=float)
            clean_idx = int(np.argmax(clean_score))
            clean_u = float(utility[clean_idx])
            best_idx = int(np.argmax(utility))
            for condition in forecast_errors:
                forecast_errors[condition].extend(
                    float(v) for v in row["candidate_forecast_mse"][condition]
                )
            for off in offsets:
                condition = f"swap_{off}"
                bus_condition = f"{condition}_bus_cut"
                ctrl_condition = f"{condition}_control_cut"
                score = np.asarray(row["scores"][condition][args.scorer], dtype=float)
                bus_score = np.asarray(row["scores"][bus_condition][args.scorer], dtype=float)
                ctrl_score = np.asarray(row["scores"][ctrl_condition][args.scorer], dtype=float)
                control_diffs[str(off)].append(float(np.max(np.abs(score - ctrl_score))))
                idx = int(np.argmax(score))
                bidx = int(np.argmax(bus_score))
                records.append(
                    {
                        "seed": int(run["seed"]),
                        "task_id": tid,
                        "offset": off,
                        "selected": idx,
                        "bus_cut_selected": bidx,
                        "clean_selected": clean_idx,
                        "selected_changed": bool(idx != clean_idx),
                        "ranking_flip_rate": ranking_flip_rate(clean_score, score),
                        "bus_cut_ranking_flip_rate": ranking_flip_rate(clean_score, bus_score),
                        "control_cut_ranking_flip_rate": ranking_flip_rate(clean_score, ctrl_score),
                        "clean_action_recovered": bool(idx == clean_idx),
                        "bus_cut_clean_action_recovered": bool(bidx == clean_idx),
                        "bus_cut_best_candidate": bool(bidx == best_idx),
                        "eef_delta": float(utility[idx] - clean_u),
                        "bus_cut_eef_delta": float(utility[bidx] - clean_u),
                        "regret": float(utility[best_idx] - utility[idx]),
                        "bus_cut_regret": float(utility[best_idx] - utility[bidx]),
                    }
                )

    by_offset = {}
    for off in offsets:
        rows = [r for r in records if r["offset"] == off]
        metrics = {
            "eef_delta_vs_clean": [r["eef_delta"] for r in rows],
            "bus_cut_eef_delta_vs_clean": [r["bus_cut_eef_delta"] for r in rows],
            "regret": [r["regret"] for r in rows],
            "bus_cut_regret": [r["bus_cut_regret"] for r in rows],
            "selected_changed": [float(r["selected_changed"]) for r in rows],
            "ranking_flip_rate": [r["ranking_flip_rate"] for r in rows],
            "bus_cut_ranking_flip_rate": [r["bus_cut_ranking_flip_rate"] for r in rows],
            "control_cut_ranking_flip_rate": [r["control_cut_ranking_flip_rate"] for r in rows],
            "bus_cut_clean_action_recovered": [float(r["bus_cut_clean_action_recovered"]) for r in rows],
            "bus_cut_best_candidate": [float(r["bus_cut_best_candidate"]) for r in rows],
        }
        out = {name: summary(values) for name, values in metrics.items()}
        for name, values in metrics.items():
            task_values = {}
            for r, value in zip(rows, values):
                task_values.setdefault(str(r["task_id"]), []).append(value)
            out[name + "_task_cluster_bootstrap"] = cluster_bootstrap(task_values)
        by_offset[str(off)] = out

    result = {
        "status": "completed_aligned_legal_goal_sweep_aggregation",
        "inputs": [str(p) for p in args.inputs],
        "seeds": [int(r["seed"]) for r in runs],
        "task_ids": task_ids,
        "task_seed_pairs": len(records) // len(offsets),
        "offsets": offsets,
        "scorer": args.scorer,
        "forecast_horizons": sorted({int(r["forecast_horizon"]) for r in runs}),
        "replay_horizons": sorted({int(r["replay_horizon"]) for r in runs}),
        "by_offset": by_offset,
        "forecast_error": {
            condition: summary(values) for condition, values in forecast_errors.items()
        },
        "control_cut_max_abs_score_diff": {
            off: float(max(vals)) if vals else None for off, vals in control_diffs.items()
        },
        "records": records,
        "inference_limits": [
            "two physical tasks and three seeds; task-cluster intervals are descriptive",
            "the offset is a legal Goal perturbation in token space, not a calibrated semantic distance",
            "30-step EEF approach progress is an intermediate real-simulator utility, not task success",
            "Bus-Cut recovery means recovery under the explicit planner and is not native-policy recovery",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
