#!/usr/bin/env python3
"""Aggregate independent aligned WorldAgen planner runs.

The unit of resampling is the physical LIBERO task.  Seed replicates remain
paired within a task so repeated initializations do not become pseudo-
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


def paired_bootstrap(task_values, n_boot=4000, seed=29):
    """Bootstrap task means; each task_values entry may contain seed repeats."""
    rng = np.random.default_rng(seed)
    task_means = np.asarray([np.mean(v) for v in task_values if len(v)], dtype=float)
    if not len(task_means):
        return {"n_tasks": 0, "mean": None, "ci95": [None, None]}
    draws = rng.choice(task_means, size=(n_boot, len(task_means)), replace=True).mean(axis=1)
    return {
        "n_tasks": int(len(task_means)),
        "mean": float(task_means.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def load(path):
    data = json.loads(Path(path).read_text())
    if data.get("status") != "completed":
        raise ValueError(f"run is not completed: {path}")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    runs = [load(p) for p in args.inputs]
    scorers = runs[0]["scorers"]
    conditions = sorted(runs[0]["rows"][0]["scores"])
    by_task = {}
    for run, path in zip(runs, args.inputs):
        for row in run["rows"]:
            tid = int(row["task_id"])
            by_task.setdefault(tid, []).append((int(run["seed"]), row, str(path)))

    result = {
        "status": "completed_aligned_replicate_aggregation",
        "inputs": [str(p) for p in args.inputs],
        "seeds": [int(r["seed"]) for r in runs],
        "forecast_horizons": sorted({int(r["forecast_horizon"]) for r in runs}),
        "replay_horizons": sorted({int(r["replay_horizon"]) for r in runs}),
        "n_tasks": len(by_task),
        "task_seed_pairs": sum(len(v) for v in by_task.values()),
        "protocol": runs[0]["protocol"],
        "scorers": scorers,
        "conditions": conditions,
        "per_scorer": {},
        "control_cut_max_abs_score_diff": {},
    }

    for scorer in scorers:
        out = {}
        for condition in conditions:
            selected_eef, regret, changed, rank, corr = [], [], [], [], []
            deltas_eef, deltas_regret = [], []
            task_delta_eef, task_delta_regret = {}, {}
            for tid, entries in by_task.items():
                task_delta_eef.setdefault(tid, [])
                task_delta_regret.setdefault(tid, [])
                for _, row, _ in entries:
                    utility = np.asarray([x["eef_approach_progress"] for x in row["candidate_replays"]], dtype=float)
                    score = np.asarray(row["scores"][condition][scorer], dtype=float)
                    clean_score = np.asarray(row["scores"]["clean"][scorer], dtype=float)
                    idx = int(np.argmax(score))
                    clean_idx = int(np.argmax(clean_score))
                    best = int(np.argmax(utility))
                    selected = float(utility[idx])
                    reg = float(utility[best] - utility[idx])
                    selected_eef.append(selected)
                    regret.append(reg)
                    changed.append(int(idx != clean_idx))
                    rank.append(int(np.where(np.argsort(-utility) == idx)[0][0] + 1))
                    if np.std(score) > 0 and np.std(utility) > 0:
                        corr.append(float(np.corrcoef(np.argsort(np.argsort(score)), np.argsort(np.argsort(utility)))[0, 1]))
                    else:
                        corr.append(0.0)
                    if condition != "clean":
                        clean_idx_utility = float(utility[clean_idx])
                        clean_reg = float(utility[best] - clean_idx_utility)
                        de = selected - clean_idx_utility
                        dr = reg - clean_reg
                        deltas_eef.append(de)
                        deltas_regret.append(dr)
                        task_delta_eef[tid].append(de)
                        task_delta_regret[tid].append(dr)
            record = {
                "selected_eef_progress": summary(selected_eef),
                "regret_vs_candidate_best": summary(regret),
                "selected_action_changed_vs_clean": summary(changed),
                "selected_true_utility_rank": summary(rank),
                "score_vs_true_utility_spearman": summary(corr),
            }
            if condition != "clean":
                record["eef_delta_vs_clean_selected_action"] = summary(deltas_eef)
                record["regret_delta_vs_clean_selected_action"] = summary(deltas_regret)
                record["task_cluster_bootstrap_eef_delta_ci95"] = paired_bootstrap(list(task_delta_eef.values()))
                record["task_cluster_bootstrap_regret_delta_ci95"] = paired_bootstrap(list(task_delta_regret.values()))
            out[condition] = record
        result["per_scorer"][scorer] = out

    for scorer in scorers:
        result["control_cut_max_abs_score_diff"][scorer] = {}
        for base in ("swap_1", "swap_5"):
            ctrl = f"{base}_control_cut"
            diffs = []
            for entries in by_task.values():
                for _, row, _ in entries:
                    a = np.asarray(row["scores"][base][scorer], dtype=float)
                    b = np.asarray(row["scores"][ctrl][scorer], dtype=float)
                    diffs.append(float(np.max(np.abs(a - b))))
            result["control_cut_max_abs_score_diff"][scorer][base] = float(max(diffs))

    random_regret = []
    for entries in by_task.values():
        for _, row, _ in entries:
            utility = np.asarray([x["eef_approach_progress"] for x in row["candidate_replays"]], dtype=float)
            random_regret.append(float(utility.max() - utility.mean()))
    result["candidate_random_baseline"] = {
        "expected_regret": summary(random_regret),
        "interpretation": "Uniform random candidate expected regret; descriptive calibration only.",
    }

    result["inference_limits"] = [
        "three seeds and six physical tasks; task-cluster bootstrap is descriptive, not a significance test",
        "three-step EEF progress is an intermediate consequence, not task completion",
        "the native policy does not use World Forecast; all decision results use the explicitly separate planner",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
