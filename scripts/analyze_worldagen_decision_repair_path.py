#!/usr/bin/env python3
"""Evaluate a fixed Clean-anchor / Bus-Cut rank repair path.

The interpolation weights are fixed before replay outcomes are read.  Scores
are converted to within-row percentile ranks, avoiding arbitrary CLIP-score
scale differences.  This is a deterministic repair diagnostic, not a learned
policy and not a claim that the clean branch is always available in deployment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rank01(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        out[order[i:j]] = ((i + j - 1) / 2) / max(len(x) - 1, 1)
        i = j
    return out


def flip_rate(a, b):
    flips = total = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            sa, sb = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if sa == 0 or sb == 0:
                continue
            total += 1
            flips += int(sa != sb)
    return float(flips / total) if total else 0.0


def summary(v):
    x = np.asarray(v, dtype=float)
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0}


def bootstrap(values_by_task, seed=41, draws=10000):
    means = np.asarray([np.mean(v) for v in values_by_task.values() if v], dtype=float)
    if not len(means):
        return {"n_tasks": 0, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    sample = rng.choice(means, size=(draws, len(means)), replace=True).mean(axis=1)
    return {"n_tasks": int(len(means)), "mean": float(means.mean()),
            "ci95": [float(np.quantile(sample, .025)), float(np.quantile(sample, .975))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--scorer", default="reference_scalar")
    ap.add_argument("--weights", default="0,0.25,0.5,0.75,1")
    args = ap.parse_args()
    weights = [float(x) for x in args.weights.split(",")]
    runs = [json.loads(p.read_text()) for p in args.inputs]
    records = []
    for run in runs:
        for row in run["rows"]:
            utility = np.asarray([x["eef_approach_progress"] for x in row["candidate_replays"]], dtype=float)
            clean = np.asarray(row["scores"]["clean"][args.scorer], dtype=float)
            clean_rank = rank01(clean)
            clean_idx = int(np.argmax(clean))
            clean_u = float(utility[clean_idx])
            best = float(utility.max())
            for off in run["goal_offsets"]:
                c = f"swap_{int(off)}"
                bc = f"{c}_bus_cut"
                bus = np.asarray(row["scores"][bc][args.scorer], dtype=float)
                bus_rank = rank01(bus)
                native = np.asarray(row["scores"][c][args.scorer], dtype=float)
                native_rank = rank01(native)
                for w in weights:
                    hybrid_rank = w * clean_rank + (1.0 - w) * bus_rank
                    idx = int(np.argmax(hybrid_rank))
                    records.append({
                        "task_id": int(row["task_id"]), "seed": int(run["seed"]), "offset": int(off), "weight_clean": w,
                        "selected": idx, "clean_selected": clean_idx,
                        "clean_action_recovered": bool(idx == clean_idx),
                        "selected_changed_vs_clean": bool(idx != clean_idx),
                        "rank_flip_vs_clean": flip_rate(clean, hybrid_rank),
                        "rank_flip_vs_native_swap": flip_rate(native_rank, hybrid_rank),
                        "eef_delta_vs_clean": float(utility[idx] - clean_u),
                        "regret": float(best - utility[idx]),
                        "bus_cut_selected": int(np.argmax(bus)),
                        "bus_cut_regret": float(best - utility[int(np.argmax(bus))]),
                    })

    by_weight = {}
    for w in weights:
        rows = [r for r in records if r["weight_clean"] == w]
        task_eef, task_reg = {}, {}
        for r in rows:
            task_eef.setdefault(str(r["task_id"]), []).append(r["eef_delta_vs_clean"])
            task_reg.setdefault(str(r["task_id"]), []).append(r["regret"])
        by_weight[str(w)] = {
            "n": len(rows),
            "eef_delta_vs_clean": summary([r["eef_delta_vs_clean"] for r in rows]),
            "regret": summary([r["regret"] for r in rows]),
            "clean_action_recovery_rate": float(np.mean([r["clean_action_recovered"] for r in rows])),
            "selected_action_changed_rate": float(np.mean([r["selected_changed_vs_clean"] for r in rows])),
            "ranking_flip_vs_clean": summary([r["rank_flip_vs_clean"] for r in rows]),
            "ranking_flip_vs_native_swap": summary([r["rank_flip_vs_native_swap"] for r in rows]),
            "eef_delta_task_cluster_bootstrap": bootstrap(task_eef),
            "regret_task_cluster_bootstrap": bootstrap(task_reg),
        }

    result = {
        "status": "completed_fixed_clean_anchor_bus_cut_repair_path",
        "inputs": [str(p) for p in args.inputs],
        "scorer": args.scorer,
        "weights": weights,
        "weight_definition": "weight_clean * rank(Clean) + (1-weight_clean) * rank(Bus-Cut)",
        "records": records,
        "by_weight": by_weight,
        "inference_limits": [
            "Weights were fixed before reading replay utility; this is an analysis diagnostic, not learned tuning.",
            "The Clean reference branch is assumed available to this repair path; deployment availability is not established.",
            "Utility is real 30-step simulator EEF progress and is never used for selection.",
            "Native policy does not consume World Forecast; all rows belong to the explicit independent planner.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
