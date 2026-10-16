#!/usr/bin/env python3
"""Candidate-cardinality sensitivity using already replayed fixed candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def flip_rate(a, b):
    flips = total = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            x, y = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if x == 0 or y == 0:
                continue
            total += 1
            flips += int(x != y)
    return float(flips / total) if total else 0.0


def summary(v):
    x = np.asarray(v, dtype=float)
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0}


def bootstrap(values_by_task, seed=43, draws=10000):
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
    args = ap.parse_args()
    runs = [json.loads(p.read_text()) for p in args.inputs]
    subsets = {"all8": list(range(8)), "linspace4": [0, 2, 4, 7]}
    records = []
    for run in runs:
        for row in run["rows"]:
            utility = np.asarray([x["eef_approach_progress"] for x in row["candidate_replays"]], dtype=float)
            for subset_name, subset in subsets.items():
                u = utility[subset]
                clean = np.asarray(row["scores"]["clean"][args.scorer], dtype=float)[subset]
                clean_idx = int(np.argmax(clean))
                clean_u = float(u[clean_idx])
                for off in run["goal_offsets"]:
                    c, bc, ctrl = f"swap_{int(off)}", f"swap_{int(off)}_bus_cut", f"swap_{int(off)}_control_cut"
                    score = np.asarray(row["scores"][c][args.scorer], dtype=float)[subset]
                    bus = np.asarray(row["scores"][bc][args.scorer], dtype=float)[subset]
                    control = np.asarray(row["scores"][ctrl][args.scorer], dtype=float)[subset]
                    idx, bidx = int(np.argmax(score)), int(np.argmax(bus))
                    records.append({
                        "task_id": int(row["task_id"]), "seed": int(run["seed"]), "offset": int(off), "subset": subset_name,
                        "candidate_indices": subset, "ranking_flip_rate": flip_rate(clean, score),
                        "control_cut_score_identity": float(np.max(np.abs(score - control))),
                        "selected_changed": bool(idx != clean_idx), "clean_action_recovered": bool(idx == clean_idx),
                        "eef_delta": float(u[idx] - clean_u), "regret": float(u.max() - u[idx]),
                        "bus_cut_regret": float(u.max() - u[bidx]), "bus_cut_best_candidate": bool(bidx == int(np.argmax(u))),
                        "random_expected_regret": float(u.max() - u.mean()),
                    })

    by_subset = {}
    for subset_name in subsets:
        rows = [r for r in records if r["subset"] == subset_name]
        by_offset = {}
        for off in sorted({r["offset"] for r in rows}):
            rr = [r for r in rows if r["offset"] == off]
            task_eef, task_reg = {}, {}
            for r in rr:
                task_eef.setdefault(str(r["task_id"]), []).append(r["eef_delta"])
                task_reg.setdefault(str(r["task_id"]), []).append(r["regret"])
            by_offset[str(off)] = {
                "n": len(rr), "eef_delta_vs_clean": summary([r["eef_delta"] for r in rr]),
                "regret": summary([r["regret"] for r in rr]),
                "bus_cut_regret": summary([r["bus_cut_regret"] for r in rr]),
                "random_expected_regret": summary([r["random_expected_regret"] for r in rr]),
                "ranking_flip_rate": summary([r["ranking_flip_rate"] for r in rr]),
                "selected_changed_rate": float(np.mean([r["selected_changed"] for r in rr])),
                "clean_action_recovery_rate": float(np.mean([r["clean_action_recovered"] for r in rr])),
                "bus_cut_best_candidate_rate": float(np.mean([r["bus_cut_best_candidate"] for r in rr])),
                "control_cut_max_abs_score_diff": float(max(r["control_cut_score_identity"] for r in rr)),
                "eef_delta_task_cluster_bootstrap": bootstrap(task_eef),
                "regret_task_cluster_bootstrap": bootstrap(task_reg),
            }
        by_subset[subset_name] = {"candidate_indices": subsets[subset_name], "by_offset": by_offset}

    result = {
        "status": "completed_candidate_cardinality_sensitivity",
        "inputs": [str(p) for p in args.inputs], "scorer": args.scorer,
        "subsets": subsets, "records": records, "by_subset": by_subset,
        "inference_limits": [
            "The four-candidate result is a fixed subset of the eight replayed candidates, not a newly sampled four-candidate protocol.",
            "All conditions share each subset; cross-subset utility comparisons are descriptive.",
            "Utility is read only after model scores and selected actions are fixed.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
