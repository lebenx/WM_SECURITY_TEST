#!/usr/bin/env python3
"""Evaluate legal-Goal attackers selected using forecast outputs only.

For each fixed task/seed, the attacker chooses one offset from 1--9 using
only the precomputed candidate score vectors.  Real replay utility is joined
after the offset and candidate have been selected.
"""
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


def summary(x):
    x = np.asarray(x, dtype=float)
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0}


def bootstrap(values_by_task, seed=47, draws=10000):
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
    offsets = sorted({int(o) for r in runs for o in r["goal_offsets"]})
    records = []
    for run in runs:
        for row in run["rows"]:
            # All attacker features are computed before reading any replay utility.
            clean_score = np.asarray(row["scores"]["clean"][args.scorer], dtype=float)
            clean_rank = rank01(clean_score)
            candidates = []
            for off in offsets:
                score = np.asarray(row["scores"][f"swap_{off}"][args.scorer], dtype=float)
                srank = rank01(score)
                candidates.append({
                    "offset": off,
                    "flip": flip_rate(clean_score, score),
                    "rank_l1": float(np.abs(srank - clean_rank).mean()),
                    "raw_z_l1": float(np.abs((score - score.mean()) / (score.std() + 1e-12) -
                                                (clean_score - clean_score.mean()) / (clean_score.std() + 1e-12)).mean()),
                })
            selection = {
                "max_pairwise_flip": max(candidates, key=lambda x: (x["flip"], -x["offset"])),
                "max_rank_l1": max(candidates, key=lambda x: (x["rank_l1"], -x["offset"])),
                "max_raw_z_l1": max(candidates, key=lambda x: (x["raw_z_l1"], -x["offset"])),
            }
            # Utility is accessed only after forecast-only selector decisions are fixed.
            utility = np.asarray([x["eef_approach_progress"] for x in row["candidate_replays"]], dtype=float)
            clean_idx = int(np.argmax(clean_score))
            clean_u = float(utility[clean_idx])
            best_u = float(utility.max())
            random_rows = []
            for off in offsets:
                score = np.asarray(row["scores"][f"swap_{off}"][args.scorer], dtype=float)
                idx = int(np.argmax(score))
                random_rows.append({
                    "offset": off, "eef_delta": float(utility[idx] - clean_u),
                    "regret": float(best_u - utility[idx]), "selected": idx,
                })
            for name, chosen in selection.items():
                off = int(chosen["offset"])
                selected = next(x for x in random_rows if x["offset"] == off)
                records.append({
                    "task_id": int(row["task_id"]), "seed": int(run["seed"]),
                    "selector": name, "selected_offset": off,
                    "forecast_selector_value": chosen["flip"] if name == "max_pairwise_flip" else chosen["rank_l1"] if name == "max_rank_l1" else chosen["raw_z_l1"],
                    "selected_candidate": selected["selected"],
                    "eef_delta_vs_clean": selected["eef_delta"],
                    "regret": selected["regret"],
                    "random_offset_mean_eef_delta": float(np.mean([x["eef_delta"] for x in random_rows])),
                    "random_offset_mean_regret": float(np.mean([x["regret"] for x in random_rows])),
                    "regret_delta_vs_random": float(selected["regret"] - np.mean([x["regret"] for x in random_rows])),
                    "eef_delta_delta_vs_random": float(selected["eef_delta"] - np.mean([x["eef_delta"] for x in random_rows])),
                    "selected_action_changed": bool(selected["selected"] != clean_idx),
                })

    by_selector = {}
    for selector in sorted({r["selector"] for r in records}):
        rows = [r for r in records if r["selector"] == selector]
        task_reg, task_eef = {}, {}
        for r in rows:
            task_reg.setdefault(str(r["task_id"]), []).append(r["regret_delta_vs_random"])
            task_eef.setdefault(str(r["task_id"]), []).append(r["eef_delta_delta_vs_random"])
        by_selector[selector] = {
            "n": len(rows),
            "selected_offset_histogram": {str(o): int(sum(r["selected_offset"] == o for r in rows)) for o in offsets},
            "eef_delta_vs_clean": summary([r["eef_delta_vs_clean"] for r in rows]),
            "regret": summary([r["regret"] for r in rows]),
            "random_offset_mean_eef_delta": summary([r["random_offset_mean_eef_delta"] for r in rows]),
            "random_offset_mean_regret": summary([r["random_offset_mean_regret"] for r in rows]),
            "regret_delta_vs_random": summary([r["regret_delta_vs_random"] for r in rows]),
            "eef_delta_delta_vs_random": summary([r["eef_delta_delta_vs_random"] for r in rows]),
            "regret_delta_vs_random_task_cluster_bootstrap": bootstrap(task_reg),
            "eef_delta_delta_vs_random_task_cluster_bootstrap": bootstrap(task_eef),
            "selected_action_changed_rate": float(np.mean([r["selected_action_changed"] for r in rows])),
            "records": [r for r in rows],
        }

    result = {
        "status": "completed_forecast_only_legal_goal_attacker",
        "inputs": [str(p) for p in args.inputs], "scorer": args.scorer,
        "offsets": offsets, "records": records, "by_selector": by_selector,
        "selection_rule": "choose offset from model score/rank features before reading candidate_replays",
        "inference_limits": [
            "The attacker selects among nine legal offsets and cannot use replay utility, future observations, or target MSE.",
            "This tests forecast-guided offset choice, not a native policy attack; native policy does not consume World Forecast.",
            "The random-offset baseline is descriptive and paired within each fixed task/seed.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
