#!/usr/bin/env python3
"""Test whether forecast-error changes track ranking and real utility changes.

This is a post-selection diagnostic over the precomputed six-task legal-Goal
sweep.  It never uses replay outcomes to choose an action; replay utility is
only joined after the forecast and ranking records are already fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def ranking_flip_rate(reference, other):
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


def rank_average(x):
    """Average ranks with ties, without requiring scipy."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def corr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": int(len(x)), "pearson": None, "spearman": None}
    return {
        "n": int(len(x)),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(rank_average(x), rank_average(y))[0, 1]),
    }


def summary(x):
    x = np.asarray(x, dtype=float)
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--scorer", default="reference_scalar")
    args = ap.parse_args()
    runs = [json.loads(p.read_text()) for p in args.inputs]
    records = []
    offsets = sorted({int(o) for r in runs for o in r["goal_offsets"]})
    for run in runs:
        for row in run["rows"]:
            clean_score = np.asarray(row["scores"]["clean"][args.scorer], dtype=float)
            clean_idx = int(np.argmax(clean_score))
            utility = np.asarray([x["eef_approach_progress"] for x in row["candidate_replays"]], dtype=float)
            clean_u = float(utility[clean_idx])
            best_u = float(utility.max())
            clean_mse = np.asarray(row["candidate_forecast_mse"]["clean"], dtype=float)
            for off in offsets:
                c = f"swap_{off}"
                b = f"{c}_bus_cut"
                score = np.asarray(row["scores"][c][args.scorer], dtype=float)
                bus_score = np.asarray(row["scores"][b][args.scorer], dtype=float)
                idx = int(np.argmax(score))
                bidx = int(np.argmax(bus_score))
                swap_mse = np.asarray(row["candidate_forecast_mse"][c], dtype=float)
                records.append({
                    "seed": int(run["seed"]), "task_id": int(row["task_id"]), "offset": off,
                    "ranking_flip_rate": ranking_flip_rate(clean_score, score),
                    "selected_changed": bool(idx != clean_idx),
                    "forecast_error_delta_all_candidates": float(swap_mse.mean() - clean_mse.mean()),
                    "forecast_error_delta_selected": float(swap_mse[idx] - clean_mse[clean_idx]),
                    "forecast_error_clean_selected": float(clean_mse[clean_idx]),
                    "forecast_error_swap_selected": float(swap_mse[idx]),
                    "eef_delta": float(utility[idx] - clean_u),
                    "regret": float(best_u - utility[idx]),
                    "bus_cut_eef_delta": float(utility[bidx] - clean_u),
                    "bus_cut_regret": float(best_u - utility[bidx]),
                })

    def correlations(rows, xkey):
        return {
            "flip": corr([r[xkey] for r in rows], [r["ranking_flip_rate"] for r in rows]),
            "eef_delta": corr([r[xkey] for r in rows], [r["eef_delta"] for r in rows]),
            "regret": corr([r[xkey] for r in rows], [r["regret"] for r in rows]),
        }

    by_offset = {}
    for off in offsets:
        rows = [r for r in records if r["offset"] == off]
        by_offset[str(off)] = {
            "n": len(rows),
            "forecast_error_delta_all_candidates": summary([r["forecast_error_delta_all_candidates"] for r in rows]),
            "forecast_error_delta_selected": summary([r["forecast_error_delta_selected"] for r in rows]),
            "correlations_all_candidates_delta": correlations(rows, "forecast_error_delta_all_candidates"),
            "correlations_selected_delta": correlations(rows, "forecast_error_delta_selected"),
        }

    # Pair means keep the nine offsets for one task-seed pair together when we
    # report the descriptive correlation check.
    pair_groups = {}
    for r in records:
        pair_groups.setdefault((r["task_id"], r["seed"]), []).append(r)
    pair_records = []
    for key, rows in pair_groups.items():
        pair_records.append({
            "task_id": key[0], "seed": key[1],
            "forecast_error_delta_all_candidates": float(np.mean([r["forecast_error_delta_all_candidates"] for r in rows])),
            "forecast_error_delta_selected": float(np.mean([r["forecast_error_delta_selected"] for r in rows])),
            "ranking_flip_rate": float(np.mean([r["ranking_flip_rate"] for r in rows])),
            "eef_delta": float(np.mean([r["eef_delta"] for r in rows])),
            "regret": float(np.mean([r["regret"] for r in rows])),
        })
    result = {
        "status": "completed_goal_sweep_forecast_error_mediation_diagnostic",
        "inputs": [str(p) for p in args.inputs],
        "scorer": args.scorer,
        "task_ids": sorted({r["task_id"] for r in records}),
        "seeds": sorted({r["seed"] for r in records}),
        "offsets": offsets,
        "records": records,
        "pooled_correlations": {
            "all_candidates_delta": correlations(records, "forecast_error_delta_all_candidates"),
            "selected_delta": correlations(records, "forecast_error_delta_selected"),
        },
        "task_seed_pair_mean_correlations": {
            "all_candidates_delta": correlations(pair_records, "forecast_error_delta_all_candidates"),
            "selected_delta": correlations(pair_records, "forecast_error_delta_selected"),
        },
        "by_offset": by_offset,
        "inference_limits": [
            "This is an associational mediation diagnostic, not an identification of a causal statistical mediator.",
            "Forecast Error is measured against replayed next observations; utility is read only after selection.",
            "Native policy does not consume World Forecast; all decision records belong to the explicit independent planner.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
