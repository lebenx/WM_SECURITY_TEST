#!/usr/bin/env python3
"""Evaluate forecast-only Goal-offset selection out of sample.

For each held-out seed, choose one legal offset using only forecast features
from the other seeds, then apply that single offset to every held-out task.
Replay utility is read only after the offset has been fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


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
            x, y = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if x == 0 or y == 0:
                continue
            total += 1
            flips += int(x != y)
    return float(flips / total) if total else 0.0


def summary(values):
    x = np.asarray(values, dtype=float)
    return {"n": int(len(x)), "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0}


def bootstrap(values_by_task, seed=71, draws=10000):
    means = np.asarray([np.mean(v) for v in values_by_task.values() if v], dtype=float)
    if not len(means):
        return {"n_tasks": 0, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    sample = rng.choice(means, size=(draws, len(means)), replace=True).mean(axis=1)
    return {"n_tasks": int(len(means)), "mean": float(means.mean()),
            "ci95": [float(np.quantile(sample, .025)), float(np.quantile(sample, .975))]}


def make_feature_rows(run, root, offsets, scorer):
    out = []
    for row in run["rows"]:
        clean = np.asarray(row["scores"]["clean"][scorer], dtype=float)
        clean_rank = rank01(clean)
        payload = torch.load(root / f"task_{int(row['task_id'])}_forecasts.pt",
                             map_location="cpu", weights_only=False)
        worlds = payload["worlds_step3"]
        clean_world = worlds["clean"].float()
        for off in offsets:
            score = np.asarray(row["scores"][f"swap_{off}"][scorer], dtype=float)
            srank = rank01(score)
            diff = (worlds[f"swap_{off}"].float() - clean_world).pow(2).mean(dim=(1, 2))
            out.append({
                "task_id": int(row["task_id"]), "seed": int(run["seed"]),
                "offset": int(off),
                "mean_latent_mse": float(diff.mean().item()),
                "max_candidate_latent_mse": float(diff.max().item()),
                "pairwise_flip": flip_rate(clean, score),
                "rank_l1": float(np.abs(srank - clean_rank).mean()),
                "raw_z_l1": float(np.abs((score-score.mean())/(score.std()+1e-12) -
                                           (clean-clean.mean())/(clean.std()+1e-12)).mean()),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path, help="matching sweep summary.json files")
    ap.add_argument("--run-roots", nargs="+", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--scorer", default="reference_scalar")
    args = ap.parse_args()
    if len(args.inputs) != len(args.run_roots):
        raise ValueError("inputs and run-roots must have the same length")
    runs = [json.loads(p.read_text()) for p in args.inputs]
    offsets = sorted({int(o) for r in runs for o in r["goal_offsets"]})
    feature_names = ["mean_latent_mse", "max_candidate_latent_mse",
                     "pairwise_flip", "rank_l1", "raw_z_l1"]
    feature_rows = []
    lookup = {}
    for run, root in zip(runs, args.run_roots):
        rows = make_feature_rows(run, root, offsets, args.scorer)
        feature_rows.extend(rows)
        for row in run["rows"]:
            lookup[(int(run["seed"]), int(row["task_id"]))] = row

    seeds = sorted({int(r["seed"]) for r in runs})
    records = []
    fold_choices = {}
    for heldout in seeds:
        train = [r for r in feature_rows if r["seed"] != heldout]
        test = [r for r in feature_rows if r["seed"] == heldout]
        choices = {}
        for feature in feature_names:
            means = {off: float(np.mean([r[feature] for r in train if r["offset"] == off]))
                     for off in offsets}
            choices[feature] = max(offsets, key=lambda o: (means[o], -o))
        fold_choices[str(heldout)] = {
            "selected_offsets": choices,
            "training_feature_means": {
                feature: {str(off): float(np.mean([r[feature] for r in train if r["offset"] == off]))
                          for off in offsets} for feature in feature_names},
        }
        for feature, chosen_offset in choices.items():
            for task_id in sorted({r["task_id"] for r in test}):
                row = lookup[(heldout, task_id)]
                utility = np.asarray([x["eef_approach_progress"] for x in row["candidate_replays"]], dtype=float)
                clean_score = np.asarray(row["scores"]["clean"][args.scorer], dtype=float)
                clean_idx = int(np.argmax(clean_score))
                clean_u = float(utility[clean_idx])
                best_u = float(utility.max())
                selected_score = np.asarray(row["scores"][f"swap_{chosen_offset}"][args.scorer], dtype=float)
                selected_idx = int(np.argmax(selected_score))
                all_rows = []
                for off in offsets:
                    idx = int(np.argmax(np.asarray(row["scores"][f"swap_{off}"][args.scorer], dtype=float)))
                    all_rows.append(float(best_u - utility[idx]))
                random_regret = float(np.mean(all_rows))
                records.append({
                    "heldout_seed": heldout, "task_id": task_id, "selector": feature,
                    "selected_offset": int(chosen_offset), "selected_candidate": selected_idx,
                    "eef_delta_vs_clean": float(utility[selected_idx] - clean_u),
                    "regret": float(best_u - utility[selected_idx]),
                    "random_offset_mean_regret": random_regret,
                    "regret_delta_vs_random": float(best_u - utility[selected_idx] - random_regret),
                    "selected_action_changed": bool(selected_idx != clean_idx),
                })

    by_selector = {}
    for selector in feature_names:
        rows = [r for r in records if r["selector"] == selector]
        task_reg = {}
        for r in rows:
            task_reg.setdefault(str(r["task_id"]), []).append(r["regret_delta_vs_random"])
        by_selector[selector] = {
            "n": len(rows),
            "heldout_seed_offset_choices": {s: fold_choices[s]["selected_offsets"][selector] for s in fold_choices},
            "selected_offset_histogram": {str(o): int(sum(r["selected_offset"] == o for r in rows)) for o in offsets},
            "eef_delta_vs_clean": summary([r["eef_delta_vs_clean"] for r in rows]),
            "regret": summary([r["regret"] for r in rows]),
            "random_offset_mean_regret": summary([r["random_offset_mean_regret"] for r in rows]),
            "regret_delta_vs_random": summary([r["regret_delta_vs_random"] for r in rows]),
            "regret_delta_vs_random_task_cluster_bootstrap": bootstrap(task_reg),
            "selected_action_changed_rate": float(np.mean([r["selected_action_changed"] for r in rows])),
            "records": rows,
        }
    result = {
        "status": "completed_out_of_sample_forecast_only_goal_attacker",
        "inputs": [str(p) for p in args.inputs], "run_roots": [str(p) for p in args.run_roots],
        "scorer": args.scorer, "offsets": offsets, "heldout_seed_folds": seeds,
        "fold_choices": fold_choices, "records": records, "by_selector": by_selector,
        "selection_rule": "for each held-out seed, choose one legal offset from forecast-only feature means on the other seeds; apply it to all held-out tasks before reading replay utility",
        "inference_limits": [
            "This is an out-of-sample planner probe, not the native WorldAgen/AlphaBrain policy.",
            "The held-out seed is never used to choose the offset, but the held-out task forecast feature is not hidden because the transfer rule uses a single global offset.",
            "Latent divergence remains model-space divergence, not physical forecast error or attack success by itself.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
