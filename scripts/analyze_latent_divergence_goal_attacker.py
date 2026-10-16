#!/usr/bin/env python3
"""Select legal Goal offsets using saved WorldAgen latent divergence only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def summary(x):
    x = np.asarray(x, dtype=float)
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0}


def bootstrap(values_by_task, seed=53, draws=10000):
    means = np.asarray([np.mean(v) for v in values_by_task.values() if v], dtype=float)
    if not len(means):
        return {"n_tasks": 0, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    sample = rng.choice(means, size=(draws, len(means)), replace=True).mean(axis=1)
    return {"n_tasks": int(len(means)), "mean": float(means.mean()),
            "ci95": [float(np.quantile(sample, .025)), float(np.quantile(sample, .975))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path, help="six-task sweep summary.json files")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--scorer", default="reference_scalar")
    ap.add_argument("--run-roots", nargs="+", required=True, type=Path,
                    help="matching all6 seed output directories containing task_*_forecasts.pt")
    args = ap.parse_args()
    if len(args.inputs) != len(args.run_roots):
        raise ValueError("inputs and run-roots must have the same length")
    runs = [json.loads(p.read_text()) for p in args.inputs]
    offsets = sorted({int(o) for r in runs for o in r["goal_offsets"]})
    records = []
    for run, root in zip(runs, args.run_roots):
        for row in run["rows"]:
            tid = int(row["task_id"])
            payload = torch.load(root / f"task_{tid}_forecasts.pt", map_location="cpu", weights_only=False)
            worlds = payload["worlds_step3"]
            clean_world = worlds["clean"].float()
            # Compute all forecast-only offset features before touching replay utility.
            features = {}
            for off in offsets:
                swap = worlds[f"swap_{off}"].float()
                diff = (swap - clean_world).pow(2).mean(dim=(1, 2))
                features[off] = {
                    "mean_latent_mse": float(diff.mean().item()),
                    "max_candidate_latent_mse": float(diff.max().item()),
                    "candidate_latent_mse": diff.tolist(),
                }
            selectors = {
                "max_mean_latent_mse": max(features, key=lambda o: (features[o]["mean_latent_mse"], -o)),
                "max_candidate_latent_mse": max(features, key=lambda o: (features[o]["max_candidate_latent_mse"], -o)),
            }
            utility = np.asarray([x["eef_approach_progress"] for x in row["candidate_replays"]], dtype=float)
            clean_score = np.asarray(row["scores"]["clean"][args.scorer], dtype=float)
            clean_idx = int(np.argmax(clean_score))
            clean_u = float(utility[clean_idx])
            best_u = float(utility.max())
            expected = []
            for off in offsets:
                score = np.asarray(row["scores"][f"swap_{off}"][args.scorer], dtype=float)
                idx = int(np.argmax(score))
                expected.append({"offset": off, "eef_delta": float(utility[idx] - clean_u),
                                 "regret": float(best_u - utility[idx]), "selected": idx})
            for selector, off in selectors.items():
                chosen = next(x for x in expected if x["offset"] == off)
                records.append({
                    "task_id": tid, "seed": int(run["seed"]), "selector": selector,
                    "selected_offset": int(off), "selected_candidate": int(chosen["selected"]),
                    "selected_mean_latent_mse": features[off]["mean_latent_mse"],
                    "selected_max_candidate_latent_mse": features[off]["max_candidate_latent_mse"],
                    "eef_delta_vs_clean": chosen["eef_delta"], "regret": chosen["regret"],
                    "random_offset_mean_eef_delta": float(np.mean([x["eef_delta"] for x in expected])),
                    "random_offset_mean_regret": float(np.mean([x["regret"] for x in expected])),
                })
    by_selector = {}
    for selector in sorted({r["selector"] for r in records}):
        rows = [r for r in records if r["selector"] == selector]
        task_reg, task_eef = {}, {}
        for r in rows:
            task_reg.setdefault(str(r["task_id"]), []).append(r["regret"] - r["random_offset_mean_regret"])
            task_eef.setdefault(str(r["task_id"]), []).append(r["eef_delta_vs_clean"] - r["random_offset_mean_eef_delta"])
        by_selector[selector] = {
            "n": len(rows),
            "selected_offset_histogram": {str(o): int(sum(r["selected_offset"] == o for r in rows)) for o in offsets},
            "selected_mean_latent_mse": summary([r["selected_mean_latent_mse"] for r in rows]),
            "eef_delta_vs_clean": summary([r["eef_delta_vs_clean"] for r in rows]),
            "regret": summary([r["regret"] for r in rows]),
            "random_offset_mean_eef_delta": summary([r["random_offset_mean_eef_delta"] for r in rows]),
            "random_offset_mean_regret": summary([r["random_offset_mean_regret"] for r in rows]),
            "regret_delta_vs_random": summary([r["regret"] - r["random_offset_mean_regret"] for r in rows]),
            "eef_delta_delta_vs_random": summary([r["eef_delta_vs_clean"] - r["random_offset_mean_eef_delta"] for r in rows]),
            "regret_delta_vs_random_task_cluster_bootstrap": bootstrap(task_reg),
            "eef_delta_delta_vs_random_task_cluster_bootstrap": bootstrap(task_eef),
            "records": rows,
        }
    result = {
        "status": "completed_latent_divergence_goal_attacker",
        "inputs": [str(p) for p in args.inputs], "run_roots": [str(p) for p in args.run_roots],
        "scorer": args.scorer, "offsets": offsets, "records": records,
        "by_selector": by_selector,
        "selection_rule": "select offset using saved WorldAgen worlds_step3 latent MSE relative to Clean before reading replay utility",
        "inference_limits": [
            "Latent divergence is a forecast-only selector feature; targets and replay outcomes are not used.",
            "This remains an explicit independent planner probe because native policy does not consume World Forecast.",
            "The latent MSE is model-space divergence, not physical forecast error or attack success by itself.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
