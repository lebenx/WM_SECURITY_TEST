#!/usr/bin/env python3
"""Held-out cross-task value probe over WorldAgen step-3 forecasts.

The probe is deliberately small and frozen: it is fit only on Clean forecasts
and real EEF outcomes from training tasks.  Test-task selections use forecast
features before any test replay outcome is read.  Goal-Swap and edge-cut
conditions are therefore evaluated as interventions on the same learned value
head, rather than as a new scorer tuned per condition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_tensor(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def forecast_features(world):
    """Low-dimensional action-independent summary of [N,196,768] patches."""
    x = np.asarray(world, dtype=np.float64).reshape(-1, 196, 16, 16, 3)
    patch_rgb = x.mean(axis=(2, 3))  # [N, 196, 3]
    global_stats = np.concatenate(
        [patch_rgb.mean(axis=1), patch_rgb.std(axis=1),
         patch_rgb.mean(axis=(1, 2), keepdims=False)[:, None],
         patch_rgb.std(axis=(1, 2), keepdims=False)[:, None]], axis=1,
    )
    return np.concatenate([patch_rgb.reshape(len(x), -1), global_stats], axis=1)


def ridge_fit(x, y, alpha=10.0):
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-8] = 1.0
    z = (x - mu) / sd
    ymu = float(y.mean())
    a = z.T @ z + alpha * np.eye(z.shape[1])
    w = np.linalg.solve(a, z.T @ (y - ymu))
    return mu, sd, ymu, w


def predict(model, x):
    mu, sd, ymu, w = model
    return ymu + ((x - mu) / sd) @ w


def summarize(v):
    x = np.asarray(v, dtype=float)
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0}


def spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+", type=Path)
    ap.add_argument("--test-tasks", required=True, help="comma-separated held-out task ids")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--alpha", type=float, default=10.0)
    args = ap.parse_args()
    test_tasks = {int(x) for x in args.test_tasks.split(",") if x}
    runs = [json.loads(p.read_text()) for p in args.summaries]
    all_tasks = sorted({int(row["task_id"]) for run in runs for row in run["rows"]})
    train_tasks = sorted(set(all_tasks) - test_tasks)
    if not train_tasks or not test_tasks:
        raise ValueError("both train and test task sets must be nonempty")

    train_x, train_y = [], []
    run_rows = []
    for summary_path, run in zip(args.summaries, runs):
        root = summary_path.parent
        for row in run["rows"]:
            tid = int(row["task_id"])
            tensor = load_tensor(root / f"task_{tid}_forecasts.pt")
            clean = forecast_features(tensor["worlds_step3"]["clean"].numpy())
            utility = np.asarray([x["eef_approach_progress"] for x in row["candidate_replays"]], dtype=float)
            if tid in train_tasks:
                train_x.append(clean)
                train_y.append(utility)
            else:
                run_rows.append((summary_path, run, row, tensor))
    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    model = ridge_fit(train_x, train_y, alpha=args.alpha)

    conditions = sorted(runs[0]["rows"][0]["scores"])
    per_condition = {}
    control_diff = {}
    random_regret = []
    for condition in conditions:
        selected, regret, changed, correlations = [], [], [], []
        delta_eef, delta_regret = [], []
        for _, _, row, tensor in run_rows:
            utility = np.asarray([x["eef_approach_progress"] for x in row["candidate_replays"]], dtype=float)
            clean_pred = predict(model, forecast_features(tensor["worlds_step3"]["clean"].numpy()))
            pred = predict(model, forecast_features(tensor["worlds_step3"][condition].numpy()))
            idx = int(np.argmax(pred))
            clean_idx = int(np.argmax(clean_pred))
            best = int(np.argmax(utility))
            selected.append(float(utility[idx]))
            regret.append(float(utility[best] - utility[idx]))
            changed.append(int(idx != clean_idx))
            correlations.append(spearman(pred, utility))
            if condition != "clean":
                clean_utility = float(utility[clean_idx])
                clean_regret = float(utility[best] - clean_utility)
                delta_eef.append(float(utility[idx] - clean_utility))
                delta_regret.append(float((utility[best] - utility[idx]) - clean_regret))
            if condition == "clean":
                random_regret.append(float(utility[best] - utility.mean()))
        rec = {
            "selected_eef_progress": summarize(selected),
            "regret_vs_candidate_best": summarize(regret),
            "selected_action_changed_vs_clean": summarize(changed),
            "score_vs_true_utility_spearman": summarize(correlations),
        }
        if condition != "clean":
            rec["eef_delta_vs_clean_selected_action"] = summarize(delta_eef)
            rec["regret_delta_vs_clean_selected_action"] = summarize(delta_regret)
        per_condition[condition] = rec

    for base in ("swap_1", "swap_5"):
        ctrl = f"{base}_control_cut"
        diffs = []
        for _, _, _, tensor in run_rows:
            a = predict(model, forecast_features(tensor["worlds_step3"][base].numpy()))
            b = predict(model, forecast_features(tensor["worlds_step3"][ctrl].numpy()))
            diffs.append(float(np.max(np.abs(a - b))))
        control_diff[base] = float(max(diffs))

    result = {
        "status": "completed_cross_task_value_probe",
        "summaries": [str(p) for p in args.summaries],
        "train_tasks": train_tasks,
        "test_tasks": sorted(test_tasks),
        "train_samples": int(len(train_y)),
        "test_task_seed_pairs": int(len(run_rows)),
        "feature": "step3 primary forecast patch RGB means per 14x14 patch plus global moments",
        "fit": "ridge on Clean forecasts only; fixed alpha",
        "alpha": args.alpha,
        "native_policy_uses_forecast": False,
        "per_condition": per_condition,
        "candidate_random_baseline": summarize(random_regret),
        "control_cut_max_abs_predicted_value_diff": control_diff,
        "scope": "Held-out independent forecast-driven value probe; not native policy evidence.",
        "inference_limits": [
            "three seeds and six tasks; one fixed task split; descriptive only",
            "three-step EEF progress is intermediate simulator utility, not task completion",
            "test replay outcomes are used only after forecast-based selection",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
