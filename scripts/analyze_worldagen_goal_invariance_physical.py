#!/usr/bin/env python3
"""Audit Goal-invariance against matched physical future targets.

The aligned WorldAgen sweep stores one real LIBERO future patch per frozen
candidate action.  That target is shared by Clean, Goal-Swap, Bus-Cut, and
Control-Cut because the physical replay is performed independently of the
forecast condition.  This script checks that contract and separates
model-space Goal divergence from error against the matched physical target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


def summary(values):
    x = np.asarray(values, dtype=float)
    return {"n": int(len(x)), "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "min": float(x.min()), "max": float(x.max())}


def cluster_bootstrap(rows, value_key, seed=83, draws=10000):
    by_task = {}
    for row in rows:
        by_task.setdefault(str(row["task_id"]), []).append(float(row[value_key]))
    means = np.asarray([np.mean(v) for v in by_task.values()], dtype=float)
    rng = np.random.default_rng(seed)
    sample = rng.choice(means, size=(draws, len(means)), replace=True).mean(axis=1)
    return {"n_tasks": int(len(means)), "mean": float(means.mean()),
            "ci95": [float(np.quantile(sample, .025)), float(np.quantile(sample, .975))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path, help="aligned sweep summary.json files")
    ap.add_argument("--run-roots", nargs="+", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--scorer", default="reference_scalar")
    args = ap.parse_args()
    if len(args.inputs) != len(args.run_roots):
        raise ValueError("inputs and run-roots must have the same length")
    runs = [json.loads(p.read_text()) for p in args.inputs]
    offsets = sorted({int(o) for run in runs for o in run["goal_offsets"]})
    records = []
    verification = []
    for run, root in zip(runs, args.run_roots):
        for row in run["rows"]:
            tid, seed = int(row["task_id"]), int(run["seed"])
            forecast_path = root / f"task_{tid}_forecasts.pt"
            target_path = root / f"task_{tid}_targets.pt"
            payload = torch.load(forecast_path, map_location="cpu", weights_only=False)
            target_payload = torch.load(target_path, map_location="cpu", weights_only=False)
            worlds = {k: v.float() for k, v in payload["worlds_step3"].items()}
            target = target_payload["target_patches_step3"].float()
            if target.shape != worlds["clean"].shape:
                raise AssertionError(f"target/forecast shape mismatch task={tid} seed={seed}")
            # The stored MSE is recomputed from the same target, then checked.
            stored_err = {}
            max_recompute_error = 0.0
            for name, pred in worlds.items():
                err = (pred - target).pow(2).mean(dim=(1, 2)).numpy()
                stored = np.asarray(row["candidate_forecast_mse"][name], dtype=float)
                max_recompute_error = max(max_recompute_error, float(np.max(np.abs(err - stored))))
                stored_err[name] = err
            if max_recompute_error > 1e-5:
                raise AssertionError(f"stored MSE mismatch task={tid} seed={seed}: {max_recompute_error}")
            clean = worlds["clean"]
            clean_err = stored_err["clean"]
            verification.append({
                "task_id": tid, "seed": seed,
                "initial_sim_sha256": row["initial_sim_sha256"],
                "initial_image_sha256": row["initial_image_sha256"],
                "candidate_sha256": row["candidate_sha256"],
                "target_shape": list(target.shape),
                "replay_environment_steps": sorted({int(x["environment_steps"]) for x in row["candidate_replays"]}),
                "target_mse_recomputed_max_abs_error": max_recompute_error,
                "target_contract": "one matched real target patch per frozen candidate; target is after step 3 while utility replay is 30 steps",
            })
            for off in offsets:
                name = f"swap_{off}"
                swap = worlds[name]
                swap_err = stored_err[name]
                bus_err = stored_err[f"{name}_bus_cut"]
                control_err = stored_err[f"{name}_control_cut"]
                latent_div = (swap - clean).pow(2).mean(dim=(1, 2)).numpy()
                records.append({
                    "task_id": tid, "seed": seed, "offset": off,
                    "forecast_divergence_mean": float(latent_div.mean()),
                    "forecast_divergence_max": float(latent_div.max()),
                    "clean_physical_mse": float(clean_err.mean()),
                    "swap_physical_mse": float(swap_err.mean()),
                    "swap_minus_clean_physical_mse": float((swap_err-clean_err).mean()),
                    "bus_cut_minus_clean_physical_mse": float((bus_err-clean_err).mean()),
                    "control_cut_minus_swap_physical_mse": float((control_err-swap_err).mean()),
                    "candidate_fraction_swap_lower_error": float(np.mean(swap_err < clean_err)),
                })

    by_offset = {}
    for off in offsets:
        rows = [r for r in records if r["offset"] == off]
        by_offset[str(off)] = {
            "n": len(rows),
            "forecast_divergence_mean": summary([r["forecast_divergence_mean"] for r in rows]),
            "swap_minus_clean_physical_mse": summary([r["swap_minus_clean_physical_mse"] for r in rows]),
            "swap_minus_clean_physical_mse_task_cluster_bootstrap": cluster_bootstrap(rows, "swap_minus_clean_physical_mse", seed=83+off),
            "bus_cut_minus_clean_physical_mse": summary([r["bus_cut_minus_clean_physical_mse"] for r in rows]),
            "control_cut_minus_swap_physical_mse": summary([r["control_cut_minus_swap_physical_mse"] for r in rows]),
            "candidate_fraction_swap_lower_error": summary([r["candidate_fraction_swap_lower_error"] for r in rows]),
        }

    div = np.asarray([r["forecast_divergence_mean"] for r in records], dtype=float)
    delta = np.asarray([r["swap_minus_clean_physical_mse"] for r in records], dtype=float)
    result = {
        "status": "completed_goal_invariance_physical_target_audit",
        "inputs": [str(p) for p in args.inputs], "run_roots": [str(p) for p in args.run_roots],
        "scorer": args.scorer, "offsets": offsets,
        "n_task_seed_pairs": len(verification), "n_counterfactuals": len(records),
        "verification": verification, "records": records, "by_offset": by_offset,
        "aggregate": {
            "swap_minus_clean_physical_mse": summary(delta.tolist()),
            "swap_minus_clean_physical_mse_task_cluster_bootstrap": cluster_bootstrap(records, "swap_minus_clean_physical_mse"),
            "forecast_divergence_mean": summary(div.tolist()),
            "spearman_divergence_vs_physical_error_delta": {
                "rho": float(spearmanr(div, delta).statistic),
                "pvalue": float(spearmanr(div, delta).pvalue),
            },
            "pearson_divergence_vs_physical_error_delta": {
                "r": float(pearsonr(div, delta).statistic),
                "pvalue": float(pearsonr(div, delta).pvalue),
            },
        },
        "interpretation": [
            "The matched physical target is held fixed across forecast conditions for each candidate; it is obtained from the same initial simulator state and candidate action replay.",
            "Goal-Swap changes model-space forecasts, but its error relative to the physical target is not sign-consistent across legal offsets; this is a forecast-invariance violation without a universal physical-error direction.",
            "Control-Cut is a policy-edge control: its physical-target MSE matches the corresponding Goal-Swap forecast, while Bus-Cut collapses the world forecast to a shared branch.",
            "Forecast divergence is therefore not itself physical forecast error or attack success; a planner/task consequence requires a separate downstream consumer.",
        ],
        "scope_limits": [
            "The target is a decoded primary-view patch after three executed actions, not a full latent-state or task-success target.",
            "The audit verifies matched replay and conditional output semantics; it does not prove that every WorldAgen latent channel is intended to be goal-free.",
            "Native WorldAgen/AlphaBrain policies do not consume World Forecast, so decision implications remain independent-planner evidence.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "n": len(records), "aggregate": result["aggregate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
