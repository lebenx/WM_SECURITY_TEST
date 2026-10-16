#!/usr/bin/env python3
"""R004: fixed prediction-only residual correction; exploratory diagnostic.

Not an attack, not an independently confirmed security mechanism. The30
evaluation roots were previously examined. Native provider permissions are
unproved; a separate forecast/calibration service is only a proposed interface.
"""
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
SOURCE = "runs/dino_action_saturation_gate_20261015/forecasts_and_encodings.npz"
EVAL = "runs/decision_leverage_dino_cpu_confirmatory"
RUN = ROOT / "runs/forecast_residual_calibration_v1"
REPORT = ROOT / "reports/forecast_residual_calibration.json"
FEATURES = ("visual", "proprio")
ARMS = ("original", "fixed_residual")
METRICS = ("terminal_reward", "cumulative_reward", "terminal_success", "ever_success", "negative_terminal_distance")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write("\n")


def main():
    started = time.monotonic()
    require(not RUN.exists() and not REPORT.exists() and not REPORT.with_suffix(".csv").exists(), "Refusing existing output")
    RUN.mkdir(parents=True)
    (RUN / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    import numpy as np
    import torch
    torch.set_num_threads(4)
    biases = {}
    # No evaluation paths or utilities are opened before correction commitment.
    with np.load(ROOT / SOURCE, allow_pickle=False) as source:
        for key in FEATURES:
            truth = torch.from_numpy(source["ground_truth_" + key].copy())
            prediction = torch.from_numpy(source["original_" + key].copy())
            require(truth.dtype == prediction.dtype == torch.float32 and tuple(truth.shape[:2]) == (6, 5)
                    and tuple(prediction.shape[:2]) == (7, 5), "Calibration shape/dtype differs")
            require(torch.isfinite(truth).all() and torch.isfinite(prediction).all(), "Nonfinite calibration data")
            biases[key] = (truth - prediction[1:]).mean(dim=1, keepdim=True).contiguous()
    np.savez_compressed(RUN / "calibrator.npz", **{"bias_" + key: value.numpy() for key, value in biases.items()})
    fit = {"status": "fitted_and_saved_before_evaluation_read", "registration": "IDEAS.md:R004",
           "fit_source": SOURCE, "fit_root": 1, "fit_candidates": 5,
           "fit_fields": ["ground_truth_visual", "original_visual", "ground_truth_proprio", "original_proprio"],
           "fit_formula": "FP32 mean overcandidate axis1 of (truth - original_forecast[1:]), separately at each of6future horizons; keep candidateaxis length1.",
           "coefficient": 1.0, "candidate_independent": True, "fit_utility_reads": 0, "evaluation_files_read_so_far": 0,
           "preserved_context": True, "goal_used_in_fit": False, "bias_shapes": {k: list(v.shape) for k, v in biases.items()},
           "primary_forecast_metric_frozen": "FP64 weightedMSE visual+0.1proprio, averaged over all6future horizons and all5candidates. Initialcontext excluded. TerminalMSE also reported separately.",
           "primary_utility": "Recorded H30 terminal native reward of argmin native goal-distance score; exact score ties choose lowest candidateindex.",
           "evaluation_seeds": list(range(1001, 1031)), "committed_at_utc": datetime.now(timezone.utc).isoformat()}
    write(RUN / "fit_commitment.json", fit)
    # Reload the saved correction; all evaluation uses this committed object.
    with np.load(RUN / "calibrator.npz", allow_pickle=False) as saved:
        biases = {key: torch.from_numpy(saved["bias_" + key].copy()) for key in FEATURES}
    receipt = json.loads((ROOT / EVAL / "summary.json").read_text())
    require(receipt["status"] == "completed" and receipt["completed_seeds"] == list(range(1001, 1031)), "Evaluation panel incomplete")
    records, csv_rows = [], []

    def score(prediction, goal):
        parts = {key: (goal[key] - prediction[key]).square().mean(dim=tuple(range(2, prediction[key].ndim)))[-1] for key in FEATURES}
        cost = parts["visual"] + .1 * parts["proprio"]
        require(cost.dtype == torch.float32 and tuple(cost.shape) == (5,) and torch.isfinite(cost).all(), "Native score schema differs")
        return cost.tolist()

    def errors(prediction, truth):
        parts = {key: (prediction[key][1:].double() - truth[key].double()).square()
                 .mean(dim=tuple(range(2, truth[key].ndim))) for key in FEATURES}
        weighted = parts["visual"] + .1 * parts["proprio"]
        return {"weighted_per_horizon_candidate": weighted.tolist(),
                "mean_all_future_candidates": weighted.mean().item(), "mean_terminal_candidates": weighted[-1].mean().item(),
                "visual_mean_all_future_candidates": parts["visual"].mean().item(),
                "proprio_mean_all_future_candidates": parts["proprio"].mean().item()}

    for seed in range(1001, 1031):
        directory = ROOT / EVAL / f"seed{seed}"
        cached = json.loads((directory / "result.json").read_text())
        require(cached["status"] == "completed" and cached["seed"] == seed, "Root incomplete")
        with np.load(directory / "features.npz", allow_pickle=False) as source:
            prediction = {key: torch.from_numpy(source["forecast_" + key].copy()) for key in FEATURES}
            truth = {key: torch.from_numpy(source["truth_" + key].copy()) for key in FEATURES}
            goal = {key: torch.from_numpy(source["goal_" + key].copy()) for key in FEATURES}
        corrected = {key: value.clone() for key, value in prediction.items()}
        for key in FEATURES:
            require(prediction[key].dtype == truth[key].dtype == goal[key].dtype == biases[key].dtype == torch.float32, "FP32 contract changed")
            require(tuple(prediction[key].shape[:2]) == (7, 5) and tuple(truth[key].shape[:2]) == (6, 5), "Evaluation horizon/batch changed")
            corrected[key][1:] += biases[key]
            require(torch.equal(corrected[key][:1], prediction[key][:1]) and torch.isfinite(corrected[key]).all(), "Context altered or nonfinite correction")
        utility = [u["h30"] for u in cached["candidate_utilities"]]
        conditions = {}
        for arm, pred in (("original", prediction), ("fixed_residual", corrected)):
            costs = score(pred, goal)
            order = sorted(range(5), key=lambda i: (costs[i], i))
            if arm == "original":
                require(costs == cached["conditions"]["0.0"]["native_costs"]
                        and order[0] == cached["conditions"]["0.0"]["selected_candidate"], "Original native scores/selection differ")
            conditions[arm] = {"native_scores": costs, "ranking": order, "selected_candidate": order[0],
                               "forecast_error": errors(pred, truth), "selected_real_utility": utility[order[0]]}
        base, calibrated = conditions["original"], conditions["fixed_residual"]
        deltas = {metric: calibrated["selected_real_utility"][metric] - base["selected_real_utility"][metric] for metric in METRICS}
        record = {"seed": seed, "conditions": conditions, "candidate_utilities": utility,
                  "delta_all_future_mse": calibrated["forecast_error"]["mean_all_future_candidates"] - base["forecast_error"]["mean_all_future_candidates"],
                  "delta_terminal_mse": calibrated["forecast_error"]["mean_terminal_candidates"] - base["forecast_error"]["mean_terminal_candidates"],
                  "selected_action_flip": calibrated["selected_candidate"] != base["selected_candidate"],
                  "ranking_flip": calibrated["ranking"] != base["ranking"], "selected_utility_deltas": deltas}
        records.append(record)
        for arm, condition in conditions.items():
            csv_rows.append({"seed": seed, "condition": arm, "selected_candidate": condition["selected_candidate"],
                             "weighted_mse_all_future_candidates": condition["forecast_error"]["mean_all_future_candidates"],
                             "weighted_mse_terminal_candidates": condition["forecast_error"]["mean_terminal_candidates"],
                             **{k: condition["selected_real_utility"][k] for k in METRICS},
                             "delta_primary_mse_vs_original": 0. if arm == "original" else record["delta_all_future_mse"],
                             "delta_terminal_reward_vs_original": 0. if arm == "original" else deltas["terminal_reward"],
                             "selected_action_flip": False if arm == "original" else record["selected_action_flip"]})
    means = {}
    for arm in ARMS:
        means[arm] = {"mean_all_future_mse": sum(r["conditions"][arm]["forecast_error"]["mean_all_future_candidates"] for r in records) / 30,
                      "mean_terminal_mse": sum(r["conditions"][arm]["forecast_error"]["mean_terminal_candidates"] for r in records) / 30,
                      **{metric: sum(r["conditions"][arm]["selected_real_utility"][metric] for r in records) / 30 for metric in METRICS}}
    delta_e = means["fixed_residual"]["mean_all_future_mse"] - means["original"]["mean_all_future_mse"]
    delta_u = means["fixed_residual"]["terminal_reward"] - means["original"]["terminal_reward"]
    baselines = {metric: {"fixed_candidates": [sum(r["candidate_utilities"][i][metric] for r in records) / 30 for i in range(5)],
                          "uniform_expected": sum(sum(u[metric] for u in r["candidate_utilities"]) / 5 for r in records) / 30,
                          "per_root_best_true_utility_oracle": sum(max(u[metric] for u in r["candidate_utilities"]) for r in records) / 30} for metric in METRICS}
    require(not torch.cuda.is_initialized(), "CUDA initialized")
    result = {"schema_version": 1, "status": "completed_exploratory_fixed_residual_calibration", "registration": "IDEAS.md:R004",
              "warning": "EXPLORATORY UPDATE DIAGNOSTIC, NOT AN ATTACK. These30 roots were already used; this is not an independent confirmation. External provider authority is only a proposed added-interface assumption.",
              "fit_commitment": "runs/forecast_residual_calibration_v1/fit_commitment.json", "calibrator": "runs/forecast_residual_calibration_v1/calibrator.npz",
              "fit_utility_or_eval_reads": 0, "fixed_coefficient": 1., "candidate_independent": True, "evaluation_roots": 30,
              "means": means, "joint_gate": {"mean_prediction_error_decreases": delta_e < 0, "mean_terminal_utility_decreases": delta_u < 0,
                                              "passed": delta_e < 0 and delta_u < 0, "delta_mean_all_future_mse": delta_e,
                                              "delta_mean_terminal_reward": delta_u},
              "root_counts": {"prediction_error_lower": sum(r["delta_all_future_mse"] < 0 for r in records),
                              "terminal_reward_lower": sum(r["selected_utility_deltas"]["terminal_reward"] < 0 for r in records),
                              "terminal_reward_tied": sum(r["selected_utility_deltas"]["terminal_reward"] == 0 for r in records),
                              "terminal_reward_higher": sum(r["selected_utility_deltas"]["terminal_reward"] > 0 for r in records),
                              "selected_action_flips": sum(r["selected_action_flip"] for r in records)},
              "fixed_baselines": baselines, "roots": records, "statistics": "Descriptive only: no bootstrap, pvalues, new population confirmation or tune-after-evaluation.",
              "scope": "Native frozen model/scorer used in an added5-action ranker; correction is fitted from one developmentroot's original prediction residuals only. Evaluated future encodings measure error but never enter corrected forecasts.",
              "activity": {"model_loads": 0, "model_calls": 0, "environment_steps": 0, "api_calls": 0, "gpu_calls": 0,
                           "hash_reaudits": 0, "cached_residual_fit_operations": 2}, "elapsed_seconds": time.monotonic() - started,
              "stop_rule": "Retain this fixed correction's result; no alternate calibration root, scale, regularization, candidate-dependent correction or refit."}
    write(REPORT, result)
    with REPORT.with_suffix(".csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    write(RUN / "summary.json", {k: result[k] for k in ("status", "warning", "means", "joint_gate", "root_counts", "activity", "elapsed_seconds")})
    print(json.dumps({k: result[k] for k in ("status", "joint_gate", "means", "root_counts", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
