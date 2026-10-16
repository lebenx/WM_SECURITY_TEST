#!/usr/bin/env python3
"""Validate and summarize the prospective DINO fixed-bank oracle diagnostic.

No model, environment, checkpoint or GPU call. Calibration roots are always
excluded. Incomplete/invalid confirmation runs cannot yield the primary claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/decision_leverage_dino_cpu_confirmatory"
DEFAULT_OUT = ROOT / "reports/decision_leverage_paper/prospective_dino"
EXPECTED_SEEDS = list(range(1001, 1031))
LAMBDAS = [0., .25, .5, .75, 1.]
BOOTSTRAP_SEED = 2026101601
BOOTSTRAP_DRAWS = 10000
METRICS = ("terminal_reward", "cumulative_reward", "negative_terminal_distance", "terminal_success", "ever_success")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_complete_panel(phase, registered_seeds, observed_seeds, run_status):
    require(phase == "confirmation", "Calibration/development roots cannot be analyzed as confirmation")
    require(registered_seeds == EXPECTED_SEEDS, "Registered confirmation panel differs from seeds1001..1030")
    require(len(observed_seeds) == len(set(observed_seeds)), "Duplicated root seeds")
    require(not set(observed_seeds) - set(EXPECTED_SEEDS), "Unexpected or calibration root in confirmation")
    missing = sorted(set(EXPECTED_SEEDS) - set(observed_seeds))
    require(not missing, "Missing confirmation roots: " + str(missing))
    require(run_status == "completed", "Run receipt is not completed")


def bootstrap_mean(values, indices=None):
    import numpy as np
    arr = np.asarray(values, dtype=np.float64)
    require(arr.ndim == 1 and len(arr) > 0, "Bootstrap requires a nonempty root vector")
    require(bool(np.isfinite(arr).all()), "Nonfinite root statistic")
    if indices is None:
        indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(arr), size=(BOOTSTRAP_DRAWS, len(arr)))
    require(indices.shape == (BOOTSTRAP_DRAWS, len(arr)), "Bootstrap must resample entire paired roots")
    means = arr[indices].mean(axis=1)
    return {"n_roots": len(arr), "mean": float(arr.mean()), "median": float(np.median(arr)),
            "std_across_roots": float(arr.std(ddof=1)) if len(arr) > 1 else None,
            "percentile_95_two_sided": np.quantile(means, [.025, .975]).tolist(),
            "percentile_95_one_sided_upper": float(np.quantile(means, .95)),
            "percentile_95_one_sided_lower": float(np.quantile(means, .05)),
            "negative_roots": int((arr < 0).sum()), "zero_roots": int((arr == 0).sum()),
            "positive_roots": int((arr > 0).sum())}


def wilson(successes, n, z=1.959963984540054):
    require(isinstance(n, int) and n > 0 and isinstance(successes, int) and 0 <= successes <= n,
            "Wilson interval requires integer0<=count<=n and n>0")
    p, zz = successes / n, z * z
    center = (p + zz / (2 * n)) / (1 + zz / n)
    half = z * math.sqrt(p * (1 - p) / n + zz / (4 * n * n)) / (1 + zz / n)
    return {"count": successes, "n_roots": n, "rate": p,
            "wilson_95_two_sided": [max(0., center - half), min(1., center + half)]}


def verify_provenance(run, allow_partial=False):
    import numpy as np
    import torch
    from run_decision_leverage_dino_cpu import verify_root

    torch.set_num_threads(4)
    summary = json.loads((run / "summary.json").read_text())
    protocol = json.loads((run / "launched_config.json").read_text())
    require(summary["phase"] == "confirmation", "Pilot excluded: this analyzer accepts confirmation phase only")
    require(summary["seeds"] == EXPECTED_SEEDS, "Run root list changed")
    require(protocol["root_partition"]["confirmatory_physical_seeds"] == EXPECTED_SEEDS, "Protocol root list changed")
    require(protocol["oracle_feature_interpolation"]["lambdas"] == LAMBDAS, "Preset lambda grid changed")
    require(protocol["oracle_feature_interpolation"]["preserve_context"] is True, "Original context must be preserved")
    require(protocol["statistics"]["independent_sample_size"] == 30, "Sample-size protocol changed")
    require(protocol["statistics"]["primary_pair"] == "actual_future_oracle minus original at each root", "Primary comparison changed")
    require(protocol["source_bank"]["count"] == 5 and protocol["source_bank"]["original_axis0"] == 0, "Candidate support changed")
    require(protocol["runtime"]["unrolls_per_root"] == 1, "Unexpected forecast intervention count")
    require(summary["protocol_sha256"] == digest(run / "launched_config.json"), "Launched protocol hash differs")
    require(summary["runner_sha256"] == digest(run / "launched_script.py"), "Launched runner hash differs")
    require(summary["runner_sha256"] == digest(ROOT / "scripts/run_decision_leverage_dino_cpu.py"),
            "Current verifier runner differs from executed source; do not import an unvalidated replacement")
    require(summary["bank_sha256"] == digest(run / "frozen_bank.npz") == protocol["source_bank"]["sha256"], "Candidate bank changed")
    for name, sha in summary["source_sha256"].items():
        require(digest(ROOT / name) == sha, "Execution dependency changed: " + name)
    for name in ("network_attempts", "training_updates", "api_calls"):
        require(summary[name] == 0, "Forbidden activity recorded: " + name)
    require(summary["runtime"]["device"] == "cpu", "Confirmation was not CPU protocol")
    require(not summary.get("cuda_initialized", False), "Execution initialized CUDA")
    roots, missing, incomplete = [], [], []
    for seed in EXPECTED_SEEDS:
        directory = run / f"seed{seed}"
        if not (directory / "result.json").exists():
            missing.append(seed); continue
        existing = json.loads((directory / "result.json").read_text())
        if existing["status"] != "completed":
            incomplete.append({"seed": seed, "status": existing["status"], "error": existing.get("error")}); continue
        report = verify_root(directory, torch, np)
        require(report["seed"] == seed, "Root seed metadata differs")
        require(report["activity"] == dict(expert_controls=100, warmup_controls=1, candidate_controls=150,
                                          encode_calls=8, unroll_calls=1), "Root activity budget differs")
        require(report["protected_checks"]["cuda_initialized"] is False, "Root CUDA check failed")
        require(all(value is True for name, value in report["protected_checks"].items() if name != "cuda_initialized"),
                "Root protected invariant failed")
        commitment = json.loads((directory / "forecast_commitment.json").read_text())
        require(commitment["candidate_outcomes_available"] is False, "Forecast commitment made after truth")
        require(commitment["sha256"] == digest(directory / "forecast_commitment.npz"), "Forecast commitment hash differs")
        require(commitment["bank_sha256"] == summary["bank_sha256"], "Commitment bank differs")
        require(commitment["interpolation_lambdas"] == LAMBDAS and commitment["score_alpha"] == .1,
                "Frozen scorer/lambda commitment differs")
        with np.load(directory / "features.npz", allow_pickle=False) as features, np.load(directory / "forecast_commitment.npz", allow_pickle=False) as old:
            require(all(np.array_equal(features[k], old[k]) for k in old.files), "Forecast/goal/context changed after commitment")
        with np.load(directory / "setup/setup.npz", allow_pickle=False) as setup:
            fingerprint = hashlib.sha256(setup["sim_qpos"].tobytes() + setup["sim_qvel"].tobytes()
                                         + setup["target"].tobytes() + setup["goal_pixels"].tobytes()).hexdigest()
        require(fingerprint == report["root_goal_fingerprint"], "Root fingerprint not bound to setup")
        roots.append(report)
    require(len({r["root_goal_fingerprint"] for r in roots}) == len(roots), "Duplicated physical root+goal")
    require([r["seed"] for r in roots] == summary["completed_seeds"], "Receipt completed root list differs")
    complete = not missing and not incomplete and summary["status"] == "completed"
    if not allow_partial:
        validate_complete_panel(summary["phase"], summary["seeds"], [r["seed"] for r in roots], summary["status"])
    require(roots, "No completed confirmation roots to summarize")
    require(not torch.cuda.is_initialized(), "Verifier initialized CUDA")
    bindings = {"summary.json": digest(run / "summary.json"), "launched_config.json": digest(run / "launched_config.json"),
                "launched_script.py": digest(run / "launched_script.py"), "frozen_bank.npz": digest(run / "frozen_bank.npz")}
    bindings.update({f"seed{r['seed']}/result.json": digest(run / f"seed{r['seed']}/result.json") for r in roots})
    return roots, summary, protocol, {"complete": complete, "missing_roots": missing, "incomplete_roots": incomplete,
                                     "source_bindings": bindings, "raw_artifact_hashes_verified": True}


def analyze_roots(roots, complete):
    import numpy as np

    require(roots, "No root observations")
    n = len(roots)
    idx = np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    curves, curve_rows, long_rows, candidate_rows = [], [], [], []
    primary_error, primary_utility, reversal, action_flips, pairwise_flips = [], [], [], [], []
    for root in roots:
        old, oracle = root["conditions"]["0.0"], root["conditions"]["1.0"]
        delta_e = oracle["terminal_forecast_errors"]["mean_weighted"] - old["terminal_forecast_errors"]["mean_weighted"]
        delta_u = oracle["metrics"]["h30"]["terminal_reward"]["selected_utility"] - old["metrics"]["h30"]["terminal_reward"]["selected_utility"]
        primary_error.append(delta_e); primary_utility.append(delta_u); reversal.append(delta_e < 0 and delta_u < 0)
        action_flips.append(oracle["selected_candidate"] != old["selected_candidate"])
        pairwise_flips.append(sum((old["ranking"].index(i) < old["ranking"].index(j))
                                  != (oracle["ranking"].index(i) < oracle["ranking"].index(j))
                                  for i in range(5) for j in range(i + 1, 5)) / 10.)
        for candidate, utility in enumerate(root["candidate_utilities"]):
            for horizon in ("h15", "h30"):
                candidate_rows.append({"physical_seed": root["seed"], "candidate": candidate, "horizon": horizon, **utility[horizon]})
        for lam in LAMBDAS:
            c = root["conditions"][str(lam)]
            for horizon in ("h15", "h30"):
                for metric in METRICS:
                    m, original = c["metrics"][horizon][metric], old["metrics"][horizon][metric]
                    long_rows.append({"physical_seed": root["seed"], "lambda": lam, "horizon": horizon, "metric": metric,
                                      "selected_candidate": c["selected_candidate"], "selected_utility": m["selected_utility"],
                                      "delta_utility_vs_original": m["selected_utility"] - original["selected_utility"],
                                      "Lev": m["Lev"], "Mis": m["Mis"], "Pred": m["Pred"], "regret": m["regret"],
                                      "uniform_expected_utility": m["uniform_expected_utility"],
                                      "terminal_forecast_mse": c["terminal_forecast_errors"]["mean_weighted"],
                                      "delta_terminal_forecast_mse": c["terminal_forecast_errors"]["mean_weighted"] - old["terminal_forecast_errors"]["mean_weighted"]})
    for lam in LAMBDAS:
        c = [r["conditions"][str(lam)] for r in roots]
        entry = {"lambda": lam, "fidelity": bootstrap_mean([v["terminal_forecast_errors"]["mean_weighted"] for v in c], idx),
                 "metrics": {}}
        for horizon in ("h15", "h30"):
            entry["metrics"][horizon] = {}
            for metric in METRICS:
                series = {key: [v["metrics"][horizon][metric][key] for v in c]
                          for key in ("Lev", "Mis", "Pred", "regret", "selected_utility", "uniform_expected_utility")}
                series["delta_utility_vs_original"] = [v["metrics"][horizon][metric]["selected_utility"]
                                                       - r["conditions"]["0.0"]["metrics"][horizon][metric]["selected_utility"]
                                                       for r, v in zip(roots, c)]
                stats = {key: bootstrap_mean(values, idx) for key, values in series.items()}
                entry["metrics"][horizon][metric] = stats
                selected, delta = stats["selected_utility"], stats["delta_utility_vs_original"]
                curve_rows.append({"lambda": lam, "horizon": horizon, "metric": metric, "n_roots": n,
                                   "mean_utility": selected["mean"], "ci95_lower": selected["percentile_95_two_sided"][0],
                                   "ci95_upper": selected["percentile_95_two_sided"][1], "paired_mean_delta": delta["mean"],
                                   "paired_delta_ci95_lower": delta["percentile_95_two_sided"][0],
                                   "paired_delta_ci95_upper": delta["percentile_95_two_sided"][1],
                                   "mean_terminal_forecast_mse": entry["fidelity"]["mean"]})
        curves.append(entry)
    baselines = {}
    for horizon in ("h15", "h30"):
        baselines[horizon] = {}
        for metric in METRICS:
            fixed = [bootstrap_mean([r["candidate_utilities"][i][horizon][metric] for r in roots], idx) for i in range(5)]
            baselines[horizon][metric] = {
                "fixed_candidates": fixed, "physical_zero_candidate_index": 3,
                "uniform_expected": bootstrap_mean([sum(u[horizon][metric] for u in r["candidate_utilities"]) / 5 for r in roots], idx),
                "per_root_best_true_utility_oracle": bootstrap_mean([max(u[horizon][metric] for u in r["candidate_utilities"]) for r in roots], idx),
                "best_fixed_candidate_on_test": {"index": max(range(5), key=lambda i: (fixed[i]["mean"], -i)),
                                                "status": "posthoc_oracle_comparator_not_a_heldout_trained_baseline"}}
    delta_error, delta_utility = bootstrap_mean(primary_error, idx), bootstrap_mean(primary_utility, idx)
    primary = {"comparison": "actual_future_oracle minus original", "utility": "h30 terminal native reward",
               "delta_forecast_mse": delta_error, "delta_selected_utility": delta_utility,
               "oracle_reversal_event": wilson(int(sum(reversal)), n),
               "forecast_error_reduction_is_manipulation_check": True,
               "claim_eligible": bool(complete and n == 30),
               "p1_offline_fixed_bank_supported": bool(delta_error["mean"] < 0 and delta_utility["percentile_95_one_sided_upper"] < 0) if complete and n == 30 else None,
               "mis_positive_does_not_decide_p1": True}
    primary["selected_action_flip_rate"] = wilson(int(sum(action_flips)), n)
    primary["pairwise_order_flip_fraction"] = bootstrap_mean(pairwise_flips, idx)
    primary["ranking_definition"] = "Selected-index changes and per-root fraction of10 pairwise stable-cost orders changed; exact ties use the registered candidate-index rule."
    primary["interpretation"] = (
        "incomplete_no_confirmatory_judgment" if not primary["claim_eligible"] else
        "offline_oracle_substitution_reduces_mean_utility_in_this_fixed_bank" if primary["p1_offline_fixed_bank_supported"] else
        "opposite_direction_mean_utility_improvement" if delta_utility["percentile_95_one_sided_lower"] > 0 else
        "no_confirmed_mean_utility_degradation_not_proof_of_absence")
    return {"primary": primary, "lambda_curves": curves, "baselines": baselines}, long_rows, curve_rows, candidate_rows


def csv_string(rows):
    require(rows, "No CSV rows")
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    return text.getvalue()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=DEFAULT_RUN)
    p.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT)
    p.add_argument("--allow-partial", action="store_true", help="Descriptive completed roots only; disables primary confirmation")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    roots, receipt, protocol, validation = verify_provenance(args.run.resolve(), args.allow_partial)
    numbers, long_rows, curve_rows, candidate_rows = analyze_roots(roots, validation["complete"])
    result = {"schema_version": 1, "status": "completed_confirmatory_analysis" if validation["complete"] else "incomplete_descriptive_only",
              "analyzer_sha256": digest(__file__),
              "source_run": str(args.run.resolve().relative_to(ROOT)), "registered_roots": EXPECTED_SEEDS,
              "included_roots": [r["seed"] for r in roots], "calibration_roots_included": [], "validation": validation,
              "statistics": {"unit": "independent physical root; all candidates/lambda arms paired",
                             "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
                             "bootstrap": "paired whole-root percentile; NumPy default_rng, common indices across statistics",
                             "primary": "one-sided95% upper bound<0 for oracle-original terminal utility, plus strictly lower forecast error",
                             "secondary": "all lambda curves and secondary endpoints descriptive, no choose-best comparison"},
              **numbers,
              "actual_execution_activity": {"receipt": {k: receipt.get(k) for k in ("model_load_calls", "network_attempts", "training_updates", "api_calls", "cuda_initialized", "invocation_elapsed_seconds", "peak_rss_kib")},
                                            "root_totals": {k: sum(r["activity"][k] for r in roots) for k in roots[0]["activity"]},
                                            "scope": receipt.get("control_count_scope")},
              "analysis_activity": {"model_calls": 0, "environment_steps": 0, "gpu_calls": 0, "api_calls": 0, "checkpoint_loads": 0},
              "limits": ["Added fixed-bank ranker; not native adaptive CEM or a deployment defense.",
                         "Actual future features are offline diagnostic oracle information. Intermediate linear mixtures may be off the feature manifold.",
                         "All selectors score H30; h15 numbers are utility of their15-control prefixes, not an H15 oracle selector.",
                         "Candidates are transferred development sequences. Roots are unused project seeds, not proven outside model training.",
                         "The best fixed candidate selected on these test outcomes is an oracle comparator; no training-root calibrated scorer is evaluated.",
                         "Incomplete outcomes never satisfy the preregistered primary test; explicit allow-partial output is descriptive only."]}
    prefix = args.out_prefix
    outputs = {prefix.with_suffix(".json"): json.dumps(result, indent=2, allow_nan=False) + "\n",
               prefix.with_name(prefix.name + "_root_metrics.csv"): csv_string(long_rows),
               prefix.with_name(prefix.name + "_curves.csv"): csv_string(curve_rows),
               prefix.with_name(prefix.name + "_candidate_utilities.csv"): csv_string(candidate_rows)}
    if args.verify_only:
        for path, content in outputs.items():
            require(path.read_text() == content.replace("\r\n", "\n"), "Analysis artifact differs: " + str(path))
    else:
        require(not any(path.exists() for path in outputs), "Refusing to overwrite existing analysis outputs")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        for path, content in outputs.items():
            with path.open("x", newline="") as handle:
                handle.write(content)
    print(json.dumps({"status": result["status"], "n_roots": len(roots), "out_prefix": str(prefix), "primary": result["primary"]}))


if __name__ == "__main__":
    main()
