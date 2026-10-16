#!/usr/bin/env python3
"""Audit cached fixed-candidate ranking against actual consequences (stdlib only).

This is a diagnostic over two existing post-hoc banks, not a new experiment,
attack success estimate, or causal mediation estimator. Truth means encoded
measured future observations passed to the unchanged native scoring function.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "reports/counterfactual_action_ranking_audit_20261016.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_bank(name, scores, utilities, lower_is_better, baseline, intervention,
               forecast_errors, metric, scope):
    """Separate truth-score mismatch from a signed forecast-selection difference.

    All utility values must already be independently measured. The caller owns
    provenance checks. This function cannot infer experimental validity from
    numbers, nor assign prediction error to the physical-content mechanism.
    """
    n = len(utilities)
    require(n >= 2 and set(scores) == {baseline, intervention, "true_future"},
            "Need two conditions plus measured-future native scoring")
    require(all(len(v) == n for v in scores.values()), "Candidate sets differ")
    require(all(math.isfinite(x) for v in [utilities, *scores.values()] for x in v),
            "Missing/nonfinite measured utility or score")
    require(set(forecast_errors) == {baseline, intervention}
            and all(len(v) == n for v in forecast_errors.values())
            and all(math.isfinite(x) and x >= 0 for v in forecast_errors.values() for x in v),
            "Forecast errors must be finite, nonnegative and candidate aligned")
    ranked = {k: sorted(range(n), key=lambda i: (v[i] if lower_is_better else -v[i], i))
              for k, v in scores.items()}
    # These particular source banks have no ties. Refuse to disguise tie-breaking
    # as a strict rank flip when reusing this diagnostic with other data.
    require(all(len(set(v)) == n for v in scores.values()), "Tied native scores: separate tie analysis needed")
    selected = {k: r[0] for k, r in ranked.items()}
    truth_i = selected["true_future"]
    best_utility = max(utilities)
    proxy_regret = best_utility - utilities[truth_i]
    rows = []
    for condition, values in scores.items():
        chosen = selected[condition]
        total_regret = best_utility - utilities[chosen]
        forecast_term = utilities[truth_i] - utilities[chosen]
        require(math.isclose(total_regret, proxy_regret + forecast_term, abs_tol=1e-12),
                "Utility regret identity failed")
        rows.append({"condition": condition, "scores": values, "rank": ranked[condition],
                     "selected_index": chosen, "selected_actual_utility": utilities[chosen],
                     "within_bank_actual_regret": total_regret,
                     "truth_score_mismatch_regret": proxy_regret,
                     "signed_forecast_selection_difference": forecast_term,
                     "forecast_errors": forecast_errors.get(condition)})
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    flips = sum((scores[baseline][i] < scores[baseline][j]) !=
                (scores[intervention][i] < scores[intervention][j]) for i, j in pairs)
    error_deltas = [b - a for a, b in zip(forecast_errors[baseline], forecast_errors[intervention])]
    return {"system": name, "scope": scope, "candidate_count": n,
            "measured_utilities": utilities, "utility_metric": metric,
            "score_lower_is_better": lower_is_better, "conditions": rows,
            "within_bank_strict_pair_flips": flips, "within_bank_pairs": len(pairs),
            "within_bank_pair_flip_fraction": flips / len(pairs),
            "population_action_ranking_flip_rate": None,
            "selected_action_changed": selected[baseline] != selected[intervention],
            "intervention_minus_baseline_actual_utility": utilities[selected[intervention]] - utilities[selected[baseline]],
            "forecast_error_intervention_minus_baseline": error_deltas,
            "forecast_error_higher_for_every_candidate": all(x > 0 for x in error_deltas),
            "forecast_error_lower_for_every_candidate": all(x < 0 for x in error_deltas),
            "truth_scoring_selects_best_measured_candidate": proxy_regret == 0,
            "bus_cut_decision_recovery": None, "matched_control_cut": None}


def analyze():
    provenance = {}

    def read(path, lines=False):
        path = Path(path)
        full = path if path.is_absolute() else ROOT / path
        data = full.read_bytes()
        provenance[str(full.relative_to(ROOT))] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        return [json.loads(line) for line in data.splitlines() if line.strip()] if lines else json.loads(data)

    cosmos = read("reports/cosmos_first_query_truth_decision_20261015.json")
    require(all(cosmos["checks"].values()), "Cosmos cached qualification failed")
    require(cosmos["status"] == "completed_cache_analysis", "Cosmos analysis incomplete")
    utilities = []
    for i, row in enumerate(cosmos["historical_utilities"]):
        episode = read(row["source_episode"])
        require(row["candidate_index"] == i and episode["status"] == "completed"
                and episode["task_id"] == 3 and episode["init_id"] == 2
                and episode["condition"] == "clean" and episode["max_steps"] == 300,
                "Cosmos actual-utility source mismatch")
        require(episode["success"] == row["success"]
                and episode["environment_steps"] == row["steps"], "Cosmos utility differs")
        utilities.append(int(episode["success"]))
    scores = {k: v["values"] for k, v in cosmos["value_rankings"].items()}
    errors = {k: [row[k]["visual_latent_mse"] for row in cosmos["forecast_errors"]]
              for k in ("clean", "swap")}
    c = audit_bank("Cosmos", scores, utilities, False, "clean", "swap", errors,
                   "historical_original_goal_full_continuation_success",
                   "One known harmful task3/init2 root, two first16-control candidates; actual full-task utility is from prior original-Goal continuations. Fixed candidate seeds31/32. Not an independently selected evaluation root.")
    c["forecast_error_metric"] = "native_joint_VAE_visual_latent_MSE_at_control16"
    c["forecast_and_utility_horizon_match"] = False
    c["causal_limit"] = "Truth replacement also changes representation distribution. The score responds to future inputs, but this does not uniquely isolate physical-content error from codec/style or off-truth Value error."

    replay = read("reports/dino_first_query_full_horizon_20261015.json")
    truth = read("reports/dino_first_query_terminal_truth_20261015.json")
    ranking = read("reports/dino_first_query_two_mean_ranking_20261015.json")
    require(truth["status"] == "completed_calibrated_two_mean_terminal_truth"
            and truth["calibration"]["all_passed"], "DINO truth not qualified")
    require(ranking["checks"]["shared_context_and_goal_exact"]
            and ranking["checks"]["original_actions_unchanged"]
            and ranking["checks"]["original_goal_unchanged"], "DINO protected inputs differ")
    names = ranking["candidate_order"]
    require(names == truth["true_future_objective"]["candidate_order"]
            == replay["frozen_scoring"]["cost_matrix"]["columns"], "DINO candidate identity differs")
    measured = []
    for i, row in enumerate(replay["candidates"]):
        require(row["index"] == i and row["name"] == names[i]
                and row["status"] == "completed30" and row["historical_prefix_exact_steps"] == 15,
                "DINO physical replay incomplete")
        trace = read(row["trace"], lines=True)
        require(len(trace) == 30 and all(t["replay_candidate"] == i for t in trace),
                "DINO trace identity/horizon mismatch")
        utility = row["utility30"]
        require(utility == truth["physical_utilities"][row["name"]]
                and trace[-1]["reward"] == utility["terminal_native_reward"]
                and sum(t["reward"] for t in trace) == utility["cumulative_native_reward"]
                and trace[-1]["success"] == utility["terminal_success"], "DINO measured rewards differ")
        measured.append(utility)
    scores = dict(zip(ranking["cost_matrix"]["rows"], ranking["cost_matrix"]["values"]))
    scores["true_future"] = truth["true_future_objective"]["costs"]
    errors = {k: [truth["terminal_forecast_errors"][n][k]["weighted_mse"] for n in names]
              for k in ("original", "canonical")}
    d = audit_bank("DINO-WM", scores, [u["terminal_native_reward"] for u in measured], True,
                   "original", "canonical", errors, "terminal_native_reward_at_control30",
                   "One post-hoc two-mean bank at a shared native root. An explicitly added fixed-bank comparison, not the entire native CEM population.30-control open-loop utility; native policy replans after15 controls. No Goal-Swap interface.")
    d["forecast_error_metric"] = "terminal_visual_feature_MSE_plus_0.1_proprio_feature_MSE"
    d["forecast_and_utility_horizon_match"] = True
    d["secondary_actual_utilities"] = measured
    d["causal_limit"] = "The unchanged native scorer prefers the lower-utility branch even on measured truth. Scalar error reduction does not establish policy improvement; this does not prove that forecast error has no causal role."
    bounds = read("reports/cosmos_first_choice_identifiability_bounds_20261015.json")
    return {"schema_version": 1, "status": "completed_cached_fixed_bank_diagnostic",
            "research_goal_status": "active_unachieved", "banks": [c, d],
            "identity": {"formula": "U(best_in_bank)-U(selected) = [U(best_in_bank)-U(truth_scored)] + [U(truth_scored)-U(selected)]",
                         "meaning": "First term is a nonnegative within-bank truth-score mismatch. Second is signed and may be negative when forecast error happens to improve the choice. This is arithmetic, not a causal mediation fraction, new scientific mechanism, or decomposition over the full action space."},
            "existing_cosmos_panel_bounds": bounds["relative_to_recorded_native"],
            "statistics": {"post_hoc_roots": 2, "comparable_replications": 0,
                           "pooled_asr": None, "confidence_interval": None, "p_value": None,
                           "reason": "Different systems, operations, utilities and selected roots. Pair-flip fractions describe each finite bank only."},
            "decision": "Retain local Cosmos forecast-input/selection harm. Reject a DINO forecast-degradation explanation on this bank. Neither demonstrates deployable Bus-Cut recovery, matched Control-Cut selectivity or independent generalization. No new GPU or repeated closed-panel run is justified by this cache analysis.",
            "activity": {"new_model_calls": 0, "new_environment_controls": 0, "api_calls": 0, "training_updates": 0},
            "input_provenance": provenance,
            "reproduce": "python scripts/analyze_counterfactual_action_ranking.py --verify-only"}


def csv_text(result):
    output = io.StringIO(newline="")
    fields = ["system", "utility_metric", "condition", "selected_index", "selected_actual_utility",
              "within_bank_actual_regret", "truth_score_mismatch_regret", "signed_forecast_selection_difference"]
    writer = csv.DictWriter(output, fields, lineterminator="\n")
    writer.writeheader()
    for bank in result["banks"]:
        for row in bank["conditions"]:
            writer.writerow({k: bank[k] if k in bank else row[k] for k in fields})
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = analyze()
    table = csv_text(result)
    if args.verify_only:
        require(json.loads(args.out.read_text()) == result, "Saved diagnostic differs")
        require(args.out.with_suffix(".csv").read_text() == table, "Saved table differs")
    else:
        require(not args.out.exists() and not args.out.with_suffix(".csv").exists(), "Output exists; use --verify-only")
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        args.out.with_suffix(".csv").write_text(table)
    print(json.dumps({"status": "verified" if args.verify_only else result["status"],
                      "banks": len(result["banks"]), "source_files": len(result["input_provenance"]),
                      "report": str(args.out), **result["activity"]}))


if __name__ == "__main__":
    main()
