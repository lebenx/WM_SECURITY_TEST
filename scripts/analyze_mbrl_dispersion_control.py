"""Read-only analysis of the frozen dispersion panel; JSON/NumPy artifacts only."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODES = ("random", "boundary", "interior")
SEEDS = (810, 811, 812, 813)
RNG_KEYS = ("torch_rng_before", "torch_rng_after", "model_rng_before", "model_rng_after")
TERMINAL = {"completed_fixed_panel", "completed_panel_with_historical_gates_failed",
            "partial_resource_deferred", "technical_error"}


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def stats(values):
    values = [v for v in values if v is not None]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "positive": sum(v > 0 for v in values), "zero": sum(v == 0 for v in values),
            "negative": sum(v < 0 for v in values)}


def analyze(run, report, verify_only=False):
    started = time.monotonic()
    source = read(run / "summary.json")
    if source is None or source.get("status") not in TERMINAL:
        raise RuntimeError("Require a terminal source summary; do not analyze a live run")
    csv_path = run / "results.csv"
    if verify_only and not (report.exists() and csv_path.exists()):
        raise FileNotFoundError("Verify-only requires existing analysis and CSV")
    if not verify_only and (report.exists() or csv_path.exists()):
        raise FileExistsError("Analysis outputs already exist; never overwrite")
    issues, issue_count = [], 0
    pair_rows, csv_rows = [], []
    negative_control = {"status": "not_measured", "exact_200_rows": None}
    cache_count = 0
    batch_evaluations = 0
    imagined_particle_transitions = 0

    def check(ok, location, field):
        nonlocal issue_count
        if not ok:
            issue_count += 1
            if len(issues) < 24:
                issues.append({"location": str(location), "check": field})
        return bool(ok)

    for mode in MODES:
        for seed in SEEDS:
            pair_start_issues = issue_count
            version = 0 if seed < 812 else 1
            history_path = ROOT / f"runs/mbrl_covariance_pinching_eval_{mode}_metrics_v{version}/episode_{seed}.json"
            history = read(history_path)
            historical_rows = history["trajectory"]
            trajectories, query_records, arms = {}, {}, {}
            for arm in ("native", "fixed_dispersion"):
                directory = run / f"{mode}_{seed}" / arm
                summary = read(directory / "summary.json", {"status": "not_run"})
                trajectory = lines(directory / "trajectory.jsonl")
                queries = lines(directory / "queries.jsonl")
                trajectories[arm], query_records[arm] = trajectory, queries
                complete = summary["status"] == "completed_episode"
                observed_return = math.fsum(row["reward"] for row in trajectory) if trajectory else None
                if complete:
                    check(bool(trajectory) and len(trajectory) == summary["real_steps"], directory, "complete_step_count")
                    check(observed_return == summary["return"], directory, "reward_sum_matches_summary")
                    check(len(trajectory) == 200 or bool(trajectory[-1]["terminated"] or trajectory[-1]["truncated"]),
                          directory, "completed_episode_has_native_stop")
                    check(len(queries) == len(trajectory), directory, "one_completed_query_per_real_control")
                check([r["step"] for r in trajectory] == list(range(len(trajectory))), directory, "contiguous_real_steps")
                check([q["query_step"] for q in queries] == list(range(len(queries))), directory, "contiguous_queries")
                historical_prefix = all(i < len(historical_rows) and row == historical_rows[i]
                                        for i, row in enumerate(trajectory))
                historical_exact = complete and trajectory == historical_rows
                if arm == "native" and complete:
                    check(historical_exact, directory, "full_historical_trajectory_exact")
                    check(observed_return == history["summary"]["return_"], directory, "historical_return_exact")
                    check(summary.get("history_gate_pass") is True, directory, "runner_historical_gate_passed")
                # Recheck every cached query's intervention and returned-action contract.
                previous_warm = None
                for q in queries:
                    step = q["query_step"]
                    path = directory / q["cache"]
                    with np.load(path, allow_pickle=False) as z:
                        cache_count += 1
                        check(z["population"].shape == (3, 100, 15, 1), path, "registered_population_shape")
                        batch_evaluations += z["population"].shape[0]
                        # Ten particles is the frozen native rollout setting; only population dimensions are cached.
                        imagined_particle_transitions += math.prod(z["population"].shape[:3]) * 10
                        check(z["dispersion"].shape == (3, 15, 1), path, "registered_dispersion_shape")
                        check(np.all(z["dispersion"][0] == np.float32(.25)), path, "native_initial_variance")
                        check(np.array_equal(z["effective_variance"], np.minimum(z["bound_variance"], z["dispersion"])),
                              path, "effective_variance_is_bound_minimum")
                        bound = np.minimum(((z["mu"] + np.float32(1)) / np.float32(2)) ** 2,
                                           ((np.float32(1) - z["mu"]) / np.float32(2)) ** 2)
                        check(np.array_equal(z["bound_variance"], bound), path, "native_bound_formula")
                        check(np.array_equal(z["mu"][1:], z["native_updated_mu"][:-1]), path, "native_mean_update_used")
                        check(np.array_equal(z["dispersion"][1:], z["returned_dispersion"][:-1]), path, "dispersion_update_used")
                        if arm == "fixed_dispersion":
                            check(np.all(z["dispersion"] == np.float32(.25))
                                  and np.all(z["returned_dispersion"] == np.float32(.25)), path, "fixed_variance_all_iterations")
                        else:
                            check(np.array_equal(z["returned_dispersion"], z["native_updated_dispersion"]), path, "native_variance_update_unchanged")
                        maxima = z["values"].max(axis=1)
                        winner = int(np.argmax(maxima))  # First strict maximum, preserving cross-iteration ties.
                        index = int(z["elite_indices"][winner, 0])
                        check(winner == q["winner_iteration"] and index == q["winner_candidate_index"], path, "strict_best_provenance")
                        check(np.array_equal(z["returned_sequence"], z["population"][winner, index]), path, "returned_sequence_from_best")
                        expected_warm = np.roll(z["returned_sequence"], -1, axis=0)
                        expected_warm[-1] = 0
                        check(np.array_equal(z["warm_after"], expected_warm), path, "best_solution_warmstart")
                        check(np.array_equal(z["warm_before"], previous_warm) if previous_warm is not None
                              else np.all(z["warm_before"] == 0), path, "warmstart_continuity")
                        previous_warm = z["warm_after"].copy()
                        if step < len(trajectory):
                            check(np.array_equal(z["obs"], trajectory[step]["obs"]), path, "cached_actual_observation")
                            check(np.array_equal(z["returned_sequence"][0], trajectory[step]["action"]), path, "cached_executed_action")
                arms[arm] = {"status": summary["status"], "complete_episode": complete,
                    "real_steps_observed": len(trajectory), "completed_return": observed_return if complete else None,
                    "partial_observed_return": observed_return if not complete else None,
                    "query_caches": len(queries), "historical_prefix_exact": historical_prefix if arm == "native" and trajectory else None,
                    "historical_complete_exact": historical_exact if arm == "native" else None,
                    "at_200_step_cap": len(trajectory) == 200 and not trajectory[-1]["terminated"] if complete else None,
                    "headroom_to_200": 200 - observed_return if complete else None,
                    "terminated": trajectory[-1]["terminated"] if complete else None,
                    "truncated": trajectory[-1]["truncated"] if complete else None,
                    "runner_history_stats_exact": summary.get("historical_stats_exact") if arm == "native" else None,
                    "runner_history_stats_allclose": summary.get("historical_stats_allclose") if arm == "native" else None}

            shared = min(len(query_records["native"]), len(query_records["fixed_dispersion"]))
            comparison = {"shared_query_indices": shared, "all_shared_rng_arrays_exact": None if shared == 0 else True,
                "same_input_queries": 0, "effective_variance_changed_same_input": 0,
                "non_ceiling_effective_variance_changed_same_input": 0, "action_changed_same_input": 0,
                "full_sequence_changed_same_input": 0, "warm_after_changed_same_input": 0,
                "first_same_input_action_change_step": None, "ceiling_locked_same_input_queries": 0,
                "first_differences": {k: None for k in ("obs", "warm_before", "returned_sequence", "warm_after", "first_action")},
                "native_only_query_tail": len(query_records["native"]) - shared,
                "fixed_only_query_tail": len(query_records["fixed_dispersion"]) - shared}
            for step in range(shared):
                prefix = run / f"{mode}_{seed}"
                with np.load(prefix / "native" / query_records["native"][step]["cache"], allow_pickle=False) as n, \
                     np.load(prefix / "fixed_dispersion" / query_records["fixed_dispersion"][step]["cache"], allow_pickle=False) as f:
                    rng_exact = all(np.array_equal(n[k], f[k]) for k in RNG_KEYS)
                    check(rng_exact, f"{mode}/{seed}/{step}", "paired_shared_query_rng_before_after")
                    comparison["all_shared_rng_arrays_exact"] &= rng_exact
                    same = all(np.array_equal(n[k], f[k]) for k in ("obs", "warm_before"))
                    for key in comparison["first_differences"]:
                        left, right = (n["returned_sequence"][0], f["returned_sequence"][0]) if key == "first_action" else (n[key], f[key])
                        if comparison["first_differences"][key] is None and not np.array_equal(left, right):
                            coordinate = tuple(int(i) for i in np.argwhere(left != right)[0])
                            comparison["first_differences"][key] = {
                                "query_step": step, "same_observation_and_warmstart": same,
                                "first_changed_coordinate": list(coordinate),
                                "native_value": left[coordinate].item(), "fixed_value": right[coordinate].item()}
                    if not same:
                        continue
                    comparison["same_input_queries"] += 1
                    check(np.array_equal(n["population"][0], f["population"][0])
                          and np.array_equal(n["values"][0], f["values"][0]), prefix, "matched_input_first_iteration_exact")
                    changed = not np.array_equal(n["effective_variance"], f["effective_variance"])
                    non_ceiling = float(n["values"][0].max()) < 15.0
                    action_changed = not np.array_equal(n["returned_sequence"][0], f["returned_sequence"][0])
                    comparison["effective_variance_changed_same_input"] += int(changed)
                    comparison["non_ceiling_effective_variance_changed_same_input"] += int(changed and non_ceiling)
                    comparison["action_changed_same_input"] += int(action_changed)
                    comparison["full_sequence_changed_same_input"] += int(not np.array_equal(n["returned_sequence"], f["returned_sequence"]))
                    comparison["warm_after_changed_same_input"] += int(not np.array_equal(n["warm_after"], f["warm_after"]))
                    if action_changed and comparison["first_same_input_action_change_step"] is None:
                        comparison["first_same_input_action_change_step"] = step
                    if not non_ceiling:
                        comparison["ceiling_locked_same_input_queries"] += 1
                        check(np.array_equal(n["returned_sequence"], f["returned_sequence"]), prefix, "ceiling_locked_sequence_exact")
            complete = all(arms[a]["complete_episode"] for a in arms)
            if mode == "interior" and seed == 813 and complete:
                exact = len(trajectories["native"]) == 200 and trajectories["native"] == trajectories["fixed_dispersion"] == historical_rows
                check(exact, "interior/813", "prespecified_negative_control_full_trajectory")
                negative_control = {"status": "measured", "exact_200_rows": exact,
                    "all_native_first_rounds_at_ceiling": all(q["first_round_ceiling"] for q in query_records["native"])}
            valid = complete and arms["native"]["historical_complete_exact"] and issue_count == pair_start_issues
            delta = arms["fixed_dispersion"]["completed_return"] - arms["native"]["completed_return"] if valid else None
            pair_rows.append({"mode": mode, "seed": seed, "valid_completed_pair": bool(valid),
                              "arms": arms, "comparison": comparison, "delta_return": delta,
                              "audit_issue_count": issue_count - pair_start_issues})
            csv_rows.append({"mode": mode, "seed": seed, "valid_completed_pair": bool(valid),
                "native_status": arms["native"]["status"], "fixed_status": arms["fixed_dispersion"]["status"],
                "native_return": arms["native"]["completed_return"], "fixed_return": arms["fixed_dispersion"]["completed_return"],
                "delta_return": delta, "native_steps_observed": arms["native"]["real_steps_observed"],
                "fixed_steps_observed": arms["fixed_dispersion"]["real_steps_observed"],
                "historical_native_exact": arms["native"]["historical_complete_exact"],
                **{k: v for k, v in comparison.items() if k != "first_differences"}})

    interactions = []
    for seed in SEEDS:
        delta = {p["mode"]: p["delta_return"] for p in pair_rows if p["seed"] == seed}
        interactions.append({"seed": seed, "delta_return": delta,
            "boundary_minus_random": delta["boundary"] - delta["random"] if None not in (delta["boundary"], delta["random"]) else None,
            "boundary_minus_interior": delta["boundary"] - delta["interior"] if None not in (delta["boundary"], delta["interior"]) else None})
    aggregates = {m: stats([p["delta_return"] for p in pair_rows if p["mode"] == m]) for m in MODES}
    aggregate_interactions = {k: stats([r[k] for r in interactions]) for k in ("boundary_minus_random", "boundary_minus_interior")}
    totals = {"observed_real_steps": sum(a["real_steps_observed"] for p in pair_rows for a in p["arms"].values()),
        "cached_native_batch_evaluations": batch_evaluations,
        "derived_imagined_particle_transitions": imagined_particle_transitions,
        "particle_transition_basis": "Cached population iteration/candidate/horizon dimensions times the registered10 particles; not an additional execution counter.",
        "shared_query_rng_checks": sum(p["comparison"]["shared_query_indices"] for p in pair_rows),
        "individual_rng_array_comparisons": 4 * sum(p["comparison"]["shared_query_indices"] for p in pair_rows),
        **{k: sum(p["comparison"][k] for p in pair_rows) for k in (
            "same_input_queries", "effective_variance_changed_same_input",
            "non_ceiling_effective_variance_changed_same_input", "action_changed_same_input",
            "full_sequence_changed_same_input", "warm_after_changed_same_input", "ceiling_locked_same_input_queries")}}
    complete = all(p["valid_completed_pair"] for p in pair_rows)
    if not complete:
        decision = "Incomplete or invalid paired panel; retain observed outcomes, no aggregate mechanism verdict."
    elif not any(p["comparison"]["non_ceiling_effective_variance_changed_same_input"] for p in pair_rows if p["mode"] == "boundary"):
        decision = "No verified effective-variance change at a matched non-ceiling boundary query; necessity control is uninformative."
    elif aggregates["boundary"]["mean"] <= 0:
        decision = "No mean boundary-return recovery under the fixed operation; do not promote selective learned-dispersion mediation."
    elif all(v["mean"] > 0 for v in aggregate_interactions.values()):
        decision = "Positive mean boundary change and positive mean capped-return interactions; retain individual signs and median. This is not a robust or standalone boundary-specific recovery conclusion, especially with ceiling-censored controls."
    else:
        decision = "Boundary return improves without positive interactions against both controls; selective recovery is not established."
    result = {"schema_version": 1, "status": "completed_read_only_analysis", "source_run": str(run.relative_to(ROOT)),
        "source_status": source["status"], "source_claimed_complete_panel": source.get("complete_panel"),
        "validated_complete_panel": complete, "pairs": pair_rows, "per_seed_interactions": interactions,
        "delta_summaries": aggregates, "interaction_summaries": aggregate_interactions,
        "negative_control": negative_control, "audit_issue_count": issue_count, "audit_issue_examples": issues,
        "cached_panel_totals": totals,
        "decision": decision,
        "limitations": [
            "Four reused development seeds and one locally trained checkpoint per source mode; no model-training replication or population inference.",
            "Query comparisons are dependent and descriptive. RNG equality after trajectory divergence does not make physical observations equal.",
            "A changed unexecuted tail of returned_sequence can change the next warm start despite an identical current first action; both sequence and first-action differences are reported.",
            "Native random and interior controls each reach the200-step ceiling in3/4 historical episodes; positive interactions can reflect unequal headroom or control deterioration.",
            "This replaces within-query learned diagonal-dispersion updates; mean-dependent bound shrinkage remains. It does not fix population covariance or establish a unique mediator.",
            "Only completed, audited pairs contribute deltas. Partial recorded reward sums are separately retained and never filled with zero.",
            "No significance tests, confidence intervals, ASR,20-percent cutoff, new seed, tuning, novel-defense or paper-readiness claim."
        ],
        "activity": {"unique_query_caches_audited": cache_count, "model_loads": 0, "environment_steps": 0,
                     "api_calls": 0, "training_updates": 0, "elapsed_seconds": time.monotonic() - started},
        "reproduction": ".venv-native-local/bin/python scripts/analyze_mbrl_dispersion_control.py --verify-only",
        "csv_null_encoding": "Empty numeric cells denote unmeasured or invalid completed outcomes, never zero."}
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
    csv_text = csv_buffer.getvalue()
    if verify_only:
        saved = read(report)
        result["activity"]["elapsed_seconds"] = saved["activity"]["elapsed_seconds"]
        with csv_path.open(newline="") as handle:
            saved_csv = handle.read()
        if result != saved or csv_text != saved_csv:
            raise AssertionError("Cached reanalysis differs from saved report or CSV")
        print(json.dumps({"verify_only": "exact_match_except_elapsed_seconds", "writes": 0,
                          "audit_issues": issue_count, "report": str(report)}), flush=True)
        return
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    with csv_path.open("x", newline="") as handle:
        handle.write(csv_text)
    print(json.dumps({"validated_complete_panel": complete, "audit_issues": issue_count,
                      "decision": decision, "delta_summaries": aggregates, "report": str(report)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/mbrl_dispersion_control_20261015")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/mbrl_dispersion_control_analysis_20261015.json")
    parser.add_argument("--verify-only", action="store_true", help="Recompute and compare existing outputs without writing")
    args = parser.parse_args()
    analyze(args.run.resolve(), args.report.resolve(), args.verify_only)
