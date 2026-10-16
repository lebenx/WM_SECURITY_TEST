#!/usr/bin/env python3
"""Audit saved native recovery traces and caches without inference or simulation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ARMS = ("clean_identity_copy", "canonical_request")
ELAPSED = [1, 16, 31, 46, 61, 76, 91]
HORIZONS = [6, 6, 6, 6, 6, 5, 2]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def display(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def trace_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def arrays(path, keys=None):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in (archive.files if keys is None else keys)}


def metrics(rows, warmup_reward):
    check(len(rows) <= 99, "Trace exceeds the registered policy budget")
    check([r["policy_step"] for r in rows] == list(range(1, len(rows) + 1)), "Policy steps are not consecutive")
    check([r["environment_step"] for r in rows] == list(range(2, len(rows) + 2)), "Trace does not exclude exactly one warmup")
    check(all(np.isfinite(r["reward"]) and np.isfinite(r["reach_distance"]) for r in rows), "Nonfinite trace metric")
    full = len(rows) == 99
    if full:
        check(rows[-1]["done"], "Full trace lacks the native terminal flag")
    total = sum(r["reward"] for r in rows)
    return {
        "completed_99_controls": full, "controls": len(rows), "reward_sum": total,
        "reward_sum_including_shared_warmup": total + warmup_reward if full and warmup_reward is not None else None,
        "terminal_reward": rows[-1]["reward"] if full else None,
        "last_observed_reward": rows[-1]["reward"] if rows else None,
        "final_success": rows[-1]["success"] if full else None,
        "last_observed_success": rows[-1]["success"] if rows else None,
        "ever_success_observed": any(r["success"] for r in rows),
        "first_success_policy_step": next((r["policy_step"] for r in rows if r["success"]), None),
        "success_controls_observed": sum(bool(r["success"]) for r in rows),
        "last_reach_distance": rows[-1]["reach_distance"] if rows else None,
    }


def verify_outcome(measured, outcome):
    check(outcome is not None, "Completed run has no saved outcome")
    for actual, saved in (("reward_sum", "reward_sum"), ("final_success", "final_success"),
                          ("ever_success_observed", "ever_success"),
                          ("last_reach_distance", "final_reach_distance")):
        check(measured[actual] == outcome[saved], f"Trace/outcome mismatch: {actual}")
    check(outcome["policy_controls_executed"] == 99 and outcome["total_environment_steps"] == 100,
          "Saved outcome has a different budget")


def trace_comparison(recovery, clean):
    """Descriptions only: unequal trajectories do not invalidate utility recovery."""
    fields = ("action", "reward", "physical", "pixels", "state", "success", "reach_distance")
    comparisons = {}
    for key in fields:
        matches = [a[key] == b[key] for a, b in zip(recovery, clean)]
        comparisons[key] = {"matching_policy_steps": sum(matches), "all_99_steps_exact": all(matches),
                            "first_difference_policy_step": next((i + 1 for i, same in enumerate(matches) if not same), None)}
    entire = [a == b for a, b in zip(recovery, clean)]
    return {"entire_trace_exact": all(entire), "entire_rows_matching": sum(entire), "fields": comparisons,
            "scope": "Post-execution comparison of existing rows only. Physical/pixel equality compares their saved digests; no new hashes or replay. Exactness is descriptive same-root repeatability, not a prerequisite for utility recovery or evidence of independent-root generalization."}


def analyze(run, source):
    summary, protocol = read(run / "summary.json"), read(run / "launched_config.json")
    check(protocol["fixed"]["physical_seed"] == 4 and protocol["fixed"]["policy_controls"] == 99, "Unexpected registered task")
    check((ROOT / protocol["source_run"]).resolve() == source, "Source override differs from the frozen protocol")
    check(Path(summary["source_run"]).resolve() == source, "Summary/source linkage differs")
    completed = summary["status"] == "completed_single_native_alias_recovery"
    source_summary = read(source / "summary.json")
    check(source_summary["status"] == "completed_native_alias_pair", "Source pair is not completed")
    check(source_summary["decision"]["consider_separate_recovery_pilot"], "Source pair did not meet the registered trigger")
    source_rows, source_metrics = {}, {}
    source_warmup = source_summary["setup"]["warmup_reward_excluded_from_policy_return"]
    for arm in SOURCE_ARMS:
        source_rows[arm] = trace_rows(source / arm / "trace.jsonl")
        source_metrics[arm] = metrics(source_rows[arm], source_warmup)
        check(source_metrics[arm]["completed_99_controls"], f"Incomplete source arm: {arm}")
        verify_outcome(source_metrics[arm], source_summary["arms"][arm]["outcome"])

    rows = trace_rows(run / "recovery/trace.jsonl")
    warmup = summary.get("setup", {}).get("warmup_reward_excluded_from_policy_return")
    measured = metrics(rows, warmup)
    queries, observations = summary["queries"], []
    query_indices = [q["index"] for q in queries]
    check(query_indices == list(range(1, len(queries) + 1)), "Query indices are not consecutive")
    check(all(r["planning_call"] in query_indices for r in rows), "Trace refers to an unrecorded query")
    counters = {k: 0 for k in ("unroll_calls", "candidate_calls", "mean_calls", "saturated_xyz_coordinates",
                              "writer_changed_coordinates", "binding_restored_coordinates")}
    for q in queries:
        index = q["index"]
        check(index <= 7, "More than seven native planning queries")
        check(q["elapsed_before"] == ELAPSED[index - 1] and q["horizon"] == HORIZONS[index - 1], "Query schedule differs")
        executed = [r for r in rows if r["planning_call"] == index]
        counts_match = len(executed) == q["executed_controls"]
        if completed or q["status"] == "completed":
            check(counts_match, "Query/trace execution counts differ")
        stats = q["request_stats"]
        for key in counters:
            counters[key] += stats[key]
        check(0 <= stats["binding_restored_coordinates"] == stats["writer_changed_coordinates"] <= stats["saturated_xyz_coordinates"], "Binding/write coordinate totals disagree")
        check(stats["unroll_calls"] == stats["candidate_calls"] + stats["mean_calls"], "Query call totals disagree")
        observation = {"query": index, "status": q["status"], "horizon": q["horizon"],
                       "executed_controls_in_trace": len(executed), "reported_execution_count_exact": counts_match,
                       "chunk_reward_sum": sum(r["reward"] for r in executed), "request_stats": stats,
                       "planning_seconds": q.get("planning_seconds"), "saved_cache_checks": None}
        if q["status"] == "completed":
            expected_controls = 9 if index == 7 else 15
            check(len(executed) == expected_controls, "Completed query has the wrong real control count")
            check(stats["candidate_calls"] == stats["mean_calls"] == 15, "Native query budget differs")
            check([r["action_in_chunk"] for r in executed] == list(range(expected_controls)), "Chunk action indices differ")
            check([r["action"] for r in executed] == q["physical_actions"][:expected_controls], "Executed controls differ from native returned-control records")
            cache = arrays(run / f"recovery/query{index}_forecast_cache.npz",
                           ("mean_trusted", "mean_writer", "mean_bound", "local_cuda_generator_after", "actual_control_offsets"))
            trusted, writer, bound = (cache[k] for k in ("mean_trusted", "mean_writer", "mean_bound"))
            check(trusted.shape == writer.shape == bound.shape == (q["horizon"], 1, 20), "Final mean request shape differs")
            check(all(v.dtype == np.float32 and np.isfinite(v).all() for v in (trusted, writer, bound)), "Invalid final mean request")
            check(np.array_equal(trusted, bound), "Saved binding differs from the current trusted mean")
            normalized = np.asarray(q["normalized_macro_actions"], dtype=np.float32)
            check(np.array_equal(normalized, trusted[:3, 0]), "Returned native actions differ from the cached original final mean")
            check(np.array_equal(trusted.reshape(-1, 4)[:, 3], writer.reshape(-1, 4)[:, 3]), "Writer changed a cached gripper coordinate")
            mean_changes = int(np.count_nonzero(trusted != writer))
            check(mean_changes <= stats["writer_changed_coordinates"], "Saved mean changes exceed all query writes")
            reference_rng = arrays(source / f"clean_identity_copy/query{index}_forecast_cache.npz", ("local_cuda_generator_after",))
            check(np.array_equal(cache["local_cuda_generator_after"], reference_rng["local_cuda_generator_after"]), "CEM RNG differs from source Clean")
            check(q["source_local_generator_exact"], "Missing runtime RNG check")
            offsets = [5] if index == 7 else [5, 10, 15]
            check(cache["actual_control_offsets"].tolist() == q["actual_forecast_aligned_control_offsets"] == offsets, "Forecast-aligned real endpoints differ")
            observation["saved_cache_checks"] = {"current_mean_equals_bound": True, "returned_actions_equal_current_native_mean": True,
                "executed_actions_equal_returned_control_records": True, "local_cuda_rng_equals_clean": True,
                "last_mean_writer_changed_coordinates": mean_changes, "actual_control_offsets": offsets}
        observations.append(observation)

    comparison = exact_trace = initial_checks = None
    if completed:
        check(measured["completed_99_controls"] and len(queries) == 7 and all(q["status"] == "completed" for q in queries), "Completed recovery lacks the full episode")
        verify_outcome(measured, summary["outcome"])
        check(warmup == source_warmup, "Shared warmup reward differs")
        check(counters["unroll_calls"] == 210 and counters["candidate_calls"] == counters["mean_calls"] == 105, "Complete native inference budget differs")
        check(summary["activity"]["policy_controls"] == 99, "Activity real-step count differs")
        check(all(summary["activity"][k] == 210 for k in ("native_unrolls", "writer_calls", "binding_calls")), "Writer/binding/inference budgets differ")
        check(counters["writer_changed_coordinates"] > 0, "No canonical intervention was attempted")
        first = arrays(run / "recovery/first_proposal.npz")
        clean_first = arrays(source / "clean_identity_copy/first_proposal.npz", ("original", "context_visual", "context_proprio"))
        check(np.array_equal(first["trusted"], first["bound"]) and np.array_equal(first["trusted"], clean_first["original"]), "First native proposal/binding differs")
        check(all(np.array_equal(first[k], clean_first[k]) for k in ("context_visual", "context_proprio")), "First native context differs")
        goals, clean_goals = arrays(run / "recovery/goal_encoding.npz"), arrays(source / "clean_identity_copy/goal_encoding.npz")
        check(goals.keys() == clean_goals.keys() and all(np.array_equal(goals[k], clean_goals[k]) for k in goals), "Original goal encodings differ")
        initial_checks = {"first_original_proposal_exact": True, "first_context_exact": True, "goal_encoding_exact": True,
                          "first_proposal_writer_changed_coordinates": int(np.count_nonzero(first["trusted"] != first["writer"]))}
        clean, canonical, recovery = source_metrics[SOURCE_ARMS[0]]["reward_sum"], source_metrics[SOURCE_ARMS[1]]["reward_sum"], measured["reward_sum"]
        check(clean > canonical, "Source return-loss denominator is not positive")
        recovered, fraction = recovery >= clean - 1e-6, (recovery - canonical) / (clean - canonical)
        decision = summary["decision"]
        check(recovered == decision["utility_recovered_to_clean_within_1e_minus6_tolerance"], "Recovery decision differs")
        check(recovery - clean == decision["recovery_minus_clean_return"] and recovery - canonical == decision["recovery_minus_canonical_return"] and fraction == decision["recovered_fraction_unclipped"], "Reported recovery arithmetic differs")
        comparison = {"utility_recovered_to_clean_within_1e_minus6_tolerance": recovered,
                      "recovery_minus_clean_return": recovery - clean, "recovery_minus_canonical_return": recovery - canonical,
                      "recovered_fraction_unclipped": fraction,
                      "decision": "Registered cumulative return recovered on this root." if recovered else "Registered cumulative return was not fully recovered on this root; no automatic extra experiment.",
                      "interpretation": "Conventional trusted-current-candidate restoration; source trace exactness is not an acceptance criterion. Success metrics and terminal distance remain separate endpoints."}
        exact_trace = trace_comparison(rows, source_rows["clean_identity_copy"])

    return {"schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "completed_native_alias_recovery_trace_analysis" if completed else "incomplete_native_alias_recovery_trace_analysis",
            "execution_status": summary["status"], "research_goal_status": "active_unachieved",
            "source": display(run / "summary.json"), "protocol": display(run / "launched_config.json"),
            "source_pair": display(source / "summary.json"), "source_pair_outcomes": source_metrics,
            "recovery_outcome": measured, "comparison": comparison, "initial_checks": initial_checks,
            "source_clean_trace_comparison": exact_trace, "queries": observations, "query_request_totals": counters,
            "runtime": {k: summary[k] for k in ("elapsed_seconds", "activity", "gpu_at_start", "first_query_episode_cost_estimate_seconds", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes") if k in summary},
            "statistics": {"independent_physical_roots": 1, "completed_new_recovery_episodes": int(completed), "source_pair_episodes": 2,
                           "statistical_test": None, "confidence_interval": None,
                           "scope": "Same development root, with a separately registered restoration after a known return loss. No ASR or independent-root repeatability claim."},
            "limits": ["Only full, valid completion permits a recovery decision. Partial reward is an observed prefix, not task failure.",
                       "Per-query total rewrite/binding statistics are checked for consistency; all-call physical equivalence and trusted-storage isolation remain runtime assertions. Saved first-population/final-mean caches do not expose every candidate request.",
                       "Normalized final-mean equality and actual-control records are independently checked; this analyzer does not reload normalization statistics, re-encode observations, or replay physics.",
                       "Restoration demonstrates conventional input integrity under the stipulated trusted boundary, not a novel defense or a deployed lower-trust interface.",
                       "No forecast-error change, same-state ranking-flip rate or lost success is inferred from trajectory differences or cumulative return alone."],
            "reproduction": {"existing_run": ".venv-jepa-native/bin/python scripts/analyze_dino_native_alias_recovery.py",
                             "other_run": ".venv-jepa-native/bin/python scripts/analyze_dino_native_alias_recovery.py --run /absolute/path/to/run --report /absolute/path/to/report.json"},
            "analysis_activity": {"model_calls": 0, "environment_steps": 0, "network_calls": 0}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/dino_native_alias_recovery_20261015")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/dino_native_alias_recovery_decision_20261015.json")
    args = parser.parse_args()
    run = args.run.resolve()
    source = (ROOT / read(run / "launched_config.json")["source_run"]).resolve()
    report = analyze(run, source)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.report), "execution_status": report["execution_status"],
                      "recovery_outcome": report["recovery_outcome"], "comparison": report["comparison"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
