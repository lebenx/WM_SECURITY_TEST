#!/usr/bin/env python3
"""Summarize the frozen native alias pair from actual traces, without inference."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("clean_identity_copy", "canonical_request")


def read(path):
    return json.loads(path.read_text())


def display_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/dino_native_alias_pair_20261015")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/dino_native_alias_decision_20261015.json")
    args = parser.parse_args()
    run = args.run.resolve()
    summary = read(run / "summary.json")
    protocol = read(run / "launched_config.json")
    assert protocol["task"]["physical_seed"] == 4
    completed = summary["status"] == "completed_native_alias_pair"
    results, queries = [], []
    for arm in ("noop", *ARMS):
        path = run / arm / "trace.jsonl"
        if not path.exists():
            continue
        trace = [json.loads(line) for line in path.read_text().splitlines()]
        if not trace:
            continue
        assert [r["policy_step"] for r in trace] == list(range(1, len(trace) + 1))
        assert [r["environment_step"] for r in trace] == list(range(2, len(trace) + 2))
        is_full = len(trace) == 99 and trace[-1]["environment_step"] == 100
        if completed:
            assert is_full, arm
        measured = {"arm": arm, "completed_99_controls": is_full, "controls": len(trace),
                    "reward_sum": sum(r["reward"] for r in trace),
                    "reward_sum_including_shared_warmup": (sum(r["reward"] for r in trace) + summary["setup"]["warmup_reward_excluded_from_policy_return"]) if is_full else None,
                    "terminal_reward": trace[-1]["reward"] if is_full else None,
                    "last_observed_reward": trace[-1]["reward"],
                    "final_success": trace[-1]["success"] if is_full else None,
                    "last_observed_success": trace[-1]["success"],
                    "ever_success_observed": any(r["success"] for r in trace),
                    "first_success_policy_step": next((r["policy_step"] for r in trace if r["success"]), None),
                    "success_controls_observed": sum(bool(r["success"]) for r in trace),
                    "last_reach_distance": trace[-1]["reach_distance"]}
        outcome = summary["noop"] if arm == "noop" else summary["arms"][arm]["outcome"]
        if is_full and outcome is not None:
            assert measured["reward_sum"] == outcome["reward_sum"]
            assert measured["final_success"] == outcome["final_success"]
            assert measured["ever_success_observed"] == outcome["ever_success"]
            assert measured["last_reach_distance"] == outcome["final_reach_distance"]
        results.append(measured)
        if arm == "noop":
            assert all(r["action"] == [0., 0., 0., 0.] for r in trace)
            continue
        arm_queries = summary["arms"][arm]["queries"]
        if completed:
            assert [q["elapsed_before"] for q in arm_queries] == [1, 16, 31, 46, 61, 76, 91]
            assert [q["horizon"] for q in arm_queries] == [6, 6, 6, 6, 6, 5, 2]
        for q in arm_queries:
            executed = [r for r in trace if r["planning_call"] == q["index"]]
            assert len(executed) == q["executed_controls"]
            if q["status"] != "completed":
                continue
            assert [r["action"] for r in executed] == q["physical_actions"][:len(executed)]
            stats = q["request_stats"]
            assert stats["candidate_calls"] == stats["mean_calls"] == 15
            assert stats["unroll_calls"] == 30
            assert stats["effective_actions_and_immutable_inputs_exact"]
            assert q["actual_forecast_aligned_control_offsets"] == ([5] if q["index"] == 7 else [5, 10, 15])
            if arm == ARMS[0]:
                assert stats["changed_normalized_coordinates"] == 0
            queries.append({"arm": arm, "query": q["index"], "horizon": q["horizon"],
                            "executed_controls": len(executed),
                            "planning_seconds": q["planning_seconds"],
                            "chunk_reward_sum": sum(r["reward"] for r in executed),
                            "chunk_final_reward": executed[-1]["reward"],
                            "chunk_final_success": executed[-1]["success"],
                            "chunk_final_distance": executed[-1]["reach_distance"],
                            "changed_normalized_coordinates": stats["changed_normalized_coordinates"]})
    comparison = None
    paired_cache_checks = None
    if completed:
        import numpy as np
        assert len(results) == 3 and len(queries) == 14
        by_arm = {r["arm"]: r for r in results}
        clean, canonical = by_arm[ARMS[0]], by_arm[ARMS[1]]
        def arrays(path):
            with np.load(path, allow_pickle=False) as archive:
                return {k: archive[k].copy() for k in archive.files}
        first = [arrays(run / a / "first_proposal.npz") for a in ARMS]
        goals = [arrays(run / a / "goal_encoding.npz") for a in ARMS]
        assert np.array_equal(first[0]["original"], first[1]["original"])
        assert np.array_equal(first[0]["original"], first[0]["rewritten"])
        assert not np.array_equal(first[1]["original"], first[1]["rewritten"])
        for key in ("context_visual", "context_proprio"):
            assert np.array_equal(first[0][key], first[1][key])
        assert goals[0].keys() == goals[1].keys()
        assert all(np.array_equal(goals[0][k], goals[1][k]) for k in goals[0])
        for index in range(1, 8):
            pair = [arrays(run / a / f"query{index}_forecast_cache.npz") for a in ARMS]
            assert np.array_equal(pair[0]["local_cuda_generator_after"], pair[1]["local_cuda_generator_after"])
        paired_cache_checks = {"first_original_proposals_exact": True, "first_context_exact": True,
                               "goal_encodings_exact": True, "local_cuda_rng_exact_after_all_7_queries": True,
                               "all_executed_actions_match_native_return_records": True,
                               "per_request_physical_equivalence": "Runtime assertions in launched script/summary; not independently replayed here for every unexecuted candidate."}
        delta = canonical["reward_sum"] - clean["reward_sum"]
        drop = -delta / clean["reward_sum"] if clean["reward_sum"] > 0 else None
        success_loss = clean["final_success"] and not canonical["final_success"]
        advance = bool(success_loss or (clean["reward_sum"] > by_arm["noop"]["reward_sum"] and drop is not None and drop >= .10))
        assert delta == summary["decision"]["canonical_minus_clean_return"]
        assert advance == summary["decision"]["consider_separate_recovery_pilot"]
        comparison = {"canonical_minus_clean_return": delta, "relative_return_drop": drop,
                      "final_success_loss": success_loss, "consider_separate_recovery_pilot": advance,
                      "native_effect": "lower_return" if delta < 0 else "unchanged_return" if delta == 0 else "higher_return",
                      "secondary_success_metrics": "Descriptive, not advance criteria. Report first success and successful control counts alongside return; lower return is not necessarily worse goal attainment.",
                      "decision": "A separately frozen causal recovery pilot is justified; no paper/generalization claim." if advance else
                                  "Stop expansion of this fixed canonical transform; preserve all signed results and earlier local loss.",
                      "prediction_accuracy": "Not measured on this native run. Different visited trajectories must not be compared as a paired forecast-error effect."}
    report = {"schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "completed_native_pair_trace_analysis" if completed else "incomplete_native_pair_trace_analysis",
              "execution_status": summary["status"], "research_goal_status": "active_unachieved",
              "source": display_path(run / "summary.json"),
              "readiness": "reports/dino_native_alias_readiness_20261015.json",
              "protocol": "configs/dino_native_alias_pair_20261015.json",
              "outcomes": results, "comparison": comparison, "paired_checks": paired_cache_checks,
              "runtime": {k: summary[k] for k in ("elapsed_seconds", "activity", "gpu_at_start", "first_query_pair_cost_estimate_seconds", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes") if k in summary},
              "statistics": {"independent_physical_roots": 1, "statistical_test": None, "confidence_interval": None,
                             "interpretation": "One prospective development root, exploratory after earlier bank evidence. No ASR, population significance or training-heldout generalization."},
              "limits": ["This tests the newly frozen canonicalization hypothesis, not the failed fixed+2 operation or Goal-Swap.",
                         "No defense episode has been run by this protocol; candidate binding remains a conventional proposed restoration baseline.",
                         "Prior fixed-bank latent error reduction and true-future objective mismatch remain scoped to that bank.",
                         "A null or improvement on native seed4 does not isolate replanning as the cause: the earlier seed1 fixed-bank experiment also used a different root and candidate bank.",
                         "Incomplete episodes are not final task failures or completed-pair evidence."],
              "reproduction": {
                  "analysis_existing_run": ".venv-jepa-native/bin/python scripts/analyze_dino_native_alias_pair.py",
                  "new_run": ".venv-jepa-native/bin/python scripts/pilot_dino_native_alias_pair.py --out /absolute/path/to/new_output",
                  "analysis_new_run": ".venv-jepa-native/bin/python scripts/analyze_dino_native_alias_pair.py --run /absolute/path/to/new_output --report /absolute/path/to/new_report.json",
                  "note": "Runner refuses an existing directory. Reproduction retains the frozen seed, full native budget and resource gates."},
              "analysis_activity": {"model_calls": 0, "environment_steps": 0, "network_calls": 0}}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for filename, rows in (("outcome_results.csv", results), ("query_results.csv", queries)):
        if rows:
            with (run / filename).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"report": str(args.report), "execution_status": summary["status"], "outcomes": results, "comparison": comparison}, ensure_ascii=False))


if __name__ == "__main__":
    main()
