#!/usr/bin/env python3
"""Read-only stdlib analysis of the fixed Cosmos task3/init2 truth diagnostic."""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "runs/cosmos_first_query_truth_replay_20261015"
INFERENCE = ROOT / "runs/cosmos_first_query_truth_inference_20261015"
REPORT = ROOT / "reports/cosmos_first_query_truth_decision_20261015.json"
CSV = REPORT.with_suffix(".csv")
CONDITIONS = ("clean", "swap", "true_future")


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def lines(path, n):
    with Path(path).open() as handle:
        return [json.loads(line) for line in itertools.islice(handle, n)]


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def analyze():
    replay, result = read(REPLAY / "summary.json"), read(INFERENCE / "summary.json")
    protocol = read(INFERENCE / "launched_config.json")
    require(read(REPLAY / "launched_config.json") == protocol, "Launched protocols differ")
    require(replay["status"] == "completed_exact_two_candidate_prefix_replay"
            and replay["qualification"]["passed"] is True and replay["exit_code"] == 0,
            "Replay did not complete its qualification")
    require(result["status"] == "completed_calibrated_first_query_truth_countercheck",
            "Inference did not complete")
    require(protocol["task_id"] == 3 and protocol["init_id"] == 2
            and protocol["candidate_indices"] == [0, 1] and protocol["candidate_seeds"] == [31, 32]
            and protocol["horizon_controls"] == 16, "Fixed panel changed")
    require(len(replay["candidates"]) == len(result["forecast_errors"]) == 2,
            "Unexpected candidate count")
    expected_calls = {"policy": 2, "future": 4, "value": 6, "vae_encode": 3,
                      "noise_calls": 12, "environment_steps": 0, "api_calls": 0,
                      "training_updates": 0, "vae_decode": 0}
    require(result["activity"] == expected_calls, "Inference call budget mismatch")
    require(result["budget"] == {k: expected_calls[k] for k in ("policy", "future", "value", "vae_encode")},
            "Recorded budget mismatch")
    require(replay["activity"]["explicit_task_controls"] == 32
            and replay["activity"]["explicit_settling_controls"] == 20
            and replay["activity"]["model_forward_calls"] == replay["activity"]["vae_calls"] == 0,
            "Replay activity mismatch")
    require(len(result["checkpoint_loads"]) == 2 and set(result["checkpoint_loads"].values()) == {1},
            "Existing checkpoints were not loaded exactly once")
    calibration = result["calibration"]
    require(calibration["all_passed"] is True
            and calibration["historical_all_passed_before_truth_encoding"] is True
            and len(calibration["checks"]) == 18 and all(v is True for v in calibration["checks"].values()),
            "Incomplete historical or codec calibration")
    require(result["same_seed_local_noise_verified"] is True, "Candidate noise check absent")
    utilities, first_refs = [], {}
    for condition in ("clean", "swap"):
        episode = read(ROOT / protocol["sources"][condition]["episode"])
        require(episode["task_id"] == 3 and episode["init_id"] == 2, "Source root differs")
        first_refs[condition] = lines(ROOT / protocol["sources"][condition]["first_query"], 1)[0]
    for i, row in enumerate(replay["candidates"]):
        require(row["candidate_index"] == i and row["candidate_seed"] == 31 + i
                and row["status"] == "completed_exact16" and row["historical_trace_exact_steps"] == 16
                and row["task_controls"] == 16 and row["last_mismatched_fields"] == []
                and all(v is True for v in row["initial_checks"].values()), "Unqualified candidate replay")
        source = protocol["sources"][f"forced{i}"]
        require(row["source_episode"] == source["episode"], "Replay outcome provenance changed")
        ep = read(ROOT / source["episode"])
        require(ep["status"] == "completed" and ep["task_id"] == 3 and ep["init_id"] == 2
                and ep["goal"] == protocol["original_goal"] and ep["condition"] == "clean"
                and ep["max_steps"] == 300 and ep["best_of_n"] == 2
                and ep["seed"] == 31 and ep["environment_seed"] == 0, "Historical continuation protocol differs")
        utility = {"candidate_index": i, "success": ep["success"], "steps": ep["environment_steps"],
                   "capped_steps": ep["environment_steps"] if ep["success"] else 300}
        require(utility == result["historical_continuation_utilities"][i], "Cached utility disagrees with source episode")
        require(row["historical_full_continuation"] == {"success": ep["success"],
                "environment_steps": ep["environment_steps"], "measured_in_this_replay": False},
                "Replay mislabels an old full-task outcome")
        old = lines(ROOT / source["trace"], 16)
        new = lines(ROOT / row["trace"], 17)
        require(len(old) == len(new) == 16 and old == new, "Recorded physical prefix fields are not exact")
        require(not any(x["done"] or x["task_success"] for x in new), "Unexpected first16 terminal event")
        first = lines(ROOT / source["first_query"], 1)[0]
        require(first["native_best_of_n"]["selected_index"] == i and first["first_step"] == first["chunk"] == 1,
                "Wrong historical candidate selection")
        require([x["action"] for x in new] == first["environment_actions"], "Replay controls are not frozen source controls")
        for condition in ("clean", "swap"):
            candidate = first_refs[condition]["native_best_of_n"]["candidates"][i]
            require(candidate["index"] == i and candidate["seed"] == 31 + i
                    and candidate["environment_actions"] == first["environment_actions"]
                    and candidate["predicted_value"] == result["scores"][condition][i],
                    "Historical action, seed or Value calibration disagrees with source")
        utility["source_episode"] = source["episode"]
        utilities.append(utility)
    require([(u["success"], u["steps"]) for u in utilities] == [(False, 300), (True, 170)],
            "Historical fixed branch outcomes changed")
    scores, rows = {}, []
    error_summary = []
    for i, record in enumerate(result["forecast_errors"]):
        require(record["candidate_index"] == i, "Forecast error candidate mismatch")
        metrics = {}
        for condition in ("clean", "swap"):
            e = record[condition]
            require(all(math.isfinite(e[k]) and e[k] >= 0 for k in (
                "visual_latent_mse", "future_wrist_slot_mse", "future_primary_slot_mse", "future_proprio_slot_mse")),
                "Nonfinite or negative cached error")
            require(math.isclose(e["visual_latent_mse"], (e["future_wrist_slot_mse"] + e["future_primary_slot_mse"]) / 2,
                                 rel_tol=1e-14, abs_tol=1e-16), "Visual MSE aggregation mismatch")
            require(e["normalized_proprio_mse"] is None, "Unexpected new proprio extractor metric")
            metrics[condition] = e
        require(metrics["clean"]["visual_latent_mse"] > 0, "Visual ratio denominator is zero")
        error_summary.append({"candidate_index": i, **metrics,
            "visual_mse_swap_over_clean": metrics["swap"]["visual_latent_mse"] / metrics["clean"]["visual_latent_mse"],
            "visual_mse_swap_minus_clean": metrics["swap"]["visual_latent_mse"] - metrics["clean"]["visual_latent_mse"]})
    for condition in CONDITIONS:
        values = result["scores"][condition]
        require(len(values) == 2 and all(math.isfinite(x) for x in values), "Invalid score pair")
        rank = sorted(range(2), key=lambda i: (-values[i], i))
        tie, selected = values[0] == values[1], rank[0]
        cached = result["ranking"][condition]
        require(cached["rank"] == rank and cached["selected_index"] == selected and cached["exact_tie"] == tie
                and cached["candidate1_minus_candidate0_value"] == values[1] - values[0], "Cached ranking inconsistent")
        utility = {k: v for k, v in utilities[selected].items() if k != "source_episode"}
        require(cached["historical_selected_utility"] == utility, "Selected utility binding differs")
        scores[condition] = {"values": values, "rank": rank, "selected_index": selected, "exact_tie": tie,
            "candidate1_minus_candidate0_value": values[1] - values[0],
            "selection_basis": "tie_driven_candidate0_priority" if tie else "strict_value_order",
            "historical_selected_utility": utilities[selected]}
        for i in range(2):
            e = error_summary[i][condition] if condition != "true_future" else {}
            rows.append({"task_id": 3, "init_id": 2, "candidate_index": i, "candidate_seed": 31 + i,
                "condition": condition, "value": values[i], "rank_position": rank.index(i) + 1,
                "selected": i == selected, "score_pair_tied": tie,
                "visual_latent_mse": e.get("visual_latent_mse"),
                "future_proprio_slot_mse": e.get("future_proprio_slot_mse"),
                "visual_mse_swap_over_clean": error_summary[i]["visual_mse_swap_over_clean"] if condition == "swap" else None,
                "historical_success": utilities[i]["success"], "historical_steps": utilities[i]["steps"],
                "historical_capped_steps": utilities[i]["capped_steps"], "new_full_episode_executed": False})
    require([scores[c]["selected_index"] for c in CONDITIONS] == [1, 0, 1]
            and not any(scores[c]["exact_tie"] for c in CONDITIONS), "Observed fixed result changed; revise interpretation explicitly")
    report = {
        "schema_version": 1, "status": "completed_cache_analysis",
        "question": protocol["question"], "scope": protocol["selection_scope"],
        "sources": {"protocol": relative(INFERENCE / "launched_config.json"),
                    "replay_summary": relative(REPLAY / "summary.json"),
                    "inference_summary": relative(INFERENCE / "summary.json"),
                    "historical_episodes": [u["source_episode"] for u in utilities],
                    "analysis_script": "scripts/analyze_cosmos_first_query_truth.py"},
        "checks": {"launched_protocols_equal": True, "fixed_root_and_two_candidates": True,
                   "recorded_physical_prefix_rows_exact": 32, "historical_utilities_match_episode_files": True,
                   "historical_control_seed_value_bindings_exact": True, "calibration_checks_passed": 18,
                   "historical_calibration_before_truth_encode": True, "native_argmax_recomputed": True,
                   "no_value_ties": True, "calls_match_frozen_budget": True},
        "historical_utilities": utilities, "value_rankings": scores, "forecast_errors": error_summary,
        "findings": {
            "swap_visual_mse_higher_for_both_candidates": all(e["visual_mse_swap_minus_clean"] > 0 for e in error_summary),
            "truth_substitution_restores_clean_selection": True,
            "clean_and_truth_choose_historically_successful_candidate": True,
            "swap_chooses_historically_failed_candidate": True,
            "selected_historical_capped_cost_swap_minus_clean": 130,
            "new_full_task_or_recovery_episodes": 0,
            "interpretation": "This fixed realization supports local predicted-future sensitivity: changing only the Future text changes the Value ranking; replacing the future slots with encoded measured consequences restores the successful candidate ordering. Compared with the separately replayed actual futures, visual forecast error is higher under Swap for each candidate; the two branches share one root and are not independent task samples. These are separate measured facts, not a mediation percentage or a universal error-to-harm law.",
            "decision": "Local causal explanation strengthened; keep this post-hoc countercheck bounded. No new attack/generalization claim and no reopening of the failed30-root expansion gate."},
        "execution": {"replay": {"session_id": 66992, "exit_code": 0,
                                  "elapsed_seconds": replay["elapsed_seconds"], "activity": replay["activity"]},
                      "inference": {"session_id": 28845, "exit_code": 0,
                                    "elapsed_seconds": result["elapsed_seconds"], "activity": result["activity"],
                                    "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
                                    "peak_cuda_reserved_bytes": result["peak_cuda_reserved_bytes"]},
                      "session_metadata_source": "Parent agent launch/terminal reports; elapsed times and activity are read from the saved summaries.",
                      "analysis": {"implementation": "Python standard library, cached JSON/JSONL only",
                                   "model_calls": 0, "environment_steps": 0, "api_calls": 0, "weight_loads": 0}},
        "limits": protocol["interpretation_rules"] + [
            "Actual full-task success/failure is reused from existing original-Goal continuations; the new physical work only reproduces32 controls, plus20 explicit settling controls.",
            "Truth substitution restores a cached-score selection, not a newly executed successful recovery episode or Bus-Cut defense.",
            "Exact initial pixels/state and recorded prefix fields do not prove equality of every unrecorded historical state; historical lossless endpoint images and full forecast latents were unavailable.",
            "Forecast errors are native joint-VAE visual features. Slot5 compares tiled normalized-proprio targets; no native9D extraction or raw physical error is claimed. All-slot means are descriptive only.",
            "The fixed candidate seeds31/32 represent one stochastic realization. Value is a discounted behavior-return proxy, not calibrated success probability.",
            "One post-hoc known harmful root cannot estimate population ASR, independent generalization, or complete closed-loop mediation. Value error on predicted latents remains possible."],
        "csv": relative(CSV),
        "reproduce": "python scripts/analyze_cosmos_first_query_truth.py --verify-only"}
    return report, rows


def csv_text(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    report, rows = analyze()
    table = csv_text(rows)
    if args.verify_only:
        require(read(REPORT) == report and CSV.read_text() == table, "Saved analysis differs from current cache recomputation")
    else:
        require(not REPORT.exists() and not CSV.exists(), "Refusing to overwrite preserved analysis")
        REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        CSV.write_text(table)
    print(json.dumps({"status": "verified" if args.verify_only else "written", "report": relative(REPORT),
                      "selections": [report["value_rankings"][c]["selected_index"] for c in CONDITIONS],
                      "visual_error_ratios": [e["visual_mse_swap_over_clean"] for e in report["forecast_errors"]]}))


if __name__ == "__main__":
    main()
