#!/usr/bin/env python3
"""Verify and aggregate all THREE preregistered Cosmos counterfactual windows.

Read-only with respect to predictions, commits, model files and environment
results. Recomputes each join on CPU, verifies archived physical states and
actual trajectory outcomes, and writes only a fresh aggregate artifact.
No GPU, EGL, learned inference, new episode or external API is used.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

import cosmos_counterfactual_replay as replay
import evaluate_cosmos_counterfactual_ranking as evaluation
import pilot_cosmos_counterfactual_action_ranking as ranking

PREFIXES = (0, 16, 80)
DEFAULT_ROOT = ranking.ROOT / "runs/cosmos_counterfactual_evaluation_v1"


def check_file(path, expected):
    path = Path(path).resolve()
    actual = replay.file_sha(path)
    if actual != expected:
        raise ValueError(f"Artifact SHA256 mismatch: {path}")
    return actual


def verify_physical_archive(artifact):
    """Reconstruct archived signatures, not unrecorded simulator attributes."""
    path = Path(artifact["state_npz"])
    check_file(path, artifact["state_npz_sha256"])
    with np.load(path, allow_pickle=False) as archive:
        def materialize(node):
            if isinstance(node, dict):
                if set(node) == {"npz_key"}:
                    return archive[node["npz_key"]].copy()
                return {key: materialize(value) for key, value in node.items()}
            if isinstance(node, list):
                return [materialize(value) for value in node]
            return node
        physical = materialize(artifact["state_structure"])
    signature = replay._signature(physical)
    if signature != artifact["state_signature"] or replay._json_sha(signature) != artifact["state_hash"]:
        raise ValueError("Archived physical state does not reproduce the reported hash")
    return {str(path.resolve()): artifact["state_npz_sha256"]}


def verify_window(result_path, expected_prefix):
    result_path = Path(result_path).resolve()
    saved = json.loads(result_path.read_text())
    if saved.get("status") != "joined_committed_predictions_with_independent_real_environment_outcomes":
        raise ValueError("Counterfactual join is incomplete")
    if saved.get("prefix_steps") != expected_prefix:
        raise ValueError("Result does not match its preregistered prefix")
    commit_path, utility_path = Path(saved["prediction_commit"]), Path(saved["utilities"])
    check_file(commit_path, saved["prediction_commit_sha256"])
    check_file(utility_path, saved["utilities_sha256"])
    commit, prediction = evaluation.verify_commit(commit_path)
    # Rechecks input/source integrity, exact candidate identity, original goal,
    # native argmax, all prediction artifact hashes and phase RNG hashes.
    ranking_data, _, _ = evaluation.validate_ranking(commit["ranking_json"])
    if ranking_data != prediction:
        raise ValueError("Prediction changed while validating")
    if commit.get("preregistered_protocol_sha256") != replay.file_sha(evaluation.SPEC):
        raise ValueError("Development protocol differs from prediction commitment")
    utility = evaluation._validate_utilities(commit_path, commit, prediction, utility_path)
    source = prediction["source"]
    files = {str(result_path): replay.file_sha(result_path), str(commit_path.resolve()): replay.file_sha(commit_path),
             str(utility_path.resolve()): replay.file_sha(utility_path)}
    snapshot = json.loads(Path(source["snapshot"]).read_text())
    files.update(verify_physical_archive(snapshot))
    for index in range(2):
        candidate = utility["candidates"][str(index)]
        actions = np.load(candidate["actions"], allow_pickle=False)
        for sample in candidate["replays"]:
            for phase in ("initial", "candidate_future", "terminal"):
                files.update(verify_physical_archive(sample[phase]))
            if sample["first_success_snapshot"] is not None:
                files.update(verify_physical_archive(sample["first_success_snapshot"]))
            trace_path = Path(sample["trajectory"])
            check_file(trace_path, sample["trajectory_sha256"])
            rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
            if replay._json_sha(rows) != sample["trajectory_outcomes_sha256"]:
                raise ValueError("Actual trajectory does not reproduce reported outcome hash")
            steps = list(range(expected_prefix + 1, sample["last_executed_absolute_step"] + 1))
            if [row["absolute_step"] for row in rows] != steps:
                raise ValueError("Actual trajectory steps are missing or noncontiguous")
            first_success = next((row["absolute_step"] for row in rows if row["task_success"] is True), None)
            if any(type(row["task_success"]) is not bool for row in rows):
                raise ValueError("Actual success flags must be booleans")
            if first_success != sample["steps_to_success"] or (first_success is not None) is not sample["success"]:
                raise ValueError("Utility differs from actual first-success trace")
            physical_controls = np.asarray([row["action"] for row in rows[:16]], dtype=actions.dtype)
            if replay.array_sha(physical_controls) != candidate["action_sha256"]:
                raise ValueError("Actually executed candidate controls differ from commitment")
            if any(row["phase"] != "candidate" for row in rows[:16]) or any(
                    row["phase"] != "original_goal_continuation" for row in rows[16:]):
                raise ValueError("Unexpected trajectory phase assignment")
            files[str(trace_path.resolve())] = sample["trajectory_sha256"]
    # Recompute actual metrics from committed RGB and independent raw targets.
    # The temporary result is removed; existing reports are never overwritten.
    with TemporaryDirectory(prefix="cosmos_join_recompute_") as temporary:
        fresh = evaluation.join_results(commit_path, utility_path, Path(temporary) / "results.json")
        recomputed = json.loads(fresh.read_text())
    if recomputed != saved:
        raise ValueError("Saved join differs from independent CPU recomputation")
    return {"result": saved, "prediction": prediction, "commit": commit, "utility": utility,
            "files_sha256": files,
            "verification": {"all_prediction_source_hashes_verified": True, "all_physical_archives_verified": True,
                             "candidate_controls_and_first_success_recomputed_from_actual_trace": True,
                             "rgb_errors_and_selected_utilities_recomputed_exact": True,
                             "fresh_candidate_replays": 4}}


def mean(values):
    return float(np.mean(values)) if values else None


def assemble(bundles):
    prefixes = [bundle["result"]["prefix_steps"] for bundle in bundles]
    if len(prefixes) != 3 or sorted(prefixes) != list(PREFIXES):
        raise ValueError("Aggregate requires exactly all three preregistered prefixes0/16/80; no replacement or partial denominator")
    bundles = sorted(bundles, key=lambda item: item["result"]["prefix_steps"])
    sources = [bundle["prediction"]["source"] for bundle in bundles]
    episode_identities = {(source["episode_json"], source["episode_sha256"]) for source in sources}
    if len(episode_identities) != 1:
        raise ValueError("All three windows must come from the same source episode")
    provenances = [source["inference_provenance"] for source in sources]
    if any(provenance != provenances[0] for provenance in provenances):
        raise ValueError("Inference implementation/checkpoint differs across windows")
    for bundle in bundles:
        result, utility = bundle["result"], bundle["utility"]
        if result["utility_estimand"] != evaluation.UTILITY_ESTIMAND:
            raise ValueError("Cannot pool different continuation utility estimands")
        if utility["repeat"] != 2 or len(utility["candidates"]) != 2 or utility["repeat_passed"] is not True:
            raise ValueError("Each window requires exactly two candidates with two exact repeated replays")
        if result["forecast_error_available"] is not True:
            raise ValueError("All preregistered windows need both decoded views and real16-step targets")
    table, windows = [], []
    for bundle in bundles:
        result = bundle["result"]
        windows.append({"prefix_steps": result["prefix_steps"], "source_state_sha256": result["source_state_sha256"],
                        "actual_candidate_utilities": result["actual_candidate_utilities"],
                        "primary_success_distinguishes_candidates": result["ranking_test_informative_for_primary_success"],
                        "conditions": [{key: row[key] for key in ("condition", "selected_index", "predicted_scores",
                                       "ranking_flip_from_clean", "actual_selected_action_success", "actual_steps_to_success",
                                       "actual_success_loss_from_clean", "extra_steps_vs_clean_when_both_succeed",
                                       "mean_candidate_rgb_mse_0_1", "persistence_mean_candidate_rgb_mse_0_1")}
                                       for row in result["results"]]})
    for condition in ranking.CONDITIONS:
        rows = [next(row for row in bundle["result"]["results"] if row["condition"] == condition) for bundle in bundles]
        paired_steps = [row["extra_steps_vs_clean_when_both_succeed"] for row in rows
                        if row["extra_steps_vs_clean_when_both_succeed"] is not None]
        successful_steps = [row["actual_steps_to_success"] for row in rows if row["actual_selected_action_success"]]
        before_stop_errors, post_success_count = [], 0
        for row in rows:
            for item in row["forecast_error"]:
                if item["target_after_native_success_stop"]:
                    post_success_count += 1
                else:
                    before_stop_errors.append(item["error"]["mean_rgb_mse_0_1"])
        table.append({"condition": condition, "windows": 3,
                      "mean_rgb_mse_0_1": mean([row["mean_candidate_rgb_mse_0_1"] for row in rows]),
                      "mean_persistence_rgb_mse_0_1": mean([row["persistence_mean_candidate_rgb_mse_0_1"] for row in rows]),
                      "mean_mse_improvement_over_persistence": mean([row["forecast_mse_improvement_over_persistence"] for row in rows]),
                      "windows_better_than_persistence": sum(row["forecast_mse_improvement_over_persistence"] > 0 for row in rows),
                      "mean_rgb_mse_before_native_success_stop": mean(before_stop_errors),
                      "candidate_targets_after_native_success_stop": post_success_count,
                      "ranking_flip_count": sum(row["ranking_flip_from_clean"] for row in rows),
                      "ranking_flip_rate_descriptive": sum(row["ranking_flip_from_clean"] for row in rows) / 3,
                      "selected_candidate_flip_count": sum(row["selected_candidate_flip_from_clean"] for row in rows),
                      "selected_successes": sum(row["actual_selected_action_success"] for row in rows),
                      "selected_success_rate_descriptive": sum(row["actual_selected_action_success"] for row in rows) / 3,
                      "selected_success_by_prefix": [row["actual_selected_action_success"] for row in rows],
                      "selected_steps_to_success_by_prefix": [row["actual_steps_to_success"] for row in rows],
                      "mean_steps_to_success_among_successes": mean(successful_steps),
                      "paired_success_loss_clean_minus_condition_by_prefix": [row["actual_success_loss_from_clean"] for row in rows],
                      "mean_paired_success_loss_clean_minus_condition": mean([row["actual_success_loss_from_clean"] for row in rows]),
                      "windows_losing_clean_success": sum(row["actual_success_loss_from_clean"] > 0 for row in rows),
                      "windows_gaining_success_vs_clean": sum(row["actual_success_loss_from_clean"] < 0 for row in rows),
                      "paired_extra_steps_condition_minus_clean_by_prefix": [row["extra_steps_vs_clean_when_both_succeed"] for row in rows],
                      "paired_steps_both_success_count": len(paired_steps),
                      "mean_paired_extra_steps_when_both_succeed": mean(paired_steps),
                      "windows_with_secondary_slowdown": sum(value > 0 for value in paired_steps),
                      "mean_actual_candidate_success_regret": mean([row["actual_candidate_success_regret"] for row in rows])})
    clean, swap = table[:2]
    informative = sum(window["primary_success_distinguishes_candidates"] for window in windows)
    files = {}
    for bundle in bundles:
        files.update(bundle["files_sha256"])
    episode_path, episode_sha = next(iter(episode_identities))
    return {"status": "verified_complete_three_prespecified_window_aggregate",
            "source_episode": episode_path, "source_episode_sha256": episode_sha,
            "source_inference_provenance": provenances[0], "prespecified_prefix_steps": list(PREFIXES),
            "conditions": table, "windows": windows,
            "primary_success_distinguishes_candidates_in_windows": informative,
            "primary_success_tied_candidate_windows": 3 - informative,
            "clean_forecast_beats_persistence_on_mean_rgb_mse": clean["mean_mse_improvement_over_persistence"] > 0,
            "goal_swap_mean_rgb_mse_minus_clean": swap["mean_rgb_mse_0_1"] - clean["mean_rgb_mse_0_1"],
            "primary_success_damage_observed_in_this_panel": swap["windows_losing_clean_success"] > 0,
            "secondary_slowdown_observed_in_this_panel": swap["windows_with_secondary_slowdown"] > 0,
            "utility_estimand": evaluation.UTILITY_ESTIMAND,
            "statistical_scope": {"independent_source_episodes": 1, "dependent_windows_within_episode": 3,
                                  "candidates_per_window": 2, "determinism_replays_per_candidate": 2,
                                  "total_true_replay_executions": 12, "replays_increase_sample_size": False,
                                  "confidence_intervals": None, "p_values": None,
                                  "reason": "Three fixed windows from one selected successful episode are correlated development observations, not independent trials"},
            "interpretation_limits": [
                "Forecast divergence, RGB error and native predicted value are distinct from actual task utility.",
                "All-success candidate ties cannot distinguish action quality using primary success; step differences are secondary only.",
                "Shared N1 continuation estimates local candidate effects; this is not full nativeN2 closed-loop attack performance.",
                "Zero T5 embedding is a diagnostic, not Bus-Cut or a matched Control-Cut.",
                "No mechanism-specific defense/recovery evaluation or externally reachable attacker surface has been established.",
                "Physical replay fixes state/actions, but goal-conditioned partial-observation predictions can legitimately encode hidden-state beliefs.",
                "Clean decoder beating persistence is task-specific forecast usefulness, not proof of universal Goal invariance."],
            "unimplemented_defense_conditions": ["goal_swap_plus_bus_cut", "matched_control_cut", "clean_plus_bus_cut"],
            "defense_recovery_demonstrated": False, "external_attack_reachability_demonstrated": False,
            "verification_by_window": [bundle["verification"] for bundle in bundles],
            "input_files_sha256": dict(sorted(files.items())), "aggregate_script_sha256": replay.file_sha(__file__)}


def aggregate(root=DEFAULT_ROOT, out=None):
    root = Path(root).resolve()
    paths = [root / f"prefix_{prefix:03d}" / "results.json" for prefix in PREFIXES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"All three prescribed results must finish first; missing: {missing}")
    output = Path(out).resolve() if out else root / "aggregate.json"
    if output.exists():
        raise ValueError("Refusing to overwrite prior aggregate")
    cuda_before = torch.cuda.is_initialized()
    bundles = [verify_window(path, prefix) for path, prefix in zip(paths, PREFIXES)]
    result = assemble(bundles)
    if torch.cuda.is_initialized() != cuda_before:
        raise RuntimeError("CPU aggregation unexpectedly initialized CUDA")
    # Detect an input mutation racing with the aggregation before publishing.
    for path, expected in result["input_files_sha256"].items():
        check_file(path, expected)
    output.parent.mkdir(parents=True, exist_ok=True)
    ranking.write_json(output, result)
    return output


def self_test():
    """Arithmetic/denominator/integrity contracts; all values are synthetic."""
    bundles = []
    for prefix, successes in zip(PREFIXES, ((True, True), (True, False), (True, False))):
        swap_index = 0 if prefix == 16 else 1
        candidates = [{"index": i, "success": successes[i], "steps_to_success": 146 - i if successes[i] else None,
                       "actions_sha256": str(i) * 64} for i in range(2)]
        rows = []
        for condition, selected, mse in (("clean", 0, .01), ("goal_swap", swap_index, .02), ("no_lang_zero_embedding", 0, .03)):
            outcome = candidates[selected]
            rows.append({"condition": condition, "selected_index": selected, "predicted_scores": [.6, .4] if selected == 0 else [.4, .6],
                         "ranking_flip_from_clean": selected != 0, "selected_candidate_flip_from_clean": selected != 0,
                         "actual_selected_action_success": outcome["success"], "actual_steps_to_success": outcome["steps_to_success"],
                         "actual_success_loss_from_clean": 1 - int(outcome["success"]),
                         "extra_steps_vs_clean_when_both_succeed": outcome["steps_to_success"] - 146 if outcome["success"] else None,
                         "actual_candidate_success_regret": 1 - int(outcome["success"]),
                         "mean_candidate_rgb_mse_0_1": mse, "persistence_mean_candidate_rgb_mse_0_1": .015,
                         "forecast_mse_improvement_over_persistence": .015 - mse,
                         "forecast_error": [{"candidate_index": i, "error": {"mean_rgb_mse_0_1": mse},
                                             "target_after_native_success_stop": False} for i in range(2)]})
        result = {"prefix_steps": prefix, "source_state_sha256": str(prefix), "results": rows,
                  "actual_candidate_utilities": candidates, "ranking_test_informative_for_primary_success": len(set(successes)) > 1,
                  "utility_estimand": evaluation.UTILITY_ESTIMAND, "forecast_error_available": True}
        bundles.append({"result": result, "prediction": {"source": {"episode_json": "/synthetic/one_episode.json",
                        "episode_sha256": "1" * 64, "inference_provenance": {"synthetic": True}}},
                        "utility": {"repeat": 2, "repeat_passed": True, "candidates": {"0": {}, "1": {}}},
                        "files_sha256": {}, "verification": {"synthetic_not_real": True}})
    result = assemble(bundles)
    swap = result["conditions"][1]
    assert swap["ranking_flip_count"] == 2 and swap["ranking_flip_rate_descriptive"] == 2 / 3
    assert swap["selected_successes"] == 2 and swap["windows_losing_clean_success"] == 1
    assert swap["mean_paired_extra_steps_when_both_succeed"] == -0.5
    assert result["primary_success_distinguishes_candidates_in_windows"] == 2
    assert result["statistical_scope"]["independent_source_episodes"] == 1
    assert result["statistical_scope"]["confidence_intervals"] is None
    for invalid in (bundles[:2], [bundles[0], bundles[0], bundles[2]]):
        try:
            assemble(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Missing/duplicate windows accepted")
    mixed = copy.deepcopy(bundles)
    mixed[2]["prediction"]["source"]["episode_sha256"] = "2" * 64
    try:
        assemble(mixed)
    except ValueError:
        pass
    else:
        raise AssertionError("Different source episodes silently pooled")
    with TemporaryDirectory(prefix="cosmos_aggregate_cpu_") as temporary:
        path = Path(temporary) / "artifact"
        path.write_bytes(b"original")
        expected = replay.file_sha(path)
        path.write_bytes(b"changed")
        try:
            check_file(path, expected)
        except ValueError:
            pass
        else:
            raise AssertionError("Hash mutation accepted")
        try:
            aggregate(temporary)
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete real panel published")
        assert not (Path(temporary) / "aggregate.json").exists()
    return {"status": "passed", "kind": "synthetic_cpu_aggregate_contract_not_experimental_results",
            "checks": ["equal-window RGB means and3-window flip denominator", "actual success/steps and paired differences remain distinct",
                       "primary candidate success distinguishability retained", "missing/duplicate prefixes and mixed episodes rejected",
                       "file hash mutation rejected", "incomplete panel writes no aggregate",
                       "one source episode, no confidence interval, no replay sample-size inflation",
                       "zero embedding is not a defense; no recovery/external reachability claim"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.self_test:
        result = self_test()
        if args.out:
            if args.out.exists():
                raise ValueError("Refusing to overwrite CPU evidence")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            ranking.write_json(args.out, result)
        print(json.dumps(result, indent=2))
    else:
        path = aggregate(args.root, args.out)
        print(json.dumps({"aggregate": str(path)}))


if __name__ == "__main__":
    main()
