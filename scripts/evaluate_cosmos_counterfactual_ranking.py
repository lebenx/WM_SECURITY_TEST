#!/usr/bin/env python3
"""Commit Cosmos ranking predictions, then obtain independent real utilities.

replay: commit completed predictions/candidates before constructing environments,
        replay each candidate twice, continue with fixed original-goal native N1
        until original task success or absolute step300, then join outcomes.
join:   CPU-only join of a prior prediction commit and completed replay artifacts.

The commit is a local ordering/integrity record, not external preregistration.
Repeated environment replay checks determinism; it does NOT increase sample n.
Forecast error uses decoded RGB against the actual t+16 observation transformed
by the released image codec. No current-slot latent is treated as future truth.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

import cosmos_counterfactual_replay as replay
import pilot_cosmos_counterfactual_action_ranking as ranking

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/cosmos_native_decision_gate_v1.json"
UTILITY_ESTIMAND = "Fixed 16-step candidate followed by shared original-Goal native N1 continuation; NOT full-episode native N2 task utility"


def _relative_artifact(base, record):
    path = (base / record["path"]).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError("Prediction artifact must remain inside its ranking directory")
    if replay.file_sha(path) != record["file_sha256"]:
        raise ValueError(f"Prediction artifact changed: {record['path']}")
    return path


def validate_ranking(path):
    """Read only model outputs and their original source prefix, never outcomes."""
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get("status") != "predictions_completed_true_outcomes_pending":
        raise ValueError("Ranking predictions must be completely written before replay")
    if data.get("original_goal") != ranking.ORIGINAL_GOAL or data.get("world_swap_goal") != ranking.SWAP_GOAL:
        raise ValueError("Wrong preregistered original/swap goal")
    if data.get("candidate_seeds") != list(ranking.SEEDS) or data.get("horizon") != 16:
        raise ValueError("Wrong fixed candidates or horizon")
    if data.get("forecast_error") is not None or data.get("selected_action_utility") is not None:
        raise ValueError("Ranking stage already contains outcomes; clean ordering cannot be established")
    if set(data["conditions"]) != set(ranking.CONDITIONS) or len(data["candidates"]) != 2:
        raise ValueError("Expected exactly two candidates and three declared conditions")
    route = data["goal_routing"]
    if route["policy"] != route["value"] or route["policy"] != route["future"]["clean"]:
        raise ValueError("Original Policy/Value goal was not held fixed")
    source = data["source"]
    _, current = ranking.load_snapshot(source["snapshot"])
    episode = json.loads(Path(current["episode_json"]).read_text())
    if episode.get("best_of_n") != 2 or source["prefix_steps"] not in (0, 16, 80):
        raise ValueError("Preregistered source requires successful N2 health and prefix0/16/80; never replace unavailable prefixes")
    for key in ("snapshot_sha256", "episode_sha256", "observation_npz_sha256", "state_sha256", "prefix_steps"):
        if source[key] != current[key]:
            raise ValueError(f"Original source prefix changed: {key}")
    files = {str(path): replay.file_sha(path)}
    def include(record):
        artifact = _relative_artifact(path.parent, record)
        files[str(artifact)] = record["file_sha256"]
        return artifact
    for key in ("observed_latent", "original_vae_latent", "model_input_observations"):
        include(data[key])
    for record in data["embeddings"].values():
        include(record)
    actions = []
    for index, candidate in enumerate(data["candidates"]):
        if candidate["index"] != index or candidate["seed"] != ranking.SEEDS[index]:
            raise ValueError("Candidate identities changed")
        array = ranking.checked_controls(np.load(include(candidate["actions"]), allow_pickle=False))
        if replay.array_sha(array) != candidate["actions"]["array_sha256"]:
            raise ValueError("Candidate controls do not match committed array hash")
        actions.append(array)
        include(candidate["normalized_actions"])
        include(candidate["action_latent"])
    for condition, result in data["conditions"].items():
        expected = ranking.rank_scores(result["scores"])
        if expected["rank"] != result["rank"] or expected["selected_index"] != result["selected_index"]:
            raise ValueError("Saved selection does not follow native argmax/tie rule")
        if result["selected_actions"] != data["candidates"][result["selected_index"]]["actions"]:
            raise ValueError("Selected action reference differs from candidate")
        if len(result["candidates"]) != 2:
            raise ValueError("Missing candidate prediction")
        for index, row in enumerate(result["candidates"]):
            candidate = data["candidates"][index]
            if row["candidate_index"] != index or row["seed"] != candidate["seed"]:
                raise ValueError("Condition changed candidate identity")
            if row["candidate_actions_sha256"] != candidate["actions"]["array_sha256"]:
                raise ValueError("Condition changed physical action candidate")
            if row["predicted_value"] != result["scores"][index]:
                raise ValueError("Saved scores differ from candidate predictions")
            for phase, n_cond, text_sha in (("future_audit", 5, route["future"][condition]),
                                             ("value_audit", 8, route["value"])):
                audit = row[phase]
                if audit["conditioning_prefix_slots"] != n_cond or audit["text_sha256"] != text_sha:
                    raise ValueError("Forecast/value conditioning or goal routing changed")
                if audit["rng"] != candidate["audit"]["rng"]:
                    raise ValueError("Candidate sampler noise differs across stages/conditions")
            include(row["future_latent"])
            include(row["value_latent"])
            if "decoded_future_images" in row:
                if set(row["decoded_future_images"]) != {"wrist", "primary"}:
                    raise ValueError("Expected both decoded future views")
                for image in row["decoded_future_images"].values():
                    include(image)
    for key in ("snapshot", "episode_json", "observation_npz"):
        files[str(Path(source[key]).resolve())] = replay.file_sha(source[key])
    return data, actions, files


def commit_predictions(ranking_json, path, *, preprocess=None):
    path = Path(path).resolve()
    if path.exists():
        raise ValueError("Refusing to replace an existing prediction commit")
    data, _, files = validate_ranking(ranking_json)
    if preprocess is None:
        import cosmos_policy_native_inference as native
        preprocess = native.preprocess_observation
    # Current observations are permissible pre-outcome inputs. Freeze the
    # prediction of no visual change on the same exact codec/scale as Cosmos.
    with np.load(data["source"]["observation_npz"], allow_pickle=False) as archive:
        current_obs = {key: archive[key].copy() for key in replay.OBS_KEYS}
    images = preprocess(current_obs)
    persistence = {}
    directory = path.parent / "persistence_inputs"
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Refusing to overwrite persistence predictions")
    directory.mkdir(parents=True, exist_ok=True)
    for index, view in enumerate(("wrist", "primary")):
        value = np.asarray(images[index])
        if value.shape != (224, 224, 3) or value.dtype != np.uint8:
            raise ValueError("Malformed current-observation codec result")
        image_path = directory / f"{view}.npy"
        np.save(image_path, value[None], allow_pickle=False)
        persistence[view] = {"path": str(image_path), "file_sha256": replay.file_sha(image_path),
                             "array_sha256": replay.array_sha(value[None])}
        files[str(image_path)] = replay.file_sha(image_path)
    for code in (SPEC, Path(__file__), Path(replay.__file__), ROOT / "scripts/cosmos_policy_native_inference.py",
                 ROOT / "scripts/cosmos_policy_codecs.py"):
        files[str(code.resolve())] = replay.file_sha(code)
    commit = {"status": "predictions_committed_before_independent_replay",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "ranking_json": str(Path(ranking_json).resolve()),
              "ranking_sha256": replay.file_sha(ranking_json),
              "files_sha256": dict(sorted(files.items())),
              "source_state_sha256": data["source"]["state_sha256"],
              "candidate_actions_sha256": [row["actions"]["array_sha256"] for row in data["candidates"]],
              "original_goal": ranking.ORIGINAL_GOAL,
              "continuation_protocol": "native N1 original-goal policy, seed31 for every call, action5, no intervention, continuous controls, stop at success or absolute step300",
              "utility_estimand": UTILITY_ESTIMAND,
              "preregistered_protocol": str(SPEC), "preregistered_protocol_sha256": replay.file_sha(SPEC),
              "persistence_predictions": persistence,
              "forecast_metric_prespecification": {
                  "prediction": "native decoded wrist/primary t+16 uint8 RGB",
                  "true_target": "independent original-state candidate replay observation at exactly t+16, official preprocessing",
                  "metric": "mean(((pred_uint8.astype(float64)-target_uint8.astype(float64))/255)**2)",
                  "aggregation": "equal average across both views; equal average across the two fixed candidates",
                  "persistence": "same current source observation after identical official preprocessing; frozen before replay",
                  "clean_usefulness": "persistence RGB MSE minus Clean forecast RGB MSE; positive means improvement on this window only",
                  "post_success_targets": "reported separately; candidate always runs full16, but mark targets after native first-success stop",
                  "no_latent_truth_approximation": True},
              "independent_replay_repeat": 2,
              "evaluation_script_sha256": replay.file_sha(__file__),
              "replay_script_sha256": replay.file_sha(replay.__file__),
              "ordering_scope": "Local program order and hashes; not external timestamping or adversarial tamper-proof registration"}
    path.parent.mkdir(parents=True, exist_ok=True)
    ranking.write_json(path, commit)
    return path


def verify_commit(path):
    path = Path(path).resolve()
    commit = json.loads(path.read_text())
    if commit.get("status") != "predictions_committed_before_independent_replay":
        raise ValueError("Not a completed prediction commit")
    for name, expected in commit["files_sha256"].items():
        if replay.file_sha(name) != expected:
            raise ValueError(f"Committed artifact changed: {name}")
    if replay.file_sha(commit["ranking_json"]) != commit["ranking_sha256"]:
        raise ValueError("Ranking JSON changed after commit")
    data = json.loads(Path(commit["ranking_json"]).read_text())
    return commit, data


class OriginalGoalContinuation:
    """N1 policy sees current observations and original goal only, never scores."""
    def __init__(self, backend, text, log_path):
        self.backend, self.text = backend, text.clone()
        self.log_path = Path(log_path)
        self.repetition = -1
        self.text_sha256 = ranking.tensor_sha(self.text)

    def reset(self):
        # Diagnostic counter only; no candidate-dependent policy state exists.
        self.repetition += 1

    @torch.inference_mode()
    def __call__(self, obs, *, original_goal, absolute_step, remaining_steps, seed):
        if original_goal != ranking.ORIGINAL_GOAL or type(seed) is not int or seed != 31:
            raise ValueError("Continuation goal/seed must remain original")
        if type(absolute_step) is not int or not 16 <= absolute_step < 300 or remaining_steps != 300 - absolute_step:
            raise ValueError("Continuation must fit the original absolute300 budget")
        if ranking.tensor_sha(self.text) != self.text_sha256:
            raise RuntimeError("Continuation goal embedding changed")
        observed, _ = self.backend.encode({key: np.asarray(obs[key]).copy() for key in replay.OBS_KEYS})
        (normalized, _), audit = ranking.checked_query(self.backend, "proposal", observed, self.text, seed, 4)
        controls = ranking.checked_controls(self.backend.controls(normalized))
        selected = controls[:remaining_steps].copy()
        record = {"replay_counter_for_logging_only": self.repetition,
                  "absolute_step": absolute_step, "remaining_steps": remaining_steps,
                  "original_goal": original_goal, "seed": seed,
                  "observations_sha256": {key: replay.array_sha(obs[key]) for key in replay.OBS_KEYS},
                  "policy_audit": audit, "proposed_actions_sha256": replay.array_sha(controls),
                  "returned_controls_sha256": replay.array_sha(selected), "returned_count": len(selected)}
        with self.log_path.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        return selected


def _validate_utilities(commit_path, commit, prediction, utility_path):
    utility_path = Path(utility_path).resolve()
    utility = json.loads(utility_path.read_text())
    if utility.get("status") != "completed_real_environment_candidate_replays" or utility.get("repeat_passed") is not True:
        raise ValueError("Independent replay did not complete exact-repeat validation")
    if utility.get("full_task_utility_complete") is not True or utility.get("continuation_enabled") is not True:
        raise ValueError("Full-horizon original-goal continuation is required for this evaluation")
    if utility.get("repeat") != 2 or utility.get("candidate_target_horizon") != 16:
        raise ValueError("Expected two repeats and exact16-step target")
    if utility.get("continuation_metadata", {}).get("prediction_commit_sha256") != replay.file_sha(commit_path):
        raise ValueError("Replay is not bound to this prior prediction commit")
    source = prediction["source"]
    for left, right in (("episode_sha256", "episode_sha256"), ("prefix_steps", "prefix_steps"),
                        ("source_snapshot_sha256", "snapshot_sha256"), ("source_state_hash", "state_sha256")):
        if utility.get(left) != source[right]:
            raise ValueError(f"Replay source mismatch: {left}")
    if utility.get("goal") != ranking.ORIGINAL_GOAL or utility.get("max_steps") != 300:
        raise ValueError("Replay changed actual original environment objective/budget")
    if set(utility["candidates"]) != {"0", "1"}:
        raise ValueError("Wrong replay candidate set")
    for index in range(2):
        row = utility["candidates"][str(index)]
        if row.get("action_sha256") != commit["candidate_actions_sha256"][index]:
            raise ValueError("Replay controls differ from committed candidate")
        if replay.file_sha(row["actions"]) != row["actions_file_sha256"]:
            raise ValueError("Replay action artifact changed")
        actions = np.load(row["actions"], allow_pickle=False)
        if replay.array_sha(actions) != row["action_sha256"]:
            raise ValueError("Replay controls have wrong array hash")
        if row.get("repeat_initial_future_terminal_trajectory_exact") is not True or len(row["replays"]) != 2:
            raise ValueError("Missing exact physical replay replication")
        if type(row.get("success")) is not bool or row.get("utility_observed_to_success_or_full_horizon") is not True:
            raise ValueError("Task utility is unknown or non-boolean")
        for repetition, sample in enumerate(row["replays"]):
            if sample["initial"]["state_hash"] != source["state_sha256"]:
                raise ValueError("Replay initial physical state changed")
            if sample["candidate_steps_executed"] != 16 or sample["success"] is not row["success"]:
                raise ValueError("Replay success or candidate horizon is inconsistent")
            if sample["steps_to_success"] != row["steps_to_success"]:
                raise ValueError("Repeated success timing differs")
            if row["success"]:
                step = row["steps_to_success"]
                if type(step) is not int or not source["prefix_steps"] < step <= 300:
                    raise ValueError("Invalid actual first-success step")
            elif row["steps_to_success"] is not None or sample["last_executed_absolute_step"] != 300:
                raise ValueError("Failure requires actual execution to step300")
            if replay.file_sha(sample["trajectory"]) != sample["trajectory_sha256"]:
                raise ValueError("Independent physical trajectory changed")
            for phase in ("initial", "candidate_future", "terminal"):
                artifact = sample[phase]
                if replay.file_sha(artifact["observations_npz"]) != artifact["observations_npz_sha256"]:
                    raise ValueError("Independent observation target changed")
                if artifact["state_hash"] != row["replays"][0][phase]["state_hash"]:
                    raise ValueError("Repeated physical state differs")
            if sample["trajectory_outcomes_sha256"] != row["replays"][0]["trajectory_outcomes_sha256"]:
                raise ValueError("Repeated action/outcome trajectory differs")
    return utility


def _image_error(record, predicted_base, actual_obs, preprocess):
    """Actual observations are independent targets, not used in selection."""
    if "decoded_future_images" not in record:
        return None
    transformed = preprocess(actual_obs)
    errors = {}
    for index, view in enumerate(("wrist", "primary")):
        image = record["decoded_future_images"][view]
        expected = np.load(_relative_artifact(predicted_base, image), allow_pickle=False)
        if expected.shape != (1, 224, 224, 3) or expected.dtype != np.uint8:
            raise ValueError("Malformed decoded native RGB prediction")
        target = np.asarray(transformed[index])
        if target.shape != (224, 224, 3) or target.dtype != np.uint8:
            raise ValueError("Malformed independently transformed true target")
        error = (expected[0].astype(np.float64) - target.astype(np.float64)) / 255.0
        errors[view] = {"rgb_mse_0_1": float(np.mean(error ** 2)), "rgb_mae_0_1": float(np.mean(np.abs(error))),
                        "predicted_image_sha256": replay.array_sha(expected), "transformed_actual_image_sha256": replay.array_sha(target)}
    return {"views": errors, "mean_rgb_mse_0_1": float(np.mean([row["rgb_mse_0_1"] for row in errors.values()])),
            "definition": "Decoded native t+16 RGB vs independent environment t+16 after official JPEG95/vertical/resize/crop preprocessing; no latent target approximation"}


def join_results(commit_path, utility_path, output_path, *, preprocess=None):
    # Verify sealed predictions BEFORE opening any future outcome artifact.
    commit_path = Path(commit_path).resolve()
    commit, prediction = verify_commit(commit_path)
    utility = _validate_utilities(commit_path, commit, prediction, utility_path)
    if preprocess is None:
        import cosmos_policy_native_inference as native
        preprocess = native.preprocess_observation
    target_obs = {}
    for index in range(2):
        target = utility["candidates"][str(index)]["replays"][0]["candidate_future"]
        with np.load(target["observations_npz"], allow_pickle=False) as archive:
            target_obs[index] = {key: archive[key].copy() for key in replay.OBS_KEYS}
    persistence = []
    for index in range(2):
        transformed = preprocess(target_obs[index])
        view_errors = {}
        for view_index, view in enumerate(("wrist", "primary")):
            value = np.load(commit["persistence_predictions"][view]["path"], allow_pickle=False)
            target = np.asarray(transformed[view_index])
            if value.shape != (1, 224, 224, 3) or target.shape != (224, 224, 3) or target.dtype != np.uint8:
                raise ValueError("Malformed persistence/true-target RGB")
            error = (value[0].astype(np.float64) - target.astype(np.float64)) / 255.0
            view_errors[view] = {"rgb_mse_0_1": float(np.mean(error ** 2)),
                                 "rgb_mae_0_1": float(np.mean(np.abs(error)))}
        persistence.append({"candidate_index": index, "views": view_errors,
                            "mean_rgb_mse_0_1": float(np.mean([value["rgb_mse_0_1"] for value in view_errors.values()]))})
    persistence_mean = float(np.mean([value["mean_rgb_mse_0_1"] for value in persistence]))
    clean_selected = prediction["conditions"]["clean"]["selected_index"]
    clean_outcome = utility["candidates"][str(clean_selected)]
    rows = []
    for condition in ranking.CONDITIONS:
        selection = prediction["conditions"][condition]
        selected = selection["selected_index"]
        outcome = utility["candidates"][str(selected)]
        errors = []
        for index, candidate in enumerate(selection["candidates"]):
            error = _image_error(candidate, Path(commit["ranking_json"]).parent, target_obs[index], preprocess)
            errors.append({"candidate_index": index, "error": error,
                           "target_after_native_success_stop": utility["candidates"][str(index)]["replays"][0]["candidate_future_after_native_success_stop"]})
        mse = [entry["error"]["mean_rgb_mse_0_1"] for entry in errors if entry["error"] is not None]
        step_difference = (outcome["steps_to_success"] - clean_outcome["steps_to_success"]
                           if outcome["success"] and clean_outcome["success"] else None)
        rows.append({"condition": condition, "selected_index": selected, "predicted_scores": selection["scores"],
                     "ranking_flip_from_clean": selection["rank"] != prediction["conditions"]["clean"]["rank"],
                     "selected_candidate_flip_from_clean": selected != clean_selected,
                     "actual_selected_action_success": outcome["success"],
                     "selected_action_utility": int(outcome["success"]),
                     "actual_steps_to_success": outcome["steps_to_success"],
                     "actual_success_loss_from_clean": int(clean_outcome["success"]) - int(outcome["success"]),
                     "extra_steps_vs_clean_when_both_succeed": step_difference,
                     "actual_candidate_success_regret": max(int(row["success"]) for row in utility["candidates"].values()) - int(outcome["success"]),
                     "forecast_error": errors, "mean_candidate_rgb_mse_0_1": float(np.mean(mse)) if len(mse) == 2 else None,
                     "persistence_mean_candidate_rgb_mse_0_1": persistence_mean,
                     "forecast_mse_improvement_over_persistence": persistence_mean - float(np.mean(mse)) if len(mse) == 2 else None,
                     "bus_cut_decision_recovery": None})
    swap = next(row for row in rows if row["condition"] == "goal_swap")
    result = {"status": "joined_committed_predictions_with_independent_real_environment_outcomes",
              "prediction_commit": str(commit_path), "prediction_commit_sha256": replay.file_sha(commit_path),
              "utilities": str(Path(utility_path).resolve()), "utilities_sha256": replay.file_sha(utility_path),
              "model": "NVIDIA Cosmos Policy LIBERO 40K torch adapter", "suite": "libero_goal",
              "task_id": 0, "init_id": 0, "seed": 31, "prefix_steps": prediction["source"]["prefix_steps"],
              "source_state_sha256": prediction["source"]["state_sha256"], "results": rows,
              "actual_candidate_utilities": [{"index": index, "success": utility["candidates"][str(index)]["success"],
                                               "steps_to_success": utility["candidates"][str(index)]["steps_to_success"],
                                               "actions_sha256": commit["candidate_actions_sha256"][index]} for index in range(2)],
              "primary_success_damage_observed_in_this_window": swap["actual_success_loss_from_clean"] > 0,
              "ranking_test_informative_for_primary_success": utility["primary_success_distinguishes_candidates"],
              "independent_windows": 1, "determinism_replays_per_candidate": 2,
              "statistical_scope": "One fixed state and two candidate actions; repeats are deterministic checks, not extra samples; no significance or general attack-success claim",
              "threat_scope": "Internal world-call context intervention; external reachability remains unproven",
              "invariance_scope": "Same simulator/controller state and complete action chunk define a common real target; a goal-conditioned partial-observation forecast can still update hidden-state beliefs",
              "utility_definition": "Actual original-task success by absolute step300 after the fixed candidate and identical original-goal native N1 continuation; not model-predicted value",
              "utility_estimand": UTILITY_ESTIMAND,
              "full_episode_native_n2_attack_evaluated": False,
              "persistence_baseline": {"candidate_errors": persistence, "mean_candidate_rgb_mse_0_1": persistence_mean,
                                       "learned_model_used": False},
              "forecast_error_available": all(row["mean_candidate_rgb_mse_0_1"] is not None for row in rows),
              "unimplemented_conditions": ["goal_swap_plus_bus_cut", "matched_control_cut"],
              "causal_bus_recovery_demonstrated": False, "external_attack_demonstrated": False,
              "evaluation_script_sha256": replay.file_sha(__file__)}
    verify_commit(commit_path)
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise ValueError("Refusing to overwrite joined results")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranking.write_json(output_path, result)
    return output_path


def execute_replay(ranking_json, out_dir, *, device="cuda", offload=False,
                   backend_factory=None, replay_function=None, preprocess=None):
    directory = Path(out_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use a fresh evaluation output directory")
    directory.mkdir(parents=True, exist_ok=True)
    commit_path = commit_predictions(ranking_json, directory / "predictions_commit.json", preprocess=preprocess)
    commit, prediction = verify_commit(commit_path)
    status_path = directory / "evaluation_status.json"
    status = {"status": "committed_before_environment_construction", "prediction_commit": str(commit_path)}
    ranking.write_json(status_path, status)
    try:
        # No simulator or true future observations are loaded before the commit.
        if backend_factory is None:
            provenance = prediction["source"]["inference_provenance"]
            for relative, expected in provenance["source_sha256"].items():
                if replay.file_sha(ROOT / relative) != expected:
                    raise ValueError(f"Continuation runtime differs from the committed proposal runtime: {relative}")
            import cosmos_policy_native_inference as native
            native.verify_artifact(native.MODEL.parent / "libero_dataset_statistics.json", ranking.STATS_SHA256)
            backend = ranking.NativeBackend(device=device, offload=offload)
            text = ranking.embeddings()["clean"]
        else:
            backend, text = backend_factory()
        if ranking.tensor_sha(text) != prediction["goal_routing"]["policy"]:
            raise ValueError("Continuation text differs from committed original-goal policy")
        continuation = OriginalGoalContinuation(backend, text, directory / "continuation_calls.jsonl")
        metadata = {"policy": "native N1 original goal", "original_goal": ranking.ORIGINAL_GOAL,
                    "utility_estimand": UTILITY_ESTIMAND,
                    "checkpoint_sha256": ranking.MODEL_SHA256, "seed": 31, "seed_rule": "same seed31 every query",
                    "action_steps": 5, "image_codec": "official raw256 vertical JPEG95 resize224 90percent center-crop",
                    "action_codec": "official unnormalization, no clipping/sign inversion/gripper binarization",
                    "backend": backend.metadata, "policy_goal_text_sha256": ranking.tensor_sha(text),
                    "prediction_commit": str(commit_path), "prediction_commit_sha256": replay.file_sha(commit_path),
                    "no_ranking_scores_or_condition_label_passed_to_continuation": True}
        source = prediction["source"]
        controls = [np.load(Path(commit["ranking_json"]).parent / row["actions"]["path"], allow_pickle=False)
                    for row in prediction["candidates"]]
        verify_commit(commit_path)
        status["status"] = "running_independent_replays_with_original_goal_continuation"
        ranking.write_json(status_path, status)
        utility_path = (replay_function or replay.replay_candidates)(source["episode_json"], source["prefix_steps"], controls,
            directory / "replays", repeat=2, continuation=continuation, continuation_metadata=metadata,
            source_snapshot=source["snapshot"])
        result = join_results(commit_path, utility_path, directory / "results.json", preprocess=preprocess)
        status.update(status="completed", results=str(result), utilities=str(utility_path))
        return result
    except BaseException as error:
        status.update(status="error", exception_type=type(error).__name__, exception=str(error))
        raise
    finally:
        ranking.write_json(status_path, status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("replay")
    run.add_argument("--ranking", type=Path, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run.add_argument("--offload", action="store_true")
    join = sub.add_parser("join")
    join.add_argument("--commit", type=Path, required=True)
    join.add_argument("--utilities", type=Path, required=True)
    join.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "replay":
        if args.offload and args.device != "cuda":
            parser.error("--offload requires --device cuda")
        path = execute_replay(args.ranking, args.out_dir, device=args.device, offload=args.offload)
    else:
        path = join_results(args.commit, args.utilities, args.out)
    print(json.dumps({"results": str(path)}))


if __name__ == "__main__":
    main()
