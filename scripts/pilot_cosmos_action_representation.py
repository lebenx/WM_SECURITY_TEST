#!/usr/bin/env python3
"""Exact action-representation gate and fixed-state Cosmos mechanism probe.

No new action proposals, simulator runs, training, or API calls. CUDA codec gate
does not load model weights. The probe consumes committed gate inputs and writes
all predictions before a separate join reads existing real replay outcomes.
These previously studied states/outcomes are development data, not a blind test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

import audit_cosmos_action_representation as projection
import cosmos_counterfactual_replay as replay
import evaluate_cosmos_counterfactual_ranking as evaluation
import pilot_cosmos_counterfactual_action_ranking as ranking

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = (0, 16, 80)
ARMS = ("clean_raw", "canonical_tile", "reverse_copies")
SOURCE_ROOT = ROOT / "runs/cosmos_counterfactual_ranking_v1"
EVALUATION_ROOT = ROOT / "runs/cosmos_counterfactual_evaluation_v1"
SPEC = ROOT / "configs/cosmos_action_representation_probe_v1.json"
LIMITS = [
    "Internal representation intervention requiring write access; no external attack demonstrated.",
    "No Goal changes: this does not establish the Goal-Bus hypothesis.",
    "Same executed actions do not prove that official forecasts promised to ignore internal latent residuals.",
    "CPU equivalence does not certify CUDA equivalence; deployment-device gate is mandatory.",
    "Predicted values and ranking flips are not actual utility or attack success.",
    "Native Value directly sees action slot4 as well as forecast slots5:8; native ranking effects need not be forecast-mediated.",
    "Fixed-seed paired sample differences do not establish a difference between output distributions.",
    "Native proposal/future/value reuse candidate seed31 or32; residual representations may couple to the shared model-sampling index. Independent forecast/value seeds are a future follow-up, not part of this registered probe.",
    "Projected canonical tensors have a small final-copy correction; they are not exactly tiled actions.",
    "Three correlated prefixes from one successful trajectory, two candidates each; no population inference.",
    "Existing outcomes were available during design. Reuse is developmental, not a new blinded or held-out experiment.",
    "Recorded utility is fixed candidate plus original-goal native N1 continuation, not full-episode native N2 utility.",
]


def fresh(directory):
    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use a fresh output directory")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def sealed(path, status, files, **extra):
    if Path(path).exists():
        raise ValueError("Refusing to overwrite a sealed manifest")
    result = {"status": status, "created_utc": datetime.now(timezone.utc).isoformat(),
              "files_sha256": dict(sorted(files.items())), **extra}
    ranking.write_json(Path(path), result)
    return result


def verify_seal(path, status):
    value = json.loads(Path(path).read_text())
    if value.get("status") != status:
        raise ValueError("Wrong manifest stage")
    for name, expected in value["files_sha256"].items():
        if replay.file_sha(name) != expected:
            raise ValueError(f"Sealed artifact changed: {name}")
    return value


def tensor_record(base, record, device="cpu"):
    path = evaluation._relative_artifact(Path(base), record)
    value = torch.load(path, map_location=device, weights_only=True)
    if ranking.tensor_sha(value) != record["tensor_sha256"]:
        raise ValueError("Tensor content differs from committed hash")
    return value


def validate_sources(source_root):
    windows, files = [], {}
    for prefix in PREFIXES:
        path = Path(source_root).resolve() / f"prefix_{prefix:03d}/ranking.json"
        data, actions, hashes = evaluation.validate_ranking(path)
        if data["source"]["prefix_steps"] != prefix:
            raise ValueError("Preselected window order changed")
        provenance = data["source"]["inference_provenance"]
        for name, expected in provenance["source_sha256"].items():
            if replay.file_sha(ROOT / name) != expected:
                raise ValueError(f"Native runtime changed since proposal: {name}")
        files.update(hashes)
        windows.append({"ranking_json": str(path), "ranking_sha256": replay.file_sha(path),
                        "source": data["source"], "data": data, "actions": actions})
    for path in (SPEC, Path(__file__), Path(projection.__file__), Path(ranking.__file__),
                 Path(evaluation.__file__), ROOT / "scripts/cosmos_policy_native_inference.py",
                 ROOT / "scripts/cosmos_policy_codecs.py"):
        files[str(path.resolve())] = replay.file_sha(path)
    return windows, files


def exact_candidate_gate(raw, saved_normalized, saved_controls, native, stats):
    """Original and alias checks use original dtype plus byte hashes, no tolerance."""
    original_hash = ranking.tensor_sha(raw)
    decoded = native.action_from_latent(raw)
    physical = native.normalize_action(decoded, stats)[0]
    exact_normalized = ranking.tensor_sha(decoded) == ranking.tensor_sha(saved_normalized)
    exact_controls = replay.array_sha(physical) == replay.array_sha(saved_controls)
    report = {"status": "passed" if exact_normalized and exact_controls else "rejected_original_decode_mismatch",
              "original_normalized_exact": exact_normalized, "original_controls_exact": exact_controls,
              "normalized_max_abs_delta": float((decoded - saved_normalized.to(decoded)).abs().max()),
              "physical_max_abs_delta": float(np.max(np.abs(physical - saved_controls))),
              "normalized_sha256": ranking.tensor_sha(decoded),
              "controls_sha256": replay.array_sha(physical), "arms": {}}
    tensors = {}
    if not exact_normalized or not exact_controls:
        for arm in ARMS:
            report["arms"][arm] = {"status": "skip_model_original_decode_mismatch"}
        return report, tensors
    tensors["clean_raw"] = raw.clone()
    report["arms"]["clean_raw"] = {"status": "passed", "normalized_exact": True,
                                    "controls_exact": True, "other_slots_exact": True}
    for arm in ARMS[1:]:
        try:
            changed, detail = projection.project_action_slot(raw, arm)
        except projection.ExactProjectionError as error:
            report["arms"][arm] = {"status": "skip_model_projection_failed", "projection": error.report}
            continue
        action = native.action_from_latent(changed)
        controls = native.normalize_action(action, stats)[0]
        exact_action = ranking.tensor_sha(action) == ranking.tensor_sha(saved_normalized)
        exact_physical = replay.array_sha(controls) == replay.array_sha(saved_controls)
        other_equal = all(ranking.tensor_sha(raw[:, :, slot]) == ranking.tensor_sha(changed[:, :, slot])
                          for slot in range(9) if slot != 4)
        eligible = exact_action and exact_physical and other_equal
        report["arms"][arm] = {"status": "passed" if eligible else "skip_model_exact_gate_failed",
            "normalized_exact": exact_action, "controls_exact": exact_physical,
            "other_slots_exact": other_equal, "representation_changed": ranking.tensor_sha(raw) != ranking.tensor_sha(changed),
            "model_bf16_action_slot_changed": ranking.tensor_sha(raw[:, :, 4].to(torch.bfloat16)) != ranking.tensor_sha(changed[:, :, 4].to(torch.bfloat16)),
            "projection": detail}
        if eligible:
            tensors[arm] = changed
    if ranking.tensor_sha(raw) != original_hash:
        raise RuntimeError("Projection mutated committed original latent")
    return report, tensors


def codec_gate(source_root, out_dir, device, evaluation_root=EVALUATION_ROOT):
    directory = fresh(out_dir)
    windows, files = validate_sources(source_root)
    import cosmos_policy_native_inference as native
    stats_path = native.MODEL.parent / "libero_dataset_statistics.json"
    native.verify_artifact(stats_path, ranking.STATS_SHA256)
    files[str(stats_path)] = replay.file_sha(stats_path)
    # Freeze bytes only, without interpreting target observations or utility values.
    # These development outcomes already exist; no claim of new blind data is made.
    references = {str((Path(evaluation_root).resolve() / f"prefix_{prefix:03d}" / relative)):
                  replay.file_sha(Path(evaluation_root).resolve() / f"prefix_{prefix:03d}" / relative)
                  for prefix in PREFIXES for relative in ("predictions_commit.json", "results.json", "replays/utilities.json")}
    sealed(directory / "source_inputs_commit.json", "source_inputs_committed_before_codec_gate", files,
           prefix_steps=list(PREFIXES), arms=list(ARMS), device=device,
           projection_bounds={"max_correction_steps": 32, "neighbor_steps": 512},
           protocol=str(SPEC), protocol_sha256=replay.file_sha(SPEC),
           existing_outcome_reference_sha256=references,
           outcome_values_not_parsed_or_used=True, limitations=LIMITS)
    status = {"status": "running_codec_gate", "device": device, "windows": [],
              "model_weights_loaded": False, "environment_created": False,
              "source_inputs_commit": str(directory / "source_inputs_commit.json"), "limitations": LIMITS}
    path = directory / "codec_gate.json"
    ranking.write_json(path, status)
    try:
        stats = native.load_stats()
        status["torch_version"] = torch.__version__
        status["device_name"] = torch.cuda.get_device_name(device) if device.startswith("cuda") else "CPU"
        for window in windows:
            data, base = window["data"], Path(window["ranking_json"]).parent
            prefix = data["source"]["prefix_steps"]
            item = {key: window[key] for key in ("ranking_json", "ranking_sha256", "source")}
            item["candidates"] = []
            for index, candidate in enumerate(data["candidates"]):
                raw = tensor_record(base, candidate["action_latent"], device)
                normalized = tensor_record(base, candidate["normalized_actions"], device)
                report, tensors = exact_candidate_gate(raw, normalized, window["actions"][index], native, stats)
                report.update(index=index, seed=candidate["seed"],
                              committed_controls_sha256=candidate["actions"]["array_sha256"],
                              committed_normalized_sha256=candidate["normalized_actions"]["tensor_sha256"],
                              committed_raw_latent_sha256=candidate["action_latent"]["tensor_sha256"])
                for arm, tensor in tensors.items():
                    record = ranking.save_tensor(directory, f"prefix_{prefix:03d}/{arm}/candidate_{index:02d}_input.pt", tensor)
                    report["arms"][arm]["latent"] = record
                    files[str((directory / record["path"]).resolve())] = record["file_sha256"]
                item["candidates"].append(report)
                print(json.dumps({"phase": "codec_gate", "prefix_steps": prefix, "candidate": index,
                                  "arms": {key: value["status"] for key, value in report["arms"].items()}}), flush=True)
                del raw, normalized, tensors
            status["windows"].append(item)
            ranking.write_json(path, status)
        status["passed_per_arm"] = {arm: sum(candidate["arms"][arm]["status"] == "passed"
            for window in status["windows"] for candidate in window["candidates"]) for arm in ARMS}
        status["status"] = "codec_gate_completed_with_explicit_rejections" if any(value != 6 for value in status["passed_per_arm"].values()) else "codec_gate_all_passed"
        ranking.write_json(path, status)
        files[str(path)] = replay.file_sha(path)
        files[str(directory / "source_inputs_commit.json")] = replay.file_sha(directory / "source_inputs_commit.json")
        sealed(directory / "gate_commit.json", "codec_gate_inputs_sealed", files,
               codec_gate=str(path), codec_gate_sha256=replay.file_sha(path),
               device=device, windows=len(windows), eligible_per_arm=status["passed_per_arm"],
               protocol=str(SPEC), protocol_sha256=replay.file_sha(SPEC),
               existing_outcome_reference_sha256=references,
               outcome_values_not_parsed_or_used=True, limitations=LIMITS)
    except BaseException as error:
        status.update(status="codec_gate_error", exception_type=type(error).__name__, exception=str(error))
        ranking.write_json(path, status)
        raise
    return directory / "gate_commit.json"


@torch.inference_mode()
def predict_one(backend, source, text, original, seed, expected_rng, directory, relative,
                raw_action_slot, clean_future=None):
    future, future_audit = ranking.checked_query(backend, "future", source, text, seed, 5)
    (value, value_latent), value_audit = ranking.checked_query(backend, "value", future, text, seed, 8)
    if future_audit["rng"] != expected_rng or value_audit["rng"] != expected_rng:
        raise RuntimeError("Proposal/future/value initial noise differs")
    score = float(value.item())
    if value.numel() != 1 or not np.isfinite(score) or not 0 <= score <= 1:
        raise RuntimeError("Native score is not one probability")
    row = {"status": "predicted", "seed": seed, "predicted_value": score,
           "future_audit": future_audit, "value_audit": value_audit,
           "future_latent": ranking.save_tensor(directory, relative + "_future.pt", future),
           "value_latent": ranking.save_tensor(directory, relative + "_value.pt", value_latent),
           "decoded_future_images": {key: ranking.save_array(directory, relative + f"_{key}.npy", pixels)
                                     for key, pixels in backend.decode(future, original).items()}}
    if clean_future is None:
        clean_future = future
    if not torch.equal(source[:, :, :4], clean_future[:, :, :4]):
        raise RuntimeError("Clean and alias observed slots differ")
    isolated = future.clone()
    isolated[:, :, 4] = raw_action_slot
    direct = clean_future.clone()
    direct[:, :, 4] = source[:, :, 4]
    for name, controlled_input, operation in (
        ("forecast_only_value_control", isolated, "rawA + aliasF: restore committed original raw action slot4, retain alias future slots5:8"),
        ("value_direct_only_control", direct, "aliasA + cleanF: use alias action slot4 and matched Clean future slots5:8"),
    ):
        if not torch.equal(controlled_input[:, :, :4], source[:, :, :4]):
            raise RuntimeError("Value control changed observation conditioning")
        if torch.equal(controlled_input[:, :, :8], future[:, :, :8]):
            controlled_score, controlled_audit, controlled_latent = score, value_audit, value_latent
            # Slot8 is unconditioned and discarded by the sampler. Record the
            # exact full input of the call being reused, not a fictitious call.
            controlled_input = future
            reused = True
        else:
            (estimate, controlled_latent), controlled_audit = ranking.checked_query(backend, "value", controlled_input, text, seed, 8)
            controlled_score = float(estimate.item())
            reused = False
        if controlled_audit["rng"] != expected_rng:
            raise RuntimeError("Value control RNG differs")
        if not np.isfinite(controlled_score) or not 0 <= controlled_score <= 1:
            raise RuntimeError("Bad value control score")
        row[name] = {"predicted_value": controlled_score, "audit": controlled_audit,
            "value_input": ranking.save_tensor(directory, relative + f"_{name}_input.pt", controlled_input),
            "value_latent": ranking.save_tensor(directory, relative + f"_{name}_output.pt", controlled_latent),
            "reuses_identical_native_conditioning": reused,
            "operation": operation + "; original Goal and RNG. Mechanism ablation, not native execution."}
    return row


def output_files(directory):
    return {str(path.resolve()): replay.file_sha(path) for path in Path(directory).rglob("*")
            if path.is_file() and path.name != "predictions_commit.json"}


def complete_ranking(rows, control=None):
    if len(rows) != 2 or [row["index"] for row in rows] != [0, 1] or any(row["status"] != "predicted" for row in rows):
        return None
    return ranking.rank_scores([row[control]["predicted_value"] if control else row["predicted_value"] for row in rows])


def probe(gate_path, out_dir, *, device="cuda", offload=False):
    gate_path = Path(gate_path).resolve()
    commit = verify_seal(gate_path, "codec_gate_inputs_sealed")
    gate = json.loads(Path(commit["codec_gate"]).read_text())
    if not gate["device"].startswith("cuda") or not device.startswith("cuda"):
        raise ValueError("Real model probe requires a CUDA codec gate and CUDA execution")
    directory = fresh(out_dir)
    sealed(directory / "probe_inputs_commit.json", "probe_inputs_committed_before_model_load",
           {**commit["files_sha256"], str(gate_path): replay.file_sha(gate_path)},
           gate_commit=str(gate_path), arms=list(ARMS), original_goal=ranking.ORIGINAL_GOAL,
           protocol=str(SPEC), protocol_sha256=replay.file_sha(SPEC),
           value_path_controls="full raw/alias action x clean/alias future 2x2",
           outcome_values_not_parsed_or_used=True, limitations=LIMITS)
    report = {"status": "loading_backend", "gate_commit": str(gate_path),
              "gate_commit_sha256": replay.file_sha(gate_path), "windows": [], "arms": list(ARMS),
              "original_goal": ranking.ORIGINAL_GOAL, "value_path_controls": "full 2x2",
              "proposal_queries_generated": 0, "api_requests": 0, "training_updates": 0,
              "forecast_error": None, "real_utility": None, "outcome_values_not_parsed_or_used": True, "limitations": LIMITS}
    path = directory / "predictions.json"
    ranking.write_json(path, report)
    try:
        backend = ranking.NativeBackend(device=device, offload=offload)
        report["backend"] = backend.metadata
        for window in gate["windows"]:
            source_ranking = Path(window["ranking_json"])
            data = json.loads(source_ranking.read_text())
            prefix, base = window["source"]["prefix_steps"], source_ranking.parent
            text = tensor_record(base, data["embeddings"]["clean"], device)
            if ranking.tensor_sha(text) != data["goal_routing"]["policy"]:
                raise RuntimeError("Forecast/Value goal differs from committed original policy Goal")
            original = tensor_record(base, data["original_vae_latent"], device)
            item = {key: window[key] for key in ("source", "ranking_json", "ranking_sha256")}
            item["conditions"] = {}
            for arm in ARMS:
                rows, scores, isolated_scores, direct_scores = [], [], [], []
                for index, checked in enumerate(window["candidates"]):
                    evidence = checked["arms"][arm]
                    if evidence["status"] != "passed":
                        rows.append({"status": "skip_model_exact_gate_failed", "index": index,
                                     "gate_status": evidence["status"]})
                        continue
                    latent = tensor_record(gate_path.parent, evidence["latent"], device)
                    candidate = data["candidates"][index]
                    # Repeat cheap equality checks after CPU save / CUDA reload.
                    normalized = tensor_record(base, candidate["normalized_actions"], device)
                    action = backend.native.action_from_latent(latent)
                    if ranking.tensor_sha(action) != ranking.tensor_sha(normalized):
                        raise RuntimeError("Gate representation no longer decodes to committed action on probe device")
                    if replay.array_sha(backend.controls(action)) != candidate["actions"]["array_sha256"]:
                        raise RuntimeError("Probe controls differ from committed controls")
                    raw = tensor_record(base, candidate["action_latent"], device)
                    if any(ranking.tensor_sha(raw[:, :, slot]) != ranking.tensor_sha(latent[:, :, slot])
                           for slot in range(9) if slot != 4):
                        raise RuntimeError("Probe representation changed another slot")
                    clean_future = None if arm == "clean_raw" else tensor_record(directory,
                        item["conditions"]["clean_raw"]["candidates"][index]["future_latent"], device)
                    row = predict_one(backend, latent, text, original, candidate["seed"], candidate["audit"]["rng"],
                                      directory, f"prefix_{prefix:03d}/{arm}/candidate_{index:02d}",
                                      raw[:, :, 4], clean_future)
                    row.update(index=index, candidate_actions_sha256=candidate["actions"]["array_sha256"])
                    if arm == "clean_raw":
                        old_clean = data["conditions"]["clean"]["candidates"][index]
                        old_future = tensor_record(base, old_clean["future_latent"])
                        current_future = tensor_record(directory, row["future_latent"])
                        row["clean_repeat_against_committed_native"] = {
                            "future_tensor_exact": ranking.tensor_sha(old_future) == ranking.tensor_sha(current_future),
                            "future_slots_mse": float((old_future[:, :, 5:8] - current_future[:, :, 5:8]).square().mean()),
                            "native_score_exact": old_clean["predicted_value"] == row["predicted_value"],
                            "native_score_absolute_delta": abs(old_clean["predicted_value"] - row["predicted_value"])}
                    rows.append(row)
                    scores.append(row["predicted_value"])
                    isolated_scores.append(row["forecast_only_value_control"]["predicted_value"])
                    direct_scores.append(row["value_direct_only_control"]["predicted_value"])
                    print(json.dumps({"phase": "representation_probe", "prefix_steps": prefix,
                                      "arm": arm, "candidate": index, "value": row["predicted_value"]}), flush=True)
                    del latent, raw, normalized
                selection = complete_ranking(rows)
                item["conditions"][arm] = {"candidates": rows, "selection": selection,
                    "ranking_status": "available" if selection else "unavailable_candidate_rejected_do_not_shrink_set",
                    "forecast_only_selection": complete_ranking(rows, "forecast_only_value_control"),
                    "value_direct_only_selection": complete_ranking(rows, "value_direct_only_control")}
            clean = item["conditions"]["clean_raw"]["selection"]
            for arm, condition in item["conditions"].items():
                choice = condition["selection"]
                condition["ranking_flip_from_clean"] = choice["rank"] != clean["rank"] if choice and clean else None
            report["windows"].append(item)
            ranking.write_json(path, report)
        verify_seal(gate_path, "codec_gate_inputs_sealed")
        report["status"] = "all_eligible_predictions_written_before_existing_outcome_join"
        ranking.write_json(path, report)
        sealed(directory / "predictions_commit.json", "representation_predictions_sealed_before_outcome_join",
               output_files(directory), predictions=str(path), predictions_sha256=replay.file_sha(path),
               gate_commit=str(gate_path), outcome_values_not_parsed_or_used=True,
               protocol=str(SPEC), protocol_sha256=replay.file_sha(SPEC),
               ordering_scope="Local program order only; pre-existing development outcomes were already available to researchers",
               limitations=LIMITS)
    except BaseException as error:
        report.update(status="probe_error", exception_type=type(error).__name__, exception=str(error))
        ranking.write_json(path, report)
        raise
    return directory / "predictions_commit.json"


def join_existing(prediction_commit, evaluation_root, out_dir):
    """CPU-only join; old independently replayed candidates remain authoritative."""
    commit = verify_seal(prediction_commit, "representation_predictions_sealed_before_outcome_join")
    gate_commit = verify_seal(commit["gate_commit"], "codec_gate_inputs_sealed")
    prediction_path = Path(commit["predictions"])
    data = json.loads(prediction_path.read_text())
    directory = fresh(out_dir)
    import cosmos_policy_native_inference as native
    result = {"status": "joining_frozen_existing_outcomes", "prediction_commit": str(Path(prediction_commit).resolve()),
              "prediction_commit_sha256": replay.file_sha(prediction_commit), "windows": [], "limitations": LIMITS,
              "utility_estimand": evaluation.UTILITY_ESTIMAND, "no_new_environment_replays": True}
    for window in data["windows"]:
        prefix = window["source"]["prefix_steps"]
        old_dir = Path(evaluation_root).resolve() / f"prefix_{prefix:03d}"
        for relative in ("predictions_commit.json", "results.json", "replays/utilities.json"):
            reference_path = old_dir / relative
            if gate_commit["existing_outcome_reference_sha256"].get(str(reference_path)) != replay.file_sha(reference_path):
                raise ValueError("Existing outcome reference differs from pre-probe frozen bytes")
        old_commit_path = old_dir / "predictions_commit.json"
        old_commit, old_prediction = evaluation.verify_commit(old_commit_path)
        if replay.file_sha(old_commit["ranking_json"]) != window["ranking_sha256"]:
            raise ValueError("Existing utility belongs to a different proposal manifest")
        for key in ("snapshot_sha256", "episode_sha256", "observation_npz_sha256", "state_sha256", "prefix_steps"):
            if old_prediction["source"][key] != window["source"][key]:
                raise ValueError(f"Existing utility source differs: {key}")
        utilities_path = old_dir / "replays/utilities.json"
        utility = evaluation._validate_utilities(old_commit_path, old_commit, old_prediction, utilities_path)
        old_results_path = old_dir / "results.json"
        old_results = json.loads(old_results_path.read_text())
        if old_results["utilities_sha256"] != replay.file_sha(utilities_path) or old_results["prediction_commit_sha256"] != replay.file_sha(old_commit_path):
            raise ValueError("Existing evaluation results do not seal these utility/commit artifacts")
        target_obs = {}
        for index in range(2):
            target = utility["candidates"][str(index)]["replays"][0]["candidate_future"]
            with np.load(target["observations_npz"], allow_pickle=False) as archive:
                target_obs[index] = {key: archive[key].copy() for key in replay.OBS_KEYS}
        item = {"prefix_steps": prefix, "source": window["source"], "conditions": {},
                "existing_utilities": str(utilities_path), "existing_utilities_sha256": replay.file_sha(utilities_path),
                "existing_results": str(old_results_path), "existing_results_sha256": replay.file_sha(old_results_path),
                "primary_success_distinguishes_candidates": utility["primary_success_distinguishes_candidates"],
                "candidate_utilities": old_results["actual_candidate_utilities"]}
        for arm, condition in window["conditions"].items():
            errors = []
            for row in condition["candidates"]:
                if row["status"] != "predicted":
                    errors.append({"index": row["index"], "status": row["status"]})
                    continue
                index = row["index"]
                if row["candidate_actions_sha256"] != old_commit["candidate_actions_sha256"][index]:
                    raise ValueError("New representation prediction controls differ from frozen replay")
                error = evaluation._image_error(row, prediction_path.parent, target_obs[index], native.preprocess_observation)
                errors.append({"index": index, "status": "scored", "forecast_error": error,
                    "target_after_native_success_stop": utility["candidates"][str(index)]["replays"][0]["candidate_future_after_native_success_stop"]})
            summaries = {}
            for field in ("selection", "forecast_only_selection", "value_direct_only_selection"):
                selection = condition[field]
                if selection is None:
                    summaries[field] = None
                    continue
                expected = ranking.rank_scores(selection["scores"])
                if expected != selection:
                    raise ValueError("Saved ranking violates native tie rule")
                outcome = utility["candidates"][str(selection["selected_index"])]
                summaries[field] = {**selection, "actual_success": outcome["success"],
                                    "actual_steps_to_success": outcome["steps_to_success"],
                                    "actual_utility": int(outcome["success"])}
            mse = [row["forecast_error"]["mean_rgb_mse_0_1"] for row in errors if row["status"] == "scored"]
            item["conditions"][arm] = {**summaries, "forecast_errors": errors,
                "mean_two_candidate_rgb_mse_0_1": float(np.mean(mse)) if len(mse) == 2 else None,
                "forecast_error_aggregation": "equal weight for both views, then equal weight for both candidates; no substitution for rejected candidates"}
        clean = item["conditions"]["clean_raw"]
        for arm, condition in item["conditions"].items():
            mse, clean_mse = condition["mean_two_candidate_rgb_mse_0_1"], clean["mean_two_candidate_rgb_mse_0_1"]
            condition["forecast_mse_delta_from_clean"] = mse - clean_mse if mse is not None and clean_mse is not None else None
            for field in ("selection", "forecast_only_selection", "value_direct_only_selection"):
                selected, baseline = condition[field], clean["selection"]
                if selected is None or baseline is None:
                    continue
                selected["ranking_flip_from_clean"] = selected["rank"] != baseline["rank"]
                selected["actual_success_loss_from_clean"] = baseline["actual_utility"] - selected["actual_utility"]
                selected["extra_steps_vs_clean_when_both_succeed"] = (
                    selected["actual_steps_to_success"] - baseline["actual_steps_to_success"]
                    if selected["actual_success"] and baseline["actual_success"] else None)
        result["windows"].append(item)
    verify_seal(prediction_commit, "representation_predictions_sealed_before_outcome_join")
    result["status"] = "joined_existing_true_outcomes_development_mechanism_probe"
    result["security_demonstrated"] = False
    result["goal_bus_demonstrated"] = False
    result["aggregate"] = {}
    for arm in ARMS:
        rows = [window["conditions"][arm] for window in result["windows"]]
        result["aggregate"][arm] = {"planned_correlated_windows": len(PREFIXES)}
        for field in ("selection", "forecast_only_selection", "value_direct_only_selection"):
            selections = [row[field] for row in rows if row[field] is not None and "ranking_flip_from_clean" in row[field]]
            result["aggregate"][arm][field] = {
                "valid_windows": len(selections), "unavailable_windows": len(PREFIXES) - len(selections),
                "ranking_flips": sum(row["ranking_flip_from_clean"] for row in selections),
                "actual_success_losses": sum(row["actual_success_loss_from_clean"] for row in selections),
                "note": "Descriptive counts on one development trajectory; no independent-sample significance test"}
    ranking.write_json(directory / "results.json", result)
    return directory / "results.json"


def self_test():
    """Tiny synthetic codec/ordering checks; no native model or CUDA initialization."""
    from types import SimpleNamespace
    checks = []
    extract, _ = projection.official_codecs()
    native = SimpleNamespace(
        action_from_latent=lambda z: extract(z, (16, 7), torch.tensor([4], device=z.device)),
        normalize_action=lambda x, stats: x.detach().float().cpu().numpy().astype(np.float64))
    raw = torch.zeros(ranking.LATENT_SHAPE)
    copies = raw[:, :, 4].reshape(1, 112, 112).clone()
    copies[:, ::2] = 0.25
    copies[:, 1::2] = -0.25
    raw[:, :, 4] = copies.reshape(1, 16, 28, 28)
    normalized = native.action_from_latent(raw)
    controls = native.normalize_action(normalized, None)[0]
    report, tensors = exact_candidate_gate(raw, normalized, controls, native, None)
    assert report["status"] == "passed" and set(tensors) == set(ARMS)
    assert all(row["status"] == "passed" for row in report["arms"].values())
    checks.append("all fixed synthetic representations retain exact decoded and physical controls")
    bad = normalized.clone(); bad[0, 0, 0] = 0.001
    rejected, candidates = exact_candidate_gate(raw, bad, controls, native, None)
    assert rejected["status"] == "rejected_original_decode_mismatch" and not candidates
    bad_controls = controls.copy(); bad_controls[0, 0] = 0.001
    rejected, candidates = exact_candidate_gate(raw, normalized, bad_controls, native, None)
    assert rejected["status"] == "rejected_original_decode_mismatch" and not candidates
    checks.append("normalized or physical control mismatch rejects every model arm without tolerance")
    with tempfile.TemporaryDirectory(prefix="cosmos_alias_contract_") as temporary:
        directory = Path(temporary)
        target = directory / "input.bin"; target.write_bytes(b"original")
        seal = directory / "seal.json"
        sealed(seal, "fixture", {str(target): replay.file_sha(target)})
        verify_seal(seal, "fixture")
        target.write_bytes(b"changed")
        try:
            verify_seal(seal, "fixture")
        except ValueError:
            checks.append("sealed source tampering rejected before model/target loading")
        else:
            raise AssertionError("Mutation accepted")
        class Stub:
            def rand(self, shape, seed=1):
                return torch.from_numpy(np.random.RandomState(seed).standard_normal(shape).astype(np.float32))
            def future(self, source, text, seed, rand):
                rand(tuple(source.shape), seed=seed)
                result = source.clone()
                result[:, :, 5:8] = source[:, :, 4].flatten()[0]
                return result
            def value(self, source, text, seed, rand):
                rand(tuple(source.shape), seed=seed)
                score = 0.5 + float(source[:, :, 4].flatten()[0]) * 0.25 + float(source[:, :, 5:8].mean()) * 0.5
                result = source.clone(); result[:, :, 8] = score
                return torch.tensor([score]), result
            def decode(self, source, original):
                return {view: np.full((1, 224, 224, 3), 100, dtype=np.uint8) for view in ("wrist", "primary")}
        backend = Stub()
        rng = ranking.AuditedRNG(backend.rand, 31)
        rng(ranking.LATENT_SHAPE, seed=31)
        expected_rng = rng.finish()
        text = torch.ones(1, 512, 1024, dtype=torch.bfloat16)
        clean = predict_one(backend, raw, text, torch.zeros_like(raw), 31, expected_rng, directory,
                            "synthetic_clean", raw[:, :, 4])
        clean_future = tensor_record(directory, clean["future_latent"])
        variant = predict_one(backend, tensors["reverse_copies"], text, torch.zeros_like(raw), 31,
                              expected_rng, directory, "synthetic_alias", raw[:, :, 4], clean_future)
        assert clean["predicted_value"] == 0.6875
        assert variant["predicted_value"] == 0.3125
        assert variant["forecast_only_value_control"]["predicted_value"] == 0.4375
        assert variant["value_direct_only_control"]["predicted_value"] == 0.5625
        for key in ("forecast_only_value_control", "value_direct_only_control"):
            value_input = tensor_record(directory, variant[key]["value_input"])
            assert ranking.tensor_sha(value_input) == variant[key]["audit"]["input_latent_sha256"]
            assert variant[key]["audit"]["rng"] == expected_rng
            assert variant[key]["audit"]["text_sha256"] == ranking.tensor_sha(text)
        checks.append("2x2 synthetic values separately identify forecast-only/direct-value routes; actual inputs, outputs, Goal and RNG hashes preserved")
        one_missing = [{**clean, "index": 0}, {"index": 1, "status": "skip_model_exact_gate_failed"}]
        assert complete_ranking(one_missing) is None
        assert complete_ranking(one_missing, "forecast_only_value_control") is None
        checks.append("one rejected candidate makes entire state/arm ranking unavailable; candidate set is never shrunk")
    assert not torch.cuda.is_initialized()
    return {"status": "passed_synthetic_cpu_checks_only", "checks": checks,
            "cuda_initialized": False, "scientific_results": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--codec-gate", action="store_true")
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--join", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--input-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--evaluation-root", type=Path, default=EVALUATION_ROOT)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--prediction-commit", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--offload", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.self_test:
        print(json.dumps(self_test(), indent=2)); return
    if args.out_dir is None:
        parser.error("--out-dir is required")
    if args.codec_gate:
        path = codec_gate(args.input_root, args.out_dir, args.device, args.evaluation_root)
    elif args.probe:
        if args.gate is None:
            parser.error("--probe requires --gate")
        path = probe(args.gate, args.out_dir, device=args.device, offload=args.offload)
    else:
        if args.prediction_commit is None:
            parser.error("--join requires --prediction-commit")
        path = join_existing(args.prediction_commit, args.evaluation_root, args.out_dir)
    print(json.dumps({"output": str(path)}))


if __name__ == "__main__":
    main()
