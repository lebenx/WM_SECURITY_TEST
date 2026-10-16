#!/usr/bin/env python3
"""One calibrated, post-hoc Cosmos first-query truth substitution; no environment.

All controls and continuation outcomes are historical. New latent tensors are
regenerated outputs, not recovered historical tensor snapshots. No retries.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import random
import signal
import socket
import subprocess
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cosmos_first_query_truth_countercheck_20261015.json"
REPLAY = ROOT / "runs/cosmos_first_query_truth_replay_20261015"
CLEAN = ROOT / "runs/cosmos_policy_native_task_gate_clean_n2_matched30_v2/libero_goal_task03_init02_clean.json"
SWAP = ROOT / "runs/cosmos_native_goal_intervention_matched30_v1/libero_goal_task03_init02_goal_swap.json"
BRANCHES = ROOT / "runs/cosmos_first_candidate_utility_matched30_v1/task03_init02"
OBS_KEYS = ("agentview_image", "robot0_eye_in_hand_image", "robot0_gripper_qpos",
            "robot0_eef_pos", "robot0_eef_quat")
SEEDS = (31, 32)
SHAPE = (1, 16, 9, 28, 28)
BUDGET = {"policy": 2, "future": 4, "value": 6, "vae_encode": 3}


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


class CalibrationFailure(RuntimeError):
    pass


class ResourceDeferred(RuntimeError):
    pass


def first_chunk(episode):
    with Path(episode["chunks"]).open() as handle:
        row = json.loads(next(handle))
    require(row["chunk"] == 1 and row["first_step"] == 1, "Not the fixed first query")
    return row


def gpu_info():
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=10)
    values = [int(x.strip()) for x in result.stdout.strip().split(",")]
    return dict(zip(("free_mib", "total_mib", "utilization_percent"), values))


def load_obs(path, np):
    with np.load(path, allow_pickle=False) as source:
        require(all(k in source for k in OBS_KEYS), "Missing original observation field")
        obs = {k: source[k].copy() for k in OBS_KEYS}
    for k in OBS_KEYS[:2]:
        require(obs[k].shape == (256, 256, 3) and obs[k].dtype == np.uint8,
                "Expected untransformed 256x256 RGB")
    for k, shape in zip(OBS_KEYS[2:], ((2,), (3,), (4,))):
        require(obs[k].shape == shape and np.isfinite(obs[k]).all(), "Invalid proprio field: " + k)
    return obs


def joint_video(native, vae, initial, endpoint, np, torch):
    """Official causal 33-frame layout, including its current-image history."""
    wrist, primary = native.preprocess_observation(initial)
    future_wrist, future_primary = native.preprocess_observation(endpoint)
    blank = np.zeros_like(primary)
    groups = (blank, wrist, primary, blank, blank, future_wrist, future_primary, blank)
    raw = np.concatenate([blank[None]] + [np.repeat(x[None], 4, axis=0) for x in groups])
    require(raw.shape == (33, 224, 224, 3), "Wrong joint raw video shape")
    video = torch.from_numpy(raw).permute(3, 0, 1, 2)[None]
    video = (video.to(vae.latents_mean) / 127.5 - 1.0).to(memory_format=torch.channels_last_3d)
    return video, raw


def main(config_path):
    protocol = read(config_path)
    out = ROOT / protocol["inference_output"]
    require((ROOT / protocol["replay_output"]).resolve() == REPLAY, "Frozen replay directory changed")
    require(protocol["task_id"] == 3 and protocol["init_id"] == 2
            and protocol["candidate_indices"] == [0, 1] and protocol["candidate_seeds"] == list(SEEDS)
            and protocol["horizon_controls"] == 16 and protocol["offload"] is False,
            "Frozen scientific panel changed")
    require(protocol["inference_call_boundary_deadline_seconds"] == 120, "Frozen wall limit changed")
    for field, expected in (("policy_sampler_calls", 2), ("future_sampler_calls", 4),
                            ("value_sampler_calls", 6), ("vae_encodes", 3), ("vae_decodes", 0)):
        require(protocol["max_activity"][field] == expected, "Frozen budget changed: " + field)
    for name, expected in (("clean", CLEAN), ("swap", SWAP)):
        require((ROOT / protocol["sources"][name]["episode"]).resolve() == expected, "Source episode changed")
    require(not out.exists(), "Output exists; refusing overwrite/retry")
    out.mkdir(parents=True)
    (out / "launched_config.json").write_bytes(config_path.read_bytes())
    (out / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    start = time.monotonic()
    counts = {**dict.fromkeys(BUDGET, 0), "noise_calls": 0, "environment_steps": 0,
              "api_calls": 0, "training_updates": 0, "vae_decode": 0}
    report = {"status": "checking_replay", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "activity": counts, "budget": BUDGET, "calibration": {"all_passed": False, "checks": {}},
              "scores": {}, "forecast_errors": None, "checkpoint_loads": {},
              "interpretation_rules": protocol["interpretation_rules"],
              "scope": "One post-hoc task3/init2 first query, two fixed historical controls. New regenerated tensors; no historical full-latent equality claim.",
              "limits": ["Historical prefix equality covers recorded fields, not every unrecorded simulator/controller variable.",
                         "Feature MSE measures the native representation, not complete physical-state error.",
                         "True-future substitution is a researcher diagnostic, not an executable oracle or a new defense.",
                         "Existing full continuation utilities are reused; no new closed-loop outcome or population inference."]}

    def persist():
        report["elapsed_seconds"] = time.monotonic() - start
        tmp = out / "summary.tmp.json"
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        tmp.replace(out / "summary.json")

    def deadline(*_):
        raise ResourceDeferred("Frozen wall-clock limit reached")

    def no_network(*_args, **_kwargs):
        counts["api_calls"] += 1
        raise RuntimeError("Network prohibited")

    signal.signal(signal.SIGALRM, deadline)
    signal.signal(signal.SIGTERM, deadline)
    signal.setitimer(signal.ITIMER_REAL, 120)
    report["hard_wall_limit_seconds"] = 120
    persist()
    try:
        import numpy as np
        replay = read(REPLAY / "summary.json")
        require(replay["status"] == "completed_exact_two_candidate_prefix_replay"
                and replay["qualification"]["passed"] is True, "Replay is not qualified and complete")
        require(len(replay["candidates"]) == 2, "Expected exactly two replay candidates")
        for i, row in enumerate(replay["candidates"]):
            require(row["candidate_index"] == i and row["historical_trace_exact_steps"] == 16,
                    "Wrong or incomplete replay prefix")
            require((ROOT / row["endpoint_obs"]).resolve() == REPLAY / f"candidate{i}_endpoint_obs.npz",
                    "Unexpected endpoint observation path")
        initial = load_obs(REPLAY / "initial_obs.npz", np)
        endpoints = [load_obs(REPLAY / f"candidate{i}_endpoint_obs.npz", np) for i in range(2)]
        clean_ep, swap_ep = read(CLEAN), read(SWAP)
        clean_ref, swap_ref = first_chunk(clean_ep), first_chunk(swap_ep)
        branch_ep = [read(BRANCHES / f"candidate{i}/libero_goal_task03_init02_clean.json") for i in range(2)]
        branch_ref = [first_chunk(ep) for ep in branch_ep]
        for ep in (clean_ep, swap_ep, *branch_ep):
            require(ep["task_id"] == 3 and ep["init_id"] == 2 and ep["seed"] == 31
                    and ep["environment_seed"] == 0 and ep["settle_steps"] == 10, "Historical root changed")
        goal, swap_goal = clean_ep["goal"], swap_ep["forecast_goal"]
        require(goal == "open the top drawer and put the bowl inside"
                and swap_goal == "put the bowl on top of the cabinet"
                and swap_ep["policy_goal"] == swap_ep["value_goal"] == goal, "Goal routing changed")
        for i in range(2):
            require(branch_ref[i]["native_best_of_n"]["selected_index"] == i, "Wrong forced branch")
            require(branch_ref[i]["native_best_of_n"]["candidates"] == clean_ref["native_best_of_n"]["candidates"],
                    "Historical first candidates differ")
            for ref in (clean_ref, swap_ref):
                require(ref["native_best_of_n"]["candidates"][i]["seed"] == SEEDS[i], "Candidate seed changed")
                require(ref["native_best_of_n"]["candidates"][i]["environment_actions"]
                        == branch_ref[i]["environment_actions"], "Historical fixed controls differ")
        report["sources"] = {"clean_episode": str(CLEAN.relative_to(ROOT)), "swap_episode": str(SWAP.relative_to(ROOT)),
                             "qualified_replay": str(REPLAY.relative_to(ROOT)),
                             "historical_branches": str(BRANCHES.relative_to(ROOT))}
        report["goals"] = {"original": goal, "forecast_swap": swap_goal}
        report["historical_continuation_utilities"] = [
            {"candidate_index": i, "success": ep["success"], "steps": ep["environment_steps"],
             "capped_steps": ep["environment_steps"] if ep["success"] else 300} for i, ep in enumerate(branch_ep)]
        require(protocol["minimum_free_gpu_mib"] == 8192 and protocol["cuda_memory_fraction"] == 0.28,
                "Frozen resource gates changed")
        require(os.environ.get("CUDA_VISIBLE_DEVICES", "0") == "0", "GPU0 required")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        require(os.environ.get("ALPHABRAIN_USE_NATIVE_TE") != "1", "Historical torch backend required")
        report["gpu_at_start"] = gpu_info(); persist()
        if report["gpu_at_start"]["free_mib"] < 8192:
            raise ResourceDeferred("GPU0 free below8192MiB; no weights loaded")
        import torch
        import cosmos_policy_native_inference as native
        require(torch.__version__ == clean_ep["provenance"]["torch_version"]
                and np.__version__ == clean_ep["provenance"]["numpy_version"], "Historical runtime version changed")
        random.seed(31); np.random.seed(31); torch.manual_seed(31); torch.cuda.manual_seed_all(31)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(0.28, 0)
        torch.cuda.reset_peak_memory_stats(0)
        require(str(native.MODEL) == clean_ep["provenance"]["checkpoint"]
                and str(native.TOKENIZER) == clean_ep["provenance"]["tokenizer"], "Checkpoint path changed")
        original_load = torch.load

        def once(path, *args, **kwargs):
            name = str(Path(path).resolve())
            require(name in (str(native.MODEL), str(native.TOKENIZER))
                    and name not in report["checkpoint_loads"], "Unexpected or repeated weight load")
            report["checkpoint_loads"][name] = 1
            return original_load(path, *args, **kwargs)

        def save_tensor(name, **items):
            torch.save({k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                        for k, v in items.items()}, out / (name + ".pt"))

        def check(name, ok):
            report["calibration"]["checks"][name] = bool(ok)
            if not ok:
                report["calibration"]["all_passed"] = False
            persist()
            if not ok:
                raise CalibrationFailure(name)

        def array_exact(actual, expected):
            a, b = np.asarray(actual), np.asarray(expected)
            return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

        def bump(kind):
            require(counts[kind] < BUDGET[kind], "Call budget exceeded: " + kind)
            counts[kind] += 1

        with ExitStack() as guards, torch.inference_mode():
            guards.enter_context(patch.object(socket.socket, "connect", side_effect=no_network))
            report["status"] = "loading_existing_checkpoint"; persist()
            with patch.object(torch, "load", side_effect=once):
                model, vae, sampler, base_rand = native.load_model(device="cuda", verify=False)
            require(len(report["checkpoint_loads"]) == 2, "Both existing weights must load once")
            report["strict_load"] = model.checkpoint_adapter_metadata
            stats = native.load_stats()
            with (native.MODEL.parent / "libero_t5_embeddings.pkl").open("rb") as handle:
                text = pickle.load(handle)
            original_text = text[goal].to(device="cuda", dtype=torch.bfloat16)
            swapped_text = text[swap_goal].to(device="cuda", dtype=torch.bfloat16)
            noises = {}

            def rand(shape, seed):
                require(tuple(shape) == SHAPE and seed in SEEDS, "Unregistered noise request")
                counts["noise_calls"] += 1
                require(counts["noise_calls"] <= 12, "Noise budget exceeded")
                value = base_rand(shape, seed=seed)
                if seed in noises:
                    require(torch.equal(value, noises[seed]), "Candidate seed-local noise changed")
                else:
                    noises[seed] = value.detach().clone()
                    save_tensor(f"noise_seed{seed}", unit_noise=value)
                return value

            def sample(kind, latent, embedding, seed):
                bump(kind)
                before, text_before = latent.clone(), embedding.clone()
                fn = {"policy": native.get_action, "future": native.get_future_state_prediction,
                      "value": native.get_value_prediction}[kind]
                result = fn(model, sampler, rand, latent, embedding, seed, 5 if kind == "policy" else 1)
                require(torch.equal(latent, before) and torch.equal(embedding, text_before), "Sampler mutated protected input")
                return result

            bump("vae_encode")
            observed, original_vae = native.encode_observation(vae, initial, stats)
            save_tensor("context", observed_latent=observed, original_vae_latent=original_vae,
                        original_text=original_text, swap_text=swapped_text)
            candidates, all_forecasts = [], {"clean": [], "swap": []}
            report["status"] = "calibrating_clean_first_query"; persist()
            for i, seed in enumerate(SEEDS):
                normalized, action_latent = sample("policy", observed, original_text, seed)
                controls = native.normalize_action(normalized, stats)[0]
                future = sample("future", action_latent, original_text, seed)
                value, value_latent = sample("value", future, original_text, seed)
                score = float(value.item())
                rms = future[:, :, 5:8].float().square().mean().sqrt().item()
                save_tensor(f"clean_candidate{i}", action_latent=action_latent, normalized_actions=normalized,
                            environment_actions=controls, future_latent=future, value_latent=value_latent,
                            value=value, seed=seed)
                ref = clean_ref["native_best_of_n"]["candidates"][i]
                check(f"clean{i}_controls_exact", array_exact(controls, np.asarray(ref["environment_actions"], dtype=np.float64)))
                check(f"clean{i}_normalized_exact", array_exact(normalized.float().cpu().numpy()[0],
                      np.asarray(branch_ref[i]["normalized_actions"], dtype=np.float32)))
                check(f"clean{i}_value_exact", score == ref["predicted_value"])
                check(f"clean{i}_future_rms_exact", rms == branch_ref[i]["future_slot_rms"])
                candidates.append({"normalized": normalized, "controls": controls, "action_latent": action_latent})
                all_forecasts["clean"].append(future)
                report["scores"].setdefault("clean", []).append(score)
                del value_latent
            check("clean_selection_exact", max(range(2), key=lambda i: report["scores"]["clean"][i])
                  == clean_ref["native_best_of_n"]["selected_index"])
            report["status"] = "calibrating_swap_first_query"; persist()
            for i, seed in enumerate(SEEDS):
                action_latent = candidates[i]["action_latent"]
                future = sample("future", action_latent, swapped_text, seed)
                value, value_latent = sample("value", future, original_text, seed)
                score = float(value.item())
                save_tensor(f"swap_candidate{i}", action_latent=action_latent, future_latent=future,
                            value_latent=value_latent, value=value, seed=seed)
                ref = swap_ref["native_best_of_n"]["candidates"][i]
                check(f"swap{i}_controls_exact", array_exact(candidates[i]["controls"], np.asarray(ref["environment_actions"], dtype=np.float64)))
                check(f"swap{i}_value_exact", score == ref["predicted_value"])
                if i == swap_ref["native_best_of_n"]["selected_index"]:
                    check("swap_selected_normalized_exact", array_exact(candidates[i]["normalized"].float().cpu().numpy()[0],
                          np.asarray(swap_ref["normalized_actions"], dtype=np.float32)))
                    check("swap_selected_future_rms_exact", future[:, :, 5:8].float().square().mean().sqrt().item()
                          == swap_ref["future_slot_rms"])
                all_forecasts["swap"].append(future)
                report["scores"].setdefault("swap", []).append(score)
                del value_latent
            check("swap_selection_exact", max(range(2), key=lambda i: report["scores"]["swap"][i])
                  == swap_ref["native_best_of_n"]["selected_index"])
            require(counts == {"policy": 2, "future": 4, "value": 4, "vae_encode": 1,
                    "noise_calls": 10, "environment_steps": 0, "api_calls": 0, "training_updates": 0, "vae_decode": 0},
                    "Calibration order or budget changed")
            report["calibration"]["all_passed"] = True
            report["calibration"]["historical_all_passed_before_truth_encoding"] = True
            report["status"] = "encoding_joint_true_futures"; persist()
            truths = []
            for i, endpoint in enumerate(endpoints):
                video, raw = joint_video(native, vae, initial, endpoint, np, torch)
                bump("vae_encode")
                joint = vae.encode(video)
                require(tuple(joint.shape) == SHAPE and torch.isfinite(joint).all(), "Invalid joint VAE latent")
                check(f"truth{i}_causal_prefix_exact", torch.equal(joint[:, :, :5], original_vae[:, :, :5]))
                actual_proprio = np.concatenate((endpoint["robot0_gripper_qpos"], endpoint["robot0_eef_pos"], endpoint["robot0_eef_quat"]))
                normalized_proprio = native.codecs.rescale_proprio(actual_proprio, stats)
                truth = candidates[i]["action_latent"].clone()
                truth[:, :, 6:8] = joint[:, :, 6:8]
                truth = native.codecs.replace_latent_with_proprio(
                    truth, torch.from_numpy(normalized_proprio)[None].to(vae.latents_mean),
                    torch.tensor([5], device="cuda"))
                require(torch.equal(truth[:, :, :5], candidates[i]["action_latent"][:, :, :5]), "Protected Value prefix changed")
                save_tensor(f"truth_candidate{i}_input", joint_vae_latent=joint, value_input=truth,
                            normalized_proprio=normalized_proprio, actual_proprio=actual_proprio)
                np.save(out / f"truth_candidate{i}_raw33.npy", raw, allow_pickle=False)
                truths.append(truth)
            report["forecast_errors"] = []
            for i, truth in enumerate(truths):
                row = {"candidate_index": i,
                       "definition": "Float64 MSE of regenerated forecast versus matching joint encoded true-future slots; no decoded RGB metric",
                       "proprio_metric_limit": "The pinned native codec exposes no extract_proprio_from_latent_sequence. Slot5 MSE is against the tiled normalized-proprio training target, not decoded9D proprio or raw physical error. No new extraction rule is introduced.",
                       "combined_metric_limit": "All-slot MSE is descriptive only; image/proprio latent scales are not a common physical utility unit."}
                for condition in ("clean", "swap"):
                    delta = (all_forecasts[condition][i][:, :, 5:8].double() - truth[:, :, 5:8].double()).cpu()
                    row[condition] = {"future_proprio_slot_mse": float(delta[:, :, 0].square().mean()),
                                      "normalized_proprio_mse": None,
                                      "future_wrist_slot_mse": float(delta[:, :, 1].square().mean()),
                                      "future_primary_slot_mse": float(delta[:, :, 2].square().mean()),
                                      "visual_latent_mse": float(delta[:, :, 1:].square().mean()),
                                      "all_future_slots_mse": float(delta.square().mean())}
                report["forecast_errors"].append(row)
            report["status"] = "scoring_two_true_futures"; persist()
            for i, seed in enumerate(SEEDS):
                value, value_latent = sample("value", truths[i], original_text, seed)
                save_tensor(f"truth_candidate{i}_value", value_input=truths[i], value_latent=value_latent, value=value, seed=seed)
                report["scores"].setdefault("true_future", []).append(float(value.item()))
                persist()
            require(all(counts[k] == v for k, v in BUDGET.items()) and counts["noise_calls"] == 12, "Incomplete final call panel")
            report["ranking"] = {}
            for condition, scores in report["scores"].items():
                order = sorted(range(2), key=lambda i: (-scores[i], i))
                report["ranking"][condition] = {"rank": order, "selected_index": order[0],
                    "exact_tie": scores[0] == scores[1], "candidate1_minus_candidate0_value": scores[1] - scores[0],
                    "selection_basis": "tie_driven_candidate0_priority" if scores[0] == scores[1] else "strict_value_order",
                    "historical_selected_utility": report["historical_continuation_utilities"][order[0]]}
            report["same_seed_local_noise_verified"] = True
            report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
            report["status"] = "completed_calibrated_first_query_truth_countercheck"
            persist()
    except BaseException as error:
        report["status"] = ("resource_deferred" if isinstance(error, ResourceDeferred) else
                            "calibration_failed" if isinstance(error, CalibrationFailure) else "error")
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        persist()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    main(args.config.resolve())
