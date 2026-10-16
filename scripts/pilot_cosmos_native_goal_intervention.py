#!/usr/bin/env python3
"""Full-task Cosmos best-of-two with an internal forecast-text intervention.

Clean retains Goal0 in every call. GoalSwap changes only the fixed-action
future call to Goal1; proposals, value scoring, environment and task metric
retain Goal0. This is an internal call-context intervention, not an established
external attacker interface or a claim that partial-observation forecasts must
be goal invariant. Task success comes only from the physical LIBERO environment.
No Bus-Cut or decision-recovery result is implemented here.

The fixed pilot uses libero_goal task0, init0/1/2, candidate seeds31/32,
5/1/1 denoising steps, 300 actions and 10 settling steps. A matched successful
Clean N=2 init0 health gate is required before execution. --self-test and
--preflight do not construct an environment or run a learned forward pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

import cosmos_policy_native_inference as inference
import pilot_cosmos_policy_native_task_gate as base

ROOT, LIBERO, CKPT_DIR = base.ROOT, base.LIBERO, base.CKPT_DIR
digest, atomic_json, frame = base.digest, base.atomic_json, base.frame
CONDITIONS = ("clean", "goal_swap")
ORIGINAL_GOAL = "open the middle drawer of the cabinet"
SWAP_GOAL = "put the bowl on the stove"
SEEDS = (31, 32)


def expected_runtime_compatibility():
    if os.environ.get("ALPHABRAIN_USE_NATIVE_TE") == "1":
        return {"backend": "native Transformer Engine", "full_runtime_parity_verified": False}
    from cosmos_policy_torch_compat import compatibility_metadata
    return compatibility_metadata()


@torch.inference_mode()
def best_of_two(model, vae, sampler, rand, obs, policy_text, forecast_text, stats, *, api=inference):
    """Same ordered calls as native best_of_n; only future text is separable."""
    observed, original = api.encode_observation(vae, obs, stats)
    candidates = []
    for index, seed in enumerate(SEEDS):
        actions, action_latent = api.get_action(model, sampler, rand, observed, policy_text, seed, 5)
        future = api.get_future_state_prediction(model, sampler, rand, action_latent, forecast_text, seed, 1)
        value, value_latent = api.get_value_prediction(model, sampler, rand, future, policy_text, seed, 1)
        candidates.append({"index": index, "seed": seed, "normalized_actions": actions,
                           "actions": api.normalize_action(actions, stats), "action_latent": action_latent,
                           "future_latent": future, "value": float(value.item())})
        del value_latent
    selected = max(range(2), key=lambda i: candidates[i]["value"])
    return {"selected_index": selected, "candidates": candidates,
            "observed_latent": observed, "original_latent": original}


def validate_clean_gate(path, offload):
    """A startup health check, not a population-level success threshold."""
    if path is None:
        raise ValueError("Supply the completed matched --clean-gate-summary (Clean best-of-2 init0)")
    path = Path(path).resolve()
    summary = json.loads(path.read_text())
    protocol, provenance = summary.get("protocol", {}), summary.get("provenance", {})
    expected = {"suite": "libero_goal", "best_of_n": 2, "seed": 31, "env_seed": 0,
                "max_steps": 300, "settle_steps": 10, "offload": offload}
    if summary.get("status") != "completed" or summary.get("condition") != "clean":
        raise ValueError("Clean N=2 health gate is not completed")
    for key, value in expected.items():
        if type(protocol.get(key)) is not type(value) or protocol[key] != value:
            raise ValueError(f"Clean health-gate protocol mismatch: {key}")
    if (provenance.get("checkpoint_sha256") != inference.MODEL_SHA256
            or provenance.get("tokenizer_sha256") != inference.TOKENIZER_SHA256
            or provenance.get("helper_sha256") != digest(Path(inference.__file__))
            or provenance.get("runner_sha256") != digest(Path(base.__file__))):
        raise ValueError("Clean health-gate checkpoint/adapter implementation mismatch")
    for key, value in {"sampler_seeds": [31, 32], "action_steps": 5,
                       "future_steps": 1, "value_steps": 1}.items():
        if provenance.get(key) != value:
            raise ValueError(f"Clean health-gate sampler mismatch: {key}")
    if provenance.get("source_sha256", {}).get("scripts/cosmos_policy_torch_compat.py") != digest(ROOT / "scripts/cosmos_policy_torch_compat.py"):
        raise ValueError("Clean health gate used a different compatibility shim")
    for relative, expected_sha in provenance.get("source_sha256", {}).items():
        if digest(ROOT / relative) != expected_sha:
            raise ValueError(f"Clean health-gate dependency changed: {relative}")
    if offload and provenance.get("offload_helper_sha256") != digest(ROOT / "scripts/cosmos_policy_cuda_offload.py"):
        raise ValueError("Clean health-gate offload implementation changed")
    if provenance.get("runtime_compatibility") != expected_runtime_compatibility():
        raise ValueError("Clean health-gate runtime backend changed (including ALPHABRAIN_USE_NATIVE_TE)")
    rows = [r for r in summary.get("results", []) if r.get("task_id") == 0 and r.get("init_id") == 0]
    if (len(rows) != 1 or rows[0].get("status") != "completed"
            or rows[0].get("condition") != "clean" or rows[0].get("success") is not True
            or rows[0].get("success_at_start") is not False or rows[0].get("goal") != ORIGINAL_GOAL):
        raise ValueError("Require exactly one nontrivial successful Clean task0/init0 health episode")
    return {"path": str(path), "sha256": digest(path), "passed": True,
            "criterion": "One matched Clean N=2 task0/init0 success; startup health only, not general task reliability.",
            "source_episode": {key: rows[0][key] for key in
                               ("initial_state_sha256", "bddl_sha256", "init_file_sha256")},
            "provenance": provenance}


def aggregate_results(results):
    aggregate = {}
    for condition in CONDITIONS:
        rows = [r for r in results if r["condition"] == condition and r["status"] == "completed"]
        aggregate[condition] = {"episodes": len(rows), "successes": sum(r["success"] for r in rows),
                                "success_rate": sum(r["success"] for r in rows) / len(rows) if rows else None}
    paired = []
    for init_id in (0, 1, 2):
        rows = {r["condition"]: r for r in results if r["init_id"] == init_id and r["status"] == "completed"}
        if all(c in rows for c in CONDITIONS):
            paired.append({"init_id": init_id, "clean_success": rows["clean"]["success"],
                           "goal_swap_success": rows["goal_swap"]["success"],
                           "success_difference_swap_minus_clean": int(rows["goal_swap"]["success"]) - int(rows["clean"]["success"]),
                           "lost_clean_success": rows["clean"]["success"] and not rows["goal_swap"]["success"]})
    aggregate["paired_real_task_success"] = paired
    aggregate["scope"] = "Three prespecified initial states of one task; no inference from forecast divergence or predicted value to actual utility."
    return aggregate


def run_episode(model, vae, sampler, rand, task, task_id, init_id, text, stats,
                args, provenance, condition):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    from libero.libero.envs import OffScreenRenderEnv
    import imageio.v2 as imageio

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    bddl = LIBERO / "libero/libero/bddl_files" / task.problem_folder / task.bddl_file
    init_path = LIBERO / "libero/libero/init_files" / task.problem_folder / task.init_states_file
    states = torch.load(init_path, map_location="cpu", weights_only=False)
    initial = np.asarray(states[init_id])
    label = f"{args.suite}_task{task_id:02d}_init{init_id:02d}_{condition}"
    forecast_goal = task.language if condition == "clean" else args.swap_goal
    output = args.out_dir
    trace_path, chunks_path = output / f"{label}.jsonl", output / f"{label}_chunks.jsonl"
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
                             camera_widths=256, render_gpu_device_id=0)
    env.task_id, env.task_name, env.task_suite_name = task_id, task.name, args.suite
    result = {
        "status": "running", "model": "NVIDIA Cosmos Policy LIBERO 40K (torch adapter)",
        "condition": condition, "forecast_goal": forecast_goal,
        "policy_goal": task.language, "value_goal": task.language, "suite": args.suite, "best_of_n": args.best_of_n,
        "task_id": task_id, "init_id": init_id, "task_name": task.name,
        "goal": task.language, "max_steps": args.max_steps, "settle_steps": args.settle_steps,
        "seed": args.seed, "environment_seed": args.env_seed,
        "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
        "bddl": str(bddl), "bddl_sha256": digest(bddl), "init_file_sha256": digest(init_path),
        "trace": str(trace_path), "chunks": str(chunks_path), "provenance": provenance,
    }
    path = output / f"{label}.json"
    atomic_json(path, result)
    writer = None
    steps, chunks, success = 0, 0, False
    started = time.monotonic()
    rewards, action_norms, eef_path = [], [], []
    torch.cuda.reset_peak_memory_stats()
    try:
        env.seed(args.env_seed)
        env.reset()
        obs = env.set_init_state(states[init_id])
        initial_sim = np.asarray(env.get_sim_state()).copy()
        dummy = [0.0] * 6 + [-1.0]
        for _ in range(args.settle_steps):
            obs, _, _, _ = env.step(dummy)
        settled_sim = np.asarray(env.get_sim_state()).copy()
        result["success_at_start"] = bool(env.check_success())
        if result["success_at_start"]:
            raise RuntimeError("Task already successful before first policy action; invalid episode")
        Image.fromarray(frame(obs)).save(output / f"{label}_initial.png")
        if args.video_stride:
            writer = imageio.get_writer(str(output / f"{label}.mp4"), fps=20 / args.video_stride,
                                        codec="libx264", quality=7, macro_block_size=16)
            writer.append_data(frame(obs))
        embedding = text[task.language].to(device="cuda", dtype=torch.bfloat16)
        forecast_embedding = text[forecast_goal].to(device="cuda", dtype=torch.bfloat16)
        with trace_path.open("w") as trace, chunks_path.open("w") as chunk_log:
            done = False
            while steps < args.max_steps:
                query_started = time.monotonic()
                selection = best_of_two(
                    model, vae, sampler, rand, obs, embedding, forecast_embedding, stats,
                )
                selected = selection["candidates"][selection["selected_index"]]
                normalized, latent = selected["normalized_actions"], selected["future_latent"]
                actions = selected["actions"][0]
                candidate_log = {
                    "selected_index": selection["selected_index"],
                    "candidates": [{"index": item["index"], "seed": item["seed"],
                                    "predicted_value": item["value"],
                                    "environment_actions": item["actions"][0].tolist()}
                                   for item in selection["candidates"]],
                }
                del selection, selected
                if actions.shape != (16, 7) or not np.isfinite(actions).all() or not torch.isfinite(latent).all():
                    raise RuntimeError("Nonfinite or malformed policy output")
                chunks += 1
                chunk_log.write(json.dumps({
                    "chunk": chunks, "first_step": steps + 1, "condition": condition,
                    "policy_goal": task.language, "forecast_goal": forecast_goal, "value_goal": task.language,
                    "normalized_actions": normalized.float().cpu().numpy()[0].tolist(),
                    "environment_actions": actions.tolist(),
                    "query_seconds": time.monotonic() - query_started,
                    "future_slot_rms": latent[:, :, 5:8].float().square().mean().sqrt().item(),
                    "cuda_allocated_bytes": torch.cuda.memory_allocated(),
                    "native_best_of_n": candidate_log,
                }, allow_nan=False) + "\n")
                chunk_log.flush()
                del latent, normalized
                for action in actions[:args.max_steps - steps]:
                    before = np.asarray(obs["robot0_eef_pos"]).copy()
                    obs, reward, done, _ = env.step(action.tolist())
                    steps += 1
                    success = bool(env.check_success())
                    after = np.asarray(obs["robot0_eef_pos"]).copy()
                    rewards.append(float(reward))
                    action_norms.append(float(np.linalg.norm(action)))
                    eef_path.append(after)
                    trace.write(json.dumps({
                        "step": steps, "chunk": chunks, "condition": condition, "action": action.tolist(),
                        "eef_before": before.tolist(), "eef_after": after.tolist(),
                        "eef_quaternion": np.asarray(obs["robot0_eef_quat"]).tolist(),
                        "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"]).tolist(),
                        "reward": float(reward), "done": bool(done), "task_success": success,
                    }, allow_nan=False) + "\n")
                    if writer is not None and (steps % args.video_stride == 0 or success or done):
                        writer.append_data(frame(obs))
                    if success or done:
                        break
                trace.flush()
                print(json.dumps({"episode": label, "steps": steps, "chunks": chunks,
                                  "success": success, "elapsed_seconds": time.monotonic() - started}), flush=True)
                if success or done:
                    break
        Image.fromarray(frame(obs)).save(output / f"{label}_final.png")
        np.savez_compressed(output / f"{label}_states.npz", initial=initial_sim, settled=settled_sim,
                            final=np.asarray(env.get_sim_state()), eef_path=np.asarray(eef_path))
        result.update(status="completed", success=success, terminal_done=bool(done),
                      environment_steps=steps, action_chunks=chunks, reward_sum=sum(rewards),
                      final_reward=rewards[-1] if rewards else None,
                      action_norm_mean=float(np.mean(action_norms)) if action_norms else None,
                      initial_image=str(output / f"{label}_initial.png"),
                      final_image=str(output / f"{label}_final.png"),
                      video=str(output / f"{label}.mp4") if writer is not None else None)
        return result
    except BaseException as error:
        result.update(status="error", exception_type=type(error).__name__, exception=str(error),
                      environment_steps=steps, action_chunks=chunks)
        raise
    finally:
        if writer is not None:
            writer.close()
        result.update(elapsed_seconds=time.monotonic() - started,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        atomic_json(path, result)
        env.close()



def self_test(report_path=None):
    """Synthetic endpoint + environment stubs only; no learned model or EGL."""
    import sys
    import tempfile
    from types import ModuleType
    from unittest.mock import patch

    calls = []
    original, swapped = torch.tensor([1.0]), torch.tensor([2.0])
    obs = {"physics": "unchanged synthetic observation"}

    def log(stage, text=None, latent=None, seed=None, steps=None):
        calls.append({"stage": stage, "text": text.tolist() if text is not None else None,
                      "latent": latent.tolist() if latent is not None else None,
                      "seed": seed, "steps": steps})

    def encode(vae, observation, stats):
        log("encode")
        return torch.zeros(1, 1, 9, 1, 1), torch.ones(1, 1, 9, 1, 1)

    def action(model, sampler, rand, latent, text, seed, steps):
        log("policy", text, latent, seed, steps)
        result = latent.clone()
        result[:, :, 4] = seed
        actions = torch.full((1, 16, 7), seed * 0.001 + text.item() * 0.01)
        return actions, result

    def future(model, sampler, rand, latent, text, seed, steps):
        log("future", text, latent, seed, steps)
        result = latent.clone()
        result[:, :, 5:8] = text.item()
        return result

    def value(model, sampler, rand, latent, text, seed, steps):
        log("value", text, latent, seed, steps)
        score = (seed - 30) / 3 if latent[0, 0, 5].item() == 1 else (33 - seed) / 3
        return torch.tensor([score]), latent.clone()

    endpoints = dict(encode_observation=encode, get_action=action,
                     get_future_state_prediction=future, get_value_prediction=value,
                     normalize_action=lambda actions, stats: actions.numpy().copy())
    with patch.multiple(inference, **endpoints):
        reference = inference.best_of_n(None, None, None, None, obs, original, {}, n=2, seed=31)
        reference_calls = list(calls)
        calls.clear()
        clean = best_of_two(None, None, None, None, obs, original, original, {})
        assert calls == reference_calls
        clean_calls = list(calls)
        assert clean["selected_index"] == reference["selected_index"]
        for expected, actual in zip(reference["candidates"], clean["candidates"]):
            for key in ("normalized_actions", "action_latent", "future_latent"):
                assert torch.equal(expected[key], actual[key])
            assert np.array_equal(expected["actions"], actual["actions"])
            assert expected["value"] == actual["value"]
        calls.clear()
        attack = best_of_two(None, None, None, None, obs, original, swapped, {})
        swap_calls = list(calls)
        assert [r["stage"] for r in calls] == ["encode", "policy", "future", "value", "policy", "future", "value"]
        assert [(r["seed"], r["steps"]) for r in calls[1:]] == [(31, 5), (31, 1), (31, 1), (32, 5), (32, 1), (32, 1)]
        for normal, changed in zip(clean_calls, swap_calls):
            if normal["stage"] in ("encode", "policy"):
                assert normal == changed
            elif normal["stage"] == "future":
                assert normal["latent"] == changed["latent"] and normal["text"] == [1.0] and changed["text"] == [2.0]
            else:
                assert normal["text"] == changed["text"] == [1.0]
        for normal, changed in zip(clean["candidates"], attack["candidates"]):
            assert torch.equal(normal["action_latent"], changed["action_latent"])
            assert np.array_equal(normal["actions"], changed["actions"])
        assert clean["selected_index"] == 1 and attack["selected_index"] == 0

        class FakeEnv:
            def __init__(self, **kwargs):
                assert kwargs["camera_heights"] == kwargs["camera_widths"] == 256
                self.reset()
            def seed(self, seed):
                assert seed == 0
            def reset(self):
                self.count, self.eef = 0, np.zeros(3)
            def observation(self):
                return {"agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
                        "robot0_eye_in_hand_image": np.ones((256, 256, 3), dtype=np.uint8),
                        "robot0_eef_pos": self.eef.copy(), "robot0_eef_quat": np.array([0., 0., 0., 1.]),
                        "robot0_gripper_qpos": np.array([0.01, -0.01])}
            def set_init_state(self, state):
                self.reset()
                return self.observation()
            def get_sim_state(self):
                return np.concatenate(([self.count], self.eef))
            def check_success(self):
                return self.count >= 12
            def step(self, command):
                assert isinstance(command, list) and len(command) == 7
                self.count += 1
                self.eef += np.asarray(command[:3])
                return self.observation(), float(self.check_success()), self.check_success(), {}
            def close(self):
                pass

        class CPUEmbedding:
            def __init__(self, tensor):
                self.tensor = tensor
            def to(self, **kwargs):
                assert kwargs == {"device": "cuda", "dtype": torch.bfloat16}
                return self.tensor

        env_module = ModuleType("libero.libero.envs")
        env_module.OffScreenRenderEnv = FakeEnv
        cuda_stubs = dict(manual_seed_all=lambda seed: None, reset_peak_memory_stats=lambda: None,
                          memory_allocated=lambda: 0, max_memory_allocated=lambda: 0,
                          max_memory_reserved=lambda: 0)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"libero.libero.envs": env_module}), patch.multiple(torch.cuda, **cuda_stubs):
            directory = Path(directory)
            (directory / "task.bddl").write_text("synthetic CPU test fixture\n")
            torch.save([np.zeros(3)], directory / "init.pth")
            task = SimpleNamespace(problem_folder=str(directory), bddl_file="task.bddl", init_states_file="init.pth", name="synthetic", language=ORIGINAL_GOAL)
            text = {ORIGINAL_GOAL: CPUEmbedding(original), SWAP_GOAL: CPUEmbedding(swapped)}
            args = SimpleNamespace(suite="libero_goal", best_of_n=2, seed=31, env_seed=0,
                                   max_steps=300, settle_steps=10, video_stride=0, swap_goal=SWAP_GOAL)
            outputs = []
            for name in ("base", "clean", "goal_swap"):
                args.out_dir = directory / name
                args.out_dir.mkdir()
                if name == "base":
                    outputs.append(base.run_episode(None, None, None, None, task, 0, 0, text, {}, args, {}))
                else:
                    outputs.append(run_episode(None, None, None, None, task, 0, 0, text, {}, args, {}, name))
            reference_rows = [json.loads(line) for line in Path(outputs[0]["trace"]).read_text().splitlines()]
            new_rows = [json.loads(line) for line in Path(outputs[1]["trace"]).read_text().splitlines()]
            assert reference_rows == [{k: v for k, v in row.items() if k != "condition"} for row in new_rows]
            assert outputs[0]["success"] is outputs[1]["success"] is True
            swapped_result = outputs[2]
            assert swapped_result["condition"] == "goal_swap" and "goal_swap" in Path(swapped_result["trace"]).name
            assert swapped_result["forecast_goal"] == SWAP_GOAL
            for line in Path(swapped_result["chunks"]).read_text().splitlines():
                chunk = json.loads(line)
                assert chunk["condition"] == "goal_swap" and chunk["forecast_goal"] == SWAP_GOAL
                assert chunk["policy_goal"] == chunk["value_goal"] == ORIGINAL_GOAL
            assert all(json.loads(line)["condition"] == "goal_swap" for line in Path(swapped_result["trace"]).read_text().splitlines())
    with tempfile.TemporaryDirectory() as directory:
        gate_path = Path(directory) / "gate.json"
        gate_fixture = {"status": "completed", "condition": "clean",
                        "protocol": {"suite": "libero_goal", "best_of_n": 2, "seed": 31, "env_seed": 0,
                                     "max_steps": 300, "settle_steps": 10, "offload": True},
                        "provenance": {"checkpoint_sha256": inference.MODEL_SHA256,
                                       "tokenizer_sha256": inference.TOKENIZER_SHA256,
                                       "helper_sha256": digest(Path(inference.__file__)),
                                       "runner_sha256": digest(Path(base.__file__)),
                                       "sampler_seeds": [31, 32], "action_steps": 5, "future_steps": 1, "value_steps": 1,
                                       "offload_helper_sha256": digest(ROOT / "scripts/cosmos_policy_cuda_offload.py"),
                                       "runtime_compatibility": expected_runtime_compatibility(),
                                       "source_sha256": {"scripts/cosmos_policy_torch_compat.py": digest(ROOT / "scripts/cosmos_policy_torch_compat.py")}},
                        "results": [{"task_id": 0, "init_id": 0, "status": "completed", "condition": "clean",
                                     "success": True, "success_at_start": False, "goal": ORIGINAL_GOAL,
                                     "initial_state_sha256": "0" * 64, "bddl_sha256": "1" * 64,
                                     "init_file_sha256": "2" * 64}]}
        atomic_json(gate_path, gate_fixture)
        assert validate_clean_gate(gate_path, True)["passed"]
        variants = []
        for location, key, replacement in (("protocol", "best_of_n", 1), ("protocol", "seed", 32),
                                           ("protocol", "offload", False), ("provenance", "helper_sha256", "stale"),
                                           ("provenance", "offload_helper_sha256", "stale"),
                                           ("provenance", "runtime_compatibility", {"backend": "changed"})):
            invalid = json.loads(json.dumps(gate_fixture))
            invalid[location][key] = replacement
            variants.append(invalid)
        invalid = json.loads(json.dumps(gate_fixture))
        invalid["results"][0]["success"] = "true"
        variants.append(invalid)
        invalid = json.loads(json.dumps(gate_fixture))
        invalid["status"] = "running"
        variants.append(invalid)
        invalid = json.loads(json.dumps(gate_fixture))
        invalid["provenance"]["source_sha256"]["scripts/cosmos_policy_codecs.py"] = "stale"
        variants.append(invalid)
        for invalid in variants:
            atomic_json(gate_path, invalid)
            try:
                validate_clean_gate(gate_path, True)
            except ValueError:
                pass
            else:
                raise AssertionError("Mismatched/incomplete Clean health gate was accepted")
    report = {"status": "passed", "device": "cpu", "gpu_used": False, "egl_used": False,
              "learned_forward_executed": False, "real_environment_constructed": False,
              "script_sha256": digest(Path(__file__)),
              "base_runner_sha256": digest(Path(base.__file__)),
              "helper_sha256": digest(Path(inference.__file__)),
              "checks": ["Clean selection calls and outputs match native best_of_n",
                         "31/32 candidate seeds and 5/1/1 steps match",
                         "Only forecast receives swapped text; proposals and value retain original goal",
                         "At matched synthetic observation, both candidate action tensors are unchanged",
                         "Clean full episode action/observation/reward traces match base runner under CPU environment stub",
                         "GoalSwap episode, filenames, action logs and chunk goal routing have accurate condition labels",
                         "Incomplete, N=1, changed seed/offload/runtime backend/dependency code and string-success Clean gates rejected"],
              "scope": "Synthetic contract and control-flow tests only. No task success, attack effect, CUDA equivalence or actual utility evidence."}
    if report_path:
        report_path = Path(report_path)
        if report_path.exists():
            raise FileExistsError(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(report_path, report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-report", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--clean-gate-summary", type=Path)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test(args.self_test_report)
        return
    if args.out_dir is None or args.video_stride < 0 or not 0 <= args.cuda_memory_fraction <= 1:
        parser.error("Require --out-dir, nonnegative video stride and valid allocator fraction")
    if len(args.conditions) != len(set(args.conditions)):
        parser.error("Duplicate conditions")
    args.suite, args.task_id, args.swap_task_id = "libero_goal", 0, 1
    args.init_ids, args.best_of_n, args.seed, args.env_seed = [0, 1, 2], 2, 31, 0
    args.max_steps, args.settle_steps, args.swap_goal = 300, 10, SWAP_GOAL
    args.out_dir = args.out_dir.resolve()
    gate = validate_clean_gate(args.clean_gate_summary, args.offload)
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task, swap_task = suite.get_task(0), suite.get_task(1)
    if task.language != ORIGINAL_GOAL or swap_task.language != SWAP_GOAL:
        raise ValueError("Benchmark language/order differs from the prespecified task0 -> Goal1 pair")
    states = suite.get_task_init_states(0)
    if len(states) < 3:
        raise ValueError("Need all three prespecified initial states")
    bddl = LIBERO / "libero/libero/bddl_files" / task.problem_folder / task.bddl_file
    init_path = LIBERO / "libero/libero/init_files" / task.problem_folder / task.init_states_file
    source_physics = gate["source_episode"]
    if (hashlib.sha256(np.asarray(states[0]).tobytes()).hexdigest() != source_physics["initial_state_sha256"]
            or digest(bddl) != source_physics["bddl_sha256"]
            or digest(init_path) != source_physics["init_file_sha256"]):
        raise ValueError("Physical task/initial states changed after the Clean health gate")
    text_path, stats_path = CKPT_DIR / "libero_t5_embeddings.pkl", CKPT_DIR / "libero_dataset_statistics.json"
    inference.verify_artifact(text_path, inference.TEXT_SHA256)
    with text_path.open("rb") as handle:
        text = pickle.load(handle)
    for goal in (ORIGINAL_GOAL, SWAP_GOAL):
        if goal not in text or not torch.isfinite(text[goal]).all():
            raise ValueError(f"Missing/nonfinite registered goal embedding: {goal}")
    if torch.equal(text[ORIGINAL_GOAL], text[SWAP_GOAL]):
        raise ValueError("Original/swap text embeddings unexpectedly identical")
    stats = inference.load_stats(stats_path)
    if digest(stats_path) != gate["provenance"]["stats_sha256"]:
        raise ValueError("Dataset statistics differ from the Clean health gate")
    provenance = dict(gate["provenance"])
    provenance.pop("execution_offload", None)
    provenance.pop("runtime_compatibility", None)
    provenance["expected_runtime_compatibility"] = expected_runtime_compatibility()
    provenance.update(runner_sha256=digest(Path(__file__)), base_runner_sha256=digest(Path(base.__file__)),
                      conditions=list(args.conditions), internal_intervention_only=True,
                      original_goal=ORIGINAL_GOAL, swap_goal=SWAP_GOAL,
                      goal_routing={"clean": {"policy": ORIGINAL_GOAL, "future": ORIGINAL_GOAL, "value": ORIGINAL_GOAL},
                                    "goal_swap": {"policy": ORIGINAL_GOAL, "future": SWAP_GOAL, "value": ORIGINAL_GOAL}},
                      clean_health_gate={k: v for k, v in gate.items() if k != "provenance"},
                      attack_surface_reachability_proven=False, bus_cut_implemented=False,
                      candidate_pairing="Same observation + seed gives matched proposals at a decision. Closed-loop states and later proposals may diverge after a different selected action; this is not a globally fixed-candidate ranking test.",
                      target_metric="Real LIBERO task0 check_success(); predicted value is only the native selection score.")
    planned = [{"task_id": 0, "init_id": i, "condition": c, "goal": ORIGINAL_GOAL,
                "forecast_goal": ORIGINAL_GOAL if c == "clean" else SWAP_GOAL}
               for i in args.init_ids for c in args.conditions]
    if args.preflight:
        print(json.dumps({"status": "cpu_preflight_passed", "planned_episodes": planned,
                          "provenance": provenance, "gpu_or_environment_started": False}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this paired native-task experiment")
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(args.out_dir.iterdir()):
        parser.error("Output directory is nonempty; existing results must be preserved")
    if args.cuda_memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    summary = {"status": "loading", "model": "NVIDIA Cosmos Policy LIBERO 40K (torch adapter)",
               "experiment": "full_task_forecast_goal_intervention", "provenance": provenance,
               "protocol": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "planned_episodes": planned, "results": []}
    path, started = args.out_dir / "summary.json", time.monotonic()
    atomic_json(path, summary)
    try:
        model, vae, sampler, rand = inference.load_model(device="cpu" if args.offload else "cuda")
        if os.environ.get("ALPHABRAIN_USE_NATIVE_TE") != "1":
            from cosmos_policy_torch_compat import compatibility_metadata
            provenance["runtime_compatibility"] = compatibility_metadata()
        else:
            provenance["runtime_compatibility"] = {"backend": "native Transformer Engine", "full_runtime_parity_verified": False}
        if args.offload:
            from cosmos_policy_cuda_offload import prepare
            model, vae, offload_manager, offload_metadata = prepare(model, vae)
            provenance["execution_offload"] = offload_metadata
            provenance["offload_helper_sha256"] = digest(ROOT / "scripts/cosmos_policy_cuda_offload.py")
            print(json.dumps({"offload_preflight": offload_metadata}), flush=True)
        summary["status"] = "running"
        for item in planned:
            summary["active_episode"] = item
            atomic_json(path, summary)
            result = run_episode(model, vae, sampler, rand, task, 0, item["init_id"], text,
                                 stats, args, provenance, item["condition"])
            summary["results"].append(result)
            summary["aggregate"] = aggregate_results(summary["results"])
            summary["elapsed_seconds"] = time.monotonic() - started
            atomic_json(path, summary)
        summary["status"] = "completed"
        summary.pop("active_episode", None)
    except BaseException as error:
        summary.update(status="error", exception_type=type(error).__name__, exception=str(error))
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        atomic_json(path, summary)
    print(json.dumps({"summary": str(path), "aggregate": summary["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()
