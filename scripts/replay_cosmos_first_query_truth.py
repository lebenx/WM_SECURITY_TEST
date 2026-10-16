#!/usr/bin/env python3
"""Replay two frozen Cosmos first chunks; no policy, VAE, or model loading.

Fresh LIBERO reset/initialization/settling matches the archived branches.
Only successful exact prefix reproduction qualifies the new endpoint images.
Physics runs on CPU; EGL rendering may use GPU graphics resources.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/cosmos_first_query_truth_countercheck_20261015.json"
OBS_KEYS = ("agentview_image", "robot0_eye_in_hand_image", "robot0_gripper_qpos",
            "robot0_eef_pos", "robot0_eef_quat")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def resolve(path):
    return (ROOT / path).resolve()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def exact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def copy_observation(obs):
    result = {key: np.asarray(obs[key]).copy() for key in OBS_KEYS}
    for key in OBS_KEYS[:2]:
        require(result[key].dtype == np.uint8 and result[key].shape == (256, 256, 3),
                f"Unexpected raw camera format: {key}")
    for key, shape in zip(OBS_KEYS[2:], ((2,), (3,), (4,))):
        require(result[key].shape == shape and np.isfinite(result[key]).all(),
                f"Unexpected proprio format: {key}")
    return result


def frame(obs):
    return np.concatenate([np.ascontiguousarray(obs[key][::-1]) for key in OBS_KEYS[:2]], axis=1)


def read_source(config, index):
    paths = config["sources"][f"forced{index}"]
    episode = json.loads(resolve(paths["episode"]).read_text())
    with resolve(paths["first_query"]).open() as handle:
        first = json.loads(next(handle))
    with resolve(paths["trace"]).open() as handle:
        trace = [json.loads(next(handle)) for _ in range(16)]
    expected = {"status": "completed", "condition": "clean", "suite": config["suite"],
                "task_id": 3, "init_id": 2, "goal": config["original_goal"],
                "seed": 31, "environment_seed": 0, "settle_steps": 10,
                "best_of_n": 2, "success_at_start": False}
    require(all(episode.get(k) == v for k, v in expected.items()), "Historical branch protocol changed")
    require(first["chunk"] == first["first_step"] == 1, "Wrong historical first query")
    selection = first["native_best_of_n"]
    require(selection["selected_index"] == index, "Historical forced branch index differs")
    require([c["index"] for c in selection["candidates"]] == [0, 1]
            and [c["seed"] for c in selection["candidates"]] == [31, 32], "Candidate identity differs")
    actions = np.asarray(selection["candidates"][index]["environment_actions"], dtype=np.float64)
    require(actions.shape == (16, 7) and np.isfinite(actions).all(), "Malformed fixed controls")
    require(exact(actions, np.asarray(first["environment_actions"], dtype=np.float64)),
            "Selected controls do not match fixed candidate")
    for offset, old in enumerate(trace):
        require(old["step"] == offset + 1 and old["chunk"] == 1
                and exact(actions[offset], np.asarray(old["action"], dtype=np.float64)),
                "Historical trace/action binding differs")
    states_path = resolve(paths["trace"]).with_name(resolve(paths["trace"]).stem + "_states.npz")
    with np.load(states_path, allow_pickle=False) as archive:
        states = {key: archive[key].copy() for key in ("initial", "settled")}
    return episode, first, trace, actions, states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-seconds", type=float, default=300.0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    frozen = {"task_id": 3, "init_id": 2, "suite": "libero_goal", "seed": 31,
              "environment_seed": 0, "settle_steps": 10, "horizon_controls": 16,
              "camera_size": 256, "candidate_indices": [0, 1], "candidate_seeds": [31, 32]}
    require(all(config.get(k) == v for k, v in frozen.items()), "Unregistered task or replay budget")
    require(config["status"] == "frozen_before_new_replay_or_model_execution", "Protocol is not frozen")
    require(tuple(config["replay_cache_schema"]["keys"]) == OBS_KEYS, "Cache schema differs")
    require(0 < args.max_seconds <= 300, "Replay deadline must be in (0,300] seconds")
    output = (args.out or resolve(config["replay_output"])).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "launched_script.py")
    shutil.copy2(args.config, output / "launched_config.json")
    started = time.monotonic()
    summary = {
        "schema_version": 1, "status": "starting", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": relative(args.config), "output": relative(output), "task_id": 3, "init_id": 2,
        "sources": config["sources"], "candidates": [], "active_candidate": None,
        "qualification": {"passed": False}, "initial_obs": None,
        "activity": {"explicit_task_controls": 0, "explicit_settling_controls": 0,
                     "environments_created": 0, "reset_calls": 0, "set_init_state_calls": 0,
                     "model_forward_calls": 0, "vae_calls": 0, "model_api_calls": 0,
                     "checkpoint_loads": 0, "training_updates": 0, "cuda_tensor_compute": False,
                     "egl_rendering": True,
                     "internal_initialization_steps": "Not instrumented; reset/set_init_state may step internally."},
        "scope": "New lossless endpoints after exact reproduction of two existing16-control branches; no new full-task outcomes.",
        "endpoint_pixel_limit": "No historical lossless endpoint exists; only initial pixels and all stored physical prefix fields can be checked exactly.",
    }
    summary_path = output / "summary.json"
    write_json(summary_path, summary)

    def timed_out(signum, _frame):
        raise TimeoutError(f"Replay interrupted by signal {signum}")

    signal.signal(signal.SIGALRM, timed_out)
    signal.signal(signal.SIGTERM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    env = None
    exit_code = 1
    try:
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
        os.environ.setdefault("MUJOCO_GL", "egl")
        require(os.environ["MUJOCO_GL"] == "egl", "Historical renderer was EGL")
        sys.path.insert(0, str(ROOT / "vendor/LIBERO"))
        import torch
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv

        require(torch.__version__ == config["torch_version"] and np.__version__ == config["numpy_version"],
                "Torch/NumPy runtime differs from frozen physical replay")
        require(not torch.cuda.is_initialized(), "No CUDA tensor context allowed in replay process")
        summary["runtime"] = {"python": sys.version, "executable": sys.executable,
                              "torch": torch.__version__, "numpy": np.__version__, "MUJOCO_GL": "egl"}
        task = benchmark.get_benchmark_dict()[config["suite"]]().get_task(3)
        require(task.language == config["original_goal"], "Task mapping changed")
        task_base = ROOT / "vendor/LIBERO/libero/libero"
        bddl = task_base / "bddl_files" / task.problem_folder / task.bddl_file
        init_path = task_base / "init_files" / task.problem_folder / task.init_states_file
        init_states = torch.load(init_path, map_location="cpu", weights_only=False)
        summary["init_state_file"] = relative(init_path)
        summary["init_state_file_is_model_checkpoint"] = False
        common_obs = None
        first_candidates = None
        for index in (0, 1):
            episode, first, old_trace, actions, old_states = read_source(config, index)
            require(task.name == episode["task_name"] and bddl == Path(episode["bddl"]), "Task identity differs")
            if first_candidates is None:
                first_candidates = first["native_best_of_n"]["candidates"]
            require(first["native_best_of_n"]["candidates"] == first_candidates,
                    "Two branches do not share the original candidate bank")
            row = {"candidate_index": index, "candidate_seed": 31 + index, "status": "starting",
                   "historical_trace_exact_steps": 0, "initial_checks": {}, "endpoint_obs": None,
                   "source_episode": config["sources"][f"forced{index}"]["episode"],
                   "historical_full_continuation": {"success": episode["success"],
                                                     "environment_steps": episode["environment_steps"],
                                                     "measured_in_this_replay": False}}
            summary["candidates"].append(row)
            summary["active_candidate"] = index
            summary["status"] = "running"
            write_json(summary_path, summary)
            random.seed(31)
            np.random.seed(31)
            torch.random.default_generator.manual_seed(31)
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
                                     camera_widths=256, render_gpu_device_id=0)
            summary["activity"]["environments_created"] += 1
            env.task_id, env.task_name, env.task_suite_name = 3, task.name, config["suite"]
            env.seed(0)
            env.reset()
            summary["activity"]["reset_calls"] += 1
            obs = env.set_init_state(init_states[2])
            summary["activity"]["set_init_state_calls"] += 1
            initial_sim = np.asarray(env.get_sim_state()).copy()
            row["initial_checks"]["initial_state_exact"] = exact(initial_sim, old_states["initial"])
            write_json(summary_path, summary)
            require(row["initial_checks"]["initial_state_exact"], "Initial simulation state mismatch")
            for _ in range(10):
                obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                summary["activity"]["explicit_settling_controls"] += 1
            settled_sim = np.asarray(env.get_sim_state()).copy()
            observed = copy_observation(obs)
            image_path = output / f"candidate{index}_initial.png"
            Image.fromarray(frame(observed)).save(image_path)
            with Image.open(resolve(episode["initial_image"])) as image:
                reference_pixels = np.asarray(image).copy()
            checks = row["initial_checks"]
            checks.update(settled_state_exact=exact(settled_sim, old_states["settled"]),
                          initial_png_pixels_exact=exact(frame(observed), reference_pixels),
                          initial_eef_exact=exact(observed["robot0_eef_pos"], np.asarray(old_trace[0]["eef_before"], dtype=np.float64)),
                          initial_success_matches=(bool(env.check_success()) == episode["success_at_start"] == False))
            row["initial_png_file_bytes_equal"] = image_path.read_bytes() == resolve(episode["initial_image"]).read_bytes()
            np.savez_compressed(output / f"candidate{index}_states.npz", initial=initial_sim, settled=settled_sim)
            if common_obs is not None:
                checks["common_initial_obs_exact"] = all(exact(observed[k], common_obs[k]) for k in OBS_KEYS)
            write_json(summary_path, summary)
            require(all(checks.values()), f"Candidate{index} initial physical/pixel parity failed")
            if common_obs is None:
                common_obs = observed
                initial_path = output / config["replay_cache_schema"]["initial"]
                np.savez_compressed(initial_path, **common_obs)
                summary["initial_obs"] = relative(initial_path)
            np.save(output / f"candidate{index}_actions.npy", actions, allow_pickle=False)
            trace_path = output / f"candidate{index}_trace.jsonl"
            row["trace"] = relative(trace_path)
            with trace_path.open("x") as handle:
                for offset, action in enumerate(actions):
                    before = np.asarray(obs["robot0_eef_pos"]).copy()
                    obs, reward, done, _ = env.step(action.tolist())
                    summary["activity"]["explicit_task_controls"] += 1
                    event = {"step": offset + 1, "chunk": 1, "action": action.tolist(),
                             "eef_before": before.tolist(), "eef_after": np.asarray(obs["robot0_eef_pos"]).tolist(),
                             "eef_quaternion": np.asarray(obs["robot0_eef_quat"]).tolist(),
                             "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"]).tolist(),
                             "reward": float(reward), "done": bool(done), "task_success": bool(env.check_success())}
                    handle.write(json.dumps(event, allow_nan=False) + "\n")
                    handle.flush()
                    mismatches = [k for k in set(event) | set(old_trace[offset]) if event.get(k) != old_trace[offset].get(k)]
                    row["last_mismatched_fields"] = mismatches
                    if not mismatches:
                        row["historical_trace_exact_steps"] += 1
                    write_json(summary_path, summary)
                    require(not mismatches, f"Candidate{index} step{offset + 1} differs: {mismatches}")
                    require(offset == 15 or not (event["done"] or event["task_success"]), "Unexpected early termination")
            endpoint = output / config["replay_cache_schema"]["endpoints"][index]
            np.savez_compressed(endpoint, **copy_observation(obs))
            Image.fromarray(frame(obs)).save(output / f"candidate{index}_endpoint.png")
            row.update(status="completed_exact16", endpoint_obs=relative(endpoint),
                       task_controls=16, last_step=event)
            env.close()
            env = None
            write_json(summary_path, summary)
        require(summary["activity"]["explicit_task_controls"] == 32
                and summary["activity"]["explicit_settling_controls"] == 20, "Replay budget not exactly completed")
        require(not torch.cuda.is_initialized(), "Unexpected CUDA tensor initialization")
        summary["status"] = "completed_exact_two_candidate_prefix_replay"
        summary["qualification"] = {"passed": True, "both_initial_observations_exact": True,
                                    "each_historical_prefix_exact_steps": 16,
                                    "historical_endpoint_pixel_equality_measured": False}
        exit_code = 0
    except BaseException as error:
        summary["status"] = "resource_incomplete" if isinstance(error, TimeoutError) else "invalid_replay_or_technical_error"
        summary["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        summary["qualification"]["passed"] = False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if env is not None:
            try:
                env.close()
            except Exception as error:
                summary["close_error"] = str(error)
        summary["active_candidate"] = None
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["exit_code"] = exit_code
        write_json(summary_path, summary)
        print(json.dumps({"status": summary["status"], "qualification": summary["qualification"],
                          "activity": summary["activity"], "output": relative(summary_path)}, allow_nan=False), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
