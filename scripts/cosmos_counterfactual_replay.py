#!/usr/bin/env python3
"""Independent real-environment outcomes for fixed Cosmos action candidates.

No model is imported by this module. Environment construction is lazy. Match
pilot_cosmos_policy_native_task_gate: fresh LIBERO reset, original init file,
256px cameras, ten [0,...,0,-1] settling actions, continuous controls unchanged.

capture_source_prefix(episode_json, prefix_steps, out_dir) -> snapshot.json Path
replay_candidates(episode_json, prefix_steps, candidates, out_dir,
                  repeat=2, continuation=None) -> utilities.json Path

The optional stateless continuation callback has this signature:
  continuation(obs, *, original_goal, absolute_step, remaining_steps, seed)
      -> real environment controls shaped [K, 7], 1 <= K <= 16.
It must use one fixed original-goal policy with explicit sampler seeds and no
candidate/attack-specific state. An optional reset() is called each replay.
Provide continuation_metadata with checkpoint/policy/seeding provenance.

Candidates always run all 16 steps to get an aligned physical forecast target.
Task utility uses first success (the native stop rule), and labels post-success
frames. Without continuation, unsuccessful short segments have unknown full-
task utility; no forecast score, geometric proxy or invented reward is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIBERO = ROOT / "vendor/LIBERO"
HASH_ALGORITHM = "sha256(json.dumps({'dtype':arr.dtype.str,'shape':list(arr.shape)},sort_keys=True,separators=(',',':')).encode()+b'\\0'+np.ascontiguousarray(arr).tobytes())"
OBS_KEYS = ("agentview_image", "robot0_eye_in_hand_image", "robot0_gripper_qpos",
            "robot0_eef_pos", "robot0_eef_quat")


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def array_sha(value):
    value = np.asarray(value)
    header = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\0" + np.ascontiguousarray(value).tobytes()).hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _fresh_dir(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_artifact(path, source):
    path = Path(path)
    if path.is_absolute():
        return path
    choices = [ROOT / path, source.parent / path, source.parent / path.name]
    for item in choices:
        if item.exists():
            return item.resolve()
    raise FileNotFoundError(path)


def _source(episode_json, prefix_steps):
    path = Path(episode_json).resolve()
    episode = json.loads(path.read_text())
    if episode.get("status") != "completed" or episode.get("success") is not True:
        raise ValueError("Source must be a completed, environment-successful native episode")
    if episode.get("condition") != "clean":
        raise ValueError("Source must be the original-goal Clean episode")
    if int(episode["settle_steps"]) != 10 or int(episode["max_steps"]) != 300:
        raise ValueError("This protocol requires native settle10 and max300")
    trace_path = _resolve_artifact(episode["trace"], path)
    trace = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    if not trace or [row["step"] for row in trace] != list(range(1, len(trace) + 1)):
        raise ValueError("Source trace has missing, duplicate or noncontiguous steps")
    if len(trace) != int(episode["environment_steps"]) or trace[-1].get("task_success") is not True:
        raise ValueError("Source trace does not match completed successful episode")
    if not isinstance(prefix_steps, int) or not 0 <= prefix_steps <= len(trace):
        raise ValueError("prefix_steps outside source trace")
    if prefix_steps + 16 > 300 or any(row["done"] or row["task_success"] for row in trace[:prefix_steps]):
        raise ValueError("Candidate prefix is terminal or lacks full16-step budget")
    actions = np.asarray([row["action"] for row in trace], dtype=np.float64)
    if actions.shape != (len(trace), 7) or not np.isfinite(actions).all():
        raise ValueError("Source has malformed/nonfinite controls")
    return {"episode": episode, "episode_path": path, "episode_sha256": file_sha(path),
            "trace": trace, "trace_path": trace_path, "trace_sha256": file_sha(trace_path),
            "actions": actions}


def _numeric_fields(obj, depth=1):
    """Copy available numeric state; explicitly identify unarchived attributes."""
    result, skipped = {}, []
    for key, value in vars(obj).items():
        if isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
            result[key] = value.copy()
        elif isinstance(value, (bool, int, float, np.number)):
            result[key] = np.asarray(value)
        elif isinstance(value, (list, tuple)):
            try:
                array = np.asarray(value)
            except (TypeError, ValueError):
                skipped.append(key)
                continue
            if array.dtype.kind in "biufc":
                result[key] = array.copy()
            else:
                skipped.append(key)
        elif value is None or isinstance(value, str):
            result[key] = value
        elif depth and "interpolator" in key and hasattr(value, "__dict__"):
            result[key] = _numeric_fields(value, depth=depth - 1)
        else:
            skipped.append(key)
    return {"values": result, "unarchived_attributes": sorted(skipped)}


def _physical_snapshot(env, obs):
    sim = env.sim
    data = sim.data
    physical = {"sim_state": np.asarray(env.get_sim_state()).copy(),
                "qpos": np.asarray(data.qpos).copy(), "qvel": np.asarray(data.qvel).copy()}
    for key in ("ctrl", "qacc", "qacc_warmstart", "act", "mocap_pos", "mocap_quat",
                "userdata", "qfrc_applied", "xfrc_applied", "body_xpos", "body_xquat"):
        if hasattr(data, key):
            physical[key] = np.asarray(getattr(data, key)).copy()
    physical["contacts"] = [
        {key: np.asarray(getattr(data.contact[i], key)).copy()
         for key in ("geom1", "geom2", "pos", "frame", "dist", "friction")
         if hasattr(data.contact[i], key)} for i in range(int(data.ncon))]
    physical["robots"] = [{"controller": _numeric_fields(robot.controller),
                           "gripper": _numeric_fields(robot.gripper)} for robot in env.robots]
    physical["observations"] = {key: np.asarray(value).copy() for key, value in obs.items()
                                if isinstance(value, (np.ndarray, np.number, int, float))
                                and np.asarray(value).dtype.kind in "biufc"}
    return physical


def _signature(value):
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": list(value.shape), "sha256": array_sha(value)}
    if isinstance(value, dict):
        return {key: _signature(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_signature(item) for item in value]
    return value


def _save_snapshot(directory, label, env, obs):
    physical = _physical_snapshot(env, obs)
    arrays = {}
    def flatten(value, prefix):
        if isinstance(value, np.ndarray):
            arrays[prefix] = value
            return {"npz_key": prefix}
        if isinstance(value, dict):
            return {key: flatten(item, f"{prefix}/{key}") for key, item in value.items()}
        if isinstance(value, list):
            return [flatten(item, f"{prefix}/{index}") for index, item in enumerate(value)]
        return value
    structure = flatten(physical, "physical")
    state_path = directory / f"{label}_state.npz"
    np.savez_compressed(state_path, **arrays)
    observations = {key: np.asarray(obs[key]).copy() for key in OBS_KEYS}
    observations["sim_state"] = physical["sim_state"]
    observation_path = directory / ("observation.npz" if label == "source" else f"{label}_observation.npz")
    np.savez_compressed(observation_path, **observations)
    return {"state_hash": _json_sha(_signature(physical)),
            "sim_state_sha256": array_sha(physical["sim_state"]),
            "state_npz": str(state_path), "state_npz_sha256": file_sha(state_path),
            "state_structure": structure, "state_signature": _signature(physical),
            "observations_npz": str(observation_path),
            "observations_npz_sha256": file_sha(observation_path),
            "observation_array_hashes": {key: array_sha(value) for key, value in observations.items()}}


def _verify_trace_observation(obs, trace, prefix_steps):
    checks = {}
    if prefix_steps < len(trace):
        checks["next_eef_before"] = float(np.max(np.abs(np.asarray(obs["robot0_eef_pos"]) - trace[prefix_steps]["eef_before"])))
    if prefix_steps:
        last = trace[prefix_steps - 1]
        for key, source_key in (("robot0_eef_pos", "eef_after"), ("robot0_eef_quat", "eef_quaternion"),
                                ("robot0_gripper_qpos", "gripper_qpos")):
            checks[source_key] = float(np.max(np.abs(np.asarray(obs[key]) - last[source_key])))
    if any(error != 0.0 for error in checks.values()):
        raise RuntimeError(f"Fresh replay differs from original native trace: {checks}")
    return checks


def _new_prefix_env(source, prefix_steps):
    """No state teleport: preserve controller history via fresh complete replay."""
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    if str(LIBERO) not in sys.path:
        sys.path.insert(0, str(LIBERO))
    import torch
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    episode = source["episode"]
    task = benchmark.get_benchmark_dict()[episode["suite"]]().get_task(int(episode["task_id"]))
    if task.name != episode["task_name"] or task.language != episode["goal"]:
        raise ValueError("Original task mapping or language changed")
    bddl = LIBERO / "libero/libero/bddl_files" / task.problem_folder / task.bddl_file
    init_path = LIBERO / "libero/libero/init_files" / task.problem_folder / task.init_states_file
    if file_sha(bddl) != episode["bddl_sha256"] or file_sha(init_path) != episode["init_file_sha256"]:
        raise ValueError("Original environment or initial-state file provenance changed")
    states = torch.load(init_path, map_location="cpu", weights_only=False)
    initial = np.asarray(states[int(episode["init_id"])])
    if hashlib.sha256(initial.tobytes()).hexdigest() != episode["initial_state_sha256"]:
        raise ValueError("Original selected initial state changed")
    random.seed(int(episode["seed"]))
    np.random.seed(int(episode["seed"]))
    torch.random.default_generator.manual_seed(int(episode["seed"]))
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
                             camera_widths=256, render_gpu_device_id=0)
    env.task_id, env.task_name, env.task_suite_name = episode["task_id"], task.name, episode["suite"]
    try:
        env.seed(int(episode["environment_seed"]))
        env.reset()
        obs = env.set_init_state(states[int(episode["init_id"])])
        for _ in range(10):
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
        if env.check_success():
            raise RuntimeError("Original task already successful before policy prefix")
        for index, action in enumerate(source["actions"][:prefix_steps]):
            obs, reward, done, _ = env.step(action.tolist())
            reference = source["trace"][index]
            if float(reward) != float(reference["reward"]) or bool(done) != bool(reference["done"]) or bool(env.check_success()) != bool(reference["task_success"]):
                raise RuntimeError(f"Source prefix outcome differs at step {index + 1}")
            _verify_trace_observation(obs, source["trace"], index + 1)
        checks = _verify_trace_observation(obs, source["trace"], prefix_steps)
        return env, obs, checks
    except BaseException:
        env.close()
        raise


def _protocol(source, prefix_steps):
    episode = source["episode"]
    keys = ("suite", "task_id", "init_id", "task_name", "goal", "seed", "environment_seed",
            "settle_steps", "max_steps", "bddl_sha256", "initial_state_sha256", "init_file_sha256")
    return {"episode_json": str(source["episode_path"]), "episode_sha256": source["episode_sha256"],
            "trace": str(source["trace_path"]), "trace_sha256": source["trace_sha256"],
            "prefix_steps": prefix_steps, **{key: episode[key] for key in keys},
            "hash_algorithm": HASH_ALGORITHM, "state_hash_definition": "sha256 of canonical recursive array signatures of archived simulator/controller/observations; unarchived attribute names are explicit",
            "camera_size": 256, "settle_action": [0.0] * 6 + [-1.0],
            "action_codec": "Environment controls unchanged; no normalization, clipping, binarization or sign inversion",
            "fresh_reset_and_full_prefix_replay": True,
            "helper_sha256": file_sha(__file__), "native_runner_sha256": file_sha(ROOT / "scripts/pilot_cosmos_policy_native_task_gate.py")}


def capture_source_prefix(episode_json, prefix_steps: int, out_dir) -> Path:
    source = _source(episode_json, prefix_steps)
    directory = _fresh_dir(out_dir)
    env, obs, checks = _new_prefix_env(source, prefix_steps)
    try:
        snapshot = {"status": "captured_original_native_prefix", **_protocol(source, prefix_steps),
                    **_save_snapshot(directory, "source", env, obs),
                    "source_trace_max_abs_errors": checks,
                    "task_success_at_prefix": bool(env.check_success())}
        snapshot.update(observation_npz=snapshot["observations_npz"],
                        observation_npz_sha256=snapshot["observations_npz_sha256"],
                        state_sha256=snapshot["state_hash"],
                        protocol={key: source["episode"][key] for key in (
                            "suite", "task_id", "init_id", "seed", "environment_seed", "settle_steps", "max_steps")})
        snapshot["protocol"]["original_goal"] = source["episode"]["goal"]
        path = directory / "snapshot.json"
        _write_json(path, snapshot)
        return path
    finally:
        env.close()


def _checked_actions(actions, *, fixed16):
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 7 or actions.dtype.kind != "f" or not np.isfinite(actions).all():
        raise ValueError("Expected finite floating environment controls [K,7]")
    if (fixed16 and actions.shape[0] != 16) or not 1 <= actions.shape[0] <= 16:
        raise ValueError("Candidates require 16 controls; continuation chunks require 1..16")
    return actions.copy()


def _step_row(env, obs, action, reward, done, absolute_step, phase):
    physical = _physical_snapshot(env, obs)
    object_poses = {key: value.tolist() for key, value in physical["observations"].items()
                    if (key.endswith("_pos") or key.endswith("_quat")) and not key.startswith("robot")}
    return {"absolute_step": absolute_step, "phase": phase, "action": action.tolist(),
            "reward": float(reward), "done": bool(done), "task_success": bool(env.check_success()),
            "eef_pos": np.asarray(obs["robot0_eef_pos"]).tolist(),
            "eef_quat": np.asarray(obs["robot0_eef_quat"]).tolist(), "object_poses": object_poses,
            "sim_state_sha256": array_sha(physical["sim_state"]),
            "qpos_sha256": array_sha(physical["qpos"]), "qvel_sha256": array_sha(physical["qvel"]),
            "controller_state_sha256": _json_sha(_signature(physical["robots"])),
            "state_hash": _json_sha(_signature(physical))}


def _one_replay(source, prefix_steps, actions, directory, continuation):
    env, obs, checks = _new_prefix_env(source, prefix_steps)
    try:
        initial = _save_snapshot(directory, "initial", env, obs)
        first_success = None
        first_success_snapshot = None
        rows = []
        absolute_step = prefix_steps
        if continuation is not None and hasattr(continuation, "reset"):
            continuation.reset()
        trace_path = directory / "trajectory.jsonl"
        with trace_path.open("w") as stream:
            def execute(action, phase):
                nonlocal obs, absolute_step, first_success, first_success_snapshot
                obs, reward, done, _ = env.step(action.tolist())
                absolute_step += 1
                row = _step_row(env, obs, action, reward, done, absolute_step, phase)
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                if row["task_success"] and first_success is None:
                    first_success = absolute_step
                    first_success_snapshot = _save_snapshot(directory, "first_success", env, obs)
                # LIBERO overrides done with task success. Reject unfamiliar
                # failure-terminal behavior instead of stepping blindly past it.
                if done and not row["task_success"]:
                    raise RuntimeError("Unexpected non-success terminal during fixed-horizon replay")
            for action in actions:
                execute(action, "candidate")
            future = _save_snapshot(directory, "candidate_future", env, obs)
            candidate_first_success = first_success
            if continuation is not None and first_success is None:
                while absolute_step < 300 and first_success is None:
                    chunk = _checked_actions(continuation(
                        {key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value for key, value in obs.items()},
                        original_goal=source["episode"]["goal"], absolute_step=absolute_step,
                        remaining_steps=300 - absolute_step, seed=int(source["episode"]["seed"])), fixed16=False)
                    for action in chunk[:300 - absolute_step]:
                        execute(action, "original_goal_continuation")
                        if first_success is not None:
                            break
        terminal = _save_snapshot(directory, "terminal", env, obs)
        observed = continuation is not None or first_success is not None or absolute_step >= 300
        return {"initial": initial, "candidate_future": future, "terminal": terminal,
                "source_trace_max_abs_errors": checks,
                "trajectory": str(trace_path), "trajectory_sha256": file_sha(trace_path),
                "trajectory_outcomes_sha256": _json_sha(rows),
                "candidate_steps_executed": 16, "candidate_first_success_step": candidate_first_success,
                "candidate_short_success": candidate_first_success is not None,
                "candidate_future_after_native_success_stop": candidate_first_success is not None and candidate_first_success < prefix_steps + 16,
                "first_success_snapshot": first_success_snapshot,
                "success": first_success is not None if observed else None,
                "steps_to_success": first_success, "steps_to_success_from_candidate": None if first_success is None else first_success - prefix_steps,
                "utility_observed_to_success_or_full_horizon": observed,
                "last_executed_absolute_step": absolute_step,
                "environment_reward_sum": sum(row["reward"] for row in rows),
                "scope": "Actual original-task first-success outcome; steps are absolute policy actions excluding settle. Reward sum is diagnostic only."}
    finally:
        env.close()


def replay_candidates(episode_json, prefix_steps: int, candidates, out_dir, repeat: int = 2,
                      continuation: Callable | None = None, *, continuation_metadata=None,
                      source_snapshot=None) -> Path:
    """Replay [N,16,7] controls, with no Goal/model/condition argument in utility."""
    if not isinstance(repeat, int) or repeat < 2:
        raise ValueError("At least two independent fresh replays are required")
    if continuation is not None and (not isinstance(continuation_metadata, dict) or not continuation_metadata):
        raise ValueError("Provide explicit fixed-policy provenance in continuation_metadata")
    source = _source(episode_json, prefix_steps)
    action_sets = [_checked_actions(item, fixed16=True) for item in candidates]
    if not action_sets:
        raise ValueError("No action candidates")
    expected = None
    if source_snapshot is not None:
        snapshot_path = Path(source_snapshot).resolve()
        expected = json.loads(snapshot_path.read_text())
        if expected["episode_sha256"] != source["episode_sha256"] or expected["prefix_steps"] != prefix_steps:
            raise ValueError("Source snapshot does not match this episode/prefix")
        observation_file = Path(expected["observations_npz"])
        if file_sha(observation_file) != expected["observations_npz_sha256"]:
            raise ValueError("Source observation artifact changed")
    directory = _fresh_dir(out_dir)
    path = directory / "utilities.json"
    summary = {"status": "running", **_protocol(source, prefix_steps), "repeat": repeat,
               "continuation_enabled": continuation is not None, "continuation_metadata": continuation_metadata,
               "source_snapshot": None if source_snapshot is None else str(Path(source_snapshot).resolve()),
               "source_snapshot_sha256": None if source_snapshot is None else file_sha(source_snapshot),
               "source_observations_npz_sha256": None if expected is None else expected["observations_npz_sha256"],
               "candidate_goal_assignment": "No goal inputs accepted: all environment/continuation goals remain original",
               "primary_utility": "Original environment first-success by absolute step300, else unknown if continuation absent",
               "secondary_utility": "Fewer absolute steps to success among successes; failed/unknown episodes have no fabricated score",
               "candidate_target_horizon": 16, "candidate_stop_on_success": False,
               "repeat_passed": False, "candidates": {}, "api_requests": 0}
    _write_json(path, summary)
    baseline_initial = expected["state_hash"] if expected is not None else None
    try:
        for index, actions in enumerate(action_sets):
            candidate_dir = directory / f"candidate_{index:02d}"
            candidate_dir.mkdir()
            action_path = candidate_dir / "actions.npy"
            np.save(action_path, actions, allow_pickle=False)
            row = {"index": index, "actions": str(action_path), "action_sha256": array_sha(actions),
                   "actions_file_sha256": file_sha(action_path), "dtype": actions.dtype.str, "replays": []}
            summary["candidates"][str(index)] = row
            for repetition in range(repeat):
                replay_dir = candidate_dir / f"repeat_{repetition:02d}"
                replay_dir.mkdir()
                result = _one_replay(source, prefix_steps, actions, replay_dir, continuation)
                row["replays"].append(result)
                if baseline_initial is None:
                    baseline_initial = result["initial"]["state_hash"]
                if result["initial"]["state_hash"] != baseline_initial:
                    raise RuntimeError("Initial simulator/controller/observation state differs across replay/candidate/source")
                first = row["replays"][0]
                if any(result[part]["state_hash"] != first[part]["state_hash"] for part in ("candidate_future", "terminal")):
                    raise RuntimeError("Repeated candidate future or continuation terminal state is not exact")
                if result["trajectory_outcomes_sha256"] != first["trajectory_outcomes_sha256"]:
                    raise RuntimeError("Repeated controls/outcomes/physical trajectory differs")
                _write_json(path, summary)
            first = row["replays"][0]
            row.update({key: first[key] for key in ("success", "steps_to_success", "steps_to_success_from_candidate",
                        "candidate_short_success", "utility_observed_to_success_or_full_horizon")})
            row["repeat_initial_future_terminal_trajectory_exact"] = True
        summary["source_state_hash"] = baseline_initial
        observed = [row for row in summary["candidates"].values() if row["utility_observed_to_success_or_full_horizon"]]
        summary["full_task_utility_complete"] = len(observed) == len(action_sets)
        success_values = [row["success"] for row in observed]
        summary["primary_success_distinguishes_candidates"] = len(set(success_values)) > 1
        summary["secondary_success_steps_distinguish_candidates"] = len({row["steps_to_success"] for row in observed if row["success"]}) > 1
        summary["ranking_information"] = ("Full-horizon actual utilities available; tied outcomes remain ties"
                                           if summary["full_task_utility_complete"] else
                                           "Incomplete full-task utility: zero short-horizon success cannot establish candidate ranking")
        summary.update(status="completed_real_environment_candidate_replays", repeat_passed=True)
    except BaseException as error:
        summary.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        _write_json(path, summary)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "replay"):
        option = sub.add_parser(name)
        option.add_argument("--episode-json", type=Path, required=True)
        option.add_argument("--prefix-steps", type=int, required=True)
        option.add_argument("--out-dir", type=Path, required=True)
        if name == "replay":
            option.add_argument("--actions", type=Path, nargs="+", required=True)
            option.add_argument("--repeat", type=int, default=2)
            option.add_argument("--source-snapshot", type=Path)
    args = parser.parse_args()
    if args.command == "capture":
        result = capture_source_prefix(args.episode_json, args.prefix_steps, args.out_dir)
    else:
        arrays = [np.load(path, allow_pickle=False) for path in args.actions]
        result = replay_candidates(args.episode_json, args.prefix_steps, arrays, args.out_dir,
                                   args.repeat, source_snapshot=args.source_snapshot)
    print(result)


if __name__ == "__main__":
    main()
