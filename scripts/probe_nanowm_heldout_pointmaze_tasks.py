#!/usr/bin/env python3
"""Frozen CPU-only PointMaze held-out task-selection and reference replay gate.

No model, planner, training, or GPU execution is performed. The only new download
is the official states.pth ZIP member; existing actions, lengths and split are
reused. Earlier random-state protocols and results are never changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/nano-world-model"
REVISION = "cc1f151a9d8c7d7afd0cdc5de6a9ae220a7c90c0"
PROTOCOL = ROOT / "reports/nanowm_heldout_task_feasibility_protocol_20261015.json"
OUTPUT = ROOT / "runs/nanowm_heldout_pointmaze_tasks_v1"
RESOURCE = ROOT / "runs/nanowm_resource_gate_v2"


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def small_sha(path):
    path = Path(path)
    if path.stat().st_size > 32 * 1024**2:
        raise ValueError("Refuse checkpoint-sized hashing")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_states(out, summary, persist):
    from probe_nanowm_pointmaze_resources import RangeFile, ARCHIVE_URL, ARCHIVE_SIZE

    remote = RangeFile()
    directory = out / "source_metadata"
    directory.mkdir()
    record = {"status": "started", "official_archive_url": ARCHIVE_URL,
              "archive_bytes": ARCHIVE_SIZE, "video_members_downloaded": 0,
              "new_member_names": [], "downloaded_bytes": 0, "range_requests": []}
    summary["download"] = record
    persist()
    try:
        with zipfile.ZipFile(remote) as archive:
            matches = [name for name in archive.namelist()
                       if Path(name).name == "states.pth" and "__MACOSX" not in name]
            if matches != ["point_maze/states.pth"]:
                raise RuntimeError(f"Unexpected states member selection: {matches}")
            info = archive.getinfo(matches[0])
            if info.file_size > 32 * 1024**2 or info.compress_size > 16 * 1024**2:
                raise RuntimeError("States metadata member exceeds frozen transfer limit")
            payload = archive.read(info)  # ZIP CRC is checked by ZipFile.
            path = directory / "states.pth"
            path.write_bytes(payload)
            record.update(status="completed", new_member_names=[info.filename],
                          path=relative(path), bytes=len(payload),
                          sha256=small_sha(path), zip_crc32=info.CRC)
        return path
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record.update(downloaded_bytes=remote.bytes_received,
                      range_requests=remote.requests)
        persist()


def freeze_selection(states_path, out, summary):
    import numpy as np
    import torch

    resource = json.loads((RESOURCE / "summary.json").read_text())
    split_path = RESOURCE / "trajectory_split.json"
    split = json.loads(split_path.read_text())
    if resource["status"] != "completed_cpu_resource_probe" or resource["source_revision"] != REVISION:
        raise RuntimeError("Required completed resource gate is unavailable")
    if split["seed"] != 42 or split["ratio"] != 0.9 or not split["disjoint"]:
        raise RuntimeError("Published training split differs from the frozen protocol")
    train_ids = set(split["train_ids"])
    val_ids = set(split["validation_ids"])
    if len(train_ids) != 1800 or len(val_ids) != 200 or train_ids.intersection(val_ids):
        raise RuntimeError("Invalid published-config trajectory partition")
    actions_path = RESOURCE / "data/point_maze/actions.pth"
    lengths_path = RESOURCE / "data/point_maze/seq_lengths.pth"
    states = torch.load(states_path, map_location="cpu", weights_only=True).float()
    actions = torch.load(actions_path, map_location="cpu", weights_only=True).float()
    lengths = torch.as_tensor(torch.load(lengths_path, map_location="cpu", weights_only=True)).tolist()
    if (states.ndim != 3 or states.shape[-1] != 4 or actions.ndim != 3
            or actions.shape[-1] != 2 or len(states) != len(actions)
            or len(states) != len(lengths)):
        raise RuntimeError(f"Unsupported metadata shapes: states={states.shape}, actions={actions.shape}")
    if len(states) != 2000 or train_ids.union(val_ids) != set(range(len(states))):
        raise RuntimeError("Split does not cover exactly the source trajectories")
    if not bool(torch.isfinite(states).all()) or not bool(torch.isfinite(actions).all()):
        raise RuntimeError("Source state or action metadata is nonfinite")
    if any(int(n) != n or n < 1 or n > states.shape[1] or n > actions.shape[1] for n in lengths):
        raise RuntimeError("Invalid sequence lengths")
    if resource["statistics"]["action_scale"] != 1.0:
        raise RuntimeError("Unexpected raw action scaling")
    cache = ROOT / resource["statistics"]["cache_path"]
    if not cache.is_file():
        raise RuntimeError("Previously audited training statistics are missing")
    valid = [i for i, n in enumerate(lengths) if i in val_ids and n >= 26]
    if not valid:
        raise RuntimeError("No eligible held-out source trajectories")
    rng = random.Random(42)
    rows = []
    for index in range(3):
        trajectory_id = rng.choice(valid)
        offset = rng.randint(0, int(lengths[trajectory_id]) - 26)
        initial = states[trajectory_id, offset].numpy().copy()
        reference = actions[trajectory_id, offset:offset + 25].numpy().copy()
        if initial.shape != (4,) or reference.shape != (25, 2):
            raise RuntimeError("Selected native goal window has an unsupported shape")
        directory = out / f"instance_{index:03d}"
        audit = directory / "reference_audit"
        audit.mkdir(parents=True)
        action_path = audit / "source_actions.npy"
        np.save(action_path, reference, allow_pickle=False)
        rows.append({"instance": index, "seed": 42 + index,
                     "trajectory_id": int(trajectory_id), "offset": int(offset),
                     "source_seq_length": int(lengths[trajectory_id]),
                     "source_split": "validation", "init_state": initial.tolist(),
                     "source_paths": {"states": relative(states_path),
                                      "actions": relative(actions_path),
                                      "lengths": relative(lengths_path),
                                      "split": relative(split_path)},
                     "reference_actions_audit_only": relative(action_path),
                     "goal_state": None, "initial_success": None,
                     "reference_verification": {"status": "not_run"},
                     "zero": {"status": "not_run"}, "clean": {"status": "not_run"}})
    summary["source_selection"] = {
        "pool_order": "ascending original global trajectory ID",
        "eligible_validation_trajectory_ids": valid,
        "eligible_count": len(valid), "rng": "random.Random(42)",
        "call_order": "choice then randint per instance, repeated exactly three times",
        "sampling_with_replacement": True,
        "training_statistics_reused": resource["statistics"]["cache_path"],
        "statistics_binding_limit": resource["statistics"]["training_run_binding_limit"]}
    write_json(out / "frozen_source_selection.json", {"source_selection": summary["source_selection"], "instances": rows})
    return rows


def simulator_state(env):
    return {"qpos": env.sim.data.qpos.copy().tolist(),
            "qvel": env.sim.data.qvel.copy().tolist(),
            "qacc_warmstart": env.sim.data.qacc_warmstart.copy().tolist(),
            "time": float(env.sim.data.time)}


def reference_rollout(env, row, actions):
    import numpy as np

    initial = np.asarray(row["init_state"], dtype=np.float32)
    obs, current = env.prepare(row["seed"], initial.copy())
    physical = np.concatenate([env.sim.data.qpos, env.sim.data.qvel])
    if not np.array_equal(current, initial) or not np.array_equal(physical, initial):
        raise RuntimeError("Reference native reset failed to restore source state")
    if obs["visual"].shape != (224, 224, 3) or obs["visual"].dtype != np.uint8:
        raise RuntimeError("Unexpected native rendered observation")
    frames = [obs["visual"].copy()]
    trace = [{"physical_step": 0, "state": np.asarray(current).tolist(),
              "simulator": simulator_state(env)}]
    for step, action in enumerate(actions, 1):
        obs, reward, done, info = env.step(action.copy())
        current = np.asarray(info["state"])
        if current.shape != (4,) or not np.isfinite(current).all() or done:
            raise RuntimeError("Reference replay stopped or produced invalid state")
        frames.append(obs["visual"].copy())
        trace.append({"physical_step": step, "state": current.tolist(),
                      "submitted_action": action.tolist(), "clipped_action": np.clip(action, -1, 1).tolist(),
                      "native_reward_audit_only": float(reward), "done": bool(done),
                      "simulator": simulator_state(env)})
    return trace, np.stack(frames)


def establish_goals(env, rows, out, persist):
    import numpy as np
    from PIL import Image

    for row in rows:
        directory = out / f"instance_{row['instance']:03d}"
        actions = np.load(ROOT / row["reference_actions_audit_only"], allow_pickle=False)
        row["reference_verification"] = {"status": "running"}
        persist()
        first, frames = reference_rollout(env, row, actions)
        write_json(directory / "reference_audit/build_trace.json", first)
        goal = np.asarray(first[-1]["state"], dtype=np.float32)
        row["goal_state"] = goal.tolist()
        row["initial_simulator_state"] = first[0]["simulator"]
        row["initial_image"] = relative(directory / "initial.png")
        row["goal_image"] = relative(directory / "goal.png")
        Image.fromarray(frames[0]).save(ROOT / row["initial_image"])
        Image.fromarray(frames[-1]).save(ROOT / row["goal_image"])
        initial = np.asarray(row["init_state"], dtype=np.float32)
        initial_metric = env.eval_state(goal, initial)
        row["initial_success"] = bool(initial_metric["success"])
        row["initial_xy_distance"] = float(np.linalg.norm(goal[:2] - initial[:2]))
        row["initial_state_dist"] = float(initial_metric["state_dist"])
        persist()
        second, verify_frames = reference_rollout(env, row, actions)
        write_json(directory / "reference_audit/verify_trace.json", second)
        same_states = first == second
        same_pixels = np.array_equal(frames, verify_frames)
        final = env.eval_state(goal, np.asarray(second[-1]["state"]))
        row["reference_verification"] = {
            "status": "completed" if same_states and same_pixels and final["success"] else "failed",
            "physical_steps_per_replay": 25, "replays": 2,
            "all_recorded_states_rewards_done_and_simulator_equal": same_states,
            "all_rendered_frames_equal": same_pixels,
            "final_success": bool(final["success"]), "final_state_dist": float(final["state_dist"]),
            "source_is_actual_environment_replay": True,
            "reachability_scope": "Constructive feasibility under these25 source actions, not planner-selected actions."}
        Image.fromarray(verify_frames[-1]).save(directory / "reference_audit/verify_final.png")
        persist()
        if not same_states or not same_pixels or not final["success"]:
            raise RuntimeError("Independent reference replay did not reproduce the frozen goal")
        print(f"REFERENCE instance={row['instance']} trajectory={row['trajectory_id']} offset={row['offset']} initial_success={row['initial_success']} verified=True", flush=True)
    keys = ("instance", "seed", "trajectory_id", "offset", "source_split", "init_state",
            "goal_state", "initial_image", "goal_image")
    targets = [{key: row[key] for key in keys} for row in rows]
    if "action" in json.dumps(targets).lower() or "reference_audit" in json.dumps(targets):
        raise RuntimeError("Reference actions leaked into planner target artifact")
    write_json(out / "planner_targets.json", targets)


def zero_controls(env, rows, out, persist):
    import numpy as np
    from PIL import Image

    for row in rows:
        directory = out / f"instance_{row['instance']:03d}/zero"
        directory.mkdir()
        initial = np.asarray(row["init_state"], dtype=np.float32)
        goal = np.asarray(row["goal_state"], dtype=np.float32)
        obs, current = env.prepare(row["seed"], initial.copy())
        initial_pixels = np.asarray(Image.open(ROOT / row["initial_image"]))
        if (not np.array_equal(current, initial) or simulator_state(env) != row["initial_simulator_state"]
                or not np.array_equal(obs["visual"], initial_pixels)):
            raise RuntimeError("Zero-control reset differs from frozen reference state or image")
        row["zero"] = {"status": "running"}
        persist()
        trace, checkpoints = [], []
        success_step = None
        for step in range(1, 51):
            action = np.zeros(2, dtype=np.float32)
            obs, reward, done, info = env.step(action)
            current = np.asarray(info["state"])
            if current.shape != (4,) or not np.isfinite(current).all() or done:
                raise RuntimeError("Unexpected zero-control terminal or nonfinite state")
            metric = env.eval_state(goal, current)
            trace.append({"physical_step": step, "state": current.tolist(),
                          "physical_action": action.tolist(), "native_reward_audit_only": float(reward),
                          "done": bool(done), "simulator": simulator_state(env),
                          "xy_distance": float(np.linalg.norm(goal[:2] - current[:2])),
                          "state_dist": float(metric["state_dist"]),
                          "is_native_success_checkpoint": step % 25 == 0})
            if step % 25 == 0:
                checkpoints.append({"physical_step": step, "success": bool(metric["success"]),
                                    "state_dist": float(metric["state_dist"]),
                                    "xy_distance": trace[-1]["xy_distance"]})
                if metric["success"]:
                    success_step = step
                    break
        write_json(directory / "physical_trace.json", trace)
        write_json(directory / "native_checkpoints.json", checkpoints)
        Image.fromarray(obs["visual"]).save(directory / "final.png")
        row["zero"] = {"status": "completed", "success": success_step is not None,
                       "physical_steps": len(trace), "success_physical_step": success_step,
                       "final_state": current.tolist(), "state_dist": float(metric["state_dist"]),
                       "xy_distance": trace[-1]["xy_distance"],
                       "capped_completion_cost": success_step if success_step is not None else 50,
                       "termination": "success_checkpoint" if success_step is not None else "budget",
                       "directory": relative(directory)}
        persist()
        print(f"ZERO instance={row['instance']} success={row['zero']['success']} physical_steps={len(trace)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    from pilot_jepa_native_task_gate import configure_process
    configure_process("cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(ROOT / ".deps/nanowm_runtime"))
    out = args.out_dir.resolve()
    if out != OUTPUT or out.exists():
        parser.error("Use the frozen new output path exactly once; preserve earlier attempts")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["status"] != "frozen_before_task_selection_and_environment_execution":
        parser.error("The preregistered protocol is missing or changed")
    out.mkdir(parents=True)
    summary = {"schema_version": 1, "status": "running_cpu_task_selection_gate",
               "protocol": {"path": relative(PROTOCOL), "sha256": small_sha(PROTOCOL)},
               "script": {"path": relative(__file__), "sha256": small_sha(__file__)},
               "source_revision": subprocess.check_output(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip(),
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "instances": [], "task_selection_gate_passed": None, "clean_gate_passed": None,
               "execution": {"model_weights_loaded": 0, "model_forwards": 0, "planner_calls": 0,
                             "gpu_used": False, "training_updates": 0, "model_api_calls": 0}}
    env = None
    rows = []
    def persist():
        summary["instances"] = rows
        write_json(out / "summary.json", summary)
    persist()
    try:
        if summary["source_revision"] != REVISION or protocol["source_revision"] != REVISION:
            raise RuntimeError("Native source revision differs from the frozen protocol")
        states_path = download_states(out, summary, persist)
        rows.extend(freeze_selection(states_path, out, summary))
        persist()
        # No environment is created before all source IDs and offsets are frozen.
        import torch
        torch.set_num_threads(1)
        from pilot_jepa_native_task_gate import isolate_d4rl_offline_env, prepare_legacy_compiler
        isolate_d4rl_offline_env()
        prepare_legacy_compiler()
        sys.path.insert(0, str(VENDOR / "src"))
        from environments.pointmaze.point_maze_wrapper import PointMazeWrapper
        from environments.pointmaze.maze_model import U_MAZE
        env = PointMazeWrapper(maze_spec=U_MAZE, reward_type="sparse", reset_target=False)
        establish_goals(env, rows, out, persist)
        zero_controls(env, rows, out, persist)
        eligible = [row["instance"] for row in rows if row["reference_verification"]["status"] == "completed"
                    and not row["initial_success"] and not row["zero"]["success"]]
        passed = len(eligible) >= 2
        summary.update(status="completed_cpu_task_selection_gate",
                       task_selection_gate_passed=passed,
                       decision="TASK_PANEL_DISCRIMINATIVE_NO_CLEAN_EXECUTED" if passed else "NO_GO_FIXED_PANEL_INSUFFICIENTLY_DISCRIMINATIVE",
                       eligible_instances=eligible, eligible_count=len(eligible),
                       initial_success_count=sum(row["initial_success"] for row in rows),
                       reference_verified_count=sum(row["reference_verification"]["status"] == "completed" for row in rows),
                       zero_success_count=sum(row["zero"]["success"] for row in rows),
                       environment_steps=150 + sum(row["zero"]["physical_steps"] for row in rows),
                       cuda_initialized=torch.cuda.is_initialized(),
                       planner_targets=relative(out / "planner_targets.json"),
                       reference_actions_excluded_from_planner_targets=True,
                       interpretation="Task-selection feasibility only. No Clean policy, forecast ranking, security damage or population inference. No sample replacement or automatic GPU continuation.")
        if summary["cuda_initialized"]:
            raise RuntimeError("Unexpected CUDA initialization in CPU-only task")
    except Exception as error:
        summary.update(status="technical_error_cpu_task_selection_gate", task_selection_gate_passed=None,
                       error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        if env is not None:
            env.close()
        summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        persist()
    print(json.dumps({key: summary[key] for key in ("status", "decision", "eligible_count", "task_selection_gate_passed", "clean_gate_passed", "environment_steps")}), flush=True)


if __name__ == "__main__":
    main()
