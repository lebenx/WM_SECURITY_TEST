#!/usr/bin/env python3
"""Matched-suite full-task WorldAgen gate with explicit input-codec ablations.

The primary protocol preserves the locally successful released-wrapper path:
LIBERO_10, 224-pixel rendering, Euler orientation, binary gripper command.
The world forecast is evaluated but is not consumed by the frozen policy.
Task success is exclusively the simulator's native success predicate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

os.environ.setdefault("LIBERO_CONFIG_PATH", str(Path(__file__).resolve().parents[1] / ".libero"))
os.environ.setdefault("MUJOCO_GL", "egl")
import pilot_worldagen_libero_closed_loop as base


def parse_ids(value: str) -> list[int]:
    result = [int(x) for x in value.split(",")]
    if not result or len(set(result)) != len(result) or min(result) < 0:
        raise argparse.ArgumentTypeError("provide unique nonnegative comma-separated IDs")
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_state(obs: dict, gripper: float, orientation: str) -> torch.Tensor:
    quaternion = np.array(obs["robot0_eef_quat"], dtype=np.float64, copy=True)
    if orientation == "euler":
        rotation = Rotation.from_quat(quaternion).as_euler("xyz", degrees=False)
    else:
        # Match LIBERO create_dataset.py -> WorldAgen load_robot_obs exactly.
        # The dataset stores axis-angle, then the loader canonicalizes those
        # three numerical values as Euler angles. This is not quat->Euler.
        from robosuite.utils.transform_utils import quat2axisangle
        axis_angle = quat2axisangle(quaternion)
        rotation = Rotation.from_euler("xyz", axis_angle, degrees=False).as_euler("xyz", degrees=False)
    state = np.concatenate([obs["robot0_eef_pos"], rotation, [gripper]])
    return torch.tensor(state[None, None], dtype=torch.float32)


def episode(model, mask, task, suite_name: str, task_id: int, init_id: int,
            condition: str, args, output: Path) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model.attention_mask.copy_(mask)
    bddl = base.LIBERO_ROOT / "libero/libero/bddl_files" / task.problem_folder / task.bddl_file
    init_path = base.LIBERO_ROOT / "libero/libero/init_files" / task.problem_folder / task.init_states_file
    init_states = torch.load(init_path, map_location="cpu", weights_only=False)
    if init_id >= len(init_states):
        raise ValueError(f"initial state {init_id} out of range for {task.name}")
    env = base.OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=args.camera_size,
        camera_widths=args.camera_size, render_gpu_device_id=0,
    )
    env.task_id, env.task_name, env.task_suite_name = task_id, task.name, suite_name
    episode_name = f"{suite_name}_task{task_id:02d}_init{init_id:02d}_{condition}"
    trace_path = output / f"{episode_name}.jsonl"
    started = time.monotonic()
    norms = []
    success = False
    steps = 0
    chunks = 0
    terminal_done = False
    try:
        env.seed(args.seed)
        env.reset()
        obs = env.set_init_state(init_states[init_id])
        initial_state_sha256 = hashlib.sha256(np.asarray(init_states[init_id]).tobytes()).hexdigest()
        # Read back the settled observation; the old evaluator left obs stale.
        for _ in range(args.settle_steps):
            obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
        success_at_start = bool(env.check_success())
        image_fn = lambda ims: base.preprocess_image(ims, image_processor=model.image_processor)
        wrapper = base.WorldAgenLiberoModelWrapper(
            base.make_args((args.max_steps + 2) // 3), model, base.clip,
            model.image_processor, torch.float32, history_len=2,
            libero_eval_max_steps=args.max_steps,
        )
        wrapper.reset()
        goal = "" if condition == "empty_goal" else task.language
        text = base.preprocess_text_calvin([goal], tokenizer=base.clip)
        last_gripper = -1.0
        with trace_path.open("w") as trace, torch.inference_mode():
            while steps < args.max_steps:
                primary = image_fn([base.Image.fromarray(obs["agentview_image"])]).unsqueeze(1)
                wrist = image_fn([base.Image.fromarray(obs["robot0_eye_in_hand_image"])]).unsqueeze(1)
                state = prepare_state(obs, last_gripper, args.orientation)
                if chunks == 0:
                    action = wrapper.step_agent(text, state, primary, wrist, torch.zeros(1, 3, 7))
                else:
                    action = wrapper.step_agent(text, state, primary, wrist)
                if not torch.isfinite(action).all():
                    raise RuntimeError("non-finite policy action")
                # Preserve native action history and native world-head call.
                # These predicted images are deliberately not used for control.
                wrapper.step_world_model(action)
                chunks += 1
                norms.append(float(action.float().norm()))
                for action_id in range(min(action.shape[1], args.max_steps - steps)):
                    act = action[0, action_id].numpy().copy()
                    before_eef = np.asarray(obs["robot0_eef_pos"]).tolist()
                    obs, reward, done, _ = env.step(act)
                    steps += 1
                    last_gripper = float(act[-1])
                    wrapper.gripper_state = np.array([last_gripper])
                    success = bool(env.check_success())
                    terminal_done = bool(done)
                    trace.write(json.dumps({
                        "step": steps, "chunk": chunks, "action": act.tolist(),
                        "eef_before": before_eef,
                        "eef_after": np.asarray(obs["robot0_eef_pos"]).tolist(),
                        "eef_quaternion": np.asarray(obs["robot0_eef_quat"]).tolist(),
                        "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"]).tolist(),
                        "reward": float(reward), "done": bool(done),
                        "task_success": success,
                    }) + "\n")
                    if success or done:
                        break
                if chunks % args.log_every_chunks == 0 or success or terminal_done:
                    trace.flush()
                    print(json.dumps({"episode": episode_name, "steps": steps,
                                      "success": success, "elapsed_seconds": time.monotonic()-started}), flush=True)
                if success or terminal_done:
                    break
        base.Image.fromarray(obs["agentview_image"]).save(output / f"{episode_name}_final.png")
        result = {
            "suite": suite_name, "task_id": task_id, "task_name": task.name,
            "goal": task.language, "init_id": init_id, "condition": condition,
            "seed": args.seed, "success": success, "success_at_start": success_at_start,
            "environment_steps": steps, "settle_steps": args.settle_steps,
            "terminal_done": terminal_done, "chunks": chunks,
            "action_norm_mean": float(np.mean(norms)) if norms else None,
            "action_norm_max": float(np.max(norms)) if norms else None,
            "initial_state_sha256": initial_state_sha256,
            "bddl_sha256": digest(bddl), "trace": str(trace_path),
            "elapsed_seconds": time.monotonic() - started,
        }
        (output / f"{episode_name}.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-ids", type=parse_ids, default=list(range(10)))
    parser.add_argument("--init-ids", type=parse_ids, default=[0, 1, 2])
    parser.add_argument("--conditions", nargs="+", choices=["native", "edge_cut", "empty_goal"], default=["native"])
    parser.add_argument("--orientation", choices=["euler", "train_axis_angle"], default="euler")
    parser.add_argument("--camera-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--settle-steps", type=int, default=5)
    parser.add_argument("--log-every-chunks", type=int, default=25)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.max_steps < 1 or args.log_every_chunks < 1 or args.settle_steps < 0:
        parser.error("invalid step/log limits")
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists() or any(output.glob("*.jsonl")):
        parser.error("output directory already contains a run; choose a new directory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    suite = base.benchmark.get_benchmark_dict()[args.suite]()
    if any(i >= suite.n_tasks for i in args.task_ids):
        parser.error(f"task ID exceeds suite size {suite.n_tasks}")
    model = base.build_model()
    native, cut = base.masks(model)
    mask_by_condition = {"native": native, "edge_cut": cut, "empty_goal": native}
    results = []
    started = time.monotonic()
    summary = {
        "status": "running", "model": "WorldAgen released LIBERO-38",
        "script_sha256": digest(Path(__file__)),
        "checkpoint_provenance": json.loads((base.MODEL_DIR / "provenance.json").read_text()),
        "protocol": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "forecast_used_for_action_selection": False,
        "gripper_codec": "binary command, gripper_width=False",
        "success_source": "LIBERO environment check_success(), never forecast scores",
        "training_updates": 0, "api_requests": 0, "results": results,
    }
    summary_path = output / "summary.json"
    for task_id in args.task_ids:
        for init_id in args.init_ids:
            for condition in args.conditions:
                print(json.dumps({"starting": {"task": task_id, "init": init_id, "condition": condition}}), flush=True)
                result = episode(model, mask_by_condition[condition], suite.get_task(task_id),
                                 args.suite, task_id, init_id, condition, args, output)
                results.append(result)
                summary["elapsed_seconds"] = time.monotonic() - started
                summary["aggregate"] = {
                    c: {"episodes": len(rows), "successes": sum(r["success"] for r in rows),
                        "success_rate": float(np.mean([r["success"] for r in rows])) if rows else None}
                    for c in args.conditions if (rows := [r for r in results if r["condition"] == c])
                }
                summary_path.write_text(json.dumps(summary, indent=2) + "\n")
                print(json.dumps(result), flush=True)
    summary["status"] = "completed"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": str(summary_path), "aggregate": summary["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()
