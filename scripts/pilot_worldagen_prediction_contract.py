#!/usr/bin/env python3
"""Small prediction-contract gate on an immutable successful native trajectory.

Only the forecast query receives counterfactual text. Policy-generated actions,
physical environment, three-step horizon, targets, and scoring stay fixed.
Uses raw native camera orientation and the released training target codec.
This tests a randomized forecast-query contract, not observation sufficiency
under the checkpoint's observational training distribution or task attacks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import pilot_worldagen_native_task_gate as gate

base = gate.base
from utils.train_utils import patchify  # noqa: E402: native patchify, no extra normalization


def array_hash(value) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if base.DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def snapshot_rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state_all() if base.DEVICE == "cuda" and torch.cuda.is_available() else []}


def numerical_fields(obj) -> dict:
    """Archive controller numerical attributes without serializing its simulator."""
    values = {}
    for key, value in vars(obj).items():
        if isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
            values[key] = value.copy()
        elif value is None or isinstance(value, (str, bool, int, float, np.number)):
            values[key] = value.item() if isinstance(value, np.generic) else value
    return values


def physical_snapshot(env, obs) -> dict:
    sim = env.sim
    robot = env.robots[0]
    contacts = []
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        contacts.append({"geom1": int(contact.geom1), "geom2": int(contact.geom2),
                         "pos": np.asarray(contact.pos).copy(),
                         "frame": np.asarray(contact.frame).copy(), "dist": float(contact.dist)})
    return {
        "sim": sim.get_state().flatten().copy(), "qpos": sim.data.qpos.copy(),
        "qvel": sim.data.qvel.copy(), "ctrl": sim.data.ctrl.copy(),
        "qacc_warmstart": sim.data.qacc_warmstart.copy(), "contacts": contacts,
        "controller": numerical_fields(robot.controller),
        "gripper": numerical_fields(robot.gripper),
        "primary": np.asarray(obs["agentview_image"]).copy(),
        "wrist": np.asarray(obs["robot0_eye_in_hand_image"]).copy(),
        "eef": np.asarray(obs["robot0_eef_pos"]).copy(),
    }


def recursive_signature(value):
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": array_hash(value)}
    if isinstance(value, dict):
        return {k: recursive_signature(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [recursive_signature(v) for v in value]
    return value


def make_at_prefix(task, source: dict, protocol: dict, actions: np.ndarray, prefix: int):
    """Recreate controller state by complete fresh reset and executed prefix."""
    seed_everything(int(source["seed"]))
    bddl = base.LIBERO_ROOT / "libero/libero/bddl_files" / task.problem_folder / task.bddl_file
    init_path = base.LIBERO_ROOT / "libero/libero/init_files" / task.problem_folder / task.init_states_file
    initial = torch.load(init_path, map_location="cpu", weights_only=False)[int(source["init_id"])]
    if gate.digest(bddl) != source["bddl_sha256"] or array_hash(initial) != source["initial_state_sha256"]:
        raise RuntimeError("source scene/initial-state provenance changed")
    env = base.OffScreenRenderEnv(bddl_file_name=str(bddl),
                                  camera_heights=int(protocol["camera_size"]),
                                  camera_widths=int(protocol["camera_size"]), render_gpu_device_id=0)
    try:
        env.task_id, env.task_name, env.task_suite_name = source["task_id"], task.name, source["suite"]
        env.seed(int(source["seed"]))
        env.reset()
        obs = env.set_init_state(initial)
        for _ in range(int(source["settle_steps"])):
            obs, *_ = env.step(np.zeros(7, dtype=np.float32))
        for action in actions[:prefix]:
            # Native wrapper executes fp16 actions. Preserve that exact dtype.
            obs, *_ = env.step(action.astype(np.float16))
        return env, obs
    except BaseException:
        env.close()
        raise


def prepare_inputs(model, obs, previous_gripper: float, candidates: torch.Tensor):
    images = [model.image_processor(base.Image.fromarray(obs[key]))
              for key in ("agentview_image", "robot0_eye_in_hand_image")]
    # First chunk is the only scored chunk. Causal masking forbids reads from
    # the duplicated, padding-only second context.
    device = next(model.parameters()).device
    batch = candidates.shape[0]
    primary = images[0][None, None].repeat(batch, 2, 1, 1, 1).to(device)
    wrist = images[1][None, None].repeat(batch, 2, 1, 1, 1).to(device)
    state = gate.prepare_state(obs, previous_gripper, "euler").repeat(batch, 2, 1).to(device)
    action = candidates.repeat(1, 2, 1).to(device)
    return primary, wrist, state, action


@torch.inference_mode()
def forecast(model, inputs, goal: str, mask, seed: int) -> torch.Tensor:
    primary, wrist, state, actions = inputs
    seed_everything(seed)
    model.attention_mask.copy_(mask)
    text = base.clip.tokenize([goal]).unsqueeze(1).repeat(primary.shape[0], 2, 1).to(primary.device)
    # The policy output is discarded. This call cannot change an executed or
    # selected action: both candidate trajectories were committed beforehand.
    _, _, world, _, _ = model(primary, wrist, state, text, actions)
    if not torch.isfinite(world).all():
        raise RuntimeError("nonfinite forecast")
    return world.reshape(primary.shape[0], 2, 2, 196, 768)[:, 0].float().cpu().clone()


def masks(model):
    native, bus = base.masks(model)
    control = native.clone()
    control[[37, 38, 39, 77, 78, 79, 77, 78, 79], [0, 0, 0, 0, 0, 0, 40, 40, 40]] = -float("inf")
    if int((torch.isfinite(native) & ~torch.isfinite(control)).sum()) != 9:
        raise RuntimeError("matched control no longer removes nine edges")
    if bool(torch.isfinite(native[:40, 40:]).any()):
        raise RuntimeError("first forecast can read the padding-only second chunk")
    return {"native": native, "bus_cut": bus, "control_cut": control}


def one_chunk(model, mask_dict, task, source, source_protocol, trace, trace_actions,
              chunk_index, goal_ids, goal_texts, args):
    prefix = chunk_index * 3
    original = torch.from_numpy(trace_actions[prefix:prefix + 3].copy()).float()
    perturbed = original.clone()
    perturbed[:, 0] = torch.clamp(perturbed[:, 0] + args.action_delta, -1, 1)
    if torch.equal(perturbed, original):
        perturbed[:, 0] = torch.clamp(perturbed[:, 0] - args.action_delta, -1, 1)
    # Quantize before both forecast and replay, matching native execution.
    candidates = torch.stack([original, perturbed]).half().float()
    if not torch.all((candidates[..., -1] == -1) | (candidates[..., -1] == 1)):
        raise RuntimeError("gripper action must remain binary +/-1")
    chunk_dir = args.output_dir / f"chunk_{chunk_index:03d}"
    chunk_dir.mkdir()
    torch.save(candidates, chunk_dir / "committed_actions.pt")
    env, obs = make_at_prefix(task, source, source_protocol, trace_actions, prefix)
    try:
        initial = physical_snapshot(env, obs)
        torch.save({"physical": initial, "rng": snapshot_rng()}, chunk_dir / "initial_snapshot.pt")
        signature = recursive_signature(initial)
        source_eef_error = float(np.max(np.abs(np.asarray(trace[prefix]["eef_before"]) - initial["eef"])))
        if source_eef_error > args.source_replay_tolerance:
            raise RuntimeError(f"clean trace prefix did not replay: EEF error {source_eef_error}")
        previous_gripper = float(trace_actions[prefix - 1, -1]) if prefix else -1.0
        inputs = prepare_inputs(model, obs, previous_gripper, candidates)
        torch.save({"primary": inputs[0].cpu(), "wrist": inputs[1].cpu(),
                    "state": inputs[2].cpu(), "action": inputs[3].cpu()}, chunk_dir / "model_inputs.pt")
        forecasts = {}
        for name, mask in mask_dict.items():
            forecasts[name] = torch.stack([
                forecast(model, inputs, goal, mask, args.forecast_seed) for goal in goal_texts
            ])
        clean_position = goal_ids.index(int(source["task_id"]))
        repeated = forecast(model, inputs, task.language, mask_dict["native"], args.forecast_seed)
        repeat_exact = torch.equal(repeated, forecasts["native"][clean_position])
        if not repeat_exact:
            raise RuntimeError("same-input same-RNG forecast failed exact-repeat control")
        torch.save({"forecasts": forecasts, "repeat": repeated}, chunk_dir / "forecasts_committed.pt")
        write_json(chunk_dir / "prediction_commitment.json", {
            "all_forecasts_saved_before_any_target_replay": True,
            "forecast_sha256": gate.digest(chunk_dir / "forecasts_committed.pt"),
            "inputs_sha256": gate.digest(chunk_dir / "model_inputs.pt"),
            "actions_sha256": array_hash(candidates.numpy()),
            "goal_ids": goal_ids, "goal_texts": goal_texts,
        })
    finally:
        env.close()

    target_list, replay_records = [], []
    for candidate_id, candidate in enumerate(candidates):
        expected = None
        for repetition in range(args.replay_repeats):
            env, obs = make_at_prefix(task, source, source_protocol, trace_actions, prefix)
            try:
                before = physical_snapshot(env, obs)
                if recursive_signature(before) != signature:
                    raise RuntimeError("fresh replay failed full numerical simulator/controller pairing")
                rewards, successes = [], []
                for action in candidate.numpy():
                    obs, reward, _, _ = env.step(action.astype(np.float16))
                    rewards.append(float(reward))
                    successes.append(bool(env.check_success()))
                after = physical_snapshot(env, obs)
                after_signature = recursive_signature(after)
                if expected is None:
                    expected = after_signature
                    processed = torch.stack([model.image_processor(base.Image.fromarray(obs[key]))
                                             for key in ("agentview_image", "robot0_eye_in_hand_image")])
                    target_list.append(patchify(processed, patch_size=16))
                    torch.save({"physical": after, "rng": snapshot_rng()},
                               chunk_dir / f"candidate_{candidate_id}_target_snapshot.pt")
                elif after_signature != expected:
                    raise RuntimeError("future simulator/controller/images differ on repeated fixed-action replay")
                replay_records.append({"candidate": candidate_id, "repetition": repetition,
                                       "initial_exact": True, "future_exact": True,
                                       "reward_sum": sum(rewards), "success_after_each_action": successes,
                                       "eef_displacement": (after["eef"] - before["eef"]).tolist(),
                                       "future_signature": after_signature})
            finally:
                env.close()
    targets = torch.stack(target_list)
    torch.save(targets, chunk_dir / "targets_native_patchify.pt")
    persistence = patchify(torch.stack([inputs[0][0, 0].cpu(), inputs[1][0, 0].cpu()]), patch_size=16)
    persistence_mse = (targets - persistence[None]).square().mean((-1, -2))
    errors = {name: (pred - targets[None]).square().mean((-1, -2)).tolist()
              for name, pred in forecasts.items()}
    effects = {name: (pred - pred[clean_position:clean_position + 1]).square().mean((-1, -2)).tolist()
               for name, pred in forecasts.items()}
    native_errors = np.asarray(errors["native"])
    nonclean = np.arange(len(goal_ids)) != clean_position
    return {
        "chunk_index": chunk_index, "prefix_steps": prefix, "forecast_horizon": 3,
        "source_trace_eef_max_abs_error": source_eef_error,
        "candidate_names": ["native_recorded_chunk", "precommitted_x_perturbation"],
        "view_axis": ["primary", "wrist"], "forecast_repeat_exact": repeat_exact,
        "target_mse_by_goal_action_view": errors, "goal_effect_mse_by_goal_action_view": effects,
        "clean_target_mse_mean": float(native_errors[clean_position].mean()),
        "persistence_target_mse_by_action_view": persistence_mse.tolist(),
        "persistence_target_mse_mean": float(persistence_mse.mean()),
        "clean_minus_persistence_target_mse_mean": float(native_errors[clean_position].mean() - persistence_mse.mean()),
        "nonclean_minus_clean_target_mse_mean": float((native_errors[nonclean] - native_errors[clean_position]).mean()),
        "bus_cut_cross_goal_exact": bool(torch.equal(forecasts["bus_cut"], forecasts["bus_cut"][:1].expand_as(forecasts["bus_cut"]))),
        "control_cut_native_forecast_exact": bool(torch.equal(forecasts["control_cut"], forecasts["native"])),
        "action_sensitivity_mse_by_goal_view": {
            name: (pred[:, 1] - pred[:, 0]).square().mean((-1, -2)).tolist()
            for name, pred in forecasts.items()},
        "bus_cut_minus_native_clean_target_mse_mean": float(np.asarray(errors["bus_cut"])[clean_position].mean() - native_errors[clean_position].mean()),
        "replays": replay_records, "artifacts": str(chunk_dir),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunk-indices", type=gate.parse_ids, default=[0, 10, 30])
    ap.add_argument("--forecast-seed", type=int, default=20261014)
    ap.add_argument("--action-delta", type=float, default=0.25)
    ap.add_argument("--replay-repeats", type=int, default=2)
    ap.add_argument("--source-replay-tolerance", type=float, default=1e-8)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.threads < 1:
        ap.error("threads must be positive")
    base.DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if base.DEVICE == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.replay_repeats < 2 or not 0 < args.action_delta <= 1:
        ap.error("require at least two replay repeats and 0 < action-delta <= 1")
    source = json.loads(args.trace.with_suffix(".json").read_text())
    source_summary = json.loads((args.trace.parent / "summary.json").read_text())
    source_protocol = source_summary["protocol"]
    if source["condition"] != "native" or not source["success"] or source["suite"] != "libero_10":
        ap.error("source must be a successful native LIBERO_10 trajectory")
    if source_protocol["orientation"] != "euler":
        ap.error("this primary protocol requires the native Euler codec")
    trace = [json.loads(line) for line in args.trace.read_text().splitlines()]
    if [r["step"] for r in trace] != list(range(1, len(trace) + 1)):
        ap.error("source trace has missing or duplicated steps")
    if max(args.chunk_indices) * 3 + 3 > len(trace):
        ap.error("requested chunk extends beyond recorded actions")
    actions = np.asarray([row["action"] for row in trace], dtype=np.float32)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error("choose an empty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suite = base.benchmark.get_benchmark_dict()[source["suite"]]()
    task = suite.get_task(int(source["task_id"]))
    goal_ids = list(range(suite.get_num_tasks()))
    goal_texts = [suite.get_task(i).language for i in goal_ids]
    protocol = {
        "status": "registered_before_inference", "script_sha256": gate.digest(Path(__file__)),
        "source_trace": str(args.trace.resolve()), "source_trace_sha256": gate.digest(args.trace),
        "source_native_success": source, "source_protocol": source_protocol,
        "checkpoint_provenance": source_summary["checkpoint_provenance"],
        "model_device": base.DEVICE, "torch_threads": args.threads,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "chunk_indices": args.chunk_indices, "forecast_horizon": 3, "goal_ids": goal_ids,
        "goal_texts": goal_texts, "clean_goal_id": source["task_id"],
        "goal_assignment": "full crossing independent of physical state and fixed action",
        "forecast_seed": args.forecast_seed, "action_delta": args.action_delta,
        "replay_repeats": args.replay_repeats, "only_first_chunk_scored": True,
        "world_query_goal_only": True, "policy_output_consumed": False,
        "environment_goal_and_bddl_fixed": True, "stop_on_success_during_three_steps": False,
        "image_codec": "native raw simulator orientation; released CLIP image_processor",
        "target_codec": "official utils.train_utils.patchify; NO extra per-patch normalization",
        "proprio_codec": "native Euler xyz and previous binary gripper command",
        "action_codec": "three fp16-representable recorded actions, gripper +/-1",
        "scope": "randomized task-agnostic forecast-query contract; not proof of training-distribution observation sufficiency or task attack",
        "training_updates": 0, "api_requests": 0,
    }
    write_json(args.output_dir / "protocol.json", protocol)
    model = base.build_model()
    mask_dict = masks(model)
    rows = []
    summary = {**protocol, "status": "running", "rows": rows}
    started = time.monotonic()
    try:
        for chunk in args.chunk_indices:
            print(json.dumps({"starting_chunk": chunk, "prefix_steps": chunk * 3}), flush=True)
            row = one_chunk(model, mask_dict, task, source, source_protocol, trace, actions,
                            chunk, goal_ids, goal_texts, args)
            rows.append(row)
            summary["elapsed_seconds"] = time.monotonic() - started
            write_json(args.output_dir / "summary.json", summary)
            print(json.dumps({k: row[k] for k in ("chunk_index", "clean_target_mse_mean",
                "nonclean_minus_clean_target_mse_mean", "bus_cut_cross_goal_exact",
                "control_cut_native_forecast_exact")}), flush=True)
        summary["status"] = "completed_prediction_contract_gate"
        summary["aggregate"] = {
            "chunks": len(rows), "physical_transitions": len(rows) * 2,
            "goal_forecasts_per_mask": len(rows) * 2 * len(goal_ids),
            "clean_target_mse_mean": float(np.mean([r["clean_target_mse_mean"] for r in rows])),
            "persistence_target_mse_mean": float(np.mean([r["persistence_target_mse_mean"] for r in rows])),
            "nonclean_minus_clean_target_mse_mean": float(np.mean([r["nonclean_minus_clean_target_mse_mean"] for r in rows])),
            "bus_cut_cross_goal_exact_all": all(r["bus_cut_cross_goal_exact"] for r in rows),
            "control_cut_native_exact_all": all(r["control_cut_native_forecast_exact"] for r in rows),
            "forecast_repeat_exact_all": all(r["forecast_repeat_exact"] for r in rows),
            "independent_episode_clusters": 1,
            "uncertainty": "descriptive pilot only; correlated chunks of one episode, no significance claim",
        }
    except Exception as error:
        summary["status"] = "failed_prediction_contract_gate"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "aggregate": summary["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()
